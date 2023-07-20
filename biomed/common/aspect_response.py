from enum import Enum
from typing import List

from biomed.biomed_env import BiomedEnv
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.feature_data_parsing import overlap_ranges
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.constants.features import LabLabel

from biomed.constants.constants import (
    DEFAULT_MIN_SCORE, NEARBY_TERM_THRESHOLD, MAX_TERM_SEPERATION_THRESHOLD,
    BiomedVersionInfo, ModelType)
from biomed.constants.stopwords import STOP_WORDS
from biomed.common.biomed_ouput import SummaryOutput, BiomedOutput, CancerOutput, AttributeWithRange, LabOutput


class AspectResponse:
    """class that holds a category name (ie: 'Lab') and a response list which is a list of BiomedOutput
    (summary/drug output)  objects. Has methods for merging nearby terms and for filtering out stop words and predicted
    terms with a score lower than the minimum threshold provided
    (this is what the output of problem.py, cancer.py etc is)"""

    def __init__(self, category_name: str, response_list: List[BiomedOutput]):
        self.category_name = category_name
        self.response_list: List[BiomedOutput] = response_list
        self.sort_response_list()

    def sort_response_list(self):
        self.response_list = sorted(self.response_list, key=lambda i: i.range)

    def remove_duplicate_responses(self):
        # requires response list be sorted by range
        for idx in range(len(self.response_list) - 1, 0, -1):
            if self.response_list[idx].range == self.response_list[idx - 1].range:
                self.response_list.pop(idx)

    def post_process(self, text: str, min_score: float = DEFAULT_MIN_SCORE):
        self.sort_response_list()
        self.merge_nearby_terms(text)
        self.filter_responses(min_score)

    def to_json(self):
        return {self.category_name: [response.to_dict() for response in self.response_list]}

    def to_versioned_json(self, model_type: ModelType = None, biomed_version: str = None):
        return self.__make_versioned_json(self.to_json(), model_type, biomed_version)

    @staticmethod
    def __make_versioned_json(json_response, model_type: ModelType = None, biomed_version: str = None):
        if model_type:
            json_response[VERSION_INFO_KEY] = BiomedVersionInfo(model_type=model_type,
                                                                biomed_version=biomed_version).to_list_dict()
        return json_response

    def filter_responses(self, min_score: float):
        """
        Filters entities with low probability
        :param min_score: minimum score below which entity is not returned
        :return: works in place, updates the AspectResponse objects response_list attribute
        """
        operations_logger.info(f'Filtering {self.category_name} responses with minimum score={min_score}')
        if not 0 <= min_score < 1:
            raise ValueError('Min score must be a valid probability, between 0 and 1')

        # takes out responses with probability < min score and takes out stop words, short responses
        for p in range(len(self.response_list) - 1, -1, -1):
            pred = self.response_list[p]
            stripped_text = pred.text.strip().lower()
            if (pred.lstm_prob < min_score or not stripped_text or stripped_text in STOP_WORDS or
                    len(stripped_text) < 2):
                self.response_list.pop(p)

    @staticmethod
    def combine_clinical_concepts(res_1: BiomedOutput, res_2: BiomedOutput, text: str) -> BiomedOutput:
        # if the first object passed in (res_1) has a preferred text, use that umls concept,
        # otherwise use res_2 umls concept
        merged_concept = res_1.combine(res_1, res_2, text)

        return merged_concept

    @staticmethod
    def nearby_pref_text_overlap(res_1: SummaryOutput, res_2: SummaryOutput, text) -> bool:
        """
        :return: true if preferred text doesnt exist for one or both values or if both have the same preferred text
        """
        # check closeby
        if (abs(res_2['range'][0] - res_1['range'][1]) < NEARBY_TERM_THRESHOLD and
                abs(res_2['range'][1] - res_1['range'][0]) < MAX_TERM_SEPERATION_THRESHOLD):
            # check prefText overlap
            if isinstance(res_1, SummaryOutput) and res_1.preferredText and res_2.preferredText:
                return res_1.preferredText == res_2.preferredText
            # if one or both dont have pref text check that there's no line separator
            # TODO: Is this desired behavior? Merge every instance of within terms if one or other has no preferredtext
            return '\n' not in text[res_1['range'][1]:res_2['range'][0]]
        return False

    @text2phenotype_capture_span()
    def merge_nearby_terms(self, text: str):
        """
        merge nearby tokens in the result to form complete problem term, e.g. Anemia of chronic disease,
        works in place, updates the AspectResponse objects response_list attribute
        """
        i = 1
        # delete copies of the same term getting tagged twice (due to multiplicity of UMLS terms).
        while i < len(self.response_list):
            if overlap_ranges(self.response_list[i].range, self.response_list[i - 1].range):
                self.response_list.pop(i)
                continue
            else:
                i += 1
        merged_result = list()
        operations_logger.debug(f'Merging Nearby Terms for aspect {self.category_name}. '
                                f'Unmerged List Length ={self.response_list}')
        if len(self.response_list) > 0:
            last_concept = self.response_list[0]
            for i in range(1, len(self.response_list)):
                current_concept = self.response_list[i]
                # if there is a nearby term that matches, then combine the terms
                merged = self.create_merged_term(last_concept, current_concept, text)
                if not merged:
                    # append the last concept to merged_result iff there are no more nearby terms that match
                    merged_result.append(last_concept)
                    last_concept = current_concept
                else:
                    last_concept = merged
            merged_result.append(last_concept)
        operations_logger.debug(
            f'Updating the response list for aspect {self.category_name} new length is {len(self.response_list)}')
        self.response_list = merged_result

    def create_merged_term(self, last_concept: SummaryOutput, current_concept: SummaryOutput, text):
        """
        Performs check for combining clinical concepts, returns None if no merge performed
        :param last_concept: last entity seen/processed
        :param current_concept: entity currently being considered for merge to lsst
        :param text: text sequence under consideration
        :return: merged entity or None
        """
        if self.nearby_pref_text_overlap(last_concept, current_concept, text) and \
                last_concept.label == current_concept.label:
            last_concept = self.combine_clinical_concepts(last_concept, current_concept, text)
            return last_concept

    @classmethod
    def from_json(cls, category_name: str, biomed_output_list: List[dict], biomed_output_class):
        response_list = [biomed_output_class(**output_dict) for output_dict in biomed_output_list]
        output = cls(category_name=category_name, response_list=response_list)
        output.sort_response_list()
        # remove duplicates
        output.remove_duplicate_responses()
        return output


class CancerResponse(AspectResponse):
    def create_merged_term(self, last_concept: CancerOutput, current_concept: CancerOutput, text):
        if CancerOutput.nearby_cancer_grade(last_concept, current_concept):
            new_last = self.combine_clinical_concepts(last_concept, current_concept, text)
        elif self.nearby_pref_text_overlap(last_concept, current_concept, text) and \
                last_concept.label == current_concept.label:
            new_last = self.combine_clinical_concepts(last_concept, current_concept, text)
        else:
            new_last = None
        return new_last


class LabResponse(AspectResponse):

    def post_process(self, text: str, min_score: float = DEFAULT_MIN_SCORE):
        self.merge_nearby_terms(text)
        self.associate_lab_attributes()
        self.remove_lab_attribute_terms()
        self.filter_responses(min_score)

    def associate_lab_attributes(self):
        lab_results = self.response_list
        self.sort_response_list()
        for i in range(len(lab_results)):
            res = lab_results[i]
            if res.label in {LabLabel.lab_interp.name, LabLabel.lab_unit.name, LabLabel.lab_value.name}:
                lab_name_idx = self.find_nearest_lab_name(i)
                if lab_name_idx >= 0:
                    # matches ctakes attribute format
                    attribute = AttributeWithRange(text=res.text, range=res.range, prob=res.lstm_prob)
                    if res.label == LabLabel.lab_interp.name:
                        lab_results[lab_name_idx].labInterp = attribute
                        lab_results[lab_name_idx].polarity = self.lab_interp_parse(res.text)
                    # if we predict a unit and ctakes didn't previously assign, assign to nearest lab
                    elif res.label == LabLabel.lab_unit.name and not lab_results[lab_name_idx].labUnit:
                        lab_results[lab_name_idx].labUnit = attribute
                    elif res.label == LabLabel.lab_value.name and not lab_results[lab_name_idx].labValue:
                        lab_results[lab_name_idx].labValue = attribute

        self.response_list = lab_results

    def remove_lab_attribute_terms(self):
        i = 0
        lab_results = self.response_list
        while i < len(lab_results):
            if not lab_results[i].is_lab_name():
                lab_results.pop(i)
            else:
                i += 1
        self.response_list: List[LabOutput] = lab_results

    def find_nearest_lab_name(self, attribute_index: int,
                              max_range_diff_association=BiomedEnv.MAX_LAB_DISTANCE.value) -> int:
        # ensure sorted
        best_lab_idx = -1
        ref_range = self.response_list[attribute_index].range
        # if there's a preceding lab name within the lab range limit assign the value to that  lab
        for i in range(attribute_index - 1, -1, -1):
            if self.response_list[i].is_lab_name():
                distance_measure = min(abs((self.response_list[i].range[0] - ref_range[1])),
                                       abs((self.response_list[i].range[1] - ref_range[0])))
                if distance_measure < max_range_diff_association:
                    best_lab_idx = i
                break
        if best_lab_idx == -1:
            # if no preceding lab asign to closest lab after the attribute if one exissts within the distance limit
            for i in range(attribute_index + 1, len(self.response_list)):
                if self.response_list[i].is_lab_name():
                    distance_measure = min(abs((self.response_list[i].range[0] - ref_range[1])),
                                           abs((self.response_list[i].range[1] - ref_range[0])))
                    if distance_measure < max_range_diff_association:
                        best_lab_idx = i
                    break

        return best_lab_idx

    @staticmethod
    def lab_interp_parse(annot_text: str) -> str:
        annot_text = annot_text.lower()
        file_interp = None
        if ('positive' in annot_text or 'critical' in annot_text or 'abnormal' in annot_text or
                annot_text == 'detected not detected'):
            file_interp = LabPolarity.pos.value
        elif 'pending' in annot_text or 'process' in annot_text or 'progress' in annot_text:
            file_interp = LabPolarity.pending.value
        elif 'neg' in annot_text or 'not detected' in annot_text or 'none' in annot_text:
            file_interp = LabPolarity.neg.value
        elif 'detected' in annot_text:
            file_interp = LabPolarity.pos.value
        return file_interp


class LabPolarity(Enum):
    pos = 'positive'
    neg = 'negative'
    pending = 'pending'
