from typing import List, Tuple, Dict

from biomed.constants.response_mapping import KEY_TO_ASPECT_OUTPUT_CLASSES
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.constants.features.label_types import CancerLabel, AllergyLabel, MedLabel

from biomed.constants.constants import DEFAULT_MIN_SCORE, ORDERED_CATEGORY_TIEBRAKER, BiomedVersionInfo, ModelType
from biomed.common.biomed_ouput import MedOutput, BiomedOutput
from biomed.common.aspect_response import AspectResponse
from text2phenotype.tasks.task_enums import TaskOperation


def check_duplicates(entry: BiomedOutput, check_list: List[BiomedOutput]):
    """
    Checks if entry token ranges overlap with any entities in check_list
    :param entry: umls summary response, a dictionary value that should have prob, range, and text values
    :param check_list: list of umls responses to check if the entry is in
    :return: if entry range overlaps with an range in check_list return the first overlapping index, else return -1
    """
    rng1 = entry.range
    for i in range(len(check_list)):
        rng2 = check_list[i].range
        if rng1[0] <= rng2[1] and rng2[0] <= rng1[1]:
            return i
    return -1


class FullSummaryResponse:
    def __init__(self, aspect_responses: List[AspectResponse] = None):
        self.aspect_responses: List[AspectResponse] = aspect_responses if aspect_responses else list()

    def __getitem__(self, item) -> AspectResponse:
        if item in self.aspects:
            return self.aspect_responses[self.aspects[item]]

    def __setitem__(self, key: str, value: AspectResponse):
        if key in self.aspects:
            self.aspect_responses[self.aspects[key]] = value

    @property
    def category_names(self) -> set:
        return {aspect.category_name for aspect in self.aspect_responses}

    @property
    def aspects(self):
        aspects = dict()
        for i in range(len(self.aspect_responses)):
            aspects[self.aspect_responses[i].category_name] = i
        return aspects

    def add(self, response: [AspectResponse, Tuple]):
        if isinstance(response, Tuple):
            for i in range(len(response)):
                self.aspect_responses.append(response[i])
        else:
            self.aspect_responses.append(response)

    def postprocess(self, text, min_score: float = DEFAULT_MIN_SCORE):
        for aspect_resp in self.aspect_responses:
            aspect_resp.post_process(text=text, min_score=min_score)

        self.remove_summary_duplicates()
        self.remove_allergies_from_meds()
        self.filter_and_merge(text=text, min_score=min_score)

    def to_json(self, task_operation: TaskOperation = None, biomed_version: str = None, model_type: ModelType = None):
        json_response = dict()
        if task_operation or model_type:
            json_response[VERSION_INFO_KEY] = BiomedVersionInfo(task_operation=task_operation,
                                                                model_type=model_type,
                                                                biomed_version=biomed_version).to_list_dict()
        for aspect_res in self.aspect_responses:
            json_response.update(aspect_res.to_json())
        return json_response

    def filter_and_merge(self, text, min_score: float = DEFAULT_MIN_SCORE):
        for i in range(len(self.aspect_responses)):
            self.aspect_responses[i].post_process(text, min_score=min_score)

    def remove_summary_duplicates(self):
        # removes terms that appear in multiple categories from teh category predicted with lower probability
        # ignores the cancer category because those terms can/should be double labeled
        for i in range(len(self.aspect_responses)):
            # loop through each aspect response (ie: category) where there are responses and category!=cancer
            current_response = self.aspect_responses[i]

            if len(current_response.response_list) > 0 and \
                    current_response.category_name != CancerLabel.get_category_label().persistent_label:

                for j in range(i + 1, len(self.aspect_responses)):
                    # loop through the responses after the list and ensure that the comparison aspect != cancer
                    comparison_response = self.aspect_responses[j]
                    if comparison_response.category_name != CancerLabel.get_category_label().persistent_label and \
                            comparison_response.response_list!=[]:
                        updated_curr, updated_comparison = self.remove_duplicates(current_response.response_list,
                                                                                  comparison_response.response_list,
                                                                                  ORDERED_CATEGORY_TIEBRAKER.get(
                                                                                      current_response.category_name,
                                                                                      0) >
                                                                                  ORDERED_CATEGORY_TIEBRAKER.get(
                                                                                      comparison_response.category_name,
                                                                                      0))
                        current_response.response_list = updated_curr
                        comparison_response.response_list = updated_comparison
                        self.aspect_responses[j] = comparison_response
                self.aspect_responses[i] = current_response

    @staticmethod
    def remove_duplicates(list_1: List[BiomedOutput],
                          list_2: List[BiomedOutput],
                          prefer_1: bool) -> Tuple[List[BiomedOutput], List[BiomedOutput]]:
        """
        Compares two lists of entities for overlapping token spans, prefer token with higher probability
        :param list_1: list of entities for comparison
        :param list_2: list of entities for comparison
        :param prefer_1: if True, in case of tied probability, prefer the token in the list_1
        :return: lists with modifications (if any)
        """
        list_len = len(list_1)
        for i in range(len(list_1)):
            index = list_len - i - 1
            # check if there are overlapping ranges between the current response (SummaryOutput object)
            # and any of the objects in the second list
            j = check_duplicates(list_1[index], list_2)
            if j >= 0:
                prob_1 = list_1[index].lstm_prob
                prob_2 = list_2[j].lstm_prob
                # remove overlapping entries based on which probability is higher
                if prob_2 > prob_1 or (not prefer_1 and prob_2 == prob_1):
                    list_1.pop(index)
                else:
                    list_2.pop(j)

        return list_1, list_2

    @staticmethod
    def filter_out_response_by_range(
            list_1: List[BiomedOutput],
            filter_biomed_output_list: List[BiomedOutput]) -> List[BiomedOutput]:
        """
        list_1: a list of biomed outputs that you want to remove all entries that overlap with rangess included in the
        filter_biomed_output_list
        filter_biomed_output_list: a list of biomed outputs (or classes that inherit from biomed output)
        that you want to use as a filter on list_1
        """
        for i in range(len(filter_biomed_output_list)):

            # check if there are overlapping ranges between the current response (SummaryOutput object)
            # and any of the objects in the second list
            j = check_duplicates(filter_biomed_output_list[i], list_1)
            if j >= 0:
                list_1.pop(j)

        return list_1

    @text2phenotype_capture_span()
    def remove_allergies_from_meds(self, tid: str = None):
        # removes any cui, concept or text identified anywhere as allergies from medications list
        operations_logger.debug('Removing Allergy Responses from the Med Category', tid=tid)

        allergy_response = self.__getitem__(AllergyLabel.get_category_label().persistent_label)
        med_response = self.__getitem__(MedLabel.get_category_label().persistent_label)

        if allergy_response is None or med_response is None:
            return

        meds = med_response.response_list.copy()
        # remove matching by cui or text or code
        for allergy in allergy_response.response_list:
            med_len = len(meds)
            for i in range(med_len):
                index = med_len - i - 1
                cui_match = allergy.cui is not None and allergy.cui == meds[index].cui
                code_match = allergy.code is not None and allergy.code == meds[index].code
                text_match = allergy.text is not None and allergy.text.lower().strip() == meds[
                    index].text.lower().strip()
                if cui_match or code_match or text_match:
                    allerg_new = meds.pop(index)
                    if not isinstance(allerg_new, MedOutput):
                        raise TypeError('Extracted Medications must be MedOutput Objects')
                    allergy_response.response_list.append(allerg_new.to_allergy())

        allergy_response.sort_response_list()
        med_response.response_list = meds
        self.aspect_responses[self.aspects[AllergyLabel.get_category_label().persistent_label]] = allergy_response
        self.aspect_responses[self.aspects[MedLabel.get_category_label().persistent_label]] = med_response

    @classmethod
    def from_json(cls, json_input: Dict[str, List[dict]]) -> 'FullSummaryResponse':
        aspect_responses = []
        for group_title in json_input:
            if group_title != VERSION_INFO_KEY:
                if group_title in KEY_TO_ASPECT_OUTPUT_CLASSES:
                    aspect_response_class, biomed_output_class = KEY_TO_ASPECT_OUTPUT_CLASSES.get(group_title)
                else:
                    operations_logger.info(f'{group_title} does not have a mapping in KEY_TO_ASPECT_OUTPUT_CLASSES,'
                                           f' using default classes')
                    aspect_response_class, biomed_output_class = AspectResponse, BiomedOutput
                aspect_resp = aspect_response_class.from_json(
                    category_name=group_title,
                    biomed_output_list=json_input[group_title],
                    biomed_output_class=biomed_output_class
                )
                aspect_responses.append(aspect_resp)
        return cls(aspect_responses=aspect_responses)


def combine_all_biomed_outputs(biomed_outputs: List[dict], remove_duplicates: bool = True) -> FullSummaryResponse:
    full_biomed_output = {}
    for biomed_json in biomed_outputs:
        # assumes normal biomed output format
        for cat_name, biomed_out_list in biomed_json.items():
            # handling case where there are multiple responses of the same model by keeping all of them
            # and then removing duplicates
            if cat_name in full_biomed_output:
                full_biomed_output[cat_name].extend(biomed_out_list)
            else:
                full_biomed_output[cat_name] = biomed_out_list or []
    summary = FullSummaryResponse.from_json(full_biomed_output)
    if remove_duplicates:
        summary.remove_summary_duplicates()
    return summary


def combine_all_biomed_output_fps(biomed_output_fps: List[str]) -> FullSummaryResponse:
    """
    :param biomed_output_fps: file paths to all json outputs,assumes they follow they follow the format of  all
    non-demographic outputs ie {CATEGORY_NAME: [{'text', 'range', 'score', etc.}]
    :return: a  Full summary response with duplicates removed
    """
    biomed_json_list = []
    for biomed_output_fp in biomed_output_fps:
        biomed_json_list.append(common.read_json(biomed_output_fp))
    return combine_all_biomed_outputs(biomed_json_list)
