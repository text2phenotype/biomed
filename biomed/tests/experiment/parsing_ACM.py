import bisect
import os
import numpy as np
from gevent import time

from biomed.common.combined_model_label import DiseaseSignSymptomLabel, DrugLabel
from biomed.common.model_test_helpers import document_cui_set, prep_cui_reports
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.testing_reports import (
    WeightedReport,
    RemovingAdjacentConfusion,
    MinusPartialAnnotation,
    FullReport,
)
from text2phenotype.common import common
from biomed import RESULTS_PATH

from typing import List, Dict
from text2phenotype.common import speech

from text2phenotype.common.data_source import DataSourceContext
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import LabelEnum


class ACMEntity:
    def __init__(self, input_dict):
        self.ID = input_dict.get("Id")
        self.start_range = input_dict["BeginOffset"]
        self.end_range = input_dict["EndOffset"]
        self.score = input_dict["Score"]
        self.category = input_dict["Category"]
        self.type = input_dict["Type"]
        self.traits = input_dict["Traits"]
        self.attributes = input_dict["Attributes"]
        self.text = input_dict["Text"]

        self._label = None

    @property
    def label(self):
        if not self._label and len(self.traits) > 0:
            self._label = self.traits[0]["Name"]
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    def __repr__(self):
        """dataclass-like repr"""
        return (
            f"{self.__class__.__name__}("
            + ", ".join([f"{name}={val}" for name, val in self.__dict__.items() if not isinstance(val, list)])
            + ")"
        )


class ACMOutput:
    CATEGORIES = {
        "ANATOMY",
        "MEDICATION",
        "MEDICAL_CONDITION",
        "TEST_TREATMENT_PROCEDURE",
        "PROTECTED_HEALTH_INFORMATION",
    }
    SS_TRAIT_NAMES = {"SIGN", "SYMPTOM"}
    DX_TRAIT_NAMES = {"DIAGNOSIS"}
    NEG = {"NEGATION"}
    MED_TYPE = {"GENERIC_NAME", "BRAND_NAME"}

    def __init__(self, dict_entry: dict):
        self.entities = [ACMEntity(entry) for entry in dict_entry["Entities"]]

    @classmethod
    def from_fp(cls, fp):
        return cls(common.read_json(fp))

    def filter_entities_by_category(self, category: str):
        """
        Filter the entities list by the target category
        :param category:
        :return:
        """
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Given category '{category}' not found in possible categories: {self.CATEGORIES}"
            )
        return [entity for entity in self.entities if entity.category == category]

    def medications(self, min_prob_threshold=0.0) -> List[ACMEntity]:
        out = []
        for med in self.filter_entities_by_category("MEDICATION"):
            if med.score >= min_prob_threshold:
                if med.type not in self.MED_TYPE:
                    continue
                med.label = DrugLabel.med.name
                out.append(med)
        return out

    def medical_conditions(self, min_prob_threshold=0.0) -> List[ACMEntity]:
        out = []
        for problem in self.filter_entities_by_category("MEDICAL_CONDITION"):
            if problem.score >= min_prob_threshold:
                trait_names = {a["Name"] for a in problem.traits}
                if len(trait_names.intersection(self.NEG)) > 0:
                    continue
                if len(trait_names.intersection(self.SS_TRAIT_NAMES)) > 0:
                    problem.label = DiseaseSignSymptomLabel.signsymptom.name

                if len(trait_names.intersection(self.DX_TRAIT_NAMES)) > 0:
                    problem.label = DiseaseSignSymptomLabel.diagnosis.name
                if len(trait_names) == 0:
                    continue

                out.append(problem)

        return out

    def match_for_gold(
        self,
        entity_list,
        token_ranges,
        token_text_list,
        label_enum: LabelEnum,
        binary_classifier: bool = False,
    ) -> Dict[int, list]:
        """
        This function takes in featureset annotation and a ground truth annotation from BRAT
        and populate the correct labeling to featureset annotation to construct training matrix and label matrix

        NOTE: it's mostly copied from DataSource.match_for_gold, so should be consistent with that method

        :param entity_list: List[ACMEntity]
        :param token_ranges: List[List[int, int]]
        :param token_text_list: List[str]
        :param label_enum: LabelEnum
        :param binary_classifier:
        :return:
        """
        output = {}
        entry_idx = None

        for entity in entity_list:
            index = bisect.bisect_left(token_ranges, [entity.start_range, entity.end_range])

            # handle the case where the last annotation didn't capture the start of the token
            if index == len(token_ranges):
                index -= 1

            while index == 0 or (index < len(token_ranges) and entity.end_range > token_ranges[index - 1][0]):

                if (
                    index
                    and entity.start_range <= token_ranges[index - 1][1]
                    and (
                        token_text_list[index - 1] in entity.text
                        or sum([brat_text in token_text_list[index - 1] for brat_text in entity.text.split()]) > 0
                    )
                ):
                    # may need to change how featureset construct training label then
                    entry_idx = index - 1

                elif (
                    index < len(token_text_list)
                    and entity.end_range >= token_ranges[index][0]
                    and (
                        token_text_list[index] in entity.text
                        or sum([brat_text in token_text_list[index] for brat_text in entity.text.split()]) > 0
                    )
                ):
                    entry_idx = index

                else:
                    operations_logger.info(
                        f"the highlighted text, {entity.text} does not match the document"
                        f" text {token_text_list[min(index, len(token_text_list) - 1)]}"
                    )
                if entry_idx is not None:
                    if binary_classifier and label_enum[entity.label].value.column_index > 0:
                        output[entry_idx] = [0, 1]
                    elif entry_idx not in output:
                        temp_vect = [0] * len(label_enum)
                        temp_vect[label_enum.from_brat(entity.label.lower()).value.column_index] = 1
                        output[entry_idx] = temp_vect
                    else:
                        old_label = label_enum.get_from_int(np.argmax(output[entry_idx]))
                        if old_label.value.order == label_enum.from_brat(entity.label.lower()).value.order:
                            operations_logger.debug(
                                "multiple annotations for the same type on the same token"
                            )
                        # if multiple annotations on token due to improper tokenizations pick the  one with lowest order
                        else:
                            operations_logger.debug(
                                f"multiple annotations of different types within same label class"
                                f" on the same token {entity.text}, token idx : {entry_idx}"
                            )
                            if old_label.value.order > label_enum.from_brat(entity.label.lower()).value.order:
                                temp_vect = [0] * len(label_enum)
                                temp_vect[label_enum.from_brat(entity.label.lower()).value.column_index] = 1
                                output[entry_idx] = temp_vect
                index += 1
                entry_idx = None
        return output

    def acm_predicted_token_list(
        self, tokens: MachineAnnotation, label_enum, min_prob_threshold: float = 0.0
    ) -> List[int]:
        # get the brat result matching the label_enum passed in
        if label_enum == DiseaseSignSymptomLabel:
            entities = self.medical_conditions(min_prob_threshold)
        elif label_enum == DrugLabel:
            entities = self.medications(min_prob_threshold)
        else:
            raise ValueError(f"Unexpected label_enum given: {label_enum}")

        matched_vectors = self.match_for_gold(entities, tokens.range, tokens.tokens, label_enum=label_enum)

        test_results = [0] * len(tokens)
        for i in matched_vectors:
            label_vector = matched_vectors[i]
            test_results[i] = np.argmax(label_vector)
        return test_results


def get_matched_ann_acm_text_docs(text_dir, ann_dir, acm_dir):
    """
    Find which files match between the raw text, the ann, and the acm output
    :param text_dir:
    :param ann_dir:
    :param acm_dir:
    :return: List[Tuple[str, str, str]]
        (text, ann, acm) filenames
    """
    matched_file_paths = []
    acm_docs = common.get_file_list(acm_dir, ".out", True)
    for acm_doc in acm_docs:
        text_doc = acm_doc.replace(acm_dir, text_dir).replace(".txt.out", ".txt")
        ann_doc = acm_doc.replace(acm_dir, ann_dir).replace(".txt.out", ".ann")
        matched_file_paths.append([text_doc, ann_doc, acm_doc])

    # keep only the files that exist as txt and annotation (and acm out)
    valid_res = [matched_text for matched_text in matched_file_paths if os.path.isfile(matched_text[0])]
    valid_res_2 = [matched_ann for matched_ann in valid_res if os.path.isfile(matched_ann[1])]
    missed_ann_files = set([a[1] for a in valid_res]) - set([a[1] for a in valid_res_2])
    if missed_ann_files:
        print(f"Skipping {len(missed_ann_files)} matched files that do not point to valid annotation files")
    print(f"Extracting matches over {len(valid_res_2)} files")
    return valid_res_2


def test(text_dir, ann_dir, acm_dir, job_id, label_enum, min_prob=0.0):
    matched_files = get_matched_ann_acm_text_docs(text_dir, ann_dir, acm_dir)
    report = WeightedReport(label_enum=label_enum, concept_feature_mapping=None)
    for text_fp, ann_fp, acm_fp in matched_files:
        print(text_fp)
        text = common.read_text(text_fp)
        tokens = speech.tokenize(text)  # now a list of token dictionary
        machine_annotation = MachineAnnotation(tokenization_output=tokens)
        acm_out = ACMOutput.from_fp(acm_fp)
        acm_predicted_list = acm_out.acm_predicted_token_list(
            tokens=machine_annotation, label_enum=label_enum, min_prob_threshold=min_prob
        )
        cep_labeled_list = BiomedDataSource.token_true_label_list(
            ann_filepath=ann_fp, label_enum=label_enum, tokens=machine_annotation
        )
        report.add_document(
            expected_category=cep_labeled_list,
            predicted_results_cat=acm_predicted_list,
            predicted_results_prob=None,
            tokens=machine_annotation,
            filename=ann_fp,
            duplicate_token_idx={},
        )
    report.write(job_id=job_id)
    return report


if __name__ == "__main__":

    # cutoff threshold to include predicted labels
    prob_threshold = 0.5

    # ## PHI data
    parent_dir = "/opt/S3/prod-nlp-train1-us-east-1"
    text_dir = "tag_tog_text/validation/mdl-phi-cyan-us-west-2"

    # ## RWD diagnosis
    # ann_dir = "tag_tog_annotations/diagnosis_signsymptom_validation/2021-02-10_combined/combined_all/mdl-phi-cyan-us-west-2"
    # acm_dir = "/opt/S3/prod-nlp-train1-us-east-1/acm_annotations/acm_rwd_annotations"
    # test_output_dir = f"acm_testing/diagnosis_{int(prob_threshold*100)}"
    # label = DiseaseSignSymptomLabel

    # ## RWD medication
    ann_dir = "tag_tog_annotations/medication_allergy/2021-02-01/CEPuser/mdl-phi-cyan-us-west-2"
    acm_dir = "/opt/S3/prod-nlp-train1-us-east-1/acm_annotations/acm_rwd_annotations"
    test_output_dir = f"acm_testing/med_{int(prob_threshold*100)}"
    label = DrugLabel

    # ## I2B2 dataset report
    # parent_dir = "/opt/S3/biomed-data"
    # text_dir = "I2B2/2010 Relation Challenge/gold_20210218/test"
    # ann_dir = "annotations/diagnosis/I2B2/2010 Relation Challenge/gold_20210218/test"
    # acm_dir = "/opt/S3/prod-nlp-train1-us-east-1/acm_annotations/i2b2_2010_text_test_out"
    # test_output_dir = f"acm_testing/diagnosis_i2b2_{int(prob_threshold*100)}"
    # label = DiseaseSignSymptomLabel

    # --------
    # run the dataset to get test results

    os.makedirs(os.path.join(RESULTS_PATH, test_output_dir), exist_ok=True)
    report = test(
        os.path.join(parent_dir, text_dir),
        os.path.join(parent_dir, ann_dir),
        acm_dir,
        job_id=test_output_dir,
        label_enum=label,
        min_prob=prob_threshold,
    )
    print(report.scipy_report_str(report.precision_recall_report()))
