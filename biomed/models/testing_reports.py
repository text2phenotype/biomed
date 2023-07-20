from abc import abstractmethod, ABC
import csv
import os
import re
from typing import List, Tuple, Dict, Set, Type
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

from text2phenotype.common import common
from text2phenotype.common.demographics import Demographics
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import LabelEnum, FeatureType, DemographicEncounterLabel
from text2phenotype.constants.features.label_types import BinaryEnum

from biomed import RESULTS_PATH
from biomed.common.biomed_ouput import BiomedOutput
from biomed.common.model_test_helpers import cui_set_for_entry, check_cui_match
from biomed.demographic.demographics_manipulation import get_best_demographics, FetchedDemographics

MAX_NEARBY_TOKEN_RANGE = 2


def _get_number(s: str):
    """try to read str and convert to float; if can't be a float, return False"""
    try:
        return float(s)
    except ValueError:
        return False


def _safe_div(x, y):
    """avoid divide by zero; doesnt work for np.ndarray yet"""
    # try/except DivideByZero misses numpy type denominators being zero, gives RuntimeWarning instead
    return x/y if y != 0 else 0


class BaseTestingReport(ABC):
    @abstractmethod
    def add_document(
            self,
            expected_category: List[int],
            predicted_results_cat: [np.ndarray, List[int]],
            tokens: MachineAnnotation,
            duplicate_token_idx: Set[int],
            predicted_results_prob: np.ndarray = None,
            filename: str = None):
        """
        :param expected_category: A list of the integer predicted categories,
        these category map to labels via the get_from_column_index method on the Label Enum
        :param predicted_results_cat: a 1d numpy array or a list of the predicted categories
        :param tokens: the machine annotation for the document
        :param duplicate_token_idx: a set of integer token indexes that ought to be ignored by the testing report,
        generally coming from the use of the 'duplicate' label
        :param predicted_results_prob: a numpy array of dimensions [num_tokens, num_classes] where the sum of each row=1
         consitutes the probaility distribution for a given token
        :param filename: the filename for which result  are  being added
        :return: nothing, updates parameters
        """
        pass

    @abstractmethod
    def write(self, job_id):
        pass


class ConfusionPrecisionMisclassReport(BaseTestingReport, ABC):
    # ALL CHILD CLASSES WILL NEED TO DEFINE THIS IN ORDER TO WRITE OUT REPORTS
    REPORT_SUFFIX = ''

    def __init__(self, label_enum: LabelEnum, concept_feature_mapping: Dict[FeatureType, str] = None):
        self.label_enum = label_enum
        self.concept_feature_mapping = concept_feature_mapping or {}
        self.labels = list(range(len(label_enum)))
        self.target_names = label_enum._member_names_
        num_classes = len(label_enum)
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.misclassification_report = [self._get_misclassification_header()]

    @abstractmethod
    def get_valid_indexes(
            self, tokens: MachineAnnotation, duplicate_token_idx, predicted_results_cat, expected_cat) -> Set[int]:
        pass

    @staticmethod
    def _get_misclassification_header() -> List[str]:
        return ['actual', 'actual_prob', 'predicted', 'predicted_prob', 'token', 'file_name', 'start', 'end']

    def add_document(
            self,
            expected_category,
            predicted_results_cat: np.ndarray,
            tokens: MachineAnnotation = None,
            duplicate_token_idx: Set = None,
            predicted_results_prob: np.ndarray = None,
            filename: str = None):
        """
        Add a document scoring to update the report
        :param expected_category: the true labels for the document content
        :param predicted_results_cat: the predicted labels for the document
        :param tokens: MachineAnnotation
            Contains the document tokens and methods for determining token validity
        :param duplicate_token_idx:
            set of tokens that are duplicates
        :param predicted_results_prob:
        :param filename: str, filename of the document
        :return:
        """
        duplicate_token_idx = duplicate_token_idx or set()
        if tokens is not None:
            valid_token_indexes = self.get_valid_indexes(
                tokens=tokens, duplicate_token_idx=duplicate_token_idx,
                predicted_results_cat=predicted_results_cat, expected_cat=expected_category)

            valid_expected, valid_predicted = self.get_valid_comparison_tokens(
                valid_indexes=valid_token_indexes,
                expected_cat=expected_category,
                predicted_cat=predicted_results_cat,
            )
        else:
            # we dont have a MachineAnnotation document, so just use everything
            valid_token_indexes = np.arange(len(expected_category))
            valid_expected = expected_category
            valid_predicted = predicted_results_cat

        self.add_to_confusion_matrix(valid_expected, valid_predicted)
        if tokens is not None and valid_token_indexes is not None:
            self.add_document_misclass(
                expected_category=expected_category,
                predicted_results_cat=predicted_results_cat,
                predicted_results_prob=predicted_results_prob,
                valid_token_indexes=valid_token_indexes,
                tokens=tokens,
                filename=filename
            )

    def add_to_confusion_matrix(self, valid_expected, valid_predicted):
        if len(valid_expected) > 0:
            file_confusion_matrix = confusion_matrix(valid_expected, valid_predicted, labels=self.labels)
            self.confusion_matrix = self.confusion_matrix + file_confusion_matrix
        else:
            operations_logger.warning("No valid tokens for document found")

    def precision_recall_report(self, binary_report: bool = False) -> pd.DataFrame:
        if binary_report:
            confusion_mat = self.binary_confusion_matrix
            label_enum = BinaryEnum
        else:
            confusion_mat = self.confusion_matrix
            label_enum = self.label_enum

        output_list = []
        for label_type in label_enum:
            label_column = label_type.value.column_index
            true_positive = confusion_mat[label_column, label_column]
            false_negative = sum(confusion_mat[label_column, :]) - true_positive
            false_positive = sum(confusion_mat[:, label_column]) - true_positive
            precision = _safe_div(true_positive, (true_positive + false_positive))
            recall = _safe_div(true_positive, (true_positive + false_negative))
            f1 = 2 * _safe_div(precision * recall, (precision + recall))
            support = true_positive + false_negative
            label_row = {'label': label_type.value.persistent_label,
                         'precision': precision,
                         'recall': recall,
                         'f1-score': f1,
                         'support': int(support)}
            output_list.append(label_row)
        df = pd.DataFrame(output_list, columns=['label', 'precision', 'recall', 'f1-score', 'support'])
        df = df.fillna(0)
        # add average line
        if sum(df.support[1:]) <= 0:
            operations_logger.warning(
                f"NO support found for report type {self.REPORT_SUFFIX},  if this is the CUI report ensure "
                f"representation features are included in annotation file")
            average_precision = 0
            average_recall = 0
            average_f1 = 0
            average_support = 0
        else:
            average_precision = np.average(df.precision[1:], weights=df.support[1:])
            average_recall = np.average(df.recall[1:], weights=df.support[1:])
            average_f1 = np.average(df["f1-score"][1:], weights=df.support[1:])
            average_support = np.sum(df.support[1:])
        df = df.append({
            'label': 'avg/total',
            'precision': average_precision,
            'recall': average_recall,
            'f1-score': average_f1,
            'support': int(average_support)}, ignore_index=True)

        return df

    @property
    def binary_confusion_matrix(self):
        binary_confusion_matrix = np.zeros((2, 2), dtype=np.int64)
        binary_confusion_matrix[0, 0] = self.confusion_matrix[0, 0]
        binary_confusion_matrix[0, 1] = (self.confusion_matrix[0, 1:]).sum()
        binary_confusion_matrix[1, 0] = (self.confusion_matrix[1:, 0]).sum()
        binary_confusion_matrix[1, 1] = (self.confusion_matrix[1:, 1:]).sum()
        return binary_confusion_matrix

    @staticmethod
    def get_valid_comparison_tokens(
            valid_indexes: Set[int],
            expected_cat: List[int],
            predicted_cat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param valid_indexes: set of indexes that should be included in testing reeport
        :param expected_cat: list of integer categories  that are the expected results
        :param predicted_cat: numpy1 d array of predicted  categories
        :return: returns a tuple of 1d numpy arrays of integers that has length == len(valid_indexes),
        the  first of which is the  expected results and the second of which is  the predicted reults
        """
        valid_index_list = sorted(list(valid_indexes))
        valid_expected = np.array(expected_cat)[valid_index_list]
        valid_predicted = np.array(predicted_cat)[valid_index_list]

        return valid_expected, valid_predicted

    def add_document_misclass(
            self,
            expected_category,
            predicted_results_cat: np.ndarray,
            tokens: MachineAnnotation,
            valid_token_indexes,
            predicted_results_prob,
            filename=None):
        """
        :param expected_category: full set of expected category for all tokens of a document
        :param predicted_results_cat: full numpy array of predicted categorie
        :param tokens: 
        :param valid_token_indexes: 
        :param predicted_results_prob: numpy  array of  size [num_tokens, num_label_classes]
        :param filename: filename that will get printed as part of the misclass report
        :return: add all the relevant lines to  the document misclasification report
        """
        if predicted_results_prob is None:
            operations_logger.warning("NO PROBABILITIES RETURNED IGNORING")
        for token_idx in valid_token_indexes:
            if expected_category[token_idx] != predicted_results_cat[token_idx]:
                expected_cat = int(expected_category[token_idx])
                predicted_cat = int(predicted_results_cat[token_idx])
                if predicted_results_prob is not None:

                    expected_cat_prob = "{:0.4f}".format(predicted_results_prob[token_idx, int(expected_cat)])
                    predicted_cat_prob = "{:0.4f}".format(predicted_results_prob[token_idx, int(predicted_cat)])
                else:
                    expected_cat_prob = "0"
                    predicted_cat_prob = "1"

                span = tokens.range[token_idx]
                line = [str(expected_cat), expected_cat_prob, str(predicted_cat), predicted_cat_prob,
                        tokens.tokens[token_idx], filename, str(span[0]), str(span[1])]
                self.misclassification_report.append(line)

    def write_misclassification_report(self, job_id):
        report_file_name = f"misclassification_report_{self.REPORT_SUFFIX}".replace('.txt', '.csv')
        file_path = os.path.join(RESULTS_PATH, job_id, report_file_name)
        operations_logger.info(f"Writing confusion matrix report to {file_path}")
        common.write_csv(self.misclassification_report, file_path)
        return file_path

    def write_confusion_matrix(self, job_id):

        matrix_file_name = f"confusion_matrix_{self.REPORT_SUFFIX}"
        file_path = os.path.join(RESULTS_PATH, job_id, matrix_file_name)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        operations_logger.info(f"Writing confusion matrix report to {file_path}")
        common.write_text(str(self.confusion_matrix), file_path)
        return file_path

    @staticmethod
    def scipy_report_str(precision_recall_df):
        digits = 3
        target_names = precision_recall_df['label']

        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, digits)

        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
        report = head_fmt.format(u'', *headers, width=width)
        report += u'\n\n'
        row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'
        for idx, row in precision_recall_df.iterrows():
            if row['label'] == 'avg/total':
                report += u'\n'
            report += row_fmt.format(*row, width=width, digits=digits)

        return report

    def write_precision_recall_report(self, job_id):
        matrix_file_name = f"report_{self.REPORT_SUFFIX}"
        file_path = os.path.join(RESULTS_PATH, job_id, matrix_file_name)
        operations_logger.info(f"Writing confusion matrix report to {file_path}")
        precision_recall_df = self.precision_recall_report()
        common.write_text(self.scipy_report_str(precision_recall_df=precision_recall_df), file_path)
        return file_path

    def write_binary_precision_recall_report(self, job_id):
        matrix_file_name = f"report_binary_{self.REPORT_SUFFIX}"
        file_path = os.path.join(RESULTS_PATH, job_id, matrix_file_name)
        operations_logger.info(f"Writing confusion matrix report to {file_path}")
        precision_recall_df = self.precision_recall_report(binary_report=True)
        common.write_text(self.scipy_report_str(precision_recall_df=precision_recall_df), file_path)
        return file_path

    def write(self, job_id: str):
        self.write_confusion_matrix(job_id=job_id)
        self.write_precision_recall_report(job_id=job_id)
        self.write_misclassification_report(job_id=job_id)
        self.write_binary_precision_recall_report(job_id=job_id)

    @staticmethod
    def parse_classification_text_to_df(report: str, label_class_enum: Type[Enum] = None) -> pd.DataFrame:
        """
        Parse report string from ConfusionPrecisionMisclassReport.scipy_report_str

        :param report: str
        :param label_class_enum:
        :return: pd.DataFrame
        """
        lines = report.splitlines()
        cols = (lines[0].split())
        if len(lines) < 2:
            raise ValueError(f"Bad report content, found less than 2 lines of text: {lines}")
        dat = np.zeros((len(lines) - 2, len(cols)))
        indexes = []
        for l in range(2, len(lines)):
            split_text = re.split(r'\s{2,}', lines[l])
            if len(split_text) == len(cols) + 1:
                dat[l - 2,] = split_text[1:]
                num_class = _get_number(split_text[0])
                if label_class_enum and (num_class or num_class == 0):
                    indexes.append(label_class_enum.get_from_int(num_class).name)
                else:
                    indexes.append(split_text[0].strip())
            elif len(split_text) == len(cols) + 2:
                dat[l - 2,] = split_text[2:]
                num_class = _get_number(split_text[1])
                if label_class_enum and num_class is not False:
                    indexes.append(label_class_enum.get_from_int(num_class).name)
                else:
                    indexes.append(split_text[1].strip())
            else:
                indexes.append(0)
        operations_logger.info(f'Indexes: {indexes}')
        df = pd.DataFrame(dat, columns=cols, index=indexes).reset_index()
        df.columns = ['class_label', 'precision', 'recall', 'f1-score', 'support']
        df.support = df.support.astype(int)
        operations_logger.info(f"df_columns: {df.columns}")
        return df[(df['class_label'] != 0) | (df.support != 0)]

    @staticmethod
    def parse_confusion_matrix_text_to_df(report: str, cols: list = None):
        """
        Parse the confusion matrix string from ConfusionPrecisionMisclassReport.write_confusion_matrix

        :param report: str
        :param cols: List[Str]: column names
        :return: pd.DataFrame
        """
        report = report.replace('\n', ' ').split('] ')
        matrix = []
        for row in report:
            row_list = row.replace('[', ' ').replace(']', '').split()
            matrix.append([int(r) for r in row_list])
        df = pd.DataFrame(matrix, columns=cols, index=cols)
        return df


class WeightedReport(ConfusionPrecisionMisclassReport):
    REPORT_SUFFIX = 'weighted.txt'

    @staticmethod
    def get_valid_indexes(tokens, duplicate_token_idx, **kwargs) -> set:
        return set(tokens.valid_tokens(duplicate_tokens=duplicate_token_idx)[1])


class RemovingAdjacentConfusion(ConfusionPrecisionMisclassReport):
    REPORT_SUFFIX = 'minus_adjacent_wrong.txt'

    @staticmethod
    def get_valid_indexes(tokens: MachineAnnotation, duplicate_token_idx, predicted_results_cat, expected_cat,
                          **kwargs) -> Set[int]:
        valid_weighted = set(tokens.valid_tokens(duplicate_tokens=duplicate_token_idx)[1])
        for i in sorted(list(valid_weighted)):
            # if we mess up  a label but part of the concept annotated was correct, continue
            if predicted_results_cat[i] != expected_cat[i]:
                for j in range(i - MAX_NEARBY_TOKEN_RANGE, i + MAX_NEARBY_TOKEN_RANGE + 1):
                    valid_index = 0 <= j < len(predicted_results_cat)
                    if valid_index and predicted_results_cat[j] == expected_cat[j] and predicted_results_cat[j] != 0:
                        valid_weighted.remove(i)
                        break
        return valid_weighted


class MinusPartialAnnotation(ConfusionPrecisionMisclassReport):
    REPORT_SUFFIX = 'minus_partial.txt'

    def get_valid_indexes(self, tokens, duplicate_token_idx, predicted_results_cat, expected_cat,
                          **kwargs) -> Set[int]:
        # get all indices of valid tokens but exclude indices that correspond to mislabelling where the missed tokens
        # were part of a concept code where we predicted another part of the concept or where the missed token ha no cui
        # and is adjacent to a correctly labeled token with a cui on the assumption that the missed token is
        # then an adjective/modifier of the concept

        valid_weighted = set(tokens.valid_tokens(duplicate_tokens=duplicate_token_idx)[1])
        for i in sorted(list(valid_weighted)):
            # if we mess up  a label but part of the concept annotated was correct, continue
            current_cuis = set(cui_set_for_entry(i, self.concept_feature_mapping, tokens))
            if predicted_results_cat[i] != expected_cat[i]:
                for j in range(i - MAX_NEARBY_TOKEN_RANGE, i + MAX_NEARBY_TOKEN_RANGE + 1):
                    match_cuis = set(cui_set_for_entry(j, self.concept_feature_mapping, tokens))
                    if 0 <= j < len(predicted_results_cat):
                        # same concept gets extracted no matter which piece is highlighted
                        # cover case where annotator labeled shortness of breath, model only caught shortness
                        adjacent_cui_match = (check_cui_match(match_cuis, current_cuis) and
                                              predicted_results_cat[j] == expected_cat[j] and
                                              predicted_results_cat[j] != 0)
                        offset_match = False

                        # cover case where annotator labels shortness, model labels breath as a symptom (or vis versa)
                        if predicted_results_cat[i] > 0:
                            offset_match = (check_cui_match(match_cuis, current_cuis) and
                                            predicted_results_cat[i] == expected_cat[j])

                        if expected_cat[i] > 0 and not offset_match:
                            offset_match = (check_cui_match(match_cuis, current_cuis) and
                                            predicted_results_cat[j] == expected_cat[i])

                        if adjacent_cui_match or offset_match:
                            valid_weighted.remove(i)
                            break
        return valid_weighted


class CuiReport(ConfusionPrecisionMisclassReport):
    REPORT_SUFFIX = 'cui_overlap.txt'

    def get_valid_indexes(self, tokens, **kwargs):
        return set(range(len(tokens)))

    @staticmethod
    def _get_misclassification_header() -> List[str]:
        return ['actual', 'predicted', 'cui', 'file_name']

    def add_document_misclass(
            self,
            expected_category,
            predicted_results_cat: np.ndarray,
            tokens: MachineAnnotation,
            valid_token_indexes: Set[int],
            predicted_results_prob=None,
            filename=None):

        for token_idx in valid_token_indexes:
            if expected_category[token_idx] != predicted_results_cat[token_idx]:
                expected_cat = expected_category[token_idx]
                predicted_cat = predicted_results_cat[token_idx]
                line = [str(expected_cat), str(predicted_cat), tokens.tokens[token_idx], filename]
                self.misclassification_report.append(line)


class DemographicsReport(BaseTestingReport):
    def __init__(self):
        self.label_enum = DemographicEncounterLabel
        self.dem_comparison_results = dict()

    def transform_to_biomed_out(self, tokens: MachineAnnotation, category) -> List[BiomedOutput]:
        out_list = []
        for idx in range(len(category)):
            if int(category[idx]) != 0:
                out_list.append(
                    BiomedOutput(
                        label=self.label_enum.get_from_int(int(category[idx])).value.persistent_label,
                        text=tokens.tokens[idx],
                        range=tokens.range[idx],
                        lstm_prob=1.0))

        return out_list

    def add_document(
            self,
            expected_category,
            predicted_results_cat: np.ndarray,
            tokens: MachineAnnotation,
            duplicate_token_idx,
            predicted_results_prob=None,
            filename=None):

        # transform category into list of biomed outputs (used by demographics transform)
        predicted_biomed_out = self.transform_to_biomed_out(tokens=tokens, category=predicted_results_cat)
        expected_biomed_out = self.transform_to_biomed_out(tokens=tokens, category=expected_category)

        # get demographic dictionary outputs
        predicted_dem_dict = FetchedDemographics(demographics_list=predicted_biomed_out)
        true_dem_dict = FetchedDemographics(demographics_list=expected_biomed_out)

        # get best demographic
        best_predicted_dem_obj = get_best_demographics(predicted_dem_dict)
        true_dem_obj = get_best_demographics(true_dem_dict)

        # add the comparisons to the obj property
        self.compare_demographics_objects(
            predicted_demographics=best_predicted_dem_obj,
            true_demographics=true_dem_obj)

    def compare_demographics_objects(self, true_demographics: Demographics, predicted_demographics: Demographics):
        # type by type comparison that outputs the intersection/len(expected) ratio
        expected_dem_dict = true_demographics.to_dict()
        predicted_dem_dict = predicted_demographics.to_dict()

        for k in expected_dem_dict.keys():
            expected_vals = set([i[0] for i in expected_dem_dict[k]])
            predicted_vals = set([i[0] for i in predicted_dem_dict[k]])

            true_positives = len(expected_vals.intersection(predicted_vals))
            false_positive = len(predicted_vals.difference(expected_vals))
            false_negatives = len(expected_vals.difference(predicted_vals))

            if k in self.dem_comparison_results:
                self.dem_comparison_results[k].append([true_positives, false_negatives, false_positive])

            else:
                self.dem_comparison_results[k] = [[true_positives, false_negatives, false_positive]]

    def write(self, job_id: str):
        file_path = os.path.join(RESULTS_PATH, job_id, 'report_demographic_post_processing.csv')
        self.demographics_report_df().to_csv(file_path)

    def demographics_report_df(self) -> pd.DataFrame:
        document_demographics_comparison = pd.DataFrame(
            self.process_dem_comparison_results()
        ).transpose().fillna(0).round(4)

        return document_demographics_comparison[['precision', 'recall', 'f1', 'support']]

    def process_dem_comparison_results(self):
        out_dict = dict()
        for k, v in self.dem_comparison_results.items():
            totals = np.array(v).sum(axis=0)
            precision = _safe_div(totals[0], (totals[0] + totals[2]))
            recall = _safe_div(totals[0], (totals[0] + totals[1]))
            support = totals[0] + totals[1]
            out_dict[k] = {'precision': precision,
                           'recall': recall,
                           'f1': 2 * _safe_div(precision * recall, (precision + recall)),
                           'support': support}
        return out_dict


class FullReport(BaseTestingReport, ABC):
    # this outputs ALL info (tokens, labels, probabilities, predictions)

    def __init__(self, label_enum: LabelEnum):
        self.label_enum = label_enum
        self.labels = list(range(len(label_enum)))
        self.target_names = label_enum._member_names_
        # list stores document-level info
        self.doc_info = {}

    def add_document(
            self,
            expected_category,
            predicted_results_cat: np.ndarray,
            tokens: MachineAnnotation,
            duplicate_token_idx,
            raw_probs,
            predicted_results_prob=None,
            filename=None):

        doc = {
            'tokens': tokens.tokens,
            'labels': expected_category,
            'predicted': predicted_results_cat,
            'prob': predicted_results_prob,
            'raw_prob': raw_probs
        }
        self.doc_info[filename] = doc

    def write(self, job_id: str):
        # writing this as gzipped pickle with pandas
        # TODO: Is this likely to cause issues?
        full_file_name = 'full_info.pkl.gz'
        file_path = os.path.join(RESULTS_PATH, job_id, full_file_name)
        pd.to_pickle(self.doc_info, file_path)
