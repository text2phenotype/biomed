from typing import Set,  List
import numpy as np

from text2phenotype.constants.features import LabelEnum
from biomed.common.biomed_ouput import BiomedOutput


class PredictResults:
    """Class to standardize what the output of any predict method ought to be, this gets used in all downstream tasks"""

    def __init__(
            self,
            predicted_probs: np.ndarray,
            predicted_cat: np.ndarray = None,
            tokens: List[str] = None,
            ranges: list = None,
            raw_probs: np.ndarray = None):
        """
        :param predicted_probs: [num_tokens, label_enum_size] prediction vectors for all tokens
        :param predicted_cat: [num_tokens] values are integers that map to label_enum column_indexes,
        if not specified will just use an argmax of the predicted_probs matrix
        :param tokens: List of token strings
        :param ranges: List of token ranges
        :param raw_probs: [num_models, num_tokens, label_enum_size] raw probability vectors, used in FullReport
        """
        self.tokens = tokens
        self.ranges = ranges
        self.predicted_probs = predicted_probs  # tokens x num classes, numbers = softmaxed output scores
        self.raw_probs = raw_probs
        self._predicted_category = predicted_cat
        self._ensemble_result = None

    def to_json(self):
        res = self.__dict__.copy()
        return res

    @property
    def predicted_category(self) -> np.ndarray:
        """
        :return: np array of dimension # tokens x 1 where value at each index is the column_index of the predicted
        category
        """
        if self._predicted_category is None:
            self._predicted_category = np.argmax(self.predicted_probs, axis=1)
        return self._predicted_category

    def token_dict_list(self, label_enum: LabelEnum, filter_nas: bool = True, bio_out_class = BiomedOutput) -> List[BiomedOutput]:
        if not self._ensemble_result and self.predicted_category is not None and self.tokens is not None and \
                self.ranges is not None and self.predicted_probs is not None:
            self._ensemble_result = []
            for i in range(len(self.tokens)):
                if self.predicted_category[i] != 0 or not filter_nas:
                    self._ensemble_result.append(
                        bio_out_class(
                            label=label_enum.get_from_int(
                                self.predicted_category[i]).value.persistent_label,
                            text=self.tokens[i],
                            range=self.ranges[i],
                            lstm_prob=(round(self.predicted_probs[i][int(self.predicted_category[i])], 3))
                        ))
        return self._ensemble_result


class UncertaintyResults:
    def __init__(
            self, average_entropy=None, uncertain_count=None, narrow_band_ratio=None,
            uncertain_tokens=None, narrow_band_width=None, uncertain_token_count=None):
        # NOTE: anything around uncertainty has not been maintatined in a year, use at your own peril
        # relevant when predicting with uncertainty
        self.average_entropy = average_entropy
        self.uncertain_count = uncertain_count
        self.narrow_band_ratio = narrow_band_ratio
        self.uncertain_tokens: Set[int] = uncertain_tokens
        self.narrow_band_width = narrow_band_width
        self.uncertain_token_count = uncertain_token_count

    def write_uncertainty_line(self, file, url_path):
        res = {'file_path': file,
               'url_path': url_path,
               'uncertain_token_count': self.uncertain_token_count,
               'narrow_band_ratio': self.narrow_band_ratio,
               'average_entropy': self.average_entropy,
               'entropy*uncertain_count': self.uncertain_token_count * self.average_entropy
               }
        return res
