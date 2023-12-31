import math
import tensorflow.keras as keras
import numpy
from typing import List, Set

from biomed.common.matrix_construction import indexes_for_token
from biomed.constants.constants import EXCLUDED_LABELS
from text2phenotype.common.featureset_annotations import Vectorization

from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features.feature_type import FeatureType
from text2phenotype.constants.features.label_types import LabelEnum

from biomed.biomed_env import BiomedEnv
from biomed.common import matrix_construction


class Mat3dGenerator(keras.utils.Sequence):
    # https://keras.io/utils/
    def __init__(self, vectors: Vectorization, num_tokens: int, max_window_size: int,
                 features: [List[FeatureType], Set[FeatureType]],
                 tid: str = None, **kwargs):

        self.batch_size = kwargs.get('batch_size', BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value)
        self.vectors = vectors
        self.num_tokens = num_tokens
        self.max_window_size = max_window_size
        self.min_window_size = kwargs.get('min_window_size', max_window_size)
        self.features = [feature for feature in features if feature not in EXCLUDED_LABELS]
        self.tid = tid
        self.start_idx = kwargs.get('start_idx', 0)
        self.feature_col_mapping = matrix_construction.create_feature_col_mapping(vectors=vectors,
                                                                                  features=self.features)
        self._valid_indices = None
        self.duplicate_tokens: set = kwargs.get('duplicate_tokens', {})
        col_count = matrix_construction.get_feature_size(self.features, vectors)
        self.default_matrix = matrix_construction.default_matrix(feature_col_mapping=self.feature_col_mapping,
                                                                 default_vectors=vectors.defaults,
                                                                 max_window_size=self.max_window_size,
                                                                 batch_size=self.batch_size,
                                                                 col_count=col_count)

    def __len__(self):
        return max(math.ceil((len(self.valid_indices) - self.min_window_size + 1 - self.start_idx) / self.batch_size),
                   1)

    def __getitem__(self, index) -> numpy.ndarray:
        start_index = self.start_idx + index * self.batch_size
        batch_token_indices = self.token_index_for_batch(index)

        indexing_help = len(self.valid_indices)-self.min_window_size
        # check that we are within expected x dimension or if there are < window_size tokens run first entry
        if start_index <= indexing_help or start_index == 0:
            if start_index + self.batch_size >= indexing_help:
                # if the expected output shape[0] for a batch is less than the batch_size,
                # create the matrix of expected size. This will only happen for index = len(predict_generator)-1
                end_index = max(indexing_help - start_index + 1, 1)
                feature_mat_3d = self.default_matrix[0:end_index, :, :].copy()
            else:
                feature_mat_3d = self.default_matrix.copy()
            self.update_matrix(feature_mat_3d, batch_token_indices)
            operations_logger.debug(f'Working on batch # {index+1}/{self.__len__()} ')
            # returns object min(batch_size, num_tokens-start-windowsize) x window_size x num vectors cols
            return feature_mat_3d

    def update_matrix(self, feature_mat_3d, token_indices):
        matrix_construction.update_default_matrix(mat_default=feature_mat_3d,
                                                  vectors=self.vectors,
                                                  feature_col_mapping=self.feature_col_mapping,
                                                  max_window_size=self.max_window_size,
                                                  token_indices=token_indices,
                                                  tid=self.tid)

    @property
    def valid_indices(self):
        if not self._valid_indices:
            self._valid_indices = sorted(list(set(range(self.num_tokens)).difference(self.duplicate_tokens)))
            operations_logger.debug(f'Number of valid indices: {len(self._valid_indices)}')
        return self._valid_indices

    def token_index_for_batch(self, batch):
        start_idx = batch * self.batch_size
        end_idx = (batch + 1) * self.batch_size + self.max_window_size
        return self.valid_indices[start_idx: end_idx]


class LabelMat3d(Mat3dGenerator):
    def __init__(self, vectors: dict, label_enum: LabelEnum, num_tokens: int, max_window_size: int, tid: str = None,
                 binary_classifier: bool = False,
                 **kwargs):
        self.batch_size = kwargs.get('batch_size', BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value)
        self.num_tokens = num_tokens
        self.max_window_size = max_window_size
        self.min_window_size = kwargs.get('min_window_size', max_window_size)
        self.tid = tid
        self.start_idx = kwargs.get('start_idx', 0)
        self.binary_classifier: bool = binary_classifier
        self.vectors = vectors
        self._valid_indices = None
        self.duplicate_tokens: set = kwargs.get('duplicate_tokens', {})

        if not self.binary_classifier:
            default_mat = numpy.zeros((self.batch_size, self.max_window_size, len(label_enum)))
        else:
            default_mat = numpy.zeros((self.batch_size, self.max_window_size, 2))
        default_mat[:, :, 0] = 1
        self.default_matrix = default_mat

    def update_matrix(self, feature_mat_3d, token_indices):
        for key, vector in self.vectors.items():
            if key in token_indices:
                # get relative index for a given batch, and get numpy version of vector
                rel_idx = token_indices.index(key)
                value_np = numpy.array(vector, dtype=numpy.float16)
                # use numpy basic indexing of the indexes for a given token
                x_dim, y_dim = indexes_for_token(rel_idx, window_size=self.max_window_size,
                                                 max_x_size=feature_mat_3d.shape[0])
                feature_mat_3d[tuple(x_dim), tuple(y_dim), :] = value_np
