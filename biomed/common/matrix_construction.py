from typing import (
    Dict,
    List,
    Tuple)

import numpy
from text2phenotype.common.featureset_annotations import Vectorization, DefaultVectors

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType


@text2phenotype_capture_span()
def get_3d_matrix(vectors: Vectorization,
                  num_tokens: int,
                  max_window_size: int,
                  feature_types: List[FeatureType],
                  **kwargs) -> Tuple[numpy.ndarray, Dict[FeatureType, range]]:
    """
    :return: 3d numpy array [number of tokens x max window size x features] and the feature column map of feats to cols
    """
    valid_indices = kwargs.get('valid_indices', list(range(num_tokens)))
    operations_logger.debug('Begin Transforming Vectors to 3D Matrix', tid=kwargs.get('tid'))
    # create mapping of features to column indices
    feature_col_mapping = create_feature_col_mapping(feature_types, vectors)
    # get z dimension
    col_count = get_feature_size(feature_types, vectors)
    # create the default 3d matrix (num tokens x window size x feature column count)
    feature_matrix_3d = default_matrix(feature_col_mapping, vectors.defaults, max_window_size, num_tokens, col_count)
    # update the mutable 3d matrix numpy.ndarray in place. Does not change dimensions
    update_default_matrix(feature_matrix_3d,
                          vectors=vectors,
                          feature_col_mapping=feature_col_mapping,
                          max_window_size=max_window_size,
                          token_indices=valid_indices, tid=kwargs.get('tid'))

    operations_logger.debug(f'Returning 3D matrix for {feature_matrix_3d.shape[0]} tokens', tid=kwargs.get('tid'))
    return feature_matrix_3d, feature_col_mapping


@text2phenotype_capture_span()
def update_default_matrix(mat_default: numpy.ndarray,
                          vectors: Vectorization,
                          feature_col_mapping: Dict[FeatureType, range],
                          max_window_size: int,
                          token_indices: List[int],
                          **kwargs):
    batch_size = mat_default.shape[0]
    # look at all tokens that would be included in matrix with dimension 0 = num_tokens - window_size
    for feature_type in feature_col_mapping:
        col_index_start = feature_col_mapping[feature_type][0]
        col_index_end = feature_col_mapping[feature_type][-1] + 1
        if vectors.check_feature(feature_type):
            # loop through vectors (non-default results),  the int(key) is the token index, the value is the vector
            for str_key, value in vectors[feature_type].items():
                # restrict the ranges you look at to be those of interest for a given batch
                key = int(str_key)
                if key in token_indices:
                    # get relative index for a given batch, and get numpy version of vector (location of relative_idx)
                    rel_idx = token_indices.index(key)
                    value_np = numpy.array(value, dtype=numpy.float16)
                    # use numpy basic indexing of the indexes for a given token
                    x_dim,  y_dim = indexes_for_token(rel_idx, window_size=max_window_size, max_x_size=batch_size)
                    mat_default[tuple(x_dim), tuple(y_dim), col_index_start:col_index_end] = value_np
        else:
            operations_logger.debug(f'Feature {feature_type} not included in the vectorization', tid=kwargs.get('tid'))


@text2phenotype_capture_span()
def default_matrix(feature_col_mapping: Dict[FeatureType, range],
                   default_vectors: DefaultVectors,
                   max_window_size: int,
                   batch_size: int,
                   col_count: int) -> numpy.ndarray:
    # creates the default matrix from the feature column mapping and the default vectors
    # default vectors is a dictionary of str: list where the str is FeatureType.name and
    # the list is the Feature.default_vector properety. This gets created and appended to the feature
    # service vectorization result
    feature_matrix_3d = numpy.zeros((batch_size, max_window_size, col_count), dtype=numpy.float16)
    for feature in feature_col_mapping.keys():
        if default_vectors[feature] is not None:
            default_vector = default_vectors[feature]
            # if the default shouldnt all be zeros, fill in the default
            if sum(default_vector) != 0:
                col_index_start = feature_col_mapping[feature][0]
                col_index_end = feature_col_mapping[feature][-1] + 1
                feature_matrix_3d[:, :, col_index_start: col_index_end] = numpy.array(default_vector,
                                                                                      dtype=numpy.float16)
    # output is batch_size x window_size x col_count
    return feature_matrix_3d


def create_feature_col_mapping(features: List[FeatureType], vectors: Vectorization) -> Dict[FeatureType, range]:
    # given a list of features output a dictionary Feature.name: range(cols) in a 3d matrix where feature vectors go
    feature_col_mapping = {}
    col_index = 0
    # always sort the features so that they are always in numerical order based on their FeatureType enum value
    for feature in sorted(features):
        default_vector = vectors.defaults[feature]
        if default_vector is None:
            raise TypeError(f'Default vector not found for feature: {feature}')
        vector_len = len(default_vector)
        feature_col_mapping[feature] = range(col_index, vector_len + col_index)
        col_index = vector_len + col_index
    return feature_col_mapping


def get_feature_size(features: List[FeatureType], vectors: Vectorization) -> int:
    # used to get z dimension for the 3d matrix
    return vectors.defaults.total_len(features)


def indexes_for_token(token_idx: int, window_size: int, max_x_size: int) -> Tuple[list, list]:
    """
    Given a token index, provide the coordinates in all windows in which it has a prediction
    the first token only has one prediction, the second has two, etc
    this assumes a single-token stride length
    """
    token_indexes = [token_idx - i for i in range(window_size) if i <= token_idx and token_idx - i < max_x_size]
    window_indexes = [i for i in range(window_size) if i <= token_idx and token_idx - i < max_x_size]
    return token_indexes, window_indexes
