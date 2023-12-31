from abc import ABC, abstractmethod
from itertools import chain, islice
from math import floor, ceil
import random
import time
from typing import Dict, List, Iterable

import numpy as np
import tensorflow as tf

from biomed.biomed_env import BiomedEnv
from biomed.common.mat_3d_generator import LabelMat3d, Mat3dGenerator
from biomed.common.voting import construct_sample_weight, sparse_sample_weight
from biomed.constants.constants import EXCLUDED_LABELS
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common import common
from text2phenotype.common.featureset_annotations import Vectorization, MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.features.label_types import LabelEnum
from text2phenotype.common.vector_cache import VectorCacheJson, VectorCache


def chunk_generator(iterable: Iterable, chunk_size: int):
    """
    From the given iterable, return chunks of the iterable of size `chunk_size` elements
    This method can receive a generator and split it into smaller generators
    of size chunk_size without consuming the original generator

    Borrowed from: https://stackoverflow.com/a/24527424
     - do not pad the chunks: if the number of remaining elements is less than the chunk size,
        the last chunk must be smaller.
     - do not walk the generator beforehand: computing the elements is expensive, and it must
        only be done by the consuming function, not by the chunker
    - which means, of course: do not accumulate in memory (no lists)

    Example:
    next() returns the first chunk from the generator as a generator, list() consumes the returned generator
    >>> tokens = ["foo", "bar", "baz", "bif", "bam"]
    >>> list(next(chunk_generator(tokens, 2)))
    ["foo", "bar"]

    :param iterable: any Iterable or Generator
    :param chunk_size: int, the number of elements to return without lookahead.
    :return: chunk (a generator itself) of initial iterable
    """
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, chunk_size - 1))


class BERTGenerator(ABC):
    """Base class for generating data formatted for BERT"""

    def __init__(
        self,
        fs_files: List[str],
        vector_cache: VectorCache,
        model_input_fields: List[str],
        window_size: int,
        class_weight: dict = None,
        shuffle_files: bool = False,
        seed: int = None,
    ):
        """
        :param fs_files: list of keys into the vector_cache, often the Feature Service file paths
        :param vector_cache: The cache registry containing the vectorized machine annotation files.
        :param model_input_fields: A list of keys in vectorcache that should be passed to the model
        :param window_size: Length of windows, required for tf.data.Dataset
        :param class_weight: dict keyed by class_id and class weight value
            {'1': 10, '2': 10}
        :param shuffle_files: Flag to indicate if files should be shuffled prior to data generation.
        :param seed: int, fixes file shuffle order
        """

        self.fs_files = fs_files
        self.vector_cache = vector_cache
        self.model_input_fields = model_input_fields
        self.window_size = window_size
        self.shuffle_files = shuffle_files
        self.seed = seed
        if class_weight is not None:
            operations_logger.warning(
                "Trying to use class weights, but BertTokenClassificationLoss does not support"
                f"this feature. class_weight={class_weight}")
        self.class_weight = None
        # self.class_weight = class_weight
        # create the generator last
        self.generator = self.make_generator()

    def gen_func(self):
        # generator for feeding to tf.data.Dataset
        if self.shuffle_files:
            # Random object sets seed for local use rather than global
            random.Random(self.seed).shuffle(self.fs_files)

        for f in self.fs_files:
            data_dict = self.vector_cache[f]
            for i, labels in enumerate(data_dict["encoded_labels"]):
                x = {input_key: data_dict[input_key][i] for input_key in self.model_input_fields}

                if self.class_weight:
                    # operations_logger.info(self.class_weight)
                    # operations_logger.debug("sample weight crafted")
                    # operations_logger.info(f"labels: {labels}")
                    # nd_labels = np.array(labels)
                    # if nd_labels.ndim == 1:
                    #     nd_labels = np.expand_dims(nd_labels, axis=0)
                    crafted_sample_weight = sparse_sample_weight(self.class_weight, labels)
                    # operations_logger.info(f"x={len(x['input_ids'])}, y={len(labels)}, s={len(crafted_sample_weight)}")
                    yield x, labels, crafted_sample_weight
                else:
                    yield x, labels

    def make_generator(self):
        """Create a tf.data.Dataset generator"""
        output_types = (
            {"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int32)
        output_shapes = (
            {
                "input_ids": tf.TensorShape([self.window_size, ]),
                "attention_mask": tf.TensorShape([self.window_size, ]),
            },
            tf.TensorShape([self.window_size, ]),
        )
        if self.class_weight:
            # if we have a class weight, we need to tell the system to expect
            # an additional element on the tuple from the generator
            output_types = output_types + (tf.float32, )
            output_shapes = output_shapes + (tf.TensorShape([self.window_size, ]), )
            print(f"Adding sample_weight to yield: {output_shapes}")
        return tf.data.Dataset.from_generator(
            lambda: self.gen_func(),
            output_types=output_types,
            output_shapes=output_shapes,
        )


class LSTMDataGenerator(ABC):
    """Base class for generating data formatted for a BiLSTM."""

    def __init__(
        self,
        ann_files: List[str],
        ma_files: List[str],
        vector_cache: VectorCacheJson,
        feature_col_size: int,
        data_source: BiomedDataSource,
        label_enum: LabelEnum,
        job_metadata: JobMetadata,
        model_metadata: ModelMetadata,
    ):
        """
        Ctor.
        :param ann_files: The list of human annotation files.
        :param ma_files: The corresponding list of machine annotation files.
        :param vector_cache: The cache registry containing the vectorized machine annotation files.
        :param feature_col_size: The feature vector length.
        """
        self.__ann_files = ann_files
        self.__ma_files = ma_files
        self._vector_cache = vector_cache
        self.__n_classes = len(label_enum) if not model_metadata.binary_classifier else 2
        self.__feature_col_size = feature_col_size
        self.__data_source = data_source
        self.__label_enum = label_enum
        self._job_metadata = job_metadata
        self.__model_metadata = model_metadata
        self.__vectorization_complete = False

    def vectorize(self) -> int:
        """
        Vectorize a collection of machine annotations.
        :return: The total number of tokens vectorized.
        """
        max_failure_pct = self._get_max_fail_pct()

        fs_client = FeatureServiceClient()

        max_failures = ceil(max_failure_pct * len(self.__ma_files))
        num_failures = 0
        token_count = 0
        for ma_file in self.__ma_files:
            tokens = self.__read_annotation_file(ma_file)
            try:
                vectors: Vectorization = self._get_vectors(tokens, fs_client)
                self._vector_cache[ma_file] = vectors.to_dict()

                token_count += len(tokens)
            except:
                num_failures += 1
                if num_failures >= max_failures:
                    raise ValueError(
                        f"{num_failures} files failed to be vectorized "
                        f"({max_failures} max allowed), destroying job"
                    )

                time.sleep(15)  # give FS a chance to reset itself

        self.__vectorization_complete = True

        return token_count

    def next_batch(self):
        if not self.__vectorization_complete:
            self.vectorize()

        max_tokens = self._get_buffer_size()

        x_train_np = np.zeros((max_tokens, self.__model_metadata.window_size, self.__feature_col_size))
        y_train_np = np.zeros((max_tokens, self.__model_metadata.window_size, self.__n_classes))

        file_iter_count = 1
        while True:
            operations_logger.info(f"Processing files for iteration {file_iter_count}...")
            file_indexes = self._get_file_indices()
            file_iter_count += 1

            buffer_size = 0
            for file_index in file_indexes:
                ma_file = self.__ma_files[file_index]
                ann_file = self.__ann_files[file_index]

                operations_logger.debug(f"Processing file {ma_file}...")
                if not self._vector_cache.exists(ma_file):
                    continue

                mat_3d_gen_x, mat_3d_gen_y = self.__get_file_matrices(ma_file, ann_file, max_tokens)
                for mat_index in range(len(mat_3d_gen_x)):
                    curr_x = mat_3d_gen_x[mat_index]
                    curr_y = mat_3d_gen_y[mat_index]
                    n_rows = curr_x.shape[0]

                    curr_offset = 0
                    while n_rows:
                        n_for_curr_batch = min(n_rows, max_tokens - buffer_size)
                        batch_end = curr_offset + n_for_curr_batch
                        train_end = buffer_size + n_for_curr_batch

                        x_train_np[buffer_size:train_end] = self.__slice_3d_mat(
                            curr_x[curr_offset:batch_end], mat_3d_gen_x.feature_col_mapping
                        )
                        y_train_np[buffer_size:train_end] = curr_y[curr_offset:batch_end]

                        buffer_size += n_for_curr_batch
                        curr_offset += n_for_curr_batch
                        if buffer_size == max_tokens:
                            for result in self._prep_batch(x_train_np, y_train_np, buffer_size):
                                yield result

                            buffer_size = 0

                        n_rows -= n_for_curr_batch

            if buffer_size:
                for result in self._prep_batch(
                    x_train_np[:buffer_size], y_train_np[:buffer_size], buffer_size
                ):
                    yield result

    def _get_file_indices(self) -> List[int]:
        return list(range(len(self.__ann_files)))

    def _get_buffer_size(self) -> int:
        return self._job_metadata.batch_size

    @abstractmethod
    def _get_max_fail_pct(self) -> float:
        return 0.0

    @staticmethod
    def __read_annotation_file(file_name: str) -> MachineAnnotation:
        return MachineAnnotation(json_dict_input=common.read_json(file_name))

    def __get_non_excluded_feats(self) -> List[FeatureType]:
        return sorted(
            [feature for feature in self.__model_metadata.features if feature not in EXCLUDED_LABELS]
        )

    def _get_vectors(self, annotation: MachineAnnotation, fs_client: FeatureServiceClient):
        return fs_client.vectorize(tokens=annotation, features=set(self.__model_metadata.features))

    def _prep_batch(self, x_train_np, y_train_np, batch_size):
        operations_logger.debug(f"yielding batch with x size {x_train_np.shape} y: {y_train_np.shape}")

        n_rows = min(batch_size, x_train_np.shape[0])
        for start in range(0, n_rows, self._job_metadata.batch_size):
            end = start + self._job_metadata.batch_size
            x, y = x_train_np[start:end], y_train_np[start:end]
            if self._job_metadata.class_weight:
                operations_logger.debug("sample weight crafted")
                crafted_sample_weight = construct_sample_weight(self._job_metadata.class_weight, y)
                yield x, y, crafted_sample_weight
            else:
                yield x, y

    def __get_file_matrices(self, ma_file, ann_file, batch_size):
        tokens = self.__read_annotation_file(ma_file)
        duplicate_tokens = self.__data_source.get_duplicate_token_idx(
            ann_file=ann_file, machine_annotations=tokens
        )

        vectors: Vectorization = Vectorization(json_input_dict=self._vector_cache[ma_file])

        label_vectors = self.__data_source.match_for_gold(
            tokens.range,
            tokens.tokens,
            self.__data_source.get_brat_label(ann_file, self.__label_enum),
            self.__label_enum,
            self.__model_metadata.binary_classifier,
        )

        x_mat = self.__get_mat3d_train_generator(
            vectors=vectors, num_tokens=len(tokens), batch_size=batch_size, duplicate_tokens=duplicate_tokens
        )

        y_mat = LabelMat3d(
            vectors=label_vectors,
            label_enum=self.__label_enum,
            num_tokens=len(tokens),
            batch_size=batch_size,
            max_window_size=self.__model_metadata.window_size,
            binary_classifier=self.__model_metadata.binary_classifier,
            duplicate_tokens=duplicate_tokens,
        )

        return x_mat, y_mat

    def __slice_3d_mat(self, mat_3d, feature_col_mapping: Dict[FeatureType, range], tid: str = None):
        features = self.__get_non_excluded_feats()
        cols = []
        for f in features:
            cols.extend(list(feature_col_mapping[f]))

        operations_logger.debug(f"Feature Size = {len(cols)}", tid=tid)
        return mat_3d[:, 0 : self.__model_metadata.window_size, tuple(cols)]

    def __get_mat3d_train_generator(
        self, vectors, num_tokens: int, batch_size: int = BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value, **kwargs
    ) -> Mat3dGenerator:
        operations_logger.debug(
            f"Creating generator, num_tokens: {num_tokens}, batch_size: {batch_size}, " f"start_idx={kwargs}"
        )

        return Mat3dGenerator(
            vectors=vectors,
            batch_size=batch_size,
            num_tokens=num_tokens,
            max_window_size=self.__model_metadata.window_size,
            features=set(self.__model_metadata.features),
            **kwargs,
        )


class LSTMTestingDataGenerator(LSTMDataGenerator):
    def _get_max_fail_pct(self) -> float:
        return self._job_metadata.max_test_failures_pct


class LSTMTrainingDataGenerator(LSTMDataGenerator):
    def _get_max_fail_pct(self) -> float:
        return self._job_metadata.max_train_failures_pct

    def _get_file_indices(self):
        file_indices = super()._get_file_indices()
        random.shuffle(file_indices)
        return file_indices

    def _get_buffer_size(self) -> int:
        return self._job_metadata.batch_size * floor(
            BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value / self._job_metadata.batch_size
        )

    def _prep_batch(self, x_train_np, y_train_np, batch_size):
        rng_state = np.random.get_state()
        np.random.shuffle(x_train_np)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train_np)

        for batch in super()._prep_batch(x_train_np, y_train_np, batch_size):
            yield batch
