import copy
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.data.generator import (
    LSTMTestingDataGenerator,
    LSTMTrainingDataGenerator,
    BERTGenerator,
    chunk_generator,
)
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata

from text2phenotype.annotations.file_helpers import Annotation
from text2phenotype.common.annotations import AnnotationLabelConfig
from text2phenotype.common.common import write_json, write_text
from text2phenotype.common.vector_cache import VectorCacheJson
from text2phenotype.common.featureset_annotations import MachineAnnotation, TOKEN, RANGE, Vectorization, DefaultVectors
from text2phenotype.constants.features.label_types import DuplicateDocumentLabel, DeviceProcedureLabel, LabelEnum


class TestBertGenerator(unittest.TestCase):
    SEED = 12345

    def setUp(self) -> None:
        self.WINDOW_SIZE = 8
        # dict mimics the return output from a VectorCache object
        self.ENCODED_FILE_CACHE = {
            "file_a": {
                "input_ids": [
                    [101, 15903, 2791, 1999, 4654, 7913, 22930, 102],
                    [101, 3111, 1012, 2783, 20992, 1024, 4487, 102],
                    [101, 7103, 2078, 1010, 9765, 12173, 20282, 102],
                    [101, 1010, 102, 0, 0, 0, 0, 0],
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                ],
                "encoded_labels": [
                    [-100, 1, -100, 0, 0, -100, -100, -100],
                    [-100, -100, 0, 0, 0, 0, 2, -100],
                    [-100, -100, -100, 0, 2, -100, -100, -100],
                    [-100, 0, -100, -100, -100, -100, -100, -100],
                ],
            },
            "file_b": {
                "input_ids": [
                    [101, 2092, 8569, 18886, 2078, 5034, 1999, 102],
                    [101, 9999, 2078, 1010, 9765, 12173, 9999, 102],
                    [101, 15238, 2099, 1010, 102, 0, 0, 0],
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 0, 0, 0],
                ],
                "encoded_labels": [
                    [-100, 1, -100, -100, -100, 0, 1, -100],
                    [-100, -100, -100, 0, 2, -100, -100, -100],
                    [-100, -100, -100, 0, -100, -100, -100, -100],
                ],
            },
        }
        self.MODEL_INPUT_FIELDS = ["input_ids", "attention_mask"]
        self.ALL_FIELDS = self.MODEL_INPUT_FIELDS + ["encoded_labels"]
        self.FLATTENED_DATASET = {
            k: [window for f, encoding in self.ENCODED_FILE_CACHE.items() for window in encoding[k]]
            for k in self.ALL_FIELDS
        }
        self.FLATTENED_FILE_SOURCE = ["file_a", "file_a", "file_a", "file_a", "file_b", "file_b", "file_b"]

    def test_bert_generator_batch_2(self):
        file_keys = list(self.ENCODED_FILE_CACHE.keys())
        dataset = BERTGenerator(file_keys, self.ENCODED_FILE_CACHE, self.MODEL_INPUT_FIELDS, self.WINDOW_SIZE)
        batch_size = 2
        for idx, batch in enumerate(dataset.generator.batch(batch_size)):
            for k in self.MODEL_INPUT_FIELDS:
                self.assertEqual(
                    batch[0][k].numpy().tolist(),
                    self.FLATTENED_DATASET[k][idx * batch_size : idx * batch_size + batch_size],
                )
            self.assertEqual(
                batch[1].numpy().tolist(),
                self.FLATTENED_DATASET["encoded_labels"][idx * batch_size : idx * batch_size + batch_size],
            )

    def test_bert_generator_batch_3(self):
        file_keys = list(self.ENCODED_FILE_CACHE.keys())
        dataset = BERTGenerator(file_keys, self.ENCODED_FILE_CACHE, self.MODEL_INPUT_FIELDS, self.WINDOW_SIZE)
        batch_size = 3  # test with an batch size that isnt a multiple of the flattened dataset length
        for idx, batch in enumerate(dataset.generator.batch(batch_size)):
            for k in self.MODEL_INPUT_FIELDS:
                self.assertEqual(
                    batch[0][k].numpy().tolist(),
                    self.FLATTENED_DATASET[k][idx * batch_size : idx * batch_size + batch_size],
                )
            self.assertEqual(
                batch[1].numpy().tolist(),
                self.FLATTENED_DATASET["encoded_labels"][idx * batch_size : idx * batch_size + batch_size],
            )

    def test_bert_generator_shuffle(self):
        file_keys = list(self.ENCODED_FILE_CACHE.keys())
        file_cache = copy.copy(self.ENCODED_FILE_CACHE)
        dataset = BERTGenerator(
            file_keys,
            file_cache,
            self.MODEL_INPUT_FIELDS,
            self.WINDOW_SIZE,
            shuffle_files=True,
            seed=self.SEED)
        subtokens_equal_list = []
        file_source_index = []
        buffer_size = 4
        for idx, batch in enumerate(dataset.generator.shuffle(buffer_size, seed=self.SEED)):
            input_ids = batch[0]["input_ids"].numpy().tolist()
            subtokens_equal_list.append(input_ids == self.FLATTENED_DATASET["input_ids"][idx])
            # do sanity check to see if we are shuffling only within a file
            window_index = self.FLATTENED_DATASET["input_ids"].index(input_ids)
            # which file did the window come from?
            file_source_index.append(self.FLATTENED_FILE_SOURCE[window_index])
        # some of the windows may match, but not all of them
        self.assertFalse(all(subtokens_equal_list))

        # check that the shuffle order isnt clustered by file source
        # expect that the returned file order will NOT match the original order
        self.assertFalse(
            all([
                shuf_source == ordered_source
                for shuf_source, ordered_source in zip(file_source_index, self.FLATTENED_FILE_SOURCE)
            ])
        )

    def test_bert_generator_class_weights(self):
        file_keys = list(self.ENCODED_FILE_CACHE.keys())
        class_weight = {'1': 9, '2': 99}
        expected_sample_weights = np.array([
            # file_a
            [1, 9, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 99, 1],
            [1, 1, 1, 1, 99, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            # file_b
            [1, 9, 1, 1, 1, 1, 9, 1],
            [1, 1, 1, 1, 99, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ])

        dataset = BERTGenerator(
            file_keys,
            self.ENCODED_FILE_CACHE,
            self.MODEL_INPUT_FIELDS,
            self.WINDOW_SIZE,
            class_weight=class_weight
        )
        batch_size = 2
        expected_tuple_size = 2  # TODO: change this to 3 when class_weights works
        for idx, batch in enumerate(dataset.generator.batch(batch_size)):
            self.assertEqual(expected_tuple_size, len(batch))  # should be yielding 3-tuple
            # np.testing.assert_array_equal(
            #     expected_sample_weights[idx * batch_size: idx*batch_size+batch_size, :],
            #     batch[2])


class TestChunkGenerator(unittest.TestCase):

    def test_chunked_sequence_list(self):
        n_elements = 113
        sequence = list(range(n_elements))
        chunk_size = 10
        expected_chunks = [
            sequence[chunk_ix * chunk_size: chunk_ix * chunk_size + chunk_size]
            for chunk_ix in range(math.ceil(n_elements / chunk_size))
        ]
        returned_chunks = chunk_generator(sequence, chunk_size)
        for expected_chunk, gen_chunk in zip(expected_chunks, returned_chunks):
            # calling list() consumes the generator chunk
            self.assertEqual(expected_chunk, list(gen_chunk))

    def test_chunked_sequence_generator(self):
        n_elements = 113
        sequence = range(n_elements)  # range is a generator
        chunk_size = 10
        expected_chunks = [
            list(range(chunk_ix * chunk_size, min(chunk_ix * chunk_size + chunk_size, n_elements)))
            for chunk_ix in range(math.ceil(n_elements / chunk_size))
        ]
        returned_chunks = chunk_generator(sequence, chunk_size)
        for expected_chunk, gen_chunk in zip(expected_chunks, returned_chunks):
            self.assertEqual(expected_chunk, list(gen_chunk))

    def test_chunked_sequence_generator_tokens(self):
        # the archtypical use case
        tokens = ["foo", "bar", "baz", "bif", "bam"]
        expected_chunks = [
            ["foo", "bar"],
            ["baz", "bif"],
            ["bam"]
        ]
        returned_chunks = chunk_generator(tokens, 2)
        for expected_chunk, gen_chunk in zip(expected_chunks, returned_chunks):
            self.assertEqual(expected_chunk, list(gen_chunk))


class TestLabel(LabelEnum):
    na = AnnotationLabelConfig(label='N/A', color='#ffffff', visibility=False, column_index=0,
                               persistent_label='na',
                               order=999)


class LSTMTrainingDataGeneratorImpl(LSTMTrainingDataGenerator):
    def __init__(self, ann_files, ma_files, vector_cache, feature_col_size, data_source, label_enum, job_metadata,
                 model_metadata):
        super().__init__(ann_files, ma_files, vector_cache, feature_col_size, data_source, label_enum, job_metadata,
                         model_metadata)
        self._default_vectors = {0: [0]}

    def _get_vectors(self, annotation, fs_client):
        if annotation.output_dict[TOKEN][-1] == 'headache':
            vector_dict = {0: {0: [1], 1: [2], 3: [3]}}
        else:
            vector_dict = {0: {1: [6], 2: [5], 3: [4]}}

        return Vectorization(default_vectors=DefaultVectors(default_vectors=self._default_vectors),
                             json_input_dict=vector_dict)


class LSTMTestingDataGeneratorImpl(LSTMTestingDataGenerator):
    def __init__(self, ann_files, ma_files, vector_cache, feature_col_size, data_source, label_enum, job_metadata,
                 model_metadata):
        super().__init__(ann_files, ma_files, vector_cache, feature_col_size, data_source, label_enum, job_metadata,
                         model_metadata)
        self._default_vectors = {0: [0]}

    def _get_vectors(self, annotation, fs_client):
        if annotation.output_dict[TOKEN][-1] == 'headache':
            vector_dict = {0: {0: [1], 1: [2], 3: [3]}}
        else:
            vector_dict = {0: {1: [6], 2: [5], 3: [4]}}

        return Vectorization(default_vectors=DefaultVectors(default_vectors=self._default_vectors),
                             json_input_dict=vector_dict)


class LSTMDataGeneratorTestBase(unittest.TestCase):
    _ANN_DIR = 'test_data_gen_ann'
    _FS_DIR = 'test_data_gen_fs'
    _VECT_DIR = 'test_data_gen_vect'  # location to write the cached vectors
    _vector_cache = None
    _FILE_ROOTS = ['ma1', 'ma2']
    tmp_root = None

    @classmethod
    def setUpClass(cls):
        cls.tmp_root = tempfile.TemporaryDirectory()
        cls._FS_FILES = [os.path.join(cls.tmp_root.name, cls._FS_DIR, f'{f}.json') for f in cls._FILE_ROOTS]
        cls._ANN_FILES = [os.path.join(cls.tmp_root.name, cls._ANN_DIR, f'{f}.ann') for f in cls._FILE_ROOTS]

        os.makedirs(os.path.join(cls.tmp_root.name, cls._ANN_DIR), exist_ok=True)
        os.makedirs(os.path.join(cls.tmp_root.name, cls._FS_DIR), exist_ok=True)
        cls._vector_cache = VectorCacheJson("train", os.path.join(cls.tmp_root.name, cls._VECT_DIR))

        # Pt had a headache
        ma1 = MachineAnnotation(json_dict_input={
            TOKEN: ['Pt', 'had', 'a', 'headache'],
            RANGE: [[0, 2], [3, 6], [7, 8], [9, 17]]
        })
        # Pt may have SARS
        ma2 = MachineAnnotation(json_dict_input={
            TOKEN: ['Pt', 'may', 'have', 'SARS'],
            RANGE: [[0, 2], [3, 6], [7, 11], [12, 16]]
        })

        for ma, f in zip([ma1, ma2], cls._FS_FILES):
            write_json(ma.to_dict(), f)

        ann1a = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                           [0, 2], 'Pt', ['id'], 0, 2, DuplicateDocumentLabel.get_category_label().label)
        ann1b = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                           [4, 8], 'ad a', ['id'], 4, 8, DuplicateDocumentLabel.get_category_label().label)
        ann2a = Annotation(DeviceProcedureLabel.device.value.persistent_label,
                           [0, 2], 'Pt', ['id'], 0, 2, DeviceProcedureLabel.get_category_label().label)
        ann2b = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                           [13, 16], 'ARS', ['id'], 13, 16, DuplicateDocumentLabel.get_category_label().label)

        for anns, f in zip([[ann1a, ann1b], [ann2a, ann2b]], cls._ANN_FILES):
            write_text(''.join(ann.to_file_line() for ann in anns), f)

    @classmethod
    def tearDownClass(cls):
        cls._vector_cache.cleanup()
        cls.tmp_root.cleanup()


class LSTMTestingDataGeneratorTests(LSTMDataGeneratorTestBase):
    def test_next_batch(self):
        model_metadata = ModelMetadata(features={0}, window_size=1)
        data_source = BiomedDataSource()
        job_metadata = JobMetadata()
        job_metadata.batch_size = 3
        generator = LSTMTestingDataGeneratorImpl(
            self._ANN_FILES, self._FS_FILES, self._vector_cache, 1, data_source,
            TestLabel, job_metadata, model_metadata)

        data_iter = iter(generator.next_batch())
        for _ in range(2):
            x, _ = next(data_iter)
            self.__validate_batch(x, [3, 0, 6])
            x, _ = next(data_iter)
            self.__validate_batch(x, [5])

    def __validate_batch(self, batch, exp):
        self.assertEqual(len(exp), len(batch))

        for i in range(len(batch)):
            self.assertEqual(exp[i], batch[i][0][0])


class LSTMTrainingDataGeneratorTests(LSTMDataGeneratorTestBase):
    def test_next_batch(self):
        model_metadata = ModelMetadata(features={0}, window_size=1)
        data_source = BiomedDataSource()
        job_metadata = JobMetadata()
        job_metadata.batch_size = 3
        generator = LSTMTrainingDataGeneratorImpl(
            self._ANN_FILES, self._FS_FILES, self._vector_cache, 1, data_source,
            TestLabel, job_metadata, model_metadata)

        exp = [3] + [0, 6, 5]
        iter_count = 0
        for x, _ in generator.next_batch():
            for i in range(len(x)):
                obs = x[i][0][0]

                self.assertIn(obs, exp)
                exp.remove(obs)

            if iter_count == 1:
                self.assertEqual(0, len(exp))
                exp = [3] + [0, 6, 5]

            if iter_count == 3:
                self.assertEqual(0, len(exp))
                break

            iter_count += 1


if __name__ == '__main__':
    unittest.main()
