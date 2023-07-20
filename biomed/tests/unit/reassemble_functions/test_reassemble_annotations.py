import json
import os
import unittest
from uuid import uuid4

from text2phenotype.common.featureset_annotations import (
    RANGE,
    TOKEN,
)

from biomed.reassembler import ReassemblerResultManager
from biomed.reassembler.reassemble_annotations import (
    ANNOTATIONS_LIST_TYPE_FEATURES_KEYS,
    reassemble_annotations,
)


class TestReassembleAnnotations(unittest.TestCase):
    TEST_DICT_TYPE_FEATURES = ['feature1', 'feature2']

    def create_chunk_data(self, tokens: list):
        """Generate fake chunk data for `annotate` task"""

        count_of_tokens = len(tokens)
        chunk_id = uuid4().hex

        data = {}

        for key in ANNOTATIONS_LIST_TYPE_FEATURES_KEYS:
            # Fill with random values
            data[key] = [uuid4().hex for _ in range(count_of_tokens)]

        data[TOKEN] = tokens

        # [[0, 1], [1, 2], [2, 3]]
        data[RANGE] = [
            [i, i+1]
            for i in range(count_of_tokens)
        ]

        for key in self.TEST_DICT_TYPE_FEATURES:
            # {"0": {"feature1": [0], "chunk_id": "<uuid>"}}
            data[key] = {str(i): {key: [i], 'chunk_id': chunk_id}
                         for i in range(count_of_tokens)}

        return data

    def test_reassemble_single_chunk(self):
        chunk1 = self.create_chunk_data(tokens=['A', 'B', 'C'])

        chunks_mapping = [
            [(0, 10), chunk1],  # [chunk_span, chunk_data]
        ]

        with ReassemblerResultManager() as res_manager:
            self.assertIsNone(res_manager.json_generator)

            reassemble_annotations(chunk_mapping=chunks_mapping, result_manager=res_manager)
            self.assertIsNotNone(res_manager.json_generator)

            result = res_manager.to_dict()

        # Result must be equal to original chunk
        self.assertDictEqual(result, chunk1)
        self.assertEqual(json.dumps(result), json.dumps(chunk1))

    def test_reassemble_multiple_chunks(self):
        chunk1 = self.create_chunk_data(tokens=['A', 'B', 'C'])
        chunk2 = self.create_chunk_data(tokens=['X', 'Y', 'Z'])

        chunks_mapping = [
            [(0, 10), chunk1],  # [chunk_span, chunk_data]
            [(10, 20), chunk2],
        ]

        expected_tokens_count = len(chunk1[TOKEN]) + len(chunk2[TOKEN])

        with self.subTest('Test ReassemblerResultManager'):
            with ReassemblerResultManager() as res_manager:
                self.assertIsNone(res_manager.json_generator)
                self.assertIsNone(res_manager._temp_directory)

                reassemble_annotations(chunk_mapping=chunks_mapping, result_manager=res_manager)
                result = res_manager.to_dict()

                # Result should be as "json_generator"
                self.assertIsNotNone(res_manager.json_generator)

                # Temporary directory should exists
                self.assertIsNotNone(res_manager._temp_directory)

                # Temporary file should be created for each chunk
                temp_dir = res_manager._temp_directory.name
                temp_files = os.listdir(temp_dir)
                self.assertEqual(len(temp_files), len(chunks_mapping) + 1)  # +1 tmp file for combined result

            # Restult manager should clean up temporary directory on exit
            self.assertFalse(os.path.isdir(temp_dir))

        # Check result data
        self.assertEqual(len(result[TOKEN]), expected_tokens_count)

        with self.subTest('Check List Type features'):
            for key in ANNOTATIONS_LIST_TYPE_FEATURES_KEYS:
                # Result should contain all values from both chunks
                self.assertListEqual(result[key], chunk1[key] + chunk2[key])

        with self.subTest('Check Ranges'):
            # Ranges must be shifted corresponds to the `chunk_span[0]` value
            self.assertListEqual(result[RANGE], [[0, 1], [1, 2], [2, 3],
                                                 [10, 11], [11, 12], [12, 13]])

        with self.subTest('Check Dict Type features'):
            expected_keys_list = [str(x) for x in range(expected_tokens_count)]

            for key in self.TEST_DICT_TYPE_FEATURES:
                # Keys in dict-type features must be corresponds to combined tokens list
                self.assertListEqual(list(result[key].keys()), expected_keys_list)

                # But values should be copied from the chunk
                self.assertListEqual(list(result[key].values()),
                                     list(chunk1[key].values()) + list(chunk2[key].values()))
