import copy
import unittest

from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.tasks.task_enums import TaskEnum

from biomed.constants.constants import OperationToModelType
from biomed.reassembler import TASK_TO_REASSEMBLER_MAPPING
from biomed.reassembler.reassemble_functions import (
    update_json_response_ranges,
    reassemble_single_list_chunk_results,
    reassemble_summary_chunk_results, get_reassemble_function,
)


class TestReassembleChunksFunctions(unittest.TestCase):
    def test_update_json_ranges(self):
        sample_output = [{"text": "hypertension", "range": [30, 42]}, {"text": "hyena", "range": [0, 5]}]
        text_range = [2, 80]
        actual_output = update_json_response_ranges(sample_output, text_range)
        expected_output = [{"text": "hypertension", "range": [32, 44]}, {"text": "hyena", "range": [2, 7]}]
        self.assertEqual(actual_output, expected_output)


    def test_single_summary_chunk(self):
        json_entry = {"DiseaseDisroder": [{"text": "hypertension",
                                           "range": [30, 42]}, {"text": "hyena", "range": [0, 5]}],
                      "AJDso": [{"text": "discipline", "range": [50, 60]}, {"text": "green", "range": [20, 25]}],
                      VERSION_INFO_KEY: [{}]}
        text_start = [[2, 0]]
        expected_data = {"DiseaseDisroder": [{"text": "hypertension",
                                              "range": [32, 44]}, {"text": "hyena", "range": [2, 7]}],
                         "AJDso": [{"text": "discipline", "range": [52, 62]},
                                   {"text": "green", "range": [22, 27]}],
                         VERSION_INFO_KEY: [{}]}
        assembled_data = reassemble_summary_chunk_results([(text_start[0], copy.deepcopy(json_entry))])
        self.assertEqual(assembled_data, expected_data)

    def test_multiple_summary_chunks(self):
        json_entry = {"DiseaseDisorder": [{"text": "hypertension",
                                           "range": [30, 42]}, {"text": "hyena", "range": [0, 5]}],
                      "AJDso": [{"text": "discipline", "range": [50, 60]}, {"text": "green", "range": [20, 25]}],
                      VERSION_INFO_KEY: [{'name': 2}]}
        text_start = [[2, 0], [62, 1], [92, 2]]
        expected_data = {"DiseaseDisorder": [{"text": "hypertension",
                                              "range": [32, 44]}, {"text": "hyena", "range": [2, 7]},
                                             {"text": "hypertension",
                                              "range": [92, 104]}, {"text": "hyena", "range": [62, 67]},
                                             {"text": "hypertension", "range": [122, 134]},
                                             {"text": "hyena", "range": [92, 97]}],
                         "AJDso": [{"text": "discipline", "range": [52, 62]},
                                   {"text": "green", "range": [22, 27]},
                                   {"text": "discipline", "range": [112, 122]},
                                   {"text": "green", "range": [82, 87]},
                                   {"text": "discipline", "range": [142, 152]},
                                   {"text": "green", "range": [112, 117]}],
                         VERSION_INFO_KEY: [{'name': 2}]
                         }
        assembled_data = reassemble_summary_chunk_results([(span, copy.deepcopy(json_entry)) for span in text_start])

        self.assertEqual(assembled_data, expected_data)

    def test_phi_tokens_reassemble_chunks(self):
        json_entry = [{'text': 'Hermione', 'range': [2, 10]}, {'text': 'Granger', 'range': [20, 26]}]
        text_chunk = [[2, 100], [100, 200]]
        expected_data = [{'text': 'Hermione', 'range': [4, 12]}, {'text': 'Granger', 'range': [22, 28]},
                         {'text': 'Hermione', 'range': [102, 110]}, {'text': 'Granger', 'range': [120, 126]}]
        assembled = reassemble_single_list_chunk_results([(span, copy.deepcopy(json_entry)) for span in text_chunk])

        self.assertEqual(assembled, expected_data)

    def test_phi_tokens_reassemble_single_chuni(self):
        json_entry = [{'text': 'Hermione', 'range': [2, 10]}, {'text': 'Granger', 'range': [20, 26]}]
        text_chunk = [[100, 2]]
        expected_data = [{'text': 'Hermione', 'range': [102, 110]}, {'text': 'Granger', 'range': [120, 126]}]

        assembled = reassemble_single_list_chunk_results([(span, json_entry) for span in text_chunk])

        self.assertEqual(assembled, expected_data)

    def test_biomed_operations_in_reassemble_key(self):
        for task_operation in OperationToModelType:
            task_enum = TaskEnum[task_operation.name]
            self.assertIsNotNone(get_reassemble_function(task_enum), task_enum)
