import unittest
from typing import Dict

from biomed.biomed_env import BiomedEnv
from biomed.reassembler import TASK_TO_REASSEMBLER_MAPPING
from biomed.reassembler.reassemble_functions import get_reassemble_function
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker
from biomed.workers.disease_sign.worker import DiseaseSignTaskWorker
from biomed.workers.family_history.worker import FamilyHistoryTaskWorker
from biomed.workers.worker_sets import SINGLE_MODEL_WORKERS
from text2phenotype.common import common
from text2phenotype.common.common import chunk_text_by_size
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.tasks.task_enums import TaskOperation, TaskEnum

from biomed.common.helpers import annotation_helper
from biomed.diagnosis.diagnosis import diagnosis_sign_symptoms
from biomed.tests.fixtures.example_file_paths import carolyn_blose_txt_filepath
from text2phenotype.tasks.task_info import ChunksIterable, TASK_MAPPING


class TestReassembleChunks(unittest.TestCase):
    text = common.read_text(carolyn_blose_txt_filepath)
    chunks = chunk_text_by_size(text, len(text) // 2)
    biomed_version = BiomedEnv.DEFAULT_BIOMED_VERSION.value

    token_vectors = [annotation_helper(chunk[1], operations=TaskOperation.biomed_operations()) for chunk in chunks]

    def confirm_single_model_type(
            self,
            chunk_mapping: ChunksIterable,
            task_worker: SingleModelBaseWorker,
            other_enum_to_chunk_mapping: Dict[TaskEnum, ChunksIterable],
    ):

        reassembled_results = get_reassemble_function(task_worker.TASK_TYPE)(
            chunk_mapping=chunk_mapping,
            other_enum_to_chunk_mapping=other_enum_to_chunk_mapping)

        self.assertIsInstance(reassembled_results, dict)
        self.assertIn(VERSION_INFO_KEY, reassembled_results)


    def test_reassemble_single_model_results(self):
        for model_worker in SINGLE_MODEL_WORKERS:
            #TODO ONCE TF VERSION BUMP/RETRAIN THIS SHOULD BE COLLAPSSED (SF)
            if model_worker is DiseaseSignTaskWorker:
                output_token_chunks = [(
                    self.chunks[i][0],
                    diagnosis_sign_symptoms(
                        tokens=self.token_vectors[i][0], vectors=self.token_vectors[i][1],
                        text=self.chunks[i][1], biomed_version=self.biomed_version)
                ) for i in range(len(self.chunks))]
            else:
                output_token_chunks = [(
                    self.chunks[i][0],
                    model_worker.get_predictions(
                        tokens=self.token_vectors[i][0], vectors=self.token_vectors[i][1],
                        text=self.chunks[i][1], biomed_version=self.biomed_version)
                ) for i in range(len(self.chunks))]
            other_enum_to_token = {}
            if TaskEnum.family_history in TASK_MAPPING[model_worker.TASK_TYPE]().model_dependencies:
                family_output_chunk = [(
                    self.chunks[i][0],
                    FamilyHistoryTaskWorker.get_predictions(
                        tokens=self.token_vectors[i][0], vectors=self.token_vectors[i][1],
                        text=self.chunks[i][1], biomed_version=self.biomed_version)
                ) for i in range(len(self.chunks))]
                other_enum_to_token[TaskEnum.family_history] = family_output_chunk

            self.confirm_single_model_type(output_token_chunks, model_worker, other_enum_to_token)
