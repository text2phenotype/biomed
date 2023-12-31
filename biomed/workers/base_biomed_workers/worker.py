import json
from abc import ABC, abstractmethod

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum
from text2phenotype.tasks.task_info import TaskInfo
from text2phenotype.tasks.rmq_worker import RMQConsumerTaskWorker

from biomed.biomed_env import BiomedEnv


class SingleModelBaseWorker(RMQConsumerTaskWorker, ABC):
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    @abstractmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization, biomed_version: str, text: str, tid: str = None) -> dict:
        pass

    def do_work(self) -> TaskInfo:
        task_result = self.init_task_result()
        tid = self.tid
        source_text = self.download_object_str(self.work_task.text_file_key)

        # get annotations from storage.
        tokens = MachineAnnotation(json_dict_input=self.get_json_results_from_storage(TaskEnum.annotate))

        # get vectors from storage
        vectors = Vectorization(json_input_dict=self.get_json_results_from_storage(TaskEnum.vectorize))

        job_task = self.get_job_task()

        output = self.get_predictions(
            tokens=tokens, vectors=vectors, biomed_version=job_task.biomed_version, text=source_text, tid=tid)

        task_result.results_file_key = self.upload_results(json.dumps(output))

        return task_result
