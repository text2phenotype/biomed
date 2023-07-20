from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import LabModelTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.lab.labs import summary_lab_value
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class LabModelTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = LabModelTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.lab
    RESULTS_FILE_EXTENSION = LabModelTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'LabModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return summary_lab_value(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
