from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import SmokingTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.smoking.smoking import get_smoking_status
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class SmokingTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = SmokingTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.smoking
    RESULTS_FILE_EXTENSION = SmokingTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'SmokingModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_smoking_status(tokens=tokens, vectors=vectors, biomed_version=biomed_version,  tid=tid)
