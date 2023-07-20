from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import PHITokenTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.deid.deid import get_phi_tokens
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class PHITokenWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.PHI_TOKEN_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.phi_tokens
    RESULTS_FILE_EXTENSION = PHITokenTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'PHITokenWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_phi_tokens(tokens=tokens, vectors=vectors, tid=tid, biomed_version=biomed_version)
