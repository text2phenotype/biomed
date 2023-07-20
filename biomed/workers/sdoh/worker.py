from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType

from biomed.biomed_env import BiomedEnv
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker
from biomed.sdoh.sdoh import get_sdoh_response
from text2phenotype.tasks.task_info import SDOHTaskInfo


class SDOHWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.SDOH_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.sdoh
    RESULTS_FILE_EXTENSION = SDOHTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'SDOHWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_sdoh_response(tokens=tokens, vectors=vectors, tid=tid, biomed_version=biomed_version)
