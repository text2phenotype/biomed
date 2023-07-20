from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import OncologyOnlyTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.cancer.cancer import get_oncology_tokens
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class OncologyOnlyTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.ONCOLOGY_ONLY_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.oncology_only
    RESULTS_FILE_EXTENSION = OncologyOnlyTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'OncologyTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_oncology_tokens(text=text, tokens=tokens, vectors=vectors, tid=tid,
                                                biomed_version=biomed_version)

