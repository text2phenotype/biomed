from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import FamilyHistoryTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.family_history.family_history_pred import family_history
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class FamilyHistoryTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = FamilyHistoryTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.family_history
    RESULTS_FILE_EXTENSION = FamilyHistoryTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'FamilyHistpryModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str,  tid: str = None) -> dict:
        return family_history(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
