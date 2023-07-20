from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import VitalSignTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.vital_signs.vital_signs import get_vital_signs
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class VitalSignTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = VitalSignTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.vital_signs
    RESULTS_FILE_EXTENSION = VitalSignTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'VitalModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_vital_signs(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
