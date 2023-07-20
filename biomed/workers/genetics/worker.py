from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import GeneticsTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.genetics.genetics import get_genetics_tokens
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class GeneticsTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = GeneticsTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.genetics
    RESULTS_FILE_EXTENSION = GeneticsTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'GeneticsModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return get_genetics_tokens(tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)

