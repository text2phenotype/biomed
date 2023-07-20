from text2phenotype.common.featureset_annotations import Vectorization, MachineAnnotation
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import CovidLabModelTaskInfo

from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker
from biomed.lab.labs import get_covid_labs
from biomed.biomed_env import BiomedEnv


class CovidLabTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = CovidLabModelTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.covid_lab
    RESULTS_FILE_EXTENSION = CovidLabModelTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'CovidLabModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,
            biomed_version: str, text: str, tid: str = None) -> dict:
        return get_covid_labs(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)



