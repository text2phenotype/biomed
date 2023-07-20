from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import DemographicsTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.demographic.demographic import get_demographic_tokens
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class DemographicsTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.DEMOGRAPHICS_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.demographics
    RESULTS_FILE_EXTENSION = DemographicsTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DemographicsTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str,  tid: str = None) -> dict:
        return get_demographic_tokens(tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
