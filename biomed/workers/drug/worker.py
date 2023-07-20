from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import DrugModelTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.drug.drug import meds_and_allergies
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class DrugModelTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = DrugModelTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.drug
    RESULTS_FILE_EXTENSION = DrugModelTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DrugModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return meds_and_allergies(tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)

