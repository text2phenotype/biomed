from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import DiseaseSignTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.diagnosis.diagnosis import diagnosis_sign_symptoms
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class DiseaseSignTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = DiseaseSignTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.disease_sign
    RESULTS_FILE_EXTENSION = DiseaseSignTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DiseaseSignModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str,  tid: str = None) -> dict:
        return diagnosis_sign_symptoms(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid, use_tf_session=False)
