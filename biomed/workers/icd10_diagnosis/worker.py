from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import ICD10DiagnosisTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.diagnosis.diagnosis import get_icd_response
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class ICD10DiagnosisWorker(SingleModelBaseWorker):
    QUEUE_NAME = ICD10DiagnosisTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.icd10_diagnosis
    RESULTS_FILE_EXTENSION = ICD10DiagnosisTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'ICD10DiagnosisWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(tokens: MachineAnnotation, vectors: Vectorization, biomed_version: str, text: str,
                        tid: str = None) -> dict:
        return get_icd_response(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid, use_tf_session=False)
