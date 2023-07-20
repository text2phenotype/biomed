from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import DeviceProcedureTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.device_procedure.api import device_procedure_predict_represent
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class DeviceProcedureTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = DeviceProcedureTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.device_procedure
    RESULTS_FILE_EXTENSION = DeviceProcedureTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DeviceProcedureModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return device_procedure_predict_represent(
            tokens=tokens,  vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
