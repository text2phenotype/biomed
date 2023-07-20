from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import ImagingFindingTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.imaging_finding.imaging_finding import imaging_and_findings
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class ImagingFindingTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = ImagingFindingTaskInfo.QUEUE_NAME
    TASK_TYPE = TaskEnum.imaging_finding
    RESULTS_FILE_EXTENSION = ImagingFindingTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'ImageFindingModelTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(
            tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str, tid: str = None) -> dict:
        return imaging_and_findings(
            tokens=tokens, vectors=vectors, biomed_version=biomed_version, text=text, tid=tid)
