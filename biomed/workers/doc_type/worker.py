from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.tasks.task_enums import TaskEnum, WorkType
from text2phenotype.tasks.task_info import DocumentTypeTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.document_section.predict import get_doc_types
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker


class DocumentTypeTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.DOC_TYPE_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.doctype
    RESULTS_FILE_EXTENSION = DocumentTypeTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DocumentTypeTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(tokens: MachineAnnotation, vectors: Vectorization,  biomed_version: str, text: str,  tid: str = None) -> dict:
        return get_doc_types(text=text, annotations=tokens, vectors=vectors, tid=tid)

