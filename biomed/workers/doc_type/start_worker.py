from biomed.biomed_env import BiomedEnv
from biomed.workers.doc_type.worker import DocumentTypeTaskWorker


if __name__ == '__main__':
    worker = DocumentTypeTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
