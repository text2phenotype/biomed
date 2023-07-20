from biomed.biomed_env import BiomedEnv
from biomed.workers.pdf_embedder.worker import PDFEmbeddingWorker


if __name__ == '__main__':
    worker = PDFEmbeddingWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
