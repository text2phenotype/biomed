from biomed.biomed_env import BiomedEnv
from biomed.workers.summary.worker import SummaryWorker


if __name__ == '__main__':
    worker = SummaryWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
