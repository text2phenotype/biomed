from biomed.biomed_env import BiomedEnv
from biomed.workers.sdoh.worker import SDOHWorker


if __name__ == '__main__':
    worker = SDOHWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
