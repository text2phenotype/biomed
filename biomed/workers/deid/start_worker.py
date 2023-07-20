from biomed.biomed_env import BiomedEnv
from biomed.workers.deid.worker import DeidTaskWorker


if __name__ == '__main__':
    worker = DeidTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
