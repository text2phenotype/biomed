from biomed.biomed_env import BiomedEnv
from biomed.workers.smoking.worker import SmokingTaskWorker


if __name__ == '__main__':
    worker = SmokingTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
