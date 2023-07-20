from biomed.biomed_env import BiomedEnv
from biomed.workers.train_test.worker import TrainTestWorker


if __name__ == '__main__':
    worker = TrainTestWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
