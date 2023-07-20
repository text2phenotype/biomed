from biomed.biomed_env import BiomedEnv
from biomed.workers.family_history.worker import FamilyHistoryTaskWorker


if __name__ == '__main__':
    worker = FamilyHistoryTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
