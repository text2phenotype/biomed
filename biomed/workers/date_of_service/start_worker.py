from biomed.biomed_env import BiomedEnv
from biomed.workers.date_of_service.worker import DateOfServiceTaskWorker


if __name__ == '__main__':
    worker = DateOfServiceTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
