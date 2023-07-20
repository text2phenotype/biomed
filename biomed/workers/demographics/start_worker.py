from biomed.biomed_env import BiomedEnv
from biomed.workers.demographics.worker import DemographicsTaskWorker


if __name__ == '__main__':
    worker = DemographicsTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
