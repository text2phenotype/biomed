from biomed.biomed_env import BiomedEnv
from biomed.workers.covid_lab.worker import CovidLabTaskWorker


if __name__ == '__main__':
    worker = CovidLabTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
