from biomed.biomed_env import BiomedEnv
from biomed.workers.bladder_risk.worker import BladderSummaryTaskWorker


if __name__ == '__main__':
    worker = BladderSummaryTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
