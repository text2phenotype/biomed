from biomed.biomed_env import BiomedEnv
from biomed.workers.bladder_risk.worker import BladderRiskTaskWorker


if __name__ == '__main__':
    worker = BladderRiskTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
