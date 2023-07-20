from biomed.biomed_env import BiomedEnv
from biomed.workers.vital_sign.worker import VitalSignTaskWorker


if __name__ == '__main__':
    worker = VitalSignTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
