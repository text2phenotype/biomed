from biomed.biomed_env import BiomedEnv
from biomed.workers.phi_token.worker import PHITokenWorker


if __name__ == '__main__':
    worker = PHITokenWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
