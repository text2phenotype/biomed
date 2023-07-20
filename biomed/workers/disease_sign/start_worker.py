from biomed.biomed_env import BiomedEnv
from biomed.workers.disease_sign.worker import DiseaseSignTaskWorker


if __name__ == '__main__':
    worker = DiseaseSignTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
