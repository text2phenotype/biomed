from biomed.biomed_env import BiomedEnv
from biomed.workers.lab.worker import LabModelTaskWorker


if __name__ == '__main__':
    worker = LabModelTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
