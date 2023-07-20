from biomed.biomed_env import BiomedEnv
from biomed.workers.genetics.worker import GeneticsTaskWorker


if __name__ == '__main__':
    worker = GeneticsTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
