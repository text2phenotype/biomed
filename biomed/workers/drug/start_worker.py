from biomed.biomed_env import BiomedEnv
from biomed.workers.drug.worker import DrugModelTaskWorker


if __name__ == '__main__':
    worker = DrugModelTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
