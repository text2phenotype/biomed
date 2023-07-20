from biomed.biomed_env import BiomedEnv
from biomed.workers.oncology_only.worker import OncologyOnlyTaskWorker


if __name__ == '__main__':
    worker = OncologyOnlyTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'
    
    worker.start()
