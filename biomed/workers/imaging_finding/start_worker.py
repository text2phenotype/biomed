from biomed.biomed_env import BiomedEnv
from biomed.workers.imaging_finding.worker import ImagingFindingTaskWorker


if __name__ == '__main__':
    worker = ImagingFindingTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
