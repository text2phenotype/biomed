from biomed.biomed_env import BiomedEnv
from biomed.workers.reassembler.worker import Reassembler


if __name__ == '__main__':
    worker = Reassembler()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
