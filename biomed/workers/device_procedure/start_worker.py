from biomed.biomed_env import BiomedEnv
from biomed.workers.device_procedure.worker import DeviceProcedureTaskWorker


if __name__ == '__main__':
    worker = DeviceProcedureTaskWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
