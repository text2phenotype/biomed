from biomed.biomed_env import BiomedEnv
from biomed.workers.icd10_diagnosis.worker import ICD10DiagnosisWorker


if __name__ == '__main__':
    worker = ICD10DiagnosisWorker()
    BiomedEnv.APPLICATION_NAME.value = f'Biomed - {worker.NAME}'

    worker.start()
