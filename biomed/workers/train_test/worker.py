import json

from text2phenotype.common.log import operations_logger
from text2phenotype.tasks.rmq_worker import RMQConsumerWorker

from biomed.biomed_env import BiomedEnv
from train_test_build import run_job


class TrainTestWorker(RMQConsumerWorker):
    QUEUE_NAME = BiomedEnv.TRAIN_TEST_TASKS_QUEUE.value

    def __init__(self):
        super().__init__()
        self.model_metadata = None
        self.job_metadata = None
        self.data_source = None
        self.ensemble_metadata = None

    def do_work(self, message: str):
        json_message = json.loads(message)
        operations_logger.info(f'Message read in {json_message}')
        try:
            self.init_metadatas(json_message)
            run_job(json_message)
        except:
            operations_logger.exception('Failed to execute train test job', exc_info=True)
            pass
