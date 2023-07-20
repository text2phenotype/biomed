from threading import Lock
from typing import Callable, List

import numpy
from text2phenotype.common.log import operations_logger


class ModelWorker:
    def __init__(self,
                 model_file_name: str,
                 prediction_function: Callable,
                 result_lock: Lock,
                 results: dict,
                 **kwargs):
        super().__init__()
        self.key = model_file_name
        self.non_locking = prediction_function
        self.lock = result_lock
        self.results = results
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        operations_logger.debug(f'Beginning Model Prediction for {self.key}', tid=self.kwargs.get('tid'))
        result = self.non_locking(*args, **kwargs, tid=self.kwargs.get('tid'))
        self.update_results(result)

    def update_results(self, result):
        operations_logger.debug(f'Acquiring Lock for model {self.key}', tid=self.kwargs.get('tid'))
        self.lock.acquire()
        try:
            self.results[self.key] = result
        finally:
            self.lock.release()
        operations_logger.debug(f'Releasing Lock for model {self.key}', tid=self.kwargs.get('tid'))


class ModelWorkerGenerator(ModelWorker):
    def run(self, tokens: List[dict], vectors):
        operations_logger.debug(f'Beginning Model Prediction for {self.key}', tid=self.kwargs.get('tid'))
        result = self.non_locking(tokens=tokens,
                                  vectors=vectors,
                                  tid=self.kwargs.get('tid'))
        self.update_results(result)

