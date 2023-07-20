import json

from text2phenotype.common.log import operations_logger
from text2phenotype.tasks.task_enums import (
    TaskEnum,
    WorkType,
)
from text2phenotype.tasks.task_info import (
    DateOfServiceTaskInfo,
    TaskInfo,
)

from biomed.biomed_env import BiomedEnv
from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import DateOfServiceOutput
from biomed.constants.constants import DATE_OF_SERVICE_CATEGORY
from biomed.constants.model_enums import ModelType
from biomed.date_of_service.model import DOSModel
from biomed.workers.base_biomed_workers.worker import RMQConsumerTaskWorker


class DateOfServiceTaskWorker(RMQConsumerTaskWorker):
    QUEUE_NAME = BiomedEnv.DOS_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.date_of_service
    RESULTS_FILE_EXTENSION = DateOfServiceTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'DateOfServiceTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    def do_work(self) -> TaskInfo:
        task_result = self.init_task_result()
        source_text = self.download_object_str(self.work_task.text_file_key)

        output = self.__get_predictions(source_text)

        task_result.results_file_key = self.upload_results(json.dumps(output))
        return task_result

    @staticmethod
    def __get_predictions(text: str) -> dict:
        model = DOSModel()

        results = []

        predictions = model.predict(text)

        for prediction in predictions:
            if text[prediction.doc_span[0]:prediction.doc_span[1]].replace('\n', ' ') != prediction.text:
                # per Ana, allow this to return even though we don't match back to the original text properly
                operations_logger.warning(f'Text location is incorrect: {prediction.span} {prediction.text}')

            results.append(DateOfServiceOutput(
                label=prediction.label,
                text=prediction.text,
                range=prediction.doc_span,
                normalized_date=prediction.normalized_date))
        return AspectResponse(
            DATE_OF_SERVICE_CATEGORY, response_list=results).to_versioned_json(ModelType.date_of_service)

