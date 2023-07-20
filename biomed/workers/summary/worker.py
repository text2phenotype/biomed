from concurrent.futures import (
    as_completed,
    Future,
    ThreadPoolExecutor,
)
import functools
import json
import threading
from typing import (
    Dict,
    Iterable,
    List,
)

from text2phenotype.common.log import operations_logger
from text2phenotype.tasks.rmq_worker import RMQConsumerWorker
from text2phenotype.tasks.mixins import WorkTaskMethodsMixin
from text2phenotype.tasks.task_enums import (
    TaskEnum,
    TaskStatus,
    WorkType,
)
from text2phenotype.tasks.task_info import (
    BiomedSummaryTaskInfo,
    SummaryTask,
)
from text2phenotype.tasks.work_tasks import DocumentTask

from biomed.biomed_env import BiomedEnv
from biomed.summary.text_to_summary import (
    summary_from_parts,
)


class SummaryWorker(WorkTaskMethodsMixin, RMQConsumerWorker):
    QUEUE_NAME = BiomedEnv.SUMMARY_TASKS_QUEUE.value
    RESULTS_FILE_EXTENSION = None
    WORK_TYPE = WorkType.document
    NAME = 'Summary'
    ROOT_PATH = BiomedEnv.root_dir

    def process_message(self):
        try:
            self.init_work_task(self.task_message)

            with self.task_update_manager():
                for task_info in self.iter_task_info():
                    task_info.started_at = None
                    task_info.completed_at = None
                    task_info.status = TaskStatus.processing
                    task_info.attempts += 1

            self.do_work()

        except Exception:
            with self.task_update_manager():
                for task_info in self.iter_task_info():
                    if task_info.status is not TaskStatus.completed_success:
                        task_info.status = TaskStatus.failed

        finally:
            self.publish_message(BiomedEnv.SEQUENCER_QUEUE.value, self.task_message)

    def do_work(self):
        storage = self.storage_client.get_container()

        job_task = self.get_job_task()
        doc_task: DocumentTask = self.work_task

        original_text = storage.get_object_content(doc_task.text_file_key).decode('utf8')
        biomed_version = job_task.biomed_version

        for task_info in self.iter_task_info():
            task = task_info.task

            try:
                self.create_summary(task, original_text, biomed_version)
            except:
                operations_logger.exception(f'Exception due creating the \'{task.value}\' summary')

                with self.task_update_manager():
                    task_info = self.get_task_info(task)
                    task_info.completed_at = self._dt_now_utc()
                    task_info.status = TaskStatus.failed
            else:
                with self.task_update_manager():
                    task_info = self.get_task_info(task)
                    task_info.completed_at = self._dt_now_utc()
                    task_info.status = TaskStatus.completed_success

    @property
    def submitted_summary_tasks(self) -> List[TaskEnum]:
        key = 'submitted_summary_tasks'

        tasks_list = getattr(self._local_data, key, None)
        if tasks_list is not None:
            return tasks_list

        if self.work_task is None:
            self.init_work_task(self.task_message)

        tasks_list = []
        for task, task_info in self.work_task.task_statuses.items():
            if isinstance(task_info, BiomedSummaryTaskInfo) \
                    and task_info.status is TaskStatus.submitted:
                tasks_list.append(task)

        setattr(self._local_data, key, tasks_list)
        return tasks_list

    def get_task_info(self, task: TaskEnum) -> BiomedSummaryTaskInfo:
        return self.work_task.task_statuses[task]

    def iter_task_info(self) -> Iterable[BiomedSummaryTaskInfo]:
        yield from map(self.get_task_info, self.submitted_summary_tasks)

    def create_summary(self, task: TaskEnum, source_text: bytes, biomed_version: str):
        with self.task_update_manager():
            task_info = self.get_task_info(task)
            task_info.started_at = self._dt_now_utc()

        for summary_task in task_info.iter_summary_tasks():
            models = [TaskEnum(m.value) for m in summary_task.models]

            # TODO: Load models results using concurrency.ThreadExecutor
            summary_parts = [self.get_json_results_from_storage(model)
                             for model in models]

            summary_data = summary_from_parts(summary_parts,
                                              text=source_text,
                                              biomed_version=biomed_version)

            result_file_key = self.upload_result(summary_task, summary_data)

            with self.task_update_manager():
                task_info = self.get_task_info(task)
                task_info.add_result_file_key(result_file_key)

        with self.task_update_manager():
            task_info = self.get_task_info(task)
            task_info.status = TaskStatus.completed_success
            task_info.completed_at = self._dt_now_utc()

    # TODO: Refactoring - move storage-related methods to a mixin

    def get_json_results_from_storage(self, task: TaskEnum):
        task_info = self.work_task.task_statuses[task]
        json_bytes = self.storage_client.container.get_object_content(task_info.results_file_key)
        return json.loads(json_bytes.decode('utf8'), strict=False)

    def upload_result(self, summary_task: SummaryTask, data: Dict) -> str:
        object_key = summary_task.get_storage_key(self.work_task.document_id)

        json_bytes = json.dumps(data).encode('utf8')
        self.storage_client.container.write_bytes(json_bytes, object_key)

        return object_key
