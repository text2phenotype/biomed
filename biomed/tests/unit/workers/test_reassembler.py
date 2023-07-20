from text2phenotype.tasks.task_enums import TaskStatus
from text2phenotype.tasks.task_info import (
    create_task_info,
    TaskEnum
)
from text2phenotype.tasks.task_message import TaskMessage
from text2phenotype.tasks.utils import (
    get_manifest_job_storage_key,
    get_metadata_document_storage_key
)
from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase

from biomed.workers.reassembler.worker import Reassembler


class TestReassembler(TaskTestCase):
    def setUp(self):
        super().setUp()
        self.worker = Reassembler()
        self.document_task = john_stevens.DOCUMENT_TASK.copy(deep=True)
        self.file_key_document = get_metadata_document_storage_key(self.document_task.document_id)

        self.job_task = john_stevens.JOB_TASK
        self.job_manifest_key = get_manifest_job_storage_key(self.job_task.job_id)

        self.message = TaskMessage(
            redis_key=self.document_task.redis_key,
            work_type=self.document_task.work_type,
        )
        self.set_initial_work_task(self.document_task)
        self.fake_redis_client.set(self.document_task.document_id, self.document_task.json())
        self.fake_redis_client.set(self.job_task.job_id, self.job_task.json())

        with open(john_stevens.EXTRACTED_TEXT_FILE, 'rb') as f:
            self.s3_container[self.document_task.document_info.text_file_key] = f.read()

        for task_enum in self.document_task.chunk_tasks:
            task_info = create_task_info(task_enum)
            self.s3_container[
                john_stevens.get_result_file_key(task_info)] = john_stevens.CHUNKS_MAPPING[task_enum].read_bytes()
            # clear fixture
            if task_enum in self.worker.work_task.task_statuses:
                del self.worker.work_task.task_statuses[task_enum]

        self.chunk_list = [john_stevens.CHUNK_TASK.copy(deep=True)]
        for chunk in self.chunk_list:
            self.fake_redis_client.set(chunk.redis_key, chunk.json())

        self.worker.download_object_str = lambda x: ""

    def test_worker_result(self):
        result = self.worker.do_work()
        self.assertIsNotNone(result)

    def test_task_statuses(self):
        self.worker.reassemble_results(self.worker.work_task, self.worker.storage_client)
        doc_task = self.worker.refresh_task(self.worker.work_task)
        for task_enum in self.document_task.chunk_tasks:
            with self.subTest(chunk_task=task_enum):
                self.assertEqual(doc_task.task_statuses[task_enum].status, TaskStatus.completed_success)

    def test_write_data_to_storage(self):
        self.worker.reassemble_results(self.worker.work_task, self.worker.storage_client)
        doc_task = self.worker.refresh_task(self.worker.work_task)
        for task_enum in set(self.document_task.chunk_tasks) - {TaskEnum.annotate, TaskEnum.vectorize}:
            with self.subTest(chunk_task=task_enum):
                results_file_key = doc_task.task_statuses[task_enum].results_file_key
                self.assertIsNotNone(results_file_key)
                self.assertIsNotNone(self.s3_container.get(results_file_key))

    def test_failed_chunk_processing(self):
        failed_task_enum = self.worker.work_task.chunk_tasks[0]
        chunk = self.chunk_list[0]
        chunk.task_statuses[failed_task_enum].status = TaskStatus.failed
        self.fake_redis_client.set(chunk.redis_key, chunk.json())

        self.worker.reassemble_results(self.worker.work_task, self.worker.storage_client)
        doc_task = self.worker.refresh_task(self.worker.work_task)

        self.assertEqual(doc_task.task_statuses[failed_task_enum].status, TaskStatus.completed_failure)
        self.assertTrue(doc_task.task_statuses[failed_task_enum].error_messages)

