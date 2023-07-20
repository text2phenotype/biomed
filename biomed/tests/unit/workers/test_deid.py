from unittest.mock import patch

from text2phenotype.tasks.task_enums import TaskEnum
from text2phenotype.tasks.task_info import PHITokenTaskInfo
from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase

from biomed.tests.unit.workers import WorkerBaseTest
from biomed.workers.deid.worker import DeidTaskWorker


class TestDeidTaskWorker(WorkerBaseTest, TaskTestCase):
    worker_class = DeidTaskWorker

    def setUp(self) -> None:
        super().setUp()
        self.s3_container[
            john_stevens.get_result_file_key(PHITokenTaskInfo)] = john_stevens.FIXTURE_PHI_TOKENS.read_bytes()
        fixture = john_stevens.FIXTURE_DEID.read_text()
        patcher = patch('biomed.workers.deid.worker.redact_text', return_value=fixture)
        patcher.start()

        self.set_initial_work_task(john_stevens.DOCUMENT_TASK)
        redacted_txt_file_key = john_stevens.get_result_file_key(self.document_task.task_statuses[TaskEnum.deid])
        redacted_txt_file_key = f'{redacted_txt_file_key}.txt'
        self.demographics_key = john_stevens.get_result_file_key((self.chunk_task.task_statuses[TaskEnum.demographics]))
        self.s3_container[redacted_txt_file_key] = john_stevens.FIXTURE_DEID.read_bytes()
        self.s3_container[self.demographics_key] = john_stevens.FIXTURE_DEMOGRAPHICS.read_bytes()

        self.addCleanup(patcher.stop)
