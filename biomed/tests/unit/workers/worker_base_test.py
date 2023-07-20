import copy

from text2phenotype.tasks.task_enums import TaskEnum
from text2phenotype.tests.fixtures import john_stevens


class WorkerBaseTest:
    worker_class = None

    def setUp(self) -> None:
        super().setUp()
        self.document_task = copy.deepcopy(john_stevens.DOCUMENT_TASK)
        self.job_task = copy.deepcopy(john_stevens.JOB_TASK)
        self.chunk_task = copy.deepcopy(john_stevens.CHUNK_TASK)
        for item in [self.job_task, self.chunk_task, self.document_task]:
            self.fake_redis_client.set(item.redis_key, item.json())

        self.worker = self.worker_class()

        self.set_initial_work_task(self.chunk_task)

        self.annotation_key = john_stevens.get_result_file_key(self.chunk_task.task_statuses[TaskEnum.annotate])
        self.vectorization_key = john_stevens.get_result_file_key(self.chunk_task.task_statuses[TaskEnum.vectorize])

        self.s3_container[john_stevens.CHUNK_TEXT_FILE_KEY] = john_stevens.EXTRACTED_TEXT_FILE.read_bytes()
        self.s3_container[john_stevens.DOCUMENT_TEXT_FILE_KEY] = john_stevens.EXTRACTED_TEXT_FILE.read_bytes()
        self.s3_container[self.annotation_key] = john_stevens.ANNOTATIONS_RESULT_FILE.read_bytes()
        self.s3_container[self.vectorization_key] = john_stevens.VECTORIZATION_RESULT_FILE.read_bytes()

    def test_do_work(self):
        result = self.worker.do_work()

        self.assertIsNotNone(result.results_file_key)
        self.assertIsNotNone(self.s3_container.get(result.results_file_key))
