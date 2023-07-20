import json
from unittest.mock import patch

from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase

from biomed.tests.unit.workers import WorkerBaseTest
from biomed.workers.oncology_only.worker import OncologyOnlyTaskWorker


class TestOncologyOnlyTaskWorker(WorkerBaseTest, TaskTestCase):

    worker_class = OncologyOnlyTaskWorker

    def setUp(self) -> None:
        super().setUp()
        fixture = json.loads(john_stevens.FIXTURE_DEMOGRAPHICS.read_text())
        patcher = patch('biomed.workers.oncology_only.worker.get_oncology_tokens', return_value=fixture)
        patcher.start()
        self.addCleanup(patcher.stop)
