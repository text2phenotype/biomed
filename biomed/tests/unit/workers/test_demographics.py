import json
from unittest.mock import patch

from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase

from biomed.tests.unit.workers import WorkerBaseTest
from biomed.workers.demographics.worker import DemographicsTaskWorker


class TestDemographicsTaskWorker(WorkerBaseTest, TaskTestCase):
    worker_class = DemographicsTaskWorker

    def setUp(self) -> None:
        super().setUp()
        fixture = json.loads(john_stevens.FIXTURE_DEMOGRAPHICS.read_text())
        patcher = patch('biomed.workers.demographics.worker.get_demographic_tokens', return_value=fixture)
        patcher.start()
        self.addCleanup(patcher.stop)
