import json
from unittest.mock import patch

from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase

from biomed.tests.unit.workers import WorkerBaseTest
from biomed.workers.phi_token.worker import PHITokenWorker


class TestPHITokenWorker(WorkerBaseTest, TaskTestCase):

    worker_class = PHITokenWorker

    def setUp(self) -> None:
        super().setUp()
        fixture = json.loads(john_stevens.FIXTURE_PHI_TOKENS.read_text())
        patcher = patch('biomed.workers.phi_token.worker.get_phi_tokens', return_value=fixture)
        patcher.start()
        self.addCleanup(patcher.stop)
