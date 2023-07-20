import unittest
from text2phenotype.tasks.task_enums import TaskOperation
from biomed.common.helpers import feature_list_helper


class TestBiomedFeatureHelper(unittest.TestCase):
    def test_works_for_all_tasks(self):
        for task in TaskOperation:
            operation_features = feature_list_helper(operations=[task.name])
            if task in TaskOperation.biomed_operations():
                self.assertGreaterEqual(len(operation_features), 0, task.name)
            else:
                self.assertEqual(len(operation_features), 0, task.name)
