import os
import pkg_resources
import tempfile
import unittest

from biomed.models import model_report
from text2phenotype.common import common


class ModelReportTests(unittest.TestCase):
    train_history_path = pkg_resources.resource_filename(
        "biomed.tests",
        "fixtures/models/train_history_diagnosis.json")
    test_results_dir = None
    __original_results_path = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.test_results_dir = tempfile.TemporaryDirectory()
        cls.__original_results_path = model_report.RESULTS_PATH
        model_report.RESULTS_PATH = cls.test_results_dir.name

    @classmethod
    def tearDownClass(cls):
        # Close the file, the directory will be removed after the test
        cls.test_results_dir.cleanup()
        model_report.RESULTS_PATH = cls.__original_results_path

    def test_plot_train_metrics(self):
        train_history = common.read_json(self.train_history_path)
        job_id = "test_job_id"
        report = model_report.ModelReportPlots("diagnosis", "test_job_id")
        report.plot_train_metrics(train_history)

        expected_output_path = os.path.join(self.test_results_dir.name, job_id)
        # check to see if we have pngs in the target location
        self.assertTrue(os.path.exists(
            os.path.join(expected_output_path, f"{report.model_name}_loss.png")
        ))


if __name__ == '__main__':
    unittest.main()
