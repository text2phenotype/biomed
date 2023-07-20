import os
import shutil
import unittest

import numpy as np
import pandas as pd

from text2phenotype.common import common
from biomed import RESULTS_PATH
from biomed.models.voter_model import VoterModel, DataMissingLabelsException
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport


class TestVoterModel(unittest.TestCase):
    DATA_X = np.random.rand(2, 100, 3)
    DATA_Y = np.random.randint(0, DATA_X.shape[2], size=100)
    CONFIG = {
        "job_id": "test_voter_model",
        "model_type": 14,
        "window_size": 1,
    }

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        test_job_ids = [cls.CONFIG["job_id"], "my_id"]
        for job_id in test_job_ids:
            try:
                shutil.rmtree(os.path.join(RESULTS_PATH, job_id))
            except FileNotFoundError:
                # we never got to the test that made the target output, so skip deleting it
                pass

    def _init_object(self, job_id=None):
        config = self.CONFIG.copy()
        if job_id:
            config["job_id"] = job_id
        model_metadata = ModelMetadata(**config)
        job_metadata = JobMetadata.from_dict(config)
        data_source = BiomedDataSource(**config)
        return VoterModel(model_metadata=model_metadata, data_source=data_source, job_metadata=job_metadata)

    def _save_data_to_results(self, job_id):
        data_path = os.path.join(RESULTS_PATH, job_id, "full_info.pkl.gz")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        n_split = 4
        split_pts = np.linspace(0, self.DATA_X.shape[1], n_split + 1, endpoint=True, dtype=int)
        split_pts = np.vstack([split_pts[:-1], split_pts[1:]]).T
        data_dict = {
            f"file_{i}": {
                "raw_prob": self.DATA_X[:, start:end, :],
                "labels": self.DATA_Y[start:end]
            }
            for i, (start, end) in enumerate(split_pts)
        }
        pd.to_pickle(data_dict, data_path)

    def test_train(self):
        # check that the interaction with the outside works as expected
        self._save_data_to_results(self.CONFIG["job_id"])
        voter = self._init_object()
        _ = voter.train()

        assert os.path.isfile(voter.model_metadata.model_file_path)
        model_metadata_json = common.read_json(
            voter.model_metadata.model_file_path + ".metadata.json")
        self.assertEqual(self.CONFIG["window_size"], model_metadata_json["window_size"])

    def test__train(self):
        # confirm the inner training loop works
        voter = self._init_object()
        model = voter._train(self.DATA_X, self.DATA_Y)
        self.assertEqual(self.DATA_X.shape[2], model.n_classes_)

    def test__train_missing_label(self):
        voter = self._init_object()
        data_y = np.random.randint(0, self.DATA_X.shape[2] - 1, size=100)
        with self.assertRaises(DataMissingLabelsException):
            _ = voter._train(self.DATA_X, data_y)

    def test_predict(self):
        self._save_data_to_results(self.CONFIG["job_id"])
        voter = self._init_object()
        # save a trained model so we can load it from memory and use it for
        _ = voter.train()
        voter.predict(self.DATA_X)

    def test_test(self):
        self._save_data_to_results(self.CONFIG["job_id"])
        voter = self._init_object()
        # save a trained model so we can load it from memory and use it for
        _ = voter.train()

        voter.test()

        _, test = voter._train_test_split_ix(self.DATA_X.shape[1])

        report_df = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(
            common.read_text(os.path.join(voter.results_job_path, "report_weighted.txt"))
        )
        # sanity check to make sure the report is catching the correct test results by count
        self.assertEqual(len(test), report_df.support.values[:-1].sum())

    def test_get_probabilities(self):
        voter = self._init_object()
        with self.assertRaises(FileNotFoundError):
            _ = voter.get_probabilities("foo")

        self._save_data_to_results(self.CONFIG["job_id"])
        x, y = voter.get_probabilities(self.CONFIG["job_id"])
        np.testing.assert_array_equal(self.DATA_X, x)
        np.testing.assert_array_equal(self.DATA_Y, y)

    def test__train_test_split_ix(self):
        voter = self._init_object()
        n_samples = 100
        train, test = voter._train_test_split_ix(n_samples)
        # mutually exclusive
        self.assertEqual(set(), set(train).intersection(set(test)))
        # complete
        self.assertEqual(set(range(n_samples)), set(train).union(set(test)))
        # idempotent
        train2, test2 = voter._train_test_split_ix(n_samples)
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(test, test2)

    def test__move_existing_reports(self):
        voter = self._init_object("my_id")
        os.makedirs(voter.results_job_path, exist_ok=True)
        common.write_text("foobar", os.path.join(voter.results_job_path, "foo.txt"))
        common.write_text("foo,bar\n1,2", os.path.join(voter.results_job_path, "table.csv"))
        common.write_text("nope", os.path.join(voter.results_job_path, "nope.nope"))
        target_dir = "baz"
        voter._move_existing_reports(target_dir)
        out_dir = os.path.join(voter.results_job_path, target_dir)
        expected_files = ["foo.txt", "table.csv"]
        self.assertEqual(set(expected_files), set(os.listdir(out_dir)))

        voter = self._init_object("not_my_id")
        with self.assertRaises(FileNotFoundError):
            voter._move_existing_reports(target_dir)

    def test_reshape(self):
        shape = (3, 100, 4)
        arr = np.arange(np.product(shape)).reshape(shape)
        arr_2d = VoterModel.reshape_3d_to_2d(arr)
        arr_3d = VoterModel.reshape_2d_to_3d(arr_2d, shape)
        np.testing.assert_array_equal(arr, arr_3d)

        # test reshape if we dont know how many tokens were dropped from the original array
        half_arr = arr[:, ::2, :]
        arr_2d = VoterModel.reshape_3d_to_2d(half_arr)
        arr_3d = VoterModel.reshape_2d_to_3d(arr_2d, (shape[0], -1, shape[2]))
        np.testing.assert_array_equal(half_arr, arr_3d)

    def test__concat_ensemble_results(self):
        my_dict = {
            'file1': {
                'labels': [1],
                'probs': [[0.1, 0.1, 0.1]]
            },
            'file2': {
                'labels': [2],
                'probs': [[0.1, 0.1, 0.1]]
            }
        }
        out = VoterModel._concat_ensemble_results(my_dict, "probs")
        np.testing.assert_array_equal(np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]), out)
        labels = VoterModel._concat_ensemble_results(my_dict, "labels")
        np.testing.assert_array_equal(np.array([1, 2]), labels)


if __name__ == '__main__':
    unittest.main()
