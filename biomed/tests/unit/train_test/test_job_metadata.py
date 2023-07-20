import unittest

from biomed.train_test.job_metadata import JobMetadata


class TestJobMetadata(unittest.TestCase):
    def test_defaults(self):
        config = {
            "job_id": "foo",
            "test": True,
            "learning_rate": 1e-5,
            "dropout": 0.5,
        }
        job_metadata = JobMetadata.from_dict(config)
        self.assertEqual(config["job_id"], job_metadata.job_id)
        self.assertEqual(config["test"], job_metadata.test)
        self.assertEqual(config["dropout"], job_metadata.dropout)
        self.assertEqual(None, job_metadata.random_seed)
        self.assertEqual(dict(), job_metadata.class_weight)
        self.assertEqual(list(), job_metadata.k_fold_subfolders)
        self.assertEqual(0, job_metadata.k_folds)

    def test_defaults_from_dict(self):
        config = {
            "job_id": "foo",
            "test": True,
            "learning_rate": 1e-5,
            "dropout": 0.5,
        }
        job_metadata = JobMetadata.from_dict(config)
        self.assertEqual(config["job_id"], job_metadata.job_id)
        self.assertEqual(config["test"], job_metadata.test)
        self.assertEqual(config["dropout"], job_metadata.dropout)
        self.assertEqual(None, job_metadata.random_seed)
        self.assertEqual(dict(), job_metadata.class_weight)
        self.assertEqual(list(), job_metadata.k_fold_subfolders)
        self.assertEqual(0, job_metadata.k_folds)

    def test_extra_fields(self):
        config = {
            "job_id": "foo",
            "model_metadata": []
        }
        job_metadata = JobMetadata.from_dict(config)
        self.assertEqual(config["job_id"], job_metadata.job_id)
        self.assertTrue("model_metadata" not in job_metadata.__dict__)


if __name__ == '__main__':
    unittest.main()
