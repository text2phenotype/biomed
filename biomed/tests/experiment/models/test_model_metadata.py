import os
import unittest

from biomed import RESULTS_PATH
from biomed.models.model_metadata import ModelMetadata
from text2phenotype.constants.features import FeatureType


class MyTestCase(unittest.TestCase):
    CONFIG = {
        "job_id": "test_model_metadata",
        "train": True,
        "test": True,
        "model_type": 13,
        "batch_size": 4,
        "window_size": 1,  # whole word window length
        "class_weight": {"1": 400, "2": 400},
        "learning_rate": 0.01,
        "epochs": 1,
        "features": [2, 3, 4],  # NOTE: this feature isnt actually used, just need something in here
        "async_mode": False,  # false b/c loading mtsamples files, not phi
        "original_raw_text_dirs": ["mtsamples/clean"],
        "ann_dirs": ["annotations"],
        "feature_set_version": "features",
        "feature_set_subfolders": ["train"],
        "testing_fs_subfolders": ["test"],
        "validation_fs_subfolders": ["test"]
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_metadata = ModelMetadata(**cls.CONFIG)

    def test_features(self):
        self.assertEqual({2, 3, 4}, self.model_metadata.features)

    def test_empty_features(self):
        # bert doesnt use explicit engineered features, so it shouldnt be listing them
        # 1. test features=[None]
        tmp_config = self.CONFIG.copy()
        tmp_config["features"] = [None]
        model_metadata = ModelMetadata(**tmp_config)
        self.assertEqual({None}, model_metadata.features)

        # 2. test features=[]
        tmp_config = self.CONFIG.copy()
        tmp_config["features"] = []
        model_metadata = ModelMetadata(**tmp_config)
        # this probably isnt what we want!!
        self.assertEqual(
            {feature_type for feature_type in FeatureType},
            model_metadata.features)

        # 3. test features=None
        tmp_config = self.CONFIG.copy()
        tmp_config["features"] = None
        model_metadata = ModelMetadata(**tmp_config)
        self.assertEqual(
            {feature_type for feature_type in FeatureType},
            model_metadata.features)

    def test_model_file_name(self):
        self.assertEqual(
            self.CONFIG["job_id"],
            os.path.dirname(self.model_metadata.model_file_name)
        )

    def test_model_file_path(self):
        self.assertEqual(
            os.path.join(RESULTS_PATH, self.model_metadata.model_file_path),
            self.model_metadata.model_file_path)

        tmp_config = self.CONFIG.copy()
        tmp_config["model_file_name"] = "foo/bar/my_model/tf_model.h5"
        model_metadata = ModelMetadata(**tmp_config)
        self.assertEqual(
            os.path.join(RESULTS_PATH, tmp_config["model_file_name"]),
            model_metadata.model_file_path)


if __name__ == '__main__':
    unittest.main()
