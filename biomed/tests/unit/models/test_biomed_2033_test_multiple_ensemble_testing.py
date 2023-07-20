import unittest

from biomed.constants.constants import ModelType
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.models.multi_train_test import MultiEnsembleTest
from biomed.train_test.job_metadata import JobMetadata


class TestMultiEnsembleMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.input_metadata = {
            "model_type": 4,
            "original_raw_text_dirs": ["original_text_dirs"],
            "feature_set_version": "feature_version",
            "ann_dirs": ["annotator_1"],
            "job_id": "base_job",
            "testing_fs_subfolders": ["e"],
            "testing_model_mapping": {
                "current_dev": ["demographic_20201115_2d3", "5122", "phi_demographic2_20201013e2"],
                "new_ensemble": ["demographic_20201115_2d3", "5122"],
            },
        }
        cls.multi_ensemble_test = MultiEnsembleTest(cls.input_metadata)

        cls.input_metadata_voting = cls.input_metadata.copy()
        cls.input_metadata_voting["voting_method"] = "threshold"
        cls.multi_ensemble_test_voter = MultiEnsembleTest(cls.input_metadata_voting)

    def test_job_count(self):
        self.assertEqual(self.multi_ensemble_test.job_count, 2)
        self.assertEqual(self.multi_ensemble_test_voter.job_count, 2)

    def test_job_data_sources(self):
        def _test_multi_ensemble(multi_ensemble: MultiEnsembleTest):
            self.assertEqual(len(multi_ensemble.data_source_list), 2)
            for data_source in multi_ensemble.data_source_list:
                self.assertEqual(data_source.feature_set_version, "feature_version")
                self.assertEqual(data_source.original_raw_text_dirs, ["original_text_dirs"]),
                self.assertEqual(data_source.ann_dirs, ["annotator_1"])
                self.assertEqual(data_source.testing_fs_subfolders, ["e"])

        _test_multi_ensemble(self.multi_ensemble_test)
        _test_multi_ensemble(self.multi_ensemble_test_voter)

    def test_job_metadatas(self):
        self.assertEqual(len(self.multi_ensemble_test.job_metadata_list), 2)

        job_ids = set()
        for job_meta in self.multi_ensemble_test.job_metadata_list:
            self.assertIsInstance(job_meta, JobMetadata)
            self.assertTrue(job_meta.test_ensemble)
            job_ids.add(job_meta.job_id)
        self.assertEqual(len(job_ids), 2)
        self.assertEqual(
            job_ids,
            {
                "base_job/current_dev",
                "base_job/new_ensemble",
            },
        )

    def test_ensemble_metadatas(self):
        self.assertEqual(len(self.multi_ensemble_test.ensemble_metadata_list), 2)
        for ens_meta in self.multi_ensemble_test.ensemble_metadata_list:
            self.assertIsInstance(ens_meta, EnsembleModelMetadata)
            self.assertEqual(ens_meta.model_type, ModelType.demographic)

    def test_model_metadatas(self):
        for model_meta in self.multi_ensemble_test.model_metadata_list:
            self.assertIsNone(model_meta)

    def _assert_ensemble_metadata(self, multi_ensemble_test, expected):
        self.assertIsInstance(multi_ensemble_test.job_ensemble_metadata_list, list)
        self.assertEqual(len(multi_ensemble_test.job_ensemble_metadata_list), 2)
        for entry in multi_ensemble_test.job_ensemble_metadata_list:
            self.assertIsInstance(entry, dict)
            self.assertEqual(
                set(entry.keys()),
                {"model_file_list", "model_name", "voting_method", "voting_model_folder"},
            )
        self.assertListEqual(multi_ensemble_test.job_ensemble_metadata_list, expected)

    def test_job_ensemble_metadata_list(self):
        expected = [
            {
                "model_file_list": [
                    "demographic_20201115_2d3",
                    "5122",
                    "phi_demographic2_20201013e2",
                ],
                "model_name": "current_dev",
                "voting_method": None,
                "voting_model_folder": None,
            },
            {
                "model_file_list": ["demographic_20201115_2d3", "5122"],
                "model_name": "new_ensemble",
                "voting_method": None,
                "voting_model_folder": None,
            },
        ]

        self._assert_ensemble_metadata(self.multi_ensemble_test, expected)

    def test_job_ensemble_metadata_list_voter(self):
        expected = [
            {
                "model_file_list": [
                    "demographic_20201115_2d3",
                    "5122",
                    "phi_demographic2_20201013e2",
                ],
                "model_name": "current_dev",
                "voting_method": self.input_metadata_voting["voting_method"],
                "voting_model_folder": None,
            },
            {
                "model_file_list": ["demographic_20201115_2d3", "5122"],
                "model_name": "new_ensemble",
                "voting_method": self.input_metadata_voting["voting_method"],
                "voting_model_folder": None,
            },
        ]

        self._assert_ensemble_metadata(self.multi_ensemble_test_voter, expected)


if __name__ == "__main__":
    unittest.main()
