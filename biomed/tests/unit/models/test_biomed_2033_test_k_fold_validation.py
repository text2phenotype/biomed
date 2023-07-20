import unittest

from biomed.models.model_metadata import ModelMetadata
from biomed.models.multi_train_test import KFoldValidation
from biomed.train_test.job_metadata import JobMetadata


class TestKFoldValidation(unittest.TestCase):

    fold_count = 5
    input_metadata = {"model_type": 4, "k_folds": fold_count, "k_fold_subfolders": ["a", "b", "c", "d", "e"],
                       "original_raw_text_dirs": ["original_text_dirs"], "feature_set_version": "feature_version",
                       "ann_dirs": ["annotator_1"], "job_id": "base_job"}

    k_fold_val = KFoldValidation(input_metadata)

    def test_num_folds(self):
        self.assertEqual(self.k_fold_val.job_count, self.fold_count)

    def test_data_source_creation(self):
        datasources = self.k_fold_val.data_source_list

        self.assertEqual(len(datasources), self.fold_count)
        fs_subfolders = {'c', 'b', 'd', 'a', 'e'}
        testing =set()

        for data_source in datasources:
            self.assertEqual(data_source.feature_set_version, 'feature_version')
            self.assertEqual(data_source.original_raw_text_dirs, ['original_text_dirs']),
            self.assertEqual(len(data_source.feature_set_subfolders), 4)
            self.assertEqual(data_source.ann_dirs, ['annotator_1'])
            self.assertEqual(len(data_source.testing_fs_subfolders), 1)
            self.assertSetEqual(
                set(data_source.testing_fs_subfolders).union(set(data_source.feature_set_subfolders)),
                fs_subfolders)
            testing.add(data_source.testing_fs_subfolders[0])

        # ensure each folder was used  as test set  once
        self.assertEqual(testing, fs_subfolders)

    def test_job_metadata_creation(self):
        job_ids = set()
        for job_meta in self.k_fold_val.job_metadata_list:
            self.assertIsInstance(job_meta, JobMetadata)
            self.assertTrue(job_meta.train)
            self.assertTrue(job_meta.test)
            job_ids.add(job_meta.job_id)
        self.assertEqual(len(job_ids), self.fold_count)
        self.assertEqual(
            job_ids,
            {'base_job/fold_0', 'base_job/fold_3', 'base_job/fold_1', 'base_job/fold_2', 'base_job/fold_4'})

    def test_model_metadata_creation(self):
        for model_meta in self.k_fold_val.model_metadata_list:
            self.assertIsInstance(model_meta, ModelMetadata)

    def test_ensemble_metadata_creation(self):
        for ens_meta in self.k_fold_val.ensemble_metadata_list:
            self.assertIsNone(ens_meta)

