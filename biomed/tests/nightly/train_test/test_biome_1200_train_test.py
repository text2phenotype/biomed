import os
import time
import unittest

from biomed.constants.constants import EXCLUDED_LABELS
from text2phenotype.constants.features.feature_type import FeatureType
from text2phenotype.common.data_source import CURRENT_FEATURE_SET_VERSION

from biomed.train_test.train_test import TrainTestJob
from biomed.models.model_metadata import ModelType
from train_test_build import prepare_metadata


class TestTrainTest(unittest.TestCase):
    job_folder = 'testing_nightly'

    def train_test(self, metadata=None, job_id=None):
        if not metadata:
            return
        model_meta, job_meta, data_source, ensemble_meta = prepare_metadata({'metadata': metadata})
        TrainTestJob(model_meta, job_meta, data_source, ensemble_meta).run()

        data_source.sync_down(f'models/train/{job_id}', os.path.join(data_source.parent_dir, job_id))
        # check that training dir was uploaded to S3
        self.assertTrue(os.path.isdir(os.path.join(data_source.parent_dir, job_id)))

        data_source.sync_down(f'models/test/{job_id}', os.path.join(data_source.parent_dir, job_id))
        self.assertTrue(os.path.isdir(os.path.join(data_source.parent_dir, job_id)))

    def test_lab_train_test(self):
        job_id = os.path.join(self.job_folder, f'testing_train_test_build_{time.time()}')
        metadata = {"model_type": ModelType.lab,
                    "original_raw_text_dirs": ["mtsamples"],
                    'ann_dirs': ['deleys.brandman/annotation_BIOMED-655'],
                    "feature_set_subfolders": ["b"],
                    "feature_set_version": CURRENT_FEATURE_SET_VERSION,
                    "features": [3, 43, 2],
                    'train': True,
                    'test': True,
                    'job_id': job_id,
                    'class_weight': {'1': 3},
                    'testing_fs_subfolders': ['a']}
        self.train_test(metadata, job_id)

    def test_openEMR_train_test_sample_weight(self):
        job_id = os.path.join(self.job_folder, f'testing_train_test_build_{time.time()}')
        metadata = {"model_type": ModelType.demographic, "original_raw_text_dirs": ["OpenEMR"],
                    "ann_dirs": [f"shannon.fee"],
                    "feature_set_version": CURRENT_FEATURE_SET_VERSION,
                    "feature_set_subfolders": ['a', 'b', 'c', 'd', 'e'],
                    "features": [f.value for f in FeatureType if
                                 f not in set(
                                     EXCLUDED_LABELS).union(
                                     {FeatureType.npi_binary, FeatureType.npi, FeatureType.document_type})],
                    'train': True, 'test': True,
                    'job_id': job_id, 'sample_weight': True, "class_weight": {"0": 1, "1": 1, "2": 10,
                                                                              "3": 20, "4": 10, "5": 20, "6": 1, "7": 1,
                                                                              "8": 1, "9": 1, "10": 1, "11": 1, "12": 1,
                                                                              "13": 1, "14": 1, "15": 1, "16": 1,
                                                                              "17": 1, "18": 1, "19": 1, "20": 1,
                                                                              "21": 1, "22": 1, "23": 1, "24": 1,
                                                                              "25": 1, "26": 1, "27": 1, "28": 1,
                                                                              "29": 1, "30": 2},
                    'testing_text_dirs': ['OpenEMR'],
                    'testing_ann_dirs': ['shannon.fee'],
                    'testing_fs_subfolders': ['a', 'b']}

        self.train_test(metadata, job_id)

    def test_OpenEMR_ensemble_test(self):
        job_id = os.path.join(self.job_folder, f'test_ensemble_build_{time.time()}')

        metadata = {"model_type": ModelType.demographic,
                    'test_ensemble': True, 'job_id': job_id,
                    'testing_ann_dirs': [f'shannon.fee'],
                    'testing_text_dirs': ['OpenEMR'],
                    'testing_fs_subfolders': ['a', 'b', 'c', 'd', 'e'],
                    'feature_set_version': CURRENT_FEATURE_SET_VERSION}

        model_meta, job_meta, data_source, ensemble_meta = prepare_metadata({'metadata': metadata})
        TrainTestJob(model_meta, job_meta, data_source, ensemble_meta).run()

        data_source.sync_down(f'models/test/{job_id}', os.path.join(data_source.parent_dir, job_id))
        self.assertTrue(os.path.isdir(os.path.join(data_source.parent_dir, job_id)))

    def test_k_fold(self):
        job_id = os.path.join(self.job_folder, f'test_k_fold_build_{time.time()}')
        metadata = {"model_type": ModelType.demographic,
                    'job_id': job_id,
                    'ann_dirs': [f'shannon.fee'],
                    'original_raw_text_dirs': ['OpenEMR'],
                    'k_fold_subfolders': ['b', 'c'],
                    'feature_set_version': CURRENT_FEATURE_SET_VERSION,
                    'k_fold_validation': True,
                    'features': [0, 1, 2, 3, 4, 5, 6, 7]
                    }
        model_meta, job_meta, data_source, ensemble_meta = prepare_metadata({'metadata': metadata})
        TrainTestJob(model_meta, job_meta, data_source, ensemble_meta).run()
        data_source.sync_down(f'models/multi/{job_id}', os.path.join(data_source.parent_dir, job_id))
        self.assertTrue(os.path.isdir(os.path.join(data_source.parent_dir, job_id)))
