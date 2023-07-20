import unittest
import datetime
import os
from typing import Tuple, List

import pandas

from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from text2phenotype.common.data_source import DataSourceContext, CURRENT_FEATURE_SET_VERSION

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common import common

from biomed.models.model_metadata import ModelMetadata
from biomed.models.testing_results.compare_results.compare_classification_report import (
    compare_testing_report)
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.train_test.job_metadata import JobMetadata
from biomed.train_test.train_test import TrainTestJob
from biomed import RESULTS_PATH
from biomed.constants.constants import ModelType
from biomed.data_sources.data_source import BiomedDataSource

from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from text2phenotype.constants.features import FeatureType


class BaseQATest(unittest.TestCase):
    job_folder = 'testing_nightly'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.modelType: ModelType = None
        cls.baseline_report: pandas.DataFrame = None
        cls.report_prefix = 'report_minus_partial'
        cls.use_threshold: bool = False
        cls.threshold_categories: list = None
        cls.use_mean_ensemble: bool = False

        cls.initialize()

        cls.feature_set_annotated_dir = datetime.datetime.now()
        cls.text_dir = 'original_text'
        cls.ann_dir = f'qa_testing_docs/{cls.modelType.name}'
        cls.matched_annotated_dir = f'{cls.ann_dir}/{cls.feature_set_annotated_dir}'
        cls.data_source: BiomedDataSource = BiomedDataSource(testing_text_dirs=[cls.text_dir],
                                                             testing_ann_dirs=[cls.ann_dir],
                                                             feature_set_version=CURRENT_FEATURE_SET_VERSION,
                                                             testing_fs_subfolders=[])
        cls.feature_set_annotated = False
        cls.model_metadata = ModelMetadata(model_type=cls.modelType)

    @property
    def label_enum(self):
        return MODEL_TYPE_2_CONSTANTS[self.modelType].label_class

    @classmethod
    def initialize(cls):
        cls.tearDownClass()

        raise unittest.SkipTest("Skip base class for all qa tests")

    def get_original_text_file(self, ann_fp):
        return ann_fp.replace(f'/{self.modelType.name}/', '/').replace('.ann', '.txt')

    def feature_set_annotate(self, features: List[FeatureType] = None):
        fs_client = FeatureServiceClient()
        ann_files = self.data_source.get_ann_files(label_enum=MODEL_TYPE_2_CONSTANTS[self.modelType].label_class,
                                                   context=DataSourceContext.testing)
        self.data_source.get_original_raw_text_files(context=DataSourceContext.testing,
                                                     orig_dir='qa_testing_docs/original_text')
        fs_header = os.path.join(self.data_source.parent_dir, self.data_source.feature_set_version)
        for ann_file in ann_files:
            txt_file = self.get_text_from_ann_file(ann_file)
            if txt_file is not None:
                annot = fs_client.annotate(common.read_text(txt_file), features=features)
                fs_fp = txt_file.replace('.txt', '.json').replace(self.data_source.parent_dir, fs_header).replace(
                    '/qa_testing_docs/', '/')
                common.write_json(annot.to_dict(), fs_fp)

    def get_text_from_ann_file(self, ann_file: str):
        txt_file = ann_file.replace('.ann', '.txt').replace(f'/{self.modelType.name}/', '/')
        if os.path.isfile(txt_file):
            return txt_file

    def create_metadata(self) -> Tuple[JobMetadata, EnsembleModelMetadata]:
        job_metadata = JobMetadata(
            job_id=os.path.join(self.job_folder,
                                f'qa_test_{self.modelType.name}_{self.use_threshold}_{self.feature_set_annotated_dir}'),
            test_ensemble=True)
        ensemble_metadata = EnsembleModelMetadata(model_type=self.modelType,
                                                  threshold_categories=self.threshold_categories,
                                                  use_mean_ensemble=self.use_mean_ensemble)
        return job_metadata, ensemble_metadata

    def run_test_ensemble_job(self, job_metadata: JobMetadata, ensemble_metadata: EnsembleModelMetadata):
        TrainTestJob(model_metadata=self.model_metadata,
                     job_metadata=job_metadata,
                     data_source=self.data_source,
                     ensemble_metadata=ensemble_metadata).run()

    def test_ensemble(self):
        job_metadata, ensemble_metadata = self.create_metadata()
        if not self.feature_set_annotated:
            features = set(ensemble_metadata.features)
            if MODEL_TYPE_2_CONSTANTS[self.modelType].required_representation_features:
                features.update(MODEL_TYPE_2_CONSTANTS[self.modelType].required_representation_features)
            self.feature_set_annotate(features=features)
        self.run_test_ensemble_job(job_metadata=job_metadata, ensemble_metadata=ensemble_metadata)
        results_folder = self.get_testing_results_folder(job_metadata.job_id)
        new_report = self.get_report_specifically(results_folder)

        diff_df = compare_testing_report(self.baseline_report,
                                         new_report)
        self.assert_diff_df_improvement(diff_df)

    def assert_diff_df_improvement(self, diff_df: pandas.DataFrame):
        for idx, row in diff_df.iterrows():
            if not row.class_label.lower() in ['na', 'avg / total']:
                self.assertGreaterEqual(row.recall_change, -.02, row.to_string())
                self.assertGreaterEqual(row.precision_change, -.05, row.to_string())
                self.assertGreaterEqual(row['f1-score_change'], -.05, row.to_string())

    def get_testing_results_folder(self, job_id: str):
        source_path = f'models/test/{job_id}'
        dest_path = f'{RESULTS_PATH}/{job_id}'
        self.data_source.sync_down(source_path, dest_path)
        return dest_path

    def get_report_specifically(self, dest_path) -> pandas.DataFrame:
        report_fn = os.path.join(dest_path, f'{self.report_prefix}.txt')
        report_txt = common.read_text(report_fn)
        print(report_txt)
        report_df = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(report_txt, label_class_enum=self.label_enum)
        return report_df
