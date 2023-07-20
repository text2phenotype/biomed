import os
import shutil
import tempfile
import unittest

from biomed import RESULTS_PATH
from biomed.constants.constants import get_version_model_folders
from biomed.constants.model_constants import ModelType
from biomed.meta.ensembler import Ensembler
from biomed.models.model_base import ModelBase
from biomed.models.model_metadata import ModelMetadata
from biomed.models.data_counts import DataCounter
from biomed.train_test.job_metadata import JobMetadata

from text2phenotype.annotations.file_helpers import Annotation
from text2phenotype.apiclients import FeatureServiceClient
from text2phenotype.common.common import read_json, write_json, write_text
from text2phenotype.common.featureset_annotations import MachineAnnotation, TOKEN, RANGE
from text2phenotype.constants.features.label_types import DuplicateDocumentLabel, DeviceProcedureLabel
from text2phenotype.common.vector_cache import VectorCacheJson


class ModelBaseTests(unittest.TestCase):
    __ANN_DIR = 'test_model_base_ann'
    __FS_DIR = 'test_model_base_fs'
    __VECT_DIR = 'test_model_base_vect'
    _vector_cache = None
    __FILE_ROOTS = ['ma1', 'ma2']
    tmp_root = None

    @classmethod
    def setUpClass(cls):
        cls.tmp_root = tempfile.TemporaryDirectory()
        cls.__FS_FILES = [os.path.join(cls.tmp_root.name, cls.__FS_DIR, f'{f}.json') for f in cls.__FILE_ROOTS]
        cls.__ANN_FILES = [os.path.join(cls.tmp_root.name, cls.__ANN_DIR, f'{f}.ann') for f in cls.__FILE_ROOTS]

        os.makedirs(os.path.join(cls.tmp_root.name, cls.__ANN_DIR), exist_ok=True)
        os.makedirs(os.path.join(cls.tmp_root.name, cls.__FS_DIR), exist_ok=True)
        cls._vector_cache = VectorCacheJson("train", cls.__VECT_DIR)

        # Pt had a headache
        ma1 = MachineAnnotation(json_dict_input={
            TOKEN: ['Pt', 'had', 'a', 'headache'],
            RANGE: [[0, 2], [3, 6], [7, 8], [9, 17]]
        })
        # Pt may have SARS
        ma2 = MachineAnnotation(json_dict_input={
            TOKEN: ['Pt', 'may', 'have', 'SARS'],
            RANGE: [[0, 2], [3, 6], [7, 11], [12, 16]]
        })

        for ma, f in zip([ma1, ma2], cls.__FS_FILES):
            write_json(ma.to_dict(), f)

        ann1a = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                           [0, 2], 'Pt', ['id'], 0, 2, DuplicateDocumentLabel.get_category_label().label)
        ann1b = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                          [4, 8], 'ad a', ['id'], 4, 8, DuplicateDocumentLabel.get_category_label().label)
        ann2a = Annotation(DeviceProcedureLabel.device.value.persistent_label,
                          [0, 2], 'Pt', ['id'], 0, 2, DeviceProcedureLabel.get_category_label().label)
        ann2b = Annotation(DuplicateDocumentLabel.duplicate.value.persistent_label,
                           [13, 16], 'ARS', ['id'], 13, 16, DuplicateDocumentLabel.get_category_label().label)

        for anns, f in zip([[ann1a, ann1b], [ann2a, ann2b]], cls.__ANN_FILES):
            write_text(''.join(ann.to_file_line() for ann in anns), f)

    @classmethod
    def tearDownClass(cls):
        cls._vector_cache.cleanup()
        cls.tmp_root.cleanup()

    def test_adjust_tokens(self):
        metadata = ModelMetadata()
        model = ModelBase(model_metadata=metadata, model_type=ModelType.drug)

        self.assertEqual(38, model._adjust_token_count(100, self.__ANN_FILES, self.__FS_FILES))

    def test_step_count_even(self):
        metadata = ModelMetadata(features={0}, window_size=1)
        model = ModelBase(model_metadata=metadata, model_type=ModelType.drug)
        model.job_metadata.batch_size = 3

        self.assertEqual(1, model._get_steps_per_epoch(model.job_metadata.batch_size))

    def test_step_count_less_than_batch_size(self):
        metadata = ModelMetadata(features={0}, window_size=1)
        model = ModelBase(model_metadata=metadata,  model_type=ModelType.drug)
        model.job_metadata.batch_size = 3

        self.assertEqual(1, model._get_steps_per_epoch(model.job_metadata.batch_size - 1))

    def test_step_count_remainder(self):
        metadata = ModelMetadata(features={0}, window_size=1)
        model = ModelBase(model_metadata=metadata, model_type=ModelType.drug)
        model.job_metadata.batch_size = 3

        self.assertEqual(3, model._get_steps_per_epoch((model.job_metadata.batch_size * 2) + 2))

    def  test_has_FSC(self):
        metadata = ModelMetadata(features={0}, window_size=1)
        model = ModelBase(model_metadata=metadata, model_type=ModelType.drug)
        model.job_metadata.batch_size = 3

        self.assertIsNotNone(model.feature_service_client)
        self.assertIsInstance(model.feature_service_client, FeatureServiceClient)

    def test_ensembler_FSC(self):
        ensembler = Ensembler(model_type=ModelType.drug,
                              model_file_list=[get_version_model_folders(ModelType.drug)[1]])
        self.assertIsNotNone(ensembler.feature_service_client)
        self.assertIsInstance(ensembler.feature_service_client, FeatureServiceClient)

    def test_support_counting(self):
        model_type = ModelType.drug
        metadata = ModelMetadata(features={0}, window_size=2)
        model = ModelBase(model_metadata=metadata, model_type=model_type)
        model.job_metadata.job_id = "test_job"
        model.feature_col_size = 16  # dont activate the FS client
        counter = DataCounter(
            model.label2id, n_features=model.feature_col_size,
            window_size=model.window_size,
            window_stride=model.model_metadata.window_stride)
        model._get_doc_support_counts(counter, self.__ANN_FILES, self.__FS_FILES)
        self.assertEqual(len(self.__ANN_FILES), len(counter.doc_token_label_counts))
        self.assertEqual(len(self.__ANN_FILES), counter.n_documents)

        context_str = "foo"
        model._add_context_support_metrics(counter, context=context_str)

        expected_context_keys = [
            "n_documents", "total_token_count", "total_valid_token_count",
            "total_num_windows", "total_token_label_counts"
        ]
        expected_keys = [
            f"{context_str}_{name}" for name in expected_context_keys
        ] + ["num_label_classes", "num_features", "window_size", "window_stride"]

        self.assertEqual(set(expected_keys), set(model._data_support_metrics.keys()))
        for k in expected_context_keys:
            self.assertEqual(getattr(counter, k), model._data_support_metrics[f"{context_str}_{k}"])

        model.write_data_support_metrics()
        expected_path = os.path.join(
            RESULTS_PATH, model.job_metadata.job_id, f"data_support_metrics_{model_type.name}.json")
        self.assertTrue(os.path.isfile(expected_path))
        metrics_out = read_json(expected_path)
        self.assertIsInstance(metrics_out["foo_total_token_label_counts"]["na"], int)



if __name__ == '__main__':
    unittest.main()
