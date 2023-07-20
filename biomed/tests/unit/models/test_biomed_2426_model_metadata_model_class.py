import unittest

from biomed.constants.model_constants import ModelType, ModelClass
from biomed.models.model_metadata import ModelMetadata


class TestBiomed2426(unittest.TestCase):
    def test_model_metadata_model_class_none(self):
        model_meta = ModelMetadata(model_type=ModelType.diagnosis)
        self.assertEqual(model_meta.model_class, ModelClass.lstm_base)

    def test_model_meta_model_class_from_str(self):
        model_meta = ModelMetadata(model_type=ModelType.diagnosis, model_class='bert')
        self.assertEqual(model_meta.model_class, ModelClass.bert)

    def test_model_meta_model_class_from_int(self):
        model_meta = ModelMetadata(model_type=ModelType.diagnosis, model_class=2)
        self.assertEqual(model_meta.model_class, ModelClass.doc_type)