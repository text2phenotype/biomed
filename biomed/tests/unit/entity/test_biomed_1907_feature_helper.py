import unittest
from biomed.constants.constants import ModelType, get_version_model_folders
from biomed.common.helpers import get_features_by_model_type
from biomed.constants.model_constants import ModelClass
from biomed.models.model_cache import ModelMetadataCache


class TestBiomed1907(unittest.TestCase):
    def test_get_feature_by_model_type(self):
        model_metadata_cache = ModelMetadataCache()
        skip = {ModelType.meta, ModelType.disability, ModelType.date_of_service}
        for model_type in ModelType:
            if model_type not in skip:
                features = get_features_by_model_type(model_type=model_type)
                self.assertIsInstance(features, set)
                if not self.all_bert_models(model_type=model_type, model_meta_cache=model_metadata_cache):
                    # oncology uses only a bert model so no features expected
                    self.assertGreaterEqual(len(list(features)), 2, model_type)

    @staticmethod
    def all_bert_models(model_type, model_meta_cache: ModelMetadataCache):
        model_folders = get_version_model_folders(model_type=model_type)
        for model_folder in model_folders:
            model_meta = model_meta_cache.model_metadata(model_type=model_type, model_folder=model_folder)
            if model_meta.model_class != ModelClass.bert:
                return False
        return True