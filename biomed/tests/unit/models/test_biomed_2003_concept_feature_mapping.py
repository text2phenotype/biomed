import unittest

from biomed.constants.constants import ModelType
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS


class TestConceptFeatureMappings(unittest.TestCase):
    def test_all_concept_feature_mappings(self):
        for model_type in ModelType:
            model_constants = MODEL_TYPE_2_CONSTANTS[model_type]
            if model_constants.production:
                concept_feature_mapping = model_constants.token_umls_representation_feat
                if concept_feature_mapping and len(concept_feature_mapping) >= 0:
                    for feature in concept_feature_mapping:
                        self.assertIsInstance(concept_feature_mapping[feature], list)
                        for entry in concept_feature_mapping[feature]:
                            self.assertIsInstance(
                                entry, str,
                                f'concept feature mapping for model type {model_type}'
                                f' has an incorrect concept_feature_mapping')
