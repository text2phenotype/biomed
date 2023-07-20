import unittest

from text2phenotype.constants.features import FeatureType
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import feature_list_helper


class TestBiomed1247(unittest.TestCase):
    def test_invalid_operation(self):
        feature_list_helper({'abc'})

    def test_get_deid_features(self):
        # test that we are getting features returned and there are >10 of them for deid, confirm that regex is in there
        features = feature_list_helper({TaskOperation.phi_tokens})
        self.assertGreater(len(list(features)), 10)
        self.assertIn(FeatureType.hospital_personnel, features)
        self.assertIn(FeatureType.patient, features)
        self.assertIn(FeatureType.gender, features)
        self.assertIn(FeatureType.age, features)
        self.assertIn(FeatureType.phi_indicator_words, features)
        self.assertIn(FeatureType.address, features)
        self.assertIn(FeatureType.hospital, features)
        self.assertIn(FeatureType.grammar, features)
        self.assertIn(FeatureType.family, features)
        self.assertIn(FeatureType.time_qualifier, features)
        self.assertIn(FeatureType.icd, features)
        self.assertIn(FeatureType.contact_info, features)

    def test_all_feature_set_greater_equal_single(self):
        features = feature_list_helper({TaskOperation.demographics, TaskOperation.phi_tokens, TaskOperation.clinical_summary})
        feature_deid = feature_list_helper({TaskOperation.phi_tokens})
        self.assertGreaterEqual(len(features), len(feature_deid))
        self.assertTrue(feature_deid.issubset(features))
