import unittest

from biomed.common.model_test_helpers import cui_set_for_entry, cuis_from_annot, check_cui_match
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.features import FeatureType


class TestCuiManipulations(unittest.TestCase):
    def test_check_cui_overlap(self):
        self.assertFalse(check_cui_match({0, 1, 2}, {3}))
        self.assertFalse(check_cui_match(set(), set()))
        self.assertTrue(check_cui_match({0, 1, 5}, {5, 7}))
        self.assertTrue(check_cui_match({0, 1}, set()))
        self.assertTrue(check_cui_match(set(), {6, 7}))

    def test_cui_from_annot(self):
        annotation = [{
            "Medication": [{
                "code": "39953003",
                "cui": "C0040329",
                "tui": ["T131"],
                "tty": [],
                "preferredText": "Tobacco",
                "codingScheme": "SNOMEDCT_US"
            }],
            "polarity": "positive"
        }, {
            "Procedure": [{
                "code": "77477000",
                "cui": "C0040405",
                "tui": ["T060"],
                "tty": [],
                "preferredText": "X-Ray Computed Tomography",
                "codingScheme": "SNOMEDCT_US"
            }],
            "polarity": "positive"}]
        expected_cui_set = {'C0040329', 'C0040405'}
        actual = cuis_from_annot(annotation, {'Medication', 'Procedure'})
        self.assertEqual(expected_cui_set, set(actual))

    def test_cui_set_for_entry(self):
        annotation = MachineAnnotation(json_dict_input={
            'clinical': {
                "53": [{
                    "Medication": [{
                        "code": "39953003",
                        "cui": "C0040329",
                        "tui": ["T131"],
                        "tty": [],
                        "preferredText": "Tobacco",
                        "codingScheme": "SNOMEDCT_US"
                    }],
                    "polarity": "positive"
                }]}})

        expected_cui_set = ['C0040329']
        actual = cui_set_for_entry(53, {FeatureType.clinical: ['Medication']}, annotation)
        self.assertEqual(actual, expected_cui_set)
