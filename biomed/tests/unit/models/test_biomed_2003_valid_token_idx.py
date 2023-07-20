import unittest

import numpy

from biomed.common.model_test_helpers import document_cui_set, prep_cui_reports
from biomed.models.testing_reports import WeightedReport, RemovingAdjacentConfusion, MinusPartialAnnotation
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.features import FeatureType, LabLabel


class TestValidTokenIndexes(unittest.TestCase):
    TOKEN_TEXT = ['Hello', 'World', 'Patient:', 'Dr.', 'Meredith', 'Grey', 'is', 'a', '43yO', 'woman', 'presenting',
                  'with', 'sob']
    ANNOTATION = MachineAnnotation(
        json_dict_input={
            'token': TOKEN_TEXT,
            'range': [(0, 5), (5, 5), (10, 8), (18, 3), (21, 8), (29, 4), (33, 2), (35, 1), (36, 4), (40, 5), (45, 10),
                      (55, 4), (59, 3)],
            FeatureType.clinical.name: {
                "3": [{"SemGroupA": [{'cui': 'abc'}, {'cui': '100'}]}], "4": [{"SemGroupA": [{'cui': '100'}]}],
            }
        })
    PREDICTED_CATEGORY = numpy.array([0, 0, 0, 0, 3, 0, 3, 2, 1, 1, 0, 0, 0])
    ACTUAL_CATEGORY = [4, 0, 0, 3, 3, 0, 0, 2, 0, 1, 1, 1, 0]
    CONCEPT_FEATURE_MAPPING = {FeatureType.clinical: ['SemGroupA']}

    def test_valid_basic(self):
        # test no duplicate ranges
        out = WeightedReport.get_valid_indexes(tokens=self.ANNOTATION, duplicate_token_idx={})
        expected = {0, 1, 2, 3, 4, 5, 8, 9, 10, 12}
        self.assertSetEqual(out, expected)
        # test with duplicate ranges

        out = WeightedReport.get_valid_indexes(tokens=self.ANNOTATION, duplicate_token_idx={0, 3, 4})
        expected = {1, 2, 5, 8, 9, 10, 12}
        self.assertSetEqual(out, expected)

    def test_valid_adjacent(self):
        out = RemovingAdjacentConfusion.get_valid_indexes(
            tokens=self.ANNOTATION,
            duplicate_token_idx={},
            predicted_results_cat=self.PREDICTED_CATEGORY,
            expected_cat=self.ACTUAL_CATEGORY)

        expected = {0, 1, 2, 4, 5, 9, 12}

        self.assertEqual(out, expected)

    def test_minus_partial_indices(self):
        actual = MinusPartialAnnotation(LabLabel, {FeatureType.clinical: ['SemGroupA']}).get_valid_indexes(
            tokens=self.ANNOTATION, duplicate_token_idx={}, predicted_results_cat=self.PREDICTED_CATEGORY,
            expected_cat=self.ACTUAL_CATEGORY)
        # ntoe that 3 is excluded bc was incorrectly labeled but we partially matched the concept, while 8 & 10 are
        # included bc there was no concept to partially match on
        expected = {0, 1, 2, 4, 5, 8, 9, 10, 12}
        self.assertEqual(actual, expected)

    def test_cui_set_for_document(self):
        actual = document_cui_set(self.ACTUAL_CATEGORY, {FeatureType.clinical: ['SemGroupA']}, self.ANNOTATION)
        expected = {'100', 'abc'}
        self.assertEqual(set(actual.keys()), expected)
        expected_dict = {'100': [3], 'abc': [3]}
        self.assertDictEqual(dict(actual), expected_dict)
        predicted = document_cui_set(self.PREDICTED_CATEGORY, {FeatureType.clinical: ['SemGroupA']}, self.ANNOTATION)
        expected_dict = {'100': [3], 'abc': [0]}
        self.assertDictEqual(dict(predicted), expected_dict)

    def test_doc_cui_transform(self):
        actual_cui_set = {'100': [3, 0], 'abc': [3], 'a': [2], 'b': [0, 1], 'c': [1, 2, 3]}
        predicted_cui_set = {'100': [0], 'abc': [3], 'a': [0], 'b': [2], 'c': [0, 2]}
        output = prep_cui_reports(actual_cui_set, predicted_cui_set)
        self.assertEqual(len(output), 3)
        self.assertIsInstance(output[0], MachineAnnotation)
        self.assertEqual(len(output[0]), 5)

        expected_actual_cats = [3, 3, 2, 1, 2]
        self.assertEqual(output[1], expected_actual_cats)
        expected_predicted_cats = [0, 3, 0, 2, 2]
        self.assertEqual(output[2], expected_predicted_cats)
