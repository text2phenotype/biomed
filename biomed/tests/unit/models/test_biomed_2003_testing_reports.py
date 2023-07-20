import unittest
import numpy
import pandas

from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.features import FeatureType, LabLabel

from biomed.models.testing_reports import (
    WeightedReport, RemovingAdjacentConfusion, ConfusionPrecisionMisclassReport, MinusPartialAnnotation, CuiReport,
    DemographicsReport, FullReport, _safe_div)
from biomed.common.model_test_helpers import document_cui_set, prep_cui_reports


class TestReportClass(unittest.TestCase):
    def assert_matrix_initialization_dimensions(self, report_class: ConfusionPrecisionMisclassReport):
        report_obj = report_class(label_enum=LabLabel)
        self.assertEqual(report_obj.confusion_matrix.shape, (5, 5))

    def test_matrix_initialization_dimensions(self):
        self.assert_matrix_initialization_dimensions(RemovingAdjacentConfusion)
        self.assert_matrix_initialization_dimensions(WeightedReport)
        self.assert_matrix_initialization_dimensions(MinusPartialAnnotation)

    def test_binary_confusion_matrix(self):
        report_obj = WeightedReport(LabLabel)
        report_obj.confusion_matrix = numpy.array([[3, 1, 0, 2, 0], [4, 1, 0, 0, 0], [0, 1, 0, 6, 0], [0, 3, 0, 0, 0],
                                     [0, 2, 0, 0, 0]])
        actual_out = report_obj.binary_confusion_matrix
        expected = numpy.array([[3, 3], [4, 13]])
        self.assertTrue((actual_out == expected).all())

    def test_binary_precision(self):
        report_obj = WeightedReport(LabLabel)
        report_obj.confusion_matrix = numpy.array([[3, 1, 0, 2, 0], [4, 1, 0, 0, 0], [0, 1, 0, 6, 0], [0, 3, 0, 0, 0],
                                                   [0, 2, 0, 0, 0]])
        actual = report_obj.precision_recall_report(binary_report=True)
        self.assertIsInstance(actual, pandas.DataFrame)
        actual_str = report_obj.scipy_report_str(actual)
        expected_precis = '           precision    recall  f1-score   support\n\n' \
                          '       na      0.429     0.500     0.462         6\n' \
                          '   non-na      0.812     0.765     0.788        17\n\n' \
                          'avg/total      0.812     0.765     0.788        17\n'

        self.assertEqual(actual_str, expected_precis)

    def test_add_to_confusion_matrix(self):
        report_obj = WeightedReport(LabLabel)
        predicted_cat = [0, 1, 0, 1, 0, 1]
        actual_cat = [0, 0, 0, 1, 0, 2]
        expected_confusion_matrix = [[3, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0]]
        self.assert_added_to_confusion_matrix(
            report_obj, predicted_cat=predicted_cat, actual_cat=actual_cat,
            expected_confusion_matrix=expected_confusion_matrix)

        second_predicted_cat = [4, 3, 1]
        second_actual_cat = [3, 4, 2]
        second_expected = numpy.array(
            [[3, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
        self.assert_added_to_confusion_matrix(
            report_obj, predicted_cat=second_predicted_cat, actual_cat=second_actual_cat,
            expected_confusion_matrix=second_expected)

    def assert_added_to_confusion_matrix(self, report_obj, predicted_cat, actual_cat, expected_confusion_matrix):
        report_obj.add_to_confusion_matrix(valid_predicted=predicted_cat, valid_expected=actual_cat)
        for i in range(len(expected_confusion_matrix)):
            for j in range(len(expected_confusion_matrix)):
                self.assertEqual(
                    expected_confusion_matrix[i][j],
                    report_obj.confusion_matrix[i][j],
                    f'Unequal Values at indexes {i}, {j}. '
                    f'{expected_confusion_matrix[i][j]} != {report_obj.confusion_matrix[i, j]}')

    def test_add_to_misclasification_report(self):
        report_obj = WeightedReport(LabLabel)
        predicted_cat = [0, 1, 0, 1, 0, 1]
        predicted_prob = numpy.array([
            [.9, .1, 0, 0, 0], [.1, .8, .1, 0, 0], [.9, .1, 0, 0, 0],
            [.4, .6, 0, 0, 0], [.9, .1, 0, 0, 0], [.25, .4, .35, 0, 0]])
        actual_cat = [0, 0, 0, 1, 0, 2]
        token = MachineAnnotation(json_dict_input={
            'token': ['a', 'mean', 'bird', 'ate', 'my', 'icecream'],
            'range': [(0, 1), (1, 4), (5, 4), (9, 3), (12, 2), (14, 8)]})
        filename = 'abc.txt'
        report_obj.add_document_misclass(
            expected_category=actual_cat,
            predicted_results_prob=predicted_prob,
            predicted_results_cat=predicted_cat,
            tokens=token,
            valid_token_indexes={0, 1, 2, 3, 4, 5},
            filename=filename

        )

        actual_misclass = report_obj.misclassification_report[1:]
        expected = [
            ['0', '0.1000', '1', '0.8000', 'mean', filename, '1', '4'],
            ['2', '0.3500', '1', '0.4000', 'icecream', filename, '14', '8']
        ]

        self.assertListEqual(actual_misclass, expected)

    def test_confusion_matrix_to_report(self):
        report_obj = WeightedReport(LabLabel)
        predicted_cat = [0, 1, 0, 1, 0, 1]
        actual_cat = [0, 0, 0, 1, 0, 2]
        report_obj.add_to_confusion_matrix(valid_predicted=predicted_cat, valid_expected=actual_cat)

        precision_recall = report_obj.precision_recall_report()
        self.assertIsInstance(precision_recall, pandas.DataFrame)
        expected_string = '            precision    recall  f1-score   support\n\n' \
                          '        na      1.000     0.750     0.857         4\n' \
                          '       lab      0.333     1.000     0.500         1\n' \
                          ' lab_value      0.000     0.000     0.000         1\n' \
                          '  lab_unit      0.000     0.000     0.000         0\n' \
                          'lab_interp      0.000     0.000     0.000         0\n\n' \
                          ' avg/total      0.167     0.500     0.250         2\n'


        self.assertEqual(report_obj.scipy_report_str(precision_recall), expected_string)

    def test_add_cui_misclasification(self):
        report_obj = CuiReport(label_enum=LabLabel)

        actual_cui_set = {'100': [3, 0], 'abc': [3], 'a': [2], 'b': [0, 1], 'c': [1, 2, 3]}
        predicted_cui_set = {'100': [0], 'abc': [3], 'a': [0], 'b': [2], 'c': [0, 2]}
        output = prep_cui_reports(actual_cui_set, predicted_cui_set)

        filename = 'abc.txt'
        report_obj.add_document_misclass(
            expected_category=output[1], predicted_results_cat=output[2], tokens=output[0], filename=filename,
            valid_token_indexes=set(range(len(actual_cui_set))))

        actual_report = report_obj.misclassification_report[1:]
        expected = [
            ['3', '0', '100', filename],
            ['2', '0', 'a', filename],
            ['1', '2', 'b', filename]
        ]
        self.assertListEqual(actual_report, expected)


class TestAddingDocuments(unittest.TestCase):
    TOKEN_TEXT = ['Hello', 'World', 'Patient:', 'Dr.', 'Meredith', 'Grey', 'is', 'a', '43yO', 'woman', 'presenting',
                  'with', 'sob']
    ANNOTATION = MachineAnnotation(
        json_dict_input={
            'token': TOKEN_TEXT,
            'range': [(0, 5), (5, 5), (10, 8), (18, 3), (21, 8), (29, 4), (33, 2), (35, 1), (36, 4), (40, 5), (45, 10),
                      (55, 4), (59, 3)],
            FeatureType.clinical.name: {
                "3": [{"SemGroupA": [{'cui': '100'}, {'cui': 'abc'}]}], "4": [{"SemGroupA": [{'cui': '100'}]}],
            }
        })
    PREDICTED_CATEGORY = numpy.array([0, 0, 0, 0, 3, 0, 3, 2, 1, 1, 0, 0, 0])
    ACTUAL_CATEGORY = [4, 0, 0, 3, 3, 0, 0, 2, 0, 1, 1, 1, 0]
    CONCEPT_FEATURE_MAPPING = {FeatureType.clinical: ['SemGroupA']}
    PREDICTED_PROBABILITY = numpy.array(
        [[.9, 0, 0, .1, 0],
         [.99, 0.01, 0, 0, 0],
         [.9, 0, 0, .1, 0],
         [.9, 0, 0, .1, 0],
         [.2, 0, 0, .8, 0],
         [.9, 0, 0, .1, 0],
         [.2, 0, .3, .4, 0],
         [.2, 0, .7, .1, 0],
         [.2, 0.4, .3, .1, 0],
         [.2, 0.4, .3, .1, 0],
         [.9, 0, 0, .1, 0],
         [.9, 0, 0, .1, 0],
         [.9, 0, 0, .1, 0]])

    def assert_reports_result(self, report_obj, expected_confusion_matrix, expected_precis_recall_str,
                              expected_misclass_report):
        report_obj.add_document(
            expected_category=self.ACTUAL_CATEGORY,
            predicted_results_cat=self.PREDICTED_CATEGORY,
            predicted_results_prob=self.PREDICTED_PROBABILITY,
            tokens=self.ANNOTATION,
            duplicate_token_idx={},
            filename='abc.txt'
        )

        # test adding second doc w/ new duplicate tokens
        report_obj.add_document(
            expected_category=self.ACTUAL_CATEGORY,
            predicted_results_cat=self.PREDICTED_CATEGORY,
            predicted_results_prob=self.PREDICTED_PROBABILITY,
            tokens=self.ANNOTATION,
            duplicate_token_idx={0, 1, 2, 3, 4, 5, 6},
            filename='123.txt'
        )
        self.assertEqual(report_obj.scipy_report_str(report_obj.precision_recall_report()), expected_precis_recall_str)
        self.assertListEqual(report_obj.misclassification_report[1:], expected_misclass_report)
        self.assertTrue((expected_confusion_matrix == report_obj.confusion_matrix).all())

    def test_weighted_report(self):
        expected_confusion_matrix = numpy.array(
            [[5, 2, 0, 0, 0],
             [2, 2, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 0, 0]])
        expected_preci_recall_str = '            precision    recall  f1-score   support\n\n' \
                                    '        na      0.556     0.714     0.625         7\n' \
                                    '       lab      0.500     0.500     0.500         4\n' \
                                    ' lab_value      0.000     0.000     0.000         0\n' \
                                    '  lab_unit      1.000     0.500     0.667         2\n' \
                                    'lab_interp      0.000     0.000     0.000         1\n\n' \
                                    ' avg/total      0.571     0.429     0.476         7\n'

        expected_misclass_report = [
            ['4', '0.0000', '0', '0.9000', 'Hello', 'abc.txt', '0', '5'],
            ['3', '0.1000', '0', '0.9000', 'Dr.', 'abc.txt', '18', '3'],
            ['0', '0.2000', '1', '0.4000', '43yO', 'abc.txt', '36', '4'],
            ['1', '0.0000', '0', '0.9000', 'presenting', 'abc.txt', '45', '10'],
            ['0', '0.2000', '1', '0.4000', '43yO', '123.txt', '36', '4'],
            ['1', '0.0000', '0', '0.9000', 'presenting', '123.txt', '45', '10']
        ]

        report_obj = WeightedReport(LabLabel)
        self.assert_reports_result(report_obj, expected_confusion_matrix, expected_preci_recall_str,
                                   expected_misclass_report)

    def test_adjacent_report(self):
        expected_confusion_matrix = numpy.array(
            [[5, 0, 0, 0, 0],
             [0, 2, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0]])
        expected_preci_recall_str = '            precision    recall  f1-score   support\n\n' \
                                    '        na      0.833     1.000     0.909         5\n' \
                                    '       lab      1.000     1.000     1.000         2\n' \
                                    ' lab_value      0.000     0.000     0.000         0\n' \
                                    '  lab_unit      1.000     1.000     1.000         1\n' \
                                    'lab_interp      0.000     0.000     0.000         1\n\n' \
                                    ' avg/total      0.750     0.750     0.750         4\n'

        expected_misclass_report = [['4', '0.0000', '0', '0.9000', 'Hello', 'abc.txt', '0', '5']]

        report_obj = RemovingAdjacentConfusion(LabLabel)
        self.assert_reports_result(report_obj, expected_confusion_matrix, expected_preci_recall_str,
                                   expected_misclass_report)

    def test_minus_partial_report(self):
        expected_confusion_matrix = numpy.array(
            [[5, 2, 0, 0, 0],
             [2, 2, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0]])
        expected_preci_recall_str = '            precision    recall  f1-score   support\n\n' \
                                    '        na      0.625     0.714     0.667         7\n' \
                                    '       lab      0.500     0.500     0.500         4\n' \
                                    ' lab_value      0.000     0.000     0.000         0\n' \
                                    '  lab_unit      1.000     1.000     1.000         1\n' \
                                    'lab_interp      0.000     0.000     0.000         1\n\n' \
                                    ' avg/total      0.500     0.500     0.500         6\n'


        expected_misclass_report = [
            ['4', '0.0000', '0', '0.9000', 'Hello', 'abc.txt', '0', '5'],
            ['0', '0.2000', '1', '0.4000', '43yO', 'abc.txt', '36', '4'],
            ['1', '0.0000', '0', '0.9000', 'presenting', 'abc.txt', '45', '10'],
            ['0', '0.2000', '1', '0.4000', '43yO', '123.txt', '36', '4'],
            ['1', '0.0000', '0', '0.9000', 'presenting', '123.txt', '45', '10']
        ]

        report_obj = MinusPartialAnnotation(LabLabel, self.CONCEPT_FEATURE_MAPPING)
        self.assert_reports_result(report_obj, expected_confusion_matrix, expected_preci_recall_str,
                                   expected_misclass_report)

    def test_cui_add_document(self):
        report_obj = CuiReport(LabLabel)
        predicted_cuis = document_cui_set(self.PREDICTED_CATEGORY, self.CONCEPT_FEATURE_MAPPING, tokens=self.ANNOTATION)
        actual_cuis = document_cui_set(self.ACTUAL_CATEGORY, self.CONCEPT_FEATURE_MAPPING, tokens=self.ANNOTATION)
        cui_tokens, actual_cui_cat, predicted_cui_cat = prep_cui_reports(
            actual_cuis, predicted_cuis)
        report_obj.add_document(
            expected_category=actual_cui_cat,
            predicted_results_cat=predicted_cui_cat,
            tokens=cui_tokens,
            duplicate_token_idx=None)

        expected_confusion_matrix = numpy.array([
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0, 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.]])
        expected_misclasification_report = []

        expected_precision_recall_report = '            precision    recall  f1-score   support\n\n' \
                                           '        na      0.000     0.000     0.000         0\n' \
                                           '       lab      0.000     0.000     0.000         0\n' \
                                           ' lab_value      0.000     0.000     0.000         0\n' \
                                           '  lab_unit      1.000     1.000     1.000         1\n' \
                                           'lab_interp      0.000     0.000     0.000         0\n\n' \
                                           ' avg/total      1.000     1.000     1.000         1\n'

        self.assertEqual(
            report_obj.scipy_report_str(
                report_obj.precision_recall_report()),
            expected_precision_recall_report)

        self.assertEqual(report_obj.misclassification_report[1:], expected_misclasification_report)
        self.assertTrue((expected_confusion_matrix == report_obj.confusion_matrix).all())

    def test_demographic_report(self):
        report_obj = DemographicsReport()
        report_obj.add_document(
            expected_category=self.ACTUAL_CATEGORY,
            predicted_results_cat=self.PREDICTED_CATEGORY,
            predicted_results_prob=self.PREDICTED_PROBABILITY,
            tokens=self.ANNOTATION,
            duplicate_token_idx={},
            filename='abc.txt'
        )

        actual = report_obj.dem_comparison_results
        expected_keys = {
            'ssn', 'mrn', 'pat_first', 'pat_middle', 'pat_last', 'pat_initials', 'pat_age', 'pat_street', 'pat_zip',
            'pat_city', 'pat_state', 'pat_phone', 'pat_email', 'insurance', 'facility_name', 'sex', 'dob', 'dr_first',
            'dr_middle', 'dr_last', 'dr_initials', 'dr_age', 'dr_street', 'dr_zip', 'dr_city', 'dr_state', 'dr_phone',
            'dr_fax', 'dr_email', 'dr_id', 'dr_org', 'ethnicity', 'race', 'language', 'pat_full_name', 'dr_full_names'}

        self.assertEqual(set(actual.keys()), expected_keys)
        for key in expected_keys:
            # assert one entry for all categories after adding one document
            self.assertEqual(len(actual[key]), 1)
            # assert len  = 3 for true poitive, false_positive, false_negative
            self.assertEqual(len(actual[key][0]), 3)

        report_df = report_obj.demographics_report_df()
        self.assertIsInstance(report_df, pandas.DataFrame)
        self.assertEqual(set(report_df.columns), {'precision', 'recall', 'f1', 'support'})
        self.assertEqual(len(report_df), len(expected_keys))

    def test_full_report(self):
        report_obj = FullReport(label_enum=LabLabel)

        expected_doc = {'tokens': self.ANNOTATION.tokens, 'labels': self.ACTUAL_CATEGORY,
                        'predicted': self.PREDICTED_CATEGORY, 'prob': self.PREDICTED_PROBABILITY,
                        'raw_prob': self.PREDICTED_PROBABILITY}
        expected_dict = {'abc.txt': expected_doc}

        report_obj.add_document(
            tokens=self.ANNOTATION,
            duplicate_token_idx=None,
            expected_category=self.ACTUAL_CATEGORY,
            predicted_results_cat=self.PREDICTED_CATEGORY,
            predicted_results_prob=self.PREDICTED_PROBABILITY,
            raw_probs=self.PREDICTED_PROBABILITY,
            filename='abc.txt'
        )

        self.assertEqual(expected_dict, report_obj.doc_info)


class TestSafeDiv(unittest.TestCase):
    def test_safe_div(self):
        self.assertEqual(2, _safe_div(4, 2))
        self.assertEqual(0.5, _safe_div(1, 2))
        self.assertEqual(0, _safe_div(0, 2))
        self.assertEqual(0, _safe_div(1, 0))
        self.assertEqual(0, _safe_div(0, 0))


if __name__ == '__main__':
    unittest.main()
