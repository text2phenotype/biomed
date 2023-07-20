import unittest

from biomed.common.biomed_ouput import BiomedOutput, SummaryOutput, MedOutput


class TestBiomed1245(unittest.TestCase):
    def test_no_input_to_dictionary(self):
        actual_summary = SummaryOutput().to_dict()
        actual_phi_dict = BiomedOutput().to_dict()
        actual_med_dict = MedOutput().to_dict()

        expected_phi_dict = {'text': None,
                             'range': [None, None],
                             'score': 0,
                             'label': None,
                             'page': None}
        expected_summary_dict = {**expected_phi_dict,
                                 'polarity': None,
                                 'code': None, 'cui': None, 'tui': None, 'vocab': None, 'preferredText': None}
        expected_drug_dict = {**expected_summary_dict, 'medFrequencyNumber': [], 'medFrequencyUnit': [],
                              'medStrengthNum': [], 'medStrengthUnit': [], 'date': None}
        self.assertDictEqual(actual_phi_dict, expected_phi_dict)

        self.assertDictEqual(actual_summary, expected_summary_dict)
        self.assertDictEqual(actual_med_dict, expected_drug_dict)

    def test_extract_values_from_umls_concept(self):
        actual_summary = SummaryOutput(umlsConcept=[{'cui': 12320, 'tui': 'asdf0812'}, {'code': 4}]).to_dict()

        expected_summary = {'text': None, 'range': [None, None], 'score': 0, 'label': None, 'polarity': None,
                            'code': None, 'cui': 12320, 'tui': 'asdf0812', 'vocab': None, 'preferredText': None,
                            'page': None}
        self.assertDictEqual(actual_summary, expected_summary)

    def test_extract_values_from_attribute(self):
        actual_drug_summary = MedOutput(attributes={'polarity': 'positive', 'medFrequencyNumber': ['10', 3, 10]}).to_dict()

        expected_drug_summary = {'text': None, 'range': [None, None], 'score': 0, 'label': None,
                                 'polarity': 'positive',
                                 'code': None, 'cui': None, 'tui': None, 'vocab': None, 'preferredText': None,
                                 'medFrequencyNumber': [10.0, 3, 10], 'medFrequencyUnit': [],
                                 'medStrengthNum': [], 'medStrengthUnit': [], 'date': None, 'page': None
                                 }

        self.assertDictEqual(actual_drug_summary, expected_drug_summary)
