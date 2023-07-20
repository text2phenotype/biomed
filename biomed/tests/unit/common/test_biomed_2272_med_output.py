import unittest

from biomed.common.biomed_ouput import MedOutput


class TestBiomed2272MedOutput(unittest.TestCase):
    def test_parsing_when_strength_already_exists(self):
        entry = MedOutput(
            text='Aspirin', preferredText='Aspirin 500 MG', range=[115, 122],
            medStrengthNum=['500', 123, 125], medStrengthUnit=['mg', 127, 129])
        expected_dict = {
            'text': 'Aspirin',
            'range': [115, 122],
            'score': 0,
            'label': None,
            'polarity': None,
            'code': None,
            'cui': None,
            'tui': None,
            'vocab': None,
            'preferredText': 'Aspirin 500 MG',
            'date': None,
            'medFrequencyNumber': [],
            'medFrequencyUnit': [],
            'medStrengthNum': [500.0, 123, 125],
            'medStrengthUnit': ['mg', 127, 129],
            'page': None

        }
        self.assertEqual(entry.to_dict(), expected_dict)

    def test_parse_from_pref_text(self):
        entry = MedOutput(
            text='Aspirin', preferredText='Aspirin 500 MG', range=[115, 122])
        expected_dict = {
            'text': 'Aspirin',
            'range': [115, 122],
            'score': 0,
            'label': None,
            'polarity': None,
            'code': None,
            'cui': None,
            'tui': None,
            'vocab': None,
            'preferredText': 'Aspirin 500 MG',
            'date': None,
            'medFrequencyNumber': [],
            'medFrequencyUnit': [],
            'medStrengthNum': [500.0, -1, -1],
            'medStrengthUnit': ['MG', -1, -1],
            'page': None

        }
        self.assertEqual(entry.to_dict(), expected_dict)


