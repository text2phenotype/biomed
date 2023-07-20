import unittest

from text2phenotype.constants.features import DemographicEncounterLabel

from biomed.common.biomed_ouput import BiomedOutput
from biomed.demographic.demographics_manipulation import FetchedDemographics


class TestDemographicNames(unittest.TestCase):
    demographics_out  = [
        BiomedOutput(
            text='Granger', lstm_prob=.7, range=[0, 7],
            label=DemographicEncounterLabel.pat_last.value.persistent_label),
        BiomedOutput(
            text='Hermione', lstm_prob=.87, range=[10, 18],
            label=DemographicEncounterLabel.pat_first.value.persistent_label),
        BiomedOutput(
            text='Jean', lstm_prob=.87, range=[20, 24],
            label=DemographicEncounterLabel.pat_first.value.persistent_label),
        BiomedOutput(
            text='Hermione', lstm_prob=.87, range=[200, 208],
            label=DemographicEncounterLabel.pat_first.value.persistent_label),
        BiomedOutput(
            text='Jean', lstm_prob=.87, range=[210, 214],
            label=DemographicEncounterLabel.pat_middle.value.persistent_label),
        BiomedOutput(
            text='Granger', lstm_prob=.7, range=[218, 223],
            label=DemographicEncounterLabel.pat_last.value.persistent_label)
    ]

    def test_collapse_suggestions(self):
        collapsed = FetchedDemographics.collapse_demographics_by_range(self.demographics_out)
        expected =  [
        BiomedOutput(
            text='Granger', lstm_prob=.7, range=[0, 7],
            label=DemographicEncounterLabel.pat_last.value.persistent_label),
        BiomedOutput(
            text='Hermione Jean', lstm_prob=.87, range=[10, 24],
            label=DemographicEncounterLabel.pat_first.value.persistent_label),
        BiomedOutput(
            text='Hermione', lstm_prob=.87, range=[200, 208],
            label=DemographicEncounterLabel.pat_first.value.persistent_label),
        BiomedOutput(
            text='Jean', lstm_prob=.87, range=[210, 214],
            label=DemographicEncounterLabel.pat_middle.value.persistent_label),
        BiomedOutput(
            text='Granger', lstm_prob=.7, range=[218, 223],
            label=DemographicEncounterLabel.pat_last.value.persistent_label)
    ]
        for i in range(len(expected)):
            self.assertEqual(expected[i].to_dict(), collapsed[i].to_dict(), f'Suggestions not equal for idx {i}')

    def test_collapse_names(self):
        collapsed_dem = FetchedDemographics.collapse_demographics_by_range(self.demographics_out)
        collapsed_names = FetchedDemographics.collapse_names(collapsed_dem)
        expected_names =[
            {'text': 'Hermione Jean Granger', 'range': [10, 7],'score': 0.7849999999999999, 'label': 'pat_name', 'page': None},
            {'text': 'Hermione Jean Granger', 'range': [200, 223], 'score': 0.8133333333333334, 'label': 'pat_name', 'page': None}]
        self.assertEqual(len(collapsed_names), 2)
        for i in range(len(expected_names)):
            self.assertEqual(expected_names[i], collapsed_names[i].to_dict())
