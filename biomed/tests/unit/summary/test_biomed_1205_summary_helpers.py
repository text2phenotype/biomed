import unittest

from biomed.cancer.cancer_represent import Grade, Qualifier
from text2phenotype.constants.features.label_types import CancerLabel

from biomed.common.biomed_ouput import SummaryOutput, CancerOutput
from biomed.common.aspect_response import AspectResponse, CancerResponse
from biomed.common.biomed_summary import FullSummaryResponse


class TestSummaryHelpers(unittest.TestCase):
    def test_merge_nearby_terms(self):
        input_data = AspectResponse('DiseaseDisorder',
                                    [SummaryOutput(text='shortness', label='problem', range=[421, 430],
                                                   attributes={'polarity': 'positive'}, lstm_prob=0.973,
                                                   umlsConcept={'code': '230145002', 'cui': 'C0013404',
                                                                'tui': 'T184', 'codingScheme': 'SNOMEDCT_US',
                                                                'preferredText': 'Dyspnea'}),
                                     SummaryOutput(text='of', label='problem', range=[431, 433],
                                                   attributes={'polarity': 'positive'}, lstm_prob=0.976,
                                                   umlsConcept={'code': '230145002', 'cui': 'C0013404',
                                                                'tui': 'T184', 'codingScheme': 'SNOMEDCT_US',
                                                                'preferredText': 'Dyspnea'}),
                                     SummaryOutput(text='breath', label='problem', range=[434, 440],
                                                   attributes={'polarity': 'positive'},
                                                   lstm_prob=0.998,
                                                   umlsConcept={'code': '230145002', 'cui': 'C0013404',
                                                                'tui': 'T184', 'codingScheme': 'SNOMEDCT_US',
                                                                'preferredText': 'Dyspnea'})])
        expected = AspectResponse('DiseaseDisorder', [SummaryOutput(text='shortness of breath',
                                                                    label='problem',
                                                                    range=[421, 440],
                                                                    attributes={'polarity': 'positive'},
                                                                    lstm_prob=0.998,
                                                                    umlsConcept={'code': '230145002', 'cui': 'C0013404',
                                                                                 'tui': 'T184',
                                                                                 'codingScheme': 'SNOMEDCT_US',
                                                                                 'preferredText': 'Dyspnea'})])
        input_data.merge_nearby_terms(f"{'a' * 421}shortness of breath")

        self.assertEqual(input_data.to_json(), expected.to_json())

    def test_pick_max_prob_duplicate(self):
        input_data = FullSummaryResponse([AspectResponse('DiseaseDisorder', [
            SummaryOutput(**{'code': '271789005', 'cui': 'C0012833', 'label': 'problem', 'lstm_prob': 0.5,
                             'polarity': 'positive', 'preferredText': 'Dizziness',
                             'range': [35, 44], 'text': 'Dizziness', 'tui':
                                 'T184', 'vocab': 'SNOMEDCT_US'})]),
                                          AspectResponse('SignSymptom', [SummaryOutput(
                                              **{'code': '271789005', 'cui': 'C0012833', 'label': 'problem',
                                                 'lstm_prob': 0.35,
                                                 'polarity': 'positive', 'preferredText': 'Dizziness',
                                                 'range': [35, 44],
                                                 'text': 'Dizziness', 'tui': 'T184', 'vocab': 'SNOMEDCT_US'})])])
        input_data.remove_summary_duplicates()
        output = input_data.to_json()

        self.assertEqual(len(output['SignSymptom']), 0)

    def test_pick_equal_prob_duplicate(self):
        # assert that in case of tie we call it a signsymptom
        input_data = FullSummaryResponse([AspectResponse('DiseaseDisorder', [
            SummaryOutput(**{'code': '271789005', 'cui': 'C0012833', 'label': 'problem', 'lstm_prob': 0.5,
                             'polarity': 'positive', 'preferredText': 'Dizziness',
                             'range': [35, 44], 'text': 'Dizziness', 'tui':
                                 'T184', 'vocab': 'SNOMEDCT_US'})]),
                                          AspectResponse('SignSymptom', [SummaryOutput(
                                              **{'code': '271789005', 'cui': 'C0012833', 'label': 'problem',
                                                 'lstm_prob': 0.5,
                                                 'polarity': 'positive', 'preferredText': 'Dizziness',
                                                 'range': [35, 44],
                                                 'text': 'Dizziness', 'tui': 'T184', 'vocab': 'SNOMEDCT_US'})])])
        input_data.remove_summary_duplicates()
        output = input_data.to_json()

        self.assertEqual(len(output['DiseaseDisorder']), 0)

    def test_remove_multiple_duplicates(self):
        # assert that in case of tie we call it a signsymptom
        input_data = FullSummaryResponse([
            AspectResponse('DiseaseDisorder', [
                SummaryOutput(**{'code': '271789005', 'cui': 'C0012833', 'label': 'problem', 'lstm_prob': 0.15,
                                 'polarity': 'positive', 'preferredText': 'Dizziness',
                                 'range': [35, 44], 'text': 'Dizziness', 'tui':
                                     'T184', 'vocab': 'SNOMEDCT_US'}),
                SummaryOutput(**{'code': '271789005', 'cui': 'C0012833', 'label': 'problem', 'lstm_prob': 0.15,
                                 'polarity': 'positive', 'preferredText': 'Dizziness',
                                 'range': [35, 44], 'text': 'Dizziness', 'tui':
                                     'T184', 'vocab': 'SNOMEDCT_US'})]),
            AspectResponse('SignSymptom', [SummaryOutput(
                **{'code': '271789005', 'cui': 'C0012833', 'label': 'problem',
                   'lstm_prob': 0.35,
                   'polarity': 'positive', 'preferredText': 'Dizziness',
                   'range': [35, 44],
                   'text': 'Dizziness', 'tui': 'T184', 'vocab': 'SNOMEDCT_US'}),
                SummaryOutput(**{'code': '271789005', 'cui': 'C0012833', 'label': 'problem',
                   'lstm_prob': 0.35,
                   'polarity': 'positive', 'preferredText': 'Dizziness',
                   'range': [35, 44],
                   'text': 'Dizziness', 'tui': 'T184', 'vocab': 'SNOMEDCT_US'})])])
        input_data.remove_summary_duplicates()
        output = input_data.to_json()

        self.assertEqual(len(output['DiseaseDisorder']), 0)

    def test_merge_nearby_grade_terms(self):
        category = CancerLabel.get_category_label().persistent_label
        input_data = CancerResponse(category, [
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                text='Grade',
                range=[0, 5],
                lstm_prob=.95,
                umlsConcept=Qualifier.represent(dict(), Grade.G9_unknown)),
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                range=[6, 7],
                text='1',
                lstm_prob=.91,
                umlsConcept=Qualifier.represent(dict(), Grade.G1_well)),
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                text='Grade',
                range=[8, 13],
                lstm_prob=.95,
                umlsConcept=Qualifier.represent(dict(), Grade.G2_moderate)),
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                range=[14, 15],
                text='3',
                lstm_prob=.91,
                umlsConcept=Qualifier.represent(dict(), Grade.G3_poor))])
        expected = CancerResponse(category, [
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                text='Grade 1',
                lstm_prob=.95,
                range=[0, 7],
                umlsConcept=Qualifier.represent(dict(), Grade.G1_well)),
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                text='Grade',
                range=[8, 13],
                lstm_prob=.95,
                umlsConcept=Qualifier.represent(dict(), Grade.G2_moderate)),
            CancerOutput(
                label=CancerLabel.grade.value.persistent_label,
                range=[14, 15],
                text='3',
                lstm_prob=.91,
                umlsConcept=Qualifier.represent(dict(), Grade.G3_poor))
        ])
        input_data.merge_nearby_terms(text='Grade 1')
        self.assertListEqual(input_data.to_json()[input_data.category_name],
                             expected.to_json()[input_data.category_name])
