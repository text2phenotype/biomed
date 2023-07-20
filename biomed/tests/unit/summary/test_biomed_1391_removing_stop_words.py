from unittest import TestCase

from biomed.common.biomed_ouput import SummaryOutput
from biomed.common.aspect_response import AspectResponse


class TestBiomed1391(TestCase):
    def test_remove_stop_words_summary_output(self):
        input = AspectResponse('abc', [SummaryOutput(text='On', lstm_prob=0.1), SummaryOutput(text='in', lstm_prob=0.1),
                                       SummaryOutput(text='mr', lstm_prob=0.1), SummaryOutput(text='Mrs.', lstm_prob=0.1)])
        input.filter_responses(min_score=0.01)
        self.assertEqual(len(input.to_json()['abc']), 0)

    def test_remove_som_stop_words(self):
        input = AspectResponse('abc', [SummaryOutput(text='Cancer', lstm_prob=0.1), SummaryOutput(text='in', lstm_prob=0.1),
                                       SummaryOutput(text='aspirin', lstm_prob=0.1), SummaryOutput(text='mr', lstm_prob=0.1),
                         SummaryOutput(text='Mrs.')])
        input.filter_responses(min_score=0.01)
        self.assertEqual(len(input.to_json()['abc']), 2)