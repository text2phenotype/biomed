import unittest

from text2phenotype.constants.features import PHILabel
from text2phenotype.entity.accuracy import Accuracy
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.deid.deid import get_phi_tokens
from biomed.common.helpers import annotation_helper


class TestBiomed68(unittest.TestCase):

    def test_lab_values_similar_to_dates(self):
        text = """
        VITAL SIGNS: 
        Weight 180, height 6 feet 1 inch, blood pressure 128/67, heart rate 74, saturation 98%; 
        Pain is 3/10 with localization of the pain in the epigastric area"""
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})
        phi = get_phi_tokens(tokens, vectors=vectors)

        expected = set()
        actual = set([token['text'] for token in phi[PHILabel.get_category_label().persistent_label]])

        score = Accuracy().compare_sets(expected, actual)

        self.assertLessEqual(len(score.falsepos), 1, f"max of 1 falsepos exceeded, {score}")
