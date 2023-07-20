import unittest

from text2phenotype.constants.features import PHILabel
from text2phenotype.entity.accuracy import Accuracy
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.deid.deid import get_phi_tokens
from biomed.common.helpers import annotation_helper


class TestBiomed142(unittest.TestCase):

    def test_short_text(self):

        text = """
        PATIENT NAME Andy McMurry
        DIAGNOSES negative for CKD or ESRD, no evidence of stage 4 cancer. 
        HOME ADDRESS 1315 Minna Street 94103       
        """

        expected = ['Andy', 'McMurry', '1315', 'Minna', 'Street', '94103']
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})

        actual = list()
        for token in get_phi_tokens(tokens=tokens, vectors=vectors)[PHILabel.get_category_label().persistent_label]:
            actual.append(token['text'])

        score = Accuracy().compare_sets(set(expected), set(actual))

        self.assertGreaterEqual(score.recall(), 0.8)
        self.assertGreaterEqual(score.precision(), 0.8)
