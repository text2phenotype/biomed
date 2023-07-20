import unittest

from biomed.common.helpers import annotation_helper
from text2phenotype.common import common
from biomed.tests.samples import JOHN_STEVENS_TXT
from biomed.drug.drug import meds_and_allergies
from text2phenotype.tasks.task_enums import TaskOperation


class TestBiomed537(unittest.TestCase):

    def test_hepc_summary(self):
        text = common.read_text(JOHN_STEVENS_TXT)
        tokens, vectors = annotation_helper(text, {TaskOperation.drug})
        res = meds_and_allergies(tokens=tokens, vectors=vectors, text=text)

        med_list_simple = set([m['text'] for m in res.get('Medication')])

        for med_text in med_list_simple:

            self.assertNotIn('http', med_text)
            self.assertNotIn('...', med_text)
            self.assertNotIn('//', med_text)

            self.assertNotIn('DEMO', med_text.upper())
            self.assertNotIn('OPENEMR'.upper(), med_text.upper())
            self.assertNotIn('text2phenotype'.upper(), med_text.upper())
