import unittest

from biomed.common.helpers import annotation_helper
from biomed.tests.fixtures.example_file_paths import stephan_garcia_txt_filepath
from biomed.drug.drug import meds_and_allergies
from text2phenotype.common import common
from text2phenotype.tasks.task_enums import TaskOperation


class TestBiomed467(unittest.TestCase):

    def test_stephan_garcia_allergic_to_erythromycin(self):

        text = common.read_text(stephan_garcia_txt_filepath)
        tokens,  vectors = annotation_helper(text,  {TaskOperation.drug})


        res = meds_and_allergies(tokens=tokens, vectors=vectors, text=text)

        self.assertIn('Allergy', res.keys())

        allergies = list(set([concept['preferredText'] for concept in res['Allergy']]))

        self.assertEqual(len(allergies), 1)

        self.assertIn('Erythromycin', allergies)
