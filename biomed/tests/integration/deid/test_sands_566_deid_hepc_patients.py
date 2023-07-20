import unittest

from text2phenotype.common import common
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.deid.deid import get_phi_tokens
from biomed.tests.samples import HCV_CONSULT_NOTE_TXT, RICARDO_HPI_TXT
from biomed.common.helpers import annotation_helper


class TestApp566(unittest.TestCase):

    def assertDEID(self, hcv_file, expected):
        """
        ECHO provided us this example from their EMR training system
        """
        text = common.read_text(hcv_file)
        tokens,  vectors = annotation_helper(text, {TaskOperation.phi_tokens
                                                    })
        phi = get_phi_tokens(tokens, vectors=vectors)
        phi_tokens = [token['text'] for token in phi['PHI']]

        for token in expected:
            self.assertTrue(token in phi_tokens, f'deid missed important PHI {token}, actual was {phi_tokens}')

    def test_consult_note(self):
        expected = ['Bob',  # patient
                    '9-26-1965',  # dob
                    '85110083',  # mrn
                    '08/24/2017',  # encounter
                    'Kevin',  # Provider, PCP
                    'Henry',  # Provider, PCP
                    'Sanjeev',  # Provider, specialist
                    'Arora']  # Provider, specialist

        self.assertDEID(HCV_CONSULT_NOTE_TXT, expected)

    def test_deid_ricardo_campos(self):
        expected = ['Ricardo',
                    'Campos',
                    '13203',
                    '01/02/1963',
                    'May', '2017',
                    '05/24/2017']

        self.assertDEID(RICARDO_HPI_TXT, expected)

    def tet_biomed_514_empty_text(self):
        """
        BIOMED-514 "get_phi_tokens() throws an exception when given empty text, it should instead return
        the same thing it would if it found no phi tokens (an empty list).'
        """
        text = ""
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})
        phi_tokens = get_phi_tokens(tokens, vectors=vectors)

        self.assertIsInstance(phi_tokens, list)
        self.assertEqual(0, len(phi_tokens))
