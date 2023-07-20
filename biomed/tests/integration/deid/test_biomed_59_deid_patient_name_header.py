import unittest

from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import annotation_helper
from biomed.deid.deid import get_phi_tokens


class TestBiomed59(unittest.TestCase):

    def assertPatientName(self, text):
        tokens, vectors = annotation_helper(text, {TaskOperation.phi_tokens})
        result = get_phi_tokens(tokens, vectors=vectors)['PHI']

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['text'], 'Andy')
        self.assertEqual(result[1]['text'], 'McMurry')

    def test_patient_name(self):
        self.assertPatientName("PATIENT: Andy McMurry")
        self.assertPatientName("PATIENT NAME: Andy McMurry")
        self.assertPatientName("Patient Name: Andy McMurry")
        self.assertPatientName("Patient Name: is Andy McMurry")
        self.assertPatientName("Patient Name is Andy McMurry")
        self.assertPatientName("Patient Name Andy McMurry")
