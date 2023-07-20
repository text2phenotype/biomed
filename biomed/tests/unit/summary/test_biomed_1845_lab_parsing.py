import unittest

from biomed.common.aspect_response import LabResponse


class TestBiomed1845(unittest.TestCase):
    pos = 'positive'
    neg = 'negative'
    pending  = 'pending'

    def test_parse_lab_interp_text(self):
        self.assertEqual(LabResponse.lab_interp_parse('DETECTED CRITICAL'), self.pos)
        self.assertEqual(LabResponse.lab_interp_parse('Detected not'), self.pos)
        self.assertEqual(LabResponse.lab_interp_parse('(abnormal)'), self.pos)
        self.assertEqual(LabResponse.lab_interp_parse('positive'), self.pos)
        self.assertEqual(LabResponse.lab_interp_parse('DETECTTED ABNORMAL NOT DETECTED'), self.pos)

        self.assertEqual(LabResponse.lab_interp_parse('NEGATIVE'), self.neg)
        self.assertEqual(LabResponse.lab_interp_parse('NOT DETECTED'), self.neg)
        self.assertEqual(LabResponse.lab_interp_parse('NOT DETECTED DETECTED'), self.neg)
        self.assertEqual(LabResponse.lab_interp_parse('DETECTTED NOT DETECTED'), self.neg)

        self.assertEqual(LabResponse.lab_interp_parse('pending'), self.pending)
        self.assertEqual(LabResponse.lab_interp_parse('in process'), self.pending)
        self.assertEqual(LabResponse.lab_interp_parse('in progress'), self.pending)

        self.assertEqual(LabResponse.lab_interp_parse('AJLDSKS'), None)
        self.assertEqual(LabResponse.lab_interp_parse('detec1ed'), None)

