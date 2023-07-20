import unittest

from text2phenotype.constants.features import CovidLabLabel

from biomed.common.biomed_ouput import LabOutput, CovidLabOutput


class TestBiomed2117(unittest.TestCase):
    def test_covid_lab_persistent_label(self):
        self.assertEqual(CovidLabLabel.lab.value.persistent_label, 'covid_lab')

    def test_reponse_lab_name(self):
        self.assertTrue(LabOutput(label='lab').is_lab_name())
        self.assertTrue(CovidLabOutput(label='covid_lab').is_lab_name())
        self.assertFalse(LabOutput(label='a').is_lab_name())

    def test_load_legacy_brat(self):
        self.assertEqual(CovidLabLabel.from_brat('lab'), CovidLabLabel.lab)
        self.assertEqual(CovidLabLabel.from_brat('covid_lab'), CovidLabLabel.lab)
