import unittest

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.constants.features import CovidLabLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.biomed_env import BiomedEnv
from biomed.common.helpers import feature_list_helper
from biomed.lab.labs import get_covid_labs
from biomed.device_procedure.api import device_procedure_predict_represent
from biomed.imaging_finding.imaging_finding import imaging_and_findings
from biomed.summary.text_to_summary import summary_from_parts, VERSION_INFO_KEY


class TestCovidTestResults(unittest.TestCase):
    txt = """plexbio
July 19, 2020
Test(s) SARS-CoV-2, NAA 


Ordered Items
SARS-CoV-2, NAA

TESTS UNITS REFERENCE INIERVAL LAB

SARS-CoV-2, NAA Detected Critical Not Detected 

Enzo Life Sciences

FDA independent review of this validation is pending. This test is
only authorized for the duration of time the declaration that
circumstances exist justifying the authorization of the emergency use
of in vitro diagnostic tests for detection of SARS-CoV-2 virus and/or
diagnosis of COVID-19 infection under section 564(b) (1) of the Act, 21
U.S.C. 360bbb-3(b) (1), unless the authorization is terminated or
revoked sooner.
"""
    COVID_LAB = CovidLabLabel.get_category_label().persistent_label

    def test_covid_yn(self):
        annotation, vectorization = FeatureServiceClient().annotate_vectorize(
            text=self.txt,
            features=feature_list_helper(
                {TaskOperation.covid_specific}))
        covid_labs = get_covid_labs(tokens=annotation, vectors=vectorization, text=self.txt)
        imaging = imaging_and_findings(tokens=annotation, vectors=vectorization, text=self.txt)
        device = device_procedure_predict_represent(tokens=annotation, vectors=vectorization, text=self.txt)
        covid_results = summary_from_parts([covid_labs, imaging, device], text=self.txt)

        expected_covid_summary = {
            'CovidLabs': [
                {'text': 'SARS-CoV-2, NAA',
                 'range': [30, 45],
                 'score': 0.9663663506507874,
                 'label': 'covid_lab',
                 'polarity': None,
                 'code': '840533007',
                 'cui': 'C5203676',
                 'tui': None,
                 'vocab': 'SNOMEDCT_US',
                 'preferredText': '2019-nCoV',
                 'date': '2020-07-19',
                 'labValue': [],
                 'labUnit': [],
                 'labInterp': [],
                 'labManufacturer': ['plexbio', 0, 7],
                 'page': 1},
                {'text': 'SARS-CoV-2, NAA',
                 'range': [63, 78],
                 'score': 0.9895656108856201,
                 'label': 'covid_lab',
                 'polarity': None,
                 'code': '840533007',
                 'cui': 'C5203676',
                 'tui': None,
                 'vocab': 'SNOMEDCT_US',
                 'preferredText': '2019-nCoV',
                 'date': '2020-07-19',
                 'labValue': [],
                 'labUnit': [],
                 'labInterp': [],
                 'labManufacturer': ['plexbio', 0, 7],
                 'page': 1},
                {'text': 'SARS-CoV-2, NAA',
                 'range': [116, 131],
                 'score': 0.9730013012886047,
                 'label': 'covid_lab',
                 'polarity': 'positive',
                 'code': '840533007',
                 'cui': 'C5203676',
                 'tui': None,
                 'vocab': 'SNOMEDCT_US',
                 'preferredText': '2019-nCoV',
                 'date': '2020-07-19',
                 'labValue': [],
                 'labUnit': [],
                 'labInterp': ['Detected Critical Not', 132, 153],
                 'labManufacturer': ['Enzo Life Sciences', 165, 183],
                 'page': 1}],
            'Findings': [],
            'DiagnosticImaging': [],
            'Device/Procedure': []}

        self.assert_biomed_out_minus_score_equal(covid_results, expected_covid_summary)

    def assert_biomed_out_minus_score_equal(self, biomed_out, expected):
        for key in biomed_out:
            if key != VERSION_INFO_KEY:
                for i in range(len(expected[key])):
                    biomed_out_entry = biomed_out[key][i]
                    expected_entry = expected[key][i]
                    if 'score' in biomed_out_entry:
                        biomed_out_entry.pop('score')
                    if 'score' in expected_entry:
                        expected_entry.pop('score')
                    self.assertDictEqual(biomed_out_entry, expected_entry)
