import unittest
import json

from biomed.summary.text_to_summary import summary_from_parts
from text2phenotype.apiclients import FeatureServiceClient

from biomed.drug.drug import meds_and_allergies
from biomed.lab.labs import summary_lab_value, get_covid_labs
from biomed.smoking.smoking import get_smoking_status
from biomed.vital_signs.vital_signs import get_vital_signs
from biomed.imaging_finding.imaging_finding import imaging_and_findings
from biomed.diagnosis.diagnosis import diagnosis_sign_symptoms
from biomed.device_procedure.api import device_procedure_predict_represent
from biomed.cancer.cancer import get_oncology_tokens


class TestFormatJsonCompatible(unittest.TestCase):
    # goal of these tests is ensuring we can write the output to json
    text = "HELLO  aspirin headache asthma diabetes type 2 Lab values"
    annotations, vectors = FeatureServiceClient().annotate_vectorize(text)

    def assert_valid_response_type(self, output):
        self.assertIsInstance(output, dict)
        self.assertIsInstance(json.dumps(output), str)
        self.assertDictEqual(json.loads(json.dumps(output)), output)

    def test_meds_an_allergies(self):
        drug_output = meds_and_allergies(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(drug_output)

    def test_labs(self):
        lab_output = summary_lab_value(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(lab_output)

    def test_vital_sign(self):
        vs_out = get_vital_signs(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(vs_out)

    def test_smoking(self):
        out = get_smoking_status(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(out)

    def test_imaging_finding(self):
        out = imaging_and_findings(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(out)

    def test_covid_lab(self):
        out = get_covid_labs(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(out)

    def test_disease_sign(self):
        out = diagnosis_sign_symptoms(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(out)

    def test_device_procedure(self):
        out = device_procedure_predict_represent(self.annotations, self.vectors, text=self.text)
        self.assert_valid_response_type(out)

    def test_cancer_output(self):
        out = get_oncology_tokens(tokens=self.annotations, vectors=self.vectors, text=self.text)
        self.assert_valid_response_type(out)


class TestNewSummaryFromParts(unittest.TestCase):
    text = "HELLO  aspirin headache \nPMH:history of asthma secondary to type 2 diabetes \n\nLab values"
    annotations, vectors = FeatureServiceClient().annotate_vectorize(text)

    def test_clinical_summary(self):
        drugs_json = json.dumps(meds_and_allergies(self.annotations, self.vectors, text=self.text))
        labs_json = json.dumps(summary_lab_value(self.annotations, self.vectors, text=self.text))
        problem_json = json.dumps(diagnosis_sign_symptoms(self.annotations, self.vectors, text=self.text))
        summary = summary_from_parts([json.loads(drugs_json), json.loads(labs_json), json.loads(problem_json)],
                                     text=self.text)

        expected = {
            'Medication': [],
            'Allergy': [],
            'Lab': [],
            'DiseaseDisorder': [
                {'text': 'asthma', 'range': [40, 46], 'score': 0.9998436042642885, 'label': 'diagnosis',
                 'polarity': 'positive', 'code': '195967001', 'cui': 'C0004096', 'tui': 'T047', 'vocab': 'SNOMEDCT_US',
                 'preferredText': 'Asthma', 'page': 1},
                {'text': 'type 2 diabetes', 'range': [60, 75], 'score': 0.9999894807178362, 'label': 'diagnosis',
                 'polarity': 'positive', 'code': '44054006', 'cui': 'C0011860', 'tui': 'T047', 'vocab': 'SNOMEDCT_US',
                 'preferredText': 'Diabetes Mellitus, Non-Insulin-Dependent', 'page': 1}]
,
            'SignSymptom': []
        }

        self.assert_biomed_out_minus_score_equal(summary, expected)

    def assert_biomed_out_minus_score_equal(self, biomed_out, expected):
        for key in biomed_out:
            if key == 'VersionInfo':
                continue

            for i in range(len(expected[key])):
                biomed_out_entry = biomed_out[key][i]
                expected_entry = expected[key][i]
                if 'score' in biomed_out_entry:
                    biomed_out_entry.pop('score')
                if 'score' in expected_entry:
                    expected_entry.pop('score')
                self.assertDictEqual(biomed_out_entry, expected_entry)
