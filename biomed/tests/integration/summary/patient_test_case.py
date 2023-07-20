from math import ceil
import unittest

import pandas as pd

from biomed.common.biomed_summary import FullSummaryResponse
from biomed.family_history.family_history_pred import family_history
from biomed.procedure.procedure import get_procedures
from biomed.sdoh.sdoh import get_sdoh_response
from text2phenotype.common.dates import parse_dates
from text2phenotype.constants.features import SignSymptomLabel, PHILabel
from text2phenotype.constants.features.label_types import FamilyHistoryLabel, ProcedureLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.helpers import annotation_helper
from biomed.deid.deid import get_phi_tokens
from biomed.deid.utils import deid_from_demographics
from biomed.demographic import demographic
from biomed.drug.drug import meds_and_allergies
from biomed.lab.labs import get_covid_labs, summary_lab_value
from biomed.diagnosis.diagnosis import diagnosis_sign_symptoms
from biomed.vital_signs.vital_signs import get_vital_signs
from biomed.smoking.smoking import get_smoking_status
from biomed.imaging_finding.imaging_finding import imaging_and_findings
from biomed.device_procedure.api import device_procedure_predict_represent
from biomed.reassembler.reassemble_functions import reassemble_demographics, reassemble_summary_chunk_results


def get_unique_vals(output_dict):
    od_len = len(output_dict)
    output_terms = [''] * od_len

    for i in range(od_len):
        if 'preferredText' in output_dict[i] and output_dict[i]['preferredText'] is not None:
            output_terms[i] = output_dict[i]['preferredText']
        elif output_dict[i]['text'] is not None:
            output_terms[i] = output_dict[i]['text']

    return list(set(output_terms))


def sample_accuracy(output_list, expected_list):
    actual_set = set([o.lower().strip() for o in output_list])
    expected_set = set([o.lower().strip() for o in expected_list])
    # get false positives count
    fp = len(actual_set.difference(expected_set))

    # get false negatives
    fn = len(expected_set.difference(actual_set))

    return fp, fn


class PatientTestCase(unittest.TestCase):
    __test__ = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.ALLERGIES = []
        cls.CONDITIONS = []
        cls.LAB_TESTS = []
        cls.MEDICATIONS = []
        cls.PHI_TOKENS = []
        cls.PHI_RANGES = []
        cls.SIGN_SYMPTOM = []
        cls.PROCEDURE = []
        cls.RACE = set()
        cls.ETHNICITY = None
        cls.txt_filepath = ""
        cls.DOB = None
        cls.summary = []
        cls.AGE = None
        cls.SSN = None
        cls.text = ""
        cls.FN = ''
        cls.LN = ''
        cls.SX = ''
        cls.hepc = []
        cls.IMAGING = []
        cls.FAMILY_HISTORY = []
        cls.SDOH = []
        cls.initialize()
        # run predictions after initializing
        tokens, vectors = annotation_helper(
            cls.text,
            {TaskOperation.phi_tokens, TaskOperation.demographics, TaskOperation.drug, TaskOperation.disease_sign,
             TaskOperation.lab, TaskOperation.covid_lab, TaskOperation.device_procedure, TaskOperation.imaging_finding,
             TaskOperation.smoking, TaskOperation.vital_signs})
        date_matches = parse_dates(cls.text)

        cls.phi_token_mapping = ([0, len(cls.text)], get_phi_tokens(tokens, vectors=vectors))
        cls.demographic_mapping = (
            [0, len(cls.text)],
            demographic.get_demographic_tokens(
                tokens,
                vectors=vectors,
                date_matches=date_matches))
        # uodate this part
        cls.phi = reassemble_summary_chunk_results(
            [cls.phi_token_mapping]+
             deid_from_demographics([cls.demographic_mapping]))
        cls.phi_list = cls.phi.get(PHILabel.get_category_label().persistent_label)

        cls.demographics = reassemble_demographics([cls.demographic_mapping])

        cls.drugs = meds_and_allergies(tokens=tokens, vectors=vectors, text=cls.text)

        cls.imaging_finding = imaging_and_findings(tokens=tokens, vectors=vectors, text=cls.text)
        cls.smoking = get_smoking_status(tokens=tokens, vectors=vectors, text=cls.text)
        cls.vital_signs = get_vital_signs(tokens=tokens, vectors=vectors, text=cls.text)
        cls.labs = summary_lab_value(tokens=tokens, vectors=vectors, text=cls.text)
        cls.covid_labs = get_covid_labs(tokens=tokens, vectors=vectors, text=cls.text)
        cls.devices = device_procedure_predict_represent(tokens=tokens, vectors=vectors, text=cls.text)
        cls.problems = diagnosis_sign_symptoms(tokens=tokens, vectors=vectors, text=cls.text)
        cls.family_history = family_history(tokens=tokens, vectors=vectors, text=cls.text)
        cls.social_history = get_sdoh_response(tokens=tokens, vectors=vectors, text=cls.text)
        cls.procedure = get_procedures(tokens=tokens, vectors=vectors, text=cls.text)


    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @classmethod
    def initialize(cls):
        cls.tearDownClass()

        raise unittest.SkipTest("Skip base class for all end-to-end patients' tests")

    def assert_valid_recall_precision(self, predicted_values:list, expected_values: list, type_label:str = None):
        afp, afn = sample_accuracy(predicted_values, expected_values)
        threshold = 0.05
        self.assertLessEqual(afp, ceil(threshold * len(predicted_values)),
                             f"{type_label} Identified that were not {type_label} (false positives)\n"
                             f"{str(set(predicted_values).difference(set(expected_values)))}")

        self.assertLessEqual(afn, ceil(threshold * len(expected_values)),
                             f"More than {threshold * 100}% of expected {type_label} are missing from predictions "
                             f"{str(set(expected_values).difference(set(predicted_values)))}")

    def test_patient_allergies(self):
        # self.initialize()

        output_allergies = get_unique_vals(self.drugs.get('Allergy', []))

        self.assert_valid_recall_precision(output_allergies, self.ALLERGIES, 'allergies')

    def test_patient_sign_symptoms(self):
        output_conditions = get_unique_vals(
            self.problems.get(SignSymptomLabel.get_category_label().persistent_label, []))
        output_conditions = [o.lower() for o in output_conditions]
        self.assert_valid_recall_precision(output_conditions, self.SIGN_SYMPTOM, 'symptoms')

    def test_patient_family_history(self):
        output_conditions = get_unique_vals(
            self.family_history.get(FamilyHistoryLabel.get_category_label().persistent_label, []))
        output_conditions = [o.lower() for o in output_conditions]
        self.assert_valid_recall_precision(output_conditions, self.FAMILY_HISTORY, 'family_history')

    def test_patient_conditions(self):
        # conditions test
        # self.initialize()

        output_conditions = get_unique_vals(self.problems.get('DiseaseDisorder', []))
        output_conditions = [o.lower() for o in output_conditions]
        self.assert_valid_recall_precision(output_conditions, self.CONDITIONS, 'diagnoses')

    def test_patient_labs(self):
        # self.initialize()

        output_labs = get_unique_vals(self.labs.get('Lab', []))
        # lab test test
        self.assert_valid_recall_precision(output_labs, self.LAB_TESTS, 'labs')

    def test_patient_sdoh(self):
        # self.initialize()
        full_resp = FullSummaryResponse.from_json(self.social_history)
        full_resp.postprocess(text=self.text)
        full_resp_json = full_resp.to_json()

        output_labs = get_unique_vals(full_resp_json.get('SocialRiskFactors', []))
        # lab test test
        self.assert_valid_recall_precision(output_labs, self.SDOH, 'Social')


    def test_patient_medications(self):
        # self.initialize()

        output_medications = get_unique_vals(self.drugs.get('Medication', []))
        # medications test
        self.assert_valid_recall_precision(output_medications, self.MEDICATIONS, 'meds')

    def test_demographics_first_name(self):
        if not self.demographics.get('pat_first'):
            self.assertIn(self.FN, ["", False, None, []],
                          msg=f"No patient first name was predicted but {self.FN} was expected")
            return
        prob = pd.DataFrame(self.demographics.get('pat_first')).max()[1]
        text = [row[0] for row in self.demographics['pat_first'] if row[1] == prob]

        self.assertEqual(text[0].lower(), self.FN.lower())

    def test_demographics_last_name(self):
        if not self.demographics.get('pat_last'):
            self.assertIn(self.LN, ["", False, None, []],
                          msg=f"No patient last name was predicted but {self.LN} was expected")
            return
        prob = pd.DataFrame(self.demographics['pat_last']).max()[1]
        text = [row[0] for row in self.demographics['pat_last'] if row[1] == prob]

        self.assertEqual(text[0].lower(), self.LN.lower())

    def test_demographics_sex(self):
        if not self.demographics.get('sex'):
            self.assertIn(self.SX, ['', None, []], msg=f"No sex was predicted but {self.SX} was expected")
            return

        prob = pd.DataFrame(self.demographics['sex']).max()[1]
        text = [row[0] for row in self.demographics['sex'] if row[1] == prob]

        self.assertEqual(text[0].lower(), self.SX.lower())

    def test_demographics_race(self):
        if not self.demographics.get('race'):
            self.assertIn(self.RACE, ['', None, [], set()], msg=f"No sex was predicted but {self.SX} was expected")
        else:
            self.assertEqual(len(self.RACE), len(self.demographics['race']))
            self.assertSetEqual(self.RACE, {entry[0] for entry in self.demographics.get('race')})

    def test_ethnicity(self):
        if not self.demographics.get('ethnicity'):
            self.assertIsNone(self.ETHNICITY)
        else:
            self.assertEqual(len(self.demographics['ethnicity']), 1)
            self.assertEqual(self.demographics['ethnicity'][0][0], self.ETHNICITY)

    def test_ssn(self):
        if not self.demographics.get('ssn'):
            if self.SSN:
                self.assertTrue(False)
            else:
                self.assertTrue(True)
            return
        self.assertEqual(self.demographics.get('ssn')[0][0], self.SSN)

    def test_age(self):
        if not self.demographics.get('pat_age'):
            if self.AGE:
                self.assertTrue(False)
            else:
                self.assertTrue(True)
            return
        self.assertEqual(self.demographics.get('pat_age')[0][0], self.AGE)

    def test_demographics_birthdate(self):
        if not self.demographics.get('dob'):
            self.assertIn(self.DOB, ["", False, None, []], msg=f"No DOB was predicted but {self.DOB} was expected")
            return
        predicted_dob = parse_dates(self.demographics['dob'][0][0])[0][0]
        self.assertEqual(self.DOB.year, predicted_dob.year)
        self.assertEqual(self.DOB.month, predicted_dob.month)
        self.assertEqual(self.DOB.day, predicted_dob.day)

    def test_phi_tokens(self):
        phi_text = [t['text'] for t in self.phi_list]
        for token in self.PHI_TOKENS:
            self.assertIn(token, phi_text)

    def test_phi_ranges(self):
        phi_ranges = [t['range'] for t in self.phi_list]
        for rngs in self.PHI_RANGES:
            self.assertIn(rngs, phi_ranges)

    def test_imaging(self):
        pred_imagings = get_unique_vals(self.imaging_finding.get('DiagnosticImaging', []))
        self.assert_valid_recall_precision(pred_imagings, self.IMAGING, 'imaging reports')

    def test_procedure(self):
        pred_procedure = get_unique_vals(self.procedure.get(ProcedureLabel.get_category_label().persistent_label, []))
        self.assert_valid_recall_precision(pred_procedure, self.PROCEDURE, 'procedures')
