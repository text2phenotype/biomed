from datetime import datetime

from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import david_vaughan_txt_filepath

from text2phenotype.common import common


class TestEndToEndDavidVaughan(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = david_vaughan_txt_filepath
        cls.text = common.read_text(cls.txt_filepath)

        cls.FN = 'David'
        cls.LN = 'Vaughan'
        cls.SX = 'Male'
        cls.DOB = datetime(year=1940, month=7, day=18)
        cls.SSN = '601-60-0606'
        cls.AGE = '77'
        cls.RACE = {'WHITE'}

        cls.ALLERGIES = []

        cls.CONDITIONS = ['anemia',
                          'thrombocytopenia',
                          'spinal fractures',
                          'chronic kidney disease stage 5',
                          'diabetes mellitus',
                          'heart failure',
                          'malnutrition',
                          'discitis']

        cls.SIGN_SYMPTOM = ['constipation', 'low back pain', 'confusion']
        cls.LAB_TESTS = ['MEASUREMENT OF TOTAL PROTEIN'] # is fine that it's catching it for now bc we don't use lab models
        cls.MEDICATIONS = []
        cls.PHI_TOKENS = ['David', 'Vaughan', '07/18/1940', '2017-07-21', '601-60-0606', '1940-07-18', '7/21/2017']
        cls.PHI_RANGES = [[3805, 3814], [3595, 3605], [3434, 3444], [2478, 2487],
                          [1912, 1916], [1909, 1910], [1905, 1908], [968, 972], [964, 966], [958, 963], [539, 549],
                          [244, 254], [227, 238], [206, 213], [203, 205], [197, 202], [158, 168], [136, 143],
                          [130, 135], [37, 47], [61, 71], [29, 34], [20, 27]]
        cls.PROCEDURE = ['hemodialysis', 'dialysis', 'Dialysis']
