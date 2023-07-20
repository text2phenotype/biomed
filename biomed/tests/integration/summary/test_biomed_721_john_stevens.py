from datetime import datetime

from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import john_stevens_txt_filepath

from text2phenotype.common import common


class TestEndToEndJohnStevens(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = john_stevens_txt_filepath
        cls.text = common.read_text(cls.txt_filepath)

        cls.FN = 'John'
        cls.LN = 'Stevens'
        cls.SX = 'Male'
        cls.DOB = datetime(year=1968, month=3, day=19)
        cls.SSN = '235-21-0677'
        cls.AGE = '49'
        cls.PHI_TOKENS = [
            '2017-07-19', 'John', 'Stevens', '03/19/1968', '1999', '1968-03-19', '7/21/2017', 'X',
            '235-21-0677']
        cls.SIGN_SYMPTOM = [
            'respiratory distress',
            'dyspnea']  # 'expiratory wheezing', 'wheezing',

        cls.ALLERGIES = []
        cls.CONDITIONS = [
            'heart failure',
            'coronary artery disease',
            'hypertensive disease',
            'cardiac arrest',
            'respiratory failure',
            'respiratory arrest',
            'chronic obstructive airway disease',
            'myocardial infarction',
            'pneumonia',
            'h/o: hypertension']
        cls.LAB_TESTS = [
            'WHITE BLOOD CELL (CELL)',
            'CARBON DIOXIDE MEASUREMENT, PARTIAL PRESSURE',
            'PO2 MEASUREMENT',
            'PH MEASUREMENT',
            'ANC']
        cls.MEDICATIONS = [
            'Antibiotic',
            'Levaquin',
            'Solu-Medrol',
            'Lisinopril 20 MG',
            'Digoxin',
            'Prednisone 20 MG',
            'Theo-24',
            'Furosemide 40 MG',
            'Acetazolamide 250 MG',
            'K-Dur',
            'Ventolin',
            'Azmacort',
            'glyceryl trinitrate', 'Lisinopril']
        cls.PHI_RANGES = [
            [20, 27], [29, 33], [36, 46], [129, 133], [134, 141], [195, 199], [200, 207], [221, 232],
            [238, 248], [860, 869], [897, 907], [1564, 1568], [4342, 4343], [5060, 5069], [5148, 5158],
            [5518, 5522], [6062, 6071]]
        cls.IMAGING = [
            'electrocardiogram', 'echocardiogram']

        cls.SDOH = [
            'been smoking up until three to',
            'three to four months previously',
            'Status: Quit',
            'Recreational',
            'been smoking up until',
            'Drugs',
            'Alcohol:',
            'Coffee:',
            'Former smoker']

        cls.PROCEDURE = ['extubated', 'electrocardiogram', 'intubated']

