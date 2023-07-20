from datetime import datetime

from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import stephan_garcia_txt_filepath

from text2phenotype.common import common


class TestEndToEndStephanGarcia(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = stephan_garcia_txt_filepath
        cls.text = common.read_text(cls.txt_filepath)

        cls.FN = 'Stephan'
        cls.LN = 'Garcia'
        cls.SX = 'Male'
        cls.DOB = datetime(year=1957, month=4, day=3)
        cls.AGE = '59'
        cls.ETHNICITY = 'Hispanic_or_Latino'

        cls.ALLERGIES = ['Erythromycin']
        cls.CONDITIONS = [
            'chronic obstructive airway disease',
            'pneumonia',
            'white coat hypertension']  # 'hypertensive disease',
        cls.SIGN_SYMPTOM = ['flecks of blood', 'productive cough']
        cls.LAB_TESTS = []
        cls.MEDICATIONS = []
        cls.PHI_TOKENS = ['Stephan', 'Garcia', '04/03/1957', '2017-07-21', '1957-04-03', '7/21/2017', 'EX1234567']
        cls.PHI_RANGES = [[20, 26], [28, 35], [38, 48], [62, 72], [131, 138], [139, 145], [160, 170], [199, 206],
                          [207, 213], [227, 236], [242, 252], [1276, 1285], [3884, 3893]]
        cls.IMAGING = ['xrays', 'xray']

        cls.FAMILY_HISTORY = ['malignant neoplasm of prostate', 'melanoma']

        cls.SDOH = [
            '50 pack years, but only smoking a few cigarettes',
            'only smoking a few cigarettes a day for past year',
            'Moderate alcohol, several hard liquor drinks',
            'a day for past year',
            'Hazardous Activities:',
            'recently',
            'Exercise history: minimal in the last 5 years',
            'extended',
            'Alcohol:',
            'weekly, does not drink',
            'Tobacco:', 'weekly, does not drink to excess.',
            'where',
            'to excess. Smoking history: 50 pack years, but',
            ': Moderate alcohol, several hard liquor drinks']

        cls.PROCEDURE = ['Prostate', 'biopsy', 'chest', 'Appendectomy']
