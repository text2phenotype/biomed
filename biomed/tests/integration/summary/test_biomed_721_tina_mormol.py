from datetime import datetime

from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import tina_mormol_txt_filepath

from text2phenotype.common import common


class TestEndToEndTinaMarmol(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = tina_mormol_txt_filepath
        cls.text = common.read_text(cls.txt_filepath)

        cls.FN = 'Tina'
        cls.LN = 'Marmol'
        cls.SX = 'Female'
        cls.DOB = datetime(year=1963, month=12, day=5)
        cls.SSN = '553-82-1234'
        cls.PHI_TOKENS = ['Tina', 'Marmol', 'March', '2016', '2017-07-19', '2017-07-21', '1963-12-05', '7/21/2017', 'X',
                          '12/05/1963']
        cls.PHI_RANGES = [[20, 26], [28, 32], [35, 45], [59, 69], [128, 132], [133, 139], [154, 164], [194, 198],
                          [202, 208], [239, 249], [536, 546],  [1352, 1357], [1358, 1362],
                          [2440, 2449], [3089, 3099], [3226, 3227], [3701, 3706], [3707, 3711],
                          [4000, 4009]]
        # confusing case  bc it's a "notify if recurrent fever or chills" which implies the presence of
        # current chills and fever, willing to let slide
        cls.SIGN_SYMPTOM = ['relapsing fever']

        cls.ALLERGIES = []
        cls.CONDITIONS = ['congestive heart failure',
                          'hypertensive disease',
                          'leukocytosis',
                          'malignant neoplasm of breast',
                          'anemia of chronic disease',
                          'heart Failure',
                          'pleural effusion disorder',
                          'diastolic dysfunction']
        cls.LAB_TESTS = ['CHLORIDE',
                         'GLUCOSE [MASS/VOLUME] IN BLOOD',
                         'CREATININE',
                         'SODIUM MEASUREMENT',
                         'CARBON DIOXIDE MEASUREMENT',
                         'THROMBOCYTE (PLATELET)',
                         'BLOOD UREA NITROGEN MEASUREMENT',
                         'POTASSIUM MEASUREMENT',
                         'PARTIAL THROMBOPLASTIN TIME, ACTIVATED',
                         'COUNTS, PLATELET',
                         'WHITE BLOOD CELL COUNT - OBSERVATION',
                         'ASSAY OF CALCIUM',
                         'INR',
                         'Magnesium',
                         'H&H',
                         'MCV']
        cls.MEDICATIONS = ['Lasix',
                           'Prinivil',
                           'Aspirin',
                           'Docusate',
                           'Klor-Con',
                           'Atenolol 25 MG',
                           'Ditropan',
                           'Colace',
                           'Oxybutynin',
                           'Potassium Chloride 10 MEQ',
                           'Lisinopril',
                           'Solu-Medrol',
                           'Diurese',
                           'oxybutynin']
        cls.IMAGING = ['Chest x-rays', 'echocardiogram']
        cls.PROCEDURE = [
            'Thoracentesis',
 'aspiration',
 'thoracentesis',
 'Chest',
 'x-rays',
 'scan',
 'Needle',
 'echocardiogram',
 'heart']
