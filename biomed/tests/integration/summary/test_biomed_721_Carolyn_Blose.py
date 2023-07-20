from datetime import datetime

from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import carolyn_blose_txt_filepath
from text2phenotype.common import common


class TestEndToEndCarolynBlose(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = carolyn_blose_txt_filepath
        cls.text = common.read_text(cls.txt_filepath)
        cls.ALLERGIES = ['Ambien', 'Cardizem', 'Ibuprofen']
        cls.MEDICATIONS = ['Lasix', 'Levaquin', 'Hydralazine', 'Ace', 'potassium', 'Fiber']
        cls.CONDITIONS = [
            'heart valve regurgitation',  # finding
            # 'systolic murmurs', finding
            'mitral valve insufficiency',
            'diverticulitis',
            'hypertensive disease',
            'pleural effusion disorder',
            'heart murmur',  # finding/signsymptom
            'hemorrhagic diarrhea',
            # 'tricuspid valve insufficiency', finding
            'colitis',
            'heart failure',
            # 'left atrial hypertrophy', finding
            'hypothyroidism',
            'cataract',
            'hypertensive disease'
        ]  #

        cls.SIGN_SYMPTOM = [
            'hematochezia',
            'heart murmur',
            'fatigue',
            'dyspnea',
            'abdominal pain',
            # 'hemorrhagic diarrhea',
            'hearing impairment',
            'blurred vision',
            'arthritis',
            'cataract',
            'muscle weakness'

        ]

        cls.AGE = '84'
        cls.SSN = '530-79-5301'
        cls.LAB_TESTS = [
            'ASSAY OF UREA NITROGEN QUANTITATIVE',
            'POTASSIUM MEASUREMENT',
            'CREATININE',
            'BNP'
        ]
        cls.FN = 'Carolyn'
        cls.LN = 'Blose'
        cls.SX = 'Female'
        cls.DOB = datetime(year=1932, month=7, day=29)
        cls.PHI_TOKENS = ['Carolyn', 'Blose', '07/29/1932', '2017-07-19', '7/21/2017',
                          '530-79-5301', '2017-07-21']
        cls.PHI_RANGES = [[6412, 6421], [6004, 6014], [6135, 6145], [6238, 6248], [5316, 5326],
                          [5271, 5280], [1363, 1372], [816, 826], [244, 254], [227, 238],
                          [208, 213], [197, 204], [158, 168], [138, 143], [130, 137], [61, 71], [37, 47],
                          [27, 34], [20, 25]]

        cls.IMAGING = ['echocardiogram', 'EKG']

        cls.SDOH = [
            'Recreational Drugs: No history of recreational',
            'not consume alcohol',
            'nonsmoker Status: Never',
            'Alcohol:',
            'a nonsmoker. Does not consume alcohol. No history',
            'Never smoker ( SNOMED-CT:',
            'drug use',
            'of recreational drug use.']

        cls.PROCEDURE = ['echocardiogram', 'Echocardiogram', 'EKG']
