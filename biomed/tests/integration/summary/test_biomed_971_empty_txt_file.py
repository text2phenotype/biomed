from biomed.tests.integration.summary.patient_test_case import PatientTestCase
from biomed.tests.fixtures.example_file_paths import empty_text_fp
from text2phenotype.common import common


class Biomed971ShortText(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = empty_text_fp
        cls.text = common.read_text(cls.txt_filepath)

        cls.FN = None
        cls.LN = None
        cls.SX = None
        cls.DOB = None

        cls.ALLERGIES = []
        cls.CONDITIONS = []
        cls.LAB_TESTS = []
        cls.MEDICATIONS = []
