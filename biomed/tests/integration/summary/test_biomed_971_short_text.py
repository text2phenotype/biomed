from biomed.tests.integration.summary.patient_test_case import PatientTestCase


class Biomed971ShortText(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = None
        cls.text = "Caroline Hannigan is 90 years old"

        cls.FN = 'Caroline'
        cls.LN = 'Hannigan'
        cls.SX = None
        cls.DOB = None
        cls.AGE = '90'

        cls.ALLERGIES = []
        cls.CONDITIONS = []
        cls.LAB_TESTS = []
        cls.MEDICATIONS = []
