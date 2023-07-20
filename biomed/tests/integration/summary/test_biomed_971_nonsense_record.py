from biomed.tests.integration.summary.patient_test_case import PatientTestCase


class Biomed971ShortText(PatientTestCase):
    __test__ = True

    @classmethod
    def initialize(cls):
        cls.txt_filepath = None
        cls.text = """ [po345hegf98  yihf;'if[ewujlkfv  poyfr9847reiohf [ou8rn9t-8 ]ap0ie jjfd aos saiut !@#$%^&*&
        (c) ✫✫✫✫✫✫✫✫ ✫ -----------
        ---------------------
        ;[0*&&*) \n \t \\n \\y
        """

        cls.FN = None
        cls.LN = None
        cls.SX = None
        cls.DOB = None

        cls.ALLERGIES = []
        cls.CONDITIONS = []
        cls.LAB_TESTS = []
        cls.MEDICATIONS = []
