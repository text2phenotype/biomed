import unittest

from biomed.tests import samples


class TestBiomed313(unittest.TestCase):

    @staticmethod
    def assertSummary(text_file, expected=None):
        return True

    @unittest.skip("Not implemented")
    def test_deleys_expert_summary(self):
        """
        https://drive.google.com/drive/folders/1yaEMBCFPmGYASs9ytTFrsIXALRP3oemi
        """
        self.assertSummary(f"{samples.RICARDO_HPI_TXT}")
        self.assertSummary(f"{samples.CAROLYN_BLOSE_TXT}")
        self.assertSummary(f"{samples.JOHN_STEVENS_TXT}")
        self.assertSummary(f"{samples.STEPHAN_GARCIA_TXT}")
        self.assertSummary(f"{samples.TINA_MARMOL_TXT}")
        self.assertSummary(f"{samples.DAVID_VAUGHN_TXT}")
