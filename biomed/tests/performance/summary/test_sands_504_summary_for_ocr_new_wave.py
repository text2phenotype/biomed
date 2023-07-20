import os
import unittest

from biomed.summary.text_to_summary import text_to_summary
from text2phenotype.common import common

from biomed.biomed_env import BiomedEnv

bluebutton_SAMPLES = os.path.join(BiomedEnv.DATA_ROOT.value, 'Customer', 'bluebutton', 'Test Data', 'Sample Medical Records', 'BIOMED')


class TestSands405(unittest.TestCase):

    @staticmethod
    def process(file_text):
        text = common.read_text(file_text)

        text_to_summary(text)

    def test_bluebutton_summary(self):
        for f in common.get_file_list(bluebutton_SAMPLES, '.txt'):
            self.process(f)
