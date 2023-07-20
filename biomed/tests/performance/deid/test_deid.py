import unittest

from text2phenotype.common import common

from biomed.deid import deid
from biomed.biomed_env import BiomedEnv


class TestDeid(unittest.TestCase):

    def test_mpi_extract_phi_using_deid(self, do_output=False):
        for f in self.read_bluebutton_text():
            text = common.read_text(f)
            # the DEID pipeline using majority vote perform better on some of the records,
            # so finding the right way to ensemble is important
            phi = deid.get_phi_tokens(text)

            if do_output:
                common.write_json(phi, f'{f}.deid.get_phi_tokens.json')

    @staticmethod
    def read_bluebutton_text():
        expected = common.get_file_list(BiomedEnv.DATA_ROOT.value + '/BIOMED', '.txt')
        expected.append(BiomedEnv.DATA_ROOT.value)

        return expected
