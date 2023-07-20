import unittest

from biomed.summary.text_to_summary import text_to_summary
from biomed.tests.samples import MTSAMPLES_DIR

from text2phenotype.common import common


class TestBiomed343(unittest.TestCase):

    def test_mtsamples(self, do_output=False):
        for f in common.get_file_list(MTSAMPLES_DIR, '.txt'):
            if f.endswith('clean.txt'):
                text = common.read_text(f)
                res = text_to_summary(text)
                if do_output:
                    common.write_json(res, f+'.text2summary-lstm.json')
