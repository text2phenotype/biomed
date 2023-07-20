import os
import unittest

from feature_service.common import bsv
from feature_service.feature_set import icd9

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common

from biomed.biomed_env import BiomedEnv


ICD9_BSV = os.path.join(BiomedEnv.DATA_ROOT.value,
                        'biomed',
                        'models',
                        'expert',
                        'problem',
                        'BIOMED-394.map_cui_icd9.bsv')
ICD10_BSV = os.path.join(BiomedEnv.DATA_ROOT.value,
                         'biomed',
                         'models',
                         'expert',
                         'problem',
                         'BIOMED-394.map_cui_icd10.bsv')


class TestBiomed394_ICD9(unittest.TestCase):

    def assertLVG(self, expected, actual):
        self.assertEqual(icd9.unique(expected), icd9.unique(actual))

    def assertFormat(self, codes, test_func, expected=True):
        for code in codes:
            self.assertEqual(expected, test_func(code), f"expected {expected} for {str(test_func)} | {code}")

    def assertICD9(self, text: str, codes: list):
        """
        :param text: str clinical
        :param codes: list of str expected ICD9
        """
        raise Exception('NOT YET IMPLEMENTED')

    def test_icd_problem_list(self):
        # self.assertICD10(problem_list, ['K21.9', 'M25.561', 'J02.9']) TODO
        # self.assertICD9(problem_list, ['530.81', '719.46', '462']) TODO
        pass

    def test_icd_vcodes(self):
        # self.assertICD9(vcodes_list, ['V70.0', 'V13.01']) TODO
        # self.assertICD10(vcodes_list, ['Z00.00', 'Z87.442']) TODO
        pass

    def test_dot(self):
        codes = ['008.5', '00.12', 'V85.36', 'E001.0']
        self.assertFormat(codes, icd9.has_dot)

    def test_is_vcode(self):
        codes = ['V91.91', 'V91.2', 'V08']

        self.assertFormat(codes, icd9.is_vcode)
        self.assertFormat(codes, icd9.is_special)

    def test_is_ecode(self):
        codes = ['E996.3', 'E987', 'E002.7']

        self.assertFormat(codes, icd9.is_ecode)
        self.assertFormat(codes, icd9.is_special)

    def test_lvg_ecode(self):
        codes = ['E002.7']  # TODO

    def test_is_numeric(self):
        codes = ['410', '4109', '41090', '00800']
        self.assertFormat(codes, icd9.is_integer)

    def test_is_digit5(self):
        codes = ['410.90', '008.00', '00800', '098.43', '09843', '173.61', '17361']
        self.assertFormat(codes, icd9.is_5digit)

    def test_lvg_5digit(self):
        self.assertLVG(['410.90', '41090'], icd9.lvg_5digit('410.90'))
        self.assertLVG(['410.90', '41090'], icd9.lvg_5digit('41090'))

    def test_is_digit4(self):
        codes = ['097.1', '196.2', '253.9', '39.98', '49.92', '596.2', '60.14', '724.0', '85.47', '97.25']
        self.assertFormat(codes, icd9.is_4digit)

    def test_lvg_4digit(self):
        self.assertLVG(['410.9', '410.90'], icd9.lvg_4digit('410.9'))
        self.assertLVG(['097.1', '097.10'], icd9.lvg_4digit('097.1'))

    def test_is_digit3(self):
        codes = ['001', '037', '999', '135', '01.0', '12.7', '24.3', '39.2', '49.5', '52.4', '68.4', '77.4', '80.9',
                 '92.1']
        self.assertFormat(codes, icd9.is_3digit)

    def test_lvg_3digit(self):
        self.assertLVG(['410', '410.0', '410.00'], icd9.lvg_3digit('410'))

    def test_is_digit2(self):
        codes = ['00', '01', '10', '25', '99']
        self.assertFormat(codes, icd9.is_2digit)

    def test_split_major_minor(self):

        self.assertEqual([410, 90], icd9.split_major_minor('410.90'))
        self.assertEqual([410, 9], icd9.split_major_minor('410.9'))
        self.assertEqual([410, None], icd9.split_major_minor('410'))

        self.assertEqual([327, 22], icd9.split_major_minor('327.22'))
        self.assertEqual([327, 2], icd9.split_major_minor('327.02'))
        self.assertEqual([327, 0], icd9.split_major_minor('327.0'))
        self.assertEqual([327, 0], icd9.split_major_minor('327.0'))
        self.assertEqual([327, None], icd9.split_major_minor('327'))

    def test_in_range_procedure(self):

        codes = ['00', '00.7', '00.71', '011.00']
        self.assertFormat(codes, icd9.in_range_procedure)

    def test_icd9_codes(self, output=True):

        vocab = bsv.parse_bsv_dict(ICD9_BSV, bsv.Columns.vocab_pref_terms, 'code')

        slang = dict()
        attrs = dict()

        for code in vocab.keys():
            operations_logger.info(f'code|{code}')
            attrs[code] = icd9.attributes(code)

            if not attrs[code]['is_normal'] and not attrs[code]['is_special']:
                self.assertTrue(code in [None, ''], f'unkown ICD9 code format? {code}')

            for synonym in icd9.lvg_code(code):
                if synonym not in vocab.keys():
                    slang[synonym] = vocab[code]

        if output:
            slang_json = f"{ICD9_BSV}.slang.json"
            operations_logger.info(slang_json)
            common.write_json(slang, slang_json)

            attrs_json = f"{ICD9_BSV}.attributes.json"
            operations_logger.info(attrs_json)
            common.write_json(attrs, attrs_json)
