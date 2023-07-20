import os
import datetime as dt
import unittest
from typing import Dict, List

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common

from feature_service.nlp.nlp_reader import ClinicalReader
from feature_service.nlp import autocode

from biomed.tests.samples import SHRINE_ICD9_BSV
from biomed.tests.timer import Timer
from biomed.biomed_env import BiomedEnv

######################################################################################################
#
# OUTPUT ( set to None if no output)
#
######################################################################################################

TEST_OUTPUT_FILE = os.path.join(BiomedEnv.DATA_ROOT.value,
                                'biomed',
                                'models',
                                'expert',
                                'problem',
                                'BIOMED-394.autocode.icd9.json')


######################################################################################################
#
# HELPER FUNCTIONS
#
######################################################################################################

def res_vocab_code(autocode_res: Dict, vocabulary='ICD9CM') -> List:
    """
    :param autocode_res:
    :param vocabulary if None then allow any vocab match, else only return code matching vocabulary
    :return dict[vocab] = [code1,code2,...]
    """
    found = list()
    for match in ClinicalReader(autocode_res).list_results():
        for c in match.concepts:
            sab = c['codingScheme']
            code = c['code']

            if vocabulary is None:
                found.append({sab: code})
            else:
                if sab == vocabulary:
                    if code not in found:
                        found.append(code)
    return found


def parse_code_text(code_text_bsv=SHRINE_ICD9_BSV) -> Dict:
    """
    :return: dict[code] = [text1, text2, ...]
    """
    parsed = dict()
    for line in common.read_text(code_text_bsv).splitlines():
        code, text = line.split('|')

        if code not in parsed:
            parsed[code] = list()

        if text not in parsed[code]:
            parsed[code].append(text)
    return parsed


######################################################################################################
#
# UNIT TEST
#
######################################################################################################

class TestBiomed394_autocode(unittest.TestCase):

    def test_icd9_autocode(self, output=TEST_OUTPUT_FILE):

        operations_logger.info(SHRINE_ICD9_BSV)

        text = common.read_text(SHRINE_ICD9_BSV)
        search = dict()
        missed = list()

        known = parse_code_text()
        progress_prct = 0
        start_time = dt.datetime.now()

        for line in text.splitlines():
            code, text = line.split('|')

            if code not in search.keys():
                search[code] = dict()

            if text not in search[code].keys():

                progress_prct = (len(search.keys())) / len(known)
                progress_time = Timer.seconds(start_time, dt.datetime.now())

                t1 = dt.datetime.now()
                res = autocode.autocode(text, autocode.PipelineURL.icd9)
                elapsed = Timer.seconds(dt.datetime.now(), t1)

                hits = res_vocab_code(res)
                search[code][text] = {'elapsed': elapsed, 'hits': hits, 'exact': (code in hits)}

                if len(hits) == 0:
                    missed.append({'code': code, 'text': text, 'hits': hits})

            if output:
                # only show progress and save files in the beginning, at the end, and once in a while :)
                if (progress_prct > .9) or (progress_prct < .01) or (10 == int(common.rng(3))):
                    operations_logger.info(f'{code}|{text}')
                    operations_logger.info(f'# {len(search)} | {progress_time}s')
                    operations_logger.info(f'{progress_prct} %')
                    common.write_json(search, TEST_OUTPUT_FILE)

                    if len(missed) > 0:
                        common.write_json(missed, TEST_OUTPUT_FILE.replace('.json', '.missed.json'))
