from typing import Dict
import unittest
from datetime import datetime

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger

from feature_service.nlp import autocode, nlp_cache
from feature_service.aspect.chunker import Chunker

from biomed.deid import deid
from biomed.tests.samples import REINSURANCE_SHORTER_TXT, REINSURANCE_LONGER_TXT


class Cursor:

    def __init__(self, txt_file, pipeline, result_key='result'):
        operations_logger.info(f"{pipeline} cursor for {txt_file} ")

        self.pipeline = pipeline
        self.result_key = result_key
        self.txt_file = txt_file
        self.json_file = f"{self.txt_file}.{self.pipeline}.json"
        self.pages = dict()
        self.skip = dict()

    def count(self, result):
        key_exists = result.get(self.result_key, None) is not None
        return 0 if not key_exists else len(result[self.result_key])

    def error(self, page_num, start, stop, e):
        operations_logger.error(e)
        self.skip[page_num] = dict()
        self.skip[page_num]['range'] = [start, stop]
        self.skip[page_num]['error'] = str(e)

    def add(self, page_num, start, stop, result, do_filter=False):
        self.pages[page_num] = dict()
        self.pages[page_num]['range'] = [start, stop]
        self.pages[page_num]['result'] = result
        self.save()

        if do_filter:
            cnt = self.count(result)
            if cnt > 0:
                operations_logger.info(f"page_num {page_num} cnt is {cnt}")
                self.pages[page_num]['result'] = filter_autocode_cui_rule(result, self.result_key)
                self.save()

    def save(self):
        operations_logger.info(f'saving {self.json_file}')
        common.write_json(self.to_dict(), self.json_file)

    def to_dict(self):
        return {'txt_file': self.txt_file,
                'pipeline': self.pipeline,
                'pages': self.pages,
                'skip': self.skip}


@unittest.skip('https://gettext2phenotype.atlassian.net/browse/BIOMED-319')
class TestCursor(unittest.TestCase):

    def run_sliding_window(self, txt_file: str, lines_per_page=50):
        """
        :param txt_file:  input file, choices:

        CL_out.pdf.txt
        ACFrOgD_BXMLzQt241zgo1syHjVENdNUJ2HEJuOc4eQsyFfhu_YdHja_d5ZoQvL90OcJZeUk5VwaWT75poFgabtk3Ux8dnIptyoQaX2tHOnjSgz3GoIErHd-9eGI0WQ=.pdf.txt

        :param lines_per_page: number of lines per page

        default: 10 thousand
        if None: entire document

        :return:
        """
        operations_logger.info(txt_file)
        operations_logger.info(datetime.now())

        text = common.read_text(txt_file)
        lines = text.splitlines()

        txt_cursor = Cursor(txt_file, 'Cursor')
        err_cursor = Cursor(txt_file, 'Error')
        aspect_cursor = Cursor(txt_file, 'AspectLabeler')
        clin_cursor = Cursor(txt_file, 'Clinical.SNOMED')
        smok_cursor = Cursor(txt_file, 'SmokingStatus')
        lab_cursor = Cursor(txt_file, 'LabValues.GENERAL', 'labValues')
        rxnorm_cursor = Cursor(txt_file, 'DrugNER.RXNORM.bilstm_drug_ner_all_prob_filter', 'drugEntities')
        phi_cursor = Cursor(txt_file, 'DEID.get_phi_tokens')
        deid_cursor = Cursor(txt_file, 'DEID.redacted')
        summary_cursor = Cursor(txt_file, 'Summary')
        chunker = Chunker()
        for r in range(0, 500):
            start, stop = (r * lines_per_page), (r * lines_per_page) + lines_per_page

            if start > len(text):
                operations_logger.info(f"stop, text is {len(text)} but start is {start}")
                return [start, stop]

            if stop > len(text):
                operations_logger.info(f"stop, text is {len(text)} but stop is {stop}")
                stop = len(text)

            fragment = '\n'.join(lines[start:stop])

            try:
                operations_logger.info("Cursor {start} {stop} ")
                txt_cursor.add(r, start, stop, {'fragment': fragment})

                operations_logger.info("AspectLabeler")
                res = chunker.predict_aspect_emb_by_section_no_enforce(fragment)
                aspect_cursor.add(r, start, stop, res)

                operations_logger.info("Clinical.SNOMED")
                res = autocode.clinical(fragment)
                clin_cursor.add(r, start, stop, res, do_filter=True)

                operations_logger.info("SmokingStatus")
                res = autocode.smoking(fragment)
                smok_cursor.add(r, start, stop, res)

                operations_logger.info("DrugNER.RXNORM")
                res = MedModel().bilstm_drug_ner_all_prob_filter(fragment)
                rxnorm_cursor.add(r, start, stop, res)

                operations_logger.info("LabValues.GENERAL")
                res = nlp_cache.lab_value(fragment)
                lab_cursor.add(r, start, stop, res, do_filter=True)

                operations_logger.info('Summary')
                res = text_to_summary(fragment)
                summary_cursor.add(r, start, stop, res)

                operations_logger.info('DEID.get_phi_tokens')
                phi = deid.get_phi_tokens(fragment)
                phi_cursor.add(r, start, stop, phi)

                operations_logger.info('DEID.redacted')

            except Exception as e:
                operations_logger.error(e)
                err_cursor.error(r, start, stop, e)

    def test_sliding_winow(self):

        VERSION = common.version_text('REINSURANCE')

        operations_logger.info(VERSION)

        self.run_sliding_window(REINSURANCE_SHORTER_TXT)
        self.run_sliding_window(REINSURANCE_LONGER_TXT)

    def run_autocoder(self, text: str, autocoder, cache_filetype, dictionary=None) -> Dict:
        """
        :param text: text fragment
        :param autocoder: autocoder function
        :return: json result or Exception if error
        """
        return nlp_cache.autocode_cache(text, autocoder, cache_filetype, dictionary)
