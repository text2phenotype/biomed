import os
import unittest

from text2phenotype.common import common
from text2phenotype.common.log import logger
from text2phenotype.apiclients.biomed import BioMedClient

from feature_service.features import sectionizer
from feature_service.feature_set import annotation
from feature_service.nlp import autocode, nlp_cache

from biomed.tests.timer import Timer


class AnnotateBatch(object):

    def __init__(self, batch_dir=None):
        """
        :param batch_dir: directory of text files
        """
        if batch_dir:
            self.batch_dir = batch_dir
        else:
            self.batch_dir = os.environ.get('BATCH_DIR', '/mnt/mimic/20190207_andy/txt/Nursing/other')

        logger.info(f'BATCH_DIR= {self.batch_dir}')

        self.samples = list()

        for root, dirs, files in os.walk(self.batch_dir, topdown=False):
            for f in files:
                if f.endswith('.txt'):
                    logger.debug(f)
                    self.samples.append(os.path.join(root, f))

        logger.info(f'{len(self.samples)} files in {self.batch_dir}')

    def process(self, annotator, save_output=True):
        """
        :param annotator: function to run, like ctakes.temporal
        :param save_output: True write output to file, False do not write output to file
        """
        logger.info(f'process= {annotator.__name__} BATCH_DIR={self.batch_dir}')

        complete = 0

        label = common.version_text(f'AnnotateBatch.{annotator.__name__}')

        if save_output:
            fout = open(f"{label}.bsv", 'w+')

            meta = [f"# NLP_HOST={os.environ.get('NLP_HOST')}"]
            meta += [f"# BATCH_DIR={self.batch_dir}"]
            meta += [f'# num|filename|len(tokens)|len(str(response))|annotator|timestamp|seconds\n']

            fout.write('\n'.join(meta))

            logger.info(f"output= {label}.bsv")

        for s in self.samples:
            logger.debug(f"{annotator.__name__} {s}")
            text = common.read_text(s)
            tokens = text.split()

            try:
                t = Timer()

                if 'annotate_text' in annotator.__name__:
                    f_annot = f"{s}.annotate_text.json"

                    if os.path.exists(f_annot):
                        logger.info(f'ok {f_annot}')
                        res = 'ok'
                    else:
                        res = annotator(text)
                        common.write_json(res, f_annot)
                else:
                    res = annotator(text)

                msg = f"{complete}|{s}|{len(tokens)}|{len(str(res))}|{annotator.__name__}|{t.stop()}|{t.elapsed()}"

                logger.debug(msg)

                complete += 1
                logger.debug(f'completed {complete} / {len(self.samples)}')

                if save_output:
                    fout.write(f"{msg}\n")
                    fout.flush()

            except Exception as e:
                logger.error(s, e)

                if save_output:
                    common.write_text(str(e), f"{s}.err")


@unittest.skip('https://gettext2phenotype.atlassian.net/browse/BIOMED-674')
class TestBiomed674(unittest.TestCase):

    def setUp(self):
        self.batch = AnnotateBatch()

    def test_ctakes(self):
        self.batch.process(autocode.clinical)
        self.batch.process(autocode.lab_value)
        self.batch.process(autocode.drug_ner)
        self.batch.process(autocode.temporal)

    def test_nlp_cache(self):
        self.batch.process(nlp_cache.clinical)
        self.batch.process(nlp_cache.lab_value)
        self.batch.process(nlp_cache.drug_ner)
        self.batch.process(nlp_cache.temporal)

    def test_biomed(self):
        self.batch.process(annotation.annotate_text)
        self.batch.process(sectionizer.match_sectionizer)

    def test_sands(self):
        sands = BioMedClient()

        self.batch.process(sands.get_clinical_summary)
        self.batch.process(sands.get_phi_tokens)
        self.batch.process(sands.get_demographics)
        self.batch.process(sands.get_demographics_from_model)
