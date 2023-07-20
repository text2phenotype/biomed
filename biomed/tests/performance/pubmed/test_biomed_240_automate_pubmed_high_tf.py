import os
import unittest
import random

from biomed.summary.text_to_summary import text_to_summary
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger

from feature_service.nlp import nlp_cache

from biomed.deid import deid
from biomed.tests.performance.pubmed import medline
from biomed.tests.samples import PUBMED_MEDLINE_DIR
from biomed.tests.timer import Timer, TimerBatch


class TestBiomed240(unittest.TestCase):

    def setUp(self):
        self.iterations = os.environ.get('TEST_ITERATIONS', 100)

    def process_abstracts(self, functor, cache_filetype) -> TimerBatch:
        operations_logger.info(f'process_abstracts {PUBMED_MEDLINE_DIR} {functor} {cache_filetype}')

        files = common.get_file_list(PUBMED_MEDLINE_DIR, 'xml')
        batch = TimerBatch()

        random.shuffle(files)

        for i in range(1, self.iterations):
            for f in files:
                abstracts = medline.get_abstracts(f)
                random.shuffle(abstracts)

                for abstract in abstracts:

                    docid = nlp_cache.hash_text(str(abstract))
                    abstract = str(abstract)

                    nlp_cache.save_text(abstract)

                    t = Timer(name=docid, content=cache_filetype)

                    nlp_cache.autocode_cache(abstract, functor, cache_filetype)

                    prct = float(batch.size() / self.iterations)

                    t.stop()
                    batch.add(t)

                    operations_logger.info(f'{t.to_dict()}')
                    operations_logger.info(f'Batch {batch.size()} of {self.iterations} ( %{prct*100} )')

                    if batch.size() >= self.iterations:
                        return batch

    def test_temporal(self):
        """
        Temporal
        """
        self.process_abstracts(functor=nlp_cache.temporal, cache_filetype='temporal.json')

    def test_biomed_240_pref_terms_text2summary(self):
        """
        Test summary ( biomed.aspect enabled text2summary )
        """
        self.process_abstracts(functor=text_to_summary, cache_filetype='pref_terms.text2summary.json')

    def test_biomed_240_deid(self):
        """
        DEID pubmed abstracts -- should return few (or 0) entries for every abstract
a       """
        self.process_abstracts(functor=deid.get_phi_tokens, cache_filetype='deid.get_phi_tokens.json')

    def test_lab_value(self):
        """
        lab_value
        """
        self.process_abstracts(functor=nlp_cache.lab_value, cache_filetype='lab_value.json')

    def test_drug_ner(self):
        """
        drug_ner
        """
        self.process_abstracts(functor=nlp_cache.drug_ner, cache_filetype='drug_ner.json')
