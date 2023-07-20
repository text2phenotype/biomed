import os
import unittest

from text2phenotype.annotations.file_helpers import AnnotationSet

from text2phenotype.common import common
from text2phenotype.entity.brat import Annotation, BratReader

from feature_service.nlp.nlp_reader import SummaryReader

from biomed.tests.samples import MTSAMPLES_DIR
from biomed.biomed_env import BiomedEnv

TEXT_EXPECTED = "T1\tdiseasesign 2 17\tCHIEF COMPLAINT\n" \
                "A1\theader T1 true\n" \
                "A2\tperson T1 patient\n" \
                "A3\treltime T1 admit\n" \
                "A4\tpolarity T1 positive\n" \
                "A5\tcode T1 false\n"


class TestBiomed343(unittest.TestCase):
    def test_to_brat_headers(self, output=True):

        for f in common.get_file_list(os.path.join(MTSAMPLES_DIR), '.txt'):
            source = common.read_text(f)
            reader = AnnotationSet()
            for match in Header().annotate(source):
                span, text_aspect = match

                for hit in text_aspect:
                    a = Annotation()

                    for text, aspect in hit.items():
                        a.text_range = span
                        a.text = text
                        a.label = aspect.replace('Aspect.', '')

                        reader.annotations[span] = a

            if output:
                common.write_text(reader.to_brat(), f.replace('clean.txt', 'clean.ann'))

    def test_read_summary_write_brat(self, output=True):
        # text2phenotype-samples/mtsamples/new_summary_mtsamples

        MTSAMPLES_SUMMARIES_DIR = os.path.join(BiomedEnv.DATA_ROOT.value, 'mtsamples', 'new_summary_mtsamples')

        for f in common.get_file_list(MTSAMPLES_SUMMARIES_DIR, '.json'):
            problem = common.read_json(f)

            reader = SummaryReader(problem['text2summary'])
            id = 0
            out = ''
            uniq = set()

            for problem in reader.problems:
                id += 1

                problem['id'] = id
                problem['aspect'] = 'diseasesign'  # NOTE: should this be changed here, and in the expected annotation?
                problem['code'] = None

                annot = Annotation(problem)

                if annot.spans not in uniq:
                    uniq.add(annot.spans)
                    out += annot.to_brat()

            if output and len(out) > 0:
                common.write_text(out, f.replace('_new_summary.json', '.txt-clean.ann'))