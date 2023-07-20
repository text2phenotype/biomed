import unittest

from feature_service.hep_c.form import autofill_hepc_form
from feature_service.nlp.nlp_reader import HepcReader

from text2phenotype.common import common
from text2phenotype.common.feature_data_parsing import from_bag_of_words

from biomed.tests import samples


class Review:
    def __init__(self):
        self.text = list()
        self.cui = list()


class TestBiomed501(unittest.TestCase):

    def test_candidates_falsepos(self, do_output=True):

        all = Review()
        uniq = Review()
        formfill = Review()

        for f in common.get_file_list(samples.MTSAMPLES_DIR, '.txt'):
            text = common.read_text(f)

            all.text += HepcReader(text).list_result_text()
            all.cui += HepcReader(text).list_concept_cuis()

            uniq.text += HepcReader(text).uniq_result_text()
            uniq.cui += HepcReader(text).uniq_concept_cuis()

            for heading, questions in autofill_hepc_form(text).items():
                for q in questions:

                    if q['evidence'] is not None:
                        suggest = q['suggest']
                        evidence = q['evidence']['text'][0]

                        hit = f"{suggest}|{evidence}"

                        formfill.text.append(hit)

            if do_output:
                common.write_json(from_bag_of_words(all.text), 'hepc_all_text_tf.json')
                common.write_json(from_bag_of_words(uniq.text), 'hepc_uniq_text_tf.json')
                common.write_json(from_bag_of_words(formfill.text), 'hepc_formfill_text_tf.json')
