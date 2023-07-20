import os
import unittest

from feature_service.nlp import autocode

from biomed.tests.performance.ctakes.base import CtakesAnnotator


class LabAnnotator(CtakesAnnotator):
    """Lab term annotator."""
    def _get_autocode_entities(self, text: str):
        # return nlp.lab_value(text)['labValues']
        return autocode.autocode(text, autocode.dest('lab_master', autocode.Pipeline.lab_value))['labValues']
        # return nlp.loinc_lab_value(text)['labValues']
        # return nlp.hepc_lab_value(text)['labValues']
        # return LabHepc().annotate(text)
        # return LabLoinc().annotate(text)


class TestRecall(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__annotator = LabAnnotator()

    def test_MIMIC(self):
        self.__test_recall('MIMIC_labs.txt', 'MIMIC_unknown_labs.txt')

    def test_mtsamples_clean(self):
        self.__test_recall('mtsamples_clean_labs.txt', 'mtsamples_clean_unknown_labs.txt')

    def test_deleys_bluebutton(self):
        self.__test_recall('deleys_bluebutton_lab.txt', 'deleys_bluebutton_unknown_labs.txt')

    def __test_recall(self, term_file, out_file):
        self.__annotator.recall_expected(os.path.join(os.path.dirname(__file__), term_file),
                                         os.path.join(os.path.dirname(__file__), out_file))


if __name__ == '__main__':
    unittest.main()
