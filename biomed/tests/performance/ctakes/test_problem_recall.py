import os
import unittest

from feature_service.nlp import autocode

from biomed.tests.performance.ctakes.base import CtakesAnnotator


class ProblemAnnotator(CtakesAnnotator):
    """Problem term annotator."""
    def _get_autocode_entities(self, text: str):
        return autocode.problem(text)['result']
        # return nlp.autocode(text, nlp.PipelineURL.snomedct)['result']


class TestRecall(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__annotator = ProblemAnnotator()

    def test_MIMIC(self):
        self.__test_recall('MIMIC_diagnoses.txt', 'MIMIC_unknown_problem.txt')

    def test_deleys_bluebutton(self):
        self.__test_recall('deleys_bluebutton_problem.txt', 'deleys_bluebutton_unknown_problem.txt')

    def test_deleys_mtsamples(self):
        self.__test_recall('deleys_mtsamples_problem.txt', 'deleys_mtsamples_unknown_problem.txt')

    def __test_recall(self, term_file, out_file):
        self.__annotator.recall_expected(os.path.join(os.path.dirname(__file__), term_file),
                                         os.path.join(os.path.dirname(__file__), out_file))


if __name__ == '__main__':
    unittest.main()
