import unittest

from biomed.summary.text_to_summary import text_to_summary
from feature_service.hep_c.answers import QuestionAnswers
from biomed.tests.samples import LAB_RESULT_TXT

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger


"""
mysql> 
select 'C0028158' into @cui ;

mysql>  
SELECT distinct C.SAB, C.STR, S.TUI, S.STY 
FROM MRCONSO C, MRSTY S 
WHERE C.SAB like '%LNC%' and C.LAT = 'ENG' and 
C.CUI = S.CUI and C.CUI = @cui order by SAB,STY;
+-----+----------+------+--------------------------+
| SAB | STR      | TUI  | STY                      |
+-----+----------+------+--------------------------+
| LNC | Nitrogen | T196 | Element, Ion, or Isotope |
+-----+----------+------+--------------------------+

select 'C0010294' into @cui ;
+-----+------------+------+-------------------------------+
| SAB | STR        | TUI  | STY                           |
+-----+------------+------+-------------------------------+
| LNC | Creatinine | T123 | Biologically Active Substance |
| LNC | Creatinine | T109 | Organic Chemical              |
+-----+------------+------+-------------------------------+

select 'C0006675' into @cui ;
+-----+---------+------+-------------------------------+
| SAB | STR     | TUI  | STY                           |
+-----+---------+------+-------------------------------+
| LNC | Calcium | T123 | Biologically Active Substance |
| LNC | Calcium | T196 | Element, Ion, or Isotope      |
| LNC | Calcium | T121 | Pharmacologic Substance       |
+-----+---------+------+-------------------------------+

select 'C0302583' into @cui ;
+-----+------+------+-------------------------------+
| SAB | STR  | TUI  | STY                           |
+-----+------+------+-------------------------------+
| LNC | Iron | T123 | Biologically Active Substance |
| LNC | Iron | T196 | Element, Ion, or Isotope      |
| LNC | Iron | T121 | Pharmacologic Substance       |
+-----+------+------+-------------------------------+

select 'C0008377' into @cui ;
+-----+-------------+------+-------------------------------+
| SAB | STR         | TUI  | STY                           |
+-----+-------------+------+-------------------------------+
| LNC | Cholesterol | T123 | Biologically Active Substance |
| LNC | Cholesterol | T109 | Organic Chemical              |
+-----+-------------+------+-------------------------------+
"""

# NOT IN LNC (LOINC lab tests)
"""
C0041980]  Uric Acid
"""


class TestAspectLab(unittest.TestCase):

    # @unittest.skip('BIOMED-275')
    def test_biomed_275_lab_values_on_common_labs_in_tabular_format(self):
        text = common.read_text(LAB_RESULT_TXT)

        res = text_to_summary(text)

        operations_logger.info(res)
        # _summary = pref_terms.text2summary(_text)

    def test_glucose(self):
        answers = QuestionAnswers()
        _cuis = answers.lookup.get('lab_glucose').keys()

        self.assertTrue('C0017725' in _cuis)
