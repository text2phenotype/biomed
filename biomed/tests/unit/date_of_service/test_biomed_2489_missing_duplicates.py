import unittest

from biomed.date_of_service.model import DOSModel


class TestBiomed2489(unittest.TestCase):
    ENCOUNTER_DATE_PAGE = """
    Patient Wellness Visit 
     Encounter Date: 08/30/2018
Visit Summary continued)
Diagnoses (continued)
COCOCOC
ovovodovodovodovodovodovodov
000000
DOVOOOOOOOOOOooooo
20000000 budovou
Comments
History of CVA (cerebrovascular accident) [397393]
Cigarette nicotine dependence with nicotine-induced disorder 1456604]
Dry eyes [1941911
Patient underweight [286926]
Vitals Recorded in This Encounter"""

    DOS_MODEL = DOSModel()

    def test_extract_encounter_date(self):
        dates = self.DOS_MODEL.predict(self.ENCOUNTER_DATE_PAGE)
        self.assertEqual(len(dates), 1)
        self.assertEqual(dates[0].label, 'encounter_date')
        self.assertEqual(dates[0].normalized_date, '2018-08-30')
        self.assertEqual(dates[0].doc_span, [50, 60])
        self.assertEqual(dates[0].span, [50, 60])
        self.assertEqual(self.ENCOUNTER_DATE_PAGE[dates[0].doc_span[0]:dates[0].doc_span[1]], dates[0].text)

    def test_multiple_duplicate_pages(self):
        num_pages = 3
        txt = "\x0c".join([self.ENCOUNTER_DATE_PAGE] * num_pages)
        dates = self.DOS_MODEL.predict(txt)
        self.assertEqual(len(dates), num_pages)
        for date in dates:
            self.assertEqual(date.label, 'encounter_date')
            self.assertEqual(date.normalized_date, '2018-08-30')
            self.assertEqual(date.span, [50, 60])

        # test doc positioning
        date_doc_ranges = [date.doc_span for date in dates]
        expected_date_doc_ranges = [
            [50, 60],
            [50 + len(self.ENCOUNTER_DATE_PAGE) + 1, 60 + len(self.ENCOUNTER_DATE_PAGE) + 1],
            [50 + 2 * (len(self.ENCOUNTER_DATE_PAGE) + 1), 60 + 2 * (len(self.ENCOUNTER_DATE_PAGE) + 1)]
        ]
        self.assertEqual(date_doc_ranges, expected_date_doc_ranges)
