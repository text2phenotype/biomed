import unittest

from biomed.summary.text_to_summary import add_page_numbers_to_predictions, get_page_indices

from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY


class TextToSummaryTest(unittest.TestCase):
    __OCR_DELIM = OCR_PAGE_SPLITTING_KEY[0]
    __text = f"""Page 1 of chart{__OCR_DELIM}{__OCR_DELIM}Page 2 was intentionally left blank"""

    def test_add_page_numbers_to_predictions(self):
        predicted = {
            "Foo": [
                {"range": [10, 18]},    # start on page 1
                {"range": [25, 30]}     # start on page 3
            ]
        }

        add_page_numbers_to_predictions(self.__text, predicted)

        predictions = predicted["Foo"]

        self.assertEqual(1, predictions[0]["page"])
        self.assertEqual(3, predictions[1]["page"])

    def test_get_page_indices(self):
        expected = [((0, 15), 1), ((16, 16), 2), ((17, len(self.__text)), 3)]

        self.assertListEqual(expected, get_page_indices(self.__text))


if __name__ == '__main__':
    unittest.main()
