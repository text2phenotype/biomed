import dateparser
import datetime
import os
import re
from typing import List
import spacy
from collections import namedtuple

from biomed.resources import LOCAL_FILES

from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY

TextStore = namedtuple('TextStore', 'text cleaned_text offset_start offset_end')


class PredictedDate:
    def __init__(self, text: str = None, label: str = None, span: List[int] = None, offset_start: int = None,
                 offset_end: int = None):
        self.text = text
        self.label = label
        self.span = span
        self.offset_start = offset_start
        self.offset_end = offset_end
        self._doc_span = None

        if text:
            self.normalized_date = self.__norm_date(text)

    @property
    def doc_span(self):
        if not self._doc_span:
            self._doc_span = [self.span[0] + self.offset_start, self.span[1] + self.offset_start]
        return self._doc_span

    @staticmethod
    def __norm_date(raw_date):
        try:
            parsed_date = str(
                dateparser.parse(raw_date, settings={'RETURN_AS_TIMEZONE_AWARE': False, 'STRICT_PARSING': True}))

            return datetime.datetime.strptime(parsed_date, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')
        except:
            pass


class DOSModel:

    def __init__(self):
        model_root = os.path.join(LOCAL_FILES, 'dos')

        # there is a priority to which models predictions trump other models.
        # this will be implied in how the models are grouped and the order they are in the list.
        # procedure/specimen > encounter > admit > discharge > report
        # this should be able to go away once we have a unified model.
        # models that are allowed to do whatever they want (order not important)
        self.__unrestricted_models = [
            os.path.join(model_root, 'procedure-date-20210428'),
            os.path.join(model_root, 'specimen-date-20210428')
        ]

        # models that that can lose if they predict the same date as another model (order IS important)
        self.__restricted_models = [
            os.path.join(model_root, 'encounter-date-20210525'),
            os.path.join(model_root, 'admission-date-20210428'),
            os.path.join(model_root, 'discharge-date-20210428'),
            os.path.join(model_root, 'report-date-20210428')
        ]

    def predict(self, text) -> List[PredictedDate]:
        predicted = {}
        text_stores = self.init_text_stores(text)

        self.__predict(text_stores, self.__unrestricted_models, predicted, True)
        self.__predict(text_stores, self.__restricted_models, predicted, False)

        return [predicted[span] for span in sorted(list(predicted.keys()))]

    @staticmethod
    def init_text_stores(text: str) -> List[TextStore]:
        result = []
        offset_start = 0  # zero based character offset of current page start relative to full text
        for t in text.split(OCR_PAGE_SPLITTING_KEY[0]):
            if not t:
                offset_start += 1
                continue
            offset_end = offset_start + len(t)
            result.append(TextStore(t, DOSModel.__clean_text(t), offset_start, offset_end))
            offset_start = offset_end + 1
        return result

    @staticmethod
    def __predict(text_stores, models, predictions, keep_duplicates):
        for model_path in models:
            model = spacy.load(model_path)
            for page in text_stores:
                doc = model(page.cleaned_text)

                if not doc.ents:
                    continue

                for ent in doc.ents:
                    entity_text = ent.text
                    page_offset_start = DOSModel.__get_original_text_start(
                        search=entity_text,
                        original_text=page.text,
                        clean_text=page.cleaned_text,
                        clean_start_offset=doc[ent.start].idx)
                    span = (page_offset_start, len(entity_text) + page_offset_start)

                    predicted = PredictedDate(entity_text, ent.label_, list(span), page.offset_start,
                                              page.offset_end)
                    if not predicted.normalized_date:
                        continue

                    if keep_duplicates or (not keep_duplicates and tuple(predicted.doc_span) not in predictions):
                        predictions[tuple(predicted.doc_span)] = predicted
            del model

    @staticmethod
    def __clean_text(text):
        """clean the text """
        text = text.replace('\n', ' ')  # make a single line
        text = re.sub('[^A-Za-z0-9\/:\-.,\(\) ]+', ' ', text)   # remove special chars
        text = re.sub(r'http\S+', ' ', text)   # remove URLs
        text = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.'
                      r'[a-z]{2,4}\b'
                      r'([-a-zA-Z0-9@:%_\+.~#?&//=]*)', ' ', text)   # remove even more URLs
        text = re.sub(r'[A-Za-z]:', r'\g<0> ', text)    # make sure there is a space after a colon
        
        return re.sub(' +', ' ', text).strip()  # remove multispaces

    @staticmethod
    def __get_original_text_start(search, original_text, clean_text, clean_start_offset):
        """Finds the first instance of a substring in the raw ocr text string, given the clean text and the start
        offset in the clean text

        :param search: word/item to be found in the text
        :param original_text: The original text to be processed.
        :param clean_text: The cleaned text.
        :param clean_start_offset: The start position of the prediction in the cleaned text.
        :return: The start position of the prediction in the original text, or the input start position if predicted
                 text cannot be located in the original text.
        """
        search = search.translate({ord(c): " " for c in r"()"})

        # find all matches of the search string in the clean text.
        # stop count when offset in clean text matches the current offset.
        match_number = 0
        for m in re.finditer(search, clean_text):
            match_number += 1

            if m.start() == clean_start_offset:
                break

        # find all matches of the search string in raw ocr text.
        # stop count when offset in raw text matches offset number from the clean text
        for m in re.finditer(search, original_text):
            match_number -= 1
            if not match_number:
                return m.start()

        return clean_start_offset
