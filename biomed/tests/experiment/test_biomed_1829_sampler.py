import unittest
import os, re
from typing import Dict, List, Set

from feature_service.features.feature import Feature
from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY
from text2phenotype.entity.attributes import Serializable
from text2phenotype.common import common
from feature_service.features.covid import CovidRegex, CovidDeviceRegex, CovidDeviceMatchHint

TEST_OUTPUT     = os.environ.get('TEST_OUTPUT', False)
DIR_BIOMED_1829 = os.environ.get('DIR_BIOMED_1829', '/mnt/inbox/COVID-BIOMED-1829')

###############################################################################
# Reader to output regex machine annotations in human readable spreadsheet
#
# feature_service.features.regex
#
###############################################################################
class RegExBaseReader(Serializable):
    """
    RegExBase Reader class, serializable to table and BSV format
    """
    def __init__(self, source=None):
        self.list_tokens = list()
        self.list_labels = list()
        self.source = None

        if source:
            self.from_json(source)

    def from_json(self, source: Dict) -> None:
        """
        :param source:
        :return:
        """
        for textspan, values in source:
            for v in values:
                if isinstance(v, dict):
                    for label, token in v.items():
                        self.list_tokens.append(token)
                        self.list_labels.append(label)
                if isinstance(v, str):
                    self.list_tokens.append(v)

    def to_tokenstring(self, unique=False)->str:
        """
        :param unique: True=uniq set of tokens, False=list each token;
        :return: string of tokens joined by ','
        """
        if unique:
            return ','.join(list(set(self.list_tokens)))
        return ','.join(self.list_tokens)

###############################################################################
#
# OCR Pages
#
###############################################################################

def list_ocr_pages(text:str)->List[str]:
    """
    :param text: OCR text provided by DataFit with the OCR_PAGE_SPLITTING_KEY
    :return: List of texts, each list item is 1 page
    """
    pattern = re.compile(OCR_PAGE_SPLITTING_KEY[0])
    pages = list()
    cursor = 0
    for match in pattern.finditer(text):
        pages.append(text[cursor:match.start()])
        cursor = match.start()

    if cursor < len(text):
        pages.append(text[cursor:])

    return pages

###############################################################################
#
# OCR Pages
#
###############################################################################

def slice_corpus_covid(corpus:List[str], human_file:str, pagesize=10)->None:
    """
    Slice a corpus of text documents into smaller documents, breaking on OCR pages
    Return spreadsheet report for human experts that suggests which pages to review

    :param corpus: list of files to slice
    :param human_file: output path to "human expert review" spreadsheet
    :param pagesize: number of OCR pages
    :return: tsv "tab seperated values" report of the corpus
    """
    spreadsheet = list()
    spreadsheet.append(".txt\tpage\trange\tCovidRegex\tCovidDeviceRegex\tCovidDeviceMatchHint")

    for f in corpus:
        filename = f.split('/')[-1]
        filename = filename.replace('.txt', '')

        pages = list_ocr_pages(common.read_text(f))
        spreadsheet.append(f"{filename}.txt\t0\t{len(pages)}")

        print(f"{f}\tpages = {len(pages)}")

        parent = os.path.join(DIR_BIOMED_1829, 'out', filename)
        if not os.path.exists(parent):
            os.makedirs(parent)

        previous = 0

        while (previous < len(pages) - 1):
            next = previous + pagesize

            if next > len(pages):
                next = len(pages)

            slice = '\n'.join(pages[previous:next])
            slice_txtfile = f"{filename}_{previous + 1}_{next}.txt"

            common.write_text(slice, os.path.join(parent, slice_txtfile))

            labtests = RegExBaseReader(CovidRegex().annotate(slice)).to_tokenstring()
            devices = RegExBaseReader(CovidDeviceRegex().annotate(slice)).to_tokenstring()
            keywords = RegExBaseReader(CovidDeviceMatchHint().annotate(slice)).to_tokenstring()

            if labtests:
                spreadsheet.append(f"{slice_txtfile}\t{previous}\t{next}\t{labtests}\t{devices}\t{keywords}")
            previous = next

        common.write_text('\n'.join(spreadsheet), human_file)


def get_pages_with_feature_annotation(text, feature: Feature) -> set:
    pages = list_ocr_pages(text)
    matching_pages = set()
    for i in range(len(pages)):
        page_annot = feature.annotate(text=pages[i])
        if len(list(page_annot)) > 0:
            matching_pages.add(i)
    return matching_pages

def get_pages_with_regex_pattern(text, regex_pattern) -> set:
    pages = list_ocr_pages(text)
    matching_pages = set()
    for i in range(len(pages)):
        if re.search(regex_pattern, pages[i], flags=re.IGNORECASE):
            matching_pages.add(i)
    return matching_pages

def get_n_page_segs(pages: Set[int], max_splice_len):
    out_segs = list()
    while len(pages) > 0:
        min_page = min(pages)
        end = min_page + max_splice_len
        out_segs.append((min_page, end))
        to_remove = {pg for pg in pages if pg < end}
        for page in to_remove:
            pages.remove(page)
    return out_segs




###############################################################################
#
# Test
#
###############################################################################
class TestBiomed1829(unittest.TestCase):

    def setUp(self) -> None:        self.samples = common.get_file_list(DIR_BIOMED_1829, '.txt')

    def test_ocr_page_identity_transform(self):
        for s in self.samples:
            expected = common.read_text(s)
            actual = ''.join(list_ocr_pages(expected))
            self.assertEqual(expected, actual)

    def slice_corpus(self):
        if TEST_OUTPUT:
            spreadsheet = os.path.join(DIR_BIOMED_1829, 'out', 'COVID-BIOMED-1829.tsv')
            slice_corpus_covid(self.samples, spreadsheet)


class TestBiomed1881(TestBiomed1829):
    def setUp(self):
        self.samples = common.get_file_list('/Users/shannon.fee/Downloads/covid-2_antibody_200', '.txt')


    def test_get_pages_with_regex(self):
        total = []
        num = 10
        for file in common.get_file_list('/Users/shannon.fee/Downloads/covid-2_antibody_200', '.txt'):
            text = common.read_text(file)
            a= get_pages_with_regex_pattern(text, f'(antibo|igg|igm|serology|\biga)')
            # print(a)
            b = get_n_page_segs(a, num)
            pages = list_ocr_pages(text)
            for seg in b:
                start_page = seg[0]
                end_page = seg[1]
                text = '\n'.join(pages[start_page:  end_page])
                new_file = os.path.split(file)[0].replace('covid-2_antibody_200', 'covid_2_antibody_splices')
                file_id = f"{os.path.split(file)[1].replace('.txt', '')}_{start_page}_{end_page}.txt"
                common.write_text(text, os.path.join(new_file, file_id))


            # print(b)

    def test_finding_cov(self):
        files =  common.get_file_list('/Users/shannon.fee/Downloads/covid_2_antibody_splices', '.txt')

        for fp in files:
            txt = common.read_text(fp)
            b = get_pages_with_feature_annotation(text=txt, feature=CovidRegex())
            if len(b)> 1:
                common.write_text(txt, fp.replace('covid_2_antibody_splices', 'covid_2_antibody_splices_cov'))