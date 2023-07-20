from enum import Enum
import os
import pickle
import string
from typing import Dict, List, Tuple

from biomed.resources import LOCAL_FILES

from text2phenotype.common.log import operations_logger
from text2phenotype.entity.attributes import Polarity
from text2phenotype.entity.concept import Serializable

from text2phenotype.constants.umls import Vocab, PROBLEM_TTY, DRUG_TTY


########################################################
#
# Qualifier
#
########################################################
class Qualifier(Serializable):
    def __init__(self, cui: str, code, preferredText: str):
        """
        :param cui: concept unique identifier
        :param code: coded value for qualifier
        :param preferredText: text for qualifier
        :return: Concept SNOMEDCT_US standard, tui type is Qualifier, tty is PT Preferred TERM
        """
        self.cui = cui
        self.code = str(code)
        self.preferredText = preferredText
        self.vocab = Vocab.SNOMEDCT_US.name
        self.tui = ['T080'] # Qualitative Concept (UMLS)

    @staticmethod
    def represent(mention:Dict, qual)->Dict:
        """
        :param mention: JSON of (un)qualified concept match
        :param qual: known Qualifier
        :return: Dictionary representation
        """
        if qual is None:
            return
        if isinstance(qual, Enum):
            qual = qual.value

        rep = mention.copy()
        if isinstance(qual, Qualifier):
            rep['cui'] = qual.cui
            rep['tui'] = qual.tui
            rep['code'] = qual.code
            rep['codingScheme'] = qual.vocab
            rep['preferredText'] = qual.preferredText
            rep['polarity'] = Polarity.POSITIVE

        return rep


########################################################
#
# Stage
#
########################################################
STAGE_VECTOR_LENGTH = 128   # this is the length for an individual stage, not the total length that gets built during encoding
CLINICAL_VECTOR_START = 0
T_VECTOR_START = STAGE_VECTOR_LENGTH
N_VECTOR_START = STAGE_VECTOR_LENGTH * 2
M_VECTOR_START = STAGE_VECTOR_LENGTH * 3


def clean_raw_stage_string(stage_str: str) -> str:
    stage_str = stage_str.lower()

    # clear any qualifying text
    raw_stage = stage_str.replace('pathologic', '').replace('clinical', '')

    # remove the word stage, and also move clinical stage to the front of the string
    # staring with "age" since I have plenty of examples of OCR have mangled the full word
    stage_split = raw_stage.split('age')
    if len(stage_split) == 1:
        raw_stage = stage_split[0]
    else:
        # make sure the clinical stage info is at the beginning of the string
        if len(stage_split) > 2:
            operations_logger.warning(f'Found multiple instances of stage: {stage_str}')

        # remove residual 'st', 't'
        for c in ['t', 's']:
            if stage_split[0].endswith(c):
                stage_split[0] = stage_split[0][:-1]
            else:
                break
        raw_stage = ''.join(reversed(stage_split))

    # clear any punctuation
    for c in string.punctuation:
        if c == '|':  # leave this since OCR confuses I/1 with |
            continue

        replace_char = ''

        raw_stage = raw_stage.replace(c, replace_char)

    return raw_stage


def encode_raw_stage_string(stage_str: str) -> List[int]:
    cleaned_stage = clean_raw_stage_string(stage_str)

    # codes the T/N/M & clinical into a single vector, where each section of the larger vector represents
    # information for a single stage component
    x_vector = [0] * STAGE_VECTOR_LENGTH * 4
    offset = CLINICAL_VECTOR_START
    for index, c in enumerate(cleaned_stage):
        ordinal = ord(c)
        if ordinal >= STAGE_VECTOR_LENGTH:  # this is just OCR junk
            continue

        if c == 't':
            offset = T_VECTOR_START
            continue

        if c == 'n':
            offset = N_VECTOR_START
            continue

        if c == 'm':
            offset = M_VECTOR_START
            continue

        x_vector[ordinal + offset] += 1

    # for clinical, see if there is a match to an expected clinical string
    # process as longest to shortest string, and only encode the first match
    clinical_values = []
    for stage in ClinicalStage:
        for stage_value in stage.value:
            clinical_values.append(stage_value)
    clinical_values.sort(key=len, reverse=True)

    raw_stage = cleaned_stage.replace(' ', '')
    for stage_value in clinical_values:
        if stage_value in raw_stage:
            for c in stage_value:
                x_vector[CLINICAL_VECTOR_START + ord(c)] += 1
            break

    return x_vector


class TStage(Enum):
    A = {'a'}
    X = {'x'}
    IN_SITU = {'is'}
    ZERO = {'0'}
    ONE = {'1'}
    TWO = {'2'}
    THREE = {'3'}
    FOUR = {'4'}


class NStage(Enum):
    X = {'x'}
    ZERO = {'0'}
    ONE = {'1'}
    TWO = {'2'}
    THREE = {'3'}


class MStage(Enum):
    X = {'x'}
    ZERO = {'0'}
    ONE = {'1'}


class ClinicalStage(Enum):
    ZERO_A = {'Oa'}     # bladder-specific stage
    I = {'i', '1'}
    IA1 = {'ia1', '1a1'}
    IA2 = {'ia2', '1a2'}
    IB1 = {'ib1', '1b1'}
    IB2 = {'ib2', '1b2'}
    IB3 = {'ib3', '1b3'}
    II = {'ii', '2'}
    IIA = {'iia', '2a'}
    IIA2 = {'iia2', '2a'}
    IIB = {'iib', '2b'}
    IIC = {'iic', '2c'}
    III = {'iii', '3'}
    IIIA = {'iiia', '3a'}
    IIIB = {'iiib', '3b'}
    IIIC = {'iiic', '3c'}
    IIIC1 = {'iiic1', '3c1'}
    IV = {'iv', '4'}
    IVA = {'iva', '4a'}
    IVB = {'ivb', '4b'}
    IVC = {'ivc', '4c'}


class Stage:
    __MODEL_DIR = os.path.join(LOCAL_FILES, 'cancer')
    __T_STAGE_CLASSIFIER = None
    __M_STAGE_CLASSIFIER = None
    __N_STAGE_CLASSIFIER = None
    __CLINICAL_STAGE_CLASSIFIER = None

    def __init__(self, t: TStage = None, n: NStage = None, m: MStage = None, clinical: ClinicalStage = None):
        self.__t = t
        self.__n = n
        self.__m = m
        self.__clinical = clinical
        
    @property
    def T(self):
        return self.__t

    @T.setter
    def T(self, value):
        self.__t = value

    @property
    def N(self):
        return self.__n

    @N.setter
    def N(self, value):
        self.__n = value

    @property
    def M(self):
        return self.__m

    @M.setter
    def M(self, value):
        self.__m = value

    @property
    def clinical(self):
        return self.__clinical

    @clinical.setter
    def clinical(self, value):
        self.__clinical = value

    @classmethod
    def from_string(cls, stage_str: str) -> 'Stage':
        cls.__load_classifiers()

        stage = cls()
        encoding = encode_raw_stage_string(stage_str)

        t_index = cls.__T_STAGE_CLASSIFIER.predict([encoding[T_VECTOR_START:(T_VECTOR_START+STAGE_VECTOR_LENGTH)]])[0]
        stage.T = cls.__prediction_to_enum(t_index, TStage)

        m_index = cls.__M_STAGE_CLASSIFIER.predict([encoding[M_VECTOR_START:(M_VECTOR_START+STAGE_VECTOR_LENGTH)]])[0]
        stage.M = cls.__prediction_to_enum(m_index, MStage)

        n_index = cls.__N_STAGE_CLASSIFIER.predict([encoding[N_VECTOR_START:(N_VECTOR_START+STAGE_VECTOR_LENGTH)]])[0]
        stage.N = cls.__prediction_to_enum(n_index, NStage)

        clinical_index = cls.__CLINICAL_STAGE_CLASSIFIER.predict([encoding[CLINICAL_VECTOR_START:(CLINICAL_VECTOR_START+STAGE_VECTOR_LENGTH)]])[0]
        stage.clinical = cls.__prediction_to_enum(clinical_index, ClinicalStage)

        return stage

    @classmethod
    def __load_classifiers(cls):
        if not cls.__T_STAGE_CLASSIFIER:
            cls.__T_STAGE_CLASSIFIER = cls.__load_classifier('20210723_t_stage_knn.pkl')
            cls.__N_STAGE_CLASSIFIER = cls.__load_classifier('20210723_n_stage_knn.pkl')
            cls.__M_STAGE_CLASSIFIER = cls.__load_classifier('20210723_m_stage_knn.pkl')
            cls.__CLINICAL_STAGE_CLASSIFIER = cls.__load_classifier('20210723_clinical_stage_knn.pkl')

    @classmethod
    def __load_classifier(cls, pkl_file):
        classifier_file = os.path.join(cls.__MODEL_DIR, pkl_file)

        operations_logger.debug(f'Loading classifier from {classifier_file}...')

        with open(classifier_file, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def __prediction_to_enum(prediction: int, enum_type: Enum):
        if prediction:
            # -1 b/c training encodes "missing" as index 0
            return enum_type[list(enum_type._member_map_)[prediction - 1]]


########################################################
#
# Grade (Cancer Histological Grade)
#
########################################################

class Grade(Enum):

    G1_well = Qualifier('C0205615', 1, 'Well Differentiated (Low grade)')
    G2_moderate = Qualifier('C0205616', 2, 'Moderately Differentiated (Intermediate grade)')
    G3_poor = Qualifier('C0205617', 3, 'Poorly Differentiated (High grade)')
    G4_undifferentiated = Qualifier('C0205618', 4, 'Undifferentiated')
    G9_unknown = Qualifier('C0456201', 9, 'Histological Grade Mention, NOS')


def __make_grade_search_list(grade: Grade, terms: List[str]) -> List[Tuple[Grade, str]]:
    return [(grade, term) for term in terms]


def parse_grade(text: str, is_stage_text: bool = False) -> Grade:
    """
    :param text: str predicted label=grade
    :param is_stage_text: Flag to indicate if this is text that was identified as Stage instead of Grade.
    :return: Grade
    """
    g1_abbr = __make_grade_search_list(Grade.G1_well, ['LG'])
    g1 = __make_grade_search_list(Grade.G1_well, ['1', 'WELL', 'LOW']) + g1_abbr
    g2 = __make_grade_search_list(Grade.G2_moderate, ['2', 'II', 'MODERATE', 'INTERMEDIATE', 'MEDIUM'])
    g3_abbr = __make_grade_search_list(Grade.G3_poor, ['HG'])
    g3 = __make_grade_search_list(Grade.G3_poor, ['3', 'III', 'POOR', 'HIGH']) + g3_abbr
    g4 = __make_grade_search_list(Grade.G4_undifferentiated, ['4', 'IV', 'UNDIFFER', 'UN-DIFF', 'ANAPLAS'])

    # note that the lists are order-specific.
    # i.e. if you change them, very bad things may happen.  yes, you.
    if is_stage_text:
        search_list = g3_abbr + g1_abbr
    else:
        search_list = g4 + g3 + g2 + g1

    text = text.upper()
    for grade, term in search_list:
        if term in text:
            return grade

    return Grade.G9_unknown

########################################################
#
# Behavior
#
########################################################

class Behavior(Enum):
    B0_benign = Qualifier('C2865391', 0, 'Benign Neoplasms')
    B1_uncertain = Qualifier('C2865391', 1, 'Neoplasms of uncertain behavior')
    B2_in_situ = Qualifier('C1265999', 2, 'In situ neoplasms')
    B3_malignant_primary = Qualifier('C4267833', 3, 'Malignant neoplasms, stated or presumed to be primary')
    B6_malignant_secondary = Qualifier('C3266877', 6, 'Malignant neoplasms, stated or presumed to be secondary')

def parse_behavior(text:str) -> Behavior:
    """
    :param text: string predicted as label=behavior
    :return: Behavior or None if not known to this parser
    """
    B0 = ['BENIGN']
    B1 = ['UNCERTAIN', 'NOS', 'UNSPEC', 'NOT SPECIFIED', 'NOT-SPEC', 'UNKN', 'INDETERMIN']
    B2 = ['SITU', 'IN-SIT', 'DCIS']
    B3 = ['INVAS', 'INVAD', 'INFLITRAT', 'INFILTRAT', 'MALIGNAN']
    B6 = ['METASTA']

    for match in B6:
        if match in text.upper():
            return Behavior.B6_malignant_secondary

    for match in B3:
        if match in text.upper():
            return Behavior.B3_malignant_primary

    for match in B2:
        if match in text.upper():
            return Behavior.B2_in_situ

    for match in B1:
        if match in text.upper():
            return Behavior.B1_uncertain

    for match in B0:
        if match in text.upper():
            return Behavior.B0_benign


########################################################
#
# Select concept from preferred vocabulary
# SNOMEDCT_US (Clinical Terms US edition) is default
#
########################################################
def select_pref_snomed(concept_list: list):
    """
    :param concept_list: list of UMLS concept dict
    :return: dict concept or None
    """
    priority = [PROBLEM_TTY.PT,
                PROBLEM_TTY.FN,
                None]

    for tty in priority:
        pref = select_pref(concept_list, Vocab.SNOMEDCT_US, tty)
        if pref:
            return pref

def select_pref_rxnorm(concept_list:list):
    """
    :param concept_list: list of UMLS concept dict ( Medications )
    :return: dict concept or None
    """
    priority = [DRUG_TTY.PSN,
                DRUG_TTY.SCD,
                DRUG_TTY.SCDC,
                DRUG_TTY.SBDF,
                DRUG_TTY.IN,
                None]

    for tty in priority:
        pref = select_pref(concept_list, Vocab.RXNORM, tty)
        if pref:
            return pref

def select_pref(concept_list: list, vocab:Vocab, tty=None):
    """
    :param concept_list: list of UMLS concept dict
    :param vocab: SNOMEDCT_US is default
    :param tty: https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html
    :return: dict concept or None
    """
    if concept_list is None: return None
    if 0 == len(concept_list): return None
    if 1 == len(concept_list): return dict(concept_list[0])

    if isinstance(tty, Enum):
        tty = str(tty.name)

    for candidate in concept_list:

        # sab is UMLS "source abbreviation" or "vocab" this goes by different names for no good reason.
        sab = candidate.get('codingScheme') or candidate.get('vocab')

        if vocab.name == sab:
            if tty is None:
                return candidate
            elif tty == candidate.get('tty'):
                return candidate
