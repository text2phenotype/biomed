import datetime
from typing import Dict, List, Union

from biomed.constants.constants import NEARBY_TERM_THRESHOLD
from text2phenotype.common.feature_data_parsing import (
    is_digit, has_numbers, is_int, is_digit_punctuation, probable_med_unit)
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features.label_types import AllergyLabel, CancerLabel, CovidLabLabel, LabLabel, GeneticsLabel


class AttributeWithRange:
    def __init__(self, input: list = None, **kwargs):
        if isinstance(input, AttributeWithRange):
            return input
        self.text: str = kwargs.get('text')
        self.range: List[int] = kwargs.get('range', [None, None])
        self.prob: float = kwargs.get('prob')
        if input is not None:
            self.ingest(input)

    def ingest(self, input_list):
        if len(input_list) in [3, 4]:
            self.text = input_list[0]
            self.range = [int(input_list[1]), int(input_list[2])]
            if len(input_list) == 4 and is_digit(input_list[3]):
                self.prob = float(input_list[3])
        elif len(input_list) > 0:
            operations_logger.warning(f"Input {input_list} is not of a valid format for interpretation")

    def to_output_list(self):
        if self.text:
            return [self.text, self.range[0], self.range[1]]
        return []

    def __bool__(self):
        return self.text is not None


class FloatAttribute(AttributeWithRange):
    def __parse_text_to_float(self):
        if isinstance(self.text, str) and is_digit(self.text):
            self.text = float(self.text)
        else:
            # if thing an attribute that should be a number is not float convertible, then don't include the attribute
            self.text = None

    def ingest(self, input_list):
        super().ingest(input_list)
        self.__parse_text_to_float()


class BiomedOutput:
    def __init__(self, **kwargs):
        self.label = kwargs.get('label')
        # used to remove consolidate text, range and prob processing when output comes in the expected ensembler format
        lstm_model_output = kwargs.get('lstm_model_output', [None, None, [None, None]])

        self.text = kwargs.get('text', lstm_model_output[1])
        self.preferredText = kwargs.get('preferredText', None)
        self.range = kwargs.get('range', lstm_model_output[2]) or [None, None]
        self.lstm_prob = kwargs.get('lstm_prob', lstm_model_output[0]) or kwargs.get('score', 0)
        self.page = kwargs.get('page')

    def __repr__(self):
        """Display the key attributes in the object repr"""
        return (
            f"{self.__class__.__name__}(" +
            f"text={self.text}, " +
            f"range={self.range}, " +
            f"label={self.label}, " +
            f"score={self.lstm_prob}, " +
            ")"
        )

    def to_dict(self):
        return {'text': self.text,
                'range': self.range,
                'score': self.lstm_prob,
                'label': self.label,
                'page':  self.page}

    def __getitem__(self, item):
        return getattr(self, item)

    def valid_range(self):
        value = True
        if len(self.range) != 2:
            value = False
        elif any([i is None for i in self.range]):
            value = False
        elif not all([is_int(j) for j in self.range]):
            value = False
        elif int(self.range[1]) <= self.range[0]:
            value = False
        return value

    @classmethod
    def combine(cls, obj_1, obj_2, text):
        return cls(**cls.create_combined_kwargs(obj_1, obj_2, text))

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        ref_range = cls.combine_range(obj_1, obj_2)
        combined_kwargs = {"lstm_prob": cls.combine_prob(obj_1, obj_2),
                           "range": ref_range,
                           "text": text[ref_range[0]: ref_range[1]].replace('\n', ' '),
                           "label": obj_1.label or obj_2.label,
                           "page": obj_1.page or obj_2.page}
        return combined_kwargs

    @classmethod
    def combine_prob(cls, obj_1, obj_2):
        # choose max prob
        return max(obj_1.lstm_prob, obj_2.lstm_prob)

    @classmethod
    def combine_range(cls, obj_1, obj_2):
        range_1 = obj_1.range
        range_2 = obj_2.range
        return [min(range_1[0], range_2[0]), max(range_1[1], range_2[1])]

    def is_lab_name(self) -> bool:
        return self.label == LabLabel.lab.value.persistent_label


class SummaryOutput(BiomedOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        umlsConcept = kwargs.get('umlsConcept', dict())
        if isinstance(umlsConcept, list):
            self.umlsConcept = umlsConcept[0]
        elif isinstance(umlsConcept, dict):
            self.umlsConcept = umlsConcept
        else:
            self.umlsConcept = dict()

        self.attributes = kwargs.get('attributes') or dict()
        self._tui = None

        self.code = kwargs.get('code', self.umlsConcept.get('code'))
        self.cui = kwargs.get('cui', self.umlsConcept.get('cui'))
        self.tui = kwargs.get('tui', self.umlsConcept.get('tui'))
        self.vocab = kwargs.get('vocab', self.umlsConcept.get('codingScheme'))
        self.preferredText = kwargs.get('preferredText', self.umlsConcept.get('preferredText'))

        self.polarity = kwargs.get('polarity', self.attributes.get('polarity'))

    # NOTE: tuis became lists with 1826 feature updates, this is to ensure tui output is string formatted to match
    # string format downtream processse expect
    @property
    def tui(self) -> str:
        return self._tui

    @tui.setter
    def tui(self, value):
        if isinstance(value, list) and len(value) >= 1:
            self._tui = value[0]
        elif isinstance(value, str):
            self._tui = value
        else:
            self._tui = None

    @staticmethod
    def ingest_attribute(val, float_attr: bool = False) -> Union[AttributeWithRange, FloatAttribute]:
        if isinstance(val, AttributeWithRange):
            return val
        elif float_attr:
            return FloatAttribute(val)
        else:
            return AttributeWithRange(val)

    def to_dict(self) -> Dict[str, str]:
        return {**super().to_dict(),
                'polarity': self.polarity,
                'code': self.code,
                'cui': self.cui,
                'tui': self.tui,
                'vocab': self.vocab,
                'preferredText': self.preferredText}

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'polarity': obj_1.polarity or obj_2.polarity,
                           'code': obj_1.code or obj_2.code,
                           'cui': obj_1.cui or obj_2.cui,
                           'tui': obj_1.tui or obj_2.tui,
                           'vocab': obj_1.vocab or obj_2.vocab,
                           'preferredText': obj_1.preferredText or obj_2.preferredText}
        return combined_kwargs


class BiomedOutputWithDate(BiomedOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._date = None
        self.date = kwargs.get('date', None)

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        if isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
            self._date = value.isoformat()
        elif value is not None:
            self._date = value

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {**super().to_dict(),
                'date': self.date}

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'date': obj_1.date or obj_2.date}
        return combined_kwargs


class SummaryOutputWithDate(SummaryOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._date = None
        self.date = kwargs.get('date', None)

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value):
        if isinstance(value, datetime.date) or isinstance(value, datetime.datetime):
            self._date = value.isoformat()
        elif value is not None:
            self._date = value

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {**super().to_dict(),
                'date': self.date}

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'date': obj_1.date or obj_2.date}
        return combined_kwargs


class LabOutput(SummaryOutputWithDate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labValue: FloatAttribute = self.ingest_attribute(
            kwargs.get('labValue', self.attributes.get('labValue')), float_attr=True)
        self.labUnit: AttributeWithRange = self.ingest_attribute(
            kwargs.get('labUnit', self.attributes.get('labUnit')))
        self.labInterp: AttributeWithRange = self.ingest_attribute(
            kwargs.get('labInterp', self.attributes.get('labInterp')))

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {**super().to_dict(),
                'labValue': self.labValue.to_output_list(),
                'labUnit': self.labUnit.to_output_list(),
                'labInterp': self.labInterp.to_output_list()}

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'labValue': obj_1.labValue or obj_2.labValue,
                           'labUnit': obj_1.labUnit or obj_2.labUnit,
                           'labInterp': obj_1.labInterp or obj_2.labInterp}
        return combined_kwargs



class CovidLabOutput(LabOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labManufacturer = kwargs.get('labManufacturer', [])

    def is_lab_name(self) -> bool:
        return self.label == CovidLabLabel.lab.value.persistent_label

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {**super().to_dict(),
                'labManufacturer': self.labManufacturer}

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'labManufacturer': obj_1.labManufacturer or obj_2.labManufacturer}
        return combined_kwargs


class MedOutput(SummaryOutputWithDate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.medFrequencyUnit: AttributeWithRange = self.ingest_attribute(
            kwargs.get('medFrequencyUnit', self.attributes.get('medFrequencyUnit')))
        self.medFrequencyNumber: FloatAttribute = self.ingest_attribute(
            kwargs.get('medFrequencyNumber', self.attributes.get('medFrequencyNumber')), float_attr=True)
        self.medStrengthNum: FloatAttribute = self.ingest_attribute(
            kwargs.get('medStrengthNum', self.attributes.get('medStrengthNum')), float_attr=True)
        self.medStrengthUnit: AttributeWithRange = self.ingest_attribute(
            kwargs.get('medStrengthUnit', self.attributes.get('medStrengthUnit')))
        self.parse_pref_text()

    def parse_pref_text(self):
        if not self.preferredText:
            return
        for token in self.preferredText.split():
            if not self.medStrengthNum:
                if is_digit_punctuation(token):
                    self.medStrengthNum = self.ingest_attribute([token, -1, -1], float_attr=True)
            if not self.medStrengthUnit:
                if probable_med_unit(token):
                    self.medStrengthUnit = self.ingest_attribute([token, -1, -1])

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {**super().to_dict(),
                'medFrequencyNumber': self.medFrequencyNumber.to_output_list(),
                'medFrequencyUnit': self.medFrequencyUnit.to_output_list(),
                'medStrengthNum': self.medStrengthNum.to_output_list(),
                'medStrengthUnit': self.medStrengthUnit.to_output_list()}

    def to_allergy(self) -> SummaryOutput:
        return SummaryOutput(attributes=self.attributes,
                             umlsConcept=self.umlsConcept,
                             text=self.text,
                             lstm_prob=self.lstm_prob,
                             label=AllergyLabel.allergy.value.persistent_label,
                             range=self.range)

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           'medFrequencyNumber': obj_1.medFrequencyNumber or obj_2.medFrequencyNumber,
                           'medFrequencyUnit': obj_1.medFrequencyUnit or obj_2.medFrequencyUnit,
                           'medStrengthNum': obj_1.medStrengthNum or obj_2.medStrengthNum,
                           'medStrengthUnit': obj_1.medStrengthUnit or obj_2.medStrengthUnit
                           }
        return combined_kwargs


class VitalSignOutput(BiomedOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        split_vital_sign = self.split_vital_sign(self.text)
        self.value = kwargs.get('value', split_vital_sign[0])
        self.unit = kwargs.get('unit', split_vital_sign[1])

    @classmethod
    def split_vital_sign(cls, token: str):
        unit, value = None, None
        if is_digit(token):
            value = float(token)
        elif not has_numbers(token):
            unit = token
        else:
            value, unit = cls.extract_first_number(token)
        return unit, value

    @staticmethod
    def extract_first_number(token: str):
        number_str = ""
        for i in range(len(token)):
            if is_digit(token[i]):
                number_str += token[i]
            elif token[i] == '.' and is_digit(token[i + 1]) and (i == 0 or is_digit(token[i - 1])):
                number_str += token[i]
            elif len(number_str) > 0:
                break

        unit = token.replace(number_str, '')
        if len(unit) == 0:
            unit = None
        return float(number_str), unit

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        combined_kwargs = {**super().create_combined_kwargs(obj_1, obj_2, text),
                           "value": obj_1.value or obj_2.value,
                           "unit": obj_1.unit or obj_2.unit}
        return combined_kwargs


class CancerOutput(SummaryOutput):
    @staticmethod
    def nearby_cancer_grade(res_1: SummaryOutput, res_2: SummaryOutput):
        # returns True if two terms next to each other are both cancer grades and are the same or one is unknown)
        if res_2['range'][0] - res_1['range'][1] < NEARBY_TERM_THRESHOLD:
            return res_1.label == CancerLabel.grade.value.persistent_label == res_2.label and (
                    res_1.code == res_2.code or res_1.code == '9' or res_2.code == '9')

    @classmethod
    def create_combined_kwargs(cls, obj_1, obj_2, text):
        if cls.nearby_cancer_grade(res_1=obj_1, res_2=obj_2) and obj_1.code == '9' and obj_2.code != '9':
            combined_kwargs = {**BiomedOutput.create_combined_kwargs(obj_1, obj_2, text),
                               'polarity': obj_2.polarity or obj_1.polarity,
                               'code': obj_2.code or obj_1.code,
                               'cui': obj_2.cui or obj_1.cui,
                               'tui': obj_2.tui or obj_1.tui,
                               'vocab': obj_2.vocab or obj_1.vocab,
                               'preferredText': obj_2.preferredText or obj_1.preferredText}
        else:
            combined_kwargs = super().create_combined_kwargs(obj_1, obj_2, text)
        return combined_kwargs


class CancerStageOutput(SummaryOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.T = self.__get_stage_repr(kwargs.get('T'))
        self.N = self.__get_stage_repr(kwargs.get('N'))
        self.M = self.__get_stage_repr(kwargs.get('M'))
        self.clinical = self.__get_stage_repr(kwargs.get('clinical'))

    def to_dict(self) -> Dict[str, Union[list, str]]:
        return {
            **super().to_dict(),
            'T': self.T,
            'N': self.N,
            'M': self.M,
            'clinical': self.clinical
        }

    @staticmethod
    def __get_stage_repr(value):
        return '' if not value else value.name


class GeneticsOutput(BiomedOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.preferredText = None
        self.polarity = None
        if self.label == GeneticsLabel.gene.value.persistent_label:
            self.__strip_polarity()

    def __strip_polarity(self):
        polarities = {'-', '+'}

        for polarity in polarities:
            if self.text.endswith(polarity):
                self.polarity = polarity
                self.preferredText = self.text[:-1]
                break
        else:
            self.preferredText = self.text

    def to_dict(self) -> Dict[str, str]:
        return {**super().to_dict(),
                'polarity': self.polarity if self.polarity else '',
                'preferredText': self.preferredText if self.preferredText else ''}


class DateOfServiceOutput(BiomedOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.normalized_date = kwargs.get('normalized_date')

    def to_dict(self):
        d = super().to_dict()

        d['preferredText'] = self.normalized_date

        return d
