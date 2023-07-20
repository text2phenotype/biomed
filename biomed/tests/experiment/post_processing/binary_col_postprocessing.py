from abc import ABC, abstractmethod
from typing import List, Dict, Union
import pandas as pd

from text2phenotype.common.feature_data_parsing import is_digit_punctuation


class PostProcessingColumn(ABC):
    def __init__(self,
                 column_name: str,
                 **kwargs):
        self.column_name = column_name
    @abstractmethod
    def bool_and_evidence_from_patient_resp(self,  full_patient_summary: dict):
        raise NotImplementedError

    @abstractmethod
    def process_patient_biomed_resp(self, full_patient_summary: dict):
        raise NotImplementedError


class BinaryColumn(PostProcessingColumn):
    def __init__(self,
                 column_name: str,
                 category_titles: List[str] = None,
                 synonyms: List[str] = None,
                 exclude_terms: List[str] = None,
                 included_label_types: List[str] = None,
                 term_abbreviations: List[str] = None):
        super().__init__(column_name=column_name)
        self.category_titles = category_titles
        self.included_label_types = included_label_types
        self.exclude_terms = exclude_terms
        self.term_synonyms = synonyms
        self.term_abbrevs = term_abbreviations

    def bool_and_evidence_from_patient_resp(self, full_patient_biomed_resp):
        binary_value = False
        evidence = []
        for cat in self.category_titles:
            for biomed_resp in full_patient_biomed_resp.get(cat, []):

                exclude = False
                if self.included_label_types is not None and biomed_resp.get('label') not in self.included_label_types:
                    continue
                for key in ['text', 'preferredText']:
                    lower_text = biomed_resp[key]
                    if not lower_text:
                        continue
                    lower_text = lower_text.lower()
                    if self.exclude_terms:
                        for term in self.exclude_terms:
                            if all([token.lower() in lower_text for token in term.split()]):
                                exclude = True
                                break
                        if exclude:
                            break
                    for term in self.term_synonyms:
                        if all([token.lower() in lower_text for token in term.split()]):
                            binary_value = True
                            evidence.append(biomed_resp)
                            break
                    if not binary_value and self.term_abbrevs is not None:
                        for term in self.term_abbrevs:
                            if term.lower().strip() == lower_text.strip():
                                binary_value = True
                                evidence.append(biomed_resp)

        if binary_value:
            print(f'For terms {self.term_synonyms} found evidence w/ raw text '
                  f'{[a["text"] for a in evidence]} and pref text {[b["preferredText"] for b in evidence]}')
        return binary_value, evidence

    def process_patient_biomed_resp(self, full_patient_summary: dict):
        binary_resp, _ = self.bool_and_evidence_from_patient_resp(full_patient_summary)
        return binary_resp

class OutputTable(ABC):
    def __init__(self,
                 binary_columns_input: List[dict] = None,
                 patient_id_col_key: str  = 'PatientId',
                 text2phenotype_uuid_col_key: str = 'uuid'):
        # add binary columns
        self.data_cols: List[PostProcessingColumn] = [
            BinaryColumn(
                column_name=entry.get('col_name'),
                category_titles=entry['summary_categories'],
                synonyms=entry['lower_text_synonyms'],
                exclude_terms=entry.get('exclude_terms'),
                term_abbreviations=entry.get('term_abbrevs')

            ) for entry in binary_columns_input]
        self.pat_id_key = patient_id_col_key
        self.text2phenotype_uuid_col_key = text2phenotype_uuid_col_key
        self.biomed_table_output = list()

    @property
    def columns(self):
        return [col.column_name for col in self.data_cols]


    def get_output_table(self):
        df = pd.DataFrame(self.biomed_table_output)
        return df

class SinglePatientRowTable(OutputTable):
    def get_output_table(self):
        df = super().get_output_table().set_index(self.text2phenotype_uuid_col_key, verify_integrity=True)
        return df

    def process_biomed_resp(self, biomed_json: Dict[str, List[dict]], patient_id: str, uuid: str):
        row_output = {col.column_name: col.process_patient_biomed_resp(biomed_json) for col in self.data_cols}
        row_output[self.pat_id_key] = patient_id
        row_output[self.text2phenotype_uuid_col_key] = uuid
        self.biomed_table_output.append(row_output)


class SingleCategoryColMapping(ABC):
    def __init__(self,
                 column_label: str = None,
                 resp_attribute_name: str = None,

                 ):
        self.column_label = column_label
        self.res_attribute_name = resp_attribute_name

    def get_non_null_values(self, evidence_list):
        return [resp.get(self.res_attribute_name) for resp in evidence_list if resp.get(self.res_attribute_name)]

    @abstractmethod
    def get_response(self, evidence_list: List[dict]) -> Union[str, float]:
        pass

class FirstValidFloat(SingleCategoryColMapping):
    def get_response(self, evidence_list: List[dict]) -> Union[str, float]:
        for entry in self.get_non_null_values(evidence_list=evidence_list):
            if is_digit_punctuation(entry):
                return entry



class LongFormatTable(OutputTable):
    def __init__(self,
                 binary_columns_input: List[dict] = None,
                 text2phenotype_uuid_col_key: str = 'uuid',
                 patient_id_col_key: str = 'PatientId',
                 category_col_mapping: dict = None):
        super().__init__(
            binary_columns_input, text2phenotype_uuid_col_key=text2phenotype_uuid_col_key,  patient_id_col_key=patient_id_col_key)
        self.category_col_mapping: category_col_mapping

    def process_biomed_resp(self, biomed_resp: Dict[str, List[dict]], patient_id: str, uuid: str):
        for col in self.data_cols:
            binary_val, evidence = col.bool_and_evidence_from_patient_resp(full_patient_summary=biomed_resp)
            if binary_val:
                row = {k: v['summarization_function'] for k, v in self.category_col_mapping}
                row = {
                    self.pat_id_key: patient_id,
                    k:
                    self.category_col_mapping
                }
