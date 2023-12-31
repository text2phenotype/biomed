import string
from typing import List

import pandas as pd
import numpy as np

from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.demographics import Demographics
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.constants.features import DemographicEncounterLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.common.biomed_ouput import BiomedOutput
from biomed.constants.constants import BiomedVersionInfo
from biomed.demographic.best_demographics import FhirDemographicEnforcement


class FetchedDemographics:
    def __init__(self,
                 demographics: Demographics = None,
                 pat_names: List[BiomedOutput] = None,
                 dr_names: List[BiomedOutput] = None,
                 demographics_list: List[BiomedOutput] = None,
                 tid: str = None):
        self.demographics = demographics
        self.pat_names: List[BiomedOutput] = pat_names
        self.dr_names: List[BiomedOutput] = dr_names
        if not self.dr_names or not self.pat_names:
            if demographics_list is not None:
                self.initialize_from_demographic_tokens(demographics_list, tid=tid)

    def ensure_list_biomed_output(self, value: list):
        for i in range(len(value)):
            if isinstance(value[i], dict):
                biomed_output = BiomedOutput(**value[i])
                value[i] = biomed_output
        return value

    def to_dict(self, biomed_version: str = None):
        demographics_dict = self.demographics.to_dict() if isinstance(self.demographics, Demographics) else self.demographics
        if self.pat_names is not None:
            pat_names = [name.to_dict() for name in self.pat_names]
        else:
            pat_names = self.pat_names
        biomed_version = BiomedVersionInfo(TaskOperation.demographics, biomed_version=biomed_version).to_dict()
        return {'demographics': demographics_dict,
                'pat_names': pat_names,
                VERSION_INFO_KEY: biomed_version}

    @staticmethod
    def get_demographic_tokens_from_prediction(weighted) -> List[BiomedOutput]:
        """
        this function takes a clinical text and return the demographic information
        :return:
        """
        demographic_tokens = list()
        for token in weighted:
            _class = list(token.keys())[0]
            _score, _text, _range = token[_class]
            if _class != DemographicEncounterLabel.na.value.column_index:
                if _text in string.punctuation:
                    pass
                else:
                    demographic_tokens.append(BiomedOutput(text=_text,
                                                           label=DemographicEncounterLabel.get_from_int(_class).name,
                                                           lstm_prob=_score,
                                                           range=_range))
        return demographic_tokens

    @staticmethod
    def transform_list_to_demographics(suggested_demographics: List[BiomedOutput]) -> Demographics:
        demographics = Demographics()
        for item in suggested_demographics:
            demo_type = item.label
            if demo_type in [DemographicEncounterLabel.pat_last, DemographicEncounterLabel.pat_first,
                             DemographicEncounterLabel.pat_middle] and '/' in item.text:
                continue
            elif demo_type in ['dr_name', 'dr_first_middle']:
                demographics.dr_full_names.append((item.text, item.lstm_prob))
            elif demo_type in ['pat_name', 'pat_first_middle']:
                demographics.pat_full_name.append((item.text, item.lstm_prob))
            elif isinstance(demo_type, str):
                demographics.add_entry(demo_type, item.text, item.lstm_prob)
            elif isinstance(demo_type, DemographicEncounterLabel):
                demographics.add_entry(demo_type.name, item.text, item.lstm_prob)
        return demographics

    @text2phenotype_capture_span()
    def initialize_from_demographic_tokens(self, demographics: List[BiomedOutput], tid: str = None):
        demographics = self.ensure_list_biomed_output(demographics)
        suggestions_collapsed = self.collapse_demographics_by_range(demographics)
        names = self.collapse_names(suggestions_collapsed)
        self.get_patient_name(names=names)
        self.get_dr_names(names=names)
        if self.dr_names:
            suggestions_collapsed.extend(self.dr_names)
        if self.pat_names:
            suggestions_collapsed.extend(self.pat_names)
        self.demographics = self.transform_list_to_demographics(suggestions_collapsed)

    # takes all predicted demographics and looks to combine names of the form first middle last, first last,
    # last first middle or last first
    # outputs a list of lists of the form [[full combined name, avg probability of all components]
    @classmethod
    def collapse_names(cls, demographics: List[BiomedOutput]) -> List[BiomedOutput]:
        names = []
        if not demographics:
            return []
        len_dem = len(demographics)
        if len_dem >= 2:
            for i in range(0, len_dem):
                entry = demographics[i]
                # starting with a first name
                if entry.label in [DemographicEncounterLabel.pat_first.name, DemographicEncounterLabel.dr_first.name]:
                    pat_first = entry
                    pat_middle = None
                    pat_last = None

                    # last name next term after  pat_first
                    if (i < len_dem - 1 and
                            demographics[i + 1].label == DemographicEncounterLabel.pat_last.value.persistent_label):
                        pat_last = demographics[i + 1]

                    # last name term before the pat_first
                    if i > 0 and demographics[i - 1].label == DemographicEncounterLabel.pat_last.value.persistent_label:
                        pat_last = demographics[i - 1]

                    # middle name next term after the pat_first
                    if i < len_dem -1 and demographics[i+1].label == DemographicEncounterLabel.pat_middle.value.persistent_label:
                        pat_middle = demographics[i+1]

                    # middle then last
                    if (pat_middle is not None and
                            i < len_dem - 2 and
                            demographics[i+2].label == DemographicEncounterLabel.pat_last.value.persistent_label):
                        pat_last = demographics[i+2]

                    collapsed_name = cls.collapse_helper(first_name=pat_first, middle_name=pat_middle, last_name=pat_last)
                    if collapsed_name.text is not None and collapsed_name.range[0] is not None and collapsed_name.lstm_prob>0:
                        names.append(collapsed_name)

        return names

    # takes labeled names with type guesses and outputs all names that are more likely to be a patient
    # name than a doctor name, Uses the fact that if a full name is found and predicted ot belong ot the patient
    # then we should only consider that combination capable of being the patients name
    # Additionally relies on assumption that while dr x patient name occurs if a name occurs multiple times nad has the
    # highest prob belonging to a dr then it is more likely a dr name
    def get_person_name(self, names: List[BiomedOutput] = None, demographics: List[BiomedOutput] = None,
                        person_type: str = 'pat', regex_vl: bool = False) -> List[BiomedOutput]:
        if not names:
            names = self.collapse_names(demographics)
        if names:
            df_a = pd.DataFrame([name.to_dict() for name in names])
            # only compare dr predictions vs patient ones (ignore the dr last, pat_first predictions for now
            max_prob_per_text_pure = df_a.iloc[list(df_a[((df_a['label'].str.contains('dr'))
                                                          | (df_a['label'].str.contains('pat')))].groupby(
                ['text']).score.idxmax())]
            max_prob_per_text_pure = pd.concat([max_prob_per_text_pure,
                                                df_a[~(df_a.label.str.contains('dr')) &
                                                     ~(df_a.label.str.contains('pat')) &
                                                     df_a.text.apply(
                                                         lambda x: x not in list(max_prob_per_text_pure.text))]])

            # gets table of name x max_probability_category
            predicted_name_rows = max_prob_per_text_pure[
                max_prob_per_text_pure.label.str.contains(person_type, regex=regex_vl)].reset_index()
            predicted_names = []

            for i in range(predicted_name_rows.shape[0]):
                predicted_name = predicted_name_rows.iloc[i]
                predicted_names.append(BiomedOutput(text=predicted_name.text,
                                                    lstm_prob=predicted_name.score,
                                                    label=predicted_name.label,
                                                    range=predicted_name.range))

            return predicted_names

    # if there is a full patient name return that, else return all names where part of them was expected
    # to be patient name
    def get_patient_name(self, names: List[BiomedOutput] = None, demographics: List[BiomedOutput] = None):
        pat_names = self.get_person_name(names, demographics, person_type='pat')
        if pat_names:
            self.pat_names = pat_names
            return
        else:
            pat_names = self.get_person_name(names, demographics, person_type='unk')
            if not pat_names:
                return
            for i in range(len(pat_names)):
                pat_names[i].label = pat_names[i].label.replace('unk', 'dr')
            self.pat_names = pat_names

    def get_dr_names(self, names: List[BiomedOutput] = None, demographics: List[BiomedOutput] = None):
        dr_names = self.get_person_name(names, demographics, person_type='dr|unk', regex_vl=True)
        if not dr_names:
            return
        for i in range(len(dr_names)):
            dr_names[i].label = dr_names[i].label.replace('unk', 'dr')
        self.dr_names = dr_names

    @staticmethod
    def collapse_demographics_by_range(demographics: List[BiomedOutput]) -> List[BiomedOutput]:
        end = len(demographics) - 1
        output = demographics.copy()
        for i in range(end):
            c_val = output[end - i]
            nxt_val = output[end - i - 1]
            if c_val.label == nxt_val.label and nxt_val.label in COLLAPSIBLE_DEMOGRAPHICS_COLS and \
                    c_val.range[0] - nxt_val.range[1] <= 3:
                combo_val = BiomedOutput(lstm_prob=max(c_val.lstm_prob, nxt_val.lstm_prob),
                                         text=f"{nxt_val.text} {c_val.text}",
                                         label=nxt_val.label,
                                         range=[nxt_val.range[0], c_val.range[1]])
                output[end - i - 1] = combo_val
                output[end - i] = None
        output = [d for d in output if d is not None]
        return output

    # outputs a dictionary with a text, score, range and type to get appended to names
    # generalizes the appending for the get_pat_name entry_q
    @staticmethod
    def collapse_helper(first_name: BiomedOutput, middle_name: BiomedOutput = None,
                        last_name: BiomedOutput = None) -> BiomedOutput:
        options = ['pat', 'dr']
        result = BiomedOutput()
        for opt in options:
            if opt in first_name.label:
                if last_name:
                    if last_name.label == f'{opt}_last':
                        result.label = f'{opt}_name'
                    else:
                        result.label = 'unk_name'
                elif middle_name:
                    if middle_name.label == f'{opt}_middle':
                        result.label = f'{opt}_first_middle'
                    else:
                        result.label = 'unk_first_middle'
        if middle_name and last_name:
            result.lstm_prob = np.mean([first_name.lstm_prob, middle_name.lstm_prob, last_name.lstm_prob])
            result.range = [first_name.range[0], last_name.range[1]]
            result.text = f"{first_name.text} {middle_name.text} {last_name.text}"
        elif last_name:
            result.lstm_prob = np.mean([first_name.lstm_prob, last_name.lstm_prob])
            result.range = [first_name.range[0], last_name.range[1]]
            result.text = f"{first_name.text} {last_name.text}"
        elif middle_name:
            result.lstm_prob = np.mean([first_name.lstm_prob, middle_name.lstm_prob])
            result.range = [first_name.range[0], middle_name.range[1]]
            result.text = f"{first_name.text} {middle_name.text}"
        return result



COLLAPSIBLE_DEMOGRAPHICS_COLS = {DemographicEncounterLabel.dr_city.value.persistent_label,
                                 DemographicEncounterLabel.dr_street.value.persistent_label,
                                 DemographicEncounterLabel.dr_phone.value.persistent_label,
                                 DemographicEncounterLabel.dr_fax.value.persistent_label,
                                 DemographicEncounterLabel.dr_org.value.persistent_label,
                                 DemographicEncounterLabel.facility_name.value.persistent_label,
                                 DemographicEncounterLabel.pat_street.value.persistent_label,
                                 DemographicEncounterLabel.pat_city.value.persistent_label,
                                 DemographicEncounterLabel.pat_phone.value.persistent_label,
                                 DemographicEncounterLabel.insurance.value.persistent_label,
                                 DemographicEncounterLabel.ssn.value.persistent_label,
                                 DemographicEncounterLabel.pat_last.value.persistent_label,
                                 DemographicEncounterLabel.pat_first.value.persistent_label,
                                 DemographicEncounterLabel.pat_middle.value.persistent_label,
                                 DemographicEncounterLabel.dob.value.persistent_label}

FHIR_DEMOGRAPHICS_TYPES = {DemographicEncounterLabel.ssn.name, DemographicEncounterLabel.mrn.name,
                           DemographicEncounterLabel.pat_email.name, DemographicEncounterLabel.sex.name,
                           DemographicEncounterLabel.dob.name, DemographicEncounterLabel.pat_phone.name,
                           DemographicEncounterLabel.pat_state.name, DemographicEncounterLabel.pat_zip.name,
                           DemographicEncounterLabel.pat_city.name, DemographicEncounterLabel.pat_street.name,
                           DemographicEncounterLabel.pat_age.name, DemographicEncounterLabel.pat_last.name,
                           DemographicEncounterLabel.pat_middle.name, DemographicEncounterLabel.pat_first.name}


def get_best_demographics(fetched_demographics: FetchedDemographics) -> Demographics:
    best_demographics = Demographics()
    if not isinstance(fetched_demographics, FetchedDemographics):
        raise TypeError(f"Expected type FetchedDemographics, received {type(fetched_demographics)}")
    pat_names = None
    # add pat names
    if fetched_demographics.pat_names and len(fetched_demographics.pat_names) >= 1:
        pat_names = sorted(fetched_demographics.pat_names, key=lambda name: name.lstm_prob)
        best_guess_pat_name = pat_names[0]
        best_demographics.add_entry('pat_full_name', best_guess_pat_name.text, best_guess_pat_name.lstm_prob)
    # add doctor names
    if fetched_demographics.dr_names:
        best_demographics.add_entry_list('dr_full_names',
                                         [(out.text, out.lstm_prob) for out in fetched_demographics.dr_names])

    # if no ssn predicted, add mrn to potential ssn list and pull out of mrns
    demographics = fetched_demographics.demographics
    if demographics is not None:
        if not demographics.ssn:
            demographics.add_entry_list('ssn', demographics.mrn)

        # loop through all demographic types in Demographics  class
        for dem_type in demographics.to_dict():
            responses = demographics[dem_type.lower()]
            if not responses:
                continue
            if dem_type in FhirDemographicEnforcement.__members__:
                if dem_type == DemographicEncounterLabel.dob.name and best_demographics.pat_age:
                    # if we have an estimated age use this to help limit DOBs
                    values = FhirDemographicEnforcement[dem_type].value(
                        responses, age_guess=int(best_demographics.pat_age[0][0]))
                # use hinting for name detection
                elif dem_type in [DemographicEncounterLabel.pat_last.name, DemographicEncounterLabel.pat_first.name,
                                  DemographicEncounterLabel.pat_middle.name] and pat_names:
                    values = FhirDemographicEnforcement[dem_type].value(responses, pat_name_full=pat_names)

                else:
                    values = FhirDemographicEnforcement[dem_type].value(responses)
                if values:
                    if isinstance(values[0], list) or isinstance(values[0], tuple):
                        for val in values:
                            best_demographics.add_entry(dem_type, val[0], val[1])
                    elif values[0]:
                            best_demographics.add_entry(dem_type, values[0], values[1])
            else:
                best_demographics.add_entry_list(dem_type, responses)

    return best_demographics
