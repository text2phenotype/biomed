import os

from typing import List
import pandas as pd

from text2phenotype.common import common
from biomed.common.biomed_ouput import BiomedOutput, MedOutput
from biomed.tests.experiment.post_processing.evidation_meds_table import med_code_to_synonym_mapping
from text2phenotype.common.log import operations_logger


def get_coded_med_output(entry: BiomedOutput):
    out = None
    for code in med_code_to_synonym_mapping:
        for expected_text in med_code_to_synonym_mapping[code]:
            if expected_text in entry.preferredText.lower():
                print(f'{expected_text}, pref_text {entry.preferredText.lower()}')
                out = (code, entry)
                break
    if not out:
        out = ('43', entry)
        operations_logger.info(f'no coded med found using other for {entry.preferredText}')
    return out


def process_medication_list(patient_id, meds_list: List[MedOutput]):
    out_rows = []
    for entry in meds_list:
        coded_entry = get_coded_med_output(entry)
        if coded_entry == '43' and not entry.preferredText:
            continue
        out_rows.append(create_nlp_med_table(patient_id=patient_id, med_code=coded_entry[0], med_entry=coded_entry[1]))
    return out_rows


def code_med_unit(med_unit: list):
    if med_unit:
        med_unit_text = med_unit[0].lower().strip().replace(' ', '')
        if med_unit_text == 'mg':
            code = 1
        elif med_unit_text == 'ml':
            code = 2
        elif med_unit_text == 'mcg':
            code = 3
        elif med_unit_text == 'units':
            code = 4
        elif med_unit_text == 'mg/kg':
            code = 5
        elif med_unit_text == 'mgc/kg/min':
            code = 6
        elif med_unit_text == 'units/kg':
            code = 7
        else:
            print(med_unit)
            code = 99
    else:
        code = 99
    return code


def create_nlp_med_table(patient_id, med_code, med_entry: MedOutput):

    output_dict = {
        'PatientID': patient_id,
        'med_start_date': med_entry.date,
        'med_end_date': med_entry.date,
        'med_name': med_code,
        'med_name_other_text': med_entry.preferredText if med_code == '43' else '',
        'med_dose': med_entry.medStrengthNum.text or 999,
        'med_dose_unit': code_med_unit(med_entry.medStrengthUnit.to_output_list()),
        'med_route': 99,
        'med_frequncy': med_entry.medFrequencyNumber.text or 99
    }
    return output_dict

def get_min_col_value(col):
    return col.dropna().min()
def get_max_col_value(col):
    return col.dropna().max()


def collapse_row(pat_id, med_name,  med_text, grouped_df: pd.DataFrame):
    if sum(grouped_df.med_dose != 999) > 0:
        med_dose = grouped_df.med_dose[grouped_df.med_dose != 999].mean()
    else:
        med_dose = 999

    output_dict = {
        'PatientID': pat_id,
        'med_start_date': get_min_col_value(grouped_df.med_start_date),
        'med_end_date': get_max_col_value(grouped_df.med_end_date),
        'med_name': med_name,
        'med_name_other_text': med_text,
        'med_dose': med_dose,
        'med_dose_unit': get_min_col_value(grouped_df.med_dose),
        'med_route': 99,
    }
    return output_dict

if __name__ == '__main__':
    med_files_dir = '/Users/shannon.fee/Documents/evidation/outbox'
    full_out_list = []
    for metadata_fp in common.get_file_list(med_files_dir, '.metadata.json', True):
        uuid = os.path.split(metadata_fp)[1].split('.')[0]
        original_file_id = os.path.split(common.read_json(metadata_fp).get('document_info').get('source'))[1].split('.')[0]
        med_file = metadata_fp.replace('.metadata.json', '.clinical_summary.json')
        if not os.path.isfile(med_file):
            med_file = metadata_fp.replace('.metadata.json', '.drug.json')
        med_json = common.read_json(med_file)

        drug_response = [MedOutput(**entry) for entry in med_json.get("Medication")]
        full_out_list.extend(process_medication_list(original_file_id, drug_response))

    df = pd.DataFrame(full_out_list)
    collapsed_vals = []
    for idx, group in df.groupby(['PatientID', 'med_name', 'med_name_other_text']):
        collapsed_vals.append(collapse_row(idx[0], idx[1], idx[2], group))
    new_df = pd.DataFrame(collapsed_vals)
    new_df.to_csv('/Users/shannon.fee/Documents/evidation/output/2021-05-20/nlp_output_medications.csv')

    # compare to abstraction
    abstractiondf = pd.read_csv('/Users/shannon.fee/Downloads/tbl_meds.csv')
    accuracy_counts = {'false_pos_count': 0, 'false_neg_count': 0, 'true_pos_count': 0}
    for patient_id in set(abstractiondf.PatientID).intersection(new_df.PatientID):
        abstracted_meds = set(abstractiondf[abstractiondf.PatientID==patient_id].med_name)
        nlp_meds = {int(a) for a in new_df[new_df.PatientID==patient_id].med_name}
        if 43 in nlp_meds:
            nlp_meds.remove(43)
        if 43 in abstracted_meds:
            abstracted_meds.remove(43)
        accuracy_counts['true_pos_count'] += len(abstracted_meds.intersection(nlp_meds))
        accuracy_counts['false_pos_count'] += len(nlp_meds.difference(abstracted_meds))
        accuracy_counts['false_neg_count'] += len(abstracted_meds.difference(nlp_meds))

    print(accuracy_counts)
    print(f'Recall: '
          f'{accuracy_counts["true_pos_count"]/(accuracy_counts["true_pos_count"]+accuracy_counts["false_neg_count"])}'
          f'\nPrecision:'
          f'{accuracy_counts["true_pos_count"]/(accuracy_counts["true_pos_count"]+accuracy_counts["false_pos_count"])}')





