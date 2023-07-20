import os
import pandas as pd

from biomed.tests.experiment.post_processing.finding_files import get_biomed_json_files
from text2phenotype.common.feature_data_parsing import is_digit_punctuation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features.label_types import SignSymptomLabel, ProblemLabel
from text2phenotype.common import common

from biomed.tests.experiment.post_processing.binary_col_postprocessing import OutputTable, SinglePatientRowTable

binary_columns = [
    {
        'col_name': 'Fever', 'lower_text_synonyms': ['fever', 'febrile'], 'exclude_terms': ['hay fever'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Sore_throat',
        'lower_text_synonyms': ['sore throat', 'itchy throat', 'throat irritation', 'scratchy throat', 'dry throat'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Cough', 'lower_text_synonyms': ['cough', 'coughing'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Headache', 'lower_text_synonyms': ['headache', 'head pain'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Muscle_Pain', 'lower_text_synonyms': ['pain', 'myalgia'],
        'exclude_terms': ['sinus pain', 'chest pain'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Chills', 'lower_text_synonyms': ['chills', 'chill'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Fatigue',
        'lower_text_synonyms': ['fatigue', 'tired', 'weary', 'energy loss', 'loss of energy', 'malaise',
                                'little energy', 'low energy'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Nasal',
        'lower_text_synonyms': [
            'nasal stuffiness', 'nasal congestion', 'blocked nose', 'stuffy nose', 'congested nose', 'runny nose',
            'rhinorrea', 'rhinorrhea', 'nasal discharge', 'snuffles', 'nasal drip', 'nose', 'nasal', 'congestion',
            'phlegm'
        ],
        'exclude_terms': ['bloody nose'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'SOB', 'lower_text_synonyms': ['shortness of breath', 'dyspnea'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Loss',
        'lower_text_synonyms': [
            'loss of taste', 'loss of smell', 'anosmia', 'no sense of smell', 'absent sense of smell', 'absent smell',
            'ageusia', 'anosmia'
        ],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Fibromyalgia', 'lower_text_synonyms': ['fibromyalgia'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'CHF', 'lower_text_synonyms': ['chf', 'congestive heart failure'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Cholesterol',
        'lower_text_synonyms': ['high cholesterol', 'Hypercholesterolaemia', 'elevated cholesterol', 'lipidemia'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'HTN', 'lower_text_synonyms': ['hypertension'],
        'term_abbrevs': ['htn'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'HxMI', 'lower_text_synonyms': ['myocardial infarction'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'HxCVA', 'lower_text_synonyms': ['stroke'],
        'term_abbrevs': ['cva'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'HIV',
        'lower_text_synonyms': ['acquired immunodeficiency syndrome ', 'human immunodeficiency virus'],
        'term_abbrevs': ['hiv', 'aids'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Thyroid', 'lower_text_synonyms': ['hypothyroidism', 'hyperthyroidism', 'overactive thyroid'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'IBD',
        'lower_text_synonyms': [
            'inflammatory bowel disease', "chron'", "chrons", 'ulcerative colitis', 'colitis gravis', 'proctocolitis',
            'enteritis',
            'enterocolitis', 'inflammatory bowel', 'irritable bowel'
        ],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Kidney', 'lower_text_synonyms': ['kidney', 'nephropathy', 'renal'],
        'exclude_terms': ['kidney calculi', 'kidney stone'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Liver', 'lower_text_synonyms': ['liver', 'cirrhosis', 'hepatitis', 'hepatocellular', 'hepatic',
                                                     'hepatorenal'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Lupus', 'lower_text_synonyms': ['lupus', 'luposa'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Migraines', 'lower_text_synonyms': ['migraine'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'MS', 'lower_text_synonyms': ['multiple sclerosis', 'disseminated sclerosis'],
        'term_abbrevs': ['ms'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Neuro', 'lower_text_synonyms': ['parkinsons', 'dementia', 'huntington'],
        'term_abbrevs': ['als'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Arthritis', 'lower_text_synonyms': ['osteoarthritis', 'degenerative arthritis', 'arthritis'],
        'exclude_terms': ['rheumat'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Osteoporosis', 'lower_text_synonyms': ['osteoporosis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Psoriasis', 'lower_text_synonyms': ['psoriasis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Rheumatoid', 'lower_text_synonyms': ['rheumatoid arthritis', 'atrophic arthritis',
                                                          'rheumatic arthritis', 'rheumatic gout'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    },  # {
    #     'col_name': 'Pregnancy', 'lower_text_synonyms': ['pregnan'],
    #     'summary_categories': [
    #         SignSymptomLabel.get_category_label().persistent_label,
    #         ProblemLabel.get_category_label().persistent_label]
    # },
    # complications
    {
        'col_name': 'Pneumonia', 'lower_text_synonyms': ['pneumonia', 'bronchitis', 'respiratory infection'],
        'exclude_terms': ['symptoms'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Lung', 'lower_text_synonyms': ['lung injury'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Asthmaflare', 'lower_text_synonyms': ['asthma'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'COPDflare', 'lower_text_synonyms': ['copd'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Sepsis', 'lower_text_synonyms': ['sepsis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Earinfxn', 'lower_text_synonyms': ['ear infection'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Sinusinfxn', 'lower_text_synonyms': ['sinus infection', 'sinusitis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Encephalitis', 'lower_text_synonyms': ['encepahlitis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Encephalopathy', 'lower_text_synonyms': ['encephalopathy'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'CVAEmbolic', 'lower_text_synonyms': ['cva embolic', 'embolic stroke'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'CVAHemm', 'lower_text_synonyms': ['cva hemmorrhagic', 'hemmorrhagic stroke'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'CVANOS', 'lower_text_synonyms': ['stroke'],
        'term_abbrevs': ['cva'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Myocarditis', 'lower_text_synonyms': ['myocarditis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Pericarditis', 'lower_text_synonyms': ['pericarditis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Hfail', 'lower_text_synonyms': ['heart failure'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'MI', 'lower_text_synonyms': ['myocardial infarction', 'heart attack'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Myositis', 'lower_text_synonyms': ['myositis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }, {
        'col_name': 'Rhabdomyolysis', 'lower_text_synonyms': ['rhabdomyolyis'],
        'summary_categories': [
            SignSymptomLabel.get_category_label().persistent_label,
            ProblemLabel.get_category_label().persistent_label]
    }
]


def get_number_value(entry_text):
    digit_tokens = []
    for token in entry_text.split():
        if is_digit_punctuation(token):
            try:
                digit = float(token.replace(',', '.').replace('(', '').replace(')', ''))
                digit_tokens.append(digit)
            except:
                operations_logger.warning(f'counldnt get float from {token}')

    return digit_tokens

def use_vital_signs_for_fever(df, fever_col_name, output_dir,  subdirs=[]):
    for uuid in df[df[fever_col_name] == False].index:
        for subdir in subdirs:
            vital_signs_fp = get_biomed_json_files(
                base_dir=os.path.join(output_dir, subdir, 'processed', 'documents'), uuid=uuid, biomed_extensions=['vital_signs'])
            if vital_signs_fp:
                break
        if not vital_signs_fp:
            continue
        vital_signs = common.read_json(vital_signs_fp[0])
        fever_bool = False
        for entry in vital_signs['VitalSigns']:
            if entry['label'] == 'temperature':
                temp_list = get_number_value(entry['text'])
                for temp in temp_list:
                    # temperature filters based on clinical definition and fact that no body can live with a
                    # fever over 110, anything higher is a ocr error or typo
                    if (temp >= 99 and temp <= 110) or (temp >= 37.8 and temp <= 46):
                        fever_bool = True
                        break
                if fever_bool:
                    break
        if fever_bool:
            df.loc[uuid, fever_col_name] = fever_bool
    return df

def create_accuracy_measurement_dict(nlp_output_df, human_abstraction_df, cols_to_measure_on):
    patient_ids_intersection = set(nlp_output_df.PatientId).intersection(set(human_abstraction_df.PatientID))

    acc_measure_dict = {
        col: {
            'true_pos': [],
            'nlp_only_pos': [],
            'abst_only_pos': [],
            'true_neg': []
        } for col in cols_to_measure_on}

    for patient_id in patient_ids_intersection:
        nlp_row = nlp_output_df[nlp_output_df.PatientId == patient_id]
        abstraction_row = human_abstraction_df[human_abstraction_df.PatientID == patient_id]

        for col in cols_to_measure_on:
            nlp_pred = bool(nlp_row[col].iloc[0])
            abst_pred = bool(abstraction_row[col].iloc[0])
            if nlp_pred:
                if nlp_pred == abst_pred:
                    acc_measure_dict[col]['true_pos'].append(patient_id)
                else:
                    acc_measure_dict[col]['nlp_only_pos'].append(patient_id)
            else:
                if nlp_pred == abst_pred:
                    acc_measure_dict[col]['true_neg'].append(patient_id)
                else:
                    acc_measure_dict[col]['abst_only_pos'].append(patient_id)
    for col in acc_measure_dict:
        true_pos = len(acc_measure_dict[col]['true_pos'])
        true_neg = len(acc_measure_dict[col]['true_neg'])
        false_pos=len(acc_measure_dict[col]['nlp_only_pos'])
        false_neg=len(acc_measure_dict[col]['abst_only_pos'])
        acc_measure_dict[col]['true_pos_count'] = true_pos
        acc_measure_dict[col]['true_neg_count'] = true_neg
        acc_measure_dict[col]['nlp_false_pos'] = false_pos
        acc_measure_dict[col]['nlp_false_neg'] = false_neg
        acc_measure_dict[col]['accuracy'] = (true_pos+true_neg)/(true_pos+true_neg+false_neg+false_pos)
        if true_pos+ false_neg > 0:
            acc_measure_dict[col]['recall'] = (true_pos)/(true_pos+false_neg)
        if true_pos+false_pos >0:
            acc_measure_dict[col]['precision'] = true_pos/(true_pos+false_pos)
    return acc_measure_dict


### SCRIPT THAT RUNS EVERYTHING #####

def main(**kwargs):
    evidation_binary_table = SinglePatientRowTable(binary_columns)
    path_to_output_folders = kwargs.get('local_outbox_dir')
    # fill in the binary table with disease_signsymptom nlp results
    for metadata_fp in common.get_file_list(path_to_output_folders, '.metadata.json', True):
        uuid = os.path.split(metadata_fp)[1].split('.')[0]
        original_file_id = os.path.split(common.read_json(metadata_fp).get('document_info').get('source'))[1].split('.')[0]
        # get file with diseasse sign info:

        disease_sign_fp = metadata_fp.replace('.metadata.json', '.disease_sign.json')
        if not os.path.isfile(disease_sign_fp):
            disease_sign_fp = metadata_fp.replace('.metadata.json', '.clinical_summary.json')
        disease_sign_json = common.read_json(disease_sign_fp)
        evidation_binary_table.process_biomed_resp(biomed_json=disease_sign_json, patient_id=original_file_id, uuid=uuid)

    nlp_df = evidation_binary_table.get_output_table()

    # adding vital signs to output
    nlp_df = use_vital_signs_for_fever(
        nlp_df, 'Fever', path_to_output_folders,
                                       ['2021_05_19', '2021-04-27', 'batch_1', 'batch_1/shannon_demo_day'])


    abstraction_csv_fp = kwargs.get('main_table_fp')
    abstraction_df = pd.read_csv(abstraction_csv_fp)

    acc_measure_dict = create_accuracy_measurement_dict(
        nlp_output_df=nlp_df,
        human_abstraction_df=abstraction_df,
        cols_to_measure_on=evidation_binary_table.columns)

    output_path = kwargs.get('output_csv_path')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(acc_measure_dict).to_csv(output_path)

if __name__ == '__main__':
    main(
        output_csv_path='/Users/shannon.fee/Documents/evidation/output/2021-05-20/diagnosis_signsymptom_accuracy.csv',
        local_outbox_dir='/Users/shannon.fee/Documents/evidation/outbox',
        main_table_fp='/Users/shannon.fee/Downloads/tbl_main (2).csv'
    )