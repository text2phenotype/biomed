import os
import pandas as pd

from text2phenotype.common import common

label_set_fp = '/Users/shannon.fee/Downloads/36k Optima Charts for NLP Pilot_COD-10983.csv'
label_set_df = pd.read_csv(label_set_fp)

job_ids = ['70548e56cc314489a966f99cf212b25b', 'e5fef2dcc5bd46309c3e21a4c909dd05']
job_path = '/Users/shannon.fee/Documents/hcc_output/ICD_10_CLINICAL_CODE'

original_path = '/Users/shannon.fee/Documents/hcc_output'

accuracy_dict = {}
new_dict = {}
for job_id in job_ids:
    icd10_ressponse = common.read_json(os.path.join(job_path, f'{job_id}.json'))
    for doc in icd10_ressponse:
        original_metadata_fp= os.path.join(original_path,  job_id, 'processed', 'documents', doc,  f'{doc}.metadata.json')
        if  os.path.isfile(original_metadata_fp):
            original_fp = os.path.split(common.read_json(original_metadata_fp)['document_info']['source'])[1].replace('.pdf', '').split('-')[0]
            new_dict[original_fp] = icd10_ressponse[doc]
        else:
            continue

        predicted_labels = {a.replace('.', '') for a in icd10_ressponse[doc]}
        actual_labels = set(label_set_df[label_set_df.ChartID==int(original_fp)].ICDCode)

        chapter_predicted =  {code[0:3] for code in predicted_labels}
        chapter_actual = {a_code[0:3] for a_code in actual_labels}

        accuracy_dict[original_fp] = {
            'full_true_positive': predicted_labels.intersection(actual_labels),
            'full_false_positive': predicted_labels.difference(actual_labels),
            'full_false_negative': actual_labels.difference(predicted_labels),
            'chapter_true_positive': chapter_predicted.intersection(chapter_actual),
            'chapter_false_positive': chapter_predicted.difference(chapter_actual),
            'chapter_false_negative': chapter_actual.difference(chapter_predicted)

        }

print(len(accuracy_dict))







