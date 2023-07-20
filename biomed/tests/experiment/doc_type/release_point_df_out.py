import os
import pandas as pd
from text2phenotype.common import common
from text2phenotype.constants.common import VERSION_INFO_KEY

BASE_DIR = '/Users/shannon.fee/Documents/release_point'

clinical_summary_fps = common.get_file_list(BASE_DIR, '.clinical_summary.json', True)
metadata_fps = common.get_file_list(BASE_DIR,'.metadata.json', True )
ORDERED_COLUMNS = ['file', 'text', 'preferredText', 'label', 'score', 'range', 'code', 'cui', 'tui', 'vocab']
EXLUDED_COLS = ['polarity']

def get_document_original_fp(metadata_fp):
    metadata = common.read_json(metadata_fp)
    document_uuid = metadata['document_info']['document_id']
    document_source = os.path.split(metadata['document_info']['source'])[1]
    return document_uuid, document_source

def clinical_summary_out(fps: str, mapping: dict):
    full_out = dict()
    for fp in fps:
        uuid  = (os.path.split(fp)[1]).split('.')[0]
        clinical_summary = common.read_json(fp)

        for key in clinical_summary:
            if key != VERSION_INFO_KEY:
                for i in range(len(clinical_summary[key])):
                    for sub_key, value in clinical_summary[key][i].items():
                        if isinstance(value, list) and len(value) > 2:
                            clinical_summary[key][i][sub_key] = value[0]
                        elif isinstance(value, list) and len(value) ==  0:
                            clinical_summary[key][i][sub_key] = 'unk'
                    clinical_summary[key][i]['file'] = mapping[uuid]
            if key in full_out:
                full_out[key].extend(clinical_summary[key])
            else:
                full_out[key] = clinical_summary[key]
    for key in full_out:
        if key != VERSION_INFO_KEY:
            out_csv_path = os.path.join(BASE_DIR, 'output_csv', f'{key}.csv')
            df = pd.DataFrame(full_out[key])
            df = df[get_ordered_cols(df.columns)]
            df.to_csv(out_csv_path, index=False)

def get_ordered_cols(df_cols):
    out_order = [None] * len(ORDERED_COLUMNS)
    for df_col in df_cols:
        if df_col in EXLUDED_COLS:
            continue
        elif df_col in ORDERED_COLUMNS:
            out_order[ORDERED_COLUMNS.index(df_col)] = df_col
        else:
            out_order.append(df_col)
    return [a for a in out_order if a is  not None]

mapping = [get_document_original_fp(fp) for fp in metadata_fps]
mapping = {a[0]: a[1] for a in mapping}
vital_signs_fps = common.get_file_list(BASE_DIR, '.vital_signs.json', True)
imaging_finding_fps = common.get_file_list(BASE_DIR, '.imaging_finding.json', True)
all_fps = clinical_summary_fps + vital_signs_fps + imaging_finding_fps
clinical_summary_out(all_fps, mapping)

