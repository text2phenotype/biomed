from feature_service.common.data_source import DataSource
from text2phenotype.constants.features import LabelEnum, LabLabel
from typing import List


def get_file_count(model_label: LabelEnum, ann_dirs: List[str], orig_raw_dirs: List[str]):
    data_source = DataSource(ann_dirs=ann_dirs, original_raw_text_dirs=orig_raw_dirs)
    ann_files = data_source.get_ann_files()
    file_count = 0
    lab_count = 0
    for ann_file in ann_files:
        parsed_ann = data_source.get_brat_label(ann_file, model_label)
        if parsed_ann:
            file_count += 1
            lab_count += len(parsed_ann)
    return lab_count, file_count



print(get_file_count(LabLabel, ann_dirs=['satjiv.kohli/BIOMED-1000-cancer/20191015'], orig_raw_dirs=['NZCR']))