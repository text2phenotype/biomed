"""
Given a TagTog project:
- download the content
- convert to .ann and .txt
- create normalized_ann

"""
import os
from typing import List, Tuple

import pandas as pd
import numpy as np

from text2phenotype.common import common
from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY
from text2phenotype.constants.features.label_types import LabLabel, deserialize_label_type
from text2phenotype.tagtog.tag_tog_async_data_source import TagTogAsyncDataSource
from text2phenotype.tagtog.tagtog_html_to_text import TagTogText
from text2phenotype.annotations.file_helpers import AnnotationSet

# =====
# this is where all of the various data files live;
# has same name as tagtog project, so tt project will also be at this level
PROJECT_ROOT = "/Users/michaelpesavento/data/ocr_compare/ocr_compare_tt_pdf"

PROJECT_NAME = "ocr_compare"
# =====

NORM_TEXT_FIELD = "normalized_text"

ANN_SUBDIR = "combined/labs"
MASTER_SUBDIR = "master/labs_raw_text"

BIOMED_API_OUTPUT = os.path.join(PROJECT_ROOT, "output/biomedAPI/processed/documents")

PDF_SOURCE_PATH = "/Users/michaelpesavento/data/ocr_compare/ocr_compare_dataset/pdfs"

LAB_SHORT_LABEL_MAP = {
    "lab": LabLabel.lab.name,
    "value": LabLabel.lab_value.name,
    "unit": LabLabel.lab_unit.name,
    "interp": LabLabel.lab_interp.name,
    "date": "lab_date",  # new add, should just be date (not time)
}


def load_tagtog_data_source(project_name, parent_dir=PROJECT_ROOT, label_type_str=None, norm_text_field=None):
    """
    Load a TagTogAsyncDataSource object
    :param project_name: name of project
    :param parent_dir: location of the project, must have project name folder in it
    :param label_type_str:
    :param norm_text_field: name of the TT entity field that contains the normalized (visually corrected) text
    :return:
    """
    label_type = deserialize_label_type(label_type_str)
    tt_annot = TagTogAsyncDataSource(
        project=project_name,
        parent_dir=parent_dir,
        include_master_annotations=True,
        require_complete=True,
        label_type=label_type,
        norm_text_field=norm_text_field
    )
    return tt_annot


def remove_labdate_ocr_anns(ann_filelist, combined_ann_dir):
    ann_repair_dir_out = combined_ann_dir + "_nodate"
    os.makedirs(ann_repair_dir_out, exist_ok=True)
    for i, ann_file in enumerate(ann_filelist):
        ann_set = AnnotationSet.from_file_content(common.read_text(ann_file))
        print(f"Clearing '{ann_file}' ({i + 1}/{len(ann_filelist)})")

        # remove the "na" annotations, which correspond to "lab_date"
        delete_keys = [key for key, entry in ann_set.directory.items() if entry.label == "na"]
        for key in delete_keys:
            del ann_set.directory[key]

        # remove any duplicate entries (matching label/text/text_range)
        ann_set.remove_duplicate_entries()

        # write ann file out
        ann_path_out = os.path.join(ann_repair_dir_out, os.path.basename(ann_file))
        common.write_text(ann_set.to_file_content(), ann_path_out)
        print(f"Wrote {ann_path_out}")


def repair_ocr_ann_date(ann_filelist, combined_ann_dir):
    ann_repair_dir_out = combined_ann_dir + "_date"
    os.makedirs(ann_repair_dir_out, exist_ok=True)
    for i, ann_file in enumerate(ann_filelist):
        ann_set = AnnotationSet.from_file_content(common.read_text(ann_file))
        print(f"Repairing '{ann_file}' ({i + 1}/{len(ann_filelist)})")

        # convert the "na" annotations in the ann set to lab_date"
        for entry in ann_set.directory.values():
            if entry.label == "na":
                entry.label = "lab_date"

        # remove any duplicate entries (matching label/text/text_range)
        ann_set.remove_duplicate_entries()

        # write ann file out
        ann_path_out = os.path.join(ann_repair_dir_out, os.path.basename(ann_file))
        common.write_text(ann_set.to_file_content(), ann_path_out)
        print(f"Wrote {ann_path_out}")


def main():
    """main"""

    # project_root = "/Users/michaelpesavento/data/ocr_compare/ocr_compare_tt_rawtext/textract"
    project_root = PROJECT_ROOT
    from_pdf = True  # use 'combined/' if from pdf, use 'master/' and different output path if not

    # initial convert from tt to ann
    tt_annot = load_tagtog_data_source(PROJECT_NAME, project_root, "LabLabel")
    tt_annot.write_raw_materials_for_annotated_materials(create_combined=from_pdf)

    source_filenames = common.get_file_list(PDF_SOURCE_PATH, ".pdf")
    source_filenames = [os.path.basename(name).replace("-", "-") for name in source_filenames]

    if from_pdf:
        base_ann_dir = os.path.join(tt_annot.annotation_parent_dir, ANN_SUBDIR)
    else:
        subproject = os.path.basename(project_root)
        base_ann_dir = os.path.join(tt_annot.annotation_parent_dir, MASTER_SUBDIR, subproject)
    ann_filelist = common.get_file_list(base_ann_dir, ".ann")
    repair_ocr_ann_date(ann_filelist, base_ann_dir)
    remove_labdate_ocr_anns(ann_filelist, base_ann_dir)

    # --------
    # Extract normed text
    tt_annot_norm = load_tagtog_data_source(PROJECT_NAME, project_root, "LabLabel", norm_text_field=NORM_TEXT_FIELD)
    tt_annot_norm.write_raw_materials_for_annotated_materials(create_combined=from_pdf)
    if from_pdf:
        base_ann_dir = os.path.join(tt_annot_norm.annotation_parent_dir, ANN_SUBDIR)
    else:
        subproject = os.path.basename(project_root)
        base_ann_dir = os.path.join(tt_annot_norm.annotation_parent_dir, MASTER_SUBDIR, subproject)
    ann_filelist = common.get_file_list(base_ann_dir, ".ann")
    repair_ocr_ann_date(ann_filelist, base_ann_dir)
    remove_labdate_ocr_anns(ann_filelist, base_ann_dir)

    # find any missing files in the .anns
    ann_filenames = [os.path.basename(name) for name in ann_filelist]
    files_not_in_ann = set(source_filenames) - set([name.replace(".ann", "") for name in ann_filenames])
    print(f"No annotations for files: {files_not_in_ann}")


if __name__ == "__main__":
    main()
