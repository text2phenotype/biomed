"""
Convert raw and normalizexd text pairs to norm_gold ann and text outputs
"""
import os
from typing import List, Dict, Type

import pandas as pd
import numpy as np

from text2phenotype.common import common
from text2phenotype.constants.features.label_types import LabLabel
from text2phenotype.annotations.file_helpers import AnnotationSet, Annotation

from biomed.tests.experiment.ocr.match_ann_to_ocr_text import (
    get_base_file_name,
    find_matched_file_in_list,
    get_page_break_char,
    get_doc_annotation_cui_frame_norm,
)

# define project constants

PROJECT_ROOT = "/Users/michaelpesavento/data/ocr_compare/ocr_compare_dataset"

SOURCES = ["tagtog", "google_ocr", "tesseract", "textract"]
SOURCE_TEXT_DIR = {
    source: os.path.join(PROJECT_ROOT, "tag_tog_text_raw_text/labs_raw_text", source) for source in SOURCES
}

TARGET_DATE_DIR = "2021-08-20"
ANNOTATIONS_ROOT = os.path.join(
    PROJECT_ROOT, f"tag_tog_annotations_raw_text/{TARGET_DATE_DIR}/master/labs_raw_text"
)
ANN_SUBSET = "_nodate"
SOURCE_ANN_RAW_DIR = {source: os.path.join(ANNOTATIONS_ROOT, f"{source}{ANN_SUBSET}") for source in SOURCES}
SOURCE_ANN_NORM_DIR = {
    source: os.path.join(
        ANNOTATIONS_ROOT.replace(TARGET_DATE_DIR, f"{TARGET_DATE_DIR}_norm"), f"{source}{ANN_SUBSET}"
    )
    for source in SOURCES
}

FEATURES_ROOT_DIR = os.path.join(PROJECT_ROOT, "features/tag_tog_text_raw_text/labs_raw_text")


# output folders:
SOURCE_TEXT_GOLD_DIR = {
    source: os.path.join(PROJECT_ROOT, "tag_tog_text_raw_text/labs_raw_text_gold", source)
    for source in SOURCES
}
SOURCE_ANN_NORM_GOLD_DIR = {
    source: os.path.join(
        ANNOTATIONS_ROOT.replace(TARGET_DATE_DIR, f"{TARGET_DATE_DIR}_norm_gold"), f"{source}{ANN_SUBSET}"
    )
    for source in SOURCES
}


def create_norm_gold_dataset(
    base_name: str,
    ann_cuis_df: pd.DataFrame,
    text_file_path: str,
    ann_norm_gold_dir_out: str,
    text_norm_gold_dir_out: str,
):
    """
    Write .txt and .ann files with the normalized text from the ann_cuis_df

    :param base_name:
    :param ann_cuis_df:
    :param text_file_path:
    :param ann_norm_gold_dir_out:
    :param text_norm_gold_dir_out:
    :return:
    """
    orig_text_str = common.read_text(text_file_path)

    new_text_str = ""
    # working assumption that the annotation set is in range sorted order, just to make things easier
    orig_text_ptr = 0

    ann_set_list = []
    for row in ann_cuis_df.itertuples():
        if row.text_range[0] != row.text_range_ann[0] or row.text_range[1] != row.text_range_ann[1]:
            print(f"Incorrect range match for row: {row}")
        if row.text != row.text_token:
            print(f"bad match in raw ann to text file: {row.text} != {row.text_token}")
        token_start = row.text_range[0]
        token_end = row.text_range[1]

        # append everything from the last ann to the current ann
        new_text_str += orig_text_str[orig_text_ptr:token_start]
        new_token_start = len(new_text_str)

        found_token = orig_text_str[row.text_range[0]:row.text_range[1]]
        new_range = [new_token_start, new_token_start + len(row.text_norm)]
        if row.text_norm.lower() != found_token.lower():
            print(f"Ann: {row.text_norm}, txt: {found_token}")
            new_text_str += row.text_norm
        else:
            new_text_str += found_token

        # do validation to confirm alignment
        new_token = new_text_str[new_range[0]:new_range[1]]
        if new_token != row.text_norm:
            print("bad alignment? ", new_token, row)

        #     # check position of new annotation
        #     if not new_ann_set.has_matching_annotation(ann.label, ann.text_range, ann.text):
        #         # likely a range mismatch
        #         print("**mismatch: ", ann, matched_ann)

        # advance the pointer to the end of the original unnormalized token
        orig_text_ptr = token_end  # get the end from the original text

        if row.Index == ann_cuis_df.shape[0] - 1:
            # at the end concatenate remaining text
            new_text_str += orig_text_str[orig_text_ptr:]

        ann_set_list.append(Annotation(text=new_token, text_range=new_range, label=row.label))
    new_ann_set = AnnotationSet.from_list(ann_set_list)

    new_ann_path = os.path.join(ann_norm_gold_dir_out, base_name + ".ann")
    new_text_path = os.path.join(text_norm_gold_dir_out, base_name + ".txt")

    common.write_text(new_text_str, new_text_path)
    common.write_text(new_ann_set.to_file_content(), new_ann_path)


def main():
    label_enum = LabLabel

    for source in SOURCES:
        ann_norm_gold_dir_out = SOURCE_ANN_NORM_GOLD_DIR[source]
        text_norm_gold_dir_out = SOURCE_TEXT_GOLD_DIR[source]
        os.makedirs(ann_norm_gold_dir_out, exist_ok=True)
        os.makedirs(text_norm_gold_dir_out, exist_ok=True)

        # get the file lists for original annotation text
        features_path = os.path.join(FEATURES_ROOT_DIR, source)
        features_file_list = common.get_file_list(features_path, ".json", recurse=True)

        text_file_list = common.get_file_list(SOURCE_TEXT_DIR[source], ".txt", recurse=True)
        ann_raw_file_list = common.get_file_list(SOURCE_ANN_RAW_DIR[source], ".ann", recurse=True)
        ann_norm_file_list = common.get_file_list(SOURCE_ANN_NORM_DIR[source], ".ann", recurse=True)

        # get the page breaks for the given text files
        file_page_break_chars = {}
        for txt_fn in text_file_list:
            # the ann filenames all had dashes swapped out, so we do the same here
            base_name = get_base_file_name(txt_fn)
            file_page_break_chars[base_name] = get_page_break_char(common.read_text(txt_fn))

        ######
        for file_ix in range(len(text_file_list)):

            text_file_path = text_file_list[file_ix]
            base_name = get_base_file_name(text_file_path)
            feature_file_path = find_matched_file_in_list(base_name, features_file_list)
            ann_raw_file_path = find_matched_file_in_list(base_name, ann_raw_file_list)
            ann_norm_file_path = find_matched_file_in_list(base_name, ann_norm_file_list)

            print("target OCR files")
            print(
                text_file_path,
                "\n",
                ann_raw_file_path,
                "\n",
                ann_norm_file_path,
            )

            page_break_chars = file_page_break_chars[base_name]
            ann_cuis_df = get_doc_annotation_cui_frame_norm(
                ann_raw_file_path,
                ann_norm_file_path,
                feature_file_path,
                page_break_chars=page_break_chars,
                label_enum=label_enum,
            )

            create_norm_gold_dataset(
                base_name, ann_cuis_df, text_file_path, ann_norm_gold_dir_out, text_norm_gold_dir_out
            )

            # end for file
        # end all files
        print(f">>> Finished source: {source}")
    # end for source
    print(">>>>>> FINISHED ALL GOLD REWRITES")


if __name__ == "__main__":
    main()
