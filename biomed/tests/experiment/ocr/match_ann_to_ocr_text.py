from collections import namedtuple
import datetime
import os
import re
from typing import List, Dict, Type

from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from text2phenotype.common import common

from text2phenotype.constants.features.label_types import LabLabel, LabelEnum
from text2phenotype.constants.features.feature_type import FeatureType
from text2phenotype.annotations.file_helpers import AnnotationSet, Annotation
from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY

from biomed.common.model_test_helpers import cui_set_for_entry, document_cui_set
from biomed.data_sources.data_source import BiomedDataSource
from biomed.constants.model_constants import LabConstants


# define project constants

project_root = "/Users/michaelpesavento/data/ocr_compare/ocr_compare_dataset"

textract_text_dir = os.path.join(project_root, "raw_text/lab/textract")
tesseract_text_dir = os.path.join(project_root, "raw_text/lab/tesseract")
googleocr_text_dir = os.path.join(project_root, "raw_text/lab/google_ocr")
tagtog_text_dir = os.path.join(project_root, "raw_text/lab/tag_tog_text")

ann_raw_dir = os.path.join(project_root, "tag_tog_annotations/labs/2021-08-06/combined/labs_nodate")
ann_norm_dir = os.path.join(project_root, "tag_tog_annotations/labs/2021-08-06_norm/combined/labs_nodate")

ANN_OUT_NAME = "matched_annotations/"

features_root_dir = os.path.join(project_root, "features")
LAB_HEPC = FeatureType.lab_hepc

CONCEPT_FEATURE_MAPPING = LabConstants.token_umls_representation_feat

# make a lighter object than MachineAnnotation for the page token sequence
Token = namedtuple("Token", ["text", "text_range", "cui"])


def sanitize_name(name):
    return name.replace("-", "_")


def get_base_file_name(name):
    """Return the base file name from a full file path"""
    return sanitize_name(os.path.basename(name)).replace(".txt", "").replace(".pdf", "").replace(".ann", "")


def get_page_break_char(raw_text):
    """Given raw text, iterate through to find the page break characters"""
    page_breaks = [0] + [
        x.start() + 1 for x in re.finditer(OCR_PAGE_SPLITTING_KEY[0], raw_text)
    ]
    if page_breaks[-1] != len(raw_text):
        page_breaks += [len(raw_text)]
    return page_breaks


def get_ann_page(ann, page_break_chars):
    """
    Find the first index that is less than end of token
    NOTE: returned pages are zero indexed!

    :param ann:
    :param page_break_chars: the list of characters each page starts at
    """
    return get_token_page(ann.text_range, page_break_chars)


def get_token_page(text_range, page_break_chars):
    """
    Find the first page_break_char that is less than the end of the token
    """
    assert page_break_chars[0] == 0, f"Expected the first page to start at 0, found {page_break_chars[0]}"

    # text_range[0] >= page_break_chars[page] and text_range[1] < page_break_chars[page+1]
    page = np.argmax(
        text_range[1] <= np.array(page_break_chars)
    )
    # we shift down the index by 1 because page_break_chars should lead with a zero
    if page and page != 0:
        page = page - 1
    else:
        page = 0

    return page


def find_matched_file_in_list(base_name, file_list):
    """Given a base name (with underbars no dashes), find matching name in file list"""
    matches = [
        file_path
        for file_path in file_list
        if sanitize_name(base_name) in sanitize_name(os.path.basename(file_path))
    ]
    if not matches:
        print(f"No match found for {base_name}")
        return ""
    elif len(matches) > 1:
        print(f"found multiple matches for {base_name}: {matches}; returning first")
    return matches[0]


def get_cui_for_token(token_ix: int, machine_ann: MachineAnnotation, concept_feature_mapping=None):
    """Get the target CUI from a token in a machine annotation object"""
    concept_feature_mapping = concept_feature_mapping or LabConstants.token_umls_representation_feat
    cuis = cui_set_for_entry(token_ix, concept_feature_mapping, machine_ann)
    # only use the first cui
    if len(cuis) >= 1:
        cui = sorted(cuis)[0]
    else:
        cui = None
    return cui


def split_brat_annotation_phrases(brat_ann: List[Annotation]) -> List[Annotation]:
    """
    Split brat annotation phrases into distinct tokens, to make matching to MachineAnnotation tokens easier

    :param brat_ann: List[Annotation]
        list of brat annotations, generally from the output of BiomedDataSource.get_brat_label()
    :return: List[Annotation]
        Any phrase annotations are split by white space and returned with adjusted text_ranges
    """
    brat_ann_split = []
    for ann in brat_ann:
        split_text = ann.text.split()
        if len(split_text) > 1:
            # assume single space between tokens?
            ann_start = ann.text_range[0]
            for text in split_text:
                text_start = ann_start + ann.text.find(text)
                brat_ann_split.append(
                    Annotation(text=text, text_range=[text_start, text_start + len(text)], label=ann.label)
                )
        else:
            brat_ann_split.append(ann)
    return brat_ann_split


def find_closest_cui_ann(ann_range_start: int, ann_page: int, ann_cuis_df: pd.DataFrame):
    """
    find closest cui for a given annotation (set via annotation text_range start)
    NOTE: the anchored cui may have nothing to do with context of the target annotation!
        Ie, the annotation may be a value for WBC, but the nearest CUI is for hematocrit
        This doesnt matter, because we just need the nearest
    """
    ann_cui_page_df = ann_cuis_df[ann_cuis_df.page == ann_page]
    # get the text range of any annotation that has a cui
    ann_has_cui = ann_cui_page_df[ann_cui_page_df.cui.notna()]
    # if no cuis to anchor, set distance to None
    if ann_has_cui.empty:
        # print(f"no cuis on page {ann_page}")
        return None, (None, None)

    # get the abs distance from our target annotation
    cui_distances = ann_has_cui.text_range.apply(lambda x: np.abs(x[0] - ann_range_start))
    matched_cui_ann = ann_cui_page_df.loc[cui_distances.idxmin()]
    min_cui_dist = matched_cui_ann.text_range[0] - ann_range_start
    return min_cui_dist, (matched_cui_ann.text, matched_cui_ann.cui)


def calc_similar_text_scores(ann_cui_page_df, token_sequence):
    """
    Return a score 0-1 for how similar a token-annotation pair is
    :param ann_cui_page_df:
    :param token_sequence:
    :return:
    """
    # get text similarity scores
    similar_scores = np.zeros((ann_cui_page_df.shape[0], len(token_sequence)))
    for i, token in enumerate(token_sequence):
        for j, ann in enumerate(ann_cui_page_df.itertuples()):
            similar_scores[j, i] = fuzz.ratio(token.text, ann.text) / 100.0
    # scale all scores to the max score across tokens
    # this reduces peaks when summed with distance scores, avoiding poor location matches
    scaled_similar = similar_scores * np.max(similar_scores, axis=1)[:, None]
    return scaled_similar


def calc_distance_scores(ann_cui_page_df, token_sequence):
    """
    Return a distance score 0-1 for how close an annotation start range and token start range are
    The score is the difference between the start ranges, normalized by the total number of tokens

    :param ann_cui_page_df:
    :param token_sequence:
    :return:
    """
    # create distance matrix; how far each each token from each annotation?
    # start with just using the character distance from the token range start
    distance_scores = np.zeros((ann_cui_page_df.shape[0], len(token_sequence)))
    for i, token in enumerate(token_sequence):
        for j, ann in enumerate(ann_cui_page_df.itertuples()):
            distance_scores[j, i] = 1 - np.abs(token.text_range[0] - ann.text_range[0]) / len(token_sequence)
    distance_scores = np.where(distance_scores < 0, 0, distance_scores)
    return distance_scores


def find_matching_cui_text(token_sequence: List[Token], cui: str, text: str):
    """
    Find the token that has a matching CUI and similar text
    :param token_sequence:
    :param cui:
    :param text:
    :return:
    """
    match = None
    same_cui = [token for token in token_sequence if token.cui == cui]
    if not same_cui:
        # print(f"No matching token for: {text}, {cui}")
        return match
    elif len(same_cui) > 1:
        similar_text = [fuzz.ratio(token.text, text) for token in same_cui]
        match = same_cui[np.argmax(similar_text)]
        # print(f"multiple matching cui: {same_cui}, using {match} for {text}")
    else:
        match = same_cui[0]
    return match


def calc_cui_distance_scores(ann_cui_page_df, token_sequence):
    # get nearest CUI distances
    cui_distance_scores = np.zeros((ann_cui_page_df.shape[0], len(token_sequence)))
    for i, token in enumerate(token_sequence):
        for j, ann_row in enumerate(ann_cui_page_df.itertuples()):
            if not np.isnan(ann_row.nearest_cui_distance) and ann_row.nearest_cui_distance is not None:
                # we need to find the new position of the anchor token
                # it will be in a different place in the new machine_ann than the original annotation
                anchor_token = find_matching_cui_text(token_sequence, cui=ann_row.nearest_cui_token_info[1],
                                                      text=ann_row.nearest_cui_token_info[0])
                if not anchor_token:
                    cui_distance_scores[j, i] = 0
                    continue
                distance_from_anchor = token.text_range[0] - anchor_token.text_range[0]
                if token.cui and not ann_row.cui:
                    # never match a token with a cui to a annotation without one
                    # TODO: unless the token is more correct than the annotation, ie annotation isnt from gold text?
                    cui_distance_scores[j, i] = 0
                else:
                    if np.isnan(ann_row.nearest_cui_distance):
                        import pdb; pdb.set_trace()
                    diff_from_cui_anchor_distance = distance_from_anchor - int(ann_row.nearest_cui_distance)
                    cui_distance_scores[j, i] = 1 - np.abs(diff_from_cui_anchor_distance) / len(token_sequence)
            else:
                # print("no cui anchors found")
                cui_distance_scores[j, i] = 0
    #     print("---")
    cui_distance_scores = np.where(cui_distance_scores < 0, 0, cui_distance_scores)
    return cui_distance_scores


def get_doc_annotation_cui_frame_norm(
        ann_raw_file_path: str,
        ann_norm_file_path: str,
        ma_file_path: str,
        page_break_chars: list,
        label_enum: Type[LabelEnum],
        concept_feature_mapping: dict = CONCEPT_FEATURE_MAPPING):
    """
    Extract a dataframe from the label annotations and machine annotations
    Used in matching the annotations to a given raw text

    :param ann_raw_file_path: absolute path to the raw annotation file
    :param ann_norm_file_path: absolute path to the norm annotation file
    :param ma_file_path: absolute path to the target machine annotation file,
        NOTE: this file should be from the SAME text that the annotations were created from
    :param page_break_chars: list of the character positions for start, page breaks, and end of a document
    :param label_enum: LabelEnum for our target label
    :param concept_feature_mapping: dict of target feature enum with the expected label name
    :return: pd.DataFrame
    """
    machine_ann = MachineAnnotation(json_dict_input=common.read_json(ma_file_path))
    true_labels = BiomedDataSource.token_true_label_list(ann_raw_file_path, machine_ann, label_enum)

    token_cuis = []
    for token_ix in range(len(machine_ann)):
        cui = get_cui_for_token(token_ix, machine_ann)
        text_range = machine_ann.range[token_ix]
        page = get_token_page(text_range, page_break_chars)
        token_cuis.append({
            "token_ix": token_ix,
            "text": machine_ann.tokens[token_ix],
            "range": text_range,
            "page": page,
            "label": true_labels[token_ix],
            "cui": cui,
        })
    token_cui_df = pd.DataFrame.from_records(token_cuis)

    # align the raw and norm ANNs
    brat_ann_raw = BiomedDataSource.get_brat_label(ann_raw_file_path, label_enum)
    brat_ann_norm = BiomedDataSource.get_brat_label(ann_norm_file_path, label_enum)
    brat_ann_raw = sorted(brat_ann_raw, key=lambda ann: ann.text_range[0])
    brat_ann_norm = sorted(brat_ann_norm, key=lambda ann: ann.text_range[0])
    assert len(brat_ann_raw) == len(brat_ann_norm), \
        f"Mismatched ann set lengths: {len(brat_ann_raw)} != {len(brat_ann_norm)}"

    brat_ann_raw_split = split_brat_annotation_phrases(brat_ann_raw)
    brat_ann_norm_split = split_brat_annotation_phrases(brat_ann_norm)
    if len(brat_ann_raw_split) != len(brat_ann_norm_split):
        longer_len = max(len(brat_ann_raw_split), len(brat_ann_norm_split))
        which_longer = np.argmax([len(brat_ann_raw_split), len(brat_ann_norm_split)])
        new_split_raw = []
        new_split_norm = []
        ptr = [0, 0]
        for i in range(longer_len):
            cur_raw = brat_ann_raw_split[ptr[0]]
            cur_norm = brat_ann_norm_split[ptr[1]]
            new_split_raw.append(cur_raw)
            new_split_norm.append(cur_norm)
            if cur_raw.text_range == cur_norm.text_range:
                ptr[0] += 1
                ptr[1] += 1
            else:
                if (cur_raw.text_range[1] - cur_raw.text_range[0]) < (cur_norm.text_range[1] - cur_raw.text_range[0]):
                    ptr[0] += 1
                elif (cur_raw.text_range[1] - cur_raw.text_range[0]) > (cur_norm.text_range[1] - cur_raw.text_range[0]):
                    ptr[1] += 1
                else:
                    ptr[0] += 1
                    ptr[1] += 1
        brat_ann_raw_split = new_split_raw
        brat_ann_norm_split = new_split_norm

    # iterate through both raw and norm to align with token text
    ann_cuis = []
    for i, (ann_raw, ann_norm) in enumerate(zip(brat_ann_raw_split, brat_ann_norm_split)):
        # sometimes the highlighted annotation token is an incomplete selection of the raw text word,
        # eg "om" in "fom" or "2.5*" in "2.5*".
        # We expand the search to check one character wider, which would miss any words that have more than one missing char
        matches = token_cui_df[
            token_cui_df.range.apply(lambda x: x[0] >= ann_raw.text_range[0] - 1 and x[1] <= ann_raw.text_range[1] + 1)
        ]
        page = get_token_page(ann_raw.text_range, page_break_chars)
        if matches.empty:
            # no matching tokens; try to find submatches
            matches = token_cui_df[(
                token_cui_df.range.apply(lambda x: x[0] <= ann_raw.text_range[0] and x[1] >= ann_raw.text_range[1])
                & token_cui_df.text.str.contains(ann_raw.text, case=False, regex=False)
            )]
            if len(matches) > 1 or matches.empty:
                print(i, page, ann_raw, ann_norm)

        cui = None
        if len(set(matches.cui)) == 1 or (len(set(matches.cui)) == 2 and None in set(matches.cui)):
            cui = list(set(matches.cui))[0]
        else:
            print(f"Found different CUIs for highlighted phrase:\n\t{ann_raw}\n\t{matches}")
            continue

        if len(set(matches.label)) > 1:
            # choose the match that has a truth label
            matches = matches[matches.label != 0]

        if len(matches) > 2:
            print(f"*** {i}, {len(matches)}")
            print(matches)
        if len(matches) > 1:
            # at this point just choose the longest match
            char_len = matches.range.apply(lambda x: x[1] - x[0])
            matches = matches[char_len == char_len.max()]
        matches = matches.iloc[0]

        ann_cuis.append({
            "token_ix": matches.token_ix,
            "text": ann_raw.text,
            "text_norm": ann_norm.text,
            "text_token": matches.text,
            "text_range": matches.range,
            "text_range_ann": ann_raw.text_range,
            "page": page,
            "label": ann_raw.label,
            "cui": cui,
            "nearest_cui_distance": None,  # fill these out in second pass
            "nearest_cui_token_info": None,
        })

    ann_cuis_df = pd.DataFrame(ann_cuis)
    # Add the CUI distance and anchor ID features
    out = ann_cuis_df.apply(
        lambda x: find_closest_cui_ann(x.text_range[0], x.page, ann_cuis_df), axis=1, result_type="expand")
    if not out.empty:
        ann_cuis_df[["nearest_cui_distance", "nearest_cui_token_info"]] = out

    return ann_cuis_df


def get_doc_annotation_cui_frame_no_norm(
        ann_file_path: str,
        ma_file_path: str,
        page_break_chars: list,
        label_enum: Type[LabelEnum],
        concept_feature_mapping: dict = CONCEPT_FEATURE_MAPPING):
    """
    Extract a dataframe from the label annotations and machine annotations
    Used in matching the annotations to a given raw text

    :param ann_file_path: absolute path to the target annotation file
    :param ma_file_path: absolute path to the target machine annotation file,
        NOTE: this file should be from the SAME text that the annotations were created from
    :param page_break_chars: list of the character positions for start, page breaks, and end of a document
    :param label_enum: LabelEnum for our target label
    :param concept_feature_mapping: dict of target feature enum with the expected label name
    :return: pd.DataFrame
    """
    machine_ann = MachineAnnotation(json_dict_input=common.read_json(ma_file_path))
    true_labels = BiomedDataSource.token_true_label_list(ann_file_path, machine_ann, label_enum)

    # get text cui by token_ix & label
    # need to get the CUIs from the text, as just a list of the annotations didnt really return CUIs appropriately
    token_cuis = []
    for token_ix in range(len(machine_ann)):
        cui = get_cui_for_token(token_ix, machine_ann, concept_feature_mapping=concept_feature_mapping)
        text_range = machine_ann.range[token_ix]
        page = get_token_page(text_range, page_break_chars)
        token_cuis.append({
            "token_ix": token_ix,
            "text": machine_ann.tokens[token_ix],
            "range": text_range,
            "page": page,
            "label": true_labels[token_ix],
            "cui": cui,
        })
    token_cui_df = pd.DataFrame.from_records(token_cuis)

    ######
    # get the annotation cuis from the machine_annotation output on the raw text
    # need to do this because for some reason the cuis arent coming out correctly on a list of ann text tokens

    # get brat ann, split phrases into individual tokens
    brat_ann = BiomedDataSource.get_brat_label(ann_file_path, label_enum)
    brat_ann = split_brat_annotation_phrases(brat_ann)
    ann_cuis = []
    for i, ann in enumerate(brat_ann):
        matches = token_cui_df[
            token_cui_df.range.apply(lambda x: x[0] >= ann.text_range[0] and x[1] <= ann.text_range[1] + 1)
        ]
        if matches.empty:
            # sanity check
            print(f"{i} no matches {ann}")
            continue

        # remove any non-annotated token matches, usually single character punctuation
        non_na_matches = matches[matches.label != 0]
        if len(non_na_matches.token_ix.values) > 1:
            print(f"Multiple token matches: {non_na_matches}")
        elif non_na_matches.empty:
            pass
            # print(f"No postive token matches for ann: {ann}; found n_matches={matches.shape[0]} ")

        cui = None
        if len(set(matches.cui)) == 1 or (len(set(matches.cui)) == 2 and None in set(matches.cui)):
            cui = list(set(matches.cui))[0]
        else:
            print(f"Found different CUIs for highlighted phrase:\n\t{ann}\n\t{matches}")
            continue

        ann_cuis.append({
            "token_ix": matches.token_ix.values[0],
            "text": ann.text,
            "text_range": ann.text_range,
            "page": get_token_page(ann.text_range, page_break_chars),
            "label": ann.label,
            "cui": cui,
            "nearest_cui_distance": None,  # fill these out in second pass
            "nearest_cui_token_info": None,
        })
    ann_cuis_df = pd.DataFrame(ann_cuis)

    # Add the CUI distance and anchor ID features
    out = ann_cuis_df.apply(
        lambda x: find_closest_cui_ann(x.text_range[0], x.page, ann_cuis_df), axis=1, result_type="expand")
    if not out.empty:
        ann_cuis_df[["nearest_cui_distance", "nearest_cui_token_info"]] = out

    return ann_cuis_df


def write_matched_ann_for_file(ann_cuis_df, ann_out_dir, base_name, machine_ann, page_break_chars, show_plots):
    # ---------------------------
    # for target file, iterate through pages and create score matrices
    # NOTE: this target text file may be different than the file used to create ann_cuis_df
    ann_list = []
    for page in range(len(page_break_chars) - 1):
        print(f"Running page: {page}")
        # page = 1

        token_sequence = [
            Token._make((
                machine_ann.tokens[i],
                machine_ann.range[i],
                get_cui_for_token(i, machine_ann, concept_feature_mapping=CONCEPT_FEATURE_MAPPING)
            ))
            for i in range(len(machine_ann))
            if
            machine_ann.range[i][0] >= page_break_chars[page] and machine_ann.range[i][1] < page_break_chars[page + 1]
        ]

        # annotations
        ann_cui_page_df = ann_cuis_df[ann_cuis_df.page == page]

        print(f"N tokens: {len(token_sequence)}, N anns: {ann_cui_page_df.shape[0]}")

        similar_scores = calc_similar_text_scores(ann_cui_page_df, token_sequence)
        distance_scores = calc_distance_scores(ann_cui_page_df, token_sequence)
        cui_distance_scores = calc_cui_distance_scores(ann_cui_page_df, token_sequence)

        # sum all of the scores; don't use distance
        score_sum = similar_scores + cui_distance_scores  # + distance_scores

        if show_plots:
            plt.matshow(similar_scores)
            plt.matshow(distance_scores)
            plt.matshow(cui_distance_scores)

            fig_size = plt.gcf().get_size_inches()
            ann_idx = 0
            fig, ax = plt.subplots(1, figsize=(fig_size[0], 2))
            ax.plot(cui_distance_scores[ann_idx, :], label="cui dist")
            ax.plot(similar_scores[ann_idx, :], label="similar")
            ax.plot(distance_scores[ann_idx, :], label="dist")
            ax.plot(score_sum[ann_idx, :], 'r', label="score")
            _ = ax.set_xlim((0, cui_distance_scores.shape[1]))
            plt.matshow(score_sum)
            plt.ion()
            plt.show()

        # Validate that the scores give us the correct annotation for each token
        token_ix_matches = np.argmax(score_sum, axis=1)
        text_ann_matches = [token_sequence[i] for i in token_ix_matches]

        is_match_count = 0
        imperfect_match_count = 0
        for i, (text_match, ann) in enumerate(zip(text_ann_matches, ann_cui_page_df.itertuples())):
            if text_match[0] != ann.text or text_match[1] != ann.text_range:
                print(i, text_match, "!=", (ann.text, ann.text_range, ann.label))
                imperfect_match_count += 1
            else:
                is_match_count += 1
            ann_list.append(
                Annotation(label=ann.label, text_range=text_match.text_range, text=text_match.text)
            )
        print(f"Perfect matches: {is_match_count}")
        print(f"Imperfect matches: {imperfect_match_count}")
    ann_set = AnnotationSet.from_list(ann_list)
    ann_set.remove_duplicate_entries()
    # write file out
    ann_file_path_out = os.path.join(ann_out_dir, base_name + ".ann")
    common.write_text(ann_set.to_file_content(), ann_file_path_out)


def main(show_plots=True):
    label_enum = LabLabel

    target_text_dir = googleocr_text_dir  # textract_text_dir, tesseract_text_dir, googleocr_text_dir

    datestr = datetime.date.today().isoformat()
    ann_out_dir = target_text_dir.replace("raw_text/", os.path.join(ANN_OUT_NAME, datestr) + os.path.sep)
    os.makedirs(ann_out_dir, exist_ok=True)

    # taken from lab model constants
    concept_feature_mapping = LabConstants.token_umls_representation_feat

    # get the file lists for original annotation text
    og_text_file_list = common.get_file_list(tagtog_text_dir, ".txt", recurse=True)
    og_ma_output_path = tagtog_text_dir.replace(project_root, features_root_dir)
    og_ma_file_list = common.get_file_list(og_ma_output_path, ".json", recurse=True)

    text_file_list = common.get_file_list(target_text_dir, ".txt", recurse=True)
    ma_output_path = target_text_dir.replace(project_root, features_root_dir)
    ma_file_list = common.get_file_list(ma_output_path, ".json", recurse=True)
    ann_raw_file_list = common.get_file_list(ann_raw_dir, ".ann", recurse=True)
    ann_norm_file_list = common.get_file_list(ann_norm_dir, ".ann", recurse=True)

    # get the page breaks for the given text files
    file_page_break_chars = {}
    for txt_fn in text_file_list:
        # the ann filenames all had dashes swapped out, so we do the same here
        base_name = get_base_file_name(txt_fn)
        file_page_break_chars[base_name] = get_page_break_char(common.read_text(txt_fn))

    ######
    for file_ix in range(len(og_text_file_list)):
        # file_ix = 1

        og_text_file_path = og_text_file_list[file_ix]
        base_name = get_base_file_name(og_text_file_path)
        og_ma_file_path = find_matched_file_in_list(base_name, og_ma_file_list)

        print("origin annotation files")
        print(og_text_file_path, "\n", og_ma_file_path)

        text_file_path = find_matched_file_in_list(base_name, text_file_list)
        ma_file_path = find_matched_file_in_list(base_name, ma_file_list)
        ann_raw_file_path = find_matched_file_in_list(base_name, ann_raw_file_list)
        ann_norm_file_path = find_matched_file_in_list(base_name, ann_norm_file_list)

        print("target OCR files")
        print(text_file_path, "\n", ann_raw_file_path, "\n", ann_norm_file_path, "\n", ma_file_path)

        page_break_chars = file_page_break_chars[base_name]
        # ann_raw_set = AnnotationSet.from_file_content(common.read_text(ann_raw_file_path))
        # ann_norm_set = AnnotationSet.from_file_content(common.read_text(ann_norm_file_path))
        machine_ann = MachineAnnotation(json_dict_input=common.read_json(ma_file_path))

        # for our target file, create the annotation dataframe with the associated cuis
        # This step can be error prone, if we have a hard time matching the file tokens to the annotations
        # For example, if we are trying to use the normalized text with shifted ranges, it won't work so well.
        # ann_cuis_df = get_doc_annotation_cui_frame_no_norm(
        #     ann_file_path, og_ma_file_path, page_break_chars=page_break_chars, label_enum=label_enum)

        ann_cuis_df = get_doc_annotation_cui_frame_norm(
            ann_raw_file_path, ann_norm_file_path, og_ma_file_path, page_break_chars=page_break_chars, label_enum=label_enum)

        write_matched_ann_for_file(ann_cuis_df, ann_out_dir, base_name, machine_ann, page_break_chars, show_plots)

        # end for file
    # end all files


if __name__ == "__main__":
    main(show_plots=False)

