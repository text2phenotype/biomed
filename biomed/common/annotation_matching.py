import datetime

import math
from typing import List
import bisect

from text2phenotype.common.featureset_annotations import IndividualFeatureOutput

MAX_TOKENS_BTWN_ANNOTATION = 30


def get_closest_nearby_annotation(
        token_index: int,
        annotation_token_indexes: List[int],
        max_token_annot_dist=MAX_TOKENS_BTWN_ANNOTATION,
        prefer_annotation_after_token: bool = False) -> [int, None]:
    """
    :param token_index: token index for the token you want to find a nearby annotation
    :param annotation_token_indexes: MachineAnnotation[FeatureType].token_indexes must be sorted
    :param max_token_annot_dist: max number of tokens between the token you want to assign and the FS annotation
    :param prefer_annotation_after_token: whether to prefer annotation after the token (within max distance range)
    when FALSE will find closest FS annotation either way,
    if TRUE will find closest FS annotation after the token within max distance range,
    if no such FS annotation exists will pick closest FS annotation before token
    :return: token index of nearest annotation
    """
    bisected_idx = bisect.bisect_left(annotation_token_indexes, token_index)
    out = None
    if bisected_idx < 0:
        return
    if bisected_idx == len(annotation_token_indexes):
        dist_after = math.inf
    else:
        closest_feat_annot_after = annotation_token_indexes[bisected_idx]
        dist_after = abs(closest_feat_annot_after - token_index)

    if dist_after < max_token_annot_dist and (prefer_annotation_after_token or bisected_idx == 0):
        out = closest_feat_annot_after
    # don't check before if first match is index = 0
    elif bisected_idx > 0:
        closest_feat_annot_before = annotation_token_indexes[bisected_idx - 1]
        dist_before = abs(closest_feat_annot_before - token_index)
        if dist_after <= dist_before and dist_after < max_token_annot_dist:
            out = closest_feat_annot_after
        elif dist_before < max_token_annot_dist:
            out = closest_feat_annot_before
    return out


def get_closest_date(token_idx: int, dates: IndividualFeatureOutput ):
    closest_date_idx = get_closest_nearby_annotation(token_idx, dates.sorted_token_indexes)
    if closest_date_idx is not None:
        closest_date = dates[closest_date_idx][0]
        closest_date_obj = datetime.date(
            year=closest_date['year'],
            month=closest_date['month'],
            day=closest_date['day'])
    else:
        closest_date_obj = None
    return closest_date_obj
