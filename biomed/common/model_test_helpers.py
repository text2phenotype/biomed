from collections import defaultdict
from typing import List, Dict, Tuple

from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.common.log import operations_logger


def cuis_from_annot(clinical_annotation: List[Dict[str, List[dict]]], sem_types_to_include: [list, set]) -> list:
    cuis = []
    for entry in clinical_annotation:
        for sem_type in entry:
            if sem_type in sem_types_to_include:
                for concept in entry[sem_type]:
                    cuis.append(concept.get('cui'))
    return cuis


def cui_set_for_entry(index, concept_feature_mapping, tokens: MachineAnnotation) -> list:
    cuis = list()
    for concept_feature in concept_feature_mapping:
        if tokens[concept_feature, index]:
            cuis.extend(cuis_from_annot(tokens[concept_feature, index], concept_feature_mapping[concept_feature]))
    return cuis


def document_cui_set(category_list, concept_feature_mapping, tokens: MachineAnnotation):
    #  where concept feature mapping is key=FeatureType and value = list[semtype_strings]
    cui_mapping = defaultdict(list)
    for i in range(len(category_list)):
        cuis = cui_set_for_entry(i, concept_feature_mapping, tokens=tokens)
        # only use the first cui
        if len(cuis) >= 1:
            cui = cuis[0]
            cui_mapping[cui].append(category_list[i])
    return cui_mapping


def check_cui_match(cuis_set_1: set, cui_set_2: set, strict: bool = False):
    match = len(cuis_set_1.intersection(cui_set_2)) > 0
    if strict:
        return match
    only_cui_1 = (len(cui_set_2) == 0 and len(cuis_set_1) > 0)
    only_cui_2 = (len(cuis_set_1) == 0 and len(cui_set_2) > 0)
    return match or only_cui_1 or only_cui_2


def prep_cui_reports(
        actual_cui_mapping: Dict[str, list],
        predicted_cui_mapping: Dict[str, list]) -> Tuple[MachineAnnotation, list, list]:
    # goal is to get teh cui sets into a format that can be interpreted by testing reports classes

    num_cuis = len(actual_cui_mapping)
    if len(actual_cui_mapping) != len(predicted_cui_mapping):
        operations_logger.warning("Cuis found for actual and predicted are not Matches")

    predicted_cat = [None] * num_cuis
    actual_cat = [None] * num_cuis

    cui_list = list(actual_cui_mapping.keys())
    cui_tokens = MachineAnnotation(json_dict_input={'token': cui_list, 'range': [(None, None)] * num_cuis})

    # loop through all cuis
    for idx in range(num_cuis):
        cui = cui_list[idx]
        actual_categories = list(set(actual_cui_mapping[cui]))
        predicted_categories = list(set(predicted_cui_mapping[cui]))

        # if only 1 category for the cui, then that single category is the value
        if len(actual_categories) == 1 and len(predicted_categories) == 1:
            actual_cat[idx] = actual_categories[0]
            predicted_cat[idx] = predicted_categories[0]
        else:
            actual_cat_set = set(actual_categories)
            predicted_cat_set = set(predicted_categories)

            # give benefit of the doubt to non-zero (aka non-na labels)
            # ie if a cui is pulled out anywhere in document it is extracted
            if 0 in actual_cat_set:
                actual_cat_set.remove(0)
            if 0 in predicted_cat_set:
                predicted_cat_set.remove(0)
            matched_cats = actual_cat_set.intersection(predicted_cat_set)
            # if there's a match on any non-na category use that, this biases towards our model
            if len(matched_cats) >= 1:
                cat = min(matched_cats)
                actual_cat[idx] = cat
                predicted_cat[idx] = cat
            # if theres no match choose min non na category by index (ensures that same output will happen every round)
            elif len(actual_cat_set) >= 1 and len(predicted_cat_set) >= 1:
                actual_cat[idx] = min(actual_cat_set)
                predicted_cat[idx] = min(predicted_cat_set)
            # take advantage of fact that if a set after removing 1 element (0) has len == 0 t
            # hen it's original length was 1
            elif len(actual_cat_set) >= 1:
                actual_cat[idx] = min(actual_cat_set)
                predicted_cat[idx] = predicted_categories[0]
            else:
                actual_cat[idx] = actual_categories[0]
                predicted_cat[idx] = min(predicted_cat_set)

    return cui_tokens, actual_cat, predicted_cat
