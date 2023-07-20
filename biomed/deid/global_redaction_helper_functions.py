import datetime
from typing import List, Set, Dict

from text2phenotype.common.featureset_annotations import IndividualFeatureOutput, MachineAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.deid import DeidGroupings
from text2phenotype.constants.features import FeatureType, PHILabel

from biomed.common.biomed_ouput import BiomedOutput
from biomed.common.biomed_summary import FullSummaryResponse
from biomed.common.aspect_response import AspectResponse
from biomed.constants.constants import SENSITIVE_DEMOGRAPHIC_STRING_CATS


def phi_from_dob_match(tokens: MachineAnnotation, dob: datetime.datetime) -> List[BiomedOutput]:
    """
    :param tokens: Machine annotation object for the text
    :param dob: The date time object you want to count as DOB
    :return: a List of BiomedOutput objects that include any date that the date comprehension feature thinks matches
    the key date
    """
    date_comprehension_annots: IndividualFeatureOutput = tokens[FeatureType.date_comprehension]
    if date_comprehension_annots is None:
        operations_logger.warning(
            "Date Comprehension Feature was not annotated for, matcihng DOBs will not be added as patient phi tokens")

    dob_phi_tokens = []
    for annot_idx in date_comprehension_annots.token_indexes:
        # note that token indexes is a set so there is NO guaranteed order of tokens coming out of this function
        if any([dates_match_strict(date_annot, dob) for date_annot in date_comprehension_annots[annot_idx]]):
            dob_phi_tokens.append(
                BiomedOutput(
                    text=tokens.tokens[int(annot_idx)],
                    range=tokens.range[int(annot_idx)],
                    label=PHILabel.patient.value.persistent_label,
                    lstm_prob=1.0
                ))

    return dob_phi_tokens


def dates_match_strict(annot: dict, date_to_match: datetime.datetime) -> bool:
    """
    :param annot: a single date comprehension annotation for a token, equivalent to MachineAnnotation[FeatureType.date_comprehension, i]
    :param date_to_match: a datetime object
    :return: a boolean of whether the dates strictly match
    """
    if not date_to_match:
        return False
    # return true if the dates strictly match
    year_match = annot.get('year') == date_to_match.year
    day_match = annot.get('day') == date_to_match.day
    month_match = annot.get('month') == date_to_match.month
    return year_match and day_match and month_match


def dob_date_time_from_demographic_json(demographic_json: dict) -> datetime.datetime:
    """
    :param demographic_json: the fully reassembled demographic json output, note that the dob key is always included,
    and when there was a dob the dob is always in mm/dd/yyyy form
    :return: returns the DOB as a datetime object
    """
    dob = demographic_json.get('dob', [])
    if not dob:
        operations_logger.warning("Demographics result has No DOB, cannot add plausible matches")
        return
    dob_date_string = dob[0][0]
    dob_datetime = datetime.datetime.strptime(dob_date_string, '%m/%d/%Y')
    return dob_datetime


def parse_demographic_string(dem_text: str):
    """
    :param dem_text: A string that corresponds to a demographic response that is identifying and must be globally redacted
    :return: a set of token level strings for global redaction, splits into separate words and allows for some
    differences in formatting of key numbers by including both strings with - and without
    """
    # parse into tokens and equivalently formatted strings
    # add all space seperated valeus
    dem_text = dem_text.lower()
    dem_set = set(dem_text.split())
    # call equivalent things with - vs without ie: 609-834-1392 == 6098341392 both should be redacted
    new_set = set()
    for entry in dem_set:
        if '-' in entry:
            new_set.add(entry.replace('-', ''))

    dem_set = dem_set.union(new_set)
    return dem_set


def get_sensitive_demographic_tokens(demographics_json: Dict[str, list]) -> Set[str]:
    """
    :param demographics_json: The full reassembled .demographics.json respons
    :return: A set of strings (ideally should be tokens), any token that strictly matches these will be redacted
    """
    sensitive_tokens = set()
    for sensitive_dem_type in SENSITIVE_DEMOGRAPHIC_STRING_CATS:
        for dem_pred in demographics_json.get(sensitive_dem_type, []):
            if dem_pred:
                dem_string = dem_pred[0]
                sensitive_tokens = sensitive_tokens.union(parse_demographic_string(dem_string))

    return sensitive_tokens


def all_strict_matching_token_strings(machine_annotation: MachineAnnotation, tokens_to_match: Set[str]) -> List[
    BiomedOutput]:
    """
    :param machine_annotation: The full reassembled machine annotation loaded from .annotation.json
    :param tokens_to_match: a set of string values if any token string matches, that token should be redacted as patient
    :return: a list of biomed outputs of PHI results that should be added to the PHI respons
    """
    additional_phi_tokens = []
    for i in range(len(machine_annotation)):
        if machine_annotation.tokens[i].lower().strip() in tokens_to_match:
            additional_phi_tokens.append(
                BiomedOutput(
                    text=machine_annotation.tokens[i],
                    label=PHILabel.patient.value.persistent_label,
                    range=machine_annotation.range[i],
                    score=1.0
                ))
    return additional_phi_tokens


def global_redact_dem_strings(demographic_json: dict, machine_annotation: MachineAnnotation) -> List[BiomedOutput]:
    """
    :param demographic_json: The full reassembled result of the demographics operation
    :param machine_annotation: the reassembled annotation file
    :return: a list of PHI tokens that should definitely be added to the PHI list response
    """
    demographics = get_sensitive_demographic_tokens(demographic_json)
    phi_list = all_strict_matching_token_strings(machine_annotation=machine_annotation, tokens_to_match=demographics)
    return phi_list


def globally_redact_phi_tokens(
        demographic_json: dict,
        machine_annotation: MachineAnnotation,
        phi_token_json: dict) -> FullSummaryResponse:
    """
    :param demographic_json: The .demographics.json file that is creatted POST-REASSEMBLER, of form
    {"pat_first": [["sam", 0.9]]}
    :param machine_annotation: The full reassembled machine annotation loaded from .annotation.json
    :param phi_token_json: the full reassembled PHI token json of form {"PHI": [{BIOMED_OUTPUT_JSON}]}
    :return: A Full PHI summary response with all instances of dates that match birthdays or tokens
    that match key patient identifiers included
    """
    phi_token_response = FullSummaryResponse.from_json(phi_token_json)
    # add DOB matches
    dob_date_time = dob_date_time_from_demographic_json(demographic_json=demographic_json)
    phi_dob_tokens = phi_from_dob_match(tokens=machine_annotation, dob=dob_date_time)
    phi_token_response[PHILabel.get_category_label().persistent_label].response_list.extend(phi_dob_tokens)
    phi_dem_matching = global_redact_dem_strings(
        demographic_json=demographic_json, machine_annotation=machine_annotation)
    phi_token_response[PHILabel.get_category_label().persistent_label].response_list.extend(phi_dem_matching)

    return phi_token_response


def filter_phi_response(
        phi_response: FullSummaryResponse,
        phi_categories_to_include: Set[PHILabel] = DeidGroupings.SAFE_HARBOR.value) -> FullSummaryResponse:
    """
    :param phi_response: The phi full summary response object loaded from the .phi_tokens.json file
    :param phi_categories_to_include: A set of the PHI labels that SHOULD be redacted, Almost always we should be using
     one of the DEIDGroupings, but this could theoretically be anything
     return: A new phi FUll Summary Response with only the values to be used for redaction
    """
    filtered_response_list = []

    phi_categories_to_include_names = {a.value.persistent_label for a in phi_categories_to_include}
    operations_logger.info(f'Filtering PHI responses for redaction to only include {phi_categories_to_include_names}')

    for phi_output in phi_response[PHILabel.get_category_label().persistent_label].response_list:
        if phi_output.label in phi_categories_to_include or phi_output.label in phi_categories_to_include_names:
            filtered_response_list.append(phi_output)

    return FullSummaryResponse(
        aspect_responses=[
            AspectResponse(
                PHILabel.get_category_label().persistent_label, filtered_response_list
            )])


def global_redact_and_filter(
        demographic_json: dict,
        machine_annotation: MachineAnnotation,
        phi_token_json: dict,
        phi_categories_to_include: Set[PHILabel] = DeidGroupings.SAFE_HARBOR.value) -> FullSummaryResponse:
    """
    :param demographic_json: The .demographics.json file that is creatted POST-REASSEMBLER, of form
    {"pat_first": [["sam", 0.9]]}
    :param machine_annotation: The full reassembled machine annotation loaded from .annotation.json
    :param phi_token_json: the full reassembled PHI token json of form {"PHI": [{BIOMED_OUTPUT_JSON}]}
    :param phi_categories_to_include: A set of PHI Labels that should be redacted
    :return: A Full PHI summary response with all instances of dates that match birthdays or tokens
    that match key patient identifiers included, and excludes any phi type that is not required for redaction
    """
    phi_res = globally_redact_phi_tokens(
        demographic_json=demographic_json, machine_annotation=machine_annotation, phi_token_json=phi_token_json)
    phi_res = filter_phi_response(phi_res, phi_categories_to_include=phi_categories_to_include)
    return phi_res


def redact_text(phi_tokens: FullSummaryResponse, text: str) -> str:
    # Create deidentified text
    text_deid = text
    for phi in phi_tokens[PHILabel.get_category_label().persistent_label].response_list:
        r = phi.range
        text_deid = text_deid[0:r[0]] + '*' * (r[1] - r[0]) + text_deid[r[1]:]

    return text_deid
