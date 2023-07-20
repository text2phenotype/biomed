import datetime
import enum
from functools import partial
import re
from typing import List

from text2phenotype.common.dates import parse_dates
from text2phenotype.common.log import operations_logger


class RaceCodesToRegexMapping(enum.Enum):
    WHITE = r"white"
    BLACK_OR_AFRICAN_AMERICAN = r"black|african"
    ASIAN = r"asia"
    AMERICAN_INDIAN_OR_ALASKA_NATIVE = r"native amer|american indian|alaska"
    NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER = r"hawaiian|pacific islander"
    MULTIPLE = r"mult"
    UNKNOWN = r"unk"


class EthnicityCodesToRegex(enum.Enum):
    Not_Hispanic_or_Latino = r"no[nt]?\W{0,4}hisp"
    Hispanic_or_Latino = r"hisp|latin"
    Unknown = 'unk'


def get_max_prob_regex_condition(data: List, regex_pattern=re.compile('[a-zA-Z]+'), strict_match: bool = False):
    # type assertion
    matches = regex_condition(data, regex_pattern, strict_match)
    if not matches:
        return
    res = sorted(matches, key=lambda i: i[1], reverse=True)[0]
    if res:
        return res[0], res[1]


def get_max_prob_list_sum(data: List[tuple]):
    res = dict()
    for pred_tuple in data:
        if pred_tuple[0] in res:
            res[pred_tuple[0]][0] += pred_tuple[1]
            if res[pred_tuple[0]][1] < pred_tuple[1]:
                res[pred_tuple[0]][1] = pred_tuple[1]
        else:
            res[pred_tuple[0]] = [pred_tuple[1], pred_tuple[1]]
    if res:
        best_response = sorted(res.items(), key=lambda item: item[1], reverse=True)[0]
        # return the best name and then the max prob for that name, ensuring the score is always < 1
        return best_response[0], best_response[1][1]


def regex_condition(data: List, regex_pattern=re.compile('[a-zA-Z]+'), strict_match: bool = False):
    matches = []
    for s in range(len(data)):
        match = regex_pattern.match(data[s][0])
        if match and (not strict_match or (strict_match and match.endpos == match.regs[0][1])):
            matches.append(data[s])
    return matches


def get_max_probability(data: List):
    res = sorted(data, key=lambda i: i[1], reverse=True)[0]
    if res:
        return res[0], res[1]


def summarize_sex(data: List):
    prob_female = 0
    prob_male = 0
    for sex in data:
        if sex[0].lower() in ['female', 'f', 'woman', 'gentlewoman']:
            prob_female += sex[1] + .0001
        elif sex[0].lower() in ['male', 'm', 'man', 'gentleman']:
            prob_male += sex[1] + .0001

    if prob_female + prob_male > 0:
        prob_female_total = prob_female / (prob_female + prob_male)
        prob_male_total = prob_male / (prob_female + prob_male)
        if prob_female_total > 0.5:
            return 'Female', prob_female_total
        elif prob_male_total > 0.5:
            return 'Male', prob_male_total


def summarize_age(data: List):
    max_prob_age = get_max_prob_regex_condition(data, re.compile(r"\d{1,3}"))
    if not max_prob_age:
        return
    # extract age as years
    if not re.match("y(ear|r|/?o)", max_prob_age[0]) and re.match("mo(nth)?|d(ay)?", max_prob_age[0]):
        if re.match("mo?(nth)?", max_prob_age[0]):
            age = extract_number(max_prob_age) / 12
        else:
            age = 0
    else:
        age = extract_number(max_prob_age[0])
    return age, max_prob_age[1]


def extract_number(string: str):
    c = re.match(r"\d{1,3}", string)
    if c:
        return c.string[c.regs[0][0]:c.regs[0][1]]


def summarize_dob(data: List, age_guess: int = 0):
    curr_year = datetime.datetime.now().year
    most_likely_dob = None
    prob = 0
    for i in data:
        dob = parse_dates(i[0])
        if dob:
            year = dob[0][0].year
            if age_guess - 2 <= curr_year - year <= 200 and i[1] >= prob:
                most_likely_dob = dob[0][0].strftime('%m/%d/%Y')
                prob = i[1]
    return most_likely_dob, prob


def summarize_name(data, full_name):
    overlap = []
    if full_name:
        for row in data:
            for name in full_name:
                if row[0] in name['text']:
                    overlap.append(row)

    return overlap


def summarize_patient_name(data: List, pat_name_full: List = None):
    overlap = summarize_name(data, pat_name_full)

    if overlap:
        return get_max_prob_list_sum(overlap)
    else:
        return get_max_prob_list_sum(data)


def get_max_prob_all_unique(data):
    if not data:
        return

    highest_prob = {}
    for entry in data:
        text = entry[0].lower()
        prob = entry[1]
        if text not in highest_prob:
            highest_prob[text] = (prob, entry[0])
        elif highest_prob[text][0] < prob:
            highest_prob[text] = (prob, entry[0])

    output = [None] * len(highest_prob)
    counter = 0
    for k, v in highest_prob.items():
        output[counter] = [v[1], v[0]]
        counter += 1
    return output


def get_standardized_race(text: str):
    race_matches = set()
    for code in RaceCodesToRegexMapping:
        if re.search(code.value, text, flags=re.IGNORECASE):
            race_matches.add(code.name)
    if len(race_matches) == 0:
        operations_logger.info(f"NO RACE CODE FOUND FOR RACE {text} RETURNING UNKNOWN")
        race_match = RaceCodesToRegexMapping.UNKNOWN.name
    elif len(race_matches) == 1:
        race_match = list(race_matches)[0]
    if len(race_matches) > 1:
        operations_logger.info(f"Multiple RACE CODES FOUND FOR RACE {text} RETURNING Multiple")
        race_match = RaceCodesToRegexMapping.MULTIPLE.name
    return race_match

def get_all_races(data: List[tuple]):
    race_to_max_prob = dict()
    for item in data:
        standardized_race = get_standardized_race(item[0])
        if standardized_race not in race_to_max_prob:
            race_to_max_prob[standardized_race] = item[1]
        elif item[1] > race_to_max_prob[standardized_race]:
            race_to_max_prob[standardized_race] = item[1]
    # if non unknown codes are return, exclude unknown from the response
    if len(race_to_max_prob) > 0 and RaceCodesToRegexMapping.UNKNOWN.name in race_to_max_prob:
        del race_to_max_prob[RaceCodesToRegexMapping.UNKNOWN.name]

    return [(race, max_prob) for race,  max_prob in race_to_max_prob.items()]


def get_standardized_ethnicity(text: str):
    if re.search(EthnicityCodesToRegex.Not_Hispanic_or_Latino.value, text, flags=re.IGNORECASE):
        ethnicity_code = EthnicityCodesToRegex.Not_Hispanic_or_Latino.name
    elif re.search(EthnicityCodesToRegex.Hispanic_or_Latino.value, text, flags=re.IGNORECASE):
        ethnicity_code = EthnicityCodesToRegex.Hispanic_or_Latino.name
    else:
        operations_logger.info(f"No standardized ethnicity code found for '{text}', returning unknown")
        ethnicity_code = EthnicityCodesToRegex.Unknown.name
    return ethnicity_code

def get_best_ethnicity(data: List[tuple]):
    ethnicity_to_max_prob = dict()
    for item in data:
        eth = get_standardized_ethnicity(item[0])
        if eth not in ethnicity_to_max_prob:
            ethnicity_to_max_prob[eth] = item[1]
        elif item[1] > ethnicity_to_max_prob[eth]:
            ethnicity_to_max_prob[eth] = item[1]
    # if there are multiple ethnicity codes, remove unknown
    if len(ethnicity_to_max_prob) != 1:
        if EthnicityCodesToRegex.Unknown.name in ethnicity_to_max_prob:
            del ethnicity_to_max_prob[EthnicityCodesToRegex.Unknown.name]
        # both not hispanic and hispanic were returned, take the not value bc it requires no/not/non in front
        if len(ethnicity_to_max_prob) > 1 and EthnicityCodesToRegex.Hispanic_or_Latino.name in ethnicity_to_max_prob:
            del ethnicity_to_max_prob[EthnicityCodesToRegex.Hispanic_or_Latino.name]

    return [(eth, max_prob) for eth, max_prob in ethnicity_to_max_prob.items()]


class FhirDemographicEnforcement(enum.Enum):
    ssn = partial(get_max_prob_regex_condition, regex_pattern=re.compile(r"\d{3}[\-\s~]?\d{2}[\s\-~]?\d{4}"),
                  strict_match=True)
    pat_first = partial(summarize_patient_name)
    pat_middle = partial(summarize_patient_name)
    pat_last = partial(summarize_patient_name)
    pat_age = partial(summarize_age)
    pat_street = partial(get_max_prob_all_unique)
    pat_city = partial(get_max_prob_all_unique)
    pat_zip = partial(get_max_prob_regex_condition, regex_pattern=re.compile(r"\d{5}([-\s]\d{4})?"), strict_match=True)
    pat_state = partial(get_max_prob_regex_condition, regex_pattern=re.compile(r"([A-Z]{2})"))
    pat_phone = partial(regex_condition,
                        regex_pattern=re.compile(r"(\d{1,2}[(\s)])?(\d{3}[\W]*)?\d{3}\W*\d{4}\W*"),
                        strict_match=True)

    pat_email = partial(regex_condition, regex_pattern=re.compile(r".*@.*\.(com?|net|edu|gov)"))
    sex = partial(summarize_sex)  # output will only be  male, female or other
    dob = partial(summarize_dob)

    dr_phone = partial(regex_condition,
                       regex_pattern=re.compile(r"(\d{1,2}[(\s)])?(\d{3}[\W]*)?\d{3}\W*\d{4}\W*"),
                       strict_match=True)
    dr_zip = partial(regex_condition, regex_pattern=re.compile(r"\d{5}([-\s]\d{4})?"), strict_match=True)

    dr_state = partial(get_max_prob_all_unique)
    dr_city = partial(get_max_prob_all_unique)
    dr_first = partial(get_max_prob_all_unique)
    dr_middle = partial(get_max_prob_all_unique)
    dr_last = partial(get_max_prob_all_unique)
    dr_street = partial(get_max_prob_all_unique)
    dr_email = partial(get_max_prob_all_unique)
    dr_fax = partial(get_max_prob_all_unique)
    dr_id = partial(get_max_prob_all_unique)
    dr_initials = partial(get_max_prob_all_unique)
    dr_org = partial(get_max_prob_all_unique)
    facility_name = partial(get_max_prob_all_unique)
    insurance = partial(get_max_prob_all_unique)
    mrn = partial(get_max_prob_all_unique)
    race = partial(get_all_races)
    ethnicity = partial(get_best_ethnicity)
