from collections import defaultdict

from text2phenotype.common.featureset_annotations import IndividualFeatureOutput

from text2phenotype.common.feature_data_parsing import overlap_ranges
from text2phenotype.constants.features import FeatureType
from text2phenotype.entity.attributes import TextSpan


PROBLEM_LIST_QUESTIONS = [
    # 'alcohol_problem',
    # 'alcohol_user',
    'anxiety',
    'asthma',
    'auto_immune_disease',
    'auto_immune_disease_ra',
    'auto_immune_disease_type',
    'brain_injury',
    'cancer',
    'chronic_pain',
    # 'cigarette_smoker',
    'cirrhosis_diagnosis',
    'cirrhosis_diagnosis_ascites',
    'cirrhosis_diagnosis_encephalopathy',
    'cirrhosis_diagnosis_variceal_bleed',
    'copd',
    'coronary_artery_disease',
    'cryoglobulinemia',
    'depression',
    'diabetes_mellitus',
    # 'drug_user',
    # 'drug_user_benzos',
    # 'drug_user_opiates',
    # 'drug_user_stimulants',
    # 'drug_user_weed',
    'hcv_diagnosis',
    'hep_b_chronic',
    'hepatocelular_carcinoma',
    'hiv',
    'hypertension',
    # 'injection_drug_user',
    'mania_hypomania',
    'peripheral_neuropathy',
    'renal_insufficiency',
    'seizure_disorder']


def filter_problem_list(form: IndividualFeatureOutput) -> dict:
    """
    Filter the form to only include diagnosis that were predicted in the clinical summary
    :param form: HEPC form results
    :param summary: predicted clinical summary (not HEPC specific)
    :return: HEPC form results, filtered by clinical summary
    """
    filtered = defaultdict(list)
    # process the annotation
    for entry_index in form.input_dict:
        entry = form[entry_index][0]
        section_question = entry[FeatureType.form.name].split('//')
        section = section_question[0]
        question = section_question[1]
        keys = set(entry.keys())
        keys.remove(FeatureType.form.name)
        text = list(keys)[0]
        evidence = entry[text]
        filtered[section].append({'suggest': question,
                        'evidence': evidence})

    return filtered


def answer_none_for_section(section: dict) -> list:
    return [{'suggest': question['suggest'], 'evidence': None} for question in section]


def filter_route_of_infection(form: dict) -> dict:
    """
    HEPC Route of Infection is being poorly predicted --
    Text2phenotype should make no claims of evidence especially about
    HIV co-infection or STD transmission at this time.

    :param form: hepc form
    :return: hepc form
    """
    conservative = form.copy()
    conservative['ROUTE_OF_INFECTION'] = answer_none_for_section(form['ROUTE_OF_INFECTION'])
    return conservative
