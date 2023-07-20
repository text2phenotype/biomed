from text2phenotype.common.demographics import Demographics
from text2phenotype.common import common
from text2phenotype.constants.features import DemographicEncounterLabel

DEMOGRAPHIC_FIELD_MAPPING = {
    DemographicEncounterLabel.dob: 'Birth_date',
    DemographicEncounterLabel.sex: 'Gender'
}

def get_demographic_out(demographic_fp):
    predicted_demographics = Demographics(**common.read_json(demographic_fp))

