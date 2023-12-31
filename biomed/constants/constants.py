from enum import IntEnum
from typing import Union

# NOTE(2021.09.05): ModelType imported from biomed.constants.constants in many places
from biomed.constants.model_enums import ModelType, VotingMethodEnum
# NOTE(mjp): importing MODEL_TYPE_2_CONSTANTS could eventually cause a circular import. Move get_model_*() to getters?
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS

from text2phenotype.common.version_info import VersionInfo
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import ProblemLabel, SignSymptomLabel, LabLabel, FeatureType, DemographicEncounterLabel
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.biomed_env import BiomedEnv

DEFAULT_MIN_SCORE = 0.10

ICD10_CATEGORY = 'ICD10ClinicalCode'
DATE_OF_SERVICE_CATEGORY = 'dos'

BIOMED_VERSION_TO_MODEL_VERSION = {
    '2.07': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.00',
        ModelType.demographic: '2.00',
        ModelType.oncology: '3.01',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.00',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: "0.00"
    },
    '2.08': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.01',
        ModelType.demographic: '2.00',
        ModelType.oncology: '3.01',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.00',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: '0.00'
    },
    'nmibc_1.00': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.01',
        ModelType.demographic: '2.00',
        ModelType.oncology: 'nmibc_1.00',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.00',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: "0.00"
    }, '2.09': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.02',
        ModelType.demographic: '2.00',
        ModelType.oncology: '3.01',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.00',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: "0.00"
    },'2.10': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.03',
        ModelType.demographic: '2.00',
        ModelType.oncology: '3.01',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.00',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: "0.00"
    }, '2.11': {
        ModelType.deid: '1.01',
        ModelType.lab: '1.03',
        ModelType.demographic: '2.00',
        ModelType.oncology: '3.01',
        ModelType.smoking: '0.00',
        ModelType.drug: '2.01',
        ModelType.vital_signs: '2.00',
        ModelType.diagnosis: '2.01',
        ModelType.covid_lab: '1.01',
        ModelType.device_procedure: '1.00',
        ModelType.imaging_finding: '2.00',
        ModelType.doc_type: '0.00',
        ModelType.date_of_service: '0.00',
        ModelType.genetics: "0.00",
        ModelType.bladder_risk: "0.00",
        ModelType.sequioa_bladder: "0.00",
        ModelType.family_history: "0.00",
        ModelType.sdoh: "0.00",
        ModelType.procedure: "0.00"
    },

}

OperationToModelType = {
    TaskOperation.phi_tokens: [ModelType.deid],
    TaskOperation.demographics: [ModelType.demographic],
    TaskOperation.oncology_only: [ModelType.oncology],
    TaskOperation.doctype: [ModelType.doc_type],
    TaskOperation.drug: [ModelType.drug],
    TaskOperation.lab: [ModelType.lab],
    TaskOperation.disease_sign: [ModelType.diagnosis],
    TaskOperation.imaging_finding: [ModelType.imaging_finding],
    TaskOperation.device_procedure: [ModelType.device_procedure, ModelType.procedure],
    TaskOperation.smoking: [ModelType.smoking],
    TaskOperation.vital_signs: [ModelType.vital_signs],
    TaskOperation.covid_lab: [ModelType.covid_lab],
    TaskOperation.date_of_service: [ModelType.date_of_service],
    TaskOperation.icd10_diagnosis: [ModelType.diagnosis],
    TaskOperation.genetics: [ModelType.genetics],
    TaskOperation.bladder_risk: [ModelType.bladder_risk, ModelType.sequioa_bladder],
    TaskOperation.family_history: [ModelType.family_history],
    TaskOperation.sdoh: [ModelType.sdoh]
}

EXCLUDED_LABELS = {'token'}

OperationToRepresentationFeature = {
    TaskOperation.icd10_diagnosis: [FeatureType.clinical_code_icd10, FeatureType.icd10_diagnosis]
}


def get_model_type_version(model_type: ModelType, biomed_version=None):
    """Get the matching version key for the given model_type and biomed_version"""
    biomed_version = biomed_version or BiomedEnv.DEFAULT_BIOMED_VERSION.value
    if biomed_version not in BIOMED_VERSION_TO_MODEL_VERSION:
        raise ValueError(f"Requested BIOMED version: {biomed_version} is not supported at this time, supported versions"
                         f" are {list(BIOMED_VERSION_TO_MODEL_VERSION.keys())}")
    model_type_version = BIOMED_VERSION_TO_MODEL_VERSION[biomed_version][model_type]
    return model_type_version


def get_version_model_folders(model_type: ModelType, biomed_version=None):
    """Get the list of model folders from the model_type constants"""
    # for a given model type if the model folder list is specified by env var, take that value
    # (and parse apart comma seperated string) else use META_MODEL_FILE_MAPPINGS
    model_type_version = get_model_type_version(model_type, biomed_version)
    model_file_list = MODEL_TYPE_2_CONSTANTS[model_type].model_version_mapping()[model_type_version]
    return model_file_list


def get_ensemble_version(model_type: ModelType, biomed_version=None) -> Union["EnsembleVersion", None]:
    """
    Get the versioned EnsembleVersion object from the model_type constants

    :param model_type: ModelType enum for the target model type
    :param biomed_version: target version string; if None, use default BIOMED version
    :return: EnsembleVersion
        Object containing all information associated with a specific model version
    """
    model_type_version = get_model_type_version(model_type, biomed_version)
    model_constants = MODEL_TYPE_2_CONSTANTS[model_type]
    if not model_constants.ensemble_version_list:
        operations_logger.warning(f"No versions found with model type {model_type}, version '{model_type_version}'")
        return None
    return model_constants.get_ensemble_version(model_type_version)


def get_ensemble_version_voting_method(model_type: ModelType, biomed_version=None) -> VotingMethodEnum:
    """
    Get the ensemble voting method enum and folder from the model_type constants
    :param model_type: ModelType enum for the target model type
    :param biomed_version: target version string; if None, use default BIOMED version
    :return: VotingMethodEnum
        The voting method enum type
    """
    return get_ensemble_version(model_type, biomed_version).voting_method


def get_ensemble_version_voter_folder(model_type: ModelType, biomed_version=None) -> str:
    """
    Get the ensemble voter folder name from the model_type constants

    :param model_type: ModelType enum for the target model type
    :param biomed_version: target version string; if None, use default BIOMED version
    :return: str
        The name of the target voting model folder within the model type directory
    """
    return get_ensemble_version(model_type, biomed_version).voter_model


class BiomedVersionInfo(VersionInfo):
    product_id = 'biomed'
    MODEL_VERSION_KEY = 'models'

    def __init__(
            self,
            task_operation: TaskOperation = None,
            biomed_version: str = None,
            model_type: ModelType = None,
            **kwargs):
        super().__init__(**kwargs)
        self.product_version = biomed_version or BiomedEnv.DEFAULT_BIOMED_VERSION.value
        self.task_operation = task_operation
        self.model_type = model_type
        if task_operation is not None:
            self.model_versions = self.task_model_version_info()
        elif model_type is not None:
            self.model_versions = {model_type.name: BIOMED_VERSION_TO_MODEL_VERSION[self.product_version][model_type]}

    def task_model_version_info(self):
        """
        :return: # json output of form:
         {
         'model_type_1': model_version
         }
        """
        biomed_version = self.product_version
        versioning_json = {}
        if self.task_operation not in OperationToModelType:
            raise ValueError(f'Task operation {self.task_operation.name} does not have a model type mapping and '
                             f'therefore there is no model version info')
        for model_type in OperationToModelType[self.task_operation]:
            versioning_json[model_type.name] = BIOMED_VERSION_TO_MODEL_VERSION[biomed_version][model_type]
        return versioning_json

    def to_dict(self):
        return {**super().to_dict(),
                'model_versions': self.model_versions}


class MatchPlausibleDemographicsEncounters(IntEnum):
    ssn = 1
    mrn = 2
    pat_first = 3
    pat_middle = 4
    pat_last = 5
    pat_street = 7
    pat_zip = 8
    pat_phone = 10
    pat_email = 11
    dr_first = 12
    dr_last = 13
    DOB = 18


DEMOGRAPHICS_TABLE_SCHEMA = """
                create table MPI
            (
              ssn varchar(500) default null,
              mrn varchar(500) default null,
              pat_first    varchar(500) default null,
              pat_middle   varchar(500) default null,
              pat_last     varchar(500) default null,
              pat_initials varchar(500) default null,
              pat_age      varchar(500) default null,
              pat_street   varchar(500) default null,
              pat_zip      varchar(500) default null,
              pat_city     varchar(500) default null,
              pat_state    varchar(500) default null,
              pat_phone    varchar(500) default null,
              pat_email    varchar(500) default null,
              insurance        varchar(500) default null,
              facility_name    varchar(500) default null,
              dr_first    varchar(500) default null,
              dr_middle   varchar(500) default null,
              dr_last     varchar(500) default null,
              dr_initials varchar(500) default null,
              dr_street   varchar(500) default null,
              dr_zip      varchar(500) default null,
              dr_city     varchar(500) default null,
              dr_state    varchar(500) default null,
              dr_phone    varchar(500) default null,
              dr_fax      varchar(500) default null,
              dr_email    varchar(500) default null,
              dr_id       varchar(500) default null,
              dr_org      varchar(500) default null,
            sex      varchar(500) default null,
            dob    varchar(500) default null,
            document_id    bigint(20) default null,
            patient_id     bigint(20) default null
             );
    alter table MPI add index(pat_first);
    alter table MPI add index(pat_last);
    alter table MPI add index(mrn);
    alter table MPI add index(ssn);
    alter table MPI add index(pat_middle);
    alter table MPI add index(pat_age);
    alter table MPI add index(pat_city);
    alter table MPI add index(pat_street);
    alter table MPI add index(pat_zip);
    alter table MPI add index(pat_email);
    alter table MPI add index(dr_first);
    alter table MPI add index(dr_last);
    alter table MPI add index(dr_phone);
    alter table MPI add index(dr_fax);
    alter table MPI add index(dr_email);
    alter table MPI add index(dob);
    """

VARS_PER_DEMOGRAPHIC = ['both_exist', 'union', 'intersection/union']

SIMILARITY_COLS = ['ssn',
                   'mrn',
                   'pat_first',
                   'pat_middle',
                   'pat_last',
                   'pat_initials',
                   'pat_age',
                   'pat_street',
                   'pat_zip',
                   'pat_city',
                   'pat_state',
                   'pat_phone',
                   'pat_email',
                   'dr_first',
                   'dr_middle',
                   'dr_last',
                   'dr_initials',
                   'dr_street',
                   'dr_zip',
                   'dr_city',
                   'dr_state',
                   'dob']

MED_BLACK_LIST = {'medication',
                  'medications',
                  'drug',
                  'drugs',
                  'generate',
                  'reconciled',
                  'frequency'}

PROBLEM_BLACK_LIST = {'moderately',
                      'degenerative',
                      'hopeless',
                      'viral',
                      'neck',
                      'extremities'}

ORDERED_CATEGORY_TIEBRAKER = {ProblemLabel.get_category_label().persistent_label: 2,
                              SignSymptomLabel.get_category_label().persistent_label: 3,
                              LabLabel.get_category_label().persistent_label: 4}

NEARBY_TERM_THRESHOLD = 5
MAX_TERM_SEPERATION_THRESHOLD = 50
SENSITIVE_DEMOGRAPHIC_STRING_CATS = {
    DemographicEncounterLabel.pat_first.value.persistent_label,
    DemographicEncounterLabel.pat_last.value.persistent_label,
    DemographicEncounterLabel.pat_middle.value.persistent_label,
    DemographicEncounterLabel.ssn.value.persistent_label,
    DemographicEncounterLabel.pat_phone.value.persistent_label,
    DemographicEncounterLabel.pat_email.value.persistent_label,
    'pat_full_name'
}
