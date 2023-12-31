import os
from text2phenotype.constants.environment import Environment, EnvironmentVariable


class BiomedEnv(Environment):
    APPLICATION_NAME = Environment.APPLICATION_NAME
    APPLICATION_NAME.value = 'Biomed Service'

    SVC_HOST = EnvironmentVariable(name='MDL_BIOM_SVC_HOST', legacy_name='BIOMED_HOST', value='0.0.0.0')
    SVC_PORT = EnvironmentVariable(name='MDL_BIOM_SVC_PORT', legacy_name='BIOMED_PORT', value=8080)
    DATA_ROOT = EnvironmentVariable(name='MDL_BIOM_DATA_ROOT', legacy_name='DATA_ROOT', value='/mnt/S3')
    PRELOAD = EnvironmentVariable(name='MDL_BIOM_PRELOAD', legacy_name='BIOMED_PRELOAD', value=False)
    USE_STORAGE_SERVICE = EnvironmentVariable(name='MDL_BIOM_USE_STORAGE_SVC',
                                              legacy_name='USE_STORAGE_SERVICE',
                                              value=True)
    DB_URL_DEID = EnvironmentVariable(name='MDL_BIOM_DB_URL_DEID',
                                      legacy_name='DB_URL_DEID',
                                      value='mysql://text2phenotype:text2phenotype@localhost/DEID')
    DB_URL_PHI = EnvironmentVariable(name='MDL_BIOM_DB_URL_PHI',
                                     legacy_name='DB_URL_PHI',
                                     value='mysql://text2phenotype:text2phenotype@localhost/PHI')
    BIOMED_USE_PREDICT_GENERATOR = EnvironmentVariable(name='MDL_BIOM_USE_PREDICT_GENERATOR', value=False)

    HEPC_FORM_FILTER = EnvironmentVariable(name='MDL_BIOM_HEPC_FORM_FILTER', legacy_name='HEPC_FORM_FILTER', value=2)
    PATIENT_MATCHING_ENABLED = EnvironmentVariable(name='MDL_BIOM_PATIENT_MATCHING_ENABLED',
                                                   legacy_name='PATIENT_MATCHING_ENABLED',
                                                   value=True)
    DB_USERNAME = EnvironmentVariable(name='MDL_SANDS_DB_USERNAME', legacy_name='DB_USERNAME')
    DB_PASSWORD = EnvironmentVariable(name='MDL_SANDS_DB_PASSWORD', legacy_name='DB_PASSWORD')
    DEMOGRAPHIC_TABLE = EnvironmentVariable(name='MDL_SANDS_DEMOGRAPHIC_TABLE', legacy_name='DEMOGRAPHIC_TABLE',
                                            value='app_documentdemographics')

    # APM
    APM_SERVICE_NAME = EnvironmentVariable(name='MDL_BIOM_APM_SERVICE_NAME',
                                           legacy_name='APM_SERVICE_NAME',
                                           value='Text2phenotype Biomed Service')
    MAX_THREAD_COUNT = EnvironmentVariable(name='MDL_BIOM_MAX_THREAD_COUNT', value=5)

    MODEL_PREDICTION_WORKERS = EnvironmentVariable(name='MDL_BIOM_MODEL_PREDICTION_WORKERS', value=1)
    # determines whether keras.predict_generator should use multiprocessing
    MODEL_USE_MULTIPROCESSING = EnvironmentVariable(name='MDL_BIOM_MODEL_USE_MULTIPROCESSING', value=False)
    # determines how many keras sequences are loaded at once into predict generator
    MODEL_SEQUENCE_MAX_QUEUE = EnvironmentVariable(name='MDL_BIOM_MODEL_SEQUENCE_MAX_QUEUE', value=2)

    INCLUDE_SMOKING_IN_CLINICAL_SUMMARY = EnvironmentVariable(name='MDL_BIOM_SMOKING_CLINICAL', value=False)
    INCLUDE_VITAL_SIGNS_IN_CLINICAL_SUMMARY = EnvironmentVariable(name='MDL_BIOM_VITAL_SIGNS_CLINICAL', value=False)

    DEFAULT_BIOMED_VERSION = EnvironmentVariable(name='MDL_BIOM_DEFAULT_MODEL_VERSION', value='2.11')

    MAX_LAB_DISTANCE = EnvironmentVariable(name='MDL_BIOM_MAX_LAB_ASSOC_CHAR_DISTANCE', value=40)

    BIOM_MODELS_PATH = EnvironmentVariable(name='MDL_BIOM_MODELS_PATH',
                                           value=os.path.join(os.path.dirname(__file__)))
    BIOMED_NON_SHARED_MODEL_PATH = EnvironmentVariable(
        name='MDL_BIOM_LOCAL_MODELS_PATH', value='/tmp')
