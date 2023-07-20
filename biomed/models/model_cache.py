import os
from typing import Dict, Union

from sklearn.externals import joblib

from biomed.models.model_metadata import ModelMetadata
from text2phenotype.common.log import operations_logger
from text2phenotype.common.singleton import SingletonCache, singleton
from text2phenotype.common import common

from biomed.biomed_env import BiomedEnv
from biomed.constants.constants import get_version_model_folders
from biomed.constants.model_enums import ModelType
from biomed.models.model_wrapper import ModelWrapper, ModelWrapperSession
from biomed.resources import CUI_RULE, TUI_RULE, HEADING_FILE, LOCAL_FILES


@singleton
class ModelCache(SingletonCache):
    def __init__(self, preload=False):
        super().__init__()
        if preload:
            self.preload()

    def model_keras(
            self,
            model_name: str,
            model_type: ModelType,
            model_file_path: str = None,
            use_tf_session: bool = False,
    ) -> ModelWrapper:
        """
        Load the keras model in a singleton cache
        :param model_name: str, the job_id and filename for the model
            often comes from stored ModelMetadata
        :param model_type: ModelType, enum of model type
        :param model_file_path: Optional[str]
            often comes from stored ModelMetadata
        :param use_tf_session: Whether or not to use the TF1 constructs of Graph and Session
            in the ModelWrapper
        :return: cached instance to ModelWrapper
        """
        full_model_path = get_full_path(model_type.name, model_name, model_file_path)
        if not self.exists(full_model_path):
            operations_logger.info(f'Caching model: {full_model_path}')
            if use_tf_session:
                operations_logger.info(f"Using ModelWrapperSession ({os.path.dirname(full_model_path)}")
                model_wrapper = ModelWrapperSession(full_model_path)
            else:
                model_wrapper = ModelWrapper(full_model_path)
            self.put(full_model_path, model_wrapper)
        else:
            operations_logger.debug(f"loading model file cached at:  {full_model_path}")
        return self.get(full_model_path)

    def model_sklearn(self, model_type: ModelType, model_folder: str, model_file_path: str = None):
        model_file = get_full_path(model_type.name, model_folder, model_file_path)
        if not self.exists(model_file):
            operations_logger.info(f'Caching sklearn model, file={model_file}')
            model = joblib.load(model_file)
            self.put(model_file, model)
        else:
            operations_logger.debug(f"loading model file cached at:  {model_file}")
        return self.get(model_file)

    def cui_rule(self) -> Dict[str, dict]:
        if not self.exists(CUI_RULE):
            self.put(CUI_RULE, common.read_json(CUI_RULE))
        return self.get(CUI_RULE)

    def tui_rule(self) -> Dict[str, dict]:
        if not self.exists(TUI_RULE):
            self.put(TUI_RULE, common.read_json(TUI_RULE))
        return self.get(TUI_RULE)

    def preload(self):
        operations_logger.info('Preloading models...')
        for model_type in ModelType:
            for model_file in get_version_model_folders(model_type):
                self.model_keras(model_file, model_type)

        operations_logger.info('Preloading models done')

    def header_file(self) -> Dict[str, dict]:
        if not self.exists(HEADING_FILE):
            self.put(HEADING_FILE, common.read_json(HEADING_FILE))
        return self.get(HEADING_FILE)


@singleton
class ModelMetadataCache(SingletonCache):
    def model_metadata(
            self,
            model_type: Union[ModelType, str],
            model_folder: str) -> ModelMetadata:
        # normalizing model type name
        if isinstance(model_type, ModelType):
            model_type_name: str = model_type.name
        elif isinstance(model_type, str):
            model_type_name: str = model_type
        else:
            raise TypeError(f'Model Type {model_type} must be an instance of ModelType or a string')

        metadata_folder = os.path.join(model_type_name, model_folder)

        if not self.exists(metadata_folder):
            operations_logger.info(f'Caching model metadata: {metadata_folder}')
            full_metadata_path = find_model_file(model_type_name, model_folder, 'h5.metadata.json')
            metadata_json = ModelMetadata(**common.read_json(full_metadata_path))
            self.put(metadata_folder, metadata_json)

        return self.get(metadata_folder)


def get_model_folder_path(model_folder_name, model_type: ModelType = None, model_type_name: str = None) -> str:
    """
    :param model_folder_name: string (for examples see ModelConstants)
    :param model_type: the model type as a enum;
        require either this or model_type_name
    :param model_type_name: stringified model type, points to model_type folder name;
        require either this or model_type
    :return: string with the absolute path to the target model folder from the model_type
    """
    assert model_type is not None or model_type_name is not None, \
        "Don't have a target model type to load, pass in either ModelType enum or model_type_name"
    if model_type and isinstance(model_type, ModelType):
        model_type_folder_name = model_type.name
    else:
        model_type_folder_name = model_type_name
    model_folder_path = os.path.join(LOCAL_FILES, model_type_folder_name, model_folder_name)

    if not os.path.isdir(model_folder_path):
        operations_logger.info(f"No model folder path found at {model_folder_path}, checking non shared dir")
        model_folder_path = os.path.join(
            BiomedEnv.BIOMED_NON_SHARED_MODEL_PATH.value, 'resources/files',
            model_type_folder_name, model_folder_name)
        if not os.path.isdir(model_folder_path):
            operations_logger.info(f"No model folder path found at {model_folder_path}")

    return model_folder_path


def find_model_file(model_type_name: str, model_folder_name: str, suffix: str):
    folder_path = get_model_folder_path(model_folder_name=model_folder_name, model_type_name=model_type_name)
    matching_file_names = common.get_file_list(folder_path, suffix)
    if len(matching_file_names) == 0:
        raise ValueError(
            f'There is  no model file associated with folder {model_folder_name} on path {folder_path}')
    elif len(matching_file_names) > 1:
        operations_logger.warning(f'Multiple matching files found: {matching_file_names}, "'
                                  f'"returning first Model File Name, {matching_file_names[0]}')
    return matching_file_names[0]


def get_full_path(model_type_name: str, model_file_name: str, model_file_path: str = None):
    """

    :param model_type_name:
        string for the model type, should point to a valid folder in biomed/resources/files
    :param model_file_name:
        Generally is formatted {model_folder_name}/{versioned_name}.h5
    :param model_file_path:
        For lstm models, expects model_file_path to have date_versioned.h5 model name on it
        For bert models, expects model_file_path to NOT have tf_model.h5 on it

    :return:
        Full absolute path to the model file
        eg: `/sss/biomed/biomed/resources/files/demographic/phi_demographic_2021_03_10_a/tf_model.h5`
    """
    if (
        not model_file_path
        or (not os.path.isfile(os.path.join(model_file_path, "tf_model.h5"))
            and not os.path.isfile(model_file_path))
    ):
        # if we dont have an explicit filepath to the .h5 file (or .pb file), load the file from the `models` folder
        model_file_path = os.path.join(
            get_model_folder_path(model_type_name=model_type_name, model_folder_name=os.path.dirname(model_file_name)),
            os.path.basename(model_file_name))

    # retest the new path and include the bert tf_model.h5 filename if needed
    return model_file_path
