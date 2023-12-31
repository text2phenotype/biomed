import os
from typing import Union

from biomed.models.model_cache import ModelMetadataCache, get_model_folder_path
from text2phenotype.common import common

from biomed.constants.model_enums import ModelType, ModelClass
from biomed.models.model_base import ModelBase
from biomed.meta.meta_lstm import MetaLSTM
from biomed.models.bert_base import BertBase
from biomed.document_section.model import DocumentTypeModel

# if you need to use a model class other than these, add a new model_class enum and the mapping to this file
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import deserialize_enum

MODEL_CLASS_2_ENUM_CLASS = {
    ModelClass.lstm_base: ModelBase,
    ModelClass.doc_type: DocumentTypeModel,
    ModelClass.bert: BertBase,
    ModelClass.meta: MetaLSTM
}


def model_folder_exists(model_folder_name, model_type: ModelType = None) -> bool:
    """
    Check if model folder exists in LOCAL_FILES
    :param model_folder_name: string (for examples see ModelConstants)
    :param model_type: the default model type, dictates default folder location
    :return: bool
    """
    return os.path.isdir(get_model_folder_path(model_folder_name, model_type))


def get_model_metadata_fp_from_model_folder(model_folder: str, model_type: ModelType = None):
    """
    :return: location of model metadata file,  assumes model metadata is the only
    .metadata.json file stored in  the model folder
    """
    return common.get_file_list(get_model_folder_path(model_folder, model_type=model_type), '.metadata.json', True)[0]


def get_model_filename_from_model_folder(model_folder: str, model_type: ModelType = None, suffix: str = None):
    """
    Get the full file path of the target model from the model_folder

    :param model_folder: folder name containing the model for the given model_type
    :param model_type: the target model_type
    :param suffix: the filename suffix to filter by, eg .h5 or .joblib
    :return: location of model file,  assumes model file is the only one with given suffix in folder
    """
    suffix = suffix or ".h5"
    return common.get_file_list(get_model_folder_path(model_folder, model_type=model_type), suffix, True)[0]


def get_model_type_from_model_folder(model_folder: str, model_type: ModelType = None) -> Union[ModelType, None]:
    """
    :return: The model type stated in the model metadata
    """

    metadata_fp = get_model_metadata_fp_from_model_folder(model_folder=model_folder, model_type=model_type)
    metadata_asserted_model_type = common.read_json(metadata_fp).get('model_type')
    if not metadata_asserted_model_type:
        raise ValueError(f"No Metadata Asserted Model Type found for model folder: {model_folder}")
    metadata_asserted_model_type = deserialize_enum(metadata_asserted_model_type, ModelType)
    return metadata_asserted_model_type


def get_model_from_model_folder(model_folder: str, base_model_type: ModelType = None):
    """
    :return: initialize a model purely from the  model folder name and the base expected model type
    """
    metadata = ModelMetadataCache().model_metadata(model_type=base_model_type, model_folder=model_folder)
    model_class = MODEL_CLASS_2_ENUM_CLASS[metadata.model_class]
    _, base_model_folder = os.path.split(model_folder)
    return model_class(model_type=base_model_type, model_folder_name=base_model_folder)
