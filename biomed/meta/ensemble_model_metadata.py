import os
import math
from typing import List, Union

from biomed.models.model_cache import ModelMetadataCache
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import deserialize_enum

from biomed import RESULTS_PATH
from biomed.constants.constants import get_version_model_folders, get_ensemble_version_voter_folder
from biomed.models.model_metadata import ModelMetadata
from biomed.constants.model_enums import ModelType, VotingMethodEnum
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from biomed.meta.voting_methods import DEFAULT_THRESHOLD, is_trained_voter


class EnsembleModelMetadata:
    def __init__(
            self,
            model_type: ModelType,
            model_file_list: List[str] = None,
            threshold: float = DEFAULT_THRESHOLD,
            threshold_categories: List[int] = None,
            voting_method: Union[str, VotingMethodEnum] = None,
            model_weights: List[float] = None,
            voting_model_folder: str = None,
            **kwargs
    ):
        """
        Init EnsembleModelMetadata
        This does some parsing of the different parameters associated with an Ensemble voting method

        :param model_type: ModelType
            The overall modeltype used by the ensemble; combines submodels under this base type
            as long as each submodel has the same output label categories
        :param model_file_list: list of model folder names to use in the ensemble.
            If undefined, loads the default models associated with the ModelType version and current
            BIOMED release version
            eg: ["5532", "diagnosis_bert/diagnosis_cbert_20210119_w64", "diagnosis_bert/phi_diagnosis_bert_la_v1"]
                for `diagnosis`
        :param threshold: float, used in threshold and threshold_category voting
            See biomed.meta.voting_methods for more information
        :param threshold_categories: list, the integer categories used for score thresholding
            See biomed.meta.voting_methods for more information
        :param voting_method: Union[str, VotingMethodEnum], which method to use for voting
            If not specified, will use the default voting method (see voting_method.setter for more info)
        :param model_weights: List[float]
            A list of model weights to be used for the model_weight_avg voting method
            There should be one coefficient for each model in the ensemble. The weights can be
            selected manually, or identified through optimization to tune the ensemble towards
            a specific target dataset.
            These weights must sum to 1 (aka be vector normed)
        :param voting_model_folder: str, the target folder name inside the model_type that contains
            a trained model for use in ensemble prediction voting.
            TODO: should this be just the folder name, and search for the first .joblib file that matches?
        :param kwargs: dict, passthrough for downstream parameters
        """
        self.model_type: ModelType = deserialize_enum(model_type, ModelType)
        self.model_metadata_cache = ModelMetadataCache()
        if not model_file_list and not kwargs.get('train'):
            model_file_list = get_version_model_folders(model_type)
        operations_logger.debug(f'Model file list: {model_file_list}')

        self.model_file_list = model_file_list
        self.threshold = threshold
        self.threshold_categories = threshold_categories

        self._voting_method = None
        self.voting_method = voting_method

        # list of model weights, must sum to 1
        # TODO: or, we automatically normalize them?
        self.model_weights = model_weights
        if self.model_weights and not math.isclose(sum(self.model_weights), 1):
            raise ValueError(f"Model weights must sum to 1, got {sum(self.model_weights)}")

        # set the voting model name if using a model-based voting_method
        # default to the name passed in first, then test for the versioned voter
        if not voting_model_folder and is_trained_voter(self.voting_method):
            biomed_version = kwargs.get("biomed_version")
            voting_model_folder = get_ensemble_version_voter_folder(model_type, biomed_version=biomed_version)
            operations_logger.info(f"Using voting_model_folder={voting_model_folder}")
            # voting_model_folder will be None if no version is listed
            if not voting_model_folder:
                raise ValueError(
                    "Must specify the `voting_model_folder` when using "
                    f"`{self.voting_method}` with model_type `{model_type}`")
        self.voting_model_folder = voting_model_folder

        self.__features = None
        self.__model_classes = None

    @property
    def voting_method(self) -> VotingMethodEnum:
        return self._voting_method

    @voting_method.setter
    def voting_method(self, value):
        # TODO(mjp): make this default cleaner!
        # TODO(mjp): given the ensemble type, we could default to the "newest" voting method in the type constant
        if value is None:
            # default to threshold, if no method is specified
            self._voting_method = VotingMethodEnum.threshold
        elif isinstance(value, VotingMethodEnum):
            # if a voting method enum has been passed use that
            self._voting_method = value
        elif isinstance(value, str):
            # get here in TrainTestJobs
            members = [mem for mem in VotingMethodEnum.__members__]
            operations_logger.info(
                f"Loading  voting method from str: '{value}', "
                f"available voting methods are {members}")
            self._voting_method = VotingMethodEnum[value]
        else:
            raise ValueError(f"Invalid voting method passed in, got {value}, expected {VotingMethodEnum.__members__}")

    def __repr__(self):
        """Display the key attributes in the object repr"""
        voting_model_folder_str = f"voting_model_folder={self.voting_model_folder}, " \
            if self.voting_model_folder else ""
        return (
            f"{self.__class__.__name__}(" +
            f"model_type={self.model_type}, " +
            f"model_file_list={self.model_file_list}, " +
            f"voting_method={self.voting_method}, " +
            f"threshold={self.threshold}, " +
            voting_model_folder_str + ")"
        )

    def save(self, job_id: str):
        path = os.path.join(RESULTS_PATH, job_id, 'ensemble_metadata.json')
        return common.write_json(self.to_json(), path)

    def get_model_metadata(self, model_folder: str) -> ModelMetadata:
        """
        Load the model metadata file from `LOCAL_FILES`
        :param model_folder: str
            Target model folder name, generally defaults to given model type as folder
            Eg, "phi_drug_20210101".
            Can include the model type, eg `drug_bert/test_drug_bert`.
            If model_type prefix isn't included, this assumes the same type as the Ensemble, self.model_type
        :return: dict
            content from the model_metadata json file
        :raises: ValueError
            if cant find model folder locally raise error caused by one of
             - model hasnt been synced
             - the folder was prefixed incorrectly,
             - no .h5 file in the folder
        """
        model_type_folder, model_name = os.path.split(model_folder)
        if model_type_folder:
            # user specifies specific model folder
            model_metadata = self.model_metadata_cache.model_metadata(
                model_type=model_type_folder, model_folder=model_name)
        else:
            model_metadata = self.model_metadata_cache.model_metadata(
                model_type=self.model_type,
                model_folder=model_name)

        return model_metadata

    @property
    def features(self):
        if self.__features:
            return self.__features
        else:
            all_features = set()
            for model_folder in self.model_file_list:
                metadata = self.get_model_metadata(model_folder)
                features = metadata.features
                if not features or features == [None]:
                    continue
                all_features.update(set(features))

            self.__features = all_features
        return all_features

    @property
    def model_class_list(self):
        """Get the model class associated with the self.model_file_list, in same order"""
        if not self.__model_classes:
            model_classes = []
            for model_folder in self.model_file_list:
                metadata = self.get_model_metadata(model_folder)
                # default to lstm if it isnt listed in the model config
                model_class_enum = metadata.model_class or MODEL_TYPE_2_CONSTANTS[metadata.model_type].model_class_enum

                model_classes.append(model_class_enum)
            self.__model_classes = model_classes
        return self.__model_classes

    def from_json(self, **kwargs):
        for key, val in kwargs.items():
            try:
                self.__setattr__(key, val)
            except AttributeError:
                pass

    def to_json(self) -> dict:
        """
        :return: dictionary of model params
        """
        # NOTE: this does not include the @property parameters, only the ones defined in the dict
        return self.__dict__.copy()

    def to_escaped_string(self):
        converted_string = common.to_escaped_string(self.to_json())
        return converted_string
