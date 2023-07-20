import json
import os
from typing import List, Optional, Set

from biomed.constants.model_constants import ModelClass
from text2phenotype.common import common
from text2phenotype.common.common import version_text
from text2phenotype.constants.common import deserialize_enum
from text2phenotype.constants.features import FeatureType

from biomed import RESULTS_PATH
from biomed.constants.constants import ModelType


class ModelMetadata:
    def __init__(self,
                 model_type: ModelType = None,
                 model_file_name: str = None,
                 window_size: int = 30,
                 window_stride: int = 1,
                 features: Set[FeatureType] = None,
                 file_ext: str = '.h5',
                 job_id: str = None,
                 model_file_path: str = None,
                 meta_classifier: bool = False,
                 base_classifier_list: List[str] = None,
                 binary_classifier: bool = False,
                 previous_model_file_name: str = None,
                 embedding_model_name: str = None,
                 model_class: str = None,
                 **kwargs):
        """
        Store the parameters used in model creation, training, and prediction

        # TODO: turn this into a dataclass?

        :param model_type: model type
        :param model_file_name: name of the model file
        :param features: List/set of features to use
            If gets None or the empty list, it will return a set of all possible features
            If gets List containing None, it will return a set containing None. This is
                used for BERT models, where no features are being used for input
        :param window_size: tokens in sequence
        :param window_stride: number of tokens to shift the window by
        :param meta_classifier: whether or not this is a meta-classifier, if it is, need to supply a list of unit
                in the correct sequence
        :param base_classifier_list: a list of unit model file name
        :param embedding_model_name: a string that matches one of the names in BertEmbeddings
        """
        self.model_type: ModelType = model_type
        self.model_file_name: str = model_file_name
        self.window_size: int = window_size
        self.window_stride: int = window_stride
        self.file_ext: str = file_ext
        # We want set of given features, otherwise have empty set;
        # no longer returns all features, since they were never all valid
        self.features: Set[FeatureType] = set(features) if features else set()
        self.job_id: str = job_id
        self._model_file_path = model_file_path
        self.meta_classifier = meta_classifier
        self.base_classifier_list = base_classifier_list
        self.model_metadata = binary_classifier
        self.previous_model_file_name = previous_model_file_name
        self.binary_classifier = binary_classifier
        self.embedding_model_name = embedding_model_name
        self.model_class: ModelClass = model_class

    def __repr__(self):
        """Display the attributes in the object repr"""
        return (
            f"{self.__class__.__name__}(" +
            ", ".join([
                f"{name}={val}"
                for name, val in self.__dict__.items()
            ]) +
            f"model_file_name={self.model_file_name}, " +
            f"model_file_path={self.model_file_path}" +
            ")"
        )

    @property
    def model_type(self) -> Optional[ModelType]:
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = deserialize_enum(value, ModelType)

    @property
    def model_class(self):
        return self._model_class

    @model_class.setter
    def model_class(self, value):
        # default model class to be lstm base
        if not value:
            value = ModelClass.lstm_base
        self._model_class = deserialize_enum(value, ModelClass)

    @property
    def features(self) -> Set[FeatureType]:
        return self._features

    @property
    def previous_model_file_name(self):
        return self._previous_model_file_name

    @previous_model_file_name.setter
    def previous_model_file_name(self, value):
        self._previous_model_file_name = value

    @features.setter
    def features(self, value):
        if value:
            enum_values = set()
            for feature in value:
                enum_values.add(deserialize_enum(feature, FeatureType))
            self._features = enum_values
        else:
            self._features = value

    @property
    def model_file_path(self) -> str:
        if not self._model_file_path:
            self._model_file_path = os.path.join(RESULTS_PATH, self.model_file_name)
        return self._model_file_path

    @model_file_path.setter
    def model_file_path(self, value: str):
        self._model_file_path = value

    @property
    def model_file_name(self) -> str:
        """
        The full filename, relative to the model_type path. Eg "my_train_jobid/trained_model.h5"
        """
        if not self._model_file_name:
            ver = version_text(self.model_type.name)
            self._model_file_name = f'{self.job_id}/{ver + self.file_ext}'
        return self._model_file_name

    @model_file_name.setter
    def model_file_name(self, value):
        self._model_file_name = value

    def update_model_file_name(self, job_id):
        ver = version_text(self.model_type.name)
        self._model_file_name = f'{job_id}/{ver + self.file_ext}'

    def update_model_file_path(self):
        self._model_file_path = os.path.join(RESULTS_PATH, self.model_file_name)

    def save(self) -> str:
        """
        :return: model_metadata file path the file was saved to
        """
        if not os.path.exists(os.path.dirname(self.model_file_path)):
            os.makedirs(os.path.dirname(self.model_file_path), exist_ok=True)
        return common.write_json(self.to_json(), f'{self.model_file_path}.metadata.json')

    def save_test(self):
        # NOTE(mjp): why is this here?
        path = os.path.join(RESULTS_PATH, self.job_id, 'model_metadata.json')
        return common.write_json(self.to_json(), path)

    def load(self, file_path: str = None):
        """:param file_path: path to model_metadata file"""
        saved = common.read_json(file_path)
        self.from_json(**saved)
        self._model_file_path = file_path.replace('.metadata.json', '')

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
        result = {'model_type': self.model_type,
                  'model_file_name': self.model_file_name,
                  'window_size': self.window_size,
                  'window_stride': self.window_stride,
                  'file_ext': self.file_ext,
                  'features': [feat.value for feat in self.features] if None not in self.features else [],
                  'job_id': self.job_id,
                  'embedding_model_name': self.embedding_model_name,
                  'model_file_path': self.model_file_path,
                  'meta_classifier': self.meta_classifier,
                  'base_classifier_list': self.base_classifier_list,
                  'previous_model_file_name': self.previous_model_file_name,
                  'model_class': self.model_class
                  }
        return result

    def to_escaped_string(self):
        converted_string = json.dumps(self.to_json()).replace('"', r'\"')
        converted_string = f'"{converted_string}"'
        return converted_string
