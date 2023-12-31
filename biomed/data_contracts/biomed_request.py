from typing import Dict, List, Optional

from text2phenotype.open_api.base_model import Model

from biomed.constants.constants import ModelType


class BiomedRequest(Model):
    def __init__(self,
                 text: str = None,
                 data: object = None,
                 models: Dict[ModelType, List[str]] = None,
                 tid: str = None):
        """
        :param data: The data of this request.
        :param models: The models to use
        :param tid: The transaction id
        """
        self.swagger_types = {
            'text': str,
            'data': object,
            'models': Dict[ModelType, List[str]],
            'tid': str
        }

        self.attribute_map = {
            'text': 'text',
            'data': 'data',
            'models': 'models',
            'tid': 'tid'
        }

        self._text = text
        self._data = data
        self._models = self.convert_enum_dict_keys(models, ModelType)
        self._tid = tid

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, text):
        self._text = text

    @property
    def data(self):
        return self._data if self._data else {}

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def models(self) -> Optional[Dict[ModelType, List[str]]]:
        return self._models

    @models.setter
    def models(self, models: Dict[ModelType, List[str]]):
        self._models = self.convert_enum_dict_keys(models, ModelType)

    @property
    def tid(self) -> str:
        return self._tid

    @tid.setter
    def tid(self, tid: str):
        self._tid = tid
