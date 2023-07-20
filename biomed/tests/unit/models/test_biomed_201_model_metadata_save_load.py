import unittest
from biomed.models.model_metadata import ModelMetadata
from biomed.constants.constants import ModelType


class TestBiomed201(unittest.TestCase):

    def test_model_metadata_serialization(self):
        window_size = 20

        expected = ModelMetadata(ModelType.deid, window_size=window_size)

        actual = ModelMetadata()
        actual.from_json(**expected.to_json())

        self.assertEqual(window_size, actual.window_size)
