import copy
import unittest

from biomed.constants import model_constants
from biomed.constants.model_enums import VotingMethodEnum
from biomed.constants.constants import (
    get_ensemble_version,
    get_model_type_version,
    get_ensemble_version_voting_method,
    get_ensemble_version_voter_folder,
)


class TestEnsembleVersion(unittest.TestCase):
    def test_defaults(self):
        ensemble = model_constants.EnsembleVersion(
            model_names=['foo']
        )
        self.assertEqual(ensemble.model_names, ['foo'])
        self.assertEqual(ensemble.voting_method, VotingMethodEnum.model_avg)
        # self.assertEqual(ensemble.voting_method, None)

    def test_voting_method(self):
        ensemble = model_constants.EnsembleVersion(
            version="test",
            model_names=['foo'],
            voting_method=VotingMethodEnum.threshold
        )
        self.assertEqual(ensemble.voting_method, VotingMethodEnum.threshold)


class TestModelConstants(unittest.TestCase):
    def test_empty_version_list(self):
        class SomethingConstant(model_constants.ModelConstants):
            ensemble_version_list = []
        self.assertEqual(SomethingConstant.ensemble_version_list, [])  # add assertion here

    def test_get_ensemble_version(self):
        ensemble = model_constants.EnsembleVersion(
            version="test",
            model_names=["foo", "baz"],
        )

        class SomethingConstant(model_constants.ModelConstants):
            ensemble_version_list = [copy.copy(ensemble)]
        self.assertEqual(SomethingConstant.get_ensemble_version("test"), ensemble)

        with self.assertRaises(KeyError):
            SomethingConstant.get_ensemble_version("now")

    def test_model_version_mapping(self):
        class SomethingConstant(model_constants.ModelConstants):
            ensemble_version_list = [
                model_constants.EnsembleVersion(
                    version="test",
                    model_names=["foo"],
                    voting_method=VotingMethodEnum.threshold
                ),
                model_constants.EnsembleVersion(
                    version="test2",
                    model_names=["foo", "bar"],
                    voting_method=VotingMethodEnum.threshold
                )
            ]
        expected_dict = {
            "test": ["foo"],
            "test2": ["foo", "bar"]
        }
        self.assertEqual(SomethingConstant.model_version_mapping(), expected_dict)

    def test_model_version_mapping_empty_ensemble(self):
        class SomethingEmpty(model_constants.ModelConstants):
            ensemble_version_list = []
        self.assertEqual(SomethingEmpty.model_version_mapping(), None)


class TestEnsembleVersionGetters(unittest.TestCase):
    # test the methods in biomed.constants.constants used to get EnsembleVersion content from model_constants

    def test_get_ensemble_version(self):
        model_type = model_constants.ModelType.diagnosis
        model_version = get_model_type_version(model_type)
        model_const = model_constants.MODEL_TYPE_2_CONSTANTS[model_type]
        expected_ensemble = model_const.get_ensemble_version(model_version)
        ensemble = get_ensemble_version(model_type)
        self.assertEqual(ensemble, expected_ensemble)

    def test_get_ensemble_version_voting_method(self):
        model_type = model_constants.ModelType.diagnosis
        model_version = get_model_type_version(model_type)
        model_const = model_constants.MODEL_TYPE_2_CONSTANTS[model_type]
        expected_voting_enum = model_const.get_ensemble_version(model_version).voting_method
        voting_enum = get_ensemble_version_voting_method(model_type)
        self.assertEqual(voting_enum, expected_voting_enum)

    def test_get_ensemble_version_voter_folder(self):
        model_type = model_constants.ModelType.diagnosis
        model_version = get_model_type_version(model_type)
        model_const = model_constants.MODEL_TYPE_2_CONSTANTS[model_type]
        expected_voter_name = model_const.get_ensemble_version(model_version).voter_model
        voter_name = get_ensemble_version_voter_folder(model_type)
        self.assertEqual(voter_name, expected_voter_name)


if __name__ == '__main__':
    unittest.main()
