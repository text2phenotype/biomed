import unittest

from biomed.models.model_cache import ModelMetadataCache
from biomed.models.model_metadata import ModelMetadata
from biomed.constants.constants import get_version_model_folders
from biomed.constants.model_constants import ModelType, ModelClass, DeidConstants
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.meta.voting_methods import DEFAULT_THRESHOLD
from biomed.constants.model_enums import VotingMethodEnum


class TestEnsembleModelMetadata(unittest.TestCase):
    """
    Tests for EnsembleMetadata. Most of these tests are restricted to use existing released models,
    as that is how the class is intended to be used
    """
    META_CACHE = ModelMetadataCache()

    def _get_model_metadata(self, model_folder_name: str, model_type: ModelType) -> ModelMetadata:
        """
        Testing utility method, pulls the local metadata file and returns the model metadata objcet
        :param model_folder_name: str
        :return: dict
        """
        return self.META_CACHE.model_metadata(model_type=model_type, model_folder= model_folder_name)

    def test_metadata_init_defaults(self):
        config = {
            "model_type": ModelType.demographic.value,
        }
        ensemble_metadata = EnsembleModelMetadata(**config)

        # sanity check that we get the default list of files for the given model type
        model_file_list = get_version_model_folders(config["model_type"])
        self.assertEqual(set(model_file_list), set(ensemble_metadata.model_file_list))

        # checking unioned features
        expected_features = {
            1, 14, 16, 17, 18, 29, 33, 34, 35, 36, 47, 48, 89, 102, 104, 105, 106, 107, 108, 109,
            110, 111, 112, 113, 114, 115, 116, 117, 165
        }
        self.assertEqual(expected_features, ensemble_metadata.features)

        # sanity check the defaults
        self.assertEqual(DEFAULT_THRESHOLD, ensemble_metadata.threshold)
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)
        self.assertEqual(ensemble_metadata.voting_method, ensemble_metadata._voting_method)
        self.assertEqual(
            [ModelClass.lstm_base] * len(ensemble_metadata.model_file_list),
            ensemble_metadata.model_class_list)

    def test_metadata_init_w_file_list(self):
        model_type = ModelType.demographic
        model_folder = "demographic_20201115_2d3"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_folder_name=model_folder, model_type=model_type)
        expected_features = set(metadata.features)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.lstm_base], ensemble_metadata.model_class_list)
        self.assertEqual(expected_features, ensemble_metadata.features)

        self.assertEqual(DEFAULT_THRESHOLD, ensemble_metadata.threshold)
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)
        self.assertEqual(ensemble_metadata.voting_method, ensemble_metadata._voting_method)

    def test_metadata_init_no_model_type(self):
        # check the requirement for a model_type
        config = {
            "model_file_list": ["demographic_20201115_2d3"]
        }
        with self.assertRaises(TypeError):
            _ = EnsembleModelMetadata(**config)

    def test_metadata_init_w_type_folder(self):
        model_type = ModelType.drug
        model_folder = "5592"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_type=model_type, model_folder_name=model_folder)
        expected_features = set(metadata.features)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.lstm_base], ensemble_metadata.model_class_list)
        self.assertEqual(expected_features, ensemble_metadata.features)

    def test_metadata_init_bert(self):
        model_type = ModelType.drug
        model_folder = "drug_cbert_20210107_w64"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_folder_name=model_folder, model_type=model_type)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.bert], ensemble_metadata.model_class_list)
        # bert has no features! the bert models NEED to not have features listed in the model_metadata
        self.assertEqual(set(metadata.features), ensemble_metadata.features)
        self.assertEqual(set(), ensemble_metadata.features)

    def test_metadata_init_bert_w_type_folder(self):
        # use drug type, but specify drug folder
        model_type = ModelType.drug
        model_folder = "drug_cbert_20210107_w64"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_type=model_type, model_folder_name=model_folder)
        expected_features = set(metadata.features)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.bert], ensemble_metadata.model_class_list)
        self.assertEqual(expected_features, ensemble_metadata.features)

    def test_metadata_init_covid_lab(self):
        # use covid_lab, which behaves similarly to lab
        model_type = ModelType.covid_lab
        model_folder = "phi_covid_labs_v2_2020_11_12_2"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_folder_name=model_folder, model_type=model_type)
        expected_features = set(metadata.features)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.lstm_base], ensemble_metadata.model_class_list)
        self.assertEqual(expected_features, ensemble_metadata.features)

    def test_metadata_init_covid_lab_w_prefix_folder(self):
        # add the prefix folder, make sure it is handled correctly
        model_type = ModelType.covid_lab
        model_folder = "phi_covid_labs_v2_2020_11_12_2"
        config = {
            "model_type": model_type.value,
            "model_file_list": [model_folder]
        }
        metadata = self._get_model_metadata(model_type=model_type, model_folder_name=model_folder)
        expected_features = set(metadata.features)

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual([model_folder], ensemble_metadata.model_file_list)
        self.assertEqual([ModelClass.lstm_base], ensemble_metadata.model_class_list)
        self.assertEqual(expected_features, ensemble_metadata.features)

    def test_voting_method(self):
        # test from enum
        mt = ModelType.diagnosis
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, voting_method=VotingMethodEnum.weighted_entropy).voting_method,
            VotingMethodEnum.weighted_entropy)
        metadata = EnsembleModelMetadata(
                model_type=mt,
                voting_method=VotingMethodEnum.threshold)
        self.assertEqual(metadata.voting_method, VotingMethodEnum.threshold)
        self.assertEqual(DEFAULT_THRESHOLD, metadata.threshold)
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, voting_method=VotingMethodEnum.threshold_categories).voting_method,
            VotingMethodEnum.threshold_categories)
        # test from string
        self.assertEqual(
            EnsembleModelMetadata(
                model_type=mt,
                voting_method='weighted_entropy').voting_method,
            VotingMethodEnum.weighted_entropy)

        metadata = EnsembleModelMetadata(
                model_type=mt,
                voting_method='threshold')
        self.assertEqual(metadata.voting_method, VotingMethodEnum.threshold)
        self.assertEqual(DEFAULT_THRESHOLD, metadata.threshold)
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, voting_method='threshold_categories').voting_method,
            VotingMethodEnum.threshold_categories)

        #  test from bool settings
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, threshold=False).voting_method, VotingMethodEnum.threshold)
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, threshold=True).voting_method, VotingMethodEnum.threshold)
        self.assertEqual(
            EnsembleModelMetadata(model_type=mt, threshold=True, threshold_categories=[1]).voting_method,
            VotingMethodEnum.threshold)

    def test_metadata_default_voting_method(self):
        # test default, should have set threshold, empty threshold categories
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["foo"],
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(DEFAULT_THRESHOLD, ensemble_metadata.threshold)
        self.assertEqual(None, ensemble_metadata.threshold_categories)
        # expect threshold
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)

        # test set threshold
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["foo"],
            "threshold": 0.1,
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(0.1, ensemble_metadata.threshold)
        self.assertEqual(None, ensemble_metadata.threshold_categories)
        # expect threshold
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)

        # test set threshold
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["foo"],
            "threshold": 0.1,
            "threshold_categories": DeidConstants.rare_varying_classes
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(0.1, ensemble_metadata.threshold)
        self.assertEqual(DeidConstants.rare_varying_classes, ensemble_metadata.threshold_categories)
        # expect threshold, even if threshold_categories is set
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)

        # test no threshold
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["foo"],
            "threshold": None,
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(None, ensemble_metadata.threshold)
        self.assertEqual(None, ensemble_metadata.threshold_categories)
        # expect weighted_entropy
        self.assertEqual(VotingMethodEnum.threshold, ensemble_metadata.voting_method)

    def test_metadata_voting_method_str(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "voting_method": "weighted_entropy",
        }

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(DEFAULT_THRESHOLD, ensemble_metadata.threshold)
        self.assertEqual(VotingMethodEnum.weighted_entropy, ensemble_metadata.voting_method)
        self.assertEqual(ensemble_metadata.voting_method, ensemble_metadata._voting_method)

    def test_metadata_voting_method_enum(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "threshold": 0.99,
            "voting_method": VotingMethodEnum.weighted_entropy,
        }

        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(0.99, ensemble_metadata.threshold)
        self.assertEqual(VotingMethodEnum.weighted_entropy, ensemble_metadata.voting_method)
        self.assertEqual(ensemble_metadata.voting_method, ensemble_metadata._voting_method)

    def test_metadata_model_weights(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "threshold": 0.99,
            "voting_method": VotingMethodEnum.model_weighted_avg,
            "model_weights": None
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(VotingMethodEnum.model_weighted_avg, ensemble_metadata.voting_method)
        self.assertEqual(None, ensemble_metadata.model_weights)

        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "threshold": 0.99,
            "voting_method": VotingMethodEnum.model_weighted_avg,
            "model_weights": [0.2, 0.3, 0.4999999999]
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(VotingMethodEnum.model_weighted_avg, ensemble_metadata.voting_method)
        self.assertEqual(config["model_weights"], ensemble_metadata.model_weights)

    def test_metadata_model_weights_bad(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "threshold": 0.99,
            "voting_method": VotingMethodEnum.model_weighted_avg,
            "model_weights": [1, 1, 1]
        }
        with self.assertRaises(ValueError):
            _ = EnsembleModelMetadata(**config)

    def test_metadata_voting_model_folder(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "voting_method": VotingMethodEnum.rf_classifier,
            "voting_model_folder": "foo"
        }
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(VotingMethodEnum.rf_classifier, ensemble_metadata.voting_method)
        self.assertEqual(config["voting_model_folder"], ensemble_metadata.voting_model_folder)

    def test_metadata_voting_model_folder_from_constant(self):
        expected_voter = "voter_rf_diagnosis_20210611"
        config = {
            "model_type": ModelType.diagnosis,
            "model_file_list": ["test_model_folder"],
            "voting_method": VotingMethodEnum.rf_classifier,
        }
        with self.assertRaises(ValueError):
            # no voting_model_folder
            _ = EnsembleModelMetadata(**config)

        config["voting_model_folder"] = expected_voter
        ensemble_metadata = EnsembleModelMetadata(**config)
        self.assertEqual(VotingMethodEnum.rf_classifier, ensemble_metadata.voting_method)
        self.assertEqual(expected_voter, ensemble_metadata.voting_model_folder)

    def test_metadata_voting_model_folder_bad_biomed_version(self):
        config = {
            "model_type": ModelType.diagnosis,
            "model_file_list": ["model_folder"],
            "voting_method": VotingMethodEnum.rf_classifier,
            "biomed_version": "99.99"
        }
        with self.assertRaises(ValueError):
            # error comes from get_model_type_version(), since we only have one possible version rn
            _ = EnsembleModelMetadata(**config)

    def test_metadata_voting_model_folder_no_model_for_type(self):
        config = {
            "model_type": ModelType.drug,
            "model_file_list": ["model_folder"],
            "voting_method": VotingMethodEnum.rf_classifier,
        }
        with self.assertRaises(ValueError):
            _ = EnsembleModelMetadata(**config)


if __name__ == '__main__':
    unittest.main()
