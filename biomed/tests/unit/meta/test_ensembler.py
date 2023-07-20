import os
import unittest

from text2phenotype.common.data_source import DataSource
from text2phenotype.apiclients.feature_service import FeatureServiceClient

from biomed.models.model_cache import ModelMetadataCache
from biomed.meta.ensembler import Ensembler
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.constants.constants import get_version_model_folders
from biomed.constants.model_constants import ModelType, MODEL_TYPE_2_CONSTANTS
from biomed.constants.model_enums import VotingMethodEnum


class TestEnsembler(unittest.TestCase):

    BASIC_CONFIG = {
        "model_type": ModelType.demographic.value,
        "test_ensemble": True,
        "train": False,
        "test": False,
        "job_id": "test_basic",
        "max_token_count": 100000,

        "feature_set_version": "dev/20210105",
        "original_raw_text_dirs": [
            "I2B2",
            "mimic_shareeclef/DISCHARGE_SUMMARY"
        ],
        "ann_dirs": [
            "tag_tog_annotations/diagnosis_signsymptom_validation/2021-01-05/CEPuser"
        ],
        "feature_set_subfolders": [
            "train/tag_tog_text"
        ],
        "testing_fs_subfolders": [
            "test/tag_tog_text"
        ]
    }
    MODEL_META_CACHE = ModelMetadataCache()

    def _get_model_metadata(self, model_type: ModelType, model_folder: str):
        """
        Testing utility method, pulls the local metadata file and returns a dict
        :param model_folder_path: str
        :return: dict
        """

        return self.MODEL_META_CACHE.model_metadata(model_type=model_type, model_folder=model_folder)

    def test_init(self):
        config = self.BASIC_CONFIG.copy()
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)

        # sanity check that we get the default list of files for the given model type
        model_file_list = get_version_model_folders(config["model_type"])
        self.assertEqual(set(model_file_list), set(ensemble_metadata.model_file_list))

        ensembler = Ensembler(
            ensemble_metadata=ensemble_metadata,
            data_source=data_source,
            job_metadata=job_metadata)

        self.assertEqual(set(model_file_list), set(ensembler.ensemble_metadata.model_file_list))
        self.assertEqual(30, ensembler.max_window_size)
        self.assertEqual(30, ensembler.min_window_size)
        self.assertIsInstance(ensembler.feature_service_client, FeatureServiceClient)
        self.assertEqual(len(model_file_list), len(ensembler.model_list))

        self.assertEqual(self.BASIC_CONFIG["model_type"], ensembler.model_metadata.model_type)

    def test_init_from_kwargs(self):
        # We can load the Ensembler from the metadata, or we can use the kwargs
        # this tests the kwargs entrance, which is often used in the token prediction functions
        config = self.BASIC_CONFIG.copy()
        model_file_list = get_version_model_folders(config["model_type"])
        with self.assertRaises(ValueError):
            # Ensembler should require a 'voting_model_folder' for a trained voter
            _ = Ensembler(
                model_type=ModelType.demographic,
                model_file_list=model_file_list,
                voting_method=VotingMethodEnum.rf_classifier,
            )

        ensembler = Ensembler(
            model_type=ModelType.demographic,
            model_file_list=model_file_list,
            voting_method=VotingMethodEnum.rf_classifier,
            voting_model_folder="voter_foo",
        )
        self.assertEqual(set(model_file_list), set(ensembler.ensemble_metadata.model_file_list))
        self.assertEqual(len(model_file_list), len(ensembler.model_list))
        self.assertEqual(VotingMethodEnum.rf_classifier, ensembler.ensemble_metadata.voting_method)

    def test_init_from_kwargs_default_voter(self):
        # Don't specify voting method via kwargs
        config = self.BASIC_CONFIG.copy()
        model_file_list = get_version_model_folders(config["model_type"])
        ensembler = Ensembler(
            model_type=ModelType.demographic,
            model_file_list=model_file_list,
        )
        self.assertEqual(set(model_file_list), set(ensembler.ensemble_metadata.model_file_list))
        self.assertEqual(len(model_file_list), len(ensembler.model_list))
        self.assertEqual(VotingMethodEnum.threshold, ensembler.ensemble_metadata.voting_method)

    def test_init_bad_model_list(self):
        config = {
            "model_type": ModelType.drug.value,
            "test_ensemble": True,
            "model_file_list": ["foo"],
            "train": False,
            "test": False,
            "job_id": "test_ensembler_bad_model_list",
        }
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)
        with self.assertRaises(ValueError):
            _ = Ensembler(
                ensemble_metadata=ensemble_metadata,
                data_source=data_source,
                job_metadata=job_metadata)

    def test_init_mismatch_label_class_no_prefix(self):
        # same, but without specifying the model type in the folder
        config = {
            "model_type": ModelType.diagnosis,
            "test_ensemble": True,
            "model_file_list": ["drug_cbert_20210107_w64"],
            "train": False,
            "test": False,
            "job_id": "test_ensembler_one_bert",
        }
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)
        with self.assertRaises(ValueError):
            # raises ValueError when it tries to look for a model folder (without model_type prefix)
            # in the wrong model_type
            _ = Ensembler(
                ensemble_metadata=ensemble_metadata,
                data_source=data_source,
                job_metadata=job_metadata)

    def test_lab_covid_lab(self):
        # covid lab has same class labels as lab, but is a different model type
        ensemble_type = ModelType.covid_lab
        model_constants = MODEL_TYPE_2_CONSTANTS[ensemble_type]
        model_name = list(model_constants.model_version_mapping().values())[0][0]
        config = {
            "model_type": ensemble_type,
            "test_ensemble": True,
            "model_file_list": [model_name],
            "train": False,
            "test": False,
            "job_id": "test_ensembler_covid_lab",
        }
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)
        ensembler = Ensembler(
            ensemble_metadata=ensemble_metadata,
            data_source=data_source,
            job_metadata=job_metadata)

        self.assertEqual(ensemble_type, ensembler.model_type)

    def test_init_bert_lstm(self):
        ensemble_type = ModelType.drug
        drug_constants = MODEL_TYPE_2_CONSTANTS[ModelType.drug]
        drug_model_names = list(drug_constants.model_version_mapping().values())[0]
        config = {
            "model_type": ensemble_type,
            "test_ensemble": True,
            "model_file_list": drug_model_names,
            "train": False,
            "test": False,
            "job_id": "test_ensembler_bert_lstm",
        }
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)
        ensembler = Ensembler(
            ensemble_metadata=ensemble_metadata,
            data_source=data_source,
            job_metadata=job_metadata)
        self.assertEqual(ensemble_type, ensembler.model_type)
        self.assertEqual(2, len(ensembler.model_list))

    # @unittest.skip("sanity check for including a model not in resources/files")
    def test_init_nonlocal_model(self):
        ensemble_type = ModelType.drug
        drug_constants = MODEL_TYPE_2_CONSTANTS[ModelType.drug]
        drug_model_name = list(drug_constants.model_version_mapping().values())[0][0]
        model_metadata = self._get_model_metadata(model_folder=drug_model_name, model_type=ModelType.drug)
        drug_bert_constants = MODEL_TYPE_2_CONSTANTS[ModelType.drug]
        drug_bert_model_name = list(drug_bert_constants.model_version_mapping().values())[0][0]
        config = {
            "model_type": ensemble_type,
            "test_ensemble": True,
            "model_file_list": [
                "drug_20201013",  # preexisting model, should work but needs to ping s3
                f"{drug_bert_model_name}"],
            "train": False,
            "test": False,
            "job_id": "test_ensembler_bert_lstm",
        }
        data_source = DataSource(**config)
        job_metadata = JobMetadata.from_dict(config)
        ensemble_metadata = EnsembleModelMetadata(**config)
        with self.assertRaises(ValueError):
            # shouldnt find the model folder "resources/files/drug/drug_20201013"
            _ = Ensembler(
                ensemble_metadata=ensemble_metadata,
                data_source=data_source,
                job_metadata=job_metadata)


if __name__ == '__main__':
    unittest.main()

