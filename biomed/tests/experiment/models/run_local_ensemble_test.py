"""
Simple script for LOCALLY loading a model and running training

You need to have the appropriate data stored locally on your localhost.
Make sure MDL_COMN_DATA_ROOT points to this location, eg
export MDL_COMN_DATA_ROOT=/opt/S3

To use the local file system, set:
export MDL_COMN_USE_STORAGE_SVC=False

Then run this script in your environment of choice!
"""
import os
import sys


from biomed.meta.ensembler import Ensembler
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from text2phenotype.common.data_source import DataSource
from biomed.train_test.job_metadata import JobMetadata
from text2phenotype.constants.environment import Environment
from text2phenotype.apiclients import FeatureServiceClient


TESTING_PARAMS = {
  "model_type": 14,
  "test_ensemble": True,
  "train": False,
  "test": False,
  "max_token_count": 100000,
  "job_id": "diagnosis_bert_ensemble_20210127_test",
  "async_mode": False,

  "model_file_list": [
    "5485",
    "diagnosis_bert/diagnosis_cbert_20210114_w64"
  ],
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


def main(config):
    """main entry point"""
    # Needs to have feature service server running locally at a known target endpoint

    text = "HELLO myocardial infarction is no bueno percocet aspirin headache asthma diabetes type 2 Lab values"
    annotations, vectors = FeatureServiceClient().annotate_vectorize(text)

    data_source = DataSource(**config)
    job_metadata = JobMetadata.from_dict(config)
    ensemble_metadata = EnsembleModelMetadata(**config)
    ensembler = Ensembler(
        ensemble_metadata=ensemble_metadata,
        data_source=data_source,
        job_metadata=job_metadata)

    res = ensembler.predict(annotations, vectors=vectors, text=text)
    # out = diagnosis_sign_symptoms(annotations, vectors, text=text)


if __name__ == "__main__":
    os.environ['MDL_COMN_USE_STORAGE_SVC'] = 'False'
    Environment.load()
    sys.exit(main(TESTING_PARAMS))
