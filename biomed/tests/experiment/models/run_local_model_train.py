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

from biomed.models.model_base import ModelBase
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.data_sources.data_source import BiomedDataSource
from biomed.constants.constants import ModelType
from text2phenotype.constants.environment import Environment

TESTING_PARAMS = {
  "learning_rate": 0.01,
  "epochs": 1,
  "async_mode": False,
  "feature_set_version": "dev/20201208",
  "feature_set_subfolders": [
    "train/tag_tog_text"
  ],
  "testing_fs_subfolders": [
    "test/tag_tog_text"
  ],
  "ann_dirs": [
    "tag_tog_annotations/diagnosis_signsymptom_validation/2020-12-08/CEPuser"
  ],
  "original_raw_text_dirs": [
    "mimic_shareeclef/DISCHARGE_SUMMARY",
    "I2B2"
  ],
  "features": [
    126,125,52,53,69,8,9,10,11,12,14,16,17,18,37,38,39,47,48,49,50,84,85,88,96,112,115,116,118,119,120,121
  ]
}


def main(params):
    """main entry point"""
    # TODO: just read in a target .json file to parse parameters.

    metadata = ModelMetadata(**params)
    ds = BiomedDataSource(
        # parent_dir="/opt/S3/prod-nlp-train1-us-east-1",\
        parent_dir="/opt/S3/biomed-data",
        **params
    )
    job_metadata = JobMetadata.from_dict(params)

    model = ModelBase(data_source=ds, model_metadata=metadata, job_metadata=job_metadata, model_type=ModelType.diagnosis)
    model.train()


if __name__ == "__main__":
    os.environ['MDL_COMN_USE_STORAGE_SVC'] = 'True'
    os.environ["AWS_PROFILE"] = "aws-rwd-dev"
    Environment.load()
    sys.exit(main(TESTING_PARAMS))
