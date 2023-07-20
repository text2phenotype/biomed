"""
Create parameter files for phi/non-phi models and datasets for a target model

This script is step 1 in running a model-dataset pairwise comparison.
The goal is to take each submodel for a model type and the associated dataset it was trained on,
and see how all of the other submodels will perform on that dataset.

In the case of `diagnosis`, there are 6 submodels (+1 ensemble), trained on 6 unique datasets.
This will give a 7x6 matrix, requiring 42 separate "test" validation runs on both PHI and non-PHI data.

Step 2 is to take the config files that are created here and run them:
- PHI params can be run manually through yacht and the OpenAPI UI,
  OR via run_model_test_configs.py, which sends API requests to a local or remote biomed service
- non-PHI data can only be run by manually copy-pasting the json-like escaped string config into
  the parameters>metadata field in Teamcity.

Step 3 is to copy all of the returned model outputs to the local machine. This can be done via S3
- non-PHI: aws-switch aws-dev
    aws s3 sync s3://biomed-data/models/test biomed/results/diseasesign_compare_20200924/dev --exclude "*" --include "diseasesign_compare*"
- PHI: aws-switch aws-phi
    aws s3 sync s3://prod-nlp-train1-us-east-1/models/test biomed/results/diseasesign_compare_20200924/phi --exclude "*" --include "diseasesign_compare*

Step 4 is to run the analysis script, `analyze_diseasesign_model_metrics`, which will create the
comparison figures

"""
import sys
import os
import pkg_resources
import itertools
from typing import List, Dict, AnyStr, Tuple
import datetime
from copy import deepcopy

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common

from biomed import RESULTS_PATH
from biomed.constants.constants import ModelType, get_version_model_folders

# =============================
# change these globals to change which model type to create configs for

MODEL_NAME = "drug"
JOB_ID_BASE = "drug_compare_20210831"
MODEL_TYPE = getattr(ModelType, MODEL_NAME)  # equivalent to ModelType.diagnosis

DISEASE_SIGN_MODELS = get_version_model_folders(MODEL_TYPE)
# =============================

MAX_TOKEN_COUNT = 100000

JOB_METADATA = {
    "train": True,
    "test": True,
    "max_token_count": MAX_TOKEN_COUNT,
}

# flag for union on all possible subfolders
USE_ALL_DATA = False

# additional parameters - for any lists in here, will duplicate configs
# ensure these parameters exist in job configs, otherwise they will be ignored
ADDTL_PARAMS = {'epochs': [3, 10, 20],
                'learning_rate': [0.01, 0.001]}


class AwsProfile:
    """Keeps track of which dataset lives with with permissions"""
    phi = "phi"
    dev = "dev"


def fs_subfolder_union(data_source_dict):
    """Find and join all available data subfolders for feature service"""
    subfolder_sets = [
        "feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders",
    ]
    all_subfolders = [subfolder
                      for subfolder_list in subfolder_sets
                      for subfolder in data_source_dict[subfolder_list]]
    all_subfolders = list(set(all_subfolders))
    data_source_dict.update({
        subfolder: all_subfolders for subfolder in subfolder_sets
    })

    return data_source_dict


def get_model_params(model_name, model_folder):
    """Get a dict with all feature indexes used for a given model name and list of model folders"""
    model_dir = pkg_resources.resource_filename(
        "biomed.resources",
        f"files/{model_name}/{model_folder}")
    model_filename = (
        [filename for filename in os.listdir(model_dir) if filename.endswith(".h5")][0]
    )
    model_params = common.read_json(os.path.join(
        model_dir, model_filename + ".metadata.json",
    ))
    return model_params


def collect_model_params(model_folder_names: List[str], model_name: str):
    """
    Given a production-released model (stored in dvi, pulled locally via `dvi pull`), get the
    parameters used to train the model.

    :param model_folder_names: List[str]
    :param model_name: string name of the model, eg 'diagnosis'
    :return: Tuple[Dict, Dict, Dict]
        (model_aws_profile, model_data_sources, model_filenames)
    """
    # load all of the dataset parameters for each production model
    dataset_keys = [
        "original_raw_text_dirs", "ann_dirs", "feature_set_version",
        "testing_text_dirs", "testing_ann_dirs",
        "validation_text_dirs", "validation_ann_dirs",
        "feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders", "class_weight"
    ]
    model_data_sources = {}  # data source dict, from data_source.json
    model_filenames = {}  # full model filenames
    model_aws_profile = {}  # which aws profile for which dataset
    for model_folder in model_folder_names:
        # this will only work if the models exist locally. eg on a local dev system, not on yacht
        model_dir = pkg_resources.resource_filename(
            "biomed.resources",
            f"files/{model_name}/{model_folder}")
        data_source_json = common.read_json(os.path.join(model_dir, "data_source.json"))
        dataset_params = {
            param_name: data_source_json.get(param_name, []) for param_name in dataset_keys
        }
        model_data_sources[model_folder] = dataset_params

        # HACK: we only take the first h5 file from the model folder
        # This will not generate expected results if there are more than one h5 files
        model_filenames[model_folder] = (
            [filename for filename in os.listdir(model_dir) if filename.endswith(".h5")][0]
        )
        model_filenames[model_folder] = model_folder

        # HACK - failure prone for any folder name that is a number and is a PHI model
        # check if the model was trained on teamcity, aka was trained on non-phi data
        # any model trained on non-phi data must have async_mode=False
        # all cyan data must have async_mode=True
        model_aws_profile[model_folder] = AwsProfile.dev if model_folder.isdigit() else AwsProfile.phi
    return model_aws_profile, model_data_sources, model_filenames


def create_config_params(
        pred_model_folder: str,
        data_source_folder: str,
        model_aws_profile: Dict[str, str],
        model_data_sources: Dict[str, Dict],
        model_name: str,
        union_data_sources: bool = USE_ALL_DATA
) -> dict:
    data_source_profile = model_aws_profile[data_source_folder]
    # get prediction model file name
    operations_logger.info("Loading model data: '{}' ({})".format(
        pred_model_folder, model_aws_profile.get("pred_model_folder", "<both>")))
    # load source data info
    operations_logger.info("setting data source from: '{}' ({})".format(
        data_source_folder, data_source_profile))

    # load the data source params used in the original model
    # and don't modify it in place!
    data_source = model_data_sources[data_source_folder].copy()

    # select all possible data sources, not just the test subfolders
    if union_data_sources:
        data_source = fs_subfolder_union(data_source)

    # our special snowflake, 4362
    if data_source_folder == "4362":
        # point the text-dirs to the correct path
        text_dir_4362 = "mimic/shareclef-ehealth-evaluation-lab-2014-task-2-disorder-attributes-in-clinical-reports-1.0/20200306/"
        data_source.update({
            target_text_dirs: [f"{text_dir_4362}{folder}" for folder in data_source[target_text_dirs]]
            for target_text_dirs in ["original_raw_text_dirs", "validation_text_dirs", "testing_text_dirs"]
        })
        data_source.update({
            target_ann_dirs: ["annotations" for folder in data_source[target_ann_dirs]]
            for target_ann_dirs in ["ann_dirs", "testing_ann_dirs", "validation_ann_dirs"]
        })

    # if no subfolders found in original data, add the train/test subfolders
    subfolder_keys = ["feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders"]
    if any([not data_source[k] for k in subfolder_keys]):
        subfolder_defaults = ["train", "test"]
        operations_logger.info(
            f"Data set for model {data_source_folder} ({data_source_profile}) has no subfolders, "
            f"defaulting to {subfolder_defaults}")
        for k in subfolder_keys:
            data_source[k] = subfolder_defaults

    if "class_weight" in data_source and not data_source['class_weight']:
        # empty weights, remove the field
        _ = data_source.pop("class_weight")

    # set the source/dest buckets here, according to the target data profile:
    if data_source_profile == AwsProfile.phi:
        buckets = {
            "source_bucket": "prod-nlp-train1-us-east-1",
            "dest_bucket": "prod-nlp-train1-us-east-1",
        }
    else:
        buckets = {
            "source_bucket": "biomed-data",
            "dest_bucket": "biomed-data",
        }
    data_source.update(buckets)

    # job metadata
    is_phi_data = data_source_profile == AwsProfile.phi
    job_metadata = JOB_METADATA.copy()
    job_id = f"{JOB_ID_BASE}_model-{pred_model_folder}_data-{data_source_folder}"
    job_metadata.update({
        "job_id": job_id,
        "async_mode": is_phi_data,  # True when is phi data source, false when non-phi
    })
    if pred_model_folder == "ensemble":
        job_metadata["test"] = False
        job_metadata["train"] = False
        job_metadata["test_ensemble"] = True

    # model metadata
    if pred_model_folder == "ensemble":
        # model file list in ensemble is the models being used for data sources
        model_file_list = list(model_data_sources.keys())
        model_metadata = {
            "model_type": MODEL_TYPE.value,
            "model_file_name": "",
            "model_file_list": model_file_list,
        }
    else:
        model_metadata = get_model_params(model_name, pred_model_folder)
        if JOB_METADATA['train']:
            del model_metadata["model_file_name"]
            del model_metadata["model_file_path"]
        else:
            # for testing, overwrite the model filename, because the model_file_name parameter is not idempotent
            # the full filename is written to the config, but when the config is read in it expects the folder name
            model_metadata["model_file_name"] = pred_model_folder
            del model_metadata["model_file_path"]

    config_json = model_metadata.copy()
    config_json.update(job_metadata)
    config_json.update(data_source)

    return config_json


def write_config_file(config_json, data_source_profile):
    """
    Write out a job config file
    Writes out an escaped string if non-phi data is being used (to txt file),
    Writes out json dict to a json file

    :param config_json: dict
    :param data_source_profile: str
    :return:
    """
    # write out the config file
    job_id = config_json.get("job_id")
    is_phi = data_source_profile == AwsProfile.phi
    file_ext = "json" if is_phi else "txt"
    config_out_path = os.path.join(
        RESULTS_PATH,
        JOB_ID_BASE,
        "configs",
        data_source_profile,  # phi/ vs dev/
        f"{job_id}.{file_ext}")
    os.makedirs(os.path.dirname(config_out_path), exist_ok=True)
    if is_phi:
        # wrap in outer dict with 'data' key:
        out_dict = {"data": config_json}
        common.write_json(out_dict, config_out_path)
    elif data_source_profile == AwsProfile.dev:
        common.write_escaped_string_txt(config_json, config_out_path)
        # may as well write a human readable version as well
        common.write_json(config_json, config_out_path + ".json")
    else:
        operations_logger.error("How did you get in here?")
    operations_logger.info(
        f"Wrote config file to: {config_out_path}"
    )
    return


def main():
    model_name = MODEL_NAME
    selected_models = DISEASE_SIGN_MODELS

    model_aws_profiles, model_data_sources, model_filenames = collect_model_params(selected_models, model_name)

    all_configs = []

    # iterate through the ensemble test datasets
    for data_source_dir in selected_models:
        config_json = create_config_params(
            "ensemble",
            data_source_dir,
            model_aws_profiles,
            model_data_sources,
            model_name,
            union_data_sources=USE_ALL_DATA  # whether or not we use just test dirs, or the whole dataset
        )
        all_configs.append([config_json, data_source_dir])

    for pred_model_dir, data_source_dir in itertools.product(selected_models, selected_models):
        config_json = create_config_params(
            pred_model_dir,
            data_source_dir,
            model_aws_profiles,
            model_data_sources,
            model_name,
            union_data_sources=USE_ALL_DATA  # whether or not we use just test dirs, or the whole dataset
        )
        all_configs.append([config_json, data_source_dir])

    # kind of hacky way to deal with all params
    # this will be used to modify job id
    param_ids = '_{}' * len(ADDTL_PARAMS)
    # get all combinations of values
    param_val_product = [x for x in itertools.product(*ADDTL_PARAMS.values())]
    # get the ids (for adding to job_id
    param_val_ids = [x for x in itertools.product(*[range(len(v)) for v in ADDTL_PARAMS.values()])]

    for config, data_source_dir in all_configs:
        write_config_file(config, model_aws_profiles[data_source_dir])
        job_id = config['job_id']
        for i, p in enumerate(param_val_product):
            config_w_params = deepcopy(config)
            add_job_id = param_ids.format(*param_val_ids[i])
            config_w_params['job_id'] = job_id + add_job_id
            # no point to write ensembles here
            if 'model_file_list' not in config_w_params:
                add_params = dict(zip(ADDTL_PARAMS, p))
                config_w_params.update(add_params)
                write_config_file(config_w_params, model_aws_profiles[data_source_dir])
    return 0


def profile_main():
    """Used in creating cProfiled output for the main() function"""
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()

    main()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats()

    timestamp = datetime.datetime.now().strftime("%Y%d%m_%H%M%S")
    with open(f'profile_stats_{timestamp}.txt', 'w+') as f:
        f.write(s.getvalue())


if __name__ == "__main__":
    """
    To run this script, feature-service and biomed must both be running locally
    """

    sys.exit(main())
    # profile_main()  # uncomment to run profiler
