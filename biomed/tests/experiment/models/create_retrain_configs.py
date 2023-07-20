import os
import pkg_resources
from typing import List, Dict, AnyStr, Tuple, Optional
from collections import defaultdict
import datetime
import time
import shutil

from text2phenotype.common.log import operations_logger
from text2phenotype.common import common
from text2phenotype.constants.features import FeatureType
from biomed import RESULTS_PATH
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS, ModelType, BertEmbeddings
from biomed.constants.constants import get_version_model_folders, BIOMED_VERSION_TO_MODEL_VERSION
from biomed.models.model_cache import get_full_path, find_model_file
from biomed.train_test.job_metadata import JobMetadata
from biomed.models.model_metadata import ModelMetadata

# should be the same for both phi and dev
TARGET_FEATURE_SET_VERSION = "fdl/20210429-mike-test"

BIOMED_VERSION = sorted(BIOMED_VERSION_TO_MODEL_VERSION.keys())[-1]

CUR_DATE_STR = datetime.datetime.now().strftime("%Y%m%d")

TRAIN_JOB_ID_BASE = f"train_configs_{CUR_DATE_STR}"

# set max token count
MAX_TOKEN_COUNT = 120000

DEFAULT_JOB_METADATA = {
    "train": True,
    "test": True,
}


class AwsProfile:
    """Keeps track of which dataset lives with which permissions"""
    phi = "phi"
    dev = "dev"


def get_model_params(model_name, sub_model_folder):
    """Get a dict with all feature indexes used for a given model name and list of model folders"""
    model_metadata_path = find_model_file(model_name, sub_model_folder, suffix=".metadata.json")
    model_params = common.read_json(model_metadata_path)
    return model_params


def get_union_feature_indexes(features_dict):
    """
    Get the union of all features used for all listed models
    :param features_dict: Dict[str, List[int]]
        Dict keyed by model folder, value of features for that model
    """
    all_features = set()
    for model_features in features_dict.values():
        if model_features:
            all_features = all_features.union(set(model_features))
    all_features = sorted(list(all_features))
    return all_features


def check_union_feature_diff(model_name, model_folders):
    """
    Given a target model and associated subfolders, return a dict of which features
     are not used from the superset union per submodel
     """
    features_dict = {
        folder: get_model_params(model_name, folder)["features"]
        for folder in model_folders}
    feature_union = get_union_feature_indexes(features_dict)
    missing_features_per_model = {
        folder: set(feature_union) - set(features_dict[folder]) for folder in model_folders
    }
    return missing_features_per_model


def create_submodel_train_config(model_type_str: str, submodel_name: str, feature_set_version: str = None):
    """
    Given a production-released model (stored in dvc, pulled locally via `dvc pull`), get the
    parameters used to train the model.

    :param model_type_str: str
        string name of the model, eg 'diagnosis'
        Should match the model folder name in biomed/resources/files
    :param submodel_name: str
        name of the submodel, should match name of .dvc file and the resulting synced folder
    :param feature_set_version: Optional[str]
        Name of the target featureset folder to use
        If None, uses the same featureset from previous production model
    :return: Tuple[Dict, Dict, Dict]
        (model_aws_profile, model_data_sources, model_filenames)
    """
    model_dir = get_full_path(model_type_str, submodel_name)

    # HACK - failure prone for any model name that doesnt actually have phi in the name
    matching_phi_substrings = ["phi", "oncology2324", "genetics2269"]
    model_aws_profile = (
        AwsProfile.phi
        if any([sub in submodel_name for sub in matching_phi_substrings])
        else AwsProfile.dev
    )

    # ---------------------------------------------------------------------------------------------
    # data source
    # load all of the data_source parameters for each production model
    dataset_keys = [
        "original_raw_text_dirs", "ann_dirs", "feature_set_version",
        "feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders"
    ]
    # this will only work if the models exist locally. eg on a local dev system, not on S3
    try:
        orig_data_source = common.read_json(os.path.join(model_dir, "data_source.json"))
    except FileNotFoundError:
        operations_logger.error("FATAL: data source file not found locally for "
                                f"'{os.path.join(model_type_str, submodel_name)}'")
        return {}, model_aws_profile
    data_source = {
        param_name: orig_data_source.get(param_name, []) for param_name in dataset_keys
    }

    # replace feature_set_version with new one, if passed in
    data_source["feature_set_version"] = feature_set_version or data_source["feature_set_version"]

    # TODO: sanity check all current models to make sure we are using train/test splits
    fs_subfolder_keys = ["feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders"]

    # TODO: do this better! maybe query the feature_set folder and get the structure?

    # if use_original_traintest:
    #     # if no subfolders found in original data, add the train/test subfolders
    #     subfolder_keys = ["feature_set_subfolders", "testing_fs_subfolders", "validation_fs_subfolders"]
    #     if any([not data_source[k] for k in subfolder_keys]):
    #         # use both train & test for testing and validation subfolders
    #         subfolder_defaults = ["train", "test"]
    #         operations_logger.info(
    #             f"Data set for model {submodel_name} ({model_aws_profile}) has no subfolders, "
    #             f"defaulting to {subfolder_defaults}")
    #         for k in subfolder_keys:
    #             data_source[k] = subfolder_defaults
    # else:
    #     # we pin the featureset and test/val folders for each of the featuresets to be consistent
    #     data_source["feature_set_subfolders"] = ["a", "b", "c", "d"]
    #     data_source["validation_fs_subfolders"] = ["e"]
    #     data_source["testing_fs_subfolders"] = ["e"]

    # ---------------------------------------------------------------------------------------------
    # model_metadata
    default_model_meta = ModelMetadata(model_type=ModelType[model_type_str]).to_json()
    orig_model_metadata = get_model_params(model_type_str, submodel_name)
    model_meta_skip_keys = ["job_id", "model_file_path", "model_file_name"]
    model_metadata = _get_diff_keys(orig_model_metadata, default_model_meta, skip_keys=model_meta_skip_keys)
    model_metadata["model_type"] = ModelType[model_type_str].value

    # ---------------------------------------------------------------------------------------------
    # job_metadata

    default_job_meta = JobMetadata().to_json()
    orig_job_metadata = common.read_json(os.path.join(model_dir, "job_metadata.json"))
    job_metadata = _get_diff_keys(orig_job_metadata, default_job_meta, skip_keys=["job_id"])
    is_phi_data = model_aws_profile == AwsProfile.phi
    job_metadata.update(DEFAULT_JOB_METADATA.copy())

    # ---------------------------------------------------------------------------------------------
    # set the job_id
    job_id = f"{model_type_str}_{CUR_DATE_STR}"
    job_id = "phi_" + job_id if is_phi_data else job_id

    # check for bert vs cbert
    bert_embedding_name = orig_model_metadata.get("embedding_model_name")
    bert_short_map = {
        BertEmbeddings.bert.name: "bert",
        BertEmbeddings.bio_clinical_bert.name: "cbert"
    }
    if bert_embedding_name:
        bert_name = bert_short_map[bert_embedding_name]
        job_id = job_id.replace("bert", bert_name)


    # Increment job id if there is more than one submodel for each model_name/aws_profile
    config_out_path = _get_submodel_config_path(model_type_str, model_aws_profile)
    os.makedirs(config_out_path, exist_ok=True)
    # NOTE: what if we have more than one phi model? Append an incrementing suffix
    # relies on there being separate model_folders in the output for each unique model_name
    idx = 0
    while job_id in _strip_json_ext(os.listdir(config_out_path)):
        if idx:
            job_id, idx = job_id.rsplit("_", 1)
            idx = int(idx)
        idx += 1
        job_id = f"{job_id}_{idx}"

    job_metadata["job_id"] = job_id

    # ---------------------------------------------------------------------------------------------
    # join separate metadata into single output file
    config_json = model_metadata.copy()
    config_json.update(job_metadata)
    config_json.update(data_source)

    return config_json, model_aws_profile


def _get_diff_keys(source_dict: dict, default_dict: dict, skip_keys: List = None):
    """
    Return the set of keys that have different values from the default dict

    :param source_dict:
        The dict object we want to compare against the default
    :param default_dict:
        The default dict, expected to be used for everything other than the changed values in source
    :param skip_keys:
        list of keys to ignore in the comparison
    :return: dict
        keys and values from the source_dict that are different than the default_dict
    """
    skip_keys = skip_keys or []
    out_dict = {}
    for k in default_dict.keys():
        if k in source_dict and \
                source_dict[k] != default_dict[k] and \
                k not in skip_keys:
            out_dict[k] = source_dict[k]
    return out_dict


def _strip_json_ext(file_list):
    """Get filenames without the extension"""
    return [s.split(".", 1)[0] for s in file_list if s.endswith(".json")]


def write_config_file(config_json, model_name: str, model_aws_profile: str):
    """
    Write out a job config file
    Writes out an escaped string if non-phi data is being used (to txt file),
    Writes out json dict to a json file

    :param model_name:
    :param config_json: dict
    :param model_aws_profile: str
    :return:
    """
    # write out the config file
    job_id = config_json.get("job_id")
    is_phi = model_aws_profile == AwsProfile.phi
    config_out_path = _get_submodel_config_path(model_name, model_aws_profile)
    os.makedirs(os.path.dirname(config_out_path), exist_ok=True)
    config_file_path = os.path.join(config_out_path, f"{job_id}.json")

    # create json or txt/json outputs, depending on training location (phi/jenkins)
    if is_phi:
        # wrap in outer dict with 'data' key:
        out_dict = {"data": config_json}
        common.write_json(out_dict, config_file_path)
    else:
        common.write_escaped_string_txt(config_json, config_file_path + ".txt")
        # may as well write a human readable version as well
        common.write_json(config_json, config_file_path)

    operations_logger.info(
        f"Wrote config file to: {config_file_path}"
    )
    return


def _get_submodel_config_path(model_name, model_aws_profile):
    config_out_path = os.path.join(
        RESULTS_PATH,
        TRAIN_JOB_ID_BASE,
        model_aws_profile,  # phi/ vs dev/
        model_name
    )
    return config_out_path


def main():
    start_time = time.time()

    # delete the results directory, so we don't continue collecting new configs
    results_root_path = os.path.join(RESULTS_PATH, TRAIN_JOB_ID_BASE)
    if os.path.exists(results_root_path):
        shutil.rmtree(results_root_path)

    # get all current models

    prod_model_lists = {
        model_type: get_version_model_folders(model_type)
        for model_type, const in MODEL_TYPE_2_CONSTANTS.items()
        if const.model_version_mapping() is not None
    }

    # hold the list of features used for each model
    # format: {'model_type': {'model_name': [1,2,3], 'model_name_2': [2,5,6], 'ALL': [1,2,3,5,6]}
    model_features_dict = defaultdict(dict)

    exclude_models = ["date_of_service"]
    # NOTE lab and covid_lab are in the same model folder, but are split as different models??
    for cur_model_type, cur_submodel_names in prod_model_lists.items():
        if cur_model_type.name in exclude_models:
            continue
        for submodel_name in cur_submodel_names:
            type_pointer, model_name = os.path.split(submodel_name)
            if type_pointer:
                cur_model_type = ModelType[type_pointer]
                submodel_name = model_name
            output_config, aws_profile = create_submodel_train_config(cur_model_type.name, submodel_name,
                                                                      feature_set_version=TARGET_FEATURE_SET_VERSION)
            if output_config == dict():
                # we weren't able to create the config
                continue

            model_features_dict[cur_model_type.name][submodel_name] = output_config.get('features')
            write_config_file(output_config, cur_model_type.name, aws_profile)

    operations_logger.info(f"Writing configs took {time.time() - start_time:0.3f} sec")

    # get union of submodel features for each model
    for model_name in model_features_dict.keys():
        if not model_features_dict[model_name]:
            i = 1
        if model_features_dict[model_name]:
            model_features_dict[model_name].update({
                "ALL": get_union_feature_indexes(model_features_dict[model_name])
            })
    # get union of all model features
    all_submodel_features_dict = {model_name: submodel_dict["ALL"] for model_name, submodel_dict in model_features_dict.items()}
    all_used_features = get_union_feature_indexes(all_submodel_features_dict)
    model_features_dict.update({"ALL_FEATURES": all_used_features})
    # sanity check: which features are not being used?
    all_feature_enum_vals = [val.value for val in FeatureType.__members__.values()]
    unused_features = sorted(list(set(all_feature_enum_vals) - set(all_used_features)))
    print(f"Features not used in the total superset of model features:\n{unused_features}")
    feature_list_path = os.path.join(
        RESULTS_PATH,
        TRAIN_JOB_ID_BASE,
        "model_features.json"
    )
    common.write_json(model_features_dict, feature_list_path)


if __name__ == "__main__":
    main()
