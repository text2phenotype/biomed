"""
Create model comparison report, save to results folder

usage:
```
python generate_model_comparisons.py my_config_file.json
```
"""
import os
import datetime
from typing import List, Tuple
import json
import argparse
from dataclasses import dataclass, field

import pandas as pd
import seaborn as sns

from biomed import RESULTS_PATH
from biomed.models.get_model import get_model_from_model_folder, get_model_type_from_model_folder, MODEL_CLASS_2_ENUM_CLASS
from biomed.constants.model_constants import ModelClass, ModelType
from biomed.common.report_helpers import (
    plot_model_perf_by_class_label,
    plot_input_size_vs_epoch_time,
    get_s3_model_report_str,
    get_s3_model_metadata_str,
    model_collection_precision_recall_f1_report,
    model_collection
)
from text2phenotype.common.aws import get_s3_client, get_matching_s3_keys
from text2phenotype.common.log import operations_logger
from text2phenotype.common import common

sns.set_theme(style="whitegrid")

S3_MODEL_TEST_PATH = "models/test"
S3_MODEL_TRAIN_PATH = "models/train"
DEFAULT_BUCKET_NAME = "biomed-data"


@dataclass
class ComparisonReportConfig:
    """
    dataclass that contains the required parameters to run a set of models to generate comparison reports

    The config json has the following expected structure:
    ```
    {
        "model_job_id_list": [
            ["diagnosis_bert_multi_ensemble_20210204_i2b2_mimic/diagnosis_prod_2.00", "diagnosis 2.00"],
            ["diagnosis_bert_multi_ensemble_20210204_i2b2_mimic/diagnosis_bert", "diagnosis_bert"],
            ["diagnosis_bert_multi_ensemble_20210204_i2b2_mimic/bert_only", "bert_only"]
        ],
        "bucket_name": "biomed-data",
        "report_names": ["report_weighted"],
        "comparison_name": "diagnosis_bert_multi_ensemble_20210204_i2b2_mimic_report",
        "target_model": "diagnosis 2.00"
    }
    ```
    """
    model_job_id_list: List[List[str]]  # required
    bucket_name: str = field(default=DEFAULT_BUCKET_NAME)
    report_names: List[str] = field(default_factory=lambda: ["report_weighted"])
    comparison_name: str = field(
        default_factory=f"model_comparison_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    target_model: str = None  # shortened model name to highlight the target we are improving
    sort_by_f1: bool = field(default=True)
    # if a model_name contains this str, we take the difference of it and any other matching subnames
    model_diff_on_name: str = None
    unpin_ylim: bool = False

    def to_dict(self) -> dict:
        """
        :return: dictionary of model params
        """
        return self.__dict__.copy()


def set_pd_display():
    pd.options.display.width = 120
    pd.set_option("max_columns", None)


# ------------------------------------------------------------------


def model_collection_support_df(model_list: List[Tuple[str, str]], bucket_name=DEFAULT_BUCKET_NAME):
    support_prefix = "data_support_metrics"
    s3_client = get_s3_client()
    support_metrics_list = []
    label_counts_by_model = {}
    for model_job_id, model_name in model_list:
        # join support data over both train and test
        model_support_json = _get_support_json(s3_client, model_job_id, "train", bucket_name=bucket_name)
        model_test_support_json = _get_support_json(s3_client, model_job_id, "test", bucket_name=bucket_name)
        model_support_json.update(model_test_support_json)

        model_support_json["model_name"] = model_name
        support_metrics_list.append(model_support_json)
    base_support_df = pd.DataFrame(support_metrics_list).set_index("model_name")
    operations_logger.info(f"loaded {base_support_df.shape[0]} models in support df")
    return base_support_df


def _get_support_json(s3_client, model_job_id, context=None, bucket_name=DEFAULT_BUCKET_NAME):
    """
    Get the data support json dict for a given job_id and context

    :param s3_client: s3 client
    :param model_job_id: str, job_id in the target bucket key prefix, eg "models/test"
    :param context: the context, could be "test", "train", "multi"
    :return:
    """
    context = context or "test"
    model_s3_key = os.path.join("models", context, model_job_id)
    test_report_data = get_s3_model_report_str(
        s3_client, bucket_name, model_s3_key, report_prefix="data_support_metrics"
    )
    model_support_json = json.loads(test_report_data)
    model_label_counts_dict = {}
    # skip the token label counts, store them elsewhere
    label_count_keys = [
        k
        for k in model_support_json.keys()
        if k.endswith("labeled_token_counts")
        or k.endswith("ann_label_counts")
        or k.endswith("token_label_counts")
    ]
    # remove the entries with lists/dicts
    for k in label_count_keys:
        model_label_counts_dict[k] = model_support_json[k].copy()
        del model_support_json[k]
    # we aren't currently using the model_label_counts_dict, but we may in the future
    return model_support_json


def model_collection_metadata_df(model_list: List[Tuple[str, str]], bucket_name=DEFAULT_BUCKET_NAME):
    """
    Load the model metadata for each model listed
    :param model_list: List[Tuple[str,str]]
        A list of tuples for the target model job_ids and display name
    :return: pd.DataFrame
    """
    s3_client = get_s3_client()
    # get the window size and batch size
    model_metadata = []
    for model_job_id, model_name in model_list:
        report_data = get_s3_model_metadata_str(s3_client, bucket_name, model_job_id)
        model_metadata_json = json.loads(report_data)
        model_metadata.append(
            {
                "model_name": model_name,
                "model_type": model_metadata_json["model_type"],
                "window_size": model_metadata_json["window_size"],
                "window_stride": model_metadata_json.get("window_stride"),  # may or may not exist
                "n_features": len(model_metadata_json["features"]),
            }
        )
    model_metadata_df = pd.DataFrame(model_metadata).set_index("model_name")
    operations_logger.info(f"loaded {model_metadata_df.shape[0]} models in support df")
    return model_metadata_df


def model_collection_history_df(model_list: List[Tuple[str, str]], bucket_name=DEFAULT_BUCKET_NAME):
    """
    Load the train history metrics data for each model listed
    :param model_list: List[Tuple[str,str]]
        A list of tuples for the target model job_ids and display name
    :return: pd.DataFrame
    """
    history_prefix = "train_history"
    s3_client = get_s3_client()
    model_histories = []
    for model_job_id, model_name in model_list:
        model_s3_key = os.path.join(S3_MODEL_TEST_PATH, model_job_id)
        report_data = get_s3_model_report_str(
            s3_client, bucket_name, model_s3_key, report_prefix=history_prefix
        )
        model_support_json = json.loads(report_data)
        df = pd.DataFrame(model_support_json)
        df["model_name"] = model_name
        df = df.reset_index().rename(columns={"index": "epoch"})
        model_histories.append(df)
    model_histories_df = pd.concat(model_histories, axis=0, sort=True)
    # do some cleaning, add in epoch times
    if "epoch_durations" in model_histories_df.columns:
        # rename col
        model_histories_df.rename(columns={"epoch_durations": "epoch_durations_sec"}, inplace=True)

    return model_histories_df


def model_history_to_epoch_durations(model_histories_df):
    """
    Translate the model history data into grouped epoch durations and test duration
    :param model_histories_df: pd.DataFrame
        from model_collection_history_df()
    :return: pd.DataFrame
    """
    if "avg_epoch_duration_sec" in model_histories_df.columns:
        model_epoch_durations_df = (
            model_histories_df.groupby("model_name")["avg_epoch_duration_sec"]
            .first()
            .to_frame()
            .rename(columns={"avg_epoch_duration_sec": "epoch_durations_sec"})
        )
    else:
        model_epoch_durations_df = (
            model_histories_df.groupby("model_name")["epoch_durations_sec"].mean().to_frame()
        )
    if "test_duration" in model_histories_df.columns:
        model_epoch_durations_df.join(
            model_histories_df.groupby("model_name")["test_duration"].first().to_frame()
        )
    return model_epoch_durations_df


def get_support_metrics(model_list: List[Tuple[str, str]], bucket_name=DEFAULT_BUCKET_NAME):
    """
    From each of the listed models, collect the support data (train and test), model metadata, and
    train history metrics.

    :param model_list: List[Tuple[str,str]]
        A list of tuples for the target model job_ids and display name
    :param bucket_name: str
        Name of the target bucket to pull the S3 files from
    :return: pd.DataFrame
    """
    support_df = model_collection_support_df(model_list, bucket_name=bucket_name)
    metadata_df = model_collection_metadata_df(model_list, bucket_name=bucket_name)
    base_support_df = support_df.join(metadata_df)

    model_histories_df = model_collection_history_df(model_list, bucket_name)
    model_epoch_durations_df = model_history_to_epoch_durations(model_histories_df)
    base_support_df = base_support_df.join(model_epoch_durations_df).reset_index()

    # do some minor alignment of the columns
    base_support_df["model_type_name"] = base_support_df["model_type"].apply(
        lambda x: ModelType(x).name
    )
    # if the model doesnt have a stride (pre 2020.12), then we assume it's an lstm and stride 1
    base_support_df["window_stride"].fillna(1, inplace=True)
    # the bert models have "total" in the name, the lstms do not.
    base_support_df["train_total_token_count"] = base_support_df["train_total_token_count"].combine_first(
        base_support_df["train_token_count"]
    )
    # TODO: find out why the test support metrics are not included
    # df has "test_token_count" for lstm, but is 0. bert does not have test_total_token_count
    # base_support_df["test_total_token_count"] = base_support_df["test_total_token_count"].combine_first(
    #     base_support_df["test_token_count"]
    # )
    base_support_df["validation_total_token_count"] = base_support_df["validation_total_token_count"].combine_first(
        base_support_df["validation_token_count"]
    )
    base_support_df = base_support_df.drop(
        columns=["train_token_count", "test_token_count", "validation_token_count"]
    )
    #
    # train_matrix_size_list = []
    # for i, model_row in base_support_df.iterrows():
    #     # model_type = get_model_type_from_model_folder()
    #     # metadata = ModelMetadataCache().model_metadata(model_type=model_type, model_folder=model_folder)
    #     # model_class = MODEL_CLASS_2_ENUM_CLASS[metadata.model_class]
    #
    #     # NB: get_model_class_enum() no longer exists, so this doesnt work
    #     model_class = get_model_class_enum(ModelType(model_row["model_type"]))
    #     # calculate the input matrix size
    #     # this will be slighty different if the model is a bert model or not...
    #     if model_class == ModelClass.bert:
    #         train_matrix_size = (
    #             model_row["train_total_num_windows"] * model_row["window_size"] * 768
    #         )
    #     else:
    #         train_matrix_size = (
    #             model_row["train_total_token_count"]
    #             * (model_row["window_size"] / model_row["window_stride"])
    #             * model_row["num_features"]
    #         )
    #     train_matrix_size_list.append(train_matrix_size)
    # base_support_df["train_matrix_size"] = train_matrix_size_list
    operations_logger.info(f"Collected support metrics from {len(model_list)} models")
    return base_support_df


def sanitize_class_label(class_label):
    sanitized_label = class_label.replace(" ", "")
    sanitized_label = (
        sanitized_label.split("/")[0] if "avg/" in sanitized_label else sanitized_label
    )
    return sanitized_label


def main(report_config: ComparisonReportConfig):
    """
    Requires authentication to the aws-dev "biomed-data" bucket
    :param report_config:
    """
    # convert list of lists to list of tuples
    target_models = [(model_job, model_name) for model_job, model_name in report_config.model_job_id_list]
    bucket_name = report_config.bucket_name
    target_model_name = report_config.target_model

    # check our s3 connection before we start writing, will raise error if we need to be authenticated
    _ = get_matching_s3_keys(
        bucket_name, prefix=os.path.join(S3_MODEL_TEST_PATH, target_models[0][0])
    )

    # set the score to sort by based on the config bool
    sort_by = "f1-score" if report_config.sort_by_f1 else None

    report_root = os.path.join(RESULTS_PATH, report_config.comparison_name)
    os.makedirs(report_root, exist_ok=True)

    for report_prefix in report_config.report_names:
        if 'misclassification' in report_prefix:
            # treating misclassification differently
            miss_df = model_collection(target_models, report_prefix, bucket=bucket_name, file_type='csv')
            miss_df.to_csv(os.path.join(report_root, f"comparison_{report_prefix}_misclassification.csv"))
            operations_logger.info(f"Created misclassification results for '{report_prefix}'")
        else:
            scores_df = model_collection_precision_recall_f1_report(target_models, report_prefix, bucket=bucket_name)
            score_type_order = scores_df.score_type.unique()

            # get the score differences between model names containing a target string, and the same name without string
            if report_config.model_diff_on_name:
                sources = [name for name in scores_df.model_name.unique() if
                           report_config.model_diff_on_name not in name]
                diff_list = []
                for source in sources:
                    source_df = scores_df[scores_df.model_name.str.contains(source)]
                    source_piv = source_df.pivot(index=["class_label", "score_type"], columns="model_name",
                                                 values="score")
                    assert source_piv.shape[1] == 2
                    col_pos = \
                    source_piv.columns[source_piv.columns.str.contains(report_config.model_diff_on_name)].values[0]
                    col_neg = \
                    source_piv.columns[~source_piv.columns.str.contains(report_config.model_diff_on_name)].values[0]
                    diff = (source_piv[col_pos] - source_piv[col_neg]).to_frame().reset_index().rename(
                        columns={0: "score"})
                    diff.insert(0, "model_name", source)
                    diff_list.append(diff)
                scores_df = pd.concat(diff_list).reset_index(drop=True)
                # fix the score_type order to the expected order
                scores_df = scores_df.set_index(["model_name", "class_label", "score_type"]) \
                    .reindex(score_type_order, level=2) \
                    .reset_index()
            scores_df = model_collection(target_models, report_prefix, bucket=bucket_name)
            # create plots for each of the class labels, except na
            possible_class_labels = scores_df.class_label.unique().tolist()
            possible_class_labels.remove("na")
            for class_label in possible_class_labels:
                fig = plot_model_perf_by_class_label(
                    scores_df,
                    class_label,
                    target_model=target_model_name,
                    report_name=report_config.comparison_name,
                    sort_by=sort_by,
                    unpin_ylim=report_config.unpin_ylim
                )
                sanitized_label = sanitize_class_label(class_label)
                fig.savefig(os.path.join(report_root, f"{report_prefix}_{sanitized_label}.png"))

                # create a CSV table with the scores, matching the figures
                pivot_scores = scores_df[scores_df.class_label == class_label].drop("class_label", axis=1)
                pivot_scores = scores_df[scores_df.class_label == class_label].drop("class_label", axis=1)\
                    .pivot(index="model_name", columns="score_type")
                pivot_scores.columns = [col[1] for col in pivot_scores.columns]
                pivot_scores = pivot_scores[["precision", "recall", "f1-score"]].sort_values("f1-score")
                pivot_scores.to_csv(os.path.join(report_root, f"comparison_{report_prefix}_{sanitized_label}.csv"))

            operations_logger.info(f"Created precision_recall_f1 results for '{report_prefix}'")

    # create the support metrics report
    try:
        support_df = get_support_metrics(target_models, bucket_name=bucket_name)
        size_time_fig = plot_input_size_vs_epoch_time(support_df)
        size_time_fig.savefig(os.path.join(report_root, f"datasize_time_regression.png"))
        operations_logger.info("Created support results")
    except ValueError as e:
        operations_logger.warning("Error in loading support data; target job may not have appropriate support files")
        operations_logger.warning(e.args[0])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Read report config and generate report output")
    parser.add_argument("config", type=str, help="path to target json config file")
    return parser.parse_args()


if __name__ == "__main__":
    set_pd_display()

    args = parse_arguments()
    if not args.config:
        raise ValueError("No config path passed in, requires input json file as config")
    if not os.path.exists(args.config):
        raise IOError(f"Target config file not found at path: {args.config}")

    operations_logger.info(f"Loading ComparisonReportConfig from {args.config}")
    report_config_dict = ComparisonReportConfig(**common.read_json(args.config))

    # hack to switch script auth to phi, if needed
    if report_config_dict.bucket_name == "prod-nlp-train1-us-east-1":
        os.environ["AWS_PROFILE"] = "aws-rwd-prod"

    main(report_config_dict)
