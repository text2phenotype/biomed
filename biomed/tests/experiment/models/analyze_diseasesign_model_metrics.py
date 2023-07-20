"""
This script is step 4 in validating model test results across different datasets.
This expects the folder containing the model results to exist at:
   biomed/results/{JOB_ID_BASE}/{dev|phi}/

Step 3 is to copy all of the returned model outputs to the local machine. This can be done via S3
- non-PHI: aws-switch aws-dev
    aws s3 sync s3://biomed-data/models/test biomed/results/diseasesign_compare_20200924/dev --exclude "*" --include "diseasesign_compare*"
- PHI: aws-switch aws-phi
    aws s3 sync s3://nlp-train1-us-east-1/models/test biomed/results/diseasesign_compare_20200924/phi --exclude "*" --include "diseasesign_compare*"

Step 4 is to run the analysis script, `analyze_diseasesign_model_metrics`, which will create the
comparison figures:
 - matrix visualizations of precision, recall, and F1 scores, over model averages and specific labels
 - bar plots of ensemble model performance on the different datasets
 - scatterplot of number of labeled token per annotation file count

These figures will be written to the figures folder in results:
   biomed/results/{JOB_ID_BASE}/figures

"""
import sys
import os
import re


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from text2phenotype.common import common

from biomed import RESULTS_PATH
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport
from biomed.common.combined_model_label import DiseaseSignSymptomLabel
from biomed.tests.experiment.models.create_iterable_model_test_datasets import AwsProfile

MAX_TOKEN_COUNT = 100000

MODEL_NAME = "diagnosis"
JOB_ID_BASE = "diseasesign_compare_20200924"

# expected string label for the model avg score over non-na labels
AVG_NAME = "avg / total"


def set_pd_display():
    pd.options.display.width = 120
    pd.set_option("max_columns", None)


def parse_model_data_names(job_id):
    match_str = ".*_(?:model-)(.*)_(?:data-)(.*)"
    model_sources = re.search(match_str, job_id).groups()
    if len(model_sources) != 2:
        raise ValueError(f"Bad string match for job_id: {job_id}; found {model_sources}")
    model_name, data_name = model_sources
    return model_name, data_name


def read_model_test_metrics(data_source_profile, overwrite_cache=False):
    out_file_path = os.path.join(
        RESULTS_PATH,
        JOB_ID_BASE,
        f"{data_source_profile}_metrics_df.pkl",
    )
    # check for cached data to avoid really slow reads
    if not overwrite_cache and os.path.exists(out_file_path):
        print(f"Found cached '{data_source_profile}' file, loading: {out_file_path}")
        metrics_df = pd.read_pickle(out_file_path)
    else:
        print(f"Reading '{data_source_profile}' metrics files to dataframe")
        config_out_path = os.path.join(
            RESULTS_PATH,
            JOB_ID_BASE,
            data_source_profile,  # phi/ vs dev/
        )
        model_output_names = sorted(os.listdir(config_out_path))

        label_enum = DiseaseSignSymptomLabel
        model_test_metrics = {}
        for job_id in model_output_names:
            # scrape the results file
            report_filename = f"report_weighted_{MODEL_NAME}.txt"
            report_file_path = os.path.join(
                RESULTS_PATH,
                JOB_ID_BASE,
                data_source_profile,
                job_id,
                report_filename)
            report_df = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(common.read_text(report_file_path), label_enum)\
                .reset_index(drop=True)

            # parse out the row and col names for our output grid
            model_name, data_name = parse_model_data_names(job_id)
            report_df.insert(0, "data_name", data_name)
            report_df.insert(0, "model_name", model_name)
            model_test_metrics[job_id] = report_df
        metrics_df = pd.concat(model_test_metrics, axis=0)
        metrics_df = metrics_df.reset_index(drop=True)
        metrics_df["support"] = metrics_df["support"].astype(int)  # convert float counts to int, for cleanliness
        metrics_df.to_pickle(out_file_path)
    return metrics_df


def read_model_support_metrics():
    model_support_metrics = []
    # do this for both support profiles
    for data_source_profile in [AwsProfile.phi, AwsProfile.dev]:
        config_out_path = os.path.join(
            RESULTS_PATH,
            JOB_ID_BASE,
            data_source_profile,  # phi/ vs dev/
        )
        model_output_names = sorted(os.listdir(config_out_path))
        for job_id in model_output_names:
            report_filename = f"data_support_metrics_{MODEL_NAME}.json"
            report_file_path = os.path.join(
                RESULTS_PATH,
                JOB_ID_BASE,
                data_source_profile,
                job_id,
                report_filename)
            model_name, data_name = parse_model_data_names(job_id)

            data_support = common.read_json(report_file_path)
            reshaped_data_support = {
                "test_num_matched_annotation_files": data_support["test_num_matched_annotation_files"],
                "model_name": model_name,
                "data_name": data_name,
            }
            reshaped_data_support.update({
                f"labeled_token_counts_{label}": int(data_support["test_labeled_token_counts"][label])
                for label in data_support["test_labeled_token_counts"].keys()
            })
            model_support_metrics.append(reshaped_data_support)
    support_df = pd.DataFrame(model_support_metrics)
    # reorder columns
    cols_to_move = ['model_name', 'data_name']
    support_df = support_df[cols_to_move + [col for col in support_df.columns if col not in cols_to_move]]
    support_df.sort_values(cols_to_move, inplace=True)
    return support_df


def create_score_matrix(metrics_df, target_metric, class_label=AVG_NAME, cmap="viridis", pin_cmap_range=True, save_fig_path=None):
    cropped_df = metrics_df[metrics_df['class_label'] == class_label][
        ['model_name', 'data_name', target_metric]]
    score_matrix_df = cropped_df.pivot(index="model_name", columns="data_name", values=target_metric)

    # create matrix figure
    fig, ax = plt.subplots()
    if pin_cmap_range:
        cmin, cmax = 0.0, 1.0
    else:
        cmin, cmax = score_matrix_df.min(), score_matrix_df.max()
    ax = sns.heatmap(score_matrix_df, cmap=cmap, ax=ax, vmin=cmin, vmax=cmax)
    plt.xticks(rotation=30, horizontalalignment='right')
    if class_label == AVG_NAME:
        ax.set_title(f"{target_metric} metrics - avg")
    else:
        ax.set_title(f"{target_metric} metrics - {class_label}")
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_fig_path:
        label = "" if class_label == AVG_NAME else f"{class_label}_"
        plt.savefig(os.path.join(save_fig_path, f"{label}{target_metric}_metrics.png"))
    return score_matrix_df, ax


def main():
    """
    Read the model metrics in for both dev and phi
    Plot some sexy graphs
    """
    overwrite_cache = False
    dev_metrics_df = read_model_test_metrics(AwsProfile.dev, overwrite_cache=overwrite_cache)
    phi_metrics_df = read_model_test_metrics(AwsProfile.phi, overwrite_cache=overwrite_cache)
    metrics_df = pd.concat([dev_metrics_df, phi_metrics_df], axis=0)

    fig_path = os.path.join(
        RESULTS_PATH,
        JOB_ID_BASE,
        "figures",
    )

    # ==========
    # create matrix plots for each of the scores: precision, recall, & F1

    precision_df, precision_ax = create_score_matrix(metrics_df, target_metric="precision", save_fig_path=fig_path)
    recall_df, recall_ax = create_score_matrix(metrics_df, target_metric="recall", save_fig_path=fig_path)
    f1_avg_df, f1_avg_ax = create_score_matrix(metrics_df, target_metric="f1-score", save_fig_path=fig_path)

    # create F1 matrices for the specific labels
    class_labels = [label for label in metrics_df.class_label.unique() if label not in ['na', AVG_NAME]]
    f1_by_label = {}
    for label in class_labels:
        f1_df, f1_ax = create_score_matrix(
            metrics_df,
            target_metric="f1-score",
            class_label=label,
            save_fig_path=fig_path)
        f1_by_label[label] = f1_df

    # create bar plot for ensemble F1
    ensemble_f1 = pd.concat(
        [f1_avg_df.loc["ensemble"].rename("avg")]
        + [f1_by_label[label].loc["ensemble"].rename(label) for label in class_labels],
        axis=1)
    ensemble_ax = ensemble_f1.plot(kind="bar")
    plt.xticks(rotation=30, horizontalalignment='right')
    thresh = 0.8
    ensemble_ax.axhline(thresh, color="r", linestyle=":", label="metric prod release threshold")
    plt.ylim((0.0, 1.0))
    plt.ylabel("F1 score")
    plt.title("F1 score by class label over dataset (ensemble)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "ensemble_f1_metrics.png"))

    # collect supporting data
    support_df = read_model_support_metrics()
    ensemble_support_df = support_df[support_df.model_name == "ensemble"] \
        .set_index("data_name") \
        .rename(columns={
            'labeled_token_counts_diagnosis': "diagnosis",
            'labeled_token_counts_signsymptom': "signsymptom"})
    ensemble_support_df['is_phi'] = ~ensemble_support_df.index.str.isdigit()
    ensemble_support_df['positive_token_counts'] = ensemble_support_df['diagnosis'] + ensemble_support_df['signsymptom']

    fig, ax = plt.subplots()
    ensemble_support_df[['diagnosis', 'signsymptom']].plot.bar(ax=ax)
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.ylabel("labeled token count")
    plt.title("Labeled token counts by positive label class")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "support_metrics_labeled_token_counts.png"))

    # look at scatterplot of number of labeled tokens per file; expect a roughly linear relationship
    ensemble_melted = pd.melt(
        ensemble_support_df.reset_index(),
        id_vars=["data_name", "test_num_matched_annotation_files", "is_phi"],
        value_vars=["diagnosis", "signsymptom"],
        var_name="class_label", value_name="labeled_token_count")
    fig, ax = plt.subplots()
    _ = sns.scatterplot(
        data=ensemble_melted,
        x="test_num_matched_annotation_files",
        y="labeled_token_count",
        hue="class_label",
        style="is_phi",
        ax=ax,
    )
    plt.savefig(os.path.join(fig_path, "scatter_filect_by_pos_labeled_token_ct.png"))

    ensemble_melted_tp_tf = pd.melt(
        ensemble_support_df.reset_index(),
        id_vars=["data_name", "test_num_matched_annotation_files", "is_phi"],
        value_vars=["positive_token_counts", "labeled_token_counts_na"],
        var_name="class_label", value_name="token_count")
    fig, ax = plt.subplots()
    _ = sns.scatterplot(
        data=ensemble_melted_tp_tf,
        x="test_num_matched_annotation_files",
        y="token_count",
        hue="class_label",
        style="is_phi",
        ax=ax,
    )
    plt.savefig(os.path.join(fig_path, "scatter_filect_by_total_labeled_token_ct.png"))

    fig, ax = plt.subplots()
    ensemble_melted[ensemble_melted['class_label'] == "diagnosis"].set_index("data_name")[
        "test_num_matched_annotation_files"].plot.bar(ax=ax)
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.ylabel("document count")
    plt.title("Document count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "support_metrics_document_counts.png"))

    fig, ax = plt.subplots()
    plt.plot(
        ensemble_support_df[ensemble_support_df['is_phi']]['labeled_token_counts_na'],
        ensemble_support_df[ensemble_support_df['is_phi']]['positive_token_counts'],
        'ro',
        label="phi")
    plt.plot(
        ensemble_support_df[~ensemble_support_df['is_phi']]['labeled_token_counts_na'],
        ensemble_support_df[~ensemble_support_df['is_phi']]['positive_token_counts'],
        'bo',
        label="non-phi")
    plt.ylabel("TP counts")
    plt.xlabel("TN counts")
    plt.title("TP vs TN counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, "support_metrics_TP_vs_TN.png"))

    return 0


if __name__ == "__main__":
    set_pd_display()
    sys.exit(main())
