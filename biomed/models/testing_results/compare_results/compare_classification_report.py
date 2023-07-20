import pandas as pd
from enum import Enum
import os
from typing import Type

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from biomed.models.testing_reports import ConfusionPrecisionMisclassReport


def compare_testing_report(report_old, report_new) -> pd.DataFrame:
    overlap = pd.merge(report_old, report_new, on=['class_label'], how='outer', suffixes=['_old', '_new'])
    overlap = overlap.fillna(0)
    cols = ['precision', 'recall', 'f1-score']
    diff_df = overlap[['class_label']].copy()

    if diff_df.shape[0] >= 2:
        for c in cols:
            diff_df[f"{c}_change"] = overlap.loc[:, f"{c}_new"] - overlap.loc[:, f"{c}_old"]
    return diff_df


def compare_confusion_matrices(confusion_matrix_old, confusion_matrix_new, report_old, report_new):
    cf_new_cols = report_new.class_label[:-1]
    cf_old_cols = report_old.class_label[:-1]
    # uses the fact that the classification report output is in order of the label number and thus in the same order
    # as the confusion matrix cols

    if report_new.shape == report_old.shape and abs(report_new.support[0] - report_old.support[0]) < 15:
        confusion_diff = confusion_matrix_new - confusion_matrix_old

    elif abs(report_new.support[0] - report_old.support[0]) < 15:
        # get types only in one of the reports
        new_classes = set(cf_new_cols)
        old_classes = set(cf_old_cols)
        diff = new_classes.difference(old_classes).union(old_classes.difference(new_classes))
        for c in diff:
            if c not in cf_new_cols:
                confusion_matrix_new[c] = 0
                confusion_matrix_new.loc[c, :] = 0
            elif c not in cf_old_cols:
                confusion_matrix_old[c] = 0
                confusion_matrix_old.loc[c, :] = 0
        confusion_diff = confusion_matrix_new - confusion_matrix_old
    else:
        confusion_diff = None
    return confusion_diff


def compare_reports(old_results_file_path,
                    new_results_file_path,
                    class_enum: Type[Enum] = None,
                    results_dir: str = None):
    report_name = os.path.split(new_results_file_path)[1]
    confusion_name = report_name.replace('report', 'confusion_matrix')

    old_conf_mat_fp = old_results_file_path.replace('report', 'confusion_matrix')
    new_conf_mat_fp = new_results_file_path.replace('report', 'confusion_matrix')
    if not os.path.isfile(old_results_file_path) or not os.path.isfile(new_results_file_path):
        operations_logger.info('old or new results files were not found')
        return

    report_old = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(
        common.read_text(old_results_file_path), class_enum).reset_index()
    report_new = ConfusionPrecisionMisclassReport.parse_classification_text_to_df(
        common.read_text(new_results_file_path), class_enum).reset_index()

    operations_logger.info(f'report_old: r{report_old.shape}, {report_old.columns}')
    operations_logger.info(f'report new: r{report_new.shape}, {report_new.columns}')
    cf_new_cols = report_new.class_label[:-1]
    cf_old_cols = report_old.class_label[:-1]

    confusion_matrix_new = ConfusionPrecisionMisclassReport.parse_confusion_matrix_text_to_df(
        common.read_text(new_conf_mat_fp), cf_new_cols)
    confusion_matrix_old = ConfusionPrecisionMisclassReport.parse_confusion_matrix_text_to_df(
        common.read_text(old_conf_mat_fp), cf_old_cols)

    confusion_diff = compare_confusion_matrices(confusion_matrix_old, confusion_matrix_new, report_old, report_new)

    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    diff_df = compare_testing_report(report_old, report_new)
    if diff_df.shape[0] >= 2:
        file_info = [''] * diff_df.shape[0]
        file_info[0:2] = [f'old_results_path:{old_results_file_path}', f'new_results_path:{new_results_file_path}']
        diff_df['file_info'] = file_info
        operations_logger.info(f"writing comparison results to"
                               f" {os.path.join(results_dir, report_name.replace('.txt', '.xlsx'))}")

        diff_df.style.applymap(color_negative_red,
                               subset=pd.IndexSlice[:, ['precision_change', 'recall_change', 'f1-score_change']]
                               ).to_excel(os.path.join(results_dir, report_name.replace('.txt', '.xlsx')))
        confusion_diff.style.applymap(color_negative_red).to_excel(
            os.path.join(results_dir, confusion_name.replace('.txt', '.xlsx'))
        )
    else:
        operations_logger.info("DIFF df shape is less than 2 rows")


def get_number(s):
    try:
        float(s)
        return float(s)
    except ValueError:
        return False


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    if isinstance(val, str):
        color = 'black'
    elif val < 0:
        color = 'red'
    elif val > 0:
        color = 'green'
    else:
        color = 'black'

    return 'color: %s' % color
