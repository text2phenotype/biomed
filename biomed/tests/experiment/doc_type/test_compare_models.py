#!/usr/bin/env python
from argparse import ArgumentParser
from collections import defaultdict
import os
import sys

from biomed.document_section.data_source import DocumentTypeDataSource

from text2phenotype.common import common
from text2phenotype.common.featureset_annotations import MachineAnnotation, DocumentTypeAnnotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.features.label_types import DocumentTypeLabel


def process_ma_files(ma_dir):
    ma_files = common.get_file_list(ma_dir, '.json', recurse=True)

    ma_map = defaultdict(dict)
    human_ann_count = 0
    for ma_file in ma_files:
        annotations = DocumentTypeAnnotation(MachineAnnotation(json_dict_input=common.read_json(ma_file)))
        doc_type_predictions = annotations.output_dict[FeatureType.document_type]

        d, f = os.path.split(ma_file)
        cohort = d.split('/')[-1]
        chart = os.path.splitext(f)[0]

        ann_file = os.path.join(d.replace('machine_annotations', 'human_annotations'), f'{chart}.ann')
        if not os.path.exists((ann_file)):
            continue

        human_anns = DocumentTypeDataSource.get_brat_label(ann_file, DocumentTypeLabel)
        human_ann_count += len(human_anns)

        label_vectors = DocumentTypeDataSource.match_for_gold(
            annotations.range,
            annotations.tokens,
            human_anns,
            DocumentTypeLabel,
            False)

        if not label_vectors:
            continue

        doc_types = []
        for i in range(len(annotations.range)):
            doc_types.append((doc_type_predictions.input_dict[i][0][0], label_vectors[i]))

        ma_map[cohort][chart] = doc_types

    operations_logger.info(f'Found {len(ma_files)} machine annotation files')
    operations_logger.info(f'Processed {human_ann_count} human annotations (document types)')

    return ma_map


def compute_counts(predictions):
    predict_counts = defaultdict(lambda: defaultdict(int))

    for charts in predictions.values():
        for chart_preds in charts.values():
            for predicted, label_vector in chart_preds:
                for truth_index in range(len(label_vector)):
                    if label_vector[truth_index]:
                        break

                if predicted == 'progress_note':
                    predicted = 'progress_notes'

                predict_counts[DocumentTypeLabel.get_from_int(truth_index).value.persistent_label][predicted] += 1

    return predict_counts


def show_confusion_matrix(predict_counts, labels):
    print('\t'.join(labels))
    for label in labels:
        counts = predict_counts[label]

        row = [label] + [str(counts[l2]) for l2 in labels]

        print('\t'.join(row))


def report_summary_metrics(predict_counts, labels):
    label_format = '{: >24}'
    precision_format = '{:9.3f}'
    recall_format = '{:6.3f}'
    f1_format = '{:8.3f}'
    support_format = '{: >7}'
    print(' '.join([label_format.format(''), 'precision', 'recall', 'f1-score', 'support']))
    total_support = 0
    label_support = dict()
    label_precision = dict()
    label_recall = dict()
    label_f1 = dict()
    for label1 in labels:
        tps = predict_counts[label1][label1]

        total_positives = 0
        label_total = 0
        for label2 in labels:
            total_positives += predict_counts[label2][label1]
            label_total += predict_counts[label1][label2]

        label_support[label1] = label_total

        total_support += label_total

        precision = tps / total_positives
        label_precision[label1] = precision

        recall = tps / label_total
        label_recall[label1] = recall

        f1 = 2 / ((1 / precision) + (1 / recall))
        label_f1[label1] = f1

        row = [label_format.format(label1),
               precision_format.format(precision),
               recall_format.format(recall),
               f1_format.format(f1),
               support_format.format(label_total)]
        print(' '.join(row))

    label_weight = {l: c / total_support for l, c in label_support.items()}
    total_precision = sum(label_precision[label] * label_weight[label] for label in labels)
    total_recall = sum(label_recall[label] * label_weight[label] for label in labels)
    total_f1 = sum(label_f1[label] * label_weight[label] for label in labels)

    row = [label_format.format('avg/total'),
           precision_format.format(total_precision),
           recall_format.format(total_recall),
           f1_format.format(total_f1),
           support_format.format(total_support)]
    print(' '.join(row))


def report_metrics(predictions):
    predict_counts = compute_counts(predictions)
    labels = [l.value.persistent_label for l in DocumentTypeLabel if l != DocumentTypeLabel.na]
    show_confusion_matrix(predict_counts, labels)
    print('\n\n')
    report_summary_metrics(predict_counts, labels)


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--ma', help='The root machine annotation directory')

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    predictions = process_ma_files(args.ma)

    report_metrics(predictions)


if __name__ == '__main__':
    main(sys.argv[1:])
