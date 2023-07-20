#!/usr/bin/env python
from argparse import ArgumentParser
from collections import defaultdict
import csv
from enum import Enum
import pickle
import string
import sys
from typing import List

from sklearn.neighbors import KNeighborsClassifier

from biomed.cancer.cancer_represent import TStage, NStage, MStage, ClinicalStage, Stage, encode_raw_stage_string,\
    STAGE_VECTOR_LENGTH, CLINICAL_VECTOR_START, T_VECTOR_START, N_VECTOR_START, M_VECTOR_START

from text2phenotype.common.log import operations_logger


################################################################################
# k-NN
################################################################################
def encode_stage(value, enum_type):
    if value:
        for index, label in enumerate(enum_type, 1):
            if label.name == value:
                return index

    return 0


def get_stage_level_vectors(full_vectors):
    xt, yt = [], []
    xn, yn = [], []
    xm, ym = [], []
    xclinical, yclinical = [], []

    for x_vector, t_vector, n_vector, m_vector, clinical_vector in full_vectors:
        xt.append(x_vector[T_VECTOR_START:(T_VECTOR_START+STAGE_VECTOR_LENGTH)])
        yt.append(t_vector)
        xn.append(x_vector[N_VECTOR_START:(N_VECTOR_START+STAGE_VECTOR_LENGTH)])
        yn.append(n_vector)
        xm.append(x_vector[M_VECTOR_START:(M_VECTOR_START+STAGE_VECTOR_LENGTH)])
        ym.append(m_vector)
        xclinical.append(x_vector[CLINICAL_VECTOR_START:(CLINICAL_VECTOR_START+STAGE_VECTOR_LENGTH)])
        yclinical.append(clinical_vector)

    return {'xt': xt, 'yt': yt,
            'xn':xn, 'yn':yn,
            'xm': xm, 'ym': ym,
            'xclinical': xclinical, 'yclinical': yclinical
            }


def encode(data) -> List[List[int]]:
    vectors = []

    for stage_str, t, n, m, clinical in data:
        x_vector = encode_raw_stage_string(stage_str)
        t_vector = encode_stage(t, TStage)
        n_vector = encode_stage(n, NStage)
        m_vector = encode_stage(m, MStage)
        clinical_vector = encode_stage(clinical, ClinicalStage)

        vectors.append((x_vector, t_vector, n_vector, m_vector, clinical_vector))

    return vectors


def run_classifiers(t_train, y_train, x_test, y_test):
    names = [f"Nearest Neighbors ({i})" for i in range(1, 11)]
    classifiers = [KNeighborsClassifier(i) for i in range(1, 11)]

    best_score = 0
    best_class_name = None
    best_classifier = None
    misclassified = []
    for name, classifier in zip(names, classifiers):
        classifier.fit(t_train, y_train)
        score = classifier.score(x_test, y_test)

        # print(f'{name}: {score}')
        if score > best_score:
            best_score = score
            best_class_name = name
            best_classifier = classifier

            misclassified = []
            for i, values in enumerate(zip(classifier.predict(x_test), y_test)):
                predicted, actual = values
                if predicted != actual:
                    misclassified.append((i, actual, predicted))

    return best_score, best_class_name, best_classifier, misclassified


def display_coded_misclassifications(misclassified, enum_type, test_data, actual_index):
    for i, actual, predicted in misclassified:
        predicted = list(enum_type._member_map_)[predicted - 1] if predicted else ''
        print(f'{test_data[i][0]}: ({test_data[i][actual_index]}) ({predicted})')


def process_model_results(score, model_name, classifier, misclassified, stage_enum, stage_label, test_data, exp_col):
    print(f'{stage_label}: {score} ({model_name})')
    display_coded_misclassifications(misclassified, stage_enum, test_data, exp_col)
    with open(f'{stage_label.lower()}_stage_knn.pkl', 'wb+') as fh:
        pickle.dump(classifier, fh)
    print('')


def run_stage_classifier(train_data, test_data):
    train_vectors = encode(train_data)
    test_vectors = encode(test_data)

    train_vectors = get_stage_level_vectors(train_vectors)
    test_vectors = get_stage_level_vectors(test_vectors)

    t_score, t_name, t_classifier, t_misclassified = run_classifiers(train_vectors['xt'], train_vectors['yt'], test_vectors['xt'], test_vectors['yt'])
    n_score, n_name, n_classifier, n_misclassified = run_classifiers(train_vectors['xn'], train_vectors['yn'], test_vectors['xn'], test_vectors['yn'])
    m_score, m_name, m_classifier, m_misclassified = run_classifiers(train_vectors['xm'], train_vectors['ym'], test_vectors['xm'], test_vectors['ym'])
    clinical_score, clinical_name, clinical_classifier, clinical_misclassified = run_classifiers(train_vectors['xclinical'],
                                                                                                 train_vectors['yclinical'],
                                                                                                 test_vectors['xclinical'],
                                                                                                 test_vectors['yclinical'])

    process_model_results(t_score, t_name, t_classifier, t_misclassified, TStage, "T", test_data, 1)
    process_model_results(n_score, n_name, n_classifier, n_misclassified, NStage, "N", test_data, 2)
    process_model_results(m_score, m_name, m_classifier, m_misclassified, MStage, "M", test_data, 3)
    process_model_results(clinical_score, clinical_name, clinical_classifier, clinical_misclassified,
                          ClinicalStage, "Clinical", test_data, 4)

    n_incorrect = len(set(x[0] for x in t_misclassified + n_misclassified + m_misclassified + clinical_misclassified))
    all_correct_count = len(test_data) - n_incorrect
    print(f'\nAll: {all_correct_count} ({all_correct_count / len(test_data)}%)')


################################################################################
# Rule-based processing
################################################################################
class _StageParser:
    @classmethod
    def parse(cls, raw_stage):
        if raw_stage:
            return cls.__parse_raw(raw_stage)
        else:
            return Stage()

    @classmethod
    def __parse_raw(cls, raw_stage: str):
        raw_stage = cls.__clean_raw_stage(raw_stage)

        for c in string.punctuation:
            if c == '|':  # leave this since OCR confuses I/1 with |
                continue

            replace_char = ' ' if c == '/' else ''

            raw_stage = raw_stage.replace(c, replace_char)

        stage = Stage()
        i = 0
        while i < len(raw_stage):
            c = raw_stage[i]

            if c == 't':
                if i < len(raw_stage) - 1:
                    stage.T = cls.__get_tnm_value(raw_stage, i, TStage)
                i += 1
            elif c == 'n':
                if i < len(raw_stage) - 1:
                    stage.N = cls.__get_tnm_value(raw_stage, i, NStage)
                i += 1
            elif c == 'm':
                if i < len(raw_stage) - 1:
                    stage.M = cls.__get_tnm_value(raw_stage, i, MStage)
                i += 1
            elif c == 'i' or c == '1':
                if i == len(raw_stage) - 1 or raw_stage[i + 1] != '-':
                    raw_clinical = cls.__get_clinical_raw_value(raw_stage, i)
                    stage.clinical = cls.__get_clinical_enum(raw_clinical)
                    i += len(raw_clinical) - 1

            i += 1

        return stage

    @staticmethod
    def __clean_raw_stage(raw_stage: str) -> str:
        return raw_stage.lower()\
            .replace('stage', '')\
            .replace('tage', '')\
            .replace('(i-)', ' ')\
            .replace('path', '')\
            .replace('l', '1')\
            .replace('o', '0') \
            .replace('|', 'i') \
            .strip()

    @classmethod
    def __get_tnm_value(cls, raw_stage: str, index: int, stage_enum: Enum) -> Enum:
        s = raw_stage[index + 1]

        return cls.__get_enum(s, stage_enum)

    @classmethod
    def __get_clinical_enum(cls, raw_clinical: str) -> Enum:
        return cls.__get_enum(raw_clinical, ClinicalStage)

    @staticmethod
    def __get_clinical_raw_value(raw_stage: str, index: int) -> str:
        return raw_stage[index:].split()[0]

    @staticmethod
    def __get_enum(raw_str: str, target_enum: Enum) -> Enum:
        for stage in target_enum:
            if raw_str in stage.value:
                return stage


def get_repr(value):
    return '' if not value else value.name


def display_misclassifications(misclassified, test_data, actual_index):
    for i, actual, predicted in misclassified:
        print(f'{test_data[i][0]}: ({test_data[i][actual_index]}) ({predicted})')


def run_parse(test_data):
    t_misclass = []
    n_misclass = []
    m_misclass = []
    clinical_misclass = []
    fails = set()
    for i, row in enumerate(test_data):
        stage_str, t, n, m, clinical = row

        stage = _StageParser.parse(stage_str)

        t_obs = get_repr(stage.T)
        if t != t_obs:
            t_misclass.append((i, t, t_obs))
            fails.add(i)

        n_obs = get_repr(stage.N)
        if n != n_obs:
            n_misclass.append((i, n, n_obs))
            fails.add(i)

        m_obs = get_repr(stage.M)
        if m != m_obs:
            m_misclass.append((i, m, m_obs))
            fails.add(i)

        clinical_obs = get_repr(stage.clinical)
        if clinical != clinical_obs:
            clinical_misclass.append((i, clinical, clinical_obs))
            fails.add(i)

    t_correct = len(test_data) - len(t_misclass)
    print(f'T: {t_correct} ({t_correct / len(test_data)}%)')
    display_misclassifications(t_misclass, test_data, 1)

    n_correct = len(test_data) - len(n_misclass)
    print(f'\nN: {n_correct} ({n_correct / len(test_data)}%)')
    display_misclassifications(n_misclass, test_data, 2)

    m_correct = len(test_data) - len(m_misclass)
    print(f'\nM: {m_correct} ({m_correct / len(test_data)}%)')
    display_misclassifications(m_misclass, test_data, 3)

    clinical_correct = len(test_data) - len(clinical_misclass)
    print(f'\nClinical: {clinical_correct} ({clinical_correct / len(test_data)}%)')
    display_misclassifications(clinical_misclass, test_data, 4)

    all_correct_count = len(test_data) - len(fails)
    print(f'\nAll: {all_correct_count} ({all_correct_count / len(test_data)}%)')


def format_pct(obs_count, total):
    pct = 100 * obs_count / total

    return f'{pct:.3}'


def show_stage_dist(train_counts, test_counts, stage_enum):
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())

    train_count = train_counts[""]
    train_pct = format_pct(train_count, train_total)

    test_count = test_counts[""]
    test_pct = format_pct(test_count, test_total)

    print(f'\tTrain\t\tTest')
    print(f'None\t{train_count} ({train_pct}%)\t{test_count} ({test_pct}%)')
    for stage in stage_enum:
        train_count = train_counts[stage.name]
        train_pct = format_pct(train_count, train_total)

        test_count = test_counts[stage.name]
        test_pct = format_pct(test_count, test_total)
        print(f'{stage.name}\t{train_count} ({train_pct}%)\t{test_count} ({test_pct}%)')

    print(f'Total:\t{train_total}\t\t{test_total}')


def get_stage_counts(data):
    counts = defaultdict(lambda: defaultdict(int))

    for stage_str, t, n, m, clinical in data:
        counts['t'][t] += 1
        counts['n'][n] += 1
        counts['m'][m] += 1
        counts['clinical'][clinical] += 1

    return counts


def show_metrics(train_data, test_data):
    train_counts = get_stage_counts(train_data)
    test_counts = get_stage_counts(test_data)

    operations_logger.info('T stage distribution')
    show_stage_dist(train_counts['t'], test_counts['t'], TStage)

    operations_logger.info('N stage distribution')
    show_stage_dist(train_counts['n'], test_counts['n'], NStage)

    operations_logger.info('M stage distribution')
    show_stage_dist(train_counts['m'], test_counts['m'], MStage)

    operations_logger.info('Clinical stage distribution')
    show_stage_dist(train_counts['clinical'], test_counts['clinical'], ClinicalStage)


def read_data(file_name):
    data = []

    with open(file_name) as fh:
        for row in csv.reader(fh):
            if len(row) > 5:
                # addresses stage raw text that contains a comma b/c I didn't quote the file properly
                row = [','.join(row[:-4])] + row[-4:]

            data.append(row)

    return data


# NOTE: data lives in s3://biomed-data/biomed-raw-data/stage_representation/
def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument('--train', required=True, help='The file containing the training data.')
    parser.add_argument('--test', required=True, help='The file containing the testing data.')

    return parser.parse_args(argv)


def __main(argv):
    args = parse_args(argv)

    train_data = read_data(args.train)
    test_data = read_data(args.test)
    show_metrics(train_data, test_data)

    print('\nTesting parsing method...')
    run_parse(test_data)
    print('')

    print('\nTesting KNN model...')
    run_stage_classifier(train_data, test_data)
    print('')


if __name__ == '__main__':
    __main(sys.argv[1:])
