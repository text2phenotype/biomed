import os
import time
from typing import List, Set, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import entropy

from text2phenotype.apiclients import FeatureServiceClient
from text2phenotype.common import common
from text2phenotype.common.data_source import DataSourceContext
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import LabelEnum, FeatureType

from biomed import RESULTS_PATH
from biomed.biomed_env import BiomedEnv
from biomed.common.mat_3d_generator import Mat3dGenerator
from biomed.common.predict_results import PredictResults, UncertaintyResults
from biomed.constants.constants import ModelType
from biomed.constants.model_constants import ModelConstants, MODEL_TYPE_2_CONSTANTS
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.data_counts import DataCounter
from biomed.models.testing_results.compare_results import compare_classification_report
from biomed.train_test.job_metadata import JobMetadata
from biomed.models.testing_reports import (
    WeightedReport, RemovingAdjacentConfusion, MinusPartialAnnotation, CuiReport, DemographicsReport, FullReport)
from biomed.common.model_test_helpers import document_cui_set, prep_cui_reports


class PredictionEngine(ABC):
    """
    Base class for all downstream things that will create predictions from MachineAnnotation,
    Has an abstract predict method and utilizes that for the testing method"""

    def __init__(self,
                 data_source: BiomedDataSource = None,
                 job_metadata: JobMetadata = None,
                 model_type: ModelType = None,
                 binary_classifier: bool = False
                 ):
        """
        :param data_source: Provided turing a training or testing build to describe the data to be used
        :param job_metadata: Provided during a training or testing build to desribe parameters for the job
        :param model_type: ModelType
        """
        self.model_type = model_type
        self.model_constants: ModelConstants = MODEL_TYPE_2_CONSTANTS[model_type]
        self.binary_classifier: bool = binary_classifier

        # used during jobs only not in production:
        self.data_source: BiomedDataSource = data_source if data_source else BiomedDataSource()
        self.job_metadata: JobMetadata = job_metadata if job_metadata else JobMetadata()
        self._data_support_metrics = {}
        self.feature_service_client: Optional[FeatureServiceClient] = None

    @abstractmethod
    def predict(self, tokens: MachineAnnotation, vectors: Vectorization = None, **kwargs) -> PredictResults:
        raise NotImplementedError

    @property
    @abstractmethod
    def testing_features(self) -> Set[FeatureType]:
        raise NotImplementedError

    @property
    @abstractmethod
    def window_size(self):
        raise NotImplementedError

    @property
    def label_enum(self) -> LabelEnum:
        return self.model_constants.label_class

    @property
    def concept_feature_mapping(self):
        return self.model_constants.token_umls_representation_feat or {}

    @property
    def num_label_classes(self):
        return len(self.label_enum) if not self.binary_classifier else 2

    @staticmethod
    def _read_annotation_file(file_name: str) -> MachineAnnotation:
        return MachineAnnotation(json_dict_input=common.read_json(file_name))

    @staticmethod
    def uncertain_tokens_to_row(predict_results: PredictResults, uncertain_token_idx, file, text: str = None) -> List[
        dict]:
        """
        :param predict_results: Predict results for the document
        :param uncertain_token_idx: List of token idxs that were "uncertain"
        :param file: the filename
        :param text: full text of the document used  for adding context
        :return: List of dictionaries used to create uncertainty pandas DF for reporting
        """
        rows = []
        for token_idx in uncertain_token_idx:
            predicted_category = predict_results.predicted_category[token_idx]
            row = {
                'predicted_category': predicted_category,
                'text': predict_results.tokens[token_idx],
                'predicted_probability': predict_results.predicted_probs[token_idx, predicted_category],
                'span': predict_results.ranges[token_idx],
                'file': file}
            # add context around uncertainty
            if text:
                start_range = row['span'][0] - 30 if row['span'][0] - 30 >= 0 else 0
                row['context'] = text[start_range:row['span'][1] + 30]
            rows.append(row)
        return rows

    @staticmethod
    def get_vectors(annotation: MachineAnnotation, fs_client: FeatureServiceClient, features: List[FeatureType]) -> Vectorization:
        try:
            vectors = fs_client.vectorize(tokens=annotation, features=features)
        except Exception as e:
            operations_logger.error(f"Failed trying to vectorize on features: {features}")
            raise e
        return vectors

    def get_mat_3d_test_generator(
            self,
            vectors: Vectorization,
            num_tokens: int,
            batch_size: int = BiomedEnv.BIOMED_MAX_DOC_WORD_COUNT.value,
            **kwargs) -> Mat3dGenerator:
        # limit generator growth such that returned list only includes generators of batch sizes
        operations_logger.debug(f'Creating generator, num_tokens: {num_tokens}, batch_size: {batch_size}, '
                                f'start_idx={kwargs}')
        generator = Mat3dGenerator(vectors=vectors, batch_size=batch_size,
                                   num_tokens=num_tokens,
                                   max_window_size=self.window_size,
                                   features=self.testing_features, **kwargs)
        return generator

    def token_true_label_list(self, ann_file_path: str, tokens: MachineAnnotation) -> List[int]:
        """
        Convert the brat annotation to a list of label indexes for each token
        :param ann_file_path:
        :param tokens: MachineAnnotation, generally from feature service
        :return: list of integers that map to the label_enum for each token
        """
        # get the brat result
        brat_res = self.data_source.get_brat_label(ann_file_path, self.label_enum)
        matched_vectors = self.data_source.match_for_gold(
            tokens.range, tokens.tokens, brat_res,
            label_enum=self.label_enum,
            binary_classifier=self.binary_classifier)

        test_results = [0] * len(tokens)
        for token_idx, label_vector in matched_vectors.items():
            test_results[token_idx] = np.argmax(label_vector)
        return test_results

    @property
    def label2id(self):
        """
        dict of label strings to id (int)
        eg: {"na": 0, "allergy": 1, "med": 2}
        :return: dict
        """
        return {self.label_enum.get_from_int(i).name: i for i in range(len(self.label_enum))}

    def test(self):
        """
        Runs a test iteration on the test dataset defined in DataSource
        Writes reports for the test results to the results folder
        :return:
        """
        # Requires that you have a model predict function implemented
        # if you are testing by document load one document at a time
        # when using by document and loading multiple files vote_majority and vote_with_weight stop working.
        test_ann_fps, test_fs_fps = self.data_source.get_matched_annotated_files(
            label_enum=self.label_enum,
            context=DataSourceContext.testing)
        test_counts = DataCounter(
            self.label2id,
            n_features=self.feature_col_size,
            window_size=self.model_metadata.window_size,
            window_stride=self.model_metadata.window_stride,
        )

        y_voted_category = np.zeros(0)
        final_pred_with_prob = np.zeros((0, self.num_label_classes))
        all_tokens = list()
        actual_y = list()
        all_rel_match_idx = list()

        # create list of the reports
        confusion_matrix_reports = [
            WeightedReport(self.label_enum),
            RemovingAdjacentConfusion(self.label_enum),
            MinusPartialAnnotation(self.label_enum, self.concept_feature_mapping)
        ]
        if self.job_metadata.full_output:
            full_report = FullReport(self.label_enum)
        if self.model_type == ModelType.demographic:
            confusion_matrix_reports.append(DemographicsReport())
        if self.concept_feature_mapping:
            cui_report = CuiReport(self.label_enum)

        if len(test_ann_fps) == 0:
            error_msg = (
                "Model test: No matching annotation+feature files found. "
                "Check the target dataset confirm existence of matching FS and ann files."
            )
            operations_logger.warning(error_msg)
            return {'y_voted_weight_category': y_voted_category,
                    'y_test_np_category': actual_y,
                    'all_tokens': all_tokens}

        max_failures = np.ceil(self.job_metadata.max_test_failures_pct * len(test_fs_fps))
        num_failures = 0
        for testing_idx in range(len(test_fs_fps)):
            testing_fp = test_fs_fps[testing_idx]
            test_ann = test_ann_fps[testing_idx]
            # get prediction
            tokens = self._read_annotation_file(testing_fp)

            vectors = None
            # skip vectorization if there are no testing_features listed
            if self.testing_features and self.feature_service_client:
                # if  model has representation features, they will be vectorized and passed to the predict method
                if len(self.testing_features) == 1 and next(iter(self.testing_features)) is None:
                    operations_logger.warning(f"Passing null feature list to vectorizer: {self.testing_features}")

                try:
                    vectors = self.get_vectors(
                        tokens,
                        fs_client=self.feature_service_client,
                        features=list(self.testing_features)
                    )
                except Exception:
                    num_failures += 1
                    if num_failures >= max_failures:
                        raise ValueError(f"{num_failures} files failed to be vectorized "
                                         f"({max_failures} max allowed), destroying job")

                    time.sleep(15)  # give FS a chance to reset itself

                    continue
            else:
                if not self.feature_service_client:
                    operations_logger.debug(
                        f"No feature_service_client for this predictor '{self.model_type.name}'")
                if not self.testing_features:
                    operations_logger.info("No features listed, skipping vectorization")

            duplicate_tokens = self.data_source.get_duplicate_token_idx(test_ann, tokens)

            prediction: PredictResults = self.predict(
                tokens=tokens,
                vectors=vectors,
                text=None,
                test_counts=test_counts)
            voted_cat = prediction.predicted_category
            voted_prob = prediction.predicted_probs

            y_true_doc_labels = self.token_true_label_list(test_ann, tokens)

            # Get the testing results for tokens
            # When not an ensemble_test, the ModelBase.predict() and BertBase.predict() methods
            # add documents to the counter.
            # When an ensemble_test, the counter never gets passed to the threaded workers, so we count here
            if test_counts:
                if not test_counts.is_predict_count:
                    n_word_tokens = len(tokens.tokens)
                    n_valid_tokens = len(tokens.valid_tokens()[1])
                    doc_window_count = 0  # ensembles dont think in terms of windows
                    test_counts.add_document(n_word_tokens, n_valid_tokens, y_true_doc_labels, doc_window_count)
                else:  # is from prediction, but we have the total token label counts so increment
                    test_counts.update_doc_label_counts(y_true_doc_labels)

            for report in confusion_matrix_reports:
                report.add_document(
                    tokens=tokens,
                    duplicate_token_idx=duplicate_tokens,
                    expected_category=y_true_doc_labels,
                    predicted_results_cat=voted_cat,
                    predicted_results_prob=voted_prob,
                    filename=test_ann
                )
            # full probability report
            if self.job_metadata.full_output:
                full_report.add_document(
                    tokens=tokens,
                    duplicate_token_idx=duplicate_tokens,
                    expected_category=y_true_doc_labels,
                    predicted_results_cat=voted_cat,
                    predicted_results_prob=voted_prob,
                    raw_probs=prediction.raw_probs,
                    filename=test_ann
                )
            # create cui report
            if self.concept_feature_mapping:
                predicted_cuis = document_cui_set(
                    category_list=voted_cat,
                    concept_feature_mapping=self.concept_feature_mapping,
                    tokens=tokens)

                actual_cuis = document_cui_set(
                    category_list=y_true_doc_labels,
                    concept_feature_mapping=self.concept_feature_mapping,
                    tokens=tokens)

                cui_tokens, actual_cui_cat, predicted_cui_cat = prep_cui_reports(
                    actual_cuis, predicted_cuis)

                cui_report.add_document(
                    expected_category=actual_cui_cat,
                    predicted_results_cat=predicted_cui_cat,
                    tokens=cui_tokens,
                    duplicate_token_idx=None,
                    filename=test_ann)

        if len(test_ann_fps) <= 0:
            operations_logger.error("No annotation feature service files found; check test data folders?")

        operations_logger.debug(
            f'Full token len: {len(all_tokens)}, final_pred_prob: {final_pred_with_prob.shape}, '
            f'final_pred_category shape: {y_voted_category.shape}, '
            f'actual test results: {len(actual_y)}, '
            f'related match indices: {len(all_rel_match_idx)}'
        )
        # ensure results path exists
        os.makedirs(os.path.join(RESULTS_PATH, self.job_metadata.job_id), exist_ok=True)
        for report in confusion_matrix_reports:
            report.write(job_id=self.job_metadata.job_id)
        if self.job_metadata.full_output:
            full_report.write(job_id=self.job_metadata.job_id)
        if self.concept_feature_mapping:
            cui_report.write(job_id=self.job_metadata.job_id)

        # update the data support metrics & write out
        self._load_preexisting_support_metrics()
        self._add_context_support_metrics(test_counts, DataSourceContext.testing)
        self.write_data_support_metrics()

        return

    def active_learning(self, url_base: str = None, dir_part_to_replace: str = None):
        # NOTE: this has not been actively maintained for ~1 year use at your own risk
        # asset uncertainty true for active learning
        self.job_metadata.return_uncertainty = True
        url_base = url_base if url_base else ""
        dir_part_to_replace = dir_part_to_replace if dir_part_to_replace else ""

        self.data_source.get_active_learning_text_files()
        matched_files = self.data_source.get_active_learning_jsons()
        uncertainty_report = []
        feature_service_client = FeatureServiceClient()
        uncertain_df = pd.DataFrame(columns=['predicted_category', 'text', 'predicted_probability', 'span', 'file'])
        for file in matched_files:
            txt_file = self.data_source.get_text_for_active_json(file)

            txt = common.read_text(txt_file) if txt_file else None

            url_path = file.replace(self.data_source.parent_dir, url_base).replace(dir_part_to_replace, '')
            tokens = MachineAnnotation(json_dict_input=common.read_json(file))
            vectors = feature_service_client.vectorize(tokens, features=self.testing_features)

            res = self.predict(tokens, use_generator=False, vectors=vectors)
            uncertainty = self.metaclassifier_return_uncertainty(res, tokens)

            uncertainty_report.append(uncertainty.write_uncertainty_line(file=file, url_path=url_path))

            uncertain_tokens = uncertainty.uncertain_tokens
            if uncertain_tokens:
                uncertain_df = uncertain_df.append(
                    self.uncertain_tokens_to_row(
                        predict_results=res,
                        uncertain_token_idx=uncertain_tokens,
                        file=file,
                        text=txt),
                    sort=False)

        report = pd.DataFrame(uncertainty_report).sort_values(by=['entropy*uncertain_count', 'average_entropy',
                                                                  'uncertain_token_count',
                                                                  'narrow_band_ratio'], ascending=False)
        report = report[['url_path', 'entropy*uncertain_count', 'average_entropy', 'narrow_band_ratio',
                         'uncertain_token_count', 'file_path']]
        if not os.path.isdir(os.path.join(RESULTS_PATH, self.job_metadata.job_id)):
            os.mkdir(os.path.join(RESULTS_PATH, self.job_metadata.job_id))

        report.to_csv(os.path.join(RESULTS_PATH, self.job_metadata.job_id, 'active_learning_list.csv'))

        uncertain_df = uncertain_df[['text', 'predicted_category', 'predicted_probability',
                                     'context', 'span', 'file']].sort_values('text')

        uncertain_tf = uncertain_df.groupby('text').predicted_category.count().reset_index().sort_values(
            'predicted_category', ascending=False)
        uncertain_tf.columns = ['text', 'frequency']
        uncertain_tf.to_csv(os.path.join(RESULTS_PATH, self.job_metadata.job_id, 'uncertain_tf.csv'))

        uncertain_df = pd.merge(uncertain_df, report[['file_path', 'url_path']], left_on=['file'],
                                right_on=['file_path'])
        uncertain_df = pd.merge(uncertain_df, uncertain_tf[['text', 'frequency']], on='text', how='left')

        uncertain_tf_category = uncertain_df.groupby(
            ['predicted_category', 'text']).predicted_probability.count().reset_index().sort_values(
            'predicted_probability', ascending=False)
        uncertain_tf_category.columns = ['Predicted Label', 'uncertain text', 'frequency']
        uncertain_tf_category.to_csv(os.path.join(RESULTS_PATH, self.job_metadata.job_id, 'uncertain_tf_cat.csv'))

        uncertain_df.to_csv(os.path.join(RESULTS_PATH, self.job_metadata.job_id, 'uncertain_tokens_with_context.csv'))

        return report

    def metaclassifier_return_uncertainty(self,
                                          res: PredictResults,
                                          tokens: MachineAnnotation) -> UncertaintyResults:
        total_entropy = 0
        uncertain_count = 0
        uncertain_tokens = set()

        num_tokens = res.predicted_probs.shape[0]

        average_prob = 0.5
        for i in range(min([num_tokens, len(tokens)])):
            total_entropy += entropy(res.predicted_probs[i])
            if np.max(res.predicted_probs[i]) < average_prob + self.job_metadata.narrow_band:
                uncertain_count += 1
                uncertain_tokens.add(i)

        average_entropy = round(float(total_entropy / num_tokens), 3)

        narrow_band_ratio = round(float(uncertain_count / num_tokens), 3)

        return UncertaintyResults(
            average_entropy=average_entropy,
            narrow_band_ratio=narrow_band_ratio,
            narrow_band_width=self.job_metadata.narrow_band,
            uncertain_tokens=uncertain_tokens,
            uncertain_token_count=uncertain_count
        )

    def compare_reports(self, fp):
        new_dir, new_name = os.path.split(fp)
        old_results_dir = new_dir.replace(self.job_metadata.job_id, self.job_metadata.comparison_test_folder)
        self.data_source.sync_down(os.path.join("models/test", self.job_metadata.comparison_test_folder),
                                   old_results_dir)
        old_results_path = os.path.join(old_results_dir, new_name)
        compare_dir = os.path.join(new_dir, self.job_metadata.comparison_test_folder)
        compare_classification_report.compare_reports(
            old_results_path, fp, results_dir=compare_dir, class_enum=self.label_enum)

    def _add_context_support_metrics(self, counts: DataCounter, context: Union[DataSourceContext, str]):
        """
        Update self._data_support_metrics with information from a DataCounter with a given context

        :param counts: DataCounter
            A counter object is used to count the job and document level information
            for a given data context
        :param context:
            The context in which the data will be used, egl train/test/validate
            Can be a string or a DataSourceContext object.
            Used as the prefix for metric key names, eg "{context}_{metric}"
        :return: None
        """
        context_str = context.value if isinstance(context, DataSourceContext) else context
        # these parameters should be identical, independent of context
        base_params = {
            "num_label_classes": counts.n_classes,
            "num_features": counts.n_features,
            "window_size": counts.window_size,
            "window_stride": counts.window_stride
        }
        if "num_label_classes" not in self._data_support_metrics:
            self._data_support_metrics.update(base_params)

        # given the current context (train/test/val/whatever), update the metrics dict
        context_counts = counts.to_json()
        skip_keys = ["n_classes", "n_features", "window_size", "window_stride", "is_predict_count"]

        # only update with the scalar values, we dont need a list of doc level counts
        scalar_counts = {
            f"{context_str}_{k}": v
            for k, v in context_counts.items()
            if isinstance(v, (float, int)) and k not in skip_keys
        }
        self._data_support_metrics.update(scalar_counts)
        self._data_support_metrics.update(
            {f"{context_str}_total_token_label_counts": counts.total_token_label_counts}
        )
        operations_logger.info(
            f"{context_str.upper()} data support metrics: "
            f"n_features={self.feature_col_size}, "
            f"n_classes={self.num_label_classes}, "
            f"n_documents={counts.n_documents}, "
            f"n_windows={counts.total_num_windows}, "
            f"total_token_count={counts.total_token_count}, "
            f"total_token_label_counts={counts.total_token_label_counts}",
            tid=self.job_metadata.job_id,
        )

    def _load_preexisting_support_metrics(self):
        """
        If we already have a data support metrics file saved, load it into self._data_support_metrics
        This allows updates across different contexts (eg train vs test)
        """
        report_file_name = f'data_support_metrics_{self.model_type.name}.json'
        support_file = os.path.join(RESULTS_PATH, self.job_metadata.job_id, report_file_name)
        if os.path.isfile(support_file):
            # if there is already a file, load it
            prior_support = common.read_json(support_file)
            self._data_support_metrics.update(prior_support)

    def write_data_support_metrics(self):
        """
        Write the support metrics for the train/validate/test datasets, eg how many tokens per file

        This method is called in several places, and will overwrite the given json with the most updated
        version of the self._data_support_metrics dict.
        """
        if not self._data_support_metrics:
            operations_logger.debug("No data support metrics found", tid=self.job_metadata.job_id)
            return
        report_file_name = f'data_support_metrics_{self.model_type.name}.json'
        file_path = os.path.join(RESULTS_PATH, self.job_metadata.job_id, report_file_name)
        common.write_json(self._data_support_metrics, file_path)
        operations_logger.debug(f"Wrote data support metrics to {file_path}", tid=self.job_metadata.job_id)
