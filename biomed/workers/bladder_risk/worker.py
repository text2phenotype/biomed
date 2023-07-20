from collections import defaultdict
import json

from biomed.biomed_env import BiomedEnv
from biomed.cancer.cancer_represent import TStage, Stage
from biomed.cancer.nmibc import get_bladder_risk_tokens, get_sequoia_risk_tokens
from biomed.constants.constants import BIOMED_VERSION_TO_MODEL_VERSION
from biomed.constants.model_enums import ModelType
from biomed.workers.base_biomed_workers.worker import SingleModelBaseWorker

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.constants.features.label_types import DocumentTypeLabel, CancerLabel, BladderRiskLabel, SequoiaBladderLabel
from text2phenotype.tasks.rmq_worker import RMQConsumerTaskWorker
from text2phenotype.tasks.task_enums import TaskEnum, WorkType, TaskOperation
from text2phenotype.tasks.task_info import BladderRiskTaskInfo, BladderSummaryTaskInfo


class _SequioaSummaryWorker:
    def __init__(self, lvi_threshold=0.9995, recurrence_threshold=0.9975, multifocal_threshold=0.995,
                 grade_threshold=0.975, cis_threshold=0.9, t_stage_thresholds=None, tumor_size_threshold=0.995,
                 behavior_threshold=0.9875):
        self.__lvi_threshold = lvi_threshold
        self.__recurrence_threshold = recurrence_threshold
        self.__multifocal_threshold = multifocal_threshold
        self.__grade_threshold = grade_threshold
        self.__cis_threshold = cis_threshold

        if not t_stage_thresholds:
            t_stage_thresholds = {
                TStage.IN_SITU.name: 0.9875,
                TStage.A.name: 0.9,
                TStage.ONE.name: 0.95,
                TStage.ZERO.name: 0
            }
        self.__t_stage_thresholds = t_stage_thresholds
        self.__tumor_size_threshold = tumor_size_threshold
        self.__behavior_threshold = behavior_threshold

    def do_work(self, doc_types, risk_predictions, onc_predictions) -> dict:
        things_you_need_to_know_about = {"pathology_report": [self.__get_path_reports(doc_types)]}

        self.__process_onc_predictions(things_you_need_to_know_about, onc_predictions)

        self.__add_binary_predictions(risk_predictions,
                                      BladderRiskLabel.lvi.value.persistent_label,
                                      things_you_need_to_know_about,
                                      threshold=self.__lvi_threshold)
        self.__add_binary_predictions(risk_predictions,
                                      BladderRiskLabel.recurrence.value.persistent_label,
                                      things_you_need_to_know_about,
                                      threshold=self.__recurrence_threshold)
        self.__add_binary_predictions(risk_predictions,
                                      SequoiaBladderLabel.bcg.value.persistent_label,
                                      things_you_need_to_know_about)
        self.__add_binary_predictions(risk_predictions,
                                      BladderRiskLabel.pui.value.persistent_label,
                                      things_you_need_to_know_about)
        self.__add_binary_predictions(risk_predictions,
                                      BladderRiskLabel.multifocal.value.persistent_label,
                                      things_you_need_to_know_about,
                                      threshold=self.__multifocal_threshold)
        things_you_need_to_know_about["max_tumor_size"] = [self.__get_max_tumor_size(risk_predictions)]

        self.__make_classification(things_you_need_to_know_about)

        return things_you_need_to_know_about

    # TODO: group by incidence?
    def __process_onc_predictions(self, results: dict, onc_predictions):
        topographies = []
        nmibc_stages = []
        mibc_stages = []
        behaviors = []
        histologies = []
        nmibc = []
        mibc = []
        cis = []

        grade_predictions = defaultdict(list)

        predictions = onc_predictions[CancerLabel.get_category_label().persistent_label]
        for i, prediction in enumerate(predictions):
            label = prediction["label"]
            if label == CancerLabel.grade.value.persistent_label:
                code = prediction["code"]
                if code != "9":     # ignore NOS
                    grade_predictions[code].append(prediction)
            elif label == CancerLabel.behavior.value.persistent_label:
                if prediction["code"] == "3" and prediction["score"] >= self.__behavior_threshold:
                    self.__append_prediction(prediction, behaviors)
            elif label == CancerLabel.topography_primary.value.persistent_label:
                if self.__is_bladder_topography(prediction):
                    self.__append_prediction(prediction, topographies)
            elif label == CancerLabel.stage.value.persistent_label:
                self.__process_stage_prediction(prediction, nmibc_stages, mibc_stages)
            elif label == CancerLabel.morphology.value.persistent_label:
                self.__process_morphology_prediction(predictions, i, nmibc, mibc, cis, histologies)

        results["bladder"] = topographies
        results[CancerLabel.grade.value.persistent_label] = self.__process_grade_predictions(grade_predictions)
        results["nmibc_stage"] = nmibc_stages
        results["mibc_stage"] = mibc_stages
        results[CancerLabel.behavior.value.persistent_label] = behaviors
        results["variant_histology"] = histologies
        results["NMIBC"] = nmibc
        results["MIBC"] = mibc

        stage_set = set(s["T"] for s in nmibc_stages)
        has_tis = TStage.IN_SITU.name in stage_set
        has_ta = TStage.A.name in stage_set or self.__has_ta_behavior(behaviors)

        if has_ta and not has_tis:
            results["CIS"] = []
        else:
            results["CIS"] = cis

    def __process_grade_predictions(self, predictions):
        target_grades = {"3", "4"}

        grades = []

        if len(predictions) == 1:
            for grade, grade_predictions in predictions.items():
                if grade in target_grades:
                    for grade_pred in grade_predictions:
                        self.__append_prediction(grade_pred, grades)
        else:
            for grade in target_grades:
                for grade_pred in predictions.get(grade, []):
                    if grade_pred['score'] >= self.__grade_threshold:
                        self.__append_prediction(grade_pred, grades)

        return grades

    def __process_morphology_prediction(self, predictions, index, nmibc, mibc, cis, histologies):
        cis_codes = {"8010/2", "8120/2"}

        prediction = predictions[index]

        text = prediction["text"].lower()
        if text == "nmibc":
            nmibc.append(prediction)
            return

        if text == "mibc":
            mibc.append(prediction)
            return

        _, cis_end = prediction['range']
        preferred_text = prediction.get("preferredText")
        if preferred_text and preferred_text in cis_codes:
            # attempt to filter out things like "CIS of skin"
            N_PREDICTIONS_TO_CHECK = 10
            MAX_DISTANCE = 50
            for npi in range(1, N_PREDICTIONS_TO_CHECK):
                if index + npi < len(predictions) - 1:
                    next_prediction = predictions[index + npi]
                    if cis_end + MAX_DISTANCE < next_prediction['range'][0]:
                        continue

                    is_topography = next_prediction['label'] in {
                        CancerLabel.topography_primary.value.persistent_label,
                        CancerLabel.topography_metastatic.value.persistent_label
                    }

                    if is_topography and not self.__is_bladder_topography(next_prediction):
                        break
            else:
                if prediction['score'] >= self.__cis_threshold:
                    cis.append(prediction)

            return

        if self.__is_variant_histology(text):
            self.__append_prediction(prediction, histologies)

    def __process_stage_prediction(self, prediction, nmibc_stages, mibc_stages):
        if not prediction["T"] and not prediction["N"] and not prediction["M"] and not prediction["clinical"]:
            # TODO: this shouldn't need to happen here, but not obvious why failing in onc API call
            predicted_stage = Stage.from_string(prediction['text'])
            if predicted_stage.T:
                prediction["T"] = predicted_stage.T.name

        t_stage = prediction["T"]
        if t_stage in self.__t_stage_thresholds:
            threshold = self.__t_stage_thresholds[t_stage]
            if prediction['score'] >= threshold:
                self.__append_prediction(prediction, nmibc_stages)
        elif t_stage == TStage.TWO.name:
            self.__append_prediction(prediction, mibc_stages)

    @staticmethod
    def __is_variant_histology(text):
        # TODO: need to figure out how to handle these
        # "lymphoma-like", "lymphoma", "lymphoma like",
        # "papillary",
        variant_histology_terms = {
            "with", "component", "variant", "differentiation", "features",
            "squamous", "micropapillary", "nested", "plasmacytoid", "neuroendocrine", "sarcomatoid",
            "sarcoma", "carcinosarcoma", "paraganglioma", "microcystic", "giant", "lipid", "small cell",
            "small-cell","lymphoepithelioma", "lymphoepithelial"
        }
        variant_histology_blacklist = {"nonsmall", "non small", "non-small"}

        for variant_term in variant_histology_terms:
            if variant_term in text:
                for blacklist_term in variant_histology_blacklist:
                    if blacklist_term in text:
                        break
                else:
                    return True

        return False

    @staticmethod
    def __is_bladder_topography(prediction):
        preferred_text = prediction.get("preferredText")

        return preferred_text and preferred_text.startswith("C67.")

    @staticmethod
    def __has_ta_behavior(behaviors):
        for behavior in behaviors:
            behavior_text = behavior['text'].lower()
            if 'non' in behavior_text and 'musc' not in behavior_text:
                return True

        return False

    def __make_classification(self, results: dict):
        high_risk = False
        evidence = []

        if results[BladderRiskLabel.lvi.value.persistent_label][0]["present"]:
            high_risk = True
            evidence.append({"text": "LVI"})

        if results["variant_histology"]:
            high_risk = True
            evidence.append({"text": "VH"})

        t_stages = {p["T"] for p in results["nmibc_stage"]}

        has_tis = TStage.IN_SITU.name in t_stages or results["CIS"]
        if has_tis:
            high_risk = True
            evidence.append({"text": "CIS"})

        has_ta = (TStage.A.name in t_stages) or \
                 self.__has_ta_behavior(results[CancerLabel.behavior.value.persistent_label])

        if (TStage.ONE.name not in t_stages) and (not has_ta) and (not has_tis):
            results["classification"] = [{"high_risk": False, "evidence": []}]
            return

        if TStage.ONE.name in t_stages and not has_ta:
            high_risk = True
            evidence.append({"text": "T1"})

        if results[CancerLabel.grade.value.persistent_label]:
            if TStage.ONE.name in t_stages:
                high_risk = True
                evidence.append({"text": "T1_HG"})

            if has_ta:
                if results[BladderRiskLabel.recurrence.value.persistent_label][0]["present"]:
                    high_risk = True
                    evidence.append({"text": "TA_HG_RECURRENCE"})

                if results["max_tumor_size"][0]["size"] > 3:
                    high_risk = True
                    evidence.append({"text": "TA_HG_SIZE"})

                if results[BladderRiskLabel.multifocal.value.persistent_label][0]["present"]:
                    high_risk = True
                    evidence.append({"text": "TA_HG_MULTI"})

                if results[BladderRiskLabel.pui.value.persistent_label][0]["present"]:
                    high_risk = True
                    evidence.append({"text": "TA_HG_PUI"})

        results["classification"] = [{"high_risk": high_risk, "evidence": evidence}]

    @staticmethod
    def __get_path_reports(doc_types):
        pages = []

        for doc_type in doc_types[TaskOperation.doctype.value]:
            if doc_type["label"] == DocumentTypeLabel.pathology.value.persistent_label:
                pages.append(doc_type["page"])

        return {'present': True if pages else False, 'evidence': [{"page": page} for page in pages]}

    def __get_max_tumor_size(self, predictions):
        max_size = 0
        max_prediction = None
        for prediction in predictions.get(BladderRiskLabel.tumor_size.value.persistent_label, []):
            if prediction['score'] < self.__tumor_size_threshold:
                continue

            try:
                tumor_size = float(prediction['text'])
                if tumor_size > max_size:
                    max_size = tumor_size
                    max_prediction = prediction
            except (TypeError, ValueError):
                pass

        return {"size": max_size, "evidence": [max_prediction]}

    @staticmethod
    def __add_binary_predictions(predictions: dict, label: str, results: dict, threshold: float = 0):
        label_predictions = [p for p in predictions[label] if p['score'] >= threshold] if label in predictions else []

        results[label] = [{'present': True if label_predictions else False, 'evidence': label_predictions}]

    @staticmethod
    def __append_prediction(prediction, predict_list):
        if len(predict_list):
            last_prediction = predict_list[-1]

            # looking for adjacent terms, but allow for some tolerance of spacing
            if prediction['label'] == last_prediction['label'] and \
                    prediction['range'][0] - last_prediction['range'][1] < 3:
                last_prediction["score"] = max(last_prediction["score"], prediction["score"])
                last_prediction["text"] += ' ' + prediction["text"]
                last_prediction['range'][1] = prediction['range'][1]
                return

        predict_list.append(prediction)


class BladderSummaryTaskWorker(RMQConsumerTaskWorker):
    ROOT_PATH = BiomedEnv.root_dir
    NAME = 'BladderSummaryTaskWorker'
    RESULTS_FILE_EXTENSION = BladderSummaryTaskInfo.RESULTS_FILE_EXTENSION
    QUEUE_NAME = BiomedEnv.BLADDER_SUMMARY_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.summary_bladder
    WORK_TYPE = WorkType.document

    def do_work(self) -> BladderSummaryTaskInfo:
        task_result = self.init_task_result()

        doc_types = self.get_json_results_from_storage(TaskEnum.doctype)
        risk_predictions = self.get_json_results_from_storage(TaskEnum.bladder_risk)
        onc_predictions = self.get_json_results_from_storage(TaskEnum.oncology_only)

        output = _SequioaSummaryWorker().do_work(doc_types, risk_predictions, onc_predictions)
        self.__add_version_info(output)

        task_result.results_file_key = self.upload_results(json.dumps(output))

        return task_result

    def __add_version_info(self, response):
        job_task = self.get_job_task()

        model_types = [ModelType.sequioa_bladder, ModelType.bladder_risk, ModelType.doc_type]
        version_info = {}
        for model_type in model_types:
            version_info[model_type.name] = BIOMED_VERSION_TO_MODEL_VERSION[job_task.biomed_version][model_type]

        response[VERSION_INFO_KEY] = [{'model_versions': version_info}]



class BladderRiskTaskWorker(SingleModelBaseWorker):
    QUEUE_NAME = BiomedEnv.BLADDER_RISK_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.bladder_risk
    RESULTS_FILE_EXTENSION = BladderRiskTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.chunk
    NAME = 'BladderRiskTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir

    @staticmethod
    def get_predictions(tokens: MachineAnnotation,
                        vectors: Vectorization,
                        biomed_version: str,
                        text: str,
                        tid: str = None) -> dict:
        predictions = get_bladder_risk_tokens(text, tid, tokens, vectors, biomed_version)

        predictions.update(get_sequoia_risk_tokens(text, tid, tokens, vectors, biomed_version))

        return predictions
