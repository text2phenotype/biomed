import os
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Callable
)

from biomed.cancer.cancer_represent import select_pref_snomed
from biomed.constants.model_constants import MODEL_TYPE_2_CONSTANTS
from biomed.models.model_cache import ModelMetadataCache
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.common import deserialize_enum
from text2phenotype.constants.features import FeatureType
from text2phenotype.constants.umls import SemTypeCtakesAsserted
from text2phenotype.ocr.data_structures import OCRPageInfo
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.constants.constants import (
    get_version_model_folders,
    ModelType,
    OperationToModelType, OperationToRepresentationFeature)
from biomed.models.model_metadata import ModelMetadata


def get_full_text_from_ocr_pages(ocr_pages: List[OCRPageInfo]) -> Tuple[str, list, list]:
    """
    :param ocr_pages: OCR'ed pages of text
    :return: Full text, coordinates of pages text, coordinates of characters
    """
    coordinate_list = []
    full_text = ''
    coord_chars = []
    for res in ocr_pages:
        full_text += res.text
        coordinate_list.extend(res.coordinates)
        for item in res.coordinates:
            item.text = item.text.replace('"', "''")  # addresses quotation marks differential
            if item.hyphen:
                item.text += '-'

            for character in item.text:
                coord_chars.append({'text': character,
                                    'coordinate': item})
    return full_text, coordinate_list, coord_chars


def annotation_helper(text: str,
                      operations: Set[TaskOperation] = None,
                      features: Set[FeatureType] = None,
                      tid: str = None) -> Tuple[MachineAnnotation, Vectorization]:
    if not features:
        features = feature_list_helper(operations, tid=tid)
    return FeatureServiceClient().annotate_vectorize(text, features, tid=tid)


@text2phenotype_capture_span()
def feature_list_helper(operations: Set[TaskOperation],
                        models: Dict[ModelType, List[str]] = None,
                        tid: str = None,
                        biomed_version: str = None) -> Set[FeatureType]:
    all_features = set()
    all_features.update(get_operation_representation_features(operations=operations))
    for op in operations:
        try:
            enum_op = deserialize_enum(op, TaskOperation)
            # error handling
            if enum_op not in OperationToModelType:
                operations_logger.exception(f'Operation {op} does not have a matched set of model types', tid=tid)
                continue
        except ValueError:
            operations_logger.exception(f'Operation {op} does not have a matched set of model types', tid=tid)
            continue

        model_types = OperationToModelType[enum_op]
        all_features.update(get_model_representation_features(model_types=model_types))
        for mt in model_types:
            all_features.update(get_features_by_model_type(model_type=mt, biomed_version=biomed_version))

    return all_features


def get_model_representation_features(model_types: List[ModelType]) -> set:
    rep_feats = set()
    for mt in model_types:
        m_const = MODEL_TYPE_2_CONSTANTS[mt]
        if m_const.token_umls_representation_feat:
            rep_feats.update(set(m_const.token_umls_representation_feat.keys()))
        if m_const.required_representation_features:
            rep_feats.update(m_const.required_representation_features)
    return rep_feats


def get_operation_representation_features(operations: Set[TaskOperation]):
    rep_feats = set()
    for operation in operations:
        if operation in OperationToRepresentationFeature:
            rep_feats.update(set(OperationToRepresentationFeature[operation]))
    return rep_feats


def get_features_by_model_type(model_type, biomed_version: str = None, model_metadata_cache: ModelMetadataCache = None):
    if not model_metadata_cache:
        model_metadata_cache = ModelMetadataCache()
    all_features = set()
    #SOURCE OF IOPS FROM SSS
    file_list = get_version_model_folders(model_type=model_type, biomed_version=biomed_version)
    for model_file in file_list:
        if len(os.path.split(model_file)[0]) > 1:
            model_type, model_file = os.path.split(model_file)
        metadata = model_metadata_cache.model_metadata(model_type=model_type, model_folder=model_file)
        all_features.update(metadata.features)
    return all_features


def get_prefered_covid_concept(
        tokens: MachineAnnotation, token_index: int,
        expected_sem_types: List[str] = [SemTypeCtakesAsserted.Entity.name,
                                         SemTypeCtakesAsserted.DiseaseDisorder.name]):
    clinical_res = tokens[FeatureType.covid_representation, token_index]
    umls_concept = None
    polarity = None
    if clinical_res:
        for response in clinical_res:
            for key, value in response.items():
                polarity = response.get('polarity')
                if key in expected_sem_types:
                    for entry in value:
                        if umls_concept is None:
                            umls_concept = entry
                        if entry['cui'] == 'C5203676':
                            umls_concept = {
                                'cui': 'C5203676',
                                'tty': 'PT',
                                'preferredText': '2019-nCoV',
                                'code': '840533007',
                                'codingScheme': 'SNOMEDCT_US'}

                            break
    return umls_concept, polarity


def get_first(entries: list):
    if len(entries) > 0:
        return entries[0]


def get_longest_pref_text(entries: list):
    umls_concept = {}
    for concept in entries:
        if len(concept.get('preferredText', '')) > len(umls_concept.get('preferredText', '')):
            umls_concept = concept
    return umls_concept


def get_by_pref_tty(entries: list):
    umls_concept = select_pref_snomed(entries)
    if not umls_concept:
        umls_concept = get_first(entries)
    return umls_concept


def get_pref_umls_concept_polarity(clinical_res, ordered_pref_sem_types: list, pick_best_match: Callable = get_first):
    if clinical_res is not None:
        for sem_type in ordered_pref_sem_types:
            for res in clinical_res:
                if sem_type in res:
                    umls_concept = pick_best_match(res[sem_type])
                    polarity = res.get('polarity')
                    attributes = res.get('attributes')
                    return umls_concept, polarity, attributes
    return None, None, None
