import itertools
import typing
from typing import (
    Dict,
    List,
    Optional,
)

from text2phenotype.common.log import operations_logger
from text2phenotype.common.version_info import VersionInfo
from text2phenotype.constants.common import VERSION_INFO_KEY
from text2phenotype.constants.features import (
    DemographicEncounterLabel,
)
from text2phenotype.tasks.task_enums import (
    TaskEnum,
    TaskOperation,
)
from text2phenotype.tasks.task_info import ChunksIterable

from biomed.biomed_env import BiomedEnv
from biomed.constants.constants import BiomedVersionInfo
from biomed.demographic.demographics_manipulation import FetchedDemographics, get_best_demographics

from biomed.reassembler.reassemble_annotations import reassemble_annotations
from biomed.common.biomed_summary import FullSummaryResponse
from biomed.constants.model_constants import ModelType
from biomed.deid.utils import iter_deid_from_demographics
from biomed.diagnosis.remove_family_history import remove_family_history_from_disease

if typing.TYPE_CHECKING:
    from .result_manager import ReassemblerResultManager


def do_nothing(*args, **kwargs):
    pass


def reassemble_summary_chunk_results(chunk_mapping: ChunksIterable,
                                     result_manager: Optional['ReassemblerResultManager'] = None,
                                     **kwargs) -> dict:
    full_response = {}
    full_version_json = None

    for chunk_span, chunk_json in chunk_mapping:
        # check chunk version infos
        full_response, full_version_json = add_to_summary_output(
            full_response=full_response, full_version_json=full_version_json,
            chunk_json=chunk_json, chunk_span=chunk_span
        )

    full_response[VERSION_INFO_KEY] = full_version_json

    if result_manager:
        result_manager.data = full_response

    return full_response


def reassemble_phi_token_chunk_results(chunk_mapping: ChunksIterable,
                                       result_manager: Optional['ReassemblerResultManager'] = None,
                                       other_enum_to_chunk_mapping: Dict[TaskEnum, ChunksIterable] = None):
    if other_enum_to_chunk_mapping and TaskEnum.demographics in other_enum_to_chunk_mapping:
        dem_chunk_to_deid = iter_deid_from_demographics(other_enum_to_chunk_mapping[TaskEnum.demographics])
        chunk_mapping = itertools.chain(chunk_mapping, dem_chunk_to_deid)
    return reassemble_summary_chunk_results(chunk_mapping=chunk_mapping, result_manager=result_manager)


def reassemble_disease_sign(chunk_mapping: ChunksIterable,
                            result_manager: Optional['ReassemblerResultManager'] = None,
                            other_enum_to_chunk_mapping: Dict[TaskEnum, ChunksIterable] = None):
    # uses Family History and Diagnosis output to get filter out family history responses
    disease_summary = reassemble_summary_chunk_results(chunk_mapping=chunk_mapping, result_manager=result_manager)
    if other_enum_to_chunk_mapping and other_enum_to_chunk_mapping.get(TaskEnum.family_history):

        disease_resp = FullSummaryResponse.from_json(disease_summary)
        # filter out family_history
        family_history_reassembled = reassemble_summary_chunk_results(
            chunk_mapping=other_enum_to_chunk_mapping.get(TaskEnum.family_history),
            result_manager=result_manager
        )
        family_resp = FullSummaryResponse.from_json(family_history_reassembled)
        disease_output = remove_family_history_from_disease(disease_resp, family_resp)
        biomed_version = disease_summary[VERSION_INFO_KEY][0].get('biomed_version')

        disease_json = disease_output.to_json(
            task_operation=TaskOperation.disease_sign,
            biomed_version=biomed_version,
            model_type=ModelType.diagnosis)
    else:
        operations_logger.warning("No Family History results found")
        disease_json = disease_summary

    if result_manager:
        result_manager.data = disease_json
    return disease_json


def reassemble_single_list_chunk_results(chunk_mapping: ChunksIterable,
                                         result_manager: Optional['ReassemblerResultManager'] = None) -> list:
    full_response = list()

    for chunk_span, chunk_json in chunk_mapping:
        full_response = add_to_individual_model_output(full_response=full_response,
                                                       chunk_span=chunk_span,
                                                       chunk_json=chunk_json)
    if result_manager:
        result_manager.data = full_response

    return full_response


def reassemble_demographics(chunk_mapping: ChunksIterable,
                            result_manager: Optional['ReassemblerResultManager'] = None,
                            biomed_version: str = None,
                            **kwargs) -> dict:
    full_suggestions = reassemble_summary_chunk_results(chunk_mapping).get(
        DemographicEncounterLabel.get_category_label().persistent_label)

    version = biomed_version or BiomedEnv.DEFAULT_BIOMED_VERSION.value
    fetched_demographics = FetchedDemographics(demographics_list=full_suggestions)
    demographics = get_best_demographics(fetched_demographics).to_final_dict()
    demographics[VERSION_INFO_KEY] = BiomedVersionInfo(TaskOperation.demographics, biomed_version=version).to_dict()

    if result_manager:
        result_manager.data = demographics

    return demographics


def update_json_response_ranges(biomed_response_list: List[dict], text_span: List[int]):
    for biomed_response in biomed_response_list:
        biomed_response['range'][0] += text_span[0]
        biomed_response['range'][1] += text_span[0]
    return biomed_response_list


def add_to_individual_model_output(full_response: list, chunk_span: list, chunk_json: list):
    updated_summary_range = update_json_response_ranges(chunk_json, chunk_span)
    full_response.extend(updated_summary_range)
    return full_response


def add_to_summary_output(full_response: dict, full_version_json: dict, chunk_span: tuple, chunk_json: dict):
    chunk_version_json = chunk_json.get(VERSION_INFO_KEY, [])
    # ensure chunk version dict  is list of dict
    if isinstance(chunk_version_json, dict):
        chunk_version_json = [chunk_version_json]
        operations_logger.warning("Still not using list[dict] output for version info in chunks")

    if not full_version_json:
        full_version_json = chunk_version_json

    check_version_info(full_version_json, chunk_version_json)

    for label_type in chunk_json:
        if label_type != VERSION_INFO_KEY:
            updated_summary_range = update_json_response_ranges(chunk_json[label_type], chunk_span)
            full_response.setdefault(label_type, []).extend(updated_summary_range)
    return full_response, full_version_json


def check_version_info(full_version_json: List[dict], chunk_version_json: List[dict]):
    if full_version_json != chunk_version_json and len(chunk_version_json) >= 1:
        full_version_info = VersionInfo(**full_version_json[0])
        chunk_version_info = VersionInfo(**chunk_version_json[0])
        if (chunk_version_info.product_id == full_version_info.product_id and
                chunk_version_info.product_version != full_version_info.product_version):
            raise ValueError(f'Different Biomed versions were used on different chunks within the document')


TASK_TO_REASSEMBLER_MAPPING: Dict[TaskEnum, callable] = {
    TaskEnum.demographics: reassemble_demographics,
    TaskEnum.doctype: reassemble_summary_chunk_results,
    TaskEnum.phi_tokens: reassemble_phi_token_chunk_results,
    TaskEnum.oncology_only: reassemble_summary_chunk_results,
    TaskEnum.drug: reassemble_summary_chunk_results,
    TaskEnum.lab: reassemble_summary_chunk_results,
    TaskEnum.covid_lab: reassemble_summary_chunk_results,
    TaskEnum.vital_signs: reassemble_summary_chunk_results,
    TaskEnum.imaging_finding: reassemble_summary_chunk_results,
    TaskEnum.disease_sign: reassemble_disease_sign,
    TaskEnum.device_procedure: reassemble_summary_chunk_results,
    TaskEnum.smoking: reassemble_summary_chunk_results,
    TaskEnum.date_of_service: reassemble_summary_chunk_results,
    TaskEnum.icd10_diagnosis: reassemble_summary_chunk_results,
    TaskEnum.genetics: reassemble_summary_chunk_results,
    TaskEnum.bladder_risk: reassemble_summary_chunk_results,
    TaskEnum.family_history: reassemble_summary_chunk_results,
    TaskEnum.sdoh: reassemble_summary_chunk_results,
}

def get_reassemble_function(task_enum: TaskEnum, include_annotations: bool= False):
    if task_enum in TASK_TO_REASSEMBLER_MAPPING:
        return TASK_TO_REASSEMBLER_MAPPING[task_enum]
    elif task_enum in {TaskEnum.annotate} and include_annotations:
        return reassemble_annotations
    else:
        return None
