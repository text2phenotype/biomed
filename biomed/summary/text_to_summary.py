from typing import Tuple

from text2phenotype.constants.common import OCR_PAGE_SPLITTING_KEY

from biomed.constants.response_mapping import KEY_TO_ASPECT_OUTPUT_CLASSES
from biomed.common.biomed_ouput import *
from biomed.common.aspect_response import *
from biomed.common.biomed_summary import FullSummaryResponse


def read_in_entry(entry: Dict[str, List[dict]], summary: FullSummaryResponse):
    for key in entry:
        if key in KEY_TO_ASPECT_OUTPUT_CLASSES:
            output_aspect_responsse, output_biom_out = KEY_TO_ASPECT_OUTPUT_CLASSES[key]

            entries = []
            for pred in entry[key]:
                if isinstance(pred, dict):
                    biom_out = output_biom_out(**pred)
                    entries.append(biom_out)

                elif isinstance(pred, BiomedOutput):
                    entries.append(pred)
                else:
                    operations_logger.info(f'{pred} is not the right format, not adding it to entries ')
            aspect_response = output_aspect_responsse(key, entries)
            summary.add(aspect_response)
    return summary


def summary_from_parts(individual_responses: List[dict], text: str, min_score=DEFAULT_MIN_SCORE,
                       biomed_version: str = None):
    summary = FullSummaryResponse()
    version_info = None
    for entry in individual_responses:
        summary = read_in_entry(entry, summary)
        if version_info is None:
            version_info = entry.get(VERSION_INFO_KEY)
        else:
            token_version_info = entry.get(VERSION_INFO_KEY)
            if isinstance(token_version_info, list) and len(token_version_info) == 1:
                token_version_info = token_version_info[0]
                for model_type in token_version_info.get('model_versions', {}):
                    version_info[0]['model_versions'][model_type] = token_version_info['model_versions'][model_type]
            else:
                operations_logger.warning(f'No version info found for response: {entry.keys()}')

    summary.postprocess(text=text, min_score=min_score)
    out_dict = summary.to_json()
    out_dict[VERSION_INFO_KEY] = version_info
    add_page_numbers_to_predictions(text, out_dict)
    return out_dict


def add_page_numbers_to_predictions(text: str, predictions: dict):
    """
    Add page numbers to API output predictions.
    :param text: The raw chart text.
    :param predictions: The set of model predictions to add page numbers to.
    """
    page_numbers = get_page_indices(text)

    for key, results in predictions.items():
        if key == VERSION_INFO_KEY:
            continue

        for result in results:
            operations_logger.debug(result)

            # range doesn't exist in demographics
            if "range" not in result:
                continue

            text_start, text_end = result["range"]

            for (page_start, page_end), page_no in page_numbers:
                if text_start < page_end and text_end >= page_start:
                    result['page'] = page_no
                    break


def get_page_indices(text: str) -> List[Tuple[Tuple[int, int], int]]:
    """
    Generate listing of chart text to page number.
    :param text: The chart text to process.
    :return: List of ((text start, text end), page number).
    """
    page_no = 1
    page_start = 0
    page_numbers = list()
    if text:
        for page_text in text.split(OCR_PAGE_SPLITTING_KEY[0]):
            page_end = page_start + len(page_text)
            page_numbers.append(((page_start, page_end), page_no))
            page_no += 1
            page_start = page_end + 1

    return page_numbers
