from PyPDF2 import PdfFileWriter


from typing import List, Set
from text2phenotype.annotations.file_helpers import TextCoordinateSet, TextCoordinate
from biomed.pdf_embedding.pdf_utilities import (
    read_and_get_full_pdf_writer,
    get_pdf_page_mapping,
    add_single_highlight_from_text_coords)

from biomed.common.biomed_summary import FullSummaryResponse
from biomed.pdf_embedding.pdf_bookmarking import create_bookmarks, Bookmark

from text2phenotype.common.log import operations_logger


def get_color_from_label(label_str):
    label_mapping = {
        'diagnosis': [1, 1, 0],
        'signsymptom': [1, .8, 0],
        'med': [.5, 1, 0],
        'medication': [.5, 1, 0],
        'allergy': [1, .4, .4],
        'temperature': [.2, .6, .8],
        'height': [.2, .6, .8],
        'weight': [.2, .6, .8],
        'BMI': [.2, .6, .8],
        'pulse_ox': [.2, .6, .8],
        'heart_rate': [.2, .6, .8],
        'respiratory_rate': [.2, .6, .8],
        'blood_pressure': [.2, .6, .8],
        'lab': [.2, .6, .8],
        'covid_lab': [.2, .6, .8],
        'ct': [.2, .6, .8],
        'xr': [.2, .6, .8],
        'us': [.2, .6, .8],
        'mri': [.2, .6, .8],
        'ecg': [.2, .6, .8],
        'echo': [.2, .6, .8],
        'other': [.2, .6, .8],
        'finding': [.2, .6, .8],
        'morphology': [1, .4, .4],
        'topography_primary': [1, .4, .4],
        'topography_metastatic': [1, .4, .4],
        'behavior': [1, .4, .4],
        'stage': [1, .4, .4],
        'grade': [1, .4, .4],
        'gene_names': [.25, .95, .96],
        'gene_interpretation': [1, 1, 1],
        'encounter_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'admission_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'discharge_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'death_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'admisssion_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'transfer_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'procedure_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'report_date': [0.792156862745098, 0.7803921568627451, 1.0],
        'specimen_date': [0.792156862745098, 0.7803921568627451, 1.0],
        None: [1, 1, 0]
    }

    if label_str in label_mapping:
        return label_mapping[label_str]
    else:
        operations_logger.info(f'No color found for label {label_str}, using dark blue color for highlight')
        return [0, 0, 1]


def create_highlights(
        biomed_summary: FullSummaryResponse,
        pdf_writer: PdfFileWriter,
        text_coord_set: TextCoordinateSet,
        image_dimensions: List[dict]
):
    """
    :param biomed_summary: a full summary  response object (see /common/biomed_summary.py_
    :param pdf_writer: pdf file writer object
    :param text_coord_set: the  TextCoordinateSet loaded from intermediate file produced as part of OCR
    :param image_dimensions: a list of len num pages where each entry {'width':float, 'height':float} of that pages img
    :return:a file writer class with the highlights included
    """

    pdf_mapping = get_pdf_page_mapping(pdf_writer)

    for aspect_resp in biomed_summary.aspect_responses:
        for biomed_resp in aspect_resp.response_list:
            biomed_resp_page = int(biomed_resp.page) - 1
            biomed_resp_range = biomed_resp.range
            text_coord_list: List[TextCoordinate] = text_coord_set.find_coords(
                biomed_resp_range[0],
                biomed_resp_range[1]
            )
            for text_coord in text_coord_list:
                if text_coord.text in biomed_resp.text:
                    pdf_writer, page = add_single_highlight_from_text_coords(
                        text_coords=text_coord,
                        img_width=image_dimensions[biomed_resp_page]['width'],
                        img_height=image_dimensions[biomed_resp_page]['height'],
                        pdf_page_height=pdf_mapping[biomed_resp_page]['height'],
                        pdf_page_width=pdf_mapping[biomed_resp_page]['width'],
                        pdf_page=pdf_mapping[biomed_resp_page]['page'],
                        pdf_writer=pdf_writer,
                        color=get_color_from_label(biomed_resp.label)
                    )
                    pdf_mapping[biomed_resp_page]['page'] = page


def embed_write_pdf(
        source_pdf_path: str,
        output_pdf_path: str,
        biomed_summary: FullSummaryResponse,
        image_dimensions: List[dict],
        text_coord_set: TextCoordinateSet,
        bookmark_hierarchy: List[Bookmark] = None,
        categories_to_include: Set[str] = None
):
    pdf_writer = read_and_get_full_pdf_writer(pdf_file_path=source_pdf_path)

    # add bookmarks
    pdf_writer = create_bookmarks(
        pdf_writer=pdf_writer,
        biomed_summary=biomed_summary,
        bookmark_hierarchy=bookmark_hierarchy,
        categories_to_include=categories_to_include)

    # add highlights
    create_highlights(
        biomed_summary=biomed_summary,
        pdf_writer=pdf_writer,
        text_coord_set=text_coord_set,
        image_dimensions=image_dimensions
    )
    with open(output_pdf_path, 'wb') as out_file:
        pdf_writer.write(out_file)

    return output_pdf_path

