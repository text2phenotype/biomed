import os
from typing import List

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger


def get_working_dir_file(base_dir, uuid, extension, subfolder=None):
    if subfolder:
        path = os.path.join(base_dir, uuid, subfolder, f'{uuid}.{extension}')
    else:
        path = os.path.join(base_dir, uuid, f'{uuid}.{extension}')
    if os.path.isfile(path):
        return path
    else:
        operations_logger.error(f"No  path found for {path}")


def get_pdf_path(base_dir, uuid):
    return get_working_dir_file(base_dir=base_dir, uuid=uuid, extension='source.pdf')


def get_png_page(base_dir, uuid, page_no):
    return get_working_dir_file(base_dir=base_dir, uuid=uuid, extension=f'page_{page_no+1:04d}.png', subfolder='pages')


def get_biomed_json_files(base_dir, uuid, biomed_extensions: List[str]):
    res_list = []
    for biomed_ext in biomed_extensions:
        biomed_out_path = get_working_dir_file(base_dir=base_dir, uuid=uuid, extension=f'{biomed_ext.strip()}.json')
        if biomed_out_path:
            res_list.append(biomed_out_path)
    return res_list


def get_text_coord_path(base_dir, uuid):
    return get_working_dir_file(base_dir=base_dir, uuid=uuid, extension='text_coordinates')


def get_job_uuids(job_manifest_fp):
    doc_info = common.read_json(job_manifest_fp).get('document_info', {})
    doc_ids = {key for key in doc_info if doc_info[key]['status'] == 'completed - success'}
    return doc_ids

