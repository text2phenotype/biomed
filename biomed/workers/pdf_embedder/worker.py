import os
import tempfile
from typing import List

from biomed.summary.text_to_summary import get_page_indices
from text2phenotype.annotations.file_helpers import (
    TextCoordinateSet
)
from text2phenotype.common.log import operations_logger

from text2phenotype.tasks.rmq_worker import RMQConsumerTaskWorker
from text2phenotype.tasks.task_enums import TaskEnum, WorkType, TaskOperation
from text2phenotype.tasks.task_info import PDFEmbedderTaskInfo, OCRTaskInfo

from biomed.biomed_env import BiomedEnv
from biomed.common.biomed_summary import FullSummaryResponse, combine_all_biomed_outputs
from biomed.pdf_embedding.create_pdf_highlight import embed_write_pdf
from biomed.pdf_embedding.pdf_utilities import get_num_pixels


class PDFEmbeddingWorker(RMQConsumerTaskWorker):
    QUEUE_NAME = BiomedEnv.PDF_EMBEDDER_TASKS_QUEUE.value
    TASK_TYPE = TaskEnum.pdf_embedder
    RESULTS_FILE_EXTENSION = PDFEmbedderTaskInfo.RESULTS_FILE_EXTENSION
    WORK_TYPE = WorkType.document
    NAME = 'PdfEmbedderTaskWorker'
    ROOT_PATH = BiomedEnv.root_dir
    INCLUDED_BIOMED_OPERATIONS = set(TaskOperation.biomed_operations()).difference(
        {TaskOperation.demographics, TaskOperation.phi_tokens}).union(
        {TaskOperation.clinical_summary, TaskOperation.oncology_summary, TaskOperation.covid_specific}
    )

    # don't include phi tokens or demographics in embedding, include all summary operations

    def get_all_operation_outputs(self) -> FullSummaryResponse:
        included_biomed_operations = set(self.work_task.operations).intersection(self.INCLUDED_BIOMED_OPERATIONS)
        operations_logger.info(f"Creating embedded PDF with values from {included_biomed_operations}")
        biomed_operation_output_list = []
        for biomed_op in included_biomed_operations:
            task_enum = TaskEnum(biomed_op.value)
            biomed_operation_output_list.append(self.get_json_results_from_storage(task_enum))
        full_biomed_summary = combine_all_biomed_outputs(biomed_operation_output_list)
        return full_biomed_summary

    def get_image_dims(self, ocr_task: OCRTaskInfo, png_out_dir: str) -> List[dict]:
        num_pages = len(ocr_task.png_pages)
        image_mapping = [None] * num_pages
        for page_no in ocr_task.png_pages:
            png_url = ocr_task.png_pages[page_no]
            local_image_path = self.storage_client.download_file(
                s3_file_key=png_url,
                local_file_name=os.path.join(png_out_dir, os.path.basename(png_url))
            )

            width, height = get_num_pixels(local_image_path)
            image_mapping[int(page_no) - 1] = {'width': width, 'height': height}

        return image_mapping

    def get_source_text(self):
        return self.download_object_str(self.work_task.text_file_key)

    def get_local_source_pdf_path(self, pdf_dir):
        source_file_key = self.work_task.document_info.source_file_key
        operations_logger.info(f'Using source file key {source_file_key}')
        local_pdf_path = os.path.join(pdf_dir, os.path.basename(source_file_key))
        return self.storage_client.download_file(s3_file_key=source_file_key, local_file_name=local_pdf_path)

    def do_work(self) -> PDFEmbedderTaskInfo:
        task_result = self.init_task_result()
        # create full biomed summary with output from all non deid/demographics requested task operations
        work_dir = tempfile.mkdtemp()
        # set up working directories
        png_out_dir = tempfile.mkdtemp(dir=work_dir)
        pdf_out_dir = tempfile.mkdtemp(dir=work_dir)

        if TaskEnum.ocr not in self.work_task.task_statuses:
            operations_logger.warning("You requested an embedded PDF but input a text file,"
                                      " we do not support embedding on text documents")
            return task_result
        else:
            full_summary_resp = self.get_all_operation_outputs()
            # PDF file
            ocr_task: OCRTaskInfo = self.work_task.task_statuses[TaskEnum.ocr]

            text_coord_set = TextCoordinateSet.from_storage(directory_filename=ocr_task.text_coords_directory_file_key,
                                                            lines_filename=ocr_task.text_coords_lines_file_key,
                                                            storage_client=self.storage_client)
            text: str = self.get_source_text()
            page_numbers = get_page_indices(text)
            text_coord_set.update_from_page_ranges(page_numbers=page_numbers)

            source_pdf_path = self.get_local_source_pdf_path(pdf_dir=pdf_out_dir)

            image_dimensions = self.get_image_dims(ocr_task=ocr_task, png_out_dir=png_out_dir)

            out_file_path = embed_write_pdf(
                source_pdf_path=source_pdf_path,
                output_pdf_path=source_pdf_path.replace('.source.pdf', f'.{self.RESULTS_FILE_EXTENSION}'),
                text_coord_set=text_coord_set,
                biomed_summary=full_summary_resp,
                image_dimensions=image_dimensions
            )
            # sync file back to s3 still needs to be done
            self.upload_results_file(local_path_to_results=out_file_path)

            return task_result
