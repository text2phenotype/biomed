import copy
import os
from unittest.mock import patch

from fakeredis import FakeStrictRedis

from biomed.pdf_embedding.pdf_utilities import get_num_pixels
from text2phenotype.annotations.file_helpers import TextCoordinateSet
from text2phenotype.common import common
from text2phenotype.constants.common import FileExtensions
from text2phenotype.tasks.document_info import DocumentInfo
from text2phenotype.tasks.task_enums import TaskOperation
from text2phenotype.tasks.task_info import OCRTaskInfo, TaskDependencies, TASK_MAPPING
from text2phenotype.tasks.tasks_constants import TasksConstants
from text2phenotype.tasks.work_tasks import DocumentTask

from biomed.tests.fixtures.example_file_paths import working_dir, uuid
from biomed.workers.pdf_embedder.worker import PDFEmbeddingWorker
from biomed.common.biomed_summary import combine_all_biomed_output_fps
from text2phenotype.tests.fixtures import john_stevens
from text2phenotype.tests.mocks.task_testcase import TaskTestCase


class TestPDFWorker(TaskTestCase):
    OCR_TASK: OCRTaskInfo = OCRTaskInfo(
        png_pages={
            '0001': f'{uuid}.page_0001.png',
            '0002': f'{uuid}.page_0002.png',
            '0003': f'{uuid}.page_0003.png',

        })

    DOCUMENT_ID = uuid

    SOURCE_TEXT_PATH = os.path.join(
        working_dir, f'{uuid}.{TasksConstants.DOCUMENT_TEXT_SUFFIX}.{FileExtensions.TXT.value}')

    SOURCE_PDF_PATH = os.path.join(working_dir, f'{uuid}.source.pdf')
    TEXT_COORD_FP = os.path.join(working_dir, f'{uuid}.text_coordinates')

    def setUp(self) -> None:
        super().setUp()
        self.fake_redis_client = FakeStrictRedis()
        self.document_task = copy.deepcopy(john_stevens.DOCUMENT_TASK_WITH_OCR)
        self.job_task = copy.deepcopy(john_stevens.JOB_TASK)
        self.chunk_task = copy.deepcopy(john_stevens.CHUNK_TASK)
        for item in [self.job_task, self.chunk_task, self.document_task]:
            self.fake_redis_client.set(item.redis_key, item.json())
        self.worker = PDFEmbeddingWorker()
        self.set_initial_work_task(self.document_task)

    def get_document_task(self, operations):
        doc_info = DocumentInfo(document_id=self.DOCUMENT_ID,
                                text_file_key=self.SOURCE_TEXT_PATH,
                                source_file_key=self.SOURCE_PDF_PATH,
                                source='test', tid='123')
        doc_task = DocumentTask(document_info=doc_info,
                                operations=operations,
                                task_statuses=TaskDependencies.get_document_tasks(operations),
                                job_id='job_test123')
        return doc_task

    def test_get_image_files(self):
        dims = self.worker.get_image_dims(
            ocr_task=self.OCR_TASK,
            png_out_dir=os.path.join(working_dir, 'pages')
        )
        self.assertEqual(
            dims,
            [{'width': 3931, 'height': 5087},
             {'width': 3931, 'height': 5087},
             {'width': 3931, 'height': 5087}]
        )

    def test_get_all_operations(self):
        operations = [TaskOperation.clinical_summary]
        work_task = self.get_document_task(operations=operations)
        included_biomed_operations = set(work_task.operations).intersection(self.worker.INCLUDED_BIOMED_OPERATIONS)
        self.assertEqual(included_biomed_operations, set(operations))
        self.assertSetEqual(
            self.worker.INCLUDED_BIOMED_OPERATIONS,
            {TaskOperation.smoking, TaskOperation.doctype, TaskOperation.date_of_service, TaskOperation.disease_sign,
             TaskOperation.drug, TaskOperation.clinical_summary, TaskOperation.covid_specific, TaskOperation.covid_lab,
             TaskOperation.device_procedure, TaskOperation.imaging_finding, TaskOperation.genetics,
             TaskOperation.icd10_diagnosis, TaskOperation.lab, TaskOperation.oncology_summary,
             TaskOperation.oncology_only,TaskOperation.family_history, TaskOperation.vital_signs,
             TaskOperation.bladder_risk, TaskOperation.sdoh
             })


    @patch('biomed.workers.pdf_embedder.worker.PDFEmbeddingWorker.get_all_operation_outputs')
    @patch('text2phenotype.annotations.file_helpers.TextCoordinateSet.from_storage')
    @patch('biomed.workers.pdf_embedder.worker.PDFEmbeddingWorker.get_local_source_pdf_path')
    @patch('biomed.workers.pdf_embedder.worker.PDFEmbeddingWorker.get_image_dims')
    @patch('biomed.workers.pdf_embedder.worker.PDFEmbeddingWorker.get_source_text')
    def test_do_work(self, mock_source_text, mock_image_dims_map, mock_pdf_fp, mock_text_coordss, mock_full_summary_out):
        task_suffixes = [TASK_MAPPING[task_operation].RESULTS_FILE_EXTENSION for task_operation in
                         self.worker.INCLUDED_BIOMED_OPERATIONS]

        biomed_fps = [os.path.join(working_dir, f'{uuid}.{task_suffix}') for task_suffix in task_suffixes]
        biomed_fps = [biomed_fp for biomed_fp in biomed_fps if os.path.isfile(biomed_fp)]

        full_summary = combine_all_biomed_output_fps(biomed_fps)
        text_coordinates = TextCoordinateSet()
        text_coordinates.fill_coordinates_from_stream(open(self.TEXT_COORD_FP, 'rb'))
        image_fps = [os.path.join(working_dir, 'pages', f'{uuid}.page_000{page_no + 1}.png') for page_no in range(3)]
        image_mapping = [None, None, None]
        for i in range(len(image_fps)):
            width, height = get_num_pixels(image_fps[i])
            image_mapping[i] = {'width': width, 'height': height}

        mock_full_summary_out.return_value = full_summary
        mock_text_coordss.return_value = text_coordinates
        mock_pdf_fp.return_value = self.SOURCE_PDF_PATH
        mock_image_dims_map.return_value = image_mapping
        mock_source_text.return_value = common.read_text(self.SOURCE_TEXT_PATH)

        res = self.worker.do_work()

        expected_embedded_pdf_path = os.path.join(working_dir, f'{uuid}.embedded.pdf')
        self.assertTrue(os.path.isfile(expected_embedded_pdf_path))

        # remove the resulting file that is written to the working_dir
        if os.path.isfile(expected_embedded_pdf_path):
            os.remove(expected_embedded_pdf_path)

