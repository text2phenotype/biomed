import unittest
import os
from typing import Dict, List

from text2phenotype.annotations.file_helpers import TextCoordinateSet
from text2phenotype.common import common

from biomed.pdf_embedding.create_pdf_highlight import create_highlights
from biomed.common.biomed_summary import combine_all_biomed_output_fps
from biomed.tests.fixtures import example_file_paths
from biomed.pdf_embedding.pdf_utilities import get_num_pixels, read_and_get_full_pdf_writer


class TestCreateHighlights(unittest.TestCase):
    FULL_SUMMARY = combine_all_biomed_output_fps([example_file_paths.working_clin_summary_fp])
    SOURCE_PDF_FP = example_file_paths.source_pdf_fp
    TEXT_COORDS = TextCoordinateSet()
    TEXT_COORDS.fill_coordinates_from_stream(open(example_file_paths.text_coords_fp))

    @staticmethod
    def create_image_mapping(working_dir) -> List[Dict[str, float]]:
        image_paths = common.get_file_list(working_dir, '.png', True)
        num_pages = len(image_paths)
        image_mapping = [None] * num_pages
        for image_path in image_paths:
            page_no = int(os.path.basename(image_path).split('.')[1].split('_')[1]) - 1
            width, height = get_num_pixels(image_path)
            image_mapping[page_no] = {'width': width, 'height': height}
        return image_mapping

    def test_create_highlights(self):
        pdf_writer = read_and_get_full_pdf_writer(self.SOURCE_PDF_FP)
        create_highlights(
            biomed_summary=self.FULL_SUMMARY,
            pdf_writer=pdf_writer,
            image_dimensions=self.create_image_mapping(working_dir=example_file_paths.working_dir),
            text_coord_set=self.TEXT_COORDS
        )
        # test the number of highlights on each page (this shouldn't change)
        self.assertEqual(len(pdf_writer.getPage(0)['/Annots']), 30)
        self.assertEqual(len(pdf_writer.getPage(1)['/Annots']), 57)
        self.assertEqual(len(pdf_writer.getPage(2)['/Annots']), 25)


