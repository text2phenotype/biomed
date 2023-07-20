import unittest

from biomed.tests.fixtures.example_file_paths import john_stevens_pdf_filepath, sample_image_path
from text2phenotype.annotations.file_helpers import TextCoordinate
from biomed.pdf_embedding.pdf_utilities import (
    get_pdf_page_mapping,
    get_num_pixels,
    read_and_get_full_pdf_writer,
    update_image_to_pdf_coords,
    create_highlight,
    add_highlight_to_page
)


class TestPDFUtilities(unittest.TestCase):
    PDF_PATH = john_stevens_pdf_filepath
    IMG_PATH = sample_image_path
    TEXT_COORD = TextCoordinate(
        text='hello',
        page_index_first=1,
        page_index_last=1,
        document_index_first=120,
        document_index_last=125,
        page=1,
        left=50,
        right=60,
        top=12,
        bottom=51,
        order=1,
        line=3
    )

    def test_read_and_get_full_pdf_writer(self):
        pdf_writer = read_and_get_full_pdf_writer(pdf_file_path=self.PDF_PATH)
        self.assertEqual(pdf_writer.getNumPages(), 3)

    def test_get_pdf_page_mapping(self):
        pdf_page_mapping = get_pdf_page_mapping(read_and_get_full_pdf_writer(self.PDF_PATH))
        self.assertEqual(len(pdf_page_mapping), 3)
        for i in pdf_page_mapping:
            self.assertEqual(i['width'], 612)
            self.assertEqual(i['height'], 792)

    def test_get_num_pixels(self):
        image_dims = get_num_pixels(self.IMG_PATH)
        self.assertEqual(image_dims, (3931, 5087))

    def test_update_image_to_pdf_coords(self):
        updated_dims = update_image_to_pdf_coords(
            text_coords=self.TEXT_COORD,
            page_image_width=500,
            page_image_height=300,
            pdf_width=100,
            pdf_height=200
        )

        self.assertEqual(updated_dims, (10.0, 166.0, 12.0, 192.0))

    def test_create_highlight(self):
        coords = (10.0, 166.0, 12.0, 192.0)
        highlight = create_highlight(coords[0], coords[1], coords[2], coords[3], color=[1, 0, 0])
        self.assertIsInstance(highlight, dict)
        self.assertSetEqual(set(highlight.keys()), {'/F', '/Type', '/Subtype', '/C', '/Rect', '/QuadPoints'})

    def test_add_highlight_to_page(self):
        pdf_writer = read_and_get_full_pdf_writer(self.PDF_PATH)
        coords = (10.0, 166.0, 12.0, 192.0)
        page_no = 0
        highlight = create_highlight(coords[0], coords[1], coords[2], coords[3], color=[1, 0, 0])
        out_pdf_writer, out_pdf_page = add_highlight_to_page(
            highlight=highlight,
            page=pdf_writer.getPage(page_no),
            pdf_writer=pdf_writer)
        self.assertEqual(len(out_pdf_page['/Annots']), 1)
        # test that the highlight dict is as expected on both the page and in the pdf writer object
        self.assertEqual(out_pdf_page['/Annots'][0].getObject(), highlight)

        self.assertEqual(out_pdf_writer.getPage(page_no)['/Annots'][0].getObject(),  highlight)
