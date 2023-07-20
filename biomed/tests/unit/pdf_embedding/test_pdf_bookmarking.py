import unittest

import PyPDF2

from biomed.common.aspect_response import AspectResponse
from biomed.common.biomed_ouput import BiomedOutput, SummaryOutput
from biomed.pdf_embedding.pdf_bookmarking import (
    BiomedGroupBookmark,
    BiomedCategoryBookmark,
    BiomedTextBookmark,
    BookmarkTree
)


class TestBiomedBookmarks(unittest.TestCase):
    BIO_OUT_1 = SummaryOutput(
        text='diabetes type I',
        preferredText='Type 1 Diabetes Mellitus',
        cui='123',
        label='diagnosis',
        page=2
    )
    BIO_OUT_2 = BiomedOutput(
        text='cancer',
        label='diagnosis',
        page=2
    )
    CATEGORY_NAME = 'DiseaseDisorder'

    def test_biomed_text_bookmark(self):
        self.assertEqual(
            BiomedTextBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1,
                parent_id=None),
            'diabetes type I')
        self.assertEqual(
            BiomedTextBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1,
                parent_id='parent'),
            'parent-diabetes type I')
        self.assertEqual(
            BiomedTextBookmark.get_bookmark_name(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1),
            'diabetes type I'
        )

    def test_biomed_group_bookmark(self):
        self.assertEqual(
            BiomedGroupBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1,
                parent_id=None),
            '123')
        self.assertEqual(
            BiomedGroupBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1,
                parent_id='parent'),
            'parent-123')
        self.assertEqual(
            BiomedGroupBookmark.get_bookmark_name(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_1),
            'Type 1 Diabetes Mellitus'
        )
        self.assertEqual(
            BiomedGroupBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2,
                parent_id=None),
            'diagnosis')
        self.assertEqual(
            BiomedGroupBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2,
                parent_id='parent'),
            'parent-diagnosis')
        self.assertEqual(
            BiomedGroupBookmark.get_bookmark_name(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2),
            'diagnosis_uncoded'
        )

    def test_biomed_category_bookmark(self):
        self.assertEqual(
            BiomedCategoryBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2,
                parent_id=None),
            'DiseaseDisorder')
        self.assertEqual(
            BiomedCategoryBookmark.get_id(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2,
                parent_id='parent'),
            'parent-DiseaseDisorder')
        self.assertEqual(
            BiomedCategoryBookmark.get_bookmark_name(
                category_name=self.CATEGORY_NAME,
                biomed_output=self.BIO_OUT_2),
            'DiseaseDisorder'
        )

    def test_init_text_bookmark_from_biomed_output(self):
        text_bookmark = BiomedTextBookmark.init_from_biomed_output(
            category_name=self.CATEGORY_NAME,
            biomed_output=self.BIO_OUT_1,
            parent_id='parent'
        )
        self.assertIsInstance(text_bookmark, BiomedTextBookmark)
        self.assertEqual(text_bookmark.page, self.BIO_OUT_1.page-1)
        self.assertEqual(text_bookmark.title, self.BIO_OUT_1.text)

    def test_init_group_bookmark_from_biomed_output(self):
        text_bookmark = BiomedGroupBookmark.init_from_biomed_output(
            category_name=self.CATEGORY_NAME,
            biomed_output=self.BIO_OUT_1,
            parent_id='parent')
        self.assertIsInstance(text_bookmark, BiomedGroupBookmark)
        self.assertEqual(text_bookmark.page, self.BIO_OUT_1.page-1)
        self.assertEqual(text_bookmark.title, self.BIO_OUT_1.preferredText)

    def test_init_category_bookmark_from_biomed_output(self):
        text_bookmark = BiomedCategoryBookmark.init_from_biomed_output(
            category_name=self.CATEGORY_NAME,
            biomed_output=self.BIO_OUT_1,
            parent_id='parent')
        self.assertIsInstance(text_bookmark, BiomedCategoryBookmark)
        self.assertEqual(text_bookmark.page, self.BIO_OUT_1.page-1)
        self.assertEqual(text_bookmark.title, self.CATEGORY_NAME)


class TestBiomedBookmarkTre(unittest.TestCase):
    BIO_OUT_1 = SummaryOutput(
        text='diabetes type I',
        preferredText='Type 1 Diabetes Mellitus',
        cui='123',
        label='diagnosis',
        page=5
    )
    BIO_OUT_2 = SummaryOutput(
        text='cancer',
        label='diagnosis',
        page=2
    )
    BIO_OUT_3 = SummaryOutput(
        text='diabetes type 1',
        preferredText='Type 1 Diabetes Mellitus',
        cui='123',
        label='diagnosis',
        page=3
    )
    CATEGORY_NAME = 'DiseaseDisorder'
    ASPECT_RESPONSE = AspectResponse(category_name=CATEGORY_NAME, response_list=[BIO_OUT_2, BIO_OUT_1, BIO_OUT_3])

    def test_base_bookmark_tree_add_biomed_output(self):
        base_bookmark_tree = BookmarkTree()
        self.assertEqual(base_bookmark_tree.hierarchy,
                         [BiomedCategoryBookmark, BiomedGroupBookmark, BiomedTextBookmark])

        base_bookmark_tree.add_biomed_output_bookmark(category_name=self.CATEGORY_NAME, biomed_output=self.BIO_OUT_1)
        # assert that when adding a base concept to the tree that we create the bookmarks for all levels
        # represented w.in heriarchy
        self.assertEqual(len(base_bookmark_tree.bookmark_id_mapping), len(base_bookmark_tree.hierarchy))
        # assert that there is only 1 top level added for the concept
        self.assertEqual(len(base_bookmark_tree.top_level_ids), 1)

        self.assertSetEqual(
            set(base_bookmark_tree.bookmark_id_mapping.keys()),
            {'DiseaseDisorder',
             f'DiseaseDisorder-{self.BIO_OUT_1.cui}',
             f'DiseaseDisorder-{self.BIO_OUT_1.cui}-{self.BIO_OUT_1.text}_{self.BIO_OUT_1.page}'}
        )

    def test_base_bookmark_tree_add_aspect_response(self):
        base_bookmark_tree = BookmarkTree()
        base_bookmark_tree.ingest_aspect_response(self.ASPECT_RESPONSE)
        self.assertEqual(base_bookmark_tree.top_level_ids, ['DiseaseDisorder'])
        self.assertEqual(len(base_bookmark_tree.bookmark_id_mapping),
                         1 + 2 + 3)  # top level + unique cuis + unique texts
        self.assertEqual(base_bookmark_tree.bookmark_id_mapping['DiseaseDisorder-123'].page, 2)

    def test_setting_book_hierarchy(self):
        bookmark_tree = BookmarkTree([BiomedGroupBookmark, BiomedCategoryBookmark])
        bookmark_tree.ingest_aspect_response(self.ASPECT_RESPONSE)
        self.assertEqual(set(bookmark_tree.top_level_ids), {'123', 'diagnosis'})
        self.assertEqual(
            len(bookmark_tree.bookmark_id_mapping),
            2 + 3)  # unique combinations of ctegory, cui and page
        self.assertEqual(bookmark_tree.bookmark_id_mapping['123'].page, 2)
        self.assertEqual(
            set(bookmark_tree.bookmark_id_mapping['123'].child_ids),
            {'123-DiseaseDisorder_5', '123-DiseaseDisorder_3'})
        self.assertEqual(bookmark_tree.bookmark_id_mapping['123-DiseaseDisorder_5'].parent_id , '123')
        self.assertSetEqual(
            {'diagnosis', 'diagnosis-DiseaseDisorder_2', '123', '123-DiseaseDisorder_5', '123-DiseaseDisorder_3'},
            set(bookmark_tree.bookmark_id_mapping.keys()))

    def test_add_bookmarks_to_pdf_writer(self):
        pdf_writer = PyPDF2.PdfFileWriter()
        for i in range(5): # 5 is the max page number in the inputs
            pdf_writer.addBlankPage(width=100, height=100)

        bookmark_tree = BookmarkTree()
        bookmark_tree.ingest_aspect_response(self.ASPECT_RESPONSE)
        pdf_writer = bookmark_tree.add_to_pdf(pdf_writer=pdf_writer)
        pdf_writer_bookmarks = [obj for obj in pdf_writer._objects if '/Title' in obj]
        expected_bookmarks = [
            {'/A': 'IndirectObject(8, 0)', '/Title': 'DiseaseDisorder', '/F': 2, '/Parent': 'IndirectObject(9, 0)',
            '/First': 'IndirectObject(12, 0)', '/Count': 2, '/Last': 'IndirectObject(14, 0)'},
            {'/A': 'IndirectObject(11, 0)', '/Title': 'Type 1 Diabetes Mellitus', '/Parent': 'IndirectObject(10, 0)',
         '/Next': 'IndirectObject(14, 0)', '/First': 'IndirectObject(16, 0)', '/Count': 2, '/Last': 'IndirectObject(18, 0)'},
            {'/A': 'IndirectObject(13, 0)', '/Title': 'diagnosis_uncoded', '/Prev': 'IndirectObject(12, 0)',
             '/Parent': 'IndirectObject(10, 0)', '/First': 'IndirectObject(20, 0)', '/Count': 1, '/Last': 'IndirectObject(20, 0)'},
            {'/A': 'IndirectObject(15, 0)', '/Title': 'diabetes type I', '/Parent': 'IndirectObject(12, 0)',
             '/Next': 'IndirectObject(18, 0)'},
            {'/A': 'IndirectObject(17, 0)', '/Title': 'diabetes type 1', '/Prev': 'IndirectObject(16, 0)',
             '/Parent': 'IndirectObject(12, 0)'},
            {'/A': 'IndirectObject(19, 0)', '/Title': 'cancer', '/Parent': 'IndirectObject(14, 0)'}]
        # as far as I can tell /Title is the bookmark name, and /Count is number of children
        self.assertEqual(len(expected_bookmarks), len(expected_bookmarks))
        # partially check title set
        expected_bookmark_titles = set([a['/Title'] for a in expected_bookmarks])
        pdf_writer_bookmark_titles = set([b['/Title'] for b in pdf_writer_bookmarks])
        self.assertSetEqual(expected_bookmark_titles, pdf_writer_bookmark_titles)
