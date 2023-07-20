import unittest

from biomed.pdf_embedding.pdf_utilities import rotate_coords

class TestRotatePDFCoods(unittest.TestCase):
    PAGE_HEIGHT = 100
    PAGE_WIDTH = 50

    def test_rotate_coords(self):
        x1, y1 = 1, 1
        self.assertEqual(
            rotate_coords(x1, y1, page_height=self.PAGE_HEIGHT, page_width=self.PAGE_WIDTH, rotation_angle=0),
            (x1, y1)
        )

        rotated_180 = rotate_coords(x1, y1, page_height=self.PAGE_HEIGHT, page_width=self.PAGE_WIDTH, rotation_angle=180)
        self.assertAlmostEqual(rotated_180[0], self.PAGE_WIDTH-x1)
        self.assertAlmostEqual(rotated_180[1], self.PAGE_HEIGHT-y1)


