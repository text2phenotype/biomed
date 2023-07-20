import unittest

from biomed.common.annotation_matching import get_closest_nearby_annotation, MAX_TOKENS_BTWN_ANNOTATION


class TestAnnotationMatching(unittest.TestCase):
    ANNOTATION_INDEXES = [3, 52, 53, 90, 120]


    def test_find_match_no_annot(self):
        annotation_indexes = []
        idx_to_find = 100
        out = get_closest_nearby_annotation(annotation_token_indexes=annotation_indexes,  token_index=idx_to_find)
        self.assertIsNone(out)

    def test_find_match(self):
        self.assertEqual(
            90, get_closest_nearby_annotation(annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=100))
        self.assertEqual(
            120, get_closest_nearby_annotation(annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=105))
        self.assertEqual(
            3, get_closest_nearby_annotation(annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=0))
        self.assertEqual(
            52, get_closest_nearby_annotation(annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=30))
        self.assertEqual(
            120, get_closest_nearby_annotation(annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=140))
        self.assertIsNone(
            get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=120 + MAX_TOKENS_BTWN_ANNOTATION + 1))

    def test_find_match_prefer_after(self):
        self.assertEqual(
            120, get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=100, prefer_annotation_after_token=True))
        self.assertEqual(
            120, get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=105, prefer_annotation_after_token=True))
        self.assertEqual(
            3, get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=0, prefer_annotation_after_token=True))
        self.assertEqual(
            52, get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=30, prefer_annotation_after_token=True))
        self.assertEqual(
            120, get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES, token_index=140, prefer_annotation_after_token=True))
        self.assertIsNone(
            get_closest_nearby_annotation(
                annotation_token_indexes=self.ANNOTATION_INDEXES,
                token_index=120 + MAX_TOKENS_BTWN_ANNOTATION + 1,
                prefer_annotation_after_token=True))
