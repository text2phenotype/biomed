import unittest
from biomed.models.bert_base import BertBase


class TestBiomed2274(unittest.TestCase):
    def test_sanitize(self):
        doc_tokens = ['년']
        self.assertNotEqual(len(doc_tokens[0]), len(doc_tokens[0].encode('utf-8')))
        sanitized = BertBase.sanitize_tokens(doc_tokens)
        self.assertEqual(len(sanitized[0]), len(sanitized[0].encode('utf-8')))
        self.assertEqual(sanitized, ['-'])

    def test_doc_encodings(self):
        doc_tokens = ['년']
        ws  = 64
        tokenizer = BertBase.load_tokenizer('bio_clinical_bert_all_notes_150000')
        doc_encodings = BertBase.get_doc_encodings(
            tokenizer, doc_tokens, max_length=ws
        )
        offset_mapping = doc_encodings['offset_mapping']
        self.assertEqual(len(offset_mapping), 1)
        self.assertEqual(len(offset_mapping[0]), ws)

        expected_tokens = [(0, 0), (0, 1)]
        expected_tokens.extend([(0, 0)] * (ws-2))
        self.assertEqual(offset_mapping[0], expected_tokens)