import unittest

from biomed.models import data_counts


class TestDataCounter(unittest.TestCase):
    LABEL2ID = {"na": 0, "foo": 10}

    def test_init(self):
        dcounter = data_counts.DataCounter(self.LABEL2ID, 2, window_size=16, window_stride=8)
        self.assertEqual(self.LABEL2ID, dcounter.label2id)
        self.assertEqual({0: "na", 10: "foo"}, dcounter.id2label)
        self.assertEqual(0, dcounter.n_documents)
        self.assertEqual(0, dcounter.total_token_count)
        self.assertEqual(0, dcounter.total_num_windows)
        self.assertEqual(2, dcounter.n_classes)
        self.assertEqual(16, dcounter.window_size)
        self.assertEqual(8, dcounter.window_stride)

    def test_update(self):
        dcounter = data_counts.DataCounter(self.LABEL2ID, n_features=2, window_size=16)
        doc_token_count = 100
        doc_val_token_count = 80
        doc_window_count = 5
        doc_token_labels = [0] * doc_token_count
        n_pos = 5
        pos_label = 10
        for i in range(5, 5 + n_pos):
            doc_token_labels[i] = pos_label

        # call update 5 x to confirm update
        for i in range(1, 6):
            dcounter.add_document(doc_token_count, doc_val_token_count, doc_token_labels, doc_window_count)
            doc_token_label_counts = dcounter.label_ids_to_counts(doc_token_labels)
            self.assertEqual(i, dcounter.n_documents)
            self.assertEqual(doc_token_count * i, dcounter.total_token_count)
            self.assertEqual(doc_val_token_count * i, dcounter.total_valid_token_count)
            self.assertEqual(doc_window_count * i, dcounter.total_num_windows)
            self.assertEqual([doc_token_count] * i, dcounter.doc_token_counts)
            self.assertEqual([doc_val_token_count] * i, dcounter.doc_valid_token_counts)
            self.assertEqual([doc_window_count] * i, dcounter.doc_num_windows)
            self.assertEqual([doc_token_label_counts] * i, dcounter.doc_token_label_counts)
            self.assertEqual(
                {k: v * i for k, v in doc_token_label_counts.items()},
                dcounter.total_token_label_counts)

    def test_to_json(self):
        doc_token_count = 100
        doc_val_token_count = 80
        doc_window_count = 5
        doc_token_labels = [0] * doc_token_count
        n_pos = 5
        pos_label = 10
        for i in range(5, 5 + n_pos):
            doc_token_labels[i] = pos_label

        dcounter = data_counts.DataCounter(self.LABEL2ID, 2, 16)
        dcounter.add_document(doc_token_count, doc_val_token_count, doc_token_labels, doc_window_count)
        expected_dict = {
            'label2id': {'na': 0, 'foo': pos_label},
            'id2label': {0: 'na', pos_label: 'foo'},
            'n_documents': 1,
            'n_classes': 2,
            'n_features': 2,
            'window_size': 16,
            'window_stride': 16,
            'total_token_count': 100,
            'doc_token_counts': [100],
            'total_valid_token_count': 80,
            'doc_valid_token_counts': [80],
            'total_num_windows': 5,
            'doc_num_windows': [5],
            'doc_token_label_counts': [{'na': 95, 'foo': 5}],
            'total_token_label_counts': {'na': 95, 'foo': 5},
            'is_predict_count': False,
        }
        self.assertEqual(expected_dict, dcounter.to_json())

    def test_label_ids_to_counts(self):
        labels = [0, 10, 0, 0, 10, 0, 10, 0]
        dcounter = data_counts.DataCounter(self.LABEL2ID, 2, 16)
        label_counts = dcounter.label_ids_to_counts(labels)
        expected_label_counts = {"na": 5, "foo": 3}
        self.assertEqual(expected_label_counts, label_counts)

        with self.assertRaises(KeyError):
            # 2 isnt in label2id
            _ = dcounter.label_ids_to_counts([0, 2, 0, 0])


if __name__ == '__main__':
    unittest.main()
