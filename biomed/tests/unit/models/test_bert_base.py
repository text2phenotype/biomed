import os
import shutil
import unittest
import pkg_resources
import tempfile
import nltk
import tensorflow as tf
from transformers import TFBertForTokenClassification

from text2phenotype.common.featureset_annotations import MachineAnnotation
from text2phenotype.constants.environment import Environment
from text2phenotype.common.speech import tokenize
from text2phenotype.common import common

from biomed.common.predict_results import PredictResults
from biomed.models.bert_base import BertBase, EMBEDDINGS_PATH
from biomed.constants.model_enums import ModelType, BertEmbeddings, ModelClass
from biomed.models.model_metadata import ModelMetadata
from biomed.train_test.job_metadata import JobMetadata
from biomed.data_sources.data_source import BiomedDataSource
from biomed.models.data_counts import DataCounter
from biomed.resources import LOCAL_FILES
from biomed import RESULTS_PATH
from biomed.models.data.generator import BERTGenerator


def _get_current_bert_model():
    """Get a current bert model from model constants, rather than hardcoding the name"""
    return 'drug_cbert_20210107_w64'


class BertEmbeddingsTests(unittest.TestCase):
    def test_bert_embeddings(self):
        expected_name_value = {
            "bert": "bert-base-uncased",
            "bio_clinical_bert": "bio_clinical_bert_all_notes_150000",
        }
        self.assertEqual(set(expected_name_value.keys()), set(BertEmbeddings._member_names_))
        for expected_name, folder_name in expected_name_value.items():
            self.assertEqual(folder_name, getattr(BertEmbeddings, expected_name).value)

    def test_bert_embeddings_location(self):
        for name in BertEmbeddings._member_names_:
            self.assertTrue(os.path.isdir(
                os.path.join(EMBEDDINGS_PATH, getattr(BertEmbeddings, name).value)))
            self.assertTrue(os.path.isfile(
                os.path.join(EMBEDDINGS_PATH, getattr(BertEmbeddings, name).value, BertBase.MODEL_FILENAME)))


class BertBaseTests(unittest.TestCase):
    parent_dir = None
    _ANN_DIR_REPO = "annotations"
    _TXT_DIR_REPO = "mtsamples"
    _FS_DIR_REPO = "features"
    _ANN_DIR = None
    _TXT_DIR = None
    _FS_DIR = None
    _MODEL_TYPE = ModelType.drug
    _MODEL_CLASS = ModelClass.bert
    BERT_OUTPUT_VECTOR_SIZE = 768  # size of a single output vector for one subtoken
    BERT_WINDOW_SIZE = 64

    current_bert_model_name = _get_current_bert_model()

    MODEL_FILE_PATH = os.path.join(LOCAL_FILES, _MODEL_TYPE.name, f"{current_bert_model_name}/tf_model.h5")

    CONFIG = {
        "job_id": current_bert_model_name,
        "train": True,
        "test": True,
        "model_type": _MODEL_TYPE,
        "batch_size": 4,
        "window_size": BERT_WINDOW_SIZE,  # whole word window length
        "window_stride": BERT_WINDOW_SIZE,
        "train_embedding_layer": False,
        "embedding_model_name": "bio_clinical_bert",
        # "class_weight": {"1": 400, "2": 400},
        "learning_rate": 5e-5,
        "epochs": 1,
        "features": [], # [None],  # NOTE: this feature isnt actually used, just need something in here
        "async_mode": False,  # false b/c loading mtsamples files, not phi
        "original_raw_text_dirs": ["mtsamples/clean"],
        "ann_dirs": ["annotations"],
        "feature_set_version": "features",
        "feature_set_subfolders": ["train"],
        "testing_fs_subfolders": ["test"],
        "validation_fs_subfolders": ["test"],
    }

    PREDICT_TEXT = """She notes her dizziness to be much worse if she does
quick positional changes such as head turning or sudden movements, no ear
fullness, pressure, humming, buzzing or roaring noted in her ears. She denies
any previous history of sinus surgery or nasal injury. She believes she has
some degree of allergy symptoms."""

    @classmethod
    def setUpClass(cls) -> None:
        # make sure nltk packages are installed
        # we shouldn't have to do this in a docker instance...
        cls._download_nltk_packages()

        cls._disable_storage_svc()

        src_txt_dir = pkg_resources.resource_filename(
            "biomed.tests", os.path.join("fixtures/data", cls._TXT_DIR_REPO)
        )
        src_ann_dir = pkg_resources.resource_filename(
            "biomed.tests", os.path.join("fixtures/data", cls._ANN_DIR_REPO)
        )
        src_fs_dir = pkg_resources.resource_filename(
            "biomed.tests", os.path.join("fixtures/data", cls._FS_DIR_REPO)
        )

        # create tmpdir to hold the test files
        cls.parent_dir = tempfile.TemporaryDirectory()
        cls.CONFIG["parent_dir"] = cls.parent_dir.name
        cls._ANN_DIR = os.path.join(cls.parent_dir.name, cls._TXT_DIR_REPO)
        cls._TXT_DIR = os.path.join(cls.parent_dir.name, cls._ANN_DIR_REPO)
        cls._FS_DIR = os.path.join(cls.parent_dir.name, cls._FS_DIR_REPO)
        shutil.copytree(src_txt_dir, cls._ANN_DIR)
        shutil.copytree(src_ann_dir, cls._TXT_DIR)
        shutil.copytree(src_fs_dir, cls._FS_DIR)
        cls.bert_base = cls._load_class_instance()

    def setUp(self) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        cls.parent_dir.cleanup()

    def tearDown(self) -> None:
        pass

    @staticmethod
    def _download_nltk_packages():
        # nltk packages are required for speech.tokenize()
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

    @classmethod
    def _load_class_instance(cls, embedding_name=None):
        data_source, job_metadata, model_metadata = cls._load_configs()
        if embedding_name:
            model_metadata.embedding_model_name = embedding_name
        bert_base = BertBase(
            model_metadata=model_metadata,
            job_metadata=job_metadata,
            data_source=data_source,
            model_type=cls._MODEL_TYPE,
        )
        return bert_base

    @staticmethod
    def _load_configs():
        """Load model, job, and data configs"""
        model_metadata = ModelMetadata(**BertBaseTests.CONFIG)
        job_metadata = JobMetadata.from_dict(BertBaseTests.CONFIG)
        data_source = BiomedDataSource(**BertBaseTests.CONFIG)
        return data_source, job_metadata, model_metadata

    @classmethod
    def _disable_storage_svc(cls):
        """force the system to read from local folders"""
        os.environ["MDL_COMN_USE_STORAGE_SVC"] = "False"
        Environment.load()

    # --------------------------------
    # test methods

    def test_bad_embedding_name(self):
        with self.assertRaises(ValueError):
            _ = self._load_class_instance("foo")

    def test_alternate_embedding_name(self):
        bert = self._load_class_instance(BertEmbeddings.bio_clinical_bert.name)
        self.assertEqual(BertEmbeddings.bio_clinical_bert.value, bert.embedding_model_name)

    def test_model_initialization(self):
        bert = self._load_class_instance()
        model = bert.initialize_model()
        # self.assertEqual("categorical_crossentropy", model.loss)
        # self.assertEqual(5, len(model.layers))

        # sanity check the max model input size
        self.assertEqual(512, self.bert_base._tokenizer.model_max_length)
        # cursory check that the tokenizer exists
        self.assertEqual("[SEP]", bert._tokenizer.sep_token)
        self.assertEqual("[CLS]", bert._tokenizer.cls_token)

        # Check that the bert layer exists
        self.assertEqual(
            model.layers[0],
            model.layers[bert.bert_layer_ix],
        )
        self.assertEqual("bio_clinical_bert_all_notes_150000", bert.embedding_model_name)

        # check that we have indeed set the bert layer to not be trainable
        self.assertFalse(model.layers[bert.bert_layer_ix].trainable)

    def test_model_load_cached_model(self):
        # NOTE this assumes a model exists in the resources/files/drug_bert folder
        # model_metadata.model_file_path = os.path.join(LOCAL_FILES, "drug_bert", model_metadata.model_file_name)
        bert = self._load_class_instance()
        bert.model_metadata.model_file_path = os.path.join(
            LOCAL_FILES, self._MODEL_TYPE.name, f"{self.current_bert_model_name}/tf_model.h5")

        # should load from resources/files/drug_bert/drug_cbert_20210107_w64
        model = bert.get_cached_model()
        # test that the model loaded correctly by checking the weights
        self.assertIsInstance(model.model, TFBertForTokenClassification)
        # TODO: should also test the existence and utility of the wrapped methods?

    def test_model_load(self):
        # consistency with reality would be to save the model before training,
        # load it, and compare the model.predict outputs
        # NOTE this assumes a model exists in the resources/files/drug_bert folder
        # should load from resources/files/drug_bert/drug_cbert_20210107_w64
        bert = self._load_class_instance()
        bert.model_metadata.model_file_path = os.path.join(
            LOCAL_FILES, self._MODEL_TYPE.name, f"{self.current_bert_model_name}/tf_model.h5")
        model = bert.load_model()
        # test that the model loaded correctly by checking the weights
        weights = model.get_weights()
        self.assertEqual((3,), weights[-1].shape)
        self.assertEqual((768, 3), weights[-2].shape)

    def test_hidden_dropout(self):
        data_source, job_metadata, model_metadata = self._load_configs()
        job_metadata.dropout = 0.5

        bert_base = BertBase(
            model_metadata=model_metadata,
            job_metadata=job_metadata,
            data_source=data_source,
            model_type=self._MODEL_TYPE,
        )
        model = bert_base.initialize_model()
        self.assertEqual(job_metadata.dropout, model.config.hidden_dropout_prob)

        bert_base.save(model)
        model_results_folder = os.path.join(
            RESULTS_PATH, os.path.dirname(bert_base.model_metadata.model_file_name)
        )
        config_path = os.path.join(model_results_folder, "config.json")
        config_out = common.read_json(config_path)
        self.assertEqual(model.config.hidden_dropout_prob, config_out["hidden_dropout_prob"])

    def test_model_train(self):
        # Train and validate on files in fixtures/data
        bert = self._load_class_instance()
        self.assertEqual(os.path.join(RESULTS_PATH, self.current_bert_model_name, self.bert_base.MODEL_FILENAME),
                         bert.model_metadata.model_file_path)
        trained_model_path = bert.train()
        self.assertEqual(os.path.dirname(bert.model_metadata.model_file_path), trained_model_path)
        assert os.path.isfile(bert.model_metadata.model_file_path),\
            f"No model file found: {bert.model_metadata.model_file_path}"

    @unittest.skip("Class weights do not work with BertTokenClassificationLoss")
    def test_model_train_class_weights(self):
        # Train and validate on files in fixtures/data
        bert = self._load_class_instance()
        bert.job_metadata.class_weight = {'1': 9, '2': 9}
        self.assertEqual(os.path.join(RESULTS_PATH, self.current_bert_model_name, self.bert_base.MODEL_FILENAME),
                         bert.model_metadata.model_file_path)
        trained_model_path = bert.train()
        self.assertEqual(os.path.dirname(bert.model_metadata.model_file_path), trained_model_path)
        assert os.path.isfile(bert.model_metadata.model_file_path),\
            f"No model file found: {bert.model_metadata.model_file_path}"

    def test_return_prediction(self):
        # set test model path to cache
        bert = self._load_class_instance()
        bert.model_metadata.model_file_path = os.path.join(
            LOCAL_FILES, self._MODEL_TYPE.name, f"{self.current_bert_model_name}/tf_model.h5")

        # make predict data
        predict_text_tokens_list = tokenize(self.PREDICT_TEXT)
        tokens = [t['token'] for t in predict_text_tokens_list]
        doc_dict = {"tokens": tokens}  # don't use word_token_labels or valid_tokens
        test_counts = DataCounter(
            bert.label2id, n_features=bert.feature_col_size, window_size=bert.model_metadata.window_size
        )
        encoded_doc_dict = bert.windowed_encodings(doc_dict, test_counts)

        # isolate target fields & predict
        x_input_dict = {name: encoded_doc_dict[name] for name in bert.model_input_fields}
        # get prediction
        y_pred_prob = bert.return_prediction(x_input_dict, encoded_doc_dict["valid_token_mask"])
        self.assertEqual((60, 3), y_pred_prob.shape)

    def test_predict_from_tokens(self):
        bert = self._load_class_instance()
        bert.model_metadata.model_file_path = os.path.join(
            LOCAL_FILES, self._MODEL_TYPE.name, f"{self.current_bert_model_name}/tf_model.h5")

        fs_file = common.get_file_list(
            os.path.join(self._FS_DIR, "test", "mtsamples/clean"),
            ".json")[0]
        machine_ann = MachineAnnotation(json_dict_input=common.read_json(fs_file))
        # force the chunk size to be smaller than the doc
        Environment.BIOMED_MAX_DOC_WORD_COUNT.value = 100
        y_pred_prob = bert.predict_from_tokens(machine_ann.tokens)

        self.assertEqual((len(machine_ann), 3), y_pred_prob.shape)


    def test_windowed_encodings_no_label_no_counter(self):
        predict_text_tokens_list = tokenize(self.PREDICT_TEXT)
        tokens = [t['token'] for t in predict_text_tokens_list]
        doc_dict = {"tokens": tokens}
        encoded_doc_dict = self.bert_base.windowed_encodings(doc_dict)

        self.assertEqual(
            {"input_ids", "attention_mask", "encoded_labels", "valid_token_mask"},
            set(encoded_doc_dict.keys()))
        self.assertIsNone(encoded_doc_dict["encoded_labels"])

    def test_model_test(self):
        bert = self._load_class_instance()
        bert.model_metadata.model_file_path = os.path.join(
            LOCAL_FILES, self._MODEL_TYPE.name, f"{self.current_bert_model_name}/tf_model.h5")
        bert.test()
        # check which files are spit out at the end?
        # should write report files to results

    def test_generator(self):
        bert_base = self._load_class_instance(embedding_name='bert')
        tokenizer = bert_base.load_tokenizer(BertEmbeddings.bert.value)
        example_file_cache = {'a':
                                  {'tokens': ["numbness", 'in', 'extremities', '.', 'CURRENT', 'MEDICATIONS', ':', 'Diovan', ',', 'estradiol', ','],
                                   'word_token_labels': [1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0]},
                              'b': {'tokens': ['Wellbutrin', 'SR', 'inhaler', ','],
                                    'word_token_labels': [1, 0, 1, 0]}
                              }
        max_window_length = 8
        expected_tokens = [['[CLS]', 'numb', '##ness', 'in', 'ex', '##tre', '##mit', '[SEP]'],
                           ['[CLS]', '##ies', '.', 'current', 'medications', ':', 'di', '[SEP]'],
                           ['[CLS]', '##ova', '##n', ',', 'est', '##rad', '##iol', '[SEP]'],
                           ['[CLS]', ',', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
                           ['[CLS]', 'well', '##bu', '##tri', '##n', 'sr', 'in', '[SEP]'],
                           ['[CLS]', '##hale', '##r', ',', '[SEP]', '[PAD]', '[PAD]', '[PAD]']]
        expected_labels = [[-100,    1, -100,    0,    0, -100, -100, -100],
                           [-100, -100,    0,    0,    0,    0,    2, -100],
                           [-100, -100, -100,    0,    2, -100, -100, -100],
                           [-100,    0, -100, -100, -100, -100, -100, -100],
                           [-100,    1, -100, -100, -100,    0,    1, -100],
                           [-100, -100, -100,    0, -100, -100, -100, -100]]
        model_input_fields= ['input_ids', 'attention_mask']
        # TODO: maybe a separate test
        encoded_file_cache = {
            k: bert_base.windowed_encodings(example_file_cache[k], window_size=max_window_length)
            for k in example_file_cache
        }
        # should be the same number of elements for each key in a cache
        for f in encoded_file_cache:
            assert len(set([len(v) for v in encoded_file_cache[f].values()]))==1
        dataset = BERTGenerator(['a', 'b'], encoded_file_cache, model_input_fields, max_window_length)
        assert isinstance(dataset.generator, tf.python.data.ops.dataset_ops.FlatMapDataset)
        for i, b in enumerate(dataset.generator.batch(1)):
            subtoken_vec = tokenizer.convert_ids_to_tokens(b[0]['input_ids'][0])
            self.assertEqual(subtoken_vec, expected_tokens[i])
            self.assertEqual(b[1].numpy().tolist()[0], expected_labels[i])
        # should be five total windows
        assert i==5

    def test_find_valid_subtoken_mask(self):
        seq = ["in", "extremities", "."]
        tokenizer = self.bert_base.load_tokenizer(BertEmbeddings.bert.value)
        encoding = tokenizer(seq, return_offsets_mapping=True, is_split_into_words=True)
        # note that we wrap the offset mapping in a list because this is a single window, not a whole doc
        valid_subtoken_mask = self.bert_base.find_valid_subtoken_mask([encoding.offset_mapping])
        expected_mask = [[False, True, True, False, False, False, True, False]]
        self.assertEqual(expected_mask, valid_subtoken_mask)

    def test_predict(self):
        tokens = MachineAnnotation(tokenize('A long time ago in a galaxy far far away'))
        bert = self._load_class_instance()
        output = bert.predict(tokens=tokens)
        self.assertIsInstance(output, PredictResults)
        self.assertEqual(len(output.predicted_probs), len(tokens))

    def test_feature_col_size(self):
        bert = self._load_class_instance()
        self.assertIsNone(bert.get_vectors())
        self.assertEqual(bert.feature_col_size, 2)
        self.assertIsNone(bert.feature_service_client)

    def test_get_doc_encodings(self):
        tokenizer = self.bert_base.load_tokenizer(BertEmbeddings.bert.value)
        tokens = [
            "numbness", 'in', 'extremities', '.', 'CURRENT', 'MEDICATIONS', ':', 'Diovan', ',', 'estradiol', ',',
            'Norvasc', ',',
            'Wellbutrin', 'SR', 'inhaler', ',']
        max_window_length = 8
        encodings = self.bert_base.get_doc_encodings(tokenizer, tokens, max_length=max_window_length)
        n_windows = len(encodings.input_ids)
        expected_n_windows = 6  # we only know this because we know how much this sequence expands
        self.assertEqual(expected_n_windows, n_windows)
        expected_input_ids = [
            [101, 15903, 2791, 1999, 4654, 7913, 22930, 102],
            [101, 3111, 1012, 2783, 20992, 1024, 4487, 102],
            [101, 7103, 2078, 1010, 9765, 12173, 20282, 102],
            [101, 1010, 4496, 12044, 2278, 1010, 2092, 102],
            [101, 8569, 18886, 2078, 5034, 1999, 15238, 102],
            [101, 2099, 1010, 102, 0, 0, 0, 0]]
        expected_subtokens = [
            ['[CLS]', 'numb', '##ness', 'in', 'ex', '##tre', '##mit', '[SEP]'],
            ['[CLS]', '##ies', '.', 'current', 'medications', ':', 'di', '[SEP]'],
            ['[CLS]', '##ova', '##n', ',', 'est', '##rad', '##iol', '[SEP]'],
            ['[CLS]', ',', 'nor', '##vas', '##c', ',', 'well', '[SEP]'],
            ['[CLS]', '##bu', '##tri', '##n', 'sr', 'in', '##hale', '[SEP]'],
            ['[CLS]', '##r', ',', '[SEP]', '[PAD]', '[PAD]', '[PAD]', '[PAD]'],
        ]
        for i in range(n_windows):
            self.assertEqual(expected_input_ids[i], encodings.input_ids[i])
            self.assertEqual(expected_subtokens[i], tokenizer.convert_ids_to_tokens(encodings.input_ids[i]))
        self.assertIn("offset_mapping", encodings.keys())
        self.assertIn("attention_mask", encodings.keys())

    def test_get_doc_encodings_short_doc(self):
        tokenizer = self.bert_base.load_tokenizer(BertEmbeddings.bert.value)
        tokens = [
            "numbness", 'in', 'extremities']
        # this should be 1 more than the length of the sequence (subtokens)
        max_window_length = 10
        encodings = self.bert_base.get_doc_encodings(tokenizer, tokens, max_length=max_window_length)
        n_windows = len(encodings.input_ids)
        expected_n_windows = 1 # should only be one window
        self.assertEqual(expected_n_windows, n_windows)
        # check that the sequence is of maximum length
        self.assertEqual(len(encodings['input_ids'][0]), max_window_length)
        # the last token should be [PAD]
        self.assertEqual(tokenizer.convert_ids_to_tokens(encodings.input_ids[0])[-1], '[PAD]')

    def test_encode_subtoken_labels_pad(self):
        # test has len(token_labels)=6, len(bert_window)=12
        doc_labels = [0, 1, 0, 2, 0, 1]  # 2d, matches whole word length, no cls or sep tags

        # encodings (transformers.BatchEncoding) needs to behave like a dict
        # 3d, includes the cls/sep token, and any extra padding, shape=(n_win, bert_window_len, 2)
        offset_mapping = [[
            [0, 0], [0, 3], [0, 6], [0, 8],
            [0, 2], [2, 3], [3, 7],  # offsets for one whole word
            [0, 5], [0, 9],
            [0, 0], [0, 0], [0, 0],
        ]]
        expected_labels = [[-100, 0, 1, 0, 2, -100, -100, 0, 1, -100, -100, -100]]
        expected_mask = [[False, True, True, True, True, False, False, True, True, False, False, False]]

        subtoken_mask = self.bert_base.find_valid_subtoken_mask(offset_mapping)
        self.assertEqual(expected_mask, subtoken_mask)
        subtoken_labels = self.bert_base.encode_doc_subtoken_labels(doc_labels, subtoken_mask)
        self.assertEqual(expected_labels, subtoken_labels)

        # test responses to padded windows from sliding_window(), -999 is padding value
        doc_labels = [0, 1, 0, 2, 0, -999]
        offset_mapping = [[
            [0, 0], [0, 3], [0, 6], [0, 8],
            [0, 2], [2, 3], [3, 7],  # offsets for one whole word
            [0, 5], [0, 0],
            [0, 0], [0, 0], [0, 0],
        ]]
        expected_labels = [[-100, 0, 1, 0, 2, -100, -100, 0, -100, -100, -100, -100]]
        expected_mask = [[False, True, True, True, True, False, False, True, False, False, False, False]]
        subtoken_mask = self.bert_base.find_valid_subtoken_mask(offset_mapping)
        self.assertEqual(expected_mask, subtoken_mask)
        subtoken_labels = self.bert_base.encode_doc_subtoken_labels(doc_labels, subtoken_mask)
        self.assertEqual(expected_labels, subtoken_labels)

    def test_encode_subtoken_labels_pad_truncation_error(self):
        # known window truncation error
        # 16 labels, extended to 24 subtoken encodings
        # the token text and subtokens are just for illustration
        token_text = (
            "[CLS] in extremities. current medications : diovan, estradiol, norvasc, "
            "wellbutrin SR inhaler, [SEP]")
        subtokens = [
            '[CLS]', 'in', 'ex', '##tre', '##mit', '##ies', '.', 'current', 'medications', ':',
            'di', '##ova', '##n', ',', 'est', '##rad', '##iol', ',', 'nor', '##vas', '##c', ',', 'well', '[SEP]'
        ]
        # Note that the windowed_word_sequence is longer than the number of subtokens,
        # which were truncated to 24 (BERT_WINDOW_LENGTH=24)
        doc_labels = [
            0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0,
            2,  # "wellbutrin", this is expanded and truncated
            # these are truncated
            2, 0, 0
        ]
        offset_mapping = [[
            [0, 0], [0, 2],  # cls, 0,
            [0, 2], [2, 5], [5, 8], [8, 11],  # 0
            [0, 1], [0, 7], [0, 11], [0, 1],  # 0, 0, 0, 0
            [0, 2], [2, 5], [5, 6],  # 2
            [0, 1],  # 0
            [0, 3], [3, 6], [6, 9],  # 2
            [0, 1],  # 0
            [0, 3], [3, 6], [6, 7],  # 2
            [0, 1], [0, 4], [0, 0]  # this is truncated!
        ]]
        expected_labels = [[-100, 0, 0, -100, -100, -100, 0, 0, 0, 0, 2, -100, -100,
              0, 2, -100, -100, 0, 2, -100, -100, 0, 2, -100]]
        expected_mask = [[False, True, True, False, False, False, True, True, True, True, True, False, False,
              True, True, False, False, True, True, False, False, True, True, False]]

        subtoken_mask = self.bert_base.find_valid_subtoken_mask(offset_mapping)
        self.assertEqual(expected_mask, subtoken_mask)
        subtoken_labels = self.bert_base.encode_doc_subtoken_labels(doc_labels, subtoken_mask)
        self.assertEqual(expected_labels, subtoken_labels)

        # try same labels with size 7 windowing in encoding, to force unusual window lengths
        windowed_word_sequence = [
            'in', 'extremities', '.', 'CURRENT', 'MEDICATIONS', ':', 'Diovan', ',', 'estradiol', ',', 'Norvasc', ',',
            'Wellbutrin', 'SR', 'inhaler', ',']
        doc_labels = [
            0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0,
            2,  # "wellbutrin", this is expanded
            2, 0, 0
        ]
        tokenizer = self.bert_base.load_tokenizer(BertEmbeddings.bert.value)
        max_window_length = 8
        windowed_encodings = self.bert_base.get_doc_encodings(tokenizer, windowed_word_sequence,
                                                              max_length=max_window_length)
        # sanity check the windowed_encodings; each window starts and ends with (0,0)
        expected_offset_mapping = [
            [(0, 0), (0, 2), (0, 2), (2, 5), (5, 8), (8, 11), (0, 1), (0, 0)],
            [(0, 0), (0, 7), (0, 11), (0, 1), (0, 2), (2, 5), (5, 6), (0, 0)],
            [(0, 0), (0, 1), (0, 3), (3, 6), (6, 9), (0, 1), (0, 3), (0, 0)],
            [(0, 0), (3, 6), (6, 7), (0, 1), (0, 4), (4, 6), (6, 9), (0, 0)],
            [(0, 0), (9, 10), (0, 2), (0, 2), (2, 6), (6, 7), (0, 1), (0, 0)]]
        self.assertEqual(expected_offset_mapping, windowed_encodings["offset_mapping"])


if __name__ == "__main__":
    unittest.main()
