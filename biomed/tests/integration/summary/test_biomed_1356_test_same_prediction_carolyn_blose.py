import unittest

from text2phenotype.common.data_source import DataSource

from biomed.common.mat_3d_generator import Mat3dGenerator
from biomed.common.helpers import annotation_helper
from biomed.meta.ensembler import Ensembler
from biomed.models.model_metadata import ModelType
from biomed.constants.constants import get_version_model_folders
from text2phenotype.tasks.task_enums import TaskOperation


class TestBiomed1356(unittest.TestCase):
    DataSource().sync_down('emr/OpenEMR', '/tmp/emr/')
    carolyn_blose_txt_filepath = '/tmp/emr/carolyn-blose.pdf.txt'

    def test_predict_generator_real_vectors(self):
        text = self.carolyn_blose_txt_filepath
        tokens, vectors = annotation_helper(text, {TaskOperation.drug})
        ensembler = Ensembler(model_type=ModelType.drug, model_file_list=[get_version_model_folders(ModelType.drug)[1]])

        mat_3d_gen = Mat3dGenerator(vectors=vectors,
                                    num_tokens=len(tokens['token']),
                                    max_window_size=20,
                                    min_window_size=1,
                                    features=ensembler.feature_list,
                                    include_all=True)

        out_full = ensembler.predict(tokens, use_generator=False, ensembler=ensembler, mat_3d=mat_3d_gen,
                                     vectors=vectors)

        output_generator = ensembler.predict(tokens, use_generator=True, ensembler=ensembler, vectors=vectors)

        self.assertTrue(((out_full.predicted_probs - output_generator.predicted_probs) < .000001).all())
