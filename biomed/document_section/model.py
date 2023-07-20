import numpy as np
from text2phenotype.common.data_source import DataSource
from text2phenotype.common.featureset_annotations import DocumentTypeAnnotation, MachineAnnotation, Vectorization

from biomed.models.model_base import ModelBase
from biomed.models.model_metadata import ModelMetadata
from biomed.constants.constants import ModelType
from biomed.train_test.job_metadata import JobMetadata


# TODO: page boundaries
# TODO: frequency of first/last
# TODO: titles?
class DocumentTypeModel(ModelBase):
    def __init__(self,
                 model_folder_name: str = None,
                 model_metadata: ModelMetadata = None,
                 data_source: DataSource = None,
                 job_metadata: JobMetadata = None,
                 **kwargs):
        super().__init__(
            model_folder_name=model_folder_name,
            model_metadata=model_metadata,
            data_source=data_source,
            job_metadata=job_metadata,
            model_type=ModelType.doc_type)

    @staticmethod
    def to_doc_type_annotation(annotation) -> DocumentTypeAnnotation:
        if not isinstance(annotation, DocumentTypeAnnotation):
            annotation = DocumentTypeAnnotation(annotation)

        return annotation

    def predict(self,
                tokens: MachineAnnotation,
                vectors: Vectorization = None,
                mat_3d: np.ndarray = None,
                feature_col_mapping: dict = None,
                text: str = None,
                tid: str = None):
        return super().predict(tokens=self.to_doc_type_annotation(tokens), vectors=vectors, mat_3d=mat_3d,
                               feature_col_mapping=feature_col_mapping, tid=tid)

    def get_vectors(self, annotation, fs_client, testing_features):
        return super().get_vectors(self.to_doc_type_annotation(annotation), fs_client, testing_features)

    def _read_annotation_file(self, file_name: str):
        return self.to_doc_type_annotation(super()._read_annotation_file(file_name))

    def token_true_label_list(self, test_ann: str, tokens):
        return super().token_true_label_list(test_ann, self.to_doc_type_annotation(tokens))

