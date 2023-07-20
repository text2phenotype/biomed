import string
from typing import List, Tuple

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.constants.features.label_types import GeneticsLabel

from biomed.biomed_env import BiomedEnv
from biomed.common.biomed_ouput import GeneticsOutput
from biomed.common.aspect_response import AspectResponse
from biomed.meta.ensembler import Ensembler
from biomed.constants.constants import ModelType, DEFAULT_MIN_SCORE, get_ensemble_version


@text2phenotype_capture_span()
def get_genetics_tokens(text: str,
                        tid: str = None,
                        min_score: float = DEFAULT_MIN_SCORE,
                        tokens: MachineAnnotation = None,
                        vectors: Vectorization = None,
                        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> dict:
    if not text:
        return {}
    ensemble_version = get_ensemble_version(ModelType.genetics, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.genetics,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    feature_service_client = FeatureServiceClient()
    ensembler_feats = set(ensembler.feature_list)
    if not tokens:
        tokens = feature_service_client.annotate(text, features=set(ensembler_feats), tid=tid)
        if not tokens:
            return {}

    if not vectors:
        vectors = feature_service_client.vectorize(tokens, features=ensembler.feature_list, tid=tid)

    res = genetics_represent(tokens,
                             use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value,
                             vectors=vectors,
                             ensembler=ensembler,
                             tid=tid)

    res.post_process(text=text, min_score=min_score)

    return res.to_versioned_json(model_type=ModelType.genetics, biomed_version=biomed_version)


@text2phenotype_capture_span()
def genetics_represent(tokens: MachineAnnotation,
                       vectors: Vectorization,
                       use_generator: bool,
                       ensembler: Ensembler,
                       **kwargs) -> AspectResponse:
    res = ensembler.predict(tokens, vectors=vectors, use_generator=use_generator, **kwargs)
    results = []
    token_list = tokens.tokens
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs

    for i in range(len(token_list)):
        if category_predictions[i] != 0 and token_list[i] not in string.punctuation:
            lstm_prob = predicted_probabilities[i, int(category_predictions[i])]

            label = GeneticsLabel.get_from_int(category_predictions[i]).value.persistent_label
            text = token_list[i]
            token_range = tokens.range[i]
            if label == GeneticsLabel.gene.value.persistent_label:
                for gene, gene_range in split_genes(text, token_range):
                    results.append(GeneticsOutput(label=label,
                                                  text=gene,
                                                  range=gene_range,
                                                  lstm_prob=lstm_prob))
            else:
                results.append(GeneticsOutput(label=label,
                                              text=text,
                                              range=token_range,
                                              lstm_prob=lstm_prob))

    return AspectResponse(category_name=GeneticsLabel.get_category_label().persistent_label,
                          response_list=results)


def split_genes(token: str, token_range: List[int]) -> List[Tuple[str, List[int]]]:
    """Split a multi-gene token into individual genes."""
    genes = []

    gene_start = token_range[0]
    for gene in token.split('/'):
        gene_end = gene_start + len(gene)
        genes.append((gene, [gene_start, gene_end]))
        gene_start = gene_end + 1

    return genes
