import string

from text2phenotype.common.featureset_annotations import MachineAnnotation, Vectorization
from text2phenotype.constants.umls import SemTypeCtakesAsserted
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.apm.metrics import text2phenotype_capture_span
from text2phenotype.constants.features.label_types import CancerLabel
from text2phenotype.constants.features.feature_type import FeatureType

from biomed.biomed_env import BiomedEnv
from biomed.common.biomed_ouput import CancerOutput, CancerStageOutput
from biomed.common.aspect_response import CancerResponse
from biomed.common.helpers import  get_pref_umls_concept_polarity, get_by_pref_tty
from biomed.meta.ensembler import Ensembler
from biomed.constants.constants import ModelType, DEFAULT_MIN_SCORE, get_ensemble_version
from biomed.constants.model_constants import OncologyConstants
from biomed.cancer.cancer_represent import Qualifier, parse_behavior, parse_grade, Grade, Stage


@text2phenotype_capture_span()
def get_oncology_tokens(text: str,
                        tid: str = None,
                        min_score: float = DEFAULT_MIN_SCORE,
                        tokens: MachineAnnotation = None,
                        vectors: Vectorization = None,
                        biomed_version: str = BiomedEnv.DEFAULT_BIOMED_VERSION.value) -> dict:
    if not text:
        return {}
    ensemble_version = get_ensemble_version(ModelType.oncology, biomed_version)
    ensembler = Ensembler(
        model_type=ModelType.oncology,
        model_file_list=ensemble_version.model_names,
        voting_method=ensemble_version.voting_method,
    )

    feature_service_client = FeatureServiceClient()
    ensembler_feats = set(ensembler.feature_list)
    ensembler_feats.update(set(OncologyConstants.token_umls_representation_feat.keys()))
    if not tokens:
        tokens = feature_service_client.annotate(text, features=ensembler_feats, tid=tid)
        if not tokens:
            return {}
    if not vectors:
        vectors = feature_service_client.vectorize(tokens, features=ensembler.feature_list, tid=tid)

    res = oncology_predict_represent(tokens,
                                     use_generator=BiomedEnv.BIOMED_USE_PREDICT_GENERATOR.value,
                                     vectors=vectors,
                                     ensembler=ensembler,
                                     tid=tid)

    res.post_process(text=text, min_score=min_score)

    return res.to_versioned_json(model_type=ModelType.oncology, biomed_version=biomed_version)


@text2phenotype_capture_span()
def oncology_predict_represent(tokens: MachineAnnotation,
                               vectors: Vectorization,
                               use_generator: bool,
                               ensembler: Ensembler,
                               **kwargs) -> CancerResponse:
    res = ensembler.predict(tokens, vectors=vectors, use_generator=use_generator, **kwargs)
    results = []
    token_list = tokens.tokens
    category_predictions = res.predicted_category
    predicted_probabilities = res.predicted_probs
    topographies = {
        CancerLabel.topography_primary.value.column_index,
        CancerLabel.topography_metastatic.value.column_index
    }
    punctuation = set(string.punctuation)

    for i in range(len(token_list)):
        predicted_category = int(category_predictions[i])
        token = token_list[i]

        # filter tokens that are all punctuation characters
        if predicted_category != 0 and not all(c in punctuation for c in token):
            lstm_prob = predicted_probabilities[i, predicted_category]

            if predicted_category == CancerLabel.stage.value.column_index:
                # parses apart tokens that contain both stage and grade (ex: TaHG)
                grade = parse_grade(token, True)
                if grade != Grade.G9_unknown:
                    results.append(CancerOutput(label=CancerLabel.grade.value.persistent_label,
                                                text=token,
                                                range=tokens.range[i],
                                                lstm_prob=lstm_prob,
                                                umlsConcept=Qualifier.represent(dict(), grade)))

                # TODO: apply rules for valid combinations (future work)
                stage = Stage.from_string(token)

                output = CancerStageOutput(label=CancerLabel.stage.value.persistent_label,
                                           text=token_list[i],
                                           range=tokens.range[i],
                                           lstm_prob=lstm_prob,
                                           T=stage.T,
                                           N=stage.N,
                                           M=stage.M,
                                           clinical=stage.clinical)
            else:
                umls_concept = None

                # if predicted to be morphology of topography add the umls output
                if predicted_category == CancerLabel.morphology.value.column_index:
                    annotation = tokens[FeatureType.morphology.name, i]
                    umls_concept, _, _ = get_pref_umls_concept_polarity(
                        annotation, [SemTypeCtakesAsserted.DiseaseDisorder.name], get_by_pref_tty)
                elif predicted_category in topographies:
                    annotation = tokens[FeatureType.topography, i]
                    umls_concept, _, _ = get_pref_umls_concept_polarity(
                        annotation, [SemTypeCtakesAsserted.AnatomicalSite.name], get_by_pref_tty)
                elif predicted_category == CancerLabel.behavior.value.column_index:
                    umls_concept = Qualifier.represent(dict(), parse_behavior(tokens['token'][i]))
                elif predicted_category == CancerLabel.grade.value.column_index:
                    umls_concept = Qualifier.represent(dict(), parse_grade(tokens['token'][i]))

                # somehow the res doesn't have y_voted_weight_prob attr, not sure if it has weighted attr
                output = CancerOutput(label=CancerLabel.get_from_int(predicted_category).value.persistent_label,
                                      text=token,
                                      range=tokens.range[i],
                                      lstm_prob=lstm_prob,
                                      umlsConcept=umls_concept)

            results.append(output)

    return CancerResponse(category_name=CancerLabel.get_category_label().persistent_label,
                          response_list=results)
