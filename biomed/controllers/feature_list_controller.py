import connexion

from biomed.biomed_env import BiomedEnv
from text2phenotype.apm.metrics import text2phenotype_capture_span

from biomed.data_contracts.biomed_request import BiomedRequest
from biomed.common.helpers import feature_list_helper
from text2phenotype.constants.common import VERSION_INFO_KEY


@text2phenotype_capture_span()
def get_full_feature_list(biomed_request: BiomedRequest = None):
    if connexion.request.is_json:
        req = BiomedRequest.from_dict(connexion.request.get_json())
    else:
        req = BiomedRequest.from_dict(biomed_request)
    operations = req.data.get('operations')
    biomed_version = req.data.get('biomed_version') or BiomedEnv.DEFAULT_BIOMED_VERSION.value
    return {'features': list(feature_list_helper(operations=operations, models=req.models, tid=req.tid,
                                    biomed_version=biomed_version)),
            VERSION_INFO_KEY: biomed_version}
