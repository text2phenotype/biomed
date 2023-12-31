from requests.exceptions import ConnectionError

from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.common.status import StatusReport, Component, Status
from text2phenotype.common.version_info import get_version_info

from biomed.biomed_env import BiomedEnv


def live():
    pass


def ready():
    status_report = StatusReport()
    biomed_status = Status.healthy
    biomed_reason = None

    feature_client = FeatureServiceClient()
    try:
        features_response = feature_client.ready()
        features_status, reason = features_response[Component.features.name]
        if features_status in [Status.healthy.name, Status.conditional.name]:
            status_report.add_status(Component.features, (Status[features_status], reason))
        else:
            status_report.add_status(Component.features, (Status.dead, reason))
            biomed_status = Status.conditional
            biomed_reason = 'feature service dead'
    except ConnectionError:
        status_report.add_status(Component.features, (Status.dead, 'Connection Error'))

    status_report.add_status(Component.biomed, (biomed_status, biomed_reason))

    return status_report.as_json(), biomed_status.value


def ready_metadata_service():
    status_report = StatusReport()
    biomed_status = Status.healthy
    status_report.add_status(Component.biomed, (biomed_status, None))
    return status_report.as_json(), biomed_status.value


def version():
    git_path = BiomedEnv.root_dir
    return get_version_info(git_path).to_dict()
