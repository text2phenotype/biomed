import connexion

from text2phenotype.common.log import operations_logger
from text2phenotype.open_api import encoder
from text2phenotype.apm.flask import configure_apm
from text2phenotype.common.cli_utils import CommonCLIArguments

from biomed.biomed_env import BiomedEnv
from biomed.models.model_cache import ModelCache


class BiomedCLIArguments(CommonCLIArguments):
    def __init__(self):
        super().__init__()
        self.parser.add_argument('--models-metadata-service',
                                 dest='models_metadata_api',
                                 action='store_true',
                                 default=False,
                                 help='Run Biomed Models Metadata service')

    @property
    def models_metadata_api(self):
        return self.args.models_metadata_api


def create_app(models_metadata_api=False):
    app = connexion.App(__name__, specification_dir='./')
    configure_apm(app.app, BiomedEnv.APM_SERVICE_NAME.value)
    operations_logger.debug(f'app.app.config = {app.app.config}')
    app.app.json_encoder = encoder.JSONEncoder

    for handler in operations_logger.logger.handlers:
        app.app.logger.addHandler(handler)

    if models_metadata_api:
        operations_logger.info('Starting Models Metadata service...')
        app.add_api('models_metadata_service_open_api.yaml')
    else:
        operations_logger.info('Starting Biomed service...')
        app.add_api('biomed_open_api.yaml')
        ModelCache(BiomedEnv.PRELOAD.value)

    return app


if __name__ == '__main__':
    cli_arguments = BiomedCLIArguments()

    app = create_app(models_metadata_api=cli_arguments.models_metadata_api)
    app.run(host=BiomedEnv.SVC_HOST.value, port=BiomedEnv.SVC_PORT.value, debug=cli_arguments.debug)
