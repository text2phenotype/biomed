import argparse
import ast
import json

from text2phenotype.common.common import json_validator
from text2phenotype.common.log import operations_logger

from biomed.constants.constants import ModelType
from biomed.data_sources.data_source import BiomedDataSource
from biomed.meta.ensemble_model_metadata import EnsembleModelMetadata
from biomed.models.model_metadata import ModelMetadata
from biomed.models.multi_train_test import KFoldValidation, MultiEnsembleTest
from biomed.train_test.job_metadata import JobMetadata
from biomed.train_test.train_test import TrainTestJob


def parse_arguments():
    """
    this parses the argument passed into the terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-job_id', type=convert_string, help='identifier for job')
    parser.add_argument('-model_type', type=convert_string, help='model type to use')
    parser.add_argument('-model_file_name', type=convert_string, help='the name of the model file')
    parser.add_argument('-test', type=convert_string, help='whether to test a model')
    parser.add_argument('-train', type=convert_string, help='whether to train a model')
    parser.add_argument('-metadata', type=convert_string, help='metadata describing model, job, and data source')
    parser.add_argument('-test_ensemble', type=convert_string,
                        help='whether to test an ensembler on provided model file list')
    return parser.parse_args()


def convert_string(value: str):
    if value == 'None':
        value = None
    elif value in ['1', 'true', 'True', 'TRUE', 'yes', True]:
        value = True
    elif value in ['0', 'false', 'False', 'FALSE', 'no', None, False]:
        value = False
    elif value in [model_type.name for model_type in ModelType]:
        value = ModelType[value]
    elif value.startswith('[') and value.endswith(']'):
        value = ast.literal_eval(value)
    elif value.startswith('{') and value.endswith('}'):
        json_validator(value)
        value = json.loads(value)
    return value


def get_args(args):
    metadata = args.get('metadata', dict())

    for k, v in metadata.items():
        args[k] = v

    return args


def prepare_metadata(args: dict):
    # note that if want to use meta-classifier then needs just pass meta to
    # the model type interface box in teamcity and NOT pass anything to metadata box
    args = get_args(args)
    model_type = args.get('model_type')

    model_metadata = ModelMetadata(**args)
    data_source = BiomedDataSource(**args)
    job_metadata = JobMetadata.from_dict(args)
    ensemble_metadata = EnsembleModelMetadata(**args)
    return model_metadata, job_metadata, data_source, ensemble_metadata


def check_bool_arg_name(parsed_args, arg_name):
    in_base = arg_name in parsed_args and parsed_args[arg_name]
    in_meta = parsed_args.get('metadata') and parsed_args['metadata'].get(arg_name)
    return in_base or in_meta


def run_job(parsed_args):
    if check_bool_arg_name(parsed_args, 'k_fold_validation'):
        parsed_args = get_args(parsed_args)
        operations_logger.info(f"Creating K Fold Validation Job with args {parsed_args}")
        job = KFoldValidation(parsed_args)

    elif check_bool_arg_name(parsed_args, 'multi_ensemble_test'):
        parsed_args = get_args(parsed_args)
        operations_logger.info(f"Creating Multi Ensemble Test Build with args {parsed_args}")
        job = MultiEnsembleTest(parsed_args)
    else:
        # meta_tuple = parsed_model_metadata, parsed_job_metadata, parsed_data_source, parsed_ensemble_metadata
        meta_tuple = prepare_metadata(parsed_args)
        job = TrainTestJob(*meta_tuple)

    job.run()


if __name__ == '__main__':
    parsed_args = vars(parse_arguments())
    run_job(parsed_args)
