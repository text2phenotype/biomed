from awscli.clidriver import create_clidriver

from biomed.constants.constants import ModelType
from typing import List
from biomed.constants.constants import MODEL_TYPE_TO_INDEX
from text2phenotype.common.log import operations_logger
from feature_service.common.data_source import FeatureServiceDataSource
import random
import os
import shutil


def get_ann_files_with_annotation(model_type: ModelType, data_source: FeatureServiceDataSource):
    ann_files = data_source.get_ann_files()
    annotated_ann_files = set()
    model_label = MODEL_TYPE_TO_INDEX[model_type]
    for ann_file in ann_files:
        if model_type is not ModelType.lab and data_source.get_brat_label(ann_file, model_label):
            annotated_ann_files.add(ann_file)
        elif model_type is ModelType.lab and data_source.get_brat_lab_name_value_unit(ann_file):
            annotated_ann_files.add(ann_file)
    return annotated_ann_files


def convert_ann_files_to_original_text_files(annotated_ann_files: set, data_source: FeatureServiceDataSource):
    text_files = set(annotated_ann_files)
    for ann_dir in data_source.ann_dirs:
        text_files = {fn.replace(ann_dir, '') for fn in text_files}

    return {txt_fn.replace('.ann', '.txt') for txt_fn in text_files}


def select_random_list_assortment(file_list: List[str], num_files: int):
    sublist = random.sample(file_list, num_files)
    return sublist


def copy_single_file_down(source_path, dest_path):
    driver = create_clidriver()

    exit_code = driver.main(('s3', 'cp', source_path, dest_path))


def get_text_files(data_source: FeatureServiceDataSource, text_files: List[str]):
    for text_file in text_files:
        try:
            copy_single_file_down(text_file.replace('/Users/shannon.fee/Documents/S3/teamcity-text2phenotype//',
                                                    's3://biomed-data/').replace(
                '/Users/shannon.fee/Documents/S3/teamcity-text2phenotype/',
                's3://biomed-data/'),
                                  text_file)

        except:
            operations_logger.warning(f'text file does not exist in s3: {text_file}')


def create_folder(text_files: List[str], ann_files: List[str], dir_name: str, model_type: ModelType):
    text_path = os.path.join(dir_name, 'original_text')

    ann_path = os.path.join(dir_name, model_type.name, 'original_text')
    if not os.path.isdir(text_path):
        os.mkdir(text_path)
    if not os.path.isdir(ann_path):
        if not os.path.isdir(os.path.join(dir_name, model_type.name)):
            os.mkdir(os.path.join(dir_name, model_type.name))
        os.mkdir(ann_path)

    for text_file in text_files:
        if os.path.isfile(text_file):
            base_split = text_file.split('teamcity-text2phenotype/')[1].split('/')
            start_after = -1
            for i in range(len(base_split)):
                if not base_split[i]:
                    start_after = i
            base_name = '_'.join(base_split[start_after + 1:])
            new_fn_txt = os.path.join(text_path, base_name)
            shutil.copy(text_file, new_fn_txt)
    for file in ann_files:
        base_split = file.split('.com/')[1].split('/')
        start_after = -1
        for i in range(len(base_split)):
            if ('BIOMED' in base_split[i] or '2019-' in base_split[i] or '2019_' in base_split[i] or
                    '20191209' in base_split[i] or '2018-' in base_split[i]):
                start_after = i
        base_name = '_'.join(base_split[start_after + 1:])
        new_fn = os.path.join(ann_path, base_name)
        shutil.copy(file, new_fn)


def create_subfolder(ann_dirs, original_raw_text_dirs, model_type, num_files=75):
    datasource = FeatureServiceDataSource(ann_dirs=ann_dirs, original_raw_text_dirs=original_raw_text_dirs)
    ann_files = get_ann_files_with_annotation(model_type, data_source=datasource)
    subset = select_random_list_assortment(ann_files, num_files)
    text_files = convert_ann_files_to_original_text_files(subset, datasource)
    get_text_files(datasource, text_files)
    create_folder(text_files=text_files, ann_files=subset, dir_name='/Users/shannon.fee/Documents/qa_docs',
                  model_type=model_type)


create_subfolder(model_type=ModelType.allergy,
                 original_raw_text_dirs=[
                     'I2B2/2014 De-identification and Heart Disease Risk Factors Challenge/gold_raw_'
                     'text/demographic/surrogates_v1/training-PHI-Gold-Set1',
                     "mimic/20190207_andy/txt/DischargeSummary/12/2",
                     "mimic/20190207_andy/txt/DischargeSummary/12",
                     "mimic/20190207_andy/txt/DischargeSummary/12/1",
                     "mimic/20190207_andy/txt/DischargeSummary/12/3",
                     "mimic/20190207_andy/txt/DischargeSummary/12/4",
                     "CMS/v1",
                     "CMS/v2"],
                 ann_dirs=["nick.colangelo/2019_05_21",
                           "nick.colangelo/BIOMED-906",
                           "nick.colangelo/BIOMED-913",
                           "nick.colangelo/BIOMED-1095",
                           "satjiv.kohli/2019_03_28", "briana.galloway/BIOMED-1215-summary"]
                 )
create_subfolder(model_type=ModelType.oncology, original_raw_text_dirs=[
    "NZCR"
],
                 ann_dirs=["despina.siolas/BIOMED-1000-cancer/20191209",
                           "despina.siolas/BIOMED-1000-cancer/20191209",
                           "despina.siolas/BIOMED-1000-cancer/20191209",
                           "despina.siolas/BIOMED-1000-cancer/20191209",
                           "despina.siolas/BIOMED-1000-cancer/20191209",
                           "despina.siolas/BIOMED-1000-cancer/20191209"]
                 )
create_subfolder(model_type=ModelType.deid, original_raw_text_dirs=[
    "CMS/v2",
    "text2phenotype_samples_himss_plus", 'ccda',
    'I2B2/2014 De-identification and Heart Disease Risk Factors Challenge/gold_raw_text/DEID/surrogates_v1/testing-PHI-Gold-fixed'
],
                 ann_dirs=['shannon.fee', 'mike.banos/2019-01-08'])

create_subfolder(model_type=ModelType.demographic, original_raw_text_dirs=[
    "CMS/v2",
    "text2phenotype_samples_himss_plus", 'ccda', 'synthea_text'
                                        'I2B2/2014 De-identification and Heart Disease Risk Factors Challenge/gold_raw_text/demographic/surrogates_v1/testing-PHI-Gold-fixed',
    'I2B2/2014 De-identification and Heart Disease Risk Factors Challenge/gold_raw_text/DEID/surrogates_v1/testing-PHI-Gold-fixed'
],
                 ann_dirs=['shannon.fee', 'mike.banos/2019-01-08',
                           'briana.galloway/BIOMED-1048'])

create_subfolder(model_type=ModelType.lab, ann_dirs=["deleys.brandman/annotation_BIOMED-655",
                                                     "briana.galloway/BIOMED-1215-summary"],
                 original_raw_text_dirs=['mtsamples/clean', 'mimic'])
create_subfolder(model_type=ModelType.drug, ann_dirs=["deleys.brandman/annotation_BIOMED-655",
                                                     "briana.galloway/BIOMED-1215-summary",
                                                     "mike.banos/2018-11-21"],
                 original_raw_text_dirs=['mtsamples/clean', 'mimic', 'I2B2'])
create_subfolder(model_type=ModelType.diagnosis, ann_dirs=["nick.colangelo/BIOMED-913",
                                                         "deleys.brandman/annotation_BIOMED-655",
                                                         "satjiv.kohli/BIOMED-913/"],
                 original_raw_text_dirs=['mtsamples/clean', 'I2B2'])
