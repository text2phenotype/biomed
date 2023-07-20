"""
inbox/COVID-BIOMED-1829/10731/10731_461_470.txt
inbox/COVID-BIOMED-1829/19196/19196_151_160.txt
inbox/COVID-BIOMED-1829/10731/10731_271_280.txt
inbox/COVID-BIOMED-1829/16457/16457_391_400.txt
inbox/COVID-BIOMED-1829/11429/11429_821_830.txt
"""
import os
import json
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
from text2phenotype.tagtog.splitting_pdfs import split_pdf
from text2phenotype.tagtog.tag_tog_annotation import TagTogEntity, TagTogAnnotationSet
from text2phenotype.tagtog.tag_tog_client import TagTogClient
from text2phenotype.tagtog.tagtog_html_to_text import TagTogText
from text2phenotype.apiclients.feature_service import FeatureServiceClient
from text2phenotype.tasks.task_enums import TaskOperation

from biomed.covid.covid_summary import tokens_to_covid_summary
from biomed.common.helpers import feature_list_helper


# Uploading Docs to Tag Tog
dir = '/Users/shannon.fee/Downloads'
file_mapping = [
    ('10731.pdf', list(range(461, 471)), '10731_461_470.pdf'),
    ('10731.pdf', list(range(271, 281)), '10731_271_280.pdf'),
    ('19196 (1).pdf', list(range(151, 161)), '19196_151_160.pdf'),
    ('16457.pdf', list(range(391, 401)), '16457_391_400.pdf'),
    ('11429.pdf', list(range(821, 831)), '11429_821_830.pdf')]

tc = TagTogClient(project='Covid_Lab_Model', proj_owner='sfee')

# create split pdfs
# for file in file_mapping:
#     split_pdf(os.path.join(dir, file[0]), pages_to_keep=file[1], out_fp=os.path.join(dir, file[2]))

# # push split pdfs to tag tog
# for file in file_mapping:
#     print(tc.push_pdf(os.path.join(dir, file[2]), 'sample_v1_output'))


# Get all Tag Tog docs by folder and process them, create ann json and push results

get_file_ids = tc.search('folder:sample_v1_output')



annotation_legen = tc.get_annotation_legend()
name_to_class_id = {v:k for k, v in annotation_legen.items()}

for doc_id in get_file_ids:
    html = tc.get_html_by_doc_id(doc_id=doc_id)
    obj = TagTogText(html)
    text = obj.raw_text
    annot, vect = FeatureServiceClient().annotate_vectorize(
        text=text,
        features=feature_list_helper([TaskOperation.covid_specific]))
    covid_labs = tokens_to_covid_summary(tokens=annot, vectors=vect, original_text=text)
    tag_tog_annot = TagTogAnnotationSet()
    tag_tog_annot.from_biomed_summary_json(covid_labs, annotation_legend=name_to_class_id, html_text_obj=obj)
    # res = tc.update_annotations(html_text=html, ann_json_dict=tag_tog_annot.to_json(obj.html_mapping_to_text),
    #                             doc_id=doc_id)

    print(covid_labs)
    print(tag_tog_annot.to_json(obj.html_mapping_to_text))
