import os

from biomed.common.aspect_response import LabResponse
from biomed.common.biomed_ouput import LabOutput
from biomed.common.helpers import feature_list_helper
from biomed.constants.constants import get_version_model_folders
from biomed.constants.model_constants import ModelType
from biomed.lab.labs import summary_lab_value
from biomed.meta.ensembler import Ensembler
from text2phenotype.annotations.file_helpers import AnnotationSet, Annotation
from text2phenotype.apiclients import FeatureServiceClient
from text2phenotype.common import common
from text2phenotype.constants.features import FeatureType, LabLabel
from text2phenotype.tagtog.tag_tog_annotation import TagTogEntity
from text2phenotype.tagtog.tag_tog_client import TagTogClient
from text2phenotype.tagtog.tagtog_html_to_text import TagTogText
from text2phenotype.tasks.task_enums import TaskOperation

ttc = TagTogClient(project='Covid_Specific', proj_owner='tagtogadmin')
#
# all_file_ids = ttc.search('lab')
# os.makedirs('/Users/shannonfee/Documents/NMIBC_WITH_LAB', exist_ok=True)
# for file in all_file_ids:
#     file_id = file['id']
#     tag_tog_html_resp= ttc.get_html_by_doc_id(file_id)
#     html = tag_tog_html_resp.text
#     tag_tog_text = TagTogText(html)
#     out_fp = f'/Users/shannonfee/Documents/NMIBC_WITH_LAB/{file_id}'
#     if ' lab ' in tag_tog_text.raw_text or 'Lab' in tag_tog_text.raw_text:
#         pdf_resp = ttc.get_orig_doc_by_doc_id(file_id)
#         if pdf_resp.status_code == 200:
#             with open(out_fp, "wb") as f:
#                 f.write(pdf_resp.content)


# from PyPDF2 import PdfFileReader, PdfFileWriter
# from math import ceil
# # split all pdfs into 5 page chunks
# os.makedirs('/Users/shannonfee/Documents/NMIBC_WITH_LAB_CHUNKS', exist_ok=True)
# for pdf_fp in common.get_file_list('/Users/shannonfee/Documents/NMIBC_WITH_LAB', '.pdf'):
#     pdf_reader = PdfFileReader(open(pdf_fp, 'rb'))
#     num_chunks = ceil(pdf_reader.numPages-1/5)
#     for chunk in range(num_chunks):
#         pdf_writer = PdfFileWriter()
#         for page_num in range(5*chunk, min(pdf_reader.numPages-1, 5*(chunk+1))):
#             pdf_writer.addPage(pdf_reader.getPage(page_num))
#
#         with open(pdf_fp.replace('.pdf', f'_chunk_{chunk}.pdf').replace('NMIBC_WITH_LAB', 'NMIBC_WITH_LAB_CHUNKS'), 'wb') as outfile:
#             pdf_writer.write(outfile)


# for pdf_fp in common.get_file_list('/Users/shannonfee/Documents/NMIBC_WITH_LAB_CHUNKS', '.pdf'):
#     ttc.push_pdf(pdf_fp, 'NMIBC_chunk')
#
#     # get lab pages
#
#     #get pdf
#     # pdf =
from text2phenotype.tagtog.helper_functions import add_tag_tog_annotation_to_doc_id, TagTogAnnotationSet, \
    add_annotation_to_doc_id

# fsclient = FeatureServiceClient()
# bad_doc_ids = []
# features = feature_list_helper({TaskOperation.lab})
# for html_fp in common.get_file_list('/Users/shannonfee/Downloads/Covid_Specific 6/plain.html/pool/NMIBC_chunk', 'html'):
#     text = TagTogText(common.read_text(html_fp)).raw_text
#     annotation, vectors = fsclient.annotate_vectorize(text=text, features=features)
#     lab_ensembler = Ensembler(model_type=ModelType.lab,
#                               model_file_list=get_version_model_folders(model_type=ModelType.lab))
#     lab_res = lab_ensembler.predict(tokens=annotation, vectors=vectors)
#     lab_aspect = LabResponse('Lab', lab_res.token_dict_list(LabLabel, bio_output_class=LabOutput))
#     lab_aspect.post_process(text)
#     if len(lab_aspect.response_list) > 0:
#         ann_list= []
#         for entry in lab_aspect.response_list:
#             ann_list.append(
#                 Annotation(
#                     label=entry.label,
#                     text=entry.text,
#                     category_label='Lab',
#                     text_range=entry.range
#
#                 ))
#
#             if entry.labUnit:
#                 lab_attrib_text = entry.labUnit.text
#                 lab_attrib_range = entry.labUnit.range
#                 ann_list.append(
#                     Annotation(
#                         label='lab_unit',
#                         text=lab_attrib_text,
#                         category_label='Lab',
#                         text_range=lab_attrib_range
#
#                     ))
#             if entry.labValue:
#                 lab_attrib_text = entry.labValue.text
#                 lab_attrib_range = entry.labValue.range
#                 ann_list.append(
#                     Annotation(
#                         label='lab_value',
#                         text=lab_attrib_text,
#                         category_label='Lab',
#                         text_range=lab_attrib_range
#
#                     ))
#             if entry.labInterp:
#                 lab_attrib_text = entry.labInterp.text
#                 lab_attrib_range =entry.labInterp.range
#                 ann_list.append(
#                     Annotation(
#                         label='lab_interp',
#                         text=lab_attrib_text,
#                         category_label='Lab',
#                         text_range=lab_attrib_range
#
#                     ))
#
#
#         ann_set = AnnotationSet.from_list(ann_list)
#         res = add_annotation_to_doc_id(
#             tag_tog_client=ttc,
#             doc_id=os.path.basename(html_fp).split('-')[1].split('.plain.html')[0],
#             ann_set=ann_set,
#             html_text=common.read_text(html_fp),
#             member_ids=['tagtogadmin']
#         )
#     else:
#         bad_doc_ids.append(os.path.basename(html_fp).split('-')[1].split('.plain.html')[0])
#         #delete doc
#         print(html_fp)
fps = ttc.search('folder:NMIBC_chunk AND -count_e_5:[1 TO *]')
print(fps)
ttc.delete_records(bad_doc_ids)



