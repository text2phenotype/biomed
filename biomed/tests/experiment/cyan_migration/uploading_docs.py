import os

from text2phenotype.annotations.file_helpers import AnnotationSet, Annotation
from text2phenotype.common import common
from text2phenotype.common.log import operations_logger
import pandas


from biomed.tests.experiment.cyan_migration.constants import (annotator_to_tag_tog_username,
                                                              tag_tog_project_folders, CovidLabLabel)
from text2phenotype.constants.features import LabLabel
from text2phenotype.tagtog.tag_tog_annotation import TagTogAnnotationSet
from text2phenotype.tagtog.tag_tog_client import TagTogClient

BASE_DIR = '/Users/shannon.fee/Downloads/mdl-phi-cyan-us-west-2'

missing_text_docs = []

def text_from_ann_path(ann_fp):
    doc_uuid = os.path.split(ann_fp)[1].split('.')[0]
    text_path = os.path.join(BASE_DIR, 'processed/documents', f'{doc_uuid}/{doc_uuid}.extracted_text.txt')
    return text_path

def upload(
        ann_fp,
        txt_fp,
        tc_client: TagTogClient,
        inverse_annotation_guide:dict,
        member_id: str='master',
        folder='us-west-2', lab: bool = False, covid_lab: bool=False, event_date_update: bool =False
):
    ann_set = AnnotationSet.from_file_content(common.read_text(ann_fp))

    if event_date_update:
        ann_set = update_event_date_label(ann_set)
    elif lab:
        ann_set = filter_lab_vs_covid_lab(ann_set)
    elif covid_lab:
        ann_set = filter_lab_vs_covid_lab(ann_set, covid=True)

    doc_id = os.path.split(ann_fp)[1].replace('.ann', '')

    ann_json: TagTogAnnotationSet = TagTogAnnotationSet()
    ann_json.from_annotation_set_for_text(
        ann_set, inverse_annotation_legend=inverse_annotation_guide, text= common.read_text(txt_fp)
        )
    if len(ann_json.entities) > 0:
        i=0
        txt= common.read_text(txt_fp)
        while txt[i] in ['\n', ' ', '\t']:
            i+=1

        txt_1 = '-'*i + txt[i:]
        txt_1 = txt_1.replace('\x0c', '\n')
        new_text_fp = f'/Users/shannon.fee/Downloads/tag_tog_upload/{tc_client.project}/{doc_id}.txt'
        os.makedirs(os.path.dirname(new_text_fp), exist_ok=True)

        ann_json_dict = ann_json.to_json({'s1v1': '1'})
        ann_json_dict['anncomplete'] = False

        ann_json_path = new_text_fp.replace('.txt', '.ann.json')
        common.write_text(txt_1, new_text_fp)
        common.write_json(ann_json_dict, ann_json_path)

        a = tc_client.push_text_ann_verbatim_from_fp(
            new_text_fp, ann_json_path, folder=folder, annotator_id=member_id,
            base_fp=f'/Users/shannon.fee/Downloads/tag_tog_upload/{tc_client.project}'
        )
        if a.ok:
            return 1
        else:
            return 2

    return 0


def upload_for_all_projects():
    failed_docs = []
    annotation_file_guide = []
    for tag_tog_project in tag_tog_project_folders:
        tc_client = TagTogClient(project=tag_tog_project, proj_owner='tagtogadmin', output='txt')
        annotation_legent = tc_client.get_annotation_legend()

        inverse_annotation_guide = {v: k for k, v in annotation_legent.items()}
        covid_lab, event_date, lab = False, False, False
        if tag_tog_project == 'Covid_Specific':
            covid_lab = True
        elif tag_tog_project == 'document_type_event_date':
            event_date = True
        elif tag_tog_project == 'lab_validation':
            lab = True

        for annotator in annotator_to_tag_tog_username:
            ann_path = os.path.join(BASE_DIR, 'annotations', annotator)
            ann_files = common.get_file_list(ann_path, '.ann', True)
            for ann_fp in ann_files:
                text_fp = text_from_ann_path(ann_fp)

                if not os.path.isfile(text_fp):
                    missing_text_docs.append(text_fp)
                    continue
                included = upload(
                    ann_fp=ann_fp, txt_fp=text_fp, member_id=annotator_to_tag_tog_username[annotator],
                    inverse_annotation_guide=inverse_annotation_guide, tc_client=tc_client, covid_lab=covid_lab,
                    lab=lab, event_date_update=event_date)
                if included == 1:
                    annotation_file_guide.append(
                        {'file_id': os.path.split(ann_fp)[1].split('.')[0], 'annotator': annotator,
                         'project': tag_tog_project})
                elif included == 2:
                    failed_docs.append((text_fp, ann_fp, tag_tog_project))

    return annotation_file_guide, failed_docs


def filter_lab_vs_covid_lab(annotation_set:AnnotationSet, covid: bool = False) -> AnnotationSet:
    bad_ids =  []
    for entry_id in annotation_set.directory:
        annot: Annotation = annotation_set.directory[entry_id]
        if annot.label == 'lab':
            if annot.category_label == LabLabel.get_category_label().persistent_label and covid:
                covid_lab = 'sars' in annot.text.lower() or 'cov' in annot.text.lower() or 'coronavirus' in annot.text.lower()
                interp = 'detec' in annot.text.lower() or 'pos' in annot.text.lower() or 'neg' in annot.text.lower()
                if not covid_lab:
                    if not interp:
                        bad_ids.append(entry_id)

                    else:
                        annot.label = 'lab_interp'
                        annotation_set.directory[entry_id] = annot
        if annot.category_label == CovidLabLabel.get_category_label().persistent_label and not covid:
            bad_ids.append(entry_id)
    for id in bad_ids:
        del annotation_set.directory[id]
    return annotation_set


DATE_TYPE_MAPPING = {'collection_date': 'specimen_date'}

def update_event_date_label(annotation_set: AnnotationSet) -> AnnotationSet:
    for entry_id in annotation_set.directory:
        annot: Annotation = annotation_set.directory[entry_id]
        if annot.label in DATE_TYPE_MAPPING:
            annot.label = DATE_TYPE_MAPPING[annot.label]
            annotation_set.directory[entry_id] = annot

    return annotation_set



# RUN PROCESS

annotation_file_guide, failed_docs = upload_for_all_projects()

#write out file of who's annotated what
out_csv_path = '/Users/shannon.fee/Documents/cyan_documents_in_tag_tog.csv'
pandas.DataFrame(annotation_file_guide).to_csv(out_csv_path)

out_failed_path = '/Users/shannon.fee/Documents/migration_failures.csv'
pandas.DataFrame(failed_docs, columns=['text', 'ann', 'project']).to_csv(out_failed_path)


