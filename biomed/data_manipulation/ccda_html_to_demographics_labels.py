import os
import datetime
from text2phenotype.common import common
from biomed.constants.constants import DemographicEncounterLabel
from text2phenotype.ccda import ccda
import re

def xml_to_demographics(ccda_xml_dir, ccda_text_dir, ccda_ann_dir):
    # get dictionary of endpoitns to filepaths

    xml_files = common.get_file_list(ccda_xml_dir, '.xml', True)
    ext_dict = dict()
    for xml in xml_files:
        split_line = os.path.split(xml)
        ext_dict[split_line[1]] = split_line[0]

    text_files = common.get_file_list(ccda_text_dir, '.txt', False)


    for txt_file in text_files:
        text = common.read_text(txt_file)
        metadata_json_file = txt_file.replace('.txt', '.metadata.json')
        metadata_json = common.read_json(metadata_json_file)
        txt_base = metadata_json['orig_file_name']
        xml_base = txt_base.replace('.pdf', '').replace('.html', '')
        xml_file = os.path.join(ext_dict.get(xml_base, ''), xml_base)
        ann_list = []
        if os.path.isfile(xml_file):
            # read ccda file
            demographics_contents = ccda.get_demographics_ccda(xml_file)
            for k, v in demographics_contents.items():
                if k == 'dob_str':
                    year = int(v[0:4])
                    month = int(v[4:6])
                    day = int(v[6:8])
                    dob = datetime.date(year=year, month=month, day=day)
                    val = dob.strftime('%B %-d, %Y')
                    a = re.finditer(r'\b({0})\b'.format(val), text)
                    for match in a:
                        ann_list.append([f'T{len(ann_list)}', f'DOB {match.start(0)} {match.end(0)}', val])
                for val in v:
                    if val:
                        a = re.finditer(r'\b({0})\b'.format(val), text)

                        for match in a:
                            if k in DemographicEncounterLabel.__members__:
                                ann_list.append([f'T{len(ann_list)}', f'{k} {match.start(0)} {match.end(0)}', val])
                            elif k == 'ids':
                                ann_list.append([f'T{len(ann_list)}', f'MRN {match.start(0)} {match.end(0)}', val])
                            elif k in ['race', 'ethnicity']:
                                ann_list.append([f'T{len(ann_list)}', f'{k} {match.start(0)} {match.end(0)}', val])
                            else:
                                continue

            final = "\n".join(["\t".join(a) for a in ann_list])
            common.write_text(final, txt_file.replace('.txt', '.ann').replace(ccda_text_dir, ccda_ann_dir))



