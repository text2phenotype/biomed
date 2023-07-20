from typing import List, Dict

from text2phenotype.constants.common import VERSION_INFO_KEY

DOS_KEY = 'dos'


def add_date_type_to_summary_output(dos_output_json: Dict[str, List[Dict]], summary_json: Dict[str, List[dict]],
                                    date_type):
    """Function that adds dates to the summary json
    If a page has an specific type of date: all entities without a data already get assigned the date of the date
    If a page does not have a date nothing from that page gets assigned a date
    """
    # get page to date mapping
    page_mapping = dict()
    for entry in dos_output_json[DOS_KEY]:
        if entry['label'] == date_type:
            page_mapping[entry['page']] = entry['preferredText']
    for key in summary_json:
        if key != VERSION_INFO_KEY:
            for biomed_entity in summary_json[key]:
                if biomed_entity['page'] in page_mapping and not biomed_entity.get('date'):
                    biomed_entity['date'] = page_mapping[biomed_entity['page']]
    return summary_json
