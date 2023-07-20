import os
import random
import lxml.etree as ET
from metapub.utils import asciify

from feature_service.nlp.nlp_reader import ClinicalReader
from text2phenotype.entity.results import ResultType
from text2phenotype.common.log import operations_logger
from text2phenotype.common import common
from feature_service.nlp import nlp_cache


def get_abstracts(medline_xml):
    """
    get text sources from medline (pubmed)
    :param medline_xml: example '/Users/andymc/code/medline/medline16n0001.xml'
    :return: list of abstract texts
    """
    operations_logger.debug('parsing ' + medline_xml)
    dom = ET.parse(medline_xml)

    abstracts = list()
    for el in dom.findall("//AbstractText"):
        abstracts.append(asciify(el.text))
    return abstracts


def process_abstracts_deprecated(medline_xml_dir, autocoder_function, cache_filetype):
    operations_logger.info(f'process_abstracts {medline_xml_dir} {autocoder_function} {cache_filetype}')

    files = common.get_file_list(medline_xml_dir, 'xml')
    success = list()
    failed = list()
    score = float()
    docid, abstract = None, None

    for i in range(1, 100):
        random.shuffle(files)
        try:
            for f in files:
                abstracts = get_abstracts(f)
                random.shuffle(abstracts)

                for abstract in abstracts:
                    abstract = str(abstract)

                    docid = nlp_cache.hash_text(abstract)

                    nlp_cache.save_text(abstract)
                    nlp_cache.autocode_cache(abstract, autocoder_function, cache_filetype)

                    success.append(docid)
                    docid, abstract = None, None

                    score = float(len(success)) / (float(len(success)) + float(len(failed)))
                    operations_logger.info(f'success {len(success)}')
                    operations_logger.info(f'failed {len(failed)}')
                    operations_logger.info(f'score {score}')

        except Exception as e:
            operations_logger.error(e)
            operations_logger.error(docid)
            failed.append(docid)
            nlp_cache.save_text(abstract, 'failed')


def save_sql_deprecated(cache_dir: str, file_type='.clinical.json', result_type=ResultType.clinical):
    """
    :param cache_dir:
    :return:
    """
    from biomed.db.sql import db

    operations_logger.info(f'reading from cache_dir {cache_dir}')

    for f in nlp_cache.list_contents(file_type, cache_dir):
        res = common.read_json(f)

        try:
            for match in ClinicalReader(res, result_type):
                for concept in match.concepts:
                    cui = concept.get('cui', "NA")
                    doc_id = os.path.basename(f)

                db.query(f"INSERT into annot (doc_id, cui, match_text) values {doc_id, cui, match.text}")
        except KeyError:
            operations_logger.error('ERROR')
            pass


def save_sql_concept_deprecated(text: str):
    """
    :param text:
    :return:
    """
    from biomed.db.sql import db

    docid = nlp_cache.hash_text(text)
    res   = nlp_cache.clinical(text)

    db.query(f"insert into doc (id) values ('{docid}')")

    for r in res['result']:
        if 'umlsConcept' in r.keys():
            for c in r['umlsConcept']:
                if 'cui' in c.keys():

                    #db.query("insert into annot (cui, tui, doc_id) values (:cui, :tui, :doc_id) ",
                    #         cui=c.get('cui', None),
                    #         tui=c.get('tui', None),
                    #         doc_id=_docid)

                    db.query("insert into concept (cui, tui, sab, code, str, doc_id) values (:cui, :tui, :sab, :code, :str, :doc_id) ",
                             cui=c.get('cui', None),
                             tui=c.get('tui', None),
                             sab=c.get('codingScheme',None),
                             code=c.get('code', None),
                             str=c.get('preferredText', None),
                             doc_id=docid)

