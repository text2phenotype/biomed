import abc
import csv
from typing import Set

from text2phenotype.common import common
from text2phenotype.common.log import operations_logger


class CtakesAnnotator(object):
    """Base for testing cTAKES recall."""
    __metaclass__ = abc.ABCMeta

    """cTAKES term annotator."""
    def recall_expected(self, term_file: str, out_file: str):
        """Check expected terms against the NLP service.
        @param term_file: The file containing the terms to process.
        @:param out_file: The file to write missed terms to.
        """
        terms = self.__read_term_file(term_file)
        operations_logger.info(f'Identified {len(terms)} unique terms.')

        term_count = 0
        med_list = []
        full_match_count = 0
        found = set()
        for term in terms:
            term_count += 1

            med_list.append(term)
            if len(med_list) == 1000:
                full_match_count += self.__call_ctakes(med_list, terms, found)
                operations_logger.info(f"Processed {term_count} terms.  {len(found)} successful...")

                med_list.clear()

        full_match_count += self.__call_ctakes(med_list, terms, found)

        missed = terms - found

        total = len(found) + len(missed)
        operations_logger.info(
            f"Did not get a result for {len(missed)} of {total} ({100 * len(missed) / total}%) labs.")
        operations_logger.info(f'Full term matches: {full_match_count}')

        common.write_text('\n'.join(sorted(missed)), out_file)
        operations_logger.info(f"Unrecognized terms written to {out_file}.\n\n")

    @abc.abstractmethod
    def _get_autocode_entities(self, text: str):
        pass

    def __call_ctakes(self, med_list, med_terms, found_terms):
        delim = '.\n####\n'
        if not len(med_list):
            return 0

        operations_logger.info(f'Annotating {len(med_list)} terms...')
        med_text = delim.join(med_list)

        return self.__process_ner_results(self._get_autocode_entities(med_text), med_text, found_terms, med_terms)

    def __process_ner_results(self, entities, med_text, found_terms, med_terms):
        full_match_count = 0
        for entity in entities:
            term = self.__get_matched_term(entity, med_text)

            if term not in med_terms:
                raise Exception(f'"{term}"')

            matched_term = entity['text'][0]
            if term.lower() == matched_term.lower():
                if term not in found_terms:
                    found_terms.add(term)
                    full_match_count += 1

                continue

            found_terms.add(term)

        return full_match_count

    @staticmethod
    def __get_matched_term(entity, med_text):
        term, start, end = entity['text']
        while start >= 0 and med_text[start] != '\n':
            start -= 1

        start += 1

        if term.endswith('.'):
            end -= 1

        while end <= len(med_text) - 1 and med_text[end:end+2] != '.\n':
            end += 1

        return med_text[start:end]

    @staticmethod
    def __read_term_file(term_file) -> Set[str]:
        """Read the query term file.
        @param term_file: The query term file.
        @return: Set of unique query terms.
        """
        meds = set()

        with open(term_file, 'r') as med_fh:
            for record in csv.reader(med_fh, doublequote=False, escapechar='\\', quoting=csv.QUOTE_ALL):
                meds.add(record[0].strip())

            operations_logger.info(f'Read {len(meds)} unique terms from file.')

        return meds
