import argparse
from typing import List, Set
import sys

from text2phenotype.common.common import get_file_list, write_text, read_text
from text2phenotype.common.log import operations_logger
from text2phenotype.entity.brat import BratReader


def __extract_terms(aspect_type: str, file_list: List[str]) -> Set[str]:
    terms = set()

    reader = BratReader()
    for f in file_list:
        text = read_text(f)
        if not len(text):
            continue

        reader.from_brat(text)

        for annotation in reader.annotations.values():
            if annotation.aspect == aspect_type:
                terms.add(annotation.text.lower())

    return terms


def __parse_args(args: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", help="The annotation file root directory.")
    parser.add_argument("--type", dest="type", help="The annotation aspect type to extract.")
    parser.add_argument("--out", dest="out", help="The unique term output file.")

    return parser.parse_args(args)


def main(args: List[str]):
    ns_args = __parse_args(args)

    ann_files = get_file_list(ns_args.dir, "ann", recurse=True)
    operations_logger.info(f"Found {len(ann_files)} annotation files.")

    terms = __extract_terms(ns_args.type, ann_files)
    operations_logger.info(f"Extracted {len(terms)} unique terms.")

    write_text('\n'.join(terms), ns_args.out)


if __name__ == '__main__':
    main(sys.argv[1:])
