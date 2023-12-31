import sys
from typing import List, Dict

from biomed.data_sources.data_source import BiomedDataSource

from text2phenotype.annotations.file_helpers import Annotation
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import DocumentTypeLabel, LabelEnum


class DocumentTypeDataSource(BiomedDataSource):
    def __init__(self, ds: BiomedDataSource):
        self.__dict__.update(ds.__dict__)

    @staticmethod
    def match_for_gold(token_ranges, token_text_list, brat_res: List[Annotation], label_enum: LabelEnum,
                       binary_classifier: bool = False) -> Dict[int, list]:
        if brat_res and type(list(label_enum)[0]) == DocumentTypeLabel:
            operations_logger.debug('Converting document type annotations...')
            brat_res = DocumentTypeDataSource.__sync_annotations(brat_res, token_ranges, token_text_list)

        operations_logger.debug('Matching for gold...')
        return BiomedDataSource.match_for_gold(token_ranges, token_text_list,
                                               brat_res, label_enum, binary_classifier)

    @staticmethod
    def __sync_annotations(brat_res, token_ranges, token_text_list):
        # filter NAs (these are not available to annotators and likely were DOS annotations)
        brat_res = sorted([a for a in brat_res if a.label != DocumentTypeLabel.na.name], key=lambda x: x.text_range)

        if not brat_res:
            return []

        new_annotations = []

        token_index = 0
        annotation = brat_res[0]
        while token_index < len(token_ranges) and annotation.text_range[0] >= token_ranges[token_index][1]:
            new_annotations.append(Annotation(DocumentTypeLabel.non_clinical.name,
                                              token_ranges[token_index],
                                              token_text_list[token_index]))

            token_index += 1

        n_annotations = len(brat_res)
        for i in range(n_annotations):
            annotation = brat_res[i]

            next_annotation_start = brat_res[i + 1].text_range[0] if i < n_annotations - 1 else sys.maxsize

            while token_index < len(token_ranges) and next_annotation_start >= token_ranges[token_index][1]:
                new_annotations.append(Annotation(annotation.label,
                                                  token_ranges[token_index],
                                                  token_text_list[token_index]))

                token_index += 1

        while token_index < len(token_ranges):
            new_annotations.append(Annotation(annotation.label,
                                              token_ranges[token_index],
                                              token_text_list[token_index]))

            token_index += 1

        return new_annotations
