import heapq
import itertools
import json
import typing
from typing import (
    BinaryIO,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import ijson

from text2phenotype.annotations.file_helpers import (
    _JsonDictGenerator,
    _JsonListGenerator,
)
from text2phenotype.common.featureset_annotations import (
    MachineAnnotation,
    RANGE,
    TOKEN,
)
from text2phenotype.common.log import operations_logger
from text2phenotype.constants.features import FeatureType
from text2phenotype.tasks.task_info import ChunksIterable

if typing.TYPE_CHECKING:
    from .result_manager import ReassemblerResultManager


ANNOTATIONS_LIST_TYPE_FEATURES_KEYS = {feature.name if isinstance(feature, FeatureType) else feature
                                       for feature in MachineAnnotation.LIST_TYPE_FEATURES} - {RANGE}

JSON_DICT_BYTE = b'{'
JSON_LIST_BYTE = b'['
KEY_SEPARATOR_BYTE = b'\x01'
NEW_LINE_BYTE = b'\n'


def reassemble_annotations(chunk_mapping: ChunksIterable,
                           result_manager: 'ReassemblerResultManager',
                           **kwargs) -> None:

    temp_files, all_keys = _prepare_chunk_data(chunk_mapping, result_manager)

    result_manager.json_generator = _create_result_json_generator(temp_files, all_keys)


def _prepare_chunk_data(chunk_mapping: ChunksIterable,
                        result_manager: 'ReassemblerResultManager') -> Tuple[List[str], List[str]]:
    """Prepare chunk data and write to temporary files"""

    all_keys: List[str] = []
    temp_files: List[str] = []
    chunk_num = 0

    token_index_offset = 0

    # This snippet is required to reduce memory consumption.
    # To avoid keeping a dictionary related to chunk which is done already,
    # we need to set this dictionary to null. In this case GC is able to remove it from memory.
    #
    # Max memory usage from memory profiler:
    #   * without gaps: ~1.7 GiB
    #   * with gaps: ~950 MiB (nearly two times less)

    def _iter_chunk_mapping_with_gaps() -> Iterable[Tuple[Optional[List[int]], Optional[dict]]]:
        """Specific iterator to reduce memory consumption"""

        for item in chunk_mapping:
            yield item
            del item
            yield None, None  # GC will release memory from the previous yield

    # Prepare chunk data and save to temporary file
    for chunk_span, chunk_data in _iter_chunk_mapping_with_gaps():
        if chunk_span is None:
            # Do nothing because it's a gap
            continue

        chunk_num += 1

        with result_manager.open_temp_file(f'chunk_{chunk_num:05d}.json', mode='wb') as f:
            operations_logger.info(f'Reassemble annotations - process chunk {chunk_num:05d}. '
                                   f'Chunk Span = {chunk_span}. Temp file = "{f.name}".')

            temp_files.append(f.name)

            # Find difference between all-keys and current dict keys
            chunk_data_keys = list(chunk_data.keys())
            new_keys_set = set(chunk_data_keys) - set(all_keys)

            # Append missed keys to the end of list
            if new_keys_set:
                all_keys += [k for k in chunk_data_keys if k in new_keys_set]

            # The order of keys should be the same between temporary files
            # These files will be merged/grouped by key iterally
            for key in all_keys:
                if key not in chunk_data:
                    continue

                data = chunk_data[key]

                # Update range spans and token indexes in the data-dict
                if key == RANGE:
                    # Update ranges
                    for span in data:
                        span[0] += chunk_span[0]
                        span[1] += chunk_span[0]

                elif key in ANNOTATIONS_LIST_TYPE_FEATURES_KEYS:
                    # No updates required
                    pass

                else:
                    # Update token indexes
                    result_data = {}

                    for token_index, feature_details in data.items():
                        result_index = str(int(token_index) + token_index_offset)
                        result_data[result_index] = feature_details

                    data = result_data

                # There is another trick to reduce memory consumption. It relates to how "heapq.merge()" works.
                # 80% of json size is a single section 'lionc_title'. While merging of temporary files
                # there is a possible situation when we keep 'loinc_title' of every chunk in memory.
                # It can be a brutal situation from memory point of view.
                # To prevent such situations the 'empty_data' line is added before real data.
                # In this case "heapq.merge()" generator will keep only short lines in memory.
                # Then it will read big lines with real data one by one (not in same time).

                # Write small-size line with empty data
                empty_data = [] if isinstance(data, list) else {}
                _write_json_line_to_file(f, key, empty_data)

                # Write real data
                _write_json_line_to_file(f, key, data)

            token_index_offset += len(chunk_data[TOKEN])

    return temp_files, all_keys


def _write_json_line_to_file(file: BinaryIO, key: str, data: Union[dict, list]) -> None:
    """Write (key, data) pair to a line of file"""

    # Format of line: {key}{KEY_SEPARATOR_BYTE}{json-encoded data}\n
    # Example: token\x01["token1", "token2"]\n

    file.write(key.encode('utf-8'))
    file.write(KEY_SEPARATOR_BYTE)
    file.write(json.dumps(data).encode('utf-8'))
    file.write(NEW_LINE_BYTE)


def _iter_json_lines_from_file(filepath: str) -> Iterable[Tuple[str, Iterable]]:
    """Iterate temporary file - (key, json_iterator)"""

    with open(filepath, 'rb') as f:
        for line in f:
            key, line = line.split(KEY_SEPARATOR_BYTE, 1)
            key = key.decode('utf-8')
            first_byte = line[:1]

            if first_byte == JSON_DICT_BYTE:
                yield key, ijson.kvitems(line, '')

            elif first_byte == JSON_LIST_BYTE:
                yield key, ijson.items(line, 'item')

            else:
                raise Exception('Unknown data in the temporary file. '
                                'Should be JSON encoded dict or list.')


def _create_result_json_generator(temp_files: List[str], all_keys: List[str]) -> _JsonDictGenerator:
    operations_logger.info(f'Reassemble annotations - combining {len(temp_files)} temp files '
                           f'into one JSON. Files list = {temp_files}.')

    # Mapping: key -> index in list
    all_keys_index = {key: index for index, key in enumerate(all_keys)}

    temp_files_iterators = map(_iter_json_lines_from_file, temp_files)

    # Merge temp files by original keys order
    merged_lines_iter = heapq.merge(*temp_files_iterators, key=lambda x: all_keys_index[x[0]])
    group_by_key_iter = itertools.groupby(merged_lines_iter, key=lambda x: x[0])

    def _iterate_kv_items() -> Iterable[Tuple[str, Union[_JsonDictGenerator, _JsonListGenerator]]]:
        for key, group in group_by_key_iter:
            operations_logger.info(f'Reassemble annotations - combining data for Key = {key}')

            # Combine all group to a single iterator
            data_iter = itertools.chain.from_iterable(json_iter for _, json_iter in group)

            # Generate JSON
            if key == RANGE or key in ANNOTATIONS_LIST_TYPE_FEATURES_KEYS:
                yield key, _JsonListGenerator(data_iter)
            else:
                yield key, _JsonDictGenerator(data_iter)

    return _JsonDictGenerator(_iterate_kv_items())
