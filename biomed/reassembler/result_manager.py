import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    IO,
    BinaryIO,
    Optional,
    Union, Dict,
)

from text2phenotype.annotations.file_helpers import (
    _JsonDictGenerator,
    _JsonGeneratorBytesStream,
    _JsonListGenerator,
)
from text2phenotype.common.log import operations_logger
from text2phenotype.tasks.task_enums import TaskEnum
from text2phenotype.tasks.task_info import ChunksIterable

from .reassemble_functions import TASK_TO_REASSEMBLER_MAPPING, get_reassemble_function


class ReassemblerResultManager:
    def __init__(self):
        self._temp_directory: TemporaryDirectory = None

        # Raw data dict
        self.data: Optional[Union[list, dict]] = None

        # JSON dict generator
        self.json_generator: Optional[Union[_JsonDictGenerator, _JsonListGenerator]] = None

        # Filpath to file with result
        self.filepath: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._temp_directory:
            self._temp_directory.cleanup()
            operations_logger.info(f'Temporary directory "{self._temp_directory.name}" '
                                   f'has been cleaned up.')

    def reassemble_chunks(self, task: TaskEnum, chunk_mapping: ChunksIterable,
                          other_enum_to_chunk_mapping: Dict[TaskEnum, ChunksIterable],
                          deid_in_task_enums: bool = False) -> None:
        reassemble_function = get_reassemble_function(task, deid_in_task_enums)

        if reassemble_function:
            reassemble_function(chunk_mapping=chunk_mapping,
                                result_manager=self,
                                other_enum_to_chunk_mapping=other_enum_to_chunk_mapping)

    def open_temp_file(self, filename: Union[str, Path], mode: str) -> IO:
        if self._temp_directory is None:
            self._temp_directory = TemporaryDirectory(prefix=f'{self.__class__.__name__}_')
            operations_logger.info(f'Temporary directory "{self._temp_directory.name}" '
                                   f'has been created for intermediate files.')

        filepath = Path(self._temp_directory.name) / filename

        # Ensure that resolved filepath is relative to temp directory
        filepath.resolve().relative_to(Path(self._temp_directory.name).resolve())

        # Ensure that nested directories exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        return filepath.open(mode=mode)

    def to_json_stream(self) -> BinaryIO:
        if self.json_generator is not None:
            # Create temp file to save intermediate JSON
            with self.open_temp_file('json-generator-result-tmp-file.json', 'wb') as f:
                self.filepath = f.name

                # Generate JSON data into this temp file
                with _JsonGeneratorBytesStream(self.json_generator, encoding='utf-8') as stream:
                    while True:
                        json_bytes = stream.read(65536)  # 64 Kib
                        if not json_bytes:
                            break
                        f.write(json_bytes)

        if self.filepath:
            return open(self.filepath, 'rb')

        return _JsonGeneratorBytesStream(self.data, encoding='utf-8')

    def to_dict(self) -> str:
        with self.to_json_stream() as stream:
            return json.load(stream)
