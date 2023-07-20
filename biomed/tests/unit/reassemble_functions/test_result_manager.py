import json
import unittest
from pathlib import Path
from uuid import uuid4

from biomed.reassembler import (
    ReassemblerResultManager,
    TASK_TO_REASSEMBLER_MAPPING
)
from biomed.reassembler.reassemble_functions import (
    do_nothing,
    reassemble_annotations,
)


class TestReassembleResultManager(unittest.TestCase):
    def test_json_stream(self):
        # `reassemble_annotations` is optional, enabled by env variable
        all_reassemble_functions = set(TASK_TO_REASSEMBLER_MAPPING.values()) | {do_nothing, reassemble_annotations}

        for func in all_reassemble_functions:
            with self.subTest(f'Test result manager with "{func.__name__}" function'):
                with ReassemblerResultManager() as res_manager:
                    chunks_mapping = iter([])  # No chunks
                    func(chunks_mapping, res_manager)

                    # JSON stream should be available as result
                    stream = res_manager.to_json_stream()
                    self.assertIsNotNone(stream)

                    # Stream content should be valid JSON string
                    result_data = json.load(stream)
                    self.assertTrue(isinstance(result_data, (dict, list, type(None))))

    def test_temporary_files(self):
        test_file_content = bytes(uuid4().hex, 'utf-8')

        with ReassemblerResultManager() as res_manager:
            self.assertIsNone(res_manager._temp_directory)

            with res_manager.open_temp_file(f'{uuid4().hex}.txt', 'wb') as f1:
                f1.write(test_file_content)

            with res_manager.open_temp_file(f'check/relative/path/with/nested/directories/{uuid4().hex}.txt', 'wb') as f2:
                f2.write(test_file_content)

            expected_files = sorted(Path(fileobj.name) for fileobj in [f1, f2])

            self.assertIsNotNone(res_manager._temp_directory)
            temp_dir = Path(res_manager._temp_directory.name)

            temp_files = sorted(temp_dir.glob('**/*.txt'))
            self.assertListEqual(temp_files, expected_files)

            # Check that files can be opened with `open_tmp_files()` method
            for t in expected_files:
                with res_manager.open_temp_file(t, 'rb') as f:
                    self.assertEqual(f.read(), test_file_content)

        # Temp directory should be cleaned up
        self.assertFalse(Path(res_manager._temp_directory.name).exists())

        for t in temp_files:
            self.assertFalse(t.exists())

    def test_temporary_files_wrong_path(self):
        # File path should be relative to temporary dicrectory
        wrong_paths = ['/etc/passwd', '../etc/passwd']

        for filepath in wrong_paths:
            with self.subTest(f'Test wrong path: "{filepath}"'), \
                 ReassemblerResultManager() as res_manager, \
                 self.assertRaises(ValueError), \
                 res_manager.open_temp_file(filepath, 'r'):
                    pass
