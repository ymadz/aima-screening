"""Synthetic archive integrity checks; no patient fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.preserve_local_experiment import copy_verified, tree_checksums, verify_archive


class PreservationTests(unittest.TestCase):
    def test_copy_is_exact_private_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, copied = root/'source', root/'archive'/'copy'
            original.write_bytes(b'synthetic reproducibility record')
            copy_verified(original, copied)
            self.assertEqual(copied.read_bytes(), original.read_bytes())
            self.assertEqual(copied.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                copy_verified(original, copied)
            self.assertEqual(original.read_bytes(), b'synthetic reproducibility record')

    def test_archive_rejects_changed_or_extra_files_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            (archive/'record.json').write_text('{}')
            expected = tree_checksums(archive)
            (archive/'checksum_manifest.json').write_text(json.dumps(expected))
            with patch('src.preserve_local_experiment.verify_saved_cv', return_value={'status':'reproduced_without_retraining','subjects':119}) as verify:
                self.assertEqual(verify_archive(archive)['files'], 1)
                verify.assert_called_once()
            (archive/'extra').write_text('unexpected')
            with patch('src.preserve_local_experiment.verify_saved_cv') as verify:
                with self.assertRaises(ValueError):
                    verify_archive(archive)
                verify.assert_not_called()
            (archive/'extra').unlink()
            (archive/'record.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_archive(archive)
