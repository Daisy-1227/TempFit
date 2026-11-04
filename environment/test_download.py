import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download import download, validate


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.data = b'wheel test content'
        self.entry = {'file': 'example-1-py3-none-any.whl', 'bytes': len(self.data),
                      'sha256': hashlib.sha256(self.data).hexdigest(), 'url': 'https://example.org/wheel'}

    def test_verified_cache_never_requests_network(self):
        (self.folder / self.entry['file']).write_bytes(self.data)
        self.assertEqual(download(self.entry, self.folder, opener=lambda *a, **k: self.fail('network called')), 'cached')

    def test_corrupt_existing_file_replaced_only_after_verification(self):
        target = self.folder / self.entry['file']
        target.write_bytes(b'corrupt')
        self.assertEqual(download(self.entry, self.folder, opener=lambda *a, **k: io.BytesIO(self.data)), 'downloaded')
        self.assertEqual(target.read_bytes(), self.data)

    def test_corrupt_response_never_installed(self):
        with self.assertRaises(RuntimeError):
            download(self.entry, self.folder, attempts=1, opener=lambda *a, **k: io.BytesIO(b'invalid'))
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_retry_after_interruption(self):
        replies = [OSError('connection interrupted'), io.BytesIO(self.data)]
        def opener(*args, **kwargs):
            value = replies.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        with patch('download.time.sleep'):
            self.assertEqual(download(self.entry, self.folder, opener=opener), 'downloaded')
        self.assertEqual(len(list(self.folder.iterdir())), 1)

    def test_oversized_response_rejected(self):
        with self.assertRaises(RuntimeError):
            download(self.entry, self.folder, attempts=1, opener=lambda *a, **k: io.BytesIO(self.data * 2))
        self.assertEqual(list(self.folder.iterdir()), [])

    def test_paths_credentials_and_http_rejected(self):
        for change in [{'file': '../escape.whl'}, {'file': '/escape.whl'}, {'file': 'a\\escape.whl'},
                       {'url': 'http://example.org/wheel'}, {'url': 'https://user:secret@example.org/wheel'}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(dict(self.entry, **change))

    def test_symlink_rejected(self):
        outside = self.folder / 'outside'
        outside.write_bytes(self.data)
        (self.folder / self.entry['file']).symlink_to(outside)
        with self.assertRaises(ValueError):
            download(self.entry, self.folder)


if __name__ == '__main__':
    unittest.main()
