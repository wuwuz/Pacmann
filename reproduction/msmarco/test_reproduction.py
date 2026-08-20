#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load_script(name: str):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_corpus.py")
embed = load_script("embed_and_reduce.py")
evaluate = load_script("evaluate.py")


class CorpusPreparationTests(unittest.TestCase):
    def test_raw_quotes_truncation_and_empty_body_filter(self):
        rows = (
            'D1\thttps://one\tA ""quoted"" title\t"A body with raw quotes"\r\n'
            "D2\thttps://two\t.\t\r\n"
            "D3\thttps://three\t\tBody three\r\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "documents.tsv.gz"
            output = root / "prefix.tsv"
            with gzip.open(archive, "wb") as target:
                target.write(rows.encode("utf-8"))
            result = prepare.prepare_documents(archive, output, limit=2)
            self.assertEqual(result["official_rows_read"], 3)
            self.assertEqual(result["documents_written"], 2)
            self.assertEqual(result["empty_body_rows_skipped"], 1)
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                'D1\tA ""quoted"" title "A body with raw quotes"\nD3\t Body three\n',
            )

    def test_historical_whitespace_normalization(self):
        docid, text = embed.normalized_document("D1\t  first\tsecond   third  \n", 1)
        self.assertEqual(docid, "D1")
        self.assertEqual(text, "first second third")


class PcaAssetTests(unittest.TestCase):
    def test_shapes_and_raw_hashes(self):
        components, mean = embed.load_pca(HERE / "assets" / "pca_192.npz")
        self.assertEqual(components.shape, (192, 768))
        self.assertEqual(mean.shape, (768,))
        self.assertEqual(
            hashlib.sha256(components.tobytes()).hexdigest(),
            "a33c1929169a437da9c514fdcf9490ab81b8761ec12c77908e3370890a5fac97",
        )
        self.assertEqual(
            hashlib.sha256(mean.tobytes()).hexdigest(),
            "1424e900221d9d002ed8d311ab61d32465fad0495b307b641e58cc527b3578b2",
        )

    def test_transform_formula(self):
        components = np.eye(2, dtype=np.float64)
        mean = np.array([1.0, 2.0], dtype=np.float64)
        encoded = np.array([[4.0, 8.0]], dtype=np.float32)
        np.testing.assert_array_equal(embed.transform(encoded, components, mean), [[3.0, 6.0]])


class EvaluationTests(unittest.TestCase):
    def test_result_reader_rejects_wrong_width(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.txt"
            result.write_text("1 2\n3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 2"):
                evaluate.read_results(result, query_count=2, k=2)

    def test_result_reader_accepts_numpy_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.npy"
            expected = np.array([[1, 2], [3, 4]], dtype=np.int32)
            np.save(result, expected)
            np.testing.assert_array_equal(
                evaluate.read_results(result, query_count=2, k=2), expected
            )


if __name__ == "__main__":
    unittest.main()
