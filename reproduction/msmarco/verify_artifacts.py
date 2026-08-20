#!/usr/bin/env python3
"""Validate generated array shapes/dtypes and write a checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_DOCID_SHA256 = "ff9c383c59cad2f59624c708c73648b74ad7bfee83d1d0232008d59b16fc47c9"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--include-graph", action="store_true")
    args = parser.parse_args()

    expected = {
        "msmarco_embeddings.npy": ((3_201_821, 192), np.dtype("float64")),
        "msmarco_queries.npy": ((1_000, 192), np.dtype("float64")),
        "msmarco_docid_permuted.npy": ((3_201_821,), None),
    }
    manifest: dict[str, object] = {}
    for name, (shape, dtype) in expected.items():
        path = args.dataset_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != shape:
            raise ValueError(f"{name}: shape {array.shape}, expected {shape}")
        if dtype is not None and array.dtype != dtype:
            raise ValueError(f"{name}: dtype {array.dtype}, expected {dtype}")
        if name.startswith("msmarco_docid") and array.dtype.kind != "U":
            raise ValueError(f"{name}: expected Unicode document IDs, found {array.dtype}")
        print(f"Hashing {name} ({path.stat().st_size / 1024**3:.2f} GiB)")
        file_sha256 = sha256_file(path)
        manifest[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes": path.stat().st_size,
            "sha256": file_sha256,
        }
        if name == "msmarco_docid_permuted.npy" and file_sha256 != EXPECTED_DOCID_SHA256:
            raise ValueError(
                f"{name}: document order/IDs differ from the paper artifact; "
                f"got {file_sha256}, expected {EXPECTED_DOCID_SHA256}"
            )

    graph_path = args.dataset_dir / "msmarco_embeddings_3201821_192_32_graph.npy"
    if args.include_graph:
        graph = np.load(graph_path, mmap_mode="r", allow_pickle=False)
        if graph.shape != (3_201_821, 32) or graph.dtype != np.dtype("int32"):
            raise ValueError(f"unexpected graph metadata: shape={graph.shape}, dtype={graph.dtype}")
        print(f"Hashing {graph_path.name}")
        manifest[graph_path.name] = {
            "shape": list(graph.shape),
            "dtype": str(graph.dtype),
            "bytes": graph_path.stat().st_size,
            "sha256": sha256_file(graph_path),
        }

    output = args.dataset_dir / "artifact-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Artifact verification passed. Manifest: {output}")


if __name__ == "__main__":
    main()
