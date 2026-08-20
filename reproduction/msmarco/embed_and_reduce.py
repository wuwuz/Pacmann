#!/usr/bin/env python3
"""Encode MS-MARCO documents/queries and apply the paper's fitted PCA."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/msmarco-distilbert-base-tas-b"
MODEL_REVISION = "996dfc6404137c6d89c7bf647a4bae62fdf8dd9a"
EXPECTED_DOCUMENTS = 3_201_821
INPUT_DIMENSIONS = 768
OUTPUT_DIMENSIONS = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--pca", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but PyTorch cannot see an NVIDIA GPU")
    return requested


def load_model(cache_dir: Path, device: str) -> SentenceTransformer:
    print(f"Downloading/loading {MODEL_ID} at revision {MODEL_REVISION}")
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
    )
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    return model


def load_pca(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as pca:
        components = np.asarray(pca["components"], dtype=np.float64)
        mean = np.asarray(pca["mean"], dtype=np.float64)
    if components.shape != (OUTPUT_DIMENSIONS, INPUT_DIMENSIONS):
        raise ValueError(f"unexpected PCA component shape: {components.shape}")
    if mean.shape != (INPUT_DIMENSIONS,):
        raise ValueError(f"unexpected PCA mean shape: {mean.shape}")
    return components, mean


def normalized_document(line: str, row: int) -> tuple[str, str]:
    fields = line.rstrip("\r\n").split("\t", 1)
    if len(fields) != 2:
        raise ValueError(f"prepared corpus row {row:,} is not docid<TAB>text")
    docid, text = fields
    # This is the historical normalization. The prepared text is already at
    # most 512 characters; the token slice is retained for exact provenance.
    return docid, " ".join(text.split()[:512])


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    encoded = np.asarray(encoded, dtype=np.float32)
    if encoded.shape != (len(texts), INPUT_DIMENSIONS):
        raise ValueError(f"unexpected encoder output shape: {encoded.shape}")
    return encoded


def transform(encoded: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (encoded.astype(np.float64) - mean) @ components.T


def prepare_documents(
    model: SentenceTransformer,
    corpus: Path,
    output_dir: Path,
    components: np.ndarray,
    mean: np.ndarray,
    batch_size: int,
    expected_documents: int,
) -> None:
    final_vectors = output_dir / "msmarco_embeddings.npy"
    final_docids = output_dir / "msmarco_docid_permuted.npy"
    partial_vectors = output_dir / "msmarco_embeddings.partial.npy"
    partial_docids = output_dir / "msmarco_docid_permuted.partial.npy"
    checkpoint_path = output_dir / "embedding-checkpoint.json"

    if final_vectors.exists() and final_docids.exists():
        vectors = np.load(final_vectors, mmap_mode="r")
        docids = np.load(final_docids, mmap_mode="r")
        if vectors.shape == (expected_documents, OUTPUT_DIMENSIONS) and docids.shape == (
            expected_documents,
        ):
            print("Document embeddings already exist with the expected shapes; skipping.")
            return
        raise ValueError("existing document outputs have unexpected shapes; move them and retry")

    processed = 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        processed = int(checkpoint["processed_documents"])
        print(f"Resuming document encoding at row {processed:,}")
        vectors = np.lib.format.open_memmap(partial_vectors, mode="r+")
        docids = np.lib.format.open_memmap(partial_docids, mode="r+")
    else:
        vectors = np.lib.format.open_memmap(
            partial_vectors,
            mode="w+",
            dtype=np.float64,
            shape=(expected_documents, OUTPUT_DIMENSIONS),
        )
        docids = np.lib.format.open_memmap(
            partial_docids,
            mode="w+",
            dtype="<U8",
            shape=(expected_documents,),
        )

    with corpus.open(encoding="utf-8", newline="") as source:
        lines = itertools.islice(source, processed, None)
        while processed < expected_documents:
            batch_lines = list(itertools.islice(lines, batch_size))
            if not batch_lines:
                break
            parsed = [normalized_document(line, processed + index + 1) for index, line in enumerate(batch_lines)]
            batch_docids = [item[0] for item in parsed]
            if any(len(docid) > 8 for docid in batch_docids):
                raise ValueError("encountered a document ID longer than the canonical <U8 dtype")
            batch_texts = [item[1] for item in parsed]
            encoded = encode(model, batch_texts, batch_size)
            reduced = transform(encoded, components, mean)
            end = processed + len(batch_lines)
            vectors[processed:end] = reduced
            docids[processed:end] = batch_docids
            processed = end
            if processed % (batch_size * 100) == 0 or processed == expected_documents:
                vectors.flush()
                docids.flush()
                checkpoint_path.write_text(
                    json.dumps({"processed_documents": processed}) + "\n", encoding="utf-8"
                )
                print(f"  encoded {processed:,}/{expected_documents:,} documents", flush=True)

    if processed != expected_documents:
        raise ValueError(f"corpus ended after {processed:,} rows; expected {expected_documents:,}")
    del vectors, docids
    partial_vectors.replace(final_vectors)
    partial_docids.replace(final_docids)
    checkpoint_path.unlink(missing_ok=True)


def prepare_queries(
    model: SentenceTransformer,
    queries_path: Path,
    output_dir: Path,
    components: np.ndarray,
    mean: np.ndarray,
    batch_size: int,
) -> None:
    output = output_dir / "msmarco_queries.npy"
    queries: list[str] = []
    with queries_path.open(encoding="utf-8") as source:
        for row, line in enumerate(source, start=1):
            fields = line.strip().split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"query row {row} is not qid<TAB>query")
            queries.append(fields[1])
    if len(queries) != 1_000:
        raise ValueError(f"expected 1,000 queries, found {len(queries)}")
    encoded = encode(model, queries, batch_size)
    reduced = transform(encoded, components, mean)
    np.save(output, reduced)
    print(f"Saved {len(queries):,} reduced queries to {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_cache.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    if device == "cpu":
        print("WARNING: CPU encoding is supported but the full corpus can take several days.")
    print(f"Encoder device: {device}")
    components, mean = load_pca(args.pca)
    model = load_model(args.model_cache, device)
    prepare_documents(
        model,
        args.corpus,
        args.output_dir,
        components,
        mean,
        args.batch_size,
        args.expected_documents,
    )
    prepare_queries(model, args.queries, args.output_dir, components, mean, args.batch_size)
    print("Embedding and dimensionality reduction completed successfully.")


if __name__ == "__main__":
    main()
