#!/usr/bin/env python3
"""Compare deterministic random samples with surviving paper artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from embed_and_reduce import encode, load_model, load_pca, normalized_document, resolve_device, transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--generated-documents", type=Path)
    parser.add_argument("--generated-queries", type=Path)
    parser.add_argument("--pca", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.generated_documents is None) != (args.generated_queries is None):
        raise ValueError(
            "--generated-documents and --generated-queries must be supplied together"
        )
    with np.load(args.reference, allow_pickle=False) as reference:
        indices = reference["indices"]
        trial_seeds = reference["trial_seeds"]
        expected_docids = reference["docids"]
        expected_text_hashes = reference["text_sha256"]
        expected_raw = reference["raw_documents"]
        expected_reduced = reference["reduced_documents"]
        expected_queries = reference["reduced_queries"]

    wanted = {int(index): position for position, index in enumerate(indices)}
    texts: list[str | None] = [None] * len(indices)
    actual_docids: list[str | None] = [None] * len(indices)
    with args.corpus.open(encoding="utf-8", newline="") as source:
        for index, line in enumerate(source):
            position = wanted.get(index)
            if position is not None:
                docid, text = normalized_document(line, index + 1)
                actual_docids[position] = docid
                texts[position] = text
            if index >= int(indices[-1]) and all(text is not None for text in texts):
                break
    if any(text is None for text in texts):
        raise ValueError("the prepared corpus ended before all validation samples were found")
    if not np.array_equal(np.asarray(actual_docids), expected_docids):
        raise ValueError("sampled document IDs do not match the paper artifacts")
    actual_hashes = np.asarray(
        [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts if text is not None]
    )
    if not np.array_equal(actual_hashes, expected_text_hashes):
        raise ValueError("sampled prepared document text does not match the paper artifacts")

    device = resolve_device(args.device)
    model = load_model(args.model_cache, device)
    components, mean = load_pca(args.pca)
    raw = encode(model, [text for text in texts if text is not None], args.batch_size)
    reduced = transform(raw, components, mean)
    generated_documents = None
    generated_queries = None
    if args.generated_documents is not None:
        generated_documents = np.load(
            args.generated_documents, mmap_mode="r", allow_pickle=False
        )
        generated_queries = np.load(args.generated_queries, mmap_mode="r", allow_pickle=False)

    trials = []
    for seed in sorted(set(map(int, trial_seeds))):
        positions = np.flatnonzero(trial_seeds == seed)
        raw_delta = np.abs(raw[positions] - expected_raw[positions])
        reduced_delta = np.abs(reduced[positions] - expected_reduced[positions])
        written_max = None
        if generated_documents is not None:
            written_delta = np.abs(
                np.asarray(generated_documents[indices[positions]])
                - expected_reduced[positions]
            )
            written_max = float(written_delta.max())
        trial = {
            "seed": seed,
            "documents": len(positions),
            "raw_max_abs": float(raw_delta.max()),
            "reduced_max_abs": float(reduced_delta.max()),
            "written_reduced_max_abs": written_max,
        }
        trials.append(trial)
        print(
            f"sample seed={seed}: {len(positions)} documents, "
            f"raw max={trial['raw_max_abs']:.3g}, "
            f"reduced max={trial['reduced_max_abs']:.3g}, "
            + (
                f"written max={written_max:.3g}"
                if written_max is not None
                else "written array not requested"
            )
        )

    query_texts = []
    with args.queries.open(encoding="utf-8") as source:
        for row, line in enumerate(source, start=1):
            fields = line.strip().split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"query row {row} is malformed")
            query_texts.append(fields[1])
    raw_queries = encode(model, query_texts, args.batch_size)
    reduced_queries = transform(raw_queries, components, mean)
    query_delta = np.abs(reduced_queries - expected_queries)
    written_query_max = None
    if generated_queries is not None:
        written_query_delta = np.abs(np.asarray(generated_queries) - expected_queries)
        written_query_max = float(written_query_delta.max())
    print(
        f"all 1,000 queries: regenerated max={query_delta.max():.3g}, "
        + (
            f"written max={written_query_max:.3g}"
            if written_query_max is not None
            else "written array not requested"
        )
    )

    maxima = [
        value
        for trial in trials
        for value in (trial["raw_max_abs"], trial["reduced_max_abs"])
    ] + [float(query_delta.max())]
    maxima.extend(
        trial["written_reduced_max_abs"]
        for trial in trials
        if trial["written_reduced_max_abs"] is not None
    )
    if written_query_max is not None:
        maxima.append(written_query_max)
    result = {
        "absolute_tolerance": args.absolute_tolerance,
        "maximum_observed_error": max(maxima),
        "query_regenerated_max_abs": float(query_delta.max()),
        "query_written_max_abs": written_query_max,
        "trials": trials,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["maximum_observed_error"] > args.absolute_tolerance:
        raise SystemExit(
            f"sample validation exceeded tolerance: {result['maximum_observed_error']:.3g} > "
            f"{args.absolute_tolerance:.3g}"
        )
    print(f"Sample validation passed. Details: {args.output}")


if __name__ == "__main__":
    main()
