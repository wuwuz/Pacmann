#!/usr/bin/env python3
"""Map Pacmann vector IDs to MS-MARCO doc IDs and compute MRR@100."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-results", required=True, type=Path)
    parser.add_argument("--docids", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--output-docids", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--minimum-mrr", type=float, default=0.24)
    parser.add_argument("--maximum-mrr", type=float, default=0.29)
    parser.add_argument("--k", type=int, default=100)
    return parser.parse_args()


def read_queries(path: Path) -> list[tuple[str, str]]:
    queries = []
    with path.open(encoding="utf-8") as source:
        for row, line in enumerate(source, start=1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2:
                raise ValueError(f"query row {row} is malformed")
            queries.append((fields[0], fields[1]))
    return queries


def read_qrels(path: Path) -> dict[str, str]:
    qrels: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for row, line in enumerate(source, start=1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"qrels row {row} is malformed")
            qrels.setdefault(fields[0], fields[2])
    return qrels


def read_results(path: Path, query_count: int, k: int) -> np.ndarray:
    with path.open("rb") as source:
        is_numpy = source.read(6) == b"\x93NUMPY"
    if is_numpy:
        results = np.load(path, allow_pickle=False)
        if results.shape != (query_count, k):
            raise ValueError(
                f"result array has shape {results.shape}; expected {(query_count, k)}"
            )
        if not np.issubdtype(results.dtype, np.integer):
            raise ValueError(f"result array must contain integers, found {results.dtype}")
        return np.asarray(results, dtype=np.int64)

    rows = []
    with path.open(encoding="utf-8") as source:
        for row, line in enumerate(source, start=1):
            values = [int(value) for value in line.split()]
            if len(values) != k:
                raise ValueError(f"result row {row} has {len(values)} values; expected {k}")
            rows.append(values)
    if len(rows) != query_count:
        raise ValueError(f"result file has {len(rows)} rows; expected {query_count}")
    return np.asarray(rows, dtype=np.int64)


def main() -> None:
    args = parse_args()
    queries = read_queries(args.queries)
    if len(queries) != 1_000:
        raise ValueError(f"expected 1,000 queries, found {len(queries)}")
    qrels = read_qrels(args.qrels)
    docids = np.load(args.docids, mmap_mode="r", allow_pickle=False)
    results = read_results(args.raw_results, len(queries), args.k)

    reciprocal_rank_sum = 0.0
    ranked = 0
    args.output_docids.parent.mkdir(parents=True, exist_ok=True)
    with args.output_docids.open("w", encoding="utf-8") as output:
        for query_index, (qid, query) in enumerate(queries):
            relevant = qrels[qid]
            output.write(f"Query: {qid} {query}\n")
            hit_rank = None
            for rank, vector_id in enumerate(results[query_index], start=1):
                if vector_id < 0 or vector_id >= len(docids):
                    returned_docid = "INVALID_VECTOR_ID"
                else:
                    returned_docid = str(docids[vector_id])
                output.write(returned_docid + "\n")
                if hit_rank is None and returned_docid == relevant:
                    hit_rank = rank
            output.write("----------\n\n")
            if hit_rank is not None:
                ranked += 1
                reciprocal_rank_sum += 1.0 / hit_rank

    mrr = reciprocal_rank_sum / len(queries)
    result = {
        "metric": "MRR@100",
        "mrr": mrr,
        "queries": len(queries),
        "ranked_queries": ranked,
        "expected_range": [args.minimum_mrr, args.maximum_mrr],
        "passed": args.minimum_mrr <= mrr <= args.maximum_mrr,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Ranked / Total = {ranked} / {len(queries)}")
    print(f"MRR@100: {mrr:.6f}")
    print(f"Expected stochastic range: [{args.minimum_mrr:.3f}, {args.maximum_mrr:.3f}]")
    if not result["passed"]:
        raise SystemExit(
            "MRR is outside the expected range. Compare all artifact hashes before interpreting the result."
        )


if __name__ == "__main__":
    main()
