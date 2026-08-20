#!/usr/bin/env python3
"""Create the exact MS-MARCO inputs used by the Pacmann paper."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED_DOCUMENTS = 3_201_821
EXPECTED_OFFICIAL_DOCUMENTS = 3_213_835
EXPECTED_SKIPPED_EMPTY_BODIES = 12_014
EXPECTED_PREFIX_SHA256 = "8c1f99f1ddb68bd6e12d05d7f829e8635c200a29371e32412bf49aa315565e0c"
EXPECTED_QUERIES_SHA256 = "bb97b748dda44cf2352fb128efcbcc0ad157f7daaa572c2f5d4cdc3f0191e47b"
EXPECTED_QRELS_SHA256 = "28e2e5ccc17cd507875ded42c77434129694594d2cd2ec48f58ec1abc13cee43"
EXPECTED_QUERIES_1000_SHA256 = "29f31149bbb653cbcba7544f12faca207de0abfb56e22eac7bf41b6b0ffa2fba"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _ in source)


def replace_atomically(partial: Path, destination: Path) -> None:
    partial.replace(destination)


def prepare_documents(archive: Path, destination: Path, limit: int | None) -> dict[str, object]:
    if destination.exists() and limit is None:
        existing = {
            "official_rows_read": EXPECTED_OFFICIAL_DOCUMENTS,
            "documents_written": count_lines(destination),
            "empty_body_rows_skipped": EXPECTED_SKIPPED_EMPTY_BODIES,
            "sha256": sha256_file(destination),
        }
        expected = {
            "official_rows_read": EXPECTED_OFFICIAL_DOCUMENTS,
            "documents_written": EXPECTED_DOCUMENTS,
            "empty_body_rows_skipped": EXPECTED_SKIPPED_EMPTY_BODIES,
            "sha256": EXPECTED_PREFIX_SHA256,
        }
        if existing == expected:
            print(f"Prepared document corpus already verified: {destination}")
            return existing
        raise ValueError(
            f"existing prepared corpus is not canonical: {destination}\n"
            "Move it away before retrying."
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    official_rows = 0
    written_rows = 0
    skipped_empty_body = 0

    print(f"Preparing documents from {archive}")
    with gzip.open(archive, "rt", encoding="utf-8", newline="") as source, partial.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for raw_line in source:
            official_rows += 1
            line = raw_line.rstrip("\r\n")
            fields = line.split("\t", 3)
            if len(fields) != 4:
                raise ValueError(
                    f"official document row {official_rows:,} has {len(fields)} fields; expected 4"
                )
            docid, _url, title, body = fields
            if body == "":
                skipped_empty_body += 1
                continue

            # This deliberately preserves the raw TSV quote characters. The
            # historical preprocessing was a raw-field transformation, not a
            # CSV-unquoting transformation.
            text = (title + " " + body).rstrip()[:512]
            output.write(f"{docid}\t{text}\n")
            written_rows += 1

            if written_rows % 100_000 == 0:
                print(f"  prepared {written_rows:,} documents", flush=True)
            if limit is not None and written_rows >= limit:
                break

    replace_atomically(partial, destination)
    result: dict[str, object] = {
        "official_rows_read": official_rows,
        "documents_written": written_rows,
        "empty_body_rows_skipped": skipped_empty_body,
        "sha256": sha256_file(destination),
    }

    if limit is None:
        expected = {
            "official_rows_read": EXPECTED_OFFICIAL_DOCUMENTS,
            "documents_written": EXPECTED_DOCUMENTS,
            "empty_body_rows_skipped": EXPECTED_SKIPPED_EMPTY_BODIES,
            "sha256": EXPECTED_PREFIX_SHA256,
        }
        if result != expected:
            raise ValueError(f"document preparation mismatch:\nactual={result}\nexpected={expected}")
    return result


def decompress_and_verify(
    archive: Path, destination: Path, expected_sha256: str, expected_rows: int
) -> dict[str, object]:
    if destination.exists():
        existing = {"rows": count_lines(destination), "sha256": sha256_file(destination)}
        expected = {"rows": expected_rows, "sha256": expected_sha256}
        if existing == expected:
            print(f"Official input already verified: {destination}")
            return existing
        raise ValueError(f"existing input is not canonical: {destination}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    with gzip.open(archive, "rb") as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    replace_atomically(partial, destination)
    result = {"rows": count_lines(destination), "sha256": sha256_file(destination)}
    expected = {"rows": expected_rows, "sha256": expected_sha256}
    if result != expected:
        raise ValueError(f"input mismatch for {destination}: actual={result}, expected={expected}")
    return result


def make_first_queries(source: Path, destination: Path, count: int = 1_000) -> dict[str, object]:
    if destination.exists():
        existing = {"rows": count_lines(destination), "sha256": sha256_file(destination)}
        expected = {"rows": count, "sha256": EXPECTED_QUERIES_1000_SHA256}
        if existing == expected:
            print(f"First-query selection already verified: {destination}")
            return existing
        raise ValueError(f"existing first-query file is not canonical: {destination}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    with source.open("rb") as input_file, partial.open("wb") as output:
        for index, line in enumerate(input_file):
            if index == count:
                break
            output.write(line)
    replace_atomically(partial, destination)
    result = {"rows": count_lines(destination), "sha256": sha256_file(destination)}
    expected = {"rows": count, "sha256": EXPECTED_QUERIES_1000_SHA256}
    if result != expected:
        raise ValueError(f"first-query selection mismatch: actual={result}, expected={expected}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents-gz", required=True, type=Path)
    parser.add_argument("--queries-gz", required=True, type=Path)
    parser.add_argument("--qrels-gz", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--limit-documents",
        type=int,
        help="development-only: stop after this many included documents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.documents_gz, args.queries_gz, args.qrels_gz):
        if not path.exists():
            raise FileNotFoundError(path)

    documents = args.output_dir / "msmarco-prefix.tsv"
    queries = args.output_dir / "msmarco-docdev-queries.tsv"
    qrels = args.output_dir / "msmarco-docdev-qrels.tsv"
    queries_1000 = args.output_dir / "msmarco-queries-1000.tsv"

    manifest = {
        "documents": prepare_documents(args.documents_gz, documents, args.limit_documents),
        "queries": decompress_and_verify(args.queries_gz, queries, EXPECTED_QUERIES_SHA256, 5_193),
        "qrels": decompress_and_verify(args.qrels_gz, qrels, EXPECTED_QRELS_SHA256, 5_193),
        "queries_1000": make_first_queries(queries, queries_1000),
    }
    manifest_path = args.output_dir / "prepared-inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prepared inputs successfully. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
