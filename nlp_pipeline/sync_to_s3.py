"""Upload NLP pipeline outputs (contracts, review queue, logs, splits) to S3."""

from __future__ import annotations

import argparse
from pathlib import Path

import boto3

DEFAULT_BUCKET = "htr-cremma-medieval"
DEFAULT_PREFIX = "nlp"

SOURCE_DIRS = ["data/contracts", "data/review", "data/splits_nlp"]


def iter_files(source_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for source_dir in source_dirs:
        root = Path(source_dir)
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file())
    return files


def upload_files(files: list[Path], bucket: str, prefix: str) -> None:
    s3 = boto3.client("s3")
    for path in files:
        key = f"{prefix}/{path.as_posix()}"
        print(f"Uploading {path} -> s3://{bucket}/{key}")
        s3.upload_file(str(path), bucket, key)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync NLP pipeline outputs to S3")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Directory to sync (repeatable). Defaults to data/contracts, data/review, data/splits_nlp",
    )
    args = parser.parse_args()

    sources = args.sources or SOURCE_DIRS
    files = iter_files(sources)
    if not files:
        print(f"No files found in: {', '.join(sources)}")
        return 1

    upload_files(files, args.bucket, args.prefix)
    print(f"Uploaded {len(files)} files to s3://{args.bucket}/{args.prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
