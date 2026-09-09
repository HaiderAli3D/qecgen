"""Small read-only inspection command for acquired pilot files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    table = pq.read_table(args.path)
    print(table.schema)
    print(json.dumps({"rows": len(table), "first_row": table.slice(0, 1).to_pylist()}, indent=2))


if __name__ == "__main__":
    main()
