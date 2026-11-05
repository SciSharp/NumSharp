"""
Write the pyarrow-made Parquet fixture the NumSharp.Interop.ParquetNet text tests read.

The ParquetNet bridge reads numeric and boolean columns and refuses text; pyarrow (the engine behind pandas'
read_parquet / to_parquet) is the writer most Parquet files in the Python ecosystem come from, so a file it wrote is
the realistic input: text, large text, binary, dictionary-encoded text, a list of text and nullable text sit next to
numeric columns of several widths. The C# suite holds every numeric column to what pyarrow's own
`table.column(name).to_numpy()` gives (dtype and values, printed below for the test to embed), and every text-like
column to the bridge's refusal.

    python test/oracle/gen_parquet_text_fixture.py      # writes test/NumSharp.Tests.Interop.ParquetNet/Fixtures/text_columns.parquet

Requires: pyarrow, numpy. The file is COMMITTED, so the C# suite needs no Python at test time. Re-run only when the
fixture changes; the script re-reads the file and prints the numpy view of each column, which is what the C# test
asserts.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NumSharp.Tests.Interop.ParquetNet", "Fixtures",
                   "text_columns.parquet")


def build() -> pa.Table:
    """Six rows: numeric columns of several widths beside every text-like column pyarrow writes."""
    names = ["alpha", "", "e" + chr(0x301) + "t" + chr(0xe9), chr(0x1F600), None, "x\x00y"]
    return pa.table({
        "id": pa.array([0, 1, 2, 3, 4, 5], pa.int64()),
        "price": pa.array([0.5, 1.25, -3.0, 1e300, -0.0, 2.5], pa.float64()),
        "ratio": pa.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], pa.float32()),
        "small": pa.array([-128, 127, 0, 1, -1, 5], pa.int8()),
        "wide": pa.array([0, 65535, 1, 2, 3, 4], pa.uint16()),
        "flag": pa.array([True, False, True, True, False, False], pa.bool_()),
        "name": pa.array(names, pa.string()),
        "name_large": pa.array(names, pa.large_string()),
        "blob": pa.array([b"\x00\x01", b"", b"\xff", b"abc", None, b"z"], pa.binary()),
        "category": pa.array(["cat", "dog", "cat", "bird", "dog", "cat"]).dictionary_encode(),
        "tags": pa.array([["a"], [], ["b", "c"], None, ["d"], ["e", "f", "g"]], pa.list_(pa.string())),
    })


def main() -> int:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    table = build()
    # One row group, no compression codec beyond pyarrow's default (snappy), dictionary encoding on (pyarrow's default),
    # so the file is what df.to_parquet() writes.
    pq.write_table(table, OUT)
    back = pq.read_table(OUT)
    assert back.equals(table), "the fixture reads back as written"
    for name in back.column_names:
        col = back.column(name)
        try:
            arr = col.to_numpy()
            print(f"{name}: arrow {col.type} -> numpy {arr.dtype} {arr.tolist()!r}")
        except Exception as e:  # pragma: no cover - list columns convert to object arrays
            print(f"{name}: arrow {col.type} -> {type(e).__name__}: {e}")
    print(f"{os.path.getsize(OUT)} bytes -> {os.path.relpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
