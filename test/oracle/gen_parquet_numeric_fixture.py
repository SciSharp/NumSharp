"""
Write the pyarrow-made NUMERIC Parquet fixtures the NumSharp.Interop.ParquetNet tests read.

The neighbours of the text fixture (gen_parquet_text_fixture.py). One ten-row table - every integer width at its extremes,
float specials (NaN, +-inf, -0.0, the smallest subnormal, the largest finite), booleans and a decimal128(10, 2) as REQUIRED
columns (Parquet.Net decodes those straight into NDArray memory), an optional int32 without nulls, nullable int64 /
float64 / uint8 / bool columns with nulls, an all-null float64, and a timestamp and a date32 the bridge has no dtype
for - written in THREE row groups (row_group_size=4: 4 + 4 + 2) under four writer configurations:

    numeric_columns.parquet             pyarrow's defaults: dictionary pages, snappy, data page v1
    numeric_columns_plain_zstd.parquet  no dictionary (PLAIN values), zstd
    numeric_columns_delta_gzip.parquet  DELTA_BINARY_PACKED integers, BYTE_STREAM_SPLIT floats, the decimal stored as an
                                        INT64 (store_decimal_as_integer), gzip
    numeric_columns_v2.parquet          data page v2 (RLE-encoded booleans: Parquet.Net 6.1.0 cannot decode them)

Three boundary files sit beside them:

    empty_columns.parquet               the same schema with ZERO rows (every column reads as an empty array of its dtype)
    wide_decimals.parquet               decimals at and past .NET decimal's 96-bit range and 28-29 significant digits
    float16_column.parquet              a float16 column (Parquet.Net 6.1.0 refuses the whole schema)

The C# suite (NumericColumnParityTests) holds every column to pyarrow's own table.column(name).to_numpy() - the dtype and
the bits - in every configuration, and the nullable columns to the bridge's documented null policy (fill default(T), a
custom fill, or raise), which differs from numpy's view on purpose (pyarrow / pandas read a nullable integer as float64
with NaN and a nullable boolean as an object array with None).

    python test/oracle/gen_parquet_numeric_fixture.py   # writes test/NumSharp.Tests.Interop.ParquetNet/Fixtures/*.parquet

Requires: pyarrow, numpy. The files are COMMITTED, so the C# suite needs no Python at test time; the script re-reads every
file, checks it against the table it wrote, and prints each column's encodings and numpy view (what the C# test embeds).
"""
from __future__ import annotations

import decimal
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "NumSharp.Tests.Interop.ParquetNet", "Fixtures")

# The integer columns (physically INT32 / INT64) and the float columns, named for the per-column encodings below.
INTEGER_COLUMNS = ["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "opt_i32", "maybe_i64", "maybe_u8", "dec"]
FLOAT_COLUMNS = ["f32", "f64", "maybe_f64", "all_null"]

# Writer configuration per file: the same table, so every file has the same numpy view.
VARIANTS = {
    "numeric_columns.parquet": dict(),
    "numeric_columns_plain_zstd.parquet": dict(use_dictionary=False, compression="zstd"),
    "numeric_columns_delta_gzip.parquet": dict(
        use_dictionary=False, compression="gzip", store_decimal_as_integer=True,
        column_encoding={**{c: "DELTA_BINARY_PACKED" for c in INTEGER_COLUMNS},
                         **{c: "BYTE_STREAM_SPLIT" for c in FLOAT_COLUMNS}}),
    "numeric_columns_v2.parquet": dict(data_page_version="2.0"),
}


def numeric_table() -> pa.Table:
    """Ten rows: each integer width's extremes, float specials, booleans, nulls, a decimal and two temporal columns."""
    def ints(lo, hi):
        return [lo, hi, 0, 1, -1 if lo < 0 else 2, 5, 7, 8, 9, 10]

    def req(name, typ):
        return pa.field(name, typ, nullable=False)

    D = decimal.Decimal
    schema = pa.schema([
        req("i8", pa.int8()), req("i16", pa.int16()), req("i32", pa.int32()), req("i64", pa.int64()),
        req("u8", pa.uint8()), req("u16", pa.uint16()), req("u32", pa.uint32()), req("u64", pa.uint64()),
        req("f32", pa.float32()), req("f64", pa.float64()), req("flag", pa.bool_()), req("dec", pa.decimal128(10, 2)),
        pa.field("opt_i32", pa.int32()),
        pa.field("maybe_i64", pa.int64()), pa.field("maybe_f64", pa.float64()), pa.field("maybe_u8", pa.uint8()),
        pa.field("maybe_flag", pa.bool_()), pa.field("all_null", pa.float64()),
        pa.field("when", pa.timestamp("ms")), pa.field("day", pa.date32()),
    ])
    return pa.table({
        "i8": ints(-128, 127),
        "i16": ints(-32768, 32767),
        "i32": ints(-2 ** 31, 2 ** 31 - 1),
        "i64": ints(-2 ** 63, 2 ** 63 - 1),
        "u8": ints(0, 255),
        "u16": ints(0, 65535),
        "u32": ints(0, 2 ** 32 - 1),
        "u64": ints(0, 2 ** 64 - 1),
        "f32": [0.1, -0.0, float("nan"), float("inf"), float("-inf"), 1e-45, 3.4028234663852886e38, 1.0, 2.5, -7.25],
        "f64": [0.1, -0.0, float("nan"), float("inf"), float("-inf"), 5e-324, 1.7976931348623157e308, 1.0, 2.5, -7.25],
        "flag": [True, False, True, True, False, False, True, False, True, False],
        "dec": [D(v) for v in ["1.25", "-3.50", "0.00", "99999999.99", "-0.01", "7.00", "8.10", "9.99", "10.00", "-12345678.90"]],
        "opt_i32": list(range(100, 110)),
        "maybe_i64": [1, None, 3, None, 5, 6, None, 8, 9, None],
        "maybe_f64": [1.5, None, -0.0, None, 2.5, None, 3.5, 4.5, None, 5.5],
        "maybe_u8": [1, None, 255, 0, None, 5, 6, 7, None, 9],
        "maybe_flag": [True, None, False, True, None, False, True, None, False, True],
        "all_null": [None] * 10,
        "when": np.arange(0, 10).astype("datetime64[ms]"),
        "day": pa.array(list(range(10)), pa.int32()).cast(pa.date32()),
    }, schema=schema)


def wide_decimals_table() -> pa.Table:
    """Decimals at and past .NET decimal's limits: 96-bit magnitude (79228162514264337593543950335), 28-29 digits."""
    D = decimal.Decimal
    return pa.table({
        "d29_0_max": pa.array([D("79228162514264337593543950335"), D("-79228162514264337593543950335")], pa.decimal128(29, 0)),
        "d10_10": pa.array([D("0.1234567891"), D("-0.0000000001")], pa.decimal128(10, 10)),
        "d38_4_precise": pa.array([D("1234567890123456789012345678.1234"), D("1.0001")], pa.decimal128(38, 4)),
        "d29_0_over": pa.array([D("79228162514264337593543950336"), D("0")], pa.decimal128(29, 0)),
        "d38_0_huge": pa.array([D("100000000000000000000000000000000000"), D("1")], pa.decimal128(38, 0)),
    })


def float16_table() -> pa.Table:
    """An int32 next to a float16 (FLOAT16 logical type, FIXED_LEN_BYTE_ARRAY(2))."""
    return pa.table({"n": pa.array([1, 2, 3], pa.int32()),
                     "half": pa.array(np.array([1.5, -0.0, 65504.0], dtype=np.float16), pa.float16())})


def check_and_print(path: str, table: pa.Table) -> None:
    """Re-read the written file, compare it with the table bit for bit, and print each column's encodings and numpy view."""
    meta = pq.ParquetFile(path).metadata
    back = pq.read_table(path)
    assert back.schema.equals(table.schema), f"{path}: the schema reads back as written"
    for name in table.column_names:
        # Table.equals treats NaN as unequal to itself, so floats are compared by their bytes and the rest by value.
        written, read = table.column(name).to_numpy(zero_copy_only=False), back.column(name).to_numpy(zero_copy_only=False)
        if written.dtype.kind in "fc":
            assert written.tobytes() == read.tobytes(), name
        else:
            assert written.tolist() == read.tolist(), name
    rg = meta.row_group(0)
    encodings = {rg.column(i).path_in_schema: f"{rg.column(i).physical_type}/{rg.column(i).compression}/"
                                                 f"{'+'.join(sorted(rg.column(i).encodings))}" for i in range(rg.num_columns)}
    print(f"== {os.path.basename(path)}: {meta.num_row_groups} row groups, {os.path.getsize(path)} bytes")
    for name in back.column_names:
        arr = back.column(name).to_numpy(zero_copy_only=False)
        print(f"   {name}: arrow {back.column(name).type} [{encodings[name]}] -> numpy {arr.dtype} {arr.tolist()!r}")


def main() -> int:
    os.makedirs(FIXTURES, exist_ok=True)
    table = numeric_table()
    for name, options in VARIANTS.items():
        path = os.path.join(FIXTURES, name)
        pq.write_table(table, path, row_group_size=4, **options)
        assert pq.ParquetFile(path).metadata.num_row_groups == 3, name
        check_and_print(path, table)
    boundaries = (("empty_columns.parquet", table.slice(0, 0)), ("wide_decimals.parquet", wide_decimals_table()),
                  ("float16_column.parquet", float16_table()))
    for name, boundary in boundaries:
        path = os.path.join(FIXTURES, name)
        pq.write_table(boundary, path)
        check_and_print(path, boundary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
