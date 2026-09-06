#!/usr/bin/env python3
"""Shared benchmark depth and dtype selection contract."""

from __future__ import annotations

from dataclasses import dataclass


ALL_DTYPES = (
    "bool", "uint8", "int8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
    "char", "float16", "float32", "float64", "decimal", "complex128",
)

DTYPE_ALIASES = {
    "bool": "bool", "boolean": "bool",
    "byte": "uint8", "u8": "uint8", "uint8": "uint8",
    "sbyte": "int8", "i8": "int8", "int8": "int8",
    "short": "int16", "i16": "int16", "int16": "int16",
    "ushort": "uint16", "u16": "uint16", "uint16": "uint16",
    "int": "int32", "i32": "int32", "int32": "int32",
    "uint": "uint32", "u32": "uint32", "uint32": "uint32",
    "long": "int64", "i64": "int64", "int64": "int64",
    "ulong": "uint64", "u64": "uint64", "uint64": "uint64",
    "char": "char",
    "half": "float16", "f16": "float16", "float16": "float16",
    "single": "float32", "float": "float32", "f32": "float32", "float32": "float32",
    "double": "float64", "f64": "float64", "float64": "float64",
    "decimal": "decimal", "dec": "decimal",
    "complex": "complex128", "c128": "complex128", "complex128": "complex128",
}

SHORT_DTYPES = {
    "bool": "bool", "uint8": "u8", "int8": "i8", "int16": "i16", "uint16": "u16",
    "int32": "i32", "uint32": "u32", "int64": "i64", "uint64": "u64", "char": "char",
    "float16": "f16", "float32": "f32", "float64": "f64", "decimal": "dec",
    "complex128": "c128",
}


@dataclass(frozen=True)
class Depth:
    name: str
    bdn_measurements: int
    bdn_warmups: int
    numpy_budget_divisor: int
    numpy_warmups: int


DEPTHS = {
    "pass": Depth("pass", bdn_measurements=1, bdn_warmups=0,
                  numpy_budget_divisor=0, numpy_warmups=0),
    "light": Depth("light", bdn_measurements=8, bdn_warmups=3,
                   numpy_budget_divisor=6, numpy_warmups=3),
    "measure": Depth("measure", bdn_measurements=50, bdn_warmups=5,
                     numpy_budget_divisor=1, numpy_warmups=10),
}


def parse_dtypes(value: str | None) -> tuple[str, ...]:
    """Parse comma-separated aliases into canonical NumSharp dtype names."""
    if value is None or not value.strip() or value.strip().lower() in {"all", "*"}:
        return ALL_DTYPES
    requested = []
    unknown = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        canonical = DTYPE_ALIASES.get(token)
        if canonical is None:
            unknown.append(raw.strip())
        elif canonical not in requested:
            requested.append(canonical)
    if unknown:
        raise ValueError(
            f"Unknown dtype(s): {', '.join(unknown)}. Choose from: {', '.join(ALL_DTYPES)}")
    if not requested:
        raise ValueError("--dtypes did not contain a dtype")
    return tuple(dtype for dtype in ALL_DTYPES if dtype in requested)


def selected(domain, requested) -> list[str]:
    requested_set = set(requested)
    return [dtype for dtype in domain if dtype in requested_set]


def matches_any(requested, *dtypes: str) -> bool:
    requested_set = set(requested)
    return any(DTYPE_ALIASES.get(str(dtype).lower(), str(dtype).lower()) in requested_set
               for dtype in dtypes)


def short_dtypes(requested) -> tuple[str, ...]:
    return tuple(SHORT_DTYPES[dtype] for dtype in requested)
