"""
gen_oracle.py — emit a committed, bytes-exact NumPy 2.4.2 oracle corpus.

The corpus is JSONL (one case per line). C# replays the operand bytes EXACTLY and
compares its op result to `expected` bit-for-bit (NaN/inf tokenized). No Python at test time.

Case schema:
  {
    "id":      "<op>/<layout>/<src>-><dst>/<n>",
    "op":      "astype",                       # OpRegistry key
    "params":  {"dtype": "int32"},             # op-specific params
    "operands":[ <operand-descriptor>, ... ],  # see layout_catalog.describe()
    "expected":{"dtype":"int32","shape":[...],"buffer":"<hex C-contiguous result>"},
    "layout":  "strided_step2_1d",
    "valueclass":"mixed"
  }

operand-descriptor = {dtype, shape, strides(elements), offset(elements), bufferSize(elements), buffer(hex of base)}
"""
import json
import os
import sys
import warnings

import numpy as np

# Overflow / invalid-value-in-cast warnings ARE the edge cases we want to capture, not errors.
np.seterr(all="ignore")
warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_catalog import LAYOUTS, PAIR_LAYOUTS, describe  # noqa: E402

# 13 NumPy-representable dtypes (Char + Decimal have no NumPy analog -> covered by
# NumSharp's Converts-oracle tests, not by this differential corpus).
ALL_DTYPES = [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64", "complex128",
]


def _expected(view, dst):
    exp = np.ascontiguousarray(view.astype(dst))
    return {"dtype": np.dtype(dst).name, "shape": [int(d) for d in view.shape], "buffer": exp.tobytes().hex()}


def gen_astype(srcs, dsts, layout_names):
    cases = []
    n = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in srcs:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for d in dsts:
                cases.append({
                    "id": f"astype/{ln}/{s}->{d}/{n}",
                    "op": "astype",
                    "params": {"dtype": np.dtype(d).name},
                    "operands": [operand],
                    "expected": _expected(view, d),
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    return cases


# Binary ops: NumPy computes the result (value AND NEP50 result dtype) — it is the oracle.
# Bit-exact today (committed green matrix).
BINARY_OPS = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,          # true_divide
}

# Known-divergent today (cataloged as [OpenBugs]): integer ÷0/mod0 throws-or-garbage vs NumPy 0,
# float //0 -> NaN vs NumPy ±inf, mixed-precision mod, complex power ~ULP/edge.
DIVMOD_POWER_OPS = {
    "floor_divide": lambda a, b: a // b,
    "mod": lambda a, b: a % b,             # NumPy: floored remainder (sign of divisor)
    "power": lambda a, b: a ** b,
}

# Comparison ops -> bool result. (NumPy raises TypeError for ordering complex; gen_binary skips those.)
COMPARISON_OPS = {
    "equal": lambda a, b: a == b,
    "not_equal": lambda a, b: a != b,
    "less": lambda a, b: a < b,
    "greater": lambda a, b: a > b,
    "less_equal": lambda a, b: a <= b,
    "greater_equal": lambda a, b: a >= b,
}

# Curated dtype pairs covering NEP50 promotion: same-type, int-width mixing, signed/unsigned,
# int->float, float widths, bool promotion, complex absorption.
DT_PAIRS = [
    ("int32", "int32"), ("int32", "int64"), ("int64", "int32"),
    ("int32", "float64"), ("float64", "int32"), ("int32", "float32"),
    ("float32", "float64"), ("float32", "float32"), ("float64", "float64"),
    ("uint8", "int8"), ("int8", "uint8"), ("uint8", "uint8"),
    ("int16", "int32"), ("uint32", "int32"), ("int32", "uint32"),
    ("bool", "int32"), ("bool", "float64"),
    ("complex128", "float64"), ("float64", "complex128"), ("complex128", "int32"),
]


# Unary ops. NumPy is the oracle for result dtype (e.g. sqrt(int)->float64, abs(complex)->float64).
UNARY_OPS = {
    "negative": np.negative, "abs": np.abs, "sign": np.sign,
    "sqrt": np.sqrt, "cbrt": np.cbrt, "square": np.square, "reciprocal": np.reciprocal,
    "floor": np.floor, "ceil": np.ceil, "trunc": np.trunc,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
}
UNARY_DTYPES = ["bool", "int32", "int64", "uint8", "float32", "float64", "complex128"]


def gen_unary(ops, dtypes, layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in layout_names:
        fn = LAYOUTS[ln]
        for s in dtypes:
            base, view = fn(np.dtype(s))
            operand = describe(base, view)
            for opname, f in ops.items():
                try:
                    r = f(view)
                except Exception:
                    skipped += 1  # NumPy raises (e.g. floor(complex)); error-parity tested separately
                    continue
                # Read the shape BEFORE ascontiguousarray (which forces ndim>=1, corrupting 0-D results).
                exp_shape = [int(d) for d in r.shape]
                exp_buf = np.ascontiguousarray(r).tobytes().hex()
                cases.append({
                    "id": f"{opname}/{ln}/{s}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [operand],
                    "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def gen_binary(ops, dt_pairs, pair_layout_names):
    cases = []
    n = 0
    skipped = 0
    for ln in pair_layout_names:
        fn = PAIR_LAYOUTS[ln]
        for (sa, sb) in dt_pairs:
            ba, va, bb, vb = fn(np.dtype(sa), np.dtype(sb))
            op_a = describe(ba, va)
            op_b = describe(bb, vb)
            for opname, f in ops.items():
                try:
                    r = f(va, vb)
                except Exception:
                    skipped += 1  # NumPy raises (e.g. int**neg); error-parity is tested separately
                    continue
                # Read the shape BEFORE ascontiguousarray (which forces ndim>=1, corrupting 0-D results).
                exp_shape = [int(d) for d in r.shape]
                exp_buf = np.ascontiguousarray(r).tobytes().hex()
                cases.append({
                    "id": f"{opname}/{ln}/{sa},{sb}/{n}",
                    "op": opname,
                    "params": {},
                    "operands": [op_a, op_b],
                    "expected": {"dtype": r.dtype.name, "shape": exp_shape, "buffer": exp_buf},
                    "layout": ln,
                    "valueclass": "mixed",
                })
                n += 1
    if skipped:
        print(f"  (skipped {skipped} cases where NumPy raised)")
    return cases


def write_jsonl(path, cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "test", "NumSharp.UnitTest", "Fuzz", "corpus"))
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        srcs = ["float64", "int32", "float32"]
        dsts = ["int32", "float64", "uint8", "int16"]
        layouts = list(LAYOUTS.keys())
        cases = gen_astype(srcs, dsts, layouts)
        write_jsonl(os.path.join(corpus_dir, "astype_smoke.jsonl"), cases)
    elif mode == "astype_full":
        cases = gen_astype(ALL_DTYPES, ALL_DTYPES, list(LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "astype_full.jsonl"), cases)
    elif mode == "binary":
        cases = gen_binary(BINARY_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "binary_arith.jsonl"), cases)
    elif mode == "divmod_power":
        cases = gen_binary(DIVMOD_POWER_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "binary_divmod_power.jsonl"), cases)
    elif mode == "comparison":
        cases = gen_binary(COMPARISON_OPS, DT_PAIRS, list(PAIR_LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "comparison.jsonl"), cases)
    elif mode == "unary":
        cases = gen_unary(UNARY_OPS, UNARY_DTYPES, list(LAYOUTS.keys()))
        write_jsonl(os.path.join(corpus_dir, "unary.jsonl"), cases)
    else:
        print(f"unknown mode '{mode}' (expected: smoke | astype_full | binary | divmod_power | comparison | unary)")
        sys.exit(2)


if __name__ == "__main__":
    main()
