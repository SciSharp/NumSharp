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
from layout_catalog import LAYOUTS, describe  # noqa: E402

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
    else:
        print(f"unknown mode '{mode}' (expected: smoke | astype_full)")
        sys.exit(2)


if __name__ == "__main__":
    main()
