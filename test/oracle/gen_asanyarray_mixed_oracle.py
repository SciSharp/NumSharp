#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for the dtype of an array built from mixed scalars
-> test/NumSharp.Tests/Creation/corpus/asanyarray_mixed.numpy242.jsonl.

np.asanyarray(object[]) in NumSharp builds an array from boxed C# values, each a scalar of its own dtype, so its NumPy
counterpart is np.array([np.int8(1), np.uint8(1), ...]): array coercion, which promotes the running dtype with each new
element in order (handle_promotion in array_coercion.c). That promotion is ORDER-SENSITIVE: {bool, float16, int8,
uint8} gives float32 in one order and float16 in another, which is why every ordering is a case.

    python test/oracle/gen_asanyarray_mixed_oracle.py      (numpy==2.4.2)

Each line is {"t": [element dtypes, in order], "r": result dtype}, dtypes written as kind + item size ('b1', 'i1', ...,
'f2', 'f4', 'f8', 'c16'), the 13 NumPy dtypes NumSharp stores. The cases are deterministic: every ordering of 2 and 3
distinct dtypes, every ordering of each 4-dtype set whose result depends on the order, and every 23rd ordering of the
other 4-dtype sets. The replay is test/NumSharp.Tests/Creation/np.asanyarray.MixedScalars.Test.cs.
"""
import itertools
import json
import os
import sys

import numpy as np

if np.__version__ != "2.4.2":
    sys.exit(f"needs numpy==2.4.2, found {np.__version__}")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "test", "NumSharp.Tests", "Creation", "corpus", "asanyarray_mixed.numpy242.jsonl")

CODES = ["b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
SAMPLE_EVERY = 23


def code(d):
    """A dtype as kind + item size, the host-independent spelling the corpus uses."""
    return f"{d.kind}{d.itemsize}"


def result(order):
    return code(np.array([np.dtype(c).type(1) for c in order]).dtype)


def main():
    cases = []
    for k in (2, 3):
        cases += [{"t": list(p), "r": result(p)} for p in itertools.permutations(CODES, k)]
    others = []
    for combo in itertools.combinations(CODES, 4):
        perms = list(itertools.permutations(combo))
        results = [result(p) for p in perms]
        if len(set(results)) > 1:
            cases += [{"t": list(p), "r": r} for p, r in zip(perms, results)]
        else:
            others += [{"t": list(p), "r": r} for p, r in zip(perms, results)]
    cases += [c for i, c in enumerate(others) if i % SAMPLE_EVERY == 0]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for c in cases:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
