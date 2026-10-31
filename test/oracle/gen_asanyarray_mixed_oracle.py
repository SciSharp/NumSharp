#!/usr/bin/env python3
"""NumPy 2.4.2 oracle for the dtype of an array built from mixed scalars
-> test/NumSharp.Tests/Creation/corpus/asanyarray_mixed.numpy242.jsonl.

np.asanyarray(object[]) in NumSharp is NumPy's array coercion of the Python list the C# list stands for, each boxed C#
value read through the boundary map (UnmanagedStorage.Coercion.Sequence.cs): a bool is a Python bool, EVERY C# integer
primitive a Python int, float and double Python floats, Complex a Python complex, and Half a NumPy float16 scalar. So
its NumPy counterpart is np.array([1, 1.0, np.float16(1), ...]), not np.array([np.int8(1), np.uint8(1), ...]): array
coercion discovers a Python int as int64 and a Python float as float64, and promotes the running dtype with each new
element in order (handle_promotion in array_coercion.c), which is order-sensitive only where a float16 meets a bool.

    python test/oracle/gen_asanyarray_mixed_oracle.py      (numpy==2.4.2)

Each line is {"t": [element dtypes, in order], "r": result dtype}, dtypes written as kind + item size ('b1', 'i1', ...,
'f2', 'f4', 'f8', 'c16'): the C# type of each element, named by the NumPy dtype NumSharp stores it in (so 'i1' is a C#
sbyte, read as a Python int). The cases are deterministic: every ordering of 2 and 3 distinct types, every ordering of
each 4-type set whose result depends on the order, and every 23rd ordering of the other 4-type sets. The replay is
test/NumSharp.Tests/Creation/np.asanyarray.MixedScalars.Test.cs.
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


def python_value(c):
    """The Python value a boxed C# 1 of the type stored as dtype `c` stands for (the boundary map)."""
    if c == "b1":
        return True
    if c[0] in "iu":
        return 1           # every C# integer primitive is a Python int
    if c == "f2":
        return np.float16(1)   # Half is a NumPy scalar
    if c in ("f4", "f8"):
        return 1.0         # float and double are Python floats
    if c == "c16":
        return 1 + 0j      # Complex is a Python complex
    raise ValueError(c)


def result(order):
    return code(np.array([python_value(c) for c in order]).dtype)


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
