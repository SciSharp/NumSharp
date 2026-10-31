#!/usr/bin/env python3
"""NumPy 1.26.4 oracle for np.find_common_type -> test/NumSharp.Tests/Logic/corpus/find_common_type.numpy1264.jsonl.

np.find_common_type was deprecated in NumPy 1.25 and REMOVED in NumPy 2.0, so its last behaviour is NumPy 1.26.4's:
the oracle needs that NumPy, not the repository's 2.4.2. Run it from a scratch environment:

    python -m venv <dir> && <dir>/Scripts/python -m pip install numpy==1.26.4    (Linux: <dir>/bin/python)
    <dir>/Scripts/python test/oracle/gen_find_common_type_oracle.py

Each line is {"a": [array dtypes], "s": [scalar dtypes], "r": result} with every dtype written as kind + item size
('b1', 'i1', 'u1', 'i2', 'u2', 'i4', 'u4', 'i8', 'u8', 'f2', 'f4', 'f8', 'c16'): the 13 NumPy dtypes NumSharp stores,
spelled so the answer does not depend on the host's C long ('int32' is 'l' on Windows, 'i' on Linux). The result is
null where NumPy returns None. The cases are deterministic: every combination of 1-3 array types alone, 1-3 scalar
types alone, 1 x 1, 1 x 2 and 2 x 1 array x scalar types, plus every 29th combination of 4 array types and of 2 x 2
array x scalar types (the full matrix, 66,443 cases, was verified once; the committed file is its regression net).
The replay is test/NumSharp.Tests/Logic/np.find_common_type.NumPy1264.Test.cs.
"""
import itertools
import json
import os
import sys
import warnings

import numpy as np

if np.__version__ != "1.26.4":
    sys.exit(f"needs numpy==1.26.4 (the last NumPy with find_common_type), found {np.__version__}")

warnings.simplefilter("ignore", DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "test", "NumSharp.Tests", "Logic", "corpus", "find_common_type.numpy1264.jsonl")

CODES = ["b1", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "f2", "f4", "f8", "c16"]
SAMPLE_EVERY = 29


def code(d):
    """A dtype as kind + item size, the host-independent spelling the corpus uses."""
    return f"{d.kind}{d.itemsize}"


def case(arrays, scalars):
    r = np.find_common_type([np.dtype(c) for c in arrays], [np.dtype(c) for c in scalars])
    return {"a": list(arrays), "s": list(scalars), "r": None if r is None else code(r)}


def main():
    for c in CODES:
        assert code(np.dtype(c)) == c, c
    cases = []
    for k in (1, 2, 3):
        cases += [case(a, []) for a in itertools.product(CODES, repeat=k)]
        cases += [case([], s) for s in itertools.product(CODES, repeat=k)]
    for ka, ks in ((1, 1), (1, 2), (2, 1)):
        cases += [case(a, s) for a in itertools.product(CODES, repeat=ka) for s in itertools.product(CODES, repeat=ks)]
    cases += [case(a, []) for i, a in enumerate(itertools.product(CODES, repeat=4)) if i % SAMPLE_EVERY == 0]
    pairs = itertools.product(itertools.product(CODES, repeat=2), itertools.product(CODES, repeat=2))
    cases += [case(a, s) for i, (a, s) in enumerate(pairs) if i % SAMPLE_EVERY == 0]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        for c in cases:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {OUT}")


if __name__ == "__main__":
    main()
