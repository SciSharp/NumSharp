"""
fuzz_random.py — seeded random differential fuzzer (offline).

Draws random operands (random ndim<=4, dims, stride permutations/signs/offsets), random
dtypes, and random ops across all tiers, computes the NumPy 2.4.2 result, and writes a
committed corpus the C# harness replays bit-exact. A fixed seed => identical corpus, so a
mismatch found in a nightly soak is reproducible and can be shrunk into corpus/regressions/.

Usage:
    python fuzz_random.py <seed> <count> [outfile]
    python fuzz_random.py 1234 2000 random_smoke.jsonl

Excludes floor_divide/mod/power (tracked separately as [OpenBugs]) so the random gate stays a
pure "should be bit-exact or documented-Misaligned" check.
"""
import json
import os
import random
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_catalog import _cbase, _fill, describe  # noqa: E402
import gen_oracle as G  # noqa: E402

DTYPES = ["bool", "int32", "int64", "uint8", "float32", "float64", "complex128"]
NP_BIN = {**G.BINARY_OPS, **G.COMPARISON_OPS}     # add/sub/mul/divide + comparisons (bit-exact tier)


def random_view(rng, dtype, max_ndim=4):
    """A random transposed/strided/reversed VIEW into a fresh C-contig base.

    Scoped to NumSharp-PRODUCIBLE layouts: NumSharp's slicing normalizes offset into the
    storage base (Shape.offset stays 0) and keeps consistent strides for size-1 dims, so we
    avoid offset!=0 slices and size-1 dims (which would carry numpy's "junk" size-1 strides).
    Reconstructing those raw numpy representations exposes ops that assume the normalized form
    (tracked separately, task #11) and is not what this fuzzer is meant to test.
    """
    ndim = rng.randint(0, max_ndim)
    if ndim == 0:
        b = _fill(1, dtype).reshape(())
        return b, b
    dims = tuple(rng.randint(3, 5) for _ in range(ndim))  # >=3 so step-slicing stays >=2 (no size-1)
    b = _cbase(dims, dtype)
    v = b
    for _ in range(rng.randint(0, 2)):
        t = rng.choice(["transpose", "step", "reverse"])
        if t == "transpose" and v.ndim >= 2:
            perm = list(range(v.ndim)); rng.shuffle(perm); v = v.transpose(perm)
        elif t == "step":
            ax = rng.randrange(v.ndim)
            sl = [slice(None)] * v.ndim; sl[ax] = slice(None, None, 2); v = v[tuple(sl)]
        elif t == "reverse":
            ax = rng.randrange(v.ndim)
            sl = [slice(None)] * v.ndim; sl[ax] = slice(None, None, -1); v = v[tuple(sl)]
    return b, v


def view_of_shape(rng, dtype, shape):
    """A shape-preserving view (contig / step-strided / reversed) of the given logical shape."""
    if len(shape) == 0:
        b = _fill(1, dtype).reshape(())
        return b, b
    style = rng.choice(["contig", "step2", "reverse"])
    if style == "step2":
        b = _cbase(tuple(d * 2 for d in shape), dtype)
        return b, b[tuple(slice(None, None, 2) for _ in shape)]
    if style == "reverse":
        b = _cbase(tuple(shape), dtype)
        return b, b[tuple(slice(None, None, -1) for _ in shape)]
    b = _cbase(tuple(shape), dtype)
    return b, b


def _case(opname, operands, r, idx):
    r = np.asarray(r)
    return {
        "id": f"{opname}/random/{idx}",
        "op": opname,
        "params": {},
        "operands": operands,
        "expected": {"dtype": r.dtype.name, "shape": [int(d) for d in r.shape],
                     "buffer": np.ascontiguousarray(r).tobytes().hex()},
        "layout": "random",
        "valueclass": "random",
    }


def gen_random(seed, count):
    rng = random.Random(seed)
    cases = []
    attempts = 0
    while len(cases) < count and attempts < count * 20:
        attempts += 1
        kind = rng.choice(["unary", "binary", "comparison", "where"])
        try:
            if kind == "unary":
                dt = rng.choice(DTYPES)
                b, v = random_view(rng, np.dtype(dt))
                opn = rng.choice(list(G.UNARY_OPS))
                cases.append(_case(opn, [describe(b, v)], G.UNARY_OPS[opn](v), len(cases)))
            elif kind in ("binary", "comparison"):
                pool = list(G.BINARY_OPS) if kind == "binary" else list(G.COMPARISON_OPS)
                ba, va = random_view(rng, np.dtype(rng.choice(DTYPES)))
                if rng.random() < 0.3:
                    bb = _fill(1, np.dtype(rng.choice(DTYPES))).reshape(()); vb = bb
                else:
                    bb, vb = view_of_shape(rng, np.dtype(rng.choice(DTYPES)), va.shape)
                opn = rng.choice(pool)
                cases.append(_case(opn, [describe(ba, va), describe(bb, vb)], NP_BIN[opn](va, vb), len(cases)))
            else:  # where
                bx, vx = random_view(rng, np.dtype(rng.choice(DTYPES)))
                bc, vc = view_of_shape(rng, np.bool_, vx.shape)
                if rng.random() < 0.3:
                    by = _fill(1, np.dtype(rng.choice(DTYPES))).reshape(()); vy = by
                else:
                    by, vy = view_of_shape(rng, np.dtype(rng.choice(DTYPES)), vx.shape)
                cases.append(_case("where", [describe(bc, vc), describe(bx, vx), describe(by, vy)],
                                   np.where(vc, vx, vy), len(cases)))
        except Exception:
            continue  # incompatible shapes / NumPy raise: drop and retry
    return cases


def main():
    if len(sys.argv) < 3:
        print("usage: python fuzz_random.py <seed> <count> [outfile]")
        sys.exit(2)
    seed = int(sys.argv[1])
    count = int(sys.argv[2])
    outfile = sys.argv[3] if len(sys.argv) > 3 else f"random_seed{seed}.jsonl"
    here = os.path.dirname(os.path.abspath(__file__))
    corpus_dir = os.path.normpath(os.path.join(here, "..", "NumSharp.UnitTest", "Fuzz", "corpus"))
    cases = gen_random(seed, count)
    G.write_jsonl(os.path.join(corpus_dir, outfile), cases)


if __name__ == "__main__":
    main()
