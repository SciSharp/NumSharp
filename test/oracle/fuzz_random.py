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
from layout_catalog import _FLOAT_POOL, _cbase, _fill, describe  # noqa: E402
import gen_oracle as G  # noqa: E402

# G10 (F18): widened from 7 dtypes to all 13 — float16 + the narrow integers were exactly the
# dtypes where the W1 widening found real bugs, yet the nightly soak never explored them.
DTYPES = ["bool", "int32", "int64", "uint8", "float32", "float64", "complex128",
          "float16", "int8", "int16", "uint16", "uint32", "uint64"]
NP_BIN = {**G.BINARY_OPS, **G.COMPARISON_OPS}     # add/sub/mul/divide + comparisons (bit-exact tier)
# G10: flat reductions (axis=None) drawn into the random space; argmax/argmin/std/var stay out
# (no flatten-argmax overload in the harness; std/var ride the documented precision excuse).
REDUCE_FLAT = {k: G.REDUCE_OPS[k] for k in ("sum", "prod", "min", "max", "mean")}


# ---------------------------------------------------------------------------
# Undefined float->integer conversions — the ONE value class this fuzzer must not emit.
#
# C leaves a float->integer conversion UNDEFINED when the value is NaN, +/-inf, or outside the
# destination's range, and NumPy performs exactly that C cast. The result is therefore the host
# toolchain's, not a NumPy contract: glibc/gcc and MSVC disagree, and so do the SIMD and scalar
# loops of ONE numpy build —
#
#     np.array([np.nan] * 8).astype(np.uint32)[0]   ->  2147483648   (gcc, vectorized loop)
#     np.float64(np.nan).astype(np.uint32)          ->  0            (same build, scalar loop)
#
# This fuzzer recomputes `expected` on whichever host it runs on, so emitting those cases makes
# the nightly soak report the host's undefined behaviour as a NumSharp bug — which it did, on
# every seed, ~950/200000 cases, unfixable by any implementation (gh: fuzz-soak 29722530598).
#
# The DETERMINISTIC corpora keep the undefined edges on purpose (see layout_catalog._FLOAT_POOL):
# they are committed as bytes and replayed, never recomputed, so they pin NumSharp's hand-written
# cast kernels against themselves. Only this recompute-on-the-host tier has to stay portable.
# ---------------------------------------------------------------------------

def _cast_defined(values, dst):
    """Elementwise mask: is converting `values` to integer dtype `dst` DEFINED in C?"""
    dst = np.dtype(dst)
    x = np.asarray(values.real if values.dtype.kind == "c" else values, dtype=np.float64)
    bits = dst.itemsize * 8
    lo = 0.0 if dst.kind == "u" else -(2.0 ** (bits - 1))
    hi = 2.0 ** (bits if dst.kind == "u" else bits - 1)   # exclusive; both exact in float64
    trunc = np.trunc(x)                                   # C converts by truncating toward zero
    return np.isfinite(x) & (trunc >= lo) & (trunc < hi)


def _defined_pool(real_dtype, dst):
    """`_FLOAT_POOL` narrowed to the values that survive `real_dtype` and convert to `dst`."""
    cand = np.array(_FLOAT_POOL, dtype=np.float64).astype(real_dtype)  # float16 saturates to inf
    keep = cand[_cast_defined(cand, dst)]
    return keep if keep.size else np.zeros(1, dtype=real_dtype)


def _defuse_cast(base, dst):
    """Rewrite, in place, the elements of `base` whose conversion to `dst` is undefined.

    `base` is the C-contiguous buffer the operand view aliases, so repairing it repairs every
    view of it. Replacements come from the pool's DEFINED edges, so the cast kernel still sees
    truncation and boundary values — only NaN/inf/out-of-range disappear.

    Complex operands are repaired through `.real`, the only part a complex->integer cast reads.
    Building a complex replacement instead would reintroduce the very values being removed:
    `1j * nan` is `nan + nan*j`, because the real part of the product is `0*nan - 1*0`.
    """
    ok = _cast_defined(base, dst)
    if ok.all():
        return
    target = base.real if base.dtype.kind == "c" else base      # writable view for complex
    reps = np.resize(_defined_pool(target.dtype, np.dtype(dst)), base.size).reshape(base.shape)
    target[...] = np.where(ok, target, reps)


def _defuse_integer_reciprocal(base):
    """NumPy's integer reciprocal is `*out = (T)(1.0 / in)`, so in == 0 converts +/-inf back to
    T — the same undefined conversion. Every other integer yields 1/x in [-1, 1], always defined.
    """
    if base.dtype.kind in "iu":
        base[...] = np.where(base == 0, 1, base)


def assert_portable(cases):
    """Raise if an undefined float->integer conversion reached the corpus.

    Runs on the SERIALIZED operand bytes, so it audits what the soak will actually replay rather
    than what the generator meant to emit — and it sits outside the per-case `except Exception`
    that would otherwise swallow the complaint.
    """
    for c in cases:
        op = c["op"]
        if op not in ("astype", "reciprocal"):
            continue
        o = c["operands"][0]
        src = np.dtype(o["dtype"])
        vals = np.frombuffer(bytes.fromhex(o["buffer"]), dtype=src)
        if op == "astype":
            dst = np.dtype(c["params"]["dtype"])
            if src.kind in "fc" and dst.kind in "iu" and not _cast_defined(vals, dst).all():
                raise AssertionError(
                    f"{c['id']}: undefined {src.name}->{dst.name} conversion reached the corpus; "
                    f"its `expected` is this host's undefined behaviour, not a NumPy contract")
        elif src.kind in "iu" and (vals == 0).any():
            raise AssertionError(
                f"{c['id']}: reciprocal({src.name} 0) reached the corpus; NumPy computes it as "
                f"({src.name})(1.0/0) — an undefined conversion whose result is host-specific")


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
        kind = rng.choice(["unary", "binary", "comparison", "where", "reduce", "astype"])
        try:
            if kind == "unary":
                dt = rng.choice(DTYPES)
                b, v = random_view(rng, np.dtype(dt))
                opn = rng.choice(list(G.UNARY_OPS))
                if opn == "reciprocal":
                    _defuse_integer_reciprocal(b)       # (T)(1.0/0) is an undefined conversion
                cases.append(_case(opn, [describe(b, v)], G.UNARY_OPS[opn](v), len(cases)))
            elif kind == "reduce":                       # G10: flat reductions over random views
                dt = rng.choice(DTYPES)
                b, v = random_view(rng, np.dtype(dt))
                opn = rng.choice(list(REDUCE_FLAT))
                case = _case(opn, [describe(b, v)], np.asarray(REDUCE_FLAT[opn](v, None, False)), len(cases))
                case["params"] = {"axis": None, "keepdims": False}
                cases.append(case)
            elif kind == "astype":                       # G10: random src->dst casts over random views
                src = rng.choice(DTYPES)
                dst = rng.choice(G.ALL_DTYPES)
                b, v = random_view(rng, np.dtype(src))
                if np.dtype(src).kind in "fc" and np.dtype(dst).kind in "iu":
                    _defuse_cast(b, np.dtype(dst))
                case = _case("astype", [describe(b, v)], v.astype(np.dtype(dst)), len(cases))
                case["params"] = {"dtype": np.dtype(dst).name}
                cases.append(case)
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
    assert_portable(cases)
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
