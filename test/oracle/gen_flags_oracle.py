"""ndarray.flags / ndarray.setflags differential oracle — NumPy 2.4.2 is the source of truth.

Emits test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl: one JSON object per case.
Replayed (no Python) by FlagsOracleTests.cs, whose FlagsOracleRecipes builds the IDENTICAL
array per recipe token and applies the IDENTICAL setflags op tokens.

Case axes:
  * ~52 layout/producer RECIPES (owned/view/F/transposed/strided/negstride/offset/composed/
    newaxis/broadcast x4/broadcast_arrays/fancy/bmask/reshape/ravel/view(dtype)/diag/diagonal/
    imag/real/astype/copy C+F/eye/frombuffer ro+rw/memmap r,r+,c,F-order,empty) — each records
    the FULL flags record (11 bools + num) and the verbatim 6-line str(flags).
  * ~9 setflags TRANSITION scenarios per recipe (w0, w1, w0+w1, a0, a0+a1, u1, one-call
    write&align, one-call align+uic -> error, one-call align+write) + u0 on owners — each
    records the error (type+message, verbatim) and the POST state (rollback checked).
  * a 13-dtype x 6-layout sweep proving the flags record is dtype-independent.

Encoding notes:
  * num is masked with 0x7FFFFFFF: NumPy's broadcast_arrays results carry the internal
    NPY_ARRAY_WARN_ON_WRITE bit (0x80000000) inside flags.num; NumSharp does not model the
    deprecation-warning machinery (its broadcast_arrays results are plainly writeable, which is
    what NumPy's will become), so the warn bit is stripped on the NumPy side.
  * ops stop at the first error, which is recorded verbatim; the post-state after an error pins
    NumPy's flagback rollback semantics.

Regenerate:  python test/oracle/gen_flags_oracle.py   (requires numpy==2.4.2)
"""
import gc
import json
import os
import tempfile
import warnings

import numpy as np

assert np.__version__ == "2.4.2", f"oracle must be generated with numpy 2.4.2, got {np.__version__}"
warnings.simplefilter("ignore")  # broadcast_arrays writeable FutureWarning

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "NumSharp.Tests", "Backends", "corpus", "flags_oracle.jsonl")

_tmpdir = tempfile.mkdtemp(prefix="ns_flags_oracle_")
_mmap_files = {}


def _mmap_path(key, arr):
    p = _mmap_files.get(key)
    if p is None:
        p = os.path.join(_tmpdir, key + ".npy")
        np.save(p, arr)
        _mmap_files[key] = p
    return p


DTYPES = {
    "bool": np.bool_, "int8": np.int8, "uint8": np.uint8, "int16": np.int16,
    "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "int64": np.int64,
    "uint64": np.uint64, "float16": np.float16, "float32": np.float32,
    "float64": np.float64, "complex128": np.complex128,
}


def build(recipe, dtype="int64"):
    """Every branch here has a 1:1 twin in FlagsOracleRecipes.Build (C#). Keep them in lockstep."""
    dt = DTYPES[dtype]
    if recipe == "c1d":           return np.arange(6).astype(dt)
    if recipe == "c2d_view":      return np.arange(12).astype(dt).reshape(3, 4)
    if recipe == "c2d_owned":     return np.zeros((3, 4), dtype=dt)
    if recipe == "f2d":           return np.asfortranarray(np.arange(12).astype(dt).reshape(3, 4))
    if recipe == "f1d":           return np.asfortranarray(np.arange(6).astype(dt))
    if recipe == "c3d":           return np.arange(24).astype(dt).reshape(2, 3, 4)
    if recipe == "rank5":         return np.zeros((2, 1, 3, 1, 4), dtype=dt)
    if recipe == "singleton_mid": return np.zeros((3, 1, 4), dtype=dt)
    if recipe == "zerod":         return np.array(5).astype(dt)
    if recipe == "zerod_view":    return np.arange(6).astype(dt)[:1].reshape(())
    if recipe == "onelem":        return np.zeros(1, dtype=dt)
    if recipe == "empty2d":       return np.zeros((0, 3), dtype=dt)
    if recipe == "empty_sliced":  return np.zeros((0, 3), dtype=dt)[::2, :]
    if recipe == "t2d":           return np.arange(12).astype(dt).reshape(3, 4).T
    if recipe == "t3d":           return np.transpose(np.arange(24).astype(dt).reshape(2, 3, 4), (2, 0, 1))
    if recipe == "strided":       return np.arange(12).astype(dt).reshape(3, 4)[..., ::2]
    if recipe == "negstride":     return np.arange(6).astype(dt)[::-1]
    if recipe == "neg2d":         return np.arange(12).astype(dt).reshape(3, 4)[::-1]
    if recipe == "slice_offset":  return np.arange(10).astype(dt)[2:7]
    if recipe == "slice_step":    return np.arange(10).astype(dt)[1:9:2]
    if recipe == "slice_composed": return np.arange(24).astype(dt).reshape(4, 6)[1:3].T
    if recipe == "row":           return np.arange(12).astype(dt).reshape(3, 4)[1]
    if recipe == "col":           return np.arange(12).astype(dt).reshape(3, 4)[:, 1]
    if recipe == "newaxis":       return np.arange(6).astype(dt)[None, :]
    if recipe == "bcast_full":    return np.broadcast_to(np.arange(3).astype(dt), (4, 3))
    if recipe == "bcast_same":    return np.broadcast_to(np.arange(3).astype(dt), (3,))
    if recipe == "bcast_scalar":  return np.broadcast_to(np.array(5).astype(dt), (2, 3))
    if recipe == "bcast_partial": return np.broadcast_to(np.arange(6).astype(dt).reshape(1, 6), (4, 6))
    if recipe == "bcast_arrays0": return np.broadcast_arrays(np.arange(3).astype(dt), np.arange(3).astype(dt).reshape(3, 1))[0]
    if recipe == "fancy":         return np.arange(12).astype(dt).reshape(3, 4)[[0, 2]]
    if recipe == "fancy1d":       return np.arange(6).astype(dt)[[0, 2, 4]]
    if recipe == "bmask":
        a = np.arange(12).astype(dt).reshape(3, 4)
        return a[np.arange(12).reshape(3, 4) % 2 == 0]
    if recipe == "reshape_view":  return np.arange(12).astype(dt).reshape(3, 4).reshape(12)
    if recipe == "reshape_copy":  return np.arange(12).astype(dt).reshape(3, 4).T.reshape(12)
    if recipe == "ravel_c":       return np.arange(12).astype(dt).reshape(3, 4).ravel()
    if recipe == "ravel_t":       return np.arange(12).astype(dt).reshape(3, 4).T.ravel()
    if recipe == "view_same":     return np.arange(6).view(np.float64)          # int64 -> float64, same itemsize
    if recipe == "view_diff":     return np.arange(6).view(np.int32)            # int64 -> int32, last axis rescales
    if recipe == "diag2d":        return np.diag(np.arange(12).astype(dt).reshape(3, 4))
    if recipe == "diagonal_m":    return np.arange(12).astype(dt).reshape(3, 4).diagonal()
    if recipe == "imag_real":     return np.imag(np.arange(6.0))
    if recipe == "real_complex":  return np.real(np.arange(4).astype(np.complex128))
    if recipe == "astype":        return np.arange(6).astype(np.int32)
    if recipe == "copy_c":        return np.arange(12).astype(dt).reshape(3, 4).copy()
    if recipe == "copy_f":        return np.arange(12).astype(dt).reshape(3, 4).copy("F")
    if recipe == "eye3":          return np.eye(3)
    if recipe == "frombuffer_ro": return np.frombuffer(bytes(range(4)), dtype=np.uint8)
    if recipe == "frombuffer_rw": return np.frombuffer(bytearray(range(4)), dtype=np.uint8)
    if recipe == "mmap_r":        return np.load(_mmap_path("m5", np.arange(5.0)), mmap_mode="r")
    if recipe == "mmap_rp":       return np.load(_mmap_path("m5", np.arange(5.0)), mmap_mode="r+")
    if recipe == "mmap_c":        return np.load(_mmap_path("m5", np.arange(5.0)), mmap_mode="c")
    if recipe == "mmap_r_f":      return np.load(_mmap_path("mF", np.asfortranarray(np.arange(6.0).reshape(2, 3))), mmap_mode="r")
    if recipe == "mmap_empty_r":  return np.load(_mmap_path("mE", np.zeros((0, 3))), mmap_mode="r")
    raise KeyError(recipe)


RECIPES = [
    "c1d", "c2d_view", "c2d_owned", "f2d", "f1d", "c3d", "rank5", "singleton_mid",
    "zerod", "zerod_view", "onelem", "empty2d", "empty_sliced",
    "t2d", "t3d", "strided", "negstride", "neg2d",
    "slice_offset", "slice_step", "slice_composed", "row", "col", "newaxis",
    "bcast_full", "bcast_same", "bcast_scalar", "bcast_partial", "bcast_arrays0",
    "fancy", "fancy1d", "bmask", "reshape_view", "reshape_copy", "ravel_c", "ravel_t",
    "view_same", "view_diff", "diag2d", "diagonal_m", "imag_real", "real_complex",
    "astype", "copy_c", "copy_f", "eye3",
    "frombuffer_ro", "frombuffer_rw",
    "mmap_r", "mmap_rp", "mmap_c", "mmap_r_f", "mmap_empty_r",
]

# The dtype-independence sweep: same flags record for every dtype.
SWEEP_RECIPES = ["c2d_view", "f2d", "t2d", "strided", "bcast_full", "negstride"]

# setflags transition scenarios applied to a FRESH instance each. Tokens are shared with C#:
#   w0/w1  = setflags(write=False/True)      a0/a1 = setflags(align=False/True)
#   u0/u1  = setflags(uic=False/True)        w0a0  = setflags(write=False, align=False)
#   a0u1   = setflags(align=False, uic=True)  a0w1 = setflags(align=False, write=True)
SCENARIOS = [
    ["w0"], ["w1"], ["w0", "w1"], ["a0"], ["a0", "a1"], ["u1"], ["w0a0"], ["a0u1"], ["a0w1"],
]


def flags_record(a):
    f = a.flags
    b = lambda v: 1 if v else 0
    return {
        "C": b(f.c_contiguous), "F": b(f.f_contiguous), "O": b(f.owndata), "W": b(f.writeable),
        "A": b(f.aligned), "X": b(f.writebackifcopy),
        "fnc": b(f.fnc), "forc": b(f.forc), "behaved": b(f.behaved),
        "carray": b(f.carray), "farray": b(f.farray),
        "num": int(f.num) & 0x7FFFFFFF,  # strip NPY_ARRAY_WARN_ON_WRITE (see module docstring)
    }


def apply_ops(a, ops):
    for op in ops:
        try:
            if op == "w0":   a.setflags(write=False)
            elif op == "w1": a.setflags(write=True)
            elif op == "a0": a.setflags(align=False)
            elif op == "a1": a.setflags(align=True)
            elif op == "u0": a.setflags(uic=False)
            elif op == "u1": a.setflags(uic=True)
            elif op == "w0a0": a.setflags(write=False, align=False)
            elif op == "a0u1": a.setflags(align=False, uic=True)
            elif op == "a0w1": a.setflags(align=False, write=True)
            else: raise KeyError(op)
        except (ValueError,) as e:
            return {"t": type(e).__name__, "m": str(e)}
    return None


def main():
    cases = []

    # 1) base records + verbatim str(flags), default dtype.
    # bcast_arrays0 skips str: NumPy renders its WARN_ON_WRITE deprecation state as
    # "WRITEABLE : True  (with WARN_ON_WRITE=True)" — machinery NumSharp deliberately does not
    # model (its broadcast_arrays results are plainly writeable, which is what NumPy's become).
    for r in RECIPES:
        a = build(r)
        case = {"id": f"{r}/base", "recipe": r, "dtype": "int64", "ops": [],
                "err": None, "f": flags_record(a)}
        if r != "bcast_arrays0":
            case["str"] = str(a.flags)
        cases.append(case)
        del a
        gc.collect()

    # 2) setflags transitions (fresh instance per scenario) + u0 on owners.
    for r in RECIPES:
        base_owns = build(r).flags.owndata
        gc.collect()
        scenarios = [list(s) for s in SCENARIOS]
        if base_owns:
            scenarios.append(["u0"])  # owners: NumPy's base-severing is a no-op, full parity
        for ops in scenarios:
            a = build(r)
            err = apply_ops(a, ops)
            cases.append({"id": f"{r}/{'.'.join(ops)}", "recipe": r, "dtype": "int64",
                          "ops": ops, "err": err, "f": flags_record(a)})
            del a
            gc.collect()

    # 3) dtype-independence sweep (base record only).
    for r in SWEEP_RECIPES:
        for dt in DTYPES:
            a = build(r, dt)
            cases.append({"id": f"{r}/dtype.{dt}", "recipe": r, "dtype": dt, "ops": [],
                          "err": None, "f": flags_record(a)})
            del a
            gc.collect()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="\n") as fh:
        for c in cases:
            fh.write(json.dumps(c, separators=(",", ":")) + "\n")

    n_err = sum(1 for c in cases if c["err"])
    print(f"wrote {len(cases)} cases ({len(RECIPES)} recipes, {n_err} error cases) -> {OUT}")

    gc.collect()
    import shutil
    shutil.rmtree(_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
