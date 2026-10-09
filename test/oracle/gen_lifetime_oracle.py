#!/usr/bin/env python3
"""
gen_lifetime_oracle.py -- the LIFETIME oracle: NumPy 2.4.2's answers for calls whose operands are TEMPORARIES.

Every other differential tier replays a case over operand arrays the harness itself holds, so an operand stays
reachable for the whole call no matter what the library does. A caller that passes a temporary
(``np.take(a, np.arange(n)[::-1])``) does not hold it: inside the library the parameter is the only reference,
and in optimized (tier-1) code the JIT stops reporting a parameter or local to the GC after its last use. When a
library method reads an operand's raw buffer pointer, then runs a kernel over that pointer without touching the
operand again, a collection during the kernel finds the operand unreachable; the finalizer of its memory block
(``UnmanagedMemoryBlock.Disposer``) then returns the buffer to the pool -- or to the OS -- while the kernel still
reads or writes it. The result is a silently wrong value, a spurious IndexError, or a fatal access violation.

This generator writes the cases that gate that class. The C# replay (``Fuzz/Lifetime/``) runs them in a child
process with tiered compilation OFF (tier-0 code reports every local live for the whole method, which hides the
bug), builds every operand marked ``temp`` afresh INSIDE the call expression, and repeats each call while a
stress thread forces full collections, drains finalizers and poisons every parked pool buffer. A repetition whose
bytes differ from the reference run (all operands held) is a lost GC anchor. The reference run itself is compared
with NumPy's answer recorded here.

Operands are RECIPES, not bytes: a case operand is ``{dtype, shape, recipe, view, temp}`` and both sides build
the same values from the recipe (operands are 2 MB each; storing their bytes would cost ~1 GB of corpus).
Recipe kinds (``i`` = the linear C-order index of the base buffer, ``q`` an exact integer):

  mod   q = (start + i*step) % mod;  value = q*scale + off   (float64 arithmetic, exact by construction)
        bool: q == 0;  integer dtypes: int(value);  float dtypes: value;  U strings: str(int(value))
  perm  q = (start + i*step) % n                              (index arrays; a permutation when gcd(step, n) == 1)
  cmod  complex: real from (start, step, mod, scale, off), imaginary from (start2, step2, mod2, scale2, off2)
  eye   2-D square: mod's value plus ``diag`` on the main diagonal

Views (applied to the base after it is built): ``""`` none, ``step2`` = [::2], ``rev`` = [::-1], ``T`` =
transpose, ``colstep`` = [:, ::2], ``off1`` = [1:].

Results are recorded as {kind, dtype (``dtype.str``), shape, sha256 of the C-order bytes}; tuples as a list of
items; byte strings (``tobytes``, ``np.save``) as {kind: bytes, sha256, len}. ``parity: false`` marks a case whose
VALUE NumSharp does not reproduce bit for bit (managed LU / Gram matrices, an introselect arrangement): the
replay checks only the lifetime contract there. ``host: true`` marks values that depend on the platform libm or
FFT twiddles; the replay checks them against NumPy only on win-amd64.

Carving (house rule: real product bugs are carved and pinned, never excused): ``_gap_of`` names the documented
lost-anchor gap a case hits (``GAPS``); a carved case goes to ``lifetime_gaps.jsonl`` with its gap id, and
the [OpenBugs] replays in ``LifetimeOracleTests`` fail while a gap is open. Both files live in
``Fuzz/Lifetime/corpus/`` (see ``CORPUS``). ``--uncarved DIR`` writes every case
to one gate file in DIR (how the carve rules were drawn, and how to re-probe after a fix).

Run (needs numpy==2.4.2):
    python test/oracle/gen_lifetime_oracle.py [--uncarved DIR]
"""
import argparse
import hashlib
import io
import json
import os
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
#: The lifetime tier lives in its OWN directory, not Fuzz/corpus: the generic corpus sweeps there (the leak gate, the
#: corpus survey, the coverage join) replay every top-level file through the ordinary case schema, and these cases
#: carry operand RECIPES, not bytes.
CORPUS = os.path.normpath(os.path.join(HERE, "..", "NumSharp.Tests.Oracle", "Fuzz", "Lifetime", "corpus"))
GAP_DIR = CORPUS

#: Elements of a 1-D operand: 2 MB of float64, so a kernel over it runs long enough (~0.2-2 ms) for a forced
#: collection to land inside it; 2-D operands are SIDE x SIDE of the same size. Calls whose window is short take larger
#: operands so a lost anchor is caught reliably: the bool-mask scans 8x elements (2 MB, a byte each), a cast, a mask
#: scatter, a float16 reduction and a factorization's input copy 4x, and the two memcpy-fast sites (np.all's SIMD scan,
#: ToMuliDimArray's copy) 16 MB. ``--bits`` changes both for a
#: classification run (a larger operand widens every window; the committed corpus uses the default).
N = 1 << 18
SIDE = 512


# =============================================================================================================
# Operand recipes
# =============================================================================================================

def mod(dtype, shape, mod_, start=0, step=1, scale=1.0, off=0.0, view="", temp=True):
    """An operand whose values follow the ``mod`` recipe.

    Parameters:
        dtype: the NumPy dtype name (``U<n>`` for strings).
        shape: the BASE shape (the view, if any, is taken from it).
        mod_: the modulus; values are ``((start + i*step) % mod_) * scale + off``.
        start, step: the linear progression over the base's C-order index ``i``.
        scale, off: exact float64 scale and offset (a power of two and a small number keep values exact).
        view: the view applied to the base (see the module docstring).
        temp: True to build the operand afresh inside every stressed call (the operand under test).

    Returns:
        The operand spec dict the corpus stores.
    """
    return {"dtype": dtype, "shape": list(shape), "view": view, "temp": temp,
            "recipe": {"k": "mod", "start": start, "step": step, "mod": mod_, "scale": scale, "off": off}}


def perm(dtype, shape, n, start=0, step=1, view="", temp=True):
    """An index operand ``(start + i*step) % n`` (a permutation of ``range(n)`` when gcd(step, n) == 1).

    Parameters:
        dtype: an integer dtype name.
        shape: the base shape.
        n: the modulus (the size of the axis the indices address).
        start, step: the progression.
        view: the view applied to the base.
        temp: True to build the operand afresh inside every stressed call.

    Returns:
        The operand spec dict.
    """
    return {"dtype": dtype, "shape": list(shape), "view": view, "temp": temp,
            "recipe": {"k": "perm", "start": start, "step": step, "n": n}}


def cmod(shape, mod_, mod2, start=0, step=1, start2=1, step2=3, scale=1.0, scale2=1.0, view="", temp=True):
    """A complex128 operand: real part from one ``mod`` progression, imaginary part from a second.

    Parameters:
        shape: the base shape.
        mod_, start, step, scale: the real part's recipe (offset 0).
        mod2, start2, step2, scale2: the imaginary part's recipe.
        view: the view applied to the base.
        temp: True to build the operand afresh inside every stressed call.

    Returns:
        The operand spec dict.
    """
    return {"dtype": "complex128", "shape": list(shape), "view": view, "temp": temp,
            "recipe": {"k": "cmod", "start": start, "step": step, "mod": mod_, "scale": scale,
                       "start2": start2, "step2": step2, "mod2": mod2, "scale2": scale2}}


def eye(n, diag, mod_, step=1, scale=1.0, temp=True):
    """A square float64 operand: a ``mod`` progression plus ``diag`` on the main diagonal (diagonally dominant
    when ``diag`` exceeds the row sums, so solve / inv / det are well conditioned).

    Parameters:
        n: the side.
        diag: the value added on the diagonal.
        mod_, step, scale: the off-diagonal progression.
        temp: True to build the operand afresh inside every stressed call.

    Returns:
        The operand spec dict.
    """
    return {"dtype": "float64", "shape": [n, n], "view": "", "temp": temp,
            "recipe": {"k": "eye", "start": 0, "step": step, "mod": mod_, "scale": scale, "off": 0.0,
                       "diag": diag}}


def build(spec):
    """Builds the NumPy array an operand spec describes (the C# replay builds the same bytes).

    Parameters:
        spec: an operand spec dict (``mod`` / ``perm`` / ``cmod`` / ``eye``).

    Returns:
        The operand array, the view applied.
    """
    shape = tuple(spec["shape"])
    size = int(np.prod(shape)) if shape else 1
    r = spec["recipe"]
    dt = spec["dtype"]
    i = np.arange(size, dtype=np.int64)
    k = r["k"]
    if k == "perm":
        arr = ((r["start"] + i * r["step"]) % r["n"]).astype(dt)
    elif k == "cmod":
        re = ((r["start"] + i * r["step"]) % r["mod"]).astype(np.float64) * r["scale"]
        im = ((r["start2"] + i * r["step2"]) % r["mod2"]).astype(np.float64) * r["scale2"]
        arr = (re + 1j * im).astype(np.complex128)
    else:
        q = (r["start"] + i * r["step"]) % r["mod"]
        v = q.astype(np.float64) * r["scale"] + r["off"]
        if k == "eye":
            n = shape[0]
            v = v.reshape(n, n)
            v[np.arange(n), np.arange(n)] += r["diag"]
            v = v.reshape(-1)
        if dt == "bool":
            arr = q == 0
        elif dt.startswith("U"):
            arr = np.array([str(int(x)) for x in v], dtype="<" + dt)
        elif np.dtype(dt).kind in "iu":
            arr = v.astype(np.int64).astype(dt)
        else:
            arr = v.astype(dt)
    arr = arr.reshape(shape)
    view = spec["view"]
    if view == "":
        return arr
    if view == "step2":
        return arr[::2]
    if view == "rev":
        return arr[::-1]
    if view == "T":
        return arr.T
    if view == "colstep":
        return arr[:, ::2]
    if view == "off1":
        return arr[1:]
    raise ValueError(f"unknown view {view!r}")


# =============================================================================================================
# Results
# =============================================================================================================

def _sha(b):
    """The lower-case hex SHA-256 of a byte string (what LifetimeItem.Sha256 computes on the C# side)."""
    return hashlib.sha256(b).hexdigest()


def _item(x):
    """Records one result item: an array (or a NumPy / Python scalar, via ``np.asarray``) or a byte string."""
    if isinstance(x, (bytes, bytearray)):
        return {"kind": "bytes", "len": len(x), "sha256": _sha(bytes(x))}
    a = np.asarray(x)
    return {"kind": "array", "dtype": a.dtype.str, "shape": list(a.shape),
            "sha256": _sha(np.ascontiguousarray(a).tobytes())}


def record(res):
    """Records a call's result: a tuple / list result item by item, anything else as one item."""
    if isinstance(res, (tuple, list)):
        return {"kind": "tuple", "items": [_item(x) for x in res]}
    return _item(res)


# =============================================================================================================
# The calls (op key -> NumPy; the C# registry Fuzz/Lifetime/LifetimeOps.cs maps the same keys)
# =============================================================================================================

def _save_bytes(a):
    """The bytes ``np.save`` writes for ``a`` (header included): the C# side compares them with NumSharp's
    ``np.save`` to a byte array, which reads the operand's buffer while the operand may be a temporary."""
    buf = io.BytesIO()
    np.save(buf, a)
    return buf.getvalue()


OPS = {
    # elementwise unary
    "sqrt": lambda a, p: np.sqrt(a[0]),
    "abs": lambda a, p: np.abs(a[0]),
    "negative": lambda a, p: np.negative(a[0]),
    "floor": lambda a, p: np.floor(a[0]),
    "ceil": lambda a, p: np.ceil(a[0]),
    "trunc": lambda a, p: np.trunc(a[0]),
    "rint": lambda a, p: np.rint(a[0]),
    "square": lambda a, p: np.square(a[0]),
    "reciprocal": lambda a, p: np.reciprocal(a[0]),
    "isnan": lambda a, p: np.isnan(a[0]),
    "isfinite": lambda a, p: np.isfinite(a[0]),
    "signbit": lambda a, p: np.signbit(a[0]),
    "invert": lambda a, p: np.invert(a[0]),
    "logical_not": lambda a, p: np.logical_not(a[0]),
    "fabs": lambda a, p: np.fabs(a[0]),
    "sign": lambda a, p: np.sign(a[0]),
    "positive": lambda a, p: np.positive(a[0]),
    "conjugate": lambda a, p: np.conjugate(a[0]),
    "real": lambda a, p: np.real(a[0]).copy(),
    "nan_to_num": lambda a, p: np.nan_to_num(a[0]),
    "round": lambda a, p: np.round(a[0], p["decimals"]),
    "exp": lambda a, p: np.exp(a[0]),
    "sin": lambda a, p: np.sin(a[0]),
    "sqrt_out": lambda a, p: np.sqrt(a[0], out=a[1]),
    "sqrt_where": lambda a, p: np.sqrt(a[0], out=a[1], where=a[2]),
    "modf": lambda a, p: np.modf(a[0]),
    "frexp": lambda a, p: np.frexp(a[0]),
    # elementwise binary
    "add": lambda a, p: np.add(a[0], a[1]),
    "subtract": lambda a, p: np.subtract(a[0], a[1]),
    "multiply": lambda a, p: np.multiply(a[0], a[1]),
    "divide": lambda a, p: np.divide(a[0], a[1]),
    "floor_divide": lambda a, p: np.floor_divide(a[0], a[1]),
    "mod": lambda a, p: np.mod(a[0], a[1]),
    "fmod": lambda a, p: np.fmod(a[0], a[1]),
    "power": lambda a, p: np.power(a[0], a[1]),
    "maximum": lambda a, p: np.maximum(a[0], a[1]),
    "minimum": lambda a, p: np.minimum(a[0], a[1]),
    "fmax": lambda a, p: np.fmax(a[0], a[1]),
    "copysign": lambda a, p: np.copysign(a[0], a[1]),
    "heaviside": lambda a, p: np.heaviside(a[0], a[1]),
    "ldexp": lambda a, p: np.ldexp(a[0], a[1]),
    "gcd": lambda a, p: np.gcd(a[0], a[1]),
    "lcm": lambda a, p: np.lcm(a[0], a[1]),
    "bitwise_and": lambda a, p: np.bitwise_and(a[0], a[1]),
    "bitwise_xor": lambda a, p: np.bitwise_xor(a[0], a[1]),
    "left_shift": lambda a, p: np.left_shift(a[0], a[1]),
    "right_shift": lambda a, p: np.right_shift(a[0], a[1]),
    "less": lambda a, p: np.less(a[0], a[1]),
    "equal": lambda a, p: np.equal(a[0], a[1]),
    "greater_equal": lambda a, p: np.greater_equal(a[0], a[1]),
    "logical_and": lambda a, p: np.logical_and(a[0], a[1]),
    "logical_xor": lambda a, p: np.logical_xor(a[0], a[1]),
    "isclose": lambda a, p: np.isclose(a[0], a[1]),
    "allclose": lambda a, p: np.allclose(a[0], a[1]),
    "array_equal": lambda a, p: np.array_equal(a[0], a[1]),
    "add_scalar": lambda a, p: np.add(a[0], 3.0),
    "add_out": lambda a, p: np.add(a[0], a[1], out=a[2]),
    "add_where": lambda a, p: np.add(a[0], a[1], out=a[2], where=a[3]),
    "divmod": lambda a, p: np.divmod(a[0], a[1]),
    "op_add": lambda a, p: a[0] + a[1],
    "op_mul_scalar": lambda a, p: a[0] * 2.0,
    "op_gt": lambda a, p: a[0] > a[1],
    "op_eq": lambda a, p: a[0] == a[1],
    "op_neg": lambda a, p: -a[0],
    "op_not": lambda a, p: ~a[0],
    "op_and": lambda a, p: a[0] & a[1],
    "clip": lambda a, p: np.clip(a[0], a[1], a[2]),
    "clip_scalars": lambda a, p: np.clip(a[0], 10.0, 500.0),
    "where3": lambda a, p: np.where(a[0], a[1], a[2]),
    "where3_scalar": lambda a, p: np.where(a[0], 7.0, a[1]),
    "where1": lambda a, p: np.where(a[0]),
    "evaluate_fma": lambda a, p: a[0] * a[1] + a[2],
    "evaluate_sum": lambda a, p: np.sum(a[0] * a[1]),
    # reductions / scans
    "sum": lambda a, p: np.sum(a[0], axis=p.get("axis")),
    "sum_dtype": lambda a, p: np.sum(a[0], dtype=np.int64),
    "prod": lambda a, p: np.prod(a[0]),
    "mean": lambda a, p: np.mean(a[0], axis=p.get("axis")),
    "std": lambda a, p: np.std(a[0]),
    "var": lambda a, p: np.var(a[0], axis=p.get("axis")),
    "max": lambda a, p: np.max(a[0], axis=p.get("axis")),
    "min": lambda a, p: np.min(a[0], axis=p.get("axis")),
    "max_initial_where": lambda a, p: np.max(a[0], initial=-1.0, where=a[1]),
    "argmax": lambda a, p: np.argmax(a[0], axis=p.get("axis")),
    "argmin": lambda a, p: np.argmin(a[0], axis=p.get("axis")),
    "all": lambda a, p: np.all(a[0], axis=p.get("axis")),
    "any": lambda a, p: np.any(a[0], axis=p.get("axis")),
    "count_nonzero": lambda a, p: np.count_nonzero(a[0], axis=p.get("axis")),
    "nansum": lambda a, p: np.nansum(a[0]),
    "nanmax": lambda a, p: np.nanmax(a[0]),
    "nanmean": lambda a, p: np.nanmean(a[0]),
    "nanvar": lambda a, p: np.nanvar(a[0]),
    "ptp": lambda a, p: np.ptp(a[0]),
    "average_weights": lambda a, p: np.average(a[0], weights=a[1]),
    "median": lambda a, p: np.median(a[0]),
    "percentile": lambda a, p: np.percentile(a[0], 30.0),
    "quantile_axis": lambda a, p: np.quantile(a[0], 0.4, axis=0),
    "nanmedian": lambda a, p: np.nanmedian(a[0]),
    "cumsum": lambda a, p: np.cumsum(a[0], axis=p.get("axis")),
    "cumprod": lambda a, p: np.cumprod(a[0]),
    "nancumsum": lambda a, p: np.nancumsum(a[0]),
    "diff": lambda a, p: np.diff(a[0]),
    "ediff1d": lambda a, p: np.ediff1d(a[0]),
    "trapezoid": lambda a, p: np.trapezoid(a[0]),
    "gradient": lambda a, p: np.gradient(a[0]),
    # selection / indexing
    "take": lambda a, p: np.take(a[0], a[1], axis=p.get("axis"), mode=p.get("mode", "raise")),
    "put": lambda a, p: (np.put(a[0], a[1], a[2]), a[0])[1],
    "place": lambda a, p: (np.place(a[0], a[1], a[2]), a[0])[1],
    "putmask": lambda a, p: (np.putmask(a[0], a[1], a[2]), a[0])[1],
    "take_along_axis": lambda a, p: np.take_along_axis(a[0], a[1], axis=0),
    "put_along_axis": lambda a, p: (np.put_along_axis(a[0], a[1], a[2], axis=0), a[0])[1],
    "choose": lambda a, p: np.choose(a[0], [a[1], a[2]]),
    "compress": lambda a, p: np.compress(a[0], a[1]),
    "extract": lambda a, p: np.extract(a[0], a[1]),
    "select": lambda a, p: np.select([a[0], a[1]], [a[2], a[3]]),
    "copyto": lambda a, p: (np.copyto(a[0], a[1]), a[0])[1],
    "copyto_where": lambda a, p: (np.copyto(a[0], a[1], where=a[2]), a[0])[1],
    "fill_diagonal": lambda a, p: (np.fill_diagonal(a[0], a[1]), a[0])[1],
    "diag": lambda a, p: np.diag(a[0]),
    "diagonal_copy": lambda a, p: np.diagonal(a[0]).copy(),
    "trace": lambda a, p: np.trace(a[0]),
    "tril": lambda a, p: np.tril(a[0]),
    "triu": lambda a, p: np.triu(a[0]),
    "nonzero": lambda a, p: np.nonzero(a[0]),
    "flatnonzero": lambda a, p: np.flatnonzero(a[0]),
    "argwhere": lambda a, p: np.argwhere(a[0]),
    "unravel_index": lambda a, p: np.unravel_index(a[0], tuple(p["dims"])),
    "ravel_multi_index": lambda a, p: np.ravel_multi_index((a[0], a[1]), (7, 7)),
    "fancy_get": lambda a, p: a[0][a[1]],
    "fancy_get2": lambda a, p: a[0][a[1], a[2]],
    "fancy_get_cols": lambda a, p: a[0][:, a[1]],
    "fancy_set": lambda a, p: (a[0].__setitem__(a[1], a[2]), a[0])[1],
    "bool_get": lambda a, p: a[0][a[1]],
    "bool_set": lambda a, p: (a[0].__setitem__(a[1], a[2]), a[0])[1],
    "slice_set": lambda a, p: (a[0].__setitem__(slice(None, None, 2), a[1]), a[0])[1],
    # sort / search / sets / counting
    "sort": lambda a, p: np.sort(a[0]),
    "argsort": lambda a, p: np.argsort(a[0], kind="stable"),
    "partition": lambda a, p: np.partition(a[0], p["kth"]),
    "argpartition": lambda a, p: np.argpartition(a[0], p["kth"]),
    "lexsort": lambda a, p: np.lexsort((a[0], a[1])),
    "searchsorted": lambda a, p: np.searchsorted(a[0], a[1]),
    "unique": lambda a, p: np.unique(a[0]),
    "unique_counts": lambda a, p: tuple(np.unique_counts(a[0])),
    "isin": lambda a, p: np.isin(a[0], a[1]),
    "intersect1d": lambda a, p: np.intersect1d(a[0], a[1]),
    "union1d": lambda a, p: np.union1d(a[0], a[1]),
    "setdiff1d": lambda a, p: np.setdiff1d(a[0], a[1]),
    "setxor1d": lambda a, p: np.setxor1d(a[0], a[1]),
    "digitize": lambda a, p: np.digitize(a[0], a[1]),
    "bincount": lambda a, p: np.bincount(a[0]),
    "bincount_weights": lambda a, p: np.bincount(a[0], weights=a[1]),
    "histogram": lambda a, p: np.histogram(a[0], bins=10)[0],
    "sort_complex": lambda a, p: np.sort_complex(a[0]),
    "nanargmax": lambda a, p: np.nanargmax(a[0]),
    # linear algebra / products
    "dot": lambda a, p: np.dot(a[0], a[1]),
    "matmul": lambda a, p: np.matmul(a[0], a[1]),
    "inner": lambda a, p: np.inner(a[0], a[1]),
    "vdot": lambda a, p: np.vdot(a[0], a[1]),
    "vecdot": lambda a, p: np.vecdot(a[0], a[1]),
    "matvec": lambda a, p: np.matvec(a[0], a[1]),
    "vecmat": lambda a, p: np.vecmat(a[0], a[1]),
    "outer": lambda a, p: np.outer(a[0], a[1]),
    "tensordot": lambda a, p: np.tensordot(a[0], a[1], axes=1),
    "einsum_mv": lambda a, p: np.einsum("ij,j->i", a[0], a[1]),
    "kron": lambda a, p: np.kron(a[0], a[1]),
    "cross": lambda a, p: np.cross(a[0], a[1]),
    "det": lambda a, p: np.linalg.det(a[0]),
    "solve": lambda a, p: np.linalg.solve(a[0], a[1]),
    "inv": lambda a, p: np.linalg.inv(a[0]),
    "norm": lambda a, p: np.linalg.norm(a[0]),
    "matrix_power": lambda a, p: np.linalg.matrix_power(a[0], 3),
    "cov": lambda a, p: np.cov(a[0]),
    "corrcoef": lambda a, p: np.corrcoef(a[0]),
    # sliding / fft
    "convolve": lambda a, p: np.convolve(a[0], a[1]),
    "correlate": lambda a, p: np.correlate(a[0], a[1]),
    "fft": lambda a, p: np.fft.fft(a[0]),
    "rfft": lambda a, p: np.fft.rfft(a[0]),
    "ifft": lambda a, p: np.fft.ifft(a[0]),
    "fft2": lambda a, p: np.fft.fft2(a[0]),
    "fftshift": lambda a, p: np.fft.fftshift(a[0]),
    # manipulation / copies / casts
    "concatenate": lambda a, p: np.concatenate([a[0], a[1]], axis=p.get("axis", 0)),
    "stack": lambda a, p: np.stack([a[0], a[1]]),
    "vstack": lambda a, p: np.vstack([a[0], a[1]]),
    "hstack": lambda a, p: np.hstack([a[0], a[1]]),
    "column_stack": lambda a, p: np.column_stack([a[0], a[1]]),
    "repeat": lambda a, p: np.repeat(a[0], 2),
    "repeat_array": lambda a, p: np.repeat(a[0], a[1]),
    "tile": lambda a, p: np.tile(a[0], 2),
    "roll": lambda a, p: np.roll(a[0], 3),
    "pad": lambda a, p: np.pad(a[0], 2),
    "insert": lambda a, p: np.insert(a[0], 5, a[1]),
    "delete": lambda a, p: np.delete(a[0], a[1]),
    "append": lambda a, p: np.append(a[0], a[1]),
    "resize": lambda a, p: np.resize(a[0], (p["n"],)),
    "trim_zeros": lambda a, p: np.trim_zeros(a[0]).copy(),
    "flatten": lambda a, p: a[0].flatten(),
    "ravel": lambda a, p: np.ravel(a[0]).copy(),
    "copy": lambda a, p: np.copy(a[0]),
    "copy_f": lambda a, p: a[0].copy(order="F"),
    "ascontiguousarray": lambda a, p: np.ascontiguousarray(a[0]),
    "asfortranarray": lambda a, p: np.asfortranarray(a[0]),
    "astype": lambda a, p: a[0].astype(p["dtype"]),
    "broadcast_to_copy": lambda a, p: np.broadcast_to(a[0], (p["side"], p["side"])).copy(),
    "rot90_copy": lambda a, p: np.rot90(a[0]).copy(),
    "flip_copy": lambda a, p: np.flip(a[0]).copy(),
    "unique_axis": lambda a, p: np.unique(a[0], axis=0),
    "interp": lambda a, p: np.interp(a[0], a[1], a[2]),
    "meshgrid": lambda a, p: tuple(np.meshgrid(a[0], a[1])),
    "array_split_piece": lambda a, p: np.array_split(a[0], 3)[1].copy(),
    # numpy.polynomial / legacy polynomial
    "polyval": lambda a, p: np.polynomial.polynomial.polyval(a[0], a[1]),
    "chebval": lambda a, p: np.polynomial.chebyshev.chebval(a[0], a[1]),
    "polyder": lambda a, p: np.polynomial.polynomial.polyder(a[0]),
    "polyint": lambda a, p: np.polynomial.polynomial.polyint(a[0]),
    "polyadd": lambda a, p: np.polynomial.polynomial.polyadd(a[0], a[1]),
    "polymul": lambda a, p: np.polynomial.polynomial.polymul(a[0], a[1]),
    "polyvander": lambda a, p: np.polynomial.polynomial.polyvander(a[0], 3),
    "mapdomain": lambda a, p: np.polynomial.polyutils.mapdomain(a[0], [0.0, 1024.0], [-1.0, 1.0]),
    "legacy_polyval": lambda a, p: np.polyval(a[0], a[1]),
    # numpy.ma
    "ma_sum": lambda a, p: np.ma.array(a[0], mask=a[1]).sum(),
    "ma_add": lambda a, p: (np.ma.array(a[0], mask=a[1]) + np.ma.array(a[2], mask=a[3])).filled(0.0),
    "ma_masked_where": lambda a, p: np.ma.masked_where(a[0], a[1]).filled(-1.0),
    # conversion / IO (the C# side returns the bytes of its .NET result)
    "tobytes": lambda a, p: a[0].tobytes(),
    "to_array": lambda a, p: np.ascontiguousarray(a[0]).tobytes(),
    "to_multidim": lambda a, p: np.ascontiguousarray(a[0]).tobytes(),
    "to_jagged": lambda a, p: np.ascontiguousarray(a[0]).tobytes(),
    "save_npy": lambda a, p: _save_bytes(a[0]),
    "view_int64": lambda a, p: a[0].view(np.int64).copy(),
    "nditer_sum": lambda a, p: np.float64(sum(a[0].ravel().tolist())),
    "flat_sum": lambda a, p: np.float64(sum(a[0].ravel().tolist())),
}


# =============================================================================================================
# The case matrix
# =============================================================================================================

def _F(**kw):
    """A float64 1-D operand with values 0..996 (``mod`` defaults overridable)."""
    return mod("float64", [N], kw.pop("mod_", 997), **kw)


def _cases():
    """Yields (op, args, params, parity, host) for every case of the matrix.

    Each call appears with ALL its array operands temporary; calls with several operands also appear with one
    operand temporary at a time (the others held), so a gap names the operand that loses its anchor.
    """
    pos = dict(mod_=97, off=1.0)                 # 1..97: safe divisors, logarithms, square roots
    small16 = dict(mod_=16)                      # 0..15: products and sums stay exact integers
    yield from [
        # ---- elementwise unary
        ("sqrt", [_F(**pos)], {}, True, False),
        ("sqrt", [mod("float32", [N], 97, off=1.0)], {}, True, False),
        ("sqrt", [mod("float16", [N], 97, off=1.0)], {}, True, False),
        # complex128 runs NEGATIVE, not sqrt: NumSharp's complex sqrt is 1 ULP off NumPy on ~10% of finite inputs
        # (sqrt(2+7j): ...0887 vs NumPy's ...0882), a value-parity gap outside this tier's scope.
        ("negative", [cmod([N // 2], 97, 89)], {}, True, False),
        ("sqrt", [_F(**pos, view="step2")], {}, True, False),
        ("sqrt", [_F(**pos, view="rev")], {}, True, False),
        ("sqrt", [mod("float64", [SIDE, SIDE], 97, off=1.0, view="T")], {}, True, False),
        ("abs", [_F(off=-498.0)], {}, True, False),
        ("negative", [_F()], {}, True, False),
        ("floor", [_F(scale=0.25)], {}, True, False),
        ("ceil", [_F(scale=0.25)], {}, True, False),
        ("trunc", [_F(scale=0.25, off=-100.0)], {}, True, False),
        ("rint", [_F(scale=0.5)], {}, True, False),
        ("square", [_F()], {}, True, False),
        ("reciprocal", [_F(**pos)], {}, True, False),
        ("reciprocal", [mod("int64", [N], 1000, off=1.0)], {}, True, False),
        ("reciprocal", [mod("int32", [N], 1000, off=1.0)], {}, True, False),
        ("isnan", [_F()], {}, True, False),
        ("isfinite", [_F()], {}, True, False),
        ("signbit", [_F(off=-498.0)], {}, True, False),
        ("invert", [mod("int64", [N], 1000)], {}, True, False),
        ("logical_not", [mod("bool", [8 * N], 3)], {}, True, False),
        ("fabs", [_F(off=-498.0)], {}, True, False),
        ("sign", [_F(off=-498.0)], {}, True, False),
        ("positive", [_F()], {}, True, False),
        ("conjugate", [cmod([N // 2], 97, 89)], {}, True, False),
        ("real", [cmod([N // 2], 97, 89)], {}, True, False),
        ("nan_to_num", [_F()], {}, True, False),
        ("round", [_F(scale=0.25)], {"decimals": 2}, True, False),
        ("exp", [mod("float32", [N], 97, scale=0.0625)], {}, True, False),
        ("sin", [mod("float32", [N], 997, scale=0.0625)], {}, True, False),
        ("exp", [_F(scale=0.0625, mod_=97)], {}, True, True),
        ("sqrt_out", [_F(**pos), _F(mod_=1, scale=0.0)], {}, True, False),
        ("sqrt_where", [_F(**pos), _F(mod_=1, scale=0.0), mod("bool", [N], 3)], {}, True, False),
        ("modf", [_F(scale=0.25, off=-50.0)], {}, True, False),
        ("frexp", [_F(**pos)], {}, True, False),
        # ---- elementwise binary
        ("add", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("subtract", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("multiply", [_F(**small16), _F(start=3, **small16)], {}, True, False),
        ("divide", [_F(), _F(**pos)], {}, True, False),
        ("floor_divide", [_F(), _F(**pos)], {}, True, False),
        ("mod", [_F(), _F(**pos)], {}, True, False),
        ("fmod", [_F(off=-498.0), _F(**pos)], {}, True, False),
        ("power", [mod("int64", [N], 7), mod("int64", [N], 4, start=1)], {}, True, False),
        ("maximum", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("minimum", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("fmax", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("copysign", [_F(), _F(start=5, mod_=991, off=-495.0)], {}, True, False),
        ("heaviside", [_F(off=-498.0), _F(start=5, mod_=991)], {}, True, False),
        ("ldexp", [_F(), mod("int32", [N], 5)], {}, True, False),
        ("gcd", [mod("int64", [N], 1000), mod("int64", [N], 7, start=1)], {}, True, False),
        ("lcm", [mod("int64", [N], 100), mod("int64", [N], 7, start=1)], {}, True, False),
        ("bitwise_and", [mod("int64", [N], 1000), mod("int64", [N], 7)], {}, True, False),
        ("bitwise_xor", [mod("int64", [N], 1000), mod("int64", [N], 7)], {}, True, False),
        ("left_shift", [mod("int64", [N], 1000), mod("int64", [N], 5)], {}, True, False),
        ("right_shift", [mod("int64", [N], 1000), mod("int64", [N], 5)], {}, True, False),
        ("less", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("equal", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("greater_equal", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("logical_and", [mod("bool", [N], 3), mod("bool", [N], 2)], {}, True, False),
        ("logical_xor", [mod("bool", [N], 3), mod("bool", [N], 2)], {}, True, False),
        ("isclose", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("allclose", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("array_equal", [_F(), _F()], {}, True, False),
        ("add", [mod("float32", [N], 997), mod("float32", [N], 991, start=5)], {}, True, False),
        ("add", [mod("float16", [N], 97), mod("float16", [N], 89, start=5)], {}, True, False),
        ("add", [cmod([N // 2], 97, 89), cmod([N // 2], 89, 83, start=5)], {}, True, False),
        ("multiply", [cmod([N // 2], 16, 16), cmod([N // 2], 16, 16, start=5)], {}, True, False),
        ("add", [mod("int64", [N], 1000), _F()], {}, True, False),
        ("add", [mod("float64", [SIDE, SIDE], 13), mod("float64", [SIDE], 11)], {}, True, False),
        ("add", [mod("float64", [SIDE, SIDE], 13), mod("float64", [SIDE, 1], 11)], {}, True, False),
        ("add", [mod("float64", [2 * N], 997, view="step2"), _F(start=5, mod_=991, view="rev")], {}, True, False),
        ("add_scalar", [_F()], {}, True, False),
        ("add_out", [_F(), _F(start=5, mod_=991), _F(mod_=1, scale=0.0)], {}, True, False),
        ("add_where", [_F(), _F(start=5, mod_=991), _F(mod_=1, scale=0.0), mod("bool", [N], 3)], {}, True, False),
        ("divmod", [_F(), _F(**pos)], {}, True, False),
        ("op_add", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("op_mul_scalar", [_F()], {}, True, False),
        ("op_gt", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("op_eq", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("op_neg", [_F()], {}, True, False),
        ("op_not", [mod("bool", [8 * N], 3)], {}, True, False),
        ("op_and", [mod("bool", [N], 3), mod("bool", [N], 2)], {}, True, False),
        ("clip", [_F(), mod("float64", [N], 1, off=10.0), mod("float64", [N], 1, off=500.0)], {}, True, False),
        ("clip_scalars", [_F()], {}, True, False),
        ("where3", [mod("bool", [N], 3), _F(), _F(start=5, mod_=991)], {}, True, False),
        ("where3", [mod("bool", [N], 3), mod("float32", [N], 997), mod("float32", [N], 991, start=5)], {}, True, False),
        ("where3", [mod("bool", [N], 3), mod("int64", [N], 1000), mod("int64", [N], 991, start=5)], {}, True, False),
        ("where3_scalar", [mod("bool", [N], 3), _F()], {}, True, False),
        ("where1", [mod("bool", [8 * N], 3)], {}, True, False),
        ("evaluate_fma", [_F(**small16), _F(start=3, **small16), _F()], {}, True, False),
        ("evaluate_sum", [_F(**small16), _F(start=3, **small16)], {}, True, False),
        # ---- reductions / scans
        ("sum", [_F()], {}, True, False),
        ("sum", [mod("float64", [SIDE, SIDE], 997)], {"axis": 0}, True, False),
        ("sum", [mod("float64", [SIDE, SIDE], 997)], {"axis": 1}, True, False),
        ("sum", [mod("float32", [N], 16)], {}, True, False),
        ("sum", [mod("int64", [N], 1000)], {}, True, False),
        ("sum_dtype", [_F()], {}, True, False),
        ("prod", [mod("float64", [N], 1, off=1.0)], {}, True, False),
        ("mean", [_F()], {}, True, False),
        ("mean", [mod("float64", [SIDE, SIDE], 997)], {"axis": 0}, True, False),
        ("std", [_F()], {}, True, False),
        ("var", [mod("float64", [SIDE, SIDE], 997)], {"axis": 1}, True, False),
        ("max", [_F()], {}, True, False),
        ("max", [mod("float64", [SIDE, SIDE], 997)], {"axis": 0}, True, False),
        ("min", [mod("float64", [SIDE, SIDE], 997)], {"axis": 1}, True, False),
        ("max", [mod("float16", [4 * N], 97)], {}, True, False),
        ("max_initial_where", [_F(), mod("bool", [N], 3)], {}, True, False),
        ("argmax", [_F()], {}, True, False),
        ("argmax", [mod("float64", [SIDE, SIDE], 997)], {"axis": 0}, True, False),
        ("argmin", [mod("float64", [SIDE, SIDE], 997)], {"axis": 1}, True, False),
        ("all", [mod("bool", [64 * N], 1)], {}, True, False),
        ("any", [mod("bool", [2 * SIDE, 4 * SIDE], 3)], {"axis": 0}, True, False),
        ("count_nonzero", [mod("bool", [8 * N], 3)], {}, True, False),
        ("count_nonzero", [mod("bool", [2 * SIDE, 4 * SIDE], 3)], {"axis": 0}, True, False),
        ("nansum", [_F()], {}, True, False),
        ("nanmax", [_F()], {}, True, False),
        ("nanmean", [_F()], {}, True, False),
        ("nanvar", [_F()], {}, True, False),
        ("ptp", [_F()], {}, True, False),
        ("average_weights", [_F(), _F(**pos)], {}, True, False),
        ("median", [_F(start=5, mod_=991)], {}, True, False),
        ("percentile", [_F(start=5, mod_=991)], {}, True, False),
        ("quantile_axis", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("nanmedian", [_F(start=5, mod_=991)], {}, True, False),
        ("cumsum", [_F()], {}, True, False),
        ("cumsum", [mod("int64", [N], 1000)], {}, True, False),
        ("cumsum", [mod("float64", [SIDE, SIDE], 997)], {"axis": 0}, True, False),
        ("cumprod", [mod("float64", [N], 1, off=1.0)], {}, True, False),
        ("nancumsum", [_F()], {}, True, False),
        ("diff", [_F()], {}, True, False),
        ("ediff1d", [_F()], {}, True, False),
        ("trapezoid", [_F()], {}, True, False),
        ("gradient", [_F()], {}, True, False),
        # ---- selection / indexing
        ("take", [mod("float64", [N], 997), perm("int64", [N], N, start=5, step=7919)], {}, True, False),
        ("take", [mod("float64", [N], 997), perm("int32", [N], N, start=5, step=7919)], {}, True, False),
        ("take", [mod("float64", [N], 997), perm("int64", [N], N, start=5, step=7919)], {"axis": 0}, True, False),
        ("take", [mod("float64", [N], 997), perm("int64", [N], N + 11, start=5, step=7919)], {"mode": "wrap"}, True, False),
        ("take", [mod("float64", [N], 997), perm("int64", [N], N + 11, start=5, step=7919)], {"mode": "clip"}, True, False),
        ("take", [mod("int16", [N], 997), perm("int64", [N], N, start=5, step=7919)], {}, True, False),
        ("put", [_F(mod_=1, scale=0.0), perm("int64", [N], N, start=5, step=7919), _F(mod_=991, off=1000.0)], {}, True, False),
        ("place", [mod("float64", [4 * N], 1, scale=0.0), mod("bool", [4 * N], 3), mod("float64", [4 * N], 991, off=1000.0)], {}, True, False),
        ("putmask", [mod("float64", [4 * N], 1, scale=0.0), mod("bool", [4 * N], 3), mod("float64", [4 * N], 991, off=1000.0)], {}, True, False),
        ("take_along_axis", [mod("float64", [N], 997), perm("int64", [N], N, start=5, step=7919)], {}, True, False),
        ("put_along_axis", [_F(mod_=1, scale=0.0), perm("int64", [N], N, start=5, step=7919), _F(mod_=991, off=1000.0)], {}, True, False),
        ("choose", [mod("int64", [N], 2), _F(), _F(start=5, mod_=991)], {}, True, False),
        ("compress", [mod("bool", [N], 3), _F()], {}, True, False),
        ("extract", [mod("bool", [N], 3), _F()], {}, True, False),
        ("select", [mod("bool", [N], 3), mod("bool", [N], 2), _F(), _F(start=5, mod_=991)], {}, True, False),
        ("copyto", [_F(mod_=1, scale=0.0), _F()], {}, True, False),
        ("copyto", [mod("float64", [4 * N], 1, scale=0.0), mod("float32", [4 * N], 997)], {}, True, False),
        ("copyto_where", [_F(mod_=1, scale=0.0), _F(), mod("bool", [N], 3)], {}, True, False),
        ("fill_diagonal", [mod("float64", [SIDE, SIDE], 1, scale=0.0), mod("float64", [SIDE], 11)], {}, True, False),
        ("diag", [mod("float64", [SIDE], 11)], {}, True, False),
        ("diagonal_copy", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("trace", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("tril", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("triu", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("nonzero", [mod("bool", [8 * N], 3)], {}, True, False),
        ("flatnonzero", [mod("bool", [8 * N], 3)], {}, True, False),
        ("argwhere", [mod("bool", [2 * SIDE, 4 * SIDE], 3)], {}, True, False),
        ("unravel_index", [perm("int64", [N], N, start=5, step=7919)], {"dims": [SIDE, SIDE]}, True, False),
        ("ravel_multi_index", [mod("int64", [N], 7), mod("int64", [N], 7, start=3)], {}, True, False),
        ("fancy_get", [mod("float64", [N], 997), perm("int64", [N], N, start=5, step=7919)], {}, True, False),
        ("fancy_get2", [mod("float64", [SIDE, SIDE], 997), perm("int64", [N], SIDE, start=5, step=7), perm("int64", [N], SIDE, start=1, step=3)], {}, True, False),
        ("fancy_get_cols", [mod("float64", [SIDE, SIDE], 997), perm("int64", [SIDE], SIDE, start=5, step=7)], {}, True, False),
        ("fancy_get", [mod("float64", [SIDE, SIDE], 997), perm("int64", [SIDE], SIDE, start=5, step=7)], {}, True, False),
        ("fancy_set", [_F(mod_=1, scale=0.0), perm("int64", [N], N, start=5, step=7919), _F(mod_=991, off=1000.0)], {}, True, False),
        ("bool_get", [_F(), mod("bool", [N], 3)], {}, True, False),
        ("bool_set", [_F(mod_=1, scale=0.0), mod("bool", [N], 2), mod("float64", [N // 2], 991, off=1000.0)], {}, True, False),
        ("slice_set", [_F(mod_=1, scale=0.0), mod("float64", [N // 2], 991, off=1000.0)], {}, True, False),
        # ---- sort / search / sets / counting
        ("sort", [_F(start=5, mod_=991)], {}, True, False),
        ("argsort", [_F(start=5, mod_=991)], {}, True, False),
        ("partition", [_F(start=5, mod_=991)], {"kth": N // 2}, False, False),
        ("argpartition", [_F(start=5, mod_=991)], {"kth": N // 2}, False, False),
        ("lexsort", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("searchsorted", [mod("float64", [N], N), _F(start=5, mod_=991)], {}, True, False),
        ("unique", [_F(start=5, mod_=991)], {}, True, False),
        ("unique_counts", [_F(start=5, mod_=991)], {}, True, False),
        ("isin", [_F(), mod("float64", [64], 64)], {}, True, False),
        ("isin", [mod("float64", [64], 64), _F()], {}, True, False),
        ("intersect1d", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("union1d", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("setdiff1d", [_F(), mod("float64", [64], 64)], {}, True, False),
        ("setxor1d", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("digitize", [_F(), mod("float64", [100], 100, scale=10.0)], {}, True, False),
        ("bincount", [mod("int64", [N], 1000)], {}, True, False),
        ("bincount_weights", [mod("int64", [N], 1000), _F(**small16)], {}, True, False),
        ("histogram", [_F()], {}, True, False),
        ("sort_complex", [cmod([N // 2], 97, 89)], {}, True, False),
        ("nanargmax", [_F()], {}, True, False),
        # ---- linear algebra / products
        ("dot", [_F(**small16), _F(start=3, **small16)], {}, True, False),
        ("dot", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("dot", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE, SIDE], 16, start=3)], {}, True, False),
        ("matmul", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE, SIDE], 16, start=3)], {}, True, False),
        ("inner", [_F(**small16), _F(start=3, **small16)], {}, True, False),
        ("vdot", [_F(**small16), _F(start=3, **small16)], {}, True, False),
        ("vecdot", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("matvec", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("vecmat", [mod("float64", [SIDE], 16, start=3), mod("float64", [SIDE, SIDE], 16)], {}, True, False),
        ("outer", [mod("float64", [SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("tensordot", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE, SIDE], 16, start=3)], {}, True, False),
        ("einsum_mv", [mod("float64", [SIDE, SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("kron", [mod("float64", [SIDE], 16), mod("float64", [SIDE], 16, start=3)], {}, True, False),
        ("cross", [mod("float64", [N // 4, 3], 16), mod("float64", [N // 4, 3], 16, start=3)], {}, True, False),
        ("det", [eye(512, 4096.0, 5)], {}, False, False),
        ("solve", [eye(512, 4096.0, 5), mod("float64", [512], 16)], {}, False, False),
        ("inv", [eye(512, 4096.0, 5)], {}, False, False),
        ("norm", [_F(**small16)], {}, True, False),
        ("matrix_power", [mod("float64", [64, 64], 3)], {}, True, False),
        ("cov", [mod("float64", [8, N // 8], 16)], {}, False, False),
        ("corrcoef", [mod("float64", [8, N // 8], 16)], {}, False, False),
        # ---- sliding / fft
        ("convolve", [_F(**small16), mod("float64", [8], 8, off=1.0)], {}, True, False),
        ("correlate", [_F(**small16), mod("float64", [8], 8, off=1.0)], {}, True, False),
        ("convolve", [mod("int64", [N], 16), mod("int64", [8], 8, off=1.0)], {}, True, False),
        ("fft", [_F(**small16)], {}, True, True),
        ("rfft", [_F(**small16)], {}, True, True),
        ("ifft", [cmod([N // 2], 16, 16)], {}, True, True),
        ("fft2", [mod("float64", [SIDE, SIDE], 16)], {}, True, True),
        ("fftshift", [_F()], {}, True, False),
        # ---- manipulation / copies / casts
        ("concatenate", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("concatenate", [mod("float64", [SIDE, SIDE], 997), mod("float64", [SIDE, SIDE], 991, start=5)], {"axis": 1}, True, False),
        ("stack", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("vstack", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("hstack", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("column_stack", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("repeat", [_F()], {}, True, False),
        ("repeat_array", [mod("float64", [4096], 997), mod("int64", [4096], 4)], {}, True, False),
        ("tile", [_F()], {}, True, False),
        ("roll", [_F()], {}, True, False),
        ("pad", [_F()], {}, True, False),
        ("insert", [_F(), mod("float64", [64], 64)], {}, True, False),
        ("delete", [_F(), perm("int64", [1000], N, start=5, step=7919)], {}, True, False),
        ("append", [_F(), _F(start=5, mod_=991)], {}, True, False),
        ("resize", [_F()], {"n": N + 7}, True, False),
        ("trim_zeros", [_F(start=1)], {}, True, False),
        ("flatten", [mod("float64", [SIDE, SIDE], 997, view="T")], {}, True, False),
        ("ravel", [mod("float64", [SIDE, SIDE], 997, view="T")], {}, True, False),
        ("copy", [_F()], {}, True, False),
        ("copy_f", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("ascontiguousarray", [mod("float64", [SIDE, SIDE], 997, view="T")], {}, True, False),
        ("asfortranarray", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("astype", [_F()], {"dtype": "float32"}, True, False),
        ("astype", [_F()], {"dtype": "int32"}, True, False),
        ("astype", [_F()], {"dtype": "float16"}, True, False),
        ("astype", [_F()], {"dtype": "bool"}, True, False),
        ("astype", [_F()], {"dtype": "complex128"}, True, False),
        ("astype", [_F(view="step2")], {"dtype": "float32"}, True, False),
        ("broadcast_to_copy", [mod("float64", [SIDE], 11)], {"side": SIDE}, True, False),
        ("rot90_copy", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("flip_copy", [_F()], {}, True, False),
        ("unique_axis", [mod("float64", [N // 64, 64], 7)], {}, True, False),
        ("interp", [_F(scale=0.5), mod("float64", [1000], 1000, scale=1.0), mod("float64", [1000], 1000, scale=2.0)], {}, True, False),
        ("meshgrid", [mod("float64", [SIDE], 11), mod("float64", [SIDE], 13)], {}, True, False),
        ("array_split_piece", [_F()], {}, True, False),
        # ---- numpy.polynomial / legacy polynomial
        ("polyval", [_F(scale=0.0009765625), mod("float64", [8], 5, off=-2.0)], {}, True, False),
        ("chebval", [_F(scale=0.0009765625), mod("float64", [8], 5, off=-2.0)], {}, True, False),
        ("polyder", [mod("float64", [N], 16)], {}, True, False),
        ("polyint", [mod("float64", [N], 16)], {}, True, False),
        ("polyadd", [mod("float64", [4096], 16), mod("float64", [4096], 16, start=3)], {}, True, False),
        ("polymul", [mod("float64", [1024], 16), mod("float64", [1024], 16, start=3)], {}, True, False),
        ("polyvander", [_F(scale=0.0009765625)], {}, True, False),
        ("mapdomain", [_F()], {}, True, False),
        ("legacy_polyval", [mod("float64", [8], 5, off=-2.0), _F(scale=0.0009765625)], {}, True, False),
        # ---- numpy.ma
        ("ma_sum", [_F(), mod("bool", [N], 3)], {}, True, False),
        ("ma_add", [_F(), mod("bool", [N], 3), _F(start=5, mod_=991), mod("bool", [N], 2)], {}, True, False),
        ("ma_masked_where", [mod("bool", [N], 3), _F()], {}, True, False),
        # ---- strings (fixed-width U)
        ("copy", [mod("U6", [N // 8], 1000)], {}, True, False),
        ("take", [mod("U6", [N // 8], 1000), perm("int64", [N // 8], N // 8, start=5, step=7919)], {}, True, False),
        ("concatenate", [mod("U6", [N // 8], 1000), mod("U6", [N // 8], 991, start=5)], {}, True, False),
        ("fancy_get", [mod("U6", [N // 8], 1000), perm("int64", [N // 8], N // 8, start=5, step=7919)], {}, True, False),
        ("where3", [mod("bool", [N // 8], 3), mod("U6", [N // 8], 1000), mod("U6", [N // 8], 991, start=5)], {}, True, False),
        # ---- conversion / IO
        ("tobytes", [_F()], {}, True, False),
        ("to_array", [_F()], {}, True, False),
        ("to_multidim", [mod("float64", [2 * SIDE, 4 * SIDE], 997)], {}, True, False),
        ("to_jagged", [mod("float64", [SIDE, SIDE], 997)], {}, True, False),
        ("save_npy", [_F()], {}, True, False),
        ("view_int64", [_F()], {}, True, False),
        ("nditer_sum", [_F()], {}, True, False),
        ("flat_sum", [_F()], {}, True, False),
    ]


def _variants(op, args):
    """The temp patterns a call is replayed with: all operands temporary, then (for several operands) each one
    alone, so a lost anchor can be attributed to the operand that loses it.

    Parameters:
        op: the op key (put-style calls keep their TARGET -- operand 0, written in place -- temporary always, since
            its result is the target itself and the target cannot be held across repetitions).
        args: the operand specs.

    Returns:
        A list of (tag, args) pairs: tag ``all`` or ``arg<k>``.
    """
    out = [("all", [dict(a, temp=True) for a in args])]
    targeted = op in TARGET_OPS
    if len(args) > 1:
        for k in range(len(args)):
            if targeted and k == 0:
                continue
            out.append((f"arg{k}", [dict(a, temp=(j == k or (targeted and j == 0))) for j, a in enumerate(args)]))
    return out


#: Calls whose operand 0 is a TARGET written in place (the result IS that operand).
TARGET_OPS = {"put", "place", "putmask", "put_along_axis", "copyto", "copyto_where", "fill_diagonal", "fancy_set",
              "bool_set", "slice_set", "sqrt_out", "sqrt_where", "add_out", "add_where"}


# =============================================================================================================
# Carving
# =============================================================================================================

#: The documented lost-anchor gaps: id -> (the NumSharp method that reads an operand's buffer and then runs its kernel
#: with no later use of the operand, the operands it loses). Drawn from three uncarved replays (2 MB operands at 48
#: repetitions, 8 MB at 24, the uncertain cases at 160) and a code read of each site: a variant is carved when one of
#: the operands the site loses is temporary, whether or not a given replay caught it (detection is a race; the code
#: is not). The [OpenBugs] test named after each id in Fuzz/Lifetime/LifetimeOracleTests.cs replays its cases.
GAPS = {
    "LT01": "np.take: ExecuteTakeKernel reads the source and the int64/int32 index buffer (Indexing/np.take.cs)",
    "LT02": "np.put: PutKernel reads the index and value buffers (Indexing/np.put.cs)",
    "LT03": "np.place: ExecutePlace reads the mask and value buffers (Indexing/np.place.cs)",
    "LT04": "np.putmask: ExecutePutMask reads the mask and value buffers (Indexing/np.putmask.cs)",
    "LT05": "np.take_along_axis: the gather reads the source and index buffers (Indexing/np.take_along_axis.cs)",
    "LT06": "np.put_along_axis: PutAlongAxisCore reads the index and value buffers (Indexing/np.put_along_axis.cs)",
    "LT07": "np.where(c, x, y): WhereKernelDispatch reads cond, x and y (APIs/np.where.cs)",
    "LT08": "np.copyto(where=): CopytoMaskedRaw reads the source and the mask (Manipulation/np.copyto.cs)",
    "LT09": "np.copyto across dtypes: NDIter.Copy's contiguous cast kernel reads the source (Backends/Iterators/NDIter.cs)",
    "LT10": "np.max(initial=, where=): RunMaxMin reads the input (Default.Reduction.MinMax.Keywords.cs)",
    "LT11": "np.max of float16: HalfMinMaxReduce reads the input (Default.Reduction.MinMax.Half.cs)",
    "LT12": "np.cumsum / np.cumprod (flat): ScanElementwiseFlat reads the input (Default.Reduction.CumAdd.cs)",
    "LT13": "np.nancumsum: NanReplace.ReplaceContiguousFloat reads the input (Backends/Kernels/NanReplace.cs)",
    "LT14": "np.searchsorted: the search kernel reads a and v (Sorting_Searching_Counting/np.searchsorted.cs)",
    "LT15": "np.digitize: the monotonicity scan and the search read x and bins (np.digitize.cs, np.searchsorted.cs)",
    "LT16": "np.bincount: MinMaxInt64 / BincountCount / BincountWeighted read x and the weights (np.bincount.cs)",
    "LT17": "np.reciprocal of integers: ReciprocalInteger reads the input (Default.Reciprocal.cs)",
    "LT18": "np.dot / np.inner of two vectors: DotInner1D reads both (Default.Dot.Fused.cs)",
    "LT19": "np.dot(matrix, vector) / np.matvec / np.vecmat: the managed product reads the matrix (Backends/Default/Math/BLAS/)",
    "LT20": "np.dot(matrix, matrix): DotNDMDSimdDouble reads both operands (Default.Dot.NDMD.cs)",
    "LT21": "np.linalg.det / inv / solve: ManagedLu reads a and b (Backends/Default/LinearAlgebra/ManagedLu.cs)",
    "LT22": "np.convolve / np.correlate: SlidingCorrelate reads the data and the kernel (Math/NDArray.SlidingDot.cs)",
    "LT23": "numpy.polynomial polyder: NDPolyCalc.TryDirectSource reads the series in place (NDPolyCalc.cs)",
    "LT24": "numpy.polynomial polyvander: NDPolyVander reads x (NDPolyVander.cs)",
    "LT25": "the ! operator of a bool array: NDArray.operator! reads its operand (NDArray.NOT.cs)",
    "LT26": "np.all of a bool array (flat): NDIter.ReduceBool reads the input (Default.All.cs, NDIter.cs)",
    "LT27": "NDArray.ToMuliDimArray: the copy reads Storage.Address with no later use (NDArrayToMultiDimArray.cs)",
}


def _temps(args):
    """The indices of the operands a stressed call builds afresh."""
    return {k for k, a in enumerate(args) if a["temp"]}


def _gap_of(op, tag, args, params):
    """Names the documented gap a case hits, or None for a gate case.

    Parameters:
        op: the op key.
        tag: the temp pattern (``all`` or ``arg<k>``).
        args: the operand specs (their ``temp`` flags say which operands the stressed call builds afresh).
        params: the call's parameters.

    Returns:
        A gap id from ``GAPS``, or None.
    """
    t = _temps(args)
    dt0 = args[0]["dtype"]
    simple = {
        "take": ("LT01", {0, 1}), "put": ("LT02", {1, 2}), "place": ("LT03", {1, 2}), "putmask": ("LT04", {1, 2}),
        "take_along_axis": ("LT05", {0, 1}), "put_along_axis": ("LT06", {1, 2}), "copyto_where": ("LT08", {1, 2}),
        "max_initial_where": ("LT10", {0}), "nancumsum": ("LT13", {0}), "searchsorted": ("LT14", {0, 1}),
        "digitize": ("LT15", {0, 1}), "bincount": ("LT16", {0}), "bincount_weights": ("LT16", {0, 1}),
        "inner": ("LT18", {0, 1}), "matvec": ("LT19", {0}), "vecmat": ("LT19", {1}),
        "det": ("LT21", {0}), "inv": ("LT21", {0}), "solve": ("LT21", {0, 1}),
        "correlate": ("LT22", {0, 1}), "polyder": ("LT23", {0}),
        "polyvander": ("LT24", {0}), "op_not": ("LT25", {0}), "to_multidim": ("LT27", {0}),
    }
    if op in simple:
        gap, lost = simple[op]
        return gap if t & lost else None
    if op == "where3":
        # The string where takes the NDIter route (anchored); the numeric dtypes take WhereKernelDispatch.
        return "LT07" if args[1]["dtype"] != "U6" and t & {0, 1, 2} else None
    if op == "copyto":
        return "LT09" if args[1]["dtype"] != args[0]["dtype"] and 1 in t else None
    if op == "max":
        return "LT11" if dt0 == "float16" and "axis" not in params and 0 in t else None
    if op in ("cumsum", "cumprod"):
        return "LT12" if "axis" not in params and 0 in t else None
    if op == "reciprocal":
        return "LT17" if dt0.startswith("int") and 0 in t else None
    if op == "dot":
        a_nd = len(args[0]["shape"])
        b_nd = len(args[1]["shape"])
        if a_nd == 1 and b_nd == 1:
            return "LT18" if t & {0, 1} else None
        if a_nd == 2 and b_nd == 1:
            return "LT19" if 0 in t else None
        return "LT20" if t & {0, 1} else None
    if op == "convolve":
        return "LT22" if 0 in t else None      # the kernel is reversed into a copy convolve owns
    if op == "all":
        return "LT26" if "axis" not in params and dt0 == "bool" and 0 in t else None
    return None


# =============================================================================================================
# Driver
# =============================================================================================================

def _materialize(args):
    """Builds fresh NumPy operands from the recipes. A TARGET operand (one the call writes in place) is therefore
    never shared between a case's variants: each variant's expected answer starts from the recipe's bytes."""
    return [build(a) for a in args]


def main():
    """Writes the gate file and the gap file (or, with ``--uncarved DIR``, every variant into one file in DIR).

    Each call of ``_cases()`` is expanded into its variants; a variant is carved when ``_gap_of`` names a gap for it,
    and its expected answer is NumPy's result over fresh operands. ``--bits`` changes the operand sizes (used by the
    classification runs: bigger operands give a collection more time to land inside a kernel)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uncarved", metavar="DIR", help="write every case to DIR/lifetime.jsonl (no carve)")
    ap.add_argument("--bits", type=int, default=18,
                    help="log2 of the 1-D operand size (even; 2-D operands are 2**(bits/2) square); default 18")
    ns = ap.parse_args()
    global N, SIDE
    N, SIDE = 1 << ns.bits, 1 << (ns.bits // 2)

    gate, gaps = [], []
    counter = {}
    for op, args, params, parity, host in _cases():
        assert op in OPS, op
        built = []
        for tag, vargs in _variants(op, args):
            n = counter.get((op, tag), 0)
            counter[(op, tag)] = n + 1
            case = {"id": f"lifetime/{op}/{tag}/{vargs[0]['dtype']}/{n}", "op": op, "args": vargs, "params": params,
                    "parity": parity, "host": host}
            built.append((case, None if ns.uncarved else _gap_of(op, tag, vargs, params)))
        # A call no gap touches keeps only its all-temporary variant (it already makes every operand temporary; the
        # one-at-a-time variants exist to attribute a gap to its operand). A call with a gap keeps every variant: the
        # carved ones pin the lost operands, the others show which operands ARE kept alive.
        if not ns.uncarved and all(g is None for _, g in built):
            built = built[:1]
        for case, gap in built:
            case["expected"] = record(OPS[op](_materialize(case["args"]), params))
            if gap:
                case["gap"] = gap
                gaps.append(case)
            else:
                gate.append(case)

    out_dir = ns.uncarved or CORPUS
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lifetime.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for c in gate:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    if not ns.uncarved:
        os.makedirs(GAP_DIR, exist_ok=True)
        with open(os.path.join(GAP_DIR, "lifetime_gaps.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for c in gaps:
                f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"lifetime: {len(gate)} gate cases, {len(gaps)} carved under {len({c['gap'] for c in gaps})} gaps",
          file=sys.stderr)


if __name__ == "__main__":
    main()
