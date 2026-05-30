"""
layout_catalog.py — the canonical catalog of memory layouts (the "44 variations").

Mirrored 1:1 by NumSharp.UnitTest/Fuzz/LayoutCatalog.cs (same layout names both sides).

Each builder takes a numpy dtype and returns (base, view):
  * `base` is ALWAYS a freshly-allocated C-contiguous ndarray. Its raw memory therefore
    equals base.tobytes(), which is what we serialize as the operand's underlying buffer.
  * `view` is the operand the op actually sees — produced from `base` using ONLY view
    operations (slicing, transpose, broadcast). Because `view` shares `base`'s buffer, the
    operand is fully described by (shape, element-strides, element-offset) into base.tobytes().

This guarantees C# can reconstruct the EXACT same logical array from the bytes alone.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Deterministic, bit-pattern-diverse value pools (include all the edges that
# broke the cast kernel: overflow, NaN, +/-inf, -0.0, type min/max boundaries).
# ---------------------------------------------------------------------------
# Critical edges are FRONT-LOADED so that even an 8-element operand exercises the
# float->int overflow/NaN/inf/-0 paths (the cvtt sentinel cases that motivated this work).
_FLOAT_POOL = [
    float("nan"), float("inf"), float("-inf"), 2147483648.0, -2147483649.0, -0.0, 0.0, 1.0,
    -1.0, 0.5, -0.5, 1.9, -1.9, 127.0, 128.0, 255.0, 256.0, 32767.0, 65535.0,
    2147483647.0, -2147483648.0, 4294967295.0, 9.0e18, -9.0e18, 1e20, -1e20, 3.5e38,
    1234.7, -1234.7, 65536.0, 2.0, 3.0, 42.0,
]
# Narrowing-wrap and sign-boundary edges front-loaded for the same reason.
_INT_POOL = [
    0, -1, 127, 128, 255, 256, -128, -129, 32767, 32768, 65535, 65536,
    2147483647, -2147483648, 1, 2, 3, 42, 99999, -99999, 1234567, -1234567,
]


def _fill(n, dt):
    """Deterministic length-n 1-D array of dtype dt drawn from the edge pools."""
    dt = np.dtype(dt)
    if dt.kind == "f":
        base = np.array(_FLOAT_POOL, dtype=np.float64).astype(dt)
    elif dt.kind == "c":
        fp = np.array(_FLOAT_POOL, dtype=np.float64)
        base = (fp + 1j * np.roll(fp, 1)).astype(dt)
    elif dt.kind == "b":
        base = (np.arange(len(_INT_POOL)) % 3 == 0).astype(dt)
    else:  # signed/unsigned int — int64->target is modular wrap (matches NumSharp int cast)
        base = np.array(_INT_POOL, dtype=np.int64).astype(dt)
    if len(base) < n:
        reps = (n + len(base) - 1) // len(base)
        base = np.tile(base, reps)
    return base[:n].copy()


def _cbase(shape, dt):
    """Fresh C-contiguous base of `shape` filled deterministically from the pools."""
    n = int(np.prod(shape)) if len(shape) else 1
    return np.ascontiguousarray(_fill(n, dt).reshape(shape))


# ---------------------------------------------------------------------------
# Layout registry
# ---------------------------------------------------------------------------
LAYOUTS = {}


def _layout(name):
    def deco(fn):
        LAYOUTS[name] = fn
        return fn
    return deco


# --- contiguous baselines -------------------------------------------------
@_layout("c_contiguous_1d")
def _(dt):
    b = _cbase((8,), dt)
    return b, b


@_layout("c_contiguous_2d")
def _(dt):
    b = _cbase((4, 5), dt)
    return b, b


@_layout("c_contiguous_3d")
def _(dt):
    b = _cbase((2, 3, 4), dt)
    return b, b


@_layout("f_contiguous_2d")
def _(dt):
    # C-contig (5,4) transposed -> (4,5) F-contiguous, same buffer, offset 0.
    b = _cbase((5, 4), dt)
    return b, b.T


@_layout("transposed_3d")
def _(dt):
    b = _cbase((2, 3, 4), dt)
    return b, b.transpose(2, 0, 1)


# --- strided / negative-stride / offset -----------------------------------
@_layout("strided_step2_1d")
def _(dt):
    b = _cbase((16,), dt)
    return b, b[::2]


@_layout("negstride_1d")
def _(dt):
    b = _cbase((8,), dt)
    return b, b[::-1]


@_layout("simple_slice_offset_1d")
def _(dt):
    b = _cbase((10,), dt)
    return b, b[2:7]


@_layout("negstride_2d_offset")
def _(dt):
    b = _cbase((4, 5), dt)
    return b, b[::-1, ::-1]


@_layout("strided_2d_cols")
def _(dt):
    b = _cbase((4, 6), dt)
    return b, b[:, ::2]


# --- broadcast (stride-0) -------------------------------------------------
@_layout("broadcast_1d_to_2d")
def _(dt):
    b = _cbase((5,), dt)
    return b, np.broadcast_to(b, (4, 5))


@_layout("broadcast_row_partial")
def _(dt):
    b = _cbase((1, 5), dt)
    return b, np.broadcast_to(b, (4, 5))


# --- degenerate shapes ----------------------------------------------------
@_layout("scalar_0d")
def _(dt):
    b = _fill(1, dt).reshape(())
    return b, b


@_layout("one_element_1d")
def _(dt):
    b = _cbase((1,), dt)
    return b, b


@_layout("empty_2d")
def _(dt):
    b = np.zeros((0, 3), dtype=dt)
    return b, b


@_layout("highrank_5d")
def _(dt):
    b = _cbase((2, 1, 3, 1, 2), dt)
    return b, b


# --- additional distinct single-array memory descriptors --------------------
@_layout("f_contiguous_3d")
def _(dt):
    b = _cbase((4, 3, 2), dt)
    return b, b.transpose(2, 1, 0)  # F-contiguous (2,3,4)


@_layout("transposed_2d")
def _(dt):
    b = _cbase((3, 5), dt)
    return b, b.T


@_layout("strided_outer_2d")
def _(dt):
    b = _cbase((8, 3), dt)
    return b, b[::2, :]  # outer strided, inner contiguous


@_layout("sliced_composed")
def _(dt):
    # offset (from slice) combined with a transpose -> non-trivial strides + offset.
    b = _cbase((6, 4), dt)
    return b, b[1:5].T


@_layout("scalar_broadcast")
def _(dt):
    # all strides zero with dim > 1 (IsScalarBroadcast); buffer is a single element.
    b = _fill(1, dt).reshape(())
    return b, np.broadcast_to(b, (3, 4))


@_layout("zerod_from_index")
def _(dt):
    # Genuine 0-D VIEW into the buffer at a non-zero offset. (Note: b[1,2,3] returns a
    # numpy SCALAR copy, not a view, so we slice-then-reshape to keep it a view.)
    b = _cbase((2, 3, 4), dt)
    return b, b.reshape(-1)[23:24].reshape(())


@_layout("singleton_dim_3d")
def _(dt):
    b = _cbase((4, 1, 5), dt)
    return b, b


@_layout("newaxis_inserted")
def _(dt):
    b = _cbase((6,), dt)
    return b, b[None, :]


@_layout("empty_composed")
def _(dt):
    b = np.zeros((0, 3), dtype=dt)
    return b, b[::2, :]


@_layout("reshape_view_2d")
def _(dt):
    # contiguous reshape returns a view (same buffer, recomputed strides).
    b = _cbase((24,), dt)
    return b, b.reshape(4, 6)


def describe(base, view):
    """Serialize a (base, view) pair into the corpus operand descriptor."""
    itemsize = base.itemsize
    base_ptr = base.__array_interface__["data"][0]
    view_ptr = view.__array_interface__["data"][0]
    offset_elem = (view_ptr - base_ptr) // itemsize
    strides_elem = [int(s // itemsize) for s in view.strides]

    # Self-validation: the operand MUST be a view into base's buffer, never a copy/scalar.
    # Verify the whole addressed range lies inside [0, base.size). A garbage offset (e.g. a
    # numpy scalar's foreign pointer) trips this immediately at generation time.
    if view.size > 0:
        lo = offset_elem + sum(min(0, s) * (d - 1) for s, d in zip(strides_elem, view.shape))
        hi = offset_elem + sum(max(0, s) * (d - 1) for s, d in zip(strides_elem, view.shape))
        if not (0 <= lo and hi < base.size):
            raise ValueError(
                f"layout produced a non-view operand: offset={offset_elem} addressed=[{lo},{hi}] "
                f"base.size={base.size}; shape={view.shape} strides={strides_elem}")
    return {
        "dtype": view.dtype.name,
        "shape": [int(d) for d in view.shape],
        "strides": strides_elem,
        "offset": int(offset_elem),
        "bufferSize": int(base.size),
        "buffer": base.tobytes().hex(),
    }


# ---------------------------------------------------------------------------
# Pairwise layouts for binary-op cases. Each builder takes (dtA, dtB) and returns
# (baseA, viewA, baseB, viewB). Operands are emitted at their NATURAL shapes; the op
# performs any broadcasting (mirroring how NumPy and NumSharp broadcast at runtime).
# ---------------------------------------------------------------------------
PAIR_LAYOUTS = {}


def _pair(name):
    def deco(fn):
        PAIR_LAYOUTS[name] = fn
        return fn
    return deco


@_pair("pp_contig_contig")        # SimdFull
def _(da, db):
    a = _cbase((4, 5), da); b = _cbase((4, 5), db)
    return a, a, b, b


@_pair("pp_contig_fortran")       # one F-contiguous operand
def _(da, db):
    a = _cbase((4, 5), da); b = _cbase((5, 4), db)
    return a, a, b, b.T


@_pair("pp_contig_strided")       # SimdChunk: inner contig, outer strided on B
def _(da, db):
    a = _cbase((4, 5), da); b = _cbase((4, 10), db)
    return a, a, b, b[:, ::2]


@_pair("pp_strided_strided")      # General: both strided
def _(da, db):
    a = _cbase((4, 10), da); b = _cbase((4, 10), db)
    return a, a[:, ::2], b, b[:, ::2]


@_pair("pp_scalar_right")         # SimdScalarRight: RHS 0-D
def _(da, db):
    a = _cbase((4, 5), da); b = _fill(1, db).reshape(())
    return a, a, b, b


@_pair("pp_scalar_left")          # SimdScalarLeft: LHS 0-D
def _(da, db):
    a = _fill(1, da).reshape(()); b = _cbase((4, 5), db)
    return a, a, b, b


@_pair("pp_broadcast_row")        # (4,5) op (5,) -> (4,5)
def _(da, db):
    a = _cbase((4, 5), da); b = _cbase((5,), db)
    return a, a, b, b


@_pair("pp_broadcast_col")        # (4,1) op (1,5) -> (4,5)
def _(da, db):
    a = _cbase((4, 1), da); b = _cbase((1, 5), db)
    return a, a, b, b


@_pair("pp_negstride_both")       # both reversed views
def _(da, db):
    a = _cbase((8,), da); b = _cbase((8,), db)
    return a, a[::-1], b, b[::-1]
