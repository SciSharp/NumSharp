#!/usr/bin/env python3
"""
gen_viewflags_oracle.py -- the array-OBJECT oracle: what NumPy 2.4.2 hands back, not only its values.

Every other differential tier records a result as (dtype, shape, C-order bytes). That contract cannot see
three things a NumPy port must still get right, and this generator gates them:

  1. VIEW versus COPY versus IDENTITY of the returned array -- ``np.ravel(c_array)`` is a view that writes
     through, ``np.asarray(a)`` IS ``a``, ``a.flatten()`` is a copy, ``np.add(a, b, out=o)`` returns ``o``.
     Recorded per operand as ``shares`` (``np.shares_memory(result, operand)``) and ``same``
     (``result is operand``).
  2. The FLAGS of the returned array -- C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA, WRITEABLE, ALIGNED -- and its
     byte STRIDES (where the elements lie: K-order results of F / permuted inputs, read-only views of
     read-only inputs, the read-only scalar a full reduction returns).
  3. SHAPE and ARRAY parameters in every form a caller can pass -- a Shape built from a view's layout, a
     ``long[]``, an ``int[]``, an ``int`` -- and secondary array parameters (indices, conditions, weights,
     sorters, bins, ...) in non-C layouts, where the existing tiers only ever passed C-contiguous arrays.

Tiers (each its own file under ../NumSharp.Tests.Oracle/Fuzz/corpus/, replayed by FuzzCorpusTests.ViewFlags.cs):

  view_flags.jsonl   op "viewflags", params.fn = the NumPy API. Views, copies, identities and fresh results of
                     194 APIs over 21 source layouts (owners in C and F order, transposed / permuted /
                     strided / reversed / offset views, read-only owners and their views, broadcast views,
                     0-d owners and 0-d views, empty arrays): sections V1-V10 and K1-K3 of gen_views.
  out_flags.jsonl    op "outflags": ``f(..., out=o)`` for every out-capable API NumSharp exposes. Slot 0 is the
                     returned array (its facet must say ``same[out] = true``), slot 1 the whole base buffer
                     behind ``o``. NumPy's refusals of an ``out`` it cannot use are recorded verbatim.
  shape_args.jsonl   op "shapearg": the shape-taking APIs (zeros / ones / empty / full / reshape /
                     ndarray.reshape / broadcast_to / resize / tile / indices / empty_like) with params.form
                     naming the C# form the replay passes (``Shape`` / ``view:<layout>`` / ``long[]`` /
                     ``int[]`` / ``int``). NumPy's answer does not depend on the form; NumSharp's must not either.
  array_args.jsonl   EXISTING op keys (take, compress, choose, searchsorted, digitize, isin, ...) whose
                     secondary array parameters are replayed in non-C layouts (F / transposed / strided /
                     reversed / offset / broadcast / 0-d / read-only), with the result facet attached.

Case schema (the ordinary FuzzCorpus schema, plus):

  operand.own    true when NumPy's operand OWNS its buffer (``base is None``). The buffer hex is then the
                 owner's MEMORY-order bytes and (strides, offset) its physical layout; the C# replay builds an
                 owning array over them instead of a view (FuzzCorpus.Reconstruct). Absent means a view.
  expected.mem   {"flags": [C, F, OWNDATA, WRITEABLE, ALIGNED], "strides": [bytes...], "shares": [per operand],
                  "same": [per operand], "scalar": true?} -- strides of an extent-1 axis and of an empty result
                 are written as 0 (NumPy assigns those per allocation path and no flag reads them). "scalar" marks a
                 NumPy SCALAR result (np.generic), whose flags are a scalar's (read-only, owning).
  expected.xshare / expected.xsame   (tuple results) the slot pairs [i, j] that share memory / are one object.

Carving (house rule: real product bugs are carved + pinned, never registry-excused). ``_gap_of`` names, per
case, the documented gap NumSharp is known to have there (GAPS). A carved case is written to
``gaps/<tier>_gaps.jsonl`` with params-free ``gap`` id instead of the gate file; FuzzCorpusTests' [OpenBugs]
replay holds every carved case to NumPy and fails while the gap is open. ``--uncarved DIR`` writes the whole
matrix (the way the carve rules were drawn and how to re-probe after a fix).

Run (needs numpy==2.4.2):
    python test/oracle/gen_viewflags_oracle.py [all|views|out|shapes|arrays] [--uncarved DIR]
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.normpath(os.path.join(HERE, "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus"))
GAP_DIR = os.path.join(CORPUS, "gaps")

ALL_DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
              "int64", "uint64", "float16", "float32", "float64", "complex128"]


# =============================================================================================================
# Values and sources
# =============================================================================================================

def _vals(n, dt):
    """Deterministic small values for an n-element array of dtype ``dt``.

    The tiers gate object semantics, not arithmetic edges, so the values are small integers (with zeros, so
    nonzero / trim_zeros / count paths see both outcomes) that every dtype represents exactly: a value
    difference can then only mean a wrong element was read, never a rounding question.

    Parameters:
        n: element count (0 allowed).
        dt: a NumPy dtype name.

    Returns:
        A 1-D C-contiguous array of ``n`` elements.
    """
    dt = np.dtype(dt)
    base = (np.arange(n) % 5) - 1                                   # -1 0 1 2 3 -1 0 ...
    if dt.kind == "b":
        return (np.arange(n) % 3 != 1).astype(dt)
    if dt.kind == "u":
        return (np.arange(n) % 5).astype(dt)                         # no negative wrap: values stay small
    if dt.kind == "c":
        return (base + 1j * ((np.arange(n) % 3) - 1)).astype(dt)
    return base.astype(dt)


def _owner(shape, dt, order="C", readonly=False):
    """A fresh array that OWNS its buffer, filled with ``_vals`` in C logical order.

    Parameters:
        shape: the array's shape.
        dt: the dtype name.
        order: 'C' or 'F' -- the memory order of the owner.
        readonly: clear WRITEABLE after filling (NumPy's ``setflags(write=False)``).

    Returns:
        The owning array (``base is None``).
    """
    n = int(np.prod(shape)) if len(shape) else 1
    arr = np.empty(shape, dtype=dt, order=order)
    arr[...] = _vals(n, dt).reshape(shape)
    if readonly:
        arr.setflags(write=False)
    return arr


def _off_view(dt):
    """A (3,4) C-contiguous view at element offset 4 of a 20-element owner, the window holding ``_vals(12)``.

    The owner's elements OUTSIDE the window hold the sentinel 7 (True for bool), a value no window holds: a kernel
    that ignores the view's offset and reads the owner from its start then sees a 7, so an order-free result (a
    sum, a unique set, a membership test) shows the misread too. With the periodic ``_vals`` fill the shifted read
    returned the same value set and np.unique of such a view passed while reading the wrong elements.

    Parameters:
        dt: the dtype name.

    Returns:
        The view (its base is the 20-element owner).
    """
    b = np.empty((20,), dtype=dt)
    b[...] = np.array(7).astype(dt)
    b[4:16] = _vals(12, dt)
    return b[4:16].reshape(3, 4)


# Source layouts: name -> (min ndim the op needs is checked by the op itself; NumPy errors are skipped)
# builder(dt) -> ndarray. Every builder returns either an OWNER or a VIEW whose ultimate owner is C- or F-
# contiguous, which is what _desc can serialize as raw memory.
SOURCES = {
    "own_c_1d": lambda dt: _owner((6,), dt),
    "own_c_2d": lambda dt: _owner((3, 4), dt),
    "own_c_3d": lambda dt: _owner((2, 3, 4), dt),
    "own_f_2d": lambda dt: _owner((3, 4), dt, order="F"),
    "own_f_3d": lambda dt: _owner((2, 3, 4), dt, order="F"),
    "view_t_2d": lambda dt: _owner((4, 3), dt).T,                          # F-contiguous view, not an owner
    "view_perm_3d": lambda dt: _owner((3, 4, 2), dt).transpose(2, 0, 1),   # permuted (2,3,4)
    "view_step_1d": lambda dt: _owner((12,), dt)[::2],
    "view_step_2d": lambda dt: _owner((3, 8), dt)[:, ::2],
    "view_rev_1d": lambda dt: _owner((6,), dt)[::-1],
    "view_rev_2d": lambda dt: _owner((3, 4), dt)[::-1, ::-1],
    "view_off_2d": lambda dt: _off_view(dt),                               # C-contiguous view at an offset
    "own_ro_2d": lambda dt: _owner((3, 4), dt, readonly=True),
    "view_ro_2d": lambda dt: _owner((3, 4), dt, readonly=True)[:, 1:3],    # view of a read-only owner
    "view_bcast_2d": lambda dt: np.broadcast_to(_owner((4,), dt), (3, 4)),  # stride 0, read-only
    "view_bsame_2d": lambda dt: np.broadcast_to(_owner((3, 4), dt), (3, 4)),  # read-only, C strides
    "own_0d": lambda dt: _owner((), dt),
    "view_0d": lambda dt: _owner((24,), dt)[23:24].reshape(()),
    "own_1x1": lambda dt: _owner((1, 1), dt),
    "own_unit_3d": lambda dt: _owner((2, 1, 3), dt),
    "own_empty_2d": lambda dt: _owner((0, 3), dt),
}

# The subsets the sections iterate.
ALL_SOURCES = list(SOURCES)
ND2_SOURCES = ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "own_ro_2d",
               "view_ro_2d", "view_bcast_2d", "view_bsame_2d", "own_1x1", "own_empty_2d"]
ND3_SOURCES = ["own_c_3d", "own_f_3d", "view_perm_3d", "own_unit_3d"]
ND1_SOURCES = ["own_c_1d", "view_step_1d", "view_rev_1d"]


def _ultimate_owner(arr):
    """Walk ``.base`` to the array that owns the memory ``arr`` addresses.

    Parameters:
        arr: an ndarray.

    Returns:
        The owning ndarray.

    Raises:
        ValueError: the chain ends in something that is not an owning ndarray (a buffer object), which the
            corpus cannot serialize.
    """
    o = arr
    while not (isinstance(o, np.ndarray) and o.flags.owndata):
        o = getattr(o, "base", None)
        if o is None:
            raise ValueError("operand does not reach an owning ndarray")
    return o


def _desc(arr):
    """Serialize one operand: an owner as its own memory, a view as its owner's memory plus its layout.

    The buffer is the owner's MEMORY-order bytes (``tobytes(order='A')`` of a C- or F-contiguous owner), the
    strides and offset are in elements and physical. ``own`` marks an owner, ``writeable`` a read-only array --
    both only when set, so the descriptor of a writeable view reads exactly like the older corpus rows.

    Parameters:
        arr: the operand (an owner, or a view of a C- or F-contiguous owner).

    Returns:
        The operand descriptor dict.

    Raises:
        ValueError: the owner is neither C- nor F-contiguous, or the view reaches outside it.
    """
    owner = _ultimate_owner(arr)
    if not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        raise ValueError("owner is not contiguous")
    isz = arr.itemsize
    if arr.size == 0:
        # An empty operand is rebuilt as a fresh owning array of its shape (FuzzCorpus.Reconstruct); only empty
        # OWNERS are used as sources, so that is exactly NumPy's operand.
        d = {"dtype": arr.dtype.name, "shape": [int(x) for x in arr.shape],
             "strides": [0] * arr.ndim, "offset": 0, "bufferSize": 0, "buffer": ""}
        d["own"] = True
        return d
    off_bytes = arr.__array_interface__["data"][0] - owner.__array_interface__["data"][0]
    strides = [int(s // isz) for s in arr.strides]
    offset = off_bytes // isz
    lo = offset + sum(min(0, s) * (dd - 1) for s, dd in zip(strides, arr.shape))
    hi = offset + sum(max(0, s) * (dd - 1) for s, dd in zip(strides, arr.shape))
    if not (0 <= lo and hi < owner.size):
        raise ValueError(f"view reaches outside its owner: [{lo},{hi}] of {owner.size}")
    d = {"dtype": arr.dtype.name, "shape": [int(x) for x in arr.shape], "strides": strides,
         "offset": int(offset), "bufferSize": int(owner.size), "buffer": owner.tobytes(order="A").hex()}
    if arr is owner:
        d["own"] = True
    if not arr.flags.writeable:
        d["writeable"] = False
    return d


# =============================================================================================================
# Results
# =============================================================================================================

def _value(r):
    """The ordinary (dtype, shape, C-order bytes) record of one result.

    Parameters:
        r: an ndarray or a NumPy scalar.

    Returns:
        The expected dict without ``mem``.
    """
    a = np.asarray(r)
    return {"dtype": a.dtype.name, "shape": [int(x) for x in a.shape],
            "buffer": np.ascontiguousarray(a).tobytes().hex()}


def _mem(r, operands):
    """The memory facet of one result against the case's operands.

    Parameters:
        r: the result (an ndarray or a NumPy scalar).
        operands: the operand arrays, in descriptor order.

    Returns:
        {"flags", "strides", "shares", "same"} plus "scalar": true for a NumPy scalar.
    """
    f = r.flags
    flags = [bool(f.c_contiguous), bool(f.f_contiguous), bool(f.owndata), bool(f.writeable), bool(f.aligned)]
    if isinstance(r, np.generic):
        return {"flags": flags, "strides": [], "shares": [False] * len(operands),
                "same": [False] * len(operands), "scalar": True}
    strides = [0 if (r.size == 0 or d <= 1) else int(s) for s, d in zip(r.strides, r.shape)]
    shares = [bool(np.shares_memory(r, o)) if isinstance(o, np.ndarray) else False for o in operands]
    same = [r is o for o in operands]
    return {"flags": flags, "strides": strides, "shares": shares, "same": same}


def _result_expected(r, operands):
    """The expected record of a single-array result: values plus the memory facet."""
    e = _value(r)
    e["mem"] = _mem(r, operands)
    return e


def _tuple_result_expected(slots, operands):
    """The expected record of a tuple result: every slot with its facet, plus cross-slot sharing / identity.

    Parameters:
        slots: the result tuple's arrays, in NumPy's order.
        operands: the operand arrays.

    Returns:
        {"kind": "tuple", "slots": [...], "xshare": [[i, j], ...], "xsame": [[i, j], ...]}.
    """
    rec = {"kind": "tuple", "slots": [_result_expected(s, operands) for s in slots]}
    xshare, xsame = [], []
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            si, sj = slots[i], slots[j]
            if si is sj:
                xsame.append([i, j])
            if isinstance(si, np.ndarray) and isinstance(sj, np.ndarray) and np.shares_memory(si, sj):
                xshare.append([i, j])
    rec["xshare"] = xshare
    rec["xsame"] = xsame
    return rec


def _exc(e):
    """NumPy's exception, verbatim: the Python class name and ``str(e)``."""
    return {"type": type(e).__name__, "text": str(e)}


def _case(cid, op, params, operands, expected, layout, valueclass):
    """One corpus case in the shared schema (see the module docstring)."""
    return {"id": cid, "op": op, "params": params, "operands": operands, "expected": expected,
            "layout": layout, "valueclass": valueclass}


def _error(cid, op, params, operands, exc, layout, kind=None):
    """One error case: NumPy raised; the replay holds NumSharp to the type and the verbatim text."""
    expected = {"kind": kind} if kind else {}
    return {"id": cid, "op": op, "params": params, "operands": operands, "expected": expected,
            "expects_throw": True, "error": _exc(exc), "layout": layout, "valueclass": "error"}


# =============================================================================================================
# Section V + K: returned arrays (op "viewflags")
# =============================================================================================================

def _ch(p, key, default):
    """A one-character order parameter, or the default."""
    return p.get(key, default)


# fn -> (callable(ops, params) -> result, expected kind). Every entry has a twin in OpRegistry.ViewFlags.cs
# (ApplyViewFlags / ApplyViewFlagsTuple), which must make the SAME NumPy call in NumSharp's spelling.
def _view_fns():
    """The fn table of the viewflags tier: the NumPy call each fn names (twinned by the C# dispatch)."""
    t = {}

    def one(name, f):
        """Register ``f(ops, params)`` under ``name`` as a single-array fn (its result gets one facet)."""
        t[name] = (f, "array")

    def tup(name, f):
        """Register ``f(ops, params)`` under ``name`` as a tuple fn (a facet per slot plus the cross-slot pairs)."""
        t[name] = (f, "tuple")

    # ---- reshaping views ----
    one("ravel", lambda o, p: np.ravel(o[0], order=p.get("order", "C")))
    one("ndarray.ravel", lambda o, p: o[0].ravel(order=p.get("order", "C")))
    one("ndarray.flatten", lambda o, p: o[0].flatten(order=p.get("order", "C")))
    one("reshape", lambda o, p: np.reshape(o[0], tuple(p["shape"])))
    one("ndarray.reshape", lambda o, p: o[0].reshape(tuple(p["shape"]), order=p.get("order", "C")))
    one("squeeze", lambda o, p: np.squeeze(o[0], axis=p.get("axis")))
    one("ndarray.squeeze", lambda o, p: o[0].squeeze(axis=p.get("axis")))
    one("expand_dims", lambda o, p: np.expand_dims(o[0], p["axis"]))
    one("atleast_1d", lambda o, p: np.atleast_1d(o[0]))
    one("atleast_2d", lambda o, p: np.atleast_2d(o[0]))
    one("atleast_3d", lambda o, p: np.atleast_3d(o[0]))
    one("broadcast_to", lambda o, p: np.broadcast_to(o[0], tuple(p["shape"])))
    # ---- axis permutations ----
    one("transpose", lambda o, p: np.transpose(o[0], axes=p.get("axes")))
    one("ndarray.transpose", lambda o, p: o[0].transpose(*(p["axes"],) if "axes" in p else ()))
    one("ndarray.T", lambda o, p: o[0].T)
    one("ndarray.mT", lambda o, p: o[0].mT)
    one("matrix_transpose", lambda o, p: np.matrix_transpose(o[0]))
    one("permute_dims", lambda o, p: np.permute_dims(o[0], axes=p.get("axes")))
    one("swapaxes", lambda o, p: np.swapaxes(o[0], p["a1"], p["a2"]))
    one("ndarray.swapaxes", lambda o, p: o[0].swapaxes(p["a1"], p["a2"]))
    one("moveaxis", lambda o, p: np.moveaxis(o[0], p["src"], p["dst"]))
    one("rollaxis", lambda o, p: np.rollaxis(o[0], p["axis"], p["start"]))
    one("flip", lambda o, p: np.flip(o[0], axis=(tuple(p["axes"]) if "axes" in p else p.get("axis"))))
    one("fliplr", lambda o, p: np.fliplr(o[0]))
    one("flipud", lambda o, p: np.flipud(o[0]))
    one("rot90", lambda o, p: np.rot90(o[0], p["k"]))
    # ---- diagonal / complex parts ----
    one("diagonal", lambda o, p: np.diagonal(o[0], p["offset"], p["axis1"], p["axis2"]))
    one("ndarray.diagonal", lambda o, p: o[0].diagonal(p["offset"], p["axis1"], p["axis2"]))
    one("diag", lambda o, p: np.diag(o[0], p["k"]))
    one("real", lambda o, p: np.real(o[0]))
    one("imag", lambda o, p: np.imag(o[0]))
    one("ndarray.real", lambda o, p: o[0].real)
    one("ndarray.imag", lambda o, p: o[0].imag)
    one("conj", lambda o, p: np.conj(o[0]))
    one("conjugate", lambda o, p: np.conjugate(o[0]))
    one("ndarray.conj", lambda o, p: o[0].conj())
    one("ndarray.conjugate", lambda o, p: o[0].conjugate())
    one("trim_zeros", lambda o, p: np.trim_zeros(o[0], p["trim"]))
    # ---- conversions: identity or copy ----
    one("asarray", lambda o, p: np.asarray(o[0], dtype=p.get("dtype"), order=p.get("order"), copy=p.get("copy")))
    one("asanyarray", lambda o, p: np.asanyarray(o[0], dtype=p.get("dtype")))
    one("ascontiguousarray", lambda o, p: np.ascontiguousarray(o[0], dtype=p.get("dtype")))
    one("asfortranarray", lambda o, p: np.asfortranarray(o[0], dtype=p.get("dtype")))
    one("asmatrix", lambda o, p: np.asarray(np.asmatrix(o[0])))    # NumSharp's asmatrix is a 2-D ndarray view
    one("array", lambda o, p: np.array(o[0], copy=p["copy"]))
    one("copy", lambda o, p: np.copy(o[0], order=p.get("order", "K")))
    one("ndarray.copy", lambda o, p: o[0].copy(order=p.get("order", "C")))
    one("require", lambda o, p: np.require(o[0], requirements=p["requirements"]))
    one("asarray_chkfinite", lambda o, p: np.asarray_chkfinite(o[0]))
    one("ndarray.astype", lambda o, p: o[0].astype(p["dtype"], order=p.get("order", "K"), copy=p.get("copy", True)))
    one("ndarray.view", lambda o, p: o[0].view(p["dtype"]) if "dtype" in p else o[0].view())
    one("ndarray.getfield", lambda o, p: o[0].getfield(p["dtype"], p["offset"]))
    one("ndarray.byteswap", lambda o, p: o[0].byteswap())
    one("nan_to_num", lambda o, p: np.nan_to_num(o[0], copy=p["copy"]))
    one("real_if_close", lambda o, p: np.real_if_close(o[0], tol=p["tol"]))
    # ---- indexing ----
    one("getitem", lambda o, p: o[0][_index_key(p["index"], o)])
    # ---- creation (no operand) ----
    one("linspace", lambda o, p: np.linspace(p["start"], p["stop"], p["num"], endpoint=p["endpoint"],
                                             dtype=p.get("dtype")))
    one("logspace", lambda o, p: np.logspace(p["start"], p["stop"], p["num"], endpoint=p["endpoint"],
                                             base=p["base"], dtype=p.get("dtype")))
    one("geomspace", lambda o, p: np.geomspace(p["start"], p["stop"], p["num"], endpoint=p["endpoint"],
                                               dtype=p.get("dtype")))
    one("arange", lambda o, p: np.arange(p["start"], p["stop"], p["step"], dtype=p.get("dtype")))
    # ---- einsum ----
    one("einsum", lambda o, p: np.einsum(p["subscripts"], *o))
    # ---- fresh results of the copy family ----
    one("take", lambda o, p: np.take(o[0], o[1], axis=p.get("axis")))
    one("compress", lambda o, p: np.compress(o[0], o[1], axis=p.get("axis")))
    one("repeat", lambda o, p: np.repeat(o[0], p["repeats"], axis=p.get("axis")))
    one("tile", lambda o, p: np.tile(o[0], tuple(p["reps"])))
    one("resize", lambda o, p: np.resize(o[0], tuple(p["shape"])))
    one("roll", lambda o, p: np.roll(o[0], p["shift"], axis=p.get("axis")))
    one("delete", lambda o, p: np.delete(o[0], p["obj"], axis=p.get("axis")))
    one("insert", lambda o, p: np.insert(o[0], p["obj"], o[1], axis=p.get("axis")))
    one("append", lambda o, p: np.append(o[0], o[1], axis=p.get("axis")))
    one("sort", lambda o, p: np.sort(o[0], axis=p.get("axis", -1)))
    one("argsort", lambda o, p: np.argsort(o[0], axis=p.get("axis", -1), kind="stable"))
    one("unique", lambda o, p: np.unique(o[0]))
    one("pad", lambda o, p: np.pad(o[0], p["width"]))
    one("argwhere", lambda o, p: np.argwhere(o[0]))
    one("flatnonzero", lambda o, p: np.flatnonzero(o[0]))
    one("diff", lambda o, p: np.diff(o[0], n=p["n"], axis=p["axis"]))
    one("cumsum", lambda o, p: np.cumsum(o[0], axis=p.get("axis")))
    one("cumprod", lambda o, p: np.cumprod(o[0], axis=p.get("axis")))
    one("concatenate", lambda o, p: np.concatenate(list(o), axis=p["axis"]))
    one("stack", lambda o, p: np.stack(list(o), axis=p["axis"]))
    one("hstack", lambda o, p: np.hstack(list(o)))
    one("vstack", lambda o, p: np.vstack(list(o)))
    one("where3", lambda o, p: np.where(o[0], o[1], o[2]))
    one("clip", lambda o, p: np.clip(o[0], o[1], o[2]))
    one("round", lambda o, p: np.round(o[0], p["decimals"]))
    one("matmul", lambda o, p: np.matmul(o[0], o[1]))
    one("dot", lambda o, p: np.dot(o[0], o[1]))
    one("outer", lambda o, p: np.outer(o[0], o[1]))
    one("kron", lambda o, p: np.kron(o[0], o[1]))
    one("inner", lambda o, p: np.inner(o[0], o[1]))
    # ---- like-creators (K order by default) ----
    one("zeros_like", lambda o, p: np.zeros_like(o[0], order=p.get("order", "K")))
    one("ones_like", lambda o, p: np.ones_like(o[0], order=p.get("order", "K")))
    one("full_like", lambda o, p: np.full_like(o[0], 2, order=p.get("order", "K")))
    one("empty_like", lambda o, p: _filled0(np.empty_like(o[0], order=p.get("order", "K"))))
    # ---- ufuncs (fresh results; the layout NpyIter gives them) ----
    for u in ["negative", "sqrt", "absolute", "isnan", "square", "logical_not", "sign", "floor", "positive",
              "invert", "exp"]:
        one(u, (lambda name: (lambda o, p: getattr(np, name)(o[0])))(u))
    for u in ["add", "subtract", "multiply", "maximum", "equal", "less", "logical_and", "bitwise_and", "power",
              "arctan2", "floor_divide"]:
        one(u, (lambda name: (lambda o, p: getattr(np, name)(o[0], o[1])))(u))
    # ---- reductions ----
    for r in ["sum", "prod", "mean", "std", "var", "max", "min", "all", "any", "ptp", "median", "average",
              "nansum", "nanmax"]:
        one(r, (lambda name: (lambda o, p: _reduce(name, o[0], p)))(r))
    for r in ["argmax", "argmin"]:
        one(r, (lambda name: (lambda o, p: getattr(np, name)(o[0], axis=p["axis"], keepdims=p.get("keepdims", False))))(r))
    # ---- tuple results ----
    tup("split", lambda o, p: np.split(o[0], p["sections"], axis=p["axis"]))
    tup("array_split", lambda o, p: np.array_split(o[0], p["sections"], axis=p["axis"]))
    tup("hsplit", lambda o, p: np.hsplit(o[0], p["sections"]))
    tup("vsplit", lambda o, p: np.vsplit(o[0], p["sections"]))
    tup("dsplit", lambda o, p: np.dsplit(o[0], p["sections"]))
    tup("unstack", lambda o, p: np.unstack(o[0], axis=p["axis"]))
    tup("ix_", lambda o, p: np.ix_(*o))
    tup("meshgrid", lambda o, p: np.meshgrid(*o, indexing=p["indexing"], sparse=p["sparse"], copy=p["copy"]))
    tup("broadcast_arrays", lambda o, p: np.broadcast_arrays(*o))
    tup("nonzero", lambda o, p: np.nonzero(o[0]))
    tup("where1", lambda o, p: np.where(o[0]))
    tup("unravel_index", lambda o, p: np.unravel_index(o[0], tuple(p["shape"]), order=p["order"]))
    # ---- V10: triangles, partitions, nan-reductions, the FFT family, creators, index tuples, ndarray methods ----
    one("tril", lambda o, p: np.tril(o[0], p["k"]))
    one("triu", lambda o, p: np.triu(o[0], p["k"]))
    one("diagflat", lambda o, p: np.diagflat(o[0], p["k"]))
    one("partition", lambda o, p: np.partition(o[0], p["kth"], axis=p["axis"]))
    one("argpartition", lambda o, p: np.argpartition(o[0], p["kth"], axis=p["axis"]))
    for r in ["nanmean", "nanstd", "nanvar", "nanmin", "nanprod", "nanargmax", "nanargmin", "nanmedian"]:
        one(r, (lambda name: (lambda o, p: getattr(np, name)(o[0], axis=p.get("axis"),
                                                              keepdims=p.get("keepdims", False))))(r))
    one("count_nonzero", lambda o, p: np.count_nonzero(o[0], axis=p["axis"], keepdims=p.get("keepdims", False)))
    for f in ["fft", "ifft", "rfft", "irfft", "hfft", "ihfft"]:
        one("fft." + f, (lambda name: (lambda o, p: getattr(np.fft, name)(o[0], axis=p["axis"])))(f))
    for f in ["fft2", "ifft2", "fftn", "ifftn", "rfft2", "rfftn", "irfft2", "irfftn"]:
        one("fft." + f, (lambda name: (lambda o, p: getattr(np.fft, name)(o[0])))(f))
    for f in ["fftshift", "ifftshift"]:
        one("fft." + f, (lambda name: (lambda o, p: getattr(np.fft, name)(o[0], axes=p.get("axes"))))(f))
    one("eye", lambda o, p: np.eye(p["N"], p.get("M"), p["k"], order=p.get("order", "C")))
    one("identity", lambda o, p: np.identity(p["n"]))
    one("tri", lambda o, p: np.tri(p["N"], p.get("M"), p["k"]))
    one("mgrid", lambda o, p: np.mgrid[tuple(_slice_of(s) for s in p["slices"])])
    tup("ogrid", lambda o, p: np.ogrid[tuple(_slice_of(s) for s in p["slices"])])
    tup("diag_indices", lambda o, p: np.diag_indices(p["n"], p["ndim"]))
    tup("tril_indices", lambda o, p: np.tril_indices(p["n"], p["k"]))
    tup("triu_indices", lambda o, p: np.triu_indices(p["n"], p["k"]))
    tup("ndarray.nonzero", lambda o, p: o[0].nonzero())
    for m in ["sum", "mean", "std", "var", "prod"]:
        one("ndarray." + m, (lambda name: (lambda o, p: getattr(o[0], name)(axis=p.get("axis"),
                                                                             keepdims=p.get("keepdims", False))))(m))
    for m in ["argmax", "argmin"]:
        one("ndarray." + m, (lambda name: (lambda o, p: getattr(o[0], name)(axis=p["axis"],
                                                                             keepdims=p.get("keepdims", False))))(m))
    one("ndarray.argsort", lambda o, p: o[0].argsort(axis=p["axis"], kind="stable"))
    one("ndarray.argpartition", lambda o, p: o[0].argpartition(p["kth"], axis=p["axis"]))
    one("ndarray.repeat", lambda o, p: o[0].repeat(p["repeats"], axis=p.get("axis")))
    one("ndarray.searchsorted", lambda o, p: o[0].searchsorted(o[1], side=p["side"]))
    one("ndarray.dot", lambda o, p: o[0].dot(o[1]))
    return t


def _slice_of(text):
    """A Python slice from the "start:stop[:step]" text the C# side hands np.mgrid / np.ogrid verbatim.

    Parameters:
        text: integer fields separated by colons; an empty field is None.

    Returns:
        The slice object.
    """
    parts = [int(x) if x else None for x in text.split(":")]
    return slice(*parts)


def _filled0(a):
    """``empty`` / ``empty_like`` results are uninitialized: both sides fill them with 0 before comparing."""
    a.fill(0)
    return a


def _reduce(name, a, p):
    """One reduction with the tier's (axis, keepdims) params; ``average`` has keepdims since NumPy 1.23."""
    f = getattr(np, name)
    return f(a, axis=p.get("axis"), keepdims=p.get("keepdims", False))


def _index_key(tokens, ops):
    """Rebuild a NumPy index tuple from the case's index tokens.

    Token forms: {"i": n} an integer, {"s": [start, stop, step]} a slice (nulls for None), {"e": 1} the
    Ellipsis, {"n": 1} np.newaxis, {"op": k} operand k (a fancy index or a boolean mask).

    Parameters:
        tokens: the token list.
        ops: the case's operand arrays.

    Returns:
        The index tuple.
    """
    key = []
    for t in tokens:
        if "i" in t:
            key.append(t["i"])
        elif "s" in t:
            key.append(slice(*t["s"]))
        elif "e" in t:
            key.append(Ellipsis)
        elif "n" in t:
            key.append(np.newaxis)
        elif "op" in t:
            key.append(ops[t["op"]])
        else:
            raise ValueError(t)
    return tuple(key)


VIEW_FNS = _view_fns()

UNARY_K = ["negative", "sqrt", "absolute", "isnan", "square", "logical_not", "sign", "floor", "positive", "exp"]
BINARY_K = ["add", "subtract", "multiply", "maximum", "equal", "less", "logical_and", "power", "arctan2",
            "floor_divide"]
REDUCE_K = ["sum", "prod", "mean", "std", "var", "max", "min", "all", "any", "ptp", "median", "average",
            "nansum", "nanmax", "argmax", "argmin"]

# Pair sources for the binary ufunc layouts: each side its own builder (and its own buffer).
PAIRS = [("own_c_2d", "own_c_2d"), ("own_f_2d", "own_f_2d"), ("own_c_2d", "own_f_2d"), ("own_f_2d", "own_c_2d"),
         ("view_t_2d", "view_t_2d"), ("own_f_3d", "own_f_3d"), ("view_perm_3d", "view_perm_3d"),
         ("view_step_2d", "view_step_2d"), ("view_rev_2d", "view_rev_2d"), ("view_off_2d", "view_off_2d"),
         ("view_bcast_2d", "own_f_2d"), ("own_f_2d", "view_bcast_2d"), ("own_0d", "own_f_2d"),
         ("own_f_2d", "own_0d"), ("own_0d", "own_0d"), ("own_ro_2d", "own_ro_2d"), ("own_empty_2d", "own_empty_2d"),
         ("own_c_1d", "view_rev_1d")]


def _ops_for(fn, sources, dt):
    """Build the operand arrays for one call from source names (each source a fresh owner or view)."""
    return [SOURCES[s](dt) for s in sources]


class _Emitter:
    """Collects the cases of one tier, numbering ids, carving the known gaps and skipping NumPy refusals.

    Attributes:
        tier: the tier name (selects the carve rules).
        op: the corpus op key every case of the tier carries.
        cases: the kept (gate) cases.
        gaps: the carved cases (each with its gap id).
        skipped: how many calls NumPy refused (value sections skip those; error sections record them).
    """

    def __init__(self, tier, op, carve):
        """An empty emitter.

        Parameters:
            tier: the tier name (views / out / shapes / arrays), which selects the carve rules of ``_gap_of``.
            op: the op key every case carries (None for the arrays tier, whose cases keep their own keys).
            carve: file the cases a rule names into ``gaps`` (False: keep every case, the ``--uncarved`` matrix).
        """
        self.tier = tier
        self.op = op
        self.carve = carve
        self.cases = []
        self.gaps = []
        self.skipped = 0
        self._n = 0

    def add(self, case):
        """File one finished case into the gate or, when a carve rule names it, the gaps file."""
        gap = _gap_of(self.tier, case) if self.carve else None
        if gap is not None:
            case = dict(case)
            case["gap"] = gap
            self.gaps.append(case)
        else:
            self.cases.append(case)

    def next_id(self, tag):
        """A unique, stable id: tier / tag / running number."""
        self._n += 1
        return f"{self.tier}/{tag}/{self._n}"


def _call_view(em, fn, params, ops, layout, dtype, record_errors=False):
    """Run one viewflags call in NumPy and record its value + facet (or skip / record its error).

    Parameters:
        em: the tier emitter.
        fn: the fn key (VIEW_FNS).
        params: the call's params (``fn`` is added here).
        ops: the operand arrays (already built).
        layout: the case's layout label.
        dtype: the dtype label used in the id.
        record_errors: record NumPy's refusal as an error case instead of skipping it.
    """
    f, kind = VIEW_FNS[fn]
    p = dict(params)
    p["fn"] = fn
    try:
        descs = [_desc(o) for o in ops]
    except ValueError:
        em.skipped += 1
        return
    cid = em.next_id(f"{fn}/{layout}/{dtype}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = f(ops, params)
    except Exception as e:  # noqa: BLE001 -- NumPy's refusal is data here
        if record_errors:
            em.add(_error(cid, em.op, p, descs, e, layout, kind="tuple" if kind == "tuple" else None))
        else:
            em.skipped += 1
        return
    if kind == "tuple":
        expected = _tuple_result_expected(list(r), ops)
    else:
        expected = _result_expected(r, ops)
    em.add(_case(cid, em.op, p, descs, expected, layout, "viewflags"))


def gen_views(carve=True):
    """The viewflags tier: returned views / copies / identities and the flags of fresh results.

    Parameters:
        carve: file the cases the carve rules name into the gaps list instead of the gate.

    Returns:
        (gate cases, carved cases).
    """
    em = _Emitter("views", "viewflags", carve)
    I64, F64, C128, B = "int64", "float64", "complex128", "bool"

    # -- V1: reshaping and permuting views over every source ----------------------------------------------
    for s in ALL_SOURCES:
        for order in [None, "C", "F", "A", "K"]:
            p = {} if order is None else {"order": order}
            _call_view(em, "ravel", p, _ops_for("ravel", [s], I64), s, I64)
            if order is not None:
                _call_view(em, "ndarray.ravel", p, _ops_for("ravel", [s], I64), s, I64)
                _call_view(em, "ndarray.flatten", p, _ops_for("ravel", [s], I64), s, I64)
        a = SOURCES[s](I64)
        n = a.size
        targets = {12: [(12,), (-1,), (4, 3), (2, 6), (2, -1), (3, 4)], 6: [(6,), (-1,), (3, 2), (2, 3)],
                   24: [(24,), (4, 6), (2, 3, 4), (6, -1), (4, 3, 2)], 1: [(1,), (), (1, 1)], 0: [(0,), (3, 0), (-1, 3)],
                   8: [(8,), (2, 4)]}.get(n, [(n,)])
        for tshape in targets:
            _call_view(em, "reshape", {"shape": list(tshape)}, _ops_for("reshape", [s], I64), s, I64)
            if -1 not in tshape:
                for order in ["C", "F", "A"]:
                    _call_view(em, "ndarray.reshape", {"shape": list(tshape), "order": order},
                               _ops_for("reshape", [s], I64), s, I64)
        nd = a.ndim
        _call_view(em, "transpose", {}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "ndarray.transpose", {}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "ndarray.T", {}, _ops_for("t", [s], I64), s, I64)
        if nd >= 2:
            axes = list(range(nd))[::-1][1:] + [nd - 1] if nd >= 3 else [1, 0]
            _call_view(em, "transpose", {"axes": axes}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "ndarray.transpose", {"axes": axes}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "permute_dims", {"axes": axes}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "ndarray.mT", {}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "matrix_transpose", {}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "swapaxes", {"a1": 0, "a2": -1}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "ndarray.swapaxes", {"a1": 0, "a2": -1}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "moveaxis", {"src": 0, "dst": -1}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "rollaxis", {"axis": -1, "start": 0}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "fliplr", {}, _ops_for("t", [s], I64), s, I64)
            for k in [1, 2, 3]:
                _call_view(em, "rot90", {"k": k}, _ops_for("t", [s], I64), s, I64)
            for off in [0, 1]:
                _call_view(em, "diagonal", {"offset": off, "axis1": 0, "axis2": 1}, _ops_for("t", [s], I64), s, I64)
                _call_view(em, "ndarray.diagonal", {"offset": off, "axis1": 0, "axis2": 1},
                           _ops_for("t", [s], I64), s, I64)
            if nd == 2:
                _call_view(em, "diag", {"k": 0}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "permute_dims", {}, _ops_for("t", [s], I64), s, I64)
        if nd >= 1:
            _call_view(em, "flipud", {}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "flip", {"axis": 0}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "expand_dims", {"axis": -1}, _ops_for("t", [s], I64), s, I64)
        if nd == 1:
            _call_view(em, "diag", {"k": 0}, _ops_for("t", [s], I64), s, I64)
            _call_view(em, "diag", {"k": 1}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "flip", {}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "expand_dims", {"axis": 0}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "squeeze", {}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "ndarray.squeeze", {}, _ops_for("t", [s], I64), s, I64)
        for ax, d in enumerate(a.shape):
            if d == 1:
                _call_view(em, "squeeze", {"axis": ax}, _ops_for("t", [s], I64), s, I64)
                _call_view(em, "ndarray.squeeze", {"axis": ax}, _ops_for("t", [s], I64), s, I64)
        for fn in ["atleast_1d", "atleast_2d", "atleast_3d"]:
            _call_view(em, fn, {}, _ops_for("t", [s], I64), s, I64)
        bshape = {0: [2, 3], 1: [2, 6], 2: [2] + list(a.shape), 3: [2] + list(a.shape)}[min(nd, 3)]
        _call_view(em, "broadcast_to", {"shape": bshape}, _ops_for("t", [s], I64), s, I64)
        _call_view(em, "broadcast_to", {"shape": list(a.shape)}, _ops_for("t", [s], I64), s, I64)

    # -- V2: complex parts and conjugates (identity for real arrays: a.real / a.conj()) ---------------------
    for dt in [F64, C128, I64]:
        for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "own_ro_2d",
                  "view_bcast_2d", "own_0d", "view_0d", "own_empty_2d"]:
            for fn in ["real", "imag", "ndarray.real", "ndarray.imag", "conj", "conjugate", "ndarray.conj",
                       "ndarray.conjugate"]:
                _call_view(em, fn, {}, _ops_for(fn, [s], dt), s, dt)

    # -- V3: conversions -- identity, view or copy ----------------------------------------------------------
    for s in ALL_SOURCES:
        for dt in [I64]:
            ops = lambda: _ops_for("conv", [s], dt)  # noqa: E731
            _call_view(em, "asarray", {}, ops(), s, dt)
            for order in ["C", "F", "A", "K"]:
                _call_view(em, "asarray", {"order": order}, ops(), s, dt)
            for cp in [True, False]:
                _call_view(em, "asarray", {"copy": cp}, ops(), s, dt, record_errors=not cp)
            _call_view(em, "asarray", {"dtype": "int64"}, ops(), s, dt)
            _call_view(em, "asarray", {"dtype": "float64"}, ops(), s, dt)
            _call_view(em, "asanyarray", {}, ops(), s, dt)
            _call_view(em, "asanyarray", {"dtype": "float64"}, ops(), s, dt)
            _call_view(em, "ascontiguousarray", {}, ops(), s, dt)
            _call_view(em, "ascontiguousarray", {"dtype": "float64"}, ops(), s, dt)
            _call_view(em, "asfortranarray", {}, ops(), s, dt)
            _call_view(em, "asfortranarray", {"dtype": "float64"}, ops(), s, dt)
            _call_view(em, "asmatrix", {}, ops(), s, dt)
            for cp in [True, False]:
                _call_view(em, "array", {"copy": cp}, ops(), s, dt)
            for order in [None, "C", "F", "A", "K"]:
                p = {} if order is None else {"order": order}
                _call_view(em, "copy", p, ops(), s, dt)
                _call_view(em, "ndarray.copy", p, ops(), s, dt)
            for req in [[], ["C"], ["F"], ["W"], ["O"], ["A"], ["C", "W"], ["F", "O"]]:
                _call_view(em, "require", {"requirements": req}, ops(), s, dt)
            _call_view(em, "asarray_chkfinite", {}, ops(), s, dt)
            for tgt, cp in [("int64", True), ("int64", False), ("float64", False), ("int32", True)]:
                for order in ["K", "C", "F"]:
                    _call_view(em, "ndarray.astype", {"dtype": tgt, "copy": cp, "order": order}, ops(), s, dt)
            _call_view(em, "ndarray.view", {}, ops(), s, dt)
            _call_view(em, "ndarray.view", {"dtype": "float64"}, ops(), s, dt)
            _call_view(em, "ndarray.view", {"dtype": "uint64"}, ops(), s, dt)
            _call_view(em, "ndarray.getfield", {"dtype": "int64", "offset": 0}, ops(), s, dt)
            _call_view(em, "ndarray.byteswap", {}, ops(), s, dt)
    for s in ["own_c_2d", "own_f_2d", "view_step_2d", "own_ro_2d", "own_0d"]:
        for cp in [True, False]:
            _call_view(em, "nan_to_num", {"copy": cp}, _ops_for("n", [s], F64), s, F64)
        for tol in [100.0]:
            _call_view(em, "real_if_close", {"tol": tol}, _ops_for("n", [s], C128), s, C128)
            _call_view(em, "real_if_close", {"tol": tol}, _ops_for("n", [s], F64), s, F64)

    # -- V4: the dtype sweep -- the flags record must not depend on the dtype -----------------------------------
    for dt in ALL_DTYPES:
        for s in ["own_c_2d", "own_f_2d", "view_step_2d", "view_bcast_2d", "own_0d"]:
            _call_view(em, "ravel", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "transpose", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "asarray", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "copy", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "ndarray.copy", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "negative" if dt != "bool" else "logical_not", {}, _ops_for("r", [s], dt), s, dt)
            _call_view(em, "sum", {"axis": 0} if s != "own_0d" else {}, _ops_for("r", [s], dt), s, dt)

    # -- V5: indexing -- basic slicing views, integer scalars, fancy and boolean copies ---------------------------
    keys_2d = [
        ("int", [{"i": 1}]), ("negint", [{"i": -1}]), ("slice", [{"s": [1, 3, None]}]),
        ("step", [{"s": [None, None, 2]}]), ("rev", [{"s": [None, None, -1]}]),
        ("ell_int", [{"e": 1}, {"i": 0}]), ("newaxis", [{"n": 1}]), ("slice_newaxis", [{"s": [None, None, None]}, {"n": 1}]),
        ("int_slice", [{"i": 0}, {"s": [1, None, None]}]), ("int_int", [{"i": 1}, {"i": 2}]),
        ("ell", [{"e": 1}]), ("empty_tuple", []),
    ]
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "own_ro_2d",
              "view_bcast_2d", "own_unit_3d", "own_c_3d"]:
        for tag, key in keys_2d:
            _call_view(em, "getitem", {"index": key}, _ops_for("g", [s], I64), s + ":" + tag, I64)
        # fancy / boolean (operand 1 is the index array)
        a = SOURCES[s](I64)
        idx = np.array([0, 1], dtype=np.int64)
        _call_view(em, "getitem", {"index": [{"op": 1}]}, [a, idx], s + ":fancy", I64)
        _call_view(em, "getitem", {"index": [{"s": [None, None, None]}, {"op": 1}]},
                   [SOURCES[s](I64), idx.copy()], s + ":slice_fancy", I64)
        mask = (np.arange(a.size).reshape(a.shape) % 2 == 0) if a.ndim else np.array(True)
        _call_view(em, "getitem", {"index": [{"op": 1}]}, [SOURCES[s](I64), np.array(mask)], s + ":mask", I64)
    for s in ["own_0d", "view_0d"]:
        for tag, key in [("ell", [{"e": 1}]), ("empty_tuple", []), ("newaxis", [{"n": 1}])]:
            _call_view(em, "getitem", {"index": key}, _ops_for("g", [s], I64), s + ":" + tag, I64)

    # -- V6: tuple results ----------------------------------------------------------------------------------------
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "own_ro_2d",
              "view_bcast_2d"]:
        for ax in [0, 1]:
            _call_view(em, "split", {"sections": 1, "axis": ax}, _ops_for("sp", [s], I64), s, I64)
            _call_view(em, "array_split", {"sections": 2, "axis": ax}, _ops_for("sp", [s], I64), s, I64)
            _call_view(em, "unstack", {"axis": ax}, _ops_for("sp", [s], I64), s, I64)
        _call_view(em, "hsplit", {"sections": 2}, _ops_for("sp", [s], I64), s, I64)
        _call_view(em, "vsplit", {"sections": 3}, _ops_for("sp", [s], I64), s, I64)
        _call_view(em, "nonzero", {}, _ops_for("nz", [s], I64), s, I64)
        _call_view(em, "where1", {}, _ops_for("nz", [s], I64), s, I64)
        _call_view(em, "argwhere", {}, _ops_for("nz", [s], I64), s, I64)
        _call_view(em, "flatnonzero", {}, _ops_for("nz", [s], I64), s, I64)
    for s in ["own_c_3d", "own_f_3d", "view_perm_3d"]:
        _call_view(em, "dsplit", {"sections": 2}, _ops_for("sp", [s], I64), s, I64)
        _call_view(em, "unstack", {"axis": 2}, _ops_for("sp", [s], I64), s, I64)
    for s1, s2 in [("own_c_1d", "own_c_1d"), ("view_step_1d", "view_rev_1d"), ("own_c_1d", "view_step_1d")]:
        _call_view(em, "ix_", {}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        _call_view(em, "broadcast_arrays", {}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        for copy in [True, False]:
            for sparse in [False, True]:
                for indexing in ["xy", "ij"]:
                    _call_view(em, "meshgrid", {"indexing": indexing, "sparse": sparse, "copy": copy},
                               [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
    # broadcast_arrays where the inputs really broadcast: a 2-D array against a 0-d, a broadcast view and a row.
    for s1, s2 in [("own_c_2d", "own_0d"), ("own_f_2d", "view_bcast_2d"), ("view_t_2d", "own_c_1d")]:
        second = SOURCES[s2](I64)
        if s2 == "own_c_1d":
            second = _owner((4,), I64)                       # a (4,) row against the (3,4) first operand
        _call_view(em, "broadcast_arrays", {}, [SOURCES[s1](I64), second], s1 + "+" + s2, I64)
    for s in ["own_c_1d", "view_step_1d", "view_rev_1d", "own_0d"]:
        for order in ["C", "F"]:
            _call_view(em, "unravel_index", {"shape": [2, 3], "order": order}, _ops_for("u", [s], "uint8"), s, "uint8")

    # -- V7: creation -- linspace's arange view, the owning casts ------------------------------------------------
    for dt in [None, "float32", "int32", "complex128"]:
        for ep in [True, False]:
            p = {"start": -1.0, "stop": 3.0, "num": 5, "endpoint": ep}
            if dt is not None:
                p["dtype"] = dt
            _call_view(em, "linspace", p, [], "none", dt or "default")
            q = dict(p, base=2.0)
            _call_view(em, "logspace", q, [], "none", dt or "default")
            # geomspace's complex128 domain is allclose, not byte-reproducible (the creation tier excludes it too).
            if dt != "complex128":
                g = {"start": 1.0, "stop": 16.0, "num": 5, "endpoint": ep}
                if dt is not None:
                    g["dtype"] = dt
                _call_view(em, "geomspace", g, [], "none", dt or "default")
        a = {"start": 0.0, "stop": 5.0, "step": 1.0}
        if dt is not None:
            a["dtype"] = dt
        _call_view(em, "arange", a, [], "none", dt or "default")

    # -- V8: einsum's view path and its fresh results ------------------------------------------------------------
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_off_2d", "own_ro_2d", "view_bcast_2d"]:
        for subs in ["ij->ij", "ij->ji", "ij->i", "ij"]:
            _call_view(em, "einsum", {"subscripts": subs}, _ops_for("e", [s], F64), s, F64)
    for s in ["own_1x1", "own_c_2d"]:
        sq = np.ascontiguousarray(np.arange(9.0).reshape(3, 3)) if s == "own_c_2d" else SOURCES[s](F64)
        _call_view(em, "einsum", {"subscripts": "ii->i"}, [sq], s + ":square", F64)
    for s in ["own_c_1d", "view_step_1d", "view_rev_1d"]:
        _call_view(em, "einsum", {"subscripts": "i->i"}, _ops_for("e", [s], F64), s, F64)
        _call_view(em, "einsum", {"subscripts": "i,i->i"}, [SOURCES[s](F64), SOURCES[s](F64)], s, F64)
        _call_view(em, "einsum", {"subscripts": "i,j->ij"}, [SOURCES[s](F64), SOURCES[s](F64)], s, F64)

    # -- V9: the copy family -- fresh results and their layout -------------------------------------------------
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_perm_3d", "view_step_2d", "view_rev_2d", "view_off_2d",
              "own_ro_2d", "view_bcast_2d", "own_0d", "own_empty_2d", "own_c_1d"]:
        a = SOURCES[s](I64)
        nd = a.ndim
        idx = np.array([0, 1], dtype=np.int64)
        for ax in ([None] + list(range(nd))) if nd else [None]:
            _call_view(em, "take", {"axis": ax}, [SOURCES[s](I64), idx.copy()], s, I64)
            _call_view(em, "repeat", {"repeats": 2, "axis": ax}, _ops_for("c", [s], I64), s, I64)
            _call_view(em, "roll", {"shift": 1, "axis": ax}, _ops_for("c", [s], I64), s, I64)
            _call_view(em, "sort", {"axis": ax}, _ops_for("c", [s], I64), s, I64)
            _call_view(em, "argsort", {"axis": ax}, _ops_for("c", [s], I64), s, I64)
            _call_view(em, "cumsum", {"axis": ax}, _ops_for("c", [s], I64), s, I64)
            _call_view(em, "delete", {"obj": 0, "axis": ax}, _ops_for("c", [s], I64), s, I64)
            if nd:
                cond = np.array([True, False, True])[: a.shape[ax] if ax is not None else 3]
                _call_view(em, "compress", {"axis": ax}, [cond.copy(), SOURCES[s](I64)], s, I64)
        if nd:
            _call_view(em, "insert", {"obj": 1, "axis": 0}, [SOURCES[s](I64), np.array(7, dtype=np.int64)], s, I64)
            _call_view(em, "append", {"axis": 0}, [SOURCES[s](I64), SOURCES[s](I64)], s, I64)
            _call_view(em, "diff", {"n": 1, "axis": -1}, _ops_for("c", [s], I64), s, I64)
        for reps in ([1] * max(nd, 1), [2] + [1] * max(nd - 1, 0)):
            _call_view(em, "tile", {"reps": reps}, _ops_for("c", [s], I64), s, I64)
        _call_view(em, "resize", {"shape": [2, 3]}, _ops_for("c", [s], I64), s, I64)
        _call_view(em, "unique", {}, _ops_for("c", [s], I64), s, I64)
        _call_view(em, "pad", {"width": 1}, _ops_for("c", [s], I64), s, I64)
        _call_view(em, "append", {}, [SOURCES[s](I64), SOURCES[s](I64)], s + ":flat", I64)
        for order in [None, "C", "F", "A"]:
            p = {} if order is None else {"order": order}
            for fn in ["zeros_like", "ones_like", "full_like", "empty_like"]:
                _call_view(em, fn, p, _ops_for("l", [s], I64), s, I64)
        for dec in [0, 1]:
            _call_view(em, "round", {"decimals": dec}, _ops_for("c", [s], F64), s, F64)
        lo = np.array(0, dtype=np.int64)
        hi = np.array(2, dtype=np.int64)
        _call_view(em, "clip", {}, [SOURCES[s](I64), lo.copy(), hi.copy()], s, I64)
        cond = (np.arange(a.size).reshape(a.shape) % 2 == 0) if nd else np.array(True)
        _call_view(em, "where3", {}, [np.array(cond), SOURCES[s](I64), SOURCES[s](I64)], s, I64)

    # -- K1/K2: ufunc results -- the layout NpyIter allocates --------------------------------------------------
    for s in ALL_SOURCES:
        for fn in UNARY_K:
            dt = B if fn == "logical_not" else F64
            _call_view(em, fn, {}, _ops_for(fn, [s], dt), s, dt)
        _call_view(em, "invert", {}, _ops_for("invert", [s], I64), s, I64)
    for s1, s2 in PAIRS:
        for fn in BINARY_K:
            dt = B if fn == "logical_and" else F64
            _call_view(em, fn, {}, [SOURCES[s1](dt), SOURCES[s2](dt)], s1 + "+" + s2, dt)
        _call_view(em, "bitwise_and", {}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        for fn in ["concatenate", "stack"]:
            for ax in [0, -1]:
                _call_view(em, fn, {"axis": ax}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        _call_view(em, "hstack", {}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        _call_view(em, "vstack", {}, [SOURCES[s1](I64), SOURCES[s2](I64)], s1 + "+" + s2, I64)
        for fn in ["matmul", "dot", "outer", "kron", "inner"]:
            _call_view(em, fn, {}, [SOURCES[s1](F64), SOURCES[s2](F64)], s1 + "+" + s2, F64)

    # -- K3: reductions -- the result's layout and the scalar a full reduction returns -----------------------------
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "own_c_3d", "own_f_3d", "view_perm_3d", "view_step_2d",
              "view_rev_2d", "view_off_2d", "view_bcast_2d", "own_ro_2d", "own_0d", "view_0d", "own_empty_2d",
              "own_unit_3d"]:
        a = SOURCES[s](F64)
        axes = [None] + list(range(a.ndim)) + ([-1] if a.ndim else [])
        for fn in REDUCE_K:
            dt = B if fn in ("all", "any") else F64
            for ax in axes:
                if fn in ("argmax", "argmin") and ax is None:
                    continue                                          # NumSharp's flat argmax returns a C# long
                for kd in [False, True]:
                    _call_view(em, fn, {"axis": ax, "keepdims": kd}, _ops_for(fn, [s], dt), s, dt)
        for ax in ([None] + list(range(a.ndim))) if a.ndim else [None]:
            _call_view(em, "cumprod", {"axis": ax}, _ops_for("cp", [s], F64), s, F64)

    _gen_views_v10(em)
    return em.cases, em.gaps


def _distinct_lanes(a, axis):
    """True when no lane of ``a`` along ``axis`` (the flattened array for None) holds a repeated value.

    argpartition's index order between equal values is the selection algorithm's choice on both sides, so its
    jobs keep only lanes whose order statistics are unique.
    """
    s = np.sort(a.reshape(-1) if axis is None else a, axis=-1 if axis is None else axis)
    return bool(np.all(np.diff(s, axis=-1 if axis is None else axis) != 0))


def _gen_views_v10(em):
    """Section V10 of the viewflags tier: the remaining fresh-result and index-tuple families.

    Triangles, diagflat, partitions, the nan-reductions and count_nonzero, the FFT family, the creators without an
    operand (eye / identity / tri / mgrid / ogrid), the index-tuple generators (diag_indices returns ONE array in
    every slot), and the ndarray methods whose np.* twins the earlier sections cover — a method may take another code
    path than its function, so its result object is gated on its own.

    Parameters:
        em: the tier emitter (cases are added to it).
    """
    I64, F64, C128 = "int64", "float64", "complex128"

    # Triangles and diagflat: fresh results whose layout NumPy derives from the input (tril/triu go through np.where).
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "view_perm_3d",
              "own_ro_2d", "view_bcast_2d", "own_1x1", "own_empty_2d", "own_c_1d"]:
        for k in [0, 1, -1]:
            _call_view(em, "tril", {"k": k}, _ops_for("tr", [s], I64), s, I64)
            _call_view(em, "triu", {"k": k}, _ops_for("tr", [s], I64), s, I64)
    for s in ["own_c_1d", "view_step_1d", "view_rev_1d", "own_c_2d", "own_f_2d", "view_t_2d", "own_0d", "view_bcast_2d"]:
        for k in [0, 1]:
            _call_view(em, "diagflat", {"k": k}, _ops_for("df", [s], I64), s, I64)

    # Partitions: only axes of length <= 3, where every position of the result is an order statistic (the
    # arrangement between kth anchors is the selection algorithm's choice on both sides).
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_perm_3d", "view_step_2d", "view_rev_2d", "view_off_2d",
              "own_ro_2d", "view_bcast_2d", "own_c_3d", "own_f_3d"]:
        a = SOURCES[s](F64)
        for ax in range(a.ndim):
            if not 2 <= a.shape[ax] <= 3:
                continue
            _call_view(em, "partition", {"kth": 1, "axis": ax}, _ops_for("pt", [s], F64), s, F64)
            if _distinct_lanes(a, ax):
                _call_view(em, "argpartition", {"kth": 1, "axis": ax}, _ops_for("pt", [s], F64), s, F64)
                _call_view(em, "ndarray.argpartition", {"kth": 1, "axis": ax}, _ops_for("pt", [s], F64), s, F64)

    # The nan-reductions and count_nonzero (the K3 reductions' siblings).
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "own_f_3d", "view_perm_3d", "view_step_2d", "view_off_2d",
              "view_bcast_2d", "own_ro_2d", "own_0d", "view_0d", "own_unit_3d"]:
        a = SOURCES[s](F64)
        axes = list(range(a.ndim)) + ([-1] if a.ndim else [])
        for fn in ["nanmean", "nanstd", "nanvar", "nanmin", "nanprod", "nanmedian"]:
            for ax in [None] + axes:
                for kd in [False, True]:
                    _call_view(em, fn, {"axis": ax, "keepdims": kd}, _ops_for(fn, [s], F64), s, F64)
        for fn in ["nanargmax", "nanargmin"]:
            for ax in axes:
                _call_view(em, fn, {"axis": ax, "keepdims": False}, _ops_for(fn, [s], F64), s, F64)
        for ax in axes:
            for kd in [False, True]:
                _call_view(em, "count_nonzero", {"axis": ax, "keepdims": kd}, _ops_for("cnz", [s], I64), s, I64)

    # The FFT family: 1-D transforms along an axis, the N-D transforms, the shifts.
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "view_perm_3d",
              "own_ro_2d", "view_bcast_2d", "own_c_1d", "view_step_1d"]:
        nd = SOURCES[s](F64).ndim
        for ax in ([-1, 0] if nd >= 2 else [-1]):
            for f in ["fft", "ifft", "irfft", "hfft"]:
                # A 6-point irfft / hfft input makes n = 10, whose twiddles no portable tier pins (see gen_out).
                if nd == 1 and f in ("irfft", "hfft"):
                    continue
                _call_view(em, "fft." + f, {"axis": ax}, _ops_for(f, [s], C128), s, C128)
            for f in ["rfft", "ihfft"]:
                _call_view(em, "fft." + f, {"axis": ax}, _ops_for(f, [s], F64), s, F64)
        if nd >= 2:
            for f in ["fft2", "ifft2", "fftn", "ifftn", "irfft2", "irfftn"]:
                _call_view(em, "fft." + f, {}, _ops_for(f, [s], C128), s, C128)
            for f in ["rfft2", "rfftn"]:
                _call_view(em, "fft." + f, {}, _ops_for(f, [s], F64), s, F64)
        for f in ["fftshift", "ifftshift"]:
            _call_view(em, "fft." + f, {}, _ops_for(f, [s], I64), s, I64)
            _call_view(em, "fft." + f, {"axes": [0]}, _ops_for(f, [s], I64), s, I64)

    # Creators without an operand.
    for m in [None, 4]:
        for k in [0, 1]:
            for order in ["C", "F"]:
                p = {"N": 3, "k": k, "order": order}
                if m is not None:
                    p["M"] = m
                _call_view(em, "eye", p, [], "none", F64)
        for k in [0, -1]:
            p = {"N": 3, "k": k}
            if m is not None:
                p["M"] = m
            _call_view(em, "tri", p, [], "none", F64)
    _call_view(em, "identity", {"n": 3}, [], "none", F64)
    for slices in [["0:2", "0:3"], ["1:7:2", "0:2"], ["0:2", "0:1", "2:5"]]:
        _call_view(em, "mgrid", {"slices": slices}, [], "none", I64)
        _call_view(em, "ogrid", {"slices": slices}, [], "none", I64)
    for ndim in [2, 3]:
        _call_view(em, "diag_indices", {"n": 3, "ndim": ndim}, [], "none", I64)
    for k in [0, 1, -1]:
        _call_view(em, "tril_indices", {"n": 3, "k": k}, [], "none", I64)
        _call_view(em, "triu_indices", {"n": 3, "k": k}, [], "none", I64)

    # ndarray methods whose np.* twins the earlier sections cover.
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_perm_3d", "view_step_2d", "view_off_2d", "view_bcast_2d",
              "own_ro_2d", "own_0d", "view_0d", "own_c_1d"]:
        a = SOURCES[s](F64)
        axes = list(range(a.ndim))
        for m in ["sum", "mean", "std", "var", "prod"]:
            _call_view(em, "ndarray." + m, {"axis": None, "keepdims": False}, _ops_for(m, [s], F64), s, F64)
            for ax in axes:
                for kd in [False, True]:
                    _call_view(em, "ndarray." + m, {"axis": ax, "keepdims": kd}, _ops_for(m, [s], F64), s, F64)
        for m in ["argmax", "argmin"]:
            for ax in axes:
                for kd in [False, True]:
                    _call_view(em, "ndarray." + m, {"axis": ax, "keepdims": kd}, _ops_for(m, [s], F64), s, F64)
        for ax in axes + ([-1] if axes else []):
            _call_view(em, "ndarray.argsort", {"axis": ax}, _ops_for("as", [s], I64), s, I64)
        for ax in [None] + axes[:1]:
            _call_view(em, "ndarray.repeat", {"repeats": 2, "axis": ax}, _ops_for("rp", [s], I64), s, I64)
        if a.ndim >= 1:
            _call_view(em, "ndarray.nonzero", {}, _ops_for("nz", [s], I64), s, I64)
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "view_rev_2d", "view_off_2d", "own_ro_2d"]:
        for side in ["left", "right"]:
            _call_view(em, "ndarray.searchsorted", {"side": side},
                       [np.array([-1.0, 0.0, 1.0, 2.5, 3.0]), SOURCES[s](F64)], "sorted+" + s, F64)
    for s in ["own_c_2d", "own_f_2d", "view_t_2d", "view_step_2d", "own_ro_2d", "view_bcast_2d"]:
        for second in ["c", "f"]:
            b = _owner((4, 2), F64, order="F" if second == "f" else "C")
            _call_view(em, "ndarray.dot", {}, [SOURCES[s](F64), b], s + "+own_" + second, F64)


# =============================================================================================================
# Section O: out= (op "outflags")
# =============================================================================================================

def _out_view(shape, dt, kind):
    """An out array of ``shape`` in the requested layout, prefilled with a marker so untouched slots show.

    Kinds: own_c (an owner), own_f (an F owner), view_off (a C view at an offset), view_step (every other
    element of a wider owner), view_t (an F view through a transpose), view_rev (a reversed view).

    Parameters:
        shape: the out's shape.
        dt: its dtype.
        kind: the layout kind.

    Returns:
        The out array, or None when the kind does not exist at this rank.
    """
    shape = tuple(shape)
    n = int(np.prod(shape)) if shape else 1
    marker = lambda m: (np.arange(m) % 4 + 7).astype(dt)  # noqa: E731 -- 7..10, distinct from every result here
    if kind == "own_c":
        o = np.empty(shape, dtype=dt)
        o[...] = marker(n).reshape(shape)
        return o
    if kind == "own_f":
        if len(shape) < 2:
            return None
        o = np.empty(shape, dtype=dt, order="F")
        o[...] = marker(n).reshape(shape)
        return o
    if kind == "view_off":
        b = marker(n + 3)
        return b[3:].reshape(shape)
    if kind == "view_step":
        if not shape:
            return None
        wide = tuple(shape[:-1]) + (shape[-1] * 2,)
        b = marker(int(np.prod(wide))).reshape(wide)
        return b[..., ::2]
    if kind == "view_t":
        if len(shape) < 2:
            return None
        b = marker(n).reshape(tuple(reversed(shape)))
        return b.T
    if kind == "view_rev":
        if not shape:
            return None
        b = marker(n).reshape(shape)
        return b[..., ::-1]
    raise ValueError(kind)


OUT_KINDS = ["own_c", "own_f", "view_off", "view_step", "view_t", "view_rev"]


def _out_fns():
    """fn -> callable(ops, params, out) for the outflags tier (twinned by OpRegistry.ViewFlags.cs)."""
    t = {}
    for u in ["add", "subtract", "multiply", "maximum", "equal", "logical_and", "bitwise_and", "arctan2"]:
        t[u] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], o[1], out=out)))(u)
    for u in ["negative", "sqrt", "absolute", "isnan", "sign", "invert", "square", "floor", "conjugate", "positive"]:
        t[u] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], out=out)))(u)
    for r in ["ptp", "median", "nanmedian"]:
        t[r] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], axis=p["axis"], out=out)))(r)
    for r in ["percentile", "quantile", "nanpercentile", "nanquantile"]:
        t[r] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], p["q"], axis=p["axis"], out=out)))(r)
    for r in ["cumsum", "cumprod", "nancumsum", "nancumprod"]:
        t[r] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], axis=p["axis"], out=out)))(r)
    for r in ["nanargmax", "nanargmin"]:
        t[r] = (lambda name: (lambda o, p, out: getattr(np, name)(o[0], axis=p["axis"], out=out)))(r)
    t["trace"] = lambda o, p, out: np.trace(o[0], out=out)
    t["clip"] = lambda o, p, out: np.clip(o[0], o[1], o[2], out=out)
    t["round"] = lambda o, p, out: np.round(o[0], p["decimals"], out=out)
    t["take"] = lambda o, p, out: np.take(o[0], o[1], axis=p["axis"], out=out)
    t["compress"] = lambda o, p, out: np.compress(o[0], o[1], axis=p["axis"], out=out)
    t["choose"] = lambda o, p, out: np.choose(o[0], [o[1], o[2]], out=out)
    t["concatenate"] = lambda o, p, out: np.concatenate([o[0], o[1]], axis=p["axis"], out=out)
    t["dot"] = lambda o, p, out: np.dot(o[0], o[1], out=out)
    t["matmul"] = lambda o, p, out: np.matmul(o[0], o[1], out=out)
    t["vecdot"] = lambda o, p, out: np.vecdot(o[0], o[1], out=out)
    t["matvec"] = lambda o, p, out: np.matvec(o[0], o[1], out=out)
    t["vecmat"] = lambda o, p, out: np.vecmat(o[0], o[1], out=out)
    t["outer"] = lambda o, p, out: np.outer(o[0], o[1], out=out)
    t["einsum"] = lambda o, p, out: np.einsum(p["subscripts"], *o, out=out)
    t["all"] = lambda o, p, out: np.all(o[0], axis=p["axis"], out=out)
    t["any"] = lambda o, p, out: np.any(o[0], axis=p["axis"], out=out)
    t["max"] = lambda o, p, out: np.max(o[0], axis=p["axis"], out=out)
    t["min"] = lambda o, p, out: np.min(o[0], axis=p["axis"], out=out)
    t["fft"] = lambda o, p, out: np.fft.fft(o[0], out=out)
    # ---- every other ufunc NumSharp gives an out= (the generic forms) ----
    for u in OUT_UNARY_UFUNCS:
        t.setdefault(u, (lambda name: (lambda o, p, out: getattr(np, name)(o[0], out=out)))(u))
    for u in OUT_BINARY_UFUNCS:
        t.setdefault(u, (lambda name: (lambda o, p, out: getattr(np, name)(o[0], o[1], out=out)))(u))
    # ---- aliases, the FFT family, multi_dot, and the ndarray methods that take out= ----
    t["around"] = lambda o, p, out: np.around(o[0], p["decimals"], out=out)
    t["concat"] = lambda o, p, out: np.concat([o[0], o[1]], axis=p["axis"], out=out)
    t["multi_dot"] = lambda o, p, out: np.linalg.multi_dot(list(o), out=out)
    for f in ["ifft", "rfft", "irfft", "hfft", "ihfft"]:
        t[f] = (lambda name: (lambda o, p, out: getattr(np.fft, name)(o[0], out=out)))(f)
    for f in ["fft2", "ifft2", "fftn", "ifftn"]:
        t[f] = (lambda name: (lambda o, p, out: getattr(np.fft, name)(o[0], out=out)))(f)
    t["ndarray.all"] = lambda o, p, out: o[0].all(axis=p["axis"], out=out)
    t["ndarray.any"] = lambda o, p, out: o[0].any(axis=p["axis"], out=out)
    t["ndarray.max"] = lambda o, p, out: o[0].max(axis=p["axis"], out=out)
    t["ndarray.min"] = lambda o, p, out: o[0].min(axis=p["axis"], out=out)
    t["ndarray.cumsum"] = lambda o, p, out: o[0].cumsum(axis=p["axis"], out=out)
    t["ndarray.cumprod"] = lambda o, p, out: o[0].cumprod(axis=p["axis"], out=out)
    t["ndarray.clip"] = lambda o, p, out: o[0].clip(o[1], o[2], out=out)
    t["ndarray.round"] = lambda o, p, out: o[0].round(p["decimals"], out=out)
    t["ndarray.take"] = lambda o, p, out: o[0].take(o[1], axis=p["axis"], out=out)
    t["ndarray.compress"] = lambda o, p, out: o[1].compress(o[0], axis=p["axis"], out=out)
    t["ndarray.choose"] = lambda o, p, out: o[0].choose([o[1], o[2]], out=out)
    t["ndarray.trace"] = lambda o, p, out: o[0].trace(out=out)
    return t


# Every unary / binary ufunc NumSharp gives an out= (the reflection inventory of np.*), by input kind: the generic
# outflags rows run each with out arrays of every layout. Names NumPy spells differently are NumPy's own aliases.
OUT_UNARY_UFUNCS = [
    "abs", "fabs", "negative", "positive", "sign", "sqrt", "cbrt", "square", "reciprocal", "floor", "ceil", "trunc",
    "rint", "fix", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh",
    "arctanh", "asinh", "acosh", "atanh", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "deg2rad", "rad2deg",
    "degrees", "radians", "spacing", "isnan", "isinf", "isfinite", "isposinf", "isneginf", "signbit", "conj",
    "conjugate", "logical_not", "invert", "bitwise_not", "bitwise_count",
]
OUT_BINARY_UFUNCS = [
    "add", "subtract", "multiply", "divide", "true_divide", "floor_divide", "mod", "remainder", "fmod", "power",
    "float_power", "maximum", "minimum", "fmax", "fmin", "arctan2", "hypot", "copysign", "nextafter", "logaddexp",
    "logaddexp2", "heaviside", "equal", "not_equal", "less", "less_equal", "greater", "greater_equal", "logical_and",
    "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm", "left_shift", "right_shift",
    "ldexp",
]
# The input dtype each ufunc's out= rows use: integer-only and bool-only loops get their own.
_OUT_INT_UFUNCS = {"invert", "bitwise_not", "bitwise_count", "bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm",
                   "left_shift", "right_shift"}
_OUT_BOOL_UFUNCS = {"logical_not", "logical_and", "logical_or", "logical_xor"}
# The loops that call the platform libm (their out= rows take an all-zero input; see gen_out).
_OUT_LIBM_UFUNCS = {"sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh",
                    "arctanh", "asinh", "acosh", "atanh", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
                    "cbrt", "power", "float_power", "hypot", "logaddexp", "logaddexp2"}


OUT_FNS = _out_fns()


def gen_out(carve=True):
    """The outflags tier: ``f(..., out=o)`` returns ``o`` and fills it, for every out-capable API.

    Each case's operands are the inputs followed by the out array (prior contents included, so the base slot
    shows which elements the call wrote). Slot 0 is the returned array with its facet (``same[out]`` true),
    slot 1 the whole base buffer of ``o``. A refused out is NumPy's verbatim error.

    Parameters:
        carve: file carved cases into the gaps list.

    Returns:
        (gate cases, carved cases).
    """
    em = _Emitter("out", "outflags", carve)
    F64, I64, B, C128 = "float64", "int64", "bool", "complex128"
    jobs = []   # (fn, params, input ops builder, input layout tag, out dtype or None (= natural))

    def inputs(sources, dt):
        """A builder of fresh input operands from source names (each call builds new arrays: an out call may alias)."""
        return lambda: [SOURCES[s](dt) for s in sources]

    for fn in ["add", "subtract", "multiply", "maximum", "equal", "arctan2"]:
        jobs.append((fn, {}, inputs(["own_c_2d", "own_c_2d"], F64), "c+c", F64))
        jobs.append((fn, {}, inputs(["own_f_2d", "view_bcast_2d"], F64), "f+bcast", F64))
    jobs.append(("logical_and", {}, inputs(["own_c_2d", "own_c_2d"], B), "c+c", B))
    jobs.append(("bitwise_and", {}, inputs(["own_c_2d", "own_c_2d"], I64), "c+c", I64))
    for fn in ["negative", "sqrt", "absolute", "isnan", "sign", "square", "floor", "conjugate", "positive"]:
        jobs.append((fn, {}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append((fn, {}, inputs(["view_t_2d"], F64), "t", F64))
    jobs.append(("invert", {}, inputs(["own_c_2d"], I64), "c", I64))
    for ax in [0, 1]:
        for fn in ["ptp", "median", "nanmedian", "cumsum", "cumprod", "nancumsum", "nancumprod", "max", "min"]:
            jobs.append((fn, {"axis": ax}, inputs(["own_c_2d"], F64), "c", F64))
        for fn in ["all", "any"]:
            jobs.append((fn, {"axis": ax}, inputs(["own_c_2d"], B), "c", B))
        for fn in ["nanargmax", "nanargmin"]:
            jobs.append((fn, {"axis": ax}, inputs(["own_c_2d"], F64), "c", I64))
        for fn in ["percentile", "quantile", "nanpercentile", "nanquantile"]:
            q = 50.0 if "percentile" in fn else 0.5
            jobs.append((fn, {"axis": ax, "q": q}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append(("take", {"axis": ax}, lambda: [SOURCES["own_c_2d"](F64), np.array([0, 2], dtype=np.int64)],
                     "c", F64))
        jobs.append(("compress", {"axis": ax}, lambda: [np.array([True, False, True]), SOURCES["own_c_2d"](F64)],
                     "c", F64))
        jobs.append(("concatenate", {"axis": ax}, inputs(["own_c_2d", "own_c_2d"], F64), "c+c", F64))
    jobs.append(("trace", {}, inputs(["own_c_2d"], F64), "c", F64))
    jobs.append(("clip", {}, lambda: [SOURCES["own_c_2d"](F64), np.array(0.0), np.array(2.0)], "c", F64))
    for dec in [0, 1]:
        jobs.append(("round", {"decimals": dec}, inputs(["own_c_2d"], F64), "c", F64))
    jobs.append(("choose", {}, lambda: [(np.arange(12).reshape(3, 4) % 2).astype(np.int64), SOURCES["own_c_2d"](F64),
                                        SOURCES["own_f_2d"](F64)], "c", F64))
    jobs.append(("dot", {}, lambda: [SOURCES["own_c_2d"](F64), np.ascontiguousarray(SOURCES["own_c_2d"](F64).T)],
                 "c+c", F64))
    jobs.append(("matmul", {}, lambda: [SOURCES["own_c_2d"](F64), np.ascontiguousarray(SOURCES["own_c_2d"](F64).T)],
                 "c+c", F64))
    jobs.append(("vecdot", {}, inputs(["own_c_2d", "own_c_2d"], F64), "c+c", F64))
    jobs.append(("matvec", {}, lambda: [SOURCES["own_c_2d"](F64), np.arange(4.0)], "c+1d", F64))
    jobs.append(("vecmat", {}, lambda: [np.arange(3.0), SOURCES["own_c_2d"](F64)], "1d+c", F64))
    jobs.append(("outer", {}, inputs(["own_c_1d", "own_c_1d"], F64), "1d+1d", F64))
    jobs.append(("einsum", {"subscripts": "ij->ji"}, inputs(["own_c_2d"], F64), "c", F64))
    jobs.append(("einsum", {"subscripts": "ij,jk->ik"},
                 lambda: [SOURCES["own_c_2d"](F64), np.ascontiguousarray(SOURCES["own_c_2d"](F64).T)], "c+c", F64))
    jobs.append(("fft", {}, inputs(["own_c_1d"], C128), "c1d", C128))
    # Every other ufunc with an out= in NumSharp, over one C input (the layouts that matter are the out's). A loop
    # that calls the platform libm takes an all-zero input, where every result is exact: on the -1..3 values
    # win-amd64 NumPy's arccosh(2), expm1(1), cbrt(2) and logaddexp(-1, 0) are not correctly rounded and log2(3) sits
    # 0.02 ULP from a midpoint, so a Linux or macOS libm would round them otherwise. This tier gates the out= object.
    covered = {j[0] for j in jobs}

    def zeros(n):
        """A builder of ``n`` fresh all-zero (3, 4) float64 inputs (the libm-backed loops' exact input)."""
        return lambda: [np.zeros((3, 4)) for _ in range(n)]

    for u in OUT_UNARY_UFUNCS:
        if u in covered:
            continue
        dt = I64 if u in _OUT_INT_UFUNCS else (B if u in _OUT_BOOL_UFUNCS else F64)
        jobs.append((u, {}, zeros(1) if u in _OUT_LIBM_UFUNCS else inputs(["own_c_2d"], dt), "c", None))
    for u in OUT_BINARY_UFUNCS:
        if u in covered:
            continue
        dt = I64 if u in _OUT_INT_UFUNCS else (B if u in _OUT_BOOL_UFUNCS else F64)
        if u == "ldexp":
            jobs.append((u, {}, lambda: [SOURCES["own_c_2d"](F64), SOURCES["own_c_2d"](I64)], "c+c", None))
        else:
            jobs.append((u, {}, zeros(2) if u in _OUT_LIBM_UFUNCS else inputs(["own_c_2d", "own_c_2d"], dt),
                         "c+c", None))
    # Aliases, the FFT family, multi_dot and the ndarray methods with out=.
    jobs.append(("around", {"decimals": 1}, inputs(["own_c_2d"], F64), "c", F64))
    for ax in [0, 1]:
        jobs.append(("concat", {"axis": ax}, inputs(["own_c_2d", "own_c_2d"], F64), "c+c", F64))
    jobs.append(("multi_dot", {}, lambda: [SOURCES["own_c_2d"](F64), np.ascontiguousarray(SOURCES["own_c_2d"](F64).T),
                                           SOURCES["own_c_2d"](F64)], "c+c+c", F64))
    # FFT lengths stay in {2, 3, 4, 6}: their twiddles are the angles fft.jsonl proves on every host (a length-6
    # irfft / hfft input would make n = 10, whose twiddles no portable tier pins), so irfft / hfft take 4 points.
    for f in ["ifft", "rfft", "ihfft"]:
        jobs.append((f, {}, inputs(["own_c_1d"], C128 if f == "ifft" else F64), "c1d", None))
    for f in ["irfft", "hfft"]:
        jobs.append((f, {}, lambda: [_owner((4,), C128)], "c1d", None))
    for f in ["fft2", "ifft2", "fftn", "ifftn"]:
        jobs.append((f, {}, inputs(["own_c_2d"], C128), "c", None))
    for ax in [0, 1]:
        jobs.append(("ndarray.all", {"axis": ax}, inputs(["own_c_2d"], B), "c", B))
        jobs.append(("ndarray.any", {"axis": ax}, inputs(["own_c_2d"], B), "c", B))
        jobs.append(("ndarray.max", {"axis": ax}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append(("ndarray.min", {"axis": ax}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append(("ndarray.cumsum", {"axis": ax}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append(("ndarray.cumprod", {"axis": ax}, inputs(["own_c_2d"], F64), "c", F64))
        jobs.append(("ndarray.take", {"axis": ax},
                     lambda: [SOURCES["own_c_2d"](F64), np.array([0, 2], dtype=np.int64)], "c", F64))
        jobs.append(("ndarray.compress", {"axis": ax},
                     lambda: [np.array([True, False, True]), SOURCES["own_c_2d"](F64)], "c", F64))
    jobs.append(("ndarray.clip", {}, lambda: [SOURCES["own_c_2d"](F64), np.array(0.0), np.array(2.0)], "c", F64))
    jobs.append(("ndarray.round", {"decimals": 1}, inputs(["own_c_2d"], F64), "c", F64))
    jobs.append(("ndarray.choose", {}, lambda: [(np.arange(12).reshape(3, 4) % 2).astype(np.int64),
                                                SOURCES["own_c_2d"](F64), SOURCES["own_f_2d"](F64)], "c", F64))
    jobs.append(("ndarray.trace", {}, inputs(["own_c_2d"], F64), "c", F64))

    for fn, params, mk, tag, odt in jobs:
        f = OUT_FNS[fn]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                natural = np.asarray(f(mk(), params, None))
        except Exception:  # noqa: BLE001
            em.skipped += 1
            continue
        for kind in OUT_KINDS:
            out = _out_view(natural.shape, natural.dtype if odt is None else natural.dtype, kind)
            if out is None:
                continue
            ops = mk()
            descs = [_desc(o) for o in ops] + [_desc(out)]
            p = dict(params)
            p["fn"] = fn
            p["out"] = len(ops)
            cid = em.next_id(f"{fn}/{tag}/out={kind}")
            owner = _ultimate_owner(out)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = f(ops, params, out)
            except Exception as e:  # noqa: BLE001
                em.add(_error(cid, "outflags", p, descs, e, f"{tag}/out={kind}", kind="tuple"))
                continue
            all_ops = list(ops) + [out]
            slot0 = _result_expected(r, all_ops)
            slot1 = _value(owner.reshape(-1) if owner.flags.c_contiguous else owner.ravel(order="K"))
            em.add(_case(cid, "outflags", p, descs, {"kind": "tuple", "slots": [slot0, slot1]},
                         f"{tag}/out={kind}", "outflags"))
    return em.cases, em.gaps


# =============================================================================================================
# Section S: shape parameters (op "shapearg")
# =============================================================================================================

# The C# forms a shape argument can take, per fn. "view:<layout>" hands the dims over as a view's Shape
# (RandomApi/ViewSizeLayouts.cs: stepped, offset stepped, reversed, broadcast, contiguous slice, transposed,
# permuted, 0-d offset) -- the class of the 2026-09-24 view-Shape allocation overrun.
VIEW_FORMS_BY_RANK = {
    0: ["view:0-d offset"],
    1: ["view:stepped", "view:offset stepped", "view:reversed", "view:broadcast", "view:contiguous slice"],
    2: ["view:stepped", "view:offset stepped", "view:reversed", "view:broadcast", "view:contiguous slice",
        "view:transposed"],
    3: ["view:stepped", "view:offset stepped", "view:reversed", "view:broadcast", "view:contiguous slice",
        "view:transposed", "view:permuted"],
}

SHAPE_FNS = {
    # fn: (callable(ops, dims, params), C# forms, needs operand?)
    "zeros": (lambda o, d, p: np.zeros(d, dtype=p.get("dtype")), ["Shape", "long[]", "int[]", "int", "view"], False),
    "ones": (lambda o, d, p: np.ones(d, dtype=p.get("dtype")), ["Shape", "long[]", "int[]", "int", "view"], False),
    "empty": (lambda o, d, p: _filled0(np.empty(d, dtype=p.get("dtype"))), ["Shape", "long[]", "int[]", "int", "view"], False),
    "full": (lambda o, d, p: np.full(d, 3, dtype=p.get("dtype")), ["Shape", "long[]", "int[]", "view"], False),
    "reshape": (lambda o, d, p: np.reshape(o[0], d), ["Shape", "long[]", "int[]", "view"], True),
    "ndarray.reshape": (lambda o, d, p: o[0].reshape(d), ["Shape", "long[]", "int[]", "view"], True),
    "broadcast_to": (lambda o, d, p: np.broadcast_to(o[0], d), ["Shape", "view"], True),
    "resize": (lambda o, d, p: np.resize(o[0], d), ["Shape", "view"], True),
    "tile": (lambda o, d, p: np.tile(o[0], d), ["long[]", "int[]"], True),
    "indices": (lambda o, d, p: np.indices(d, dtype=p.get("dtype", int)), ["Shape", "long[]", "int[]", "view"], False),
    "empty_like": (lambda o, d, p: _filled0(np.empty_like(o[0], shape=d)), ["Shape", "view"], True),
}


def _shape_forms(fn, dims, dt):
    """The C# forms one (fn, dims, dtype) case is replayed with: the fn's fixed forms plus its view layouts.

    zeros / ones / empty take a dtype only beside a Shape, so their long[] / int[] / int forms are emitted for the
    default float64 alone (an int32 array form would have to cast afterwards, a different call).

    Parameters:
        fn: the shapearg fn.
        dims: the dims (negative entries are NumPy refusals; a Shape cannot carry them).
        dt: the dtype of the case.

    Returns:
        The form names, in a fixed order.
    """
    forms = []
    for f in SHAPE_FNS[fn][1]:
        if f == "view":
            if all(x >= 0 for x in dims):
                forms += VIEW_FORMS_BY_RANK[min(len(dims), 3)]
        elif f in ("long[]", "int[]", "int") and fn in ("zeros", "ones", "empty") and dt != "float64":
            continue
        elif f == "int":
            if len(dims) == 1:
                forms.append("int")
        elif f == "Shape" and any(x < 0 for x in dims):
            continue                                           # a Shape cannot hold a negative extent
        else:
            forms.append(f)
    return forms


def gen_shapes(carve=True):
    """The shapearg tier: every shape-taking API called with each C# form of the same dims.

    NumPy's result depends only on the dims, so each (fn, dims) result is computed once and emitted per form;
    NumPy's refusals (a negative extent, a size that does not reshape) are recorded verbatim for the forms that
    can carry the invalid dims.

    Parameters:
        carve: file carved cases into the gaps list.

    Returns:
        (gate cases, carved cases).
    """
    em = _Emitter("shapes", "shapearg", carve)
    dims_by_fn = {
        "zeros": [[], [0], [3], [2, 3], [2, 0, 3], [1, 1], [2, 3, 4], [-1], [2, -3]],
        "ones": [[], [3], [2, 3], [2, 3, 4], [-2]],
        "empty": [[], [3], [2, 3], [2, 3, 4], [0, 2]],
        "full": [[], [3], [2, 3], [2, 3, 4]],
        "reshape": [[12], [3, 4], [4, 3], [2, 3, 2], [2, 2, 3], [-1], [3, -1], [5], [2, -1, -1]],
        "ndarray.reshape": [[12], [3, 4], [2, 3, 2], [-1], [4, -1], [7]],
        "broadcast_to": [[3, 4], [2, 3, 4], [4], [5, 4]],
        "resize": [[3, 4], [2, 3], [5], [2, 3, 4], [0]],
        "tile": [[2], [2, 1], [1, 2, 1], [0]],
        "indices": [[2, 3], [3], [2, 1, 2], [0, 2]],
        "empty_like": [[2, 3], [5], [2, 3, 4], []],
    }
    for fn, dimsets in dims_by_fn.items():
        f, _forms, needs = SHAPE_FNS[fn]
        for dims in dimsets:
            for dt in (["float64", "int32"] if not needs else ["int64"]):
                # The operand a shape-taking method or function works on: a (3,4) owner, or a (4,) row for
                # broadcast_to (so every target above is a legal broadcast).
                ops = [SOURCES["own_c_2d"](dt)] if needs else []
                if fn == "broadcast_to":
                    ops = [_owner((4,), dt)]
                p = {"dims": dims}
                if not needs:
                    p["dtype"] = dt
                for form in _shape_forms(fn, dims, dt):
                    q = dict(p, fn=fn, form=form)
                    descs = [_desc(o) for o in ops]
                    cid = em.next_id(f"{fn}/{'x'.join(map(str, dims)) or 'scalar'}/{dt}/{form}")
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            r = f(ops, tuple(dims) if form != "int" else dims[0], p)
                    except Exception as e:  # noqa: BLE001
                        em.add(_error(cid, "shapearg", q, descs, e, form))
                        continue
                    em.add(_case(cid, "shapearg", q, descs, _result_expected(r, ops), form, "shapearg"))
    # indices(sparse=True): a tuple of open-grid views of ONE arange each.
    for dims in [[2, 3], [3], [2, 1, 2]]:
        for form in ["Shape", "long[]", "int[]"] + VIEW_FORMS_BY_RANK[min(len(dims), 3)]:
            q = {"fn": "indices_sparse", "dims": dims, "form": form, "dtype": "int64"}
            r = np.indices(tuple(dims), dtype=np.int64, sparse=True)
            em.add(_case(em.next_id(f"indices_sparse/{'x'.join(map(str, dims))}/{form}"), "shapearg", q, [],
                         _tuple_result_expected(list(r), []), form, "shapearg"))
    return em.cases, em.gaps


# =============================================================================================================
# Section P: secondary array parameters in non-C layouts (EXISTING op keys)
# =============================================================================================================

def _arr_source(kind, base):
    """A secondary operand in the requested layout, holding ``base``'s values in C logical order.

    Kinds: c (a C owner), f (an F owner), t (an F view through a transpose), step (every other element),
    rev (reversed), off (a C view at an offset), ro (a read-only owner), bcast (a read-only broadcast of the
    first row -- only when the array's rows repeat), 0d (a 0-d owner, for 1-element arrays).

    Parameters:
        kind: the layout kind.
        base: a C-contiguous array whose VALUES the operand must hold.

    Returns:
        The operand, or None when the kind cannot hold these values at this shape.
    """
    shape = base.shape
    dt = base.dtype
    if kind == "c":
        return np.ascontiguousarray(base).copy()
    if kind == "f":
        if base.ndim < 2:
            return None
        return np.asfortranarray(base).copy(order="F")
    if kind == "t":
        if base.ndim < 2:
            return None
        return np.ascontiguousarray(base.T).T
    if kind == "step":
        if base.ndim == 0:
            return None
        wide = np.zeros(tuple(shape[:-1]) + (shape[-1] * 2,), dtype=dt)
        wide[..., ::2] = base
        return wide[..., ::2]
    if kind == "rev":
        if base.ndim == 0:
            return None
        return np.ascontiguousarray(base[..., ::-1])[..., ::-1]
    if kind == "off":
        # The prefix holds the sentinel 7 (see _off_view): a read that ignores the offset sees it.
        flat = np.empty(base.size + 3, dtype=dt)
        flat[:3] = np.array(7).astype(dt)
        flat[3:] = base.reshape(-1)
        return flat[3:].reshape(shape)
    if kind == "ro":
        c = np.ascontiguousarray(base).copy()
        c.setflags(write=False)
        return c
    if kind == "bcast":
        if base.ndim < 1 or base.shape[0] < 2 or not np.all(base == base[:1]):
            return None
        return np.broadcast_to(np.ascontiguousarray(base[:1]).copy(), shape)
    if kind == "0d":
        if base.size != 1:
            return None
        return np.ascontiguousarray(base.reshape(())).copy()
    raise ValueError(kind)


ARR_KINDS = ["c", "f", "t", "step", "rev", "off", "ro", "bcast", "0d"]


def gen_arrays(carve=True):
    """The array_args tier: existing op keys with their SECONDARY array operands in non-C layouts.

    Each job names an existing corpus op key (whose OpRegistry entry is the direct NumPy call), its params, the
    first operand(s) (C-contiguous) and the secondary operand's VALUES; the secondary operand is then replayed in
    every layout kind that can hold those values. The facet is attached because these results are fresh arrays
    (or, for take / compress of a 0-d, scalars) whose flags must match too.

    Parameters:
        carve: file carved cases into the gaps list.

    Returns:
        (gate cases, carved cases).
    """
    em = _Emitter("arrays", None, carve)
    a2 = _owner((3, 4), "float64")
    a1 = np.ascontiguousarray(np.array([0.0, 1.5, 3.0, 4.5, 6.0]))
    sorted1 = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    jobs = [
        # (op key, params, operand builder taking the secondary operand, secondary values, position label)
        ("take", {"axis": 0}, lambda s: [a2.copy(), s], np.array([[0, 2], [1, 0]], dtype=np.int64), "indices"),
        ("take", {"axis": 1}, lambda s: [a2.copy(), s], np.array([[3, 0], [1, 1]], dtype=np.int64), "indices"),
        ("take_along_axis", {"axis": 1}, lambda s: [a2.copy(), s],
         np.array([[0, 3, 1], [2, 2, 0], [3, 1, 0]], dtype=np.int64), "indices"),
        ("compress", {"axis": 0}, lambda s: [s, a2.copy()], np.array([True, False, True]), "condition"),
        ("extract", {}, lambda s: [s, a2.copy()], (np.arange(12).reshape(3, 4) % 3 == 0), "condition"),
        ("choose", {"nc": 2}, lambda s: [s, a2.copy(), (a2 * 2).copy()],
         (np.arange(12).reshape(3, 4) % 2).astype(np.int64), "index"),
        ("searchsorted", {"side": "left"}, lambda s: [sorted1.copy(), s],
         np.array([[0.5, 2.0, 9.0], [3.0, 4.0, 1.0]]), "v"),
        ("searchsorted", {"side": "right"}, lambda s: [np.array([5.0, 1.0, 3.0, 2.0, 8.0]), np.array([2.0, 7.5]), s],
         np.array([1, 3, 2, 0, 4], dtype=np.int64), "sorter"),
        ("digitize", {"right": False}, lambda s: [a2.copy(), s], np.array([0.0, 1.0, 2.5]), "bins"),
        ("isin", {}, lambda s: [a2.copy(), s], np.array([[0.0, 2.0], [3.0, -1.0]]), "test_elements"),
        ("bincount", {"minlength": 0}, lambda s: [np.array([0, 1, 1, 3, 2], dtype=np.int64), s],
         np.array([0.5, 1.0, 1.5, 2.0, 2.5]), "weights"),
        ("interp", {}, lambda s: [np.array([0.5, 1.5, 2.5]), s, np.array([10.0, 20.0, 30.0])],
         np.array([0.0, 1.0, 3.0]), "xp"),
        ("interp", {}, lambda s: [np.array([0.5, 1.5, 2.5]), np.array([0.0, 1.0, 3.0]), s],
         np.array([10.0, 20.0, 30.0]), "fp"),
        ("inner", {}, lambda s: [a2.copy(), s], _owner((2, 4), "float64"), "b"),
        ("kron", {}, lambda s: [_owner((2, 2), "float64"), s], _owner((2, 3), "float64"), "b"),
        ("matvec", {}, lambda s: [a2.copy(), s], np.array([1.0, -1.0, 2.0, 0.5]), "x2"),
        ("vecmat", {}, lambda s: [np.array([1.0, -1.0, 2.0]), s], a2.copy(), "x2"),
        ("cross", {}, lambda s: [_owner((2, 3), "float64"), s], np.array([[1.0, 0.0, 2.0], [0.5, 3.0, -1.0]]), "b"),
        ("insert", {"obj": 1, "axis": 0}, lambda s: [a2.copy(), s], np.array([7.0, 8.0, 9.0, 10.0]), "values"),
        ("append", {"axis": 0}, lambda s: [a2.copy(), s], _owner((2, 4), "float64"), "values"),
        ("lexsort", {"axis": -1}, lambda s: [np.array([3, 1, 2, 1, 3], dtype=np.int64), s],
         np.array([0, 1, 1, 0, 0], dtype=np.int64), "key"),
        ("convolve", {"mode": "full"}, lambda s: [a1.copy(), s], np.array([1.0, -2.0, 0.5]), "v"),
        ("correlate", {"mode": "valid"}, lambda s: [a1.copy(), s], np.array([1.0, -2.0, 0.5]), "v"),
        ("einsum", {"subscripts": "ij,jk->ik"}, lambda s: [a2.copy(), s], _owner((4, 2), "float64"), "operand1"),
        ("concatenate", {"axis": 0}, lambda s: [a2.copy(), s], _owner((2, 4), "float64"), "second"),
        ("stack", {"axis": 0}, lambda s: [a2.copy(), s], _owner((3, 4), "float64"), "second"),
        ("where", {}, lambda s: [s, a2.copy(), (a2 * -1).copy()], (np.arange(12).reshape(3, 4) % 3 == 0), "condition"),
        ("clip", {}, lambda s: [a2.copy(), s, np.array(2.0)], np.array([[0.0, 0.5, 1.0, 1.5]] * 3), "a_min"),
        ("polyval", {}, lambda s: [np.array([1.0, -2.0, 0.5]), s], np.array([[0.0, 1.0], [2.0, -1.0]]), "x"),
        ("trapezoid", {"axis": -1}, lambda s: [a2.copy(), s], np.array([0.0, 1.0, 3.0, 6.0]), "x"),
        ("put_along_axis", {"axis": 1}, lambda s: [a2.copy(), s, np.array(9.0)],
         np.array([[0], [2], [3]], dtype=np.int64), "indices"),
        ("place", {}, lambda s: [a2.copy(), s, np.array([7.0, 8.0])], (np.arange(12).reshape(3, 4) % 4 == 1), "mask"),
        ("putmask", {}, lambda s: [a2.copy(), s, np.array([7.0, 8.0])], (np.arange(12).reshape(3, 4) % 4 == 1), "mask"),
        ("select", {"nc": 1}, lambda s: [s, a2.copy(), np.array(0.0)], (np.arange(12).reshape(3, 4) % 2 == 0), "condlist"),
        ("ravel_multi_index", {"dims": [3, 4]}, lambda s: [s, np.array([3, 0, 1], dtype=np.int64)],
         np.array([2, 0, 1], dtype=np.int64), "multi_index"),
        # ---- more positions of the keys above ----
        ("clip", {}, lambda s: [a2.copy(), np.array(0.5), s], np.array([[1.0, 1.5, 2.0, 2.5]] * 3), "a_max"),
        ("where", {}, lambda s: [(np.arange(12).reshape(3, 4) % 3 == 0), s, (a2 * -1).copy()], a2.copy(), "x"),
        ("where", {}, lambda s: [(np.arange(12).reshape(3, 4) % 3 == 0), a2.copy(), s], (a2 * -1).copy(), "y"),
        ("choose", {"nc": 2}, lambda s: [(np.arange(12).reshape(3, 4) % 2).astype(np.int64), a2.copy(), s],
         (a2 * 2).copy(), "choice1"),
        ("compress", {"axis": 1}, lambda s: [np.array([True, False, True, True]), s], a2.copy(), "a"),
        ("extract", {}, lambda s: [(np.arange(12).reshape(3, 4) % 3 == 0), s], a2.copy(), "arr"),
        ("select", {"nc": 1}, lambda s: [(np.arange(12).reshape(3, 4) % 2 == 0), s, np.array(0.0)], a2.copy(),
         "choicelist"),
        ("put_along_axis", {"axis": 1}, lambda s: [a2.copy(), np.array([[0], [2], [3]], dtype=np.int64), s],
         np.array([[9.0], [8.0], [7.0]]), "values"),
        # ---- more keys: the products, the set routines, the stacks, cov / corrcoef, ldexp ----
        ("dot", {}, lambda s: [a2.copy(), s], _owner((4, 2), "float64"), "b"),
        ("matmul", {}, lambda s: [a2.copy(), s], _owner((4, 2), "float64"), "b"),
        ("outer", {}, lambda s: [np.array([1.0, -2.0, 0.5]), s], _owner((2, 2), "float64"), "b"),
        ("tensordot", {"axes": 1}, lambda s: [a2.copy(), s], _owner((4, 2), "float64"), "b"),
        ("union1d", {}, lambda s: [a1.copy(), s], np.array([[0.0, 2.0], [7.5, 1.5]]), "ar2"),
        ("intersect1d", {}, lambda s: [a1.copy(), s], np.array([[0.0, 3.0], [7.5, 6.0]]), "ar2"),
        ("setxor1d", {}, lambda s: [a1.copy(), s], np.array([[0.0, 3.0], [7.5, 6.0]]), "ar2"),
        ("setdiff1d", {}, lambda s: [a1.copy(), s], np.array([[0.0, 3.0], [7.5, 6.0]]), "ar2"),
        ("column_stack", {}, lambda s: [a2.copy(), s], _owner((3, 2), "float64"), "second"),
        ("dstack", {}, lambda s: [a2.copy(), s], _owner((3, 4), "float64"), "second"),
        ("cov", {}, lambda s: [_owner((2, 4), "float64"), s], np.array([[1.0, 0.0, 2.0, 2.0], [0.5, 3.0, -1.0, 0.0]]), "y"),
        ("corrcoef", {}, lambda s: [_owner((2, 4), "float64"), s], np.array([[1.0, 0.0, 2.0, 2.0], [0.5, 3.0, -1.0, 0.0]]),
         "y"),
        ("ldexp", {}, lambda s: [a2.copy(), s], (np.arange(12).reshape(3, 4) % 3 - 1).astype(np.int64), "x2"),
    ]
    for op, params, mk, values, pos in jobs:
        npf = _ARRAY_JOB_NUMPY[op]
        for kind in ARR_KINDS:
            sec = _arr_source(kind, np.ascontiguousarray(values))
            if sec is None:
                continue
            ops = mk(sec)
            try:
                descs = [_desc(o) for o in ops]
            except ValueError:
                em.skipped += 1
                continue
            cid = em.next_id(f"{op}/{pos}={kind}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = npf(ops, params)
            except Exception:  # noqa: BLE001
                em.skipped += 1
                continue
            expected = _result_expected(r, ops)
            em.add(_case(cid, op, params, descs, expected, f"{pos}={kind}", "arrayarg"))
    return em.cases, em.gaps


def _mutated(arr):
    """The in-place mutators' result as OpRegistry returns it: the mutated first operand itself."""
    return arr


# The NumPy call each array_args op key makes -- the SAME call the existing generator paired with the key in
# OpRegistry (read there before adding a job: a key whose registry entry composes calls cannot carry a facet).
_ARRAY_JOB_NUMPY = {
    "take": lambda o, p: np.take(o[0], o[1], axis=p["axis"], mode=p.get("mode", "raise")),
    "take_along_axis": lambda o, p: np.take_along_axis(o[0], o[1], axis=p["axis"]),
    "compress": lambda o, p: np.compress(o[0], o[1], axis=p["axis"]),
    "extract": lambda o, p: np.extract(o[0], o[1]),
    "choose": lambda o, p: np.choose(o[0], list(o[1:1 + p["nc"]])),
    "searchsorted": lambda o, p: np.searchsorted(o[0], o[1], side=p["side"], sorter=o[2] if len(o) > 2 else None),
    "digitize": lambda o, p: np.digitize(o[0], o[1], right=p["right"]),
    "isin": lambda o, p: np.isin(o[0], o[1]),
    "bincount": lambda o, p: np.bincount(o[0], o[1] if len(o) > 1 else None, minlength=p["minlength"]),
    "interp": lambda o, p: np.interp(o[0], o[1], o[2]),
    "inner": lambda o, p: np.inner(o[0], o[1]),
    "kron": lambda o, p: np.kron(o[0], o[1]),
    "matvec": lambda o, p: np.matvec(o[0], o[1]),
    "vecmat": lambda o, p: np.vecmat(o[0], o[1]),
    "cross": lambda o, p: np.cross(o[0], o[1]),
    "insert": lambda o, p: np.insert(o[0], p["obj"], o[1], axis=p["axis"]),
    "append": lambda o, p: np.append(o[0], o[1], axis=p["axis"]),
    "lexsort": lambda o, p: np.lexsort(o, axis=p["axis"]),
    "convolve": lambda o, p: np.convolve(o[0], o[1], mode=p["mode"]),
    "correlate": lambda o, p: np.correlate(o[0], o[1], mode=p["mode"]),
    "einsum": lambda o, p: np.einsum(p["subscripts"], *o),
    "concatenate": lambda o, p: np.concatenate([o[0], o[1]], axis=p["axis"]),
    "stack": lambda o, p: np.stack([o[0], o[1]], axis=p["axis"]),
    "where": lambda o, p: np.where(o[0], o[1], o[2]),
    "clip": lambda o, p: np.clip(o[0], o[1], o[2]),
    "polyval": lambda o, p: np.polyval(o[0], o[1]),
    "trapezoid": lambda o, p: np.trapezoid(o[0], o[1], axis=p["axis"]),
    "put_along_axis": lambda o, p: (np.put_along_axis(o[0], o[1], o[2], axis=p["axis"]), _mutated(o[0]))[1],
    "place": lambda o, p: (np.place(o[0], o[1], o[2]), _mutated(o[0]))[1],
    "putmask": lambda o, p: (np.putmask(o[0], o[1], o[2]), _mutated(o[0]))[1],
    "select": lambda o, p: np.select(list(o[:p["nc"]]), list(o[p["nc"]:2 * p["nc"]]), o[2 * p["nc"]]),
    "ravel_multi_index": lambda o, p: np.ravel_multi_index((o[0], o[1]), tuple(p["dims"])),
    "dot": lambda o, p: np.dot(o[0], o[1]),
    "matmul": lambda o, p: np.matmul(o[0], o[1]),
    "outer": lambda o, p: np.outer(o[0], o[1]),
    "tensordot": lambda o, p: np.tensordot(o[0], o[1], axes=p["axes"]),
    "union1d": lambda o, p: np.union1d(o[0], o[1]),
    "intersect1d": lambda o, p: np.intersect1d(o[0], o[1]),
    "setxor1d": lambda o, p: np.setxor1d(o[0], o[1]),
    "setdiff1d": lambda o, p: np.setdiff1d(o[0], o[1]),
    "column_stack": lambda o, p: np.column_stack(list(o)),
    "dstack": lambda o, p: np.dstack([o[0], o[1]]),
    "cov": lambda o, p: np.cov(o[0], o[1] if len(o) > 1 else None),
    "corrcoef": lambda o, p: np.corrcoef(o[0], o[1] if len(o) > 1 else None),
    "ldexp": lambda o, p: np.ldexp(o[0], o[1]),
}


# =============================================================================================================
# Carving: the gaps NumSharp is known to have (filled from the replay of the uncarved matrix)
# =============================================================================================================

# gap id -> one-line description. Drawn from the replay of the uncarved matrix on 2026-10-07 (every carved case was
# checked against that replay: each one diverged, and no kept case did). FuzzCorpusTests.ViewFlags.cs replays
# gaps/<tier>_gaps.jsonl under [TestCategory("OpenBugs")] and prints, per id, how many carved cases still diverge;
# NumSharp.Tests/OpenBugs.ViewFlags.cs reproduces the value bugs through public calls (np.split / np.unstack
# children are C-contiguous views at a non-zero offset). When a gap is fixed, drop its rule and regenerate.
# Redrawn 2026-10-12, when branch strings was rebased onto journey4: its fixes made VF03, VF05, VF11, VF18, VF23,
# VF24, VF25, VF28, VF33, VF35, VF39 and VF40 match NumPy (their rules are dropped) and parts of VF01 (unravel_index
# of a 0-d), VF08 (np.all / np.any / np.count_nonzero), VF12 (np.asarray / np.asanyarray), VF26 (np.resize, np.tile
# of an empty input) and VF32 (np.roll, np.fft.fftshift / ifftshift); the replay of the gap files found which.
GAPS = {
    "VF01": "a NumPy SCALAR result (ufuncs, clip, round, nan_to_num, conj, dot / inner / kron / matmul of 0-d or 1-D "
            "pairs) comes back as a WRITEABLE 0-d array (matmul's also does not own its data)",
    "VF02": "element access that NumPy answers with a scalar COPY (a[i, j], z[()], np.flip(z), np.nan_to_num(z, "
            "copy=False) of a 0-d z) returns a writeable 0-d VIEW of the operand",
    "VF04": "np.squeeze / ndarray.squeeze with no length-1 axis return a new view where NumPy returns the operand itself",
    "VF06": "ufunc results over a dense permuted 3-D input are C-ordered; NumPy keeps the input's permuted (K) layout",
    "VF07": "np.arctan2 of F-ordered inputs returns a C-ordered result; NumPy's (and NumSharp's np.add's) is F",
    "VF08": "np.median / np.nanmedian along an axis of an F or permuted 3-D input return a C-ordered result; NumPy "
            "keeps the input's axis order",
    "VF09": "np.cumsum / np.cumprod along an axis of a permuted (cumprod also F) input return a C-ordered result",
    "VF10": "np.diff of an F or permuted input returns a C-ordered result; NumPy keeps the input's order",
    "VF12": "np.array(a, copy=True) of a permuted or broadcast input is C-ordered; NumPy copies in K order",
    "VF13": "np.array(a, copy=False) returns a new view where NumPy returns the operand itself",
    "VF14": "copies of a read-only array stay READ-ONLY: np.array(ro, copy=True) and np.positive(ro) of a read-only "
            "owner or a same-shape broadcast",
    "VF16": "ndarray.astype(same dtype, copy=False, order='C' or 'F') of a strided 1-D view returns the operand; "
            "NumPy copies to satisfy the requested order",
    "VF17": "np.broadcast_arrays returns new views where NumPy returns the operands themselves, and a read-only "
            "broadcast operand comes back WRITEABLE",
    "VF19": "np.dot / np.inner of a 0-d with an F 2-D array: -0.0 where NumPy's scalar daxpy gives +0.0, and an F "
            "result where NumPy's is C (known: OpenBugs.Reductions Dot_SizeOneOperand_*)",
    "VF20": "np.einsum('i,j->ij'): -0.0 where NumPy's zero-filled accumulator gives +0.0, F-ordered for a C operand "
            "(known: OpenBugs.Reductions Einsum_ZeroFilledAccumulator_*)",
    "VF21": "a[:, idx] (a slice beside an index array) is a C-ordered owning copy; NumPy returns MapIter's "
            "transposed view (known: OpenBugs.FancyIndex.cs)",
    "VF22": "a[:, idx] on a C-contiguous view at a non-zero offset (an np.split child) THROWS "
            "IndexOutOfRangeException",
    "VF26": "results NumPy returns as a VIEW of an internal temporary (np.tile with unit reps of a 0-d input, "
            "np.kron with a 0-d, ndarray.getfield of an empty array, np.nanmedian along an axis — "
            "MaskedArray.filled's _data view) come back owning their data",
    "VF27": "np.matvec / np.vecmat return a view of an internal buffer (OWNDATA false); NumPy's result owns its data",
    "VF29": "np.median / np.nanmedian of a 0-d with keepdims=True are read-only; NumPy returns a writeable 0-d array",
    "VF30": "np.nansum / np.nanprod with axis=None of a view at a non-zero offset (a 0-d np.unstack child, a 2-D "
            "np.split child) read the WRONG elements (the owner's, from its start)",
    "VF31": "np.imag / ndarray.imag of a real F-ordered array (fresh zeros) are C-ordered; NumPy's are K (F)",
    "VF32": "np.nan_to_num(copy=True) of an F input is C-ordered; NumPy allocates in K order",
    "VF34": "np.round of a permuted or broadcast input: decimals=0 is C-ordered where NumPy keeps K, decimals=1 keeps "
            "the input order where NumPy's is C",
    "VF36": "the np.fft transforms (fft / ifft / rfft / irfft / hfft / ihfft along an axis and the 2-D / N-D forms) of "
            "an F, transposed, permuted or broadcast input return C-ordered results; NumPy's gufunc output follows "
            "the input's layout (K)",
    "VF37": "np.nanmean along an axis with keepdims=True returns a view of a reduced buffer (OWNDATA false); NumPy's "
            "result owns its data",
    "VF38": "np.tril / np.triu of a dense permuted 3-D input return a C-ordered result; NumPy's (an np.where) keeps "
            "the input's layout",
    "VO01": "np.linalg.multi_dot writes into an out= that is not C-contiguous (F, stepped, transposed, reversed); "
            "NumPy's final np.dot refuses it with ValueError 'output array is not acceptable (must have the right "
            "datatype, number of dimensions, and be a C-Array)', which np.dot itself reproduces",
    "VS01": "np.zeros / ones / empty / full given a view's Shape of a dense F or permuted layout allocate that "
            "layout; NumPy's np.zeros(x.shape) is always C-ordered",
    "VA01": "np.searchsorted / ndarray.searchsorted read a C-contiguous v / sorter at a non-zero offset (an np.split "
            "child) WRONG",
    "VA02": "np.take_along_axis with F-ordered indices returns a C-ordered result; NumPy's follows the indices (F)",
    "VA03": "np.where reads a C-contiguous condition / x / y at a non-zero offset (an np.split child) WRONG",
    "VA04": "np.cov / np.corrcoef results do not own their data (OWNDATA false): both end with ndarray.squeeze, which "
            "in NumSharp returns a new view where NumPy returns the array itself (VF04)",
    "VA05": "np.ldexp reads a C-contiguous exponent array x2 at a non-zero offset (an np.split child) WRONG",
}

# The ufuncs of the tier (unary and binary), for the rules keyed on "a ufunc".
_UFUNC_FNS = set(UNARY_K) | set(BINARY_K) | {"invert", "bitwise_and"}
# The fns whose 0-d result NumPy hands back as a scalar and NumSharp as a writeable 0-d array (VF01).
_VF01_FNS = _UFUNC_FNS | {"clip", "round", "conj", "conjugate", "ndarray.conj", "ndarray.conjugate", "dot", "inner",
                          "kron", "matmul"}


def _is_scalar_result(expected):
    """True when NumPy returned a scalar (np.generic) — for a tuple, when any slot is one."""
    if "mem" in expected:
        return bool(expected["mem"].get("scalar"))
    return any(s.get("mem", {}).get("scalar") for s in expected.get("slots", []))


def _gap_of(tier, case):
    """The documented gap a case falls into (a GAPS id), or None when NumSharp is expected to match NumPy there.

    Every rule names its id's description in GAPS; the rules were drawn from the replay of the uncarved matrix and
    carve exactly the cases that diverged there (FuzzCorpusTests' gaps replay reports any carved case that matches).

    Parameters:
        tier: views / out / shapes / arrays.
        case: the finished case dict.

    Returns:
        A GAPS id or None.
    """
    p = case["params"]
    fn = p.get("fn", case["op"])
    lay = case["layout"]
    e = case.get("expected") or {}
    if tier == "out":
        # The out tier's one divergence is an error NumPy raises and NumSharp does not, so its rule runs before the
        # early return below (which keeps every other error case in the gate).
        if fn == "multi_dot" and case.get("expects_throw"):
            return "VO01"
        return None
    if case.get("expects_throw"):
        return None
    if tier == "views":
        scalar = _is_scalar_result(e)
        axis = p.get("axis")
        if fn in _VF01_FNS and scalar and not (fn == "nan_to_num"):
            return "VF01"
        if fn == "nan_to_num" and scalar:
            return "VF01" if p.get("copy") else "VF02"
        if fn in ("getitem", "flip") and scalar:
            return "VF02"
        if fn in ("squeeze", "ndarray.squeeze") and "axis" not in p and e["mem"]["same"][0]:
            return "VF04"
        if fn in _UFUNC_FNS and lay in ("view_perm_3d", "view_perm_3d+view_perm_3d"):
            return "VF06"
        if fn == "arctan2" and e["mem"]["flags"][1] and not e["mem"]["flags"][0]:
            return "VF07"
        if fn in ("median", "nanmedian") and axis is not None and (
                lay == "own_f_3d" or (lay == "view_perm_3d" and axis != 0)):
            return "VF08"
        if axis is not None and ((fn == "cumsum" and lay == "view_perm_3d") or (
                fn == "cumprod" and lay in ("own_f_2d", "own_f_3d", "view_t_2d", "view_perm_3d"))):
            return "VF09"
        if fn == "diff" and lay in ("own_f_2d", "view_t_2d", "view_perm_3d"):
            return "VF10"
        if lay in ("view_perm_3d", "view_bcast_2d") and fn == "array" and p.get("copy") is True:
            return "VF12"
        if fn == "array" and p.get("copy") is False:
            return "VF13"
        if lay in ("own_ro_2d", "view_bsame_2d") and ((fn == "array" and p.get("copy") is True) or fn == "positive"):
            return "VF14"
        if (fn == "ndarray.astype" and p.get("copy") is False and p.get("dtype") == "int64"
                and p.get("order") in ("C", "F") and lay in ("view_step_1d", "view_rev_1d")):
            return "VF16"
        if fn == "broadcast_arrays":
            return "VF17"
        if fn in ("dot", "inner") and lay in ("own_0d+own_f_2d", "own_f_2d+own_0d"):
            return "VF19"
        if fn == "einsum" and p.get("subscripts") == "i,j->ij":
            return "VF20"
        if fn == "getitem" and lay.endswith(":slice_fancy"):
            return "VF22" if lay.startswith("view_off_2d") else "VF21"
        if ((fn == "tile" and lay == "own_0d" and p["reps"] == [1])
                or (fn == "kron" and lay == "own_0d+own_f_2d")
                or (fn == "ndarray.getfield" and lay == "own_empty_2d")
                or (fn == "nanmedian" and axis is not None and not p.get("keepdims"))):
            return "VF26"
        if fn in ("median", "nanmedian") and lay in ("own_0d", "view_0d") and p.get("keepdims"):
            return "VF29"
        if fn in ("nansum", "nanprod") and axis is None and lay in ("view_0d", "view_off_2d"):
            return "VF30"
        if fn in ("imag", "ndarray.imag") and lay in ("own_f_2d", "view_t_2d") and case["operands"][0]["dtype"] != "complex128":
            return "VF31"
        if fn == "nan_to_num" and p.get("copy") and lay == "own_f_2d":
            return "VF32"
        if fn == "round" and ((lay == "view_perm_3d") or (lay == "view_bcast_2d" and p["decimals"] == 1)):
            return "VF34"
        # hfft conjugates first, and NumPy's np.conjugate of a broadcast input is C-ordered: that one input matches.
        if (fn.startswith("fft.") and fn not in ("fft.fftshift", "fft.ifftshift")
                and lay in ("own_f_2d", "view_t_2d", "view_perm_3d", "view_bcast_2d")
                and not (fn == "fft.hfft" and lay == "view_bcast_2d")):
            return "VF36"
        # The strided, broadcast and permuted inputs reduce through a materialized copy whose result owns its data.
        if (fn == "nanmean" and axis is not None and p.get("keepdims")
                and lay not in ("view_step_2d", "view_bcast_2d", "view_perm_3d")):
            return "VF37"
        if fn in ("tril", "triu") and lay == "view_perm_3d":
            return "VF38"
        if fn == "ndarray.searchsorted" and lay == "sorted+view_off_2d":
            return "VA01"
        if fn == "where3" and lay == "view_off_2d":
            return "VA03"
        return None
    if tier == "shapes":
        # Only where the view's dense layout is a real F / permuted one: a (1,1) or an empty transposed view's Shape is
        # also C, so NumSharp keeps NumPy's answer there.
        if (fn in ("zeros", "ones", "empty", "full") and p["form"] in ("view:transposed", "view:permuted")
                and p["dims"] in ([2, 3], [2, 3, 4])):
            return "VS01"
        return None
    if tier == "arrays":
        if fn in ("matvec", "vecmat"):
            return "VF27"
        if fn == "searchsorted" and lay in ("v=off", "sorter=off"):
            return "VA01"
        if fn == "take_along_axis" and lay in ("indices=f", "indices=t"):
            return "VA02"
        if fn == "where" and lay in ("condition=off", "x=off", "y=off"):
            return "VA03"
        if fn in ("cov", "corrcoef"):
            return "VA04"
        if fn == "ldexp" and lay == "x2=off":
            return "VA05"
        return None
    return None


# =============================================================================================================
# Main
# =============================================================================================================

TIERS = {
    "views": ("view_flags.jsonl", gen_views),
    "out": ("out_flags.jsonl", gen_out),
    "shapes": ("shape_args.jsonl", gen_shapes),
    "arrays": ("array_args.jsonl", gen_arrays),
}


def write_jsonl(path, cases):
    """Write cases as JSONL (LF line ends, compact separators) and report the count."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    """Generate the requested tiers (all by default); see the module docstring for the flags."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("mode", nargs="?", default="all", choices=["all"] + list(TIERS))
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    args = ap.parse_args()
    modes = list(TIERS) if args.mode == "all" else [args.mode]
    for m in modes:
        name, gen = TIERS[m]
        if args.uncarved:
            cases, _ = gen(carve=False)
            write_jsonl(os.path.join(args.uncarved, name), cases)
            continue
        cases, gaps = gen(carve=True)
        write_jsonl(os.path.join(CORPUS, name), cases)
        write_jsonl(os.path.join(GAP_DIR, name.replace(".jsonl", "_gaps.jsonl")), gaps)


if __name__ == "__main__":
    main()
