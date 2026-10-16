"""Randomized soak of the string oracle's computing sections (strings phase 2): fresh random texts, NA kinds, layouts,
sizes and keyword combinations for every family the committed sections gate one fixed matrix of — the sorts and
partitions, ``np.lexsort``, the searches, ``np.unique`` and its Array-API forms, ``np.isin`` and the set routines,
``np.sort_complex``, the histograms, the quantile family and ``np.median`` (fixed-width ``S`` / ``U`` and NEP 55
``StringDType``), and ``StringDType``'s comparisons, arithmetic, truth readers, reductions and numpy.ma members.

Every case is recorded by ``gen_strings_oracle.add`` (NumPy 2.4.2's answer or its exact error) under the op keys the C#
replay already knows, in a section named ``soak<seed>``, and written to ``<out>/soak_<seed>.jsonl``.
``StringOracleTests.StringSoak`` replays every ``*.jsonl`` file of the directory in ``NUMSHARP_STRING_ORACLE_SOAK_DIR``
with the committed sections' comparator; every case is phase 1, so a deferral is a failure.

Usage::

    python test/oracle/soak_strings_oracle.py --seeds 1,2,3 --out DIR [--scale 1.0] [--families sort,sets]

The script avoids the inputs that crash or misbehave in NumPy 2.4.2 itself (the committed generator documents each):
``np.lexsort`` over a ``StringDType`` key that is not contiguous along the axis (an access violation), the argsort of a
broadcast ``StringDType`` view holding arena texts, ``np.searchsorted`` over a haystack text outside its cell, the
``StringDType``-to-complex cast (zeros only), and the nan-forms' ``apply_along_axis`` object result for a ``None`` row.
Each family of each seed runs in its own process (``--worker``), so a NumPy crash costs one family, not the soak.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gen_strings_oracle as g  # noqa: E402 -- the case recorder, the operand describers and the NA specs

TD, TDN, TDNAN, TDS, TDC = g.TD, g.TDN, g.TDNAN, g.TDS, g.TDC
T_SPECS = [TD, TDN, TDNAN, TDS, TDC]
_ABSENT = g._ABSENT

# ----------------------------------------------------------------------------------------------------------------------
# Random values
# ----------------------------------------------------------------------------------------------------------------------

# Characters of the texts: a small Latin core (so duplicates, prefixes and equal leading bytes are frequent), the NUL
# (embedded NULs order texts: 'a' < 'a\x00' < 'ab'), a space, two- / three- / four-byte UTF-8 characters.
CHARS = ["a", "b", "c", "a", "b", "z", "\x00", " ", "é", "中", "\U0001F600", "A", "0", "1", "x"]
# Bytes of an S text: the Latin core, the NUL, a space, high bytes (unsigned order) and DEL.
BYTES = [b"a", b"b", b"c", b"a", b"z", b"\x00", b" ", b"\xe9", b"\xff", b"\x7f", b"A", b"0"]


def _length(rng, long_ok=True):
    """A text length: mostly 0-3 characters, then the 15 / 16-byte cell boundary, arena lengths and (rarely) heap ones.

    :param rng: the generator.
    :param long_ok: whether a heap-sized length (250+) may be drawn.
    :returns: the length in characters.
    """
    r = rng.random()
    if r < 0.45:
        return int(rng.integers(0, 4))
    if r < 0.70:
        return int(rng.integers(4, 9))
    if r < 0.88:
        return int(rng.integers(13, 18))
    if r < 0.98 or not long_ok:
        return int(rng.integers(18, 40))
    return int(rng.integers(250, 310))


def rand_text(rng, long_ok=True):
    """A random ``str``.

    :param rng: the generator.
    :param long_ok: whether a heap-sized text may be drawn.
    :returns: the text.
    """
    return "".join(CHARS[int(rng.integers(0, len(CHARS)))] for _ in range(_length(rng, long_ok)))


def rand_bytes(rng):
    """A random ``bytes`` of at most 20 bytes.

    :param rng: the generator.
    :returns: the bytes.
    """
    n = int(rng.integers(0, 4)) if rng.random() < 0.6 else int(rng.integers(4, 21))
    return b"".join(BYTES[int(rng.integers(0, len(BYTES)))] for _ in range(n))


def vocabulary(rng, k, kind, short=False):
    """``k`` random texts plus derived ones (a prefix, a NUL appended, a character appended), so a draw from it repeats
    texts and meets prefixes.

    :param rng: the generator.
    :param k: the number of base texts.
    :param kind: ``"T"`` / ``"U"`` (str) or ``"S"`` (bytes).
    :param short: texts of at most 15 UTF-8 bytes only (a StringDType cell): broadcast views and searched haystacks.
    :returns: the list.
    """
    out = []
    for _ in range(k):
        if kind == "S":
            t = rand_bytes(rng)
            out += [t, t[:-1], t + b"\x00", t + b"a"]
        else:
            t = rand_text(rng, long_ok=not short)
            out += [t, t[:-1], t + "\x00", t + "a"]
    if short and kind != "S":
        out = [t for t in out if len(t.encode("utf-8")) <= 15]
        if not out:
            out = ["", "a"]
    return out


def rand_values(rng, n, kind, spec=None, na_rate=0.0, short=False):
    """``n`` values for an array: draws from a fresh vocabulary (70 %) or fresh texts, and NA objects at ``na_rate``.

    :param rng: the generator.
    :param n: the count.
    :param kind: ``"T"``, ``"U"`` or ``"S"``.
    :param spec: the StringDType spec (``"T"``).
    :param na_rate: the probability of an NA object (a spec with one).
    :param short: cell-sized texts only.
    :returns: the values.
    """
    vocab = vocabulary(rng, int(rng.integers(1, 8)), kind, short)
    vals = []
    for _ in range(n):
        if kind == "T" and spec is not None and spec.na is not _ABSENT and rng.random() < na_rate:
            vals.append(spec.na)
        elif rng.random() < 0.7 or kind == "S":
            vals.append(vocab[int(rng.integers(0, len(vocab)))])
        else:
            t = rand_text(rng, long_ok=not short)
            vals.append(t if not short or len(t.encode("utf-8")) <= 15 else t[:3])
    return vals


def make_base(vals, kind, spec=None):
    """A 1-D C-contiguous base array of ``vals``.

    :param vals: the values.
    :param kind: ``"T"``, ``"U"`` or ``"S"``.
    :param spec: the StringDType spec (``"T"``).
    :returns: the array (an empty one keeps the dtype).
    """
    if kind == "T":
        dt = spec.build()
        return np.array(vals, dtype=dt) if vals else np.zeros(0, dtype=dt)
    if kind == "U":
        w = max([len(v) for v in vals] + [1])
        return np.array(vals, dtype=f"<U{w}") if vals else np.zeros(0, dtype="<U1")
    w = max([len(v) for v in vals] + [1])
    return np.array(vals, dtype=f"|S{w}") if vals else np.zeros(0, dtype="|S1")


# ----------------------------------------------------------------------------------------------------------------------
# Random layouts: every view addresses its base's buffer (describe() requires it)
# ----------------------------------------------------------------------------------------------------------------------

def _dims(rng, k):
    """``k`` random dimension lengths of 1-5.

    :param rng: the generator.
    :param k: the rank.
    :returns: the list.
    """
    return [int(rng.integers(1, 6)) for _ in range(k)]


# Each layout returns (base length, view builder) for a random size.
def layout_spec(rng, name):
    """The base length and view builder of one random instance of the layout ``name``.

    :param rng: the generator.
    :param name: the layout kind.
    :returns: ``(n, build)``.
    """
    if name == "c1":
        n = int(rng.integers(0, 40)) if rng.random() < 0.9 else int(rng.integers(40, 400))
        return n, lambda b: b
    if name == "step":
        n = int(rng.integers(2, 30))
        s, k = int(rng.integers(0, 2)), int(rng.integers(2, 4))
        return n, lambda b: b[s::k]
    if name == "rev":
        # A base of two or more: b[::-1] of a one-element base is rebuilt as the base itself (see "slice").
        n = int(rng.integers(2, 30))
        return n, lambda b: b[::-1]
    if name == "revstep":
        n = int(rng.integers(3, 30))
        k = int(rng.integers(2, 4))
        return n, lambda b: b[n - 1::-k]
    if name == "slice":
        # Never the whole base: the harness rebuilds a 1-D stride-1 view of every cell as the base itself (the descriptor
        # cannot tell them apart), so its OWNDATA would be True where NumPy's view says False.
        n = int(rng.integers(3, 30))
        i = int(rng.integers(0, n // 2))
        j = int(rng.integers(i, n + 1))
        if i == 0 and j == n:
            j = n - 1
        return n, lambda b: b[i:j]
    if name == "c2":
        r, c = _dims(rng, 2)
        extra = int(rng.integers(0, 3))
        return r * c + extra, lambda b: b[extra:].reshape(r, c)
    if name == "f2":
        r, c = _dims(rng, 2)
        return r * c, lambda b: b.reshape(c, r).T
    if name == "c3":
        d = _dims(rng, 3)
        return int(np.prod(d)), lambda b: b.reshape(d)
    if name == "t3":
        d = _dims(rng, 3)
        perm = list(rng.permutation(3))
        return int(np.prod(d)), lambda b: b.reshape(d).transpose(perm)
    if name == "s2":
        r, c = int(rng.integers(2, 7)), int(rng.integers(2, 7))
        return r * c, lambda b: b.reshape(r, c)[::2, 1::2]
    if name == "r2":
        r, c = _dims(rng, 2)
        return r * c, lambda b: b.reshape(r, c)[::-1, ::-1]
    if name == "col":
        r, c = int(rng.integers(1, 6)), int(rng.integers(2, 6))
        j = int(rng.integers(0, c))
        return r * c, lambda b: b.reshape(r, c)[:, j]
    if name == "row":
        r, c = int(rng.integers(2, 6)), int(rng.integers(1, 6))
        i = int(rng.integers(0, r))
        return r * c, lambda b: b.reshape(r, c)[i]
    if name == "nx":
        r, c = _dims(rng, 2)
        return r * c, lambda b: b.reshape(r, c)[:, None, :]
    if name == "bc":
        r, c = int(rng.integers(2, 5)), int(rng.integers(1, 6))
        return c, lambda b: np.broadcast_to(b, (r, c))
    if name == "bcol":
        r, c = int(rng.integers(1, 6)), int(rng.integers(2, 5))
        return r, lambda b: np.broadcast_to(b[:, None], (r, c))
    if name == "z0":
        n = int(rng.integers(1, 6))
        i = int(rng.integers(0, n))
        return n, lambda b: b[i, ...]
    if name == "one":
        # A base of two or more: b[0:1] of a one-element base is the whole base (see "slice").
        n = int(rng.integers(2, 6))
        i = int(rng.integers(0, n))
        return n, lambda b: b[i:i + 1]
    if name == "d5":
        a, c, d = _dims(rng, 3)
        return a * c * d, lambda b: b.reshape(1, a, 1, c, d)
    if name == "e1":
        return 0, lambda b: b
    if name == "e2":
        k = int(rng.integers(1, 4))
        return 0, lambda b: b.reshape(0, k)
    if name == "e3":
        r, c = _dims(rng, 2)
        return r * c, lambda b: b.reshape(r, c)[:, c:]
    raise ValueError(name)


LAYOUT_NAMES = ["c1", "c1", "c1", "step", "rev", "revstep", "slice", "c2", "c2", "f2", "c3", "t3", "s2", "r2", "col",
                "row", "nx", "bc", "bcol", "z0", "one", "d5", "e1", "e2", "e3"]
BROADCAST = ("bc", "bcol")
# The layouts that are not C-contiguous: NumPy 2.4.2's np.argsort(v, axis=None) of such a StringDType view misorders
# its arena texts (np.argsort(v.ravel()), np.sort(v, axis=None), an axis, the partitions and ndarray.sort all order them
# right; measured over 12,000 random views), so the sort family skips that one call when the view holds an arena text —
# a NumPy bug NumSharp does not reproduce.
NON_CONTIGUOUS = ("step", "rev", "revstep", "f2", "t3", "s2", "r2", "col", "bc", "bcol")


def rand_operand_built(rng, kind, spec=None, layouts=None, na_rate=None, short=False):
    """A random (base, view) operand of ``kind`` and the builder that made the view (so an in-place call can be given a
    fresh copy of the same operand).

    :param rng: the generator.
    :param kind: ``"T"``, ``"U"`` or ``"S"``.
    :param spec: the StringDType spec (``"T"``; random when None).
    :param layouts: the layout names to draw from (``LAYOUT_NAMES`` by default).
    :param na_rate: the NA probability (random when None).
    :param short: cell-sized texts only.
    :returns: ``((base, view), layout name, spec, build)``.
    """
    name = (layouts or LAYOUT_NAMES)[int(rng.integers(0, len(layouts or LAYOUT_NAMES)))]
    if kind == "T" and spec is None:
        spec = T_SPECS[int(rng.integers(0, len(T_SPECS)))]
    if na_rate is None:
        na_rate = [0.0, 0.1, 0.3][int(rng.integers(0, 3))]
    n, build = layout_spec(rng, name)
    # A broadcast StringDType view holds cell texts only (NumPy's argsort misorders a broadcast view's arena texts).
    vals = rand_values(rng, n, kind, spec, na_rate, short=short or (kind == "T" and name in BROADCAST))
    base = make_base(vals, kind, spec)
    return (base, build(base)), name, spec, build


def rand_operand(rng, kind, spec=None, layouts=None, na_rate=None, short=False):
    """A random (base, view) operand of ``kind`` (``rand_operand_built`` without the builder).

    :param rng: the generator.
    :param kind: ``"T"``, ``"U"`` or ``"S"``.
    :param spec: the StringDType spec (``"T"``; random when None).
    :param layouts: the layout names to draw from (``LAYOUT_NAMES`` by default).
    :param na_rate: the NA probability (random when None).
    :param short: cell-sized texts only.
    :returns: ``((base, view), layout name, spec)``.
    """
    op, name, spec, _ = rand_operand_built(rng, kind, spec, layouts, na_rate, short)
    return op, name, spec


def label(kind, spec, name):
    """A case's layout label: the dtype kind (with the StringDType NA kind) and the layout name.

    :param kind: ``"T"``, ``"U"`` or ``"S"``.
    :param spec: the StringDType spec, or None.
    :param name: the layout name.
    :returns: the label.
    """
    return f"{g._t_label(spec) if kind == 'T' else kind}:{name}"


def has_none_na(spec, view):
    """Whether a StringDType view holds a ``None``-like NA (its NA object is None, and some element is missing).

    :param spec: the spec, or None.
    :param view: the array.
    :returns: the flag.
    """
    if spec is None or spec.na is not None or view.size == 0:
        return False
    flat = np.array(view, copy=True, order="C").reshape(-1)
    cells = flat.tobytes()
    return any(cells[16 * i + 15] & 0x80 for i in range(flat.size))


def pick(rng, seq):
    """A random element of ``seq``.

    :param rng: the generator.
    :param seq: a non-empty sequence.
    :returns: the element.
    """
    return seq[int(rng.integers(0, len(seq)))]


def rand_kind(rng, kinds=("T", "T", "U", "S")):
    """A random text kind (StringDType twice as often as each fixed-width kind).

    :param rng: the generator.
    :param kinds: the kinds to draw from.
    :returns: the kind.
    """
    return pick(rng, kinds)


# ----------------------------------------------------------------------------------------------------------------------
# Families
# ----------------------------------------------------------------------------------------------------------------------

def fam_sort(rng, sec, count):
    """The sort family: np.sort / np.argsort (axis, kind, stable), np.partition / np.argpartition (kth), ndarray.sort."""
    for _ in range(count):
        kind = rand_kind(rng)
        (base, view), name, spec, build = rand_operand_built(rng, kind)
        lab = label(kind, spec, name)
        # np.argsort(view, axis=None) of a non-contiguous StringDType view with an arena text: NumPy's misorder (above).
        arena_view = kind == "T" and name in NON_CONTIGUOUS             and any(len(t.encode("utf-8", "surrogatepass")) > 15 for t in _flat_texts(view))
        axes = [None] + list(range(-view.ndim, view.ndim)) if view.ndim else [None, -1, 0]
        ax = pick(rng, axes)
        kw = {}
        r = rng.random()
        if r < 0.25:
            kw["kind"] = pick(rng, ["quicksort", "stable", "mergesort", "heapsort", None])
        elif r < 0.4:
            kw["stable"] = pick(rng, [True, False, None])
        op = (base, view)
        if not (ax is None and arena_view):
            g.add(sec, "argsort_kw", params=g._sort_params(kw, axis=ax), operands=[op], layout=lab,
                  call=lambda a, x=ax, k=kw: np.argsort(a, axis=x, **k))
        g.add(sec, "sort_kw", params=g._sort_params(kw, axis=ax), operands=[op], layout=lab,
              call=lambda a, x=ax, k=kw: np.sort(a, axis=x, **k))
        if view.ndim:
            n_axis = view.size if ax is None else view.shape[ax]
            kth = int(rng.integers(-n_axis, n_axis)) if n_axis else int(rng.integers(-1, 2))
            if rng.random() < 0.1:
                kth = n_axis + int(rng.integers(0, 3))  # out of bounds: NumPy's ValueError
            g.add(sec, "partition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=lab,
                  call=lambda a, x=ax, k=kth: np.partition(a, k, axis=x))
            g.add(sec, "argpartition_kw", params={"axis": ax, "kth": kth}, operands=[op], layout=lab,
                  call=lambda a, x=ax, k=kth: np.argpartition(a, k, axis=x))
        if ax is not None and view.ndim and view.flags.writeable:
            # The in-place sort mutates its operand: a fresh copy of the same base, viewed by the same builder.
            fresh = base.copy()
            g.add(sec, "ndsort_kw", params=g._sort_params(kw, axis=ax), operands=[(fresh, build(fresh))], layout=lab,
                  mutate=0, call=lambda a, x=ax, k=kw: a.sort(axis=x, **k))


def fam_lexsort(rng, sec, count):
    """np.lexsort over one to three keys of StringDType / U / S / integer / float kinds (StringDType keys contiguous
    along the axis: NumPy 2.4.2 crashes otherwise)."""
    for _ in range(count):
        n = int(rng.integers(0, 40)) if rng.random() < 0.85 else int(rng.integers(40, 300))
        two_d = rng.random() < 0.25 and n >= 2
        rows = int(rng.integers(1, 4)) if two_d else 1
        keys, labels = [], []
        for _ in range(int(rng.integers(1, 4))):
            k = pick(rng, ["T", "T", "U", "S", "i", "f"])
            if k in ("T", "U", "S"):
                spec = pick(rng, T_SPECS) if k == "T" else None
                vals = rand_values(rng, rows * n, k, spec, pick(rng, [0.0, 0.2, 0.5]))
                arr = make_base(vals, k, spec)
                labels.append(g._t_label(spec) if k == "T" else k)
            elif k == "i":
                arr = rng.integers(0, 3, rows * n)
                labels.append("i")
            else:
                arr = rng.integers(0, 3, rows * n).astype(float)
                arr[rng.random(rows * n) < 0.1] = np.nan
                labels.append("f")
            if two_d:
                arr = arr.reshape(rows, n)
            keys.append(g.t_plain(arr) if arr.dtype.kind == "T" else g.plain(arr))
        ax = -1 if not two_d else pick(rng, [-1, 1])
        g.add(sec, "lexsort", params={"axis": ax}, operands=keys, layout=",".join(labels) + f":n{n}x{rows}",
              call=lambda *ks, x=ax: np.lexsort(ks, axis=x))


def fam_search(rng, sec, count):
    """np.searchsorted (sides, sorters, the method form) over sorted or unsorted haystacks of cell texts, and
    np.digitize."""
    for _ in range(count):
        kind = rand_kind(rng)
        spec = pick(rng, T_SPECS) if kind == "T" else None
        n = int(rng.integers(0, 30))
        hay = make_base(rand_values(rng, n, kind, spec, pick(rng, [0.0, 0.2]), short=True), kind, spec)
        if rng.random() < 0.8:
            try:
                hay = hay[np.argsort(hay, kind="stable")]
            except Exception:  # noqa: BLE001 -- a None-like NA: searched as is
                pass
        hay_op = (hay, hay)
        kkind = kind if rng.random() < 0.8 else rand_kind(rng)
        (kb, kv), kname, kspec = rand_operand(rng, kkind)
        side = pick(rng, ["left", "right"])
        lab = f"{label(kind, spec, 'hay')}{n}:{label(kkind, kspec, kname)}"
        if n and rng.random() < 0.25:
            perm = rng.permutation(n)
            shuffled = hay[perm].copy()
            sorter = np.argsort(perm).astype(np.int64)
            g.add(sec, "searchsorted_kw", params={"side": side, "sorter": True},
                  operands=[(shuffled, shuffled), (kb, kv), g.own(sorter)], layout=lab + ":sorter",
                  call=lambda h, k, s, sd=side: np.searchsorted(h, k, side=sd, sorter=s))
        elif rng.random() < 0.2:
            g.add(sec, "ndsearchsorted", params={"side": side, "sorter": False}, operands=[hay_op, (kb, kv)],
                  layout=lab, call=lambda h, k, sd=side: h.searchsorted(k, side=sd))
        else:
            g.add(sec, "searchsorted_kw", params={"side": side, "sorter": False}, operands=[hay_op, (kb, kv)],
                  layout=lab, call=lambda h, k, sd=side: np.searchsorted(h, k, side=sd))
        # np.digitize: text data against numeric or text bins.
        (xb, xv), xname, xspec = rand_operand(rng, rand_kind(rng))
        if rng.random() < 0.5:
            bins = g.num_operand(np.sort(rng.random(int(rng.integers(0, 4)))) if rng.random() < 0.7
                                 else rng.random(int(rng.integers(0, 4))))
        else:
            bk = rand_kind(rng)
            bspec = pick(rng, T_SPECS) if bk == "T" else None
            bvals = rand_values(rng, int(rng.integers(0, 5)), bk, bspec, 0.2)
            barr = make_base(bvals, bk, bspec)
            bins = (barr, barr)
        right = bool(rng.random() < 0.5)
        g.add(sec, "digitize", params={"right": right}, operands=[(xb, xv), bins], layout=f"{xname}",
              call=lambda a, b, r=right: np.digitize(a, b, right=r))


FLAGS = [(False, False, False), (True, False, False), (False, True, False), (False, False, True), (True, True, True),
         (True, False, True), (False, True, True), (True, True, False)]


def fam_unique(rng, sec, count):
    """np.unique (every flag, axis, equal_nan, sorted) and the Array-API unique family over every layout."""
    for _ in range(count):
        kind = rand_kind(rng)
        (base, view), name, spec = rand_operand(rng, kind)
        lab = label(kind, spec, name)
        ri, rv, rc = pick(rng, FLAGS)
        ax = None
        if view.ndim and rng.random() < 0.3:
            ax = int(rng.integers(-view.ndim, view.ndim))
        params = {"ri": ri, "rv": rv, "rc": rc, "axis": ax}
        kw = {}
        if rng.random() < 0.4:
            params["equal_nan"] = kw["equal_nan"] = bool(rng.random() < 0.5)
        if rng.random() < 0.3:
            params["sorted"] = kw["sorted"] = bool(rng.random() < 0.5)
        # The values-only call takes NumPy's hash route (has_strncmp_ties); a 1-D input's axis is only checked, NumPy's
        # flat path runs.
        if not (ri or rv or rc) and (ax is None or view.ndim == 1) and has_strncmp_ties(view):
            params["hash_ties"] = True
        g.add(sec, "unique", params=params, operands=[(base, view)], layout=lab,
              call=lambda a, i=ri, v=rv, c=rc, x=ax, k=kw: np.unique(a, return_index=i, return_inverse=v,
                                                                     return_counts=c, axis=x, **k))
        fname = pick(rng, ["unique_values", "unique_counts", "unique_inverse", "unique_all"])
        g.add(sec, "unique_api", params={"name": fname}, operands=[(base, view)], layout=lab,
              call=lambda a, f=getattr(np, fname): f(a))


def _flat_texts(values):
    """The str values of a (possibly nested) list or an array, in order (NA objects that are not str skipped).

    :param values: a list (nested for 2-D), an ndarray, or None.
    :returns: the list of str values.
    """
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        if values.dtype.kind not in "TU":
            return []
        values = values.ravel().tolist()
    out = []
    for v in values:
        if isinstance(v, (list, tuple)):
            out.extend(_flat_texts(list(v)))
        elif isinstance(v, str):
            out.append(v)
    return out


def has_strncmp_ties(*value_sets):
    """Whether the texts hold two DIFFERENT texts that NumPy's NpyString_cmp calls equal (strncmp over the common
    length stops at a NUL both share, then the sizes decide: 'a\x00b' and 'a\x00c' are tied). NumPy's values-only
    np.unique hashes the full bytes and keeps both, then sorts them unstably, so their order and the representative the
    set routines keep are the C++ hash set's: such a case is marked ``hash_ties`` and the replay accepts a tied element.

    :param value_sets: lists / arrays of values (any kinds; only str values count).
    :returns: True when a tie exists.
    """
    seen = {}
    for vs in value_sets:
        for t in _flat_texts(vs):
            b = t.encode("utf-8", "surrogatepass")
            k = b.find(b"\x00")
            key = (b[:k + 1] if k >= 0 else b, len(b))
            if seen.setdefault(key, b) != b:
                return True
    return False


def _set_operand(rng, prefer=None):
    """A set-routine operand: StringDType / U / S text (any layout), or (rarely) numbers.

    :param rng: the generator.
    :param prefer: the kind to use, or None for a random one.
    :returns: ``((base, view), label)``.
    """
    k = prefer or pick(rng, ["T", "T", "U", "S", "T", "i", "f"])
    if k in ("i", "f"):
        arr = rng.integers(-2, 4, int(rng.integers(0, 8)))
        if k == "f":
            arr = arr.astype(float)
        return g.own(arr), k
    (b, v), name, spec = rand_operand(rng, k, layouts=["c1", "c1", "c2", "f2", "step", "rev", "z0", "e1", "bc"])
    return (b, v), label(k, spec, name)


def fam_sets(rng, sec, count):
    """np.isin (assume_unique, invert, kind) and np.intersect1d / union1d / setdiff1d / setxor1d (assume_unique,
    return_indices) between StringDType, U, S and numbers."""
    for _ in range(count):
        same = rng.random() < 0.6
        x, xl = _set_operand(rng)
        y, yl = _set_operand(rng, prefer=xl.split(":")[0][0] if same and xl[0] in "TUS" else None)
        lab = f"{xl}|{yl}"
        au = bool(rng.random() < 0.2)
        iv = bool(rng.random() < 0.3)
        kd = pick(rng, [None, None, "sort", "table"])
        g.add(sec, "isin", params={"assume_unique": au, "invert": iv, "kind": kd}, operands=[x, y], layout=lab,
              call=lambda a, b, u=au, i=iv, k=kd: np.isin(a, b, assume_unique=u, invert=i, kind=k))
        name = pick(rng, ["intersect1d", "union1d", "setdiff1d", "setxor1d"])
        ri = bool(name == "intersect1d" and rng.random() < 0.3)
        au2 = bool(name != "union1d" and rng.random() < 0.2)
        params = {"name": name, "assume_unique": au2, "return_indices": ri}
        # union1d, and intersect1d / setdiff1d without assume_unique or indices, run the values-only np.unique (its hash
        # route: has_strncmp_ties) and keep its order / representative.
        if (name == "union1d" or (name in ("intersect1d", "setdiff1d") and not au2 and not ri)) \
                and has_strncmp_ties(x[1], y[1]):
            params["hash_ties"] = True
        if name == "intersect1d":
            call = lambda a, b, u=au2, r=ri: np.intersect1d(a, b, assume_unique=u, return_indices=r)
        elif name == "union1d":
            call = lambda a, b: np.union1d(a, b)
        else:
            call = lambda a, b, f=getattr(np, name), u=au2: f(a, b, assume_unique=u)
        g.add(sec, "setop", params=params, operands=[x, y], layout=lab, call=call)


COMPLEX_TEXTS = ["1", "-2.5", "3+4j", "j", "-j", "1e1", "nan", "inf", "-inf", "0", "+0", " 2 ", "1+2J", "x", "",
                 "(1+2j)", "1_0", "0x10", "1e1e1", "2j", "-0"]


def fam_sort_complex(rng, sec, count):
    """np.sort_complex over U / S numeric texts (the parse and its errors) and StringDType zeros (NumPy 2.4.2's
    StringDType-to-complex cast stores the real part in both parts: only a +0 real compares as a value — "-0" would give
    NumPy an imaginary -0, which NumSharp's correct cast does not reproduce)."""
    for _ in range(count):
        kind = pick(rng, ["T", "U", "S"])
        n = int(rng.integers(0, 8))
        if kind == "T":
            spec = pick(rng, [TD, TDN, TDNAN, TDS])
            pool = ["0", "+0", "0.0", " 0 ", "x", "1e1e1"] if rng.random() < 0.3 else ["0", "+0", "0.0", " 0 "]
            vals = [pick(rng, pool) for _ in range(n)]
            if spec.na is not _ABSENT and n and rng.random() < 0.3:
                vals[int(rng.integers(0, n))] = spec.na
            arr = make_base(vals, "T", spec)
            lab = g._t_label(spec)
        else:
            vals = [pick(rng, COMPLEX_TEXTS) for _ in range(n)]
            if kind == "S":
                vals = [v.encode() for v in vals]
            arr = make_base(vals, kind)
            spec, lab = None, kind
        if arr.size >= 4 and rng.random() < 0.3:
            arr = arr[: (arr.size // 2) * 2].reshape(2, -1)
        op = g.t_plain(arr) if arr.dtype.kind == "T" else g.plain(arr)
        g.add(sec, "sort_complex", operands=[op], layout=f"{lab}:{arr.shape}", call=lambda a: np.sort_complex(a))


def fam_hist(rng, sec, count):
    """np.histogram / histogram_bin_edges over text data: text edges of every kind, integer and 'auto' bins, ranges,
    density and weights."""
    for _ in range(count):
        kind = rand_kind(rng)
        spec = pick(rng, T_SPECS[:4]) if kind == "T" else None
        r = rng.random()
        ek = (kind if rng.random() < 0.8 else rand_kind(rng)) if r < 0.6 else None
        # Text edges search the SORTED DATA (np.histogram's _search_sorted_inclusive): with a StringDType side NumPy
        # 2.4.2's binary search loads the haystack's cells with the KEYS' allocator, so an arena text in the data reads
        # garbage (MemoryError, or counts from foreign bytes). The data then holds cell texts only.
        vals = rand_values(rng, int(rng.integers(0, 12)), kind, spec, pick(rng, [0.0, 0.2]),
                           short=ek is not None and "T" in (kind, ek))
        data = make_base(vals, kind, spec)
        ops = [(data, data)]
        params = {"fn": pick(rng, ["histogram", "histogram", "histogram_bin_edges"]), "range": None,
                  "density": bool(rng.random() < 0.3)}
        if r < 0.6:
            espec = pick(rng, T_SPECS[:4]) if ek == "T" else None
            edges = make_base(rand_values(rng, int(rng.integers(0, 5)), ek, espec, 0.1), ek, espec)
            if rng.random() < 0.7 and edges.size:
                try:
                    edges = edges[np.argsort(edges, kind="stable")]
                except Exception:  # noqa: BLE001 -- an unsortable edge list stays as drawn
                    pass
            params["bins_op"] = len(ops)
            params["bins"] = None
            ops.append((edges, edges))
        elif r < 0.85:
            params["bins"] = int(rng.integers(1, 4))
        else:
            params["bins"] = "auto"
        if rng.random() < 0.2:
            params["range"] = [0.0, 1.0]
        if data.size and rng.random() < 0.2:
            params["weights_op"] = len(ops)
            w = np.ones(data.size) if rng.random() < 0.7 else make_base(["x"] * data.size, "T", TD)
            ops.append((w, w))

        def call(*views, p=params):
            a = views[0]
            b = views[p["bins_op"]] if "bins_op" in p else p["bins"]
            w = views[p["weights_op"]] if "weights_op" in p else None
            rr = tuple(p["range"]) if p["range"] is not None else None
            if p["fn"] == "histogram_bin_edges":
                return np.histogram_bin_edges(a, b, rr, w)
            return np.histogram(a, b, rr, density=p["density"], weights=w)

        g.add(sec, "hist", params=params, operands=ops, layout=label(kind, spec, f"n{data.size}"), call=call)


METHODS = ["inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf", "hazen",
           "weibull", "linear", "median_unbiased", "normal_unbiased", "lower", "higher", "midpoint", "nearest"]


def fam_quant(rng, sec, count):
    """np.quantile / np.percentile / np.nanquantile / np.nanpercentile (every method, q forms, axes, keepdims, out=)
    and np.median / np.nanmedian over text of every layout."""
    for _ in range(count):
        kind = rand_kind(rng)
        (base, view), lname, spec = rand_operand(rng, kind)
        name = pick(rng, ["quantile", "nanquantile", "percentile", "nanpercentile"])
        axis = None
        if view.ndim and rng.random() < 0.5:
            axis = int(rng.integers(-view.ndim, view.ndim))
            if view.ndim >= 2 and rng.random() < 0.3:
                axis = (0, view.ndim - 1)
        # NumPy's nan forms along an axis return an object array for a None-like NA row (apply_along_axis): skipped.
        if name.startswith("nan") and axis is not None and has_none_na(spec, view):
            continue
        scale = 100.0 if "percentile" in name else 1.0
        qk = pick(rng, ["f", "f", "fl", "arr", "i"])
        params = {"name": name, "qkind": qk, "method": pick(rng, METHODS), "keepdims": bool(rng.random() < 0.3),
                  "axis": list(axis) if isinstance(axis, tuple) else axis}
        ops = [(base, view)]
        if qk == "arr":
            q = (rng.integers(0, 2, int(rng.integers(0, 3))) * scale).astype(int if rng.random() < 0.5 else float)
            params["q_op"] = 1
            ops.append(g.own(q))
        elif qk == "fl":
            params["q"] = [float(x) for x in np.round(rng.random(int(rng.integers(1, 3))), 2) * scale]
        elif qk == "i":
            params["q"] = int(rng.integers(0, 2))
        else:
            params["q"] = float(np.round(rng.random(), 2) * scale)

        # NumPy's weights= (method inverted_cdf; another method is its ValueError): fractional weights of the data's shape
        # or of the reduced axes' shape, sometimes with zeros (a leading zero weight never answers q = 0).
        if rng.random() < 0.35:
            if rng.random() < 0.85:
                params["method"] = "inverted_cdf"
            if axis is None or rng.random() < 0.5:
                wshape = view.shape
            else:
                axt = axis if isinstance(axis, tuple) else (axis,)
                wshape = tuple(view.shape[a] for a in axt)
            w = np.round(rng.uniform(0.0, 2.0, wshape), 3)
            if w.size and rng.random() < 0.2:
                w.reshape(-1)[int(rng.integers(0, w.size))] = 0.0
            params["w_op"] = len(ops)
            ops.append(g.plain(w))

        def call(*views, p=params):
            a = views[0]
            qq = views[p["q_op"]] if p["qkind"] == "arr" else p["q"]
            ax = tuple(p["axis"]) if isinstance(p["axis"], list) else p["axis"]
            kw = {"axis": ax, "method": p["method"], "keepdims": p["keepdims"]}
            if "w_op" in p:
                kw["weights"] = views[p["w_op"]]
            return getattr(np, p["name"])(a, qq, **kw)

        g.add(sec, "quant_kw", params=params, operands=ops, layout=label(kind, spec, lname), call=call)
        # np.median / np.nanmedian through the statistics op.
        mname = pick(rng, ["median", "nanmedian"])
        max_ = None if axis is None or isinstance(axis, tuple) else axis
        if mname == "nanmedian" and max_ is not None and has_none_na(spec, view):
            continue
        kd = bool(rng.random() < 0.3)
        mparams = {"name": mname, "axis": max_, "keepdims": kd}
        g.add(sec, "t_stat", params=mparams, operands=[(base, view)], layout=label(kind, spec, lname),
              call=lambda a, f=getattr(np, mname), x=max_, k=kd: f(a, axis=x, keepdims=k))


CMP = ["equal", "not_equal", "less", "less_equal", "greater", "greater_equal"]


def _pair(rng, kinds=("T", "T", "U")):
    """Two operands for a binary StringDType loop whose shapes broadcast (or, rarely, do not), each StringDType or U.

    :param rng: the generator.
    :param kinds: the kinds to draw from for each operand.
    :returns: ``(op1, op2, label)``.
    """
    k1, k2 = pick(rng, kinds), pick(rng, kinds)
    if k1 == "U" and k2 == "U":
        k2 = "T"  # a U pair is no StringDType loop (section strcomp gates it)
    spec1 = pick(rng, T_SPECS) if k1 == "T" else None
    # The same descriptor on both sides most of the time (the common case), another one otherwise (NA reconciliation).
    spec2 = (spec1 if spec1 is not None and rng.random() < 0.6 else pick(rng, T_SPECS)) if k2 == "T" else None
    shape_kind = pick(rng, ["same", "same", "row", "col", "scalar", "bad"])
    r, c = int(rng.integers(1, 5)), int(rng.integers(1, 6))
    shapes = {"same": ((r, c), (r, c)), "row": ((r, c), (c,)), "col": ((r, c), (r, 1)), "scalar": ((r, c), ()),
              "bad": ((r, c), (c + 1,))}[shape_kind]
    if rng.random() < 0.5:
        shapes = shapes[::-1]

    def build(k, spec, shape):
        n = int(np.prod(shape)) if shape else 1
        vals = rand_values(rng, n, k, spec, pick(rng, [0.0, 0.2]))
        if k == "U":
            vals = [v for v in vals]
        b = make_base(vals, k, spec)
        v = b.reshape(shape) if shape else b.reshape(())
        if v.ndim == 2 and rng.random() < 0.3:
            v = b.reshape(shape[::-1]).T if shape[0] == shape[1] else v
        return (b, v)

    o1, o2 = build(k1, spec1, shapes[0]), build(k2, spec2, shapes[1])
    lab = f"{g._t_label(spec1) if k1 == 'T' else k1}{shapes[0]},{g._t_label(spec2) if k2 == 'T' else k2}{shapes[1]}"
    return o1, o2, lab


def fam_tcomp(rng, sec, count):
    """StringDType comparisons (StringDType or U on either side, broadcasting), out= / where= forms, and the logical
    ufuncs."""
    for _ in range(count):
        o1, o2, lab = _pair(rng)
        name = pick(rng, CMP)
        g.add(sec, "ufunc", params={"name": name}, operands=[o1, o2], layout=lab,
              call=lambda x, y, f=getattr(np, name): f(x, y))
        if rng.random() < 0.25:
            try:
                shape = np.broadcast_shapes(o1[1].shape, o2[1].shape)
            except ValueError:
                continue
            odt = pick(rng, ["?", "?", "i1", "f8", "U5"])
            out = g.target(np.zeros(shape, dtype=odt)) if shape else g.target(np.zeros((), dtype=odt))
            if rng.random() < 0.5 and shape:
                mask = rng.random(shape) < 0.5
                g.add(sec, "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "where": 3, "dtype": None},
                      operands=[o1, o2, out, g.plain(mask)], layout=lab + f":where:{odt}",
                      call=lambda x, y, o, w, f=getattr(np, name): f(x, y, out=o, where=w))
            else:
                g.add(sec, "ufunc_kw", params={"name": name, "nin": 2, "out": 2, "dtype": None},
                      operands=[o1, o2, out], layout=lab + f":out:{odt}",
                      call=lambda x, y, o, f=getattr(np, name): f(x, y, out=o))
        if rng.random() < 0.2:
            lname = pick(rng, ["logical_and", "logical_or", "logical_xor"])
            g.add(sec, "ufunc", params={"name": lname}, operands=[o1, o2], layout=lab,
                  call=lambda x, y, f=getattr(np, lname): f(x, y))


def fam_tarith(rng, sec, count):
    """StringDType add / maximum / minimum (StringDType or U on either side) and multiply by integer arrays."""
    for _ in range(count):
        r = rng.random()
        if r < 0.6:
            o1, o2, lab = _pair(rng)
            name = pick(rng, ["add", "add", "maximum", "minimum"])
            g.add(sec, "ufunc", params={"name": name}, operands=[o1, o2], layout=lab,
                  call=lambda x, y, f=getattr(np, name): f(x, y))
        else:
            (b, v), lname, spec = rand_operand(rng, "T", layouts=["c1", "c2", "f2", "bc", "z0", "e1"])
            shape = v.shape
            reps = rng.integers(-1, 4, shape) if shape else np.array(int(rng.integers(-1, 4)))
            # NumPy 2.4.2 HANGS on an empty StringDType text times a negative count (its size_t repeat loop runs 2^64
            # times copying nothing; NumSharp returns the empty text, pinned by the unit tests): such pairs get a
            # count of 0 instead.
            empty = np.array([x == "" for x in np.array(v, copy=True).reshape(-1).tolist()], dtype=bool).reshape(shape)
            reps = np.where(empty & (np.broadcast_to(reps, shape) < 0), 0, reps)
            rop = g.num_operand(reps)
            if rng.random() < 0.5:
                g.add(sec, "ufunc", params={"name": "multiply"}, operands=[(b, v), rop], layout=label("T", spec, lname),
                      call=lambda x, y: np.multiply(x, y))
            else:
                g.add(sec, "ufunc", params={"name": "multiply"}, operands=[rop, (b, v)], layout=label("T", spec, lname),
                      call=lambda x, y: np.multiply(x, y))


def red_case(sec, name, operands, axis=None, keepdims=False, out_index=None, initial=_ABSENT, where_index=None,
             layout=""):
    """One ``t_red`` case (the committed ``_t_red_case`` in another section).

    :param sec: the section.
    :param name: the NumPy function.
    :param operands: the operands.
    :param axis: the axis.
    :param keepdims: keepdims.
    :param out_index: the out's operand index.
    :param initial: the initial value.
    :param where_index: the mask's operand index.
    :param layout: the label.
    """
    params = {"name": name, "axis": axis, "keepdims": keepdims}
    if out_index is not None:
        params["out"] = out_index
    if initial is not _ABSENT:
        params["initial"] = g.enc(initial)
    if where_index is not None:
        params["where"] = where_index

    def call(*ops):
        return g._t_red(name, ops[0], axis, keepdims, None, ops[out_index] if out_index is not None else None, initial,
                        ops[where_index] if where_index is not None else None)

    g.add(sec, "t_red", params=params, operands=operands, call=call, layout=layout)


def fam_treduce(rng, sec, count):
    """StringDType reductions: np.sum / cumsum / max / min / argmax / argmin and their nan forms, axis, keepdims,
    initial= and where=, the truth readers (any / all / count_nonzero / nonzero / flatnonzero / argwhere), astype(bool)."""
    for _ in range(count):
        (b, v), lname, spec = rand_operand(rng, "T")
        lab = label("T", spec, lname)
        name = pick(rng, ["sum", "cumsum", "max", "min", "argmax", "argmin", "nansum", "nancumsum", "nanargmax",
                          "nanargmin", "amax", "amin"])
        axis = None
        if v.ndim and rng.random() < 0.6:
            axis = int(rng.integers(-v.ndim, v.ndim))
        kd = bool(rng.random() < 0.3) and name not in ("cumsum", "nancumsum")
        initial = _ABSENT
        where_index = None
        ops = [(b, v)]
        if name in ("max", "min", "amax", "amin") and rng.random() < 0.3:
            initial = rand_text(rng, long_ok=False)
            if v.ndim and rng.random() < 0.5:
                mask = rng.random(v.shape) < 0.5
                where_index = len(ops)
                ops.append(g.plain(mask))
        red_case(sec, name, ops, axis=axis, keepdims=kd, initial=initial, where_index=where_index, layout=lab)
        tname = pick(rng, ["any", "all", "count_nonzero"] + ([] if v.ndim == 0 else ["nonzero", "flatnonzero",
                                                                                      "argwhere"]))
        g.add(sec, "call", params={"name": tname}, operands=[(b, v)], layout=lab,
              call=lambda a, f=getattr(np, tname): f(a))
        if v.ndim and rng.random() < 0.5:
            tn = pick(rng, ["any", "all", "count_nonzero"])
            ta = int(rng.integers(-v.ndim, v.ndim))
            tk = bool(rng.random() < 0.5)
            g.add(sec, "truth_axis", params={"name": tn, "axis": ta, "keepdims": tk}, operands=[(b, v)], layout=lab,
                  call=lambda a, f=getattr(np, tn), x=ta, k=tk: f(a, axis=x, keepdims=k))
        if rng.random() < 0.3:
            g.add(sec, "astype", params={"dtype": "?", "casting": "unsafe", "order": "K", "copy": True},
                  operands=[(b, v)], layout=lab, call=lambda a: a.astype(bool, order="K", casting="unsafe", copy=True))


def fam_ma(rng, sec, count):
    """numpy.ma over StringDType masked arrays with random masks: sort / argsort / median / unique and the reductions,
    and the binary comparisons / extremes / set routines."""
    unary = g._strma_unary_table()
    binary = g._strma_binary_table()
    unames = ["sort", "argsort", "median", "unique", "sort_end_false", "sum", "max", "min", "argmax", "argmin",
              "cumsum", "count", "all", "any", "compressed"]
    unames = [n for n in unames if n in unary]
    bnames = ["equal", "not_equal", "less", "greater_equal", "maximum", "minimum", "add", "isin", "intersect1d",
              "union1d", "setdiff1d", "setxor1d"]
    for _ in range(count):
        spec = pick(rng, T_SPECS)
        n = int(rng.integers(0, 10))
        two_d = n >= 4 and rng.random() < 0.3

        def marg():
            vals = rand_values(rng, n, "T", spec, pick(rng, [0.0, 0.2]))
            mask = None if rng.random() < 0.3 else [bool(x) for x in rng.random(n) < 0.3]
            if two_d:
                half = n // 2
                vals = [vals[:half], vals[half:2 * half]]
                mask = None if mask is None else [mask[:half], mask[half:2 * half]]
            return g.MArg(vals if n else np.zeros(0, spec.build()), mask, spec=spec.build() if n else None)

        x = marg()
        uname = pick(rng, unames)
        uparams = {"name": uname}
        # numpy.ma's unique and set routines run np.unique, whose values-only form hashes (has_strncmp_ties).
        if uname == "unique" and has_strncmp_ties(x.data):
            uparams["hash_ties"] = True
        g.add_ma("ma1", params=uparams, args=[x], layout=f"{g._t_label(spec)}:n{n}", phase=1,
                 call=lambda a, f=unary[uname]: f(a), section=sec)
        if rng.random() < 0.6:
            y = marg()
            bname = pick(rng, [b for b in bnames if b in binary])
            bparams = {"name": bname}
            if bname in ("intersect1d", "union1d", "setdiff1d") and has_strncmp_ties(x.data, y.data):
                bparams["hash_ties"] = True
            g.add_ma("ma2", params=bparams, args=[x, y], layout=f"{g._t_label(spec)}:n{n}", phase=1,
                     call=lambda a, b, f=binary[bname]: f(a, b), section=sec)


FAMILIES = {
    "sort": (fam_sort, 400),
    "lexsort": (fam_lexsort, 200),
    "search": (fam_search, 300),
    "unique": (fam_unique, 400),
    "sets": (fam_sets, 400),
    "sort_complex": (fam_sort_complex, 120),
    "hist": (fam_hist, 200),
    "quant": (fam_quant, 400),
    "tcomp": (fam_tcomp, 400),
    "tarith": (fam_tarith, 300),
    "treduce": (fam_treduce, 400),
    "ma": (fam_ma, 300),
}


def run_worker(seed, family, scale, out_path):
    """One family of one seed in this process: generate its cases and write them as JSON lines.

    :param seed: the seed.
    :param family: the family name.
    :param scale: the count multiplier.
    :param out_path: the output file.
    """
    fn, count = FAMILIES[family]
    rng = np.random.default_rng([seed, sum(map(ord, family))])
    g.CASES.clear()
    g._IDS.clear()
    g._OP_COUNT.clear()
    fn(rng, f"soak{seed}{family}", max(1, int(count * scale)))
    with open(out_path, "wb") as f:
        for c in g.CASES:
            f.write(json.dumps(c, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("ascii") + b"\n")


def main():
    """Parses the arguments and runs every (seed, family) pair as its own worker process, then merges each seed's
    families into ``<out>/soak_<seed>.jsonl`` and writes ``manifest.json``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--worker", nargs=3, metavar=("SEED", "FAMILY", "FILE"))
    args = ap.parse_args()
    if args.worker:
        run_worker(int(args.worker[0]), args.worker[1], args.scale, args.worker[2])
        return
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"numpy": np.__version__, "seeds": [], "scale": args.scale, "families": {}, "crashed": []}
    for seed in [int(s) for s in args.seeds.split(",") if s]:
        merged = out / f"soak_{seed}.jsonl"
        total = 0
        with open(merged, "wb") as m:
            for fam in args.families.split(","):
                part = out / f".part_{seed}_{fam}.jsonl"
                r = subprocess.run([sys.executable, __file__, "--scale", str(args.scale), "--out", str(out),
                                    "--worker", str(seed), fam, str(part)], capture_output=True, text=True)
                if r.returncode != 0 or not part.exists():
                    manifest["crashed"].append({"seed": seed, "family": fam, "code": r.returncode,
                                                "stderr": r.stderr[-2000:]})
                    continue
                data = part.read_bytes()
                m.write(data)
                n = data.count(b"\n")
                total += n
                manifest["families"][f"{seed}:{fam}"] = n
                part.unlink()
        manifest["seeds"].append({"seed": seed, "cases": total})
        print(f"seed {seed}: {total} cases -> {merged}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    if manifest["crashed"]:
        print(f"{len(manifest['crashed'])} worker(s) crashed: see manifest.json", file=sys.stderr)


if __name__ == "__main__":
    main()
