#!/usr/bin/env python3
"""
gen_relayout_oracle.py -- fill the ndarray-KIND gaps of the committed oracle corpus.

Every committed differential case pins one call of one op on operands of one layout. A survey of the whole corpus
(2026-10-07) found that most ops see only some of the ndarray kinds a caller can pass: of 1,775 (op key, operand
position) pairs, 42 % ever saw a C-contiguous view at a non-zero offset (the kind behind the np.split-child wrong
values), 0.1 % a permuted view at an offset, 2 % a stepped-and-reversed view, 0 % a broadcast at an offset, 0.2 % an
empty view. This generator closes those gaps: it re-emits existing cases with one operand moved into a missing kind,
and lets NumPy compute the new expected result.

How a case is re-emitted (``gen``):

  1. The SURVEY (``survey``) classifies every operand descriptor of every root corpus file into one of the 19 KINDS
     (``kind_of``: c, f, permuted, strided -- each also ``*_offset``, at an extra offset --, c_offset, reversed,
     strided_neg, bcast / bcast_offset, scalar_bcast / scalar_bcast_offset, 0d_owner, 0d_view, one_1d, empty,
     empty_view) plus the tags highrank / readonly, and records which CELLS -- (op, position, kind, dtype class:
     bool / signed / unsigned / float16 / float32-64 / complex) -- the corpus has.
  2. A SOURCE case qualifies only when a generic NumPy call reproduces its recorded result byte for byte
     (``reproduces``): the operands are rebuilt from their descriptors (``rebuild``), the op key resolves to a NumPy
     callable (``resolve``: np.*, np.linalg.*, np.fft.*, a module-qualified key, or an ``ndarray.`` method; ``SPECIAL``
     adapts the corpus conventions that are not a plain call), every param must be a keyword that callable accepts,
     and NumPy's answer (or its exact error type and text) must equal the case's. The dispatcher therefore never
     guesses a call: a case it cannot reproduce is not used.
  3. For each operand position and each TRANSFORM (``TRANSFORMS``) whose cell (op, position, resulting kind, the
     operand's dtype class) is empty, the operand is rebuilt in that kind holding the SAME logical values
     (value-preserving transforms), or -- for the elementwise and reduction families only, where every value is valid
     input (``_value_free_ops``) -- with derived values: a broadcast of its first row / element (also at an offset),
     its first element as a one-element 1-D array or a 0-d owner, axis 0 cut to length 0 (a fresh empty array, or an
     empty view inside a sentinel buffer). NumPy is called again on the new operands and its result (or error)
     becomes the variant's expected record. One variant fills a cell; an emptied operand fills two (``EMPTYING``:
     one where the other operands still broadcast with it, one where NumPy raises "could not be broadcast").
  4. Each variant is marked ``layout_sensitive`` when NumPy's own outcome changes once the moved operand is a fresh
     C-ordered copy; the others are held by FuzzCorpusTests.RelayoutLayoutInvariance to the contract that NumSharp's
     outcome does not change either (no excuse table in between). ``_upstream_residual`` drops the variants that
     would only repeat a divergence another tier documents (NEP50's weak 0-d promotion in a ufunc / np.where call --
     the relayout tiers take no NEP50 excuse --, the flat np.mean of a broadcast float operand), and ``_gap_of``
     carves the known NumSharp gaps (``GAPS``) by emulating each bug, so the carve is exact.

The variants keep the source's op key, params and valueclass (so the replay dispatches them through the same
OpRegistry entry, under the same excuse rules but the NEP50 0-d branches), with ``layout`` =
``relayout:<kind>@<position>`` and an id naming the source tier. Layout buffers hold the sentinel 7 (True for bool)
wherever the view does not reach (an offset prefix, the gaps of a stepped view, an empty view's buffer), so a kernel
that reads outside the view sees a value the operand never holds.

Outputs (../NumSharp.Tests.Oracle/Fuzz/corpus/):

  relayout.jsonl          variants of cases from the portable tiers, replayed by FuzzCorpusTests.Relayout.cs
  relayout_host.jsonl     variants of cases from the host-libm tiers (the files the replay runs through
                          RunHostLibmCorpus, read from the C# source), replayed the same way
  gaps/relayout_gaps.jsonl, gaps/relayout_host_gaps.jsonl
                          the variants a carve rule (``_gap_of``) names: NumSharp is known to differ there; each case
                          carries its gap id and is replayed by an [OpenBugs] test with the ordinary verdict AND the
                          layout-invariance verdict

Excluded sources: the stateful random tiers, the index_* / ma_* / decimal_* schemas, the backend-pinned tiers, the
array-object tiers (they own their layouts), np.evaluate trees and iterator traces (no single NumPy call),
np.frombuffer (its operand stands for Python bytes), and every operand of dtype char or decimal (no NumPy analog).
Run it after every other generator: its sources are their files.

Run (needs numpy==2.4.2; deterministic: a rerun is byte-identical):
    python test/oracle/gen_relayout_oracle.py [--uncarved DIR] [--report]
"""
import argparse
import collections
import inspect
import json
import os
import re
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layout_catalog import describe  # noqa: E402

FUZZ = os.path.normpath(os.path.join(HERE, "..", "NumSharp.Tests.Oracle", "Fuzz"))
CORPUS = os.path.join(FUZZ, "corpus")
GAP_DIR = os.path.join(CORPUS, "gaps")

# Files that are never sources: their own schemas (index_*, ma_*, decimal_*), stateful streams, backend pins, the
# array-object tiers, and the corpora this generator writes.
SKIP_PREFIXES = ("index_", "ma_", "decimal_", "random_", "generator_", "relayout")
SKIP_FILES = {"view_flags.jsonl", "out_flags.jsonl", "shape_args.jsonl", "matmul_parity.jsonl",
              "linalg_parity.jsonl", "polyalgebra_parity.jsonl", "polyroots_parity.jsonl", "evaluate.jsonl",
              "iter.jsonl", "random_smoke.jsonl"}
# Ops that are not a single NumPy call over ndarray operands: the random streams, np.evaluate trees, and
# np.frombuffer, whose operand stands for a Python bytes object -- the replay hands NumSharp the operand's logical
# bytes (FuzzCorpus.ResultBytes, a byte[]), so no layout of it ever reaches NumSharp, while NumPy, handed the
# ndarray itself, refuses a non-contiguous one ("ndarray is not C-contiguous").
SKIP_OPS = {"rnd", "grnd", "random_api", "seed", "get_state", "set_state", "evaluate", "frombuffer"}


def _host_libm_files():
    """The tiers the C# replay runs through ``RunHostLibmCorpus`` (hard-gated on win-amd64 only).

    Read from the replay source, as coverage/oracle_evidence.py does, so a tier pinned later is routed correctly
    without an edit here.

    Returns:
        The set of corpus file names.
    """
    pattern = re.compile(r'RunHostLibmCorpus\("([A-Za-z0-9_.]+\.jsonl)"\)')
    names = set()
    for name in sorted(os.listdir(FUZZ)):
        if name.startswith("FuzzCorpusTests") and name.endswith(".cs"):
            with open(os.path.join(FUZZ, name), encoding="utf-8-sig") as f:
                names.update(pattern.findall(f.read()))
    return names


def _value_free_ops():
    """The op keys of the elementwise and reduction families, where ANY operand values are valid input.

    Only these take the value-changing transforms (a broadcast of the first row / element, a one-element or 0-d
    operand, an emptied axis): their NumPy result is defined and deterministic for every input. Other ops may need
    structure those transforms destroy (a sorted array, in-range indices, distinct sort keys whose tie order is the
    algorithm's choice, a shape their params name). Read from gen_oracle's own tables, so the list follows them.

    Returns:
        The set of op keys.
    """
    import gen_oracle as go  # noqa: E402 -- 3 s import, only for its tables
    keys = set()
    for table in ("UNARY_OPS", "UNARY_EXTRA_OPS", "BINARY_OPS", "DIVMOD_POWER_OPS", "COMPARISON_OPS", "REDUCE_OPS",
                  "NAN_REDUCE_OPS", "STAT_REDUCE_OPS", "SCAN_OPS", "NAN_SCAN_OPS", "LOGIC_UNARY_OPS", "LOGIC_BIN_OPS",
                  "LOGICAL_BIN_OPS", "LOGADDEXP_OPS", "BITWISE_BIN_OPS", "SHIFT_OPS", "GCDLCM_OPS", "ISCOMPLEX_OPS"):
        keys.update(getattr(go, table).keys())
    return keys


# =============================================================================================================
# Survey: the kind of an operand descriptor
# =============================================================================================================

KINDS = ["c", "c_offset", "f", "f_offset", "permuted", "permuted_offset", "reversed", "strided", "strided_offset",
         "strided_neg", "bcast", "bcast_offset", "scalar_bcast", "scalar_bcast_offset", "0d_owner", "0d_view",
         "one_1d", "empty", "empty_view"]


def kind_of(d):
    """The ndarray kind of one operand descriptor, plus its tags.

    The ``*_offset`` kinds are the same layouts with their first element past the buffer's start -- the offset a
    view of a split child, of a row of a bigger array, or of a slice that starts past 0 carries. Only the EXTRA
    offset counts: a reversed axis needs an offset to reach its first element, so a reversed or stepped-and-reversed
    view stays ``reversed`` / ``strided_neg`` whatever its offset. A kernel that forgets ``Shape.offset`` reads such
    an operand from the owner's start, which is the bug class these kinds exist to reach.

    Parameters:
        d: the descriptor (dtype, shape, element strides, element offset, bufferSize, buffer, own?, writeable?).

    Returns:
        (kind, tags): kind is one of KINDS; tags a list of "highrank" (ndim >= 5) and "readonly".
    """
    shape = d.get("shape") or []
    strides = d.get("strides") or []
    offset = d.get("offset", 0) or 0
    nd = len(shape)
    size = 1
    for x in shape:
        size *= x
    tags = []
    if nd >= 5:
        tags.append("highrank")
    if d.get("writeable") is False:
        tags.append("readonly")
    if nd == 0:
        return ("0d_owner" if d.get("own") or offset == 0 else "0d_view"), tags
    if size == 0:
        # An empty VIEW keeps the buffer it was cut from (NumPy's a[3:3] points into a); a fresh empty array has none.
        return ("empty_view" if (d.get("bufferSize") or 0) > 0 and d.get("buffer") else "empty"), tags
    if nd == 1 and size == 1:
        return "one_1d", tags
    big = [(s, n) for s, n in zip(strides, shape) if n > 1]
    # The offset a negative stride needs to reach the view's first element; anything beyond it is an EXTRA offset.
    extra = offset - sum((n - 1) * -s for s, n in big if s < 0)
    if big and all(s == 0 for s, _ in big):
        return ("scalar_bcast_offset" if offset > 0 else "scalar_bcast"), tags
    if any(s == 0 for s, _ in big):
        return ("bcast_offset" if extra > 0 else "bcast"), tags
    want_c, acc = [], 1
    for n in reversed(shape):
        want_c.append(acc)
        acc *= n
    want_c.reverse()
    if all(s == w for s, w, n in zip(strides, want_c, shape) if n > 1):
        return ("c" if offset == 0 else "c_offset"), tags
    want_f, acc = [], 1
    for n in shape:
        want_f.append(acc)
        acc *= n
    if all(s == w for s, w, n in zip(strides, want_f, shape) if n > 1):
        return ("f_offset" if offset > 0 else "f"), tags
    neg = any(s < 0 for s, _ in big)
    acc, dense = 1, True
    for s, n in sorted(((abs(s), n) for s, n in big), key=lambda t: t[0]):
        if s != acc:
            dense = False
            break
        acc *= n
    if dense:
        return ("reversed" if neg else ("permuted_offset" if offset > 0 else "permuted")), tags
    return ("strided_neg" if neg else ("strided_offset" if offset > 0 else "strided")), tags


def _corpus_files():
    """Every root corpus file name, in a fixed order (host-pin metadata files excluded)."""
    return sorted(n for n in os.listdir(CORPUS) if n.endswith(".jsonl") and not n.endswith(".host.jsonl"))


def _norm_op(c):
    """The op label a survey pair is keyed by: the array-object oracle's vehicle keys split by params.fn."""
    op = c.get("op")
    if op in ("viewflags", "outflags", "shapearg"):
        return f"{op}:{(c.get('params') or {}).get('fn')}"
    return op


def dtype_class(dtype):
    """The dtype CLASS a fill cell is keyed by: b (bool), i (signed), u (unsigned), f2 (float16), f (float32 /
    float64), c (complex), or the name itself for the non-NumPy dtypes (char, decimal)."""
    if dtype == "bool":
        return "b"
    if dtype.startswith("int"):
        return "i"
    if dtype.startswith("uint"):
        return "u"
    if dtype == "float16":
        return "f2"
    if dtype.startswith("float"):
        return "f"
    if dtype.startswith("complex"):
        return "c"
    return dtype


def survey():
    """Which kinds and tags every (op, operand position) pair of the committed corpus has seen, per dtype class.

    The generator's own output files are left out, so a regeneration fills the same gaps (it is deterministic).

    Returns:
        (kinds, tags): dicts (op, position) -> {kind or tag -> set of dtype classes}.
    """
    kinds = collections.defaultdict(lambda: collections.defaultdict(set))
    tags = collections.defaultdict(lambda: collections.defaultdict(set))
    for name in _corpus_files():
        if name.startswith("relayout"):
            continue
        with open(os.path.join(CORPUS, name), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                op = _norm_op(c)
                if not op:
                    continue
                for i, d in enumerate(c.get("operands") or []):
                    if isinstance(d, dict) and "shape" in d:
                        k, t = kind_of(d)
                        cls = dtype_class(d.get("dtype", ""))
                        kinds[(op, i)][k].add(cls)
                        for tag in t:
                            tags[(op, i)][tag].add(cls)
    return kinds, tags


# =============================================================================================================
# The self-validating NumPy dispatcher
# =============================================================================================================

def rebuild(d):
    """The NumPy operand a descriptor names, built exactly as FuzzCorpus.Reconstruct builds NumSharp's.

    Parameters:
        d: the operand descriptor.

    Returns:
        A view over a 1-D base holding the buffer (an owning copy when ``own`` is set; a fresh empty array for a
        zero-size shape without a buffer -- with one, an empty VIEW into it, as the C# side builds since the
        empty_view kind), read-only when ``writeable`` is false.
    """
    dt = np.dtype(d["dtype"])
    shape = tuple(d["shape"])
    if shape and int(np.prod(shape)) == 0 and not d.get("buffer"):
        return np.empty(shape, dtype=dt)
    base = np.frombuffer(bytes.fromhex(d["buffer"]), dtype=dt).copy()
    isz = dt.itemsize
    v = np.ndarray(shape, dtype=dt, buffer=base, offset=d.get("offset", 0) * isz,
                   strides=tuple(s * isz for s in d["strides"]))
    if d.get("own"):
        v = np.array(v, copy=True, order="K")
    if d.get("writeable") is False:
        v.setflags(write=False)
    return v


_MODULES = {"emath": np.emath, "ma": np.ma, "linalg": np.linalg, "fft": np.fft,
            "polyutils": np.polynomial.polyutils}
for _b in ["polynomial", "chebyshev", "legendre", "laguerre", "hermite", "hermite_e"]:
    _MODULES[_b] = getattr(np.polynomial, _b)


def resolve(op):
    """The NumPy callable an op key names.

    Parameters:
        op: the corpus op key.

    Returns:
        (target, method): target is a callable, or a method NAME when ``method`` is true (``ndarray.<name>``:
        operand 0 is the receiver); (None, False) when the key names nothing callable.
    """
    if op.startswith("ndarray."):
        return op[len("ndarray."):], True
    head, _, rest = op.partition(".")
    if rest:
        m = _MODULES.get(head)
        return (getattr(m, rest, None) if m is not None else None), False
    for m in (np, np.linalg, np.fft):
        f = getattr(m, op, None)
        if callable(f) and not inspect.isclass(f) and not inspect.ismodule(f):
            return f, False
    return None, False


_UFUNC_KW = {"out", "where", "dtype", "casting", "order", "subok", "axes", "axis", "keepdims"}


def accepts(f, key):
    """True when the callable takes ``key`` as a keyword (a ufunc's keywords, or the Python signature's)."""
    if isinstance(f, np.ufunc):
        return key in _UFUNC_KW
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return False
    return key in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _split_piece(fn):
    """An adapter for the split family's ``piece`` convention: the call's result tuple, indexed by params.piece."""
    def run(ops, p):
        """Split operand 0 by params.sections (along params.axis when given) and return piece params.piece."""
        kw = {"axis": p["axis"]} if "axis" in p else {}
        return fn(ops[0], p["sections"], **kw)[p["piece"]]
    return run


def _list_operands(fn):
    """An adapter for the functions that take their operands as ONE sequence (stack, concatenate, ...)."""
    def run(ops, p):
        """Pass every operand as one list, the params as keywords."""
        return fn(list(ops), **p)
    return run


def _unwrap(ops, p):
    """np.unwrap with the harness's ``period_is_int`` flag (the period passed as a Python int when set)."""
    kw = {"axis": p["axis"]}
    if "discont" in p:
        kw["discont"] = p["discont"]
    if "period" in p:
        kw["period"] = int(p["period"]) if p.get("period_is_int") else p["period"]
    return np.unwrap(ops[0], **kw)


def _searchsorted(ops, p):
    """np.searchsorted with an optional sorter operand (operand 2)."""
    kw = {"side": p["side"]} if "side" in p else {}
    if len(ops) > 2:
        kw["sorter"] = ops[2]
    return np.searchsorted(ops[0], ops[1], **kw)


# Op keys whose corpus convention is not a plain call: (operands, params) -> NumPy's result, mirroring the
# OpRegistry entry of the key (each adapter is still validated by reproducing the recorded results).
SPECIAL = {
    "astype": lambda ops, p: ops[0].astype(p["dtype"]),
    "round_": lambda ops, p: np.round(ops[0], p["decimals"]),
    "unwrap": _unwrap,
    "modf_frac": lambda ops, p: np.modf(ops[0])[0],
    "modf_int": lambda ops, p: np.modf(ops[0])[1],
    "swapaxes": lambda ops, p: np.swapaxes(ops[0], p["a1"], p["a2"]),
    "moveaxis": lambda ops, p: np.moveaxis(ops[0], p["src"], p["dst"]),
    "flip": lambda ops, p: np.flip(ops[0], axis=tuple(p["axes"]) if "axes" in p else p.get("axis")),
    "trim_zeros": lambda ops, p: np.trim_zeros(ops[0], p["trim"], axis=tuple(p["axes"]) if "axes" in p else p.get("axis")),
    "resize": lambda ops, p: np.resize(ops[0], tuple(p["shape"])),
    "stack": _list_operands(np.stack),
    "concatenate": _list_operands(np.concatenate),
    "hstack": _list_operands(np.hstack),
    "vstack": _list_operands(np.vstack),
    "dstack": _list_operands(np.dstack),
    "column_stack": _list_operands(np.column_stack),
    "split": _split_piece(np.split),
    "array_split": _split_piece(np.array_split),
    "hsplit": _split_piece(np.hsplit),
    "vsplit": _split_piece(np.vsplit),
    "dsplit": _split_piece(np.dsplit),
    "choose": lambda ops, p: np.choose(ops[0], list(ops[1:1 + p["nc"]]), mode=p.get("mode", "raise")),
    "select": lambda ops, p: np.select(list(ops[:p["nc"]]), list(ops[p["nc"]:2 * p["nc"]]), ops[2 * p["nc"]]),
    "tensordot": lambda ops, p: np.tensordot(ops[0], ops[1], axes=(p["axesA"], p["axesB"]) if "axesA" in p else p["axes"]),
    "multi_dot": lambda ops, p: np.linalg.multi_dot(list(ops)),
    "insert": lambda ops, p: np.insert(ops[0], p["obj"], ops[1], axis=p.get("axis")),
    "ravel_multi_index": lambda ops, p: np.ravel_multi_index(tuple(ops), tuple(p["dims"]), mode=p.get("mode", "raise"),
                                                             order=p.get("order", "C")),
    "unravel_index": lambda ops, p: np.unravel_index(ops[0], tuple(p["shape"]), order=p.get("order", "C"))[p["piece"]],
    "ix_": lambda ops, p: np.ix_(*ops)[p["which"]],
    "median": lambda ops, p: np.median(ops[0], axis=tuple(p["axes"]) if "axes" in p else p.get("axis"),
                                       keepdims=p.get("keepdims", False)),
    "nanmedian": lambda ops, p: np.nanmedian(ops[0], axis=tuple(p["axes"]) if "axes" in p else p.get("axis"),
                                             keepdims=p.get("keepdims", False)),
    "average": lambda ops, p: np.average(ops[0], axis=tuple(p["axes"]) if "axes" in p else p.get("axis"),
                                         keepdims=p.get("keepdims", False)),
    "std_ddof": lambda ops, p: np.std(ops[0], axis=p.get("axis"), ddof=p["ddof"]),
    "var_ddof": lambda ops, p: np.var(ops[0], axis=p.get("axis"), ddof=p["ddof"]),
    "ravel_f": lambda ops, p: np.ravel(ops[0], order="F"),
    "searchsorted": _searchsorted,
    "flatten": lambda ops, p: ops[0].flatten(),
    "block": lambda ops, p: np.block([[ops[0], ops[1]], [ops[2], ops[3]]]),
    "ndarray.reshape": lambda ops, p: ops[0].reshape(tuple(p["shape"])),
    "ndarray.clip": lambda ops, p: ops[0].clip(p["lo"], p["hi"]),
    "ndarray.swapaxes": lambda ops, p: ops[0].swapaxes(p["a1"], p["a2"]),
    "ndarray.T": lambda ops, p: ops[0].T,
    "ndarray.real": lambda ops, p: ops[0].real,
    "ndarray.imag": lambda ops, p: ops[0].imag,
}


def call(op, params, ops):
    """Call the NumPy function an op key names with the case's operands (positional) and params (keywords).

    Parameters:
        op: the op key.
        params: the case params; each must be a keyword of the callable (or the key has a SPECIAL adapter).
        ops: the operand arrays, in descriptor order.

    Returns:
        NumPy's result; for an in-place function that returns None, operand 0 (the mutated array).

    Raises:
        LookupError: the key names no callable, or a param is not one of its keywords (the case is not a plain
            call, and is not used).
        Exception: whatever NumPy raises for these operands (an error outcome).
    """
    if op in SPECIAL:
        try:
            r = SPECIAL[op](ops, params or {})
        except KeyError as e:
            raise LookupError(f"adapter param {e}") from None
        return ops[0] if r is None else r
    f, method = resolve(op)
    if f is None:
        raise LookupError("unresolved op key")
    args = list(ops)
    if method:
        target = getattr(args[0], f, None)
        if target is None:
            raise LookupError("no such ndarray method")
        f, args = target, args[1:]
    kw = {}
    for k, v in (params or {}).items():
        if not accepts(f, k):
            raise LookupError(f"param {k} is not a keyword")
        kw[k] = tuple(v) if isinstance(v, list) and k in ("axis", "axes", "shape", "newshape", "s") else v
    r = f(*args, **kw)
    return ops[0] if r is None else r


def serialize(r, kind=None):
    """The expected record of one array result: (dtype, shape, C-order bytes), plus the source's kind tag."""
    a = np.asarray(r)
    exp = {"dtype": a.dtype.name, "shape": [int(x) for x in a.shape],
           "buffer": np.ascontiguousarray(a).tobytes().hex()}
    if kind:
        exp["kind"] = kind
    return exp


def _exc(e):
    """NumPy's exception, verbatim: the Python class name and ``str(e)``."""
    return {"type": type(e).__name__, "text": str(e)}


def _outcome(op, params, ops):
    """NumPy's outcome of one call, in a form two calls can be compared by.

    Parameters:
        op: the op key.
        params: the case params.
        ops: the operand arrays (an in-place function mutates them; pass fresh ones).

    Returns:
        ("ok", expected record) or ("err", exception type, text); None when the key is not a plain call.
    """
    try:
        return ("ok", serialize(call(op, params, ops)))
    except LookupError:
        return None
    except Exception as e:  # noqa: BLE001 -- NumPy's refusal is an outcome
        return ("err", type(e).__name__, str(e))


def reproduces(c):
    """True when the generic call reproduces the case's recorded outcome exactly.

    Parameters:
        c: a corpus case (default or ``scalar`` result kind, or an error case).

    Returns:
        True for the same (dtype, shape, bytes), or the same exception type and text for an error case.
    """
    exp = c.get("expected") or {}
    if exp.get("kind") not in (None, "scalar"):
        return False
    ops = [rebuild(d) for d in c["operands"]]
    try:
        r = call(c["op"], c.get("params"), ops)
    except LookupError:
        return False
    except Exception as e:  # noqa: BLE001 -- NumPy's refusal is an outcome
        err = c.get("error") or {}
        return bool(c.get("expects_throw")) and err.get("type") == type(e).__name__ and err.get("text") == str(e)
    if c.get("expects_throw"):
        return False
    s = serialize(r)
    return s["dtype"] == exp.get("dtype") and s["shape"] == exp.get("shape") and s["buffer"] == exp.get("buffer")


# =============================================================================================================
# Transforms: an operand in another kind
# =============================================================================================================

def _sentinel(dt):
    """The value the layout buffers hold outside the view: 7 (True for bool)."""
    return np.array(7).astype(dt)


def _t_offset(v):
    """A C-contiguous view at element offset 3 (a 0-d input gives a 0-d view at offset 3)."""
    base = np.empty(v.size + 3, dtype=v.dtype)
    base[:3] = _sentinel(v.dtype)
    base[3:] = np.asarray(v).reshape(-1)
    return base, base[3:].reshape(v.shape)


def _t_f(v):
    """An F-contiguous view: the C-contiguous transpose of the values, transposed back."""
    base = np.ascontiguousarray(np.asarray(v).T)
    return base, base.T


def _t_permuted(v):
    """A dense view whose axes are a rotation of a C-contiguous array's (neither C nor F; ndim >= 3)."""
    nd = v.ndim
    perm = tuple(list(range(1, nd)) + [0])
    inv = tuple(int(i) for i in np.argsort(perm))
    base = np.ascontiguousarray(np.asarray(v).transpose(perm))
    return base, base.transpose(inv)


def _t_reversed(v):
    """A dense view with every axis reversed (negative strides)."""
    rev = (slice(None, None, -1),) * v.ndim
    base = np.ascontiguousarray(np.asarray(v)[rev])
    return base, base[rev]


def _t_strided(v):
    """Every other element of a buffer twice as wide along the last axis (the gaps hold the sentinel)."""
    base = np.empty(v.shape[:-1] + (2 * v.shape[-1],), dtype=v.dtype)
    base[...] = _sentinel(v.dtype)
    base[..., ::2] = v
    return base, base[..., ::2]


def _t_strided_neg(v):
    """Every other element of a twice-as-wide buffer, walked backward along the last axis."""
    base = np.empty(v.shape[:-1] + (2 * v.shape[-1],), dtype=v.dtype)
    base[...] = _sentinel(v.dtype)
    base[..., ::-2] = v
    return base, base[..., ::-2]


def _t_readonly(v):
    """A C-contiguous owner with WRITEABLE cleared (NumPy's setflags(write=False))."""
    base = np.ascontiguousarray(v).copy()
    base.setflags(write=False)
    return base, base


def _t_bcast(v):
    """VALUE-CHANGING: the first row repeated along axis 0 (a stride-0 axis; read-only)."""
    base = np.ascontiguousarray(np.asarray(v)[:1]).copy()
    return base, np.broadcast_to(base, v.shape)


def _t_scalar_bcast(v):
    """VALUE-CHANGING: the first element repeated everywhere (every stride 0; read-only)."""
    base = np.ascontiguousarray(np.asarray(v).reshape(-1)[:1]).copy()
    return base, np.broadcast_to(base.reshape(()), v.shape)


def _prefixed(values):
    """A 1-D buffer of the sentinel three times, then ``values`` (raveled in C order): the base of an offset view.

    Parameters:
        values: the array whose elements the view will hold.

    Returns:
        (base, payload): the whole buffer and its view ``base[3:]`` (1-D, at element offset 3).
    """
    flat = np.asarray(values).reshape(-1)
    base = np.empty(flat.size + 3, dtype=flat.dtype)
    base[:3] = _sentinel(flat.dtype)
    base[3:] = flat
    return base, base[3:]


def _t_f_offset(v):
    """An F-contiguous view at element offset 3 (the sentinel before it): an F array cut from a bigger buffer."""
    a = np.asarray(v)
    base, payload = _prefixed(a.T)
    return base, payload.reshape(a.shape[::-1]).T


def _t_permuted_offset(v):
    """A dense, axis-rotated view (neither C nor F; ndim >= 3) at element offset 3."""
    a = np.asarray(v)
    perm = tuple(list(range(1, a.ndim)) + [0])
    inv = tuple(int(i) for i in np.argsort(perm))
    moved = a.transpose(perm)
    base, payload = _prefixed(moved)
    return base, payload.reshape(moved.shape).transpose(inv)


def _t_strided_offset(v):
    """Every other element along the last axis, starting at element 1 (NumPy's ``a[..., 1::2]``): a stepped view at
    a non-zero offset; the sentinel fills the even slots."""
    base = np.empty(v.shape[:-1] + (2 * v.shape[-1],), dtype=v.dtype)
    base[...] = _sentinel(v.dtype)
    base[..., 1::2] = v
    return base, base[..., 1::2]


def _t_bcast_offset(v):
    """VALUE-CHANGING: the first row repeated along axis 0, the row read from element offset 3 of its buffer."""
    a = np.asarray(v)
    base, payload = _prefixed(a[:1])
    return base, np.broadcast_to(payload.reshape(a[:1].shape), a.shape)


def _t_scalar_bcast_offset(v):
    """VALUE-CHANGING: the first element repeated everywhere, read from element offset 3 (a broadcast split child)."""
    base, payload = _prefixed(np.asarray(v).reshape(-1)[:1])
    return base, np.broadcast_to(payload.reshape(()), v.shape)


def _t_one_1d(v):
    """VALUE-CHANGING: the first element as a one-element 1-D array (the size-1 shape some paths confuse with 0-d)."""
    base = np.ascontiguousarray(np.asarray(v).reshape(-1)[:1]).copy()
    return base, base


def _t_0d_owner(v):
    """VALUE-CHANGING: the first element as a 0-d array that owns its data (NumPy's ``np.array(x)``)."""
    base = np.array(np.asarray(v).reshape(-1)[0])
    return base, base


def _t_empty(v):
    """VALUE-CHANGING: a fresh empty array of the operand's shape with axis 0 cut to length 0."""
    base = np.empty((0,) + tuple(v.shape[1:]), dtype=v.dtype)
    return base, base


def _t_empty_view(v):
    """VALUE-CHANGING: an empty view at element offset 3 into a sentinel buffer, axis 0 cut to length 0 and the
    other axes kept (C strides). NumSharp's empty slices keep a non-zero offset (np.split's last piece of a split
    at the end, ``a["3:3"]``); its offset addresses no element, so a kernel that reads one anyway sees 7.

    Built with np.ndarray(buffer=, offset=) because NumPy's own empty slice ``base[3:3]`` does NOT advance its data
    pointer (it stays at the owner's start), so slicing cannot place the view at offset 3.
    """
    rest = tuple(int(x) for x in v.shape[1:])
    k = int(np.prod(rest)) if rest else 1
    base = np.empty(3 + max(k, 1), dtype=v.dtype)
    base[...] = _sentinel(v.dtype)
    isz = base.itemsize
    strides, acc = [], isz
    for n in reversed((0,) + rest):
        strides.append(acc)
        acc *= max(n, 1)
    view = np.ndarray((0,) + rest, dtype=v.dtype, buffer=base, offset=3 * isz, strides=tuple(reversed(strides)))
    return base, view


# name -> (builder, minimum ndim, value-preserving). Each needs size >= 1; the resulting kind is classified after
# the build (kind_of), so e.g. an offset 0-d operand fills the 0d_view kind. The value-changing ones run only for
# the elementwise and reduction families (``_value_free_ops``).
TRANSFORMS = {
    "c_offset": (_t_offset, 0, True),
    "f": (_t_f, 2, True),
    "f_offset": (_t_f_offset, 2, True),
    "permuted": (_t_permuted, 3, True),
    "permuted_offset": (_t_permuted_offset, 3, True),
    "reversed": (_t_reversed, 1, True),
    "strided": (_t_strided, 1, True),
    "strided_offset": (_t_strided_offset, 1, True),
    "strided_neg": (_t_strided_neg, 1, True),
    "readonly": (_t_readonly, 0, True),
    "bcast": (_t_bcast, 2, False),
    "bcast_offset": (_t_bcast_offset, 2, False),
    "scalar_bcast": (_t_scalar_bcast, 1, False),
    "scalar_bcast_offset": (_t_scalar_bcast_offset, 1, False),
    "one_1d": (_t_one_1d, 1, False),
    "0d_owner": (_t_0d_owner, 1, False),
    "empty": (_t_empty, 1, False),
    "empty_view": (_t_empty_view, 1, False),
}

# The transforms whose operand is a fresh array that owns its data (NumPy's np.array(x)): their descriptor carries
# ``own`` so the C# side builds an owner too, not a view over a one-element base.
OWNING_TRANSFORMS = {"one_1d", "0d_owner"}

# The transforms that cut axis 0 to length 0, and the suffix of the second cell they fill: the variant whose other
# operands no longer broadcast with the emptied one (gen() keeps one of each per (op, position, dtype class)).
EMPTYING = {"empty", "empty_view"}
BCAST_ERR = "!bcast_error"


def _predicted_kind(tname, nd, size):
    """The kind a transform is expected to produce for an operand of ``nd`` dimensions and ``size`` elements.

    Used only to skip work early; the built operand is classified again by ``kind_of``.

    Parameters:
        tname: the transform name.
        nd: the operand's ndim.
        size: its element count (>= 1).

    Returns:
        The kind (or "readonly" for the tag transform), or None when the transform cannot change this operand's
        kind (an offset or reversed one-element 1-D array is still one_1d; a stepped one-element array too).
    """
    if tname == "readonly":
        return "readonly"
    if nd == 0:
        return "0d_view" if tname == "c_offset" else None
    if tname in ("0d_owner", "empty", "empty_view"):
        # These replace the shape, so they reach their kind from any operand of ndim >= 1 (size 1 included).
        return tname
    if tname == "one_1d":
        # A (1,), (1, 1), ... operand becomes (1,); only a one-element 1-D operand already is one.
        return None if (nd == 1 and size == 1) else "one_1d"
    if size == 1:
        return None
    return tname


# =============================================================================================================
# Carving (filled from the replay of the uncarved matrix)
# =============================================================================================================

# gap id -> description; drawn from the replay of the uncarved matrix (2026-10-07: 70 of 16,295 variants, every one
# an operand the kernel reads from its OWNER'S START -- the C-contiguous-at-an-offset class of the array-object
# oracle's VA01 / VA03 / VA05 / VF39 / VF40, reached here through more ops, dtypes and positions). Every rule also
# needs ``_misread_observable``: a misread that leaves NumPy's answer unchanged cannot be seen, so it is not carved.
# FuzzCorpusTests.Relayout.cs replays gaps/relayout*_gaps.jsonl under [TestCategory("OpenBugs")] with the ordinary
# verdict AND the layout-invariance verdict. When a gap is fixed, drop its rule and regenerate.
GAPS = {
    "RL01": "np.searchsorted / ndarray.searchsorted read a C-contiguous v (operand 1) or sorter (operand 2) at a "
            "non-zero offset from the owner's start (WRONG indices); np.digitize passes its x as searchsorted's v, and "
            "np.isin passes a complex128 / float16 element the same way (its sort + searchsorted route), so both "
            "inherit it (same root cause as the array-object oracle's VA01)",
    "RL02": "np.where reads a C-contiguous condition (bool) / x / y already of the result dtype at a non-zero offset "
            "from the owner's start (WRONG values; a cast operand is copied first and is read right), and so does its "
            "all-0-d route for a 0-d y at an offset; np.nancumsum / np.nancumprod of float16 / complex128 run "
            "np.where(np.isnan(a), fill, a) and inherit it (same root cause as VA03)",
    "RL03": "np.ldexp reads a C-contiguous mantissa x1 already of the result dtype, or an int32 / int64 exponent x2, at "
            "a non-zero offset from the owner's start (WRONG values): MaterializeContiguous passes such an operand "
            "through np.ascontiguousarray unchanged and the kernel reads .Address without Shape.offset (VA05, x1 too)",
    "RL04": "np.unique / np.unique_values of a C-contiguous array at a non-zero offset read the owner from its start "
            "(WRONG unique set), and np.intersect1d / np.setdiff1d / np.setxor1d inherit it through either operand "
            "(same root cause as VF40)",
    "RL05": "ndarray.repeat of a 0-d view at a non-zero offset (an np.unstack child of a 1-D array) repeats the owner's "
            "FIRST element, not the view's (same root cause as VF39; the ordinary replay's registry excuses it as a "
            "known repeat bug, so only the layout-invariance verdict sees it)",
    "RL06": "the NDArray comparison operators (==, !=, <, >, <=, >=) return np.empty(lhs.Shape, bool) for an EMPTY left "
            "operand without broadcasting it against the right one: no error where NumPy raises 'operands could not be "
            "broadcast together', a result shaped like the left operand where NumPy broadcasts both, and the untyped "
            "empty wrapper behind MakeGeneric<bool>() is left to the finalizer (one pooled buffer per call; the leak "
            "sweep's KnownEscapes lists the families that only leak)",
    "RL07": "np.logaddexp / np.logaddexp2 return a CLONE of an empty operand (either position) without broadcasting or "
            "promoting it: no error where NumPy raises 'operands could not be broadcast together', and the empty "
            "operand's own dtype (int32, uint8) where NumPy's float loop gives float64 / float16",
}

# The registry's comparison op keys, replayed through the NDArray operators (OpRegistry: ops[0] == ops[1], ...).
COMPARISON_OPERATOR_OPS = {"equal", "not_equal", "less", "greater", "less_equal", "greater_equal"}

# The kinds whose first element sits past the buffer's start by an EXTRA offset (kind_of): the operands a kernel
# that forgets Shape.offset misreads.
OFFSET_KINDS = {"c_offset", "0d_view", "f_offset", "permuted_offset", "strided_offset", "bcast_offset",
                "scalar_bcast_offset"}


def _misread_observable(case, pos, kind, expected_outcome):
    """Would a kernel that reads the moved operand from its OWNER'S START give a different answer?

    The model of the offset-view bug class: the operand is read with its own strides but at offset 0, i.e. the
    window holds the three sentinels in front of the values. NumPy is asked for that window's answer; when it equals
    the true answer the misread cannot be seen (an order-free result of a shifted window, a sentinel that happens to
    leave the result unchanged), so a NumSharp kernel that misreads still matches and the variant must not be
    carved.

    Parameters:
        case: the finished variant (its operands are the descriptors NumSharp replays).
        pos: the moved operand's position.
        kind: its kind.
        expected_outcome: NumPy's true outcome, as ``_outcome`` forms it.

    Returns:
        True / False for the offset kinds; None for every other kind (the model does not apply).
    """
    if kind not in OFFSET_KINDS:
        return None
    ops = [rebuild(x) for x in case["operands"]]
    ops[pos] = rebuild({**case["operands"][pos], "offset": 0})
    return _outcome(case["op"], case.get("params"), ops) != expected_outcome


def _gap_of(case, misread):
    """The documented gap a variant falls into (a GAPS id), or None when NumSharp is expected to match NumPy.

    Parameters:
        case: the finished variant case.
        misread: ``_misread_observable``'s verdict (None outside the offset kinds).

    Returns:
        A GAPS id or None.
    """
    op = case["op"]
    kind, pos = case["layout"].split(":", 1)[1].split("@")
    pos = int(pos)
    if kind in ("empty", "empty_view"):
        return _empty_gap(case, op, pos)
    if not misread:
        return None
    dts = [d["dtype"] for d in case["operands"]]
    result_dt = (case.get("expected") or {}).get("dtype")
    if kind == "c_offset":
        if (op == "searchsorted" and pos in (1, 2)) or (op == "ndarray.searchsorted" and pos == 1) \
                or (op == "digitize" and pos == 0):
            return "RL01"
        # isin's complex / float16 route (sort + searchsorted) runs only when the element is already the common dtype.
        if op == "isin" and pos == 0 and dts[0] in ("complex128", "float16") \
                and np.result_type(*[np.dtype(t) for t in dts[:2]]).name == dts[0]:
            return "RL01"
        if op == "where" and ((pos == 0 and dts[0] == "bool") or (pos in (1, 2) and dts[pos] == result_dt)):
            return "RL02"
        if op == "ldexp" and ((pos == 0 and dts[0] == result_dt) or (pos == 1 and dts[1] in ("int32", "int64"))):
            return "RL03"
        if (op in ("unique", "unique_values") and pos == 0) \
                or (op in ("intersect1d", "setdiff1d", "setxor1d") and pos in (0, 1)):
            return "RL04"
    if kind in ("c_offset", "0d_view") and op in ("nancumsum", "nancumprod") and pos == 0 \
            and dts[0] in ("float16", "complex128"):
        return "RL02"
    if kind == "0d_view" and op == "ndarray.repeat" and pos == 0:
        return "RL05"
    return None


def _empty_gap(case, op, pos):
    """The gap of a variant whose moved operand is EMPTY, by emulating the two NumSharp early returns that skip the
    ufunc machinery for an empty operand (RL06, RL07): the variant is carved exactly when NumPy's outcome differs from
    what that early return produces.

    Parameters:
        case: the finished variant case.
        op: its op key.
        pos: the position of the emptied operand.

    Returns:
        "RL06", "RL07" or None.
    """
    emptied = case["operands"][pos]
    truth = ("err", case["error"]["type"], case["error"]["text"]) if case.get("expects_throw") \
        else ("ok", {k: v for k, v in case["expected"].items() if k != "kind"})
    if op in COMPARISON_OPERATOR_OPS and pos == 0:
        # operator ==(NDArray lhs, NDArray rhs) & co: `if (lhs.size == 0) return np.empty(lhs.Shape, bool)`.
        early = ("ok", {"dtype": "bool", "shape": emptied["shape"], "buffer": ""})
        return "RL06" if early != truth else None
    if op in ("logaddexp", "logaddexp2"):
        # DefaultEngine.LogAddExp: `if (x1.size == 0) return x1.Clone(); if (x2.size == 0) return x2.Clone();`.
        first_empty = next(d for d in case["operands"] if 0 in d["shape"])
        early = ("ok", {"dtype": first_empty["dtype"], "shape": first_empty["shape"], "buffer": ""})
        return "RL07" if early != truth else None
    return None


def _upstream_residual(case, view, expected_outcome):
    """True when a variant lands in a NumSharp divergence another tier already owns and documents.

    Two such divergences reach the relayout matrix, and a variant there would only repeat them, so it is not emitted
    and its cell stays open for a source that does not hit them:

      * the flat np.mean of a float / complex operand with a stride-0 axis (the plan's finding 10: NumSharp folds a
        broadcast operand in its own order, NumPy per row), unless the sum is order-free -- the mean_loops tier drops
        exactly these cases with the same predicate, and OpenBugs.Reductions.cs pins them;
      * a multi-operand ufunc or np.where call whose NumPy answer changes once every 0-d operand is a Python scalar
        of its value (NEP 50's weak promotion, which NumSharp applies to 0-d arrays where NumPy promotes them as
        arrays: the registry's documented NEP50 branches). The relayout tier takes NO such excuse
        (MisalignedRegistry exempts it), so every 0-d variant it keeps must have the same answer either way, and a
        dtype or value difference there is a real one -- the blanket used to excuse any difference of a case with
        a 0-d operand, which hid np.logaddexp returning its empty int32 operand where NumPy returns float64. Only
        ufuncs and np.where have NEP 50's weak scalars; another function converts a Python scalar its own way
        (np.searchsorted discovers int64), so no such model applies to it and its 0-d variants are simply kept.

    Parameters:
        case: the finished variant case.
        view: the moved operand (NumPy array).
        expected_outcome: NumPy's outcome for the variant, as ``_outcome`` forms it.

    Returns:
        True to drop the variant.
    """
    p = case.get("params") or {}
    descs = case["operands"]
    if len(descs) >= 2 and any(len(d["shape"]) == 0 for d in descs) \
            and (case["op"] == "where" or (case["op"] not in SPECIAL and isinstance(resolve(case["op"])[0], np.ufunc))):
        ops = [rebuild(x) for x in descs]
        weak = [o.item() if o.ndim == 0 else o for o in ops]
        if _outcome(case["op"], p, weak) != expected_outcome:
            return True
    if case["op"] != "mean" or p.get("axis") is not None or view.dtype.kind not in "fc":
        return False
    if not any(st == 0 and sh > 1 for st, sh in zip(view.strides, view.shape)):
        return False
    import gen_oracle as go  # noqa: E402 -- the mean_loops tier's own predicates, so both tiers draw one line
    return not go._mean_order_free(view, None, go._mean_loop_of(view, p.get("dtype")))


# =============================================================================================================
# Generation
# =============================================================================================================

def gen(carve=True, report=False):
    """Re-emit reproducible source cases with one operand in each missing kind.

    Parameters:
        carve: file the variants a carve rule names into the gaps lists.
        report: print the per-kind fill counts.

    Returns:
        dict file name -> list of cases, for relayout.jsonl, relayout_host.jsonl and their gap files.
    """
    kinds_seen, tags_seen = survey()
    host = _host_libm_files()
    free = _value_free_ops()
    out = {"relayout.jsonl": [], "relayout_host.jsonl": [], "relayout_gaps.jsonl": [], "relayout_host_gaps.jsonl": []}
    filled = collections.Counter()
    n = 0

    def missing(op_key, pos, kind, cls):
        """True when the cell (op, position, kind or tag, dtype class) has no case yet (committed or emitted)."""
        seen = tags_seen if kind == "readonly" else kinds_seen
        return cls not in seen[(op_key, pos)].get(kind, ())

    def mark(op_key, pos, kind, cls):
        """Record an emitted variant, so each cell is filled by exactly one variant."""
        seen = tags_seen if kind == "readonly" else kinds_seen
        seen[(op_key, pos)][kind].add(cls)

    for name in _corpus_files():
        if name in SKIP_FILES or name.startswith(SKIP_PREFIXES):
            continue
        dest = "relayout_host.jsonl" if name in host else "relayout.jsonl"
        with open(os.path.join(CORPUS, name), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                op = c.get("op")
                if not op or op in SKIP_OPS or c.get("expects_throw"):
                    continue
                descs = c.get("operands") or []
                if not descs or any(not isinstance(d, dict) or d.get("dtype") in ("char", "decimal") for d in descs):
                    continue
                key_op = _norm_op(c)
                # Cheap pre-check before the (costly) reproduction test: does any position still have an empty cell
                # (op, position, kind, this operand's dtype class) this case could fill? The kind is PREDICTED here and
                # verified after the build (kind_of), which skips a transform whose operand lands in a filled cell.
                wanted = []
                for pos, d in enumerate(descs):
                    nd = len(d.get("shape") or [])
                    size = int(np.prod(d["shape"])) if nd else 1
                    if size == 0:
                        continue
                    for tname, (_, min_nd, keep_values) in TRANSFORMS.items():
                        if nd < min_nd or (not keep_values and op not in free):
                            continue
                        predicted = _predicted_kind(tname, nd, size)
                        cls = dtype_class(d["dtype"])
                        if predicted is None or not (missing(key_op, pos, predicted, cls)
                                                     or (tname in EMPTYING and missing(key_op, pos,
                                                                                       predicted + BCAST_ERR, cls))):
                            continue
                        wanted.append((pos, tname))
                if not wanted:
                    continue
                try:
                    if not reproduces(c):
                        continue
                except Exception:  # noqa: BLE001 -- a descriptor NumPy cannot rebuild is not a source
                    continue
                for pos, tname in wanted:
                    builder, _, keep_values = TRANSFORMS[tname]
                    d = descs[pos]
                    try:
                        v = rebuild(d)
                        base, view = builder(v)
                        nd_desc = describe(base, view)
                    except Exception:  # noqa: BLE001 -- the transform does not apply to this operand
                        continue
                    if tname in OWNING_TRANSFORMS:
                        nd_desc["own"] = True
                    kind = "readonly" if tname == "readonly" else kind_of(nd_desc)[0]
                    cls = dtype_class(d["dtype"])
                    if not missing(key_op, pos, kind, cls) \
                            and not (tname in EMPTYING and missing(key_op, pos, kind + BCAST_ERR, cls)):
                        continue
                    ops = [rebuild(x) for x in descs]
                    ops[pos] = view
                    new_descs = list(descs)
                    new_descs[pos] = nd_desc
                    exp_kind = (c.get("expected") or {}).get("kind")
                    n += 1
                    cid = f"relayout/{name[:-6]}/{op}/{kind}@{pos}/{n}"
                    layout = f"relayout:{kind}@{pos}"
                    try:
                        r = call(op, c.get("params"), ops)
                        case = {"id": cid, "op": op, "params": c.get("params") or {}, "operands": new_descs,
                                "expected": serialize(r, exp_kind), "layout": layout,
                                "valueclass": c.get("valueclass", "mixed")}
                    except LookupError:
                        continue
                    except Exception as e:  # noqa: BLE001 -- NumPy's refusal for this layout is the outcome
                        case = {"id": cid, "op": op, "params": c.get("params") or {}, "operands": new_descs,
                                "expected": {"kind": exp_kind} if exp_kind else {}, "expects_throw": True,
                                "error": _exc(e), "layout": layout, "valueclass": "error"}
                    expected_outcome = ("err", case["error"]["type"], case["error"]["text"]) \
                        if case.get("expects_throw") else ("ok", {k: v for k, v in case["expected"].items() if k != "kind"})
                    if _upstream_residual(case, view, expected_outcome):
                        continue
                    # An emptied operand fills TWO cells per (op, position, dtype class): one where the other operands
                    # still broadcast with it (a value), one where they no longer do (NumPy's "could not be broadcast"
                    # error -- the edge where a code path that returns early for an empty operand skips the shape
                    # check). Every other transform fills one.
                    cell = kind + BCAST_ERR if (tname in EMPTYING and case.get("expects_throw")
                                                and "could not be broadcast" in case["error"]["text"]) else kind
                    if not missing(key_op, pos, cell, cls):
                        continue
                    mark(key_op, pos, cell, cls)
                    filled[cell] += 1
                    # Is NumPy's answer a property of the VALUES, or does the layout change it? Recomputed with the
                    # moved operand as a fresh C-ordered copy (every operand rebuilt: an in-place op mutated the first
                    # set). Where the answers agree, NumSharp's must not depend on the layout either -- the contract
                    # FuzzCorpusTests.RelayoutLayoutInvariance holds it to, with no excuse table in between; where
                    # they differ (a reduction whose summation order follows the memory order, a read-only operand an
                    # in-place op refuses) the variant says so and that test skips it.
                    canon_ops = [rebuild(x) for x in new_descs]
                    canon_ops[pos] = np.array(canon_ops[pos], copy=True, order="C")
                    sensitive = _outcome(op, c.get("params"), canon_ops) != expected_outcome
                    if sensitive:
                        case["layout_sensitive"] = True
                    gap = _gap_of(case, _misread_observable(case, pos, kind, expected_outcome)) if carve else None
                    if gap:
                        case["gap"] = gap
                        out[dest.replace(".jsonl", "_gaps.jsonl")].append(case)
                    else:
                        out[dest].append(case)
    if report:
        for k, v in sorted(filled.items()):
            print(f"  filled {k:12s} {v}")
    return out


def write_jsonl(path, cases):
    """Write cases as JSONL (LF line ends, compact separators) and report the count."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, separators=(",", ":")) + "\n")
    print(f"wrote {len(cases)} cases -> {path}")


def main():
    """Generate the relayout tiers (see the module docstring for the flags)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    ap.add_argument("--report", action="store_true", help="print the per-kind fill counts")
    args = ap.parse_args()
    out = gen(carve=not args.uncarved, report=args.report)
    if args.uncarved:
        for name in ("relayout.jsonl", "relayout_host.jsonl"):
            write_jsonl(os.path.join(args.uncarved, name), out[name])
        return
    for name, cases in out.items():
        folder = GAP_DIR if name.endswith("_gaps.jsonl") else CORPUS
        write_jsonl(os.path.join(folder, name), cases)


if __name__ == "__main__":
    main()
