#!/usr/bin/env python3
"""
gen_matrix_operand_oracle.py -- what NumPy's np.* functions return for a numpy.matrix OPERAND.

NumPy propagates the matrix subclass through most of its functions, each by its own mechanism: the fromnumeric
wrappers call the object's METHOD (np.sum(m, 0) is m.sum(0), a 1 x 3 matrix), the ufuncs wrap their result
(__array_wrap__), np.dot / np.concatenate / np.linalg pick the subclass of the operand with the highest priority,
and other functions strip it on purpose with asarray (np.ravel, np.diagonal, np.where, np.einsum return plain
arrays). A few cannot hold their result in two dimensions and raise (np.stack). So there is no uniform rule: the
rule per function is NumPy's, and this tier records it.

How a case is made (the machinery is gen_relayout_oracle's):

  1. A SOURCE case of the committed corpus qualifies when a generic NumPy call reproduces its recorded result byte
     for byte (gen_relayout_oracle.reproduces) and it has an operand with exactly two dimensions.
  2. That operand is re-typed in place as a matrix (``operand.view(np.matrix)``: the same memory, layout and
     values), one position at a time, plus one variant with every 2-D operand as a matrix when there are two or
     more. The other operands stay plain arrays.
  3. NumPy is called again. Its result is recorded with its CLASS (``expected.cls``: matrix / ndarray / scalar),
     dtype, shape and bytes; its exception with type and text.

The variants keep the source's op key, params and valueclass, so the replay dispatches them through the same
OpRegistry entry with the same value excuses; ``layout`` is ``matrix@<position>`` (``matrix@all``), which the
harness reads to hold the class and the shape with no excuse (MisalignedRegistry.IsMatrixOperandCase). The operand
descriptors carry ``matrix: true``, so the replay hands NumSharp an NDMatrix through an NDArray reference.

A budget keeps the tier small: at most 2 variants per (op, position, dtype class, set of param keys, axis /
keepdims values) and 8 per (op, position, axis / keepdims values). The axis and keepdims VALUES are part of the
cell, because a matrix reduction answers each of them differently (a scalar, a row, a column, a TypeError).

EVERY MATRIX CALL RUNS IN A WORKER PROCESS. NumPy 2.4.2 corrupts its own heap when a function that allocates its
result as the operand's subclass cannot finalize it: np.stack((m, m)) and np.dstack((m, m)) raise "shape too large
to be a matrix." and a few such calls later the process dies in PyDataMem_UserFREE ("PyCapsule_GetPointer called
with invalid PyCapsule object"; 400 calls in a fresh process always crash it, np.concatenate / np.hstack /
np.vstack never do). An expected value computed after such a call cannot be trusted. So the parent only reads the
corpus and reproduces the SOURCE cases (plain arrays), and a worker (``--worker``) makes the matrix calls; the
parent restarts the worker after every call that raised that text, and after a crash, where the variant is dropped
and counted.

Outputs (../NumSharp.Tests.Oracle/Fuzz/corpus/):

  matrix_operand.jsonl        variants of cases from the portable tiers
  matrix_operand_host.jsonl   variants of cases from the host-libm tiers (replayed through RunHostLibmCorpus)
  gaps/matrix_operand_gaps.jsonl, gaps/matrix_operand_host_gaps.jsonl
                              the variants a carve rule (``_gap_of``) names: NumSharp is known to differ there

Run it after the other generators (its sources are their files; its own files, the relayout files and the matrix
tier are never sources). Needs numpy==2.4.2; deterministic.

    python test/oracle/gen_matrix_operand_oracle.py [--uncarved DIR] [--report]
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import warnings

import numpy as np

np.seterr(all="ignore")
warnings.simplefilter("ignore")

assert np.__version__ == "2.4.2", f"corpus must be generated with numpy 2.4.2, got {np.__version__}"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_relayout_oracle as R  # noqa: E402
from gen_matrix_oracle import desc  # noqa: E402  (the operand descriptor, with ``matrix: true`` for a matrix)

CORPUS = R.CORPUS
GAP_DIR = R.GAP_DIR

# Never sources: what gen_relayout_oracle skips, plus the matrix tiers themselves.
SKIP_PREFIXES = R.SKIP_PREFIXES + ("matrix",)

PER_CELL = 2      # variants per (op, position, dtype class, param keys, axis / keepdims values)
PER_PAIR = 8      # variants per (op, position, axis / keepdims values)

# The parameters that decide the SHAPE NumPy gives a matrix result; their values are part of a budget cell.
SHAPE_PARAMS = ("axis", "axes", "keepdims")

# Op keys without the ``ndarray.`` prefix whose call is a METHOD of operand 0 on both sides
# (gen_relayout_oracle.SPECIAL: ``ops[0].astype(...)``, ``ops[0].flatten()``). Like the prefixed keys they test a
# member through an NDArray reference, which keeps the plain-array meaning in NumSharp.
METHOD_KEYS = ("astype", "flatten")

# NumPy's text for a result that cannot be a matrix. A call that raised it may have damaged the heap (see the
# module docstring), so the worker that made it is retired.
TOO_LARGE = "shape too large to be a matrix."

# The carve: gap id -> why NumSharp differs there. A rule in _gap_of files a variant under one id. Only MO01 is a
# limit of NumSharp; in MO02 - MO05 NumPy's own answer is an accident of matrix semantics inside its Python code
# (its `*` is the matrix product there, its ravel() stays two-dimensional, it hands keepdims to a matrix method),
# and NumSharp returns what the function returns for the same values as a plain array, shaped by the function's
# matrix rule.
GAPS = {
    "MO01": "np.isclose returns NDArray<bool> in C#: a typed array cannot be a matrix object, so the result is plain",
    "MO02": "np.count_nonzero(m, axis): NumPy raises TypeError (it passes keepdims to matrix.sum for every call "
            "with an axis); NumSharp returns the counts",
    "MO03": "np.median / percentile / quantile and their nan forms: NumPy partitions, indexes and multiplies the "
            "matrix with matrix semantics (a 1 x N result without an axis, 3-d and 4-d matrix objects under keepdims, "
            "a row for axis 1, float64 for float16 / float32, TypeError or ValueError); NumSharp returns the scalar, "
            "row or column of the plain result",
    "MO04": "np.roll and np.append without an axis, np.ediff1d, np.vdot, np.union1d: ravel() of a matrix stays "
            "1 x N inside NumPy (the roll does nothing, ediff1d is 0 x N, append / vdot / union1d raise); NumSharp "
            "computes with the flattened values",
    "MO05": "np.polyval, np.i0, np.sinc, np.trapezoid: `*` inside NumPy's implementation is the matrix product "
            "(a different value, a float64 result for float16 / float32 through np.dot's dtype, or an alignment "
            "error); NumSharp computes elementwise",
}

_NARROW_FLOATS = ("float16", "float32")


def _expected(result):
    """The expected record of one NumPy result.

    Parameters:
        result: what NumPy returned: a matrix, an ndarray, or a scalar.

    Returns:
        (dtype, shape, C-order bytes) plus ``cls``: "matrix", "ndarray" or "scalar".
    """
    if isinstance(result, np.ndarray):
        record = R.serialize(result)
        record["cls"] = "matrix" if isinstance(result, np.matrix) else "ndarray"
        return record
    record = R.serialize(np.asarray(result))
    record["cls"] = "scalar"
    return record


def _axis_of(p):
    """The axis argument of the reduction adapters: a tuple for ``axes``, else ``axis`` (None when absent).

    Parameters:
        p: the case params.

    Returns:
        The value to pass as ``axis=``.
    """
    return tuple(p["axes"]) if "axes" in p else p.get("axis")


def _keepdims_kw(p):
    """``keepdims=True`` as a keyword dict, or nothing: a C# caller cannot say ``keepdims=False`` explicitly.

    Parameters:
        p: the case params.

    Returns:
        ``{"keepdims": True}`` when the case asks for it, else ``{}``.
    """
    return {"keepdims": True} if p.get("keepdims") else {}


# gen_relayout_oracle's adapters for these keys pass keepdims=False explicitly, which is the same call for a plain
# array but not for a matrix (see numpy_call). The same adapters with the keyword left out when it is False.
KEEPDIMS_ADAPTERS = {
    "median": lambda ops, p: np.median(ops[0], axis=_axis_of(p), **_keepdims_kw(p)),
    "nanmedian": lambda ops, p: np.nanmedian(ops[0], axis=_axis_of(p), **_keepdims_kw(p)),
    "average": lambda ops, p: np.average(ops[0], axis=_axis_of(p), **_keepdims_kw(p)),
}


def numpy_call(op, params, ops):
    """NumPy's call for one matrix variant, spelled the way the C# replay can spell it.

    NumPy's fromnumeric wrappers forward ``keepdims`` to the operand's method only when the caller passed it, and
    ``matrix.sum`` takes no such keyword: ``np.sum(m, 0)`` is a 1 x 3 matrix, ``np.sum(m, 0, keepdims=False)``
    raises ``TypeError: matrix.sum() got an unexpected keyword argument 'keepdims'``. C# has default parameters,
    so a caller who writes ``np.sum(m, 0)`` and one who writes ``np.sum(m, 0, keepdims: false)`` make the same
    call. The corpus records ``keepdims`` in every reduction case; a False value therefore stands for "not
    passed", and only ``keepdims=True`` reaches NumPy.

    Parameters:
        op: the op key.
        params: the case params.
        ops: the operands (some of them matrices).

    Returns:
        NumPy's result.

    Raises:
        LookupError: the key is not a plain NumPy call.
        Exception: whatever NumPy raises for these operands.
    """
    if op in KEEPDIMS_ADAPTERS:
        return KEEPDIMS_ADAPTERS[op](ops, params)
    passed = {k: v for k, v in params.items() if not (k == "keepdims" and v is False)}
    return R.call(op, passed, ops)


def _gap_of(case):
    """The gap id of a variant NumSharp is known to answer differently, or None.

    Parameters:
        case: the variant (op, params, operands, expected or error).

    Returns:
        A key of GAPS, or None for a gate case.

    The rules are exact: an uncarved replay fails on every case a rule names and on no other case (checked in both
    directions when a rule changes).
    """
    op = case["op"]
    p = case.get("params") or {}
    ops = case["operands"]
    dtype0 = ops[0]["dtype"]
    refused = bool(case.get("expects_throw"))
    if op == "isclose":
        return "MO01"
    if op == "count_nonzero":
        return "MO02"
    if op in ("nanmedian", "nanpercentile", "nanquantile"):
        return "MO03"
    if op == "median":
        # NumPy's answer is the row / column only for an axis without keepdims, and not where it raises.
        return "MO03" if p.get("keepdims") or p.get("axis") is None or refused else None
    if op in ("percentile", "quantile"):
        row_instead_of_column = p.get("axis") == 1 and ops[0]["shape"][0] != 1
        if dtype0 in _NARROW_FLOATS or p.get("keepdims") or p.get("axis") is None or row_instead_of_column:
            return "MO03"
        return None
    if op in ("roll", "ediff1d", "vdot", "union1d"):
        return "MO04"
    if op == "append":
        return "MO04" if refused else None
    if op in ("polyval", "i0"):
        return "MO05"
    if op == "sinc":
        return "MO05" if dtype0 in _NARROW_FLOATS else None
    if op == "trapezoid":
        # y is the matrix: with an x the products are matrix products; without one the scalar dx times a narrow
        # float matrix is np.dot's float64.
        if case["layout"] == "matrix@0" and (len(ops) == 2 or dtype0 in _NARROW_FLOATS):
            return "MO05"
        return None
    return None


def _variants(positions):
    """The operand-position sets to re-type: each 2-D position alone, then all of them together.

    Parameters:
        positions: the positions of the 2-D operands, ascending.

    Returns:
        A list of (label, tuple of positions).
    """
    out = [(str(p), (p,)) for p in positions]
    if len(positions) > 1:
        out.append(("all", tuple(positions)))
    return out


def worker():
    """The worker loop: one request per stdin line, one answer per stdout line.

    A request is ``{"op", "params", "operands", "group"}``: the source case's operand descriptors and the
    positions to re-type as matrices. The answer is ``{"operands", "expected"}``, ``{"operands", "error"}`` or
    ``{"skip": reason}`` (the op key is not a plain NumPy call, or an operand cannot be serialized).

    Returns:
        None. The loop ends when stdin closes.
    """
    for line in sys.stdin:
        request = json.loads(line)
        ops = [R.rebuild(d) for d in request["operands"]]
        for i in request["group"]:
            ops[i] = ops[i].view(np.matrix)
        try:
            answer = {"operands": [desc(o) for o in ops]}
        except ValueError:
            answer = {"skip": "operand cannot be serialized"}
        else:
            try:
                answer["expected"] = _expected(numpy_call(request["op"], request["params"], ops))
            except LookupError:
                answer = {"skip": "not a plain call"}
            except Exception as e:  # noqa: BLE001 -- NumPy's refusal is the expected outcome
                answer["error"] = R._exc(e)
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()


class Worker:
    """A worker process and its pipes; ``ask`` restarts it when it died or made a heap-damaging call."""

    def __init__(self):
        """Start with no process (the first ``ask`` starts one).

        Returns:
            None.
        """
        self.process = None
        self.crashes = 0
        self.restarts = 0

    def _start(self):
        """Start a fresh worker process.

        Returns:
            None.
        """
        self.process = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        self.restarts += 1

    def retire(self):
        """Stop the current worker, so the next request gets a fresh process.

        Returns:
            None.
        """
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=30)
            except Exception:  # noqa: BLE001 -- a damaged process may not exit cleanly
                self.process.kill()
            self.process = None

    def ask(self, request):
        """Send one request and read its answer.

        Parameters:
            request: the request dict (see ``worker``).

        Returns:
            The answer dict, or None when the worker died while it computed this request (the variant is
            dropped: NumPy crashed on it).
        """
        if self.process is None:
            self._start()
        try:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except (BrokenPipeError, OSError):
            line = ""
        if not line:
            self.crashes += 1
            self.retire()
            return None
        answer = json.loads(line)
        if answer.get("error", {}).get("text") == TOO_LARGE:
            self.retire()
        return answer


def gen(carve=True, report=False):
    """Re-emit reproducible source cases with their 2-D operands as matrices.

    Parameters:
        carve: file the variants a carve rule names into the gap lists.
        report: print how NumPy answered per op key, and the worker statistics.

    Returns:
        dict file name -> list of cases.
    """
    host = R._host_libm_files()
    out = {"matrix_operand.jsonl": [], "matrix_operand_host.jsonl": [],
           "matrix_operand_gaps.jsonl": [], "matrix_operand_host_gaps.jsonl": []}
    per_cell = collections.Counter()
    per_pair = collections.Counter()
    answers = collections.defaultdict(collections.Counter)
    pool = Worker()
    n = 0

    for name in R._corpus_files():
        if name in R.SKIP_FILES or name.startswith(SKIP_PREFIXES):
            continue
        dest = "matrix_operand_host" if name in host else "matrix_operand"
        with open(os.path.join(CORPUS, name), encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                op = c.get("op")
                if not op or op in R.SKIP_OPS or c.get("expects_throw"):
                    continue
                # An ``ndarray.<method>`` key calls the method through an NDArray reference, where NumSharp's
                # matrix semantics do not apply (they follow the static type). The matrix tier holds the members.
                if op.startswith("ndarray.") or op in METHOD_KEYS:
                    continue
                descs = c.get("operands") or []
                if not descs or any(not isinstance(d, dict) or d.get("dtype") in ("char", "decimal") for d in descs):
                    continue
                positions = [i for i, d in enumerate(descs) if len(d.get("shape", ())) == 2]
                if not positions:
                    continue
                # A 0-d operand next to another operand is where NumSharp's documented NEP 50 divergence lives (a
                # 0-d array is weak there, strong in NumPy). The harness switches that blanket excuse off for this
                # tier (MisalignedRegistry.IsMatrixOperandCase), so its sources stay out.
                if len(descs) >= 2 and any(len(d.get("shape", ())) == 0 for d in descs):
                    continue
                params = c.get("params") or {}
                keys = tuple(sorted(params))
                # The matrix rule of a function follows its axis and keepdims arguments (a reduction along axis 0
                # is a row, along axis 1 a column, without an axis a scalar; keepdims raises), so each value of
                # them is a cell of its own. A budget by parameter NAMES alone kept only the axis=None cases.
                shape_args = tuple(json.dumps(params.get(k), sort_keys=True) for k in SHAPE_PARAMS)
                wanted = []
                for label, group in _variants(positions):
                    cls = R.dtype_class(descs[group[0]]["dtype"])
                    cell = (op, label, cls, keys, shape_args)
                    if per_cell[cell] < PER_CELL and per_pair[(op, label, shape_args)] < PER_PAIR:
                        wanted.append((label, group, cell))
                if not wanted or not R.reproduces(c):
                    continue
                for label, group, cell in wanted:
                    answer = pool.ask({"op": op, "params": params, "operands": descs, "group": list(group)})
                    if answer is None:
                        answers[op]["NUMPY CRASHED"] += 1
                        continue
                    if "skip" in answer:
                        continue
                    variant = {"op": op, "params": params, "operands": answer["operands"],
                               "layout": f"matrix@{label}", "source": c.get("id")}
                    if c.get("valueclass"):
                        variant["valueclass"] = c["valueclass"]
                    if "error" in answer:
                        if "object at 0x" in answer["error"]["text"]:
                            continue
                        variant["expected"] = {}
                        variant["expects_throw"] = True
                        variant["error"] = answer["error"]
                        answers[op]["error"] += 1
                    else:
                        variant["expected"] = answer["expected"]
                        answers[op][answer["expected"]["cls"]] += 1
                    n += 1
                    variant["id"] = f"{op}/matrix@{label}/{descs[group[0]]['dtype']}/{n}"
                    per_cell[cell] += 1
                    per_pair[(op, label, shape_args)] += 1
                    gap = _gap_of(variant) if carve else None
                    if gap:
                        variant["gap"] = gap
                        out[dest + "_gaps.jsonl"].append(variant)
                    else:
                        out[dest + ".jsonl"].append(variant)
    pool.retire()

    if report:
        for op in sorted(answers):
            print(f"  {op:28s} " + "  ".join(f"{k}={v}" for k, v in sorted(answers[op].items())))
        print(f"  {len(answers)} op keys; worker starts {pool.restarts}, crashes {pool.crashes}")
    return out


def main():
    """Generate the matrix-operand tiers; see the module docstring for the flags.

    Returns:
        None.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--uncarved", metavar="DIR", help="write the whole matrix (no carve) into DIR instead")
    ap.add_argument("--report", action="store_true", help="print NumPy's answers per op key")
    ap.add_argument("--worker", action="store_true", help="internal: serve matrix calls on stdin / stdout")
    args = ap.parse_args()
    if args.worker:
        worker()
        return
    out = gen(carve=not args.uncarved, report=args.report)
    if args.uncarved:
        for name in ("matrix_operand.jsonl", "matrix_operand_host.jsonl"):
            R.write_jsonl(os.path.join(args.uncarved, name), out[name])
        return
    for name, cases in out.items():
        folder = GAP_DIR if name.endswith("_gaps.jsonl") else CORPUS
        R.write_jsonl(os.path.join(folder, name), cases)


if __name__ == "__main__":
    main()
