"""Validate the gap map's findings against their two ground truths: the C# compiler and NumPy's runtime.

Usage (from this directory, after dump_modules.cs, binding_probe.py and signature_diff.py):
    python -I validate_findings.py <modules_dump.json> <types_dump.json> <NumSharp.dll> <diff.csv> \
        <probe_results.json> <work dir> <validation.json>

signature_diff.py decides most marks from reflection and from NumPy's signature as it READS it: the runtime signature,
completed from the .pyi stubs and the numpydoc sections. Both readings can be wrong. A stub can promise a keyword the
runtime refuses (``ndarray.std(correction=...)``), a docstring can name a keyword the C code spells differently
(``ndarray.setfield(val, ...)`` is ``value`` at run time), a stub can promise a tuple axis the runtime refuses
(``matrix.argmax(axis=(0, 1))``), and a C# class inherits overloads its own member list does not show
(``NDMatrix.cumsum(out: o)`` binds ``NDArray.cumsum``). This tool asks the ground truth for every finding:

- **Probe artifacts.** A compile-probe failure whose code is not a binding verdict (CS0246 type not found, CS0103 name
  not found, ...) means the probe FILE is wrong; the finding it carries is not evidence.
- **M, by the compiler.** For each missing keyword-capable NumPy parameter, the name's base call (the probe's R call)
  is compiled again with the parameter added by name (``np.sum(a, initial: default)``). CS1739 naming the parameter
  confirms M. A call that compiles refutes it: some overload, possibly an inherited one, takes that name.
- **M and X, by NumPy.** NumPy checks keyword NAMES before it looks at a value (probed for C functions, ufuncs,
  dispatchers, methods, numpy.ma wrappers and constructors), so each check calls the NumPy callable with ``None`` for
  every required positional parameter and one keyword. An X parameter must be refused ("unexpected keyword argument" /
  "is an invalid keyword argument"); an M parameter must not be. A control keyword that cannot exist is tried first:
  when NumPy does not refuse it, the placeholders stopped the call before its keywords were checked, and the verdict
  comes from a real call (``REAL_CALLS``) or stays inconclusive.
- **Q and L, by NumPy.** Q says NumPy takes a sequence of integers there and L that NumPy takes None: each is called
  with a real operand and the tuple / None (``SEQUENCE_CALLS``, ``NULL_CALLS``, else a generic call on a 3-D operand).

The JSON lists every verdict; ``summary`` counts them per mark. signature_diff.py and parity_diff.py read the refuted
findings (``refuted``) to leave them out of the marks and the change list.
"""
import argparse
import csv
import faulthandler
import io
import json
import os
import re
import shutil
import sys
import tempfile
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import binding_probe as bp  # noqa: E402  (the probe's callee and value writers)
import signature_diff as sd  # noqa: E402  (NumPy's side and the groups)

np = sd.np

#: Compile errors that are not a binding verdict: the probe file itself is wrong (a missing using, a name it invented).
ARTIFACT_CODES = frozenset({"CS0246", "CS0103", "CS0234", "CS1002", "CS1003", "CS1026", "CS1513", "CS1525"})

#: A keyword no NumPy callable takes: the control that tells whether a placeholder call reaches the keyword check.
BOGUS = "__gap_map_validation__"

#: Phrases NumPy uses when it refuses a keyword's NAME (C functions, ufuncs, Python functions, methods, constructors).
REFUSAL = ("unexpected keyword argument", "is an invalid keyword argument", "takes no keyword arguments",
           "keyword arguments not accepted")


def _cube():
    """A small 3-D float64 operand every axis value of the checks fits.

    Returns:
        ``arange(1, 25).reshape(2, 3, 4)``.
    """
    return np.arange(1.0, 25.0).reshape(2, 3, 4)


def _npz():
    """An open ``NpzFile`` over an in-memory archive (``np.load`` of a ``np.savez`` result).

    Returns:
        The NpzFile.
    """
    buffer = io.BytesIO()
    np.savez(buffer, x=np.arange(3.0))
    buffer.seek(0)
    return np.load(buffer)


#: The NumPy instance whose methods an object group's spellings compare against.
INSTANCES = {
    "ndarray": lambda: np.arange(1.0, 7.0).reshape(2, 3),
    "ndarray (NDArray<T>)": lambda: np.arange(1.0, 7.0).reshape(2, 3),
    "np.matrix": lambda: np.matrix([[1.0, 2.0], [3.0, 4.0]]),
    "np.char.chararray": lambda: np.char.array(["ab", "cd"]),
    "np.ma.MaskedArray": lambda: np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]),
    "np.dtype": lambda: np.dtype("f8"),
    "np.nditer": lambda: np.nditer(np.arange(3.0)),
    "np.broadcast": lambda: np.broadcast(np.arange(3.0), np.arange(3.0)),
    "np.flatiter": lambda: np.arange(6.0).flat,
    "np.ndindex": lambda: np.ndindex(2, 3),
    "np.ndenumerate": lambda: np.ndenumerate(np.arange(3.0)),
    "np.poly1d": lambda: np.poly1d([1.0, 2.0, 3.0]),
    "np.vectorize": lambda: np.vectorize(lambda x: x),
    "np.finfo": lambda: np.finfo(np.float64),
    "np.iinfo": lambda: np.iinfo(np.int32),
    "np.lib.npyio.NpzFile": _npz,
    "np.random.Generator": lambda: np.random.default_rng(0),
    "np.random.SeedSequence": lambda: np.random.SeedSequence(0),
    "np.random.BitGenerator": lambda: np.random.PCG64(0),
    "np.random.MT19937": lambda: np.random.MT19937(0),
    "np.random.PCG64": lambda: np.random.PCG64(0),
    "np.random.PCG64DXSM": lambda: np.random.PCG64DXSM(0),
    "np.random.Philox": lambda: np.random.Philox(0),
    "np.random.SFC64": lambda: np.random.SFC64(0),
}


def _enter_printoptions(**kwargs):
    """Enter and leave ``np.printoptions`` (a context manager checks its keywords only when it is entered).

    Args:
        **kwargs: The keywords to pass.
    """
    with np.printoptions(**kwargs):
        pass


#: Real calls for the names whose placeholder call stops before NumPy checks keywords: each takes the keyword to test
#: (with the value None) and calls NumPy with valid operands.
REAL_CALLS = {
    ("np", "frompyfunc"): lambda kw: np.frompyfunc(lambda x: x, 1, 1, **kw),
    ("np", "printoptions"): lambda kw: _enter_printoptions(**kw),
    ("ndarray", "(constructor)"): lambda kw: np.ndarray((2,), **kw),
    ("ndarray (NDArray<T>)", "(constructor)"): lambda kw: np.ndarray((2,), **kw),
    ("np.ma", "hstack"): lambda kw: np.ma.hstack([np.ma.array([1.0]), np.ma.array([2.0])], **kw),
    ("np.ma", "vstack"): lambda kw: np.ma.vstack([np.ma.array([1.0]), np.ma.array([2.0])], **kw),
    ("np.ma", "row_stack"): lambda kw: np.ma.row_stack([np.ma.array([1.0]), np.ma.array([2.0])], **kw),
    ("np.ma", "stack"): lambda kw: np.ma.stack([np.ma.array([1.0]), np.ma.array([2.0])], **kw),
}

#: Values a real call passes for a keyword whose None means something else than "no value" to NumPy's runtime check.
REAL_VALUES = {"offset": 0, "casting": "same_kind"}

#: Calls that pass a tuple where NumPy's stubs allow a sequence of integers (Q), per (group, name, parameter).
SEQUENCE_CALLS = {
    ("np", "roll", "shift"): lambda: np.roll(_cube(), shift=(1, 1), axis=(0, 1)),
    ("np", "roll", "axis"): lambda: np.roll(_cube(), shift=(1, 1), axis=(0, 1)),
    ("np", "squeeze", "axis"): lambda: np.squeeze(np.ones((1, 1, 3)), axis=(0, 1)),
    ("ndarray", "squeeze", "axis"): lambda: np.ones((1, 1, 3)).squeeze(axis=(0, 1)),
    ("np.ma", "squeeze", "axis"): lambda: np.ma.squeeze(np.ma.ones((1, 1, 3)), axis=(0, 1)),
    ("np.ma.MaskedArray", "squeeze", "axis"): lambda: np.ma.ones((1, 1, 3)).squeeze(axis=(0, 1)),
    ("np.matrix", "squeeze", "axis"): lambda: np.matrix([[1.0]]).squeeze(axis=(0, 1)),
    ("np.char.chararray", "squeeze", "axis"): lambda: np.char.array([[["ab"]]]).squeeze(axis=(0, 1)),
    ("np.char.chararray", "(constructor)", "shape"): lambda: np.char.chararray((2, 3)),
    ("np", "trim_zeros", "axis"): lambda: np.trim_zeros(np.pad(np.ones((2, 2)), 1), axis=(0, 1)),
    ("np", "gradient", "axis"): lambda: np.gradient(_cube(), axis=(0, 1)),
    ("np.ma", "expand_dims", "axis"): lambda: np.ma.expand_dims(np.ma.array(_cube()), axis=(0, 1)),
    ("np.ma", "compress", "axis"): lambda: np.ma.compress([True, False], np.ma.array(_cube()), axis=(0, 1)),
    ("np.ma.MaskedArray", "compress", "axis"): lambda: np.ma.array(_cube()).compress([True, False], axis=(0, 1)),
}

#: Calls that pass None where NumPy's default is None (L), per (group, name, parameter).
NULL_CALLS = {
    ("np", "arange", "stop"): lambda: np.arange(5, stop=None),
    ("np.ma", "arange", "stop"): lambda: np.ma.arange(5, stop=None),
    ("np", "histogram2d", "density"): lambda: np.histogram2d(np.ones(4), np.ones(4), density=None),
    ("np", "histogramdd", "density"): lambda: np.histogramdd(np.ones((5, 2)), density=None),
    ("np", "ndarray", "order"): lambda: np.ndarray((2,), order=None),
    ("np", "trim_zeros", "axis"): lambda: np.trim_zeros(np.ones(3), trim="fb", axis=None),
    ("np.linalg", "norm", "axis"): lambda: np.linalg.norm(np.ones((2, 2)), ord=None, axis=None),
    ("np.random", "hypergeometric", "size"): lambda: np.random.hypergeometric(5, 5, 3, size=None),
    ("np.random", "logseries", "size"): lambda: np.random.logseries(0.5, size=None),
    ("np.random", "dirichlet", "size"): lambda: np.random.dirichlet([1.0, 2.0], size=None),
    ("np.random", "multivariate_normal", "size"): lambda: np.random.multivariate_normal([0.0], [[1.0]], size=None),
}


def names_in(cell):
    """The parameter names of a ``"; "``-joined cell, NumPy's spelling (``axes (C#: premute)`` -> ``axes``).

    Args:
        cell: The CSV cell.

    Returns:
        The names.
    """
    return [re.sub(r" \(C#: [^)]*\)$", "", part.strip()) for part in cell.split("; ") if part.strip()]


def parse_signature(text):
    """Parse a signature the gap map prints (``sum(a, axis=None, *, where=<no value>)``).

    Args:
        text: The ``numpy_signature`` cell.

    Returns:
        ``[(name, kind, required)]``, kind one of ``pos_only`` / ``pos_or_kw`` / ``kw_only`` / ``var_pos`` /
        ``var_kw``; None when the text holds no parameter list.
    """
    if "(" not in text or not text.endswith(")"):
        return None
    inner = text[text.index("(") + 1:-1]
    params, kind = [], "pos_or_kw"
    for token in ([t.strip() for t in sd.split_top(inner)] if inner.strip() else []):
        if token == "/":
            params = [(n, "pos_only" if k == "pos_or_kw" else k, r) for n, k, r in params]
        elif token == "*":
            kind = "kw_only"
        elif token.startswith("**"):
            params.append((token[2:], "var_kw", False))
        elif token.startswith("*"):
            params.append((token[1:], "var_pos", False))
            kind = "kw_only"
        else:
            name, has_default, _ = token.partition("=")
            params.append((name.strip(), kind, not has_default))
    return params


def numpy_callable(group, name):
    """The NumPy callable one gap-map spelling compares against.

    Args:
        group: The group key (``np``, ``ndarray``, ``np.ma.MaskedArray``, ...).
        name: The member name, or ``(constructor)`` for the class call.

    Returns:
        The callable, or None when NumPy has none to call.
    """
    if name == sd.CONSTRUCTOR:
        for key, path, _, _ in sd.OBJECT_GROUPS:
            if key == group:
                return sd.resolve_numpy_path(path)
        return {"ndarray": np.ndarray, "np.random.Generator": np.random.Generator}.get(group)
    factory = INSTANCES.get(group)
    if factory is not None:
        return getattr(factory(), name, None)
    try:
        return getattr(sd.resolve_module(group), name, None)
    except AttributeError:
        return None


def keyword_outcome(call, keyword):
    """Run one call and say whether NumPy refused the keyword's NAME.

    Args:
        call: A function of one dict (the keyword arguments) that calls NumPy.
        keyword: The keyword to pass.

    Returns:
        ``(verdict, detail)``: ``"refused"`` when NumPy names the keyword as unknown (or takes no keywords at all), else
        ``"accepted"`` with what ended the call.
    """
    saved = np.get_printoptions()
    try:
        call({keyword: REAL_VALUES.get(keyword)})
        return "accepted", "ok"
    except Exception as error:  # noqa: BLE001  (every outcome but the refusal is an acceptance of the name)
        text = str(error)
        refused = isinstance(error, (TypeError, ValueError)) and any(
            phrase in text for phrase in REFUSAL) and (keyword in text or "takes no keyword" in text
                                                      or "not accepted" in text)
        return ("refused" if refused else "accepted"), f"{type(error).__name__}: {text[:150]}"
    finally:
        np.set_printoptions(**saved)


#: TypeError phrases with which a callable refuses one more POSITIONAL argument (C's PyArg_ParseTuple, Python's own).
ARITY_REFUSAL = ("positional argument", "takes at most", "takes no arguments", "takes exactly", "takes from")


def cs_param_names(best):
    """The parameter names of a C# signature cell (``NDArray conj(NDArray out = null)`` -> ``["out"]``).

    Args:
        best: The CSV's ``best`` cell: the closest overload as ``Return name(Type p = default, ...)``.

    Returns:
        The names in declaration order (``@`` and modifiers dropped); an empty list when the cell has no parameters.
    """
    inside = best[best.index("(") + 1:best.rindex(")")] if "(" in best else ""
    names, depth, current = [], 0, ""
    for ch in inside + ",":
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            declaration = current.split("=", 1)[0].strip()
            if declaration:
                names.append(declaration.split()[-1].lstrip("@"))
            current = ""
        else:
            current += ch
    return names


def positional_outcome(func, placeholders):
    """Whether NumPy takes one more POSITIONAL argument after ``placeholders`` positional ``None`` values.

    A C method parsed with ``PyArg_ParseTuple`` (``ndarray.conj``'s ``"|O&:conjugate"``) can take an argument by position
    that its docstring and its keyword parser do not show; the keyword probe alone reads it as absent.

    Args:
        func: The NumPy callable.
        placeholders: How many positional values precede the probed one.

    Returns:
        ``(verdict, detail)``: ``"refused"`` when NumPy's arity check names the extra argument, else ``"accepted"``.
    """
    saved = np.get_printoptions()
    try:
        func(*([None] * placeholders), None)
        return "accepted", "ok"
    except Exception as error:  # noqa: BLE001  (anything but the arity refusal means the argument was taken)
        text = str(error)
        refused = isinstance(error, TypeError) and any(phrase in text for phrase in ARITY_REFUSAL)
        return ("refused" if refused else "accepted"), f"{type(error).__name__}: {text[:150]}"
    finally:
        np.set_printoptions(**saved)


def check_numpy_keywords(rows):
    """Validate M (must be accepted) and X (must be refused) against NumPy's runtime.

    An X parameter must be refused as a keyword and, when its C# slot follows NumPy's positional parameters, as one more
    positional argument too (:func:`positional_outcome`): a C method may take by position what it refuses by name.

    Args:
        rows: The CSV rows with a NumPy counterpart.

    Returns:
        ``{"m_confirmed", "m_refuted", "x_confirmed", "x_refuted", "inconclusive", "by_slot"}`` verdict lists.
    """
    out = {"m_confirmed": [], "m_refuted": [], "x_confirmed": [], "x_refuted": [], "inconclusive": [],
           "by_slot": []}
    for row in rows:
        if row["mark_M"] != "●" and row["mark_X"] != "●":
            continue
        key = (row["module"], row["name"])
        params = parse_signature(row["numpy_signature"])
        func = numpy_callable(*key)
        if func is None or params is None:
            out["inconclusive"].append({"key": key, "reason": "NumPy's callable could not be reached"})
            continue
        placeholders = sum(1 for _, kind, required in params if required and kind in ("pos_only", "pos_or_kw"))
        kinds = {name: kind for name, kind, _ in params}
        placeholder_call = (lambda kw, f=func, n=placeholders: f(*([None] * n), **kw))
        call = placeholder_call
        if keyword_outcome(placeholder_call, BOGUS)[0] != "refused":
            call = REAL_CALLS.get(key)
            if call is None or keyword_outcome(call, BOGUS)[0] != "refused":
                out["inconclusive"].append({"key": key, "reason": "the call stops before NumPy checks keywords",
                                            "m": names_in(row["missing_any_overload"]) if row["mark_M"] == "●" else [],
                                            "x": names_in(row["numsharp_only_params"]) if row["mark_X"] == "●" else []})
                continue
        if row["mark_X"] == "●":
            positional = sum(1 for _, kind, _ in params if kind in ("pos_only", "pos_or_kw"))
            cs_names = cs_param_names(row["best"])
            for name in names_in(row["numsharp_only_params"]):
                verdict, detail = keyword_outcome(call, name)
                # A keyword NumPy refuses may still be its parameter BY POSITION (a C method's undocumented optional
                # argument): when the C# parameter sits right after NumPy's positional ones and NumPy has no *args, one
                # more positional value tells (2026-11-11: ndarray.conj / conjugate take `out` so, a.conj(o) is o).
                if (verdict == "refused" and not any(kind == "var_pos" for _, kind, _ in params)
                        and name in cs_names and cs_names.index(name) == positional):
                    taken, positional_detail = positional_outcome(func, positional)
                    if taken == "accepted":
                        verdict, detail = "accepted", f"refused as a keyword, accepted by position ({positional_detail})"
                out["x_confirmed" if verdict == "refused" else "x_refuted"].append(
                    {"key": key, "param": name, "detail": detail})
        if row["mark_M"] == "●":
            for name in names_in(row["missing_any_overload"]):
                if name.startswith("*") or kinds.get(name) in ("pos_only", "var_pos", "var_kw"):
                    out["by_slot"].append({"key": key, "param": name})
                    continue
                verdict, detail = keyword_outcome(call, name)
                out["m_confirmed" if verdict == "accepted" else "m_refuted"].append(
                    {"key": key, "param": name, "detail": detail})
    return out


#: A valid value for each required parameter of the random samplers (a generic call fills the others with 1.0): every
#: value stays inside the sampler's domain, so no rejection sampler can loop (logseries needs p < 1, zipf a > 1).
SAMPLER_ARGUMENTS = {
    "a": 2.0, "p": 0.5, "pvals": [0.5, 0.5], "alpha": [1.0, 1.0], "mean": 1.0, "cov": [[1.0]], "n": 5,
    "ngood": 5, "nbad": 5, "nsample": 3, "colors": [2, 2], "left": 0.0, "mode": 0.5, "right": 1.0, "low": 1,
    "kappa": 1.0, "mu": 0.0, "lam": 1.0, "df": 2.0, "dfnum": 2.0, "dfden": 2.0, "nonc": 1.0, "shape": 2.0,
}


def sampler_call(func, signature, param, value):
    """A call of a random sampler with every required parameter set to a domain-valid value and one keyword.

    Args:
        func: The NumPy sampler (a ``numpy.random`` function or a ``Generator`` / ``RandomState`` method).
        signature: Its ``numpy_signature`` cell, which names the required parameters.
        param: The keyword to test.
        value: Its value.

    Returns:
        A function of no arguments.
    """
    params = parse_signature(signature) or []
    required = [n for n, kind, req in params if req and kind in ("pos_only", "pos_or_kw")]
    args = [SAMPLER_ARGUMENTS.get(n, 1.0) for n in required]
    if "multivariate_normal" in signature:
        args = [[0.0], [[1.0]]]
    return lambda: func(*args, **{param: value})


def generic_value_call(group, name, param, value, signature=""):
    """A call with a real operand and one keyword value, for the names without an entry in the call tables.

    Args:
        group: The group key.
        name: The member name.
        param: The keyword.
        value: The value to pass.
        signature: The row's ``numpy_signature`` (the random samplers' required parameters come from it).

    Returns:
        A function of no arguments, or None when no operand fits the group.
    """
    cube = _cube()
    if group in ("np.random", "np.random.Generator") and name != sd.CONSTRUCTOR:
        func = numpy_callable(group, name)
        return sampler_call(func, signature, param, value) if func is not None else None
    if group in ("np", "np.ma", "np.linalg"):
        owner = {"np": np, "np.ma": np.ma, "np.linalg": np.linalg}[group]
        func = getattr(owner, name, None)
        operand = np.ma.array(cube) if group == "np.ma" else cube
        if func is None:
            return None
        if "percentile" in name or "quantile" in name:
            q = 0.5 if "quantile" in name else 50.0
            return lambda: func(operand, q, **{param: value})
        return lambda: func(operand, **{param: value})
    if group.startswith("np.random.") and name == sd.CONSTRUCTOR:
        cls = numpy_callable(group, name)
        return (lambda: cls(**{param: value})) if cls is not None else None
    receivers = {"ndarray": cube, "np.ma.MaskedArray": np.ma.array(cube),
                 "np.matrix": np.matrix([[1.0, 2.0], [3.0, 4.0]])}
    receiver = receivers.get(group)
    if receiver is None:
        return None
    method = getattr(receiver, name, None)
    return (lambda: method(**{param: value})) if method is not None else None


#: What a refusal of the tested VALUE says (a tuple where one integer is wanted, an unsupported axis); any other error
#: of a generic call is about the operand the check chose, so it leaves the finding unconfirmed rather than refuted.
VALUE_REFUSAL = re.compile(r"tuple|axis|sequence|integer|None", re.I)


def signature_default_is_none(group, name, param):
    """Whether NumPy's introspectable runtime signature gives ``param`` the default None.

    Args:
        group: The group key.
        name: The member name.
        param: The parameter.

    Returns:
        True or False from ``inspect.signature``, or None when the callable has no introspectable signature.
    """
    func = numpy_callable(group, name)
    try:
        parameter = sd.inspect.signature(func).parameters.get(param) if func is not None else None
    except (TypeError, ValueError):
        return None
    if parameter is None:
        return None
    return parameter.default is None


def check_values(rows, mark, pattern, table, value):
    """Validate Q (NumPy takes a tuple there) or L (NumPy takes None there).

    L is first read from NumPy's runtime signature (a default of None is the claim itself); a parameter without an
    introspectable signature, and every Q, is checked with a real call (``table``, else a generic call on a 3-D
    operand). A failing call refutes the finding only when its message is about the tested value
    (``VALUE_REFUSAL``); other failures leave it unreached.

    Args:
        rows: The CSV rows.
        mark: ``"Q"`` or ``"L"``.
        pattern: The regex that pulls the parameter names out of the probe cell.
        table: ``SEQUENCE_CALLS`` or ``NULL_CALLS``.
        value: The value a generic call passes: ``(0, 1)`` or None.

    Returns:
        ``{"confirmed", "refuted", "unreached"}`` verdict lists.
    """
    out = {"confirmed": [], "refuted": [], "unreached": []}
    column = {"Q": "probe_sequence", "L": "probe_null"}[mark]
    for row in rows:
        if row[f"mark_{mark}"] not in (("●", "◐") if mark == "Q" else ("●", "◐", "◦")):
            continue
        # A finding starts with its parameter (`axis`: ..., `axis` as a sequence: ...); the combined "None values in
        # one call" finding names its parameters only inside the call (axis: null). Backticked words elsewhere in a
        # finding are C# types (`int`, `Shape`) or keywords (`default`), not parameters.
        names = re.findall(pattern, row[column]) or re.findall(r"(\w+): null", row[column])
        for param in dict.fromkeys(names):
            key = (row["module"], row["name"], param)
            if mark == "L" and key not in table:
                by_signature = signature_default_is_none(*key)
                if by_signature is not None:
                    (out["confirmed"] if by_signature else out["refuted"]).append(
                        {"key": key, "detail": "runtime signature default"})
                    continue
            call = table.get(key) or generic_value_call(row["module"], row["name"], param, value,
                                                        row["numpy_signature"])
            if call is None:
                out["unreached"].append({"key": key})
                continue
            try:
                call()
                out["confirmed"].append({"key": key})
            except Exception as error:  # noqa: BLE001  (sorted below: the value's refusal or the operand's)
                detail = f"{type(error).__name__}: {str(error)[:150]}"
                (out["refuted"] if VALUE_REFUSAL.search(str(error)) else out["unreached"]).append(
                    {"key": key, "detail": detail})
    return out


def type_spelling(group):
    """The C# expression that names a group's type or facade in a ``nameof``: ``np.linalg``, ``NDMaskedArray``, ...

    Args:
        group: The group key.

    Returns:
        The expression, or None for a group the probe cannot spell.
    """
    if group in bp.INSTANCES:
        return bp.INSTANCES[group][0]
    if group == "ndarray":
        return "NDArray"
    if group.startswith("np"):
        return bp.MODULE_ACCESS.get(group, group)
    return None


def check_m_with_compiler(dump, types, rows, dll, work):
    """Validate every M parameter by compiling the base call with that parameter added by name, and every NumPy name
    a group lacks by compiling ``nameof`` of it (CS0117 confirms the absence).

    Args:
        dump: The module dump.
        types: The type dump.
        rows: The CSV rows.
        dll: The NumSharp.dll the dumps were taken from.
        work: A scratch directory for the probe project.

    Returns:
        ``{"confirmed", "refuted", "other", "by_slot", "unprobed", "absent_confirmed", "absent_refuted"}`` verdict
        lists; a refuted entry carries the expression that compiled.

    Raises:
        RuntimeError: From ``binding_probe.compile_errors`` (the compiler could not run).
    """
    wanted = {(r["module"], r["name"]): names_in(r["missing_any_overload"]) for r in rows if r["mark_M"] == "●"}
    _, analyzed, _, surface, _ = sd.run_analysis(dump, types, sd.load_stub_index())
    os.makedirs(work, exist_ok=True)
    local_dll = os.path.join(work, "NumSharp.dll")
    shutil.copyfile(dll, local_dll)
    lines = ["#nullable disable", "using System;", "using System.Collections.Generic;", "using System.IO;",
             "using System.Numerics;", "using NumSharp;", "using NumSharp.Generic;", "using NumSharp.IO;",
             "using NumSharp.Backends;", "namespace Probe {", "public static class P {"]
    for cs_type, local in sorted(set(bp.INSTANCES.values())):
        lines.append(f"static {cs_type} {local};")
    checks = []
    out = {"confirmed": [], "refuted": [], "other": [], "by_slot": [], "unprobed": [], "absent_confirmed": [],
           "absent_refuted": []}
    absent_checks = []
    for group, info in surface.items():
        spelled = type_spelling(group)
        if spelled is None:
            continue
        for name in info["np_absent"]:
            if not name.isidentifier() or name in bp.CS_KEYWORDS:
                continue  # A dunder or keyword name cannot be a C# member access; nothing to compile.
            lines.append(f"static void n{len(absent_checks):05d}() {{ _ = nameof({spelled}.{name}); }}")
            absent_checks.append({"key": (group, name), "expr": f"nameof({spelled}.{name})", "line": len(lines)})
    for row in analyzed:
        key = (row["module"], row["name"])
        if key not in wanted or row.get("np_params") is None:
            continue
        kinds = {p.name: p.kind for p in row["np_params"]}
        target = bp.callee(row, row["overload_list"])
        found, reason = bp.probes_for(row)
        base = None if reason else next((args for kind, _, args in found if kind == "R"), None)
        for name in wanted[key]:
            # The cell writes NumPy's *args / **kwargs with their stars, which no named argument spells; one syntax
            # error would also stop csc before it binds any other call of the file.
            kind = kinds.get(name.lstrip("*"), sd.P.VAR_POSITIONAL if name.startswith("*") else None)
            if kind in (sd.P.POSITIONAL_ONLY, sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD):
                out["by_slot"].append({"key": key, "param": name})
                continue
            if target is None or base is None:
                out["unprobed"].append({"key": key, "param": name, "reason": reason or "no C# expression"})
                continue
            call = f"{target}({', '.join(list(base) + [bp.cs_name(name) + ': default'])})"
            lines.append(f"static void m{len(checks):05d}() {{ {call}; }}")
            checks.append({"key": key, "param": name, "call": call, "line": len(lines)})
    lines += ["}", "}"]
    with open(os.path.join(work, "Probes.cs"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    errors = bp.compile_errors(work, "latest", local_dll)
    for check in checks:
        found_errors = errors.get(check.pop("line"), [])
        check["errors"] = [list(e) for e in found_errors]
        if not found_errors:
            out["refuted"].append(check)
        elif any(code == "CS1739" and f"'{check['param']}'" in message for code, message in found_errors):
            out["confirmed"].append(check)
        else:
            out["other"].append(check)
    for check in absent_checks:
        found_errors = errors.get(check.pop("line"), [])
        check["errors"] = [list(e) for e in found_errors]
        # CS0117: the type has no member of that name; CS1061: the same through an instance-property facade (np.ma,
        # np.random are objects); CS0234 / CS0426: no such namespace / nested type; CS0571: the name is only a
        # property's hidden accessor (get_fill_value is NDMaskedArray.fill_value's getter, not a callable method).
        absent = any(code in ("CS0117", "CS1061", "CS0234", "CS0426", "CS0571") for code, _ in found_errors)
        out["absent_confirmed" if absent else "absent_refuted"].append(check)
    return out


def probe_artifacts(probes):
    """The compile-probe failures that are not binding verdicts.

    Args:
        probes: The probe records of probe_results.json.

    Returns:
        One entry per failing probe whose errors include an ``ARTIFACT_CODES`` code.
    """
    out = []
    for probe in probes:
        for lang in ("12", "14"):
            codes = [e[0] for e in probe["errors"].get(lang, [])]
            if any(code in ARTIFACT_CODES for code in codes):
                out.append({"key": (probe["group"], probe["name"]), "kind": probe["kind"], "call": probe["call"],
                            "lang": lang, "errors": probe["errors"][lang]})
    return out


def main():
    """Run every check and write the validation JSON.

    Raises:
        RuntimeError: From the compiler check.
    """
    parser = argparse.ArgumentParser(description="Validate the gap map's findings against the compiler and NumPy.")
    for name in ("dump", "types", "dll", "csv", "probe", "work", "out"):
        parser.add_argument(name)
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    faulthandler.dump_traceback_later(300, repeat=True)  # names the call if NumPy ever hangs on a check
    with open(args.dump, encoding="utf-8") as handle:
        dump = json.load(handle)
    with open(args.types, encoding="utf-8") as handle:
        types = json.load(handle)
    with open(args.csv, encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["numpy_kind"] != "absent"]
    with open(args.probe, encoding="utf-8") as handle:
        probes = json.load(handle)["probes"]
    work = os.path.abspath(args.work)
    result = {"artifacts": probe_artifacts(probes),
              "m_compiler": check_m_with_compiler(dump, types, rows, os.path.abspath(args.dll), work)}
    here = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="gap-map-validation-"))  # whatever a NumPy call writes lands in a scratch folder
    try:
        result["numpy_keywords"] = check_numpy_keywords(rows)
        result["q_numpy"] = check_values(rows, "Q", r"`(\w+)` as a sequence", SEQUENCE_CALLS, (0, 1))
        result["l_numpy"] = check_values(rows, "L", r"(?:^|; )`(\w+)`:", NULL_CALLS, None)
    finally:
        os.chdir(here)
    faulthandler.cancel_dump_traceback_later()
    kw = result["numpy_keywords"]
    refuted = {
        "M": sorted({(*x["key"], x["param"]) for x in result["m_compiler"]["refuted"] + kw["m_refuted"]}),
        "X": sorted({(*x["key"], x["param"]) for x in kw["x_refuted"]}),
        "Q": sorted({tuple(x["key"]) for x in result["q_numpy"]["refuted"]}),
        "L": sorted({tuple(x["key"]) for x in result["l_numpy"]["refuted"]}),
        "B": sorted({tuple(x["key"]) for x in result["artifacts"]}),
        # A NumPy name the gap map lists as absent although a C# member access of it compiles.
        "absent": sorted({tuple(x["key"]) for x in result["m_compiler"]["absent_refuted"]}),
    }
    result["refuted"] = {mark: [list(item) for item in items] for mark, items in refuted.items()}
    result["summary"] = {
        "artifacts": len(result["artifacts"]),
        "m_compiler": {k: len(v) for k, v in result["m_compiler"].items()},
        "numpy_keywords": {k: len(v) for k, v in kw.items()},
        "q_numpy": {k: len(v) for k, v in result["q_numpy"].items()},
        "l_numpy": {k: len(v) for k, v in result["l_numpy"].items()},
        "refuted": {mark: len(items) for mark, items in refuted.items()},
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1, default=list)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
