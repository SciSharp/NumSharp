"""Compile NumPy-style calls of every compared np.* name against NumSharp, to find the calls that do not bind.

Usage (from this directory):
    python -I binding_probe.py <modules_dump.json> <types_dump.json> <NumSharp.dll> <work dir> <results.json>

Reflection (signature_diff.py) can say that a parameter exists in SOME overload; only the compiler can say whether a
NumPy call, written the way a port writes it, binds to one. The probes of a name are written from NumPy's signature (the
effective one signature_diff.py builds: runtime signature plus the stub / numpydoc parameters it hides):

  R  the required parameters only, positionally (a required keyword-only one by name): ``np.sum(a)``. NumPy's ``*args``
     gets the closest overload's own parameters at that slot (``np.einsum("x", new NDArray[] { a })``), and a
     parameter NumPy marks "not passed" (``np._NoValue``) counts as required when every overload requires it
     (``np.vectorize``'s ``pyfunc``, optional in NumPy only for the decorator form).
  S  the required ones plus NumPy's optional positional parameters up to one of them, positionally and in NumPy's
     order, with ordinary values: ``np.sum(a, 1)``, ``np.sum(a, 1, np.float64)``, ... (NumPy code writes
     ``a.sum(0)``, ``np.linspace(0, 1, 5)``). Fails when no overload has NumPy's order (a value of the wrong type for
     the slot), which reflection only checks for the closest overload.
  P  the required ones plus ONE optional parameter by name with an ordinary value (NumPy's default, or a value of the
     C# type where that default is None): ``np.empty(shape, device: "x")``. Fails when the parameter only binds
     together with another one (CS7036: ``device`` lives in an overload that requires ``dtype``).
  N  the required ones plus ONE parameter whose NumPy default is None, passed as ``null`` (``axis: null``): fails when
     the C# type cannot hold null (CS1503, CS0037) or the overloads tie on it (CS0121).
  D  for an N probe of a ``Shape`` parameter: the same call with ``default`` instead of ``null``.
  A  the required ones, then every optional NumPy parameter the C# side names, by name and with NumPy's default:
     ``np.sum(a, axis: null, dtype: null, @out: null)``. One call holding them all is the owner's one-overload rule;
     it fails when the parameters are split across overloads (CS1739) or the overloads tie (CS0121). A ``Shape``
     parameter whose NumPy default is None gets ``default``, NumSharp's documented spelling of ``size=None``.
  V  for an A probe that passes ``null``: the same call with the P values instead, which tells a combination that does
     not bind at all from one that only refuses the None values together.
  Q  for a parameter NumPy's stubs let take a sequence of integers (``axis=(0, 1)``, ``size=(2, 3)``, ``shape``):
     the call with ``new[] { 0, 1 }`` there (an optional one by name, a required one in its slot).
  U  the same with the C# tuple ``(0, 1)`` (NumSharp's ``Shape`` converts from tuples). Only when neither Q nor U
     binds does signature_diff.py report a Q mark.
  Y  for a ``dtype`` parameter: the call with a dtype string, ``dtype: "float64"`` (NumPy's ``dtype='float64'``). It
     binds where the parameter is a ``DType``, which converts from a string, and not where it is an ``NPTypeCode`` or
     a ``Type``.
  K  for NumPy's ``*args``: the call with several values at its slot, the way NumPy code passes them
     (``a.transpose(1, 0)``, ``np.ma.atleast_1d(x, y)``, ``np.meshgrid(x, y, z)``). R writes the closest overload's
     own parameters there, so it binds where a C# array parameter takes the elements as ONE array
     (``a.transpose(new int[] { 1 })``); K binds only where they can be passed one by one (``params``, or enough C#
     parameters). ``several_varargs_values`` chooses the values.
  W  for a parameter NumPy marks "not passed" (``np._NoValue``) that R passes because every overload requires it: the
     call without it, a call NumPy defines (``np.ma.where(condition)`` returns ``condition.nonzero()``,
     ``np.ma.clip(a)`` and ``mk.clip()`` clip with no bound). The forms C# cannot express are exempt
     (``OMISSION_NOT_PORTED``).

Each probe's argument VALUES are chosen from the C# parameter types (the closest overload's first; a delegate gets a
typed lambda, ``new Func<NDArray, int, NDArray>((p0, p1) => default(NDArray))``), so a failure is a binding failure, not a
wrong literal. A required parameter NumSharp renamed is passed positionally (the rename itself is signature_diff's M
mark). The file is compiled twice: LangVersion 12 (the default of a net8.0 consumer, which ignores
[OverloadResolutionPriority]) and the latest (C# 14). A call that fails only at 12 relies on that attribute. Names whose
call needs a value this tool cannot write (an open generic type) are not probed and are listed with the reason.

The results JSON (a list of probes with their call text and the errors per language version) is read by
``signature_diff.py --probe``, which adds the B (a NumPy-style call does not bind), L (``null`` refused where NumPy's
default is None) and Q (a sequence refused where NumPy's stubs allow one) marks.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signature_diff as sd  # noqa: E402  (the comparison this tool probes)

#: C# reserved keywords: a parameter with such a name is written ``@name`` in a named argument.
CS_KEYWORDS = frozenset(
    "abstract as base bool break byte case catch char checked class const continue decimal default delegate do double "
    "else enum event explicit extern false finally fixed float for foreach goto if implicit in int interface internal is "
    "lock long namespace new null object operator out override params private protected public readonly ref return "
    "sbyte sealed short sizeof stackalloc static string struct switch this throw true try typeof uint ulong unchecked "
    "unsafe ushort using virtual void volatile while".split())

#: How a module group's functions are reached in C# (np.char needs the verbatim identifier).
MODULE_ACCESS = {"np.char": "np.@char"}

#: Object groups: the C# type, and the local of that type the probe calls instance members on.
INSTANCES = {
    "ndarray": ("NDArray", "a"), "ndarray (NDArray<T>)": ("NDArray<double>", "ad"),
    "np.random.Generator": ("Generator", "gen"), "np.matrix": ("NDMatrix", "mx"),
    "np.char.chararray": ("NDCharArray", "ca"), "np.ma.MaskedArray": ("NDMaskedArray", "mk"),
    "np.dtype": ("DType", "dt"), "np.nditer": ("np.NDIterator", "it"), "np.broadcast": ("np.Broadcast", "bc"),
    "np.flatiter": ("np.FlatIterator", "fi"), "np.ndindex": ("np.NDIndex", "ni"),
    "np.ndenumerate": ("np.NDEnumerate", "ne"), "np.poly1d": ("poly1d", "p1"), "np.vectorize": ("Vectorized", "vz"),
    "np.finfo": ("finfo", "fn"), "np.iinfo": ("iinfo", "ii"), "np.lib.npyio.NpzFile": ("NpzFile", "nz"),
    "np.random.SeedSequence": ("SeedSequence", "ss"), "np.random.BitGenerator": ("BitGenerator", "bg"),
    "np.random.MT19937": ("MT19937", "mt"), "np.random.PCG64": ("PCG64", "pc"),
    "np.random.PCG64DXSM": ("PCG64DXSM", "pd"), "np.random.Philox": ("Philox", "ph"),
    "np.random.SFC64": ("SFC64", "sf"),
}

#: Optional positional parameters NumPy only accepts together with the next one (``where(condition, [x, y], /)``:
#: "either both or neither of x and y should be given"), per (group, name): a positional probe never stops at them.
JOINT_POSITIONAL = {("np", "where"): frozenset({"x"}), ("np.ma", "where"): frozenset({"x"})}

#: Names whose NumPy call without its "not passed" (``np._NoValue``) parameters has no C# counterpart by design, with
#: the reason; the W probe is not written for them. ``np.vectorize()`` without ``pyfunc`` returns a decorator
#: (``@np.vectorize(otypes=[float])`` over a ``def``): C# has no decorator syntax, so a port passes the function.
OMISSION_NOT_PORTED = {
    ("np", "vectorize"): "without `pyfunc` NumPy returns a decorator; C# has no decorator syntax",
}

#: One C# value per simple C# parameter type, the value a port would pass.
SIMPLE_VALUES = {
    "NDArray": "a", "object": "a", "NDMatrix": "mx", "NDCharArray": "ca", "NDMaskedArray": "mk",
    "int": "1", "long": "1", "short": "1", "byte": "1", "sbyte": "1", "uint": "1", "ulong": "1", "ushort": "1",
    "double": "1.0", "float": "1.0f", "decimal": "1m", "bool": "true", "string": "\"x\"", "char": "'C'",
    "DType": "np.float64", "Shape": "new Shape(2)", "Slice": "Slice.All", "Complex": "Complex.One",
    "Half": "(Half)1", "BigInteger": "BigInteger.One", "NDArray<double>": "ad",
}


#: A generic method's or type's own type parameter inside a friendly C# type (``T``, ``T1``, ``TR2``, ``TResult``,
#: ``TKey``): a value of such a type cannot be written without choosing the type argument, which the probe does not do.
_TYPE_PARAMETER = re.compile(r"(?<![\w.])T(?:\d+|R\d*|Result|Key|Value|Arg|DType|From|To|Out|In)?(?![\w.])")


def split_type_arguments(text):
    """Split a generic type's argument list at its top-level commas.

    Args:
        text: The text between the outer ``<`` and ``>`` (``NDArray, Func<int, int>, NDArray``).

    Returns:
        The argument types, stripped (``["NDArray", "Func<int, int>", "NDArray"]``).
    """
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def delegate_value(cs_type):
    """A C# delegate value of a delegate type: a typed lambda that returns ``default`` of its result type.

    The lambda is wrapped in ``new <type>(...)`` so it carries its type and also converts to a ``Delegate`` parameter.
    A ``Delegate`` parameter itself gets the one-array function a port of a Python callable would pass.

    Args:
        cs_type: ``Func<...>``, ``Action``, ``Action<...>`` or ``Delegate``.

    Returns:
        The expression, or None for a delegate type this tool does not write (``Predicate<T>``, an expression tree).
    """
    if cs_type == "Delegate":
        return "new Func<NDArray, NDArray>(p0 => p0)"
    if cs_type == "Action":
        return "new Action(() => { })"
    match = re.match(r"^(Func|Action)<(.*)>$", cs_type)
    if not match:
        return None
    args = split_type_arguments(match.group(2))
    if match.group(1) == "Action":
        names = ", ".join(f"p{i}" for i in range(len(args)))
        return f"new {cs_type}(({names}) => {{ }})"
    names = ", ".join(f"p{i}" for i in range(len(args) - 1))
    return f"new {cs_type}(({names}) => default({args[-1]}))"


def value_for(cs_type):
    """A C# expression of a given C# parameter type, as a port would pass it.

    Args:
        cs_type: The friendly C# type from the dump (``NDArray``, ``int?``, ``long[]``, ``np.MeshgridResult``,
            ``Func<NDArray, int, NDArray>``).

    Returns:
        The expression, or None when no such value can be written here (an open generic type, an unsupported
        delegate type).
    """
    if cs_type in SIMPLE_VALUES:
        return SIMPLE_VALUES[cs_type]
    base = cs_type[:-1] if cs_type.endswith("?") else cs_type
    if base in SIMPLE_VALUES:
        return SIMPLE_VALUES[base]
    if _TYPE_PARAMETER.search(base):
        return None
    if base.endswith("[]"):
        # One element of the element type: `new long[] { 1 }`, a jagged `new int[][] { new int[] { 1 } }`.
        element = base[:-2]
        inner = value_for(element)
        return None if inner is None else f"new {element}[] {{ {inner} }}"
    if base.startswith(("Func<", "Action", "Delegate", "Predicate", "Expression<")) or base in ("Func", "Delegate"):
        return delegate_value(base)
    return f"default({base})"


def cs_name(name):
    """Write a parameter name as a C# named argument's name (``out`` -> ``@out``).

    Args:
        name: The parameter name.

    Returns:
        The name, verbatim-escaped when it is a C# keyword.
    """
    return "@" + name if name in CS_KEYWORDS else name


def find_cs_param(overloads, best_index, np_param, slot, positional=False):
    """The C# parameter that serves one NumPy parameter, the closest overload's first.

    Args:
        overloads: The name's overload dictionaries.
        best_index: The closest overload's index (signature_diff).
        np_param: NumPy's Param.
        slot: Its position among NumPy's positional parameters (None for a keyword-only one).
        positional: True when the probe passes the value positionally (a required parameter). Then a NumPy name that no
            overload spells (a rename, already an M mark) is served by the parameter at the same slot, because a
            positional argument does not need the name, and a ``params`` array serves it too: the probe passes the
            whole array, as a port passes NumPy's sequence (``np.vstack((a, b))`` -> ``np.vstack(new[] { a, b })``).

    Returns:
        The C# parameter dictionary, or None when no overload has one.
    """
    order = [best_index] + [i for i in range(len(overloads)) if i != best_index]
    for i in order:
        cs = overloads[i]["params"]
        if np_param.kind == sd.P.POSITIONAL_ONLY:
            if slot is not None and slot < len(cs) and (positional or not cs[slot]["params"]):
                return cs[slot]
        else:
            for c in cs:
                if c["name"] == np_param.name and (positional or not c["params"]):
                    return c
    if positional and slot is not None:
        for i in order:
            cs = overloads[i]["params"]
            if slot < len(cs):
                return cs[slot]
    return None


def required_by_every_overload(overloads, slot):
    """Whether every C# overload has a required, non-``params`` parameter at a slot.

    NumPy marks some parameters "not passed" (``np._NoValue``) rather than required: ``np.vectorize(pyfunc=<no value>)``
    exists for the decorator form ``@np.vectorize(otypes=...)``, which C# does not have. When no overload can be
    called without the parameter, a port passes it, so the probe does too.

    Args:
        overloads: The name's overload dictionaries.
        slot: The positional slot.

    Returns:
        True when the slot is required in all overloads.
    """
    return bool(overloads) and all(
        slot < len(ov["params"]) and not ov["params"][slot]["optional"] and not ov["params"][slot]["params"]
        for ov in overloads)


def several_varargs_values(overloads, best_index, start, np_names):
    """The arguments a port writes at NumPy's ``*args`` slot when it passes several values there (the K probe).

    Walks the closest overload's parameters from the slot, as the R probe does, to find the parameter that carries
    NumPy's elements:

    - a parameter with a NumPy name ends the slot without a carrier;
    - an array parameter (``params`` or not, optional or not) carries them: two elements of its element type
      (``a.transpose(1, 1)`` for ``int[] premute``, ``np.einsum("x", a, a)`` after ``string subscripts``);
    - an optional parameter carries them: two values of its type (``np.gradient(a, a, a)`` for ``object spacing``);
    - a required parameter without a NumPy name keeps the R probe's value and is remembered.

    When the walk ends without a carrier, the last remembered parameter gets one more value of its type
    (``np.meshgrid(a, a, a)`` after R's ``np.meshgrid(a, a)``, ``np.ma.atleast_1d(a, a)`` after ``object a``). When
    nothing sits at the slot (``item()``), the elements are two values of another overload's ``params`` element type
    there, else two ``1`` placeholders: no overload has a parameter for them, so any value fails to bind (CS1501).

    Args:
        overloads: The name's overload dictionaries.
        best_index: The closest overload's index (signature_diff).
        start: The index of NumPy's ``*args`` slot in the C# parameter list (the positional values written before it).
        np_names: NumPy's parameter names other than ``*args`` / ``**kwargs``.

    Returns:
        The argument texts for the slot (at least two), or None when a value cannot be written (an open generic type,
        an unsupported delegate type).
    """
    written, last = [], None
    for c in overloads[best_index]["params"][start:]:
        if c["name"] in np_names:
            break
        cs_type = c["type"]
        if cs_type.endswith("[]") or c["optional"]:
            # The carrier: an array takes the elements (one array in R, one by one here), an optional parameter the
            # first of them.
            v = value_for(cs_type[:-2] if cs_type.endswith("[]") else cs_type)
            return None if v is None else written + [v, v]
        v = value_for(cs_type)
        if v is None:
            return None
        written.append(v)
        last = cs_type
    if last is not None:
        return written + [value_for(last)]
    for ov in overloads:
        cs = ov["params"]
        if start < len(cs) and cs[start]["params"] and cs[start]["type"].endswith("[]"):
            v = value_for(cs[start]["type"][:-2])
            if v is not None:
                return [v, v]
    return ["1", "1"]


def default_value(np_param, cs_param):
    """NumPy's default of an optional parameter, written as a C# value of the C# parameter's type.

    Args:
        np_param: NumPy's Param (it has a default).
        cs_param: The C# parameter that serves it.

    Returns:
        The C# expression, or None when the default cannot be written (``<no value>``, a tuple, a class).
    """
    text = np_param.default
    if text in ("<no value>", "<optional>", "...") or text is sd.REQUIRED:
        return None
    value = sd.literal(text)
    t = cs_param["type"].rstrip("?")
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if t in ("int", "long", "short", "byte", "sbyte", "uint", "ulong", "ushort"):
            return str(int(value)) if value == int(value) else None
        if t == "float":
            return repr(value) + "f"
        if t == "decimal":
            return repr(value) + "m"
        return repr(value) if "." in repr(value) or "e" in repr(value) else repr(value) + ".0"
    if isinstance(value, str):
        if t == "char" and len(value) == 1:
            return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
        return json.dumps(value)
    return None


def probes_for(row):
    """The probes of one compared name.

    Args:
        row: A compared row of ``signature_diff.run_analysis`` (it carries ``np_params``, ``overload_list`` and
            ``best_index``).

    Returns:
        ``(probes, reason)``: a list of ``(kind, param, args)`` with ``args`` the argument list text, and the reason
        when the name cannot be probed (else None).
    """
    params, overloads, best = row["np_params"], row["overload_list"], row.get("best_index", 0)
    policy = row["policy"]
    positional = [p for p in params if p.kind in (sd.P.POSITIONAL_ONLY, sd.P.POSITIONAL_OR_KEYWORD)]
    required = []
    written = set()  # the NumPy parameters the call already carries
    at = {}  # a required positional parameter's index in `required` (the Q probes replace its value)
    unset = []  # (index in `required`, name) of the "not passed" parameters R passes only because C# requires them
    for slot, p in enumerate(positional):
        unset_only = p.default == "<no value>" and required_by_every_overload(overloads, slot)
        if p.default is not sd.REQUIRED and not unset_only:
            continue
        c = find_cs_param(overloads, best, p, slot, positional=True)
        if c is None:
            return [], f"no C# parameter for NumPy's required `{p.name}`"
        v = value_for(c["type"])
        if v is None:
            return [], f"cannot write a {c['type']} for `{p.name}`"
        if unset_only:
            unset.append((len(required), p.name))
        at[p.name] = len(required)
        required.append(v)
        written.add(p.name)
    positional_count = len(required)
    var_positional = next((p for p in params if p.kind == sd.P.VAR_POSITIONAL), None)
    if var_positional is not None:
        # NumPy's *args carries the call's positional arguments (einsum's subscripts and operands, meshgrid's arrays),
        # and a port passes them: write the closest overload's parameters at that slot that carry no NumPy name, a
        # params array as one element.
        np_names = {p.name for p in params if p.kind not in (sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD)}
        cs = overloads[best]["params"]
        for c in cs[len(required):]:
            if c["name"] in np_names:
                break
            if c["params"]:
                v = value_for(c["type"][:-2]) if c["type"].endswith("[]") else None
                if v is not None:
                    required.append(v)
                break
            if c["optional"]:
                break
            v = value_for(c["type"])
            if v is None:
                return [], f"cannot write a {c['type']} for NumPy's *{var_positional.name}"
            required.append(v)
    var_end = len(required)  # the R values at NumPy's *args slot are required[positional_count:var_end]
    for p in params:
        # A keyword-only parameter without a default is required too, and only reachable by name.
        if p.kind == sd.P.KEYWORD_ONLY and p.default is sd.REQUIRED:
            c = find_cs_param(overloads, best, p, None)
            if c is None:
                return [], f"no C# parameter for NumPy's required keyword `{p.name}`"
            v = value_for(c["type"])
            if v is None:
                return [], f"cannot write a {c['type']} for `{p.name}`"
            required.append(f"{cs_name(p.name)}: {v}")
            written.add(p.name)
    probes = [("R", None, list(required))]
    if var_positional is not None:
        # NumPy code passes several values at *args (`a.transpose(1, 0)`, `np.meshgrid(x, y, z)`); R only shows that the
        # slot's C# parameters bind, which an array parameter does with ONE array of the elements.
        np_names = {p.name for p in params if p.kind not in (sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD)}
        several = several_varargs_values(overloads, best, positional_count, np_names)
        if several is not None:
            probes.append(("K", var_positional.name, required[:positional_count] + several + required[var_end:]))
    elif unset and (row["module"], row["name"]) not in OMISSION_NOT_PORTED:
        # NumPy defines the call without its "not passed" parameters (`np.ma.where(condition)`, `np.ma.clip(a)`); R
        # passes them because every C# overload requires them. Only a trailing run of the positional values can be
        # left out without moving a later value into its slot.
        dropped = {i for i, _ in unset}
        if dropped == set(range(positional_count - len(dropped), positional_count)):
            probes.append(("W", ",".join(name for _, name in unset),
                           [v for i, v in enumerate(required) if i not in dropped]))
    if var_positional is None:
        # NumPy code passes the leading optional parameters positionally (`a.sum(0)`, `np.linspace(0, 1, 5)`), so each
        # further positional slot gets a probe holding every slot up to it, in NumPy's order. A C# overload with
        # another order refuses the value (or, with equal types, takes it for another parameter, which only the slot
        # difference of the diff report shows). The required keyword-only arguments stay named, after them.
        head, tail, extra = required[:positional_count], [], required[positional_count:]
        joint = JOINT_POSITIONAL.get((row["module"], row["name"]), frozenset())
        for slot in range(positional_count, len(positional)):
            p = positional[slot]
            if p.name in policy:
                break  # an excluded parameter has no C# slot: a positional call cannot go past it
            c = find_cs_param(overloads, best, p, slot)
            if c is None:
                break
            v = default_value(p, c)
            value = v if v not in (None, "null") else value_for(c["type"])
            if value is None:
                break
            tail.append(value)
            if p.name not in joint:
                probes.append(("S", p.name, head + tail + extra))
    named, typed = [], []
    for p in params:
        if p.kind in (sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD, sd.P.POSITIONAL_ONLY) or p.name in policy:
            continue
        if p.name in written:
            continue
        slot = positional.index(p) if p in positional else None
        c = find_cs_param(overloads, best, p, slot)
        if c is None:
            continue
        v = default_value(p, c)
        is_none = sd.literal(p.default) is None and p.default not in ("<no value>", "<optional>", "...")
        if v == "null" and c["type"] == "Shape":
            # Shape is a struct: NumSharp spells NumPy's size=None as `default` (a documented house idiom), so the
            # all-parameters call uses it; the N and D probes below report how `null` fares.
            v = "default"
        # The value the presence probe passes: NumPy's default when it is an ordinary value, else one of the C# type
        # (None says nothing about whether the parameter binds, only about null).
        present = v if v not in (None, "null") else value_for(c["type"])
        if present is not None:
            probes.append(("P", p.name, required + [f"{cs_name(p.name)}: {present}"]))
        if v is not None:
            named.append((p.name, v))
            typed.append((p.name, present if v == "null" else v))
        if is_none:
            probes.append(("N", p.name, required + [f"{cs_name(p.name)}: null"]))
            if c["type"] == "Shape":
                probes.append(("D", p.name, required + [f"{cs_name(p.name)}: default"]))
    if named:
        probes.append(("A", None, required + [f"{cs_name(n)}: {v}" for n, v in named]))
        if any(v == "null" for _, v in named) and all(v is not None for _, v in typed):
            probes.append(("V", None, required + [f"{cs_name(n)}: {v}" for n, v in typed]))
    by_name = {p.name: p for p in params}
    dtype = by_name.get("dtype")
    if dtype is not None and dtype.name not in policy and dtype.kind not in (sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD):
        # NumPy code names a dtype by string (`dtype='int32'`): it binds where the parameter is a DType (which converts
        # from a string), not where it is an NPTypeCode or a Type (the T mark's parameters).
        if dtype.name in at:
            args = list(required)
            args[at[dtype.name]] = "\"float64\""
            probes.append(("Y", dtype.name, args))
        elif dtype.kind != sd.P.POSITIONAL_ONLY:
            slot = positional.index(dtype) if dtype in positional else None
            if find_cs_param(overloads, best, dtype, slot) is not None:
                probes.append(("Y", dtype.name, required + ["dtype: \"float64\""]))
    for name in row.get("sequence_params", []):
        # NumPy takes a sequence of integers here (`axis=(0, 1)`, `size=(2, 3)`): a port passes an int array, or the
        # C# tuple NumSharp's Shape converts from. A required one replaces its value in the required-only call.
        p = by_name.get(name)
        if p is None or p.name in policy or p.kind in (sd.P.VAR_POSITIONAL, sd.P.VAR_KEYWORD):
            continue
        for kind, value in (("Q", "new[] { 0, 1 }"), ("U", "(0, 1)")):
            if p.name in at:
                args = list(required)
                args[at[p.name]] = value
            elif p.kind == sd.P.POSITIONAL_ONLY:
                break  # an optional positional-only parameter cannot be named, and its slot may be another's in C#
            else:
                slot = positional.index(p) if p in positional else None
                if find_cs_param(overloads, best, p, slot) is None:
                    break  # no overload has it: the M mark
                args = required + [f"{cs_name(p.name)}: {value}"]
            probes.append((kind, p.name, args))
    return probes, None


def callee(row, overloads):
    """The C# expression a probe calls: a facade function, an instance member, a static member or a constructor.

    Args:
        row: The row (its ``module`` is the group key, ``name`` the member).
        overloads: Its overload dictionaries.

    Returns:
        The callee text (``np.linalg.norm``, ``a.reshape``, ``new NDArray``), or None for a group this tool cannot
        reach.
    """
    group, name = row["module"], row["name"]
    if group in INSTANCES:
        cs_type, local = INSTANCES[group]
        if name == sd.CONSTRUCTOR:
            return f"new {cs_type}"
        if all(ov["static"] and not ov.get("extension") for ov in overloads):
            return f"{cs_type}.{name}"
        return f"{local}.{name}"
    if name == sd.CONSTRUCTOR:
        return None
    return f"{MODULE_ACCESS.get(group, group)}.{name}"


CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <OutputType>Library</OutputType>
    <LangVersion>{lang}</LangVersion>
    <Nullable>disable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
    <NoWarn>$(NoWarn);CS0618;CS0612;CS0649;CS0169;CS8019;NU1900</NoWarn>
    <EnableDefaultCompileItems>false</EnableDefaultCompileItems>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="..\\Probes.cs" />
    <Reference Include="NumSharp"><HintPath>{dll}</HintPath></Reference>
  </ItemGroup>
</Project>
"""

#: An MSBuild error line of the probe file.
ERROR_LINE = re.compile(r"Probes\.cs\((\d+),\d+\): error (CS\d+): (.*?)(?: \[[^\]]*\])?$")


def compile_errors(work, lang, dll):
    """Compile the probe file at one language version and collect its errors per line.

    Args:
        work: The work directory (holds Probes.cs).
        lang: The LangVersion value (``12.0`` or ``latest``).
        dll: The NumSharp.dll to reference.

    Returns:
        ``{line: [(code, message)]}``.

    Raises:
        RuntimeError: dotnet failed without reporting a single error of the probe file (a broken setup, not a probe).
    """
    folder = os.path.join(work, "lang" + lang.replace(".", ""))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "Probes.csproj"), "w", encoding="utf-8") as fh:
        fh.write(CSPROJ.format(lang=lang, dll=dll))
    proc = subprocess.run(["dotnet", "build", os.path.join(folder, "Probes.csproj"), "-c", "Release", "-nologo",
                           "-v", "q", "-clp:ErrorsOnly;NoSummary"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    errors = {}
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        m = ERROR_LINE.search(line.strip())
        if m:
            entry = (m.group(2), m.group(3))
            bucket = errors.setdefault(int(m.group(1)), [])
            if entry not in bucket:
                bucket.append(entry)
    if proc.returncode != 0 and not errors:
        raise RuntimeError(f"dotnet build failed at LangVersion {lang}:\n{proc.stdout[-3000:]}\n{proc.stderr[-2000:]}")
    return errors


def main():
    """Write the probe file, compile it at C# 12 and at the latest C#, and write the results JSON.

    Raises:
        RuntimeError: From ``compile_errors``.
    """
    parser = argparse.ArgumentParser(description="Compile NumPy-style calls against NumSharp.")
    parser.add_argument("dump")
    parser.add_argument("types")
    parser.add_argument("dll", help="the NumSharp.dll the dumps were taken from")
    parser.add_argument("work", help="a scratch directory for the probe projects")
    parser.add_argument("out", help="the results JSON to write")
    args = parser.parse_args()
    with open(args.dump, encoding="utf-8") as fh:
        dump = json.load(fh)
    with open(args.types, encoding="utf-8") as fh:
        types = json.load(fh)
    _, rows, _, _, _ = sd.run_analysis(dump, types, sd.load_stub_index())
    os.makedirs(args.work, exist_ok=True)
    dll = os.path.join(args.work, "NumSharp.dll")
    shutil.copyfile(args.dll, dll)
    lines = ["#nullable disable", "using System;", "using System.Collections.Generic;", "using System.IO;",
             "using System.Numerics;", "using NumSharp;", "using NumSharp.Generic;", "using NumSharp.IO;",
             "namespace Probe {", "public static class P {"]
    for cs_type, local in sorted(set(INSTANCES.values())):
        lines.append(f"static {cs_type} {local};")
    probes, skipped = [], []
    for row in rows:
        if row.get("np_params") is None:
            continue
        target = callee(row, row["overload_list"])
        if target is None:
            skipped.append({"group": row["module"], "name": row["name"], "reason": "no C# expression reaches it"})
            continue
        found, reason = probes_for(row)
        if reason:
            skipped.append({"group": row["module"], "name": row["name"], "reason": reason})
            continue
        for kind, param, call_args in found:
            call = f"{target}({', '.join(call_args)})"
            lines.append(f"static void p{len(probes):05d}() {{ {call}; }}")
            probes.append({"group": row["module"], "name": row["name"], "kind": kind, "param": param, "call": call,
                           "line": len(lines), "errors": {}})
    lines += ["}", "}"]
    with open(os.path.join(args.work, "Probes.cs"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    for lang, label in (("12.0", "12"), ("latest", "14")):
        errors = compile_errors(args.work, lang, dll)
        for p in probes:
            p["errors"][label] = [list(e) for e in errors.get(p["line"], [])]
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"probes": probes, "skipped": skipped}, fh, indent=0)
    failed12 = sum(1 for p in probes if p["errors"]["12"])
    failed14 = sum(1 for p in probes if p["errors"]["14"])
    print(f"{len(probes)} probes ({len(skipped)} names skipped): {failed12} fail at C# 12, {failed14} at C# 14")


if __name__ == "__main__":
    main()
