"""Compare every function NumSharp exposes under np.* with NumPy 2.4.2's signature, and write the differences.

Usage (from this directory; the committed outputs are docs/plans/numpy-signature-diff.md and .csv):
    dotnet run --no-cache dump_modules.cs <scratch>/modules_dump.json
    python -I signature_diff.py <scratch>/modules_dump.json <md> <csv>
with <md> = ../../docs/plans/numpy-signature-diff.md and <csv> = ../../docs/plans/numpy-signature-diff.csv. The dump
is compiled from the working tree, so regenerate after any change to a public np.* signature.

The dump comes from ``dump_modules.cs`` (same directory): every ``[ModuleName]`` module of NumSharp.Core (np, ndarray,
np.linalg, np.fft, np.random, np.ma, np.emath, np.char, np.strings, np.dtypes, np.typing, np.polynomial and its
bases) plus the object API np.random.Generator. NumPy 2.4.2 must be importable.

Why one signature per name: the owner's rule of 2026-10-17 is "one whole api per function as numpy provides, remove
others". NumPy has ONE signature per function, so a name with several public C# overloads differs from NumPy by that
alone, and its CLOSEST overload (the fewest edits from NumPy's signature) is what the single surviving overload must
become. The report therefore lists, per name, the overload count and the closest overload's parameter differences.

How parameters are matched (the rules of docs/plans/api-parameter-parity.md §1):
  * positional-only (before NumPy's "/"): by SLOT. Its name cannot be passed in Python, so a C# name that differs is
    not a difference;
  * positional-or-keyword: by NAME, and it must sit at NumPy's slot, or a positional port binds the wrong argument
    (np.max(a, 1, o) binding o to keepdims);
  * keyword-only (after "*"): by NAME, in any position;
  * NumPy's ``*args``: a C# ``params`` array, or the unmatched C# parameters at that slot ("varargs" when they are not
    a params array, because a ported call then cannot pass the arguments one by one);
  * NumPy's ``**kwargs``: any C# parameters left over (keywords NumPy accepts through it; not a difference).

Difference kinds of the closest overload:
  missing      NumPy has the parameter, no overload of the name does.
  split        NumPy has the parameter, the closest overload does not, another overload does (merge them).
  renamed      a missing positional parameter and an extra C# parameter at the same slot.
  extra        the overload has a parameter NumPy does not.
  slot         a positional-or-keyword parameter at another position.
  required     NumPy defaults it, C# requires it (a ported call that omits it does not compile).
  default      both default it, to different values.
  null_default one side defaults to None/null and the other to a value (C# often spells "not given" as null; the
               behavior may still agree: only a probe can tell).
  varargs      NumPy's *args spelled as fixed C# parameters that are not a params array. Often forced: a C# params
               array must be the LAST parameter and nothing optional may follow it, so NumPy's keyword-only
               parameters after *args (einsum's out=, savez's allow_pickle=) cannot share a signature with it.
  policy       missing, but excluded by the parity plan (§3): like=, subok=, a ufunc's casting= / order= /
               signature=, the field order= of sort / argsort / partition / argpartition, einsum_path's einsum_call=.
  optional     C# defaults what NumPy requires (harmless, listed only in the CSV).
  kwargs       C# parameters NumPy takes through **kwargs (listed only in the CSV).

Where NumPy's signature comes from:
  * inspect.signature, for functions, ufuncs, classes, Cython methods and numpy.ma's wrappers that set __signature__;
  * the docstring's first line for ndarray methods (their documented signature: inspect folds their keywords into
    **kwargs, and NumPy's C forwarding accepts some keywords positionally that the documentation makes keyword-only;
    the documented form is the contract a port follows);
  * the wrapped callable for numpy.ma's _MaskedUnaryOperation / _MaskedBinaryOperation / _DomainedBinaryOperation,
    which forward (*args, **kwargs) to it after their own ``a`` (and ``b``);
  * ndarray's documented method for np.ma.copy / diagonal / repeat / swapaxes, whose ``(a, *args, **params)`` reaches
    it through MaskedArray's forwarding methods.
"""
import ast
import csv
import inspect
import json
import math
import re
import sys

import numpy as np

P = inspect.Parameter

#: Marker for "the parameter has no default": a NumPy parameter every call must pass.
REQUIRED = object()

#: Excluded everywhere (plan §3): NumPy's __array_function__ protocol and ndarray subclasses, neither of which
#: NumSharp has.
POLICY_ALL = frozenset({"like", "subok"})

#: Ufunc keywords NumSharp's ufunc surface does not model (plan §3).
POLICY_UFUNC = frozenset({"casting", "order", "signature"})

#: Per-name exclusions (plan §3): order= is the field order of a structured dtype (NumSharp has none);
#: einsum_call= is NumPy-internal.
POLICY_BY_NAME = {"sort": frozenset({"order"}), "argsort": frozenset({"order"}), "partition": frozenset({"order"}),
                  "argpartition": frozenset({"order"}), "einsum_path": frozenset({"einsum_call"})}

#: numpy.ma classes whose instances forward (*args, **kwargs) to the callable in their ``f`` attribute, with the
#: number of their own leading parameters (``a``, or ``a, b``).
MA_FORWARDERS = {"_MaskedUnaryOperation": 1, "_MaskedBinaryOperation": 2, "_DomainedBinaryOperation": 2}

#: Difference kinds in display order; HARD_KINDS make the function's signature differ. "split" is a parameter the
#: closest overload lacks while ANOTHER overload of the name has it (merging the overloads is the fix, not an
#: implementation), "missing" one that no overload has.
HARD_KINDS = ("missing", "split", "renamed", "extra", "slot", "required", "default", "varargs")
ALL_KINDS = HARD_KINDS + ("null_default", "policy", "optional", "kwargs", "notes")

#: Edits each difference costs when the closest overload is chosen, weighted by the work: a missing parameter needs
#: an implementation (2), a rename / move / removal / new default value is one edit (1), making a required parameter
#: optional is half of one (0.5), a null default is a quarter (a likely stand-in for NumPy's value), and policy /
#: optional / kwargs / notes are free. So among the overloads of one name the fullest wins: `mean(a)` lacks five of
#: NumPy's parameters (10) where `mean(a, int? axis, ...)` lacks two, misplaces one and requires two (6.0).
COST = {"missing": 2.0, "split": 2.0, "renamed": 1.0, "extra": 1.0, "slot": 1.0, "required": 0.5, "default": 1.0,
        "varargs": 1.0, "null_default": 0.25, "policy": 0.0, "optional": 0.0, "kwargs": 0.0, "notes": 0.0}


class Soft:
    """A NumPy default that C# cannot write as a constant (a class such as ``dtype=float``, a function, an object).

    C# spells such a default as null, so null against a Soft value is not a difference; any other C# default is.
    """

    def __init__(self, text):
        """Keep NumPy's text of the default.

        Args:
            text: The default as NumPy shows it (its repr).
        """
        self.text = text


class Param:
    """One parameter of NumPy's signature.

    Attributes:
        name: The parameter's name.
        kind: An ``inspect.Parameter`` kind: POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY or
            VAR_KEYWORD.
        default: ``REQUIRED``, or the default as text: its repr, ``<no value>`` for ``np._NoValue`` (NumPy does not
            pass the argument on, so any C# default matches it) or ``<optional>`` for a docstring's bracketed
            parameter without a stated default.
    """

    def __init__(self, name, kind, default):
        """Store the three fields.

        Args:
            name: The parameter's name.
            kind: The ``inspect.Parameter`` kind.
            default: ``REQUIRED`` or the default's text.
        """
        self.name, self.kind, self.default = name, kind, default

    def text(self):
        """Render the parameter as a Python signature does.

        Returns:
            ``*name``, ``**name``, ``name=default`` or ``name``.
        """
        prefix = "*" if self.kind == P.VAR_POSITIONAL else "**" if self.kind == P.VAR_KEYWORD else ""
        return prefix + self.name + ("" if self.default is REQUIRED else "=" + self.default)


def default_text(value):
    """Convert an ``inspect`` default to the text a ``Param`` keeps.

    Args:
        value: ``inspect.Parameter.default``.

    Returns:
        ``REQUIRED`` for no default, ``<no value>`` for ``np._NoValue``, else the default's repr.
    """
    if value is P.empty:
        return REQUIRED
    if value is np._NoValue:
        return "<no value>"
    return repr(value)


def from_inspect(sig):
    """Convert an ``inspect.Signature`` to Params, dropping a bound or unbound ``self``.

    Args:
        sig: The signature.

    Returns:
        The list of Params in signature order.
    """
    return [Param(p.name, p.kind, default_text(p.default)) for p in sig.parameters.values() if p.name != "self"]


def split_top(text):
    """Split a parameter list at its top-level commas: commas inside brackets, braces, parentheses or quotes do not
    split (``shape=(2, 3)`` stays one piece).

    Args:
        text: The text between a signature's parentheses.

    Returns:
        The pieces, unstripped.
    """
    parts, depth, quote, cur = [], 0, None, []
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def from_doc(doc, name):
    """Parse the signature line of a NumPy docstring, such as ``a.min(axis=None, out=None, *, keepdims=<no value>)``
    or ``a.view([dtype][, type])``, into Params.

    A bracketed parameter is optional without a stated default (``<optional>``). A ``/`` makes the parameters before it
    positional-only, a ``*`` or ``*args`` makes the ones after it keyword-only.

    Args:
        doc: The docstring (may be empty).
        name: The function's name, located in the docstring as ``name(`` after a start, a space or a dot.

    Returns:
        The list of Params, or None when the docstring has no such line.
    """
    if not doc:
        return None
    m = re.search(r"(?:^|\s|\.)" + re.escape(name) + r"\(", doc)
    if not m:
        return None
    start, depth, i = m.end(), 1, m.end()
    while i < len(doc) and depth:
        depth += {"(": 1, ")": -1}.get(doc[i], 0)
        i += 1
    inner = " ".join(doc[start:i - 1].split())
    tokens = []
    # "[dtype][, type]": move each bracket's leading comma out so the split sees "[dtype]" and "[type]".
    for raw in split_top(inner.replace("[,", ",[")):
        optional = "[" in raw or "]" in raw
        token = raw.replace("[", "").replace("]", "").strip()
        if token:
            tokens.append((token, optional))
    params, keyword_only = [], False
    for token, optional in tokens:
        if token == "/":
            # Everything before "/" is positional-only.
            for p in params:
                if p.kind == P.POSITIONAL_OR_KEYWORD:
                    p.kind = P.POSITIONAL_ONLY
            continue
        if token == "*":
            keyword_only = True
            continue
        if token.startswith("**"):
            params.append(Param(token[2:], P.VAR_KEYWORD, REQUIRED))
            continue
        if token.startswith("*"):
            params.append(Param(token[1:], P.VAR_POSITIONAL, REQUIRED))
            keyword_only = True
            continue
        pname, _, dflt = token.partition("=")
        pname, dflt = pname.strip(), dflt.strip()
        if not re.match(r"^[A-Za-z_]\w*$", pname):
            continue
        default = dflt if dflt else ("<optional>" if optional else REQUIRED)
        params.append(Param(pname, P.KEYWORD_ONLY if keyword_only else P.POSITIONAL_OR_KEYWORD, default))
    return params


def format_params(params):
    """Render Params as a Python signature's parameter list, with the ``/`` and ``*`` markers.

    Args:
        params: The Params in order.

    Returns:
        The text between the parentheses.
    """
    shown = []
    slash_pending = any(p.kind == P.POSITIONAL_ONLY for p in params)
    star_done = False
    for p in params:
        if slash_pending and p.kind != P.POSITIONAL_ONLY:
            shown.append("/")
            slash_pending = False
        if p.kind == P.VAR_POSITIONAL:
            star_done = True
        if p.kind == P.KEYWORD_ONLY and not star_done:
            shown.append("*")
            star_done = True
        shown.append(p.text())
    if slash_pending:
        shown.append("/")
    return ", ".join(shown)


def ma_forwarded(obj):
    """The signature a numpy.ma forwarder accepts: its own ``a`` (``a, b``), then the wrapped callable's parameters
    after its first one (two), which ``__call__(self, a, *args, **kwargs)`` passes on.

    Args:
        obj: Any NumPy object.

    Returns:
        ``(params, wrapped)``, or None when ``obj`` is not such a forwarder or its callable has no signature.
    """
    count = MA_FORWARDERS.get(type(obj).__name__)
    wrapped = getattr(obj, "f", None)
    if count is None or not callable(wrapped):
        return None
    try:
        inner = from_inspect(inspect.signature(wrapped))
    except (TypeError, ValueError):
        return None
    head = [Param("a", P.POSITIONAL_OR_KEYWORD, REQUIRED)]
    if count == 2:
        head.append(Param("b", P.POSITIONAL_OR_KEYWORD, REQUIRED))
    return head + inner[count:], wrapped


def numpy_signature(module_key, name, obj):
    """Find NumPy's signature of one name.

    Args:
        module_key: The dump's module key (``np``, ``ndarray``, ``np.linalg``, ``np.random.Generator``, ...).
        name: The function's name.
        obj: NumPy's object of that name in the module, or None when NumPy has none.

    Returns:
        ``(kind, params, text, is_ufunc)``: kind is one of absent, module, property, attribute, method, function,
        class, ufunc, ma-forward, ma-method; params is a list of Params or None when NumPy states no signature; text is the
        signature as shown in the report; is_ufunc says whether the ufunc exclusions apply.
    """
    if obj is None:
        return "absent", None, "", False
    if inspect.ismodule(obj):
        return "module", None, "", False
    if module_key == "ndarray":
        raw = inspect.getattr_static(np.ndarray, name, None)
        if type(raw).__name__ in ("getset_descriptor", "member_descriptor", "property"):
            return "property", None, "", False
        params = from_doc(getattr(obj, "__doc__", "") or "", name)
        if params is not None:
            return "method", params, f"a.{name}({format_params(params)})", False
    forwarded = ma_forwarded(obj)
    if forwarded is not None:
        params, wrapped = forwarded
        return "ma-forward", params, f"{name}({format_params(params)})", isinstance(wrapped, np.ufunc)
    if isinstance(obj, np.ufunc):
        kind = "ufunc"
    elif inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "method" if module_key.endswith("Generator") else "function"
    else:
        return "attribute", None, "", False
    try:
        params = from_inspect(inspect.signature(obj))
    except (TypeError, ValueError):
        params = from_doc(getattr(obj, "__doc__", "") or "", name)
    if params is None:
        return kind, None, "(NumPy states no signature)", kind == "ufunc"
    if module_key == "np.ma" and [(p.name, p.kind) for p in params] ==             [("a", P.POSITIONAL_OR_KEYWORD), ("args", P.VAR_POSITIONAL), ("params", P.VAR_KEYWORD)]:
        # np.ma.copy / diagonal / repeat / swapaxes are _frommethod wrappers of MaskedArray methods that are themselves
        # _arraymethod forwarders: the call reaches ndarray's method, so its documented parameters are the real ones.
        method = from_doc(getattr(getattr(np.ndarray, name, None), "__doc__", "") or "", name)
        if method is not None:
            # A method parameter that is positional-only (swapaxes' axis1, axis2) can only arrive through *args, which
            # also makes `a` effectively positional; Python cannot order a keyword-capable `a` before it otherwise.
            head = Param("a", P.POSITIONAL_ONLY if any(p.kind == P.POSITIONAL_ONLY for p in method)
                         else P.POSITIONAL_OR_KEYWORD, REQUIRED)
            return "ma-method", [head] + method, f"{name}({format_params([head] + method)})", False
    return kind, params, f"{name}({format_params(params)})", kind == "ufunc"


def literal(text):
    """Turn NumPy's default text into a comparable Python value.

    Args:
        text: A repr from ``default_text``.

    Returns:
        None, a bool, a float (for every number), a str, a tuple / list, or a ``Soft`` for a value C# cannot write.
    """
    if text in ("inf", "-inf", "nan"):
        return float(text)
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return Soft(text)
    if isinstance(value, bool) or value is None or isinstance(value, (str, tuple, list)):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return Soft(text)


def cs_default_value(prm):
    """Turn a C# parameter's default (from the dump) into a comparable Python value.

    Args:
        prm: A parameter dictionary of the dump.

    Returns:
        None for null, a bool, a float, a str (a C# string or char), or ``("enum", text)`` / ``("other", text)``.
    """
    kind, text = prm["defaultKind"], prm["default"]
    if kind == "null":
        return None
    if kind == "bool":
        return text == "true"
    if kind == "number":
        try:
            return float(text)
        except ValueError:
            return ("other", text)
    if kind == "string":
        return json.loads(text)
    if kind == "char":
        return text[1:-1]
    return ("enum" if kind == "enum" else "other", text)


def same_value(npv, csv_value):
    """Decide whether a NumPy default and a C# default are the same value.

    Args:
        npv: From ``literal``.
        csv_value: From ``cs_default_value``.

    Returns:
        True when they agree: equal bools (a bool never equals a number), equal numbers (NaN equals NaN), equal
        strings, or a C# enum whose member name is NumPy's string ignoring case and underscores.
    """
    if isinstance(npv, bool) or isinstance(csv_value, bool):
        return type(npv) is type(csv_value) and npv == csv_value
    if isinstance(npv, float) and isinstance(csv_value, float):
        return npv == csv_value or (math.isnan(npv) and math.isnan(csv_value))
    if isinstance(npv, str) and isinstance(csv_value, str):
        return npv == csv_value
    if isinstance(npv, str) and isinstance(csv_value, tuple) and csv_value[0] == "enum":
        member = csv_value[1].rsplit(".", 1)[-1]
        return member.replace("_", "").lower() == npv.replace("_", "").lower()
    return False


def compare_default(np_param, cs_param):
    """Classify the difference between two defaults of one parameter.

    Args:
        np_param: NumPy's Param (it has a default).
        cs_param: The C# parameter dictionary (it has a default).

    Returns:
        ``(kind, text)``: kind is None (no difference), ``default`` (two values differ) or ``null_default`` (one side
        is None / null, the other a value); text shows both defaults.
    """
    text = np_param.default
    if text in ("<no value>", "<optional>"):
        return None, ""
    npv, csv_value = literal(text), cs_default_value(cs_param)
    shown = f"{text} vs {cs_param['default']}"
    if np_param.name == "where" and npv is True and csv_value is None:
        return None, ""  # where=True is "no mask", which NumSharp spells null.
    if isinstance(npv, Soft):
        return (None, "") if csv_value is None else ("default", shown)
    if npv is None and csv_value is None:
        return None, ""
    if npv is None or csv_value is None:
        return "null_default", shown
    if isinstance(npv, (tuple, list)):
        return "default", shown
    return (None, "") if same_value(npv, csv_value) else ("default", shown)


def check_requiredness(np_param, cs_param, diff):
    """Record whether one matched parameter is required on one side only, or defaults differently.

    Args:
        np_param: NumPy's Param.
        cs_param: The matched C# parameter dictionary.
        diff: The difference dictionary to extend (``required``, ``optional``, ``default``, ``null_default``).
    """
    np_required = np_param.default is REQUIRED
    cs_required = not cs_param["optional"]
    if np_required and not cs_required:
        diff["optional"].append(np_param.name)
    elif not np_required and cs_required:
        diff["required"].append(np_param.name)
    elif not np_required:
        kind, text = compare_default(np_param, cs_param)
        if kind:
            diff[kind].append(f"{np_param.name}: {text}")


def compare(np_params, overload, policy):
    """Compare one C# overload with NumPy's parameters.

    Args:
        np_params: NumPy's Params.
        overload: One overload dictionary of the dump.
        policy: The names this function excludes by policy.

    Returns:
        A dictionary from every kind of ``ALL_KINDS`` to the list of its findings.
    """
    cs = overload["params"]
    names = [c["name"] for c in cs]
    np_names = {p.name for p in np_params}
    used = set()
    diff = {k: [] for k in ALL_KINDS}
    positional = [p for p in np_params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    unmatched_positional = {}
    for slot, p in enumerate(positional):
        if p.kind == P.POSITIONAL_ONLY:
            # The name cannot be passed in Python, so any C# parameter at the slot serves, unless it carries the name
            # of another NumPy parameter (then it is that parameter, misplaced).
            fits = slot < len(cs) and slot not in used and not cs[slot]["params"] \
                and (cs[slot]["name"] == p.name or cs[slot]["name"] not in np_names)
            idx = slot if fits else None
        else:
            idx = names.index(p.name) if p.name in names else None
            if idx is not None and idx != slot:
                diff["slot"].append(f"{p.name}: NumPy #{slot}, C# #{idx}")
        if idx is None:
            unmatched_positional[slot] = p
            continue
        used.add(idx)
        check_requiredness(p, cs[idx], diff)
    unmatched_keyword = []
    for p in np_params:
        if p.kind != P.KEYWORD_ONLY:
            continue
        idx = names.index(p.name) if p.name in names else None
        if idx is None or idx in used:
            unmatched_keyword.append(p)
            continue
        used.add(idx)
        check_requiredness(p, cs[idx], diff)
    var_positional = next((p for p in np_params if p.kind == P.VAR_POSITIONAL), None)
    var_keyword = next((p for p in np_params if p.kind == P.VAR_KEYWORD), None)
    absorbed = []
    if var_positional is not None:
        # *args takes the run of unmatched C# parameters that starts at its slot; the run ends at a C# parameter that
        # carries the name of NumPy's **kwargs (savez's `IDictionary<string, NDArray> kwds` spells **kwds, not *args).
        i = len(positional)
        stop_names = np_names - {var_positional.name}
        while i < len(cs) and i not in used and cs[i]["name"] not in stop_names:
            absorbed.append(i)
            if cs[i]["params"]:
                break
            i += 1
        used.update(absorbed)
        star = "*" + var_positional.name
        spelled = ", ".join(("params " if cs[i]["params"] else "") + f"{cs[i]['type']} {cs[i]['name']}" for i in absorbed)
        forwarder = [p.kind for p in np_params] == [P.VAR_POSITIONAL, P.VAR_KEYWORD]
        if not absorbed:
            diff["missing"].append(star)
        elif any(cs[i]["params"] for i in absorbed):
            diff["notes"].append(f"{star} as {spelled}")
        elif forwarder:
            diff["notes"].append(f"NumPy forwards (*args, **kwargs); C# spells {spelled}")
        else:
            diff["varargs"].append(f"{star} as {spelled} (no params array)")
    if var_keyword is not None:
        rest = [i for i in range(len(cs)) if i not in used]
        diff["kwargs"].extend(cs[i]["name"] for i in rest)
        used.update(rest)
    extra_at = {i: c["name"] for i, c in enumerate(cs) if i not in used}
    for slot, p in unmatched_positional.items():
        if p.name in policy:
            diff["policy"].append(p.name)
        elif slot in extra_at:
            diff["renamed"].append(f"{p.name} -> {extra_at.pop(slot)}")
        else:
            diff["missing"].append(p.name)
    for p in unmatched_keyword:
        (diff["policy"] if p.name in policy else diff["missing"]).append(p.name)
    diff["extra"].extend(extra_at[i] for i in sorted(extra_at))
    return diff


def split_missing(diff, overloads, best_index):
    """Move the closest overload's missing parameters that another overload of the name has into ``split``: the fix
    for them is merging the overloads, not implementing the parameter.

    Args:
        diff: The closest overload's difference dictionary (changed in place).
        overloads: All overload dictionaries of the name.
        best_index: The closest overload's index.
    """
    others = [ov for i, ov in enumerate(overloads) if i != best_index]
    names = {c["name"] for ov in others for c in ov["params"]}
    has_params = any(c["params"] for ov in others for c in ov["params"])
    keep = []
    for name in diff["missing"]:
        present = has_params if name.startswith("*") else name in names
        (diff["split"] if present else keep).append(name)
    diff["missing"] = keep


def cost(diff, overload):
    """The number of edits that would make an overload NumPy's signature, used to pick the closest overload.

    Args:
        diff: The overload's difference dictionary.
        overload: The overload dictionary (a generic, obsolete or hidden overload costs a little more, so a plain
            overload wins a tie).

    Returns:
        The weighted edit count.
    """
    total = sum(COST[k] * len(v) for k, v in diff.items())
    if overload["generic"]:
        total += 0.5
    if overload["obsolete"]:
        total += 0.5
    if overload["hidden"]:
        total += 0.25
    return total


def cs_signature(name, ov):
    """Render one C# overload as C# source reads, with its binding attributes.

    Args:
        name: The method's name.
        ov: The overload dictionary.

    Returns:
        ``Return name<T>(Type p = default, ...)`` with ``[ORPA n, obsolete, hidden]`` marks when present.
    """
    parts = []
    for c in ov["params"]:
        piece = ("params " if c["params"] else "") + (c["byRef"] + " " if c["byRef"] else "") + f"{c['type']} {c['name']}"
        if c["optional"]:
            piece += " = " + (c["default"] if c["default"] is not None else "default")
        parts.append(piece)
    generic = "<" + ", ".join(ov["generic"]) + ">" if ov["generic"] else ""
    marks = []
    if ov["orpa"] is not None:
        marks.append(f"ORPA {ov['orpa']}")
    if ov["obsolete"]:
        marks.append("obsolete")
    if ov["hidden"]:
        marks.append("hidden")
    return f"{ov['returns']} {name}{generic}(" + ", ".join(parts) + ")" + (" [" + ", ".join(marks) + "]" if marks else "")


def resolve_module(key):
    """Find NumPy's module or class for a dump key by attribute access (``np.emath`` is an attribute, not an
    importable module; ``np.polynomial`` loads lazily).

    Args:
        key: ``np``, ``ndarray`` or a dotted ``np.``-path.

    Returns:
        The module or class.

    Raises:
        AttributeError: NumPy has no such attribute (the dump names a module NumPy 2.4.2 lacks).
    """
    if key == "ndarray":
        return np.ndarray
    target = np
    for part in key.split(".")[1:]:
        target = getattr(target, part)
    return target


def numpy_public_names(target):
    """NumPy's public names of a module or class, without submodules.

    Args:
        target: From ``resolve_module``.

    Returns:
        The names: a module's ``__all__`` (its ``dir`` without underscores when it has none), a class's ``dir``
        without underscores.
    """
    if inspect.isclass(target):
        return {n for n in dir(target) if not n.startswith("_")}
    names = getattr(target, "__all__", None)
    if names is None:
        names = [n for n in dir(target) if not n.startswith("_")]
    return {n for n in names if not inspect.ismodule(getattr(target, n, None))}


def policy_for(name, is_ufunc):
    """The parameter names one function excludes by policy (plan §3).

    Args:
        name: The function's name.
        is_ufunc: Whether NumPy's object is a ufunc (or a numpy.ma forwarder of one).

    Returns:
        The set of excluded names.
    """
    return POLICY_ALL | (POLICY_UFUNC if is_ufunc else frozenset()) | POLICY_BY_NAME.get(name, frozenset())


def status_of(diff, overload_count):
    """Summarize one name's state for the report.

    Args:
        diff: The closest overload's difference dictionary, or None when NumPy states no signature.
        overload_count: The number of C# overloads.

    Returns:
        ``identical``, or the parts that apply joined by ``; ``: ``differs``, ``null default``, ``policy only``,
        ``N overloads``.
    """
    parts = []
    if diff is not None:
        if any(diff[k] for k in HARD_KINDS):
            parts.append("differs")
        else:
            if diff["null_default"]:
                parts.append("null default")
            if diff["policy"]:
                parts.append("policy only")
    if overload_count > 1:
        parts.append(f"{overload_count} overloads")
    return "; ".join(parts) if parts else "identical"


def main():
    """Read the dump, compare every name, and write the Markdown report (and the CSV).

    Raises:
        SystemExit: Wrong number of command-line arguments.
    """
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: signature_diff.py <modules_dump.json> <out.md> [<out.csv>]")
    with open(sys.argv[1], encoding="utf-8") as fh:
        dump = json.load(fh)
    rows, summary, surface = [], [], {}
    for key in sorted(dump, key=lambda k: (k != "np", k != "ndarray", k)):
        info = dump[key]
        target = resolve_module(key)
        stats = {"module": key, "names": len(info["methods"]),
                 "overloads": sum(len(v) for v in info["methods"].values())}
        for k in ("numpy", "identical", "multi", "differs", "single_differs", "null_only", "policy_only", "cs_only",
                  "np_absent", "no_sig") + HARD_KINDS:
            stats[k] = 0
        cs_only = []
        for name, overloads in info["methods"].items():
            obj = getattr(target, name, None)
            kind, params, sig_text, is_ufunc = numpy_signature(key, name, obj)
            row = {"module": key, "name": name, "numpy_kind": kind, "numpy_signature": sig_text,
                   "overloads": len(overloads), "orpa": sum(1 for o in overloads if o["orpa"] is not None),
                   "generic": sum(1 for o in overloads if o["generic"]),
                   "cs_signatures": [cs_signature(name, o) for o in overloads], "best": "", "diff": None}
            if kind == "absent":
                cs_only.append(name)
                stats["cs_only"] += 1
                continue
            stats["numpy"] += 1
            if len(overloads) > 1:
                stats["multi"] += 1
            if params is None:
                stats["no_sig"] += 1
                row["status"] = (f"NumPy {kind}, C# method" if kind in ("property", "attribute", "module")
                                 else "NumPy states no signature")
                rows.append(row)
                continue
            policy = policy_for(name, is_ufunc)
            scored = []
            for index, ov in enumerate(overloads):
                d = compare(params, ov, policy)
                scored.append((cost(d, ov), index, d))
            scored.sort(key=lambda t: (t[0], t[1]))
            _, best_index, best = scored[0]
            split_missing(best, overloads, best_index)
            row["best"] = row["cs_signatures"][best_index]
            row["diff"] = best
            row["status"] = status_of(best, len(overloads))
            if any(best[k] for k in HARD_KINDS):
                stats["differs"] += 1
                if len(overloads) == 1:
                    stats["single_differs"] += 1
            elif best["null_default"]:
                stats["null_only"] += 1
            elif best["policy"]:
                stats["policy_only"] += 1
            if row["status"] == "identical":
                stats["identical"] += 1
            for k in HARD_KINDS:
                if best[k]:
                    stats[k] += 1
            rows.append(row)
        cs_members = set(info["methods"]) | set(info["properties"]) | set(info["fields"]) | set(info.get("types", []))
        absent = sorted(numpy_public_names(target) - cs_members)
        stats["np_absent"] = len(absent)
        surface[key] = {"cs_only": sorted(cs_only), "np_absent": absent}
        summary.append(stats)
    write_markdown(sys.argv[2], summary, rows, surface)
    if len(sys.argv) == 4:
        write_csv(sys.argv[3], rows)
    totals = {k: sum(s[k] for s in summary) for k in summary[0] if k != "module"}
    print(json.dumps(totals))


def fmt_diff(diff, kinds):
    """Render chosen kinds of a difference dictionary for a Markdown table cell.

    Args:
        diff: The difference dictionary.
        kinds: The kinds to show, in order.

    Returns:
        ``**kind**: `a`, `b`<br>...``, or an empty string.
    """
    labels = {"missing": "missing", "split": "in another overload only", "renamed": "renamed", "extra": "extra",
              "slot": "slot", "required": "C# requires",
              "default": "default", "varargs": "varargs", "null_default": "null default", "policy": "policy",
              "notes": "note", "kwargs": "via **kwargs", "optional": "C# defaults"}
    parts = [f"**{labels[k]}**: " + ", ".join(f"`{esc(x)}`" for x in diff[k]) for k in kinds if diff[k]]
    return "<br>".join(parts)


def esc(text):
    """Escape a text for a Markdown table cell.

    Args:
        text: Any text.

    Returns:
        The text with ``|`` escaped.
    """
    return text.replace("|", "\\|")


#: The summary table's columns: (statistics key, header).
SUMMARY_COLUMNS = [
    ("names", "C# names"), ("overloads", "C# overloads"), ("numpy", "with a NumPy counterpart"),
    ("identical", "identical"), ("multi", "more than one overload"), ("differs", "closest overload differs"),
    ("single_differs", "one overload, differs"), ("missing", "missing"), ("split", "in another overload only"),
    ("renamed", "renamed"), ("extra", "extra"),
    ("slot", "slot"), ("required", "C# requires"), ("default", "default"), ("varargs", "varargs"),
    ("null_only", "only null defaults"), ("policy_only", "only policy"), ("no_sig", "not compared"),
    ("cs_only", "NumSharp-only names"), ("np_absent", "NumPy names absent"),
]


def write_markdown(path, summary, rows, surface):
    """Write the report.

    Args:
        path: The Markdown file to write (UTF-8, LF line ends).
        summary: The per-module statistics.
        rows: One row per C# name with a NumPy counterpart.
        surface: Per module, the NumSharp-only names and the NumPy names NumSharp lacks.
    """
    lines = []
    w = lines.append
    w("# NumSharp `np.*` signatures against NumPy 2.4.2")
    w("")
    w("Generated by `tools/api-parameter-parity/signature_diff.py` from `dump_modules.cs`: every `[ModuleName]` module")
    w("of NumSharp.Core and the object API `np.random.Generator`. Do not edit it by hand; rerun the two tools. The")
    w("script's docstring has the full rules. In short: NumPy has ONE signature per function, so a name with several")
    w("C# overloads differs by that alone (the owner's rule of 2026-10-17: \"one whole api per function as numpy")
    w("provides, remove others\"); the CLOSEST overload, the one the fewest edits away from NumPy's signature, is what")
    w("the single surviving overload must become, and its parameters are diffed: positional-only by slot,")
    w("positional-or-keyword by name and slot, keyword-only by name. `*args` needs a `params` array; C# parameters")
    w("that NumPy takes through `**kwargs` are not differences.")
    w("")
    w("Kinds: **missing** (NumPy has it, no overload does), **in another overload only** (the closest overload lacks")
    w("it, another overload has it: merging fixes it), **renamed** (another name at that slot), **extra** (C# only),")
    w("**slot** (a positional parameter elsewhere: a positional port binds the wrong argument), **C# requires** (NumPy")
    w("defaults it: a port that omits it does not compile), **default** (two different values), **varargs** (`*args`")
    w("as fixed parameters, no `params` array; often forced, because a C# `params` array must come last and nothing")
    w("optional may follow it, so NumPy's keyword-only parameters after `*args` cannot share its signature), **null")
    w("default** (one side None/null: often C#'s spelling of \"not given\", same behavior unless a probe says")
    w("otherwise), **policy** (missing but excluded by the parity plan §3: `like`, `subok`, a ufunc's `casting` /")
    w("`order` / `signature`, the field `order` of the sorts, `einsum_call`).")
    w("")
    w("## Summary")
    w("")
    w("| module | " + " | ".join(label for _, label in SUMMARY_COLUMNS) + " |")
    w("|---" * (len(SUMMARY_COLUMNS) + 1) + "|")
    for s in summary:
        w(f"| `{s['module']}` | " + " | ".join(str(s[k]) for k, _ in SUMMARY_COLUMNS) + " |")
    w("| **total** | " + " | ".join(str(sum(s[k] for s in summary)) for k, _ in SUMMARY_COLUMNS) + " |")
    w("")
    w("*closest overload differs* counts names whose closest overload has a hard difference (missing, in another")
    w("overload only, renamed, extra, slot, C# requires, default, varargs); each kind is counted again in its own")
    w("column. *one overload, differs*:")
    w("the name already has one overload and that one differs. *only null defaults* / *only policy*: no hard")
    w("difference, only those. *not compared*: NumPy states no signature, or the name is a property there.")
    w("")
    for s in summary:
        key = s["module"]
        mod_rows = [r for r in rows if r["module"] == key]
        w(f"## `{key}`")
        w("")
        hard_rows = [r for r in mod_rows if r["diff"] and any(r["diff"][k] for k in HARD_KINDS)]
        if hard_rows:
            w(f"### Signature differences ({len(hard_rows)})")
            w("")
            w("| name | NumPy | closest C# overload | differences |")
            w("|---|---|---|---|")
            for r in hard_rows:
                more = f"<br>(+{r['overloads'] - 1} more overloads, below)" if r["overloads"] > 1 else ""
                cell = fmt_diff(r["diff"], HARD_KINDS + ("null_default", "policy", "notes"))
                w(f"| `{r['name']}` | `{esc(r['numpy_signature'])}` | `{esc(r['best'])}`{more} | {cell} |")
            w("")
        soft_rows = [r for r in mod_rows if r["diff"] and not any(r["diff"][k] for k in HARD_KINDS)
                     and r["diff"]["null_default"]]
        if soft_rows:
            w(f"### Only null defaults ({len(soft_rows)})")
            w("")
            w("| name | NumPy | closest C# overload | null defaults |")
            w("|---|---|---|---|")
            for r in soft_rows:
                w(f"| `{r['name']}` | `{esc(r['numpy_signature'])}` | `{esc(r['best'])}` | "
                  f"{fmt_diff(r['diff'], ('null_default', 'policy', 'notes'))} |")
            w("")
        policy_rows = [r for r in mod_rows if r["diff"] and not any(r["diff"][k] for k in HARD_KINDS)
                       and not r["diff"]["null_default"] and r["diff"]["policy"]]
        if policy_rows:
            groups = {}
            for r in policy_rows:
                groups.setdefault(", ".join(r["diff"]["policy"]), []).append(r["name"])
            w(f"### Only policy-excluded parameters ({len(policy_rows)})")
            w("")
            for missing, names in sorted(groups.items()):
                w(f"- lacking `{missing}`: " + ", ".join(f"`{n}`" for n in names))
            w("")
        multi_rows = [r for r in mod_rows if r["overloads"] > 1]
        if multi_rows:
            w(f"### More than one overload ({len(multi_rows)}; NumPy has one signature each)")
            w("")
            for r in multi_rows:
                w(f"- `{r['name']}`: NumPy `{r['numpy_signature']}`")
                w("")
                w("  ```csharp")
                for sig in r["cs_signatures"]:
                    w("  " + sig)
                w("  ```")
            w("")
        no_sig = [r for r in mod_rows if r["diff"] is None]
        if no_sig:
            w(f"### Not compared ({len(no_sig)})")
            w("")
            for r in no_sig:
                w(f"- `{r['name']}`: {r['status']}")
            w("")
        same = [r["name"] for r in mod_rows if r.get("status") == "identical"]
        if same:
            w(f"### Identical to NumPy ({len(same)})")
            w("")
            w(", ".join(f"`{n}`" for n in same))
            w("")
        cs_only = surface[key]["cs_only"]
        if cs_only:
            lower = [n for n in cs_only if n[:1].islower()]
            upper = [n for n in cs_only if not n[:1].islower()]
            w(f"### NumSharp-only names ({len(lower)} NumPy-style, {len(upper)} PascalCase C# members)")
            w("")
            if lower:
                w(", ".join(f"`{n}`" for n in lower))
                w("")
            if upper:
                w("PascalCase: " + ", ".join(f"`{n}`" for n in upper))
                w("")
        absent = surface[key]["np_absent"]
        if absent:
            w(f"### NumPy public names that `{key}` lacks ({len(absent)})")
            w("")
            w(", ".join(f"`{n}`" for n in absent))
            w("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def write_csv(path, rows):
    """Write one CSV row per compared name, with every difference kind and every overload.

    Args:
        path: The CSV file to write (UTF-8, LF line ends).
        rows: The rows from ``main``.
    """
    fields = ["module", "name", "numpy_kind", "status", "overloads", "orpa", "generic", "numpy_signature", "best",
              *ALL_KINDS, "all_overloads"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        # LF like the repository stores text: the csv module's default "\r\n" made git warn on every regeneration.
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            d = r["diff"] or {}
            writer.writerow({
                "module": r["module"], "name": r["name"], "numpy_kind": r["numpy_kind"], "status": r["status"],
                "overloads": r["overloads"], "orpa": r["orpa"], "generic": r["generic"],
                "numpy_signature": r["numpy_signature"], "best": r["best"],
                **{k: "; ".join(d.get(k, [])) for k in ALL_KINDS},
                "all_overloads": " || ".join(r["cs_signatures"]),
            })


if __name__ == "__main__":
    main()
