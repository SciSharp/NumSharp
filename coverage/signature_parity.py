"""Signature parity: NumPy's signature of each credited catalog row against the C# overloads of its NumSharp member.

Why this module exists
----------------------
Availability is a NAME claim. ``generate_coverage.direct_target`` credits a NumPy API when the annotated NumSharp
module has a member of the same name, a reviewed alias (overrides.json) credits it through another name, and nothing
after that looked at a parameter: the inventory tool carried each overload only as display text
(``numsharp_signatures``), which the artifact printed beside NumPy's signature without ever comparing the two. So a
row read "available" while a NumPy call ported word for word did not compile or bound the wrong argument:
``np.sum`` had no ``out=`` / ``initial=`` / ``where=`` and seven overloads, ``np.transpose`` named its parameter
``premute``, ``np.can_cast`` took ``NPTypeCode``, ``numpy.acos`` was credited to ``np.arccos`` although
``np.acos(x)`` does not compile. A standalone comparison (``tools/api-parameter-parity/signature_diff.py``, the
owner's request of 2026-10-17) found 182 rows missing a parameter, 82 with a parameter NumPy lacks, 310 with several
overloads and 7 taking a dtype as ``NPTypeCode`` / ``Type``, every one of them "available / exact" in the artifact.
Schema 7 of ``NumSharp.Tools.ApiInventory`` carries every overload's parameters as data, and this module turns them
into a per-row verdict, so the artifact reports what it used to hide.

That comparison still missed what ``docs/plans/API_DIFF_GAP.md`` (the gap map of ``tools/api-parameter-parity``)
finds: NumPy parameters a runtime signature hides behind ``(*args, **kwargs)`` (read from NumPy's stubs), the
constructors, extension methods and typed ``NDArray<T>`` spellings, ``numpy.char.chararray``, and the compile
probe's B / L / Q marks. Generator 1.16.0 therefore joins the gap map itself onto the rows (``attach`` with a
``signature_gap.GapMap``): one engine decides the marks, the inventory reports them per API, and this module's own
comparison is kept for the credited rows the gap map does not reach.

The record (``row["signature"]``)
---------------------------------
Every credited NumPy row that names a call (a function, ufunc or method, or a class credited to a factory method)
gets one, except the Python protocols, whose C# spelling is an operator, indexer or cast rather than a call; with
the gap map, a protocol method NumSharp declares under its own name (``NDArray.__getitem__``) and a class with a C#
constructor get one too, and so does a NumSharp-only member the gap map marks (O / T):

* ``status``: ``identical`` (no mark below and no closest-overload difference), ``differs``, or ``not-compared``
  with a ``reason`` (NumPy states no signature, the credit is a property or an instance member of another surface;
  a credit the inventory index cannot resolve is not a verdict but a join failure, ``WIRING_REASONS``, and the
  generator refuses to publish it);
* ``spelling``: ``exact``, or ``alias`` when the row is credited through another name (mark **N**: NumPy's
  spelling does not compile);
* ``overloads``: the number of C# overloads (mark **O** above 1: NumPy has one signature per function, the owner's
  rule of 2026-10-17 "one whole api per function as numpy provides, remove others");
* ``missing``: NumPy parameters NO overload offers (mark **M**; a rename counts, ``"axes (C#: premute)"``, because
  NumPy's keyword does not compile); ``excluded``: missing ones the parity plan excludes (docs/plans/
  api-parameter-parity.md §3: ``like``, ``subok``, a ufunc's ``casting`` / ``order`` / ``signature``, the field
  ``order`` of the sorts, ``einsum_path``'s ``einsum_call``), reported but not counted against the row;
* ``foreign``: C# parameter names NumPy does not have, in any overload (mark **X**);
* ``typecode``: parameters that take a dtype as ``NPTypeCode`` or ``Type`` instead of ``DType`` (mark **T**);
* ``order`` / ``required`` / ``default`` / ``varargs`` / ``null_default``: the differences of the CLOSEST overload
  (the one the fewest weighted edits from NumPy's signature, what the one surviving overload must become): a
  positional parameter at another slot, a parameter NumPy defaults that C# requires, two different default values,
  ``*args`` spelled without a ``params`` array, and (informational) a null default where NumPy has a value;
* ``closest`` and ``numpy``: the closest overload's text and the NumPy signature it was compared with;
* ``source``: ``gap-map`` when the record comes from the API diff gap map (generator 1.16.0: every C# spelling
  ``tools/api-parameter-parity`` compares, joined by ``attach``; see ``signature_gap``), ``inventory`` when it is
  this module's own reflection comparison (a credited row the gap map does not reach: never compile-probed);
* ``marks``: the gap map's symbol per mark (``M`` ● / ◦, ``X`` ●, ``O`` the overload count, ``T`` ● and / or ◇,
  ``B`` / ``Q`` ● / ◐, ``L`` ● / ◐ / ◦, ``—`` not probed), the strongest over the spellings;
* ``generic``: the generic overloads whose type argument picks the dtype (mark T ◇, ``<T>(int[] shape)``);
* ``binding`` / ``null`` / ``sequence``: the compile probe's findings (marks B, L, Q) for the primary spelling, each
  the call and the compiler's reason;
* ``spellings``: one entry per C# spelling of the API (``np.ndarray(...)``, ``new NDArray(...)``,
  ``new NDArray<T>(...)``; ``a.reshape`` and ``NDArray<T>.reshape``), each with its own marks and details.

Matching rules (§1 of the plan): a positional-only NumPy parameter is matched by SLOT (Python cannot pass it by
name), a positional-or-keyword one by NAME and must sit at NumPy's slot, a keyword-only one by name anywhere;
``*args`` is matched by a ``params`` array or by the C# parameters at its slot; C# parameters NumPy takes through
``**kwargs`` are not foreign. NumPy's signature comes from ``inspect.signature``, except the documented first
docstring line for ndarray methods (inspect folds their keywords into ``**kwargs``), the wrapped callable for
numpy.ma's ufunc wrappers, and ndarray's documented method for the numpy.ma functions and MaskedArray methods that
forward ``(*args, **params)`` to it.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from object_surfaces import resolve_api_id
from signature_gap import (CONSTRUCTOR, TYPED_NDARRAY_GROUP, GapMap, catalog_id, csharp_spelling, is_module_group,
                           split_findings, split_list)

P = inspect.Parameter

#: Marker for "the parameter has no default": a NumPy parameter every call must pass.
REQUIRED = object()

#: Excluded everywhere (plan §3): NumPy's __array_function__ protocol and ndarray subclasses; NumSharp has neither.
POLICY_ALL = frozenset({"like", "subok"})

#: Ufunc keywords NumSharp's ufunc surface does not model (plan §3).
POLICY_UFUNC = frozenset({"casting", "order", "signature"})

#: Per-name exclusions (plan §3): order= is the field order of a structured dtype (NumSharp has none); einsum_call=
#: is NumPy-internal.
POLICY_BY_NAME = {
    "sort": frozenset({"order"}), "argsort": frozenset({"order"}), "partition": frozenset({"order"}),
    "argpartition": frozenset({"order"}), "einsum_path": frozenset({"einsum_call"}),
}

#: numpy.ma classes whose instances forward (*args, **kwargs) to the callable in their ``f`` attribute, with the
#: number of their own leading parameters (``a``, or ``a, b``).
MA_FORWARDERS = {"_MaskedUnaryOperation": 1, "_MaskedBinaryOperation": 2, "_DomainedBinaryOperation": 2}

#: Parameter shapes of the numpy.ma forwarders that reach ndarray's method of the same name: numpy.ma's
#: ``_frommethod`` functions ``(a, *args, **params)`` and MaskedArray's ``_arraymethod`` methods ``(*args, **params)``
#: (after ``self``).
_FORWARD_SHAPES = (
    (("a", P.POSITIONAL_OR_KEYWORD), ("args", P.VAR_POSITIONAL), ("params", P.VAR_KEYWORD)),
    (("args", P.VAR_POSITIONAL), ("params", P.VAR_KEYWORD)),
)

#: C# parameter types that select a dtype the pre-DType way (FriendlyType spelling of the inventory tool): since the
#: dtype system's Stage B every dtype parameter is meant to be ONE ``DType`` (an NPTypeCode, a Type or a dtype string
#: converts to it implicitly), so a parameter still typed NPTypeCode / Type is a leftover.
TYPECODE_TYPE = re.compile(r"^(NumSharp\.NPTypeCode|System\.Type)\??(\[\])*$")

#: The closest-overload difference kinds that make a row ``differs`` (beside the marks M, X, O, T, N).
CLOSEST_HARD_KINDS = ("order", "required", "default", "varargs")

#: Edits each closest-overload difference costs when the closest overload is chosen, weighted by the work: a missing
#: parameter needs an implementation (2), a rename / move / removal / new default value is one edit (1), making a
#: required parameter optional is half of one (0.5), a null default a quarter. So among one name's overloads the
#: fullest wins: ``mean(a)`` lacks five of NumPy's parameters (10) where ``mean(a, int? axis, ...)`` lacks two,
#: misplaces one and requires two (6.0).
COST = {"missing": 2.0, "renamed": 1.0, "extra": 1.0, "order": 1.0, "required": 0.5, "default": 1.0,
        "varargs": 1.0, "null_default": 0.25, "excluded": 0.0, "optional": 0.0, "kwargs": 0.0}


class Soft:
    """A NumPy default C# cannot write as a constant (a class such as ``dtype=float``, a function, an object).

    C# spells such a default as null, so null against a Soft value is not a difference; any other C# default is.
    """

    def __init__(self, text: str) -> None:
        """Keep NumPy's text of the default.

        Args:
            text: The default as NumPy shows it (its repr).
        """
        self.text = text


class Param:
    """One parameter of NumPy's signature.

    Attributes:
        name: The parameter's name.
        kind: An ``inspect.Parameter`` kind.
        default: ``REQUIRED``, or the default as text: its repr, ``<no value>`` for ``np._NoValue`` (NumPy does not
            pass the argument on, so any C# default matches it) or ``<optional>`` for a docstring's bracketed
            parameter without a stated default.
    """

    def __init__(self, name: str, kind: Any, default: Any) -> None:
        """Store the three fields.

        Args:
            name: The parameter's name.
            kind: The ``inspect.Parameter`` kind.
            default: ``REQUIRED`` or the default's text.
        """
        self.name, self.kind, self.default = name, kind, default

    def text(self) -> str:
        """Render the parameter as a Python signature does.

        Returns:
            ``*name``, ``**name``, ``name=default`` or ``name``.
        """
        prefix = "*" if self.kind == P.VAR_POSITIONAL else "**" if self.kind == P.VAR_KEYWORD else ""
        return prefix + self.name + ("" if self.default is REQUIRED else "=" + self.default)


@dataclass(frozen=True)
class NumPySignature:
    """NumPy's parameters of one row, ready for comparison.

    Attributes:
        params: The parameters in signature order (``self`` dropped).
        is_ufunc: Whether the plan's ufunc exclusions apply (a ufunc, or a numpy.ma wrapper of one).
        text: The signature as shown in the record (``a.sum(axis=None, ...)`` for an ndarray method).
    """

    params: tuple[Param, ...]
    is_ufunc: bool
    text: str


def default_text(np: Any, value: Any) -> Any:
    """Convert an ``inspect`` default to the text a ``Param`` keeps.

    Args:
        np: The imported numpy module (its ``_NoValue`` sentinel is recognised).
        value: ``inspect.Parameter.default``.

    Returns:
        ``REQUIRED`` for no default, ``<no value>`` for ``np._NoValue``, else the default's repr.
    """
    if value is P.empty:
        return REQUIRED
    if value is getattr(np, "_NoValue", object()):
        return "<no value>"
    return repr(value)


def from_inspect(np: Any, sig: inspect.Signature) -> list[Param]:
    """Convert an ``inspect.Signature`` to Params, dropping a bound or unbound ``self``.

    Args:
        np: The imported numpy module.
        sig: The signature.

    Returns:
        The Params in signature order.
    """
    return [Param(p.name, p.kind, default_text(np, p.default)) for p in sig.parameters.values() if p.name != "self"]


def split_top(text: str) -> list[str]:
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


def from_doc(doc: str | None, name: str) -> list[Param] | None:
    """Parse the signature line of a NumPy docstring, such as ``a.min(axis=None, out=None, *, keepdims=<no value>)``
    or ``a.view([dtype][, type])``, into Params.

    A bracketed parameter is optional without a stated default (``<optional>``). A ``/`` makes the parameters before
    it positional-only, a ``*`` or ``*args`` makes the ones after it keyword-only. ndarray methods need this:
    ``inspect`` reports their keywords as ``**kwargs``, and their C forwarding is looser than their documentation
    (``a.sum(0, None, None, True)`` takes keepdims positionally while the docs make it keyword-only), so the
    documented form is the contract a port follows.

    Args:
        doc: The docstring (may be empty or None).
        name: The function's name, located in the docstring as ``name(`` after a start, a space or a dot.

    Returns:
        The Params, or None when the docstring has no such line.
    """
    if not doc:
        return None
    match = re.search(r"(?:^|\s|\.)" + re.escape(name) + r"\(", doc)
    if not match:
        return None
    start, depth, i = match.end(), 1, match.end()
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
    params: list[Param] = []
    keyword_only = False
    for token, optional in tokens:
        if token == "/":
            # Everything before "/" is positional-only.
            for param in params:
                if param.kind == P.POSITIONAL_OR_KEYWORD:
                    param.kind = P.POSITIONAL_ONLY
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


def format_params(params: Sequence[Param]) -> str:
    """Render Params as a Python signature's parameter list, with the ``/`` and ``*`` markers.

    Args:
        params: The Params in order.

    Returns:
        The text between the parentheses.
    """
    shown = []
    slash_pending = any(p.kind == P.POSITIONAL_ONLY for p in params)
    star_done = False
    for param in params:
        if slash_pending and param.kind != P.POSITIONAL_ONLY:
            shown.append("/")
            slash_pending = False
        if param.kind == P.VAR_POSITIONAL:
            star_done = True
        if param.kind == P.KEYWORD_ONLY and not star_done:
            shown.append("*")
            star_done = True
        shown.append(param.text())
    if slash_pending:
        shown.append("/")
    return ", ".join(shown)


def ma_forwarded(np: Any, obj: Any) -> tuple[list[Param], Any] | None:
    """The signature a numpy.ma ufunc wrapper accepts: its own ``a`` (``a, b``), then the wrapped callable's
    parameters after its first one (two), which ``__call__(self, a, *args, **kwargs)`` passes on.

    Args:
        np: The imported numpy module.
        obj: Any NumPy object.

    Returns:
        ``(params, wrapped)``, or None when ``obj`` is not such a wrapper or its callable has no signature.
    """
    count = MA_FORWARDERS.get(type(obj).__name__)
    wrapped = getattr(obj, "f", None)
    if count is None or not callable(wrapped):
        return None
    try:
        inner = from_inspect(np, inspect.signature(wrapped))
    except (TypeError, ValueError):
        return None
    head = [Param("a", P.POSITIONAL_OR_KEYWORD, REQUIRED)]
    if count == 2:
        head.append(Param("b", P.POSITIONAL_OR_KEYWORD, REQUIRED))
    return head + inner[count:], wrapped


def numpy_parameters(np: Any, api_id: str) -> NumPySignature | None:
    """Find NumPy's signature of one catalog row.

    Args:
        np: The imported numpy module.
        api_id: The row id (``numpy.sum``, ``numpy.ndarray.sum``, ``numpy.ma.add``, ``numpy.random.Generator.normal``).

    Returns:
        The signature, or None when the id does not resolve or NumPy states no signature for the object.
    """
    resolved, obj = resolve_api_id(np, api_id)
    if not resolved or obj is None:
        return None
    short = api_id.rsplit(".", 1)[-1]
    if api_id.startswith("numpy.ndarray.") and callable(obj):
        documented = from_doc(getattr(obj, "__doc__", "") or "", short)
        if documented is not None:
            return NumPySignature(tuple(documented), False, f"a.{short}({format_params(documented)})")
    forwarded = ma_forwarded(np, obj)
    if forwarded is not None:
        params, wrapped = forwarded
        return NumPySignature(tuple(params), isinstance(wrapped, np.ufunc), f"{short}({format_params(params)})")
    try:
        params = from_inspect(np, inspect.signature(obj))
    except (TypeError, ValueError):
        params = from_doc(getattr(obj, "__doc__", "") or "", short)
    if params is None:
        return None
    if tuple((p.name, p.kind) for p in params) in _FORWARD_SHAPES and hasattr(np.ndarray, short):
        # numpy.ma.copy / diagonal / repeat / swapaxes and MaskedArray's forwarding methods reach ndarray's method:
        # its documented parameters are the real ones. A positional-only method parameter (swapaxes' axis1, axis2)
        # can only arrive through *args, which also makes a leading ``a`` positional in effect.
        method = from_doc(getattr(getattr(np.ndarray, short), "__doc__", "") or "", short)
        if method is not None:
            if params[0].name == "a":
                head = Param("a", P.POSITIONAL_ONLY if any(p.kind == P.POSITIONAL_ONLY for p in method)
                             else P.POSITIONAL_OR_KEYWORD, REQUIRED)
                method = [head] + method
            return NumPySignature(tuple(method), False, f"{short}({format_params(method)})")
    return NumPySignature(tuple(params), isinstance(obj, np.ufunc), f"{short}({format_params(params)})")


def literal(text: str) -> Any:
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


def cs_default_value(parameter: Mapping[str, Any]) -> Any:
    """Turn a C# parameter's default (inventory schema 7) into a comparable Python value.

    Args:
        parameter: One ``parameters`` entry of an overload.

    Returns:
        None for null / ``default``, a bool, a float, a str (a C# string or char), ``("missing", text)`` for an
        ``[Optional]`` parameter without a constant, or ``("enum", text)`` / ``("other", text)``.
    """
    kind, text = parameter.get("defaultKind"), parameter.get("default")
    if kind == "missing":
        return ("missing", text)
    if kind == "null" or text is None:
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


def same_value(npv: Any, cs_value: Any) -> bool:
    """Decide whether a NumPy default and a C# default are the same value.

    Args:
        npv: From ``literal``.
        cs_value: From ``cs_default_value``.

    Returns:
        True when they agree: equal bools (a bool never equals a number), equal numbers (NaN equals NaN), equal
        strings, or a C# enum whose member name is NumPy's string ignoring case and underscores.
    """
    if isinstance(npv, bool) or isinstance(cs_value, bool):
        return type(npv) is type(cs_value) and npv == cs_value
    if isinstance(npv, float) and isinstance(cs_value, float):
        return npv == cs_value or (math.isnan(npv) and math.isnan(cs_value))
    if isinstance(npv, str) and isinstance(cs_value, str):
        return npv == cs_value
    if isinstance(npv, str) and isinstance(cs_value, tuple) and cs_value[0] == "enum":
        member = cs_value[1].rsplit(".", 1)[-1]
        return member.replace("_", "").lower() == npv.replace("_", "").lower()
    return False


def compare_default(np_param: Param, cs_param: Mapping[str, Any]) -> tuple[str | None, str]:
    """Classify the difference between two defaults of one parameter.

    Args:
        np_param: NumPy's Param (it has a default).
        cs_param: The C# parameter (it has a default).

    Returns:
        ``(kind, text)``: kind is None (no difference: also for an ``[Optional]`` C# parameter without a constant,
        whose omitted argument the member reads as NumPy's default), ``default`` (two values differ) or
        ``null_default`` (one side is None / null, the other a value); text shows both defaults.
    """
    text = np_param.default
    if text in ("<no value>", "<optional>"):
        return None, ""
    if cs_param.get("defaultKind") == "missing":
        # An [Optional] parameter without a constant (C# cannot write NumPy's [] or 0 as the default of an object
        # parameter): an omitted argument arrives as Type.Missing, which the member replaces with NumPy's default.
        return None, ""
    npv, cs_value = literal(text), cs_default_value(cs_param)
    shown = f"{np_param.name}: {text} vs {cs_param.get('default')}"
    if np_param.name == "where" and npv is True and cs_value is None:
        return None, ""  # where=True is "no mask", which NumSharp spells null.
    if isinstance(npv, Soft):
        return (None, "") if cs_value is None else ("default", shown)
    if npv is None and cs_value is None:
        return None, ""
    if npv is None or cs_value is None:
        return "null_default", shown
    if isinstance(npv, (tuple, list)):
        return "default", shown
    return (None, "") if same_value(npv, cs_value) else ("default", shown)


def _check_requiredness(np_param: Param, cs_param: Mapping[str, Any], diff: dict[str, list[str]]) -> None:
    """Record whether one matched parameter is required on one side only, or defaults differently.

    Args:
        np_param: NumPy's Param.
        cs_param: The matched C# parameter.
        diff: The difference dictionary to extend (``required``, ``optional``, ``default``, ``null_default``).
    """
    np_required = np_param.default is REQUIRED
    cs_required = not cs_param.get("optional")
    if np_required and not cs_required:
        diff["optional"].append(np_param.name)
    elif not np_required and cs_required:
        diff["required"].append(np_param.name)
    elif not np_required:
        kind, text = compare_default(np_param, cs_param)
        if kind:
            diff[kind].append(text)


def compare_overload(np_params: Sequence[Param], overload: Mapping[str, Any], policy: frozenset[str]) -> dict[str, list[str]]:
    """Compare ONE C# overload with NumPy's parameters (the closest-overload view).

    Args:
        np_params: NumPy's Params.
        overload: One ``overloads`` entry of the inventory (schema 7).
        policy: The parameter names this function excludes by policy.

    Returns:
        A dictionary from each kind (``missing``, ``renamed``, ``extra``, ``order``, ``required``, ``default``,
        ``varargs``, ``null_default``, ``excluded``, ``optional``, ``kwargs``) to its findings.
    """
    cs = list(overload.get("parameters") or [])
    names = [c["name"] for c in cs]
    np_names = {p.name for p in np_params if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)}
    used: set[int] = set()
    diff: dict[str, list[str]] = {k: [] for k in COST}
    positional = [p for p in np_params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    unmatched_positional: dict[int, Param] = {}
    for slot, param in enumerate(positional):
        if param.kind == P.POSITIONAL_ONLY:
            # The name cannot be passed in Python, so any C# parameter at the slot serves, unless it carries the
            # name of another NumPy parameter (then it is that parameter, misplaced).
            fits = (slot < len(cs) and slot not in used and not cs[slot].get("isParams")
                    and (cs[slot]["name"] == param.name or cs[slot]["name"] not in np_names))
            index = slot if fits else None
        else:
            index = names.index(param.name) if param.name in names else None
            if index is not None and index != slot:
                diff["order"].append(f"{param.name}: NumPy #{slot}, C# #{index}")
        if index is None:
            unmatched_positional[slot] = param
            continue
        used.add(index)
        _check_requiredness(param, cs[index], diff)
    unmatched_keyword: list[Param] = []
    for param in np_params:
        if param.kind != P.KEYWORD_ONLY:
            continue
        index = names.index(param.name) if param.name in names else None
        if index is None or index in used:
            unmatched_keyword.append(param)
            continue
        used.add(index)
        _check_requiredness(param, cs[index], diff)
    var_positional = next((p for p in np_params if p.kind == P.VAR_POSITIONAL), None)
    var_keyword = next((p for p in np_params if p.kind == P.VAR_KEYWORD), None)
    if var_positional is not None:
        # *args takes the run of unmatched C# parameters that starts at its slot; the run ends at a C# parameter
        # that carries the name of another NumPy parameter, **kwargs's own name included (savez's
        # `IDictionary<string, NDArray> kwds` spells **kwds, not part of *args).
        stop = np_names | ({var_keyword.name} if var_keyword is not None else set())
        absorbed = []
        i = len(positional)
        while i < len(cs) and i not in used and cs[i]["name"] not in stop:
            absorbed.append(i)
            if cs[i].get("isParams"):
                break
            i += 1
        used.update(absorbed)
        forwarder = [p.kind for p in np_params] == [P.VAR_POSITIONAL, P.VAR_KEYWORD]
        if not absorbed:
            diff["missing"].append("*" + var_positional.name)
        elif not any(cs[i].get("isParams") for i in absorbed) and not forwarder:
            spelled = ", ".join(f"{cs[i]['type']} {cs[i]['name']}" for i in absorbed)
            diff["varargs"].append(f"*{var_positional.name} as {spelled} (no params array)")
    if var_keyword is not None:
        rest = [i for i in range(len(cs)) if i not in used]
        diff["kwargs"].extend(cs[i]["name"] for i in rest)
        used.update(rest)
    extra_at = {i: c["name"] for i, c in enumerate(cs) if i not in used}
    for slot, param in unmatched_positional.items():
        if param.name in policy:
            diff["excluded"].append(param.name)
        elif slot in extra_at:
            diff["renamed"].append(f"{param.name} -> {extra_at.pop(slot)}")
        else:
            diff["missing"].append(param.name)
    for param in unmatched_keyword:
        (diff["excluded"] if param.name in policy else diff["missing"]).append(param.name)
    diff["extra"].extend(extra_at[i] for i in sorted(extra_at))
    return diff


def _cost(diff: Mapping[str, list[str]], overload: Mapping[str, Any]) -> float:
    """The weighted number of edits that would make an overload NumPy's signature.

    Args:
        diff: The overload's difference dictionary (``compare_overload``).
        overload: The overload (a generic, obsolete or hidden overload costs a little more, so a plain one wins a
            tie).

    Returns:
        The weighted edit count.
    """
    total = sum(COST[kind] * len(values) for kind, values in diff.items())
    if overload.get("genericParameters"):
        total += 0.5
    if overload.get("obsolete"):
        total += 0.5
    if overload.get("hidden"):
        total += 0.25
    return total


def closest_overload(np_params: Sequence[Param], overloads: Sequence[Mapping[str, Any]],
                     policy: frozenset[str]) -> tuple[int, dict[str, list[str]]]:
    """Pick the overload the fewest weighted edits from NumPy's signature: what the single surviving overload must
    become under the owner's one-signature rule.

    Args:
        np_params: NumPy's Params.
        overloads: The credited member's overloads (at least one).
        policy: The excluded parameter names.

    Returns:
        ``(index, diff)``: the closest overload's index (the earliest on a tie) and its difference dictionary.
    """
    scored = []
    for index, overload in enumerate(overloads):
        diff = compare_overload(np_params, overload, policy)
        scored.append((_cost(diff, overload), index, diff))
    scored.sort(key=lambda item: (item[0], item[1]))
    _, index, diff = scored[0]
    return index, diff


def union_marks(np_params: Sequence[Param], overloads: Sequence[Mapping[str, Any]],
                policy: frozenset[str]) -> dict[str, list[str]]:
    """The union view over ALL overloads: which NumPy parameters no overload offers, and which C# parameter names
    NumPy does not have.

    A NumPy parameter is missing only when no overload has it; a C# parameter is foreign when any overload has it,
    except at a positional-only slot (Python never sees the name), as a spelling of ``*args``, or as a keyword NumPy
    takes through ``**kwargs``. A NumPy parameter whose slot holds a foreign C# name in some overload is reported
    with that name (a rename: NumPy's keyword does not compile, and the C# name is foreign).

    Args:
        np_params: NumPy's Params.
        overloads: The credited member's overloads.
        policy: The excluded parameter names (reported apart).

    Returns:
        ``missing`` (with ``"axes (C#: premute)"`` for a rename), ``excluded`` and ``foreign`` (sorted).
    """
    # Only the named parameters match by name: a C# `params object[] args` IS NumPy's `*args` whatever its name.
    np_names = {p.name for p in np_params if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)}
    positional = [p for p in np_params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    var_positional = next((p for p in np_params if p.kind == P.VAR_POSITIONAL), None)
    var_keyword = next((p for p in np_params if p.kind == P.VAR_KEYWORD), None)
    offered: set[str] = set()
    renamed: dict[str, set[str]] = {}
    foreign: set[str] = set()
    for overload in overloads:
        cs = list(overload.get("parameters") or [])
        names = [c["name"] for c in cs]
        for slot, param in enumerate(positional):
            at_slot = cs[slot] if slot < len(cs) and not cs[slot].get("isParams") else None
            if param.kind == P.POSITIONAL_ONLY:
                if at_slot is not None and (at_slot["name"] == param.name or at_slot["name"] not in np_names):
                    offered.add(param.name)
            elif param.name in names:
                offered.add(param.name)
            elif at_slot is not None and at_slot["name"] not in np_names:
                renamed.setdefault(param.name, set()).add(at_slot["name"])
        for param in np_params:
            if param.kind == P.KEYWORD_ONLY and param.name in names:
                offered.add(param.name)
        for i, c in enumerate(cs):
            if c["name"] in np_names:
                continue
            if var_keyword is not None and c["name"] == var_keyword.name:
                continue  # C# spells NumPy's **kwds as one parameter of that name (savez's dictionary).
            if i < len(positional) and positional[i].kind == P.POSITIONAL_ONLY:
                continue  # A positional-only slot: Python never sees the name.
            if var_positional is not None and (c.get("isParams") or i >= len(positional)):
                offered.add("*" + var_positional.name)  # C# spells what NumPy's *args carries.
                continue
            if var_keyword is not None:
                continue  # A keyword NumPy accepts through **kwargs.
            foreign.add(c["name"])
    missing, excluded = [], []
    for param in np_params:
        if param.kind == P.VAR_KEYWORD:
            continue  # C# has no **kwargs; what it would carry is matched by name above.
        key = "*" + param.name if param.kind == P.VAR_POSITIONAL else param.name
        if key in offered:
            continue
        if param.name in policy:
            excluded.append(param.name)
        elif param.name in renamed:
            missing.append(f"{param.name} (C#: {', '.join(sorted(renamed[param.name]))})")
        else:
            missing.append(key)
    return {"missing": missing, "excluded": excluded, "foreign": sorted(foreign)}


def typecode_parameters(overloads: Iterable[Mapping[str, Any]]) -> list[str]:
    """The parameters of a member's overloads that take a dtype as NPTypeCode or Type.

    Args:
        overloads: The member's overloads.

    Returns:
        The distinct ``"Type name"`` texts, in declaration order.
    """
    found: list[str] = []
    for overload in overloads:
        for parameter in overload.get("parameters") or []:
            text = f"{parameter['type']} {parameter['name']}"
            if TYPECODE_TYPE.match(parameter["type"]) and text not in found:
                found.append(text)
    return found


def policy_for(name: str, is_ufunc: bool) -> frozenset[str]:
    """The parameter names one function excludes by policy (plan §3).

    Args:
        name: The function's own name (the last segment of its id).
        is_ufunc: Whether NumPy's object is a ufunc (or a numpy.ma wrapper of one).

    Returns:
        The excluded names.
    """
    return POLICY_ALL | (POLICY_UFUNC if is_ufunc else frozenset()) | POLICY_BY_NAME.get(name, frozenset())


def member_index(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index every reflected method by the target id the catalog rows use (``NumSharp.np.sum``,
    ``NumSharp.np.linalg.norm``, ``NumSharp.Generator.normal``): the type's CLR name with ``+`` read as ``.``, then the
    member name.

    Args:
        inventory: The reflected NumSharp inventory (schema 7).

    Returns:
        Target id -> member (``kind``, ``static``, ``overloads``, ...). A module facade's entry wins over the same
        type's object-inventory entry: the facade is reflected DeclaredOnly, exactly what its rows credit.
    """
    index: dict[str, dict[str, Any]] = {}
    for collection in ("objectTypes", "modules"):
        for type_data in (inventory.get(collection) or {}).values():
            prefix = type_data["type"].replace("+", ".")
            for kind in ("methods", "properties", "fields"):
                for member in type_data.get(kind) or []:
                    index[f"{prefix}.{member['name']}"] = member
    return index


def _not_compared(spelling: str, reason: str, numpy_text: str = "") -> dict[str, Any]:
    """A record for a credited row whose parameters cannot be compared.

    Args:
        spelling: ``exact`` or ``alias``.
        reason: Why the row was not compared.
        numpy_text: NumPy's signature text, when known.

    Returns:
        The record.
    """
    record = {"status": "not-compared", "reason": reason, "spelling": spelling}
    if numpy_text:
        record["numpy"] = numpy_text
    return record


def assess(np: Any, row: Mapping[str, Any], member: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Compute one row's signature-parity record.

    Args:
        np: The imported numpy module.
        row: A catalog row (its ``id``, ``kind``, ``surface``, ``availability``, ``numsharp_target``).
        member: The credited NumSharp member (``member_index``), or None when the inventory has no such member.

    Returns:
        The record (see the module docstring), or None when the row takes no part: not NumPy's, not credited, a
        Python protocol, or not a call (a property, constant, module, or a class credited to a type).
    """
    if row.get("origin") != "numpy" or not row.get("numsharp_target") or row.get("python_protocol"):
        return None
    kind = row.get("kind")
    member_kind = (member or {}).get("kind")
    if kind not in ("function", "ufunc", "callable", "method") and not (kind == "class" and member_kind == "method"):
        return None
    spelling = "alias" if row.get("availability") == "alias" else "exact"
    if member is None:
        return _not_compared(spelling, "the credited member is not in the inventory")
    if member_kind != "method":
        return _not_compared(spelling, f"credited to a {member_kind or 'non-method member'}")
    overloads = member.get("overloads") or []
    if not overloads:
        return _not_compared(spelling, "the inventory carries no overload data for the credited member")
    signature = numpy_parameters(np, row["id"])
    if signature is None:
        return _not_compared(spelling, "NumPy states no signature")
    # The array is the receiver on one side and the first argument on the other when a module function is credited
    # to an NDArray instance method (numpy.astype -> NDArray.astype) or an ndarray method to a module function (the
    # generator's auto_alternative, numpy.ndarray.X -> np.X): the parameter lists cannot be aligned, so they are not
    # compared. (np.random's functions are instance methods too, of the NumPyRandom facade object; their receiver
    # is the module, not an argument, so they align and are compared.)
    target = row["numsharp_target"]
    ndarray_row = row["id"].startswith("numpy.ndarray.")
    ndarray_target = target.startswith("NumSharp.NDArray.")
    if ndarray_row != ndarray_target and not row.get("object_surface"):
        return _not_compared(spelling, f"credited through {target}, whose receiver is an argument of the NumPy call "
                             "or the other way round", signature.text)
    short = row["id"].rsplit(".", 1)[-1]
    policy = policy_for(short, signature.is_ufunc)
    params = list(signature.params)
    marks = union_marks(params, overloads, policy)
    index, diff = closest_overload(params, overloads, policy)
    record: dict[str, Any] = {"status": "identical", "spelling": spelling, "overloads": len(overloads)}
    for key in ("missing", "excluded", "foreign"):
        if marks[key]:
            record[key] = marks[key]
    typecode = typecode_parameters(overloads)
    if typecode:
        record["typecode"] = typecode
    for key in CLOSEST_HARD_KINDS + ("null_default",):
        if diff[key]:
            record[key] = diff[key]
    hard = (spelling == "alias" or len(overloads) > 1 or marks["missing"] or marks["foreign"] or typecode
            or any(diff[key] for key in CLOSEST_HARD_KINDS))
    record["status"] = "differs" if hard else "identical"
    record["closest"] = overloads[index].get("signature", "")
    record["numpy"] = signature.text
    return record


#: Not-compared reasons that mean the JOIN failed rather than the row: the credited target is not where
#: ``member_index`` looks, or the inventory lost its overload data. ``attach`` reports them so the generator can
#: refuse to publish (a wiring slip would otherwise turn hundreds of rows "not compared" and shrink every count
#: silently, the very failure this module exists to end).
WIRING_REASONS = ("the credited member is not in the inventory",
                  "the inventory carries no overload data for the credited member")

#: Gap-map rows with a NumPy counterpart that join no catalog row, each with the reviewed reason. Any OTHER unjoined
#: row fails the generator (the inventory would silently lose a finding of ``API_DIFF_GAP.md``), and an entry that
#: stops occurring fails it too (stale), so the list cannot hide a new gap.
GAP_JOIN_EXEMPT = {
    ("np.random.MT19937", "_legacy_seeding"): "NumPy's name is private (a leading underscore): the catalog lists "
                                              "public NumPy APIs only, while NumSharp exposes the member publicly",
}

#: The gap map's mark symbols, strongest first. ``●`` a difference at both C# versions (or the plain mark), ``◐`` only
#: at C# 12 (an overload set that relies on [OverloadResolutionPriority], which a net8.0 consumer's C# 12 ignores),
#: ``◦`` the weak form (M: only plan-excluded parameters missing; L: only ``Shape`` refuses ``null`` and ``default``
#: binds), ``◇`` (T) a generic overload whose type argument picks the dtype.
_SYMBOL_RANK = {"●": 3, "◐": 2, "◦": 1}

#: Marks the gap map's compile probe adds, with the symbols that count as a difference. ``—`` (not probed) and ``?``
#: (the probe could not write a call) are not differences.
PROBE_MARKS = {"B": ("●", "◐"), "L": ("●", "◐", "◦"), "Q": ("●", "◐")}

#: The mark letters in display order: the reflection marks, the probe marks, then N (credited through another name).
MARK_ORDER = "MXOTBLQN"


def _strongest(symbols: Iterable[str]) -> str:
    """The strongest of several mark symbols.

    Args:
        symbols: Symbols from the gap map (``●``, ``◐``, ``◦``, ``""``, ``—``, ``?``).

    Returns:
        The highest-ranked of ``●`` / ``◐`` / ``◦``; else ``—`` when every spelling was unprobed, else ``""``.
    """
    symbols = list(symbols)
    ranked = [s for s in symbols if s in _SYMBOL_RANK]
    if ranked:
        return max(ranked, key=_SYMBOL_RANK.__getitem__)
    return "—" if symbols and all(s in ("—", "?") for s in symbols) else ""


def spelling_from_gap(row: Mapping[str, str], group_types: Mapping[str, str], via: str | None = None) -> dict[str, Any]:
    """One C# spelling of a NumPy API, from one gap-map row.

    Args:
        row: A row of the gap map's CSV.
        group_types: ``GapMap.group_types`` (for the C# spelling's receiver).
        via: The catalog id whose gap row this spelling was adopted from (same C# member, same NumPy object), or None.

    Returns:
        ``csharp`` (``np.sum``, ``a.sum``, ``new NDArray(...)``), ``group`` / ``name`` (the gap map's keys),
        ``numpy_kind`` (``absent`` for a NumSharp-only name), ``marks`` (the gap map's symbol per mark), ``status``
        (its closest-overload status) and the non-empty detail lists:
        ``missing``, ``excluded``, ``foreign``, ``typecode``, ``generic`` (T ◇ selectors), ``binding`` / ``null`` /
        ``sequence`` (the probe's B / L / Q findings), ``order``, ``required``, ``default``, ``varargs``, ``split``,
        ``renamed``, ``extra``, ``null_default``, ``numpy_added`` (parameters read from NumPy's stubs / docs),
        ``extension_hosts``; ``overloads``, ``closest``, ``numpy`` and ``via``.
    """
    lists = {
        "missing": "missing_any_overload", "excluded": "missing_excluded", "foreign": "numsharp_only_params",
        "typecode": "typecode_params", "generic": "typecode_generic_selectors", "binding": "probe_binding",
        "null": "probe_null", "sequence": "probe_sequence", "order": "slot", "required": "required",
        "default": "default", "varargs": "varargs", "split": "split", "renamed": "renamed", "extra": "extra",
        "null_default": "null_default", "numpy_added": "numpy_params_added", "extension_hosts": "extension_hosts",
    }
    spelling: dict[str, Any] = {
        "csharp": csharp_spelling(row["module"], row["name"], group_types),
        "group": row["module"], "name": row["name"], "numpy_kind": row["numpy_kind"], "status": row["status"],
        "marks": {mark: row[f"mark_{mark}"] for mark in "MXOTBLQ"},
        "overloads": int(row["overloads"] or 0), "closest": row["best"], "numpy": row["numpy_signature"],
    }
    for key, column in lists.items():
        # A probe finding can hold "; " in its own text, so those cells are split by how a finding starts.
        values = split_findings(row[column]) if key in ("binding", "null", "sequence") else split_list(row[column])
        if values:
            spelling[key] = values
    if via:
        spelling["via"] = via
    return spelling


def spelling_differs(spelling: Mapping[str, Any]) -> bool:
    """Whether one C# spelling differs from NumPy's signature (the gap map's "has a mark", M ◦ aside).

    A spelling missing only plan-excluded parameters (M ◦) does not differ: the parity plan leaves those parameters out
    of scope. Every other mark does, and so does a closest-overload difference (the gap map never shows one without a
    mark, but a hard difference is a difference whatever the marks say).

    Args:
        spelling: From ``spelling_from_gap``.

    Returns:
        True when the spelling carries M ●, X, O, T (● or ◇), B / Q (● or ◐), L (● / ◐ / ◦) or a hard
        closest-overload difference.
    """
    marks = spelling["marks"]
    if marks["M"] == "●" or marks["X"] == "●" or marks["O"] or marks["T"]:
        return True
    if any(marks[mark] in symbols for mark, symbols in PROBE_MARKS.items()):
        return True
    return any(spelling.get(kind) for kind in ("order", "required", "default", "varargs", "split", "renamed", "extra"))


def _primary(spellings: Sequence[Mapping[str, Any]], target: str, group_types: Mapping[str, str]) -> Mapping[str, Any]:
    """The spelling a record's top-level fields describe.

    Args:
        spellings: The row's spellings (at least one).
        target: The row's credited C# member (``NumSharp.np.sum``), or empty.
        group_types: ``GapMap.group_types``.

    Returns:
        The spelling of the credited member when one is; else a module function or ``ndarray`` member, then another
        class's member, then a constructor, then the typed ``NDArray<T>`` spelling (the first of the best kind).
    """
    def rank(spelling: Mapping[str, Any]) -> int:
        """Order the spellings by how directly they spell the row's NumPy API.

        Args:
            spelling: One spelling.

        Returns:
            0 for the credited member, then 1 (a function or ndarray member), 2 (another class's member),
            3 (a constructor), 4 (the typed receiver).
        """
        member = group_types.get(spelling["group"], "").replace("+", ".") + "." + spelling["name"]
        if target and member == target:
            return 0
        if spelling["group"] == TYPED_NDARRAY_GROUP:
            return 4
        if spelling["name"] == CONSTRUCTOR:
            return 3
        return 1 if spelling["group"] == "ndarray" or is_module_group(spelling["group"]) else 2

    return min(spellings, key=rank)


def record_from_spellings(spellings: Sequence[Mapping[str, Any]], spelling_kind: str, target: str,
                          group_types: Mapping[str, str]) -> dict[str, Any]:
    """The signature record of a row from its C# spellings (gap-map rows).

    The row-level marks are the union over the spellings (the strongest symbol of each mark); the list fields
    ``missing`` / ``excluded`` / ``foreign`` / ``typecode`` are the union too (a port may write any of the spellings),
    and ``overloads``, ``closest``, ``numpy`` and the closest-overload lists describe the primary spelling
    (``_primary``). Every spelling keeps its own detail under ``spellings``.

    Args:
        spellings: The row's spellings (at least one).
        spelling_kind: ``exact``, or ``alias`` when the row is credited through another name (mark N).
        target: The row's credited C# member, or empty.
        group_types: ``GapMap.group_types``.

    Returns:
        The record: ``status`` (``identical``, ``differs``, or ``not-compared`` when no spelling has a NumPy signature
        to compare), ``spelling``, ``source`` (``gap-map``), ``marks``, the union lists, the primary spelling's fields,
        and ``spellings``.
    """
    primary = _primary(spellings, target, group_types)
    compared = [s for s in spellings if s["marks"]["M"] != "—"]
    marks = {mark: _strongest(s["marks"][mark] for s in spellings) for mark in "MXBLQ"}
    marks["O"] = str(max(s["overloads"] for s in spellings)) if any(s["marks"]["O"] for s in spellings) else ""
    t_symbols = "".join(s["marks"]["T"] for s in spellings)
    marks["T"] = ("●" if "●" in t_symbols else "") + ("◇" if "◇" in t_symbols else "")
    record: dict[str, Any] = {"spelling": spelling_kind, "source": "gap-map", "marks": marks,
                              "overloads": primary["overloads"]}
    for key in ("missing", "excluded", "foreign", "typecode", "generic"):
        union: list[str] = []
        for spelling in spellings:
            union.extend(value for value in spelling.get(key, []) if value not in union)
        if union:
            record[key] = union
    for key in ("order", "required", "default", "varargs", "null_default", "binding", "null", "sequence"):
        if primary.get(key):
            record[key] = list(primary[key])
    record["closest"] = primary["closest"]
    record["numpy"] = primary["numpy"]
    record["spellings"] = [dict(s) for s in spellings]
    if not compared:
        record["status"] = "not-compared"
        record["reason"] = "the gap map compares no NumPy signature here (" + primary["status"] + ")"
    else:
        record["status"] = "differs" if spelling_kind == "alias" or any(spelling_differs(s) for s in compared) \
            else "identical"
    return record


def inventory_marks(record: Mapping[str, Any]) -> dict[str, str]:
    """The mark symbols of a record the inventory computed itself (no gap-map spelling: never compile-probed).

    Args:
        record: From ``assess``.

    Returns:
        The gap map's symbol per mark: ``M`` ``●`` / ``◦``, ``X`` ``●``, ``O`` the overload count above one, ``T``
        ``●``, and ``—`` for B / L / Q (not probed).
    """
    return {"M": "●" if record.get("missing") else "◦" if record.get("excluded") else "",
            "X": "●" if record.get("foreign") else "",
            "O": str(record["overloads"]) if record.get("overloads", 1) > 1 else "",
            "T": "●" if record.get("typecode") else "", "B": "—", "L": "—", "Q": "—"}


def same_numpy_object(np: Any, first: str, second: str) -> bool:
    """Whether two catalog ids name the same NumPy object (``numpy.acos`` and ``numpy.arccos``,
    ``numpy.matrix.fill`` and ``numpy.ndarray.fill``).

    Args:
        np: The imported numpy module.
        first: A catalog id.
        second: Another catalog id.

    Returns:
        True when both resolve and are the same object once a bound method is unwrapped to its function.
    """
    resolved_a, a = resolve_api_id(np, first)
    resolved_b, b = resolve_api_id(np, second)
    if not (resolved_a and resolved_b) or a is None:
        return False
    return getattr(a, "__func__", a) is getattr(b, "__func__", b)


def gap_member_id(row: Mapping[str, str], group_types: Mapping[str, str]) -> str | None:
    """The C# member a gap-map row compares, in the catalog's target spelling (``NumSharp.np.sum``).

    Args:
        row: A gap-map row.
        group_types: ``GapMap.group_types``.

    Returns:
        ``<CLR type with + read as .>.<name>``, or None for a constructor, the typed group (its members are not a
        catalog target) or a group of unknown type.
    """
    clr = group_types.get(row["module"])
    if not clr or row["name"] == CONSTRUCTOR or row["module"] == TYPED_NDARRAY_GROUP:
        return None
    return clr.replace("+", ".") + "." + row["name"]


def declared_member_ids(member: Mapping[str, Any] | None, target: str) -> list[str]:
    """The C# members a credited target stands for when its overloads are inherited.

    ``NumSharp.NDMatrix.fill`` is ``NDArray.fill`` (NDMatrix declares no ``fill``): the gap map compares it once, under
    the declaring type's group, so the adoption looks it up there.

    Args:
        member: The target's inventory member (``member_index``), or None.
        target: The credited target id.

    Returns:
        The target, then ``<declaring type>.<name>`` when every overload is declared by one other type.
    """
    ids = [target]
    declaring = (member or {}).get("declaringTypes") or []
    owner, _, name = target.rpartition(".")
    if len(declaring) == 1 and declaring[0].replace("+", ".") != owner:
        ids.append(declaring[0].replace("+", ".") + "." + name)
    return ids


def attach(rows: list[dict[str, Any]], inventory: Mapping[str, Any], np: Any, gap: GapMap | None = None) -> list[str]:
    """Attach the signature-parity record to every row that takes part (``row["signature"]``).

    Without a gap map, every record is the inventory's own comparison (``assess``: reflection only, no compile probe).
    With one, every C# spelling the gap map compares joins the catalog row of the NumPy API it implements, and the
    record is built from those spellings (``record_from_spellings``), so the inventory reports exactly the gap map's
    marks. The joins, in order:

    1. by id: a NumPy row of the gap map joins ``signature_gap.catalog_id`` (a constructor joins the class row, the
       typed ``NDArray<T>`` group joins ``numpy.ndarray``'s rows);
    2. by C# member: a NumSharp-only row of the gap map (and a NumPy one whose catalog id the catalog does not list,
       such as ``numpy.random.get_bit_generator``, outside NumPy's ``__all__`` and documentation) joins the
       NumSharp-only catalog row of the same member;
    3. by adoption: a credited row no gap row joined adopts the spellings of a row that compares the SAME C# member
       (or the type that declares it, for an inherited member) against the SAME NumPy object, because the verdict
       depends on nothing else: ``numpy.acos`` adopts ``numpy.arccos``'s (and keeps its N mark),
       ``numpy.matrix.fill`` adopts ``numpy.ndarray.fill``'s, ``numpy.random.RandomState.beta`` ``numpy.random.beta``'s.

    Rows left without a spelling keep the inventory's own record (``source: inventory``, B / L / Q not probed).

    Args:
        rows: The catalog rows (changed in place).
        inventory: The reflected NumSharp inventory (schema 7).
        np: The imported numpy module.
        gap: The gap map (``signature_gap.run``), or None for the inventory's own comparison only.

    Returns:
        Problem lines: a row whose record carries a ``WIRING_REASONS`` reason (``"numpy.x -> NumSharp.np.x:
        reason"``); a gap-map row with a NumPy counterpart that joins no catalog row and is not in
        ``GAP_JOIN_EXEMPT``; an exemption that no longer occurs. Empty when everything joined.
    """
    index = member_index(inventory)
    problems = []
    by_id = {row["id"]: row for row in rows}
    joined: dict[str, list[dict[str, Any]]] = {}
    exempt_seen = set()
    if gap is not None:
        types = gap.group_types
        numsharp_rows = {row["numsharp_target"]: row for row in rows
                         if row.get("origin") == "numsharp" and row.get("numsharp_target")}
        for gap_row in gap.rows:
            key = (gap_row["module"], gap_row["name"])
            if gap_row["numpy_kind"] != "absent":
                cid = catalog_id(*key)
                if cid in by_id:
                    if not by_id[cid].get("numsharp_target"):
                        # The gap map compares a C# spelling of an API the catalog calls missing: the two disagree on
                        # what exists (numpy.char.chararray read "missing" while NDCharArray implemented it).
                        problems.append(f"gap map {key[0]} / {key[1]}: compares a C# member for {cid}, which the "
                                        "catalog does not credit")
                    joined.setdefault(cid, []).append(spelling_from_gap(gap_row, types))
                    continue
            member = gap_member_id(gap_row, types)
            if member in numsharp_rows:
                joined.setdefault(numsharp_rows[member]["id"], []).append(spelling_from_gap(gap_row, types))
                continue
            if gap_row["numpy_kind"] == "absent":
                continue  # A NumSharp-only member of an object type: the catalog lists those of the facades only.
            if key in GAP_JOIN_EXEMPT:
                exempt_seen.add(key)
                continue
            problems.append(f"gap map {key[0]} / {key[1]}: joins no catalog row ({catalog_id(*key)})")
        problems.extend(f"stale gap-map exemption {key[0]} / {key[1]}: it joins a row now, or the gap map no longer "
                        "lists it" for key in GAP_JOIN_EXEMPT if key not in exempt_seen)
        # The C# member each joined NumPy spelling compares, for the adoption below.
        by_member: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for gap_row in gap.rows:
            member = gap_member_id(gap_row, types)
            if member and gap_row["numpy_kind"] != "absent" and catalog_id(gap_row["module"], gap_row["name"]) in by_id:
                by_member.setdefault(member, []).append((catalog_id(gap_row["module"], gap_row["name"]), gap_row))
    for row in rows:
        target = row.get("numsharp_target") or ""
        member = index.get(target)
        record = assess(np, row, member)
        spellings = joined.get(row["id"])
        if gap is not None and spellings is None and target and row.get("origin") == "numpy" \
                and not row.get("python_protocol"):
            for candidate in declared_member_ids(member, target):
                adopted = [spelling_from_gap(gap_row, gap.group_types, via=cid) for cid, gap_row
                           in by_member.get(candidate, ()) if cid != row["id"] and same_numpy_object(np, row["id"], cid)]
                if adopted:
                    spellings = adopted
                    break
        if spellings:
            kind = (record or {}).get("spelling") or ("alias" if row.get("availability") == "alias" else "exact")
            record = record_from_spellings(spellings, kind, target, gap.group_types)
        elif record is not None:
            record["source"] = "inventory"
            record["marks"] = inventory_marks(record)
        if record is None:
            continue
        row["signature"] = record
        if record.get("reason") in WIRING_REASONS:
            problems.append(f"{row['id']} -> {row['numsharp_target']}: {record['reason']}")
    return problems


def marks(record: Mapping[str, Any] | None) -> str:
    """The marks of one record as letters, in ``MARK_ORDER``.

    Args:
        record: A row's signature record, or None.

    Returns:
        ``"MXOTBLQN"``-style letters: M a missing NumPy parameter (● only; M ◦, plan-excluded only, is no mark), X a
        parameter NumPy lacks, O more than one overload, T a dtype as NPTypeCode / Type or picked by a generic type
        argument, B a NumPy-style call that does not bind (● or ◐), L ``null`` refused for NumPy's None (● / ◐ / ◦), Q
        an integer sequence refused (● or ◐), N credited through another name. A not-compared record carries N at
        most, plus the O / T of a NumSharp-only member's spellings. Empty for no mark or no record.
    """
    if not record:
        return ""
    symbols = record.get("marks") or {}
    alias = record.get("spelling") == "alias"
    if record.get("status") == "not-compared":
        letters = "".join(letter for letter in "OT" if symbols.get(letter))
        return letters + ("N" if alias else "")
    letters = ""
    letters += "M" if symbols.get("M") == "●" or (not symbols and record.get("missing")) else ""
    letters += "X" if symbols.get("X") == "●" or (not symbols and record.get("foreign")) else ""
    letters += "O" if symbols.get("O") or (not symbols and record.get("overloads", 1) > 1) else ""
    letters += "T" if symbols.get("T") or (not symbols and record.get("typecode")) else ""
    for letter, counted in PROBE_MARKS.items():
        letters += letter if symbols.get(letter) in counted else ""
    letters += "N" if alias else ""
    return letters


#: The gap map's own count keys (``signature_diff.mark_counts``), in its order: the reconciliation counts the
#: spellings the artifact carries under the same keys, so the two lists can be compared line by line.
GAP_COUNT_KEYS = ("numpy", "M", "M_excluded_only", "X", "O", "T", "T_generic", "B", "B_12", "L", "L_12", "L_default",
                  "Q", "no_mark")


def spelling_counts(symbols: Mapping[str, str]) -> dict[str, int]:
    """Count one spelling's marks under the gap map's keys (``signature_diff.mark_counts``' rules).

    Args:
        symbols: A spelling's ``marks`` (the gap map's symbol per mark).

    Returns:
        ``GAP_COUNT_KEYS`` -> 0 or 1 (``numpy`` is always 1: the spelling has a NumPy counterpart).
    """
    m, b, n, q, t = symbols["M"], symbols["B"], symbols["L"], symbols["Q"], symbols["T"]
    flags = {
        "numpy": 1, "M": m == "●", "M_excluded_only": m == "◦", "X": symbols["X"] == "●", "O": bool(symbols["O"]),
        "T": "●" in t, "T_generic": "◇" in t, "B": b == "●", "B_12": b == "◐", "L": n == "●", "L_12": n == "◐",
        "L_default": n == "◦", "Q": q in ("●", "◐"),
    }
    flags["no_mark"] = not (m in ("●", "◦") or flags["X"] or flags["O"] or t or b in ("●", "◐")
                            or n in ("●", "◐", "◦") or flags["Q"])
    return {key: int(flags[key]) for key in GAP_COUNT_KEYS}


def reconcile(gap: GapMap, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that the artifact carries every finding of the gap map: count the spellings the rows carry under the gap
    map's own keys, and say where each spelling with a NumPy counterpart that the rows do not carry went.

    Args:
        gap: The gap map.
        rows: The emitted (projected) rows.

    Returns:
        ``tool`` (the gap map's own counts), ``carried`` (the same keys over the spellings the rows carry, each
        spelling once however many rows carry it), ``probe`` (the compile probe's counts), ``elsewhere`` (spellings carried by a
        NumSharp-only row: NumPy has the name outside its ``__all__`` and documentation) and ``outside`` (spellings no
        emitted row carries: an exemption, or a row the dashboard projection leaves out, ``numpy.lib`` /
        ``numpy.testing``), each a list of ``{"group", "name", "id", "reason"}``.
    """
    carried = {key: 0 for key in GAP_COUNT_KEYS}
    seen, elsewhere = set(), []
    for row in rows:
        for spelling in (row.get("signature") or {}).get("spellings", []):
            key = (spelling["group"], spelling["name"])
            # A spelling counts once however many rows carry it (an adopted copy is the same finding as its
            # source row's, and still carries it when the projection drops the source); a NumSharp-only spelling
            # has no NumPy counterpart, and the gap map's counts leave those out too.
            if key in seen or spelling.get("numpy_kind") == "absent":
                continue
            seen.add(key)
            for count_key, value in spelling_counts(spelling["marks"]).items():
                carried[count_key] += value
            if row.get("origin") == "numsharp":
                elsewhere.append({"group": key[0], "name": key[1], "id": row["id"],
                                  "reason": "NumPy has the name outside its __all__ and documentation; the catalog "
                                            "lists the NumSharp member as NumSharp-only"})
    emitted = {row["id"] for row in rows}
    outside = []
    for gap_row in gap.rows:
        key = (gap_row["module"], gap_row["name"])
        if gap_row["numpy_kind"] == "absent" or key in seen:
            continue
        cid = catalog_id(*key)
        reason = GAP_JOIN_EXEMPT.get(key) or ("the dashboard projection leaves the row out (numpy.lib and "
                                              "numpy.testing are not shown, by the owner's rule)"
                                              if cid not in emitted else "carried by no emitted row")
        outside.append({"group": key[0], "name": key[1], "id": cid, "reason": reason})
    return {"tool": {key: gap.marks.get(key, 0) for key in GAP_COUNT_KEYS}, "carried": carried, "probe": gap.probe,
            "elsewhere": elsewhere, "outside": outside}


def classify(row: Mapping[str, Any]) -> str:
    """A row's signature-parity class, the single definition the summary counts and the dashboard mirror.

    Args:
        row: A catalog row.

    Returns:
        ``identical``, ``differs``, ``not-compared``, or ``none`` (the row takes no part).
    """
    record = row.get("signature")
    return record["status"] if record else "none"


def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Count the signature-parity classes and marks over one scope's rows.

    Args:
        rows: The rows of one scope.

    Returns:
        ``signature_identical`` / ``signature_differs`` / ``signature_not_compared``; the per-mark row counts over the
        compared rows (``signature_missing`` M, ``signature_foreign`` X, ``signature_overloaded`` O,
        ``signature_typecode`` T ●, ``signature_generic`` T ◇, ``signature_typecode_any`` T with either symbol (a row
        can carry both, so the two counts above do not add up to it), ``signature_binding`` B, ``signature_null`` L,
        ``signature_sequence`` Q, ``signature_excluded_only`` rows missing only plan-excluded parameters) and
        ``signature_other_name`` (N, every record); the closest-overload counts (``signature_order``,
        ``signature_required``, ``signature_default``, ``signature_varargs``); ``signature_unprobed`` (compared rows
        whose record is the inventory's own comparison: the gap map's compile probe never saw them); and
        ``signature_percent``: identical rows over every row of the scope (the denominator of ``coverage_percent``).
    """
    tally = {key: 0 for key in (
        "signature_identical", "signature_differs", "signature_not_compared", "signature_missing",
        "signature_foreign", "signature_overloaded", "signature_typecode", "signature_generic",
        "signature_typecode_any", "signature_binding", "signature_null", "signature_sequence", "signature_other_name",
        "signature_excluded_only", "signature_order", "signature_required", "signature_default", "signature_varargs",
        "signature_unprobed")}
    for row in rows:
        record = row.get("signature")
        if not record:
            continue
        status = record["status"]
        tally["signature_" + status.replace("-", "_")] += 1
        if record.get("spelling") == "alias":
            tally["signature_other_name"] += 1
        if status == "not-compared":
            continue
        letters = marks(record)
        symbols = record.get("marks") or {}
        tally["signature_missing"] += "M" in letters
        tally["signature_foreign"] += "X" in letters
        tally["signature_overloaded"] += "O" in letters
        tally["signature_typecode"] += "●" in symbols.get("T", "") or (not symbols and bool(record.get("typecode")))
        tally["signature_generic"] += "◇" in symbols.get("T", "")
        tally["signature_typecode_any"] += "T" in letters
        tally["signature_binding"] += "B" in letters
        tally["signature_null"] += "L" in letters
        tally["signature_sequence"] += "Q" in letters
        tally["signature_excluded_only"] += bool(record.get("excluded")) and "M" not in letters
        tally["signature_unprobed"] += record.get("source") == "inventory"
        for key in CLOSEST_HARD_KINDS:
            tally["signature_" + key] += bool(record.get(key))
    total = len(rows)
    tally["signature_percent"] = round(tally["signature_identical"] * 100 / total, 1) if total else 0.0
    return tally
