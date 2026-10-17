"""Compare every function NumSharp exposes under np.* with NumPy 2.4.2's signature, and write the differences.

Usage (from this directory; the committed outputs are docs/plans/numpy-signature-diff.md / .csv and
docs/plans/numpy-signature-marks.md):
    dotnet run --no-cache dump_modules.cs <scratch>/modules_dump.json <scratch>/types_dump.json
    python -I binding_probe.py <scratch>/modules_dump.json <scratch>/types_dump.json \
        ../../src/NumSharp.Core/bin/Debug/net10.0/NumSharp.dll <scratch>/probe <scratch>/probe_results.json
    python -I signature_diff.py <scratch>/modules_dump.json <md> <csv> <marks> \
        --types <scratch>/types_dump.json --probe <scratch>/probe_results.json
with <md> = ../../docs/plans/numpy-signature-diff.md, <csv> = ../../docs/plans/numpy-signature-diff.csv and
<marks> = ../../docs/plans/numpy-signature-marks.md. The dump is compiled from the working tree (dump_modules.cs builds
NumSharp.Core Debug, the DLL the probe then references), so regenerate after any change to a public np.* signature.
Without ``--types`` only the module facades are compared (the first pass); without ``--probe`` the B / L / Q marks are
left out.

The marks table (the owner's requests of 2026-10-17) lists EVERY C# name once, NumSharp-only names included, with
marks taken over ALL its overloads together (``marks_for``): M, a NumPy parameter no overload offers (a rename counts;
parameters the plan excludes are shown apart); X, a parameter NumPy does not have in some overload; O, more than one
overload; T, a dtype taken as NPTypeCode or Type instead of DType (or picked by a generic type argument). The second
pass adds what reflection cannot see, from compiling NumPy-style calls (binding_probe.py): B, such a call does not
bind; L, ``null`` is refused where NumPy's default is None; Q, a sequence is refused where NumPy's stubs allow one.

The module dump comes from ``dump_modules.cs`` (same directory): every ``[ModuleName]`` module of NumSharp.Core (np,
ndarray, np.linalg, np.fft, np.random, np.ma, np.emath, np.char, np.strings, np.dtypes, np.typing, np.polynomial and its
bases) plus the object API np.random.Generator; its type dump (second output) every exported type, from which the
second pass takes NumPy's other classes (OBJECT_GROUPS), constructors, extension methods and the typed NDArray<T>.
NumPy 2.4.2 must be importable; its ``.pyi`` stubs and numpydoc sections fill in the parameters a runtime signature
hides behind ``(*args, **kwargs)`` (``--no-stubs`` turns that off).

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
import argparse
import ast
import csv
import importlib
import inspect
import json
import math
import os
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

#: C# parameter types that select a dtype the pre-DType way: since the dtype system's Stage B every dtype parameter is
#: meant to be ONE `DType` (an NPTypeCode, a Type or a dtype string converts to it implicitly), so a parameter still
#: typed NPTypeCode / Type (nullable or array forms included) is a leftover the marks table flags.
TYPECODE_TYPE = re.compile(r"^(NPTypeCode|Type)\??(\[\])*$")

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
        annotation: The type annotation's source text when the parameter comes from a ``.pyi`` stub
            (``_ShapeLike | None``), else None.
    """

    def __init__(self, name, kind, default, annotation=None):
        """Store the fields.

        Args:
            name: The parameter's name.
            kind: The ``inspect.Parameter`` kind.
            default: ``REQUIRED`` or the default's text.
            annotation: The stub annotation's text, or None.
        """
        self.name, self.kind, self.default, self.annotation = name, kind, default, annotation

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


def split_top(text, separator=","):
    """Split a parameter list at its top-level commas: commas inside brackets, braces, parentheses or quotes do not
    split (``shape=(2, 3)`` stays one piece).

    Args:
        text: The text between a signature's parentheses.
        separator: The character to split at (``"|"`` splits a type annotation into its union's members).

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
        if ch == separator and depth == 0:
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


#: Descriptor types of attributes that are not callables (a property, a C getset / member slot).
PROPERTY_DESCRIPTORS = ("getset_descriptor", "member_descriptor", "property")

#: Descriptor types of methods implemented in C: inspect folds their keywords into **kwargs (ndarray.sum) or finds no
#: signature (dtype.newbyteorder), so the docstring's first line, NumPy's documented signature, is read first.
C_METHOD_DESCRIPTORS = ("method_descriptor", "builtin_function_or_method", "wrapper_descriptor",
                        "classmethod_descriptor")


def numpy_signature(module_key, name, obj, target=None):
    """Find NumPy's signature of one name.

    Args:
        module_key: The dump's group key (``np``, ``ndarray``, ``np.linalg``, ``np.random.Generator``,
            ``np.ma.MaskedArray``, ...).
        name: The function's name.
        obj: NumPy's object of that name in the module, or None when NumPy has none.
        target: The module or class the name was looked up in; for a class, a C-implemented method reads its
            docstring first and a non-callable attribute is a property. None reads ``module_key == "ndarray"`` as the
            ndarray class (the first pass's call form).

    Returns:
        ``(kind, params, text, is_ufunc)``: kind is one of absent, module, property, attribute, method, function,
        class, ufunc, ma-forward, ma-method; params is a list of Params or None when NumPy states no signature; text is the
        signature as shown in the report; is_ufunc says whether the ufunc exclusions apply.
    """
    if obj is None:
        return "absent", None, "", False
    if inspect.ismodule(obj):
        return "module", None, "", False
    if target is None and module_key == "ndarray":
        target = np.ndarray
    if inspect.isclass(target):
        raw = inspect.getattr_static(target, name, None)
        if type(raw).__name__ in PROPERTY_DESCRIPTORS or (
                raw is not None and not callable(raw) and not isinstance(raw, (staticmethod, classmethod))):
            return "property", None, "", False
        if type(raw).__name__ in C_METHOD_DESCRIPTORS:
            params = from_doc(getattr(obj, "__doc__", "") or "", name)
            if params is not None:
                prefix = "a" if target is np.ndarray else target.__name__
                return "method", params, f"{prefix}.{name}({format_params(params)})", False
    forwarded = ma_forwarded(obj)
    if forwarded is not None:
        params, wrapped = forwarded
        return "ma-forward", params, f"{name}({format_params(params)})", isinstance(wrapped, np.ufunc)
    if isinstance(obj, np.ufunc):
        kind = "ufunc"
    elif inspect.isclass(obj):
        kind = "class"
    elif callable(obj):
        kind = "method" if inspect.isclass(target) else "function"
    else:
        return "attribute", None, "", False
    try:
        params = from_inspect(inspect.signature(obj))
    except (TypeError, ValueError):
        params = from_doc(getattr(obj, "__doc__", "") or "", name)
    if params is None:
        return kind, None, "(NumPy states no signature)", kind == "ufunc"
    if module_key == "np.ma" and [(p.name, p.kind) for p in params] == [
            ("a", P.POSITIONAL_OR_KEYWORD), ("args", P.VAR_POSITIONAL), ("params", P.VAR_KEYWORD)]:
        # np.ma.copy / diagonal / repeat / swapaxes are _frommethod wrappers of MaskedArray methods that are themselves
        # _arraymethod forwarders: the call reaches ndarray's method, so its documented parameters are the real ones.
        method = from_doc(getattr(getattr(np.ndarray, name, None), "__doc__", "") or "", name)
        if method is not None:
            # A method parameter that is positional-only (swapaxes' axis1, axis2) can only arrive through *args, which
            # also makes `a` effectively positional; Python cannot order a keyword-capable `a` before it otherwise.
            head = Param("a", P.POSITIONAL_ONLY if any(p.kind == P.POSITIONAL_ONLY for p in method)
                         else P.POSITIONAL_OR_KEYWORD, REQUIRED)
            return "ma-method", [head] + method, f"{name}({format_params([head] + method)})", False
    prefix = "" if not inspect.isclass(target) else ("a." if target is np.ndarray else f"{target.__name__}.")
    if (inspect.isclass(target) and target is not np.ndarray and issubclass(target, np.ndarray)
            and params and all(p.kind in (P.VAR_POSITIONAL, P.VAR_KEYWORD) for p in params)
            and hasattr(np.ndarray, name)):
        # A subclass method that forwards (*args, **params) to ndarray's own (MaskedArray.copy / .diagonal / .repeat
        # are _arraymethod wrappers): the call reaches ndarray's method, so its documented parameters are the real ones.
        method = from_doc(getattr(getattr(np.ndarray, name), "__doc__", "") or "", name)
        if method is not None:
            return "method", method, f"{prefix}{name}({format_params(method)})", False
    return kind, params, f"{prefix}{name}({format_params(params)})", kind == "ufunc"


def numpy_class_call(cls):
    """NumPy's signature for calling a class (its constructor), the counterpart of a C# constructor.

    Args:
        cls: A NumPy class.

    Returns:
        The list of Params, or None when neither inspect nor the class docstring states one.
    """
    try:
        return from_inspect(inspect.signature(cls))
    except (TypeError, ValueError):
        return from_doc(getattr(cls, "__doc__", "") or "", cls.__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Second-pass sources for NumPy's parameters: the type stubs NumPy ships (.pyi) and the numpydoc "Parameters" sections.
# A runtime signature that forwards (*args, **kwargs) hides the parameters it accepts (np.printoptions forwards every
# keyword to set_printoptions, so inspect sees none of them); both sources name them.
# ----------------------------------------------------------------------------------------------------------------------

#: Stub modules (by dotted name, prefix match) searched for a facade's module-level function when the function has no
#: Python code to point at its own module (a C function such as np.empty). Order is preference: the first module that
#: defines the name wins. Modules of other facades are left out, so `np.add` does not pick up numpy.char's `add`.
STUB_SCOPES = {
    "np": ("numpy._core.multiarray", "numpy._core.numeric", "numpy._core.fromnumeric", "numpy._core.function_base",
           "numpy._core.shape_base", "numpy._core.numerictypes", "numpy._core.arrayprint", "numpy._core.einsumfunc",
           "numpy._core._ufunc_config", "numpy._core._asarray", "numpy._core.getlimits", "numpy._core.printoptions",
           "numpy.lib._", "numpy.matrixlib", "numpy"),
    "np.ma": ("numpy.ma.core", "numpy.ma.extras"),
    "np.random": ("numpy.random.mtrand", "numpy.random._generator", "numpy.random.bit_generator", "numpy.random"),
    "np.linalg": ("numpy.linalg._linalg",),
    "np.fft": ("numpy.fft._pocketfft", "numpy.fft._helper"),
    "np.char": ("numpy._core.defchararray",),
    "np.strings": ("numpy._core.strings",),
    "np.emath": ("numpy.lib._scimath_impl",),
    "np.dtypes": ("numpy.dtypes",),
}


def stub_args(node):
    """The parameters of one stub ``def`` as Params (self / cls dropped).

    Args:
        node: An ``ast.FunctionDef``.

    Returns:
        The Params in signature order, each with its annotation's text; a parameter with a default gets the default's
        source text (``...`` in most stubs), one without gets ``REQUIRED``.
    """
    a = node.args

    def annotation(arg):
        return ast.unparse(arg.annotation) if arg.annotation is not None else None

    positional = [(x, P.POSITIONAL_ONLY) for x in a.posonlyargs] + [(x, P.POSITIONAL_OR_KEYWORD) for x in a.args]
    first_default = len(positional) - len(a.defaults)
    out = []
    for i, (arg, kind) in enumerate(positional):
        default = ast.unparse(a.defaults[i - first_default]) if i >= first_default else REQUIRED
        out.append(Param(arg.arg, kind, default, annotation(arg)))
    if a.vararg is not None:
        out.append(Param(a.vararg.arg, P.VAR_POSITIONAL, REQUIRED, annotation(a.vararg)))
    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
        out.append(Param(arg.arg, P.KEYWORD_ONLY, REQUIRED if default is None else ast.unparse(default),
                         annotation(arg)))
    if a.kwarg is not None:
        out.append(Param(a.kwarg.arg, P.VAR_KEYWORD, REQUIRED, annotation(a.kwarg)))
    return [p for p in out if p.name not in ("self", "cls")]


#: The members of a stub annotation's union that make the parameter take a sequence of integers: an axis tuple, a
#: shape tuple (``_ShapeLike`` is ``SupportsIndex | Sequence[SupportsIndex]``), a size tuple.
_INT_SEQUENCE = ("_ShapeLike", "Sequence[int]", "Sequence[SupportsIndex]", "Iterable[int]", "Iterable[SupportsIndex]",
                 "tuple[int", "tuple[SupportsIndex")


def takes_int_sequence(annotation):
    """Whether a stub annotation lets the parameter be a sequence of integers.

    Only the union's top-level members count, so ``ndarray[tuple[int], ...]`` (an array's shape type argument) does
    not.

    Args:
        annotation: The annotation's source text, or None.

    Returns:
        True when one top-level member starts with an integer-sequence type.
    """
    if not annotation:
        return False
    return any(member.strip().startswith(_INT_SEQUENCE) for member in split_top(annotation, "|"))


#: The key under which ``load_stub_index`` keeps a stub module's class bases (``{Class: [base names]}``); no stub
#: name can collide with it.
STUB_BASES = "<class bases>"


def load_stub_index():
    """Parse every NumPy ``.pyi`` stub (tests excluded) into one index.

    Two stub idioms hide a signature from a plain ``Class.name`` / ``name`` lookup, and both are resolved here: a
    method declared on a stub-only base class (``ndarray.sum`` is ``_ArrayOrScalarCommon.sum``; each module's class
    bases are kept under ``STUB_BASES`` for ``stub_overloads``), and a module-level alias of a function
    (``nansum = sum`` after ``from numpy._core.fromnumeric import sum``), which gets its target's overloads.

    Returns:
        ``{module: {qualname: [overload Params lists]}}``, module being the stub's dotted module name
        (``numpy._core.fromnumeric``), qualname ``name`` or ``Class.name``; every ``@overload`` of a name is one list.
        ``index[module][STUB_BASES]`` maps the module's classes to their base class names.
    """
    root = os.path.dirname(np.__file__)
    index, aliases = {}, []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("tests", "testing", "_pyinstaller", "f2py", "typing")]
        for fn in filenames:
            if not fn.endswith(".pyi"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)[:-4].replace(os.sep, ".")
            module = "numpy" if rel == "__init__" else "numpy." + (rel[:-9] if rel.endswith(".__init__") else rel)
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
            defs, bases, imports = {}, {}, {}

            def visit(body, prefix):
                """Collect the defs of one body, recursing into classes and ``if`` blocks (version checks).

                Args:
                    body: A list of statements.
                    prefix: ``""`` at module level, ``"Class."`` inside a class.
                """
                for node in body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.setdefault(prefix + node.name, []).append(stub_args(node))
                    elif isinstance(node, ast.ClassDef):
                        # A base's generic arguments do not matter: `ndarray[_ShapeT, _DTypeT]` is the class ndarray.
                        bases[prefix + node.name] = [ast.unparse(b).split("[", 1)[0].rsplit(".", 1)[-1]
                                                     for b in node.bases]
                        visit(node.body, prefix + node.name + ".")
                    elif isinstance(node, ast.If):
                        visit(node.body, prefix)
                        visit(node.orelse, prefix)
                    elif isinstance(node, ast.ImportFrom) and not prefix and node.module:
                        for alias in node.names:
                            imports[alias.asname or alias.name] = (node.module, alias.name)
                    elif (isinstance(node, ast.Assign) and not prefix and len(node.targets) == 1
                          and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Name)):
                        aliases.append((module, node.targets[0].id, node.value.id, imports))

            visit(tree.body, "")
            defs[STUB_BASES] = bases
            index[module] = defs
    # Aliases resolve after every module is parsed, because their target may live in a module read later.
    for module, name, target, imports in aliases:
        defs = index[module]
        if name in defs:
            continue
        if target in defs:
            defs[name] = defs[target]
        elif target in imports:
            source, original = imports[target]
            if original in index.get(source, {}):
                defs[name] = index[source][original]
    return index


def module_of_code(obj):
    """The dotted module of the Python file that implements a NumPy function, through its dispatcher or wrapper.

    NumPy's public functions report ``__module__ == "numpy"`` (set_module), so the module is read from the code
    object's file instead, which names the stub to read (``numpy/_core/fromnumeric.py`` -> ``numpy._core.fromnumeric``).

    Args:
        obj: A NumPy callable.

    Returns:
        The dotted module name, or None for a C function (no code object) or a file outside the numpy package.
    """
    impl = getattr(obj, "_implementation", None) or obj
    try:
        impl = inspect.unwrap(impl)
    except ValueError:
        pass
    code = getattr(impl, "__code__", None)
    if code is None:
        return None
    root = os.path.dirname(np.__file__)
    path = os.path.normcase(os.path.abspath(code.co_filename))
    if not path.startswith(os.path.normcase(root)) or not path.endswith(".py"):
        return None
    rel = os.path.relpath(path, os.path.normcase(root))[:-3].replace(os.sep, ".")
    return "numpy" if rel == "__init__" else "numpy." + (rel[:-9] if rel.endswith(".__init__") else rel)


def stub_overloads(index, group_key, target, name, obj):
    """Find the stub overloads NumPy's stubs declare for one name of a group.

    Args:
        index: From ``load_stub_index``.
        group_key: The group (``np``, ``np.ma``, ``ndarray``, ``np.ma.MaskedArray``, ...).
        target: The NumPy module or class of the group.
        name: The member name.
        obj: NumPy's object (None when absent).

    Returns:
        ``(module, qualname, [Params lists])`` or None when no stub declares it.
    """
    if inspect.isclass(target):
        # A class member: search every stub module for `Class.name`, the class's own module first, then for the
        # member on the class's stub bases, nearest first (ndarray's reductions are declared on the stub-only
        # `_ArrayOrScalarCommon`, MaskedArray's inherited methods on `ndarray`).
        own = getattr(target, "__module__", "") or ""
        modules = sorted(index, key=lambda m: (m != own, not own.startswith(m), m))
        queue, seen = [target.__name__], set()
        while queue:
            cls = queue.pop(0)
            if cls in seen:
                continue
            seen.add(cls)
            qual = f"{cls}.{name}"
            for m in modules:
                if qual in index[m]:
                    return m, qual, index[m][qual]
            for m in modules:
                queue.extend(b for b in index[m].get(STUB_BASES, {}).get(cls, ()) if b not in seen)
        return None
    candidates = []
    home = module_of_code(obj) if obj is not None else None
    if home in index:
        candidates.append(home)
    for prefix in STUB_SCOPES.get(group_key, ("numpy." + group_key[3:],) if group_key.startswith("np.") else ()):
        candidates.extend(m for m in sorted(index) if (m == prefix or m.startswith(prefix)) and m not in candidates)
    for m in candidates:
        for qual in (name, f"RandomState.{name}") if group_key == "np.random" else (name,):
            if qual in index.get(m, {}):
                return m, qual, index[m][qual]
    return None


#: A numpydoc section heading line followed by its underline.
_DOC_HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z ]*)\s*\n\s*-{3,}\s*$", re.M)


def doc_parameter_names(doc):
    """The parameter names a numpydoc docstring's ``Parameters`` section documents.

    Args:
        doc: A docstring (may be None).

    Returns:
        The names in order (``*args`` / ``**kwargs`` entries keep their stars), or an empty list without the section.
    """
    if not doc:
        return []
    text = inspect.cleandoc(doc)
    headings = list(_DOC_HEADING.finditer(text))
    for i, h in enumerate(headings):
        if h.group(1).strip() != "Parameters":
            continue
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        names = []
        for line in text[h.end():end].splitlines():
            # An entry starts at column 0: "name : type", "name1, name2 : type" or a bare "name"; its description is
            # indented below it.
            if not line or line[0].isspace():
                continue
            head = line.split(" : ")[0].split(":")[0].strip()
            for piece in head.split(","):
                piece = piece.strip()
                if re.match(r"^\*{0,2}[A-Za-z_]\w*$", piece):
                    names.append(piece)
        return names
    return []


def effective_params(params, source, stub, doc_names):
    """Add the parameters a runtime signature hides to it, from the stubs and the documentation.

    The runtime signature stays the base. Stub / doc names are added only where the runtime one cannot show them: a
    pure forwarder ``(*args, **kwargs)`` takes the stub's whole signature, a ``**kwargs`` takes the stub's and the
    documented keyword names it lacks, and a docstring-sourced signature (a C method) takes the stub names its one
    documented line lacks.

    A stub's POSITIONAL-ONLY parameters are never added: they cannot be passed by name, and the stubs use them to spell
    the fixed-arity overloads of a ``*args`` signature (``atleast_1d(a0, a1, /)``, ``ndarray.reshape(size0, size1, /)``,
    ``ndarray.item(i0, /)``), names no caller can write. A runtime ``(*args)`` without ``**kwargs`` hides no name either,
    so it is never replaced. For the same reason a documented name listed before the ``*args`` entry is not added
    (``einsum``'s ``subscripts`` travels in ``*operands``).

    Args:
        params: The runtime Params (from ``numpy_signature``).
        source: ``"doc"`` when ``params`` came from a docstring line, ``"sig"`` otherwise.
        stub: The stub overload lists (``stub_overloads``' third item) or None.
        doc_names: From ``doc_parameter_names``.

    Returns:
        ``(params, added)``: the effective Params and the ``"name (stub)"`` / ``"name (doc)"`` texts of the added ones.
    """
    kinds = [p.kind for p in params]
    has_kw = P.VAR_KEYWORD in kinds
    forwarder = has_kw and all(k in (P.VAR_POSITIONAL, P.VAR_KEYWORD) for k in kinds)
    union = []
    for overload in stub or []:
        for p in overload:
            if p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY) and p.name not in {q.name for q in union}:
                union.append(p)
    added = []
    if forwarder and union:
        # The stub's first overload in full (its keyword-capable parameters): their kinds and defaults say how the
        # forwarded parameters are passed.
        base = [Param(p.name, p.kind, p.default) for p in stub[0]
                if p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)]
        names = {p.name for p in base}
        for p in union:
            if p.name not in names:
                base.append(Param(p.name, P.KEYWORD_ONLY, p.default))
        added = [f"{p.name} (stub)" for p in base if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)]
        return base, added
    out = list(params)
    names = {p.name for p in out}
    if has_kw or source == "doc":
        for p in union:
            if p.name not in names:
                # A docstring signature lists positional parameters in order, so a stub name it lacks is a keyword;
                # behind **kwargs every added name is a keyword too.
                insert = Param(p.name, P.KEYWORD_ONLY, p.default)
                at = next((i for i, q in enumerate(out) if q.kind == P.VAR_KEYWORD), len(out))
                out.insert(at, insert)
                names.add(p.name)
                added.append(f"{p.name} (stub)")
    if has_kw:
        # A name documented BEFORE the *args entry is one of the positional arguments *args carries (einsum documents
        # `subscripts` ahead of `operands`): no caller passes it by name, so it is not a keyword. Without a documented
        # *args entry the order says nothing, and nothing is set aside.
        var_positional = next((p.name for p in out if p.kind == P.VAR_POSITIONAL), None)
        carried = []
        if var_positional is not None:
            for n in doc_names:
                if n.lstrip("*") == var_positional:
                    break
                carried.append(n)
            else:
                carried = []
        for n in doc_names:
            if not n.startswith("*") and n not in names and n not in carried:
                at = next((i for i, q in enumerate(out) if q.kind == P.VAR_KEYWORD), len(out))
                out.insert(at, Param(n, P.KEYWORD_ONLY, "<optional>"))
                names.add(n)
                added.append(f"{n} (doc)")
    return out, added


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


def typecode_params(overloads):
    """The parameters (and return types) of a name's overloads that take a dtype as NPTypeCode or Type, and the
    generic overloads whose type argument selects the dtype.

    A generic overload selects the dtype by a CLR type when the caller must spell its type argument: no parameter
    mentions it, so C# cannot infer it (``iinfo<T>()``, ``zeros<T>(shape)``). Where a parameter carries it
    (``array<T>(T[] data)``) the element type comes from the data, NumPy's own inference, and it is not counted.

    Args:
        overloads: All overload dictionaries of the name.

    Returns:
        ``(params, returns, selectors)``: the distinct ``"Type name"`` texts of such parameters in declaration order,
        the distinct NPTypeCode / Type return types (a result NumPy gives as a dtype), and one ``"<T>(...)"`` text per
        generic overload whose type argument cannot be inferred.
    """
    found = []
    for ov in overloads:
        for c in ov["params"]:
            text = f"{c['type']} {c['name']}"
            if TYPECODE_TYPE.match(c["type"]) and text not in found:
                found.append(text)
    returns = sorted({ov["returns"] for ov in overloads if TYPECODE_TYPE.match(ov["returns"])})
    selectors = []
    for ov in overloads:
        for g in ov["generic"]:
            # A whole-word search: `T` must not match inside `Tuple` or `TDType`.
            if not any(re.search(rf"(?<![A-Za-z0-9_]){re.escape(g)}(?![A-Za-z0-9_])", c["type"]) for c in ov["params"]):
                text = f"<{', '.join(ov['generic'])}>(" + ", ".join(f"{c['type']} {c['name']}" for c in ov["params"]) + ")"
                if text not in selectors:
                    selectors.append(text)
                break
    return found, returns, selectors


def marks_for(np_params, overloads, policy):
    """The union view over ALL overloads of one name: which NumPy parameters no overload offers, and which C#
    parameter names NumPy does not have.

    This differs from ``compare`` (one overload at a time): a NumPy parameter is missing only when no overload has it,
    and a C# parameter is reported when any overload has it. The same matching rules apply: a positional-only NumPy
    parameter is offered by any C# parameter at its slot (and any name there is fine), NumPy's ``*args`` is offered by
    a params array or by the C# parameters at and after its slot that carry no NumPy name, and C# parameters NumPy
    takes through ``**kwargs`` are not foreign. A NumPy parameter whose slot holds a foreign C# name in some overload is
    reported with that name (a rename: the NumPy keyword does not compile, and the C# name is foreign).

    Args:
        np_params: NumPy's Params.
        overloads: All overload dictionaries of the name.
        policy: The names this function excludes by policy (reported apart from the missing ones).

    Returns:
        A dictionary: ``missing`` (NumPy parameters no overload offers, ``"axes (C#: premute)"`` for a rename),
        ``missing_policy`` (the excluded ones no overload offers), ``extra`` (the foreign C# parameter names, sorted).
    """
    # Only the named parameters match by name: a C# `params object[] args` IS NumPy's `*args` whatever its name, so the
    # names of NumPy's *args / **kwargs must not take part (`common_type(*arrays)` against `params NDArray[] arrays`).
    np_names = {p.name for p in np_params if p.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)}
    positional = [p for p in np_params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    var_positional = next((p for p in np_params if p.kind == P.VAR_POSITIONAL), None)
    var_keyword = next((p for p in np_params if p.kind == P.VAR_KEYWORD), None)
    offered, renamed, extra = set(), {}, set()
    for ov in overloads:
        cs = ov["params"]
        names = [c["name"] for c in cs]
        for slot, p in enumerate(positional):
            at_slot = cs[slot] if slot < len(cs) and not cs[slot]["params"] else None
            if p.kind == P.POSITIONAL_ONLY:
                # Only the slot counts: Python cannot pass this parameter by name.
                if at_slot is not None and (at_slot["name"] == p.name or at_slot["name"] not in np_names):
                    offered.add(p.name)
            elif p.name in names:
                offered.add(p.name)
            elif at_slot is not None and at_slot["name"] not in np_names:
                renamed.setdefault(p.name, set()).add(at_slot["name"])
        for p in np_params:
            if p.kind == P.KEYWORD_ONLY and p.name in names:
                offered.add(p.name)
        for i, c in enumerate(cs):
            if c["name"] in np_names:
                continue
            if i < len(positional) and positional[i].kind == P.POSITIONAL_ONLY:
                continue  # A positional-only slot: Python never sees the name.
            if var_positional is not None and (c["params"] or i >= len(positional)):
                offered.add("*" + var_positional.name)  # C# spells what NumPy's *args carries.
                continue
            if var_keyword is not None:
                continue  # A keyword NumPy accepts through **kwargs.
            extra.add(c["name"])
    missing, missing_policy = [], []
    for p in np_params:
        if p.kind == P.VAR_KEYWORD:
            continue  # C# has no **kwargs; what it would carry is matched by name above.
        key = "*" + p.name if p.kind == P.VAR_POSITIONAL else p.name
        if key in offered:
            continue
        if p.name in policy:
            missing_policy.append(p.name)
        elif p.name in renamed:
            missing.append(f"{p.name} (C#: {', '.join(sorted(renamed[p.name]))})")
        else:
            missing.append(key)
    return {"missing": missing, "missing_policy": missing_policy, "extra": sorted(extra)}


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
        The names: a module's ``__all__`` (its ``dir`` without underscores when it has none, or an empty one:
        ``numpy.lib.format`` sets ``__all__ = []`` and documents ``read_array`` / ``write_array`` / ``open_memmap``
        as its API), a class's ``dir`` without underscores.
    """
    if inspect.isclass(target):
        return {n for n in dir(target) if not n.startswith("_")}
    names = getattr(target, "__all__", None)
    if not names:
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


#: Object APIs: NumPy classes whose methods are np.*.* API, each with the CLR type that implements it (the coverage
#: tool's object_surfaces.OBJECT_SURFACES map, plus numpy.char.chararray -> NDCharArray and the typed NDArray<T>), as
#: (group key, path under numpy, CLR type, own_only). own_only: the C# type derives from another compared type (NDArray,
#: BitGenerator), so NumPy names it lacks are taken from what NumPy's class defines itself (inherited ones belong to the
#: base group), while every member the C# type declares is still compared.
OBJECT_GROUPS = (
    ("ndarray (NDArray<T>)", "ndarray", "NumSharp.Generic.NDArray`1", True),
    ("np.matrix", "matrix", "NumSharp.NDMatrix", True),
    ("np.char.chararray", "char.chararray", "NumSharp.NDCharArray", True),
    ("np.ma.MaskedArray", "ma.MaskedArray", "NumSharp.NDMaskedArray", False),
    ("np.dtype", "dtype", "NumSharp.DType", False),
    ("np.nditer", "nditer", "NumSharp.np+NDIterator", False),
    ("np.broadcast", "broadcast", "NumSharp.np+Broadcast", False),
    ("np.flatiter", "flatiter", "NumSharp.np+FlatIterator", False),
    ("np.ndindex", "ndindex", "NumSharp.np+NDIndex", False),
    ("np.ndenumerate", "ndenumerate", "NumSharp.np+NDEnumerate", False),
    ("np.poly1d", "poly1d", "NumSharp.poly1d", False),
    ("np.vectorize", "vectorize", "NumSharp.Vectorized", False),
    ("np.finfo", "finfo", "NumSharp.finfo", False),
    ("np.iinfo", "iinfo", "NumSharp.iinfo", False),
    ("np.lib.npyio.NpzFile", "lib.npyio.NpzFile", "NumSharp.IO.NpzFile", False),
    ("np.random.SeedSequence", "random.SeedSequence", "NumSharp.SeedSequence", False),
    ("np.random.BitGenerator", "random.BitGenerator", "NumSharp.BitGenerator", False),
    ("np.random.MT19937", "random.MT19937", "NumSharp.MT19937", True),
    ("np.random.PCG64", "random.PCG64", "NumSharp.PCG64", True),
    ("np.random.PCG64DXSM", "random.PCG64DXSM", "NumSharp.PCG64DXSM", True),
    ("np.random.Philox", "random.Philox", "NumSharp.Philox", True),
    ("np.random.SFC64", "random.SFC64", "NumSharp.SFC64", True),
)

#: The CLR type of a module group whose constructors / extension methods also belong to it (`new NDArray(...)` is
#: NumPy's `np.ndarray(...)` class call; an extension method on NDArray binds as `a.name(...)`).
MODULE_TYPES = {"ndarray": "NumSharp.NDArray", "np.random.Generator": "NumSharp.Generator"}

#: The row name constructors get in a group (NumPy's class call).
CONSTRUCTOR = "(constructor)"

#: NumPy's public modules (the coverage tool's EXTENDED_SUBMODULES and headline surfaces, which follow NumPy's own
#: test_public_api.py list), with the facade group that implements each one or None. A module without a facade is
#: reported with its names, and with the names NumSharp has elsewhere.
NUMPY_PUBLIC_MODULES = (
    ("", "np"), ("random", "np.random"), ("linalg", "np.linalg"), ("fft", "np.fft"), ("lib.scimath", "np.emath"),
    ("polynomial", "np.polynomial"), ("polynomial.polyutils", "np.polynomial.polyutils"),
    ("polynomial.polynomial", "np.polynomial.polynomial"), ("polynomial.chebyshev", "np.polynomial.chebyshev"),
    ("polynomial.legendre", "np.polynomial.legendre"), ("polynomial.hermite", "np.polynomial.hermite"),
    ("polynomial.hermite_e", "np.polynomial.hermite_e"), ("polynomial.laguerre", "np.polynomial.laguerre"),
    ("ma", "np.ma"), ("char", "np.char"), ("strings", "np.strings"), ("dtypes", "np.dtypes"), ("typing", "np.typing"),
    ("lib.stride_tricks", None), ("lib.array_utils", None), ("rec", None), ("lib.recfunctions", None),
    ("exceptions", None), ("ma.mrecords", None), ("ctypeslib", None), ("lib.format", None), ("lib.introspect", None),
    ("lib", None), ("lib.mixins", None), ("version", None), ("f2py", None), ("testing", None),
    ("testing.overrides", None),
)


def resolve_numpy_path(path):
    """Walk a dotted path under numpy by attribute access.

    Args:
        path: ``""`` for numpy itself, else a path such as ``ma.MaskedArray`` or ``lib.npyio.NpzFile``.

    Returns:
        The object, or None when an attribute along the path is missing.
    """
    target = np
    for part in [p for p in path.split(".") if p]:
        target = getattr(target, part, None)
        if target is None:
            return None
    return target


def extension_overloads(types, clr):
    """The extension methods every exported static class declares for one CLR type, without their ``this`` parameter.

    Args:
        types: The type dump (``dump_modules.cs``' second output).
        clr: The extended type's full name.

    Returns:
        ``{name: [overload dictionaries]}``; each overload is a copy whose first parameter is dropped (``a.f(x)``
        passes ``a`` as it) and whose ``host`` names the declaring class.
    """
    found = {}
    for host, info in types.items():
        for name, overloads in info["methods"].items():
            for ov in overloads:
                if ov.get("extension") and ov.get("extends") == clr:
                    copy = dict(ov)
                    copy["params"] = ov["params"][1:]
                    copy["host"] = host
                    found.setdefault(name, []).append(copy)
    return found


def build_groups(dump, types):
    """Every group the comparison covers: each module facade, then (with the type dump) each object API.

    Args:
        dump: The module dump (``dump_modules.cs``' first output).
        types: The type dump, or None for the first pass's module facades only.

    Returns:
        A list of dictionaries: ``key``, ``target`` (NumPy's module or class), ``methods`` (name -> overloads,
        constructors under ``CONSTRUCTOR``), ``members`` (every name the C# side has, inherited NumSharp members
        included), ``own_only`` (see ``OBJECT_GROUPS``), ``clr`` (the CLR type or the facade type).
    """
    groups = []
    for key in sorted(dump, key=lambda k: (k != "np", k != "ndarray", k)):
        info = dump[key]
        methods = {n: list(v) for n, v in info["methods"].items()}
        members = set(methods) | set(info["properties"]) | set(info["fields"]) | set(info.get("types", []))
        clr = MODULE_TYPES.get(key)
        if types is not None and clr in types:
            if types[clr]["ctors"]:
                methods[CONSTRUCTOR] = list(types[clr]["ctors"])
            for name, extra in extension_overloads(types, clr).items():
                methods.setdefault(name, []).extend(extra)
                members.add(name)
        groups.append({"key": key, "target": resolve_module(key), "methods": methods, "members": members,
                       "own_only": False, "clr": clr or info["type"]})
    if types is None:
        return groups
    for key, path, clr, own_only in OBJECT_GROUPS:
        info = types.get(clr)
        target = resolve_numpy_path(path)
        if info is None or target is None:
            continue
        methods = {n: list(v) for n, v in info["methods"].items()}
        if info["ctors"]:
            # A generic class's constructors take the class's type argument as their dtype (`new NDArray<double>(shape)`),
            # so they carry its generic parameters, which typecode_params reads as a dtype selector.
            methods[CONSTRUCTOR] = [dict(c, generic=list(info["genericParams"])) for c in info["ctors"]]
        for name, extra in extension_overloads(types, clr).items():
            methods.setdefault(name, []).extend(extra)
        members = set(methods) | set(info["properties"]) | set(info["fields"]) | set(info["types"])
        # Members a C# subclass inherits from a NumSharp base are there for the caller too (NDMatrix.reshape is
        # NDArray.reshape), so they count as present when NumPy's names are checked.
        base = info["base"]
        while base in types:
            members |= set(types[base]["methods"]) | set(types[base]["properties"]) | set(types[base]["fields"])
            base = types[base]["base"]
        groups.append({"key": key, "target": target, "methods": methods, "members": members, "own_only": own_only,
                       "clr": clr})
    return groups


def new_stats(key, methods):
    """The zeroed statistics of one group.

    Args:
        key: The group key.
        methods: The group's name -> overloads map.

    Returns:
        The statistics dictionary ``write_markdown`` reads.
    """
    stats = {"module": key, "names": len(methods), "overloads": sum(len(v) for v in methods.values())}
    for k in ("numpy", "identical", "multi", "differs", "single_differs", "null_only", "policy_only", "cs_only",
              "np_absent", "no_sig") + HARD_KINDS:
        stats[k] = 0
    return stats


def analyze_group(group, stub_index, rows, marked):
    """Compare every name of one group with NumPy and append its rows.

    Args:
        group: From ``build_groups``.
        stub_index: From ``load_stub_index``, or None to read runtime signatures only (the first pass).
        rows: The compared rows (with NumPy's signature) to extend.
        marked: Every row (NumSharp-only names included) to extend.

    Returns:
        ``(stats, cs_only)``: the group's statistics and its NumSharp-only names.
    """
    key, target = group["key"], group["target"]
    stats = new_stats(key, group["methods"])
    cs_only = []
    for name, overloads in group["methods"].items():
        if name == CONSTRUCTOR:
            obj = target if inspect.isclass(target) else None
            params = numpy_class_call(target) if obj is not None else None
            label = "ndarray" if target is np.ndarray else getattr(target, "__name__", key)
            kind, is_ufunc, source = ("class" if obj is not None else "absent"), False, "sig"
            sig_text = f"{label}({format_params(params)})" if params is not None else ""
        else:
            obj = getattr(target, name, None)
            kind, params, sig_text, is_ufunc = numpy_signature(key, name, obj, target)
            raw = inspect.getattr_static(target, name, None) if inspect.isclass(target) else None
            source = "doc" if type(raw).__name__ in C_METHOD_DESCRIPTORS else "sig"
            if params is not None and name.startswith("__") and name.endswith("__"):
                # A protocol method is called by Python's syntax (`m[i]`, `m * x`), never with keywords, so its
                # parameter names cannot matter: compare them by slot, like positional-only ones.
                params = [Param(p.name, P.POSITIONAL_ONLY, p.default) if p.kind == P.POSITIONAL_OR_KEYWORD else p
                          for p in params]
        row = {"module": key, "name": name, "numpy_kind": kind, "numpy_signature": sig_text,
               "overloads": len(overloads), "orpa": sum(1 for o in overloads if o["orpa"] is not None),
               "generic": sum(1 for o in overloads if o["generic"]),
               "cs_signatures": [cs_signature(name, o) for o in overloads], "best": "", "diff": None,
               "numpy_added": [], "extension_hosts": sorted({o["host"] for o in overloads if o.get("host")})}
        # The marks are kept for EVERY name, NumSharp-only ones included (their overload count and NPTypeCode / Type
        # parameters still apply); missing / foreign parameters need NumPy's signature, so they stay None.
        tc_params, tc_returns, tc_selectors = typecode_params(overloads)
        row["marks"] = None
        row["typecode"], row["typecode_returns"], row["typecode_selectors"] = tc_params, tc_returns, tc_selectors
        marked.append(row)
        if kind == "absent":
            row["status"] = "NumSharp only"
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
        if stub_index is not None:
            # Second pass: what the runtime signature hides behind (*args, **kwargs), from the stubs and the docs.
            if name == CONSTRUCTOR:
                stub = None
                for qual in ("__init__", "__new__"):
                    stub = stub_overloads(stub_index, key, target, qual, getattr(target, qual, None))
                    if stub is not None:
                        break
                doc = getattr(target, "__doc__", "")
            else:
                stub = stub_overloads(stub_index, key, target, name, obj)
                doc = getattr(obj, "__doc__", "")
            params, added = effective_params(params, source, stub[2] if stub else None, doc_parameter_names(doc))
            if added:
                row["numpy_added"] = added
                head = sig_text.split("(", 1)[0]
                row["numpy_signature"] = f"{head}({format_params(params)})"
            # The parameters NumPy's stubs let take a sequence of integers (`axis=(0, 1)`, `size=(2, 3)`), which
            # binding_probe.py passes as one (the Q mark). Stub names the signature does not have (the fixed-arity
            # `size0, size1, /` spellings of *args) are left out.
            names = {p.name for p in params}
            row["sequence_params"] = sorted({p.name for overload in (stub[2] if stub else []) for p in overload
                                             if p.name in names and takes_int_sequence(p.annotation)})
        policy = policy_for(name, is_ufunc)
        row["marks"] = marks_for(params, overloads, policy)
        # Kept for binding_probe.py, which writes NumPy-style C# calls from them (not written to the CSV).
        row["np_params"], row["overload_list"], row["policy"] = params, overloads, policy
        scored = []
        for index, ov in enumerate(overloads):
            d = compare(params, ov, policy)
            scored.append((cost(d, ov), index, d))
        scored.sort(key=lambda t: (t[0], t[1]))
        _, best_index, best = scored[0]
        split_missing(best, overloads, best_index)
        row["best"] = row["cs_signatures"][best_index]
        row["best_index"] = best_index
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
    return stats, cs_only


def numpy_names_for(group):
    """The NumPy names a group is checked against for "NumPy names NumSharp lacks".

    Args:
        group: From ``build_groups``.

    Returns:
        A module's public names; a class's public names (only its own ones for an ``own_only`` group).
    """
    target = group["target"]
    if inspect.isclass(target) and group["own_only"]:
        return {n for n in vars(target) if not n.startswith("_")}
    return numpy_public_names(target)


def elsewhere_index(groups):
    """Map every C# member name to the groups that have it, to say where NumSharp has a name a group lacks.

    Args:
        groups: From ``build_groups``.

    Returns:
        ``{name: [group keys]}``.
    """
    index = {}
    for g in groups:
        for n in g["members"]:
            index.setdefault(n, []).append(g["key"])
    return index


def missing_modules(groups):
    """NumPy's public modules that no group implements, with their names and where NumSharp has any of them.

    Args:
        groups: From ``build_groups``.

    Returns:
        A list of ``(module path, names, {name: [group keys]})`` for every module without a facade.
    """
    where = elsewhere_index(groups)
    out = []
    for path, facade in NUMPY_PUBLIC_MODULES:
        if facade is not None:
            continue
        module = resolve_numpy_path(path)
        if module is None:
            # A public submodule `import numpy` does not load (numpy.lib.recfunctions, numpy.ma.mrecords,
            # numpy.lib.format) is still public API: import it by name.
            try:
                module = importlib.import_module("numpy." + path)
            except ImportError:
                out.append((path, [], {}))
                continue
        names = sorted(numpy_public_names(module))
        out.append((path, names, {n: where[n] for n in names if n in where}))
    return out


def run_analysis(dump, types, stub_index):
    """Compare every group with NumPy (the whole comparison, without writing anything).

    Args:
        dump: The module dump.
        types: The type dump, or None for the module facades only.
        stub_index: From ``load_stub_index``, or None for runtime signatures only.

    Returns:
        ``(groups, rows, summary, surface, marked)``: the groups, the compared rows, the per-group statistics, the
        per-group NumSharp-only / absent names, and every row.
    """
    groups = build_groups(dump, types)
    rows, summary, surface, marked = [], [], {}, []
    where = elsewhere_index(groups)
    for group in groups:
        stats, cs_only = analyze_group(group, stub_index, rows, marked)
        absent = sorted(numpy_names_for(group) - group["members"])
        stats["np_absent"] = len(absent)
        surface[group["key"]] = {"cs_only": sorted(cs_only), "np_absent": absent,
                                 "elsewhere": {n: [k for k in where.get(n, []) if k != group["key"]] for n in absent}}
        summary.append(stats)
    return groups, rows, summary, surface, marked


def attach_probe_results(marked, results):
    """Attach binding_probe.py's results to the rows they belong to, with their B, L and Q findings.

    Args:
        marked: Every row from ``run_analysis``. Each gets ``probes`` (its probes), ``probe_skipped`` (the reason it
            was not probed, or None), ``probe_b``, ``probe_n`` and ``probe_q`` (``classify_probes``).
        results: ``binding_probe.py``'s JSON: ``probes`` (each with ``group``, ``name``, ``kind`` — ``R`` the required
            parameters only, ``S`` the optional positional ones positionally up to one, ``P`` one optional parameter
            with an ordinary value, ``N`` one parameter as None, ``D`` the same with ``default``, ``A`` every NumPy
            parameter the C# side names, ``V`` that call without the None values, ``Q`` / ``U`` a sequence parameter as
            an int array / a C# tuple, ``Y`` the dtype as a string —, ``param``, ``call`` and ``errors``, the
            compiler's ``[code, message]`` pairs per language version: ``{"12": [...], "14": [...]}``) and ``skipped``
            (``group``, ``name``, ``reason`` for the names it could not write a call for).
    """
    by_row, skipped = {}, {}
    for p in results["probes"]:
        by_row.setdefault((p["group"], p["name"]), []).append(p)
    for s in results.get("skipped", []):
        skipped[(s["group"], s["name"])] = s["reason"]
    for r in marked:
        r["probes"] = by_row.get((r["module"], r["name"]), [])
        r["probe_skipped"] = skipped.get((r["module"], r["name"]))
        r["probe_b"], r["probe_n"], r["probe_q"] = classify_probes(r["probes"])


#: Compiler errors a probe gets from the value None itself (an argument ``null``): a struct parameter (CS0037), another
#: value type or a type null does not convert to (CS1503), or two overloads null fits equally well (CS0121).
NULL_CODES = frozenset({"CS0037", "CS1503", "CS0121"})


def short_error(code, message):
    """Shorten a compiler error to the part a marks table cell needs.

    Args:
        code: The CS code.
        message: The compiler's message.

    Returns:
        A short text: ``needs `dtype```, ``no overload takes 2 arguments``, ``ambiguous: `f(int?)` / `f(int[])```, ...
        A CS1503 that is not about the ``null`` argument names the compiler's closest candidate, because the compiler
        reports the first argument that candidate refuses, not the one that rules every overload out.
    """
    message = message.replace("NumSharp.Backends.", "").replace("NumSharp.", "")
    if code == "CS1503" and "'<null>'" not in message:
        m = re.search(r"Argument (\d+): cannot convert from '([^']+)' to '([^']+)'", message)
        if m:
            return f"no overload fits; the closest one wants `{m.group(3)}` for argument {m.group(1)}"
    patterns = (
        ("CS7036", r"required parameter '(\w+)'", "needs `{0}`"),
        ("CS1501", r"takes (\d+) arguments?", "no overload takes {0} arguments"),
        ("CS1739", r"parameter named '(\w+)'", "the best overload has no `{0}`"),
        ("CS1744", r"Named argument '(\w+)'", "`{0}` is already given positionally"),
        ("CS0037", r"Cannot convert null to '([^']+)'", "null -> `{0}`"),
        ("CS1503", r"Argument (\d+): cannot convert from '([^']+)' to '([^']+)'", "argument {0}: `{1}` -> `{2}`"),
        ("CS0121", r"methods or properties: '(.+?)' and '(.+?)'$", "ambiguous: `{0}` / `{1}`"),
    )
    for c, pattern, text in patterns:
        if c == code:
            m = re.search(pattern, message)
            if m:
                return text.format(*m.groups())
    return f"{code}: {message[:100]}"


def classify_probes(probes):
    """Turn a name's compile probes into its B (binding) and N (None) findings.

    A probe that fails because a smaller probe of the name already fails is not reported again: when the
    required-only call (R) fails at a language version, every other probe fails with it there; when a parameter does
    not bind alone (P), its None probe (N) and the all-parameters call (A) fail with it; when the all-parameters call
    with ordinary values (V) fails, the same call with NumPy's None values (A) fails too.

    Args:
        probes: The name's probes (``attach_probe_results``).

    Returns:
        ``(b, n, q)``: lists of findings ``{"what", "call", "code", "message", "lang"}``, ``lang`` being ``"both"``,
        ``"12"`` (C# 12 only: the call relies on [OverloadResolutionPriority], which a net8.0 consumer's C# 12 ignores)
        or ``"14"``. B: a NumPy-style call that does not bind for a reason other than None (R, S, P, Y, V, or A without
        None values); L (the ``n`` list): a parameter whose None NumSharp refuses (an N probe, or the None values only
        together), with
        ``default_ok`` True when ``default`` binds where ``null`` does not (NumSharp's ``Shape`` idiom for
        ``size=None``); Q: a parameter NumPy lets take a sequence of integers where neither ``new[] { 0, 1 }`` nor
        ``(0, 1)`` binds.
    """
    by_kind = {(p["kind"], p["param"]): p for p in probes}

    def fails(p, lang):
        return bool(p and p["errors"].get(lang))

    def finding(p, what, blocked):
        langs = [lang for lang in ("12", "14") if fails(p, lang) and lang not in blocked]
        if not langs:
            return None
        code, message = p["errors"]["14" if "14" in langs else "12"][0]
        return {"what": what, "call": p["call"], "code": code, "message": message,
                "lang": "both" if len(langs) == 2 else langs[0]}

    b, n = [], []
    r = by_kind.get(("R", None))
    r_blocked = {lang for lang in ("12", "14") if fails(r, lang)}
    if r is not None:
        f = finding(r, "required parameters only", set())
        if f:
            b.append(f)
    p_blocked_any = set(r_blocked)
    for p in probes:
        if p["kind"] != "P":
            continue
        f = finding(p, f"`{p['param']}` alone", r_blocked)
        if f:
            b.append(f)
            p_blocked_any |= {lang for lang in ("12", "14") if fails(p, lang)}
    for p in probes:
        if p["kind"] != "Y":
            continue
        own = by_kind.get(("P", p["param"]))
        f = finding(p, f"`{p['param']}` as a dtype string",
                    r_blocked | {lang for lang in ("12", "14") if fails(own, lang)})
        if f:
            b.append(f)
    # The positional calls grow one slot at a time, so only the first failing one is new: every longer call holds
    # it. A P failure does not explain an S one: the positional call also carries the parameters before it (`a.sum(1,
    # np.float64)` passes the `axis` that `a.sum(dtype: ...)` lacks), so the order is a finding of its own.
    s_blocked = set(r_blocked)
    for p in probes:
        if p["kind"] != "S":
            continue
        f = finding(p, f"positionally up to `{p['param']}`", s_blocked)
        if f:
            b.append(f)
        s_blocked |= {lang for lang in ("12", "14") if fails(p, lang)}
    n_failed = set()
    for p in probes:
        if p["kind"] != "N":
            continue
        own = by_kind.get(("P", p["param"]))
        blocked = r_blocked | {lang for lang in ("12", "14") if fails(own, lang)}
        f = finding(p, f"`{p['param']}`", blocked)
        if not f:
            continue
        if f["code"] in NULL_CODES:
            d = by_kind.get(("D", p["param"]))
            f["default_ok"] = d is not None and not fails(d, "12") and not fails(d, "14")
            n.append(f)
            n_failed |= {lang for lang in ("12", "14") if fails(p, lang)}
        else:
            f["what"] = f"`{p['param']}: null`"
            b.append(f)
    a, v = by_kind.get(("A", None)), by_kind.get(("V", None))
    if v is not None:
        f = finding(v, "all NumPy parameters in one call", p_blocked_any)
        if f:
            b.append(f)
        else:
            f = finding(a, "the None values in one call", p_blocked_any | n_failed)
            if f:
                f["default_ok"] = False
                n.append(f)
    elif a is not None:
        f = finding(a, "all NumPy parameters in one call", p_blocked_any)
        if f:
            b.append(f)
    q = []
    for p in probes:
        if p["kind"] != "Q":
            continue
        # A sequence is refused only when neither spelling binds: the int array (Q) nor the C# tuple (U).
        u = by_kind.get(("U", p["param"]))
        own = by_kind.get(("P", p["param"]))
        blocked = r_blocked | {lang for lang in ("12", "14") if fails(own, lang)}
        blocked |= {lang for lang in ("12", "14") if u is not None and not fails(u, lang)}
        f = finding(p, f"`{p['param']}` as a sequence", blocked)
        if f:
            q.append(f)
    return b, n, q


def probe_symbol(findings, none=False):
    """The mark symbol of a list of B, L or Q findings.

    Args:
        findings: From ``classify_probes``.
        none: True for L findings (they know ``default_ok``).

    Returns:
        "●" when one fails at the latest C#, "◐" when every one fails at C# 12 only, "◦" (L only) when every refused
        None is a ``Shape`` that binds ``default``, else "".
    """
    if not findings:
        return ""
    strong = [f for f in findings if not (none and f.get("default_ok"))]
    if any(f["lang"] != "12" for f in strong):
        return "●"
    if strong:
        return "◐"
    return "◦"


def probe_text(findings):
    """Render B, L or Q findings for a table cell.

    Args:
        findings: From ``classify_probes``.

    Returns:
        One finding per ``<br>``-separated line: ``what: short error`` with the call for the R / A probes, and
        ``(C# 12 only)`` / ``(default binds)`` notes.
    """
    parts = []
    for f in findings:
        text = f"{f['what']}: {short_error(f['code'], f['message'])}"
        if f["what"] in ("required parameters only", "all NumPy parameters in one call",
                         "the None values in one call") or f["what"].startswith("positionally"):
            text += f" — `{f['call']}`"
        if f["lang"] == "12":
            text += " (C# 12 only)"
        if f.get("default_ok"):
            text += " (`default` binds)"
        parts.append(text)
    return "<br>".join(esc(p) for p in parts)


def main():
    """Read the dumps, compare every name, and write the Markdown report (and the CSV, and the marks table).

    Raises:
        SystemExit: Wrong command-line arguments (argparse).
    """
    parser = argparse.ArgumentParser(description="Diff NumSharp's np.* signatures against NumPy 2.4.2.")
    parser.add_argument("dump", help="dump_modules.cs' first output (the module facades)")
    parser.add_argument("md", help="the diff report to write")
    parser.add_argument("csv", nargs="?", help="the CSV to write")
    parser.add_argument("marks", nargs="?", help="the marks table to write")
    parser.add_argument("--types", help="dump_modules.cs' second output: adds the object APIs, constructors, "
                                        "extension methods and the typed NDArray<T> (the second pass)")
    parser.add_argument("--no-stubs", action="store_true",
                        help="read runtime signatures only (no .pyi stubs, no numpydoc Parameters sections)")
    parser.add_argument("--probe", help="binding_probe.py's results: adds the B (a NumPy-style call does not "
                                        "compile), L (NumPy's None is refused as null) and Q (a sequence is refused "
                                        "where NumPy's stubs allow one) marks")
    args = parser.parse_args()
    with open(args.dump, encoding="utf-8") as fh:
        dump = json.load(fh)
    types = None
    if args.types:
        with open(args.types, encoding="utf-8") as fh:
            types = json.load(fh)
    stub_index = None if args.no_stubs else load_stub_index()
    groups, rows, summary, surface, marked = run_analysis(dump, types, stub_index)
    if args.probe:
        with open(args.probe, encoding="utf-8") as fh:
            attach_probe_results(marked, json.load(fh))
    modules_missing = missing_modules(groups)
    write_markdown(args.md, summary, rows, surface)
    if args.csv:
        write_csv(args.csv, marked)
    if args.marks:
        write_marks_markdown(args.marks, marked, [s["module"] for s in summary], surface, modules_missing)
    totals = {k: sum(s[k] for s in summary) for k in summary[0] if k != "module"}
    print(json.dumps(totals))
    print(json.dumps(mark_counts(marked)))


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
    w("of NumSharp.Core and the object API `np.random.Generator`, and (with `--types`, the second pass) NumPy's other")
    w("classes as their C# types (`np.matrix` -> `NDMatrix`, `np.ma.MaskedArray` -> `NDMaskedArray`, the bit")
    w("generators, ...), constructors as `(constructor)`, extension methods, and the typed `NDArray<T>`. NumPy's")
    w("parameters come from its runtime signature, completed from its `.pyi` stubs and numpydoc sections where that")
    w("signature forwards `(*args, **kwargs)`. Do not edit it by hand; rerun the tools. The")
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


def mark_flags(row):
    """The marks of one name, as the marks table shows them.

    Args:
        row: A row from ``main`` (``marks`` is None for a NumSharp-only name or one without a NumPy signature).

    Returns:
        A dictionary: ``M`` is "●" when a NumPy parameter no overload offers is not excluded by the plan, "◦" when
        only excluded ones are missing, "—" when NumPy states no signature; ``X`` "●" when an overload has a parameter
        NumPy lacks ("—" likewise); ``O`` the overload count when above one; ``T`` "●" when an overload takes a dtype as
        NPTypeCode or Type, "◇" when the only such form is a generic overload whose type argument picks the dtype
        (``iinfo<T>()``), "●◇" for both; ``B``, ``L`` and ``Q`` the compile probe's marks (``probe_symbol``), "—" when
        the name was not probed ("?" when the probe could not write a call for it, "" when no probe ran at all).
    """
    m = row["marks"]
    no_signature = row["numpy_kind"] == "absent" or m is None
    typed = "●" if row["typecode"] else ""
    generic = "◇" if row.get("typecode_selectors") else ""
    if "probes" not in row:
        b = n = q = ""
    elif row.get("probe_skipped"):
        b = n = q = "?"
    elif not row["probes"]:
        b = n = q = "—"
    else:
        b, n = probe_symbol(row["probe_b"]), probe_symbol(row["probe_n"], none=True)
        q = probe_symbol(row["probe_q"])
    return {
        "M": "—" if no_signature else ("●" if m["missing"] else "◦" if m["missing_policy"] else ""),
        "X": "—" if no_signature else ("●" if m["extra"] else ""),
        "O": str(row["overloads"]) if row["overloads"] > 1 else "",
        "T": typed + generic,
        "B": b,
        "L": n,
        "Q": q,
    }


def mark_counts(marked):
    """Count the marks over every name.

    Args:
        marked: Every row from ``main``.

    Returns:
        A dictionary of counts: names with a NumPy counterpart, each mark among them (the probe's B / N by symbol),
        no mark at all, and the NumSharp-only names with their overload and NPTypeCode / Type marks.
    """
    counts = {"numpy": 0, "M": 0, "M_excluded_only": 0, "X": 0, "O": 0, "T": 0, "T_generic": 0, "B": 0, "B_12": 0,
              "L": 0, "L_12": 0, "L_default": 0, "Q": 0, "no_mark": 0, "not_compared": 0, "cs_only": 0,
              "cs_only_O": 0, "cs_only_T": 0}
    for r in marked:
        f = mark_flags(r)
        if r["numpy_kind"] == "absent":
            counts["cs_only"] += 1
            counts["cs_only_O"] += bool(f["O"])
            counts["cs_only_T"] += bool(f["T"])
            continue
        counts["numpy"] += 1
        counts["M"] += f["M"] == "●"
        counts["M_excluded_only"] += f["M"] == "◦"
        counts["X"] += f["X"] == "●"
        counts["O"] += bool(f["O"])
        counts["T"] += "●" in f["T"]
        counts["T_generic"] += "◇" in f["T"]
        counts["B"] += f["B"] == "●"
        counts["B_12"] += f["B"] == "◐"
        counts["L"] += f["L"] == "●"
        counts["L_12"] += f["L"] == "◐"
        counts["L_default"] += f["L"] == "◦"
        counts["Q"] += f["Q"] in ("●", "◐")
        counts["not_compared"] += f["M"] == "—"
        # A name missing only plan-excluded parameters still misses parameters, so M ◦ is a mark too, and so is every
        # probe mark: "no mark" means the name matches NumPy in every respect the table checks.
        counts["no_mark"] += not (f["M"] in ("●", "◦") or f["X"] == "●" or f["O"] or f["T"]
                                  or f["B"] in ("●", "◐") or f["L"] in ("●", "◐", "◦") or f["Q"] in ("●", "◐"))
    return counts


def write_marks_markdown(path, marked, modules, surface, modules_missing=()):
    """Write every name with its marks: missing NumPy parameters, foreign parameters, more than one overload,
    NPTypeCode / Type dtype parameters, and (with ``--probe``) NumPy-style calls that do not compile and None values
    NumSharp refuses; per group the NumPy names NumSharp has no member for (and where NumSharp has them instead); and
    NumPy's public modules that have no NumSharp facade at all.

    Args:
        path: The Markdown file to write (UTF-8, LF line ends).
        marked: Every row from ``main``.
        modules: The group keys in report order.
        surface: Per group, ``np_absent`` (NumPy's public names the group lacks) and ``elsewhere`` (name -> the
            groups that have it).
        modules_missing: From ``missing_modules``: ``(module path, names, {name: [group keys]})`` per NumPy module
            without a facade.
    """
    probed = any("probes" in r for r in marked)
    lines = []
    w = lines.append
    w("# NumSharp `np.*` names marked against NumPy 2.4.2")
    w("")
    w("Generated by `tools/api-parameter-parity/signature_diff.py` (4th argument) from the same comparison as")
    w("[`numpy-signature-diff.md`](numpy-signature-diff.md). Do not edit it by hand. Every C# name of every")
    w("`[ModuleName]` module, of `np.random.Generator` and (with `--types`) of every object API (`np.ma.MaskedArray`,")
    w("`np.matrix`, `np.char.chararray`, `np.dtype`, `np.nditer`, the bit generators, ...) is listed once, NumPy's names")
    w("first, then the names NumPy does not have. Constructors are the row `(constructor)` (NumPy's class call). The")
    w("marks look at ALL overloads of a name together (extension methods included):")
    w("")
    w("| mark | meaning |")
    w("|---|---|")
    # One table row per line: a Markdown table row cannot continue on the next line.
    w("| **M** ● | NumSharp is missing a NumPy parameter: no overload offers it. A rename counts, because NumPy's "
      "keyword does not compile: `axes (C#: premute)` |")
    w("| **M** ◦ | only parameters the parity plan excludes are missing (`like`, `subok`, a ufunc's `casting` / "
      "`order` / `signature`, the field `order` of the sorts, `einsum_call`) |")
    w("| **X** ● | NumSharp has a parameter NumPy does not have, in at least one overload (names at a positional-only "
      "slot, C# spellings of `*args`, and keywords NumPy takes through `**kwargs` are not counted) |")
    w("| **O** n | the name has n C# overloads, where NumPy has one signature |")
    w("| **T** ● | an overload takes a dtype as `NPTypeCode` or `Type` (nullable and array forms included) instead "
      "of `DType` |")
    w("| **T** ◇ | a generic overload whose type argument picks the dtype (`iinfo<T>()`): the caller names a CLR type "
      "instead of a `DType` |")
    if probed:
        w("| **B** ● | a NumPy-style call does not compile: the required parameters alone, the optional positional "
          "ones positionally in NumPy's order, one optional parameter alone (it binds only together with another "
          "one), the dtype as a string (`dtype: \"float64\"`), or all NumPy parameters in one call (they are split "
          "across overloads, or the overloads tie) |")
        w("| **B** ◐ | such a call compiles at the latest C# but not at C# 12, a net8.0 consumer's default: it relies "
          "on `[OverloadResolutionPriority]`, which C# 12 ignores |")
        w("| **L** ● | passing `null` where NumPy's default is None does not compile (a value-type parameter, or two "
          "overloads `null` fits equally well) |")
        w("| **L** ◐ | the same at C# 12 only (`[OverloadResolutionPriority]`) |")
        w("| **L** ◦ | only `Shape` parameters refuse `null`, and `default` (NumSharp's spelling of `size=None`) "
          "binds |")
        w("| **Q** ● | NumPy's stubs let a parameter take a sequence of integers (`axis=(0, 1)`, `size=(2, 3)`), and "
          "neither `new[] { 0, 1 }` nor the tuple `(0, 1)` binds there |")
    w("| — | not applicable: NumPy states no signature for the name, or NumPy has no such name |")
    w("")
    w("Matching: positional-only NumPy parameters by slot, the others by name. `*args` is offered by a `params`")
    w("array or by C# parameters at its slot. NumPy's parameters come from its runtime signature; where that signature")
    w("forwards `(*args, **kwargs)`, the parameters it hides are added from NumPy's type stubs (`.pyi`) and the")
    w("numpydoc `Parameters` section, shown as `+ name (stub)` / `+ name (doc)` under the signature. The column *missing")
    w("on NumSharp* lists the M parameters (excluded ones after `excluded:`), *NumSharp-only parameters* the X ones,")
    w("*NPTypeCode / Type* the T parameters (an NPTypeCode / Type return type, and the generic `<T>` overloads).")
    if probed:
        w("")
        w("B, L and Q come from `binding_probe.py`, which writes NumPy-style calls of every compared name and compiles")
        w("them against NumSharp at C# 12 and at the latest C#: the required parameters only (`np.sum(a)`), the optional")
        w("positional ones positionally in NumPy's order (`np.sum(a, 1, np.float64)`), each optional parameter alone by")
        w("name with an ordinary value (`np.empty(shape, device: \"x\")`) and, where NumPy's default is None, with `null`")
        w("(and `default` for a `Shape`), every NumPy parameter in one call with NumPy's defaults (and again with ordinary")
        w("values where those are None), a sequence where the stubs allow one, and the dtype as a string. A failure")
        w("already explained by a smaller call is not repeated. The column *compile probe* lists what failed, with the")
        w("compiler's reason.")
    w("")
    counts_by_module = {k: mark_counts([r for r in marked if r["module"] == k]) for k in modules}
    w("## Summary")
    w("")
    probe_headers = " B ● | B ◐ | L ● | L ◐ | L ◦ | Q |" if probed else ""
    w("| group | names with a NumPy counterpart | M ● | M ◦ only | X | O | T ● | T ◇ |" + probe_headers
      + " no mark | not compared | NumSharp-only names | of them O | of them T | NumPy names NumSharp lacks |")
    w("|---" * (14 + (6 if probed else 0)) + "|")
    keys = ["numpy", "M", "M_excluded_only", "X", "O", "T", "T_generic"]
    if probed:
        keys += ["B", "B_12", "L", "L_12", "L_default", "Q"]
    keys += ["no_mark", "not_compared", "cs_only", "cs_only_O", "cs_only_T"]
    for k in modules:
        c = counts_by_module[k]
        w(f"| `{k}` | " + " | ".join(str(c[x]) for x in keys) + f" | {len(surface[k]['np_absent'])} |")
    total = mark_counts(marked)
    w("| **total** | " + " | ".join(str(total[x]) for x in keys)
      + f" | {sum(len(surface[k]['np_absent']) for k in modules)} |")
    w("")
    if probed:
        w("*no mark*: no M (● or ◦), X, O, T, B, L or Q: the name matches NumPy in every respect the table checks.")
    else:
        w("*no mark*: no M (● or ◦), X, O or T: the name matches NumPy in all four respects.")
    w("")
    for k in modules:
        mod_rows = [r for r in marked if r["module"] == k]
        compared = sorted((r for r in mod_rows if r["numpy_kind"] != "absent"), key=lambda r: r["name"])
        only = sorted((r for r in mod_rows if r["numpy_kind"] == "absent"), key=lambda r: r["name"])
        absent = surface[k]["np_absent"]
        if not compared and not only and not absent:
            continue
        w(f"## `{k}`")
        w("")
        if compared:
            w(f"### Names NumPy has ({len(compared)})")
            w("")
            if probed:
                w("| name | M | X | O | T | B | L | Q | NumPy | missing on NumSharp | NumSharp-only parameters "
                  "| NPTypeCode / Type | compile probe |")
                w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            else:
                w("| name | M | X | O | T | NumPy | missing on NumSharp | NumSharp-only parameters | NPTypeCode / Type |")
                w("|---|---|---|---|---|---|---|---|---|")
            for r in compared:
                f = mark_flags(r)
                m = r["marks"]
                missing = ", ".join(f"`{esc(x)}`" for x in m["missing"]) if m else ""
                if m and m["missing_policy"]:
                    missing += ("<br>" if missing else "") + "excluded: " + ", ".join(f"`{x}`" for x in m["missing_policy"])
                extra = ", ".join(f"`{esc(x)}`" for x in m["extra"]) if m else ""
                numpy_text = f"`{esc(r['numpy_signature'])}`" if m else r.get("status", "")
                if r.get("numpy_added"):
                    numpy_text += "<br>+ " + ", ".join(esc(x) for x in r["numpy_added"])
                name_text = f"`{r['name']}`"
                if r.get("extension_hosts"):
                    name_text += "<br>(extension: " + ", ".join(h.rsplit(".", 1)[-1] for h in r["extension_hosts"]) + ")"
                if probed:
                    found = probe_text(r.get("probe_b", []) + r.get("probe_n", []) + r.get("probe_q", []))
                    if r.get("probe_skipped"):
                        found = "not probed: " + esc(r["probe_skipped"])
                    w(f"| {name_text} | {f['M']} | {f['X']} | {f['O']} | {f['T']} | {f['B']} | {f['L']} | {f['Q']} "
                      f"| {numpy_text} | {missing} | {extra} | {typecode_text(r)} | {found} |")
                else:
                    w(f"| {name_text} | {f['M']} | {f['X']} | {f['O']} | {f['T']} | {numpy_text} | {missing} | {extra} "
                      f"| {typecode_text(r)} |")
            w("")
        if only:
            w(f"### Names NumPy does not have ({len(only)})")
            w("")
            w("| name | O | T | NPTypeCode / Type |")
            w("|---|---|---|---|")
            for r in only:
                f = mark_flags(r)
                w(f"| `{r['name']}` | {f['O']} | {f['T']} | {typecode_text(r)} |")
            w("")
        if absent:
            w(f"### NumPy names NumSharp lacks ({len(absent)})")
            w("")
            where = surface[k].get("elsewhere", {})
            w(", ".join(f"`{n}`" + (f" (in {', '.join(f'`{x}`' for x in where[n])})" if where.get(n) else "")
                        for n in absent))
            w("")
    if modules_missing:
        w("## NumPy modules without a NumSharp facade")
        w("")
        w("NumPy's public modules (the coverage tool's list, which follows NumPy's `test_public_api.py`) that no")
        w("`[ModuleName]` type implements. Their names are absent as `np.<module>.<name>`; a name NumSharp has under")
        w("another group is noted with that group.")
        w("")
        # The loop variable must not be `path`: that is the file this function writes.
        for module_path, names, where in modules_missing:
            w(f"### `np.{module_path}` ({len(names)} names)")
            w("")
            w(", ".join(f"`{n}`" + (f" (in {', '.join(f'`{x}`' for x in where[n])})" if n in where else "")
                        for n in names) if names else "(not importable in NumPy 2.4.2)")
            w("")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def typecode_text(row):
    """Render a row's NPTypeCode / Type parameters, return types and generic dtype selectors for a table cell.

    Args:
        row: A row from ``main``.

    Returns:
        ``\\`NPTypeCode typeCode\\`, ...``, ``returns \\`NPTypeCode\\``` and ``\\`<T>(...)\\``` when present, or an
        empty string.
    """
    parts = [f"`{esc(x)}`" for x in row["typecode"]]
    if row["typecode_returns"]:
        parts.append("returns " + ", ".join(f"`{x}`" for x in row["typecode_returns"]))
    parts.extend(f"`{esc(x)}`" for x in row.get("typecode_selectors", []))
    return ", ".join(parts)


def write_csv(path, rows):
    """Write one CSV row per C# name (NumSharp-only names included), with every difference kind of the closest
    overload, the marks over all overloads (with the compile probe's findings when it ran), and every overload.

    Args:
        path: The CSV file to write (UTF-8, LF line ends).
        rows: Every row from ``main``.
    """
    fields = ["module", "name", "numpy_kind", "status", "overloads", "orpa", "generic", "numpy_signature", "best",
              *ALL_KINDS, "mark_M", "mark_X", "mark_O", "mark_T", "mark_B", "mark_L", "mark_Q", "missing_any_overload",
              "missing_excluded", "numsharp_only_params", "typecode_params", "typecode_returns",
              "typecode_generic_selectors", "numpy_params_added", "extension_hosts", "probe_binding", "probe_null",
              "probe_sequence", "all_overloads"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        # LF like the repository stores text: the csv module's default "\r\n" made git warn on every regeneration.
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            d = r["diff"] or {}
            m = r["marks"] or {}
            f = mark_flags(r)
            writer.writerow({
                "module": r["module"], "name": r["name"], "numpy_kind": r["numpy_kind"], "status": r["status"],
                "overloads": r["overloads"], "orpa": r["orpa"], "generic": r["generic"],
                "numpy_signature": r["numpy_signature"], "best": r["best"],
                **{k: "; ".join(d.get(k, [])) for k in ALL_KINDS},
                "mark_M": f["M"], "mark_X": f["X"], "mark_O": f["O"], "mark_T": f["T"], "mark_B": f["B"],
                "mark_L": f["L"], "mark_Q": f["Q"],
                "missing_any_overload": "; ".join(m.get("missing", [])),
                "missing_excluded": "; ".join(m.get("missing_policy", [])),
                "numsharp_only_params": "; ".join(m.get("extra", [])),
                "typecode_params": "; ".join(r["typecode"]),
                "typecode_returns": "; ".join(r["typecode_returns"]),
                "typecode_generic_selectors": "; ".join(r.get("typecode_selectors", [])),
                "numpy_params_added": "; ".join(r.get("numpy_added", [])),
                "extension_hosts": "; ".join(r.get("extension_hosts", [])),
                # The cell text of the marks table without its Markdown line breaks and table escapes.
                "probe_binding": probe_text(r.get("probe_b", [])).replace("<br>", "; ").replace("\\|", "|"),
                "probe_null": probe_text(r.get("probe_n", [])).replace("<br>", "; ").replace("\\|", "|"),
                "probe_sequence": probe_text(r.get("probe_q", [])).replace("<br>", "; ").replace("\\|", "|"),
                "all_overloads": " || ".join(r["cs_signatures"]),
            })


if __name__ == "__main__":
    main()
