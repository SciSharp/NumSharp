"""Inventory public NumPy object contracts separately from namespace exports.

The NumPy class/runtime instance is authoritative for member discovery. CLR matching
uses an explicit owner map; sharing a name with np or NDArray does not implement a
ufunc, MaskedArray, matrix, or random.Generator member. These rows never contribute
to the five-surface headline. No random draw, file-system IO, or global RNG mutation
is required to discover the instance-only fields in NumPy's public objects.

Ufunc protocol methods (``add.reduce``, ``maximum.accumulate``, ``add.resolve_dtypes``, ...)
are catalogued per export, from the pinned NumPy itself: the method list is read from the
``numpy.ufunc`` type, and the scan covers every namespace the generator compares, so a
ufunc exported from a submodule (``numpy.strings.str_len``) and a ufunc-like wrapper
(numpy.ma's masked ``ma.add``) are inventoried like the top-level ufuncs.

Python protocols are public API too (``a + b`` is ``ndarray.__add__``, ``with np.nditer(...)`` is
``nditer.__enter__``), so every protocol a NumPy class defines is catalogued, discovered from the
class itself (numpy_protocols). Each object surface lists the protocols it inherits from NumPy
classes, like its other members. NumPy's scalar types (numpy.generic and its 34 subclasses) and the
remaining public classes (numpy.errstate, numpy.ma.mvoid, numpy.exceptions.AxisError, ...) list
each member once, at the class that defines it (scalar_type_rows, class_member_rows): a scalar type
repeats numpy.generic's 118 members, so listing inherited members there would repeat each one 35
times. A protocol is credited when NumSharp has a member with the protocol's own name, or when C#
spells the protocol with an operator, an indexer or a cast of the same meaning, or through a
reviewed adapter whose NumSharp documentation states that it ports the protocol (protocol_credit).
A protocol slot NumPy fills only to refuse every call (``ndarray.__delitem__``) is marked
``not_applicable``, as a rejected ufunc method is.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ObjectSurface:
    """One public NumPy class whose members the catalog lists per owner, with its NumSharp counterpart.

    :ivar path: the class's dotted path relative to numpy (``ma.MaskedArray``); also the row surface.
    :ivar category: the capability category every member row of the class gets.
    :ivar clr_type: the full CLR name of the NumSharp type that implements the class (nested types
        keep ``+``), or None when NumSharp has none; members are credited only against this type.
    :ivar disposition: why the surface is out of headline scope (see DISPOSITION_NOTE).
    :ivar protocols: Python special members to list although numpy_protocols does not discover them,
        because they are attributes rather than protocol hooks (a ufunc's ``__name__`` and ``__doc__``).
    """

    path: str
    category: str
    clr_type: str | None = None
    disposition: str = "candidate"
    protocols: tuple[str, ...] = ()


OBJECT_SURFACES = (
    # ndarray's public members are headline rows (generate_coverage.public_exports emits them first, so
    # they are skipped here as already seen); this surface adds the Python protocols ndarray defines
    # (__add__, __getitem__, __array__, ...), credited against NDArray like any object member.
    ObjectSurface("ndarray", "Array protocols", "NumSharp.NDArray"),
    ObjectSurface("ufunc", "Universal function objects", disposition="subsystem",
                  protocols=("__call__", "__name__", "__doc__")),
    ObjectSurface("random.Generator", "Random generators", "NumSharp.Generator"),
    ObjectSurface("random.RandomState", "Legacy random state", "NumSharp.NumPyRandom"),
    ObjectSurface("random.SeedSequence", "Random seeding", "NumSharp.SeedSequence"),
    ObjectSurface("random.BitGenerator", "Random bit generators", "NumSharp.BitGenerator"),
    ObjectSurface("random.PCG64", "Random bit generators", "NumSharp.PCG64"),
    ObjectSurface("random.PCG64DXSM", "Random bit generators", "NumSharp.PCG64DXSM"),
    ObjectSurface("random.MT19937", "Random bit generators", "NumSharp.MT19937"),
    ObjectSurface("random.Philox", "Random bit generators", "NumSharp.Philox"),
    ObjectSurface("random.SFC64", "Random bit generators", "NumSharp.SFC64"),
    ObjectSurface("nditer", "Array iterators", "NumSharp.np+NDIterator"),
    ObjectSurface("flatiter", "Array iterators", "NumSharp.np+FlatIterator"),
    ObjectSurface("ndindex", "Array iterators", "NumSharp.np+NDIndex",
                  protocols=("__iter__", "__next__")),
    ObjectSurface("ndenumerate", "Array iterators", "NumSharp.np+NDEnumerate",
                  protocols=("__iter__", "__next__")),
    ObjectSurface("broadcast", "Array iterators", "NumSharp.np+Broadcast"),
    ObjectSurface("dtype", "Dtype objects", "NumSharp.DType"),
    ObjectSurface("finfo", "Machine limits", "NumSharp.finfo"),
    ObjectSurface("iinfo", "Machine limits", "NumSharp.iinfo"),
    ObjectSurface("ndarray.flags", "Array flags", "NumSharp.NDArrayFlags"),
    ObjectSurface("poly1d", "Polynomials", "NumSharp.poly1d"),
    # numpy.polynomial's six series classes are NumSharp's ND* series classes (one sealed class each over NDPolyBase, the
    # port of ABCPolyBase): their members, the base class's included, are credited against the concrete class.
    ObjectSurface("polynomial.polynomial.Polynomial", "Polynomial objects", "NumSharp.NDPolynomial"),
    ObjectSurface("polynomial.chebyshev.Chebyshev", "Polynomial objects", "NumSharp.NDChebyshev"),
    ObjectSurface("polynomial.legendre.Legendre", "Polynomial objects", "NumSharp.NDLegendre"),
    ObjectSurface("polynomial.hermite.Hermite", "Polynomial objects", "NumSharp.NDHermite"),
    ObjectSurface("polynomial.hermite_e.HermiteE", "Polynomial objects", "NumSharp.NDHermiteE"),
    ObjectSurface("polynomial.laguerre.Laguerre", "Polynomial objects", "NumSharp.NDLaguerre"),
    # NumSharp's numpy.ma implementation: NDMaskedArray is a standalone class (no NDArray base),
    # so every member it credits is its own masked-array code, never an inherited ndarray member.
    ObjectSurface("ma.MaskedArray", "Masked arrays", "NumSharp.NDMaskedArray"),
    ObjectSurface("matrix", "Matrix objects", "NumSharp.NDMatrix"),
    ObjectSurface("recarray", "Record arrays", disposition="subsystem"),
    ObjectSurface("record", "Record arrays", disposition="subsystem"),
    # numpy.char's chararray is NumSharp's NDCharArray (an NDArray subclass, like NDMatrix): its members, the
    # inherited ones included, are credited against it. Before generator 1.16.0 the surface had no owner, so all of
    # chararray read "missing" while NDCharArray implemented it (and the API diff gap map compared it).
    ObjectSurface("char.chararray", "String array objects", "NumSharp.NDCharArray"),
    ObjectSurface("memmap", "Memory-mapped arrays", disposition="subsystem"),
    ObjectSurface("lib.npyio.DataSource", "Data sources"),
    ObjectSurface("lib.npyio.NpzFile", "Archive objects", "NumSharp.IO.NpzFile"),
    ObjectSurface("__array_namespace_info__", "Array API introspection"),
    ObjectSurface("testing.suppress_warnings", "Test support", disposition="tooling"),
    # np.vectorize returns NumSharp.Vectorized: the object NumPy's vectorize instances correspond to.
    ObjectSurface("vectorize", "Function wrappers", "NumSharp.Vectorized",
                  protocols=("__call__",)),
    ObjectSurface("lib.NumpyVersion", "Runtime & diagnostics", disposition="tooling"),
    # Public classes whose members NumPy documents (or whose module NumPy lists as public) but which
    # had no object surface before generator 1.13.0, found by auditing the catalog against NumPy's
    # documentation index and public-module list. NumSharp implements none of them yet.
    ObjectSurface("busdaycalendar", "Date & time", disposition="subsystem"),
    ObjectSurface("ma.mrecords.MaskedRecords", "Masked record arrays", disposition="subsystem"),
    ObjectSurface("lib.Arrayterator", "Array iterators", disposition="tooling"),
    ObjectSurface("lib.user_array.container", "Array subclass helpers", disposition="tooling"),
    ObjectSurface("testing.clear_and_catch_warnings", "Test support", disposition="tooling"),
    # numpy.rec.format_parser's one public attribute, dtype, exists only on an instance (it is set in
    # __init__), so the class needs a sample like finfo does; a scan of the class object misses it.
    ObjectSurface("rec.format_parser", "Record arrays", disposition="subsystem"),
)

OBJECT_SURFACE_PATHS = tuple("numpy." + surface.path for surface in OBJECT_SURFACES)

# Reviewed spelling adapters, not case-folded matching. Method/property shape must
# still agree. Mapping-like Keys/Values properties cannot satisfy keys()/values().
MEMBER_ALIASES = {
    ("flatiter", "base"): "Base",
    ("lib.npyio.NpzFile", "files"): "Files",
    ("lib.npyio.NpzFile", "allow_pickle"): "AllowPickle",
    ("lib.npyio.NpzFile", "max_header_size"): "MaxHeaderSize",
    ("lib.npyio.NpzFile", "zip"): "Zip",
    ("lib.npyio.NpzFile", "close"): "Close",
}


# ---------------------------------------------------------------------------------------------------
# Python protocols ("dunder" members) NumPy defines.
# ---------------------------------------------------------------------------------------------------

#: Members every Python class has, or that Python and Cython write for their own bookkeeping: never NumPy
#: API, whichever NumPy class carries them.
PROTOCOL_MACHINERY = frozenset({
    # Construction and the class object itself: the class row already stands for its constructor.
    "__new__", "__init__", "__init_subclass__", "__subclasshook__", "__class__", "__module__", "__dict__",
    "__weakref__", "__slots__", "__annotations__", "__qualname__", "__name__", "__doc__", "__orig_bases__",
    "__parameters__", "__abstractmethods__", "__type_params__", "__firstlineno__", "__static_attributes__",
    # The attribute-access hooks every object implements.
    "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__dir__",
    # The garbage collector's finalizer: Python calls it, user code never does.
    "__del__",
    # A cache Python writes into a class's namespace while it runs (copyreg stores __slotnames__ the
    # first time an instance is pickled): listing it would make the catalog depend on what ran first.
    "__slotnames__",
    # Cython's implementation helpers on numpy.random's classes: the C vtable capsule and the pickling
    # shims Cython generates behind __reduce__ and __setstate__.
    "__pyx_vtable__", "__reduce_cython__", "__setstate_cython__",
})

#: Protocol names only Python 2 calls. numpy.lib.user_array.container still defines __oct__ and __hex__ (Python 3's
#: oct() and hex() call __index__ instead); Python 3 never invokes any of these, so they are not NumPy 2 API.
PYTHON2_PROTOCOLS = frozenset({
    "__div__", "__rdiv__", "__idiv__", "__nonzero__", "__long__", "__oct__", "__hex__", "__unicode__",
    "__getslice__", "__setslice__", "__delslice__", "__cmp__", "__rcmp__", "__coerce__",
})


def is_protocol_name(name: str) -> bool:
    """Tell whether a member name is a Python special name such as ``__add__``.

    :param name: a member name.
    :returns: True for a name that starts and ends with two underscores around at least one character.
    """
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def numpy_defined(cls: Any) -> bool:
    """Tell whether a class is NumPy's own rather than Python's or another package's.

    :param cls: a class (any object is accepted; a non-class reads as not NumPy-defined).
    :returns: True when the class's ``__module__`` is ``numpy`` or a ``numpy.*`` module (numpy.ndarray,
        numpy.random._generator.Generator, numpy.polynomial._polybase.ABCPolyBase).
    """
    module = getattr(cls, "__module__", None) or ""
    return inspect.isclass(cls) and (module == "numpy" or module.startswith("numpy."))


def numpy_protocols(cls: type, *, own_only: bool = False) -> tuple[str, ...]:
    """List the Python protocols NumPy defines on a class, read from the classes themselves.

    A name counts when a NumPy class (numpy_defined) declares it in its own namespace: Python's object
    machinery (PROTOCOL_MACHINERY) and Python 2 names (PYTHON2_PROTOCOLS) never count, and neither do
    protocols a class inherits from Python itself or another package (a NumPy class built on
    collections.abc.Mapping does not make Mapping's __eq__ NumPy API). A protocol whose resolved value
    is None is NumPy declaring it absent (``ndarray.__hash__ = None``: arrays are unhashable), so it is
    left out too.

    :param cls: the class to read.
    :param own_only: True lists only the protocols ``cls`` declares itself (the defining-owner rule
        scalar_type_rows and class_member_rows use); False also lists those it inherits from NumPy
        classes (the per-owner rule of object surfaces: numpy.matrix lists ndarray's __add__).
    :returns: the protocol names, sorted.
    """
    owners = (cls,) if own_only else inspect.getmro(cls)
    names: set[str] = set()
    for owner in owners:
        if not numpy_defined(owner):
            continue
        for name in vars(owner):
            if is_protocol_name(name) and name not in PROTOCOL_MACHINERY and name not in PYTHON2_PROTOCOLS:
                names.add(name)
    # getattr_static resolves through the MRO without running a descriptor, so the None check sees the
    # value the class really uses (a subclass that restores __hash__ overrides its base's None).
    return tuple(sorted(name for name in names if inspect.getattr_static(cls, name, None) is not None))


def own_public_names(cls: type) -> tuple[str, ...]:
    """List the public (non-underscore) members a class declares in its own namespace.

    :param cls: the class to read.
    :returns: the names, sorted; inherited members are not included.
    """
    return tuple(sorted(name for name in vars(cls) if not name.startswith("_")))


def member_kind(holder: Any, name: str) -> str:
    """Classify a member as a method or a property, without evaluating a descriptor.

    :param holder: the class (or a sample instance) that exposes the member.
    :param name: the member name.
    :returns: ``"method"`` for a callable (a function, a slot wrapper, a static or class method),
        ``"property"`` for everything else (a property, a getset descriptor, a plain attribute).
    """
    raw = inspect.getattr_static(holder, name)
    return "method" if callable(raw) or isinstance(raw, (staticmethod, classmethod)) else "property"


#: Python operator protocols C# spells with an operator of the same symbol, as (CLR operator method, the
#: operand that must be the owner type). "left" is the forward form (``a + x``); "right" the reflected
#: form (``x + a``), which Python calls only when ``x`` is of another type, so it needs an overload with the
#: owner on the right and another type on the left, or an owner-only overload plus an implicit conversion
#: of the left operand to the owner (``5 << a`` binds ``operator <<(NDArray, NDArray)``). Without either,
#: C# binds another type's operator: ``2.0 + p`` with only ``operator +(poly1d, poly1d)`` reaches NDArray's
#: + through poly1d's conversion to its coefficient array, which adds to every coefficient, not polyadd.
#: "either" is a comparison, which Python reflects by itself; "unary" a one-operand operator. In-place protocols
#: (__iadd__, ...) have no entry: C#'s ``a += x`` rebinds ``a`` to a new array instead of writing into the
#: one ``a`` names, so views of ``a`` would not see the result. Python operators without a C# symbol
#: (``//``, ``**``, ``@``, ``divmod``, ``abs``) have none either.
OPERATOR_PROTOCOLS: dict[str, tuple[str, str]] = {
    "__add__": ("op_Addition", "left"), "__radd__": ("op_Addition", "right"),
    "__sub__": ("op_Subtraction", "left"), "__rsub__": ("op_Subtraction", "right"),
    "__mul__": ("op_Multiply", "left"), "__rmul__": ("op_Multiply", "right"),
    "__truediv__": ("op_Division", "left"), "__rtruediv__": ("op_Division", "right"),
    "__mod__": ("op_Modulus", "left"), "__rmod__": ("op_Modulus", "right"),
    "__and__": ("op_BitwiseAnd", "left"), "__rand__": ("op_BitwiseAnd", "right"),
    "__or__": ("op_BitwiseOr", "left"), "__ror__": ("op_BitwiseOr", "right"),
    "__xor__": ("op_ExclusiveOr", "left"), "__rxor__": ("op_ExclusiveOr", "right"),
    "__lshift__": ("op_LeftShift", "left"), "__rlshift__": ("op_LeftShift", "right"),
    "__rshift__": ("op_RightShift", "left"), "__rrshift__": ("op_RightShift", "right"),
    "__eq__": ("op_Equality", "either"), "__ne__": ("op_Inequality", "either"),
    "__lt__": ("op_LessThan", "either"), "__le__": ("op_LessThanOrEqual", "either"),
    "__gt__": ("op_GreaterThan", "either"), "__ge__": ("op_GreaterThanOrEqual", "either"),
    "__neg__": ("op_UnaryNegation", "unary"), "__pos__": ("op_UnaryPlus", "unary"),
    "__invert__": ("op_OnesComplement", "unary"),
}

#: The C# symbol each operator method is declared with (``public static NDArray operator +(...)``); used
#: only to find the declaring source file.
_OPERATOR_SYMBOLS = {
    "op_Addition": "+", "op_Subtraction": "-", "op_Multiply": "*", "op_Division": "/", "op_Modulus": "%",
    "op_BitwiseAnd": "&", "op_BitwiseOr": "|", "op_ExclusiveOr": "^", "op_LeftShift": "<<",
    "op_RightShift": ">>", "op_Equality": "==", "op_Inequality": "!=", "op_LessThan": "<",
    "op_LessThanOrEqual": "<=", "op_GreaterThan": ">", "op_GreaterThanOrEqual": ">=",
    "op_UnaryNegation": "-", "op_UnaryPlus": "+", "op_OnesComplement": "~",
}

#: Python conversion protocols C# spells as a cast, with the result types that implement each (in the
#: inventory's FriendlyType spelling). ``int(a)`` is ``(long)a`` or another integral cast, ``float(a)`` a
#: floating cast, ``complex(a)`` a cast to Complex, ``bool(a)`` a cast to bool. __index__ has no entry: a
#: C# integral cast truncates a float array, which operator.index refuses.
CONVERSION_PROTOCOLS: dict[str, frozenset[str]] = {
    "__int__": frozenset({"sbyte", "byte", "short", "ushort", "int", "uint", "long", "ulong"}),
    "__float__": frozenset({"System.Half", "float", "double"}),
    "__complex__": frozenset({"System.Numerics.Complex"}),
    "__bool__": frozenset({"bool"}),
}

#: Python's subscript protocols, which C# spells with the indexer ``this[...]`` (CLR property ``Item``),
#: with the accessor the indexer must have.
INDEXER_PROTOCOLS = {"__getitem__": "get;", "__setitem__": "set;"}

#: Protocols whose C# spelling is a named member, credited only where NumSharp's own documentation of
#: that member says it ports the protocol; keyed (owner path, protocol), valued (CLR member, a signature
#: fragment one overload must contain or None, the evidence). A general rule would be wrong here:
#: NumSharp's ToString() returns str() on NDArray but repr() on finfo and poly1d, and NpzFile's
#: enumeration yields key/value pairs where Python's iter(npz) yields keys.
REVIEWED_PROTOCOL_ADAPTERS: dict[tuple[str, str], tuple[str, str | None, str]] = {
    ("ndarray", "__str__"): ("ToString", "ToString()", "NDArray.ToString() is NumPy's str() (np.array_str)."),
    ("ndarray", "__repr__"): ("ToString", "ToString(bool ",
                              "NDArray.ToString(true) is NumPy's repr() (np.array_repr)."),
    ("matrix", "__str__"): ("ToString", "ToString()", "NDMatrix inherits NDArray.ToString(), NumPy's str()."),
    ("matrix", "__repr__"): ("ToString", "ToString(bool ",
                             "NDMatrix inherits NDArray.ToString(true), NumPy's repr(), printed as matrix(...)."),
    ("ma.MaskedArray", "__str__"): ("ToString", "ToString()",
                                    "NDMaskedArray.ToString() ports MaskedArray.__str__."),
    ("ma.MaskedArray", "__repr__"): ("ToString", "ToString(bool ",
                                     "NDMaskedArray.ToString(true) ports MaskedArray.__repr__."),
    ("dtype", "__str__"): ("ToString", "ToString()", "DType.ToString() is str(dtype)."),
    ("dtype", "__repr__"): ("ToString", "ToString(bool ", "DType.ToString(true) is repr(dtype)."),
    ("dtype", "__hash__"): ("GetHashCode", None,
                            "DType.GetHashCode() follows hash(dtype): class and canonical byte order."),
    ("finfo", "__repr__"): ("ToString", "ToString()", "finfo.ToString() renders NumPy's finfo(...) repr."),
    ("iinfo", "__repr__"): ("ToString", "ToString()", "iinfo.ToString() renders NumPy's iinfo(...) repr."),
    ("poly1d", "__repr__"): ("ToString", "ToString()", "poly1d.ToString() is NumPy's repr: poly1d([...])."),
    ("poly1d", "__call__"): ("Call", None, "poly1d.Call(x) is NumPy's p(x) (poly1d.__call__)."),
    ("vectorize", "__call__"): ("Call", None, "Vectorized.Call(args) is NumPy's vfunc(*args)."),
    ("lib.npyio.NpzFile", "__repr__"): ("ToString", "ToString()", "NpzFile.ToString() is NumPy's repr."),
    ("lib.npyio.NpzFile", "__enter__"): ("Dispose", None,
                                         "C#'s using statement over the IDisposable NpzFile ports `with np.load(f)`."),
    ("lib.npyio.NpzFile", "__exit__"): ("Dispose", None,
                                        "C#'s using statement over the IDisposable NpzFile ports `with np.load(f)`."),
    ("random.Generator", "__str__"): ("ToString", "ToString()", "Generator.ToString() is str(Generator): Generator(PCG64)."),
    ("random.RandomState", "__str__"): ("ToString", "ToString()",
                                        "NumPyRandom.ToString() is str(RandomState): RandomState(MT19937)."),
    ("random.SeedSequence", "__repr__"): ("ToString", "ToString()", "SeedSequence.ToString() is NumPy's repr(SeedSequence)."),
    ("ndarray.flags", "__str__"): ("ToString", "ToString()", "NDArrayFlags.ToString() is str(a.flags)."),
    ("ndarray.flags", "__repr__"): ("ToString", "ToString()", "NDArrayFlags.ToString() is repr(a.flags), the same text."),
    ("flatiter", "__iter__"): ("GetEnumerator", None,
                               "FlatIterator enumerates in C order with the shared cursor of NumPy's iter(f) is f."),
    ("flatiter", "__next__"): ("next", None, "FlatIterator.next() is NumPy's next(f)."),
    ("ndindex", "__iter__"): ("GetEnumerator", None, "NDIndex returns itself as its iterator, NumPy's iter(i) is i."),
    ("ndenumerate", "__iter__"): ("GetEnumerator", None,
                                  "NDEnumerate returns itself as its iterator, NumPy's iter(e) is e."),
    ("broadcast", "__iter__"): ("GetEnumerator", None, "Broadcast returns itself as its iterator, NumPy's iter(b) is b."),
    ("nditer", "__iter__"): ("GetEnumerator", None,
                             "NDIterator enumerates through its live cursor, as NumPy's iteration over an nditer does."),
    ("nditer", "__enter__"): ("Dispose", None,
                              "C#'s using statement over the IDisposable NDIterator ports `with np.nditer(...) as it`."),
    ("nditer", "__exit__"): ("Dispose", None,
                             "NDIterator.Dispose (close()) resolves write-backs at the end of the block, as __exit__ does."),
}

_DELETE_REFUSED = ("NumPy fills the item-deletion slot only to refuse it: `del x[...]` raises for every "
                   "index on this type (CPython derives __delitem__ from the same C slot as __setitem__).")
_ORDERING_REFUSED = ("NumPy's comparison slot on this type supports only == and !=: every ordering "
                     "comparison returns NotImplemented, so Python raises TypeError (CPython derives all six "
                     "comparison methods from one C slot).")

#: Protocols that exist on a NumPy type only because CPython creates them from a C slot the type fills
#: for a sibling operation, and that NumPy then refuses for every call. Keyed (owner, protocol): the
#: owner is an object surface path, a canonical scalar type name or a class path. Their rows are kept
#: (the protocol is public and can be called) and marked ``not_applicable``, as a ufunc method NumPy
#: rejects is. test_object_surfaces calls every entry on a NumPy instance and asserts the refusal.
REFUSED_PROTOCOLS: dict[tuple[str, str], str] = {
    **{(owner, "__delitem__"): _DELETE_REFUSED for owner in (
        "ndarray", "matrix", "recarray", "char.chararray", "memmap", "ma.MaskedArray",
        "ma.mrecords.MaskedRecords", "record", "flatiter", "nditer", "ndarray.flags", "void")},
    **{(owner, name): _ORDERING_REFUSED for owner in ("ndarray.flags", "dtypes.StringDType")
       for name in ("__lt__", "__le__", "__gt__", "__ge__")},
}


def protocol_refusal(owner: str, name: str) -> str | None:
    """Return why NumPy refuses every call of a protocol on an owner, or None when calls can succeed.

    :param owner: an object surface path, a canonical scalar type name or a class path.
    :param name: the protocol name.
    :returns: the reason for the row's note (REFUSED_PROTOCOLS), or None.
    """
    return REFUSED_PROTOCOLS.get((owner, name))


@dataclass(frozen=True)
class ProtocolCredit:
    """How a NumSharp type implements one NumPy protocol under a C# spelling (always an alias).

    :ivar member: the CLR member that implements it (``op_Addition``, ``Item``, ``ToString``, ...).
    :ivar signatures: the matching overload signatures, for the row's ``numsharp_signatures``.
    :ivar member_entry: the inventory member entry, so a named member's source is located like any
        other member's; None for operators, casts and indexers, located through ``pattern`` instead.
    :ivar declaring_types: the CLR types that declare the matching overloads (source evidence).
    :ivar pattern: the C# declaration to search the declaring types' files for, or None.
    :ivar note: the sentence the row's note gets, naming the C# spelling.
    """

    member: str
    signatures: tuple[str, ...]
    member_entry: dict[str, Any] | None
    declaring_types: tuple[str, ...]
    pattern: re.Pattern[str] | None
    note: str


def _owner_operand(parameter_types: Sequence[str], clr_type: str, position: str) -> bool:
    """Tell whether an operator overload takes the owner type at the operand position a protocol needs.

    :param parameter_types: the overload's operand types, in the inventory's FriendlyType spelling.
    :param clr_type: the owner's CLR full name (the same spelling: nested types keep ``+``).
    :param position: ``left``, ``right``, ``either`` or ``unary`` (see OPERATOR_PROTOCOLS).
    :returns: True when the owner type itself (not a base type an inherited overload names) is there; for
        ``right``, only when the left operand is another type (the case Python's reflected call handles).
    """
    if position == "unary":
        return len(parameter_types) == 1 and parameter_types[0] == clr_type
    if len(parameter_types) != 2:
        return False
    if position == "left":
        return parameter_types[0] == clr_type
    if position == "right":
        return parameter_types[1] == clr_type and parameter_types[0] != clr_type
    return clr_type in parameter_types


def _converts_implicitly_to(operators: Mapping[str, Sequence[Mapping[str, Any]]], clr_type: str) -> bool:
    """Tell whether a type declares an implicit conversion to itself from another type.

    :param operators: the type's operators by CLR name (protocol_credit's map of the inventory entry).
    :param clr_type: the type's CLR full name.
    :returns: True when some ``op_Implicit`` returns the type from an operand of another type (NDArray's
        conversions from every C# scalar), so a value of that type can stand where the type is expected.
    """
    return any(overload["returnType"] == clr_type and overload["parameterTypes"] != [clr_type]
               for overload in operators.get("op_Implicit", []))


def _operator_pattern(op_name: str) -> re.Pattern[str]:
    """Build the C# declaration pattern of an operator or conversion method, to find its source file.

    :param op_name: a CLR operator name from OPERATOR_PROTOCOLS, or ``op_Explicit``/``op_Implicit``.
    :returns: a pattern for ``operator <symbol>(`` (one operand for a unary operator, two for a binary
        one, so unary and binary ``-`` are told apart) or for ``explicit operator``/``implicit operator``.
    """
    if op_name in {"op_Explicit", "op_Implicit"}:
        return re.compile(r"\b" + op_name[3:].lower() + r"\s+operator\b")
    symbol = re.escape(_OPERATOR_SYMBOLS[op_name])
    operands = r"\([^,()]*\)" if op_name in {"op_UnaryNegation", "op_UnaryPlus", "op_OnesComplement"} else r"\([^)]*,"
    return re.compile(r"\boperator\s*" + symbol + r"\s*" + operands)


def protocol_credit(owner: str, name: str, clr_type: str, type_inventory: Mapping[str, Any]) -> ProtocolCredit | None:
    """Find the NumSharp spelling of a NumPy protocol on the owner's CLR type, if it has one.

    Called only after an exact member of the protocol's own name was looked for (NDArray.__getitem__
    credits ndarray.__getitem__ directly). The checks run in order: a reviewed adapter for this owner
    (REVIEWED_PROTOCOL_ADAPTERS), then the general C# spellings, which hold for every type: an operator
    with the owner at the right operand (OPERATOR_PROTOCOLS), a cast from the owner to a matching type
    (CONVERSION_PROTOCOLS), and the indexer with the needed accessor (INDEXER_PROTOCOLS). An operator must
    name the owner type itself, because a base type's operator returns the base class (NDArray's + on two
    matrices would not return a matrix); a cast may be the base type's, whose scalar result is the same.

    :param owner: the NumPy owner path (``ndarray``, ``ma.MaskedArray``, ``exceptions.AxisError``).
    :param name: the protocol name.
    :param clr_type: the owner's NumSharp CLR full name.
    :param type_inventory: that type's entry of the inventory's ``objectTypes`` (schema 6: members and
        operators, inherited ones included).
    :returns: the credit, or None when NumSharp has no spelling of the protocol on this type.
    :raises ValueError: when a reviewed adapter names a member or overload the type no longer declares,
        so the reviewed table cannot silently go stale.
    """
    members = {member["name"]: member for collection in ("methods", "properties", "fields")
               for member in type_inventory.get(collection, [])}
    reviewed = REVIEWED_PROTOCOL_ADAPTERS.get((owner, name))
    if reviewed is not None:
        clr_member, fragment, evidence = reviewed
        member = members.get(clr_member)
        signatures = [signature for signature in (member or {}).get("signatures", [])
                      if fragment is None or fragment in signature]
        if not signatures:
            raise ValueError(f"Reviewed protocol adapter {owner}.{name} -> {clr_type}.{clr_member} matches no "
                             "declaration; update REVIEWED_PROTOCOL_ADAPTERS.")
        return ProtocolCredit(clr_member, tuple(signatures), member, tuple(member.get("declaringTypes", [clr_type])),
                              None, "Reviewed protocol adapter: " + evidence)

    operators = {operator["name"]: operator["overloads"] for operator in type_inventory.get("operators", [])}
    if name in OPERATOR_PROTOCOLS:
        op_name, position = OPERATOR_PROTOCOLS[name]
        candidates = operators.get(op_name, [])
        overloads = [overload for overload in candidates
                     if _owner_operand(overload["parameterTypes"], clr_type, position)]
        where = {"left": "left operand", "right": "right operand (the reflected form)",
                 "either": "an operand", "unary": "operand"}[position]
        if not overloads and position == "right" and _converts_implicitly_to(operators, clr_type):
            # `5 << a` compiles through operator <<(NDArray, NDArray): C# converts the left operand with the
            # owner's implicit conversion. Without such a conversion (poly1d) an owner-only overload does not
            # serve another left type, and C# would bind some other type's operator instead.
            overloads = [overload for overload in candidates if overload["parameterTypes"] == [clr_type, clr_type]]
            where = "right operand, the left one converted implicitly to this type (the reflected form)"
        if overloads:
            symbol = _OPERATOR_SYMBOLS[op_name]
            return ProtocolCredit(
                op_name, tuple(overload["signature"] for overload in overloads), None,
                tuple(sorted({overload["declaringType"] for overload in overloads})), _operator_pattern(op_name),
                f"C# spells this protocol with operator {symbol} ({op_name}), declared with this type as the {where}.")
    if name in CONVERSION_PROTOCOLS:
        targets = CONVERSION_PROTOCOLS[name]
        # A cast converts FROM the owner: the operand is the owner type, or the base type that declares the
        # conversion (the object inventory flattens a base type's static operators into its subclasses, and C#
        # applies a base class's conversion to a subclass: (int)matrix uses NDArray's explicit operator, with
        # the same scalar result). Unlike a binary operator, an inherited cast cannot return the wrong class.
        found = [(op_name, overload) for op_name in ("op_Explicit", "op_Implicit")
                 for overload in operators.get(op_name, [])
                 if len(overload["parameterTypes"]) == 1
                 and overload["parameterTypes"][0] in {clr_type, overload["declaringType"]}
                 and overload["returnType"] in targets]
        if found:
            op_name = found[0][0]
            return ProtocolCredit(
                op_name, tuple(overload["signature"] for _, overload in found), None,
                tuple(sorted({overload["declaringType"] for _, overload in found})), _operator_pattern(op_name),
                "C# spells this protocol as a cast from this type ("
                + ", ".join(sorted({overload["returnType"] for _, overload in found})) + ").")
    if name in INDEXER_PROTOCOLS:
        indexer = members.get("Item")
        accessor = INDEXER_PROTOCOLS[name]
        signatures = [signature for signature in (indexer or {}).get("signatures", [])
                      if "[" in signature and accessor in signature]
        if signatures:
            return ProtocolCredit(
                "Item", tuple(signatures), None, tuple(indexer.get("declaringTypes", [clr_type])),
                re.compile(r"\bthis\s*\["),
                "C# spells this protocol with the indexer this[...] ({} accessor).".format(accessor.rstrip(";")))
    return None


# The ufunc methods whose call NumPy refuses for some ufunc SHAPES (arity, output count,
# core signature). Any other method (resolve_dtypes, or one a future NumPy adds) gets no
# shape rule: resolve_dtypes resolves loop dtypes for every shape, and an unknown method
# must not inherit rules nobody reviewed for it.
_SHAPE_RULED_UFUNC_METHODS = frozenset({"reduce", "accumulate", "reduceat", "outer", "at"})

# Shared tail of every protocol row's note: why NumSharp can never match the member today.
_NO_UFUNC_OBJECT = ("NumSharp exposes NumPy's ufuncs as functions, not first-class ufunc objects, "
                    "so no object member can match; related np.sum/cumsum/outer operations do not "
                    "implement it. Catalogued outside the headline scope.")


def ufunc_protocol_methods(np: Any) -> tuple[str, ...]:
    """Return the public methods of NumPy's ufunc type, read from the pinned NumPy.

    Reading the type instead of listing names by hand keeps the per-export rows complete:
    the hand-written list held five names and missed ``resolve_dtypes`` (NumPy 1.24+), so
    ``numpy.add.resolve_dtypes`` had no row. A method a later NumPy adds appears on the next
    generation the same way. Properties (``identity``, ``nin``, ``nout``, ``nargs``, ``ntypes``,
    ``types``, ``signature``) are excluded: each is the same contract on every ufunc and is
    catalogued once as a ``numpy.ufunc.<name>`` object member.

    :param np: the imported, version-pinned ``numpy`` module.
    :returns: the method names, sorted; at NumPy 2.4.2 ``("accumulate", "at", "outer",
        "reduce", "reduceat", "resolve_dtypes")``.
    """
    # getattr_static reads the descriptor on the type: a method descriptor is callable,
    # a getset descriptor (identity, nin, ...) is not, and nothing is evaluated.
    return tuple(sorted(
        name for name in dir(np.ufunc)
        if not name.startswith("_") and callable(inspect.getattr_static(np.ufunc, name))
    ))


def ufunc_method_rejection(ufunc: Any, method: str) -> str | None:
    """Return why NumPy rejects every call of ``method`` on ``ufunc``, or None when it may succeed.

    The rules come from NumPy's ufunc method entry points (probed at 2.4.2): reductions,
    accumulations and ``outer`` need a binary ufunc; reductions, accumulations and ``at`` need a
    single output (``outer`` returns one array per output); ``at`` takes only unary or binary
    ufuncs; and no method but ``resolve_dtypes`` runs on a generalized ufunc with a core
    signature. A None result only means the shape is accepted: operands, dtypes and casting can
    still make a given call fail.

    :param ufunc: a ``numpy.ufunc`` instance.
    :param method: a ufunc method name from ``ufunc_protocol_methods``.
    :returns: the rejection text for the row's note, or None.
    """
    if method not in _SHAPE_RULED_UFUNC_METHODS:
        return None
    if ufunc.signature is not None:
        return "NumPy rejects this method for generalized ufuncs with a core signature."
    if method != "at" and ufunc.nin != 2:
        return "NumPy requires a binary ufunc for this method."
    if method != "outer" and ufunc.nout != 1:
        return "NumPy requires a single-output ufunc for this method."
    if method == "at" and ufunc.nin not in {1, 2}:
        return "NumPy at supports only unary and binary ufuncs."
    return None


def _public_names(module: Any, top_level: bool) -> list[str]:
    """List the public export names of one namespace, using the rule its catalog rows use.

    :param module: the namespace module (numpy itself or a submodule).
    :param top_level: True for numpy itself, whose historical per-ufunc rows walk ``vars(np)``.
    :returns: sorted, de-duplicated names without a leading underscore.
    """
    if top_level:
        names = vars(module)
    else:
        # Submodule rows (generate_coverage.public_exports / extended_surface_rows) come from
        # __all__ when the module declares it, else from dir(): use the same list so every
        # protocol row's owner export is itself a catalogued row.
        names = getattr(module, "__all__", None) or dir(module)
    return sorted({name for name in names if not name.startswith("_")})


def ufunc_protocol_rows(
    np: Any, seen_ids: set[str], namespaces: Sequence[tuple[str, str]], *,
    signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per ufunc method on every ufunc-like export of the given namespaces.

    Two kinds of export carry NumPy's ufunc protocol:

    * ``numpy.ufunc`` instances. Every public method of the type exists on each of them,
      whether or not their shape allows a call, so each gets one row per method; the row's
      ``applicability`` is ``not_applicable`` when NumPy rejects every call (``sqrt.reduce``)
      and ``conditional`` otherwise.
    * Wrapper objects that implement part of the protocol: numpy.ma's masked ufuncs. ``ma.add``
      has reduce, accumulate and outer; the comparisons only accumulate and outer, because
      NumPy sets their ``reduce`` to None; ``ma.maximum``/``ma.minimum`` have reduce and outer;
      the domained and unary wrappers have none. Only methods that are callable get rows.

    Every row is ``missing``: NumSharp exposes ufuncs as functions, so no NumSharp member can
    implement an object member. Rows never enter the headline scope.

    :param np: the imported, version-pinned ``numpy`` module.
    :param seen_ids: ids emitted so far; updated in place, and an id already present is skipped
        (the first emitter of an id keeps it, as everywhere in the catalog).
    :param namespaces: ``(import path relative to numpy, display surface)`` pairs. The empty path
        is numpy itself, whose rows keep the historical ``ufunc`` surface; a submodule's rows use
        its display surface (``("lib.scimath", "emath")`` yields ``numpy.emath.<name>.<method>``).
    :param signature: renders a NumPy callable's signature (generate_coverage.numpy_signature).
    :param documentation_url: resolves a documented URL or its nearest documented parent; None
        links the ufunc reference page.
    :returns: the new rows, in namespace order, then export name, then method name.
    :raises ValueError: when a namespace path does not resolve to a module, so a misspelt or
        removed submodule fails the run instead of silently dropping its rows.
    """
    methods = ufunc_protocol_methods(np)
    rows: list[dict[str, Any]] = []
    for path, surface in namespaces:
        try:
            module = _resolve(np, path) if path else np
        except (AttributeError, ImportError) as error:
            # _resolve imports numpy.<path> when the attribute walk fails; a namespace the
            # generator lists but NumPy lacks must fail the run with the namespace's name.
            raise ValueError(f"Ufunc protocol namespace numpy.{path} does not resolve.") from error
        if not inspect.ismodule(module):
            raise ValueError(f"Ufunc protocol namespace numpy.{path} is not a module.")
        for name in _public_names(module, top_level=not path):
            obj = getattr(module, name, None)
            if isinstance(obj, np.ufunc):
                implemented = methods
            elif obj is None or inspect.isclass(obj) or inspect.ismodule(obj):
                # Classes (MaskedArray, chararray) are object surfaces of their own, and a
                # module is a namespace; neither is a ufunc-like callable.
                continue
            else:
                # getattr_static sees the INSTANCE override NumPy uses to remove a method
                # (numpy.ma sets equal.reduce = None) and evaluates no property.
                implemented = tuple(method for method in methods
                                    if callable(inspect.getattr_static(obj, method, None)))
                if not implemented:
                    continue
            owner = f"numpy.{surface}.{name}" if path else f"numpy.{name}"
            for method in implemented:
                row_id = f"{owner}.{method}"
                if row_id in seen_ids:
                    continue
                seen_ids.add(row_id)
                if isinstance(obj, np.ufunc):
                    rejection = ufunc_method_rejection(obj, method)
                    if rejection:
                        note = rejection
                    elif method in _SHAPE_RULED_UFUNC_METHODS:
                        note = "NumPy permits this protocol shape; supported operands and dtype resolution still apply."
                    elif method == "resolve_dtypes":
                        note = ("NumPy resolves loop dtypes for every ufunc shape; it raises only for dtype "
                                "combinations the ufunc has no loop for.")
                    else:
                        note = ("Discovered on numpy.ufunc without a reviewed shape rule, so it is recorded "
                                "as conditional.")
                    doc_url = (documentation_url("ufunc", method, "method") if documentation_url
                               else "https://numpy.org/doc/stable/reference/ufuncs.html")
                else:
                    rejection = None
                    note = (f"{owner} is a ufunc wrapper object that implements this method; NumPy's "
                            "wrappers implement only some of the ufunc methods.")
                    # No generated page exists for a wrapper's method: link the nearest
                    # documented parent (never a fabricated URL).
                    doc_url = (documentation_url(surface, f"{name}.{method}", "method") if documentation_url
                               else "https://numpy.org/doc/stable/reference/ufuncs.html")
                rows.append({
                    "id": row_id, "origin": "numpy", "surface": surface if path else "ufunc",
                    "name": f"{name}.{method}", "kind": "method",
                    "numpy_signature": signature(getattr(obj, method), method),
                    "documentation_url": doc_url,
                    "in_default_scope": False, "extended": True, "object_surface": True,
                    "object_type": owner, "disposition": "subsystem",
                    "category": "Universal function objects", "availability": "missing",
                    "support": "missing", "status": "missing", "numsharp_target": None,
                    "numsharp_signatures": [], "numsharp_obsolete": False,
                    "numsharp_source_paths": [], "numsharp_source_urls": [],
                    "applicability": "not_applicable" if rejection else "conditional",
                    # Marks the per-export ufunc method rows for the summary's ufunc-protocol counts;
                    # a refused Python protocol carries an applicability too, so that field alone no
                    # longer identifies these rows.
                    "ufunc_protocol": True,
                    "notes": note + " " + _NO_UFUNC_OBJECT,
                })
    return rows


def _resolve(np: Any, path: str) -> Any:
    current = np
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError:
            current = importlib.import_module(current.__name__ + "." + part)
    return current


def resolve_api_id(np: Any, api_id: str) -> tuple[bool, Any]:
    """Resolve a ``numpy.*`` catalog id to the object it names, without raising.

    Walks attributes from the numpy package and imports a dotted prefix only when the attribute is
    missing (a submodule is an attribute of its package only once something imported it).

    :param np: the imported numpy module.
    :param api_id: a catalog id such as ``numpy.ma.mvoid`` or ``numpy.byte.__add__``.
    :returns: ``(True, obj)`` when every segment resolves, else ``(False, None)``.
    """
    if not api_id.startswith("numpy."):
        return False, None
    parts = api_id.split(".")
    obj: Any = np
    for index in range(1, len(parts)):
        try:
            obj = getattr(obj, parts[index])
            continue
        except Exception:
            pass
        try:
            obj = importlib.import_module(".".join(parts[: index + 1]))
        except Exception:
            return False, None
    return True, obj


def _samples(np: Any) -> tuple[dict[str, Any], Any]:
    """Instances expose finfo/iinfo and archive fields absent from class dir().

    :param np: the imported numpy module.
    :returns: sample instances by object surface path (a member set in ``__init__``, such as finfo.bits or
        format_parser.dtype, is found on them), and the open .npz archive behind the NpzFile sample, which
        the caller must close.
    """
    buffer = io.BytesIO()
    np.savez(buffer, sample=np.array([0], dtype=np.int64))
    buffer.seek(0)
    archive = np.load(buffer, allow_pickle=False)
    samples = {
        "ufunc": np.add,
        "dtype": np.dtype(np.float64),
        "finfo": np.finfo(np.float64),
        "iinfo": np.iinfo(np.int64),
        "ndarray.flags": np.array([0], dtype=np.int64).flags,
        "ndenumerate": np.ndenumerate(np.array([0], dtype=np.int64)),
        "random.SeedSequence": np.random.SeedSequence(0),
        "lib.npyio.NpzFile": archive,
        "vectorize": np.vectorize(lambda value: value),
        "lib.NumpyVersion": np.lib.NumpyVersion(np.__version__),
        # format_parser sets its public dtype attribute in __init__; any formats list builds it.
        "rec.format_parser": np.rec.format_parser(["f8", "i4"], None, None),
    }
    for spec in OBJECT_SURFACES:
        if spec.path.startswith("polynomial."):
            samples[spec.path] = _resolve(np, spec.path)([1, 0])
    return samples, archive


class _SourceIndex:
    """Locate the declaring CLR type and member; retain inherited ownership."""

    def __init__(self, root: Path) -> None:
        """Read every C# file of NumSharp.Core once.

        :param root: the repository root; files under ``obj`` and ``bin`` are skipped.
        """
        self.root = root
        self.files = [
            (path, path.read_text(encoding="utf-8-sig"))
            for path in sorted((root / "src" / "NumSharp.Core").rglob("*.cs"),
                               key=lambda path: path.as_posix())
            if not {"obj", "bin"}.intersection(path.parts)
        ]
        self.type_files: dict[str, list[tuple[Path, str]]] = {}
        # locate_credit's results by (declaring types, declaration pattern).
        self.credit_paths: dict[tuple[tuple[str, ...], str], list[str]] = {}

    def for_type(self, clr_type: str) -> list[tuple[Path, str]]:
        if clr_type not in self.type_files:
            namespace, _, nested_name = clr_type.rpartition(".")
            # The namespace and every containing type disambiguate nested CLR
            # hosts such as np.NDIterator from similarly named engine types.
            namespace_pattern = re.compile(r"\bnamespace\s+" + re.escape(namespace) + r"\b")
            patterns = [re.compile(r"\b(?:class|struct|interface)\s+" +
                                   re.escape(name.split("`")[0]) + r"\b")
                        for name in nested_name.split("+")]
            self.type_files[clr_type] = [
                (path, text) for path, text in self.files
                if namespace_pattern.search(text) and all(p.search(text) for p in patterns)
            ]
        return self.type_files[clr_type]

    def locate(self, clr_type: str, member: dict[str, Any] | None = None) -> list[str]:
        owners = member.get("declaringTypes", [clr_type]) if member else [clr_type]
        paths: set[str] = set()
        declaration = None
        if member:
            suffix = r"\s*(?:<[^>]+>)?\s*\(" if member["kind"] == "method" else r"\s*(?:=>|\{|;|=)"
            declaration = re.compile(r"\bpublic\s+[^;{}\n]*?\b@?" +
                                     re.escape(member["name"]) + suffix)
        for owner in owners:
            for path, text in self.for_type(owner):
                if declaration is None or declaration.search(text):
                    paths.add(path.relative_to(self.root).as_posix())
        return sorted(paths)

    def locate_credit(self, clr_type: str, credit: ProtocolCredit) -> list[str]:
        """Locate the source of the declaration a protocol credit names.

        :param clr_type: the owner's CLR type (used for a named member's lookup).
        :param credit: the credit from protocol_credit.
        :returns: the repository-relative files declaring it: a named member (ToString, Call) is
            located like any member; an operator, cast or indexer by its C# declaration pattern in the
            files of the types that declare the matching overloads. Empty when none is found, which the
            caller reports as missing source evidence.
        """
        if credit.member_entry is not None:
            return self.locate(clr_type, credit.member_entry)
        if credit.pattern is None:
            return []
        # Several protocols share one declaration (__add__ and __radd__ are both operator +), and a type
        # such as NDArray spans hundreds of partial files, so each (types, pattern) search runs once.
        key = (credit.declaring_types, credit.pattern.pattern)
        if key not in self.credit_paths:
            # Every pattern names the C# keyword it starts at; files without it cannot match, and a
            # substring test is far cheaper than a regex search that begins with a word boundary.
            keyword = "this" if credit.member == "Item" else "operator"
            paths: set[str] = set()
            for owner in credit.declaring_types:
                for path, text in self.for_type(owner):
                    if keyword in text and credit.pattern.search(text):
                        paths.add(path.relative_to(self.root).as_posix())
            self.credit_paths[key] = sorted(paths)
        return list(self.credit_paths[key])


@functools.lru_cache(maxsize=4)
def _source_index(root: Path) -> _SourceIndex:
    """Return the source index of a repository root, built once per process.

    Reading every .cs file of NumSharp.Core and matching each CLR type's declaring files takes seconds,
    and the object surfaces, enrich_object_classes and class_member_rows each need the same index. The
    files do not change while the generator runs.

    :param root: the repository root.
    :returns: the shared index.
    """
    return _SourceIndex(root)


def enrich_object_classes(
    rows: list[dict[str, Any]], inventory: dict[str, Any], root: Path, source_base_url: str,
) -> int:
    """Repair existing missing class rows using the reviewed object-owner map.

    Namespace exports are emitted before object members. A preexisting class row
    therefore wins duplicate-ID suppression, even when its CLR counterpart has a
    different name (numpy.flatiter -> NumSharp.np.FlatIterator). Enrich only missing
    class rows; retain their scope, identity, category and explicit support rules.

    :param rows: the catalog rows; matching class rows are updated in place.
    :param inventory: the reflected NumSharp inventory (its ``objectTypes`` map).
    :param root: the repository root, for source evidence.
    :param source_base_url: URL prefix that turns a repository path into a source link.
    :returns: how many class rows were credited.
    :raises ValueError: when the inventory has no ``objectTypes`` map, or a matched class has no source file.
    """
    object_types = inventory.get("objectTypes")
    if not isinstance(object_types, dict):
        raise ValueError("Object inventory requires schema 5's 'objectTypes' map.")
    specs = {"numpy." + spec.path: spec for spec in OBJECT_SURFACES if spec.clr_type}
    sources = _source_index(root)
    enriched = 0
    for row in rows:
        if row.get("origin") != "numpy" or row.get("kind") != "class" or row.get("status") != "missing":
            continue
        spec = specs.get(row["id"])
        if spec is None or spec.clr_type not in object_types:
            continue
        paths = sources.locate(spec.clr_type)
        target = spec.clr_type.replace("+", ".")
        if not paths:
            raise ValueError(f"Object class {row['id']} matched {target} without source evidence.")
        note = f"Class availability uses the explicit corresponding CLR object type {target}."
        row.update({
            "availability": "exact", "support": "declared", "status": "available",
            "numsharp_target": target, "numsharp_signatures": ["public class " + target],
            "numsharp_obsolete": False, "numsharp_source_paths": paths,
            "numsharp_source_urls": [source_base_url + path for path in paths],
            "notes": (row.get("notes", "") + " " + note).strip(),
        })
        enriched += 1
    return enriched


def object_surface_rows(
    np: Any, inventory: dict[str, Any], seen_ids: set[str], *, root: Path,
    source_base_url: str, signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
    ufunc_namespaces: Sequence[tuple[str, str]] = (),
) -> list[dict[str, Any]]:
    """Return complete extended coverage rows, updating ``seen_ids`` in place.

    ``objectTypes`` must come from inventory schema 5 or later. It is keyed by exact CLR
    full name and carries TypeInventory methods/properties/fields, including
    inherited public members and their ``declaringTypes`` for source evidence; schema 6
    adds each type's ``operators``, which credit Python operator protocols.

    Every object surface lists its public members (``dir`` of the class or of a sample
    instance), the special members its spec names, and every Python protocol NumPy defines
    on the class or a NumPy base class (numpy_protocols). A protocol is credited by a member
    of its own name, else by protocol_credit's C# spellings; a protocol NumPy refuses for
    every call is marked ``not_applicable`` (protocol_refusal).

    :param np: the imported, version-pinned ``numpy`` module.
    :param inventory: the NumSharp reflection inventory (schema 5 or 6).
    :param seen_ids: ids emitted so far; updated in place.
    :param root: repository root, for source-file evidence.
    :param source_base_url: URL prefix that turns a repository path into a source link.
    :param signature: renders a NumPy callable's signature.
    :param documentation_url: resolves a documented URL or its nearest parent; None uses
        stable family pages.
    :param ufunc_namespaces: submodules, besides numpy itself, whose ufunc-like exports get
        per-method protocol rows: ``(import path relative to numpy, display surface)`` pairs,
        normally every namespace the generator compares (``UFUNC_PROTOCOL_NAMESPACES``).
    :returns: the object-member rows, then the ufunc protocol rows.
    :raises ValueError: when the inventory lacks schema 5's ``objectTypes``, when a matched
        member has no source evidence, when a reviewed protocol adapter is stale, or when a
        ufunc namespace does not resolve.
    """
    object_types = inventory.get("objectTypes")
    if not isinstance(object_types, dict):
        raise ValueError("Object inventory requires schema 5's 'objectTypes' map.")
    sources = _source_index(root)
    samples, archive = _samples(np)
    rows: list[dict[str, Any]] = []

    def emit(spec: ObjectSurface, name: str, kind: str, obj: Any,
             member: dict[str, Any] | None, class_row: bool = False,
             credit: ProtocolCredit | None = None, refusal: str | None = None) -> None:
        """Append one object row unless its id was already emitted.

        :param spec: the object surface the row belongs to.
        :param name: the member name (or the class's short name for the class row).
        :param kind: ``class``, ``method`` or ``property``.
        :param obj: the NumPy object, for the signature.
        :param member: the inventory member of the protocol's or member's own name (or its MEMBER_ALIASES
            spelling), or None.
        :param class_row: True for the surface's class row.
        :param credit: a protocol's C# spelling (protocol_credit), used when ``member`` is None.
        :param refusal: why NumPy refuses every call of the protocol (protocol_refusal), or None.
        :raises ValueError: when a credited member or class has no source evidence.
        """
        row_id = "numpy." + spec.path + ("" if class_row else "." + name)
        if row_id in seen_ids:
            return
        seen_ids.add(row_id)
        matched_class = class_row and spec.clr_type in object_types
        matched = bool(member) or matched_class or credit is not None
        clr_name = member["name"] if member else credit.member if credit else name
        # A protocol credited through its C# spelling is always an alias: the NumSharp member has another
        # name (op_Addition, Item, ToString) even when it is the exact port.
        alias = matched and not class_row and (credit is not None or clr_name != name)
        target = spec.clr_type.replace("+", ".") if matched else None
        if target and not class_row:
            target += "." + clr_name
        if credit is not None:
            paths = sources.locate_credit(spec.clr_type, credit)
        else:
            paths = sources.locate(spec.clr_type, member) if matched else []
        if matched and not paths:
            raise ValueError(f"Object API {row_id} matched {target} without source evidence.")
        notes = "Public NumPy object contract, catalogued outside the headline scope."
        if member or credit:
            notes += " Availability reflects a compiled declaration; it does not assert full behavioral parity."
        elif not matched_class:
            notes += " No matching public member on the corresponding NumSharp object type."
        if spec.path == "ufunc":
            notes += " Top-level NumSharp operations do not expose NumPy's first-class ufunc object protocol."
        # Only an uncredited iteration protocol gets this note: a reviewed adapter (ndindex.__iter__ ->
        # GetEnumerator) explains its own C# spelling below.
        if name in {"__iter__", "__next__"} and not matched:
            notes += " NumSharp exposes .NET enumeration; Python protocol names are not credited as exact members."
        if credit is not None:
            notes += " " + credit.note
        elif alias:
            notes += f" Explicit spelling adapter: {name} is exposed as {clr_name}."
        if refusal:
            notes += " " + refusal
        if class_row:
            parent, _, short = spec.path.rpartition(".")
            doc_surface, doc_name = parent or "np", short
        else:
            doc_surface, doc_name = spec.path, name
        if documentation_url:
            doc_url = documentation_url(doc_surface, doc_name, kind)
        else:
            # Stable family pages, never speculative generated member URLs.
            family = "random/index.html" if spec.path.startswith("random.") else "index.html"
            doc_url = "https://numpy.org/doc/stable/reference/" + family
        row = {
            "id": row_id, "origin": "numpy", "surface": doc_surface if class_row else spec.path,
            "name": name, "kind": kind,
            "numpy_signature": signature(obj, name) if kind in {"class", "method"} else "property",
            "documentation_url": doc_url, "in_default_scope": False, "extended": True,
            "object_surface": True, "object_type": "numpy." + spec.path,
            "disposition": spec.disposition, "category": spec.category,
            "availability": "alias" if alias else "exact" if matched else "missing",
            "support": "declared" if matched else "missing",
            "status": "available" if matched else "missing",
            "numsharp_target": target,
            "numsharp_signatures": (member["signatures"] if member else list(credit.signatures) if credit
                                    else ["public class " + target] if matched_class else []),
            "numsharp_obsolete": member["obsolete"] if member else False,
            "numsharp_source_paths": paths,
            "numsharp_source_urls": [source_base_url + path for path in paths],
            "notes": notes,
        }
        if not class_row and is_protocol_name(name):
            row["python_protocol"] = True
        if refusal:
            # The ufunc protocol rows' field and value: NumPy rejects every call of this member.
            row["applicability"] = "not_applicable"
        rows.append(row)

    try:
        for spec in OBJECT_SURFACES:
            sample = samples.get(spec.path)
            cls = type(sample) if spec.path == "ndarray.flags" else _resolve(np, spec.path)
            host = sample if sample is not None else cls
            members = {
                member["name"]: member
                for collection in ("methods", "properties", "fields")
                for member in object_types.get(spec.clr_type, {}).get(collection, [])
            }
            type_inventory = object_types.get(spec.clr_type) if spec.clr_type else None
            # Constructors/classes are independently searchable, even for types
            # nested under submodules whose export list excludes class objects.
            emit(spec, spec.path.rsplit(".", 1)[-1], "class", cls, None, class_row=True)
            names = {name for name in dir(host) if not name.startswith("_")}
            names.update(name for name in spec.protocols if hasattr(cls, name))
            # The Python protocols NumPy defines on the class or a NumPy base class: `a + b`, `a[i]`,
            # `len(a)` and `with ...` are API exactly like the named members above.
            names.update(numpy_protocols(cls))
            for name in sorted(names):
                raw = inspect.getattr_static(host, name)
                # A dtype's `type` value is callable but is an attribute. Inspect
                # the descriptor, never the runtime value, to distinguish these.
                instance_field = sample is not None and name in getattr(sample, "__dict__", {})
                kind = "method" if not instance_field and (
                    callable(raw) or isinstance(raw, (staticmethod, classmethod))) else "property"
                obj = getattr(host, name) if kind == "method" else raw
                clr_name = MEMBER_ALIASES.get((spec.path, name), name)
                member = members.get(clr_name)
                if member and (member["kind"] == "method") != (kind == "method"):
                    member = None
                # A protocol without a member of its own name may still be spelled in C#: an operator,
                # a cast, the indexer, or a reviewed named member (protocol_credit). Only protocols are
                # looked up this way; a named NumPy member needs a NumSharp member of the same name.
                credit = None
                if member is None and type_inventory is not None and is_protocol_name(name):
                    credit = protocol_credit(spec.path, name, spec.clr_type, type_inventory)
                emit(spec, name, kind, obj, member, credit=credit, refusal=protocol_refusal(spec.path, name))

        # Every ufunc method on every ufunc-like export (106 top-level ufunc objects at the
        # pin, aliases included, plus submodule ufuncs and numpy.ma's masked wrappers) is
        # independently searchable (add.reduce, maximum.accumulate, add.resolve_dtypes, ...).
        # A method exists even when the ufunc's arity/core signature makes calling it
        # invalid; the rows record that distinction instead of hiding the method.
        rows.extend(ufunc_protocol_rows(
            np, seen_ids, (("", ""), *ufunc_namespaces),
            signature=signature, documentation_url=documentation_url,
        ))
    finally:
        archive.close()
    return rows


# ---------------------------------------------------------------------------------------------------
# NumPy's scalar types and the remaining public classes: members at the class that defines them.
# ---------------------------------------------------------------------------------------------------

#: NumPy's canonical scalar type names, the name NumPy's documentation gives each scalar class
#: ("Canonical name: numpy.byte"). numpy.int32, numpy.int64, numpy.intp, numpy.uint and the other sized or
#: platform names each alias one of these classes, but WHICH one depends on the platform's C long:
#: numpy.int32 is numpy.long on Windows and numpy.intc on Linux. A row named numpy.long.__add__ denotes the
#: same C type everywhere, so the artifact is identical on every platform. numpy.record, the remaining
#: scalar class, is an object surface.
CANONICAL_SCALAR_TYPES = (
    # The abstract hierarchy, root first.
    "generic", "number", "integer", "signedinteger", "unsignedinteger", "inexact", "floating",
    "complexfloating", "flexible", "character",
    # One concrete class per C type, then the flexible, object and time kinds.
    "bool", "byte", "short", "intc", "long", "longlong", "ubyte", "ushort", "uintc", "ulong", "ulonglong",
    "half", "single", "double", "longdouble", "csingle", "cdouble", "clongdouble",
    "object_", "bytes_", "str_", "void", "datetime64", "timedelta64",
)

#: The text generate_coverage.numpy_signature returns when a NumPy object exposes no signature at runtime.
SIGNATURE_UNAVAILABLE = "Signature unavailable from runtime introspection"

#: Why no NumSharp member can implement a scalar-type member (every scalar row's note ends with it).
_NO_SCALAR_OBJECT = ("NumSharp has no NumPy scalar objects: a NumPy scalar is a C# primitive or a 0-d NDArray "
                     "there, so no NumSharp member implements a scalar-type member. Catalogued outside the "
                     "headline scope.")


def scalar_type_owners(np: Any) -> list[tuple[str, type]]:
    """Resolve NumPy's canonical scalar classes and check that they are every scalar class NumPy exports.

    :param np: the imported, version-pinned numpy module.
    :returns: (canonical name, class) pairs in CANONICAL_SCALAR_TYPES order.
    :raises ValueError: when a canonical name is not a scalar class or two names share a class (the list
        no longer matches this NumPy), or when numpy exports a scalar class that is neither listed nor an
        object surface (a scalar type NumPy added); either would silently drop or merge member rows.
    """
    owners: list[tuple[str, type]] = []
    for name in CANONICAL_SCALAR_TYPES:
        cls = getattr(np, name, None)
        if not (inspect.isclass(cls) and issubclass(cls, np.generic)):
            raise ValueError(f"numpy.{name} is not a NumPy scalar type on this NumPy.")
        owners.append((name, cls))
    if len({id(cls) for _, cls in owners}) != len(owners):
        raise ValueError("Two canonical scalar type names denote the same class on this NumPy.")
    known = {id(cls) for _, cls in owners}
    # A scalar class an object surface owns (numpy.record) lists its members there, per owner.
    for spec in OBJECT_SURFACES:
        try:
            obj = _resolve(np, spec.path)
        except (AttributeError, ImportError):
            continue
        if inspect.isclass(obj) and issubclass(obj, np.generic):
            known.add(id(obj))
    unknown = sorted(name for name in getattr(np, "__all__", ())
                     if inspect.isclass(obj := getattr(np, name, None)) and issubclass(obj, np.generic)
                     and id(obj) not in known)
    if unknown:
        raise ValueError("NumPy exports scalar classes outside CANONICAL_SCALAR_TYPES: " + ", ".join(unknown))
    return owners


def scalar_type_rows(
    np: Any, seen_ids: set[str], *, signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """Return one row per member each NumPy scalar type defines, at the class that defines it.

    A scalar class's members are the public names and NumPy protocols in its own namespace
    (own_public_names, numpy_protocols with ``own_only``): numpy.generic's 118 members once, then
    what each subclass adds or redefines (each concrete numeric type defines its own arithmetic
    slots, numpy.integer adds is_integer). Listing inherited members on every one of the 34 classes
    would repeat numpy.generic's members 34 times. Rows share the ``generic`` surface (the root of
    the hierarchy) and name the owner in ``name`` (``byte.__add__``), as ufunc protocol rows share
    ``ufunc``. Every row is missing (_NO_SCALAR_OBJECT); a protocol NumPy refuses for every call is
    marked ``not_applicable``.

    :param np: the imported, version-pinned numpy module.
    :param seen_ids: ids emitted so far; updated in place, and an id already present is skipped.
    :param signature: renders a NumPy callable's signature.
    :param documentation_url: resolves a documented URL or its nearest documented parent; None links
        NumPy's scalars reference page.
    :returns: the rows, in CANONICAL_SCALAR_TYPES order, then member name. A method without a runtime
        signature on this platform takes the signature every other scalar class gives it, when they agree.
    :raises ValueError: from scalar_type_owners when the canonical list no longer matches NumPy.
    """
    rows: list[dict[str, Any]] = []
    for canonical, cls in scalar_type_owners(np):
        names = set(own_public_names(cls)) | set(numpy_protocols(cls, own_only=True))
        for name in sorted(names):
            row_id = f"numpy.{canonical}.{name}"
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            kind = member_kind(cls, name)
            refusal = protocol_refusal(canonical, name)
            notes = (f"Member of NumPy's scalar type numpy.{canonical}, catalogued at the class that defines it "
                     "(its subclasses inherit it). " + _NO_SCALAR_OBJECT)
            if refusal:
                notes += " " + refusal
            row = {
                "id": row_id, "origin": "numpy", "surface": "generic", "name": f"{canonical}.{name}",
                "kind": kind,
                # getattr on the class returns the descriptor or a bound class method; nothing is called.
                "numpy_signature": signature(getattr(cls, name), name) if kind == "method" else "property",
                "documentation_url": (documentation_url(canonical, name, kind) if documentation_url
                                      else "https://numpy.org/doc/stable/reference/arrays.scalars.html"),
                "in_default_scope": False, "extended": True, "object_surface": True,
                "object_type": "numpy." + canonical, "disposition": "subsystem", "category": "Scalar types",
                "availability": "missing", "support": "missing", "status": "missing",
                "numsharp_target": None, "numsharp_signatures": [], "numsharp_obsolete": False,
                "numsharp_source_paths": [], "numsharp_source_urls": [], "notes": notes,
            }
            if is_protocol_name(name):
                row["python_protocol"] = True
            if refusal:
                row["applicability"] = "not_applicable"
            rows.append(row)
    # NumPy documents a few C methods only on the scalar classes that have a sized name: numpy.intc.bit_count
    # has no docstring (so no signature) on Windows, where intc is the one integer class without a sized alias,
    # and does on Linux, where intc IS numpy.int32. Where every scalar class that exposes a signature for a
    # member agrees on it, the classes without one take it, so the rows read the same on every platform.
    shared: dict[str, set[str]] = {}
    for row in rows:
        if row["kind"] == "method" and row["numpy_signature"] != SIGNATURE_UNAVAILABLE:
            shared.setdefault(row["name"].rpartition(".")[2], set()).add(row["numpy_signature"])
    for row in rows:
        known = shared.get(row["name"].rpartition(".")[2], set())
        if row["numpy_signature"] == SIGNATURE_UNAVAILABLE and len(known) == 1:
            row["numpy_signature"] = next(iter(known))
    return rows


def class_member_rows(
    np: Any, catalog_rows: Iterable[Mapping[str, Any]], seen_ids: set[str], inventory: Mapping[str, Any], *,
    root: Path, source_base_url: str, signature: Callable[[Any, str], str],
    documentation_url: Callable[[str, str, str], str] | None = None,
    types_by_simple_name: Mapping[str, str] | None = None,
    category_of: Callable[[Mapping[str, Any]], str] | None = None,
    excluded: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the members of every other public NumPy class, at the class that defines them.

    Object surfaces list their classes' members per owner, and scalar_type_rows the scalar types';
    every other catalogued NumPy class (numpy.errstate, numpy.ma.mvoid, numpy.exceptions.AxisError,
    numpy.dtypes.StringDType, ...) gets a row per public name and NumPy protocol in its own namespace.
    Its subclasses' rows hold only what they add: numpy.dtypes.Float64DType adds nothing to numpy.dtype,
    whose members the dtype object surface lists. A class reached under two names (an alias) is listed
    once, under the first id in sorted order. Classes that are not NumPy's own (numpy.ctypeslib.c_intp is
    ctypes', numpy.testing.TestCase unittest's) and typing aliases (numpy.typing.NDArray) get no rows.

    A member is credited against the NumSharp type of the class's simple name, the type match that
    credits the class row (numpy.exceptions.AxisError -> NumSharp.AxisError): by a member of the same
    name, or for a protocol by protocol_credit. Matching is case-sensitive: NumSharp.AxisError.Axis does
    not credit AxisError.axis.

    :param np: the imported, version-pinned numpy module.
    :param catalog_rows: the rows emitted so far; their ``kind == "class"`` NumPy rows are the owners.
    :param seen_ids: ids emitted so far; updated in place, and an id already present is skipped.
    :param inventory: the NumSharp reflection inventory (``objectTypes`` supplies members and operators).
    :param root: repository root, for source-file evidence.
    :param source_base_url: URL prefix that turns a repository path into a source link.
    :param signature: renders a NumPy callable's signature.
    :param documentation_url: resolves a documented URL or its nearest documented parent; None links the
        NumPy reference index.
    :param types_by_simple_name: NumSharp exported types by simple name (generate_coverage's
        numsharp_types_by_simple_name); None credits nothing.
    :param category_of: the category a class's member rows get, from the class row; None keeps the
        class row's own category.
    :param excluded: tells whether an id is deliberately left out of the catalog (a NumPy 1.x
        compatibility API); its members are skipped too. None excludes nothing.
    :returns: the rows, ordered by owner id, then member name.
    :raises ValueError: when a credited member has no source evidence, or a reviewed protocol adapter
        is stale (from protocol_credit).
    """
    object_types = inventory.get("objectTypes") or {}
    types_by_simple_name = types_by_simple_name or {}
    # Classes listed elsewhere: every object surface's class (per owner) and every scalar type.
    listed: set[int] = set()
    for spec in OBJECT_SURFACES:
        try:
            obj = _resolve(np, spec.path)
        except (AttributeError, ImportError):
            continue
        if inspect.isclass(obj):
            listed.add(id(obj))
    sources: _SourceIndex | None = None
    rows: list[dict[str, Any]] = []
    for class_row in sorted((row for row in catalog_rows
                             if row.get("origin") == "numpy" and row.get("kind") == "class"),
                            key=lambda row: row["id"]):
        api_id = class_row["id"]
        if excluded is not None and excluded(api_id):
            continue
        found, cls = resolve_api_id(np, api_id)
        if (not found or not numpy_defined(cls) or issubclass(cls, np.generic)
                or id(cls) in listed):
            continue
        listed.add(id(cls))  # an alias of this class (a later id) is the same owner
        owner = api_id[len("numpy."):]
        clr_type = types_by_simple_name.get(api_id.rsplit(".", 1)[-1])
        type_inventory = object_types.get(clr_type) if clr_type else None
        members = {member["name"]: member for collection in ("methods", "properties", "fields")
                   for member in (type_inventory or {}).get(collection, [])}
        category = category_of(class_row) if category_of else class_row.get("category", "Types")
        names = set(own_public_names(cls)) | set(numpy_protocols(cls, own_only=True))
        for name in sorted(names):
            row_id = f"{api_id}.{name}"
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            kind = member_kind(cls, name)
            member = members.get(name)
            if member and (member["kind"] == "method") != (kind == "method"):
                member = None
            credit = None
            if member is None and type_inventory is not None and is_protocol_name(name):
                credit = protocol_credit(owner, name, clr_type, type_inventory)
            matched = member is not None or credit is not None
            paths: list[str] = []
            if matched:
                sources = sources or _source_index(root)  # built only when something is credited
                paths = sources.locate_credit(clr_type, credit) if credit else sources.locate(clr_type, member)
                if not paths:
                    raise ValueError(f"Class member {row_id} matched {clr_type} without source evidence.")
            refusal = protocol_refusal(owner, name)
            notes = (f"Public member of the NumPy class {api_id}, catalogued at the class that defines it "
                     "(subclasses inherit it). Catalogued outside the headline scope.")
            if matched:
                notes += " Availability reflects a compiled declaration; it does not assert full behavioral parity."
                if credit is not None:
                    notes += " " + credit.note
            elif clr_type:
                notes += f" No matching public member on the corresponding NumSharp type {clr_type}."
            else:
                notes += " NumSharp has no type corresponding to this class."
            if refusal:
                notes += " " + refusal
            target = (clr_type.replace("+", ".") + "." + (credit.member if credit else member["name"])
                      if matched else None)
            row = {
                "id": row_id, "origin": "numpy", "surface": owner, "name": name, "kind": kind,
                "numpy_signature": signature(getattr(cls, name), name) if kind == "method" else "property",
                "documentation_url": (documentation_url(owner, name, kind) if documentation_url
                                      else "https://numpy.org/doc/stable/reference/index.html"),
                "in_default_scope": False, "extended": True, "object_surface": True, "object_type": api_id,
                "disposition": class_row.get("disposition") or "candidate", "category": category,
                "availability": "alias" if credit is not None else "exact" if matched else "missing",
                "support": "declared" if matched else "missing",
                "status": "available" if matched else "missing",
                "numsharp_target": target,
                "numsharp_signatures": (member["signatures"] if member else list(credit.signatures) if credit
                                        else []),
                "numsharp_obsolete": member["obsolete"] if member else False,
                "numsharp_source_paths": paths,
                "numsharp_source_urls": [source_base_url + path for path in paths],
                "notes": notes,
            }
            if is_protocol_name(name):
                row["python_protocol"] = True
            if refusal:
                row["applicability"] = "not_applicable"
            rows.append(row)
    return rows
