---
paths:
  - "src/NumSharp.Core/Polynomial/Package/NDPolyBase*.cs"
  - "src/NumSharp.Core/Polynomial/Package/NDPoly{Class,Printing}.cs"
  - "src/NumSharp.Core/Polynomial/Package/ND{Polynomial,Chebyshev,Legendre,Laguerre,Hermite,HermiteE}.cs"
  - "test/NumSharp.Tests/Polynomial/Polynomial{Class,ClassPrinting,Printing}Tests.cs"
  - "test/NumSharp.Tests.Oracle/Fuzz/corpus/poly{class,print}*.jsonl"
---

# The polynomial package (4/4): the series classes and printing

## Printing helpers: `set_default_printstyle`, `polyutils.format_float`

- `polyutils.format_float(object x, bool parens = false)` (`Polynomial/Package/NDPolyPrinting.cs`, NumPy's statements):
  a non-floating x is `str(x)` (`NDPyObject.Str`: `'3'`, `'True'`, `'None'`, `'(1+2j)'`, `'[1.5]'`, an array's str, a
  0-d array too, since NumPy tests `type(x)`; a `char` is np.uint16 and prints its number); NaN / inf are the print
  options' `nanstr` / `infstr` (an infinity's SIGN is dropped: `-inf` prints `inf`); scientific from `|x| >= 1e8` or
  below `10**min(0, -(precision-1)//2)`, Python literals that NEP 50 compares in x's OWN dtype (a float16 sees 1e8 as inf
  and is never scientific from above); `floatmode='fixed'` is exact digits with trim `'k'`, every other mode the shortest
  digits with trim `'0'`; `sign='+'` only (`' '` prints nothing); `parens` wraps a scientific result only. A C# `float`
  is np.float32 here (`0.1f` prints `'0.1'`), a `Half` np.float16, a `decimal` its float64. Prints with the Dragon4 port
  (`io-printing.md`).
- `set_default_printstyle(object style)`: `style not in ('unicode', 'ascii')` under Python's `==` (a one-element U /
  StringDType array compares by its text; an empty or larger array raises NumPy's truth-value ValueError first; a number,
  None, bytes, a list or a tuple is never equal), else `ValueError("Unsupported format string '{str(style)}'. Valid
  options are 'ascii' and 'unicode'")`. ONE process-wide flag (`NDPolyPrinting.UseUnicode`, NumPy's
  `ABCPolyBase._use_unicode` class attribute), default `!OperatingSystem.IsWindows()` (NumPy's `os.name != 'nt'`); the
  series classes read it when they print. The oracle registry restores the flag after every case.
- Gates: `polyprint.jsonl` (format_float over x of every kind under every print option it reads,
  set_default_printstyle over every style kind; `set_default_printstyle` is a bare op key with its own
  `coverage/oracle_map.json` entry, since a `polynomial.` key names the submodule),
  `Polynomial/PolynomialPrintingTests.cs` (`[DoNotParallelize]`: the style is one process-wide flag).

## The series classes: `Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE`

- Types (`Polynomial/Package/`): `NDPolyBase` (abstract, the port of `ABCPolyBase`: `NDPolyBase.cs` state, members,
  lifetime; `.Protocol.cs` Python's binary-operator protocol; `.Algebra.cs` convert / cast / fit / fromroots / roots /
  integ / deriv / ...; `.Printing.cs` the four printers) and six sealed classes (`NDPolynomial`, `NDChebyshev`,
  `NDLegendre`, `NDLaguerre`, `NDHermite`, `NDHermiteE`), each with its class object (`NDPolyClass`). Every member calls
  the package's module functions; nothing has a loop of its own. `np.polynomial.Chebyshev(...)` and
  `np.polynomial.chebyshev.Chebyshev(...)` are `new NDChebyshev(...)`.
- NumPy's names, case included (gate `PolynomialClassTests.MemberNames_AreNumPys_CaseIncluded`, against NumPy 2.4.2's
  `dir(p)`): `p(x)` is `p.__call__(x)` (an array result) or `p.__call__(q)` (composition, a series); `//`, `divmod`, `**`
  are `__floordiv__` / `__rfloordiv__`, `__divmod__` / `__rdivmod__` (a `(quo, rem)` tuple), `__pow__`; `len(p)` /
  `iter(p)` are `__len__()` and `foreach` over the coefficients; `str` / `format` / `repr` are `__str__()` = `ToString()`,
  `__format__(spec)` = `IFormattable` (`$"{p:unicode}"`), `__repr__()`; `_repr_latex_()`; `__getstate__()` /
  `__setstate__(dict)` (pickling's dictionary, in NumPy's `__dict__` order). The class methods are statics
  (`NDChebyshev.fit(...)`, `fromroots`, `identity`, `basis`, `cast`, `interpolate`, Chebyshev's 55th member). The only
  other public names are C# protocol members: `Class`, `Dispose`, `Equals` (NumPy's `==`) with `==` / `!=`, `ToString`,
  `GetHashCode` (raises TypeError, `__hash__ = None`), `GetEnumerator` (`foreach`), `Deconstruct` (a full fit's
  `(series, status)`).
- The class object (`Polynomial/Package/NDPolyClass.cs`): NumPy's `Chebyshev` is `NDChebyshev.Class`, also every series'
  `__class__` (`ReferenceEquals(p.__class__, NDChebyshev.Class)` is `p.__class__ is Chebyshev`; one object per class). It
  holds the class attributes under NumPy's names (`NDChebyshev.Class.domain` / `window`: float64 `[-1, 1]`, Laguerre's
  `[0, 1]`; `maxpower` 100; `basis_name`, read-only since the classes are sealed) and `__name__` / `__qualname__` /
  `__module__` / `ToString()` (`str(cls)`, `<class 'numpy.polynomial.chebyshev.Chebyshev'>`), and it converts implicitly
  to `Type`, so `p.convert(kind: q.__class__)` ports verbatim. It is an object because C# gives a name in a type to ONE
  member: a static `NDChebyshev.domain` would hide the instance `p.domain`. Footgun: the class attributes are
  process-wide; a test that assigns one is `[DoNotParallelize]` and restores it in a `finally`.
- The operator protocol (`.Protocol.cs`, `PyBinary`): the left operand's forward method, NotImplemented on any exception
  of the coefficient arithmetic, the right operand's reflected method only when the types differ, then CPython's
  fallbacks (`can only concatenate list (not "Polynomial") to list`, `can't multiply sequence by non-int of type
  'Polynomial'`, a str's `%` formatting, ndarray's `operand 'Polynomial' does not support ufuncs (__array_ufunc__=None)`,
  `unsupported operand type(s)`). `_get_coefficients` checks types, domains, windows, symbols in NumPy's order; `/`
  divides by numbers only and `__rtruediv__` always declines; `+p` is `p` itself; `**` honours an instance or class
  `maxpower` (`power > maxpower` with NEP 50); `int()` of a NumPy scalar for `cutdeg` / `truncate` / `basis` / `**` takes
  a complex's real part and compares with the complex.
- Operator overloads, and why each exists (each class declares the set typed with itself, NDPolyBase the base set):
  forward `(series, object | NDArray | object[] | Array | string)`, reflected `(NDArray | Array | object[] | IList |
  ITuple | string, series)`. `object` alone loses to NDArray's own `(object, NDArray)` operator, and C# prefers converting
  a value to `NDArray` over `object`, so without the `Array` and `string` overloads `p + new[] { 1.0, 2.0 }` and
  `p + "ab"` would convert their operand at the call site and leave that temporary to the GC. Footguns: `p + null` is
  ambiguous (write `(object)null`, NumPy's None); `p == arr` is ambiguous between the series' and NDArray's `==` (write
  `p == (object)arr` or `p.__eq__(arr)`); a `(series, NDArray)` equality overload would make `p == null` ambiguous;
  `object o = 5; o / (dynamic)p` binds `o` as object, so bind both operands dynamically.
- Lifetimes: a series is `IDisposable` and owns its coefficients, its own domain / window and a full fit's statistics;
  getters return the stored array itself (NumPy's `p.coef is p.coef`). Assigning `coef` / `domain` / `window` stores a
  VIEW of the given array (the caller's array stays the caller's) and RELEASES the replaced one, as `poly1d` releases
  replaced coefficients: a reference read earlier is released with it (copy it to keep it); assigning the current array
  back changes nothing. `__setstate__` stores views of the state's arrays and releases what it replaces. The class
  attributes are one shared array per class (a series built without a domain reads it, NumPy's `p.domain is
  Chebyshev.domain`), holding one extra buffer reference like the module constants, so a caller's `Dispose` of
  `p.domain` cannot free it; assigning `NDChebyshev.Class.domain` stores a VIEW with such a reference (NumPy stores the
  array itself), never releases the array it replaces (series may still hold it), and refuses null. A derived series (an
  operator's result, `copy`) owns copies.
- Printing: NumPy's `_generate_string` (line breaking at the print options' linewidth, the term printers of both styles,
  the scaled-symbol argument of a mapped domain), `_repr_latex_` with NumPy's real `\text{}` scalars (format_float under
  the print options), `repr`'s `domain=` / `window=` / `symbol=` through the array printer.
- What the classes rely on below them: the symbol is Python's `str.isidentifier` (`PyUnicodeData.IsIdentifier`, the
  XID_Start / XID_Continue tables of `tools/gen_unicode_tables.py`, NFKC); a float32 series' complex roots are NumPy's
  complex64 (carried in complex128), mapped back through a float32 domain in complex64 arithmetic
  (`NDPolySeries.MapDomainWith(xIsComplex64)`, `PolyComplex64AffineComplex`); integ's str / None constants raise NumPy's
  TypeErrors (`NDPolyCalc.RefuseNonNumericConstant`); `NDPolyEval.ValAtScalar` has a float64 lane (`ValAtScalarF64`): a
  float64 series at a float64 scalar (`p(0.5)`, integ's lbnd) runs the step trees on doubles through the same cached
  scalar kernels `PolyNumber.Binary` calls (the same bits, NaN payloads included) instead of a `PolyNumber` per operand.
- Divergences (documented, not in the corpus): an object-dtype series (a Python int past uint64 among the coefficients)
  is NotSupported; `str % series` with a format conversion (a str result); a series as integ's lower bound; None or a
  `BigInteger` as a left operand; `x == p` with a non-series `x` binds that type's operator; a `__setstate__` value of
  another type than the attribute holds is a TypeError where NumPy stores anything; no RankWarning from `fit`. Pickling
  is the speed floor: `__getstate__` copies three arrays as NumPy does, `__setstate__` stores a view per array where
  NumPy assigns one `__dict__`.
- Gates: `polyclass.jsonl` (portable, NumPy's errors typed strictly by `FuzzCorpusTests.Kinds.PolyClassErrorMatches`:
  NumPy's TypeError / ValueError / ... or the house type of the same error) and `polyclass_parity.jsonl` (host-pinned
  BLAS / LAPACK-bound cases: products in np.convolve's vector regime, roots of degree two and more, fit);
  `Polynomial/PolynomialClassTests.cs` (NumPy's `test_classes.py` and `test_symbol.py` for every class, the C# pins, the
  class object, NumPy's member names) and `PolynomialClassPrintingTests.cs` (`test_printing.py`'s class part), both
  `[DoNotParallelize]`. Coverage and leaks: `coverage/object_surfaces.py` maps the six surfaces to the ND* classes (every
  member credited by its exact name); `coverage/oracle_map.json` names the constructor keys (`chebyshev.Chebyshev` → the
  class row), the `_repr_latex_` keys and the package's re-exported aliases (`numpy.polynomial.Chebyshev`); the reflection
  tools report an `[Optional]` parameter without a constant (`trim`'s `tol`, `fromroots`' `domain`) as default kind
  `missing`, which signature parity reads as NumPy's default. The leak gate's owners include the six classes, NDPolyBase
  and NDPolyClass; `LeakSurfaceCoverageTests.PolyClassMemberNames` maps the C# spellings to the tier's keys (a key that
  no longer names a live member is reported stale), and the ownership ledger watches the twelve class attributes
  (`Ownership/SharedArrays.cs`).
