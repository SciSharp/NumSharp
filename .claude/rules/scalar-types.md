---
paths:
  - "src/NumSharp.Core/DTypes/{NDScalarType,np.scalar_types}.cs"
  - "src/NumSharp.Core/APIs/{np.typing,np.finfo,np.iinfo}.cs"
  - "src/NumSharp.Core/Logic/{np.issubdtype,np.type_checks}.cs"
  - "test/NumSharp.Tests/DTypes/{NDScalarTypeTests,CTypeClassesTests}.cs"
---

# Scalar types: `NDScalarType` and `np.typing`

NumPy's scalar-type hierarchy over NumSharp's descriptors (the descriptors themselves: `dtypes.md`).

`DTypes/NDScalarType.cs`, `DTypes/np.scalar_types.cs`, `APIs/np.typing.cs`; gates `DTypes/NDScalarTypeTests.cs`,
`DTypes/CTypeClassesTests.cs`, the surface tier's scalar-type section (zero excuses) and the DType oracle's C-type classes.
- Model: `np.generic`, `np.number`, `np.integer`, `np.signedinteger`, `np.unsignedinteger`, `np.inexact`, `np.floating`,
  `np.complexfloating`, `np.flexible`, `np.character` are the ten abstract `NDScalarType` singletons (identity is what
  issubclass compares). A CONCRETE class is spelled by its descriptor (`np.float64` is a `DType`); the concrete
  `NDScalarType` behind it is reached by the implicit `DType → NDScalarType` conversion (NumPy's `dtype.type`), built once
  per `DTypeMeta` and cached in `DTypeMeta._scalarType` (compare-exchange). Parents come from the kind (NumPy's
  `SINGLE_INHERIT` table: bool / datetime64 under generic, timedelta64 under signedinteger); NumSharp's `decimal`
  (floating) and `char` (unsignedinteger) are `numsharp.decimal` / `numsharp.char`. The way back is explicit (`(DType)t`,
  `np.dtype(t)`): an abstract class raises NumPy's per-class `TypeError` from `scalarapi.c` (np.number and np.flexible
  quote other classes' names, as NumPy does), so `np.zeros(3, np.integer)` is a compile error where NumPy raises at run
  time.
- Python's built-in classes are modelled (`NDScalarType.PyObject`, `PyInt`, `PyBool` (base `int`), `PyFloat`,
  `PyComplex`, `PyStr`, `PyBytes`, `PyMemoryView`; module `builtins`), so every `__mro__` is NumPy's whole tuple by
  Python's C3 linearisation: `numpy.generic.__base__` is `object`, and NumPy's `DUAL_INHERIT` table gives
  `numpy.float64` `[floating, float]`, `numpy.complex128` `[complexfloating, complex]`, `numpy.bytes_` / `numpy.str_`
  `[bytes|str, character]` (the Python base FIRST, `DUAL_INHERIT2`, so their `__base__` is Python's). Keyed by NumPy's
  type number, not the kind: `float32`, `complex64` and the integers have no Python base. A built-in's
  `__subclasses__()` lists the modelled classes only.
- The dtype-less NumPy classes `np.complex64` (= `np.csingle`, type number 14), `np.object_` (17) and `np.void` (20) are
  concrete `NDScalarType`s in their NumPy places (`object_` last under generic, `void` under flexible): type tests,
  `np.finfo(np.complex64)` (float32's) and `np.isdtype` kinds work; `np.dtype(cls)` / `(DType)cls` raise
  `NotSupportedException`. A dtype class registered later under one of those type numbers ADOPTS the class (same
  object). Not modelled: `numpy.record`.
- The string classes: `U` is `numpy.str_` and `S` `numpy.bytes_`, both under `np.character`. A `StringDType` (`T`)
  descriptor's class is Python's own `str` (`<class 'str'>`, module `builtins`, OUTSIDE `np.generic`), which `np.str_`
  derives from, so `issubdtype('U5', StringDType())` is True and `issubdtype(StringDType(), np.character)` False;
  `np.isdtype` refuses a `StringDType` descriptor and Python's `str` with NumPy's `_preprocess_dtype` texts.
- Class surface: `__name__`, `__qualname__`, `__module__`, `__base__`, `__bases__`, `__mro__`, `mro()` (a fresh list),
  `__subclasses__()` (NumPy's registration order, abstract children first; cached per registry state through
  `DTypeRegistry.AllSnapshot`, which a registration republishes), `__instancecheck__`, `__subclasscheck__` (of a class, a
  `DType` or a C# element `Type`), `IsAbstract`, `IsBuiltin`, `ToString()` = `<class 'numpy.integer'>` (`<class
  'float'>` for a built-in).
- isinstance: NumSharp has no scalar objects, so a boxed C# value of a dtype's EXACT element type is a scalar of that
  dtype (`np.floating.__instancecheck__(1.5)` is True; Python's `1.0` is not a `numpy.floating`), and so is an UNMARKED
  0-d array (a computed 0-d result, `a[0]`, `NDArray.Scalar`); an explicit `np.array(5)` is not
  (`NDArray.IsExplicitZeroDimArray`), nor is a boxed enum. A boxed C# value is ALSO the Python object it stands for in
  the NEP 50 boundary map (`NDScalarType.PythonClassOf`: integers and `BigInteger` → `int`, `bool` → `bool`,
  `float`/`double` → `float`, `Complex` → `complex`, `string` → `str`, `byte[]` → `bytes`, `np.MemoryView` →
  `memoryview`; `Half`/`char`/`decimal` are NumPy scalars only). A 0-d array follows NumPy's MROs: a float64 scalar is a
  Python `float`, an int64 one is not an `int`, as in NumPy. Every non-null value (classes too) is an `object`.
- `np.issubdtype`'s string argument is a DTYPE string, as in NumPy (`dtype(arg).type`): `"integer"` raises
  `NotSupportedException("data type 'integer' not understood")`, and `issubdtype(np.float32, "float")` is False
  (`"float"` is float64). Write the class: `np.issubdtype(x, np.floating)`. A null argument is NumPy's `None` (float64).
  A Python built-in class is read as NumPy's `dtype(cls).type` (`NDScalarType.NumPyClass`: `int` → the pointer-sized
  int64, `float` → float64, `object` → object_, `memoryview` → void), so `issubdtype(float, np.floating)` is True and
  `issubdtype(bool, int)` False. Overloads `(NDScalarType, NDScalarType)`, `(DType, NDScalarType)`, `(NDScalarType,
  DType)`. Trap: `np.issubdtype(NPTypeCode.X, np.bool_)` is CS0121-ambiguous (`DType → NPTypeCode` is implicit, so both
  `(DType, DType)` and `(NPTypeCode, NPTypeCode)` apply); spell one side as a string or a `DType`.
- `np.isdtype` with classes compares by IDENTITY (NumPy's `dtype in processed_kinds`): `isdtype(np.int64, np.integer)` is
  False, and an abstract class is in no named kind. The mixed tuple form is `isdtype(dtype, params object[])` (names,
  `DType`s, classes, `Type`s; NumPy's TypeError text `... but is a <class 'int'>.` for anything else). The named kinds are
  NumPy's `sctypes` sets (`NDScalarType.IsInKind`): complex64 is "complex floating" / "numeric", object_ and void in
  none. A Python built-in class is no NumPy dtype: refused as the argument and as a kind (`... but it is a <class
  'type'>.`).
- `np.issctype` (NumPy 1.x's, kept on purpose: `Logic/np.type_checks.cs`) is True for every dtype with a storage lane,
  the string classes included (NumPy 1.x answered False only for `object_`); a datetime descriptor answers False (no
  storage lane), and `NDScalarType` classes are not accepted.
- `np.finfo` / `np.iinfo` of an abstract class raise NumPy's ValueErrors, which name `object_` because NumPy retries with
  the argument's type (`data type <class 'numpy.object_'> not inexact`, `Invalid integer data type 'O'.`). Of a dtype-less
  or built-in class they read NumPy's class (`finfo(np.complex64)` is float32's, `finfo(float)` float64's, `iinfo(int)`
  int64's) and refuse with its name or kind (`Invalid integer data type 'c'.`).
- Relatives: `np.typecodes` (NumPy's strings in NumPy's order, read-only); `np.sctypeDict` (NumPy's names → NumSharp's
  descriptors, `np.sctypeDict["float64"]` IS `np.float64`, in NumPy's key order for the host's C long model; the string
  names `bytes_`/`bytes`/`a` and `str_`/`str`/`unicode` map to the unsized `dtype('S')` / `dtype('<U')`; the dtype-less
  classes' names have no key, the C-type names map to their own classes, `long`/`ulong` follow the host's C long);
  `np.True_` / `np.False_` (C# bools); `np.ScalarType` (NumPy's construction: Python's seven classes, then
  `sorted(dict.fromkeys(sctypeDict.values()), key=(kind.lower(), itemsize))` stably, the dtype-less classes included -
  31 classes on both models, `intc` before `int32` on Windows, `longlong` after `int64` on Linux).
- `np.typing` (`[ModuleName("np.typing")]`): C# has no exported type aliases, so each member names the C# type that plays
  the alias's role in NumSharp's signatures: `NDArray` → `typeof(NDArray<>)` (`NumSharp.Generic`; `npt.NDArray[np.float64]`
  is `NDArray<double>`), `ArrayLike` → `typeof(NDArray)` (every array_like parameter), `DTypeLike` → `typeof(DType)` (every
  dtype parameter); `NBitBase` is a nested `[Obsolete]` abstract class (deprecated in NumPy 2.3, NumPy's warning text).
