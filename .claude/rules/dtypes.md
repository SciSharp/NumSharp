---
paths:
  - "src/NumSharp.Core/DTypes/**"
  - "src/NumSharp.Core/Creation/np.dtype.cs"
  - "src/NumSharp.Core/Logic/np.{find_common_type,issubdtype,isdtype,can_cast,result_type,promote_types}*.cs"
  - "src/NumSharp.Core/APIs/np.typing.cs"
  - "src/NumSharp.Core/Backends/NPTypeCode.cs"
  - "test/NumSharp.Tests/DTypes/**"
  - "test/oracle/gen_dtype_oracle.py"
---

# The dtype system

NumPy 2.x's two-level model (NEP 40/41/42/43/50/56), in `src/NumSharp.Core/DTypes/`. Gates: `test/NumSharp.Tests/DTypes/*`
and the `dtype_text` tier.

## Model

- `DTypeMeta` is the DType CLASS (`type(np.dtype('f8'))` = `np.dtypes.Float64DType`): one live object per class, with the
  NEP 42 slots as virtuals under NumPy's names (`common_dtype` (null = `NotImplemented`), `common_instance`,
  `default_descr`, `ensure_canonical`, `discover_descr_from_pyobject`, `is_known_scalar_type`, the NEP 55 reservations
  `HasReferences`/`get_clear_loop`/`get_fill_zero_loop`), the C fields `singleton`/`type_num`/`flags`, the Python class
  attributes `type`/`_abstract`/`_parametric`/`_legacy`/`_is_numeric`, and the per-class-pair `CastingImpl` table
  (`castingimpls` + `within_dtype_castingimpl`, looked up by `GetCastingImpl` = `PyArray_GetCastingImpl`; NEP 43
  `ArrayMethod`: `resolve_descriptors` + `get_strided_loop`, which no kernel calls; `casting` is the impl's minimal
  safety).
- `DType` is the INSTANCE (the descriptor, `numpy.dtype`): `Meta` + `byteorder` + the parametric payload
  (`DatetimeMetadata`), with NumPy's surface: `num`/`kind`/`char`/`itemsize`/`alignment`/`name`/`str`/`descr`/
  `isbuiltin`/`isnative`/`hasobject`/`flags`/`newbyteorder`; `ToString()` = `str(dtype)`, `ToString(true)` =
  `repr(dtype)`.
- `NPTypeCode` is the storage/kernel discriminator: every storage class exposes it as `Meta.TypeCode` and kernels
  dispatch on it. A class without storage (the datetime pair) has `TypeCode == Empty`: its descriptors parse, print,
  promote and cast-check, but converting one to `NPTypeCode`/`Type` raises `NotSupportedException` (NumSharp has no
  datetime storage).
- Classes: `LegacyBuiltinDTypeMeta` × 16 (the 15 storage types plus the vestigial `String` slot; NumPy type numbers of
  the host's C `long` model: Windows `int32 = 7 'l'`, `int64 = 9 'q'`, LP64 `int32 = 5 'i'`, `int64 = 7 'l'`;
  `Half = 23`; Decimal 256 / Char 257 are in the user-defined range, so their descriptors report `isbuiltin == 2` and repr
  `dtype(decimal)`), `DatetimeDTypeMeta` × 2 (`DateTime64DType` 21 `'M'`, `TimeDelta64DType` 22 `'m'`, parametric),
  `PyScalarDTypeMeta` × 3 (`_PyLongDType`/`_PyFloatDType`/`_PyComplexDType`, abstract: the NEP 50 weak literals).
- `DTypeRegistry` owns them (`FromTypeCode`/`FromScalarType`/`FromTypeNum`/`FromName`, `Register` for further classes,
  casting-impl registration). Lookups take no lock (`ConcurrentDictionary` tables, a `Volatile` code table, `All` copies
  an array that each registration replaces); only `Register` locks, and it checks every key before its first write, so a
  refused class leaves no trace. `np.dtypes` is the NumPy-named facade with NumPy's aliases (`ByteDType`, `IntDType`, the
  platform `LongDType`, `LongLongDType`, `LongDoubleDType` → Float64, …).

## Promotion and casting

- `DTypePromotion` is `common_dtype.c`/`convert_datatype.c`: `CommonDType` (the two-sided protocol, then
  `DTypePromotionError`, a `TypeError`, with NumPy's text), `PromoteDTypeSequence` (`reduce_dtypes_to_most_knowledgeable`,
  order-independent), `PromoteTypes` (the same native descriptor object → itself, NumPy's pointer test; else the common
  class → its default descriptor, or for a parametric class both operands cast to it and `common_instance` taken),
  `CastDescrToDType`, and `ResultType` = NEP 50: every `NDArray` (0-d too), `DType`, `NPTypeCode`, `Type`, dtype string,
  `bool`/`char`/`Half`/`decimal` is STRONG; C# integer / `float` / `double` / `Complex` literals are WEAK
  (`result_type(int8_dtype, 300)` → int8, `+ 1.5` → float64; a lone literal → int64/float64/complex128).
- For the storage builtins `CommonDType` is the frozen `np._nptypemap_arr_arr` table (deferring to the larger type
  number, as NumPy does). `np._FindCommonType(NDArray, NDArray)` (binary-operator promotion),
  `NDExprTypeRules.PromoteStrong` and `NDIterCasting.PromoteTypes` are separate engines; the LIST forms of
  `np._FindCommonType` are `np.find_common_type`'s NumPy 1.26.4 port (below).
- `DTypeCasting` is `PyArray_CanCastTypeTo`/`CheckCastSafety`/`GetCastInfo`/`EquivTypes`/`MinCastSafety` over the impl
  table. The builtin impls: same class → `equiv`; `promote(from, to) == to` → `safe`; kind order `b<u<i<f<c` →
  `same_kind`; else `unsafe` (Char↔UInt16 are distinct classes, never `equiv`). The table also expresses byte order
  (`can_cast('>i4', 'i4', 'equiv')` True, `'no'` False) and the datetime unit rules (`time_to_time`: exact 10³ᵏ folds are
  no-casts, generic → concrete `safe`, concrete → generic `unsafe`, the timedelta years/months barrier `unsafe`,
  finer-and-divides `safe`, else `same_kind`; numeric → `m8` is `safe` for bool and integers except 64-bit unsigned
  (`same_kind`), floats `unsafe`; anything ↔ `M8` `unsafe`). Casting strings are case-sensitive, with NumPy's `casting
  must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')` ValueError.

## `np.dtype(string)`

A port of `_convert_from_str`:
- a comma string or sub-array is unsupported; the byte-order character is consumed (`'|'` reads native; a non-native
  prefix is KEPT: `np.dtype(">i4")` has `byteorder '>'`, `isnative false`, `str ">i4"`, `repr dtype('>i4')`,
  `isbuiltin 0`); an explicit host byte order is folded into `'='` (`np.dtype("<f8")` is the float64 singleton);
- datetime typestrs (`M8`, `m8`, `M`, `m`, `datetime64[10ns]`, `timedelta64[s/2]` → `m8[500ms]`, the divisor → multiple
  table including NumPy's `M8[W/11]` → `M8[0Y]`) through `DatetimeMetaData.Parse`, with `Invalid datetime metadata
  string "[5]" at position 2` / `Invalid datetime unit in metadata string "[5x]"` / `divisor (7) is not a multiple of a
  lower-unit …` (a zero divisor raises where CPython crashes);
- one-character codes (the `'n'`/`'N'` intp letters included; a lone control character below `'\x18'` is a TYPE NUMBER
  to `PyArray_DescrFromType`: `'\x0c'` is float64); `kind+size` parsed with `strtol` semantics (`i04`, `i 4`, `i+4` are
  int32; C whitespace only, so `'f\xa08'` is invalid; `b1` is bool; `i3`/`f16`/`?1` are invalid);
- then the name table, looked up by the WHOLE string as NumPy's `typeDict` is, so a byte-order prefix before a NAME is
  "not understood" (`'<float64'`, `'|bool'`) while it is a byte order before a code (`'<f8'`, `'>d'`).
- Invalid strings raise `NotSupportedException` with NumPy's `data type %R not understood` / `Alias %R was removed in
  NumPy 2.0. Use a name without a digit at the end.` wording, quoted by Python's `repr` (`data type "a'b" not
  understood`); a valid NumPy dtype of a class NumSharp lacks (complex64, `S`/`U`/`V`/`O`, `T`, `O4`/`O8`, `csingle`)
  says "is not supported by NumSharp". The 24 C# spellings (`Int32`, `Single`, `Decimal`, …) are NumSharp's superset,
  asserted refused by NumPy.
- Builtins are singletons (`np.dtype("i8")` is the same object every call, `isbuiltin 1`), and so are the datetime
  letters (`np.dtype("M")`); datetime typestrs (`M8`, `M8[ns]`) and non-native descriptors are fresh.
- `DType` equality is STRUCTURAL (class + canonical byte order + unit: `M8[ns] != M8[s]`, `M8[1s] == M8[s]`,
  `>i4 != i4`, an explicit `'<'` equals `'='` on a little-endian host); the hash is class + canonical byte order, so
  `hash(M8[ns]) == hash(M8[s])` as in NumPy. `Equals(object/string/Type/NPTypeCode)` coerces like `dtype.__eq__`
  (`d.Equals("garbage")` is false, never an error). `<`/`<=`/`>`/`>=` are NumPy's safe-cast ordering.
- Null is NumPy's `None` wherever NumPy runs `PyArray_DescrConverter` (`_convert_from_any`'s first test,
  `PyArray_DescrFromType(NPY_DEFAULT_TYPE)`): every `np.dtype` overload (`np.dtype((DType)null)` is the float64
  singleton; `copy: true` / `metadata:` then apply as for any argument) and every `np.result_type` operand (a STRONG
  float64: `np.result_type((DType)null, np.int8)` is float64), through one rule, `np.DefaultTypeDescr`. Not in
  `np.promote_types` / `np.can_cast` (NumPy's `DescrConverter2` refuses None with TypeErrors, ported), not in equality
  (`descr == null` is the engine's null test, so `DType.Double.Equals(null)` is false where NumPy's `np.dtype('f8') ==
  None` is True), and not in a `DType dtype = null` PARAMETER (that means "not given"; each function picks its default).
  A bare `null` literal is CS0121 between `np.dtype(string)` and `np.dtype(Type)`: write it typed.

## Traps (pinned by tests)

1. Never add `==(DType, string)` operators: a `null` literal binds them over `==(DType, DType)`, and every
   `descr == null` inside the engine silently answers false. The coercing string comparison lives on `Equals(string)`.
2. C# scalar literals bind NumSharp's scalar → `NDArray` conversion before `params object[]`, so
   `np.result_type(int8_array, 300)` reaches the STRONG `(NDArray, NDArray)` overload; the weak rule is reachable through
   the dtype: `np.result_type(a.dtype, 300)`. A dtype STRING likewise binds string → `NDArray` (a character array) and is
   ambiguous against `params DType[]`, which is why `result_type(params string[])`, `can_cast(string, DType, …)`,
   `issubdtype(string, string)` and `isdtype(string, …)` exist.
3. `promote_types('m8[Y]', 'M8[D]')` is `M8[D]`: `PromoteTypes` casts the timedelta operand to the DATETIME class first
   (keeping its unit) and only then runs the GCD, so the strict years/months barrier bites on `m8 × m8` only.
4. NumPy's `dtype.__eq__` (`PyArray_EquivTypes`) is asymmetric for the metric-prefix folds (`M8[1000ms] == M8[s]` but
   not the reverse); `DType.Equals` stays structural, and the NumPy relation is `DTypeCasting.EquivTypes`
   (`[Misaligned]`).
5. `result_type(f4, 1j)` is complex128 (NumPy: complex64): NumSharp has one complex width (`[Misaligned]`).
6. The legacy `DType._kind_list_map` (`'?'` for bool, `'S'` for char) is what `np._FindCommonType` orders kinds through
   and must not move; the descriptor's own `kind` is `'b'`/`'u'`.

## NumPy's names, case for case

Every member of `DType`, `DTypeMeta` and `DTypeFlags` that NumPy names keeps NumPy's spelling: Python attributes as
Python has them (`dtype.isalignedstruct`, `DType._abstract`, `DType.type`), C fields and NEP 42 slots as `dtype_api.h` /
`dtypemeta.h` have them (`singleton`, `type_num`, `default_descr()`, `within_dtype_castingimpl`, `castingimpls`), C
macros as `NPY_*` (`DTypeFlags.NPY_DT_LEGACY`…, the byte-order characters `DType.NPY_NATIVE`/`NPY_IGNORE`/`NPY_SWAP`/
`NPY_LITTLE`/`NPY_BIG`/`NPY_NATBYTE`/`NPY_OPPBYTE`), and C function names keep NumPy's PascalCase after the `PyArray_`
prefix (`GetCastingImpl`, `DTypePromotion.CommonDType`). PascalCase members (`Name`, `ModulePath`, `TypeCode`, `Kind`,
`Instantiate()` for the class call, `Meta`, `DatetimeMetadata`) have no NumPy name.
- The other ports follow the same rule: `ArrayMethod` has the fields of `PyArrayMethodObject` and
  `PyBoundArrayMethodObject` (`name`, `nin`, `nout`, `casting`, `flags`, `dtypes`, `resolve_descriptors()`,
  `get_strided_loop()`), `ArrayMethodContext` those of `PyArrayMethod_Context` (`method`, `descriptors`),
  `DatetimeMetaData` those of `PyArray_DatetimeMetaData` (`@base`, `num`); `CastingImpl.From`/`To`/`IsWithinDType` and
  the static helpers that port NumPy-internal C functions (`DatetimeMetaData.Divides`, `DTypeCasting.ParseCasting`, …)
  keep their names.
- Ported members: `isalignedstruct` (False), `__len__()` (0 fields), `__getitem__(key)` (NumPy's KeyError, quoted like
  Python's `str(KeyError)`), `metadata` and `np.dtype(dtype, align, copy, metadata)` (`copy` copies only a singleton;
  `metadata` always copies and merges without overwriting existing keys; equality, hash and printing ignore it).
- `newbyteorder` is `PyArray_DescrNewByteorder` (a requested host order stays EXPLICIT: `np.dtype('>i4').newbyteorder()`
  is `dtype('<i4')`; errors quote the whole string; `None` is TypeError); `isnative` is `PyArray_ISNBO`; `repr` takes the
  short form whenever the stored byte order is explicit; `ensure_canonical` returns a FRESH native copy, never the
  singleton (`np.result_type(np.dtype('>f8')).isbuiltin` is 0); `promote_types`' fast path is pointer identity.
- Gate: `test/oracle/gen_dtype_oracle.py` → `test/NumSharp.Tests/DTypes/corpus/dtype_oracle.{llp64,lp64}.jsonl` (NumPy's
  member names including the C slots parsed from the headers, the C enums of `ndarraytypes.h`, the fields of the structs
  `ArrayMethod`, `ArrayMethodContext` and `DatetimeMetaData` port (`StructPorts`, each unported field listed with its
  reason), descriptors × every attribute + str / repr / len / `d[0]`, errors, comparison pairs, promotions, classes,
  casting resolutions, dtype strings), one file per C `long` model (Windows NumPy / Linux NumPy under WSL: regenerate
  both), replayed by `DTypes/DTypeOracleTests.cs`, which picks the host's model. Exemptions are listed with reasons and
  retire themselves (each must stay absent or divergent): the dunders C# has no use for (`__bool__`, pickle, typing,
  `__mul__`/`__rmul__` sub-arrays), the unported slots (`setitem`, `getitem`, `finalize_descr`, `get_constant`,
  ArrFuncs, `sort_meth`/`argsort_meth`), the absent classes (`Complex64DType`, `StrDType`, `BytesDType`, `StringDType`,
  `ObjectDType`, `VoidDType`), the C-integer / long-double classes NumPy keeps apart and NumSharp collapses, the datetime
  `type`, and the asymmetric `M8[1000ms] == M8[s]` (trap 4).

## C enums, type characters, `np.find_common_type`

- `NPY_TYPECHAR` (`Creation/np.dtype.cs`) is NumPy 2.4.2's `ndarraytypes.h` enum member for member: `NPY_VSTRINGLTR =
  'T'` (NEP 55), `NPY_DEPRECATED_STRINGLTR2 = 'a'`, and no `NPY_INTPLTR` / `NPY_UINTPLTR` (intp is `'n'`; `'p'` / `'P'` /
  `'n'` / `'N'` parse but have no member: cast the character, `(NPY_TYPECHAR)'n'`). The kind letters share values with
  type characters as in C (`'b'` is the bool kind and int8, `'c'` the complex kind and NumPy's `S1`): compare values, never
  names (`ToString()` of a shared value is unspecified).
- `NPTypeCode.ToTypeCode(NPY_TYPECHAR)` IS `np.dtype(c).GetTypeCode()`: `'D'` is complex128, `'F'` is refused, `'l'`
  follows the C `long` model, a control character below `'\x18'` is a type number. `ToTYPECHAR` is the lane's descriptor
  character (`'D'` for complex128, never the kind letter `'c'`). NumSharp's decimal and char have no NumPy character and
  REFUSE (`NotSupportedException`): their descriptors' own `char` borrows other dtypes' letters (decimal `'q'`, char
  `'c'`).
- The header gate: the oracle's `members` record carries every C enum of `ndarraytypes.h` (`c_enums`, parsed from
  `refs/numpy`), and `DTypeOracleTests.CEnums_MirrorNumPysHeader` holds the mirrored ones (`NPY_TYPES`, `NPY_TYPECHAR`,
  `NPY_SCALARKIND`, `NPY_ORDER`, `NPY_CASTING`, `NPY_DATETIMEUNIT`) to NumPy's names, values and order. Exemptions:
  NumSharp's `NPY_ORDER` numbering (K = 0 … A = 3 where NumPy has A = -1 … K = 2: renumbering would change what
  `default(NPY_ORDER)` means), NumPy 2.4's `NPY_SAME_VALUE_CASTING` and `_NPY_ERROR_OCCURRED_IN_CAST`, and the enums
  NumSharp takes as strings (`NPY_SORTKIND`, `NPY_SELECTKIND`, `NPY_SEARCHSIDE`, `NPY_CLIPMODE`, `NPY_CORRELATEMODE`) or
  lacks (`NPY_BUSDAY_ROLL`).
- `np.find_common_type` is NumPy 1.26.4's (`Logic/np.find_common_type.cs`; deprecated in NumPy 1.25, removed in 2.0).
  Each list's maximum is the FIRST candidate of `?bBhHiIlLqQefdgFDGO` every entry casts to safely (a search, not a
  promotion: `[i1, u1, f2]` is float16 where a left-to-right `promote_types` chain gives float32); a scalar type of the
  same or a lower kind never moves it (`([i1], [i8])` is int8, `np.result_type(i1, i8)` int64); a scalar maximum of a
  HIGHER kind goes to `_find_common_coerce` (`find_common_type([u1], [i2])` is int16). A list holding decimal or char
  promotes by NumSharp's table. Gate: `Logic/corpus/find_common_type.numpy1264.jsonl` (from a numpy==1.26.4 environment,
  `test/oracle/gen_find_common_type_oracle.py`).
- `np.asanyarray(object[])` of mixed scalars promotes element by element like NumPy's `handle_promotion`, so the result
  follows element ORDER (`[i1, u1, f2]` is float32, `[f2, i1, u1]` float16), and a float16 among other types converts.
  Gate: `Creation/corpus/asanyarray_mixed.numpy242.jsonl` (`test/oracle/gen_asanyarray_mixed_oracle.py`).

## `DType` is the one dtype spelling

- `NDArray.dtype` returns `DType` (`Storage.Descr`, NumPy's `PyArrayObject.descr`): the class singleton for a builtin
  lane, so `a.dtype == b.dtype` / `a.dtype == np.float64` are cheap structural compares and
  `ReferenceEquals(a.dtype, np.float64)` holds; a parametric descriptor is carried explicitly through
  `Alias`/`Clone`/`ReplaceData` and reset when the lane changes.
- The `np.float64` family (`np.bool_` … `np.clongdouble`, `np.@decimal`, `np.intp` / `np.@long` / `np.@uint`, the
  throwing `complex64` / `csingle` / `chars`) are `DType` descriptors, not `Type`s: NumSharp has no scalar-type objects
  (a C# `double` IS the scalar), so the descriptor is the one currency, and `np.float64.itemsize` / `.name` / `.kind`
  read as in NumPy.
- Every `dtype`/`typeCode` parameter across `np.*`/NDArray takes one `DType` (creation, IO, `frombuffer`, `indices`,
  `trace`, `clip`, `cumsum`/`cumprod`/`prod`/`sum`, `nancumsum`/`nancumprod`, `amax`/`amin`/`std`/`var`,
  `cov`/`corrcoef`, `getfield`/`setfield`, `finfo`/`iinfo`, Generator/`randint`/`uniform`, `concatenate`/`concat`,
  `matmul`/`vecdot`/`matvec`/`vecmat`/`einsum` (`GufuncGuard.ToLoop`), `astype`/`view`, the `NDArray(DType, …)`
  constructors, `isdtype`, `find_common_type(DType[]…)`); a `Type`, `NPTypeCode`, `NPTypeCode?` or dtype string converts
  implicitly, so `np.zeros(3, typeof(int))`, `np.zeros(3, NPTypeCode.Int32)`, `np.zeros(3, "i4")` and
  `np.zeros(3, np.int32)` bind the one overload.
- `Type` remains where it means the concrete element type of a storage lane: `TensorEngine.GetStorage(Type)` /
  `Cast(…, Type, …)`, `UnmanagedStorage.Allocate/Cast/AliasAs(Type)`, `NDArray.ReplaceData`, `UnmanagedStorage.DType`
  (the CLR type; `Descr` is the descriptor), plus the engine-internal `NPTypeCode` currency
  (`DefaultEngine.GetStorage(NPTypeCode)`, the `finfo(NPTypeCode)` constructor, `NpyFormat.DtypeInfo`). The promotion
  functions return `NPTypeCode` for typed pairs and `DType` for their `DType` overloads.
- Design constraints, each pinned by a test:
  1. `DType → Type` is an EXPLICIT conversion (`(Type)a.dtype` or NumPy's `a.dtype.type`); `Type → DType` is implicit.
     `System.Type` declares its own `==`, so with conversions in both directions `a.dtype == typeof(double)` would be
     CS0034-ambiguous, and `==(DType, Type)` overloads would make `descr == null` ambiguous. One-way implicit resolves
     `a.dtype == typeof(double)`, `typeof(double) == a.dtype`, `a.dtype == "f8"` and `a.dtype == NPTypeCode.Double` to
     the structural, coercing `DType.==`, and `Assert.AreEqual(typeof(double), a.dtype)` infers `T = DType`
     (`DTypeDescriptorTests.TypeConversion_IsExplicit_SoTypeEqualityResolvesToDType`). A `Type` parameter fed with
     `.dtype` needs `.dtype.type` or `(Type)` (`Convert.ChangeType`, kernel-factory lookups, `Type.Name` in messages →
     `dtype.type.Name`).
  2. Static initialization: `np`'s field initializers run `DType.From` → `DTypeRegistry`'s cctor, which registers the
     15 × 15 `BuiltinCastingImpl`s whose minimal `casting` reads `np._nptypemap_arr_arr`, a table `static np()` fills
     after the field initializers. `ArrayMethod.casting` is therefore `virtual`, and `BuiltinCastingImpl` computes it on
     first read (both orders, np first and registry first, are tested). The `static np()` tables are built as
     `Dictionary<(DType, DType), DType>` and the `Type`/`NPTypeCode`-keyed maps derived from them.
  3. `isdtype` has only the NumPy-exact `DType` overloads (NumPy rejects `issubdtype`'s vocabulary: `kind argument is a
     string, but 'floating' is not a known kind name.`, ValueError); the `NDArray` convenience overloads route through
     `arr.dtype`.
  4. Tests: `test/NumSharp.Tests/Utilities/DTypeAssertions.cs` adds `Should(this DType)` → `DTypeAssertions`
     (`Be(DType)` (a `Type`/`NPTypeCode`/string converts in), `Be<T>()`, `NotBe`, `BeOneOf`) in the `NumSharp.Tests`
     namespace, so enclosing-namespace lookup binds every `x.dtype.Should()` ahead of AwesomeAssertions'
     `Should(this object)`.
- Consequences for callers: `Type t = a.dtype` and `Type t = np.float64` need a cast; `a.dtype.Name` is
  `a.dtype.name` (NumPy) or `a.dtype.type.Name` (C#); `np.X` where a `Type` is required (`Marshal.SizeOf(np.float64)`) is
  `np.float64.type`; reflection over `np.array(Array, Type, …)` uses `typeof(DType)`; `np.isdtype(x, "floating")` raises.

## Scalar types: `NDScalarType` and `np.typing`

`DTypes/NDScalarType.cs`, `DTypes/np.scalar_types.cs`, `APIs/np.typing.cs`; gates `DTypes/NDScalarTypeTests.cs` and the
surface tier's scalar-type section (zero excuses).
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
- Class surface: `__name__`, `__qualname__`, `__module__`, `__base__`, `__bases__`, `__mro__`, `mro()` (a fresh list),
  `__subclasses__()` (NumPy's registration order, abstract children first; cached per registry state through
  `DTypeRegistry.AllSnapshot`, which a registration republishes), `__instancecheck__`, `__subclasscheck__` (of a class, a
  `DType` or a C# element `Type`), `IsAbstract`, `ToString()` = `<class 'numpy.integer'>`. Python's built-in bases are not
  modelled: MROs end at generic, and `np.generic.__base__` is null (NumPy: `object`).
- isinstance: NumSharp has no scalar objects, so a boxed C# value of a dtype's EXACT element type is a scalar of that
  dtype (`np.floating.__instancecheck__(1.5)` is True; Python's `1.0` is not a `numpy.floating`), and so is an UNMARKED
  0-d array (a computed 0-d result, `a[0]`, `NDArray.Scalar`); an explicit `np.array(5)` is not
  (`NDArray.IsExplicitZeroDimArray`), nor is a boxed enum.
- `np.issubdtype`'s string argument is a DTYPE string, as in NumPy (`dtype(arg).type`): `"integer"` raises
  `NotSupportedException("data type 'integer' not understood")`, and `issubdtype(np.float32, "float")` is False
  (`"float"` is float64). Write the class: `np.issubdtype(x, np.floating)`. A null argument is NumPy's `None` (float64).
  Overloads `(NDScalarType, NDScalarType)`, `(DType, NDScalarType)`, `(NDScalarType, DType)`. Trap:
  `np.issubdtype(NPTypeCode.X, np.bool_)` is CS0121-ambiguous (`DType → NPTypeCode` is implicit, so both
  `(DType, DType)` and `(NPTypeCode, NPTypeCode)` apply); spell one side as a string or a `DType`.
- `np.isdtype` with classes compares by IDENTITY (NumPy's `dtype in processed_kinds`): `isdtype(np.int64, np.integer)` is
  False, and an abstract class is in no named kind. The mixed tuple form is `isdtype(dtype, params object[])` (names,
  `DType`s, classes, `Type`s; NumPy's TypeError text `... but is a <class 'int'>.` for anything else).
- `np.finfo` / `np.iinfo` of an abstract class raise NumPy's ValueErrors, which name `object_` because NumPy retries with
  the argument's type (`data type <class 'numpy.object_'> not inexact`, `Invalid integer data type 'O'.`).
- Relatives: `np.typecodes` (NumPy's strings in NumPy's order, read-only); `np.sctypeDict` (NumPy's names → NumSharp's
  descriptors, `np.sctypeDict["float64"]` IS `np.float64`; absent classes have no key, collapsed names map onto the class
  they collapse onto, `long`/`ulong` follow the host's C long); `np.True_` / `np.False_` (C# bools); `np.longdouble` (the
  float64 descriptor, as `np.clongdouble` is complex128).
- `np.typing` (`[ModuleName("np.typing")]`): C# has no exported type aliases, so each member names the C# type that plays
  the alias's role in NumSharp's signatures: `NDArray` → `typeof(NDArray<>)` (`NumSharp.Generic`; `npt.NDArray[np.float64]`
  is `NDArray<double>`), `ArrayLike` → `typeof(NDArray)` (every array_like parameter), `DTypeLike` → `typeof(DType)` (every
  dtype parameter); `NBitBase` is a nested `[Obsolete]` abstract class (deprecated in NumPy 2.3, NumPy's warning text).
- Not provided: `np.ScalarType` (NumPy's tuple mixes Python built-in classes with NumPy's; `np.isscalar` covers its use),
  and the complex64 / object_ / bytes_ / void classes (no such dtypes). `StrDType` (`np.dtype("string")`) is `numpy.str_`
  under character.
