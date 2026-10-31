---
paths:
  - "src/NumSharp.Core/DTypes/**"
  - "src/NumSharp.Core/Creation/np.dtype.cs"
  - "src/NumSharp.Core/Logic/np.{find_common_type,issubdtype,isdtype,can_cast,result_type,promote_types,type_checks}*.cs"
  - "src/NumSharp.Core/APIs/np.typing.cs"
  - "src/NumSharp.Core/Backends/NPTypeCode.cs"
  - "src/NumSharp.Core/Utilities/NDByteOrder.cs"
  - "test/NumSharp.Tests/DTypes/**"
  - "test/oracle/gen_{dtype,find_common_type,asanyarray_mixed}_oracle.py"
---

# The dtype system

NumPy 2.x's two-level model (NEP 40/41/42/43/50/55/56), in `src/NumSharp.Core/DTypes/`. Gates: `test/NumSharp.Tests/DTypes/*`
and the `dtype_text` tier. The string classes' storage and computation: `strings.md`; the scalar-type classes
(`NDScalarType`, `np.typing`, issubdtype / isdtype / issctype / finfo / iinfo of a class): `scalar-types.md`.

## Model

- `DTypeMeta` is the DType CLASS (`type(np.dtype('f8'))` = `np.dtypes.Float64DType`): one live object per class, with the
  NEP 42 slots as virtuals under NumPy's names (`common_dtype` (null = `NotImplemented`), `common_instance`,
  `default_descr`, `ensure_canonical`, `discover_descr_from_pyobject`, `is_known_scalar_type`, the NEP 55 reservations
  `HasReferences`/`get_clear_loop`/`get_fill_zero_loop`), the C fields `singleton`/`type_num`/`flags`, the Python class
  attributes `type`/`_abstract`/`_parametric`/`_legacy`/`_is_numeric`, and the per-class-pair `CastingImpl` table
  (`castingimpls` + `within_dtype_castingimpl`, looked up by `GetCastingImpl` = `PyArray_GetCastingImpl`; NEP 43
  `ArrayMethod`: `resolve_descriptors` + `get_strided_loop`, which every storage class pair implements and the converting
  walks call; `casting` is the impl's minimal safety).
- `DType` is the INSTANCE (the descriptor, `numpy.dtype`): `Meta` + `byteorder` + the parametric payload
  (`DatetimeMetadata`, a string class's width, `StringDType`'s `VStringParameters`), with NumPy's surface:
  `num`/`kind`/`char`/`itemsize`/`alignment`/`name`/`str`/`descr`/`isbuiltin`/`isnative`/`hasobject`/`flags`/
  `newbyteorder`; `ToString()` = `str(dtype)`, `ToString(true)` = `repr(dtype)`.
- `NPTypeCode` is the storage/kernel discriminator: every storage class exposes it as `Meta.TypeCode` and kernels
  dispatch on it. Several classes can share one lane: the three string classes share `NPTypeCode.String`, and each
  lane-sharing C-type class shares its owner's lane, so a lane names no ONE class - tell classes apart by the
  descriptor (`descr.Meta is VStringDTypeMeta`), never by the code. `DType.From(NPTypeCode.String)` (and
  `(DType)NPTypeCode.String`) refuse (`StringLaneHasNoDType`: the lane carries neither the string class nor the width).
  A class without storage (the datetime pair) has `TypeCode == Empty`: its descriptors parse, print, promote and
  cast-check, but converting one to `NPTypeCode`/`Type` raises `NotSupportedException` (NumSharp has no datetime storage).
- Classes: `LegacyBuiltinDTypeMeta` for the 15 numeric storage types (NumPy type numbers of the host's C `long` model:
  Windows `int32 = 7 'l'`, `int64 = 9 'q'`, LP64 `int32 = 5 'i'`, `int64 = 7 'l'`; `Half = 23`; Decimal 256 / Char 257
  are in the user-defined range, so their descriptors report `isbuiltin == 2` and repr `dtype(decimal)`) and for the
  lane-sharing C-type classes (below); `StringDTypeMeta` × 2 (`BytesDType` `'S'` 18 and `StrDType` `'U'` 19, whose
  singletons are NumPy's unsized builtins `S0` and `U0`) and `VStringDTypeMeta` (NEP 55's `StringDType`, `'T'` 2056,
  parametric, no singleton: every descriptor is fresh); `DatetimeDTypeMeta` × 2 (`DateTime64DType` 21 `'M'`,
  `TimeDelta64DType` 22 `'m'`, parametric); `PyScalarDTypeMeta` × 3 (`_PyLongDType`/`_PyFloatDType`/`_PyComplexDType`,
  abstract: the NEP 50 weak literals).
- `DTypeRegistry` owns them (`FromTypeCode`/`FromScalarType`/`FromTypeNum`/`FromName`, `Register` for further classes,
  casting-impl registration). Lookups take no lock (`ConcurrentDictionary` tables, a `Volatile` code table, `All` copies
  an array that each registration replaces); only `Register` locks, and it checks every key before its first write, so a
  refused class leaves no trace. `DTypeRegistry.FromTypeCode(NPTypeCode.String)` is the `StrDType` CLASS (the lane's
  default class), where the descriptor-level `DType.From` refuses. `np.dtypes` is the NumPy-named facade with NumPy's
  aliases (`ByteDType`, the platform `LongDType`, …) and the C-type classes of the host's model (`IntDType` /
  `UIntDType` on Windows, `LongLongDType` / `ULongLongDType` on Linux, `LongDoubleDType` / `CLongDoubleDType`).

## The C-type classes

NumPy keeps a C-type class beside the sized class of one width and kind: `np.longdouble` / `np.clongdouble` everywhere,
Windows' `np.intc` / `np.uintc` (int32 is the C long there), Linux's `np.longlong` / `np.ulonglong` (int64 is the C long
there). Each is a LANE-SHARING `LegacyBuiltinDTypeMeta` (`DTypeRegistry.LongDouble` / `CLongDouble` / `CInt` / `CUInt` /
`CLongLong` / `CULongLong`; `ownsLane: false`, so `OwnsTypeCode` / `OwnsScalarType` are false): it stores in the owner's
lane and reports its own type number, character, scalar type and class (`np.dtype('g')` is `num 13, char 'g', name
'float64', type longdouble, LongDoubleDType`). NumPy's semantics, probed on Windows and Linux NumPy 2.4.2:
- `np.issubdtype(np.float64, np.longdouble)` is False; the MROs and `__subclasses__()` lists are NumPy's per model
  (`floating` lists `longdouble`, `complexfloating` `clongdouble`, Windows' `signedinteger` `intc` before `int32`).
- Equality is `PyArray_EquivTypes`: same-kind same-size NumPy numerics are EQUAL and hash equally
  (`LegacyBuiltinDTypeMeta.IsEquivalentNumeric`, the hash `(kind, itemsize, byte order)`), never the same object.
- Their casts are NumPy's `numeric_copy_or_byteswap` at `NPY_EQUIV_CASTING`, resolved as a no-cast for natives
  (`can_cast(np.longdouble, np.float64, 'no')` is True).
- Promotion of two NumPy numeric classes reads NumPy's own table of TYPE NUMBERS (`DTypes/NumPyPromotionTable.cs`, a port
  of `_npy_can_cast_safely_table` and the table loop of `initialize_casting_tables`, sizes by model), so the answer is a
  CLASS: `promote(float64, longdouble)` is `longdouble`, and Windows' `promote(int16, uint16)` is the C int `'i'` (equal
  to int32). Its LANE always equals the frozen storage table's, which every kernel promotes by (`CTypeClassesTests`
  checks every owner pair); a weak complex lifts `longdouble` to `clongdouble`. `np.result_type` over arrays is
  descriptor-level (`np.result_type(x_g, x_g)` is `'g'`).
- Arrays KEEP the class through creation (`zeros`/`ones`/`empty`/`full`/`arange`/`*_like`), views (every
  `UnmanagedStorage` sub-view carries it: `AdoptLaneSharingDescr`), copies (`copy`, `flatten` in C / F / K,
  `ascontiguousarray`), `astype`, `view(dtype)` (`SetLaneDescr`: `x.view(np.float64)` of a `'g'` array is float64), joins
  (`concatenate`/`stack`/`hstack`: the class of `np.result_type`), fancy and boolean gathers
  (`NDArray.KeepLaneSharingClass`) and the routines built on those (`tile`, `repeat`, `flip`, `sort`, `roll`, ...).
- `memoryview.format` is the descriptor's character (`'g'`, `"Zg"`, Windows' intc `'i'`; int64 on LP64 is `'l'`).
  Printing names the dtype of a non-default class (`array([0., 0., 0.], dtype=float64)` for longdouble:
  `ArrayFormatter.DtypeIsImplied` reads the class); `np.finfo(np.longdouble).dtype` is `'g'`, `np.iinfo(np.intc).dtype`
  `'i'`.
- Divergences (`[Misaligned]`): COMPUTING routines (ufuncs, reductions, `clip`, ...) return their storage lane's class
  (NumPy registers loops per class, so its `x + x` of a longdouble array is longdouble, NumSharp's float64; the values
  are equal and `(x + x).dtype == np.longdouble` holds). NumSharp's long double is the 8-byte double on every host
  (Windows NumPy's; Linux NumPy's is the 16-byte x87 type, so `np.dtype('float128')` / `'f16'` raise here, as on Windows,
  and Linux's `promote(longdouble, complex128)` is `clongdouble` there, `complex128` here).

## Promotion and casting

- `DTypePromotion` is `common_dtype.c`/`convert_datatype.c`: `CommonDType` (the two-sided protocol, then
  `DTypePromotionError`, a `TypeError`, with NumPy's text), `PromoteDTypeSequence` (`reduce_dtypes_to_most_knowledgeable`,
  order-independent), `PromoteTypes` (the same native descriptor object → itself, NumPy's pointer test; else the common
  class → its default descriptor, or for a parametric class both operands cast to it and `common_instance` taken),
  `CastDescrToDType`, and `ResultType` = NEP 50: every `NDArray` (0-d too), `DType`, `NPTypeCode`, `Type`, dtype string,
  `bool`/`char`/`Half`/`decimal` is STRONG; C# integer / `float` / `double` / `Complex` literals are WEAK
  (`result_type(int8_dtype, 300)` → int8, `+ 1.5` → float64; a lone literal → int64/float64/complex128).
- The storage builtins' kernel promotion is the frozen `np._nptypemap_arr_arr` table (deferring to the larger type
  number, as NumPy does). `np._FindCommonType(NDArray, NDArray)` (binary-operator promotion),
  `NDExprTypeRules.PromoteStrong` and `NDIterCasting.PromoteTypes` are separate engines over it. The internal list
  helpers (`np._FindCommonType_Array`: NDIter's `COMMON_DTYPE` and its allocated operand) run NumPy 2's
  `PyArray_PromoteDTypeSequence` (`PromoteTypeSequence`: `int8, uint8, float16` is float16 in every order, where a
  left fold gives float32); only the public `np.find_common_type` is NumPy 1.26.4's search (below).
- `DTypeCasting` is `PyArray_CanCastTypeTo`/`CheckCastSafety`/`GetCastInfo`/`EquivTypes`/`MinCastSafety` over the impl
  table. The builtin impls: same class → `equiv`; `promote(from, to) == to` → `safe`; kind order `b<u<i<f<c` →
  `same_kind`; else `unsafe` (Char↔UInt16 are distinct classes, never `equiv`). The table also expresses byte order
  (`can_cast('>i4', 'i4', 'equiv')` True, `'no'` False) and the datetime unit rules (`time_to_time`: exact 10³ᵏ folds are
  no-casts, generic → concrete `safe`, concrete → generic `unsafe`, the timedelta years/months barrier `unsafe`,
  finer-and-divides `safe`, else `same_kind`; numeric → `m8` is `safe` for bool and integers except 64-bit unsigned
  (`same_kind`), floats `unsafe`; anything ↔ `M8` `unsafe`). `DTypeCasting.ParseCasting` is NumPy's
  `PyArray_CastingConverter`: names are case-sensitive, an unknown one is NumPy's `casting must be one of 'no', 'equiv',
  'safe', 'same_kind', 'unsafe' (got '…')` ValueError, and null (Python's None) its `casting must be str, not
  NoneType` TypeError.
- The converting walks take their loop from `DTypeCasting.GetDTypeTransferFunction` (NumPy's
  `PyArray_GetDTypeTransferFunction`): the class pair's impl (NumPy's `TypeError` "Cannot cast data from X to Y."
  without one), its `resolve_descriptors` (a loop descriptor other than the given one would need NumPy's multi-step
  wrapping, which native storage never needs: `NotSupportedException`), then its `get_strided_loop`, as an `NDCastInfo`
  (`NPY_cast_info`: `func`, `context`, `descriptors`, `Run(src, srcStride, dst, dstStride, count)` with byte strides; one
  walk at a time, a parse loop owns its scratch). A `StringDType` side's allocator rides on
  `ArrayMethodContext.Allocators` (NumPy keeps it on the descriptor, NumSharp on the cells' buffer), and its loops lock
  both allocators for each call. The callers: the raw assignment of numbers ↔ text (`NDTextConversion.Walk` over
  `UnmanagedStorage.RawArrayAssignArray`, NumPy's `raw_array_assign_array` / `raw_array_wheremasked_assign_array`), the
  iterator's string buffer transfers (`NDIterBufferManager.StringTransfer`, resolved once per fill or write-back; a
  `StringDType` operand's is cached in `NDIterVStringContext` and dropped whenever its buffer allocator changes), the
  text-number reductions' conversions and np.copyto's scalar store into `S` / `U` (NumPy's `PyArray_CastRawArrays`). A
  caller never chooses a converter by the operands' lanes. The bulk routes keep their own kernels: `NDStringCopy.Copy`'s
  unit views, `NDVStringCopy.Copy`, `NDVStringCasting.Copy` (its `S → StringDType` setup reserves a fresh destination's
  arena once, which a per-run loop cannot see), the numeric scalar store's `ConvertValue`, and setitem's parse of a
  Python list's strings (`TextReprStyle.PythonObject`).
- `ndarray.astype` parses its arguments in NumPy's order: the order, then the casting name, before anything else (an
  unknown name raises even where `copy=False` would return the array itself). Its sixth name is `'same_value'`, and a
  NUMERIC cast under it raises `NotSupportedException` (NumPy 2.4's value-checking casts are not ported; pinned in
  `OpenBugs.cs`). A refused rule says `Cannot cast scalar from …` for a 0-d array and `array data` otherwise
  (`DTypeCasting.InvalidCastError`, NumPy's `npy_set_invalid_cast_error`, shared by every assignment route).

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
  to `PyArray_DescrFromType`: `'\x0c'` is float64, `'\x0d'` the longdouble class, `'\x12'` bytes, `'\x13'` str, while
  `'\x11'` object and `'\x14'` void are refused); `kind+size` parsed with `strtol` semantics (`i04`, `i 4`, `i+4` are
  int32; C whitespace only, so `'f\xa08'` is invalid; `b1` is bool; `i3`/`f16`/`?1` are invalid);
- the string spellings are the string classes: `S`, `S10`, `a`, `a5`, `c` (`S1`), `U`, `U32`, `T` / `<T` (a new
  `StringDType()`), byte orders included (`=U2` is `<U2`, `|S3`, `>S` reads native);
- then the name table, looked up by the WHOLE string as NumPy's `typeDict` is, so a byte-order prefix before a NAME is
  "not understood" (`'<float64'`, `'|bool'`, `'<str'`, `'<bytes'`) while it is a byte order before a code (`'<f8'`,
  `'>d'`); `'str'` is `<U0`, `'bytes'` / `'bytes_'` / `'a'` the bytes class, `'cdouble'` complex128.
- Invalid strings raise `NotSupportedException` with NumPy's `data type %R not understood` / `Alias %R was removed in
  NumPy 2.0. Use a name without a digit at the end.` wording, quoted by Python's `repr` (`data type "a'b" not
  understood`); a valid NumPy dtype of a class NumSharp lacks (complex64 / `F` / `csingle`, `V` / `void`, `O` /
  `object` / `O4` / `O8`) says "is not supported by NumSharp". The 24 C# spellings (`Int32`, `Single`, `Decimal`, …) are
  NumSharp's superset, asserted refused by NumPy.
- Builtins are singletons (`np.dtype("i8")` is the same object every call, `isbuiltin 1`), and so are the datetime
  letters (`np.dtype("M")`) and the unsized string classes (`np.dtype("U")`, `np.dtype("S")`); datetime typestrs (`M8`,
  `M8[ns]`), sized strings and non-native descriptors are fresh.
- `DType` equality is STRUCTURAL (class + canonical byte order + unit: `M8[ns] != M8[s]`, `M8[1s] == M8[s]`,
  `>i4 != i4`, an explicit `'<'` equals `'='` on a little-endian host); the hash is class + canonical byte order, so
  `hash(M8[ns]) == hash(M8[s])` as in NumPy (the C-type classes: above). `Equals(object/string/Type/NPTypeCode)` coerces
  like `dtype.__eq__` (`d.Equals("garbage")` is false, never an error). `<`/`<=`/`>`/`>=` are NumPy's safe-cast ordering.
- Null is NumPy's `None` wherever NumPy runs `PyArray_DescrConverter` (`_convert_from_any`'s first test,
  `PyArray_DescrFromType(NPY_DEFAULT_TYPE)`): every `np.dtype` overload (`np.dtype((DType)null)` is the float64
  singleton; `copy: true` / `metadata:` then apply as for any argument) and every `np.result_type` operand (a STRONG
  float64: `np.result_type((DType)null, np.int8)` is float64), through one rule, `np.DefaultTypeDescr`. Not in
  `np.promote_types` / `np.can_cast` (NumPy's `DescrConverter2` refuses None with TypeErrors, ported), not in equality
  (`descr == null` is the engine's null test, so `DType.Double.Equals(null)` is false where NumPy's `np.dtype('f8') ==
  None` is True), and not in a `DType dtype = null` PARAMETER (that means "not given"; each function picks its default).
  A bare `null` literal is CS0121 between `np.dtype(string)` and `np.dtype(Type)`: write it typed.

## The DType element slots

`DTypeMeta` carries NumPy's per-element questions under NumPy's names, internal until a user-defined DType needs them
public. They run once per call or per Python value; bulk moves keep their typed kernels.
- `setitem(descr, value, item, buffer)` / `getitem(descr, item, buffer)` are NEP 42's slots 7 and 8 (`PyArray_Pack`'s last
  step): the numeric classes through `ReductionInitial.PackStore` / `Box` (a non-native descriptor stores and reads the
  swapped bytes), `S` / `U` through the text element codec, `StringDType` through its buffer's allocator (NumSharp keeps
  the allocator on the buffer, `VStringArraySlice`, not on the descriptor, hence `buffer`; the cell must be one of the
  buffer's).
- `copyswapn(descr, dst, dstride, src, sstride, n, swap)` is the legacy ArrFuncs byte-order copy. It swaps each unit: the
  item, a complex number's halves, a `U` element's code units. `ndarray.byteswap` runs through it, as NumPy's
  `PyArray_Byteswap` does (`UnmanagedStorage.Byteswap`: one call over a one-segment array, one per inner run of an
  EXTERNAL_LOOP iterator otherwise), and so do the non-native decodes of `np.frombuffer` and binary `np.fromfile`.
  Kernels (`Utilities/NDByteOrder.cs`): a contiguous run copied and swapped in ONE SIMD pass when source and destination
  do not overlap (`CopyReverseRun`; an overlapping source keeps NumPy's memmove, then the swap), a contiguous run in
  place with SIMD (`ReverseRun`), strided items one `BinaryPrimitives.ReverseEndianness` per unit (`ReverseStrided`).
  `StringDType` refuses a source; in place its swap is the identity (a cell has no byte order; NumPy 2.4.2 crashes the
  process in `ndarray.byteswap` of one).
- `fill(descr, buffer, length)` is `PyArray_Arange`'s fill, NumPy's `@NAME@_fill` operation for operation (C's promotion
  for the integers, `HALF_fill` in float32, complex numbers by parts, `BOOL_fill`'s TypeError); `HasFill` is NumPy's
  `funcs->fill != NULL` (false for `S`, `U` and `StringDType`).
- `nonzero(descr, item)` is the truth `np.nonzero` reads: float16 by its bits (a NaN nonzero, a signed zero not), decimal
  by its value, an NA cell of `StringDType` only for a NaN-like NA object.
- `HasScanFunc` / `HasFromStr` are NumPy's `scanfunc != NULL` / `fromstr != NULL`, the tests behind the text modes of
  `np.fromfile` / `np.fromstring` (`Unable to read character files of that array type` / `don't know how to read
  character strings with that array type`): true for the numeric classes, NumSharp's decimal and char included. The
  scanners stay one typed kernel per scan family (`NumPyTextScan`).
- `DTypeMeta.NPY_LIST_PICKLE` (0x02) is the descriptor flag that refuses `ndarray.tofile`'s binary mode
  (`PyArray_ToFile`'s `PyDataType_FLAGCHK`); `StringDType` carries it.
- The classes without storage (the datetime pair, the abstract Python scalar classes) refuse every slot but `fill`.
  Gate `DTypes/DTypeSlotTests.cs`: every class against the path its slot comes from (the flat iterator's store,
  `np.nonzero`, the slices' `SetIndex` / `GetIndex`) and NumPy's bytes; `copyswapn` against an independent byte reversal
  over its three swap paths and an overlapping source in both directions.

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
- The string ports too: `VStringParameters` has the fields of `PyArray_StringDTypeObject` (`na_object`, `coerce`,
  `has_nan_na`, `has_string_na`, `na_name`, `default_string`; `HasNAObject` has no field, NumPy reads a null
  `na_object`), the cell's constants are `VStringDTypeMeta.SIZEOF_NPY_PACKED_STATIC_STRING` /
  `ALIGNOF_NPY_PACKED_STATIC_STRING`, and the item flag is `DTypeMeta.NPY_NEEDS_INIT`. `VStringAllocator` (the port of
  `static_string.c`) carries the macro names (`NPY_STRING_MISSING`, `NPY_STRING_INITIALIZED`, `NPY_STRING_OUTSIDE_ARENA`,
  `NPY_STRING_LONG`, `NPY_STRING_FLAG_MASK`, `NPY_SHORT_STRING_SIZE_MASK`, `NPY_SHORT_STRING_MAX_SIZE`,
  `NPY_MEDIUM_STRING_MAX_SIZE`, `NPY_MAX_STRING_SIZE`, `SIZEOF_NPY_PACKED_STATIC_STRING`), and its `NpyString_*` ports
  take the function's name after the prefix (`load`, `pack`, `pack_null`, `pack_empty`, `newsize`, `newemptysize`,
  `free`, `dup`, `isnull`, `size`, `share_memory`, `acquire_allocator`, `release_allocator`, and `acquire_allocators` /
  `release_allocators` with a two-allocator and a list overload each); the ports of `static_string.c`'s unprefixed
  helpers (`is_short_string`, `arena_malloc`, …) keep NumSharp names. The pythonnet binding `NumpyStringApi` names the C
  API slots it calls the same way; `np.strings.str_len` is the ufunc's name, `np.ClipmodeConverter` is
  `PyArray_ClipmodeConverter`, `StringCastingImpls.GetDatetimeISO8601StrLen` is `NpyDatetime_GetDatetimeISO8601StrLen`.
  Not renamed: `NDArrayMethodFlags` (NumPy's `NPY_METH_*`) and the CPython ports (pickle, codecs, `str()`).
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
  reason), `static_string.c`'s macros and its `NpyString_` functions (`MacroHolders`: a macro may have several holders;
  `FunctionPorts`: self-retiring exemptions), descriptors × every attribute + str / repr / len / `d[0]`, errors,
  comparison pairs, promotions (each result's class too: char and type number, since the repr cannot tell Windows' C int
  `'i'` from int32's `'l'`), classes, casting resolutions, dtype strings), one file per C `long` model (Windows NumPy /
  Linux NumPy under WSL: regenerate both; the generator reads NumPy's headers from `refs/numpy`, or `NUMPY_SRC` in a
  worktree without one), replayed by `DTypes/DTypeOracleTests.cs`, which picks the host's model. The C-type classes of
  each model are compared like every other class. Exemptions are listed with reasons and retire themselves (each must
  stay absent or divergent): the dunders C# has no use for (`__bool__`, pickle, typing, `__mul__`/`__rmul__`
  sub-arrays), the unported slots (`setitem`, `getitem`, `finalize_descr`, `get_constant`, ArrFuncs,
  `sort_meth`/`argsort_meth`), the absent classes (`Complex64DType`, `ObjectDType`, `VoidDType`), the long-double pair
  where Linux NumPy's is the 16-byte x87 type (`LongDoubleClasses`: the attributes compared where NumPy's size equals
  NumSharp's, `num` and `char` always), the datetime `type`, and the asymmetric `M8[1000ms] == M8[s]` (trap 4).

## C enums, type characters, `np.find_common_type`

- `NPY_TYPECHAR` (`Creation/np.dtype.cs`) is NumPy 2.4.2's `ndarraytypes.h` enum member for member: `NPY_VSTRINGLTR =
  'T'` (NEP 55), `NPY_DEPRECATED_STRINGLTR2 = 'a'`, and no `NPY_INTPLTR` / `NPY_UINTPLTR` (intp is `'n'`; `'p'` / `'P'` /
  `'n'` / `'N'` parse but have no member: cast the character, `(NPY_TYPECHAR)'n'`). The kind letters share values with
  type characters as in C (`'b'` is the bool kind and int8, `'c'` the complex kind and NumPy's `S1`): compare values, never
  names (`ToString()` of a shared value is unspecified).
- `NPTypeCode.ToTypeCode(NPY_TYPECHAR)` IS `np.dtype(c).GetTypeCode()`: `'D'` is complex128, `'F'` is refused, `'l'`
  follows the C `long` model, a control character below `'\x18'` is a type number, and the string characters (`'S'`,
  `'a'`, `'c'`, `'U'`, `'T'`) are the `NPTypeCode.String` lane (the class and width are lost: keep the descriptor).
  `ToTYPECHAR` is the lane's descriptor character (`'D'` for complex128, never the kind letter `'c'`). NumSharp's decimal
  and char have no NumPy character and REFUSE (`NotSupportedException`): their descriptors' own `char` borrows other
  dtypes' letters (decimal `'q'`, char `'c'`); so does the String lane, which names no one dtype (ask the array's own
  `a.dtype.@char`).
- The header gate: the oracle's `members` record carries every C enum of `ndarraytypes.h` (`c_enums`, parsed from
  `refs/numpy`), and `DTypeOracleTests.CEnums_MirrorNumPysHeader` holds the mirrored ones (`NPY_TYPES`, `NPY_TYPECHAR`,
  `NPY_SCALARKIND`, `NPY_ORDER`, `NPY_CASTING`, `NPY_DATETIMEUNIT`, and `NPY_SORTKIND`, the sort kinds of
  `Backends/Default/Sorting/SortKind.cs` with NumPy's nine names: aliases and the new-style names included) to NumPy's
  names, values and order. Exemptions: NumSharp's `NPY_ORDER` numbering (K = 0 … A = 3 where NumPy has A = -1 … K = 2:
  renumbering would change what `default(NPY_ORDER)` means), NumPy 2.4's `NPY_SAME_VALUE_CASTING` and
  `_NPY_ERROR_OCCURRED_IN_CAST`, and the enums NumSharp takes as strings (`NPY_SELECTKIND`, `NPY_SEARCHSIDE`,
  `NPY_CLIPMODE`, `NPY_CORRELATEMODE`) or lacks (`NPY_BUSDAY_ROLL`).
- `np.find_common_type` is NumPy 1.26.4's (`Logic/np.find_common_type.cs`; deprecated in NumPy 1.25, removed in 2.0).
  Each list's maximum is the FIRST candidate of `?bBhHiIlLqQefdgFDGO` every entry casts to safely (a search, not a
  promotion: `[i1, u1, f2]` is float16 where a left-to-right `promote_types` chain gives float32); a scalar type of the
  same or a lower kind never moves it (`([i1], [i8])` is int8, `np.result_type(i1, i8)` int64); a scalar maximum of a
  HIGHER kind goes to `_find_common_coerce` (`find_common_type([u1], [i2])` is int16). The internal
  `_can_coerce_all(dtypelist, start)` takes NumPy's `start`: an index into that ladder, not into the list. A list holding
  decimal or char promotes by NumSharp's table. Gate: `Logic/corpus/find_common_type.numpy1264.jsonl` (from a
  numpy==1.26.4 environment, `test/oracle/gen_find_common_type_oracle.py`).
- `np.asanyarray(object[])` of mixed scalars reads each value through the boundary map (a `bool` is a Python bool, every
  C# integer primitive a Python int, `float` and `double` Python floats, `Complex` a Python complex, `Half` / `char` /
  `decimal` NumPy scalars) and promotes them as NumPy's array coercion does: a Python int is discovered as int64, which
  widens a float16 to float64 in either order, so no ordering of these types changes the result; the element order still
  decides among NumPy scalars. Gate: `Creation/corpus/asanyarray_mixed.numpy242.jsonl`
  (`test/oracle/gen_asanyarray_mixed_oracle.py`).

## `DType` is the one dtype spelling

- `NDArray.dtype` returns `DType` (`Storage.Descr`, NumPy's `PyArrayObject.descr`): the class singleton for a builtin
  lane, so `a.dtype == b.dtype` / `a.dtype == np.float64` are cheap structural compares and
  `ReferenceEquals(a.dtype, np.float64)` holds; a parametric descriptor is carried explicitly through
  `Alias`/`Clone`/`ReplaceData` and reset when the lane changes.
- The `np.float64` family (`np.bool_` … `np.clongdouble`, `np.@decimal`, `np.intp` / `np.@long` / `np.@uint`, `np.str_`
  (`<U0`), `np.bytes_` (`S0`), the throwing `np.chars`) are `DType` descriptors, not `Type`s: NumSharp has no scalar-type
  objects (a C# `double` IS the scalar), so the descriptor is the one currency, and `np.float64.itemsize` / `.name` /
  `.kind` read as in NumPy. `np.complex64` / `np.csingle` / `np.object_` / `np.void` are dtype-less `NDScalarType`
  classes (below), so `np.zeros(3, np.complex64)` is a compile error.
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
