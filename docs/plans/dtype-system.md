# NumSharp DType system — NEP 41/42/43/50-aligned descriptor + DType-class architecture

> Design and staged plan for replacing "a dtype is an `NPTypeCode`" with NumPy 2.x's two-level model
> (a **DType class** with behaviour slots, and **descriptor instances** that can carry parameters),
> so that `datetime64[unit]`, `timedelta64[unit]`, fixed-width strings (`U`/`S`), NEP 55
> `StringDType` and future user dtypes fit without touching the 15-type kernel switches — while every
> existing `NPTypeCode`-based call keeps compiling and keeps its value/dtype parity.
>
> Source of truth: `refs/numpy/` (2.4.2) — `numpy/_core/src/multiarray/{dtypemeta.[ch], common_dtype.c,
> convert_datatype.c, abstractdtypes.c, array_method.h, descriptor.c, datetime.c}`,
> `numpy/_core/include/numpy/dtype_api.h`, `numpy/_core/_dtype.py`, `numpy/dtypes.py`, and the NEPs
> under `refs/numpy/doc/neps/nep-00{40,41,42,43,50,55,56}*.rst`. Every behaviour below was probed live
> against `numpy==2.4.2` (win-amd64) before being written down.

---

## 0. Why

NumSharp today describes a dtype with ONE integer — `NPTypeCode` (a `System.TypeCode`-numbered enum:
Boolean=3 … Decimal=15, Half=16, String=18, Complex=128). It is the storage discriminator
(`UnmanagedStorage._typecode`, the typed-slice union), the kernel key (`DirectILKernelGenerator.GetTypeSize/
GetClrType`, every `switch (typecode)`), the promotion-table key (`np.find_common_type.cs`), the
casting key (`NDIterCasting.CanCast`), and the whole descriptor surface (`DType` is a `(Type, NPTypeCode)`
spelling shim whose equality is `typecode == typecode`). That is exactly NumPy 1.x's `PyArray_Descr`
world that NEP 40 describes and NEP 41–43 replaced, and it cannot express what the next dtypes need:

| Need | Why an enum cannot carry it |
|---|---|
| `datetime64[ns]` vs `datetime64[s]` | same storage (int64), same type number, DIFFERENT dtype — the unit + multiplier live on the **instance** (`PyArray_DatetimeMetaData`, NumPy's `c_metadata`). `M8[ns] == M8[s]` is `False`; promotion is the unit GCD; `can_cast(M8[D], M8[s], 'safe')` is `True` but the reverse is `False`. |
| `U10`, `S5` | itemsize is a per-instance parameter; `promote_types(U5, U10) → U10`; `promote_types(int32, U) → U11` is decided by the **cast** (`int32` prints in ≤11 chars). |
| `StringDType()` (NEP 55) | non-legacy layout (num 2056), per-instance `na_object`/`coerce`, reference-holding items (`hasobject`, clear/fill-zero loops). |
| new ufunc loops for a new dtype | today a loop is a `case NPTypeCode.X:` inside 63 partial IL generators; NEP 43 says a loop is an `ArrayMethod` registered against DType classes and looked up by `resolve_descriptors`/`get_loop`. |

## 1. NumPy's model (what we are aligning to)

```
DTypeMeta  (the CLASS — np.dtypes.Float64DType, type(np.dtype('f8')))
  singleton, type_num, scalar_type (DType.type), flags {LEGACY, ABSTRACT, PARAMETRIC, NUMERIC}
  slots: discover_descr_from_pyobject, is_known_scalar_type, default_descr, common_dtype,
         common_instance, ensure_canonical, setitem, getitem, get_clear_loop, get_fill_zero_loop,
         finalize_descr, get_constant, within_dtype_castingimpl, castingimpls{DTypeMeta→ArrayMethod},
         (legacy ArrFuncs: compare/argmax/sort/argsort/fill/nonzero/…), sort_meth, argsort_meth
Descr      (the INSTANCE — np.dtype('f8'), np.dtype('M8[ns]'))
  typeobj, kind, type(char), byteorder, type_num, flags(item flags), elsize, alignment, metadata,
  hash, [legacy: subarray, fields, names, c_metadata (datetime unit!)]
ArrayMethod / CastingImpl  (NEP 43 — one object per (in-dtypes → out-dtypes) implementation)
  name, nin, nout, casting (minimal safety), flags, resolve_descriptors(given → loop descrs, safety,
  view_offset), get_strided_loop(context, aligned, strides → inner loop), get_reduction_initial
```

The algorithms we port verbatim:

* **`PyArray_CommonDType(a, b)`** (`common_dtype.c`): `a.common_dtype(b)`; on `NotImplemented` try
  `b.common_dtype(a)`; if still `NotImplemented` → `DTypePromotionError("The DTypes %S and %S do not have
  a common DType. For example they cannot be stored in a single array unless the dtype is `object`.")`.
* **`PyArray_PromoteTypes(descr1, descr2)`** (`convert_datatype.c`): identical-native-legacy fast path;
  `common = CommonDType(class1, class2)`; if `common` is not parametric → `common.default_descr()` (loses
  metadata); else `CastDescrToDType` each input to `common` (via that cast's `resolve_descriptors`) and
  `common.common_instance(d1, d2)`.
* **`PyArray_PromoteDTypeSequence`** — `reduce_dtypes_to_most_knowledgeable` (pairwise low/high
  reduction that swaps on `NotImplemented` and clears an operand when `common(low, high) is low`),
  then the "main dtype" promotes every survivor; failure text: `The DType %S could not be promoted by
  %S. This means that no common DType exists for the given inputs. For example they cannot be stored in
  a single array unless the dtype is `object`. The full list of DTypes is: (%S, …)`.
* **`PyArray_ResultType`** (NEP 50): Python scalars contribute only their abstract class (`_PyLongDType`,
  `_PyFloatDType`, `_PyComplexDType`) and NO descriptor; the abstract classes' `common_dtype`
  (`abstractdtypes.c`: `int_common_dtype`/`float_common_dtype`/`complex_common_dtype`) do the "weak"
  work (`int8 + 300 → int8` at result_type level — no overflow check there); an abstract survivor
  falls back to its default (`int64`/`float64`/`complex128`).
* **`can_cast(from, to, casting)`** = `PyArray_CanCastTypeTo`: unsized flexible `to` (`U0`) is treated
  as the class; `PyArray_CheckCastSafety`: fetch the `CastingImpl(from.class, to.class)` (none →
  `False`), short-circuit when `MinCastSafety(impl.casting, requested) == requested`, else
  `resolve_descriptors` → `MinCastSafety(safety, requested) == requested`. `MinCastSafety` = the LARGER
  (less safe) enum value (`no=0 < equiv=1 < safe=2 < same_kind=3 < unsafe=4`).
* **datetime**: `compute_datetime_metadata_greatest_common_divisor` (unit GCD with the
  years/months-vs-linear barrier that is STRICT for timedelta operands and RELAXED for datetime),
  `datetime_type_promotion`, `can_cast_datetime64_units`/`can_cast_timedelta64_units` +
  `datetime_metadata_divides`, `time_to_time_resolve_descriptors` (NO/EQUIV for equal or exact
  10³ᵏ-fold metric prefixes, SAFE from generic, UNSAFE to generic or across the timedelta barrier, SAFE
  towards finer units when the multipliers divide, else SAME_KIND), `datetime_common_dtype`
  (datetime absorbs timedelta; the builtin promotion table gives `timedelta × {bool, signed ints,
  uint8/16/32} → timedelta`, everything else with a number → error), the metadata grammar
  `[<num><unit>]` / `[<unit>/<den>]` with the divisor→multiple table (`m8[s/2]` → `m8[500ms]`,
  `M8[Y/2]` → `M8[6M]`, `m8[D/3]` → `m8[8h]`), and the `str`/`name`/`repr` spellings
  (`<M8[ns]`, `datetime64[ns]`, `dtype('<M8[ns]')`).

Probed facts worth pinning (all 2.4.2): builtins are singletons (`np.dtype('i8') is np.dtype('i8')`),
parametric instances are not; `hash(M8[ns]) == hash(M8[s])` even though they differ; `dtype('i4') ==
'i4' == np.int32` (coercing equality) but `== int` is `False` and `== 'garbage'` is `False` (no error);
`dtype('>i4')`: byteorder `'>'`, `isnative False`, `isbuiltin 0`, `str() → '>i4'`,
`promote_types('>i4','>i4') → int32` (native, the identical fast path requires native byte order);
1-byte types always report byteorder `'|'`; `np.dtype(np.dtypes.Int8DType)` is `dtype('O')` (a class
object is just an object to `np.dtype`) — the class is instantiated as `Int8DType()`, and parametric
legacy classes refuse (`DateTime64DType()` → "Preliminary-API: … can only be instantiated using
`np.dtype(...)`"). **`np.dtype('M8[s/0]')` crashes the interpreter** (integer division by zero in
`convert_datetime_divisor_to_multiple`); NumSharp raises the metadata `TypeError` instead.

## 2. NumSharp architecture (target)

```
NumSharp.DTypeMeta                (abstract) — the DType CLASS. One live object per dtype class.
 ├─ LegacyBuiltinDTypeMeta        the 15 storage-backed types + the vestigial NPTypeCode.String; data-driven
 │                                (name, TypeNum, TypeCode, ScalarType, Kind, TypeChar, ItemSize, Alignment,
 │                                flags, aliases). common_dtype = the existing frozen promotion table.
 ├─ PyScalarDTypeMeta (internal)  _PyLongDType/_PyFloatDType/_PyComplexDType — NEP 50 weak C# literals
 └─ DatetimeDTypeMeta             DateTime64DType / TimeDelta64DType — PARAMETRIC; unit metadata on the
                                  instance; GCD promotion; unit casting rules. (descriptor-level in stage A)
NumSharp.DType                    (sealed) — the descriptor INSTANCE (was the spelling shim; extended):
                                  Meta, num, kind, char, byteorder, itemsize, alignment, name, str, descr,
                                  isbuiltin, isnative, hasobject, flags, metadata, Parameters (c_metadata),
                                  structural equality + coercing Equals(object) (Type/NPTypeCode/string),
                                  singletons per builtin, ToString()=str(dtype), ToString(repr:true).
NumSharp.DTypeRegistry            static — NPTypeCode/Type/type_num/char/name → DTypeMeta; Register().
NumSharp.DTypePromotion           static — CommonDType, PromoteDTypeSequence, PromoteTypes, ResultType,
                                  CastDescrToDType, CastToDTypeAndPromoteDescriptors (NEP 42 + NEP 50).
NumSharp.ArrayMethod              (abstract, NEP 43) name/nin/nout/Casting/Flags/DTypes,
 └─ CastingImpl                     ResolveDescriptors(given → loop, view_offset) + GetStridedLoop(...).
    ├─ BuiltinCastingImpl           builtin→builtin (safety from the promotion table + kind order; loop =
    │                               the existing IL cast kernels — the engine still drives them directly)
    └─ TimeToTime/DatetimeToTimedelta/NumericToTime/TimeToNumeric CastingImpls (descriptor-level)
NumSharp.DTypeCasting             static — GetCastingImpl, GetCastInfo, CheckCastSafety, CanCastTypeTo,
                                  MinCastSafety, casting-string parsing.
np.dtypes                         nested static class, [ModuleName("np.dtypes")] — Int8DType … Complex128DType,
                                  DateTime64DType, TimeDelta64DType, the C-name aliases (ByteDType, IntDType,
                                  LongDType (platform), LongLongDType, LongDoubleDType→Float64 …), NumSharp-only
                                  DecimalDType/CharDType.
np.dtype(...)                     factory: string grammar ported from descriptor.c `_convert_from_str`
                                  (byteorder prefix incl. non-native '>', datetime typestr, sized codes, names)
                                  + Type/NPTypeCode/DType/DTypeMeta overloads.
np.promote_types / result_type / can_cast / issubdtype / isdtype / datetime_data — DType overloads on top of
                                  the existing NPTypeCode ones (which stay, routed to the same engine).
DTypePromotionError : TypeError   numpy.exceptions.DTypePromotionError, verbatim texts.
```

### 2.1 Compatibility contract (why existing code keeps working)

* `NPTypeCode` stays the storage/kernel discriminator. Every `DTypeMeta` that has storage exposes
  `TypeCode`; `DType → NPTypeCode` implicit conversion is unchanged; every existing
  `NPTypeCode`/`Type` overload stays. Kernels keep dispatching on `NPTypeCode`; a future
  storage-backed dtype adds ONE enum member (`DateTime64 = 21`, `TimeDelta64 = 22`, mirroring
  `NPY_DATETIME`/`NPY_TIMEDELTA`) and its meta registers the code — the enum is an optimisation the
  registry hands out, not the identity.
* `DType` keeps its public fields and implicit conversions (`Type`/`NPTypeCode`/`NPTypeCode?`/`string`
  in; `Type`/`NPTypeCode` out — *Stage B made the `Type` out-conversion EXPLICIT, see §5*). For the 15 builtins, structural equality collapses to typecode equality
  — bit-identical behaviour to today. `Equals(object)` additionally coerces `Type`/`NPTypeCode`/`string`
  (NumPy's `dtype.__eq__` coerces its operand), so `dtype.Should().Be(typeof(int))` and
  `dtype == np.int32` are true.
* `NDArray.dtype` still returns `Type` in stage A (204 test sites assert `Assert.AreEqual(typeof(X),
  nd.dtype)`, which would silently fail with a descriptor on the right). Flipping it to `DType` is stage
  B, its own commit, after the Creation/IO families gain `DType` overloads (a `DType` argument is
  otherwise AMBIGUOUS between a `Type` and an `NPTypeCode` overload — the trap recorded in memory).
  *(Stage B did this — §5. Two predictions here were wrong in a useful way: with `DType → Type` made
  EXPLICIT, `Assert.AreEqual(typeof(X), nd.dtype)` infers `T = DType` and passes, so those sites needed no
  rewrite; and the `Type`+`NPTypeCode` pairs were not given a third `DType` overload but REPLACED by it.)*
* `np.dtype(string)` accepts everything it accepted (all `DTypeStringParityTests` mappings, the NumSharp
  PascalCase aliases) and now ALSO datetime/timedelta typestrs and non-native byte orders. Strings /
  void / object / StringDType still raise `NotSupportedException` (no meta registered) with the same
  texts. Invalid strings keep `NotSupportedException` (pinned) but adopt NumPy's
  `data type 'X' not understood` wording.
* The three existing promotion engines (`np.find_common_type` tables, `NDExprTypeRules.PromoteStrong`,
  `NDIterCasting.PromoteTypes`) and the two casting-safety engines (`np.can_cast`, `NDIterCasting.CanCast`)
  are NOT collapsed in stage A: the builtin meta's `common_dtype` IS the `_nptypemap_arr_arr` table and
  the builtin `CastingImpl` safety IS `np.can_cast`'s rule (`safe ⇔ promote(from,to)==to`; `same_kind ⇔
  kind order`), so the `dtype_text` fuzz tier (promote_types / result_type / can_cast / isdtype /
  issubdtype, 2,618 cases) gates that the new engine reproduces the old answers bit-for-bit. Unifying
  the iterator's copies onto the engine is a follow-up with its own A/B (their `i4+f4` answers differ:
  `NDIterCasting.PromoteTypes` says `f4`, NumPy says `f8` — a latent VIRTUAL-operand divergence).

### 2.2 Type numbers, kinds, chars (NumSharp ↔ NumPy)

| NumSharp | NumPy class | num | kind | char | notes |
|---|---|---|---|---|---|
| Boolean | BoolDType | 0 | `b` (was `'?'` — fixed, AuditV2 T1.52) | `?` | |
| SByte / Byte | Int8DType / UInt8DType | 1 / 2 | i / u | b / B | |
| Int16 / UInt16 | Int16DType / UInt16DType | 3 / 4 | i / u | h / H | |
| Int32 / UInt32 | Int32DType / UInt32DType | LP64 **5 / 6**, LLP64 **7 / 8** | i / u | LP64 `i / I`, LLP64 `l / L` | **platform-resolved** (see below); `IntDType` alias; `LongDType` on Windows |
| Int64 / UInt64 | Int64DType / UInt64DType | LP64 **7 / 8**, LLP64 **9 / 10** | i / u | LP64 `l / L`, LLP64 `q / Q` | **platform-resolved**; `LongLongDType` alias; `LongDType` on LP64 |
| Half / Single / Double | Float16/32/64DType | 23 / 11 / 12 | f | e / f / d | `LongDoubleDType` → Float64 (`'g'` collapses) |
| Complex | Complex128DType | 15 | c | D | `CLongDoubleDType` → Complex128; complex64 stays unsupported |
| Decimal | DecimalDType (NumSharp) | 256 | f | q (legacy `ToTYPECHAR`) | `isbuiltin 2` (user-defined range), name `decimal`, str `<f16` |
| Char | CharDType (NumSharp) | 257 | u (was `'S'`) | c | promotes/prints as uint16 everywhere; name `char`, str `<u2` |
| String (vestigial) | — | 19 | U | none | no storage; only the `"string"`/`"String"` aliases reach it |
| — | DateTime64DType | 21 | M | M | parametric, descriptor-level in stage A |
| — | TimeDelta64DType | 22 | m | m | parametric, descriptor-level in stage A |

**The four C-integer types are platform-resolved (2026-09-13), so `dtype.num`/`.char` are byte-identical to the
LOCAL NumPy.** `np.dtype('int32')`/`np.dtype('int64')` resolve to whichever C integer is that width: on LP64
(Linux/macOS, C `long` = 64-bit) `int32 == NPY_INT (5,'i')` and `int64 == NPY_LONG (7,'l')`; on LLP64 (Windows / any
32-bit process, C `long` = 32-bit) `int32 == NPY_LONG (7,'l')` and `int64 == NPY_LONGLONG (9,'q')` — verified against
numpy 2.4.2. `DTypeRegistry` maps NumSharp's single Int32/UInt32/Int64/UInt64 onto the platform-correct `(num, char)`
via the existing `CLongIs32Bit` switch (the same switch the `LongDType`/`LongLongDType` aliases already used — the nums
are now CONSISTENT with those aliases, where before they were hard-coded LP64). Everything else (bool 0, float32 11,
float64 12, complex128 15, half 23, str 19, datetime 21, timedelta 22, Decimal 256, Char 257) is platform-independent.

**`NPY_TYPES` (the identity enum) landed alongside `NPTypeCode` (the computational switch).** `Creation/np.dtype.cs`
now carries `public enum NPY_TYPES` — a byte-identical mirror of NumPy's C `NPY_TYPES` (`ndarraytypes.h`), sibling to
the existing `NPY_TYPECHAR`, plus the NumSharp user-range extras `NUMSHARP_DECIMAL`(256)/`NUMSHARP_CHAR`(257).
`DType.type_num` returns it (`(NPY_TYPES)Meta.TypeNum`); `DType.num` stays `int` (NumPy's `.num` is an int);
`DType.From(NPY_TYPES)` is the `PyArray_DescrFromType` reverse (rejecting a number with no NumSharp class on the
platform). `NPTypeCode` is untouched — it remains the storage/kernel discriminator every backend `switch`/IL generator
dispatches on; the `DType` descriptor is served ALONGSIDE it into the kernels (Stage C). Gate: `DTypeDescriptorTests`
(`Builtin_Surface` + `type_num`) and `NpDtypesModuleTests` are now platform-aware; full DType/promotion/parity suite
1523 green, `dtype_text` fuzz tier + `OracleSurfaceCoverage` green.

## 3. Stages

**Stage A — this change.** Everything in §2 except the `NDArray.dtype` flip and the iterator/engine
unification: `DTypeMeta` + registry + slots, the extended `DType`, NEP 42 promotion and NEP 50
`result_type` on descriptors, the NEP 43 `ArrayMethod`/`CastingImpl` seam with the builtin impl,
`np.dtypes`, the full dtype-string grammar, `DTypePromotionError`, and the datetime64/timedelta64
metas at the DESCRIPTOR level (parse, format, equality, `np.datetime_data`, GCD promotion, unit cast
safety) as the parametric proof. Allocating storage for a datetime descriptor raises
`NotSupportedException` naming the missing piece. Gates: `test/NumSharp.Tests/DTypes/*` (all probed
against 2.4.2) + the existing `dtype_text` fuzz tier + the full suite.

**Stage B — `NDArray.dtype → DType`.** Add `DType` overloads to every API that still has a
`Type` + `NPTypeCode` pair (Creation, IO, `np.indices`, `linalg.arrayapi`, `corrcoef`, manipulation,
`concatenate`), rewrite the `Assert.AreEqual(typeof(X), nd.dtype)` test sites, then change the property
type; `UnmanagedStorage` gains a `Descr` field (today it is derived from `_typecode`).

**Stage C — datetime64/timedelta64 storage + kernels** (`docs/plans/datetime64.md` has the full
NumPy analysis): `NPTypeCode.DateTime64/TimeDelta64`, an int64 slice lane, the NaT sentinel, the
arithmetic/comparison/min-max ufunc loops registered as `ArrayMethod`s against the datetime metas
(NEP 43 dispatch for non-legacy dtypes), ISO-8601 parse/format, `arange`, `astype` between units, the
`.npy` descr. The `ArrayMethod.GetStridedLoop` contract is `NDInnerLoopFunc` — NumSharp's existing
`PyUFuncGenericFunction` analog — so NDIter drives new loops without a new driver.

**Stage D — strings.** `StrDType` (`U`) / `BytesDType` (`S`) as parametric legacy metas (itemsize
parameter, `string_unicode_common_dtype`/`common_instance` = max length, the int→string length casts),
then NEP 55 `StringDType` (arena storage, `hasobject`, `get_clear_loop`/`get_fill_zero_loop` slots —
already declared on `DTypeMeta`).

## 4. NEP compliance map

| NEP | Requirement | NumSharp |
|---|---|---|
| 40 | legacy dtype layout is what to move away from | `NPTypeCode` demoted to a per-meta storage key |
| 41 | dtypes are classes; scalars are not dtype instances; `np.dtype[np.float64]` class getter | `DTypeMeta` instances are the classes; `DType.type`/`Meta.ScalarType` is the C# scalar type; `DTypeRegistry.FromScalarType(typeof(double))` ≙ `np.dtype[np.float64]` |
| 42 | `__common_dtype__`/`__common_instance__`/`default_descr`/`discover_descr_from_pyobject`/`is_known_scalar_type`/`ensure_canonical`/`getitem`/`setitem`; abstract & parametric flags; `CastingImpl.resolve_descriptors`; `np.dtype('S')`-class-vs-instance | the same slots as virtual methods; `DTypeFlags`; `CastingImpl`; unsized/generic descriptors stand for the class |
| 43 | `ArrayMethod` (`resolve_descriptors`, `get_loop`, context, flags, casting), promoters keyed on DType classes | `ArrayMethod`/`CastingImpl`/`ArrayMethodContext`; loop contract = `NDInnerLoopFunc`; flags = existing `NDArrayMethodFlags` |
| 50 | weak Python scalars; `result_type(int8, 300) → int8`; 0-D arrays are strong | `PyScalarDTypeMeta` for C# literals; `np.result_type(params object[])` treats every `NDArray` as strong (the operator path's 0-D weakness stays the documented `[Misaligned]`) |
| 55 | variable-width `StringDType` | slots reserved (`HasReferences`/`GetClearLoop`), num 2056 reserved, stage D |
| 56 | `np.isdtype`, `np.dtypes` namespace, `promote_types`/`result_type` on dtypes | `np.dtypes` facade; `DType` overloads |

## 5. Implementation status

**Stage A landed (2026-09-08).** Everything in §2/§3-A is in `src/NumSharp.Core/DTypes/` plus
`Exceptions/DTypePromotionError.cs`, with the factory/grammar in `Creation/np.dtype.cs` and the `DType`
overloads in `Logic/np.{promote_types,result_type,can_cast,issubdtype,type_checks}.cs`:

| File | Port of |
|---|---|
| `DTypeMeta.cs`, `DTypeFlags.cs` | `PyArray_DTypeMeta` + `NPY_DType_Slots` (slots as virtuals; `NPY_DT_*` flag bits) |
| `LegacyBuiltinDTypeMeta.cs` | `dtypemeta_wrap_legacy_descriptor` (data-driven, one per `NPTypeCode`) |
| `PyScalarDTypeMeta.cs` | `abstractdtypes.c` (`_PyLongDType`/`_PyFloatDType`/`_PyComplexDType`, the three `*_common_dtype`) |
| `DatetimeDTypeMeta.cs`, `DatetimeMetaData.cs` | the datetime slots + `datetime.c` metadata grammar / GCD / divides / unit casting rules |
| `ArrayMethod.cs`, `CastingImpls.cs` | NEP 43 `ArrayMethod`/`castingimpl`; `add_numeric_cast`, `time_to_time_resolve_descriptors`, `datetime_to_timedelta_resolve_descriptors`, `PyArray_AddLegacyWrapping_CastingImpl` |
| `DTypeRegistry.cs` | `_builtin_descrs` / `typenum_to_dtypemeta` / `PyArray_InitializeNumericCasts` / `PyArray_InitializeDatetimeCasts` |
| `DTypePromotion.cs` | `PyArray_CommonDType`, `reduce_dtypes_to_most_knowledgeable`, `PyArray_PromoteDTypeSequence`, `PyArray_PromoteTypes`, `PyArray_CastDescrToDType`, `PyArray_ResultType` (NEP 50) |
| `DTypeCasting.cs` | `PyArray_GetCastInfo`, `PyArray_CheckCastSafety`, `PyArray_CanCastTypeTo`, `PyArray_EquivTypes`, `PyArray_MinCastSafety`, the casting-string converter |
| `DType.cs` | the descriptor: `numpy.dtype`'s surface, `_dtype.py`'s `__str__`/`__repr__`/`_name_get`, `arraydescr_newbyteorder` |
| `np.dtypes.cs`, `np.datetime_data.cs` | `numpy.dtypes` (NEP 56) and `np.datetime_data` |

Gates: `test/NumSharp.Tests/DTypes/*` (5 suites — descriptor surface, string grammar, promotion, casting,
`np.dtypes`), the updated `Creation/DTypeStringParityTests.cs` and the three `AuditV2` T1.51/T1.52/T1.53
reproductions (no longer `[OpenBugs]`), the `dtype_text` fuzz tier (2,618 cases, unchanged answers), the full
suite (14,823 green) and `OracleSurfaceCoverageTests` (`datetime_data` classified sibling-owned).

Things learned while landing it that the design above did not anticipate:

* **`np.dtype('M')` / `np.dtype('m')` are valid** — `PyArray_DescrFromType` maps the datetime LETTERS to the
  generic descriptors, so the parity tests that pinned them as unsupported were wrong, not conservative.
* **Do not add `==(DType, string)` operators.** A `null` literal binds them over `==(DType, DType)`, so every
  `descr == null` check inside the engine silently answered false and the next dereference NRE'd. The coercing
  string comparison lives on `Equals(string)`; `dtype == "i4"` still works through the implicit conversion.
* **C# scalar literals bind NumSharp's scalar→`NDArray` conversion before `params object[]`**, so
  `np.result_type(int8_array, 300)` reaches the strong `(NDArray, NDArray)` overload, not the weak NEP 50 path.
  The weak rule is reachable through the dtype: `np.result_type(a.dtype, 300)` → `int8`. A dtype STRING would
  likewise bind the string→`NDArray` (character array) conversion and be ambiguous against `params DType[]`, so
  `result_type(params string[])`, `can_cast(string, DType, …)`, `issubdtype(string, string)` and
  `isdtype(string, …)` exist purely to give a string argument the dtype grammar.
* `promote_types('m8[Y]', 'M8[D]')` is `M8[D]`, not an error: `PromoteTypes` first casts the timedelta operand
  to the DATETIME class (keeping its unit), and only then runs the GCD, so the strict years/months barrier
  bites on `m8 × m8` alone.
* NumPy's `dtype.__eq__` (`PyArray_EquivTypes`) is asymmetric for the exact 10³ᵏ metric-prefix folds
  (`M8[1000ms] == M8[s]` but not the reverse); `DType.Equals` is structural and the NumPy relation is exposed as
  `DTypeCasting.EquivTypes` — the one `[Misaligned]` in the descriptor suite.
* The old `result_type(params NPTypeCode[])` fold (`_can_coerce_all`) dropped every other pair for four or
  more operands (`result_type(i1, i1, f8, i1)` was `int8`); the public entry points now run the NEP 42 sequence
  reduction. The three internal engines (`np._FindCommonType*`, `NDExprTypeRules`, `NDIterCasting`) are
  untouched, as planned.

**Stage B landed (2026-09-09) — `DType` is the one dtype spelling end to end.**

* **`NDArray.dtype` returns `DType`** — `UnmanagedStorage.Descr`, NumPy's `PyArrayObject.descr`. For a builtin lane it is the
  class singleton (resolved lazily from `_typecode`, so a fresh result array pays nothing until asked), so `a.dtype == b.dtype`
  and `a.dtype == np.float64` are cheap structural compares and `ReferenceEquals(a.dtype, np.float64)` holds; a parametric
  descriptor (`UnmanagedStorage(DType)`, `Allocate(Shape, DType, bool)`, `TensorEngine.GetStorage(DType)` — the
  `PyArray_NewFromDescr` entries) is stored as-is, travels through `Alias`/`Clone`/`ReplaceData`, and is dropped when the
  lane changes. `UnmanagedStorage.DType` (the CLR element `Type`) and `TypeCode` stay the kernel currency.
* **The `np.float64` family are `DType` descriptors** (`np.bool_`…`np.clongdouble`, `np.@decimal`, the platform aliases
  `intp`/`uintp`/`@long`/`@ulong`/`@uint`, the throwing `complex64`/`csingle`/`chars`). NumSharp has no scalar-type objects (a
  C# `double` IS the scalar), so the descriptor is the single currency; `np.float64.itemsize`/`.name`/`.kind` read like NumPy.
* **Every `dtype`/`typeCode` parameter takes ONE `DType`** and the `Type`/`NPTypeCode` twins are gone: the Creation family,
  IO (`fromfile`/`fromstring`/`loadtxt`; `frombuffer`'s `DType` entry delegates to its `NPTypeCode` implementation), `indices`,
  `trace`, `clip`, `cumsum`/`cumprod`/`prod`/`sum`, `nancumsum`/`nancumprod`, `amax`/`amin`/`std`/`var`, `cov`/`corrcoef`,
  `getfield`/`setfield`, `finfo`/`iinfo`, `Generator.*`/`randint`/`uniform`/`SeedSequence`, `concatenate`/`concat`,
  `matmul`/`vecdot`/`matvec`/`vecmat`/`einsum` (via `GufuncGuard.ToLoop(NDArray, DType)`), `astype`/`view`, the seven
  `NDArray(NPTypeCode, …)` ctors, `isdtype`, and `find_common_type(DType[] …)`. Kept as `Type` on purpose (the concrete
  element type of a lane, never a user request): `TensorEngine.GetStorage(Type)`/`Cast(…, Type, …)`,
  `UnmanagedStorage.Allocate/Cast/AliasAs(Type)`, `NDArray.ReplaceData`. Kept returning `NPTypeCode`: the typed
  `result_type`/`promote_types`/`can_cast` pairs and `find_common_type`/`_FindCommonType*` (the `DType` overloads return `DType`).
* **`DType → Type` became an EXPLICIT conversion** (`Type → DType` stays implicit) — §2.1's "implicit out" was wrong once
  `NDArray.dtype` was a `DType`: `System.Type` declares its own `==`, so with conversions in both directions
  `a.dtype == typeof(double)` is CS0034-ambiguous between `Type.==` and `DType.==`, and `==(DType, Type)` overloads would make
  `descr == null` ambiguous instead (neither target is better when both directions convert). One-way implicit resolves
  `a.dtype == typeof(double)` / `typeof(double) == a.dtype` / `== "f8"` / `== NPTypeCode.Double` to the structural, coercing
  `DType.==`, and makes `Assert.AreEqual(typeof(double), a.dtype)` infer `T = DType` — which is why the 310 MSTest sites the
  plan expected to rewrite needed no change. The cost is `(Type)a.dtype` / `a.dtype.type` wherever a `Type` is required by
  signature (kernel-factory lookups, `Convert.ChangeType`, `Type.Name` in messages → `dtype.type.Name`).
* **A static-initialization cycle had to be broken.** `np`'s field initializers now run `DType.From` → `DTypeRegistry`'s
  constructor, which registered the 15×15 `BuiltinCastingImpl`s whose minimal `Casting` read `np._nptypemap_arr_arr` — filled by
  `static np()` AFTER the field initializers — so the registry saw null and the process died with
  `TypeInitializationException`. `ArrayMethod.Casting` is `virtual` and `BuiltinCastingImpl` computes it on first read (a
  constant; the laziness is unobservable). The `static np()` tables are built as `Dictionary<(DType, DType), DType>` (the
  aliases ARE descriptors) and the frozen `Type`/`NPTypeCode`-keyed maps are derived. Both orders are probed in fresh
  processes (np-first and registry-first).
* **`isdtype` folded onto the NumPy-exact `DType` overloads**: the old `NPTypeCode`/`Type` twins accepted `issubdtype`'s
  vocabulary (`"floating"`, `"integer"`) which NumPy's `isdtype` rejects (`kind argument is a string, but 'floating' is not a
  known kind name.`); the `NDArray` convenience overloads route through `arr.dtype`.
* **Tests:** `test/NumSharp.Tests/Utilities/DTypeAssertions.cs` — `Should(this DType)` → `DTypeAssertions` (`Be(DType)`,
  `Be<T>()`, `NotBe`, `BeOneOf`), declared in the `NumSharp.Tests` namespace so enclosing-namespace lookup binds every
  `x.dtype.Should()` ahead of AwesomeAssertions' `Should(this object)`; ~1,000 existing `.dtype.Should().Be(…)` sites compile
  unchanged. Gates: full suite net10.0 14,832 / net8.0 14,824 green, `FuzzMatrix` 30/30 + Oracle 176/176 (the `dtype_text`
  tier unchanged), Interop 638/638.
* **Breaking (accepted):** `Type t = a.dtype` / `Type t = np.float64` need a cast; `a.dtype.Name` → `a.dtype.name` (NumPy) or
  `a.dtype.type.Name` (C#); `np.X` where a `Type` is required by signature → `np.X.type`; reflection over
  `np.array(Array, Type, …)` → `typeof(DType)`; `np.isdtype(x, "floating")` raises.

**Stage D phase 1 landed (2026-10-07, branch `strings`) — fixed-width `S` / `U` storage, byte-perfect with NumPy 2.4.2.**

* **Metas.** `StringDTypeMeta` (`DTypes/StringDTypeMeta.cs`) is NumPy's `BytesDType` (`'S'`, num 18, one byte per character)
  and `StrDType` (`'U'`, num 19, one UCS-4 code point per character), PARAMETRIC: the length lives on the descriptor
  (`S5` → itemsize 5, `U5` → 20), `CommonInstance` takes the longer one, `S + U → U`, and a number promotes to text of its
  `REQUIRED_STR_LEN` (bool 5, unsigned {1:3, 2:5, 4:10, 8:20}, signed +1, float 32, complex 64). `StringCastingImpls.cs`
  ports `string_to_string_resolve_descriptors` (longer = safe, shorter = same_kind, equal = no / a `U` byte swap = equiv)
  and the numeric → string length casts. The unsized descriptors (`np.dtype("U")`, `'S'`, itemsize 0) parse, print and
  promote, and adapt to the operand (`PyArray_AdaptDescriptorToArray`) wherever NumPy sizes them.
* **Storage.** One storage lane, `NPTypeCode.String`, holds every width of both classes: `StringArraySlice` is an element
  view over an ordinary `ArraySlice<byte>` (so pooling, ARC lifetime and views are the numeric lanes'), and the
  descriptor (`UnmanagedStorage.Descr`) carries the class and width. `NPTypeCode.String` alone therefore names NO dtype —
  every allocation of a string result takes a descriptor. The codec (`Backends/Strings/NDStringCodec.cs`) is NumPy's
  `STRING_/UNICODE_ getitem/setitem`: NUL padding, trailing NULs stripped on read, embedded NULs kept, a C# `string` is
  `str`, a `byte[]` is `bytes`, supplementary-plane text is one UCS-4 unit.
* **Movement.** `NDStringCopy` runs every string copy (`NDIter.Copy`, `copyto`, `astype`, slice assignment,
  concatenate, take/put/place/putmask, where, pad …) on the numeric kernels through UNIT VIEWS: a string array of shape
  `D` is copied as a `uint8` / `uint32` array of shape `D + (chars,)`, so broadcasting, every layout, the overlap guard
  and the SIMD copy kernels are the numeric ones; `S ↔ U` is NumPy's legacy ASCII cast with Python's
  `UnicodeEncodeError` / `UnicodeDecodeError` texts.
* **Printing.** `Backends/Printing/PyRepr.cs` is CPython's `unicode_repr` / `PyBytes_Repr` / `str.isprintable` / `len`,
  so `str` / `repr` / `array2string` of a string array are NumPy's bytes.
* **Computing on text is the next phase.** Where NumPy REFUSES a call (no string loop: `np.sum`, `np.sqrt`, gufuncs),
  NumSharp raises NumPy's exception and text; where NumPy COMPUTES (comparisons, concatenation, truth tests, sorting,
  searching, text ↔ number parsing), it throws `StringComputationNotSupportedException` (a `NotSupportedException`).
  Both come after NumPy's earlier checks, in NumPy's order (`Backends/Strings/NDStringUfuncs{,.Precedence}.cs`):
  the `dtype=` class check, a read-only `out=`, a non-bool `where=`, the axis (three rules — `ufunc.reduce`, the counting
  statistics, the ravelling arg / scan / sort family), the quantile family's q / axis / method / empty-axis order, the
  output cast and the broadcast of a computing loop. An unsized string `dtype=` request gets NumPy's per-ufunc answer
  (`No loop matching …`, the class-form `UFuncTypeError`, the string `add` formatting numbers as text, the string
  `multiply` 'out' text). The add family ignores a NUMERIC request over text (the string add loop resolves from the
  operands: `np.sum(u, dtype='f8')` is the add.reduce refusal), every other reduction parses (next phase).
* **`.npy`.** `|S5` / `<U3` / `>U3` files read and write byte for byte (mmap included). **Breaking:** a `<U1` file reads
  back as a `<U1` string array, not `Char`; `Char` is still written as `<U1`.
* **Gate.** The string oracle (`test/oracle/gen_strings_oracle.py` → `NumSharp.Tests.Oracle/Strings/corpus/strings.jsonl`,
  14,175 cases, 8,281 of them NumPy errors compared by type map + verbatim text; 14 sections: dtype, creation, astype,
  views, movement, mutation, access, printing, ufuncs, reductions, strdtype, strprec, straxis, strcov). A phase-2 case
  (NumPy returns, or raises a parse error) passes as "deferred" while NumSharp throws
  `StringComputationNotSupportedException` and is held to NumPy's answer the moment NumSharp computes it — the next
  phase is gated by the same corpus. Unit tests: `Backends/Strings/StringCallOrderTests.cs` (one per precedence rule),
  `Documentation/FundamentalsStringsBytesDocTests.cs`, the frombuffer / getfield / file I/O string tests.
* **Movement perf** (NPY/NS, two pinned P-cores, 100K / 1M `<U8`): copy 5.0 / 3.7×, strided / reversed / transposed
  copies 2.5–4.4×, `astype` width changes 2.7–4.2×, `S ↔ U` 44–86×, concatenate 3.3–3.6×, take / fancy / mask 1.7–3.1×,
  `np.where` 3.7–5.8×. Same-layout copies are one block move; non-overlapping copies over at most two effective axes run
  the `ElementCopy` IL kernel (constant-size cpblk + initblk per element); `np.where` runs the `WhereBytes` kernel;
  integer-index and prefix-mask gathers take the fancy-index kernel route; fully-written `U` results skip the NEEDS_INIT
  zero fill. Floors: 1K calls (the per-call NDArray floor), `copyto` / fill (memory-bound), `AsStringArray` at 1M
  (.NET GC of live strings).
* **Traps found.**
  - NumPy's ufunc DISPATCH CACHE makes an answer depend on call history: after
    `np.multiply(u, int8, dtype='U')` (whose promoter casts int8 → int64 and is cached per DType tuple),
    `np.prod(int8_array, dtype='U')` finds the string repeat loop ('out' kwarg text), where a fresh process names no
    loop. The generator runs the reductions before the binary ufuncs, so the corpus records the fresh answer.
  - A test corpus of "string" cases must not include numeric-only calls: they test numeric behavior (reduction 0-d
    results are read-only reduction scalars in NumSharp, NumPy's generator sees a writeable 0-d array).
  - `GetString` / `SetString` (the `Char` vector API) used to `Debug.Assert` the dtype, so a Release build read a `U`
    array's UCS-4 bytes as UTF-16; they refuse a non-`Char` array now.

**Not yet:** datetime storage/kernels/NaT/ISO (C — `docs/plans/datetime64.md`); Stage D phase 2 (computing on text —
comparisons, `np.strings`, sorting, `add`/`multiply` loops, text ↔ number casts, the gated corpus is ready) and NEP 55
`StringDType` (phase 3).
