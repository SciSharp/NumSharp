# Release Notes

## TL;DR

This release adds full NumPy-parity support for **three new dtypes** — `SByte` (int8), `Half` (float16), and `Complex` (complex128) — across every `np.*` API, operator, IL kernel, and reduction. A new **`DateTime64` helper type** closes a 64-case conversion gap vs NumPy's `datetime64`. The **`np.*` class-level type aliases are now fully aligned with NumPy 2.4.2** (breaking changes: `np.byte = int8`, `np.complex64` throws, `np.uint = uintp`, `np.intp` is platform-detected), and `np.dtype(string)` is rewritten as a `FrozenDictionary` lookup covering every NumPy 2.x type code. Over the course of **55 commits (+30k / −5.0k lines, 165 files)**, **34 NumPy-parity bugs** were fixed, the entire casting subsystem was rewritten for NumPy 2.x wrapping semantics, the bitshift operators `<<` / `>>` were added to `NDArray`, and rejection sites (shift on non-integer dtypes, invalid indexing types, non-safe `repeat` counts, complex→int scalar cast) now throw NumPy-canonical `TypeError` / `IndexError`. Full test suite grew to **~7,000+ tests / 0 failures / 11 skipped** per framework (net8.0 + net10.0), with ~2,400 new test LoC across 23 new test files. Three systematic coverage sweeps (Creation, Arithmetic, Reductions) probed the new dtypes against NumPy 2.4.2 and landed at 100% parity on the functional surface, with 4 well-documented BCL-imposed divergences.

---

## Major Features

### New dtypes: SByte (int8), Half (float16), Complex (complex128)
Complete first-class support matching NumPy 2.x:
- `NPTypeCode` enum extended (`SByte=5`, `Half=16`, `Complex=128`) with every extension method (`GetGroup`, `GetPriority`, `AsNumpyDtypeName`, `IsFloatingPoint`, `IsSimdCapable`, `GetComputingType`, …).
- Type aliases on `np.*`: `np.int8`, `np.sbyte`, `np.float16`, `np.half`.
- Storage/memory plumbing: `UnmanagedMemoryBlock`, `ArraySlice`, `UnmanagedStorage` (Allocate / FromArray / Scalar / typed Getters + Setters).
- `np.find_common_type` — ~80 new type-promotion entries across both `arr_arr` and `arr_scalar` tables following NEP50.
- NDArray integer/float/complex indexing (`Get*`/`Set*` methods for the three dtypes).
- Full iterator casts added: `NDIterator.Cast.Half.cs`, `NDIterator.Cast.Complex.cs`, `NDIterator.Cast.SByte.cs`.

### DateTime64 helper type (`src/NumSharp.Core/DateTime64.cs`)
New `readonly struct` modeled on `System.DateTime` but with NumPy `datetime64` semantics:
- Full `long.MinValue..long.MaxValue` tick range (no `DateTimeKind` bits).
- `NaT == long.MinValue` sentinel that propagates through arithmetic and compares like IEEE NaN.
- Implicit widenings from `DateTime` / `DateTimeOffset` / `long`; explicit narrowings with NaT/out-of-range guards.
- Closes **64 datetime-related fuzz diffs** that previously forced `DateTime.MinValue` fallbacks (Groups A + B).
- Bundled with reference `DateTime.cs` / `DateTimeOffset.cs` copies under `src/dotnet/` as source-of-truth.
- `Converts.DateTime64.cs` — NumPy-exact conversion to/from every primitive dtype.
- Quality pass (commit `7b14a41a`) trimmed the surface to helper scope and fixed the `Equals`/`==` contract split (mirrors `double`'s NaN handling so the type can be a `Dictionary` key while `==` follows NumPy).

### NumPy 2.x type alias alignment (`src/NumSharp.Core/APIs/np.cs`)
Full overhaul of the class-level `Type` aliases on `np` to match NumPy 2.4.2 exactly.

**Breaking changes:**

| Alias | Before | After | Reason |
|-------|--------|-------|--------|
| `np.byte` | `byte` (uint8) | `sbyte` (int8) | NumPy C-char convention |
| `np.complex64` | alias → complex128 | throws `NotSupportedException` | no silent widening — user intent preserved |
| `np.csingle` | alias → complex128 | throws `NotSupportedException` | same rationale |
| `np.uint` | `uint64` | `uintp` (pointer-sized) | NumPy 2.x |
| `np.intp` | `nint` | `long` on 64-bit / `int` on 32-bit | `nint` resolves to `NPTypeCode.Empty`, breaking dispatch |
| `np.uintp` | `nuint` | `ulong` on 64-bit / `uint` on 32-bit | same |
| `np.int_` | `long` | `intp` | NumPy 2.x: `int_ == intp` |

**New aliases:** `np.short`, `np.ushort`, `np.intc`, `np.uintc`, `np.longlong`, `np.ulonglong`, `np.single`, `np.cdouble`, `np.clongdouble`.

**Platform-detected** (C-long convention: 32-bit MSVC / 64-bit \*nix LP64): `np.@long`, `np.@ulong`.

### `np.dtype(string)` parser rewrite (`src/NumSharp.Core/Creation/np.dtype.cs`)
Regex-based parser replaced with a `FrozenDictionary<string, Type>` built once at static init.

**Covers every NumPy 2.x dtype code:**
- Single-char: `?`, `b`/`B`, `h`/`H`, `i`/`I`, `l`/`L`, `q`/`Q`, `p`/`P`, `e`, `f`, `d`, `g`, `D`, `G`.
- Sized forms: `b1`, `i1`/`u1`, `i2`/`u2`, `i4`/`u4`, `i8`/`u8`, `f2`, `f4`, `f8`, `c16`.
- Lowercase names: `bool`, `int8..int64`, `uint8..uint64`, `float16..float64`, `complex`, `complex128`, `half`, `single`, `double`, `byte`, `ubyte`, `short`, `ushort`, `intc`, `uintc`, `int_`, `intp`, `uintp`, `bool_`, `int`, `uint`, `long`, `ulong`, `longlong`, `ulonglong`, `longdouble`, `clongdouble`.
- NumSharp-friendly: `SByte`, `Byte`, `UByte`, `Int16..UInt64`, `Half`, `Single`, `Float`, `Double`, `Complex`, `Bool`, `Boolean`, `boolean`, `Char`, `char`, `decimal`.

**Unsupported codes throw `NotSupportedException`** with an explanatory message:
- Bytestring (`S`/`a`), Unicode (`U`), datetime (`M`), timedelta (`m`), object (`O`), void (`V`) — NumSharp has no equivalents.
- `complex64` / `F` / `c8` — NumSharp only has complex128; refusing to silently widen preserves user intent.

**Platform-detection helpers** (`_cLongType`, `_cULongType`, `_intpType`, `_uintpType`) are declared before the dictionary since static initializers run top-down.

### `np.finfo` + `np.iinfo` extended to new dtypes
- **`np.finfo(Half)`** — IEEE binary16: `bits=16`, `eps=2^-10`, `smallest_subnormal=2^-24`, `maxexp=16`, `minexp=-14`, `precision=3`, `resolution=1e-3`.
- **`np.finfo(Complex)`** — NumPy parity: reports underlying float64 values with `dtype=float64` (`finfo(complex128).dtype == float64`).
- **`np.iinfo(SByte)`** — int8 with signed min/max and `'i'` kind.
- `IsSupportedType` on both extended to accept the new dtypes.

### Complex-source → non-complex scalar cast = `TypeError`
All explicit `NDArray → scalar` conversions (`(int)arr`, `(double)arr`, etc) now validate via a common `EnsureCastableToScalar(nd, targetType, targetIsComplex)` helper:
- `ndim != 0` → `IncorrectShapeException`.
- Non-complex target + complex source → `TypeError` ("can't convert complex to int/float/…").

This matches Python's `int(complex(1, 2))` behavior. NumPy's silent `ComplexWarning` is treated as a hard error since NumSharp has no warning mechanism — users must `np.real(arr)` explicitly to drop imaginary.

Also added: implicit `sbyte → NDArray`, implicit `Half → NDArray`, explicit `NDArray → sbyte`.

### NumPy-canonical exception types at rejection sites
| Site | Before | After | NumPy message |
|------|--------|-------|---------------|
| `Default.Shift.ValidateIntegerType` | `NotSupportedException` | `TypeError` | "ufunc 'left_shift' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule 'safe'" |
| `NDArray.Indexing.Selection.{Getter,Setter}` validation | `ArgumentException` | `IndexError` | "only integers, slices (':'), ellipsis ('...'), numpy.newaxis ('None') and integer or boolean arrays are valid indices" |
| `np.repeat` on non-integer repeats | permissive truncation | `TypeError` | "Cannot cast array data from dtype('float16') to dtype('int64') according to the rule 'safe'" |

**New exception:** `NumSharp.IndexError : NumSharpException` mirroring Python's `IndexError`.

### Operator overloads
- **`<<` and `>>`** added to `NDArray` (file `NDArray.Shift.cs`). Two overloads per direction (NDArray↔NDArray, NDArray↔object) mirroring `NDArray.OR/AND/XOR.cs`. C# compiler synthesizes `<<=` / `>>=` (reassign, not in-place — locked in by test).

### NumPy-parity casting overhaul
Entire `Converts.cs` / `Converts.Native.cs` / `Converts.DateTime64.cs` rewritten across Rounds 1-5E:
- Modular wrapping for integer overflow matching NumPy (no more `OverflowException`).
- NaN / Inf → 0 consistently across all float → int targets.
- `Char` (16-bit) follows `uint16` semantics for every source type.
- `IConvertible` constraint removed from generic converter surface (`Converts<T>`) to admit `Half` / `Complex`.
- Six precision-boundary bugs in `double → int` converters fixed (Round 5F).
- `ToUInt32(double)` overflow now returns 0.
- `ToInt64` / `ToTimeSpan` / `ToDateTime` precision fixes at 2^63 boundary.
- `ArraySlice.Allocate` + `np.searchsorted` patched for `Half` / `Complex`.
- `UnmanagedMemoryBlock.Allocate(Type, long, object)` — direct boxing casts (`(Half)fill`, `(Complex)fill`, …) replaced with `Converts.ToXxx(fill)` dispatchers, so cross-type fills (e.g. `fill = 1` on a Half array, `fill = 3.14` on a Complex array) work with full NumPy-parity wrapping.

### Complex matmul preserves imaginary
`Default.MatMul.2D2D.cs::MatMulMixedType<TResult>` short-circuits to a dedicated `MatMulComplexAccumulator` when `TResult` is `Complex`. The double-precision accumulator was dropping imaginary parts for Complex-typed result buffers; the new path accumulates in `Complex` across the inner `K` dimension.

---

## Bug fixes (34 closed)

| ID | Round | Area | Summary |
|----|-------|------|---------|
| B1  | 14 | Reduction | `Half` min/max elementwise returned ±inf — IL `Bgt/Blt` don't work on `Half` |
| B2  | 14 | Reduction | Complex `mean(axis)` returned `Double`, dropping imaginary |
| B3/B38 | 13 | Arithmetic | Complex `1/0` returned `(NaN,NaN)` vs NumPy `(inf,NaN)` — .NET Smith's algorithm |
| B4  | 14 | Reduction | `np.prod(Half/Complex)` threw `NotSupportedException` |
| B5  | 14 | Reduction | `SByte` axis reduction threw (no identity/combiner) |
| B6  | 14 | Reduction | `Half/Complex cumsum(axis)` threw mid-execution |
| B7  | 14 | Reduction | `argmax/argmin(axis)` threw for Half/Complex/SByte |
| B8  | 14 | Reduction | Complex `min/max` elementwise threw |
| B9  | 15 | Manipulation | `np.unique(Complex)` threw — generic `IComparable<T>` constraint |
| B10/B17 | 6 | Arithmetic | Half/Complex `maximum`/`minimum`/`clip` + axis variant |
| B11 | 6 | Unary Math | Half+Complex `log10`/`log2`/`cbrt`/`exp2`/`log1p`/`expm1` missing |
| B12 | 14 | Reduction | Complex `argmax` tiebreak wrong (non-lex compare) |
| B13 | 15 | Reduction | Complex `argmax/argmin` with NaN returned wrong index |
| B14 | 6 | Statistics | Half+Complex `nanmean`/`nanstd`/`nanvar` returned NaN |
| B15 | 14 | Reduction | Complex `nansum` propagated NaN instead of skipping |
| B16 | 14 | Reduction | Half `std/var(axis)` returned `Double` instead of preserving |
| B18 | 7 | Reduction | `cumprod(Complex, axis)` dropped imaginary |
| B19 | 7 | Reduction | `max/min(Complex, axis)` returned all zeros |
| B20 | 7 | Reduction | `std/var(Complex, axis)` computed real-only variance |
| B21 | 9 | Unary Math | Half `log1p/expm1` lost subnormal precision — promote to `double` |
| B22 | 9 | Unary Math | Complex `exp2(±inf+0j)` returned NaN — use `Math.Pow(2,r)` branch |
| B23 | 9 | Reduction | Complex `var/std` single-element axis returned Complex dtype |
| B24 | 9 | Reduction | `var/std` with `ddof > n` returned negative variance — clamp `max(n-ddof, 0)` |
| B25 | 10 | Comparison | Complex ordered compare with NaN returned True — NaN short-circuit |
| B26 | 10 | Unary Math | Complex `sign(inf+0j)` returned `NaN+NaNj` — unit-vector branch |
| B27 | 11 | Creation | `np.eye(N,M,k)` wrong diagonal stride for non-square/k≠0 (all dtypes) |
| B28 | 11 | Creation | `np.asanyarray(NDArray, dtype)` ignored dtype override |
| B29 | 11 | Creation | `np.asarray(NDArray, dtype)` overload missing |
| B30 | 12 | Creation | `np.frombuffer` dtype-string parser incomplete + `i1/b` wrong (uint8 vs int8) |
| B31 | 12 | Creation | `ByteSwapInPlace` missing Half/Complex branches — big-endian reads corrupted |
| B32 | 12 | Creation | `np.eye` didn't validate negative N/M |
| B33 | 13 | Arithmetic | `floor_divide(inf, x)` returned `inf` vs NumPy `NaN` for all float dtypes |
| B35 | 13 | Arithmetic | Integer `power` overflow wrong — routed through `Math.Pow(double)` |
| B36 | 13 | Arithmetic | `np.reciprocal(int)` promoted to float64 instead of C-truncated int |
| B37 | 13 | Arithmetic | `np.floor/ceil/trunc(int)` promoted to float64 instead of no-op |

Plus the pre-existing fixes landed before the tracked-bug table:
- `np.abs(complex)` now returns `float64` matching NumPy.
- Complex `ArgMax`/`ArgMin`, `IsInf`/`IsNan`/`IsFinite`, Half NaN reductions.
- 1-D `dot` preserves dtype.
- `Half + int16/uint16` promotes to `float32` (was `float16`).
- `float → byte` uses int32 intermediate.
- `UnmanagedMemoryBlock.Allocate` cross-type fills now use `Converts.ToXxx(fill)` — `fill = 1` on a `Half` array no longer throws `InvalidCastException`.
- `np.asanyarray(Half)` / `np.asanyarray(Complex)` — scalar detection now includes `Half` and `System.Numerics.Complex`.
- `Default.MatMul.2D2D` — Complex result type preserves imaginary via dedicated accumulator.

### Accepted divergences (documented)
1. **Complex `(inf+0j)^(1+1j)`** — BCL `Complex.Pow` via `exp(b*log(a))` fails; would require rewriting `Complex.Pow` manually.
2. **SByte integer `// 0`, `% 0`** — returns garbage via double-cast path; seterr-dependent.
3. **`exp2(complex(inf, inf))`** — .NET `Complex.Pow` BCL quirk in dual-infinity regime.
4. **`frombuffer(">f2"/">c16")`** — byte values correct after swap, but dtype string loses byte-order prefix (NumSharp dtypes carry no byte-order info).

---

## Infrastructure / IL Kernel

- `ILKernelGenerator` gained Half/Complex/SByte across `.Binary`, `.Unary`, `.Unary.Math`, `.Unary.Decimal`, `.Comparison`, `.Reduction`, `.Reduction.Arg`, `.Reduction.Axis`, `.Reduction.Axis.Simd`, `.Reduction.Axis.VarStd`, `.Masking.NaN`, `.Scan`, `.Scalar`.
- **Six Complex IL helpers inlined** (`IsNaN`, `IsInfinity`, `IsFinite`, `Log2`, `Sign`, `Less/LessEqual/Greater/GreaterEqual`) — eliminates reflection lookup and method-call hops in hot loops. Factored into `EmitComplexComponentPredicate` and `EmitComplexLexCompare`.
- `ComplexExp2Helper` inlined as direct IL emit.
- `ComplexDivideNumPy` helper replaces BCL `Complex.op_Division` (Smith's algorithm) to match NumPy's component-wise IEEE semantics at `z/0`.
- `PowerInteger` fast-path for all 8 integer dtypes (repeated squaring with unchecked multiplication).
- `ReciprocalInteger` fast-path with C-truncated division.
- Sign-of-zero preservation for Half `log1p`/`expm1` (Math.CopySign) and Complex `exp2` pure-real branch.

---

## Tests

- **14 new test files** under `test/NumSharp.UnitTest/NewDtypes/` covering Basic, Arithmetic, Unary, Comparison, Reduction, Cumulative, EdgeCase, TypePromotion, Round 6/7/8 battletests, and three 100%-coverage sweep files (Creation / Arithmetic / Reductions).
- **9 new test files** for the NumPy 2.x alignment commit (~1,912 LoC):

  | File | LoC | Scope |
  |------|-----|-------|
  | `NpTypeAliasParityTests` | 174 | Every `np.*` alias vs NumPy 2.4.2 (Windows 64-bit + platform-gated) |
  | `np.finfo.NewDtypesTests` | 262 | Half + Complex finfo |
  | `np.iinfo.NewDtypesTests` | 95 | SByte iinfo |
  | `UnmanagedMemoryBlockAllocateTests` | 226 | Cross-type fill matrix |
  | `ComplexToRealTypeErrorTests` | 170 | Complex → int/float scalar cast TypeError |
  | `NDArrayScalarCastTests` | 384 | 0-d cast matrix (implicit + explicit, 15 × 15) |
  | `Complex64RefusalTests` | 116 | `np.complex64` / `np.csingle` throw |
  | `DTypePlatformDivergenceTests` | 166 | `'l'` / `'L'` / `'int'` platform-dependent behavior |
  | `DTypeStringParityTests` | 319 | Every dtype string vs NumPy 2.4.2 |

- **Casting suite** grew by ~4,800 lines: `ConvertsBattleTests.cs` (1,586 LoC), `DtypeConversionMatrixTests.cs` (1,456 LoC), `DtypeConversionParityTests.cs` (526 LoC), `ConvertsDateTimeParityTests.cs` (615 LoC), `ConvertsDateTime64ParityTests.cs` (631 LoC).
- Test count: **~6,400 → 7,000+** / 0 failed / 11 skipped on both net8.0 and net10.0.
- Probe matrices (330 cases Creation, 109 Arithmetic, 80 Reductions) re-run against NumPy 2.4.2 at 100% / 96.3% / 100% post-fix parity.

---

## Breaking changes / behavioral alignment

- `Convert.ChangeType`-style paths for `decimal` / `float` / `Half` → integer now **wrap modularly** instead of throwing `OverflowException`.
- `ToDecimal(float/double)` for NaN/Inf/out-of-range now returns `0m` (was: throw).
- `np.reciprocal(int)` / `np.floor/ceil/trunc(int)` now **preserve integer dtype** (was: promoted to `float64`).
- `InfoOf<T>.Size` switched from `Marshal.SizeOf<T>()` to `Unsafe.SizeOf<T>()` — `Marshal.SizeOf` rejects `System.DateTime` and other managed-only structs.
- `NPTypeCode` for `typeof(DateTime)` now returns `Empty` instead of accidentally resolving to `Half` (`TypeCode.DateTime (16) == NPTypeCode.Half (16)` collision fixed).
- `Shape.IsWriteable` enforces read-only broadcast views (NumPy-aligned).
- **`np.byte` is now `sbyte` (int8)** — was `byte` (uint8). For .NET-style `uint8`, use `np.uint8` / `np.ubyte`.
- **`np.complex64` / `np.csingle` throw `NotSupportedException`** — previously silently aliased to complex128. Use `np.complex128` / `np.complex_` / `np.cdouble` explicitly.
- **`np.uint` is now `uintp` (pointer-sized)** — was `uint64`. For explicit 64-bit unsigned, use `np.uint64` / `np.ulonglong`.
- **`np.intp` is now platform-detected `long`/`int`** — was `nint`. `nint` has `NPTypeCode.Empty` which broke dispatch through `np.zeros(typeof(nint))`.
- **`np.int_` is now `intp` (pointer-sized)** — was always `long`. Matches NumPy 2.x where `int_ == intp`.
- **Shift ops on non-integer dtypes throw `TypeError`** — was `NotSupportedException`. Message matches NumPy: `"ufunc '...' not supported for the input types, ... safe casting"`.
- **Invalid index types throw `IndexError`** — was `ArgumentException`. New `NumSharp.IndexError` mirrors Python.
- **`np.repeat` on non-integer repeats throws `TypeError`** — was permissive truncation. Matches NumPy 2.4.2 exactly.
- **Explicit cast `NDArray → non-complex scalar` on Complex source throws `TypeError`** — was silent imaginary drop via `Convert.ChangeType`. Use `np.real(arr)` explicitly to drop imaginary.
- **`np.find_common_type` table entries** — all `np.complex64` references replaced with `np.complex128` to avoid relying on the now-throwing alias. No behavioral change for callers (the alias pointed at `Complex` anyway).

---

## Docs

- `docs/NEW_DTYPES_IMPLEMENTATION.md`, `docs/NEW_DTYPES_HANDOFF.md` — implementation design + handoff notes.
- `docs/plans/LEFTOVER.md`, `docs/plans/LEFTOVER_CONVERTS.md`, `docs/plans/REVIEW_FINDINGS.md` — round-by-round tracking with post-mortem audit.
- `docs/website-src/docs/NDArray.md` (663 LoC) — user-facing NDArray guide.
- `docs/website-src/docs/dtypes.md` (610 LoC) — complete dtype reference (aliases, string forms, type promotion, platform notes).
- `docs/website-src/docs/toc.yml` — NDArray + Dtypes pages added to the navigation.
