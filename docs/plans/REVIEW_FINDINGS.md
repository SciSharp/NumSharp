# worktree-half Review Findings

Systematic file-by-file review of the 97 files changed on `worktree-half` branch.
Compared against merge-base `70210083` (merge PR #609 from `worktree-mstests`).

Legend: ✅ OK  |  ⚠️ minor concern  |  🐛 bug  |  📝 missing tests  |  ❓ needs verification

---

## Resolved Findings (addressed 2026-04-18)

### 🔧 np.dtype(string) — full NumPy 2.x parity rewrite

**Problem:** The pre-existing parser had ~35 NumPy-parity bugs across single-char codes,
sized variants, and named forms. Examples:
- `np.dtype("b")` returned Byte (NumPy: int8/SByte)
- `np.dtype("B")` **threw** (NumPy: uint8/Byte)
- `np.dtype("i1")` returned Byte (NumPy: int8)
- `np.dtype("u1")` returned UInt16 (NumPy: uint8)
- `np.dtype("uint8")` returned UInt64 (regex matched "uint"+"8")
- Most single-char codes (`h`, `H`, `I`, `l`, `L`, `q`, `Q`, `g`, `F`, `D`, `G`, `p`, `P`) threw
- `np.dtype("c")` returned Complex (NumPy: S1, 1-byte string — now NotSupportedException)
- `np.dtype("S")` / `"U"` returned Char (NumPy: bytestring/unicode — now NotSupportedException)

**Fix:** `src/NumSharp.Core/Creation/np.dtype.cs` — replaced regex-based parser with
`FrozenDictionary<string, Type>` lookup. Covers every valid NumPy 2.x dtype string
(143 map entries), rejects invalid/unsupported forms, handles byte-order prefixes.

**Tests:** `test/NumSharp.UnitTest/Creation/DTypeStringParityTests.cs` — 153 tests,
each expectation cross-checked against `python -c "import numpy as np; np.dtype('...')"`.
Updated existing `np.dtype.Test.cs` to match NumPy parity. Also fixed
`np.finfo.BattleTest.cs::FInfo_String_Float` (was expecting 32-bit; NumPy: 64-bit).

**Adaptations from NumPy:**
- Complex64 ('F', 'c8', 'complex64') widens to NumSharp's Complex (complex128).
- 'l'/'L' and 'int'/'uint' match Windows NumPy (C long → int32).
- Accepts .NET PascalCase aliases (SByte, Byte, Int16, ..., Half, Complex).

### 🔧 NDArray cast operators — sbyte/Half/Complex

**Problem:** `NdArray.Implicit.ValueTypes.cs` had 13 existing scalar casts but was
missing `sbyte`, `Half`, `Complex` explicit-from-NDArray operators. Also missing implicit
`sbyte → NDArray` and `Half → NDArray` operators.
Users could not write `(Half)nd[0]`, `(Complex)nd[0]`, `(sbyte)nd[0]`.

**Fix:** Added 5 operators (2 implicit scalar→NDArray, 3 explicit NDArray→scalar).
All explicit operators require `ndim == 0` and throw `IncorrectShapeException` otherwise
(matches NumPy 2.x strict — even single-element 1-d/2-d arrays throw, per
`"only 0-dimensional arrays can be converted to Python scalars"`).

**Tests:** `test/NumSharp.UnitTest/Casting/NDArrayScalarCastTests.cs` — 40 tests covering:
- Implicit scalar → NDArray (all 3 new types)
- Explicit NDArray → scalar round-trips
- Boundary values (sbyte MinValue/MaxValue, Half NaN/±Inf, Complex zero/one/imaginary)
- Cross-type conversion (int→Half, Complex→Half drops imaginary, etc.)
- ndim validation (1-d single-element still throws, 2-d (1,1) still throws)
- 2-D indexing round-trips
- Composition with arithmetic

**Test totals:**
- 153 dtype parity tests (new) + 40 cast tests (new) + 4 finfo tests (new/fixed) = **197 new tests**
- Full project test suite: **6271 passed, 0 failed, 11 skipped** (both net8.0 + net10.0)

### 🔧 UnmanagedMemoryBlock.Allocate(count, fill) — fixed

Previously used direct casts like `(Half)fill` which throw `InvalidCastException`
if `fill` is boxed as the wrong type (e.g. `Allocate(Half, 10, 42)` where `42` is boxed int).
Now routes every dtype through `Converts.ToXxx(fill)` — same pattern as sibling
`ArraySlice.Allocate`. Supports cross-type fills per NumPy's casting rules.

**Tests:** `test/NumSharp.UnitTest/Backends/Unmanaged/UnmanagedMemoryBlockAllocateTests.cs` —
24 tests covering: same-type fill, cross-type fills (int→Half, double→Half, Half→Complex,
Half→Int32, Complex→Double), boundary values (SByte MinValue/MaxValue), NaN/Inf preservation.

### 🔧 np.finfo(Half) / np.finfo(Complex) — fixed

**Problem:** `np.finfo(NPTypeCode.Half)` and `np.finfo(NPTypeCode.Complex)` threw
`"not inexact"` — `IsFloatType` in `np.finfo.cs:164` only allowed Single/Double/Decimal.

**Fix:** Added Half and Complex cases with NumPy-parity machine constants:
- Half: bits=16, eps=2^-10, epsneg=2^-11, max=65504, smallest_normal=2^-14, smallest_subnormal=2^-24, precision=3, resolution=1e-3, maxexp=16, minexp=-14.
- Complex: reports underlying float64 precision per NumPy convention (bits=64, dtype=Double, all values match float64). This is the NumPy behavior — `np.finfo(np.complex128).dtype == np.float64`.

**Tests:** `test/NumSharp.UnitTest/APIs/np.finfo.NewDtypesTests.cs` — 42 tests covering
each machine-limit field, all 5 constructor overloads (NPTypeCode, Type, generic,
NDArray, string), string aliases (float16/half/e/f2 and complex128/complex/D/c16),
plus negative tests that integer dtypes still throw.

### 🔧 np.iinfo(SByte) — fixed

**Problem:** `np.iinfo(NPTypeCode.SByte)` threw — `IsIntegerType` was missing the SByte case.

**Fix:** Added SByte to `IsIntegerType` and to `GetTypeInfo` with bits=8, min=-128,
max=127, kind='i'.

**Tests:** `test/NumSharp.UnitTest/APIs/np.iinfo.NewDtypesTests.cs` — 16 tests covering
all constructor overloads, string aliases (int8/sbyte/b/i1), and negative tests that
Half and Complex still throw.

### 📋 Net test count across all fixes

| File | Tests |
|---|---|
| `DTypeStringParityTests.cs` | 153 |
| `NDArrayScalarCastTests.cs` | 40 |
| `np.finfo.BattleTest.cs` (updated + 2 new) | +3 |
| `np.finfo.NewDtypesTests.cs` | 42 |
| `np.iinfo.NewDtypesTests.cs` | 16 |
| `UnmanagedMemoryBlockAllocateTests.cs` | 24 |
| **Total new/changed** | **~278 tests** |

Test suite: **6353 pass, 0 fail** (net8.0 + net10.0).

---

## Round 2 fixes (2026-04-18, user-directed)

### 🔧 Reject complex64 outright (no silent widening)

**Before:** NumSharp silently widened `np.complex64` / `"c8"` / `"F"` / `"complex64"` to `Complex` (complex128). This hid user intent — someone wanting 32-bit precision would unknowingly get 64-bit.

**After:**
- `np.complex64` — now a computed property that throws `NotSupportedException` with guidance to use `np.complex128`.
- `np.dtype("complex64")` / `"c8"` / `"F"` → throw `NotSupportedException` via `_unsupported_numpy_codes` set.
- `np.dtype("complex128")` / `"D"` / `"c16"` / `"complex"` / `"G"` (long-double complex collapses to 128) → still work.

**Internal callers:** `find_common_type.cs` had ~58 references to `np.complex64` (as alias for Complex). All rewritten to `np.complex128` so internal lookups still succeed.

**Tests:** `test/NumSharp.UnitTest/Creation/Complex64RefusalTests.cs` — 10 tests covering direct access, dtype strings, finfo strings, and positive cases for `complex128`/`D`/`c16`/`complex`/`G`.

### 🔧 Platform-dependent int dtype clarification + fix

**Was incorrect before:** I claimed `"int"` → Int32 as "Windows convention". That was wrong per NumPy 2.4.2.

**Actual NumPy 2.x behavior** (verified against `python -c "np.dtype(...)"` on Windows 64-bit):

| Spelling | Win 64 | Linux 64 | Explanation |
|---|---|---|---|
| `int_`, `intp`, `int`, `p`, `P` | int64/uint64 | int64/uint64 | NumPy 2.x made these pointer-sized |
| `longlong`, `q`, `Q` | int64/uint64 | int64/uint64 | C `long long` always 64-bit |
| **`long`, `l`, `L`, `ulong`** | **int32/uint32** | **int64/uint64** | **C `long` differs: MSVC=32, gcc LP64=64** |
| `i`, `I`, `i4`, `u4` | int32/uint32 | int32/uint32 | fixed per NumPy spec |

**Fix:** `src/NumSharp.Core/Creation/np.dtype.cs` — introduced `_cLongType`/`_cULongType` (platform-detected via `RuntimeInformation.IsOSPlatform(OSPlatform.Windows)`) and `_intpType`/`_uintpType` (via `IntPtr.Size == 8`). Remapped `"int"` → intp (was Int32), `"long"`/`"l"` → C long (platform-dependent), kept `"longlong"`/`"q"` as always-64-bit.

**Tests:** `test/NumSharp.UnitTest/Creation/DTypePlatformDivergenceTests.cs` — 22 tests, each asserting the expected dtype per-platform via runtime detection. Runs green on Windows and should remain correct on Linux/Mac once CI tests them.

### 🔧 Complex → non-Complex scalar cast throws TypeError

**Before:** `(int)complex_nd` / `(Half)complex_nd` / `(double)complex_nd` silently discarded imaginary via `Converts.ChangeType`. No warning, no signal.

**After:** All 14 non-Complex explicit cast operators on `NDArray` call a new `EnsureCastableToScalar(...)` helper that:
- Checks `ndim == 0` (as before)
- If the target is non-Complex, rejects Complex-typed source arrays with `TypeError("can't convert complex to {type}")` — matches Python's `int(complex)` / `float(complex)` semantics

**Rationale:** NumPy 2.x emits `ComplexWarning` and silently drops imaginary, but NumSharp has no warning mechanism. Treating NumPy's warning as a hard error is the strict NumPy-parity interpretation. Users who actually want the real part should call `np.real(nd)` before casting.

**Applies to:** bool, sbyte, byte, short, ushort, int, uint, long, ulong, char, float, double, decimal, Half — 14 operators guard against Complex source.

**Does NOT apply to:**
- Complex → Complex (identity, always OK)
- Any non-Complex → Complex (widening, always OK)
- `nd.astype(real)` (array-level cast — separate code path, unchanged for now; matches NumPy's silent-drop behavior)

**Tests:** `test/NumSharp.UnitTest/Casting/ComplexToRealTypeErrorTests.cs` — 25 tests covering:
- Complex → each of 14 real types throws
- Zero-imaginary still throws (NumPy: `int(3+0j)` throws too)
- Complex → Complex identity works
- Real → Complex widening still works (for int, sbyte, Half, double)
- Shape guard still fires before type guard (1-d Complex → int throws IncorrectShapeException first)

### 📋 Final net test count + suite status

| File | Tests |
|---|---|
| `DTypeStringParityTests.cs` | 156 |
| `DTypePlatformDivergenceTests.cs` | 22 |
| `Complex64RefusalTests.cs` | 10 |
| `NDArrayScalarCastTests.cs` | 47 |
| `ComplexToRealTypeErrorTests.cs` | 25 |
| `np.finfo.NewDtypesTests.cs` | 43 |
| `np.iinfo.NewDtypesTests.cs` | 16 |
| `UnmanagedMemoryBlockAllocateTests.cs` | 24 |
| `np.finfo.BattleTest.cs` (updated) | +3 |
| `find_common_type.Test.cs` (c8 → c16) | updated |
| `np.iinfo.BattleTest.cs` (int → intp) | updated |
| **Total new/changed** | **~345 tests** |

Test suite: **6420 pass, 0 fail, 11 skip** (net8.0 + net10.0).

---

## Phase 1: Core type system (6 files)

### 1. `src/NumSharp.Core/Backends/NPTypeCode.cs`  ✅ (with bug-fixes to pre-existing issues)

- Added `SByte = 5` (int8), `Half = 16` (float16), fixed `Complex = 128` docstring.
- **Pre-existing bug fixed:** `IsNumerical` had `val == 129` (Complex is 128, not 129).
- **Pre-existing bug fixed:** `NPY_BYTELTR` was wrongly mapped to `Byte`; NumPy's 'b' = int8 = SByte. Now correct.
- **Pre-existing bug fixed:** `NPY_UBYTELTR` was wrongly mapped to `Char`; NumPy's 'B' = uint8 = Byte. Now correct.
- **Pre-existing bug fixed:** `NPY_HALFLTR` ('e') fell through to Single. Now returns Half.
- **Pre-existing bug fixed:** Complex's `AsNumpyDtypeName()` returned `"complex64"` — `System.Numerics.Complex` is two float64 = `complex128`. Fixed.
- Switch coverage added for all 12 + new 3 types across: `AsType`, size lookup, `IsFloatingPoint`, `IsInteger`, `IsSigned`, priority table, power order, `GetDefault`, `GetOne`, `IsSimdCapable`, `GetComputingType`.
- `GetComputingType(SByte) = Int64` matches NumPy NEP50. `GetComputingType(Half) = Half` (NumPy preserves float16 for sum). `GetComputingType(Complex) = Complex` ✓.
- `IsSimdCapable`: SByte=true (has `Vector<sbyte>`), Half=false (no `Vector<Half>` in .NET), Complex=false ✓.
- ❓ Pre-existing oddity: `NPY_CFLOATLTR` ('F'=complex64) still maps to `Single` (should be Complex fallback) — not this branch's concern.
- ❓ `Byte` in `powerOrder` still returns 0 (unchanged pre-existing issue, alongside String/Char=0). Unrelated.

### 2. `src/NumSharp.Core/Utilities/InfoOf.cs`  ✅

- Size switch: SByte=1, Half=2 added. Complex falls through to default `Marshal.SizeOf<T>()` (= 16 at runtime — verified).
- Zero uses `default(T)` — works for all 15 types.
- MaxValue/MinValue from `NPTypeCode.MaxValue()` (wrapped in try/catch) — works correctly.

### 3. `src/NumSharp.Core/Utilities/NumberInfo.cs`  ✅

- Added `SByte.MaxValue/MinValue`, `Half.MaxValue/MinValue`.
- Complex was already handled at switch top: `new Complex(double.MaxValue, double.MaxValue)` / `...MinValue...`. Sentinel values, not mathematically meaningful (no complex ordering), but usable as reduction seeds.
- Fixed pre-existing docstring typo ("min value" → "max value" on MaxValue method).

### 4. `src/NumSharp.Core/Creation/np.dtype.cs`  ⚠️ (partial)

- Added `sbyte`/`half`/`complex128` entries to kind dictionary:
  - SByte → 'i' (signed int kind), Byte → 'u' (unsigned kind — pre-existing bug FIX: was 'b' = boolean kind), Half → 'f' (float kind) ✓
- Added DType creation cases for SByte/Half.
- Added pre-flight string switch: `"int8"/"sbyte"`, `"float16"/"half"`, `"complex128"/"complex"` → works.
- Added `"e"`, `"float16"`, `"Half"`, `"half"` aliases. Added `"uint8"`, `"complex128"` to existing.
- Added `size=2, type="f"` → Half (so `"f2"` works as NumPy's float16).
- 🐛 **Bug (pre-existing, not fixed by branch):**
  - `np.dtype("b")` returns **Byte** — NumPy: int8/SByte.
  - `np.dtype("B")` **THROWS** — NumPy: uint8/Byte.
  - `np.dtype("i1")` returns **Byte** — NumPy: int8/SByte.
  - `np.dtype("u1")` returns **UInt16** — NumPy: uint8/Byte.
- Users hitting these four forms get wrong dtype or crash. The branch added SByte but kept the old `"b"` / `"i1"` mappings that collide with NumPy's int8 conventions.

### 5. `src/NumSharp.Core/Logic/np.find_common_type.cs`  ✅

- Added full 15 entries each for (int8, *), (float16, *) rows and (int, X→int8)/(X→float16) column entries in both `typemap_arr_arr` and `typemap_arr_scalar`.
- Cross-verified 42 promotion pairs against NumPy 2.x `np.promote_types(...)` — **all match**.
- Note: `np.complex64` in NumSharp source refers to `System.Numerics.Complex` (complex128 in NumPy). Naming is confusing but semantically correct.
- ⚠️ Observation (not this branch): `typemap_arr_scalar` rules differ from NEP50 in general (NumPy 2.x scalars follow normal promotion). Pre-existing design, not altered by this branch.

### 6. `src/NumSharp.Core/Utilities/Converts\`1.cs`  ✅

- Added static cached `ToHalf(T)` / `ToComplex(T)` / `From(Half)` / `From(Complex)` methods.
- Each uses `Converts.FindConverter<T, Half>()` / `<T, Complex>()` — consistent with existing `ToByte`/`ToInt32`/etc. pattern.
- Uses `System.Numerics` using statement added at top.

---

## Phase 2: Memory/Storage (7 files)

### 7. `src/NumSharp.Core/Backends/Unmanaged/ArraySlice.cs`  ✅ (+ bug fix)

- All 7 switch statements (Scalar, Scalar w/ object, FromArray, Allocate x3, Allocate(Type)) cover SByte/Half/Complex.
- **Bug fix (pre-existing):** Two Scalar switches previously used `((IConvertible)val).ToXxx(InvariantCulture)` — throws for Half/Complex. Now routed via `Converts.ToXxx(val)` — handles all 15 dtypes.
- Added `ArraySlice.FromArray(sbyte[])`, `FromArray(Half[])`, `FromArray(Complex[])` overloads.

### 8. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.cs`  ⚠️ (minor)

- FromArray, Allocate(count), Allocate(count, fill) all have SByte/Half/Complex cases.
- ⚠️ `Allocate(count, fill)` uses direct cast `(Half)fill` / `(Complex)fill` — throws `InvalidCastException` if caller boxes wrong type (e.g. passes `int` for Half). Compare to `ArraySlice.Allocate` which uses `Converts.ToHalf`.
- Not a show-stopper since this is an internal API; public entry goes through `ArraySlice.Allocate`.

### 9. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs`  ✅

- Two switches updated. Non-generic `CastTo<TOut>` now covers SByte/Half/Complex.
- Generic `CastTo<TOut>` refactored from static `CastTo<TIn, TOut>(source)` call → instance `((IMemoryBlock<T>)source).CastTo<T, TOut>()` to use the generic converter path. Semantically equivalent, supports new types cleanly.

### 10. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.cs`  ✅

- Added `_arraySByte`, `_arrayHalf`, `_arrayComplex` fields.
- `SetInternalArray(array)` and `SetInternalArray(ArraySlice)` both get SByte/Half/Complex cases.
- Address pointer cast via `(byte*)field.Address` is consistent ✓.

### 11. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Getters.cs`  ✅

- 3 object-returning switches (GetValue int[], long[], TransformOffset) — all 15 dtypes.
- 6 new typed direct getters: `GetSByte(int[])`, `GetSByte(params long[])`, `GetHalf(...)×2`, `GetComplex(...)×2` ✓

### 12. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Setters.cs`  ✅

- 1 object-returning switch (SetValue) — all 15 dtypes.
- 6 new typed direct setters: `SetSByte(...)×2`, `SetHalf(...)×2`, `SetComplex(...)×2`.
- All respect `ThrowIfNotWriteable()` (broadcast-view protection) ✓.

### 13. `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Cloning.cs`  ✅

- `AliasAs(NPTypeCode)` switch covers all 15 dtypes including SByte/Half/Complex ✓.

### 🐛 Cross-cutting gap (NOT in diff, should have been): `src/NumSharp.Core/Casting/Implicit/NdArray.Implicit.ValueTypes.cs`

- File is UNCHANGED by this branch.
- Has implicit scalar → NDArray casts for 13 types (bool, byte, short, ushort, int, uint, long, ulong, char, float, double, decimal, **Complex**).
  - Missing: **sbyte**, **Half** implicit operators.
- Has explicit NDArray → scalar casts for 12 types (bool through decimal).
  - **Missing: sbyte, Half, Complex explicit operators.**
- **User-facing impact:**
  - `(sbyte)nd[0]` — compile error
  - `(Half)nd[0]` — compile error
  - `(Complex)nd[0]` — compile error
  - `NDArray x = (sbyte)42` — compile error
  - `NDArray x = (Half)3.14` — compile error
- **Workaround:** `nd.Storage.GetSByte(0)` / `GetHalf(0)` / `GetComplex(0)` — works but less ergonomic.
- **Should be fixed** to complete the dtype API surface — currently users can create arrays of SByte/Half/Complex but can't cast scalars back out with simple syntax.

