# NumSharp `nditer` Branch — Quality Audit V2

**Branch:** `nditer` (compared to `master`)
**Date:** 2026-05-13 (audit) + 2026-05-13 (fact-check pass)
**Methodology:** 8 parallel agents auditing file-by-file, plus 12 orchestrator spot-checks. All findings verified via `python -c` (NumPy 2.4.2) and `dotnet_run` (NumSharp). **A second pass dispatched 8 fact-check agents that wrote failing `[OpenBugs]` tests for each confirmed bug** — those tests live under `test/NumSharp.UnitTest/AuditV2/AuditV2_*.cs`.
**Scope:** 189 src files changed; 24,958 insertions / 7,493 deletions.

---

## Fact-check status (2026-05-13)

| Category | Count |
|---|---|
| Tier 1 findings filed | 65 (T1.1-T1.66, excluding T1.20 which is a sub-table header) |
| **Confirmed** (failing test written) | **60** |
| **False positive / revoked** | **4** — T1.21, T1.41, T1.45, T1.66 |
| **Latent / not reproducible** | **1** — T1.40 |
| **Newly discovered during fact-check** | **4** — T1.24 HASINDEX variant; NPY_TYPECHAR enum value collisions (`b`, `c` shared); Char→'c'→Complex round-trip collision (T1.65 sibling); UnmanagedStorage.GetByte assert on in-bounds SByte 1-D index 0 |

**Per-batch test files** (each `[OpenBugs]` test asserts NumPy-correct behavior — fails today, passes when bug fixed):
- `AuditV2_Iterators.cs` (Batch 1) — 12 tests
- `AuditV2_ILKernelSimd.cs` (Batch 2) — 17 tests (12 failing + 4 sibling guards + 1 perf marker)
- `AuditV2_MathReductions.cs` (Batch 3) — confirmed 9 (T1.3, T1.4, T1.5, T1.14, T1.22, T1.28, T1.35, T1.37, T1.55)
- `AuditV2_LogicShapeStorage.cs` (Batch 4) — 9 tests (T1.10, T1.13, T1.29, T1.42, T1.57, T1.64 + perf measurements)
- `AuditV2_NDArrayCreation.cs` (Batch 5) — 14 tests
- `AuditV2_ManipulationApis.cs` (Batch 6) — 17 tests (NB: has known build error lines 327-328 — non-blocking, fixed during aggregation)
- `AuditV2_MathSelectionSorting.cs` (Batch 7) — 8 tests (T1.15a/b, T1.27a-c, T1.32, perf marker; NB: known build errors lines 73, 75 — fixed during aggregation)
- `AuditV2_CastingRandomUtilities.cs` (Batch 8) — confirmed 12 of 13 assigned

**Refinements measured during fact-check:**
- **T1.6** — narrowed to **scalar path only**. SIMD path (Vector*.LessThanOrEqual) is correct; the bug is in `EmitComparisonOperation` lines 1009-1036 only.
- **T1.15** — `SetIndicesNDNonLinear` call site (line 471-472) is commented-out `TODO`. User-facing fancy-indexed transposed setter actually hits `Debug.Assert` in `SetIndicesND` first (DEBUG) or silently writes wrong offsets (Release). The `NotImplementedException` is dead code today; both symptoms need fixing.
- **T1.16 ≡ T1.43** — same defect at `Default.Cast.cs:23` listed twice. T1.43's "TensorEngine reassignment" claim is technically true but functionally a no-op alias.
- **T1.36** — "garbage" mischaracterization. Values are deterministic mathematical truncations (`base^-1 → 0` except `±1 → ±1`). The API contract divergence is real (NumPy raises ValueError unconditionally).
- **T1.46** — concrete repro: `np.array([100], np.uint8) + 1000` → uint8 result with values 233/234/235 (silent wrap). NumPy raises `OverflowError`.
- **T1.58** — perf gap exists but is not catastrophic at small sizes (<2× SIMD baseline in fact-check).
- **T1.65** — also exposes Char→'c'→Complex collision: `NPY_CHARLTR == NPY_COMPLEXLTR == 'c'`. Round-trip via TYPECHAR is broken for Char and Decimal.

**Perf ratio corrections (measured during fact-check):**

| Operation | Original audit | Fact-check measurement | Verdict |
|---|---|---|---|
| `argsort` 2D Double 1000×1000 axis=-1 | 184× | 151× | Same magnitude, regression real |
| `np.all` 1M Int32 contiguous | 13× | 3.1× | Audit overstated |
| `np.nonzero` 1M Double | 29× | 17.5× | Slight overstatement, still severe |

---

## Auditing criteria (per file/function)

1. **NumPy structural parity** — does the implementation match NumPy's C source structure?
2. **NumPy behavioral parity** — `python -c` ground truth verified against NumSharp output
3. **Performance across compute cases** — contiguous, slightly strided, heavily strided, broadcast, scalar-on-one-side / both, F-contiguous, SIMD-eligible
4. **NumPy ≥10× better?** — orders-of-magnitude gaps
5. **IL generation utilization** — avoids switch-per-type? uses ILKernelGenerator/NpFunc?
6. **dtype coverage** — all 15 dtypes (Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex; plus Char8/DateTime64 helper types)
7. **API parameter parity** — same signature, defaults, edge cases as NumPy
8. **Wasted copies** — does NumSharp materialize where NumPy doesn't?
9. **Iterator/kernel path utilization** — should this use NDIterator/NpyIter/ILKernelGenerator?
10. **Missing functionality** — what does NumPy expose that we don't?

---

## TIER 1 — Correctness bugs (crashes, wrong output, silent corruption)

### T1.1 — `Iternext()` ignores EXLOOP, reads past buffer  ★★★

**File:** `src/NumSharp.Core/Backends/Iterators/NpyIter.cs:1985-2003`
**Source:** Group 1
**Severity:** Bug (data corruption + perf 20×)

`Iternext()` calls `state.Advance()` unconditionally — no branch for `EXLOOP`, no branch for buffered-non-reduce refill. EXLOOP path advances 1 element at a time (NDim-1 extra iterations) while the kernel believes it received `Shape[NDim-1]` elements per call → 3-12× buffer overrun on transposed/non-coalescible arrays.

**Reproduction:** `dotnet_run` with `arange(12).reshape(3,4).T` + `EXTERNAL_LOOP` returns 12 iterations vs NumPy's 4. Verified.

**Remediation:** Add EXLOOP + buffered-non-reduce branches to `Iternext()`, OR delete the public method and force callers onto `GetIterNext()` (the bridge already does this correctly).

---

### T1.2 — `Iternext()` BUFFERED non-reduce path segfaults on arrays > BufferSize  ★★★

**File:** `src/NumSharp.Core/Backends/Iterators/NpyIter.cs:1985-2003`
**Source:** Group 1
**Severity:** Bug (AccessViolationException)

For `BUFFERED + !REDUCE`, `Iternext()` falls through to `state.Advance()` which doesn't refill the buffer. `GetDataPtr(int op)` (line 2723-2756) then computes a pointer past the buffer boundary.

**Reproduction:** `arange(20000).astype(Int32)` cast to Double via BUFFERED iterator → `AccessViolationException` on second buffer fill. Verified.

**Remediation:** Implement `BufferedNonReduceIternext()` mirroring NumPy's `npyiter_buffered_iternext` — refill READ operands, flush WRITE operands across boundaries.

---

### T1.3 — `np.power` on sliced/broadcast integer arrays CRASHES  ★★

**File:** `src/NumSharp.Core/Backends/Default/Math/Default.Power.cs:50-128`
**Source:** Group 3 + orchestrator V1
**Severity:** Bug (InvalidOperationException)

`PowerInteger` uses `lhs.Unsafe.Address` which throws on sliced/broadcast arrays. Even if address were available, the loop ignores strides → silent corruption.

**Reproduction:**
```csharp
var a = np.arange(20).astype(NPTypeCode.Int32);
np.power(a["::2"], np.arange(10));  // → InvalidOperationException
```

**Remediation:** Guard fast-path with `Shape.IsContiguous && !IsBroadcasted && offset==0`, OR emit a stride-aware integer Power IL kernel (preferred).

---

### T1.4 — `np.reciprocal` on sliced integer arrays CRASHES (same pattern as T1.3)

**File:** `src/NumSharp.Core/Backends/Default/Math/Default.Reciprocal.cs:24-95`
**Source:** Group 3
**Severity:** Bug

Identical bug pattern: `ReciprocalInteger` uses `Unsafe.Address`, ignores strides. Crashes on `np.reciprocal(arr["::2"])` and `np.reciprocal(np.broadcast_to(...))`.

**Remediation:** Same as T1.3.

---

### T1.5 — `np.dot(N-D, 1-D)` for N≥3 returns wrong shape and values  ★★

**File:** `src/NumSharp.Core/Backends/Default/Math/BLAS/Default.Dot.cs:60`
**Source:** Group 3
**Severity:** Bug (silent wrong output)

Code hardcodes `axis: 1` despite a `//TODO!` comment. For `(2,3,4) @ (4,)`:
- NumPy returns shape `(2, 3)` with sum-along-last-axis values
- NumSharp returns shape `(2, 4)` with completely wrong values

**Remediation:** Replace with `axis: lhs.ndim - 1`. Add tests covering N-D × 1-D for N=2,3,4,5.

---

### T1.6 — `NaN <= x` and `NaN >= x` return True (IL comparison kernel bug, SCALAR PATH ONLY)  ★★

**Status:** CONFIRMED (narrowed) — `AuditV2_ILKernelSimd.cs::T1_6_NaN_LessEqual_*` (6 tests)
**File:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Comparison.cs:1009-1036` (scalar path)
**Source:** Group 2 + fact-check Batch 2
**Severity:** Bug

LessEqual/GreaterEqual emit `!Cgt`/`!Clt` in the scalar path. For NaN: `Cgt(NaN, x) = false` then `!false = true`. NumPy spec: all NaN comparisons return False.

**Important narrowing:** The SIMD path is CORRECT (`Vector*.LessThanOrEqual`/`GreaterThanOrEqual` propagate NaN as false). Only the scalar tail / small-array path emits the wrong sequence. Sibling tests for `<`, `>`, `==`, `!=` confirmed correct (would catch regressions when fix lands).

**Reproduction:**
```csharp
var nan = np.array(new float[] { float.NaN });
(nan <= np.array(new float[]{1.0f})).GetValue<bool>(0);  // True — WRONG
```

**Remediation:** Emit `LessEqual` as `(Clt OR Ceq)` and `GreaterEqual` as `(Cgt OR Ceq)`. NaN-NaN-NaN → false ∨ false = false.

---

### T1.7 — `np.array(NDArray, copy=False)` aliases by default — NumPy default is `copy=True`  ★★★

**File:** `src/NumSharp.Core/Creation/np.array.cs`
**Source:** Group 5
**Severity:** Bug (production-breaking silent corruption)

Default behavior diverges from NumPy. Constructing `b = np.array(a)` aliases storage; subsequent `b[0] = 999` silently mutates `a[0]` in NumSharp but not NumPy.

**Remediation:** Change default to `copy=true`. Existing aliasing users must pass `copy=False` explicitly.

---

### T1.8 — `np.concatenate` dtype promotion violates NEP50  ★★

**File:** `src/NumSharp.Core/Creation/np.concatenate.cs`
**Source:** Group 5
**Severity:** Bug (silent precision loss)

`f32 + i64 → f32` in NumSharp; NumPy promotes to `f64`. The `NPTypeCode.CompareTo` group-vs-size logic doesn't implement NEP50.

**Remediation:** Route through `np._FindCommonArrayType` (NEP50-compliant).

---

### T1.9 — `np.concatenate` throws on mixed SByte/Half/Complex inputs

**File:** `src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs:CopyStridedToStridedWithCast`
**Source:** Group 5
**Severity:** Bug (crash)

`CopyStridedToStridedWithCast` lacks branches for SByte/Half/Complex (3 of 15 dtypes). Mixed-dtype concatenation involving these throws `NotSupportedException`.

**Remediation:** Add SByte/Half/Complex cases to `CopyStridedToStridedWithCast` and `IsSafeCast`/`ReadAsDouble`/`WriteFromDouble`.

---

### T1.10 — `Shape` is `readonly struct` with a MUTATING set indexer  ★★

**File:** `src/NumSharp.Core/View/Shape.cs`
**Source:** Group 4
**Severity:** Bug (breaks immutability, invalidates cached flags)

`shape[i] = x` mutates the underlying `long[] dimensions`, breaking `readonly struct` semantics and invalidating cached `_flags`/`size`/`_hashCode`. Causes subtle correctness issues when shapes are passed around.

**Remediation:** Remove the set indexer. Force all mutations to go through Shape construction (or a builder).

---

### T1.11 — `np.fmax`/`np.fmin` propagate NaN instead of skipping  ★

**File:** `src/NumSharp.Core/Math/np.maximum.cs`, `np.minimum.cs`
**Source:** Orchestrator V2
**Severity:** Bug

`np.fmax(5, NaN)` returns NaN; NumPy returns 5 (skip NaN semantics). `np.maximum/minimum` now correctly propagate (was wrong in prior audit).

**Remediation:** Compose `fmax` via `where(isnan(a), b, where(isnan(b), a, max(a,b)))` or dedicated IL kernel.

---

### T1.12 — `NpyIterCasting` missing SByte/Half/Complex in `IsSafeCast`/`ReadAsDouble`/`WriteFromDouble`

**File:** `src/NumSharp.Core/Backends/Iterators/NpyIterCasting.cs`
**Source:** Group 1
**Severity:** Bug (silent throw)

Three dtypes silently throw `NotSupportedException` for buffered cast paths.

---

### T1.13 — `UnmanagedStorage.SetValue(object, ...)` / `CopyTo` missing SByte/Half/Complex

**File:** `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Setters.cs`
**Source:** Group 4
**Severity:** Bug

Throws `NotSupportedException` for these dtypes.

---

### T1.14 — `np.convolve` accumulates in `double`, loses int64 precision

**File:** `src/NumSharp.Core/Math/NdArray.Convolve.cs:138-188`
**Source:** Group 7
**Severity:** Bug

`double sum` regardless of T. For Int64/UInt64/Decimal beyond 2^53, accumulator is lossy:
- NumSharp: `convolve([(1<<53)+1, (1<<53)+1], [1])` = `[9007199254740992, ...]`
- NumPy: `[9007199254740993, ...]` (preserves exact int64)

**Remediation:** Use native-typed accumulator per T. Add `Boolean` case (NumPy treats as bitwise OR).

---

### T1.15 — `NDArray.Indexing.Selection.Setter.SetIndicesNDNonLinear` throws `NotImplementedException` (DEAD CODE)

**Status:** CONFIRMED with refinement — `AuditV2_MathSelectionSorting.cs::T1_15a/b`
**File:** `src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Setter.cs:617`
**Source:** Group 7 + fact-check Batch 7
**Severity:** Bug

The `NotImplementedException` body exists but the **call site (line 471-472) is commented out** as a TODO. The user-facing fancy-indexed transposed setter actually hits `Debug.Assert(dstOffsets.size == values.size)` in `SetIndicesND` first (DEBUG build fires the assert; Release silently writes wrong offsets). The `NotImplementedException` is dead code today — both symptoms need fixing.

---

### T1.16 — `Default.Cast.cs` mutates caller's storage when `copy=false` (consolidates T1.43)

**Status:** CONFIRMED — `AuditV2_NDArrayCreation.cs::T1_16/T1_43_*`
**File:** `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Cast.cs:23`
**Source:** Group 4 + fact-check Batch 5
**Severity:** Bug (behavioral divergence)

When `copy=false`, replaces `nd.Storage` and reassigns `nd.TensorEngine` (the latter being a no-op alias since `engine = nd.TensorEngine` is captured at line 15). All four branches (empty / scalar / size=1 / general) have the defect. NumPy's `astype(copy=False)` returns a new wrapper around the same storage, never mutates the original.

**Note:** T1.43 below is the same defect (listed twice in the original audit).

---

### T1.17 — `np.expand_dims` drops new axis for empty arrays

**File:** `src/NumSharp.Core/Manipulation/np.expand_dims.cs:7-12`
**Source:** Group 6
**Severity:** Bug

Early-return for `a.size == 0` skips the axis insertion. `np.expand_dims(np.array([]), 0)` returns shape `(0,)` instead of `(1, 0)`.

---

### T1.18 — `np.copyto` ignores `casting` and `where` parameters (silently truncates)

**File:** `src/NumSharp.Core/Manipulation/np.copyto.cs:16`
**Source:** Group 6
**Severity:** Bug (silent precision loss)

NumPy default `casting='same_kind'` rejects float→int8 copy with TypeError. NumSharp silently truncates.

---

### T1.19 — `NpFunc` caches by MethodHandle.Value, ignoring instance target  ★

**File:** `src/NumSharp.Core/Utilities/NpFunc.cs`
**Source:** Group 8
**Severity:** Bug (landmine — silent wrong-instance dispatch)

Instance methods called with different target instances silently invoke the first cached instance. All 24 production call sites use static methods today (safe), but the API is public.

**Remediation:** Include `method.Target` in cache key, or reject non-static delegates.

---

### T1.20 — Other correctness gaps

| # | File | Issue | Severity |
|---|---|---|---|
| T1.21 | `Default.ATan2.cs:110-120` | ~~Promotes int8/uint8 inputs to Half (NumPy: Double)~~ **FALSE POSITIVE** — NumPy 2.4.2 actually maps int8/uint8/bool→float16, int16/uint16→float32, int32+→float64. NumSharp matches. | REVOKED |
| T1.22 | `Default.Ceil/Floor/Truncate.cs` | Boolean input promoted to Double (NumPy keeps bool) | Bug |
| T1.23 | `NpyIter.GetIterView` | Throws OverflowException on 0-d arrays | Bug |
| T1.24 | `NpyIter.EnableExternalLoop` | Doesn't validate MULTI_INDEX/HASINDEX (NumPy raises ValueError) | Bug |
| T1.25 | `NDIterator.cs` broadcast ctor | Produces wrong strides → OOM on broadcast views | Bug |
| T1.26 | `np.finfo.minexp` | Off-by-one for float32 (-125 vs -126), float64 (-1021 vs -1022) | Bug |
| T1.27 | `np.searchsorted` | `binarySearchRightmost` is actually leftmost; missing `side`/`sorter`; multidim silently accepted | Bug |
| T1.28 | `np.negative` | Accepts bool (NumPy rejects); throws on uint8/uint16/uint32/uint64/Char (NumPy wraps); operator `-byte_arr` works but `np.negative(byte_arr)` throws — inconsistent paths | Bug |
| T1.29 | `Shape.OWNDATA` | Flag declared (0x0004) but never set anywhere — `Shape.OwnsData` always false | Bug |
| T1.30 | `ArrayConvert.cs` (4576 LoC) | Inner `ToX(Array)` switches only handle 13/15 dtypes — SByte/Half/Complex throw `ArgumentOutOfRangeException` | Bug (~45 missing cases) |
| T1.31 | `randint(low, high=-1)` | `-1` is a "high omitted" sentinel; breaks legal `randint(-10, -1, 3)` | Bug |
| T1.32 | `np.modf` | Public field name typo: `Intergral` (should be `Integral`) | Bug (API break to fix) |
| T1.33 | `NPTypeCode.AsNumpyDtypeName(Char)` | Returns `"uint8"` but Char is 2 bytes (UTF-16) — interop hazard | Bug |
| T1.34 | `NpyExpr` Const/Where/Call | Only support 12 dtypes (no SByte/Half/Complex). `NpyExpr.Const(Half).Compile` → NotSupportedException | Bug |
| T1.35 | `Default.MatMul.cs:19-21` | `np.matmul(1D, 2D)` rejected with `NotSupportedException`; NumPy prepends 1 to first dim. Comment describes correct NumPy behavior right above the throw | Bug |
| T1.36 | `Default.Power.cs` | `int ** -int` returns silent garbage (e.g. `[0,1,-1,0]`); NumPy raises `ValueError("Integers to negative integer powers are not allowed")` | Bug |
| T1.37 | `Default.Reduction.Std/Var:VarSimdHelper line 42` | `if (size <= ddof) return NaN` hardcoded; NumPy returns `+inf` from raw `0/0` IEEE division | Bug |
| T1.38 | `NpyIterCasting.IsSafeCast` | `IsSignedInteger` doesn't list SByte; `IsFloatingPoint` doesn't list Half. `IsSafeCast(SByte→Int32)` returns false despite NumPy declaring safe; `IsSafeCast(Half→Single)` returns false despite safe widening | Bug |
| T1.39 | `NpyIterCasting.ReadAsDouble`/`WriteFromDouble` | Int64/UInt64 → double loses precision above 2^53 (NumPy correctly upcasts). Also missing SByte/Half/Complex (covered as T1.12) | Bug |
| T1.40 | `NpyIter.Copy()` | ~~Multi-axis buffered+reduce copies may not deep-copy chain~~ **LATENT — NOT REPRODUCIBLE.** Stress-tested with 2D+3D multi-axis reduce; all pointer chains preserved. | LATENT |
| T1.41 | `NpyIter.Shape property:2494-2520` | ~~Returns post-coalesce internal shape, not original `itershape`~~ **FALSE POSITIVE** — NumPy 2.4.2 also returns `(12,)` for `np.nditer(arange(12).reshape(3,4)).shape`. Only differs with MULTI_INDEX, where NumSharp matches. | REVOKED |
| T1.42 | `Shape.Equals` | Compares only `dimensions`, not strides/offset/bufferSize. Two semantically-different shapes (e.g. C-contig vs transposed) hash equal | Bug (semantic) |
| T1.43 | `Default.Cast.cs:23` (`copy=false` path) | **DUPLICATE OF T1.16** — same defect described twice. TensorEngine reassignment is a no-op alias (engine = nd.TensorEngine captured at line 15). | DUPLICATE |
| T1.44 | `Default.NDArray.cs::CreateNDArray(Shape, Type, Array, char order)` | Passes `order` to NDArray ctor without resolution through `OrderResolver`. Callers passing `'A'`/`'K'` get literal char in NDArray | Bug |
| T1.45 | `OrderResolver 'K'` | ~~Non-contig source falls back to `'C'`~~ **FALSE POSITIVE** — NumPy 2.4.2 also returns C-contig for strided `a[:, ::2].copy(order='K')`. NumSharp's "conservative fallback" matches. | REVOKED |
| T1.46 | `np.find_common_type` for `uint8_arr + 1000` | Returns uint8 (silent overflow to 232/233/234 with concrete repro); NumPy raises `OverflowError: Python integer 1000 out of bounds for uint8`. NEP50 fits-check missing | Bug (NEP50 strictness) |
| T1.47 | `np.find_common_type._can_coerce_all(arr, start)` | Wrong `Array.Copy` call: `Array.Copy(dtypelist, start, sub, len, len)` — 4th arg should be 0. Currently unreachable but landmine if NPTypeCode ordering changes | Bug (latent) |
| T1.48 | `np.ascontiguousarray(scalar)` / `np.asfortranarray(scalar)` | On 0-D scalar input, returns ndim=0; NumPy promotes to ndim=1 | Bug |
| T1.49 | `np.asanyarray` | Missing `IEnumerable<sbyte>`, `IEnumerable<Half>`, `IEnumerable<Complex>` cases → `NotSupportedException` | Bug |
| T1.50 | `np.arange(0, 5, 1, NPTypeCode.Boolean)` | Returns alternating bool[] for any length; NumPy raises `TypeError` for length > 2 | Bug |
| T1.51 | `DType.byteorder` | Parsed `<`/`>`/`=`/`|` prefix is stripped but always returns `'='`. NumPy preserves the parsed prefix | Bug |
| T1.52 | `DType.kind` | Confuses TYPECHAR with kind code: `bool` returns `'?'` (NumPy: `'b'`), `Char` returns `'S'` (NumPy: `'U'`). NumPy uses `biufcmMOSUV` kind alphabet | Bug |
| T1.53 | `DType.name` | Returns C# typename (`"Int32"`, `"Double"`, `"Boolean"`); NumPy returns `"int32"`, `"float64"`, `"bool"` | Bug |
| T1.54 | `np.frombuffer 'F'/'c8'/'complex64'` | Silently maps to Complex128 (single→double widen). Inconsistent with `np.dtype('F')` which throws NotSupportedException. If user buffer is complex64 (8 bytes/elem), NumSharp reads 16 bytes/elem | Bug |
| T1.55 | `np.copyto(dst_sbyte, src_float)` | Throws `NotSupportedException: Unsupported type: SByte` from cast layer. NumPy errors with type message but doesn't crash | Bug |
| T1.56 | `np.array(Array, ...)` default `ndmin=1` | NumPy default `ndmin=0`. Rare path (0-D inputs) silently differs | Bug (low impact) |
| T1.57 | `UnmanagedStorage.SetValue(object, ...) / SetData(NDArray, ...)` | Object-overload `#else` branch covers only 12 dtypes — missing SByte/Half/Complex → `NotSupportedException` on those dtypes via object setters (covered partially as T1.13) | Bug |
| T1.58 | `Default.BooleanMask` fallback gather kernel | Uses `Buffer.MemoryCopy(src, dst, elemSize, elemSize)` per matched element (~1µs/element overhead). For dtype Half/Complex this calls into `Buffer.MemoryCopy(src, dst, 2/16, 2/16)` per element | Perf bug |
| T1.59 | `np.where(condition)` 1-arg | Returns `NDArray<long>[]`; NumPy returns a `tuple`. Type signature diverges | API divergence |
| T1.60 | `np.where(cond, 1, 2)` | Returns `int32`; NumPy returns `int64`. Documented cross-language divergence | Divergence (porting risk) |
| T1.61 | `np.copyto` unwriteable dst | Throws `NumSharpException`; NumPy raises `ValueError` | Exception type mismatch |
| T1.62 | `np.iinfo(bool)` | Accepted (returns bits=8, min=0, max=1); NumPy 2.x raises `ValueError: Invalid integer data type 'b'`. Documented NumSharp extension | API divergence |
| T1.63 | `np.iinfo(UInt64).max` | Clamped to `long.MaxValue` because public `max` field is typed `long`. Callers reading `info.max` for uint64 silently get wrong value (`maxUnsigned` field has the real value) | Bug |
| T1.64 | `np.arr.flags.OWNDATA` | Always `False`; NumPy: `True` for arrays that own their data. Inert today (correctness checked via `_baseStorage`) | Bug |
| T1.65 | `NPTypeCode` TYPECHAR collisions | `Decimal → 'q' (NPY_LONGLONGLTR) → Int64` (round-trip identity lost). **Also discovered:** `NPY_BYTELTR == NPY_GENBOOLLTR == 'b'` and `NPY_CHARLTR == NPY_COMPLEXLTR == 'c'` — Char round-trip resolves to Complex. | Bug |
| T1.66 | `np.dtype('float')` | ~~NumPy 2.x deprecates this string~~ **FALSE POSITIVE** — NumPy 2.4.2 accepts `np.dtype('float')` returning `dtype('float64')` with no warnings. Behavior matches. | REVOKED |
| T1.67 | `NpyIter.EnableExternalLoop` HASINDEX (C_INDEX/F_INDEX) variant | **NEWLY DISCOVERED.** Same root cause as T1.24 — also fails to reject the HASINDEX flag. Test: `AuditV2_Iterators.cs::T1_24_EnableExternalLoop_Must_Reject_CIndex`. | Bug |
| T1.68 | `NPTypeCode.cs:485-487` unreachable code | **NEWLY DISCOVERED.** `return NPTypeCode.Decimal;` after `return NPTypeCode.Complex;` — dead code on Complex branch. | Bug (latent) |
| T1.69 | `UnmanagedStorage.GetByte(new int[]{0})` on (3,) SByte array | **NEWLY DISCOVERED.** Fires `Debug.Assert("Memory corruption expected")` at `UnmanagedStorage.Getters.cs:475` for in-bounds 1-D index 0. | Bug |

**Tier 1 totals after fact-check (2026-05-13):**
- **Confirmed: 60 / 65** (failing tests written under `test/NumSharp.UnitTest/AuditV2/`)
- **False positives: 4** (T1.21, T1.41, T1.45, T1.66) — verified NumSharp matches NumPy 2.4.2
- **Latent: 1** (T1.40) — no concrete failure mode reproducible
- **Duplicate: 1** (T1.43 ≡ T1.16) — same defect listed twice
- **Newly discovered: 3** (T1.67, T1.68, T1.69) added above

---

## TIER 2 — Performance regressions (≥10× slower than NumPy)

| Operation | NumPy | NumSharp | Ratio | File | Source |
|---|---|---|---|---|---|
| `argsort` 2D Double 1000×1000 axis=-1 | 12.5 ms | **2,305 ms** | **184× orig; 151× re-measured** | `ndarray.argsort.cs` (LINQ) | Group 7 + fact-check Batch 7 |
| `argsort` 2D Double 1000×1000 axis=0 | 18.7 ms | **2,769 ms** | **148×** | same | Group 7 |
| `matmul` Double 1024×1024 | 5.6 ms | **1,064 ms** | **190×** | `SimdMatMul.Double.cs` | V5 |
| `matmul` Double 512×512 | 1.6 ms | 163 ms | 102× | same | V5 |
| `matmul` Float 1024×1024 | 3.7 ms | 180 ms | 49× | same | V5 |
| `matmul` (float SIMD path) GFLOPS | 113.6 | 12.84 | 8.85× | `SimdMatMul.cs` | Group 2 |
| `convolve('full')` Double 10K×100 | 0.40 ms | 27 ms | **67×** | `NdArray.Convolve.cs` | Group 7 |
| `clip` Double 1000×1000 Transposed | 1.55 ms | 104 ms | **67×** | `np.clip.cs` → `ClipNDArray` | Group 7 |
| `bool-mask setter` 1000×1000 + 1D mask | 0.09 ms | 3.28 ms | **36×** | `NDArray.Indexing.Masking.cs:281-318` | Group 7 |
| `nonzero` 1M Double | 22.7 ms | **662 ms (orig); 285 ms (re-measured)** | **29× orig; 17.5× re-measured** | `ILKernelGenerator.Masking.cs:194` (`List<long[]>` per elem) | Group 4 + fact-check Batch 4 |
| `dot` 1D@1D 1M Double | (NumPy fast) | (NumSharp slow) | **22×** | `Default.Dot.cs` | Group 3 |
| `searchsorted` 1M sorted, 100K Int32 | 6.0 ms | 112 ms | 18.7× | `np.searchsorted.cs` (boxed per probe) | Group 7 |
| `ravel('F')` of F-contig (3000× regression) | (view, O(1)) | (forced copy) | **3000×** | `np.ravel.cs:30-34` | Group 6 |
| `shift` strided | (NumPy fast) | (NumSharp materializes) | 15× | `Default.Shift.cs:72,79` | Group 3 |
| `dot` ND@MD strided | (NumPy fast) | (NumSharp materializes) | 17× | `Default.Dot.cs` | Group 3 |
| `std` axis 1000×1000 Double | 2.84 ms | 113 ms | **40×** | (axis path) | V7 |
| `add` F-order 1000×1000 ×10 | 17.3 ms | 225 ms | **13×** | F-contig kernel path in `ILKernelGenerator.Binary.cs` | V9 |
| `all`/`any` 1M Int32 contiguous | 21.5 ms (orig); 21 ms (fact-check) | 270 ms (orig); 64 ms (fact-check) | **13× orig; 3.1× re-measured** | `NpyAllKernel<T>` scalar loop (no SIMD) | Group 4 + fact-check Batch 4 |
| `eye(5000)` Float64 | 3.4 ms | 31 ms | **9.1×** | `np.eye.cs` (boxing per diag elem) | V12 |
| `astype('F')` 1000×1000 f32→f64 | 1.5 ms | 13.1 ms | **9×** | `Backends/NDArray.cs:493-499` (2-pass alloc) | Group 5 |
| `copy('F')` 1000×1000 int64 | 1 ms | 11.7 ms | **13×** | `NDArray.Copy.cs` via NpyIter coordinate decode | Group 5 |
| `tile(arange(100), 10000)` | 1.35 ms | 9 ms | **6.7×** | `np.tile.cs` (8 allocs per call) | V12 |
| `sum Half 1M ×10` | 24 ms | 121 ms | **5×** | (Half SIMD missing) | V11 |
| `searchsorted` 1M sorted, 100K Int32 queries | 6.0 ms | 112.4 ms | **18.7×** | (boxed `Storage.GetValue`) | Group 7 |

**Common root causes:**
1. **Hand-rolled LINQ loops** with allocation per element (argsort, nonzero, repeat, nanmean)
2. **Boxing via `Storage.GetValue(int)` + `Converts.ToDouble(object)`** in inner loops (searchsorted, convolve, clip-general)
3. **No SIMD path for F-contig / strided / broadcast** binary ops — fall through to scalar
4. **MatMul kernel** is 108 LoC; NumPy uses OpenBLAS (multi-threaded, AVX-512, cache-blocked GEMM)
5. **Double-allocation patterns** for F-order (`Cast → C-contig copy → F-contig copy`) — astype/eye/copy/concatenate
6. **Per-element coordinate decode** in `NpyIterCasting.CopyStridedToStridedWithCast` even for same-dtype copy
7. **NpFunc dispatch ~20× slower than manual switch** (~32ns/call vs 1.5ns); acceptable for kernel-level, unusable for per-element

### Additional perf findings (pass 2)

| Operation | NumPy | NumSharp | Ratio | File | Source |
|---|---|---|---|---|---|
| `np.dot` 1D@1D 1M Double | (NumPy fast) | (2-pass: `*` + ReduceAdd alloc) | **22×** | `Default.Dot.cs:64-72` | Group 3 |
| `NpyIter.Iternext()` per-element 10M Int64 | 4.8 ms | 94.3 ms | **20×** | `NpyIter.cs:1985-2003` | Group 1 (P1) |
| `Default.Cast` general path 1M int32→f64 | 14 ms | 29 ms | 2× | `UnmanagedMemoryBlock.Casting.cs:122` | Group 4 |
| `UnmanagedMemoryBlock.Casting.cs` (2238 LoC) per-element delegate call cast | (NumPy fast) | (scalar, no SIMD per type-pair) | 2-10× | `UnmanagedMemoryBlock.Casting.cs` | Group 4 |
| `Default.ClipNDArray.cs:158-173` (general) 1M int32 strided | 13 ms / 10 iters | 185 ms / 10 iters | **14×** | `Default.ClipNDArray.cs` | Group 3 |
| `Default.Shift.cs` strided 1M | (NumPy fast) | 31 ms vs 2 ms contig | **15×** | `Default.Shift.cs:72,79` | Group 3 |
| `Default.Dot.NDMD.cs` Generic path strided | (NumPy fast) | (boxes via `GetValue(coords) + Converts.ToDouble`) | **17×** | `Default.Dot.NDMD.cs:325-385` | Group 3 |
| NDIterator legacy wrapper (1M int64 transposed) | (direct ptr: 2.5 ms) | 17.8 ms | **7×** | `NDIterator.cs` | Group 1 (P2) |
| Bool full mask 1M Double | 0.77 ms | 5.94 ms | **7.7×** | `Default.BooleanMask.cs` | Group 7 |
| Fancy 1D index 1M src, 100K Int32 indices Double | 0.40 ms | 2.04 ms | **5.1×** | `NDArray.Indexing.Selection.Getter.cs` (`FetchIndices` LINQ + virtual dispatch) | Group 7 |
| `multivariate_normal(NDArray cov)` | (NumPy fast) | (per-element `cov.GetDouble(i,j)`) | N²× | `np.random.multivariate_normal.cs:300` | Group 8 |
| `Converts<T>.ToInt32(T)` cached delegate | 6.7 ns (direct) | 62 ns (delegate) | ~10× | `Converts'1.cs` | Group 8 |
| `NpFunc.Invoke` vs manual switch | 1.5 ns (switch) | 31.7 ns | ~20× | `NpFunc.cs` | Group 8 |
| `eye(10000)` Double (root cause: zeros init) | 7 ms | 181 ms | **26×** | `UnmanagedMemoryBlock.Fill(T)` scalar loop, no SIMD/InitBlockUnaligned | Group 5 |
| `Default.Reduction.Nan.cs:ExecuteNanAxisReductionScalar` strided 1024² f64 | (NumPy fast) | (per-elem GetAtIndex boxing) | ~3-4× | `Default.Reduction.Nan.cs:439-506` | Group 3, 7 |

---

## TIER 3 — API parity gaps (missing parameters / behaviors)

| Function | Missing | File | Source |
|---|---|---|---|
| `np.repeat` | `axis` parameter (always ravels) | `np.repeat.cs` | Group 6 + V3 |
| `np.searchsorted` | `side`, `sorter`, multidim `a` validation | `np.searchsorted.cs` | Group 7 |
| `np.argsort` | `kind` ('quicksort'/'mergesort'/'stable'/'heapsort'), `order` | `ndarray.argsort.cs` | Group 7 |
| `np.linspace` | `retstep`, `axis`, `device` (NumPy 2.x) | `np.linspace.cs` | Group 5 |
| `np.expand_dims` | Tuple-axis support (only int accepted) | `np.expand_dims.cs` | Group 6 |
| `np.unique` | `return_index`, `return_inverse`, `return_counts`, `axis`, `equal_nan` | `NDArray.unique.cs`, `np.unique.cs` | Group 6 |
| `np.all`/`np.any` | Tuple-axis, `out=`, `where=`, `keepdims=` (no-axis overload) | `np.all.cs`, `np.any.cs` | Group 6 |
| `np.copyto` | `casting=`, `where=` | `np.copyto.cs` | Group 6 |
| `np.concatenate` | `out=`, `dtype=`, `casting=` | `np.concatenate.cs` | Group 5 |
| `np.full`/`np.ones`/`np.zeros` | `order=` parameter | `np.full.cs`, etc. | Group 5 |
| `np.dot`/`np.matmul`/`np.power`/`np.clip`/`np.modf`/`np.negative`/binary ufuncs | `out=` parameter | Many | Group 3, 7 |
| `np.matmul(1D, 2D)` | Rejected with NotSupportedException (NumPy supports as broadcast prepend) | `Default.MatMul.2D2D.cs` | Group 3 |
| `np.clip` | NumPy `min=`/`max=` aliases (NumPy 2.x) | `np.clip.cs` | Group 7 |
| F-order `reshape` | `-1` placeholder not supported | `NdArray.ReShape.cs:34-37` | Group 5 |

**iinfo/finfo deviations:**
- `np.iinfo(bool)` accepted (NumPy 2.x raises ValueError) — `iinfo.cs:84` documented divergence
- `np.iinfo(UInt64).max` clamped to long.MaxValue — public `max` field is `long`, can't fit `2^64-1`

### Additional API parity gaps (pass 2)

| Function | Missing | File | Source |
|---|---|---|---|
| `np.argsort` | `kind` (quicksort/mergesort/heapsort/stable), `order`, `stable` keyword (NumPy 2.x) | `ndarray.argsort.cs` | Group 7 |
| `np.nanmean`/`nanstd`/`nanvar` | `dtype=`, `out=`, `where=`. Plus `mean=`, `correction=` (NumPy 2.x for nanstd/nanvar) | `np.nan*.cs` | Group 7 |
| `np.modf` | tuple `out=(None, None)`, `where=` | `np.modf.cs` | Group 7 |
| `np.convolve` | `out=` parameter; FFT path for large kernels | `NdArray.Convolve.cs` | Group 7 |
| `np.dot`/`np.matmul` | `axes=` parameter (NumPy 2.x) | `Default.MatMul.cs` | Group 3 |
| `np.clip` | NumPy 2.x `min=`/`max=` keyword aliases; default-None bounds; `np.clip(a)` no-bound copy | `np.clip.cs` | Group 7 |
| `np.asarray` | `copy=` parameter (NumPy 2.x); `like=`; `dtype` as string | `np.asarray.cs` | Group 5 |
| `np.empty`/`np.empty_like`/`np.ones_like`/`np.zeros_like`/`np.full_like` | `device=`, `like=`, `subok=`, `shape=` (for `*_like`) | `np.empty.cs`, etc. | Group 5 |
| `np.frombuffer` | `like=` parameter | `np.frombuffer.cs` | Group 5 |
| `np.iinfo(UInt64).max` typed `long` | Should expose `ulong max` or `BigInteger max` field for uint64 | `np.iinfo.cs` | Group 6 |
| `np.linspace` | `device=` parameter (already noted: missing `retstep`, `axis`) | `np.linspace.cs` | Group 5 |
| `np.dtype.newbyteorder` | Throws `NotSupportedException` unconditionally; NumPy supports byte-order specification | `np.dtype.cs` | Group 5 |
| `NDIterator/NpyIter` Python-API methods | `iterationneedsapi`, `has_delayed_bufalloc`, `value` (none); `itviews` buggy on 0-d (T1.23) | `NpyIter.cs` | Group 1 |
| `np.where(condition)` 1-arg | NumPy returns tuple; NumSharp returns `NDArray<long>[]` (cosmetic) | `np.where.cs` | Group 6 |

---

## TIER 4 — Missing functionality (NumPy APIs absent)

Per `.claude/CLAUDE.md` "Missing Functions" plus newly identified:

| Category | Functions |
|---|---|
| Sorting | `np.sort` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.pad` |
| Splitting | (now implemented on this branch — strike from list) |
| Diagonal | `np.diag`, `np.diagonal`, `np.trace` |
| Cumulative | `np.diff`, `np.gradient`, `np.ediff1d` |
| Rounding | `np.round` (only `round_`/`around` exist) |
| Datetime | `np.datetime64` unit support (Y/M/W/D/h/m/s/ms/us/ns/ps/fs/as), `np.timedelta64`, `np.array(strings, dtype='datetime64[D]')` |
| Dtypes | Char8 and DateTime64 NOT registered as NPTypeCode — `new NDArray(new Char8[]{...})` throws NotSupportedException |
| Iterator | NpyIter `GROWINNER` flag set but `CalculateGrowInnerSize` never called; NumPy nditer's REUSE_REDUCE_LOOPS, more itershape variants |
| NaN kernels | Only Float/Double — Half/Complex fall back to scalar |

---

## TIER 5 — Refactor / code quality

| # | File | Issue |
|---|---|---|
| R1 | `src/NumSharp.Core/Backends/Iterators/NpyAxisIter.cs` (492 LoC) | Parallel implementation of NpyIter with 64-dim cap — could be replaced by NpyIter.MultiNew with `op_axes` |
| R2 | `NpyIter.cs` (3469 LoC) | Split into more partial files (.Construction, .MultiIndex, .Lifecycle, .Debug) |
| R3 | `NpyIterKernels.NpyIterExecution` | Dead code with TODOs left behind from migration |
| R4 | `TryCoalesceInner` | Unused |
| R5 | `Default.Cast.cs` 4-branch ladder | Special cases for empty/scalar/(1,)/general duplicate logic; should route general path through `NpyIter.Copy` |
| R6 | `np.find_common_type.cs:_can_coerce_all` | Wrong `Array.Copy` index (`Array.Copy(dtypelist, start, sub, len, len)` — 4th arg should be 0); landmine even though currently unreachable |
| R7 | `NPTypeCode.cs:485-487` | Unreachable `return NPTypeCode.Decimal` after Complex case |
| R8 | `NPTypeCode.cs:531-532` | Decimal → NPY_LONGLONGLTR ('q' = int64) — round-trip loses Decimal identity |
| R9 | `ILKernelGenerator.Binary.cs` | Missing `sbyte` in `IsSimdSupported<T>` — perf loss (Vector256<sbyte> is supported) |
| R10 | `ILKernelGenerator.Reduction.Axis.Simd.cs:75-85` | Heap allocates `long[]` per call; sibling files use `stackalloc Span<long>` |
| R11 | `ILKernelGenerator.cs` | EmitVectorDeg2Rad/Rad2Deg emit unnecessary Stloc/Ldloc pair (multiply is commutative) |
| R12 | `np.tile.cs` | 8 allocations per call (could use slot-reused buffers) |
| R13 | `Arrays.cs::AppendAt` | Clones source then mutates wrong reference (currently unused — landmine) |
| R14 | `Arrays.cs::Slice<T>` | Dead code (identical if/else branches) |
| R15 | `NDArray.Implicit.cs::implicit operator NDArray(string)` | Non-NumPy parsing of `"[1,2,3]"` — feature, not bug, but easy to break test fixtures |
| R16 | `ILKernelGenerator.Masking.cs:38` | `NonZeroSimdHelper<T>` exists but is dead code; `Default.NonZero.cs:51` always calls the slow path |
| R17 | `np.unique.cs` non-contig path | Closure allocation per element via `Func<long, long>` |
| R18 | `Converts.cs` `ChangeType<TIn,TOut>` 12×12 ladder | Doesn't include SByte/Half/Complex — falls through to slower boxed path |
| R19 | `Default.MatMul.Strided.cs` (512 LoC new) | Generic INumber<T> kernel; should be IL-emitted per dtype pair for full SIMD |
| R20 | `Default.Reduction.CumAdd.cs:130-131` | Unconditional `arr.copy()` on non-contig elementwise cumsum; NpyIter now available to avoid |
| R21 | `Default.Reduction.CumMul.cs` | Same pattern as CumAdd — unconditional copy on non-contig |
| R22 | `Default.Reduction.Add.cs:HandleTrivialAxisReduction:161-178` | Uses `GetAtIndex`/`SetAtIndex` per element for axis-of-size-1 reduce; should be `memcpy` |
| R23 | `Default.Reduction.ArgMax.cs:192` | Half/Complex axis fallback creates one NDArray view per slice (1000 view allocations for (1000,1000) axis=0) |
| R24 | `Default.MatMul.Strided.cs:375` | `accBuf = new double[N]` per call — should be poolable via ArrayPool |
| R25 | `Default.Clip.cs:47` | `Cast(copy: true)` always materializes even when input is already contig + dtype-matched |
| R26 | `np.unique.cs` non-contig path (~145-151) | Closure allocation per element via `Func<long, long>` — could call method directly |
| R27 | `np.unique.cs` non-contig path | Materializes `flat` copy then dispatches by typecode — 2-pass; NumPy does 1-pass over iterator |
| R28 | `np.tile.cs:91-104` | 8 allocations per call (reshape × 2, broadcast_to, copy, plus 4× `new long[]` for outShape/interleaved/etc.) |
| R29 | `np.repeat.cs:81-85,203-208` | Per-element `Converts.ToInt64(repeatsFlat.GetAtIndex(i))` — boxes twice |
| R30 | `np.where.WhereImpl:157-162` | Reallocates `NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2))` per call; could be a static field per type |
| R31 | `np.find_common_type.cs:1058-1090` | Four duplicated `_can_coerce_all` overloads (array, list, array-with-start, list-with-start) — collapse to a single helper taking `ReadOnlySpan<NPTypeCode>` |
| R32 | `np.empty.cs::empty(Shape, char order, Type)` | `'A'`/`'K'` orders throw `ArgumentException` — minor but should match NumPy's "only 'C' or 'F' permitted" message exactly |
| R33 | `Converts.cs::ChangeType<TIn,TOut>` 12×12 ladder | Missing SByte/Half/Complex source cases — falls through to slower boxed path. Could expand to 15×15=225 |
| R34 | `Arrays.cs::AppendAt` | Bug: clones source then mutates the wrong reference. Currently unused but landmine |
| R35 | `Arrays.cs::Slice<T>` | Dead code — both `if (len > 700_000)` branches identical (presumably future Parallel.For scaffold) |
| R36 | `NdArray.Implicit.Array.cs` | `implicit operator NDArray(string)` parses `"[1,2,3]"` — non-NumPy behavior. Worth marking obsolete or replacing with explicit `np.array(str)` |
| R37 | `ArraySlice.cs::DangerousFree()` | Publicly callable; can corrupt other live slices over the same MemoryBlock. Should be `internal` or `[Obsolete]` |
| R38 | `UnmanagedMemoryBlock<T>.GetHashCode():947` | Truncates `long Count` and 64-bit address to int — distinct blocks at different addresses with same low 32 bits hash-collide |
| R39 | `ArraySlice.cs::FromArray(T[], bool copy=false)` | Default `copy=false` uses `GCHandle.Alloc(arr, Pinned)` — mutations to original `T[]` visible through slice. Sharp edge for callers expecting copy semantics |
| R40 | `np.ones.cs::case NPTypeCode.String: one = "1"` | `np.ones(shape, dtype='string')` makes no NumPy sense; NumSharp quirk. Document or remove |
| R41 | `np.empty/empty_like` on F-contig source | Relies on Shape strides driving sequential init — fragile if `np.ones`/`np.zeros` ever materializes via row-major |
| R42 | `NpyAxisIter.cs` (492 LoC) | Parallel implementation of NpyIter with 64-dim cap — could be replaced by NpyIter.MultiNew with op_axes |
| R43 | `NpyIterKernels.NpyIterExecution` | Dead code with TODOs left behind from migration; `NpyIterPathSelector.Strided` (AVX2 gather) not wired into main path |
| R44 | `NpyIterCoalescing.cs:120-167::TryCoalesceInner` | Appears not invoked from main path; possibly leftover |

---

## Summary statistics

| Metric | Count (audit) | Count (after fact-check) |
|---|---|---|
| Files audited | 189 (full coverage) | 189 |
| Tier 1 findings filed | **66** | 65 numbered (T1.20 is header) + 3 newly discovered = **68** |
| **Tier 1 — Confirmed correctness bugs (with failing test)** | — | **60 + 3 new = 63** |
| Tier 1 — False positives (revoked) | — | **4** — T1.21, T1.41, T1.45, T1.66 |
| Tier 1 — Latent (not reproducible) | — | **1** — T1.40 |
| Tier 1 — Duplicates | — | **1** — T1.43 ≡ T1.16 |
| Tier 2 perf regressions ≥2× | **35+** (15+ at ≥10× threshold) | 35+ (3 ratios re-measured lower; see fact-check status section) |
| Tier 3 API parity gaps | **30+** functions | unchanged |
| Tier 4 missing functionality categories | **6** | unchanged |
| Tier 5 refactor opportunities | **44** | unchanged |

**Failing tests filed:** 8 test files under `test/NumSharp.UnitTest/AuditV2/` totaling **119 tests** (115 currently failing as `[OpenBugs]` documenting real bugs, 3 sibling-guard tests passing, 1 inconclusive perf marker). Build: clean on `net8.0` and `net10.0`. Run via:
```bash
dotnet test --no-build --filter "TestCategory=OpenBugs&FullyQualifiedName~AuditV2"
```

---

## Domain-by-domain coverage

| # | Domain | Files | LoC delta | Report path | Top severity finding |
|---|---|---|---|---|---|
| 1 | Iterators (NpyIter, NpyAxisIter, NpyExpr, NDIterator) | 17 | ~12,000+ | `audit_v2/01_iterators.md` | T1.1, T1.2 — segfault + 20× perf |
| 2 | IL kernels + SIMD | 25 | ~3,300+ | `audit_v2/02_ilkernel_simd.md` | T1.6 — NaN comparison bug |
| 3 | DefaultEngine math/reductions/BLAS | 27 | ~3,800+ | `audit_v2/03_default_math_reductions.md` | T1.3, T1.4, T1.5 — crashes + wrong shape |
| 4 | Logic/Shape/Storage | 21 | ~1,500+ | `audit_v2/04_logic_shape_storage.md` | T1.10 — Shape mutating set; nonzero 29×; all/any 13× |
| 5 | NDArray core + Creation | 22 | ~1,300+ | `audit_v2/05_ndarray_creation.md` | T1.7 — `np.array` copy default; T1.8 NEP50 violation |
| 6 | Manipulation/APIs/Logic | 20 | ~1,400+ | `audit_v2/06_manipulation_apis_logic.md` | T1.17, finfo.minexp off-by-one, ravel('F') 3000× |
| 7 | Math/Selection/Sorting/Stats | 15 | ~1,300+ | `audit_v2/07_math_ops_selection_sorting_stats.md` | argsort 184×, searchsorted broken |
| 8 | Casting/Random/Utilities/Primitives | 22 | ~5,400+ | `audit_v2/08_casting_random_utilities.md` | T1.19 NpFunc landmine; ArrayConvert ~45 missing cases |
| **Total** | **8 domains** | **189** | **24,958 / 7,493** | | |

---

## Priority-ordered action plan

### Phase 1 — Critical correctness (block release)

| # | Task | Effort | Files |
|---|---|---|---|
| 1 | Fix `Iternext()` EXLOOP + buffered-non-reduce branches (T1.1, T1.2) | 2-3 d | `NpyIter.cs:1985-2003` |
| 2 | Fix IL NaN comparison `<=`/`>=` (T1.6) | 1 h | `ILKernelGenerator.Comparison.cs` |
| 3 | Fix `np.array` default `copy=True` (T1.7) | 1 d (with migration warning) | `np.array.cs` |
| 4 | Fix `Default.Dot.cs` axis hardcoded (T1.5) | 2 h | `Default.Dot.cs:60` |
| 5 | Guard `PowerInteger`/`ReciprocalInteger` on `Unsafe.Address` (T1.3, T1.4) | 1 h | `Default.Power.cs`, `Default.Reciprocal.cs` |
| 6 | Remove `Shape` mutating set indexer (T1.10) | 1 d | `Shape.cs` |
| 7 | Add SByte/Half/Complex to `NpyIterCasting` (T1.9, T1.12) | 1 d | `NpyIterCasting.cs` |
| 8 | Add SByte/Half/Complex to `UnmanagedStorage.SetValue`/`CopyTo` (T1.13) | 1 d | `UnmanagedStorage.Setters.cs` |
| 9 | Fix `np.concatenate` NEP50 + missing dtypes (T1.8, T1.9) | 1 d | `np.concatenate.cs` |
| 10 | Fix `np.convolve` int64 precision (T1.14) | 1 d | `NdArray.Convolve.cs` |
| 11 | Implement `SetIndicesNDNonLinear` (T1.15) | 2 d | `NDArray.Indexing.Selection.Setter.cs` |
| 12 | Fix NpFunc cache target inclusion (T1.19) | 1 h | `NpFunc.cs` |

**Phase 1 total: ~14 days**

### Phase 2 — High-value performance (10-200× gaps)

| # | Task | Expected gain |
|---|---|---|
| 13 | Replace `ndarray.argsort` LINQ with stride-aware `LongIntroSort` over typed pointers | 100-200× |
| 14 | Cache-blocked SIMD matmul (4×4 register tiles, 64-element blocks) + multi-threading | 10-50× |
| 15 | Rewrite `np.nonzero` to use `NonZeroSimdHelper<T>` (already exists, dead code) | 5-30× |
| 16 | Add SIMD `AllSimdHelper<T>`/`AnySimdHelper<T>` paths | 5-13× |
| 17 | Fix `np.clip` general/transposed path: route through NpyIter | 5-15× |
| 18 | Fix `ravel('F')` to return view when F-contig | 3000× |
| 19 | Single-pass cast+layout for `astype(_, 'F')` | 9× |
| 20 | Replace `Storage.GetValue` boxing in `np.searchsorted`/`convolve` with typed IL | 5-20× |
| 21 | F-contig fast path in `ILKernelGenerator.Binary.cs` (no scalar fallthrough) | 13× |
| 22 | Add SIMD broadcast-add fast path | 5× |
| 23 | Replace `flat.SetAtIndex` in `np.eye` with typed pointer writes | 10× |
| 24 | NpFunc `np.nan{mean,std,var}_axis*` route through `TryGetNanAxisReductionKernel` | 3-12× |

**Phase 2 total: ~3 weeks** (90% of files reuse existing infrastructure)

### Phase 3 — API parity gaps

| # | Task | Effort |
|---|---|---|
| 25 | `np.repeat axis=` | 0.5 d |
| 26 | `np.searchsorted side=, sorter=` | 1 d |
| 27 | `np.expand_dims` tuple axis | 0.5 d |
| 28 | `np.copyto casting=, where=` | 0.5 d |
| 29 | `np.unique return_index/inverse/counts/axis/equal_nan` | 2 d |
| 30 | `np.all/np.any` tuple axis + `where=`, `out=` | 1 d |
| 31 | `np.linspace retstep=, axis=` | 0.5 d |
| 32 | `out=` parameter for `np.dot`/`np.matmul`/`np.power`/`np.clip`/`np.modf`/`np.negative` | 1 w |
| 33 | `np.matmul(1D, 2D)` support | 0.5 d |
| 34 | F-order reshape `-1` placeholder | 0.5 d |

### Phase 4 — Missing functions

| # | Task | Effort |
|---|---|---|
| 35 | `np.sort` | 2 d |
| 36 | `np.flip` family | 0.5 d each |
| 37 | `np.diag`/`np.diagonal`/`np.trace` | 1 d each |
| 38 | `np.diff`/`np.gradient`/`np.ediff1d` | 1 d each |
| 39 | `np.pad` | 2 d |
| 40 | `np.rot90` | 0.5 d |
| 41 | `np.round` (alias) | 0.5 h |
| 42 | DateTime64 unit support + parser | 1 w |
| 43 | Register Char8/DateTime64 as NPTypeCode | 2 d |

### Phase 5 — Refactor/code quality

- R1-R19 listed above; ~2 weeks combined.

---

## Audit V1 corrections (overstated/inaccurate claims found by V2)

| V1 claim | V2 measurement | Correction |
|---|---|---|
| `nanmean/std/var` axis "100-1000× slower" | Actual: 3-4× slower | Still worth fixing; not catastrophic |
| `linspace` integer "10-30× slower due to `Converts.ToInt32` boxing" | Actual: 2.3× slower; `ToInt32(double)` is a static overload (no boxing) | Proposed `(int)(start + i*step)` fix would BREAK NumPy NaN/overflow parity. **Do not apply.** |
| `np.eye` "boxes per diagonal element" | Only ~10K `SetAtIndex` calls for `eye(10000)`; real bottleneck is `np.zeros` scalar init (~178ms of 181ms total) | Fix `np.zeros` init, not `eye` |
| Bug 1 "np.maximum returns wrong (5 instead of NaN) for NaN" | Maximum/minimum now correct (NaN-propagating); `fmax`/`fmin` still wrong (should skip NaN) | Only `fmax`/`fmin` needs the fix |
| Bug 2 "PowerInteger silently produces wrong output on strided" | Now CRASHES (`InvalidOperationException`) — guarded by `Unsafe.Address` check | Same root cause, worse symptom; still must be fixed |

---

## Independently verified ground-truth (orchestrator spot-checks)

(V1-V12 from earlier audit — all preserved and confirmed by domain agents.)

### V1 — `np.power` on sliced integer arrays CRASHES

See T1.3.

### V2 — `np.maximum`/`np.minimum` correct, `np.fmax`/`np.fmin` wrong

See T1.11.

### V3 — `np.repeat` axis parameter missing

See Tier 3.

### V4 — `np.linspace` integer dtype perf ~2.3× NumPy (not 10-30×)

| Case | NumPy | NumSharp | Ratio |
|---|---|---|---|
| `linspace(0, 1000, 1M, int32)` | 3.0 ms | 7 ms | 2.3× |
| `linspace(0, 1000, 1M, float64)` | 1.7 ms | 3 ms | 1.8× |

### V5 — `np.matmul` 50-190× slower at large sizes

See Tier 2.

### V6 — `np.argsort` 225× slower (V2 measured 184× at 1000×1000)

See T2.

### V7 — Axis reductions 9-40× slower

See Tier 2.

### V8 — Contiguous flat reductions are baseline-fast (1ms for sum 1M f64)

### V9 — Memory-layout sensitivity sweep (add 1000×1000 f64 ×10)

| Layout | NumPy | NumSharp | Ratio |
|---|---|---|---|
| Contiguous | 15.8 ms | 19 ms | 1.2× ✓ |
| Strided `::2, ::2` | 6.3 ms | 20 ms | 3.2× |
| Broadcast `(1000,1)+(1,1000)` | 16.7 ms | 81 ms | 4.8× |
| F-order via `.T` | 17.3 ms | **225 ms** | **13×** |
| Contig + scalar | 15.8 ms | 23 ms | 1.5× ✓ |

### V10 — Broadcast scalar add 4.8× slower

See V9.

### V11 — New dtype perf (Half/Complex)

| Op | NumPy | NumSharp | Ratio |
|---|---|---|---|
| `add Half 1M ×10` | 28.9 ms | 58 ms | 2.0× |
| `sum Half 1M ×10` | 24.0 ms | **121 ms** | **5.0×** |

### V12 — Creation/Manipulation perf

| Op | NumPy | NumSharp | Ratio |
|---|---|---|---|
| `eye(5000)` f64 | 3.4 ms | 31 ms | 9.1× |
| `unique 1M int32` | 2.6 ms | 10 ms | 3.8× |
| `where 1M ×10` | 17.9 ms | 27 ms | 1.5× ✓ |
| `tile arange(100) ×10000` | 1.35 ms | 9 ms | 6.7× |

---

## Methodology notes

**NumPy version:** 2.4.2
**.NET SDK:** 10.0.101
**Platform:** Windows 11 / win32
**Date:** 2026-05-13

**Per-agent file count:**

| Group | Files | Findings count by severity |
|---|---|---|
| 1. Iterators | 17 | 8 bugs, 6 parity-gaps, 1 perf catastrophic, 4 refactors, 5 clean |
| 2. IL kernels + SIMD | 25 | 1 bug (NaN cmp), 8 parity-gaps (SIMD coverage), 4 perf, 8 clean |
| 3. DefaultEngine math/reduction/BLAS | 27 | 3 bugs (Power/Reciprocal/Dot), 5 parity-gaps, 5 perf, 6 clean |
| 4. Logic/Shape/Storage | 21 | 4 critical (OWNDATA, Shape mutation, all/any perf, nonzero perf), 5 medium, 7 minor |
| 5. NDArray core + Creation | 22 | 3 critical (np.array copy, concat dtype, concat crash), 6 medium, ~15 minor |
| 6. Manipulation/APIs/Logic | 20 | 7 HIGH (finfo, expand_dims, copyto, repeat, ravel, unique), 8 MEDIUM, 6 LOW |
| 7. Math/Selection/Sorting/Stats | 15 | 13 bugs, 7 perf gaps (argsort 184×, convolve 67×, etc.) |
| 8. Casting/Random/Utilities/Primitives | 22 | 1 critical (NpFunc cache), 2 high, 3 medium, ~15 verified-clean |
| **Total** | **189** | **66 bugs, 35+ perf, 30+ parity, 6 missing, 44 refactor** |

**Major themes across all groups:**

1. **Boxing in inner loops** — 7+ ops box per element (`GetValue/SetValue/GetAtIndex` + `Converts.ToDouble/ChangeType`). Resolved by IL kernels or `NpFunc.Invoke` (kernel-level only).
2. **Missing SByte/Half/Complex** — 12+ paths assume 12 dtypes (not 15): `CopyToBuffer`, `IsSafeCast`, `ReadAsDouble`/`WriteFromDouble`, `ConstNode`/`WhereNode`/`CallNode`, NaN kernels, `ArrayConvert.cs` inner switches, `UnmanagedStorage.SetValue(object)`/`CopyTo`, `np.asanyarray IEnumerable<>` cases, `Converts.cs ChangeType<TIn,TOut>` 12×12 ladder.
3. **F-order penalties** — `astype`/`copy`/`eye`/`ravel`/`concatenate` use 2-allocation patterns (cast→copy('F')) costing 9-3000× slowdown.
4. **Hand-rolled LINQ + virtual dispatch** — argsort (184×), nonzero (29×), all/any (13×), bool-mask setter (36×), repeat per-element boxing.
5. **NpyIter not used where it should be** — `nan{mean,std,var}_axis*` 1500 LoC could shrink to ~150 by routing through existing `ILKernelGenerator.TryGetNanAxisReductionKernel`.
6. **MatMul gap to OpenBLAS** — 8-190× depending on size; expected without native BLAS link but could be improved 3-5× via multi-threading + AVX-512 + better cache blocking.
7. **`Iternext()` ignores EXLOOP + buffered-non-reduce** — single root cause of buffer overruns AND 20× perf gap in iterator-driven user code; the bridge layer avoids both bugs but they remain in the public API.

**Audit V1 ("NDITER_BRANCH_QUALITY_AUDIT.md") corrections summary:**

| V1 claim | V2 measurement | Status |
|---|---|---|
| `nanmean/std/var` axis "100-1000× slower" | 3-4× slower | V1 overstated |
| `linspace` integer "10-30× slower (boxing)" | 2-2.5× slower | V1 fix would BREAK NumPy NaN parity — do not apply |
| `np.eye` "boxes per diagonal element" dominates | Actual root cause: `np.zeros` scalar init loop (no SIMD) | V1 wrong direction |
| Bug 1 "np.maximum returns 5 for (5,NaN)" | maximum/minimum now correct; only `fmax`/`fmin` wrong | V1 partially outdated |
| Bug 2 "PowerInteger silently wrong on strided" | Now CRASHES (`InvalidOperationException`) | Worse symptom; same root cause |
| `argsort` "100-1000× slower" | 18× (1D) to 184× (2D 1000×1000) | V1 in right range |
| `searchsorted` boxing | 19× slower (confirmed) | V1 accurate |
| Bug 5 "Iternext ignores EXLOOP" | Confirmed + split into 2 distinct bugs (EXLOOP + BUFFERED non-reduce) with segfault repro | V1 understated severity |

**Reference reports:** `docs/plans/audit_v2/01_iterators.md` through `08_casting_random_utilities.md` — full per-file, per-function detail with line numbers and reproductions.

**Prior audit:** `docs/plans/NDITER_BRANCH_QUALITY_AUDIT.md` (V1) — see "Audit V1 corrections" section above for findings the V2 audit revised.
