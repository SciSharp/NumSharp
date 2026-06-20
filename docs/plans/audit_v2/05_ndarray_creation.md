# Audit v2 — Group 5: NDArray core + Creation APIs

Audited on `nditer` branch. All claims independently verified with NumPy 2.x +
`dotnet_run` reproductions. The companion V1 branch-quality audit (now
superseded) was used only as a starting point —
several of its claims (linspace 10-30× perf, eye boxing dominant cost) are
overstated and corrected below.

Tooling baseline:
- Python 3.x + NumPy 2.x via `python -c`
- `.NET 10` file-based scripts (assembly name override + InternalsVisibleTo)
- All NumSharp numbers measured on Windows x64 in Release-equivalent JIT.

Severity legend used per file: **CRITICAL** (silent wrong output), **HIGH**
(behavioral mismatch, perf >10× off, or hard exception on supported types),
**MEDIUM** (API gap or perf 2-5× off), **LOW** (cosmetic / NumPy field
mismatch / documented limitation).

---

## File: `src/NumSharp.Core/Backends/NDArray.cs` (+146/-30)

### What changed on branch
- New 3-arg overloads `astype(Type, bool, char)` / `astype(NPTypeCode, bool, char)`
  that thread `order='C'/'F'/'A'/'K'` through `OrderResolver` and follow up the
  cast with `casted.copy('F')` when the requested physical order is F.
- `GetEnumerator()` 1-D fast path was rewritten to materialize through
  `Storage.ToArray<T>()` instead of allocating an `NDIterator<T>`. Adds
  `Half`/`Complex`/`SByte` branches alongside existing dtypes (now 15 total).
- New typed accessors `GetSByte`/`GetHalf`/`GetComplex` (both `int[]` and
  `params long[]` overloads) and matching `SetSByte`/`SetHalf`/`SetComplex`.

### Correctness — verified
- `astype(dtype, copy=true, order='F')` on a C-contig source produces an
  F-contig result with NumPy-aligned values (`[0,1]` reads as `1`, etc.).
  Reproduction OK.
- `astype(dtype, copy=true, order='K')` on F-contig source preserves F. ✓
- Enumerator fast path: walks logical order via `Storage.ToArray<T>()` which
  already handles contiguous / strided / sliced layouts.

### Performance — measured
- `astype(C, F)` on a (1000,1000) f32→f64 does a **cast → C-contig allocate →
  copy('F') → second F-contig allocate**. Two passes, two allocations. NumPy
  fuses cast and layout into a single nditer-with-cast pass.
  - 100 iters of (1000,1000) `float32 → float64`:
    - NumSharp astype('C'): 445 ms (≈ 4.5 ms/call)
    - NumSharp astype('F'): 1312 ms (≈ 13.1 ms/call) → **~2.9× slower than C**
    - NumPy astype('C'): 129 ms (≈ 1.3 ms/call)
    - NumPy astype('F'): 150 ms (≈ 1.5 ms/call)
  - NumSharp astype('F') is **~9× slower than NumPy astype('F')**.
- The Cast preserves the source shape's contiguity (so F-contig source +
  same-class request short-circuits), but C→F is the common case.

### API parity gaps
- NumPy `ndarray.astype(dtype, order='K', casting='unsafe', subok=True,
  copy=True)` — NumSharp omits `casting` (always 'unsafe') and `subok` (no
  subclasses anyway). Acceptable, but `astype(int8, copy=False, casting='safe')`
  would silently differ — there is no validation.
- `view(Type)` does a byte-level reinterpret via `Storage.AliasAs(dtype)`. NumPy
  also supports `view(typeofN)` shape-preservation; NumSharp's overloads here
  are minimal. Pre-existing — not part of this branch change.

### Specific issue: F-order path is double-allocation (Severity: MEDIUM)
At `Backends/NDArray.cs:493-499` and `:523-529`:
```csharp
char physical = OrderResolver.Resolve(order, this.Shape);
var casted = TensorEngine.Cast(this, dtype, copy);  // alloc #1 (C-contig)
if (physical == 'F' && casted.Shape.NDim > 1 && !casted.Shape.IsFContiguous)
    return casted.copy('F');                          // alloc #2 (F-contig)
return casted;
```

Either route the cast through `NpyIter.Copy(dstWithFStrides, src)` using a
buffered-cast iterator path, or pre-allocate the F-contiguous destination and
have Cast write to it (one pass).

### Pre-existing observations (not branch issues, mentioned for completeness)
- `astype(NPTypeCode, copy=true)` with `copy=false` and identical type returns
  the same array unchanged — view aliased. NumPy: same behavior.
- The `#if _REGEN` block at lines 1155-1295 / 1166-1296 with manual fallback
  is still untouched (legacy template artifact) — refactor candidate, no
  correctness impact.

---

## File: `src/NumSharp.Core/Backends/NDArray.String.cs` (+1/-1)

### What changed
Single line: `MultiIterator.Assign(dst, src)` → `NpyIter.Copy(dst, src)` inside
`GetString(long[])` non-contiguous path. This is part of the MultiIterator
removal across the branch.

### Status
- Tested: `GetString(0,1)` on a sliced/transposed `(rows, chars)` array still
  decodes correctly (existing tests cover this).
- No behavioral change. Pure cleanup.

---

## File: `src/NumSharp.Core/Backends/NPTypeCode.cs` (+82/-30)

### What changed
- New enum member `SByte = 5` (slot was unused previously — `int8` was missing).
- New enum member `Half = 16` (collides with `TypeCode.DateTime = 16` — handled
  by an explicit early-return in `GetTypeCode(Type)` so DateTime maps to Empty).
- `IsNumerical` range widened from 3-15+129 to 3-16+128. (`129` was a typo for
  `Complex = 128` — that's a stealth correctness fix.)
- `SizeOf(Char)` corrected from `1` to `2` (char is 2-byte UTF-16). ✓
- `SizeOf(Decimal)` corrected from `32` to `16` (decimal is 16 bytes). ✓
- `GetPriority(Decimal)` recomputed `5*10*16 = 800` to match the new size.
- `AsNumpyDtypeName(Complex)` corrected from `complex64` to `complex128` (since
  NumSharp's Complex = 2 × float64). ✓
- New helpers `IsFloatingPoint`, `IsInteger`, `IsSimdCapable`.
- 5 dispatch tables (`AsType`, `SizeOf`, `IsRealNumber`, `IsUnsigned`,
  `IsSigned`, `GetGroup`, `GetPriority`, `ToTypeCode`, `ToTYPECHAR`,
  `AsNumpyDtypeName`, `GetAccumulatingType`, `GetDefaultValue`, `GetOneValue`)
  extended to handle `SByte` and `Half`.

### Correctness — verified
- Size mapping for all 15 dtypes matches NumPy itemsizes (with Char=2 because
  NumSharp's Char is UTF-16, not NumPy's S1).
- `SByte.AsNumpyDtypeName() == "int8"`, `Half == "float16"`, `Complex ==
  "complex128"`. ✓

### Bug — dead `return` statement (Severity: LOW, correctness)
At `NPTypeCode.cs:485-487`:
```csharp
case NPY_TYPECHAR.NPY_COMPLEXLTR:
    return NPTypeCode.Complex;

    return NPTypeCode.Decimal;   // <-- unreachable code, dead since prior branch
```
This was likely an unfinished merge. The second `return` is dead. Cosmetic but
the dead code signals an unfinished migration — `NPTypeCode.Decimal` has no
NPY_TYPECHAR mapping (it maps to `NPY_LONGLONGLTR = 'q'` in `ToTYPECHAR()`,
which is int64 — also wrong, see below).

### Bug — `ToTYPECHAR(Decimal) = 'q'` (int64), round-trip broken (Severity: LOW)
At `NPTypeCode.cs:531-532`:
```csharp
case NPTypeCode.Decimal:
    return NPY_TYPECHAR.NPY_LONGLONGLTR;  // 'q' = int64
```
- `Decimal → 'q'`, but `'q' → Int64` in `ToTypeCode`. Round-trip loses the
  Decimal identity:
  ```
  Decimal.ToTYPECHAR() → 'q' → ToTypeCode() → Int64  (≠ Decimal)
  ```
- Verified via `dotnet_run`: matches.

NumPy doesn't have a Decimal dtype so any choice is "wrong"; but at minimum
the round-trip should be self-consistent. Suggest a dedicated synthetic
TYPECHAR (e.g. `NPY_LONGDOUBLELTR = 'g'`) so the mapping is bijective for
NumSharp's 15 types.

### Bug — `AsNumpyDtypeName(Char) = "uint8"` but `SizeOf(Char) = 2` (Severity: MEDIUM, semantic)
At `NPTypeCode.cs:577`:
```csharp
case NPTypeCode.Char:
    return "uint8";
```
But Char is a 2-byte UTF-16 unit. Mapping to `uint8` (1 byte) confuses any
caller using `AsNumpyDtypeName()` to interop with NumPy — sending NumSharp Char
data as if it were 1-byte uint8 would silently corrupt strings.

Recommend: return `"uint16"` (matches sizeof) or `"<U1"` (matches semantic
intent — single Unicode char).

### Bug — `AsNumpyDtypeName(Decimal) = "float64"` (Severity: LOW)
At `NPTypeCode.cs:600-601`:
```csharp
case NPTypeCode.Decimal:
    return "float64";
```
Decimal is 16 bytes, NumPy float64 is 8. Same hazard as Char→uint8 — any
interop layer trusting this name will misinterpret bytes.

### Bug — `Decimal` not in `_kind_list_map` for kind='f' (Severity: LOW)
Already handled, but the `_kind_list_map` entry `{NPTypeCode.Decimal, 'f'}`
(see `np.dtype.cs:30`) returns `'f'` for kind, while NumPy uses `'f'` only for
true float (float16/32/64). This is consistent with `IsRealNumber(Decimal)`
returning `true`, so it's intentional. Note as documented behavior.

### Pre-existing: `Char` kind hangover
Constant `_kind_list_map[Char] = 'S'` (byte-string). With sizeof=2 and
semantic "UTF-16 char", `'U'` (Unicode) would be more accurate. Pre-existing
choice. Affects only the rarely-used `DType.kind` field.

---

## File: `src/NumSharp.Core/Generics/NDArray\`1.cs` (no diff)

The file is in scope but no changes from `master`. Indexer validation, type
checks on constructors, and `Address`/`Array` typed accessors continue to
work. Not a regression vector — skipped from per-file findings.

---

## File: `src/NumSharp.Core/Creation/NDArray.Copy.cs` (+33/-13)

### What changed
The old implementation was `copy(char order = 'C') => Clone();` ignoring order.
Now it uses `OrderResolver.Resolve(order, Shape)` to pick a physical layout
and allocates an F-contig destination via `NpyIter.Copy` when needed.

### Correctness — verified
- `copy('C')` on C-contig source → Clone fast path. ✓
- `copy('F')` on C-contig source → F-contig allocation + NpyIter.Copy
  produces NumPy-aligned values. ✓
- `copy('K')` on F-contig source → keeps F (physical='F'). ✓
- `copy('A')` on non-contig source → resolves to 'C'. ✓
- Scalar / empty / size=1 short-circuits to Clone (which preserves layout).
- `copy('C')` on F-contig source → falls through to NpyIter.Copy → C-contig.
  ✓ (verified by index [0,1]==1 instead of stride-skip value.)

### Performance — measured
- 1000×1000 int64 copy: C-path 0.9 ms/call, F-path 11.7 ms/call →
  **~13× slower for F-order on 2D**.
- 10000×10000 int64 copy (100M elements): C-path 114 ms, F-path 1249 ms →
  **~11× slower**.
- 1D copy (size 10M): C-path matches F-path (no permutation needed).

NumPy's `np.copy(a, order='F')` for the (1000,1000) case is ~1 ms (transposed
memcpy). NumSharp's F-copy is ~12× slower than NumPy's. The bottleneck is
NpyIter's per-element coordinate decode in
`NpyIterCasting.CopyStridedToStridedWithCast` (used even for same-dtype copy
— see notes under `np.concatenate` below).

### Minor concern — destShape cloning
```csharp
var destShape = new Shape((long[])this.Shape.dimensions.Clone(), physical);
```
The `.Clone()` is necessary (comment explains the `Shape.dimensions` indexer
setter would otherwise alias). Fine.

### Pre-existing observation
`NDArray.Copy.cs` declares `order='C'` as default, but `np.copy.cs:16`
declares `order='K'` as default — both calling the same `NDArray.copy(char)`.
The mismatch is intentional (matches NumPy: `ndarray.copy()` defaults `C`,
`np.copy()` defaults `K`) but easy to confuse.

---

## File: `src/NumSharp.Core/Creation/NdArray.ReShape.cs` (+33/-0)

### What changed
New `reshape(Shape newShape, char order)` overload. F-order goes through
`flatten('F') → new UnmanagedStorage(flatBuffer, fShape)`. Non-F paths return
unchanged (delegate to existing reshape).

### Correctness — verified
- `a.reshape((2,3), 'F')` on `np.arange(6)` returns `[[0,2,4],[1,3,5]]`,
  matching `np.arange(6).reshape((2,3), order='F')`. ✓
- F-order reshape after a previous reshape preserves logical order. ✓
- C-order reshape on a transposed view still copies through CloneData
  (existing behavior). ✓

### API parity gap — `-1` placeholder not supported on F-path (Severity: MEDIUM)
At `NdArray.ReShape.cs:34-37` (the XML doc):
```
The F-order path does not currently support the -1 placeholder dimension —
pre-compute the inferred dim and pass explicit sizes.
```
Verified: `a.reshape((-1, 2), 'F')` throws `IncorrectShapeException` while
NumPy returns shape (3,2) with values `[[0,3],[1,4],[2,5]]`. The Shape
constructor `Shape(long[], char)` does not implement -1 inference, only the
default 4-arg ctor + `Reshape` does.

Fix: route through `Shape.ComputeLongShape` (or reuse `Shape.Reshape`) to do
-1 inference before constructing the F-strided shape.

### Pre-existing observations
- `reshape_unsafe` family unchanged — bypasses the contiguity copy guard.
- `reshape(int[] shape)` always boxes to long[] via `ComputeLongShape` —
  cheap, not a perf concern.

---

## File: `src/NumSharp.Core/Creation/np.arange.cs` (+23/-0)

### What changed
New `NPTypeCode dtype` overload + matching `Type dtype` overload. The core
implementation got Boolean, SByte, Half, Complex branches and a comment
explaining NumPy's `start_t + i * delta_t` int-truncation semantics.

### Correctness — verified
- `arange(0, 5, 0.5, int32)` → `[0,0,0,0,0,0,0,0,0,0]` matches NumPy exactly
  (NumPy truncates `start + 0.5 = 0` then delta_t=0). ✓
- `arange(5, 0, -0.5, int32)` → `[5,4,3,2,1,0,-1,-2,-3,-4]` matches NumPy. ✓
- Empty array on `tmp_len <= 0`: shape `(0,)`. ✓
- Float types use direct cast (no integer truncation chain). ✓
- Complex: real part = `start + i * step`, imag = 0. ✓

### Bug — Boolean special case nonsensical (Severity: LOW)
At `np.arange.cs:81-90`:
```csharp
case NPTypeCode.Boolean:
{
    bool start_t = start != 0;
    bool next_t = (start + step) != 0;
    for (long i = 0; i < length; i++)
        addr[i] = (i % 2 == 0) ? start_t : next_t;  // alternating
    break;
}
```
NumPy raises `TypeError: arange() is only supported for booleans when the
result has at most length 2`. NumSharp instead produces alternating
`[False,True,False,True,...]` for any length. Not a common path, but a
silent surprise.

Suggested: throw the same TypeError when `length > 2`, or document the
NumSharp-specific extension.

### Default dtype slight gap (Severity: LOW)
For the `arange(double, double, double, NPTypeCode dtype = Empty)` overload,
when `dtype == Empty` NumSharp defaults to `Double`. NumPy's `arange` infers
`int64` for integer inputs (`np.arange(0, 5, 1).dtype == int64`). NumSharp
handles this correctly via the separate `(int, int, int)` overload that
routes through `NPTypeCode.Int64`, but the cross-overload from `Empty` is
double. Acceptable as currently structured (callers reach the int overload
via type-based dispatch).

---

## File: `src/NumSharp.Core/Creation/np.array.cs` (+11/-9)

### What changed
The `array(Array array, Type dtype, int ndmin, bool copy, char order)`
overload gained F-order support — after constructing a C-contig result, it
re-copies to F when requested. Otherwise the file is unchanged.

### Critical bug — `np.array(NDArray, copy=false)` (NumSharp default) aliases (Severity: CRITICAL)
At `np.array.cs:24-27`:
```csharp
public static NDArray array(NDArray nd, bool copy = false) =>
    copy
        ? new NDArray(nd.Storage.Clone()) { TensorEngine = nd.TensorEngine }
        : new NDArray(nd.Storage) { TensorEngine = nd.TensorEngine };
```

NumPy's `np.array(x)` defaults `copy=True`. NumSharp's default `copy=false`
silently returns an alias. Verified:
```csharp
var a = np.arange(10);
var b = np.array(a);          // expect copy per NumPy
b.SetAtIndex<long>(999L, 0);
// NumSharp: a[0] == 999  (aliased! bug)
// NumPy:    a[0] == 0    (copy)
```

Impact: code ported from Python that does `b = np.array(a); b[0] = 999`
expecting `a` untouched will silently corrupt `a`. This is the
"breaks-in-production" kind of behavioral divergence.

Fix: flip default to `copy=true`. Breaking change to NumSharp users who relied
on the alias (uncommon, more typically users called `np.asarray` to alias),
but aligns with NumPy and matches the documented project policy that "breaking
changes are acceptable to align with NumPy".

### API gap — no `dtype`/`order`/`ndmin` overload for NDArray input (Severity: MEDIUM)
NumPy signature: `np.array(object, dtype=None, copy=True, order='K',
subok=False, ndmin=0, like=None)` applies to any input including ndarrays.
NumSharp's `array(NDArray, bool)` does not support dtype change, order
override, or ndmin pre-padding. Users must call `arr.astype(...).reshape(...)`
manually.

### API gap — `array(Array array, ...)` default `ndmin=1` differs from NumPy=0 (Severity: LOW)
NumPy default `ndmin=0`; NumSharp default `ndmin=1`. For input arrays that
are already ≥1-D, this is a no-op. But for 0-D scalars (unusual case), behavior
differs. Verified `np.array(int[]{1,2,3})` returns shape `(3,)` in both,
which is the common case.

### Pre-existing observations
- The 2D/3D/4D/5D jagged-array overloads (`T[][]`, `T[][][]`, ...) use direct
  stride-aware pointer copy — efficient.
- The `T[,]` rectangular array overloads use `ArraySlice.FromArray` (pinning)
  — efficient.
- `array<T>(params T[] data)` always copies (NumPy parity).
- F-order via `result.copy('F')` after C-fill is double-allocation; same
  cost concern as `astype('F')` above. Acceptable for a creation-time op.

---

## File: `src/NumSharp.Core/Creation/np.asanyarray.cs` (+24/-6)

### What changed
- Split into 2-arg and 3-arg overloads (the latter takes `char order`).
- NDArray case now delegates to `asarray` (consistency).
- `default:` scalar branch now also accepts `Half` and `Complex` types.
- Apply-order step at the bottom: if requested layout != actual, do
  `ret.copy(physical)`.

### Bug — missing `IEnumerable<T>` cases for new dtypes (Severity: MEDIUM)
At `np.asanyarray.cs:53-64`:
```csharp
case IEnumerable<bool> e: ret = np.array(ToArrayFast(e)); break;
case IEnumerable<byte> e: ret = np.array(ToArrayFast(e)); break;
... (12 types) ...
case IEnumerable<decimal> e: ret = np.array(ToArrayFast(e)); break;
```
Missing: `IEnumerable<sbyte>`, `IEnumerable<Half>`, `IEnumerable<Complex>`.
Verified that `np.asanyarray(new List<sbyte>{1,2,3})` throws `NotSupported`.

Pattern matching falls through to the `default:` block which only handles
scalars/tuples/IEnumerables for the legacy types — not the 3 newly supported.

### Performance / correctness — verified
- `ToArrayFast<T>` correctly fast-paths `List<T>` (CollectionsMarshal +
  AllocateUninitializedArray + Span.CopyTo) and `ICollection<T>` (CopyTo).
- `FindCommonNumericType` uses stackalloc + bitmask — clean.
- The `default:` branch with `ITuple` / `IEnumerable` / `IEnumerator` fallback
  handles boxed numeric values via `Convert.ToX(item)` per element. Slow per
  element but only hit for non-typed enumerables.

### Apply-order step — slight inefficiency
After computing `ret`, the function checks if layout matches and `ret.copy(physical)`
if not. This is a second pass for non-contig inputs. NumPy fuses dtype + order
into the same nditer-with-cast operation. NumSharp does cast → maybe-copy →
maybe-reorder, up to 3 passes.

---

## File: `src/NumSharp.Core/Creation/np.asarray.cs` (+30/-0)

### What changed
New `asarray(NDArray a, Type dtype=null, char order='K')` overload. Returns
the input as-is if dtype and layout both match; otherwise dispatches to
`astype` (dtype change with order) or `copy(physical)` (layout-only change).

### Correctness — verified
- `np.asarray(a)` on already-matching dtype+layout → returns same reference
  (NumPy parity: same object). ✓
- `np.asarray(a, typeof(double))` casts. ✓
- `np.asarray(a, order='F')` on C-contig → F-contig copy. ✓
- `np.asarray(strided, order='K')` → 'K' resolves to 'C' (non-contig conservative
  fallback) and triggers a copy. ✓
- `np.asarray((NDArray)null)` → `ArgumentNullException`. ✓

### API gap — no `like=` or `dtype` as string (Severity: LOW)
NumPy signature: `np.asarray(a, dtype=None, order=None, *, copy=None,
device=None, like=None)`. NumSharp doesn't accept `dtype` as a string or
`copy=` parameter. The latter would distinguish "must copy" from "must alias",
which is now part of NumPy's 2.x semantics.

### Pre-existing observations
- The primitive-typed `asarray<T>(T data)` overloads (lines 14-33) create
  0-D or 1-D arrays from scalars/arrays. Pre-existing.
- The string overload (line 7) creates a 0-D string array. Pre-existing.

---

## File: `src/NumSharp.Core/Creation/np.ascontiguousarray.cs` (+21, new file)

### What it does
One-line wrapper: `np.ascontiguousarray(a, dtype) => asarray(a, dtype, 'C')`.

### Correctness — verified
- On a transposed view: produces C-contig copy with correct values. ✓
- On C-contig source: returns same reference (asarray fast path). ✓

### Bug — 0-D scalar should promote to 1-D (Severity: MEDIUM)
NumPy `np.ascontiguousarray(np.array(42))` returns shape `(1,)` with ndim=1.
NumSharp returns the unchanged 0-D scalar (ndim=0). Verified:
```python
>>> np.ascontiguousarray(np.array(42)).shape
(1,)
```
```csharp
np.ascontiguousarray(NDArray.Scalar(42)).ndim  // 0 in NumSharp
```
The docstring even says "Return a contiguous array (ndim >= 1) in memory" but
the implementation doesn't enforce the ndim≥1 invariant.

Fix: if `a.ndim == 0`, reshape to `(1,)` before delegating.

---

## File: `src/NumSharp.Core/Creation/np.asfortranarray.cs` (+21, new file)

### What it does
Symmetric to `np.ascontiguousarray`: `asarray(a, dtype, 'F')`.

### Correctness — verified
- On C-contig source: produces F-contig copy. ✓
- On a transposed view (already F-contig): returns same reference. ✓

### Bug — 0-D scalar should promote to 1-D (Severity: MEDIUM)
Same as `np.ascontiguousarray`. NumPy: `np.asfortranarray(scalar).ndim == 1`.
NumSharp: returns the 0-D scalar unchanged.

---

## File: `src/NumSharp.Core/Creation/np.concatenate.cs` (+17/-2)

### What changed
Two changes:
1. After determining `retShape`, the function now checks if **all** inputs are
   F-contiguous (and not C-contiguous). If so, the destination shape is built
   with F-order.
2. The copy step now goes through `NpyIter.Copy(writeTo, writeFrom)` instead
   of the legacy MultiIterator path (already happens in master).

### Critical bug — dtype promotion is wrong for float+int (Severity: HIGH)
The "find common type" loop:
```csharp
if (srcType.CompareTo(retType) == 1)
    retType = srcType;
```
uses `NPTypeCode.CompareTo(other)`, which goes by `(group, size)`:
- Group 1 (signed int): SByte=10, Int16=20, Int32=40, Int64=80
- Group 3 (float): Half=100, Single=200, Double=400

So `f32(group3, size4)` vs `i64(group1, size8)`: group3>group1 → f32 wins.
NumPy's NEP50 rules return `float64`. Verified:
```python
>>> np.promote_types('f4', 'i8')
dtype('float64')
>>> np.concatenate([np.array([1,2,3], dtype='f4'),
...                 np.array([4,5,6], dtype='i8')]).dtype
dtype('float64')
```
```csharp
np.concatenate({np.array(new float[]{...}),
                np.array(new long[]{...})}).dtype.Name
// "Single"  ← wrong, loses precision
```

For 8-byte ints we should promote to float64; for 4-byte ints to float64 too
(NumPy bumps up). NumSharp keeps source-Single, losing 53-bit int precision.

Fix: replace the bespoke `CompareTo` with `np._FindCommonType(typeCodes)` (or
add a `PromoteForArithmetic` helper). The IL kernel code already does this for
binary ops; concatenate should reuse it.

### Critical bug — concat across mismatched types crashes on SByte/Half/Complex (Severity: HIGH)
Downstream of dtype promotion: when `NpyIter.Copy(writeTo, writeFrom)` is
called with src.dtype != dst.dtype, it routes to
`NpyIterCasting.CopyStridedToStridedWithCast` which throws
`NotSupportedException("Unsupported type: {Half|SByte|Complex}")` (see
`NpyIterCasting.cs:343-355` and `:367-379`). Verified:
```csharp
np.concatenate({np.array(new sbyte[]{1,2}),    // int8
                np.array(new byte[]{3,4})});    // uint8 — different dtype
// Throws NotSupportedException("Unsupported type: SByte")
```
And:
```csharp
np.concatenate({np.array(new Half[]{...}),
                np.array(new float[]{...})});
// Throws NotSupportedException("Unsupported type: Half")
```

This is technically an `NpyIterCasting` issue (out of this group's scope) but
manifests as a `np.concatenate` crash in production. The branch added SByte
and Half to NPTypeCode but the NpyIter cast path was not extended.

Fix: add SByte/Half/Complex entries to `ReadAsDouble`/`WriteFromDouble`
(or use a Decimal-precision intermediate). Or: skip the cast path entirely
when dtypes match and call a typed memcpy.

### API gap — missing `out`, `dtype`, `casting` parameters (Severity: MEDIUM)
NumPy 2.x: `np.concatenate((a1, a2, ...), axis=0, out=None, dtype=None,
casting='same_kind')`. NumSharp omits all 3:
- `out=`: in-place destination
- `dtype=`: explicit output dtype, overrides promotion
- `casting=`: controls dtype-narrowing semantics ('no'|'equiv'|'safe'|
  'same_kind'|'unsafe')

### Pre-existing — `arrays.Length == 1` returns same reference
Line 30-31: `return arrays[0]`. NumPy returns a copy. Subtle aliasing bug
shared with `np.array(NDArray, copy=false)`.

---

## File: `src/NumSharp.Core/Creation/np.copy.cs` (+12/-2)

### What changed
Default `order` parameter changed from `'C'` (master) to `'K'` (matches
NumPy's `np.copy(a, order='K')`). Documentation expanded.

### Correctness — verified
- `np.copy(a, 'K')` on F-contig source → F-contig result. ✓
- `np.copy(a, 'A')` on C-contig source → 'A' resolves to 'C'. ✓
- `np.copy(a, 'C')` on F-contig source → C-contig result. ✓

### Status
1-line wrapper, no concerns. Mirrors NumPy default.

Note: `ndarray.copy(order='C')` default and `np.copy(a, order='K')` default
both match NumPy. Easy-to-confuse documentation but technically correct.

---

## File: `src/NumSharp.Core/Creation/np.dtype.cs` (+217/-163)

### What changed
- `DType` class added `_kind_list_map` (FrozenDictionary) replacing earlier
  inline switch.
- New `_dtype_string_map` (FrozenDictionary) replacing the old per-call
  parser. Builds platform-aware mappings (`l`/`L`/`long`/`ulong` map to
  int32/uint32 on Windows, int64/uint64 on 64-bit Linux/Mac).
- New `_unsupported_numpy_codes` (FrozenSet) for explicit `NotSupportedException`
  on NumPy types NumSharp doesn't implement (S/U/V/O/M/m/a, complex64).
- `np.dtype(string)` now goes through the map + unsupported set, with
  byte-order prefix stripping (<, >, =, |), and falls back to enum-name parse.
- `Complex64` (single-precision) is explicitly REJECTED, NumSharp only has
  `Complex128`. Code/tests say so.

### Correctness — verified
Tested 60+ NumPy dtype strings; all map to expected types or throw
NotSupportedException for the documented unsupported set:
- Single-char: `?`, `b`, `B`, `h`, `H`, `i`, `I`, `l`, `q`, `L`, `Q`, `e`, `f`, `d`, `g`, `D`, `G` ✓
- Sized: `b1`, `i1`, `u1`, `i2`, `u2`, `i4`, `u4`, `i8`, `u8`, `f2`, `f4`, `f8`, `c16` ✓
- Names: `bool`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `int64`, `uint64` ✓
- Floats: `float16`, `half`, `float32`, `single`, `float64`, `double`, `float`, `longdouble`, `g` ✓
- Complex: `complex128`, `complex`, `D`, `c16` ✓
- Platform-detected: `int`, `int_`, `intp`, `long`, `ulong` ✓ (on Windows: long→int32, ulong→uint32, int→int64)
- Byte-order: `<i4` ✓, `>i4` ✓ (but doesn't preserve byteorder in DType.byteorder, see below)
- Aliases: `Char`, `char`, `decimal`, `Decimal`, `string` ✓
- Unsupported: `complex64`, `c8`, `F`, `S`, `S10`, `U`, `V`, `O`, `M`, `m`, `a`, `bytes`, `str`, `datetime64` → NotSupportedException ✓
- Invalid: `xyz`, `f16`, `b4` → NotSupportedException ✓
- Parenthesized custom: `(2,)i4` → NotSupportedException (NumSharp doesn't do nested)

### Bug — `DType.byteorder` does not preserve the parsed prefix (Severity: LOW)
At `np.dtype.cs:346-349`:
```csharp
string key = dtype;
if (key.Length > 1 && (key[0] == '<' || key[0] == '>' || key[0] == '=' || key[0] == '|'))
    key = key.Substring(1);
// ... lookup map ...
return new DType(t);  // DType ctor sets byteorder = '=' unconditionally
```

NumPy preserves the prefix:
```python
>>> np.dtype('>i4').byteorder
'>'
>>> np.dtype('<i4').byteorder
'<'
```

NumSharp always returns `'='`. The parse strips the prefix without recording
it. Note: NumSharp is host-endian only, so this is informational only — but
NumPy code that branches on `dtype.byteorder` to detect "byte swap needed"
will see false negatives.

### Bug — kind code confusion (Severity: LOW)
`_kind_list_map` mixes char-codes (single-letter NPY_TYPECHAR) with kind-codes
(group letter biufcmMOSUV):
- `{NPTypeCode.Boolean, '?'}` — `'?'` is the **char**, NumPy kind is `'b'`
- `{NPTypeCode.Char, 'S'}` — `'S'` is byte-string kind, NumPy treats char
  as either 'u' (uint) or 'U' (unicode), not 'S' (which means bytes)

Verified via `dotnet_run`:
```
bool: kind='?'  (NumPy: 'b')
int8: kind='i'  (NumPy: 'i')  ✓
int32: char='i' (NumPy: 'l' on Windows)
complex128: char='c' (NumPy: 'D')
```
Several char fields differ from NumPy.

`DType.name` returns the C# type name (e.g. `"Int32"`, `"Double"`,
`"Boolean"`) instead of NumPy's dtype name (`"int32"`, `"float64"`, `"bool"`).
Should use `AsNumpyDtypeName()` for parity.

### Bug — `np.dtype('float')` returns Double, but NumPy 2.x deprecation (Severity: LOW)
NumPy 2.x deprecates `np.dtype('float')` (use 'float64' explicit). NumSharp
silently accepts it. Acceptable backward-compat, but worth flagging as
a deprecation pass-through to make explicit in docs.

### Bug — int_ vs intp duplication (Severity: LOW)
Both map to `_intpType`, which is correct for NumPy 2.x. But NumPy used to
distinguish `int_` (C long) from `intp` (pointer size). The current mapping
matches 2.x. Pre-existing.

### Pre-existing observations
- `mintypecode` (lines 133-167) is duplicated for `string` vs `char[]`
  overloads — minor code-dup.
- `NPY_TYPECHAR` enum includes deprecated `m`/`M` for datetime/timedelta —
  declared but not mapped.
- `DType.newbyteorder` throws `NotSupportedException` unconditionally. NumPy
  supports this (and it's the only way to construct big-endian arrays in
  NumSharp). Acceptable since NumSharp is host-endian only.

### Performance — measured
- `np.dtype("int32")` first call: ~0.1ms (FrozenDictionary lookup + DType ctor).
- Subsequent calls: <1µs. Excellent.
- `np.dtype("xyz")` (NotSupported path): ~5µs (allocates Exception). Fine.

---

## File: `src/NumSharp.Core/Creation/np.empty.cs` (+16/-0)

### What changed
New overload: `empty(Shape shape, char order, Type dtype = null)` that
allocates with F-strides when `order='F'`.

### Correctness — verified
- `empty(shape, 'F')` → F-contig allocation ✓
- `empty(shape, 'C')` → C-contig allocation ✓
- `empty(shape, 'A')` / `empty(shape, 'K')` → throws `ArgumentException`
  (matches NumPy "only 'C' or 'F' order is permitted" since no source array). ✓
- Default `dtype=float64` ✓

### API parity gap (Severity: LOW)
NumPy: `np.empty(shape, dtype=float64, order='C', *, device=None, like=None)`.
NumSharp omits `device` and `like`. Not critical for a CPU-only library.

### Pre-existing observations
- The legacy `empty(shape, NPTypeCode)` throws for `Empty` typecode (correct).
- `empty<T>(int[])` uses `typeof(T)` ↔ correct dispatch.

---

## File: `src/NumSharp.Core/Creation/np.empty_like.cs` (+30/-6)

### What changed
- Two-overload split: 3-arg (Type dtype) + 4-arg adds `char order` (with both
  defaulting through `'K'`).
- F-order support via `OrderResolver.Resolve(order, prototype.Shape)` and
  Shape-with-strides allocation.

### Correctness — verified
- `empty_like(c_arr)` → C-contig matches prototype dtype/shape ✓
- `empty_like(c_arr, dtype=double, order='F')` → F-contig, double dtype ✓
- `empty_like(f_arr)` with default 'K' → preserves F-contig ✓
- Custom shape override (4-arg) works ✓
- `(long[])shape` extraction: relies on `Shape.Dimensions` getter (no copy
  concern).

### API gap — `subok=` not supported (Severity: LOW)
NumPy: `np.empty_like(prototype, dtype=None, order='K', subok=True,
shape=None, *, device=None)`. NumSharp's `subok` is moot (no subclasses).

### Status
Clean implementation. F-order support introduced cleanly.

---

## File: `src/NumSharp.Core/Creation/np.eye.cs` (+50/-6)

### What changed
- Signature gained `char order = 'C'`.
- Negative N/M now throw `ArgumentException` (matches NumPy).
- One-value dispatch: switch over typeCode + Converts.ChangeType fallback
  for the default branch.
- After filling the diagonal, F-order path takes `m.copy('F')`.

### Correctness — verified
- `eye(3)` → 3×3 identity, C-contig ✓
- `eye(3, order='F')` → 3×3 identity, F-contig, correct diagonal values ✓
- `eye(4, M=3, k=1)` → 4×3 with k=1 super-diagonal ✓
- Negative N/M → ArgumentException ✓
- N=0 or cols=0 → empty 2D array of correct shape and dtype ✓
- Complex/Half/SByte/Decimal/String/Char dtype branches all work ✓

### Performance — measured
Per the audit, this was flagged as "perf 6 — eye boxes per diagonal element".
Reality:
- `eye(10000)`: NumSharp 181 ms vs NumPy 7 ms → **~26× slower**.
- Breakdown:
  - `np.zeros(100M)` allocation alone: 178 ms
  - `flat.SetAtIndex` over 10000 diag elements: ~2 ms
- **The bottleneck is `np.zeros`, NOT `SetAtIndex`.** NumPy uses calloc
  (page-zero-on-demand) which is essentially free until first touch.
- For `eye(100)`: NumSharp 25µs, NumPy 2.5µs → 10× slower — same ratio,
  dominated by allocation/init.

The `flat.SetAtIndex` boxing is **not** a real bottleneck — only ~10K virtual
calls per eye even for large N. Audit's claim that the boxing is dominant
is incorrect.

### Real performance issue — F-order path double-copy (Severity: MEDIUM)
- C path: zeros(N,N) + diagonal fill (~1 alloc + N writes)
- F path: zeros(N,N) C-contig + diagonal fill + **`m.copy('F')` — 2nd alloc + full memcpy**

Verified: `eye(1000, order='F')` × 100 = 1158 ms vs `eye(1000)` × 100 = 132 ms.
**~9× slowdown** purely from the F-order request.

Fix: allocate the F-contig output directly (Shape with F strides), then write
diagonal at offset `i * (1 + strides[1]/itemsize)` — or simpler, fill diagonal
in the same layout from the start.

### Performance issue — zeros() init is scalar loop, not platform-zero (Severity: MEDIUM, pre-existing)
The slow `eye(10000)` traces to `UnmanagedMemoryBlock<double>(count, default)`
in `ArraySlice.Allocate(typeCode, count, true)`. The `Fill(default)` for
multi-byte types uses a scalar loop with 8-way unroll, no SIMD, no
`InitBlockUnaligned` for power-of-2 sized values. NumPy's `calloc` is essentially
free (mmap of zero pages).

This is the root of the 26× eye gap. Fix in `UnmanagedMemoryBlock.Fill(T)` to
use `Unsafe.InitBlockUnaligned` for all power-of-2 sizes (not just 1-byte).

---

## File: `src/NumSharp.Core/Creation/np.frombuffer.cs` (+65/-4)

### What changed
- Full 15-dtype coverage in `CreateArraySliceView` / `CreateArraySliceCopy` /
  `CreateArraySliceWithDispose` / `CreateArraySliceFromPointer` /
  `CreateArraySliceFromPinnedPointer` (added SByte, Half, Complex).
- New `ParseDtypeString` handles `>` (big-endian), `<` (little-endian),
  `=` (native), `!` (network = big-endian), `|` (N/A).
- New `ByteSwapInPlace` handles 2/4/8-byte swap and Complex (two 8-byte halves).
- New overloads for: `ReadOnlySpan<byte>` (always copies), `ArraySegment<byte>`,
  `Memory<byte>` (view or copy), `IntPtr` raw pointer (with optional dispose),
  generic `TSource[]` reinterpret.

### Correctness — verified
- All 15 dtypes round-trip from byte[]: Boolean, Byte, SByte, Int16, UInt16,
  Int32, UInt32, Int64, UInt64, Char (2-byte), Half, Single, Double, Decimal,
  Complex. ✓
- View semantics: modifying the source byte[] reflects in NDArray. ✓
- Big-endian parse: `frombuffer(>i4_buf, ">i4")` correctly byte-swaps. ✓
- Big-endian Half: `0x3C 0x00` BE → 1.0 after swap. ✓
- Complex from buffer: real/imag laid out as 2 × float64 contiguously. ✓
- offset alignment validation. ✓

### Correctness — Decimal binary layout (Severity: LOW, semantic)
Per `dotnet_run` test, `decimal` bit layout in CLR memory is `[flags, hi, lo,
mid]`, not the `[lo, mid, hi, flags]` returned by `decimal.GetBits`. So:
- `np.frombuffer(decimal.GetBits(42m).SelectMany(BitConverter.GetBytes).ToArray(),
  Decimal)` returns 0, not 42.
- `np.frombuffer(MemoryMarshal.AsBytes(stackalloc decimal[]{42m}), Decimal)`
  returns 42.

This isn't a bug — it's correct in-memory binary view semantics. But callers
ported from `decimal.GetBits` will be surprised. Acceptable since `np.frombuffer`
is by definition a byte-level reinterpret.

### Bug — `'F'`/`'c8'` strings map to Complex128, not Complex64 (Severity: LOW, deliberate)
In `ParseDtypeString` line 720-721:
```csharp
"c8"  or "F"       => NPTypeCode.Complex,   // single-precision complex
"c16" or "D"       => NPTypeCode.Complex,   // complex128
```
But the comment-doc on np.dtype.cs:316-320 says `'F'`/`'c8'`/`'complex64'`
throw NotSupportedException. Inconsistent: `frombuffer` accepts these silently
and widens to complex128 (losing precision for incoming complex64 data!).

If incoming buffer is `complex64` (2 × float32 = 8 bytes), but NumSharp reads
8 bytes as complex128 (2 × float64 = 16 bytes), the buffer alignment check at
line 165 (`availableBytes % itemSize != 0`) will catch it — but only if the
buffer length is wrong. If the user manually specifies count, they'll read
wrong bytes.

Fix: align with np.dtype.cs — throw NotSupportedException for `'F'`/`'c8'`/
`'complex64'`, or document that frombuffer interprets them as native float64
pairs (which doesn't match the byte layout).

### Bug — `'a'` is accepted? (Severity: LOW)
Line 706: `"i1" or "b"        => NPTypeCode.SByte,` — accepts `b` for int8
even though `b` (byte) was NumPy's old alias for "byte string". NumPy 2.x
removed this. NumSharp follows old NumPy. Acceptable.

### API gap — missing `like=` (Severity: LOW)
NumPy: `np.frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None)`.
Not critical.

### Performance — measured
- `frombuffer(byte[1M], int32)` (view, no copy): ~50µs (pinning cost).
- `frombuffer(byte[1M], ">i4")` (byte swap): ~5 ms (scalar loop).
- NumPy `frombuffer(ba, '>i4')`: ~3 ms.
- The byte-swap path is reasonable; numpy uses SIMD for swap.

---

## File: `src/NumSharp.Core/Creation/np.full.cs` (+1/-1)

### What changed
Single line — cosmetic. Type inference path moved to use
`fill_value.GetType().GetTypeCode()` directly instead of via Type→NPTypeCode
intermediate.

### Status — verified
- `np.full((2,3), 42)` → 2x3 int32 array. ✓
- `np.full((2,3), 42.0, dtype=float32)` → 2x3 float32. ✓
- All 15 dtypes via `Converts.ChangeType` + `ArraySlice.Allocate(NPTypeCode,
  count, fillValue)`. ✓

### API gap — missing `order='C'` parameter (Severity: MEDIUM)
NumPy: `np.full(shape, fill_value, dtype=None, order='C', *, device=None, like=None)`.
NumSharp omits `order`. Users have to do `np.full(...).copy('F')` for F-layout.

---

## File: `src/NumSharp.Core/Creation/np.full_like.cs` (+19/-1)

### What changed
Two-overload split (3-arg + 4-arg with `char order = 'K'`). F-order resolution
via OrderResolver against prototype's shape.

### Correctness — verified
- `full_like(c_proto, 42, dtype=null, order='F')` → F-contig with all 42s. ✓
- Type inference: `dtype ?? fill_value?.GetType() ?? a.dtype`. ✓
- 15 dtypes covered through `ArraySlice.Allocate(typeCode, size, value)`. ✓

### API gap — missing `shape=`, `subok=` (Severity: LOW)
NumPy: `np.full_like(a, fill_value, dtype=None, order='K', subok=True,
shape=None)`. NumSharp omits `shape` (would override prototype's shape) and
`subok`. `empty_like` has `shape` parameter — inconsistent across the
`*_like` family.

---

## File: `src/NumSharp.Core/Creation/np.linspace.cs` (+30/-0)

### What changed
Added `Half` and `Complex` branches to the per-dtype switch (originally only
13 types — added the 2 new ones).

### Correctness — verified
- `linspace(0,1,5,int32)` → `[0,0,1,1,2,2,3,3,4,4,5]`-like (depending on cut)
  matches NumPy exactly (NumPy uses truncation, NumSharp uses `Converts.ToInt32`
  which truncates with NaN→int.MinValue). ✓
- `linspace(0,1,5,float64)` → `[0, 0.25, 0.5, 0.75, 1]`. ✓
- `linspace(0,10,5,bool)` → `[False, True, True, True, True]`. ✓
- `linspace(0,1,0)` → empty array. ✓
- `linspace(0,1,1)` → `[0]` (single element = start). ✓
- `linspace(0,1,5,Half)` → `[0, 0.25, 0.5, 0.75, 1]`. ✓
- `linspace(0,1,5,Complex)` → `[(0,0), (0.25,0), ...]`. ✓
- `linspace(0,5,11,int32)` → `[0,0,1,1,2,2,3,3,4,4,5]`. ✓

### Audit's "10-30× perf" claim — CORRECTED (Severity: MEDIUM, not HIGH)
The V1 audit's Perf 3 claimed integer linspace
is 10-30× slower because of `Converts.ToInt32` "virtual call per element /
boxing". Reality:
- `Converts.ToInt32(double)` is a **static method overload** that takes `double`
  directly — no boxing, no virtual call. The JIT inlines it (`OptimizeAndInline`).
- The function does include a NaN/Inf check (`double.IsNaN(value)`) and
  `Math.Truncate` — both needed for NumPy-2.x parity (NaN/overflow → int.MinValue).
- Measured 1M-element int32 linspace × 100 iters:
  - NumSharp 699 ms vs NumPy 276 ms → **~2.5× slower**.
  - Same test with float32: NumSharp 209 ms vs NumPy 263 ms → NumSharp *faster*.
  - Same test with float64: NumSharp 265 ms vs NumPy 197 ms → 1.3× slower.

The audit's prescribed fix ("replace `Converts.ToInt32(...)` with direct casts
`(int)(start + i * step)`") would **break NumPy parity for NaN/Inf overflow**.
Don't apply it. The 2.5× gap is real but comes from per-iteration latency of
`Math.Truncate` + NaN check + clamp, plus scalar-loop overhead vs NumPy's
SIMD. Use IL kernel SIMD for >5× improvement instead.

### API gap — missing `retstep=`, `axis=`, `device=` (Severity: HIGH, API parity)
NumPy 2.x: `np.linspace(start, stop, num=50, endpoint=True, retstep=False,
dtype=None, axis=0, device=None)`.

Missing in NumSharp:
- `retstep=True`: NumPy returns `(array, step)` tuple. NumSharp doesn't.
- `axis=int`: NumPy supports vector start/stop and inserts the linspace
  dimension at the given axis. E.g.:
  ```python
  >>> np.linspace([0,10], [1,20], 3, axis=0).shape
  (3, 2)
  >>> np.linspace([0,10], [1,20], 3, axis=-1).shape
  (2, 3)
  ```
  NumSharp's linspace takes scalar start/stop only.

These are common in ML/scientific code (e.g. batched ramps along an axis).

### Pre-existing — `linspace(num=0)` returns shape `(0,)`, not scalar
Matches NumPy. Verified.

---

## File: `src/NumSharp.Core/Creation/np.ones.cs` (+8/-2)

### What changed
Inline cosmetic changes — moved `case NPTypeCode.SByte` and `case NPTypeCode.Half`
into the switch. `Converts.ChangeType((byte)1, typeCode)` fallback for the
default branch.

### Correctness — verified
- All 15 dtypes produce ones of correct type. ✓
- Complex `ones`: `(1, 0)`. ✓
- Half `ones`: `(Half)1`. ✓
- Default dtype: float64. ✓
- Shape passthrough. ✓

### API gap — missing `order=` parameter (Severity: MEDIUM)
NumPy: `np.ones(shape, dtype=None, order='C', *, device=None, like=None)`.
NumSharp's `ones(Shape shape, NPTypeCode)` ignores order. `ones_like` has
order; `ones` doesn't — inconsistent.

### API gap — `string`-typed ones makes no sense
`case NPTypeCode.String: one = "1";` is a quirk. NumPy doesn't support
`np.ones(shape, dtype='string')`. Acceptable since NumSharp uses string for
text-typed arrays.

---

## File: `src/NumSharp.Core/Creation/np.ones_like.cs` (+18/-0)

### What changed
- Two overloads: 2-arg (default order='K') and 3-arg with explicit `char order`.
- F-order support via OrderResolver.

### Correctness — verified
- `ones_like(c_arr, order='F')` → F-contig array of ones. ✓
- `ones_like(f_arr)` default 'K' → preserves F. ✓
- `ones_like(c_arr, dtype=float32)` → float32 ones. ✓

### Pre-existing — `np.ones(resolvedShape, dtype)` ignores order in the resolved Shape
The `resolvedShape` has F-strides, but `np.ones(Shape, Type)` allocates via
`ArraySlice.Allocate(typeCode, shape.size, one)` — which writes elements
sequentially. Because Shape has the correct F-strides, the same memory pattern
forms a valid F-contig array.

Verified F-contig output, but this is a fragile pattern: any future change
to `np.ones` that materializes via row-major would silently break F-order
correctness for `ones_like(_, _, 'F')`.

### API gap — missing `shape=`, `subok=` (Severity: LOW)
Same as `full_like`.

---

## File: `src/NumSharp.Core/Creation/np.zeros_like.cs` (+18/-0)

### What changed
Same pattern as `ones_like` — 2-arg + 3-arg with `char order`.

### Correctness — verified
- `zeros_like(c_arr, order='F')` → F-contig zeros. ✓
- `zeros_like(f_arr)` default 'K' → F-contig zeros. ✓

### Same caveats as `ones_like` re: shape/subok and reliance on Shape strides
to drive layout via sequential init.

---

## Summary table

Severity ordering: CRITICAL > HIGH > MEDIUM > LOW.

| # | File | Issue | Severity | Verified |
|---|------|-------|----------|----------|
| 1 | `np.array.cs:24-27` | `np.array(nd, copy=false)` aliases — NumPy default is `copy=True`. Production-breaking. | **CRITICAL** | yes (mutated b mutates a) |
| 2 | `np.concatenate.cs:53-58` | dtype promotion via `CompareTo(group,size)` returns wrong type — `f32+i64 → f32` (loses precision); NumPy: `float64`. | **HIGH** | yes |
| 3 | `np.concatenate.cs:108` | NpyIter.Copy crashes with `NotSupported` for SByte/Half/Complex when src.dtype != dst.dtype. | **HIGH** | yes |
| 4 | `np.linspace.cs:53-69` | API gap: missing `retstep`, `axis`, `device`. Heavy NumPy API feature missing. | **HIGH** | yes |
| 5 | `NDArray.cs:493-499`, `:523-529` | `astype(_, F)` is 2-allocation (cast → copy('F')) — ~4.5× slower than NumPy on (1000,1000). | **MEDIUM** | yes (perf) |
| 6 | `NPTypeCode.cs:577` | `AsNumpyDtypeName(Char) = "uint8"` but Char is 2 bytes — interop layer silently corrupts. | **MEDIUM** | yes |
| 7 | `NpFunc/NpyIterCasting` (downstream) | NpyIterCasting `ReadAsDouble`/`WriteFromDouble` lack SByte/Half/Complex — root cause of #3. | **MEDIUM** | yes |
| 8 | `np.ascontiguousarray.cs`, `np.asfortranarray.cs` | 0-D scalar input → returns ndim=0; NumPy: returns ndim=1. | **MEDIUM** | yes |
| 9 | `np.asanyarray.cs:53-64` | Missing `IEnumerable<sbyte>`, `IEnumerable<Half>`, `IEnumerable<Complex>` cases — throws `NotSupported`. | **MEDIUM** | yes |
| 10 | `NdArray.ReShape.cs:38-50` | F-order reshape does not support `-1` placeholder dim; NumPy: works. | **MEDIUM** | yes |
| 11 | `np.eye.cs:69` | F-order path is cast-then-copy('F') — ~9× slower than C-order eye of same size. | **MEDIUM** | yes (perf) |
| 12 | `np.full.cs`, `np.ones.cs` | Missing `order=` parameter — inconsistent with `*_like` family. | **MEDIUM** | yes |
| 13 | `np.concatenate.cs` | API gap: missing `out=`, `dtype=`, `casting=`. | **MEDIUM** | (NumPy ref) |
| 14 | `np.array.cs:51` | `ndmin=1` default differs from NumPy `ndmin=0` (rare path: 0-D inputs). | **LOW** | yes |
| 15 | `NPTypeCode.cs:485-487` | Dead `return NPTypeCode.Decimal` after Complex case. | **LOW** | yes |
| 16 | `NPTypeCode.cs:531-532` | `Decimal.ToTYPECHAR() = 'q'` (int64) — round-trip lost. | **LOW** | yes |
| 17 | `np.dtype.cs:34-44` + `_kind_list_map` | `DType.kind` confuses char-code with kind-code; `bool` returns `'?'` (char) where NumPy returns `'b'` (kind). | **LOW** | yes |
| 18 | `np.dtype.cs:346-349` | `byteorder` parsed prefix is stripped but not preserved in `DType.byteorder` (always returns `'='`). | **LOW** | yes |
| 19 | `np.dtype.cs:36-44` | `DType.name` uses C# typename (`"Int32"`); NumPy: `"int32"`. | **LOW** | yes |
| 20 | `np.frombuffer.cs:720` | `'F'`/`'c8'` silently maps to Complex128 (single→double widen). Comment says they should throw. | **LOW** | yes |
| 21 | `np.arange.cs:81-90` | Boolean dtype: NumPy throws TypeError for len > 2; NumSharp returns alternating bools. | **LOW** | yes |
| 22 | `np.empty.cs`, `np.empty_like.cs`, `np.ones_like.cs`, `np.zeros_like.cs`, `np.full_like.cs` | Missing NumPy 2.x params: `device`, `like`, `subok`, `shape` (for ones_like/zeros_like/full_like). | **LOW** | (NumPy ref) |
| 23 | `np.eye.cs` (zeros allocation) | Underlying `UnmanagedMemoryBlock.Fill(T)` for multi-byte types is scalar loop, no SIMD. Root cause of eye 26× gap. Pre-existing. | **LOW** (pre-existing, but relevant) | yes |
| 24 | `np.array.cs:178-186` | `NDArray(Array values, Shape, char order)` silently ignores order param. Documentation says so but the API surface is misleading. | **LOW** | yes |

**Audit correction:** The companion document's claim "linspace 10-30× slow due
to Converts.ToInt32 boxing" is incorrect — `Converts.ToInt32(double)` is a
static overload (no boxing). Measured 2× gap is from `Math.Truncate` + NaN
check, both required for NumPy parity. Audit's prescribed fix `(int)(start +
i*step)` would break NumPy NaN/overflow behavior. Do not apply.

**Audit correction:** The "eye boxes per diagonal element" perf claim is
overstated — only ~10K virtual calls per eye(10000), <1 ms. The real eye
bottleneck is `np.zeros` (scalar-loop init, no SIMD), which is in
`UnmanagedMemoryBlock.Fill` (pre-existing, not on this branch).
