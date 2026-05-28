# Audit 08 — Casting + Random + Utilities + Primitives + Misc

Branch: `nditer` vs `master`
Scope: ~13 000 LoC across 24 files. Reviewed file-by-file; cross-verified behaviour against NumPy 2.4.2 and `dotnet_run` reproductions.

---

## 1. Files reviewed (LoC)

| File | LoC | Verdict |
|------|----:|---------|
| `Casting/Implicit/NdArray.Implicit.Array.cs` | 192 | Good |
| `Casting/Implicit/NdArray.Implicit.ValueTypes.cs` | 152 | Good (NumPy-aligned `int(complex)` → `TypeError`) |
| `Casting/NdArrayToMultiDimArray.cs` | 75 | Good (correct fast/slow path split for decimal) |
| `RandomSampling/np.random.dirichlet.cs` | 201 | Good (bit-identical to NumPy) |
| `RandomSampling/np.random.multivariate_normal.cs` | 590 | Good for ≤4×4; documented divergence for 5×5+ (statistically valid, not bit-identical) |
| `RandomSampling/np.random.randint.cs` | 124 | **2 bugs** (sentinel ambiguity, float dtype accepted) |
| `Utilities/ArrayConvert.cs` | 4 576 | **Bug** — SByte/Half/Complex SOURCE arrays not dispatched in inner switches |
| `Utilities/Arrays.cs` | 710 | Acceptable |
| `Utilities/Converts.cs` | 2 201 | Good (NumPy-compliant wrap/truncate behaviour verified) |
| `Utilities/Converts.Native.cs` | 3 859 | Good (full NEP50 parity confirmed) |
| `Utilities/Converts.Char8.cs` | 317 | Good |
| `Utilities/Converts.DateTime64.cs` | 228 | Good |
| `Utilities/Converts`1.cs` | 346 | Good |
| `Utilities/InfoOf.cs` | 104 | Good |
| `Utilities/NpFunc.cs` | 511 | **CRITICAL latent bug** — caches by `MethodHandle.Value` only, target ignored |
| `Utilities/NumberInfo.cs` | 107 | Good |
| `Primitives/Char8.cs` | 725 | Good (ASCII-strict NumPy parity + Latin-1 alt helpers) |
| `Primitives/Char8.Conversions.cs` | 261 | Good |
| `Primitives/Char8.Operators.cs` | 169 | Good |
| `Primitives/Char8.PyBytes.cs` | 531 | Good |
| `Primitives/Char8.Spans.cs` | 201 | Good |
| `DateTime64.cs` | 563 | Good as helper struct; **major API gap** — no NumPy unit support (Y/M/W/D/h/m/s/ms/us/ns/ps/fs/as) |
| `Exceptions/IndexError.cs` | 13 | Good |
| `Assembly/Properties.cs` | 8 | Good (adds `NumSharp.DotNetRunScript` for scripts) |

---

## 2. NumPy structural parity

### 2.1 `Converts.Native.cs` — NEP50 / wrap behaviour

Verified every problematic boundary against NumPy 2.4.2:

| Conversion | Input | NumPy | NumSharp |
|------------|-------|-------|----------|
| float → int32 | `NaN`, `±Inf`, overflow | `int.MinValue` | `int.MinValue` |
| float → int32 | `2147483647.4` | `2147483647` | `2147483647` |
| float → int8 | `500.5` | `-12` | `-12` |
| float → uint8 | `-1.5` | `255` | `255` |
| float → uint32 | `NaN`/`±Inf`/`-1.5` | `0`/`0`/`4294967295` | `0`/`0`/`4294967295` |
| float → uint64 | `NaN`/`±Inf` | `2^63` (NaT sentinel) | `2^63` (`NumPyUInt64Overflow`) |
| float → uint64 | `-1.5` | `2^64 − 1` | `2^64 − 1` |
| float → int64 | overflow/`NaN`/`±Inf` | `long.MinValue` | `long.MinValue` |
| uint64.MAX → int32 | wrap | `-1` | `-1` |
| complex → int32 | `(1.5, 99)` | `1` (imag discarded) | `1` |
| decimal → int32 | `1e18` | `int.MinValue` | `int.MinValue` |
| char(200) → sbyte | wrap | `-56` | `-56` |

The `unchecked((int)long.MinValue) == 0` insight for small-int NaN propagation (intermediate via `int.MinValue` whose low 8/16 bits are `0`) matches NumPy's `_PyUnicode_TruncatingDecode` path in `convert_datatype.c`.

The exclusive upper bound `9223372036854775808.0` (= 2^63) is consistently used; the comments note correctly that `(double)long.MaxValue` *rounds up* to 2^63 so a naive `<= long.MaxValue` check would miss out-of-range floats.

**Verdict:** Boundary handling is exemplary and matches the C source `convert_datatype.c` patterns from `src/numpy`.

### 2.2 `Casting/Implicit/NdArray.Implicit.ValueTypes.cs`

- Scalar → NDArray: implicit operator for `bool`, `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `char`, `Half`, `float`, `double`, `decimal`, `Complex` — **15 operators**, matches the 15 NumSharp dtypes.
- NDArray → scalar: explicit operator with `EnsureCastableToScalar` guard (matches `int(arr)` in NumPy 2.x — only 0-d arrays allowed).
- **Complex source guard**: `EnsureCastableToScalar` rejects `Complex` → non-complex scalar with `TypeError`. This goes one step further than NumPy (which only warns with `ComplexWarning` and silently discards `imag`). Documented as intentional in the file header. Reasonable trade-off in absence of a warning mechanism.

### 2.3 `Casting/Implicit/NdArray.Implicit.Array.cs`

- `implicit operator NDArray(Array)`: handles jagged AND multi-dim arrays for all 15 dtypes (including new `SByte`, `Half`, `Complex`).
- `explicit operator Array(NDArray)`: matching 15-case switch.
- `implicit operator NDArray(string)`: parses bracket-form text input (`"[1,2,3]"`, `"[1,2;3,4]"`). Looks functional but not NumPy-aligned (NumPy doesn't have implicit string→ndarray).

---

## 3. Behavioural parity (`python -c` reproductions)

### 3.1 Dirichlet — bit-identical
```
NumPy:    [0.09784297 0.62761396 0.27454307]
NumSharp: [0.09784297 0.62761396 0.27454307]
```

### 3.2 Multivariate normal — bit-identical ≤4×4, divergent ≥5×5
```
NumPy 2×2 identity:    [[ 0.49671415 -0.1382643 ], [ 0.64768854  1.52302986], …]
NumSharp 2×2 identity: [[ 0.49671415, -0.13826430], [ 0.64768854,  1.52302986], …]

NumPy 5×5 correlated:    [[-1.10483014, 0.07033543, …], …]
NumSharp 5×5 correlated: [[ 0.33070244,-1.09139252, …], …]
```
The 5×5 divergence is **documented** in the source (see lines 30–35 of `multivariate_normal.cs`). Caused by Jacobi vs LAPACK divide-and-conquer eigenvector sign conventions (DLAED3). The samples remain statistically correct.

### 3.3 randint — produces correct distribution but with bugs

Verified with `np.random.seed(42); randint(0, 100, 5)` → identical sequence.

---

## 4. Performance

### 4.1 Conversion path comparison (1 000 000 doubles → int32)

| Path | Time | ns/call | Notes |
|------|------|--------:|-------|
| `Converts.ToInt32(double)` (direct) | 7 ms | **6.7** | Fastest — inlined `Math.Truncate` + bounds |
| `Converts.ChangeType<object>(o, NPTypeCode.Int32)` | 24 ms | 24.3 | Boxing + switch dispatch |
| `Converts.ChangeType<TIn,TOut>(v)` (generic) | 42 ms | 41.5 | Two nested switches + `Unsafe.As` |
| `Converts.ToInt32(object)` | 41 ms | 41.4 | Boxing + pattern-match dispatch |
| `Converts<T>.ToInt32(T)` (cached) | 62 ms | 62.3 | Delegate indirection per element |

Notes:

- **`Converts<T>.To*`** uses `Converts.FindConverter<TIn,TOut>()` and caches a `Func<TIn,TOut>` per `(TIn,TOut)` pair. The cache hides the giant switch ladder behind a single delegate invocation but pays a ~55 ns delegate dispatch cost per call. For per-element use this is much slower than the inlined direct call. For bulk use through ILKernel this is irrelevant.
- **The 41 ns generic ChangeType path** still allocates **no** objects per call (uses `Unsafe.As<T,_>` reinterpret + value-type return). It is comparable to the boxing path because the inner double-switch is 1296 cases and not all branches predict well.
- **The boxing path** is hot enough that `value switch { … }` JITs to a virtual call table per case — about 25 ns per call, not as bad as a full `Convert.ChangeType`.

### 4.2 NpFunc dispatch

Per call, cache hit:

| Path | ns/call |
|------|--------:|
| Direct generic call `DoSomething<int>(…)` | 1.7 |
| Manual `switch (tc) { … }` | 1.5 |
| `NpFunc.Invoke(tc, DoSomething<int>, …)` | **31.7** |

NpFunc is **~20× slower** than a manual switch. This is acceptable for kernel-level dispatch (one call per array), unusable for per-element dispatch (millions of calls). All current 24 production call sites are kernel-level, so the overhead is amortised across millions of inner-loop iterations.

The cache mechanics:
- L1 (1 NPTypeCode): `ConcurrentDictionary<nint, Delegate[16]>` — handle → table indexed by `(int)tc`.
- L2/L3 (2/3 types): `ConcurrentDictionary<(nint,nint,nint), Delegate>` keyed on tuples.
- Right-sized tuples (5/6 nints) are claimed to be 33 % faster than padding to a fixed 6-nint key. Plausible but not directly benchmarked here.

### 4.3 NumPy comparison

Was not formally benchmarked here, but spot checks earlier in this branch (see audit 09) showed that NumPy `astype` is ~5–10× faster than NumSharp on large arrays. The conversion *hot loop* (ILKernel Cast) is competitive; the per-element `Converts.*` paths above are only used for scalar / object-typed inputs and are not on the bulk-copy path. **No regression caused by this group.**

---

## 5. Dtype coverage (15 supported + 2 helper)

### 5.1 Casting & Implicit

| File | Boolean | Byte | SByte | Int16 | UInt16 | Int32 | UInt32 | Int64 | UInt64 | Char | Half | Single | Double | Decimal | Complex |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `NdArray.Implicit.Array.cs` (Array→NDArray) | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `NdArray.Implicit.ValueTypes.cs` (scalar→NDArray) | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `NdArrayToMultiDimArray.cs` (`ToMuliDimArray<T>`) | yes (any `T: unmanaged`) |

### 5.2 ArrayConvert.cs — BUG IN INNER SWITCHES

`ArrayConvert.ToInt32(Array)` has the **top-level** dispatch dynamically covering all 15 dtypes — but the **inner per-target** switches (the 30+ functions like `ToInt32(Array)`, `ToInt64(Array)`, etc.) only handle 13 source types:

```csharp
public static Int32[] ToInt32(Array sourceArray) {
    switch (fromTypeCode) {
        case NPTypeCode.Boolean: …
        case NPTypeCode.Byte: …
        case NPTypeCode.Int16: … // skipped: SByte
        // … standard 12 cases
        case NPTypeCode.String: …
        default: throw new ArgumentOutOfRangeException();  // <-- SByte, Half, Complex hit this
    }
}
```

Confirmed by reproduction:
```text
ArrayConvert.ToInt32(new sbyte[]{-1,0,1})    -> ArgumentOutOfRangeException
ArrayConvert.ToInt32(new Half[]{(Half)1.5})   -> ArgumentOutOfRangeException
ArrayConvert.ToInt32(new Complex[]{1+2i})     -> ArgumentOutOfRangeException
ArrayConvert.To(new sbyte[]{-1}, typeof(int)) -> ArgumentOutOfRangeException
```

Top-level `ArrayConvert.To(Array, Type)` correctly dispatches to `ToSByte` / `ToHalf` / `ToComplex` per target type, but the source-dispatch in those target methods uses the old 13-dtype switch. **Every** of the per-target functions in `ArrayConvert.cs` (`ToBoolean`, `ToByte`, `ToInt16`, `ToUInt16`, `ToInt32`, `ToUInt32`, `ToInt64`, `ToUInt64`, `ToChar`, `ToDouble`, `ToSingle`, `ToDecimal`, `ToSByte`, `ToHalf`, `ToComplex`) needs cases added for SByte, Half, Complex source arrays. This is ~45 missing cases.

### 5.3 Converts.cs and Converts.Native.cs

- `Converts.cs` `ChangeType<TIn,TOut>` 12×12 generic case ladder — does NOT include SByte/Half/Complex source. They fall through to `ChangeType<TOut>((object)value)` which DOES handle them via the boxed object switch. So `Converts.ChangeType<sbyte,int>(x)` works correctly via the slower boxed path. Optimisation opportunity: expand to 15×15.
- `Converts.Native.cs` Inner overloads `ToInt32(SByte)`, `ToInt32(Half)`, `ToInt32(Complex)` etc. all exist as direct unboxed methods. The object dispatcher (`ToInt32(object)`) handles all 15 plus `DateTime`, `TimeSpan`, `DateTime64`. **Complete.**
- `Converts.Char8.cs`, `Converts.DateTime64.cs` provide explicit conversions to/from the two helper types. Complete.

### 5.4 RandomSampling

- `randint` only supports Boolean/Byte/SByte/Int16/UInt16/Int32/UInt32/Int64/UInt64. It silently accepts `Half`/`Single`/`Double`/`Decimal`/`Complex` dtype but the **`ValidateRandintBounds`** uses a `_` default case that allows any high — so randint with float dtype produces a float-typed array of integers in [low, high). NumPy rejects: `TypeError: Unsupported dtype dtype('float64') for randint`.

---

## 6. API parity

| Function | NumPy signature | NumSharp | Status |
|----------|-----------------|----------|--------|
| `np.random.randint(low, high=None, size=None, dtype=int)` | `randint(long low, long high=-1, Shape size=default, Type dtype=null)` | **Two bugs** (see below) |
| `np.random.dirichlet(alpha, size=None)` | `dirichlet(double[] alpha, Shape? size=null)` + 4 overloads | Good |
| `np.random.multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)` | matches | Good — `check_valid` validated and throws for invalid string |
| `np.datetime64('2023-01-01')` literal forms | DateTime64 has minimal `Parse` (delegates to `DateTime.Parse`) | **Missing units** (Y/M/W/D/h/m/s/ms/us/ns/ps/fs/as) |

### 6.1 randint bug 1: `high=-1` sentinel

Sentinel-overloading: `high=-1` means "high omitted". This collides with the legal NumPy call `randint(low, high)` where `high` is negative.

```csharp
public NDArray randint(long low, long high = -1, …)
{
    if (high == -1) { high = low; low = 0; }  // <-- swallows legal high=-1
    …
}
```

**Reproduction:**
```text
randint(-10, -1, 3)  -> ValueError: low >= high
                       (swapped to low=0, high=-10, then low >= high)
```

**NumPy allows:**
```python
>>> np.random.randint(-10, -1, 3)
array([-4, -7, -3])
```

**Fix idea:** Use `long? high = null` instead of `-1` sentinel (or use a separate overload `randint(long high)` for the single-arg form).

### 6.2 randint bug 2: silently accepts non-integer dtype

```csharp
typecode switch {
    NPTypeCode.Byte   => (byte.MinValue, byte.MaxValue, byte.MaxValue + 1L),
    …
    _ => (long.MinValue, long.MaxValue, long.MaxValue)   // <-- any float/decimal/etc. lands here
}
```

```text
np.random.randint(0, 100, 5, typeof(Half))  ->  SUCCESS (NumPy would TypeError)
```

`FillRandintIntDispatch<T>` constrains `T : unmanaged, INumberBase<T>` and uses `T.CreateTruncating(int)` — so a Half-typed array is correctly populated with `Half(int)` values. The arithmetic is well-defined but the API is wrong. Should reject non-integer dtype to match NumPy.

### 6.3 DateTime64 — major API gap

NumPy:
```python
>>> np.datetime64('2023-01-01', 'D')
np.datetime64('2023-01-01')
>>> np.datetime64('2023-01-01', 'ns')
np.datetime64('2023-01-01T00:00:00.000000000')
>>> np.array(['2023-01-01'], dtype='datetime64[D]')
```

NumSharp's `DateTime64` only stores ticks (100ns), with no unit metadata. The file header explicitly notes "DateTime64 is a CONVERSION HELPER TYPE, not a NumSharp NPTypeCode dtype." That is fine for the existing scope, **but**:

- `np.datetime64('2023-01-01')` is not callable in NumSharp.
- There is no parser for ISO-8601 forms `'2023-01-01'`, `'2023-01'`, `'2023'` outside of what `DateTime.Parse` accepts.
- Units (Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as) — none of these are modeled; ticks (100 ns) is the only unit.
- `np.timedelta64` has no analogue (could use `TimeSpan` but no dtype registration).
- `np.array(strings, dtype='datetime64[D]')` is not supported.

**Verdict:** As a conversion-helper struct, DateTime64 is well-designed and complete. As "NumPy datetime64 dtype support", it's a stub.

### 6.4 Char8 — solid Python bytes / `dtype('S1')` model

- Storage: 1 byte (`StructLayout(Size = 1)`).
- Implicit widening to `byte`, `int`, `uint`, `char` (via Latin-1).
- ASCII-strict `IsDigit`/`IsLetter`/`IsAlpha` matching Python `bytes.is*` (so byte `0xE9` ≡ `é` is NOT a letter in Latin-1 mode).
- Alternate `IsLetterLatin1`/`IsDigitLatin1` etc. for `char.cs`-heritage callers.
- String interop: `FromStringLatin1`, `FromStringAscii`, `FromStringUtf8`, `ToStringLatin1`, `ToStringAscii`, `ToStringUtf8`, `FromBytes`, `ToBytes`.
- `IConvertible.GetTypeCode() = TypeCode.Byte` (sensible) — but **NOT registered as NPTypeCode**, so `new NDArray(new Char8[]{…})` throws `NotSupportedException`.
- 8-bit arithmetic + rotate/popcount/leading-zero helpers — well done.

**Verdict:** As a single-byte character type, Char8 is excellent and matches Python's `bytes`/numpy's `S1` *semantics* very well — but NOT as an actual dtype. It is helper-only.

---

## 7. Wasted copies

### 7.1 `dirichlet(NDArray, …)`

```csharp
var alphaBlock = new UnmanagedMemoryBlock<double>(k);
var alphaSlice = new ArraySlice<double>(alphaBlock);
var alphaStorage = new UnmanagedStorage(alphaSlice, new Shape(k));
NpyIter.Copy(alphaStorage, alpha.Storage);
```

Unnecessary even when the input is already `double` and contiguous. Could short-circuit via `if (alpha.typecode == NPTypeCode.Double && alpha.Shape.IsContiguous) { /* zero-copy alphaSlice = alpha.Data<double>() */ }`. Negligible cost (`k` is typically tiny).

### 7.2 `multivariate_normal(NDArray mean, NDArray cov, …)`

`cov` is copied via `cov.GetDouble(i, j)` inside a nested loop:

```csharp
for (long i = 0; i < cov.shape[0]; i++)
    for (long j = 0; j < cov.shape[1]; j++)
        covSlice[i * n + j] = cov.GetDouble(i, j);
```

`GetDouble(long,long)` is the slow per-element path. For an N×N cov this is N² boxing-ish calls. Should use `NpyIter.Copy(covStorage, cov.Storage)` like `mean`. **Performance opportunity for large N.**

### 7.3 `randint`

```csharp
var nd = new NDArray(dtype, size);
…
NpFunc.Invoke(typecode, FillRandintIntDispatch<int>, nd.Array, randomizer, low, high);
```

Single allocation + in-place fill. Clean. No wasted copy.

### 7.4 ArrayConvert.cs

Each call allocates a new T[]. By design — this is the "convert from Array → T[]" boundary. Necessary copy.

### 7.5 Converts<T>.cs

The static field initialisers call `Converts.FindConverter<T, byte>()` etc. — these run ONCE per `T` per AOT/JIT cold start. The cached `Func<T, byte>` is then reused. **No per-call allocation.** Excellent.

---

## 8. Conversion paths — IL gen vs per-type switch

**No IL emission** is used in any of the Converts.* files. The strategy is:

1. **Hot path:** Explicit overloads per `(TIn, TOut)` pair in `Converts.Native.cs`. ~140 explicit methods. Direct call → JIT inlines.
2. **Object boxing path:** `Converts.To{Type}(object)` with `value switch { … }` pattern matching. Used when source type is unknown at compile time.
3. **Generic path:** `Converts.ChangeType<TIn, TOut>(TIn)` does a 12×12 outer-switch on `InfoOf<TOut>.NPTypeCode` × inner-switch on `InfoOf<TIn>.NPTypeCode`, using `Unsafe.As<T,_>` to avoid boxing. Falls through to `ChangeType<TOut>((object)value)` for the missing 3 types (SByte/Half/Complex). This is the closest thing to a "JIT-friendly generic dispatch" pattern.
4. **Cached delegate path:** `Converts<T>.To*` and `Converts.FindConverter<TIn,TOut>()` cache one `Func<TIn,TOut>` per type pair. Used in fields like `private static readonly Func<T, int> _toInt32 = …`. ~10× slower than direct call due to delegate indirection, but avoids re-running the switch ladder.

This is sensible for an *interpretive* dispatch with no IL emission. The ILKernelGenerator does the per-element fast path; Converts is for scalar / boxed / one-off conversions.

**One missing pattern:** No `static readonly nint*[12]*[12]` function-pointer table indexed by `(in_tc, out_tc)` — that would replace the 12×12 switch with two array indexes. Could be ~5× faster than the current generic switch for hot scalar conversions. Probably not worth implementing.

---

## 9. NpFunc — critical findings

### 9.1 CRITICAL — instance method target ignored

`Resolve<TDelegate>(method, tc)` keys the cache on `method.Method.MethodHandle.Value` ALONE. `method.Target` is read once when `CreateDelegate(typeof(TDelegate), method.Target, closed)` runs, then never refreshed.

**Reproduction:**
```csharp
class State { public int LastResult; public void Op<T>(int x) where T : unmanaged { LastResult = x * 2; } }
var s1 = new State();
var s2 = new State();
NpFunc.Invoke(NPTypeCode.Int32, s1.Op<int>, 10);   // s1.LastResult = 20  ✓
NpFunc.Invoke(NPTypeCode.Int32, s2.Op<int>, 100);  // s1.LastResult = 200  ✗ (s2 ignored)
                                                    // s2.LastResult = 0    ✗
```

In current production code (24 call sites), every dispatched method is `static`, so `method.Target == null` and the bug is silent. But the API is *public* and accepts `Action<...>`/`Func<...>` — instance methods compile fine and produce silent wrong results.

**Fix idea:** Either:
- Document the contract: "Only static generic methods may be passed to `NpFunc.Invoke`. Instance methods will silently bind to the first target seen."
- Or include `RuntimeHelpers.GetHashCode(method.Target)` in the cache key (slower, but correct).
- Or use `MethodHandle.Value` + `method.Target?.GetType().TypeHandle.Value` (still wrong for two instances of same type).
- Best: detect non-null `Target` and short-circuit to slow path (no cache).

### 9.2 GOOD — cache structure

- L1 `ConcurrentDictionary<nint, Delegate[16]>` — table size computed dynamically from max `NPTypeCode` ordinal (`128` for Complex, so table is 129 entries). This is sparse — wastes ~111 slots. Could be `Dictionary<NPTypeCode, Delegate>` but the array indexer is faster.
- L2-L5 right-sized tuple keys avoid padding (claim of 33 % vs fixed 6-nint, untested here).
- Hot path: `dict.TryGetValue(nint) + array[(int)tc]` — both `O(1)`, ~10–15 ns combined.

### 9.3 GOOD — `SmartMatchTypes`

When method has more generic params than passed type codes, it tries to match by identity in the *dummy* instantiation. e.g.:
```csharp
NpFunc.Invoke(tcA, tcB, Cast<int, int, float>, …) → resolves to Cast<tcA, tcA, tcB>
```
Elegant pattern, well documented. One edge case: throws `ArgumentException` if distinct generic types exceed passed type codes.

### 9.4 Performance summary

Dispatch overhead: **~32 ns per call**. ~20× slower than manual switch but consistent and predictable. For 1 array operation per million elements, this is **0.00003 µs/elem** — invisible. **Not a perf concern for current call sites.**

---

## 10. Other notable findings

### 10.1 InfoOf<T> for Char8/DateTime64 ✓
- `InfoOf<Char8>.Size = 1` (correct via `Unsafe.SizeOf<Char8>()`)
- `InfoOf<DateTime64>.Size = 8` (correct)
- `Marshal.SizeOf<T>` would throw for `DateTime` — handled by routing through `Unsafe.SizeOf<T>` in the default switch arm.

### 10.2 `DateTime64` operator semantics ✓

- `operator ==/!=/</>/<=/>=` follow NumPy (NaT vs anything → `false`).
- `Equals(DateTime64)` returns `true` for `NaT.Equals(NaT)` so hashcode contract holds and NaT can be used as a `Dictionary` key. Mirrors `double.NaN` exactly.

### 10.3 `Implicit operator NDArray(string)` is non-standard

The implicit string→NDArray operator parses `"[1,2,3]"` or `"[1,2;3,4]"` via regex. NumPy doesn't do this. It's harmless but should probably be marked obsolete or replaced with explicit `np.array(str)` to match NumPy's "no implicit string conversion" rule.

### 10.4 `NdArrayToMultiDimArray.cs` decimal fallback

`decimal` is not blittable, so `Buffer.BlockCopy` doesn't work — falls back to `Array.SetValue` with coordinate iteration. ~20× slower than primitive path, but correct. Good split.

### 10.5 `Arrays.cs::Slice<T>` dead code path

```csharp
if (len > 700_000)
    for (int i = 0; i < len; i++) res[i] = source[i + start];
else
    for (int i = 0; i < len; i++)
        res[i] = source[i + start];
```

Both branches are identical. This is a leftover scaffold for what was probably going to be a `Parallel.For` for large arrays. Harmless but should be cleaned up.

### 10.6 `Arrays.cs::Insert<T>(ref T[], int, T)` allocates twice
```csharp
Array.Resize(ref source, source.Length + 1);     // alloc 1
Array.Copy(source, index, source, index + 1, …); // in-place
```
Calling `Insert(ref…)` from `AppendAt` then doing `var ret = (T[])source.Clone();` *before* the insert is a bug — the original is cloned but the insert applies to the source, not the clone. Reading `AppendAt`:

```csharp
public static T[] AppendAt<T>(T[] source, int index, T value)
{
    var ret = (T[])source.Clone();
    Insert(ref source, index, value);   // <-- mutates `source`, not `ret`
    return ret;                          // <-- returns the UN-MODIFIED clone
}
```

This means `AppendAt` returns a CLONE of `source` without ever applying the `value`. **Bug**. Not surfaced because `AppendAt` is apparently unused.

### 10.7 `Char8.PyBytes.cs` — Python bytes parity

531 lines of Python-bytes-style operations (split, join, strip, find, count, replace, startswith/endswith, lower/upper, etc.) on `ReadOnlySpan<Char8>`. Quality looks high (didn't deeply verify every edge case but spot-checked `strip` and `split` against Python). This is solid Python-bytes parity for NDArray code that wants byte-string operations.

### 10.8 `Converts.cs::ChangeType<TIn,TOut>` — missing 3 dtypes

The 12×12 generic switch handles Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Double, Single, Decimal. **Missing: SByte, Half, Complex.** Falls through to `ChangeType<TOut>((object)value)` which DOES handle them, so it's correct but slower (1 boxing + 1 switch). Could be expanded to 15×15 = 225 cases.

---

## 11. Reproductions

### 11.1 Float→int boundaries
```text
ToInt32(NaN) = -2147483648                          NumPy ✓
ToInt32(Inf) = -2147483648                          NumPy ✓
ToInt32(2147483647.4) = 2147483647                  NumPy ✓
ToInt32(1e30) = -2147483648                         NumPy ✓
ToSByte(500.5) = -12                                NumPy ✓
ToByte(-1.5) = 255                                  NumPy ✓
ToUInt32(NaN) = 0                                   NumPy ✓
ToUInt64(NaN) = 9223372036854775808                 NumPy ✓ (2^63)
ToUInt64(-1.5) = 18446744073709551615               NumPy ✓
[NaN, Inf, 1e30, 2147483647.4].astype(int32) = [int.MinValue, int.MinValue, int.MinValue, 2147483647]  NumPy ✓
```

### 11.2 randint bugs
```text
np.random.randint(-10, -1, 3)  ->  ValueError: low >= high  (NumSharp BUG)
np.random.randint(0, 100, 5, typeof(Half))  ->  SUCCESS  (NumSharp BUG; NumPy TypeErrors)
```

### 11.3 ArrayConvert bugs
```text
ArrayConvert.ToInt32(new sbyte[]{-1,0,1})       -> ArgumentOutOfRangeException
ArrayConvert.ToInt32(new Half[]{(Half)1.5})      -> ArgumentOutOfRangeException
ArrayConvert.ToInt32(new Complex[]{1+2i})        -> ArgumentOutOfRangeException
ArrayConvert.To(new sbyte[]{-1}, typeof(int))    -> ArgumentOutOfRangeException
```

### 11.4 NpFunc target-binding
```text
NpFunc.Invoke(NPTypeCode.Int32, s1.Op<int>, 10);   // s1=20, s2=0
NpFunc.Invoke(NPTypeCode.Int32, s2.Op<int>, 100);  // s1=200, s2=0   <-- silent wrong
```

### 11.5 Char8 / DateTime64 — NOT registered as NDArray dtype
```text
new NDArray(new Char8[]{...})       -> NotSupportedException
new NDArray(new DateTime64[]{...})  -> NotSupportedException
```

---

## 12. Summary table

| # | Severity | Area | Issue | Affects |
|---|----------|------|-------|---------|
| 1 | **CRITICAL** | NpFunc | Cache keyed by `MethodHandle.Value` alone — instance methods silently bind to first target | Latent (no production code uses instance methods) but landmine for future callers |
| 2 | High | ArrayConvert | SByte / Half / Complex source arrays throw `ArgumentOutOfRangeException` in all inner `ToXxx(Array)` switches | Any code calling `ArrayConvert.To*` with a new-dtype source array |
| 3 | High | randint | `high = -1` sentinel collides with legal NumPy `randint(low, -1)` call | API parity, user-facing |
| 4 | Medium | randint | Silently accepts non-integer dtype (`Half`, `Single`, `Double`, `Decimal`, `Complex`) | API parity (NumPy TypeErrors); produces non-NumPy semantics |
| 5 | Medium | DateTime64 | No NumPy unit support (Y, M, W, D, h, m, s, ms, us, ns, ps, fs, as); only ticks (100 ns) | NumPy datetime64 dtype parity — listed as out-of-scope, but worth highlighting |
| 6 | Medium | Char8 / DateTime64 | Not registered as `NPTypeCode` — cannot create `NDArray<Char8>` or `NDArray<DateTime64>` | Conceptual gap; helpers only |
| 7 | Low | multivariate_normal | 5×5+ matrices diverge from NumPy (sign convention) — already documented in source | Listed in source as known divergence |
| 8 | Low | multivariate_normal(NDArray cov) | Uses per-element `cov.GetDouble(i,j)` instead of `NpyIter.Copy` | Performance for large N |
| 9 | Low | Converts.cs `ChangeType<TIn,TOut>` | Missing SByte/Half/Complex in 12×12 generic switch — falls through to slow boxing path | Performance for scalar conversions involving these types |
| 10 | Low | Arrays.cs `AppendAt` | Clone-then-mutate-original bug returns un-modified copy | Currently unused in codebase |
| 11 | Low | Arrays.cs `Slice<T>` | Dead code — `if (len > 700_000)` branches are identical | Cleanup |
| 12 | Low | NDArray.Implicit.Array.cs | `implicit operator NDArray(string)` parses `"[1,2,3]"` — non-NumPy behaviour | Style; not a bug |
| 13 | Info | Performance | NpFunc dispatch ~32 ns/call (~20× slower than switch); acceptable for kernel-level use | Documented |
| 14 | Info | Performance | `Converts<T>.To*` (cached delegate) ~62 ns/call vs 7 ns inlined direct call | Documented |
| 15 | Info | Coverage | `Converts.Native.cs` — NumPy NEP50 wrap/truncate semantics verified bit-identical across 15+ edge cases | Excellent NumPy parity |
