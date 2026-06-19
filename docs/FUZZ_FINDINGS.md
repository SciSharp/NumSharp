# NumPy Differential-Fuzzer Findings

Every NumSharp-vs-NumPy-2.4.2 divergence surfaced by the differential fuzzer
(`test/NumSharp.UnitTest/Fuzz/`, Plan A). Each is **bit-exact verified** against NumPy as the
oracle. Dispositions:

- **FIXED** — corrected in-tree.
- **BUG** — confirmed correctness/parity defect, documented in `MisalignedRegistry` so the gate stays
  green, tracked for fix. Remove the classifier branch (or `[OpenBugs]` tag) when fixed.
- **INTENDED** — accepted NumSharp-vs-NumPy difference (maintainer decision); stays `[Misaligned]`.
- **SCOPING** — divergence only reachable by feeding NumSharp a representation its own API never
  produces; the fuzzer is scoped around it.

Char and Decimal are excluded from the differential corpus (no NumPy analog).

## Summary

| # | Area | Issue | Disposition | Task |
|---|------|-------|-------------|------|
| 1 | cast | `complex → bool` drops the imaginary part | **FIXED** | — |
| 2 | floor_divide/mod | integer `÷0` / `mod 0` throws or returns garbage (NumPy → 0) | **FIXED** | F1 |
| 3 | floor_divide | float `//0` → NaN (NumPy → ±inf) | **FIXED** | F1 |
| 4 | mod | `mod(float32, float64)` computed in float32 then widened | **FIXED** | F1 |
| 5 | power | complex power ~1 ULP + gross inf/NaN edge | BUG | #7→F5 |
| 6 | comparison | `<=` / `>=` return True for a NaN operand (NumPy → False) | **FIXED** | F2 |
| 7 | unary | NEP50 unary float promotion: int → float64 (NumPy: width-based) | **FIXED (transcendental, F3a)** / square·floor·ceil·trunc·reciprocal pending F3b | F3 |
| 8 | unary | `negative(uint*)` throws (NumPy wraps modulo) | **FIXED** | F4 |
| 9 | unary | `reciprocal(int)` wrong; non-contig throws (**float non-contig FIXED**; int value/non-contig pending F3b) | BUG | F3b |
| 10 | unary | complex `square` cancellation; complex `sin/cos/tan/log` inf/NaN edge | BUG | #9 |
| 11 | reduction | `min/max` skip NaN (NumPy propagates) | **FIXED** (flat + axis, float32/double) | F2-red |
| 12 | reduction | complex axis reduction throws (**2-D+ FIXED**; 1-D vector-shape pending) | BUG | #10 |
| 13 | reduction | bool `min/max` along an axis returns True where NumPy → False | **FIXED** | F8 |
| 14 | reduction | `sum/mean/std/var` accumulation order (vs NumPy pairwise/two-pass) | BUG | #10 |
| 15 | reduction | result dtype (NEP50 accumulator width / complex→real) | BUG | #10 |
| 16 | where | complex `np.where` throws ("Zero-push unsupported for Complex") | **FIXED** | F4 |
| 17 | binary | bool arithmetic: `True + True → 2` (NumPy → bool True) | **FIXED** | F6 |
| 18 | binary | size-1 result collapses to 0-D (NumPy keeps `[1]`) | **FIXED** | F7 |
| 19 | binary | complex `multiply`/`divide` cancellation + ~1 ULP | BUG | #12 |
| 20 | promotion | 0-D array operand promoted weakly (NEP50: full participant) | INTENDED | — |
| 21 | divide | complex division ~1 ULP (`npy_cdivide` vs `System.Numerics.Complex`) | INTENDED | — |
| 22 | views | ops ignore raw NumPy `offset!=0` / junk size-1 strides | SCOPING | #11 |
| 23 | reduction | NpyIter reduce path ignored `Shape.offset` → sliced/negative-stride views read wrong cells / OOB garbage | **FIXED** | F8 |
| 24 | reduction | non-contiguous flat `prod(char)` overflows (uint16 accumulator) while contiguous promotes to uint64 | BUG | — |
| 25 | reduction | broadcast-reduce FOLD (`sum`/`prod`/`min`/`max`/`mean` over stride-0 views) — **VERIFIED CORRECT**, ~6,600 fuzz cases (15 dtypes × layouts × axes × keepdims) vs NumPy & materialized copy | **OK** | — |
| 26 | nan-reduction | `nansum`/`nanmax`/`nanmin`/`nanprod` AXIS reduction on NON-contiguous **half** views (transpose/broadcast/reversed) reads wrong cells (float32/float64 correct on identical layout) | BUG | — |
| 27 | nan-reduction | `nanmax`/`nanmin` on **complex** do NOT skip NaN — gated to float only, fall through to `ReduceAMax`/`AMin` which propagate NaN (even contiguous) | BUG | — |
| 28 | nan-reduction | `nansum` **complex** AXIS reduction for ndim≥3 reads UNINITIALIZED memory (incrementor writes only a 1-D subset of the multi-D `new NDArray(...,false)` output; even contiguous) | BUG | — |
| 29 | arg-reduction | `argmax`/`argmin` on **decimal** return a boundary index not the extreme (even contiguous); on **char** throw `NotSupportedException` (NumPy uint16 supports it) | BUG | — |

---

## Broadcast-reduce adversarial sweep (2026-06-20)

Stress-test of the broadcast-reduce fold (commit `878240c3`) and every adjacent reduction family
over broadcast/strided views. Harness: `benchmark/poc/bcast_ref.py`+`bcast_check.cs` (flat fold,
1,009 cases vs NumPy), `bcast_ax_ref.py`+`bcast_ax_check.cs` (axis/arg/var/nan/all/any/ptp/median,
7,614 cases vs NumPy), `bcast_consistency.cs` (broadcast-vs-materialized-copy, 4,727 cases, self-contained).
Reproductions: `test/NumSharp.UnitTest/OpenBugs.BroadcastReduce.cs`.

**Fold verdict (#25): bug-free.** `sum`/`prod`/`min`/`max`/`mean` over every broadcast layout
(prepend / inner / interleaved / scalar-collapse / 3-D / 5-D / non-contiguous remainder), every axis,
keepdims on/off, all 15 dtypes — **0 divergences** vs NumPy AND vs the materialized copy. `var`/`std`
(compose on the same infra), `all`/`any`/`count_nonzero`, `ptp`/`median`, and non-decimal
`argmax`/`argmin` over broadcast are likewise consistent.

The 48 broadcast-vs-copy divergences + the NumPy-parity failures all fall into the PRE-EXISTING,
fold-disjoint families #26–#29 (NaN-aware engine `Default.Reduction.Nan.cs` + the arg dtype switch).
None are introduced by the fold (which only edits `ExecuteElementReduction`). Edge left unfiled:
2 cases of float/double `nanmin` over an all-NaN broadcast slice returning ±inf vs NaN (layout-dependent).

---

## FIXED

### 1. `complex → bool` dropped the imaginary part

`NpyIterCasting.ConvertValue` read `c.Real` for every complex→real/bool cast, so a complex number
with zero real part but non-zero imaginary part converted to `False`. NumPy: `bool(z) == (z != 0)` —
truthy if **either** part is non-zero. Latent in `astype`, `np.where`, `np.copyto`, `np.concatenate`.

```python
np.array([complex(0, 1), complex(-0.0, -2147483649.0)]).astype(bool)   # [ True,  True]
```
```csharp
np.array(new[]{ new Complex(0,1), new Complex(-0.0,-2147483649.0) }).astype(NPTypeCode.Boolean)
// before fix: [False, False]   after fix: [True, True]
```

**Fix:** special-case complex→bool to `c.Real != 0 || c.Imaginary != 0` (matches the correct
`Converts.ToBoolean(Complex)`). Commit `908ee7f9`. Found by T1 cast matrix (15 cells).

---

## BUGS (confirmed parity defects, tracked for fix)

### 2. Integer `÷0` / `mod 0` — throws or returns garbage — **FIXED (F1)**

**Fixed** by the `NpyDivision` helper (ports NumPy `floor_div_@TYPE@` / integer remainder):
integer ÷0 and mod-0 now return **0**, signed floor rounds toward −∞, and `MIN // -1` wraps to
`MIN`. Routed through both IL emission paths (`EmitFloorDivideOperation` / `EmitModOperation`,
same-type and mixed). The `binary_divmod_power` corpus is now bit-exact for floor_divide/mod and
runs CI-gated (`[FuzzMatrix]`).

NumPy integer division/modulo by zero returns **0** (with a RuntimeWarning). NumSharp threw or
returned a sentinel.

```python
np.array([1,-7,5], np.int32) // np.array([0,0,0], np.int32)   # [0, 0, 0]
np.array([1,-7,5], np.int32) %  np.array([0,0,0], np.int32)   # [0, 0, 0]
np.array([5,9], np.uint8)    // np.array([0,0], np.uint8)     # [0, 0]
```
```csharp
np.floor_divide(i32, zeros)  // [2147483647, -2147483648, 2147483647]  (saturated sentinel)
np.mod(i32, zeros)           // [1, -7, 5]                              (returns the dividend)
np.floor_divide(u8, zeros)   // THROWS DivideByZeroException
```
Found by T2 `Binary_DivModPower` (`[OpenBugs]`). Task **#7**.

### 3. Float `//0` → NaN instead of ±inf — **FIXED (F1)**

**Fixed** — the float helper ports CPython's `npy_divmod` (fmod → sign-fixup → snap-to-nearest):
`b == 0` returns `a / b` (±inf, or nan for 0/0), never a forced NaN. Edge cases verified bit-exact:
`0.7 // 0.1 == 6.0`, `-2.0 // inf == -1.0`, `inf // 2.0 == nan`, `1e308 // 1e-300 == inf`.


```python
np.array([1.0,-1.0,0.0]) // np.array([0.0,0.0,0.0])   # [ inf, -inf,  nan]
```
```csharp
np.floor_divide(f, zeros)   // [NaN, NaN, NaN]   (loses the ±inf sign result)
```
Float `%0` is correctly `nan` on both sides; mod sign convention is correctly floored
(`mod(-7,3) == 2`). Task **#7**.

### 4. Mixed-precision `mod` loses precision — **FIXED (F1)**

**Fixed** — once `mod`/`floor_divide` route through the `NpyDivision` helpers, the promoted
result dtype (float64) drives the computation: `mod(f32, f64)` now yields `float64` bit-exact with
NumPy (e.g. `[0.10000000000000009, 0.600000047683716, 0.3000003814697272]`). `add/sub/mul/div`
already promoted correctly.

### 5. Complex `power` — ~1 ULP + gross edge

`complex ** {float,complex,int}` differs from NumPy by ~1 ULP in places, plus gross edge
divergences (NumPy NaN where NumSharp returns 0) for inf/zero bases. Task **#7**.

### 6. `<=` / `>=` return True for NaN — **FIXED (F2)**

**Fixed** — the scalar comparison emitted `a <= b` as `!(a > b)` using the *ordered* `Cgt`
(`Clt` for `>=`), which yields false for a NaN operand and negates to **true**. Switching to the
*unordered* `Cgt_Un` / `Clt_Un` for float operands makes a NaN compare yield true, so the negation
is false — matching IEEE/NumPy. Verified bit-exact across scalar, SIMD (NaN mid-vector), strided,
and float32 paths; the comparison matrix runs CI-gated with no excused divergence.

IEEE/NumPy: every ordered comparison with NaN is False (only `!=` is True). `<`, `>`, `==`, `!=`
handled NaN correctly; `<=` and `>=` returned **True**.

```python
np.array([np.nan]) <= np.array([1.0])    # [False]
np.array([np.nan]) >= np.array([1.0])    # [False]
```
```csharp
(np.array(new[]{double.NaN}) <= np.array(new[]{1.0}))   // [True]
```
Found by T3 `Comparison` (122 cases, all the NaN element). Likely `a <= b` implemented as
`!(a > b)` / `a < b || a == b`. Task **#8**.

### 7. NEP50 unary float promotion — **FIXED (transcendental, F3a)** / rest pending F3b

The **transcendental** ufuncs (`sqrt/cbrt/exp/exp2/expm1/log/log10/log1p/log2/sin/cos/tan/sinh/cosh/
tanh/arcsin/arccos/arctan/deg2rad/rad2deg`) now use NumPy's width-based float promotion via the new
`ResolveUnaryFloatReturnType`: bool/int8/uint8 → float16, int16/uint16 → float32, int32+ → float64,
float/complex preserved. 364 of the 494 unary dtype divergences cleared bit-exact; the transcendental
branch of `MisalignedRegistry` is removed so a regression now fails the gate. Half/Single value
diffs vs NumPy's float16/float32 libm remain within 2 ULP (excused as algorithm difference).

**Still pending (F3b):** the dtype-**preserving** ufuncs `square/floor/ceil/trunc/round/reciprocal`
still widen integer input to float64 instead of preserving the integer dtype (needs integer
identity / `x*x` / int-reciprocal kernels — 130 dtype divergences, scoped in the classifier).

| op(input) | NumPy | NumSharp (now) |
|-----------|-------|----------------|
| `sqrt(bool)` / `sqrt(uint8)` | float16 | **float16** ✓ |
| `sqrt(int16)` | float32 | **float32** ✓ |
| `sqrt(int32)` | float64 | float64 ✓ |
| `square(uint8)` | uint8 | float64 (F3b) |
| `floor(int32)` | int32 | float64 (F3b) |

Found by T4 `Unary` (494 → 130 dtype divergences). Task **#9** / F3.

### 8. `negative` on unsigned integers throws — **FIXED (F4)**

**Fixed** — `np.negative(nd)` called the legacy hand-written `NDArray.negative()` which threw
`NotSupportedException` for every unsigned dtype and required a flat `Address` (so non-contiguous
also failed). It now routes through `nd.TensorEngine.Negate(nd)` — the same engine path the unary
`-` operator and `nd.negate()` already used — whose IL kernel negates unsigned via two's-complement
wrap (`-1u -> 255`) and handles strided/non-contiguous operands through NpyIter. Verified bit-exact
on contiguous and strided uint8/uint16/uint32; the unary matrix no longer excuses it.

```python
np.negative(np.array([1], np.uint8))    # [255]  (wraps modulo)
```

### 9. `reciprocal` — integer result wrong; non-contiguous throws — **partially resolved**

`reciprocal` on a non-contiguous **float** operand (transposed / strided) now works (resolved).
**Still pending (F3b):** `reciprocal(int)` returns `0` (float64) where NumPy returns the integer
reciprocal with the `÷0` sentinel, *and* `reciprocal(int)` on a non-contiguous operand still throws
`InvalidOperationException: Can't return a memory address...` because the int→float reciprocal path
needs a flat Address — both fixed once F3b gives reciprocal an integer-preserving strided kernel.
The classifier now excuses only the integer-input cases (float non-contig is gate-verified).

### 10. Complex unary — `square` cancellation; `sin/cos/tan/log` edge

`square(complex)` suffers catastrophic cancellation in `re² − im²` (NumSharp → exactly 0 where
NumPy retains precision). `sin/cos/tan/log` of complex differ on inf/NaN-involving inputs
(NumPy `(NaN, +inf)` vs NumSharp `(NaN, NaN)`). `System.Numerics.Complex` vs NumPy's `npy_c*`.
Task **#9**.

### 11. Reductions skip NaN (NumPy propagates) — **FIXED (flat + axis min/max)**

`sum/mean/std/var` already propagated NaN (arithmetic: `NaN op x == NaN`). Only `min/max` skipped it,
because the SIMD path used hardware `Vector.Min/Max` (MINPS/MAXPS drop a NaN operand). The **flat**
(`axis=null`) min/max reduction is now **fixed**: `EmitVectorBinaryReductionOp`, the horizontal
tree-reduce (`EmitVectorReductionOp`), and the C# `CombineVectors256/128` all emit the
NaN-propagating form `ConditionalSelect(Equals(a,a) & Equals(b,b), MinMax(a,b), a+b)` for float/
double — verified across every size (3..257) and NaN position, double + float32. The scalar tail
already used `Math.Min/Max` (propagates).

The **axis** (vertical/strided) SIMD min/max kernel now also propagates NaN — verified bit-exact for
float32 and double, `axis=0` and `axis=1` (`np.amin(m, 1)` over a row containing NaN → `nan`).

```python
np.min(np.array([np.nan, -np.inf, 1.0]))      # nan  (NumSharp: nan ✓)
np.amin(np.array([[np.nan,-1,2],[3,4,5]]), 1) # [nan, 3]  (NumSharp: [nan, 3] ✓)
```
Found by T5 `Reduce`. Task **#10** / F2-reductions.

### 12. Complex axis reduction throws — **partially resolved (2-D+ FIXED)**

`sum/mean/prod/std/var/min/max` of a complex **2-D+** array along an axis now works bit-exact
(`std(complex2d, axis=0)` → real float64, verified). **Still pending:** reducing a **1-D** complex
array along its only axis still throws `InvalidOperationException: Can't construct
NDCoordinatesAxisIncrementor with a vector shape`. The classifier excuses only the 1-D Threw case;
the 2-D matrix cases are gate-verified (value diffs fall to the summation-precision branch).

### 13. bool `min`/`max` along an axis is wrong — **FIXED (F8)**

`max`/`min` of a bool array reduced along an axis returned `True` at every position where NumPy
returns `False` (all-False group). Root cause: the axis scalar reducer
(`CreateAxisReductionKernelScalar<bool,bool>`) seeded its accumulator from
`GetIdentityValueTyped<bool>`, which computed a `double` identity (`Max → double.NegativeInfinity`)
and funneled it through `ConvertFromDouble<bool>` (`value != 0`). `double.NegativeInfinity != 0` is
**True**, so the Max accumulator started `True` and `Math.Max(1.0, ≤1.0)` never dropped below 1 →
every group reduced to True. `amin` was coincidentally correct (its `PositiveInfinity → True` seed is
the correct Min identity); integer/byte/char were correct (their `(int)NegInf`/`(byte)NegInf` casts
are valid Max seeds — only bool's `!= 0` test corrupted it). Fixed by an explicit bool identity block
in `GetIdentityValueTyped` (`Max → false`, `Min → true`). The double-bridge combine is correct once
the seed is right. Verified bit-exact vs NumPy across axis 0/1 incl. all-False groups.

```python
np.max(np.array([[True,False,True],[False,False,True]]), axis=0)  # [T,F,T]  (NumSharp: [T,F,T] ✓)
```
Found by T5 `Reduce` / re-confirmed by the min/max parity sweep (`benchmark/poc/minmax_*`).

### 23. NpyIter reduce path ignored `Shape.offset` — **FIXED (F8)**

Any reduction routed through the NpyIter REDUCE path (`ExecuteAxisReductionNpyIter` →
`NpyIterRef.NewReduce`, which serves double/single `sum`+`mean`, and complex/decimal `sum`/`prod`/
`min`/`max`/`mean`, half `sum`+`mean`) read from the **buffer base** instead of the view's logical
origin for any input whose offset lives in `Shape.offset`: sliced views (`a[1:3,1:3]`) and
negative-stride views (`a[::-1]`, `a[:,::-1]`). Root cause: in `NpyIter.cs` the **op_axes** operand
base-pointer branch set `basePtr = (byte*)arr.Address` without adding `arrShape.offset * elemBytes`,
while the standard-broadcast branch did add it. So a strided reduce read the wrong cells; and once
`FlipNegativeStrides` moved the (already-wrong) pointer by the flip offset, it read/wrote **out of
bounds** → garbage / NaN / denormals. Contiguous, transpose, F-order and positive-strided
(offset-in-`Address`) views all have `Shape.offset == 0`, so they were unaffected — which is why the
bug hid. Fixed by adding the offset in the op_axes branch (matching the standard branch), using
`NPTypeCode.SizeOf()` (1 byte for bool, never `arr.dtypesize`).

```python
a = np.arange(12).reshape(3,4)
np.sum(a[::-1], axis=0)      # [12,15,18,21]   (was [0,1,2,3])
np.sum(a[1:3,1:3], axis=0)   # [14,16]          (was [4,6])
```
Pinned by `ReduceOffsetStrideParityTests` + the sum/prod/mean parity sweep
(`benchmark/poc/reduce_*`, 1616/1620 — the 4 are #24).

### 24. non-contiguous flat `prod(char)` overflows

`np.prod` of a **non-contiguous** char view (`a.T`, `a[::-1]`, …) over the whole array accumulates in
the 16-bit char width and overflows (e.g. `24⁶ → 0`), whereas the **contiguous** flat path promotes
to `uint64` and the **axis** path is fine. `int16` (and all other narrow ints) promote correctly on
the same non-contiguous flat path — only `char` is wrong. char has no NumPy analog (modelled as
`uint16`); pre-existing and unrelated to the reduce-offset fix. Not yet fixed.

### 14. Reduction accumulation order

`sum/mean/std/var` floating results differ from NumPy's pairwise summation / two-pass variance
(rounding from accumulation order). Magnitude is data-dependent (ill-conditioned sums diverge more).
Task **#10**.

### 15. Reduction result dtype

Some reduction result dtypes differ from NumPy (NEP50 accumulator width; complex→real for
`std/var`). Task **#10**.

### 16. Complex `np.where` throws — **FIXED (F4)**

`np.where(cond, x, y)` threw `NotSupportedException: Zero-push unsupported for Complex` whenever the
promoted result was complex — `NpyExpr.EmitPushZeroPublic` had no `Complex` case (the WhereNode pushes
a typed zero for the unselected branch). **Fixed** by adding `Complex` (`Complex.Zero` static field)
and `Half` cases. Both-complex `where` already worked; the throw only hit the *mixed* promotion
(`where(cond, complex, float)`). Now bit-exact: `where([T,F,T], complex, float) == [(1+2j),(8+0j),(5+6j)]`.

### 17. bool arithmetic computes the integer result — **FIXED (F6)**

NumPy's bool dtype has no integer add/multiply ufunc loop: `+` is logical **OR**, `*` is logical
**AND** (`True + True == True`, raw byte 1). NumSharp computed byte arithmetic, so `True + True`
stored **2** in a bool slot. **Fixed** in `ExecuteBinaryOp`: when both operands are bool
(`resultType == Boolean`), `Add` is remapped to `BitwiseOr` and `Multiply` to `BitwiseAnd` before
kernel dispatch, so every SIMD/scalar path writes a normalized 0/1 byte. `-` has no bool loop and
throws on both sides. Verified bit-exact scalar + SIMD (32-wide).

```python
np.array([True]) + np.array([True])    # [ True]  (bool, byte 1)
```

### 18. size-1 result collapses to 0-D — **FIXED (F7)**

`Shape.Broadcast`'s size-1 fast path treated a 1-D `[1]` like a 0-D scalar and broadcast it to the
*other* operand's dimensions — so `[1] + 0-D scalar` adopted the scalar's `[]` shape, dropping a
rank (`[1] → []`). The path was asymmetric: `scalar + [1]` was already correct. **Fixed** by guarding
the size-1 collapse on `rightShape.NDim >= leftShape.NDim` (and the symmetric `leftShape.NDim >=
rightShape.NDim`), so the result keeps `ndim == max(ndims)`. Verified across `[1]+scalar`, `[1]+[2,3]`,
`[3]+[1]`, `[2,1]+[3]`, `[[1]]+scalar`, etc.; full suite green (the broadcast change is core).
Found by the A2 random fuzzer.

### 19. Complex binary `multiply`/`divide` cancellation

Random complex data triggered catastrophic cancellation and >1 ULP differences in `multiply`
(and `divide`) that the fixed pool missed — same `System.Numerics.Complex` vs `npy_c*` root as the
unary/divide complex issues. Found by the A2 random fuzzer. Task **#12**.

---

## INTENDED (accepted divergences — `[Misaligned]`)

### 20. NEP50 weak-scalar promotion

NumSharp treats **any 0-D array operand** as a weak scalar (the array operand's dtype drives the
result). NEP50 makes 0-D arrays full promotion participants; only Python scalar literals are weak.
NumSharp cannot distinguish the two (both are 0-D `NDArray`), and `arr + 5` ergonomics were chosen
over strict parity. Array+array promotion is correct across all layouts.

```python
np.arange(5, dtype=np.int32) + np.array(7, np.int64)   # int64
np.arange(5, dtype=np.uint8) + np.array(3, np.int8)    # int16
```
```csharp
arrI32 + zerodI64    // Int32   (NumPy: Int64)
arrU8  + zerodI8     // Byte    (NumPy: Int16)
```
Maintainer decision: keep as Misaligned.

### 21. Complex division ~1 ULP

`complex / {float,complex,int}` differs from NumPy's `npy_cdivide` by one ULP
(`System.Numerics.Complex` uses different scaling). Complex add/sub/mul are bit-exact (modulo
cancellation, #19). Maintainer decision: keep as Misaligned (bit-exact complex division is
impractical and platform-sensitive).

---

## SCOPING (unreachable via NumSharp's API)

### 22. Ops vs raw NumPy stride/offset representation

When fed a byte-reconstructed view carrying NumPy's **raw** representation — `Shape.offset != 0`, or
a size-1 dimension with NumPy's arbitrary "junk" stride — some ops read the wrong element:

- `np.where(cond, int64[1]@offset1, scalar)` reads offset 0.
- `subtract` on a `[5,1,3,1]` view with transposed size-1 strides yields wrong values.

But NumSharp's own slicing **normalizes** the offset into the storage base (`x["1:2"].Shape.offset
== 0`) and keeps consistent size-1 strides, so these representations never arise through the API —
native `where`/`subtract` on the same logical views are correct. The fuzzer is therefore scoped to
NumSharp-producible layouts. Per the DOD ("ops must handle `Shape.offset`"), the open question is
whether to harden the `where`/binary kernels to honor arbitrary `Shape.offset` + size-1 strides, or
document that NumSharp normalizes offset and never produces these states. Task **#11**.
