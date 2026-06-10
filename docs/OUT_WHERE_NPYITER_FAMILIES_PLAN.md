# `out=` / `where=` for the NpyIter-routed families — design & elaboration

**Date:** 2026-06-10 · **Branch:** `nditer` · **NumPy:** 2.4.2 (every behavioral claim below probed this session, outputs verbatim) · **Baseline:** Wave 2.1 shipped `out=`/`where=` on the 18-function elementwise core (8 binary + 10 unary, commit 5962a5e1); Wave 6.1 shipped `np.evaluate(…, out:)`.

**Scope:** thread the ufunc `out=`/`where=` parameters through the operation families that **already execute through NpyIter Tier-3B** for non-trivial layouts — comparisons, predicates, bitwise, the remaining ~20 unary math ops, and `arctan2`. This is **signature threading + per-family dtype-rule wiring + probe-pinned tests**, not new iterator machinery: the masked ForEach driver, ARRAYMASK plumbing, windowed cast flush, COPY_IF_OVERLAP, and validation helpers all exist and are battle-tested (suite 9596).

**Explicitly OUT of scope** (different problem — route migration, not parameter threading):

| excluded | why |
|---|---|
| `left_shift` / `right_shift` | hand-rolled loops in `Default.Shift.cs`, never touch NpyIter; need route migration first |
| `maximum` / `minimum` | clip-composed, no NpyIter route |
| all reductions (`sum`/`prod`/…, `out=` on reduce) | Wave 5 (reductions through the core) |
| `modf` | multi-output ufunc — needs the Phase-3 multi-output operand contract; see §4.4 |
| `casting=` / `order=` / `dtype=`-as-ufunc-kwarg | separate ufunc-kwargs surface; `out`/`where` first |

---

## 1. WHY

### 1.1 NumPy API parity — the exact affected surface

Every NumPy ufunc accepts `out=` and `where=`. After Wave 2.1, NumSharp honors them on 18 functions; the families below — all already NpyIter-routed — do not. The audit (this session, grepped, not assumed):

**Category A — function exists at np.* level, lacks the parameters (32 names):**

| family | np.* functions | current return | engine route today |
|---|---|---|---|
| comparisons (6) | `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal` (`Logic/np.comparison.cs`, thin operator wrappers) | `NDArray<bool>` | `ExecuteComparisonOp` → Tier-3B (`DefaultEngine.CompareOp.cs:60`) |
| predicates (3) | `isnan`, `isfinite`, `isinf` (`Logic/np.is.cs:55-78`) | `NDArray<bool>` | `ExecuteUnaryOp(a, UnaryOp.IsNan/…, NPTypeCode.Boolean)` (`Default.IsNan.cs:27`, `Default.IsInf.cs:23`, `Default.IsFinite.cs:26`) — **verified: the unary ladder, whose Wave-2.1 out/where branch already exists** |
| unary math (21) | `log2`, `log10`, `log1p`, `exp2`, `expm1`, `cbrt`, `sign`, `floor`, `ceil`, `trunc`, `reciprocal`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `deg2rad`, `rad2deg`, `round_`, `around` | `NDArray` | all → `ExecuteUnaryOp` (one `Default.<Op>.cs` each) |
| invert (1) | `invert` (`Math/np.invert.cs`) | `NDArray` | `ExecuteUnaryOp(BitwiseNot / LogicalNot)` (`Default.Invert.cs:15-26`) |
| arctan2 (1) | `arctan2` (`Math/np.tan.cs:88-101`) | `NDArray` | **own Direct path** — `ExecuteATan2Op` + `MixedTypeKernel` (`Default.ATan2.cs:47-103`), *not* the binary ladder, *not* NpyIter. Audit correction: the out/where route for it goes through `ExecuteBinaryUfuncInto` (§4.3.4) |

**Category B — function missing entirely at np.* level (3 names):**

`np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor` **do not exist** (verified by grep: only `np.left_shift`/`np.right_shift`/`np.invert` exist in `Math/`; `np.bitwise_and(a, b)` is CS0117). Only the `&`/`|`/`^` operators and the engine methods (`DefaultEngine.BitwiseOp.cs:14-33` → `ExecuteBinaryOp`) exist, despite the project docs listing the names as supported. NumPy parity means **creating them, with `out=`/`where=` from day one**.

**Totals: 35 np.* names touched (3 created, 32 gaining overloads); 35 TensorEngine signatures (6 comparison + 3 predicate + 3 bitwise + 22 unary [21 ops, Round×2] + ATan2).**

### 1.2 Performance — the families inherit the Wave-2 win for free

Wave 2.1 measured the production end-to-end: **`np.add(a, b, out)` 446 ns vs 834 ns allocating** at N=1K. The delta is the result-allocation lifecycle: ≈804 ns of the allocating path is result allocation, of which ≈500 ns is the **two finalizable objects per result** (`~NDArray` + Disposer registration, finalizer-queue churn, extra-gen survival — Wave 2.3 profiling, roadmap §5). The Wave-2.4 buffer pool reclaims pages but cannot remove the finalizer lifecycle; `out=` is the idiomatic zero-alloc escape hatch.

These families construct **the same iterator with the same kernels** as the core 18 — the comparison Into-path reuses `npy_cmp_*` cache keys, bitwise rides `ExecuteBinaryUfuncInto` unchanged, the unary batch rides `ExecuteUnaryUfuncInto` unchanged — so the ~390 ns/call saving transfers without new kernel work. Measured family baselines (roadmap + this branch): bitwise contig/strided 1.9–2.9× *faster* than NumPy; comparison controls at parity-or-faster except one known gap (column-broadcast comparison 2.4× behind NumPy — **adjacent but out-of-scope here**; it is an inner-kernel issue on the no-out route, unaffected by parameter threading).

### 1.3 Composition

- `where=` enables conditional in-place updates without boolean-indexing temporaries: `np.floor(x, out: x, where: mask)` is one pass, zero temps; today it is mask-index → compute → scatter (three allocations).
- Comparisons with `out=` feed preallocated mask pipelines: `np.less(a, b, out: maskBuf)` then `np.add(x, y, out: dst, where: maskBuf)` — a steady-state loop with **zero** allocations.
- Comparison `out=` with a numeric dtype is NumPy's idiom for branchless 0/1 arrays (probe A1/A2: `True→1.0`).

### 1.4 Consistency / teachability

Half the elementwise API honoring ufunc kwargs and half not is a trap. `np.add(a, b, out)` compiles; `np.less(a, b, out)` is CS1501; `np.bitwise_and(a, b)` is CS0117 — three different failure modes for what NumPy users perceive as one calling convention. The Wave 2.1 core set the contract (semantics, error texts, tests); this work finishes the surface that is already NpyIter-backed.

### 1.5 Cost honesty — why this is LOW risk

Everything hard already exists and is suite-pinned:
- masked execution: `ForEach`'s mask-TRUE run decomposition (Wave 2.1), masked windowed flush (`CopyWindowFromBufferMasked`, Wave 1.3) incl. multi-window >8192;
- dtype-mismatched out: the Wave-4 windowed buffer machinery (`BUFFERED|GROWINNER|DELAY_BUFALLOC` + UNSAFE), Advance bug (b) fixed (Wave 1.2/1.4);
- aliasing: COPY_IF_OVERLAP + write-back (Wave 1.1);
- validation: `ValidateWhereMask` / `ValidateOutCast` / `ResolveUfuncIterationShape` with NumPy texts verbatim (`DefaultEngine.UfuncOut.cs:121-225`);
- 0-d EXLOOP iterators: `ResolveInnerLoopCount` fixed in Wave 2.1;
- flag arrays: static readonly per-op configs (Wave 2.2 discipline) — **the comparison Into-path needs zero new flag arrays**: its operand layout `[in, in, out(, mask)]` is identical to the binary one, so `s_ufuncBinaryOutFlags` / `s_ufuncBinaryOutMaskedFlags` are reused as-is.

The only genuinely new design surface is the comparison **return type** (§4.1.3). Everything else is plumbing.

---

## 2. NumPy 2.4.2 probe evidence

All probes run this session on NumPy 2.4.2 (`python probe_*.py`). Texts verbatim; trailing spaces preserved where NumPy emits them.

### 2.1 Comparison `out=` dtype matrix (block A1) — bool casts same_kind to **everything numeric**

```
a = arange(4, f64); b = a + 0.5   (a < b everywhere)
less(f64,f64,out=bool)      -> OK bool       [ True,  True,  True,  True]
less(f64,f64,out=int8)      -> OK int8       [1, 1, 1, 1]
less(f64,f64,out=uint8)     -> OK uint8      [1, 1, 1, 1]
less(f64,f64,out=int16)     -> OK int16      [1, 1, 1, 1]
less(f64,f64,out=int32)     -> OK int32      [1, 1, 1, 1]
less(f64,f64,out=int64)     -> OK int64      [1, 1, 1, 1]
less(f64,f64,out=float16)   -> OK float16    [1., 1., 1., 1.]
less(f64,f64,out=float32)   -> OK float32    [1., 1., 1., 1.]
less(f64,f64,out=float64)   -> OK float64    [1., 1., 1., 1.]
less(f64,f64,out=complex128)-> OK complex128 [1.+0.j, 1.+0.j, 1.+0.j, 1.+0.j]
greater(x,y,out=f64 prior)  -> [0. 1. 0. 1.]   (False→0.0, True→1.0; returns the SAME object)
```

No numeric out dtype fails: `can_cast(bool, uint32, 'same_kind') → True`, `can_cast(bool, complex128, 'same_kind') → True` (probe D1). NumSharp's `NpyIterCasting.IsSafeCast` ends with `if (srcType == Boolean) return true;` for *any* destination (`NpyIterCasting.cs:131-133`), and same_kind ⊇ safe — so **`ValidateOutCast(Boolean, X, name)` never throws across NumSharp's 15 dtypes** (incl. the Char/Decimal extensions). The call stays for structural parity, not because a NumSharp-reachable error exists.

### 2.2 Comparisons compare at `result_type(lhs, rhs)` (block A3)

```
greater(int64 [2^53+1], float64 [2^53]) -> [False]
equal  (int64 [2^53+1], float64 [2^53]) -> [ True]      # both cast to f64; 2^53+1 rounds to 2^53
result_type(int64, float64) -> float64
```

NumPy promotes both operands to the common dtype and compares there (it does **not** do exact int-vs-float comparison for arrays). This is exactly what NumSharp's comparison route already does — `np._FindCommonScalarType(lhsType, rhsType)` with per-element converts fused inside the kernel (`CompareOp.cs:394-427`) — so the Into-path inherits correct semantics with the same body.

### 2.3 Comparison `where=` (blocks A4–A6, D12)

```
less(a,b,out=f64 prior=-5, where=[T,F,T,F]) -> [ 1. -5.  1. -5.]
greater(a,b,out=i8 prior=9, where=[T,F,T,F]) -> [0, 9, 0, 9]            # cast + mask compose
less(a,b,where=m)  (no out) -> masked-on slots True/True; dtype bool; shape (4,)
   UserWarning: 'where' used without 'out', expect unitialized memory in output.
   If this is intentional, use out=None.                                 # note NumPy's typo "unitialized"
less((4,),(4,),where=(2,4)-mask) -> shape (2, 4) bool                    # where JOINS the output shape
less((2,4),(4,),out=(2,4) prior-True, where=row(4,)) -> mask broadcast over rows
```

Masked-off slots without `out` are **unobservable garbage** — proven dramatically by bitwise (B5): the uninitialized slots contained `1090519040` (= 0x41000000, a stale float 8.0f bit pattern) and `1065353216` (0x3F800000, float 1.0f). Tests may only assert masked-on slots.

### 2.4 Comparison out shape / aliasing / layout (blocks A7–A10)

```
less(a,b,out=(5,))  -> ValueError: operands could not be broadcast together with shapes (4,) (4,) (5,) 
less(a,b,out=(1,))  -> ValueError: non-broadcastable output operand with shape (1,) doesn't match the broadcast shape (4,)
less(a,b,out=(2,4)) -> OK — inputs broadcast UP, rows repeat
less(aa, 4.0, out=aa)                    -> [1. 1. 1. 1. 0. 0. 0. 0.]   # full alias f64 out, well-defined
less(aa2[:-1], aa2[1:], out=aa2[:-1])    -> [1. 1. 1. 1. 1. 1. 1. 7.]   # partial overlap, COPY_IF_OVERLAP
less(0d, 0d, out=0d bool)  -> True  ()   # reference identity preserved
less(0d, 0d, out=0d f64)   -> 1.0
less(empty, empty, out=empty bool/f64) -> OK
out=big[::2]      -> [T F T F T F T F]                                  # strided write-through
F-order out       -> F_CONTIGUOUS preserved, values correct
transposed out    -> OK
out=buf[3:7] f64  -> [0 0 0 1 1 1 1 0 0 0]                              # offset slice + cast
```

Identical texts/rules to the Wave 2.1 binary pins — `ResolveUfuncIterationShape` already produces all of them (incl. the trailing space). The unary variant lists only `(input) (out) `: `sinh(f8(4,), out=(5,))` → `operands could not be broadcast together with shapes (4,) (5,) ` (block C8).

### 2.5 Predicates (blocks A11, D8, E1, E2)

```
v = [1.0, nan, inf, -inf]
isnan(v,out=bool)   -> [False,  True, False, False]
isnan(v,out=uint8)  -> [0, 1, 0, 0]
isnan(v,out=int32)  -> [0, 1, 0, 0]
isnan(v,out=f64)    -> [0., 1., 0., 0.]
isfinite(v,out=bool)-> [ True, False, False, False]
isinf(v,out=bool)   -> [False, False,  True,  True]
isnan(i32, out=bool/f64) -> all 0                       # int inputs: valid, all-False
isnan(v,out=f64 prior=-1, where=[T,F,T,F]) -> [ 0., -1.,  0., -1.]
isnan(v,out=(5,))   -> ValueError: operands could not be broadcast together with shapes (4,) (5,) 
isnan(v,where=int)  -> TypeError: Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'
isnan(v,out=i32 strided big[::2]) -> writes through, [0,1,0,0] interleaved
isnan(complex128 [1+1j, 2, nan+1j, 3-2j]) -> [False, False,  True, False]; out=u1 -> [0,0,1,0]
isnan(float16) -> OK; out=i32 -> [0,1,0,0]
```

### 2.6 Bitwise (blocks B1–B5, D1)

```
i4a=[0b1100,0b1010,0b1111,0b0001]; i4b=[0b1010,0b1100,0b0101,0b0011]
bitwise_and(i4,i4,out=int32)  -> [8, 8, 5, 1]
bitwise_and(i4,i4,out=int64)  -> [8, 8, 5, 1]
bitwise_and(i4,i4,out=int16)  -> [8, 8, 5, 1]                       # same_kind narrowing OK
bitwise_and(i4,i4,out=uint32) -> UFuncTypeError: Cannot cast ufunc 'bitwise_and' output from dtype('int32') to dtype('uint32') with casting rule 'same_kind'
bitwise_and(i4,i4,out=float64)-> [8., 8., 5., 1.]                   # int→float IS same_kind
bitwise_and(i4,i4,out=float32)-> [8., 8., 5., 1.]
bitwise_and(i4,i4,out=bool)   -> UFuncTypeError: … from dtype('int32') to dtype('bool') with casting rule 'same_kind'
bitwise_and(i4,i8)            -> loop int64                          # inputs promote first
bitwise_and(i4,i8,out=i4)     -> [8,8,5,1] int32                     # i8 loop → same_kind narrowing
bitwise_or / bitwise_xor (i4,i4,out=i4) -> [14,14,15,3] / [6,6,10,2]
bitwise_and(bool,bool)        -> loop bool [T,F,F,F]
bitwise_and(bool,bool,out=i4) -> [1, 0, 0, 0]
bitwise_and(bool,bool,out=f8) -> [1., 0., 0., 0.]
bitwise_and(u4,u4,out=u4/i8/f8) -> all OK                            # u→wider i IS same_kind
bitwise_and(i4,i4,out,where=[T,F,T,F]) prior=-1 -> [ 8, -1,  5, -1]
bitwise_and(i4,i4,where=m) no out -> [8, 1090519040, 5, 1065353216]  # garbage slots + the UserWarning
```

**same_kind is directional across int kinds** (probe D1): `can_cast(i4,u4,'same_kind') → False`, `can_cast(u4,i4,'same_kind') → True`, `can_cast(u4,i8) → True`, `can_cast(i4,bool) → False`. NumSharp's `IsSameKindCast` already encodes exactly this (`NpyIterCasting.cs:155-191`: "int → int for every signedness pair EXCEPT signed → unsigned"), so validation falls out of the existing matrix.

### 2.7 Bitwise/invert no-loop errors and validation ORDER (blocks B4, B6, C8)

```
bitwise_and(f8,f8)            -> TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
bitwise_and(f8,f8,out=i4)     -> same no-loop TypeError                # no-loop BEATS bad-out-cast
bitwise_and(f8,f8,out=(5,))   -> same no-loop TypeError                # no-loop BEATS bad-out-shape
bitwise_and(f8,f8,where=int)  -> TypeError: Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'
                                                                       # bad-where BEATS no-loop
bitwise_and(i4,i4,out=(5,))   -> ValueError: operands could not be broadcast together with shapes (4,) (4,) (5,) 
invert(i4)        -> [-13, -11, -16,  -2]
invert(i4,out=i8) -> int64 same values;  invert(i4,out=f8) -> [-13.,-11.,-16.,-2.]
invert(u4)        -> [4294967294, …] uint32;  invert(u4,out=i4) -> [-2,-3,-4,-5]   # u→i same_kind
invert(bool)      -> logical not [F,F,T,T];  invert(bool,out=i4) -> [0, 0, 1, 1]
invert(f8)        -> TypeError: ufunc 'invert' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
invert(f8,out=i4) -> same no-loop TypeError
invert(f8,where=int) -> the where 'safe' TypeError                      # where first again
invert(i4,out prior=-1,where=[T,F,T,F]) -> [-13, -1, -16, -1]
sinh(f8,out=i4,where=int)     -> the where 'safe' TypeError             # where BEATS out-cast
sinh(f8,out=i4)               -> UFuncTypeError: Cannot cast ufunc 'sinh' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
sinh(f8(4,),out=(5,)f8)       -> ValueError: operands could not be broadcast together with shapes (4,) (5,) 
less(a,b,out=(5,),where=int)  -> the where 'safe' TypeError             # where BEATS bad-out-shape
less(a,b,out=(4,),where=(3,)) -> ValueError: operands could not be broadcast together with shapes (4,) (4,) (4,) (3,) 
less(a,b,out=(4,),where=(2,4))-> ValueError: non-broadcastable output operand with shape (4,) doesn't match the broadcast shape (2,4)
```

**Pinned validation order:** ① `where` bool conversion (argument parsing, `_wheremask_converter` ufunc_object.c:579-600) → ② loop resolution (no-loop TypeError) → ③ out same_kind cast (UFuncTypeError) → ④ broadcast/shape (iterator construction, ValueError). NumSharp's `ExecuteBinaryUfuncInto` already runs `ValidateWhereMask` first then `ValidateOutCast` then shapes — matching ①③④; the bitwise/invert no-loop check must be inserted **between** them (§4.2.2).

### 2.8 Unary loop dtype comes from INPUTS — proven by precision (block C1)

```
sinh(i32) -> float64;  sinh(i8) -> float16                            # the f16/f32/f64 tier
sinh(i4,out=f8/f4/f2) -> OK (f64 loop, same_kind down-casts)
sinh(i4,out=i4) -> UFuncTypeError: Cannot cast ufunc 'sinh' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
sinh(i1,out=f8) -> [0., 1.17480469, 3.62695312, 10.015625]            # f16 PRECISION in an f64 out!
sinh(f4,out=f8) -> [0., 1.17520118, 3.62686038, 10.01787472]          # f32 precision
sinh(f8)        -> [0., 1.17520119, 3.62686041, 10.01787493]          # f64 precision
```

`sinh(i1, out=f8)` runs the **float16 loop** (input tier) and casts the f16 values up — the out dtype does not promote the loop. This is the single most load-bearing semantic for the unary batch: NumSharp's `ResolveUnaryFloatReturnType` (`DefaultEngine.ResolveUnaryReturnType.cs:36-76`: bool/i8/u8→f16, i16/u16/char→f32, i32+→f64, floats preserved) computes the loop dtype **before** the out/where branch, and out only constrains the write-back cast. The existing `ExecuteUnaryUfuncInto` already implements exactly this.

Tier representatives (block C2): `log2(i4,out=f8)→[0,1,2,3]`, `exp2(i8)→float16 [1,2,4,8]`, `cbrt(i16)→float32`, `deg2rad(i32)→float64`, `arcsin(i8,out=f2)→[0,1.57,0,1.57]`, and (C9) `tanh(f8,out=f4)`, `arctan(i4,out=f8)`, `expm1(f8,out=f8)`, `log1p(i4,out=f8)` all OK.

### 2.9 Dtype-preserving unary + the integer identity loops (block C3)

```
floor(i4)            -> int32 [0,1,2,3]                                # identity loop, dtype preserved
floor(i4,out=i4)     -> int32 [0,1,2,3]
floor(i4,out=f8)     -> [0., 1., 2., 3.]                               # int loop → cast f8
floor(f8,out=i4)     -> UFuncTypeError: Cannot cast ufunc 'floor' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
floor(f8,out=f4)     -> [ 1., -2.,  2., -1.]
ceil(i4,out=i4) / trunc(i4,out=i4) -> identity
floor(bool)          -> bool [T,F,T,F]                                 # even bool has the identity loop
sign(i4)             -> int32 [-1, 0, 1, -1];  sign(i4,out=f8) -> [-1., 0., 1., -1.]
sign(f8,out=i4)      -> UFuncTypeError: … ufunc 'sign' … float64 → int32 …
sign(bool)           -> UFuncTypeError: ufunc 'sign' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None
floor(i4,out prior=-99,where=[T,F,T,F]) -> [  0, -99,   2, -99]
```

`floor/ceil/trunc` on integers are **identity loops at the input dtype** — with `out=f8` the ints are computed (identity) then cast. NumSharp's int shortcut (`Default.Floor.cs:16-17`: `Cast(nd, nd.GetTypeCode, copy: true)`) bypasses the unary ladder entirely; §4.3.2 designs the out/where route for it.

### 2.10 `reciprocal` — semantics divergence found (block C4)

```
reciprocal(i4 [1,2,-3,0])        -> [1, 0, 0, -2147483648]  RuntimeWarning: invalid value encountered in reciprocal
reciprocal(i4,out=i4)            -> same
reciprocal(i4,out=f8)            -> [ 1., 0., 0., -2.14748365e+09]    # int loop → cast (the MinValue casts!)
reciprocal(f8 [1,2,4,8],out=f8)  -> [1., 0.5, 0.25, 0.125]
reciprocal(bool)                 -> int8 [1, 0, 1, 0]  (divide-by-zero + invalid warnings)   # bool→i8 loop!
```

**Bug collected:** `DefaultEngine.ReciprocalInteger` (`Default.Reciprocal.cs:25-100`) returns **0** for `x == 0` and its comment claims NumPy parity — NumPy 2.4.2 returns **int.MinValue** (`-2147483648` for i4) with a RuntimeWarning. Additionally `ReciprocalInteger` walks `nd.Unsafe.Address` linearly, and `_Unsafe.Address` **throws** `InvalidOperationException` for sliced/broadcast inputs (`NDArray.Unmanaged.cs:43-50`) — `np.reciprocal` on a strided int view throws today. Both predate this work; §4.3.3 routes the out= path so the fix lands together.

### 2.11 `round` with decimals is a COMPOSITION in NumPy (blocks C5, D9)

```
round(f8 [1.44,1.55,2.5,-2.5], 1)            -> [ 1.4,  1.6,  2.5, -2.5]
round(f8, 1, out=f8)                         -> same, returns out (identity confirmed)
round(f8, out=f8)  (decimals=0)              -> [ 1.,  2.,  2., -2.]            # rint / banker's
round(f8, 1, out=f4)                         -> float32 [1.4, 1.6, 2.5, -2.5]
round(i4, 1, out=i4)                         -> identity ints
round(i4, -1)                                -> [20, 20, 40, 40]                 # real work on ints, banker's
round(f8, out=i4)  (decimals=0)              -> UFuncTypeError: Cannot cast ufunc 'rint' output from …
round(f8, 1, out=i4)                         -> UFuncTypeError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
```

The `'multiply'` ufunc name in the decimals≠0 error is the smoking gun: NumPy implements `round(x, d≠0)` as a **multiply → rint → divide composition with `out=` threaded through** the constituent ufuncs. `round(x, 0, out)` is the `rint` ufunc directly. §4.3.3 mirrors this.

### 2.12 `arctan2` (block C6)

```
arctan2(i4,i4)          -> float64 [0., 1.57079633, -1.57079633, 0.78539816]
arctan2(i1,i1)          -> float16;   arctan2(i2,i2) -> float32          # the tier
arctan2(i4,i4,out=f8)   -> OK;  out=f4 -> OK (f64 loop → same_kind down)
arctan2(i4,i4,out=i4)   -> UFuncTypeError: Cannot cast ufunc 'arctan2' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'
arctan2(f8,f8,out prior=-1,where=[T,F,T,F]) -> [ 0., -1., -1.57079633, -1.]
```

NumPy's ufunc name is **`arctan2`** — NumSharp's `UfuncName(BinaryOp)` default arm would produce `"atan2"` (`op.ToString().ToLowerInvariant()`, `UfuncOut.cs:88`). An explicit `BinaryOp.ATan2 => "arctan2"` mapping is required (§4.3.4).

### 2.13 `positive` / `negative` on bool (block C7) — loop errors win over out

```
positive(bool)            -> UFuncTypeError: ufunc 'positive' did not contain a loop with signature matching types <class 'numpy.dtypes.BoolDType'> -> None
positive(bool,out=bool)   -> same error                                 # out does not change it
negative(bool)            -> TypeError: The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.
negative(bool,out=i4)     -> same error
```

### 2.14 Scalar / 0-d `where`, complex comparisons, `out=` tuple form (blocks D5–D7, E1, E6)

```
add(a,b,out prior=-1,where=True)             -> full write [0.5, 2.5, 4.5, 6.5]
add(a,b,out prior=-1,where=False)            -> untouched [-1,-1,-1,-1]
less(a,b,out prior=-1,where=np.array(False)) -> untouched (0-d bool mask, stride-0 broadcast)
less(0d,0d,out=0d f64 prior=-1,where=0d False) -> array(-1.)             # 0-d everything
less(A(2,3), col(2,1), out=(2,3) bool)       -> [[T,F,F],[T,F,F]]        # col-broadcast inputs + out
equal(c128,c128)            -> bool [T,F,F,T];  out=f8 -> [1.,0.,0.,1.]
less/greater(c128,c128)     -> WORK in 2.4.2 (real-then-imag), RuntimeWarning on NaN operands
less(a,b,out=(o,)) tuple kw  -> OK                                       # Python-only form, C# N/A
```

### 2.15 Every out-cast error ufunc name pinned (final probe)

`log2, log10, log1p, exp2, expm1, cbrt, tanh, cosh, sinh, arcsin, arccos, arctan, deg2rad, rad2deg, trunc, ceil, floor, reciprocal, rint` — all produce
`Cannot cast ufunc '<name>' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'`, and **`np.round(x, out=i4)` / `np.around(x, out=i4)` say `'rint'`** (not "round"/"around"). `np.modf(x, o1, o2)` and `out=(o1, o2)` both work in NumPy (block D11) — recorded as the excluded multi-output evidence.

---

## 3. The shared engine contract (recap of what gets reused)

Reference architecture: `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.UfuncOut.cs`.

| concern | mechanism | where |
|---|---|---|
| where must be bool | `ValidateWhereMask` — 'safe' rule text verbatim | UfuncOut.cs:121-128 |
| out same_kind from loop dtype | `ValidateOutCast` — UFuncTypeError text with ufunc name | UfuncOut.cs:135-145 |
| out joins broadcast, never stretched; where joins output shape | `ResolveUfuncIterationShape` — all four NumPy texts incl. trailing space | UfuncOut.cs:160-225 |
| masked execution | trailing READONLY\|ARRAYMASK operand + WRITEMASKED\|NO_BROADCAST out; `ForEach` decomposes inner chunks into mask-TRUE runs (NumPy ufunc_object.c:2190-2226: `op[nop] = wheremask`, outputs WRITEMASKED, masked `execute_ufunc_loop`) | UfuncOut.cs:49-68, 322-339 |
| dtype-mismatched out | out becomes a CAST operand: `BUFFERED\|GROWINNER\|DELAY_BUFALLOC` + `NPY_UNSAFE_CASTING` (loop dtypes already validated — NumPy passes unsafe identically, ufunc_object.c:1078-1083), kernel writes loop dtype into the window, flush casts (and masks) on write-back | UfuncOut.cs:289-304 |
| aliasing | `EXTERNAL_LOOP\|COPY_IF_OVERLAP` + `OVERLAP_ASSUME_ELEMENTWISE_PER_OP`; write-back on Dispose | UfuncOut.cs:296, Wave 1.1 |
| flag arrays | static readonly (`s_ufuncBinaryOutFlags`, `s_ufuncBinaryOutMaskedFlags`, `s_ufuncUnaryOutFlags`, `s_ufuncUnaryOutMaskedFlags`) — no per-call `new[]` for call-invariant data | UfuncOut.cs:42-68 |
| trivial-loop interaction | branch sits **before** scalar×scalar and the trivial bypass. NumPy nuance, stated precisely: a wheremask always forces the full iterator (ufunc_object.c:2213-2227); a provided `out` *without* where still attempts `check_for_trivial_loop` (c:2235-2247). NumSharp Wave 2.1 routes **all** out/where to the iterator — semantically identical, conservative, measured fine (446 ns e2e). An out-aware trivial bypass is a possible later micro-opt, not part of this plan. | BinaryOp.cs:102-110, UnaryOp.cs:68-75 |
| 0-d operands | `ResolveInnerLoopCount` 0-d EXLOOP fix shipped in Wave 2.1 | roadmap §5 Wave 2.1 |
| bool→X flush casts | `TryGetCastKernel` returns null for Boolean-involved pairs (`DirectILKernelGenerator.Cast.cs:102-104`); `CopyRunFromBuffer` then falls back to the scalar `NpyIterCasting.CopyWithCast` (`NpyIterBufferManager.cs:1073→1093`) — **correct, scalar-speed**. Optional later: SIMD bool→{int,float} widening cast kernels (perf only). | cited |

---

## 4. HOW — per family

### 4.1 Comparisons (`==  !=  <  <=  >  >=`) + predicates (`isnan/isfinite/isinf`)

The only family with a real design problem. Current ladder (`DefaultEngine.CompareOp.cs:26-105`): scalar×scalar (`ExecuteComparisonScalarScalar`) → trivial-contig bypass (`TryTrivialContiguousComparisonOp`) → NpyIter Tier-3B (`TryExecuteComparisonOpViaNpyIter`, cache keys `npy_cmp_{op}_{lhsType}_{rhsType}`) → Direct fallback → looser-F `copy('F')` post-step.

#### 4.1.1 Semantics to implement (all probed, §2.1–2.4)

1. Loop dtype is **Boolean** (the comparison computes at `result_type(lhs, rhs)` *inside* the kernel — fused converts, the Wave-4 measured winner for cheap ops — and emits bool).
2. `out` of ANY numeric dtype is valid (bool→X always same_kind); `True→1`, `False→0` at the out dtype, through the windowed flush.
3. `where` slots keep prior out contents; without out, fresh **uninitialized** result (NumPy UserWarning noted; NumSharp does not warn — consistent with Wave 2.1's binary behavior, `fillZeros:false`).
4. out joins the broadcast (inputs broadcast UP), never stretched; where joins the output shape; all four error texts identical to the binary ones (§2.4 — including the where-shape error listing `inputs… out where ` in that order, matching `ResolveUfuncIterationShape`'s existing construction).
5. out aliasing an input is well-defined (bool result cast into the f64 input — probe A8) → COPY_IF_OVERLAP handles it; the iterator sees a WRITE operand overlapping a READ operand and forces the temp.
6. With out provided, NumPy returns **out as-is** — the post-kernel `ShouldProduceFContigOutput`/`copy('F')` layout steps are **skipped** on the Into path (they would clone, breaking reference identity).

#### 4.1.2 New engine method — `ExecuteComparisonUfuncInto`

Mirrors `ExecuteBinaryUfuncInto` (UfuncOut.cs:241-343) with these deltas:

```
ExecuteComparisonUfuncInto(lhs, rhs, ComparisonOp op, lhsType, rhsType, @out, where):
  ValidateWhereMask(where)                                  // ① — first, NumPy parse order
  name = UfuncName(op)                                      // NEW: UfuncName(ComparisonOp) switch:
                                                            // Equal→"equal", NotEqual→"not_equal",
                                                            // Less→"less", LessEqual→"less_equal",
                                                            // Greater→"greater", GreaterEqual→"greater_equal"
                                                            // (ToString().ToLower() would give "notequal" — wrong)
  if (@out != null) ValidateOutCast(Boolean, @out.typecode, name)   // ③ — never throws for our 15 dtypes; structural
  (leftShape, _) = Broadcast(lhs.Shape, rhs.Shape)
  iterShape = ResolveUfuncIterationShape(leftShape.Clean(), [lhs, rhs], @out, where)   // ④ texts
  target = @out ?? new NDArray(Boolean, iterShape.Clean(), fillZeros: false)
  if (target.size == 0) return target
  scalarBody = EXACTLY TryExecuteComparisonOpViaNpyIter's   // CompareOp.cs:407-427: same-dtype fast arm,
                                                            // else stash-rhs → convert both to
                                                            // _FindCommonScalarType → EmitComparisonOperation
  vectorBody = null                                         // bool output breaks the Tier-3B same-type
                                                            // invariant (CompareOp.cs:429-432) — unchanged
  cacheKey = $"npy_cmp_{op}_{lhsType}_{rhsType}"            // SHARED with the existing Tier-3B route:
                                                            // same body ⇒ same compiled kernel, zero new entries
  outNeedsCast = target.typecode != Boolean
  globalFlags = EXTERNAL_LOOP | COPY_IF_OVERLAP
  if (outNeedsCast) globalFlags |= BUFFERED | GROWINNER | DELAY_BUFALLOC; casting = UNSAFE
  if (where == null):
      opDtypes = outNeedsCast ? [lhsType, rhsType, Boolean] : null
      iter = MultiNew(3, [lhs, rhs, target], …, s_ufuncBinaryOutFlags, opDtypes)      // REUSED array
      iter.ExecuteElementWiseBinary(lhsType, rhsType, Boolean, scalarBody, null, cacheKey)
  else:
      opDtypes = outNeedsCast ? [lhsType, rhsType, Boolean, Empty] : null
      iter = MultiNew(4, [lhs, rhs, target, where], …, s_ufuncBinaryOutMaskedFlags, opDtypes)  // REUSED
      iter.ExecuteElementWise([lhsType, rhsType, Boolean], scalarBody, null, cacheKey)
  return target
```

Key points, spelled out:
- **Flag arrays:** the operand layout `[READONLY, READONLY, WRITEONLY|NO_BROADCAST(|WRITEMASKED), (ARRAYMASK)]` is byte-identical to the binary config — `s_ufuncBinaryOutFlags` / `s_ufuncBinaryOutMaskedFlags` are reused; **no new static arrays** (Wave 2.2 discipline holds at zero cost).
- **Kernel/cache sharing:** the scalar body and `npy_cmp_*` key are the existing Tier-3B route's — first out= call on an already-used comparison (op, lhs, rhs) triple hits the warm cache.
- **bool→X flush:** out=f64/i32/… makes the out operand a CAST operand (`op_dtypes` request Boolean); the kernel writes bools into the window; `CopyRunFromBuffer` casts on flush via the scalar `CopyWithCast` fallback (Boolean pairs have no SIMD cast kernel — §3 last row). For `where` + cast simultaneously, the Wave-1.3 masked windowed copy-back (`CopyWindowFromBufferMasked`) already composes both — pinned for binary by `Where_MultiWindow_CastOut_MaskHoldsAcrossWindows` (20 005 elements, 3 windows).
- **Where the branch goes** (`ExecuteComparisonOp`, CompareOp.cs:26): immediately after `lhsType/rhsType` are read and **before** the scalar×scalar arm and the trivial bypass:
  ```csharp
  if (@out is not null || where is not null)
      return ExecuteComparisonUfuncInto(lhs, rhs, op, lhsType, rhsType, @out, where);
  ```
  Rationale: NumPy's masked execution never takes the trivial loop (ufunc_object.c:2213-2227); scalar×scalar with a 0-d out must route through the iterator for reference identity + write-masking (0-d EXLOOP works post-Wave-2.1; probe A9/D7 pin the expected values). Wave 2.1 precedent: `UnaryOp.cs:68-75`, `BinaryOp.cs:102-110`.
- **No F post-step on the Into path** (§4.1.1 point 6).

#### 4.1.3 THE design decision — return type

Engine comparisons return `NDArray<bool>` (TensorEngine.cs:151-156); NumPy's `np.less(a, b, out=x)` returns **x itself**, which may be float64. C# cannot overload on return type, and `NDArray<bool>` cannot wrap a float buffer. Options:

| option | description | verdict |
|---|---|---|
| **(i) out-taking overloads return plain `NDArray`** | Keep the existing 6 abstracts untouched (`NDArray<bool> Less(lhs, rhs)`), ADD 6 new abstracts `NDArray Less(NDArray lhs, NDArray rhs, NDArray @out, NDArray where = null)` (out **required** at the engine level — the 2-arg call binds the old exact-arity overload, so no CS0121). The returned reference is the caller's out (identity), or a fresh bool-typecode `NDArray` when only `where` was given. | **RECOMMENDED.** Matches NumPy exactly (the static type of `np.less(a,b,out=f64)` *is* "an array", dtype is runtime data); zero source breaks (existing callers keep `NDArray<bool>`); the no-out/no-where forms keep their typed sugar. The only asymmetry — `np.less(a, b, @out: boolArr)` statically returns `NDArray` not `NDArray<bool>` — is resolved by the caller's own reference (`r` *is* `boolArr`) or `MakeGeneric<bool>()`. |
| (ii) restrict out to Boolean dtype | `NDArray<bool> Less(…, NDArray<bool> @out, …)` | **REJECT.** Diverges from NumPy — probe A1 shows ALL ten numeric out dtypes succeed and are used idiomatically (0/1 arrays). Would also silently forbid the cast-flush path that already exists. Documented for completeness only. |
| (iii) generic gymnastics | `NDArray<T> Less<T>(…, NDArray<T> @out, …) where T : unmanaged` | **REJECT.** Dtype is runtime information in NumSharp (NPTypeCode), so every call site needs a 15-arm type switch to name `T`; TensorEngine abstracts cannot be generic-virtual over 15 instantiations without massive surface; no NumPy analog (NumPy has no static dtype either). Pure cost, no parity gain. |

**TensorEngine consequence (option i):** +6 abstract overloads (Compare/NotEqual/Less/LessEqual/Greater/GreaterEqual with required `@out`), +3 for predicates (`NDArray IsNan(NDArray a, NDArray @out, NDArray where = null)` etc.). Breaking for third-party TensorEngine implementors (abstract = must implement) — acceptable per project policy ("Breaking Changes OK"); alternatively `virtual` with a `NotSupportedException` default à la `Evaluate` (TensorEngine.cs:136-144) if implementor-compat is preferred. Recommendation: **abstract** — DefaultEngine is the only engine in-tree, and silent non-support of ufunc kwargs is worse than a compile break.

#### 4.1.4 Predicates ride the existing unary Into-path — verified

`IsNan/IsFinite/IsInf` already call `ExecuteUnaryOp(a, UnaryOp.IsNan/…, NPTypeCode.Boolean)` (Default.IsNan.cs:27), and `ExecuteUnaryOp`'s out/where branch (UnaryOp.cs:72-75) precedes everything — so threading `@out`/`where` through the three engine methods reaches `ExecuteUnaryUfuncInto` with `outputType = Boolean` and **no further engine work**:
- `bufferedPromoting` excludes predicates (`IsUnaryPredicateOp`, UfuncOut.cs:384-388) → the scalar body emits `EmitUnaryScalarOperation(op, inputType)` which itself produces bool — already correct for `isnan(i4)` (all-False, probe A11) and `isnan(c128)`/`isnan(f2)` (probe E1/E2).
- `out` of non-bool dtype → `outNeedsCast` → windowed bool→X flush, same as comparisons.
- `ValidateOutCast(Boolean, X, "isnan"/"isinf"/"isfinite")` — names map cleanly from `UnaryOp.ToString().ToLowerInvariant()` (IsNan→"isnan" ✓).
- **Trap:** `Default.IsNan.cs` ends with `.MakeGeneric<bool>()` — the out-taking overload must NOT call it (out may be f64). Same return-type pattern as 4.1.3(i): existing `NDArray<bool> IsNan(a)` unchanged; new `NDArray IsNan(a, @out, where = null)` returns the target.
- Empty-input early-return in `ExecuteUnaryOp` (UnaryOp.cs:37-43) already defers to the Into path when out/where present.

#### 4.1.5 np.* API for comparisons + predicates

```csharp
// Logic/np.comparison.cs — add one overload per function (×6); out REQUIRED not for
// ambiguity (the 2-arg exact match wins anyway) but for API clarity & house symmetry:
public static NDArray less(NDArray x1, NDArray x2, NDArray @out, NDArray where = null)
    => x1.TensorEngine.Less(x1, x2, @out, where);
// equal / not_equal / less_equal / greater / greater_equal identically.

// Logic/np.is.cs — ×3:
public static NDArray isnan(NDArray a, NDArray @out, NDArray where = null)
    => a.TensorEngine.IsNan(a, @out, where);
```

C# ambiguity analysis: the existing comparison surface is `(NDArray,NDArray)`, `(NDArray,object)`, `(object,NDArray)` — a 3/4-argument overload collides with none of them, and `np.less(a, b)` still binds the exact 2-arg form (candidates that fill no defaults beat candidates that do). Predicates: existing `(NDArray)` likewise wins for 1-arg calls. The `where`-only call is `np.less(a, b, null, mask)` — accepting `@out: null` mirrors `np.add`'s shipped shape (`Where_WithoutOut_MaskedSlotsComputed` test).

### 4.2 Bitwise `&  |  ^` + `invert`

#### 4.2.1 Engine work is near-zero — verified

`BitwiseAnd/Or/Xor` are thin wrappers over `ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseAnd/Or/Xor)` (`DefaultEngine.BitwiseOp.cs:14-33`). The Wave 2.1 branch inside `ExecuteBinaryOp` —

```csharp
// BinaryOp.cs:107-110
if (@out is not null || where is not null)
    return ExecuteBinaryUfuncInto(lhs, rhs, op, lhsType, rhsType, resultType, @out, where);
```

— is **not gated by op** (verified by reading the full ladder head, BinaryOp.cs:44-110): it sits after `_FindCommonType` + the bool `+/*`→`|/&` remap + the divide/power promotions, and fires for any `BinaryOp`. `UfuncName(BinaryOp)` already maps `BitwiseAnd→"bitwise_and"`, `BitwiseOr→"bitwise_or"`, `BitwiseXor→"bitwise_xor"` (UfuncOut.cs:85-87 — confirmed present). `CanUseSimdForOp` includes the three bitwise ops (`DirectILKernelGenerator.MixedType.cs:203-211`) — same-dtype bitwise out= gets the **full SIMD vector body**, unlike comparisons. So the engine work is:

1. `BitwiseAnd/Or/Xor(lhs, rhs)` → `BitwiseAnd(lhs, rhs, NDArray @out = null, NDArray where = null)` (TensorEngine + DefaultEngine, trailing optionals — house pattern, non-breaking).
2. **The no-loop guard** (next section).

Validation already-correct pieces: `bitwise_and(i4,i4,out=u4)` must throw — falls out of `ValidateOutCast` because `IsSameKindCast` already rejects signed→unsigned (NpyIterCasting.cs:175-177; probe §2.6); `out=f8` passes (int→float same_kind); `out=bool` throws (int→bool rejected, NpyIterCasting.cs:154 doc + probe). `bool&bool` inputs: `_FindCommonType` → Boolean loop; `out=i4` → bool→int32 flush cast → `[1,0,0,0]` (probe B3). Mixed `i4&i8` → i8 loop → out=i4 same_kind narrowing (probe B2).

#### 4.2.2 The no-loop validation (bitwise float/complex/decimal inputs)

Today `f8 & f8` falls through the dispatch ladder until the kernel emit fails — surfacing as a NumSharp-internal `NotSupportedException` ("IL kernel not available …"), not NumPy's TypeError. The pinned text and order (§2.7):

- text: `ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''` (note the doubled quotes around safe — NumPy verbatim; the identical string already exists in-tree at `NpyExpr.Typing.cs:443-446` and `Default.Shift.cs:54` — reuse, do not re-derive);
- order: the no-loop error fires **before** out-cast and out-shape errors but **after** the where bool check.

Placement: at the top of the three `BitwiseAnd/Or/Xor` engine methods (covering both the out and no-out routes in one place), guarded on `resultType` not integer/bool — i.e., *before* delegating to `ExecuteBinaryOp`. `ValidateWhereMask` must run first to preserve order ①②: simplest faithful sequence inside the engine method is `ValidateWhereMask(where); ValidateBitwiseLoop(lhsType, rhsType, name); ExecuteBinaryOp(...)`. (`ExecuteBinaryUfuncInto` will call `ValidateWhereMask` again — idempotent, nanoseconds.)

#### 4.2.3 np.* creation (the missing three)

New file `Math/np.bitwise.cs`:

```csharp
public static NDArray bitwise_and(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null)
    => x1.TensorEngine.BitwiseAnd(x1, x2, @out, where);
// bitwise_or / bitwise_xor identically.
```

No existing overloads → no ambiguity → the **clean-binary optional pattern** (like `np.add`). Scalar conveniences (`bitwise_and(NDArray, object)`) may be added to match the comparison surface, but the operators already cover that usage; keep the minimum that closes CS0117.

#### 4.2.4 `invert` (unary BitwiseNot / LogicalNot)

`Default.Invert.cs:15-26` dispatches bool→`ExecuteUnaryOp(LogicalNot, Boolean)`, ints→`ExecuteUnaryOp(BitwiseNot, typeCode ?? input)`. Threading `@out`/`where` through both arms reaches the existing unary Into-branch. Specifics:
- dtype rules: int preserve (i4→i4; out=i8/f8 cast, out=u4-from-i4 rejected, u4-loop→i4 out accepted — probe §2.7 pins all four), bool preserve (`invert(bool,out=i4)` → `[0,0,1,1]`).
- **no-loop guard** for float/complex/decimal inputs with the `'invert'` text (pinned §2.7; today the ladder fails with the non-parity internal error). Order: where-check → no-loop → out-cast, same as bitwise; the existing in-tree string at `NpyExpr.Typing.cs:506-508` is the verbatim source.
- `UfuncName(UnaryOp)` additions required: `BitwiseNot → "invert"`, `LogicalNot → "logical_not"` (defaults would yield "bitwisenot"/"logicalnot").
- np.*: `np.invert(x, NPTypeCode?)/(x, Type)` exist → **required-out overload** `np.invert(NDArray x, NDArray @out, NDArray where = null)` (optional-out would CS0121 against `(x, NPTypeCode? outType = null)` on 1-arg calls).

### 4.3 Remaining unary math (~20 ops) + `arctan2`

#### 4.3.1 The batch — pure signature threading

All of these already flow `Default.<Op>.cs` → `ExecuteUnaryOp(nd, UnaryOp.X, resolvedOutputType)` → (with out/where) `ExecuteUnaryUfuncInto`, which composes the promoting buffered-cast route (sqrt(i32)-class) with a provided out (the out operand simply also requests the compute dtype; the flush casts and masks). **The engine Into-path needs zero changes for the regular ops.** Work per op = TensorEngine abstract + DefaultEngine override gain trailing `NDArray @out = null, NDArray where = null` + pass-through, + np.* overload.

Exact TensorEngine members (22 signatures; `TensorEngine.cs` lines as of this branch) and their override files:

| # | TensorEngine member (NPTypeCode overload) | Default override file | loop-dtype rule | ufunc name for errors |
|---|---|---|---|---|
| 1 | `Log2(nd, NPTypeCode?)` | `Default.Log2.cs` | float tier | `log2` ✓(auto) |
| 2 | `Log10(nd, NPTypeCode?)` | `Default.Log10.cs` | float tier | `log10` ✓ |
| 3 | `Log1p(nd, NPTypeCode?)` | `Default.Log1p.cs` | float tier | `log1p` ✓ |
| 4 | `Exp2(nd, NPTypeCode?)` | `Default.Exp2.cs` | float tier | `exp2` ✓ |
| 5 | `Expm1(nd, NPTypeCode?)` | `Default.Expm1.cs` | float tier | `expm1` ✓ |
| 6 | `Cbrt(nd, NPTypeCode?)` | `Default.Cbrt.cs` | float tier | `cbrt` ✓ |
| 7 | `Sign(nd, NPTypeCode?)` | `Default.Sign.cs` | preserve | `sign` ✓; sign(bool) no-loop error §2.13 |
| 8 | `Floor(nd, NPTypeCode?)` | `Default.Floor.cs` | preserve (int identity) | `floor` ✓ — §4.3.2 |
| 9 | `Ceil(nd, NPTypeCode?)` | `Default.Ceil.cs` | preserve (int identity) | `ceil` ✓ — §4.3.2 |
| 10 | `Round(nd, NPTypeCode?)` | `Default.Round.cs:16` | preserve (int identity) | **`rint`** (probe §2.15 — explicit map needed). Adjacent divergence: no int shortcut — `round_(i4)` runs `ResolveUnaryReturnType` → `GetComputingType` → **f64** (NPTypeCode.cs:615-621), NumPy preserves int32; audit while threading |
| 11 | `Round(nd, int decimals, NPTypeCode?)` | `Default.Round.cs:25` | composition | `multiply`/`rint`/`divide` per stage — §4.3.3 |
| 12 | `Truncate(nd, NPTypeCode?)` | `Default.Truncate.cs` | preserve (int identity) | **`trunc`** (explicit map; default "truncate" wrong). NOTE: unlike Floor/Ceil it has **no** int shortcut today — `trunc(i4)` runs `ResolveUnaryReturnType` → `GetComputingType(Int32)` = **Double** (NPTypeCode.cs:615-621), diverging from NumPy's int32-preserve (probe §2.9); fix lands with the §4.3.2 identity-emit work |
| 13 | `Reciprocal(nd, NPTypeCode?)` | `Default.Reciprocal.cs` | preserve-int / float | `reciprocal` ✓ — §4.3.3 |
| 14 | `Sinh(nd, NPTypeCode?)` | `Default.Sinh.cs` | float tier | `sinh` ✓ |
| 15 | `Cosh(nd, NPTypeCode?)` | `Default.Cosh.cs` | float tier | `cosh` ✓ |
| 16 | `Tanh(nd, NPTypeCode?)` | `Default.Tanh.cs` | float tier | `tanh` ✓ |
| 17 | `ATan(nd, NPTypeCode?)` | `Default.ATan.cs` | float tier | **`arctan`** (explicit; "atan" wrong) |
| 18 | `ACos(nd, NPTypeCode?)` | `Default.ACos.cs` | float tier | **`arccos`** |
| 19 | `ASin(nd, NPTypeCode?)` | `Default.ASin.cs` | float tier | **`arcsin`** |
| 20 | `Deg2Rad(nd, NPTypeCode?)` | `Default.Deg2Rad.cs` | float tier | `deg2rad` ✓ |
| 21 | `Rad2Deg(nd, NPTypeCode?)` | `Default.Rad2Deg.cs` | float tier | `rad2deg` ✓ |
| 22 | `Invert(nd, NPTypeCode?)` | `Default.Invert.cs` | preserve / bool | **`invert`** / **`logical_not`** — §4.2.4 |

(The parallel `(nd, Type dtype)` overloads delegate to the NPTypeCode ones and need no change. "✓(auto)" = `UnaryOp.ToString().ToLowerInvariant()` already yields the NumPy name; the **seven bold names** need explicit `UfuncName(UnaryOp)` arms: `Round→rint`, `Truncate→trunc`, `ASin→arcsin`, `ACos→arccos`, `ATan→arctan`, `BitwiseNot→invert`, `LogicalNot→logical_not`. Plus `UfuncName(BinaryOp)`: `ATan2→arctan2`.)

The float-tier rule is `ResolveUnaryFloatReturnType` (bool/i8/u8→f16, i16/u16/char→f32, i32+→f64, floats/decimal preserved — `DefaultEngine.ResolveUnaryReturnType.cs:36-76`), already invoked by every override **before** `ExecuteUnaryOp` — so the loop dtype is fixed by inputs and out only constrains the cast, exactly the probed `sinh(i1,out=f8)`-shows-f16-precision semantic (§2.8). For inputs that promote (e.g. `log2(i4)` → f64 loop), `bufferedPromoting` engages the Wave-4 windowed input cast and the same-dtype SIMD body — already composing with out (UfuncOut.cs:379-447).

np.* overload pattern — the **required-out rule** applies to every function in this batch (each has a `(x, NPTypeCode? dtype = null)` overload; an optional-out overload makes the 1-arg call ambiguous — both candidates would fill defaults — CS0121; precedent: `np.sqrt(x, NDArray @out, NDArray where = null)` in `np.sqrt.cs`):

```csharp
public static NDArray floor(NDArray x, NDArray @out, NDArray where = null)
    => x.TensorEngine.Floor(x, (NPTypeCode?)null, @out, where);
```

`round_`/`around` get both arities: `(x, NDArray @out, …)` and `(x, int decimals, NDArray @out, …)`. `np.exp2`/`np.expm1` technically could take an optional out (their non-nullable `NPTypeCode` overload doesn't collide with a bare 1-arg call) — standardize on required-out anyway for one teachable rule across the unary surface.

#### 4.3.2 Special case: floor/ceil/trunc/round on integer inputs

`Default.Floor.cs:16-17` / `Default.Ceil.cs:16-17` shortcut integer inputs to `Cast(nd, nd.GetTypeCode, copy: true)` — bypassing the ladder, so the threaded parameters would be silently dropped. NumPy ground truth (§2.9): these are **identity loops at the input dtype** (`floor(i4,out=i4)` identity; `floor(i4,out=f8)` ints-cast-to-f8; `floor(i4,out,where)` keeps prior at masked-off; even `floor(bool)` → bool). Design options:

- **(a) RECOMMENDED — teach the emitter the identity loop:** `EmitUnaryScalarOperation(Floor/Ceil/Truncate/Round, <integer dtype>)` emits nothing (value already on the stack = the result), mirroring NumPy registering identity inner loops for integer floor/ceil/trunc/rint. Then delete the int shortcut for the out/where path (keep it for the plain path — it's a fast memcpy) and let `ExecuteUnaryOp(nd, op, inputType, @out, where)` flow normally: out=i4 → same-dtype write; out=f8 → windowed i4→f8 flush; where → masked. One emitter switch-arm, fixes all four ops × all 8 int dtypes + bool.
- (b) special-case in `ExecuteUnaryUfuncInto` (swap the scalar body for an identity emit when int + rounding-op) — works but hides a dtype rule inside the Into path that the plain ladder then lacks.

`reciprocal(bool)` → **int8 loop** in NumPy (probe §2.10, `[1,0,1,0]`) — a quirk; NumSharp today routes bool through `ResolveUnaryReturnType` → float-ish. Out-of-scope correction; note only (the bool input is pathological).

#### 4.3.3 Special cases: reciprocal(int) and round(decimals≠0)

**`reciprocal` int inputs** (`Default.Reciprocal.cs:19-20` → `ReciprocalInteger`): hand loop with two collected defects (§2.10: returns 0 for 1/0 where NumPy 2.4.2 returns int.MinValue; throws on strided views via `_Unsafe.Address`). For this plan the **minimal correct out/where route** is: compute `tmp = Reciprocal(nd)` via the existing path, then masked-copy `tmp` into out through the established Into machinery (a copy-shaped `ExecuteUnaryUfuncInto` with identity body, or `np.copyto`-equivalent) — one temp, correct semantics, no new emitter work. The **better fix** — an integer-division scalar body (`1/x` with the div-by-zero → MinValue hardware semantic, matching NumPy's C loop) emitted into the Into path — also fixes both pre-existing defects and removes `ReciprocalInteger`; recommended if the slice has budget, but the temp+copy fallback is acceptable to ship the API. Float inputs need nothing (already on the ladder).

**`round(x, decimals≠0, out, where)`** (`Default.Round.cs:25-58`, hand loop after a Cast): mirror NumPy's probed composition (§2.11 — the `'multiply'` error name proves the structure): `t = multiply(x, 10^d)` → `Round(t, out: t)` (rint, in-place, already supported once #10 threads) → `divide(t, 10^d, out: @out, where: where)` — multiply/divide **already have out=/where=** from Wave 2.1, so the composition is three shipped calls. decimals==0 routes to the plain Round ufunc path (name `rint`). Integer inputs with decimals ≥ 0: identity (probe `round(i4,1,out=i4)`); negative decimals on ints do real banker's work in NumPy (`round(i4,-1)` → `[20,20,40,40]`) — NumSharp's current decimals path doesn't implement negative-decimals-on-int either; out= threading does not change that pre-existing gap (note, don't fix here).

#### 4.3.4 `arctan2` — binary, but NOT on the binary ladder (audit correction)

`Default.ATan2.cs` has its own Direct route (`ExecuteATan2Op` → `MixedTypeKernel` via `ClassifyATan2Path`); it never touches `ExecuteBinaryOp` or NpyIter. The out/where route does **not** require migrating that: `ExecuteBinaryUfuncInto` is op-agnostic and its bodies support ATan2 today — `EmitScalarOperation` special-cases `BinaryOp.ATan2 → EmitATan2Operation` (Math.Atan2 call, `DirectILKernelGenerator.cs:1175-1178`; decimal via `DecimalMath.ATan2`, c:1514-1517), and the mixed-dtype arm (`EmitMixedScalarBody`, BinaryOp.cs:563-583) converts both operands to the result dtype then calls the same emitter. `CanUseSimdForOp(ATan2) = false` → scalar body, correct (Math.Atan2 has no vector form). So:

```csharp
public override NDArray ATan2(NDArray y, NDArray x, NPTypeCode? typeCode = null,
                              NDArray @out = null, NDArray where = null)
{
    var resultType = typeCode ?? PromoteATan2Binary(y.GetTypeCode, x.GetTypeCode);  // i1→f16, i2→f32, i4+→f64 — probed §2.12
    if (@out is not null || where is not null)
        return ExecuteBinaryUfuncInto(y, x, BinaryOp.ATan2,
                                      y.GetTypeCode, x.GetTypeCode, resultType, @out, where);
    …existing Direct path unchanged…
}
```

Plus the `UfuncName(BinaryOp.ATan2) => "arctan2"` mapping (§2.12 — the switch default would emit `atan2`). `atan2(i4,i4,out=f4)`: loop f64 → f4 same_kind ✓ (probed); `out=i4` → the pinned `'arctan2'` UFuncTypeError. np.*: `np.arctan2(y, x, NDArray @out, NDArray where = null)` — **required out** (`(y, x, NPTypeCode? dtype = null)` exists → CS0121 otherwise). Bonus: the Into route gives arctan2 its **first** NpyIter-backed execution (strided/broadcast atan2 currently runs the Direct coordinate walker) — out= callers get iterator-quality layout handling for free.

#### 4.3.5 `positive` / `negative` bool guards (adjacent, 5 lines)

Probed §2.13: both fail at **loop resolution** regardless of out. `np.negative` already has out= (Wave 2.1); `np.positive(nd)` has neither out nor the bool guard. While threading the batch, add the two pinned error texts (`positive`: no-loop UFuncTypeError text; `negative`: the special "boolean negative" TypeError — text already in-tree for evaluate at `NpyExpr.Typing.cs:438-441` in subtract form; the negative-specific wording is pinned in §2.13). `np.positive` out= support itself is a 2-liner on the Negate…no — positive is identity: `ExecuteUnaryUfuncInto` with an identity body (same emit as §4.3.2(a)) — include it in the batch as the 22nd-bis op if desired; it is listed by NumPy as a ufunc with out/where. (Not counted in the 35; call it optional scope.)

### 4.4 `modf` — EXCLUDED, with rationale

NumPy signature `np.modf(x, out1, out2)` / `out=(a, b)` (probed working, §2.15/D11). Two WRITE operands means the Into-path contract changes shape: `ResolveUfuncIterationShape` must validate two outs, the iterator gets `[in, out1, out2(, mask)]`, and the kernel needs the multi-output per-chunk signature — exactly the "multi-output (Modf)" migration step already scheduled in the Phase-3 family order (CLAUDE.md migration priority: "… → copy → multi-output (Modf) → selection"). Bolting a second WRITEMASKED output onto the current single-out helpers would duplicate that work ahead of its design. **Decision: defer to the Phase-3 Modf migration; when Modf moves to per-chunk kernels, `out=`/`where=` lands with it.** (NumSharp's `ModF` returns a tuple today; no API removal involved.)

---

## 5. Scenario & edge-case coverage matrix (the test plan)

Families: **CMP** = comparisons ×6 ops, **PRED** = isnan/isfinite/isinf, **BIT** = and/or/xor, **INV** = invert, **U-P** = unary dtype-preserving (floor/ceil/trunc/round/sign/reciprocal), **U-F** = unary tier-promoting (logs/exps/hyp/arc/deg-rad/cbrt), **AT2** = arctan2. Every cell's expected outcome is pinned by the §2 evidence block noted. ✔ = test required; (✔) = one representative per family suffices; — = N/A.

| # | scenario | CMP | PRED | BIT | INV | U-P | U-F | AT2 | expected (probe) |
|---|---|---|---|---|---|---|---|---|---|
| S1 | out same-dtype contiguous, identity + values | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | same instance returned; values per §2.1/2.5/2.6/2.9/2.8/2.12 |
| S2 | out same_kind cast (small) | ✔ bool→{i1,u1,i4,f4,f8,c16} | ✔ bool→{u1,i4,f8} | ✔ i4→{i2,i8,f4,f8} | ✔ i4→{i8,f8}, u4→i4 | ✔ i4→f8 (identity-cast) | ✔ f64→{f4,f2} | ✔ f64→f4 | A1/A11/B1/§2.7/C3/C1/C6 |
| S3 | out cast, **>8192 elements (multi-window flush)** | ✔ (20 005, bool→f4) | (✔) | ✔ (i4 loop→f8 out) | — | — | ✔ (i4→f64 loop→f4 out: input-cast + out-cast windows) | — | values derivable; machinery pinned by Wave-1.3/2.1 multi-window tests |
| S4 | out invalid cast — exact text w/ ufunc name | — (none reachable: bool→all-numeric same_kind, §2.1) | — (same) | ✔ i4→u4, i4→bool | ✔ f8 input → no-loop instead | ✔ floor/sign(f8,out=i4) | ✔ sinh(i4/f8, out=i4) | ✔ out=i4 `'arctan2'` | B1/§2.7/C3/C1/C6 texts verbatim |
| S5 | out wrong shape — `operands could not be broadcast … ` (trailing space) | ✔ (4,)(4,)(5,) | ✔ (4,)(5,) unary form | ✔ | (✔) | (✔) | (✔) | (✔) | A7/D8/B4/C8 |
| S6 | out smaller → `non-broadcastable output operand…` | ✔ (1,) vs (4,) | (✔) | (✔) | (✔) | (✔) | (✔) | (✔) | A7 |
| S7 | out LARGER — inputs broadcast UP | ✔ (4,)→(2,4) rows repeat | (✔) | (✔) | (✔) | (✔) | (✔) | (✔) | A7 row 3 |
| S8 | out strided view / F-order / transposed / offset slice | ✔ all four | ✔ strided (D8) | (✔) | — | — | (✔) | — | A10 |
| S9 | out aliasing input — full alias (dtype-cast write into f64 input) | ✔ | (✔) | ✔ in-place `a &= b` shape | ✔ in-place | ✔ `floor(x,out:x)` | ✔ | (✔) | A8 row 1; consume after `using` (COPY_IF_OVERLAP write-back at Dispose) |
| S10 | out partial overlap (slices of same buffer) | ✔ | — | ✔ | — | — | (✔) | — | A8 row 2 |
| S11 | 0-d scalars with 0-d out (+ 0-d where) | ✔ bool + f64 outs | (✔) | (✔) | — | — | (✔) | (✔) | A9/D7 |
| S12 | empty arrays with out | ✔ | (✔) | (✔) | — | — | (✔) | — | A9 |
| S13 | where dense/sparse/all-false/all-true masks | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | A4/B5/§2.7/C3/C1/C6; where=all-false leaves out untouched (D5) |
| S14 | where stride-0 / 0-d broadcast mask | ✔ np.array(false) scalar | — | (✔) | — | — | (✔) | — | D5/D7 |
| S15 | where joins output shape (no out) + masked-on-only asserts | ✔ (2,4) join | (✔) | ✔ garbage-slot lesson | — | — | (✔) | — | A5; B5 (garbage = stale float bit patterns — never assert masked-off) |
| S16 | where + out: masked-off keep prior | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | ✔ | A4/A11/B5/§2.7/C3/C1/C6 |
| S17 | where non-bool — 'safe' text | ✔ | ✔ | ✔ | ✔ | (✔) | ✔ | (✔) | A12/B4/§2.7/C8 |
| S18 | where + out-cast composed (masked windowed flush), small AND >8192 | ✔ bool→f8 + i1 (D12) | ✔ (A11) | ✔ | — | — | ✔ multi-window | — | D12/A11; multi-window machinery = Wave-1.3 masked copy-back |
| S19 | validation ORDER: bad-where beats bad-out-shape / out-cast / no-loop | ✔ | ✔ | ✔ | ✔ | — | ✔ | — | A12 / B4 / §2.7 (C8: `sinh(f8,out=i4,where=int)` → where text) |
| S20 | validation ORDER: no-loop beats bad-out (bitwise/invert float inputs) | — | — | ✔ out=i4 AND out=(5,) | ✔ | — | — | — | B4/§2.7 |
| S21 | where-shape errors list `inputs… out where ` / where-stretches-out | ✔ both texts | — | (✔) | — | — | — | — | A12 rows 4-5 |
| S22 | return-type/static-type checks (out=bool → NDArray (is the out instance); out=f64 → NDArray; no-out forms keep `NDArray<bool>`) | ✔ | ✔ | — | — | — | — | — | §4.1.3 design |
| S23 | mixed-dtype inputs with out (loop = result_type) | ✔ i4 vs f8 → i1 out (D2); 2^53 precision pin (A3) | — | ✔ i4&i8→i4 out (B2) | — | — | — | ✔ i4,i4 tier (C6) | D2/A3/B2/C6 |
| S24 | bool×bool inputs with int/float out | ✔ equal(bool,bool,out=i4) (D4) | — | ✔ B3 | ✔ invert(bool,out=i4) | — | — | — | D4/B3/§2.7 |
| S25 | dtype edges: Half / Complex / Decimal | ✔ f2 cmp (D3), c128 equal+ordered cmp (E1) | ✔ isnan(f2/c128) (E1/E2) | Decimal/float → no-loop ✔ | same ✔ | Decimal round/floor: NumSharp ext — document supported-or-throw | Half via tier ✔ | Decimal via DecimalMath ✔ | D3/E1/E2; NumSharp-only dtypes (Char/Decimal) probed impossible — pin NumSharp behavior + `[Misaligned]` where it extends NumPy |
| S26 | int-identity loops: floor/ceil/trunc/round(i4) with out=i4 / out=f8 / where | — | — | — | — | ✔ C3 ([0,-99,2,-99] where-pin) | — | — | C3 |
| S27 | reciprocal int semantics with out (1/0 → MinValue NumPy) | — | — | — | — | ✔ + `[OpenBugs]`/`[Misaligned]` for the existing 0-vs-MinValue divergence | — | — | C4 |
| S28 | round decimals≠0 with out (+f4 out), negative decimals int | — | — | — | — | ✔ C5/D9 ('multiply' composition) | — | — | C5/D9 |
| S29 | loop-dtype-from-inputs precision pin (sinh(i1,out=f8) shows f16 values) | — | — | — | — | — | ✔ THE pin | — | C1 |
| S30 | positive/negative(bool) error parity (with/without out) | — | — | — | — | ✔ both texts | — | — | C7 |

**Regression gates (every slice):** `ctor_probe.cs` interleaved A/B — construction stays ~181-183 ns (the Wave-1.3 OR-sweep lesson: per-op validation loops must be gated, not unconditionally extended); `variation_probe.{cs,py}` grid neutral; full suite (9596 baseline) green; **all timing via `dotnet run -c Release`** (the Debug-default invalidator — Debug numbers are void, roadmap header + handover §4.12).

Estimated new tests: CMP ~28, PRED ~10, BIT+INV ~18, unary batch ~25, AT2 ~6 → **~85-90 tests** (one file per slice following `UfuncOutWhereTests.cs` conventions: NumPy-pinned values, texts verbatim incl. trailing spaces, `Assert.IsTrue(ReferenceEquals(r, o))` identity checks).

---

## 6. Risks, ordering, effort

### 6.1 Risk register

| risk | exposure | mitigation |
|---|---|---|
| out/where branch placed after a bypass → mask/identity silently skipped | comparisons (scalar×scalar + trivial arms), floor/ceil int shortcut, reciprocal-int shortcut, Round-decimals path — **every pre-ladder shortcut in a touched override must be audited** | branch FIRST (NumPy rule, ufunc_object.c:2213; Wave 2.1 precedent); S1/S11/S26 tests catch regressions |
| bool→X cast kernel coverage in the flush | comparison/predicate out≠bool | NO gap: `TryGetCastKernel` null for Boolean pairs → scalar `CopyWithCast` fallback (NpyIterBufferManager.cs:1025-1045 fill / 1073-1093 flush). Perf-only; optional SIMD bool-widening kernels later. S3 multi-window tests exercise it |
| validation order divergence | bitwise/invert (no-loop position), all (where-first) | pinned matrix §2.7: where → no-loop → out-cast → shape; engine sequence per §4.2.2; S19/S20/S21 |
| `NDArray<bool>` API surface | comparison/predicate consumers | option (i): additive overloads, zero source breaks; TensorEngine gains abstracts (engine implementors break — accepted policy, or `virtual`+throw à la `Evaluate`) |
| ufunc-name drift in error texts | 7 UnaryOp + 1 BinaryOp + 6 ComparisonOp names whose `ToString().ToLowerInvariant()` is wrong (`rint`, `trunc`, `arcsin`, `arccos`, `arctan`, `invert`, `logical_not`, `arctan2`, `not_equal`, `less_equal`, `greater_equal`) | explicit switch arms; S4 texts pin every name |
| ctor-cost creep from new validation | small-N path (the +11.9 ns Wave-1.3 incident) | new checks live in the np.*/engine glue (only on out/where calls), NOT in iterator construction; ctor_probe gate |
| comparison F-layout post-step cloning the user's out | reference-identity break | Into path never runs `ShouldProduceFContigOutput` (§4.1.1.6); S1 identity asserts |
| `where`-without-out reading uninitialized slots in tests | flaky CI | the B5 garbage evidence is the rule: assert masked-on slots only (S15) |
| pre-existing adjacent defects surfacing in review | `ReciprocalInteger` (0 vs MinValue; strided throw), bitwise/invert float internal error text, col-broadcast comparison 2.4× perf gap | collected & documented here (§2.10, §4.2.2, §1.2); fix-or-tag `[OpenBugs]`/`[Misaligned]` in the owning slice; perf gap stays out-of-scope |

### 6.2 Implementation order (3 slices, each independently shippable)

1. **Bitwise first** (`&|^` + np.bitwise_* creation + no-loop guards). Near-zero engine work — validates that the Wave-2.1 binary branch generalizes beyond arithmetic with no op-gating surprises, and closes the CS0117 hole. Files: `TensorEngine.cs` (3 sigs), `DefaultEngine.BitwiseOp.cs`, new `Math/np.bitwise.cs`, `UfuncOut.cs` (no-loop helper), tests (~18). **Smallest possible diff, highest doc-mismatch payoff.**
2. **Unary batch** (22 sigs + invert + arctan2 + positive/negative guards). Mechanical threading ×22 override files + np.* overloads ×22 files (one per existing np.<op>.cs) + the §4.3.2 identity-emit arm + §4.3.3 compositions + `UfuncName` additions + the ATan2 Into hookup. Tests ~31. Big-but-shallow; every op rides the proven `ExecuteUnaryUfuncInto`.
3. **Comparisons + predicates last** — the only genuinely new Into-path (`ExecuteComparisonUfuncInto`) + the return-type decision + 9 engine overloads + np.* overloads ×9. Tests ~38. Last because it carries the one design decision and the most scenario rows; by then slices 1-2 have re-validated every shared mechanism.

Rough size: slice 1 ≈ 5 files / ~250 LOC; slice 2 ≈ 48 files / ~700 LOC (mostly 6-line overloads); slice 3 ≈ 14 files / ~600 LOC. Each gated by ctor_probe + variation grid + full suite.

### 6.3 Restated non-goals

Shifts and `maximum`/`minimum` need route migration onto NpyIter before parameter threading is meaningful (their measured plan is separate); reductions' `out=` is Wave 5; `modf` waits for the Phase-3 multi-output contract (§4.4); `casting=`/`order=`/`subok=`/`dtype=` ufunc kwargs are a later surface; the column-broadcast comparison perf gap (2.4×) is an inner-kernel work item independent of this plan.
