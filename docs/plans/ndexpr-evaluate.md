# NDExpr / `np.evaluate` — state, gaps, and the plan to a high-quality, high-performance, broad-coverage fused-expression engine

> Written 2026-09-07 on branch `exprs` (worktree on `journey3` @ `b648ee53`). Every number below
> comes from `benchmark/fusion/probes/expr_probe.cs` + `numpy_twins.py` (added with this plan;
> Release, pinned to one P-core, `DOTNET_TC_CallCountingDelayMs=0`, NumPy 2.4.2 on the same core).
> Ratios follow the house convention **NPY/NS = NumPy_ms / NumSharp_ms, higher = NumSharp faster**;
> "unfused/fused" is NumSharp's own unfused chain divided by the fused call. File:line citations are
> into this checkout; NumPy citations are into `refs/numpy/` (the submodule is populated in the main
> checkout, not in this worktree).

---

## 0. Bottom lines

- **What exists is sound.** `np.evaluate` is a real fused-expression engine: an immutable `NDExpr`
  tree is bound (array leaves deduplicated by reference), typed **per node** with NumPy 2.x
  `result_type` semantics including NEP50 weak literals, compiled once into a cached Tier-3B inner
  loop (4×-unrolled SIMD + scalar-strided fallback + AVX2 gather), and driven by ONE NDIter pass with
  ufunc-grade `out=` (same_kind cast, COPY_IF_OVERLAP). Root reductions (`Sum/Prod/Min/Max/Mean`, flat
  or single-axis, `keepdims`) fold the expression per element without a temporary. 233 unit tests
  across five files pin it; the whole 15-dtype × 18-shape sweep in §2.2 runs without a crash except
  where NumPy itself has no loop.
- **Three correctness defects** (§2.1, §3.1): `Where`/`LogicalNot`/any zero-test over an **int8**
  operand throws `Zero-push unsupported for SByte` (a real crash, currently *excused* in the fuzz
  registry as W1-E); `Abs(complex128)` returns complex `(|z|, 0)` where NumPy returns **float64**;
  and fused float **reductions are not NumPy-exact** (4-accumulator fold; f16/f32 accumulate in f64;
  429 ULP off on a broadcast input). A fourth is a shared engine gap: `int64` vs `uint64`
  comparisons are not exact (NumPy 2.x has dedicated `qQ`/`Qq` loops).
- **Performance is bimodal.** Homogeneous float arithmetic chains are excellent — `(a-b)/(a+b)`
  **4.3×** NumPy, `mean((a-b)²)` **7.9×**, `leaky-relu` **4.7×**, `sum(a*b)` **3.3×** at 4M — but
  any tree containing a **comparison, `Where`, `Min/Max`, a Half operand, or a mixed dtype** drops to
  the scalar path: `a > 0.5` is **0.16× NumPy at 100K**, `maximum(a,b)` **0.46×**, `f2*f2+f2` is
  **0.15× the unfused chain**, and the **fixed cost is ~1.0 µs + 3 KB of garbage per call** (unfused
  chain 0.36 µs / 0.9 KB; NumPy 0.52 µs), so every 1K-element case loses to the chain it replaces.
- **Coverage is ~60 % of the elementwise surface NumSharp already ships** (§3.2): no `Cast`, no
  `positive/conjugate/real/imag`, no `fmax/fmin/copysign/nextafter/logaddexp*/shifts`, no logical
  and/or/xor with NumPy's nonzero semantics, no `any/all/argmax/argmin/std/var/nan*` reductions, no
  multi-axis reductions, no `where=`, no comparison operators, literals only for `int/long/float/double`.
- **The plan** (§4) is seven phases ordered by measured loss: **P0** correctness + a NumPy
  differential oracle tier for `evaluate`; **P1** typed vector emission for masks (the biggest measured
  loss); **P2** NumPy-exact pairwise + SIMD reductions and the missing reduction kinds; **P3** fixed
  cost (compiled expressions, ≤ 450 ns / ≤ 600 B); **P4** node/API coverage to the full np.\* surface
  incl. `Cast`, `where=`, operators, literals; **P5** Half and mixed-width SIMD; **P6** optimizer
  passes (CSE, constant folding, parameter leaves), opt-in threading, string front-end, diagnostics.
  Each phase has acceptance numbers a re-run of the probe verifies.

### 0.1 Landed status (2026-09-08, branch `exprs`; the half point — P0 → P1 → P3 done, P2 → P4 → P5 → P6 open)

| Phase | Status | Commit | What landed / evidence |
|---|---|---|---|
| **P0** correctness + oracle | landed | `53602da9` | G1–G8 + G10 fixed (int8 zero-push, `Abs(complex)→float64`, exponent-array sign check, complex `Min/Max`, typed literals, exact `int64`/`uint64` compares, engine dispatch, `Call` dtype map). **`evaluate.jsonl`** tier: 14,702 cases, floor 11,800, green; `MisalignedRegistry` E1–E5. **E1** (float `Sum/Prod/Mean` 4-accumulator fold ≤16 ULP, complex ≤64 ULP of magnitude) and **E5** (complex min/max NaN identity on negative-stride views — NumPy reduces in logical order, the KEEPORDER iterator in memory order) are *pending Phase 2* and must be deleted with it, not widened. |
| **P1** typed vector emission | landed | `939d0636` | "Vector v2": one lane dtype W per kernel, Boolean-typed nodes as lane masks (`NDExpr.Vector.cs`), the fused inner-loop shell with bool operand expansion / packed bool output / hoisted stride-0 operands / AVX2 gather (`DirectILKernelGenerator.InnerLoop.Fused.cs`), `NDExpr.ForceScalar` test hook + the 47-tree × 8-lane × 8-layout vector-vs-scalar metamorphic sweep (`NDEvaluateVectorTests`, 11). Measured NPY/NS at 100K / 4M: `a>0.5` 0.65 / 1.26–1.45 (was 0.16 / 0.71), `where(a>b,a,b)` ~1.0 / 2.6, `maximum(a,b)` 0.9 / 1.8, `(a>0.2)&(b<0.8)` 1.26 / 1.85, leaky relu 17.6 / 5.2, f32 `af>0.5` 8× the unfused engine compare. |
| **P3** fixed cost | landed | `feat(evaluate): NDExpr Phase 3 — per-root compiled program, CompiledExpression handle, N-ary input broadcast` | `NDExprProgram` per-root cache (`NDExpr.Program.cs`), identical-dims fast path, single-allocation N-ary input broadcast with NumPy's every-operand error text, `NDExpr.Compile()` / `Compile(params NPTypeCode[])` → `CompiledExpression` (strict signature), the 0-d-operand parameter form pinned (`NDEvaluateProgramTests`, 14). Numbers below. |
| **6.1** structural program cache | landed | (2026-09-13 landing) | `NDExprProgramCache` + `NDExpr.Structure.cs`: a tree rebuilt per call finds its program by structural hash + node-by-node verification — rebuilt `a*b+c` 995 → **464–508 ns / 864 B** at n = 8, fused ≥ unfused on **13/15** rows at 1K (was 5/15). `Call` slot identity fixed. Numbers under "Both residuals closed" below. |
| **3.3** parameters (0-d inputs hoisted into the kernel aux block) | landed | (2026-09-13 landing) | `NDExpr.Params.cs`: a 0-d input is a kernel parameter, not a stride-0 operand — `a*b+k` 681 → **388 ns**, zero per-element cost, one kernel per structure (the windows compile 0 kernels over 20 distinct M). `NDEvaluateParamTests` (12). |
| P2 reductions (NumPy-exact pairwise, SIMD folds, Any/All/Arg/Std/Var/Nan*/Ptp/Average, tuple axis) | open | | E1/E5 excuses wait on it; `sum(af*bf)` f32 0.60× unfused and `max(a*b)` 0.44× NumPy at 100K are its cells. |
| P4 coverage (Cast, Positive/Conjugate/Real/Imag/…, FMax/FMin/CopySign/NextAfter/LogAddExp/shifts, logical and/or/xor, Select/Clip, `< > <= >=` operators, `where=`/`casting=`/`order=`/`dtype=`) | open | | |
| P5 Half + mixed-width SIMD | open | | `f2*f2+f2` 0.15× unfused; `i4*2+f8` scalar. |
| P6 optimizer (CSE / constant folding / unary-only delegation / opt-in threading / string front-end / `Explain`) | open | | the structural hash (6.1) is also what would make the inline-rebuilt spelling below cheap. |

**Phase 3 measured** (n = 8, ns / managed B per call; Release, one pinned P-core, `DOTNET_TC_CallCountingDelayMs=0`; probe section D, two runs):

| Call | before | after | reference |
|---|---:|---:|---|
| `np.evaluate(expr)` prebuilt tree | 1006 / 2896 | **404–419 / 584** | acceptance ≤ 450 / ≤ 600 ✓; unfused `a*b+c` 353 / 928; NumPy 524 |
| `np.evaluate(expr, out=)` | 831 / 2432 | **266–339 / 184–248** | unfused two-pass `out=` 332 / 400; NumPy 515 |
| `np.evaluate(Sum(a*b))` prebuilt | 775 / 2648 | **295 / 464** | unfused `np.sum(a*b)` 348 / 1048; NumPy 1717 |
| `np.evaluate(Where(a>b,a,b))` prebuilt | 935 / 2968 | **369–382 / 576** | unfused `np.where` 432 / 1040; NumPy 934 |
| positional `np.evaluate(expr, ops)` prebuilt | — | **377 / 584** | |
| `a*b+k`, `k` a 0-d operand (parameter form) | 823 / 1560 | **668 / 1368** | unfused `a*b+2.5` 453 / 1488 — closed below: **388 / 624** |
| `np.evaluate((NDExpr)a*b+c)` rebuilt per call | 1008 / 3048 | 901–1276 / 3128 (unchanged) | closed below: **464–508 / 864** |

**Section C at 1K after Phase 3** (µs per call, many-iteration timing; `unfused/fused`, > 1 = fused wins). With a
**prebuilt** tree fusion wins 16 of 22 rows — `a*b+c` 1.32, `(a-b)/(a+b)` 1.82, `sqrt(a*a+b*b)` 2.05,
`where` 1.28, `a>0.5` 1.53, `(a>0.2)&(b<0.8)` 3.55, leaky relu 3.23, `exp(a)*b` 1.08, `exp(af)*bf` 1.17,
`sin*cos` 1.20, `i4*2+f8` 1.69, `i8*i8+1` 2.44, strided 1.80, `sum(a*b)` 1.19, `mean((a-b)²)` 2.75,
`sum(ax1)` 1.24 — and the six it loses are each another phase's cell or have nothing to fuse:
`maximum(a,b)` 0.65 and `abs(a)` 0.67 (a single op — the engine's direct kernel route has a lower fixed
cost than an NDIter pass, P6.3), `f2*f2+f2` 0.18 (P5), `sum(af*bf)` 0.86 / `max(a*b)` 0.83 /
`sum(ax0)` 0.96 (P2's scalar folds). With the tree **rebuilt per call** only 9 of 22 rows win (see
residual 1). The 100K / 4M columns are unchanged from Phase 1 (fixed cost is invisible there).

**Both residuals closed (2026-09-13, landed on master together with the branch):**

1. **The inline-rebuilt spelling — Phase 6.1's structural cache** (`NDExpr.Structure.cs`,
   `NDExprProgramCache` in `NDExpr.Program.cs`). Every node folds its identity into a 64-bit
   order-sensitive hash while collecting the distinct array leaves in BindArrays' order (so an
   `ArrayNode` hashes as the input index it binds to); the hash + dtype signature + 0-d mask +
   `ForceScalar` picks a candidate program in a process-wide `ConcurrentDictionary`, and a hit is
   VERIFIED node by node against the candidate's bound tree (`StructureEquals`) — a hash-only match
   would run the wrong kernel. Two allocation-free walks replace bind + typing + string key. One entry
   per hash (a genuine collision evicts, never mis-pairs), 4096-entry cap with a wholesale clear (the
   kernels stay cached). Measured (n = 8, ns / B per call, pinned P-core,
   `DOTNET_TC_CallCountingDelayMs=0`, probe `benchmark/fusion/probes/fc_probe.cs`):
   `np.evaluate((NDExpr)a*b+c)` rebuilt **464–508 / 864** (old NDExpr 995 / 3056; Phase-3-only
   1580 / 3136 — the per-root cache had made the MISS path dearer, since `Build` also runs the vector
   plan); `out=` rebuilt **353–363 / 456**; `Sum(a*b)` rebuilt **368–388 / 704**; `Where(a>b,a,b)`
   rebuilt **448–456 / 880**. Section C at 1K with the rebuilt spelling: fused ≥ unfused on **13 of 15**
   rows (old NDExpr 5/15, Phase-3-only 6/15); the two losses are `maximum(a,b)` 0.59× and `abs(a)`
   0.63× — single-op trees against the direct kernel's lower fixed cost (P6.3). The library consumers
   (`np.sinc`, the windows, `trapezoid`, `gradient`) spell their trees inline and get this for free (the
   sinc tree at 1K 4.9 → 3.4 µs, 1.7× the unfused chain). A rode-along correctness fix: a `Call` node's
   delegate/target slot is now part of the kernel signature (two closures over one lambda body shared a
   kernel, and the second read the first's captured state), and `DelegateSlots` dedups registration by
   identity, so a field-held delegate rebuilt into a tree per call keeps one slot and one kernel.
2. **The 0-d operand — 3.3's parameter form, done properly** (`NDExpr.Params.cs`). A 0-d input is no
   longer a stride-0 iterator operand: the host copies its element into the kernel's aux block (16 bytes
   per parameter; after the accumulator slot for reductions) and the kernel prologue loads it once into
   a local — the scalar for the scalar body, `Vector.Create` / a lane mask for the vector body — so an
   `InputNode` reading it costs what a literal costs. The mask is part of the program identity (hash,
   `StructureEquals` verification, kernel cache key suffix `|p0110`), re-validated on the per-root fast
   path (`ndarray.resize` of a 0-d array re-resolves; a compiled handle refuses), and empty when EVERY
   input is 0-d (the result is 0-d; the iterator path stays). Measured: `a*b+k` prebuilt **388 / 624**
   (was 668–681 / 1368–1376; old NDExpr 1255 / 3752), rebuilt **481 / 896**; per element the stride-0
   operand had cost 8–10 % on BOTH kernel paths (the fused shell's per-load broadcast branch; the
   strided scalar fallback's stride multiply) — the hanning tree over a 0-d `M-1` now runs at literal
   speed (371.7 vs 372.1 µs at 100K, 3.72 vs 3.72 ms at 1M), and `np.hanning` over 20 distinct M compiles
   0 kernels where the literal compiled 20 (0.46 ms vs 3.64 ms). The windows (`np.windows.cs`) spell
   every M-dependent value as a 0-d operand. NDIter's stride-0 construction cost noted here before is
   thereby bypassed for evaluate (it still applies to the ufunc routes — `docs/NDITER_PERF_CONTINUATION.md`).

The named weak `NDExpr.Param("k")` leaf is still not spelled: the 0-d array IS the parameter
mechanism, and it is a **strong** scalar (`np.float64(k)` semantics: `int32 * 0-d int64 → int64`) where
a literal is weak — a sweep that must keep the literal's promotion still needs a weak-leaf form (Phase 6
optional). Gates: `NDEvaluateProgramTests` (21 — structural cache identity, capacity, thread-safety,
`Call` slots), `NDEvaluateParamTests` (12 — every dtype × both kernel paths, reductions, aliasing `out`,
the all-0-d fallback, the resize guard, positional forms, 16-byte slots), the `evaluate.jsonl` tier
(its `pp_scalar_*` pair layouts and `scalar_0d` drive the parameter path bit-exact).

---

## 1. What `np.evaluate` / NDExpr is today (as built)

### 1.1 Files

| File | Lines | Role |
|---|---:|---|
| `src/NumSharp.Core/APIs/np.evaluate.cs` | 58 | public surface: `evaluate(expr, out=)`, `evaluate(expr, operands, out=)`; rejects non-writeable `out` |
| `src/NumSharp.Core/Backends/Iterators/NDExpr.cs` | 1320 | node classes + scalar/vector emission, legacy `Compile`, `DelegateSlots` |
| `src/NumSharp.Core/Backends/Iterators/NDExpr.Typing.cs` | 610 | NumPy `result_type` pass (`NDExprTypeRules`, weak literals), `CompileNumPy` |
| `src/NumSharp.Core/Backends/Iterators/NDExpr.Evaluate.cs` | 980 | `ArrayNode`, binding, operator overloads, `ReduceNode` + the flat/axis reduce kernels |
| `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.Evaluate.cs` | 483 | the host: broadcast, F-order preservation, iterator config, `out=` cast, reduce seeding |
| `src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop.cs` | 1240 | the Tier-3B kernel shell every tree compiles into |
| `src/NumSharp.Core/Backends/Iterators/NDIter.Execution.Custom.cs` | 573 | Tier 3A/3B/3C entry points (`ExecuteExpression`, the EXTERNAL_LOOP foot-gun guard) |

Tests (233 methods): `NDEvaluateTests` (27, the NumPy-probed semantic pins), `NDExprCallTests` (33),
`NDExprExtensiveTests` (107), `NDIterCustomOpTests` (14), `NDIterCustomOpEdgeCaseTests` (52).
Benchmark gate: `benchmark/fusion/{evaluate_bench.cs,evaluate_bench.py,fusion_sheet.py}` →
`fusion_results.md` (a fixed-expression report, appended to every `run_benchmark.py` run).
User docs: `docs/website-src/docs/NDIter.md` ("Tier 3C — Expression DSL" + "Fused kernels in
production"), `il-generation.md`, `.claude/CLAUDE.md` → "Fused Expressions".

### 1.2 Pipeline

```
NDExpr tree (immutable; NDArray leaves via Arr()/implicit; literals via Const()/implicit)
  → BindArrays          ArrayNode → InputNode(i), one operand per DISTINCT NDArray reference
  → ResolveNumPyTypes   node → dtype side table (reference-keyed Dictionary), NEP50 per node
  → CompileNumPy        cache key "NDExpr:<signature>:in=…:out=…|np" → NDInnerLoopFunc
                         scalar body always; vector body iff EVERY operand and EVERY node share one
                         SIMD-capable dtype AND every node's op has a vector emit (SupportsSimdAt)
  → DefaultEngine       Broadcast inputs → ResolveUfuncIterationShape(out) → result alloc
                         (F-contiguous when all non-scalar inputs are strictly F) → NDIterRef.MultiNew
                         (EXTERNAL_LOOP | COPY_IF_OVERLAP; BUFFERED+UNSAFE when out needs a cast)
  → iter.ForEach(kernel)
Reductions: root ReduceNode → CompileReduceKernel (flat: 4 scalar accumulators into a 16-byte aux
slot, seeded by the host) or CompileAxisReduceKernel (REDUCE_OK iterator; pinned stripe vs slab)
```

### 1.3 Node catalog as built (SIMD = has a vector emit today)

| Family | Nodes | SIMD | Notes |
|---|---|:-:|---|
| Leaves | `Input(i)`, `Arr(nd)`, `Const(int/long/float/double)` | ✓ | literals are NEP50 weak; no `bool/uint/ulong/Half/decimal/Complex` spelling |
| Binary arithmetic | `Add Subtract Multiply Divide` | ✓ | bool add/multiply are logical or/and; divide of ints → f64 |
| Binary arithmetic | `Mod Power FloorDivide ATan2` | — | NumPy floored mod, `NDIntegerPower` wrapping, `npy_floor_divide` port, `Math.Atan2` |
| Bitwise | `BitwiseAnd BitwiseOr BitwiseXor` | ✓ | float/complex/decimal → NumPy no-loop TypeError text |
| Select | `Min(a,b) Max(a,b) Clamp Where` | — | branchy scalar; NaN-propagating like `np.maximum` |
| Unary arithmetic | `Negate Abs Square Reciprocal Deg2Rad Rad2Deg BitwiseNot` | ✓ | |
| Unary arithmetic | `Sqrt` | ✓ | |
| Unary transcendental | `Exp Exp2 Log Sin Cos Tanh` | f32 only | the NumPy-ported bit-exact float32 kernels (`NDFloatMath`) |
| Unary transcendental | `Expm1 Log2 Log10 Log1p Tan Sinh Cosh ASin ACos ATan Asinh Acosh Atanh Cbrt Sign` | — | CRT calls |
| Rounding | `Floor Ceil Round Truncate` | f32/f64 when the BCL has the method | identity on ints (NumPy loops) |
| Predicates | `IsNaN IsFinite IsInf LogicalNot` | — | bool result, tested at the child's dtype |
| Comparison | `Equal NotEqual Less LessEqual Greater GreaterEqual` | — | bool result; compared at `result_type(l, r)` |
| Call | `Call(Func<…>)`, `Call(Delegate)`, `Call(MethodInfo[, target])` | — | 12 CLR signature dtypes (no SByte/Half/Complex) |
| Reductions (root only) | `Sum Prod Min Max Mean` × {flat, `axis`, `keepdims`} | — | NumPy reduce dtypes; f16/f32 sum/prod/mean accumulate in f64 |
| Operators | `+ - * / % & | ^`, unary `- ~ !` | | mixed `NDExpr`/`NDArray`/`int`/`long`/`double` overloads; **no** `< > <= >=` |

### 1.4 What the shell gives every tree for free — and what it cannot

`DirectILKernelGenerator.CompileInnerLoop` (`InnerLoop.cs:163`) wraps the tree's scalar/vector
bodies in: a contiguous 4×-unrolled SIMD loop + 1-vector remainder + scalar tail; for binary trees a
scalar-broadcast SL/SR path (`Vector.Create(*scalar)` hoisted); an **AVX2 hardware-gather** path for
strided inputs of 32/64-bit dtypes; a scalar contiguous loop for mixed dtypes; and the scalar strided
fallback. That is why §2.5 shows no layout cliff any more (F, T, `[:, ::2]`, `[::2, :]` and
broadcast all fused-faster-than-unfused at 4M). Two structural limits matter for the plan:

1. **The SIMD gate is "all operands AND all nodes one dtype"** (`CanSimdAllOperands`,
   `InnerLoop.cs:456`; `CompileNumPy`'s `homogeneous` check, `NDExpr.Typing.cs:328`). A comparison
   node types as `Boolean`, so **any tree with a comparison, `Where`, predicate or logical node is
   scalar-only**, and so is every mixed-dtype tree (`i4*2+f8`) and every Half/Decimal/Complex tree.
2. **`np.evaluate` calls `iter.ForEach(kernel)` directly** (`DefaultEngine.Evaluate.cs:195`), not
   the packed-key `ExecuteElementWise` entry points, so it never reaches the 2-D block kernels
   (`Is2DElementwiseShape` / masked variant) that the production ufunc routes use for narrow strided
   rows and `where=` masks.

---

## 2. Measured state (2026-09-07; reproduce with the probe)

### 2.1 Semantics probes (section A)

| Id | Probe | Result | Verdict |
|---|---|---|---|
| A1 | `Where(int8 cond, f8, f8)` | `NotSupportedException: Zero-push unsupported for SByte` | **BUG** — `WhereNode.EmitPushZeroPublic` (`NDExpr.cs:948`) has Byte/Int16/UInt16 but not SByte; `int16` cond works |
| A2 | `LogicalNot(int8)` | same throw | **BUG** (same root); uint8 fine |
| A3 | `Abs(complex128)` | `Complex (5, 0)`; NumPy `float64 5.0` | **DIVERGENCE** — `UnaryNode.InferType` treats Abs as dtype-preserving; emitter returns `Complex(|z|,0)` |
| A4 | `Sign(complex)` = z/|z|; `Square`/`Exp(complex)` bit-equal `np.square`/`np.exp`; `IsNaN(complex)` true iff a part is NaN | OK | |
| A5 | `Power(i4, i4 with negatives)` | silently computes (`(-32)^-32 → 1`); NumPy `ValueError: Integers to negative integer powers are not allowed.` | documented divergence (only literal exponents are checked) |
| A6 | `Sum/Prod/Mean(complex)` | OK, equals `np.sum` | |
| A6c | `Min(complex)` | throws NSE; NumPy returns the lexicographic min `(nan+0j)` here | **GAP** |
| A7 | `f2*f2+f2`, `Exp(f2)`, `Sqrt(f2)` fused vs unfused | bit-equal | OK (per-op Half rounding preserved) — but 0.15× the unfused speed, §2.3 |
| A8 | `Sum(af*bf)` f32, N=100 003 | fused `0x47429D72` = **NumPy**; NumSharp unfused `np.sum` `0x47429D75` (3 ULP) | fused matches by coincidence (f64 accumulate → round); the core's flat f32 sum is off NumPy |
| A9b | `Sum(a*b)` f64, N=100 003 | NumPy `…D784`, unfused `…D783` (1 ULP), fused `…D780` (4 ULP) | **fused not NumPy-exact**; core flat sum drifts too (16 ULP at 4M, 0 up to N=1000 — Appendix B) |
| A9c | `Sum(a*b, axis=0)` 2-D | 0 ULP vs `np.sum` | the axis path is sequential per column on both sides |
| A11 | `Greater(u8[2⁶³+1], i8[2⁶³−1])` | fused False, unfused False; NumPy **True** (`generate_umath.py:537` `qQ->?` loops) | shared ENGINE gap: mixed 64-bit compare promotes to f64 |
| A12 | `u8 + 18446744073709551615UL` | binds to `double` → f64 result; NumPy uint64 (wraps to 0) | literal spelling gap |
| A13 | all-0-d inputs; `Sum` over a `(0,3)` axis; `out=` aliasing a strided view; `out=` into a transposed target; F-layout preserved (`a.T*2` is F like NumPy) | OK | |
| A15 | int `Divide/Mod/FloorDivide` by 0 → `-inf / 0 / 0` | OK = NumPy | |
| A16 | `Sqrt/Exp/Mean(decimal)` | OK (decimal bridge) | |
| A19 | comparison written into `out=f8` | OK (bool same_kind→f8) | |
| A21 | `Sum(broadcast(1000×64) * b)` | fused 429 ULP off unfused | order-of-summation divergence over stride-0 chunks |
| A22 | `Max(a with NaN, b)` vs `np.maximum` | bit-equal | |
| A23 | `bool+1 → int64`, `bool*2.5 → f64` | OK = NumPy | |

Not expressible today (API gaps): tuple `axis`, `keepdims` on the flat reduction, `where=`,
`casting=`, `order=`, a `Cast`/`astype` node, comparison operators.

### 2.2 Support sweep — 15 dtypes × 18 expression shapes (section B)

Every cell computes except: `Negate(bool)` (NumPy raises too), `a&a` on Half/Single/Double/Decimal
(NumPy TypeError too), `Floor(complex)` / `a%3` on complex (NumPy no-loop too), and **`Min(complex)`**
(NumPy computes → gap A6c). Result dtypes follow NumPy for all 13 NumPy dtypes (bool→int64 sums,
uint→uint64, int8/uint8→float16 unary tiers, int16→float32, …). `Decimal` and `Char` print under
their NumPy proxy names (`float64`, `uint16`) — the arrays are genuinely decimal/char.

### 2.3 Fused vs unfused vs NumPy (section C; twin section C)

`fused` = `np.evaluate`, `unfused` = the equivalent NumSharp `np.*` chain, `NPY/NS` = NumPy ÷ fused.
The 1K column is dominated by the fixed cost (§2.4) and is noisy at the µs scale; treat it as
"loses/wins", not as a ratio.

| Expression | 100K fused ms | 100K unfused/fused | 100K NPY/NS | 4M fused ms | 4M unfused/fused | 4M NPY/NS |
|---|---:|---:|---:|---:|---:|---:|
| `a*b+c` f64 | 0.050 | 1.83 | 1.45 | 5.07 | 2.25 | **2.59** |
| `(a-b)/(a+b)` f64 | 0.041 | 3.50 | 11.9¹ | 4.63 | 4.28 | **4.30** |
| `sqrt(a*a+b*b)` f64 | 0.056 | 2.65 | 11.7¹ | 5.58 | 3.93 | **4.45** |
| `where(a>b,a,b)` f64 | 0.065 | **0.82** | **0.89** | 4.89 | 1.33 | 1.92 |
| `maximum(a,b)` f64 | 0.066 | **0.57** | **0.46** | 5.10 | **0.69** | 1.22 |
| `a>0.5` (bool out) | 0.039 | 0.95 | **0.16** | 2.46 | 1.17 | **0.71** |
| `(a>0.2)&(b<0.8)` | 0.071 | 1.14 | **0.39** | 4.60 | 1.42 | **0.93** |
| leaky relu `where(a>.5,a,.01a)` | 0.036 | 3.66 | 9.5¹ | 3.23 | 3.95 | **4.72** |
| `exp(a)*b` f64 | 0.277 | 1.16 | 1.03 | 11.66 | 1.83 | 1.66 |
| `exp(af)*bf` f32 | 0.054 | 1.36 | 1.20 | 2.32 | 2.32 | **3.03** |
| `sin(a)*cos(a)` f64 | 0.401 | 1.27 | 1.45 | 17.00 | 1.96 | 1.53 |
| `i4*2+f8` mixed | 0.041 | 1.47 | 1.83 | 4.06 | 1.25 | 2.52 |
| `i8*i8+1` int64 | 0.047 | 1.18 | 1.11 | 3.31 | 3.69 | **3.65** |
| `f2*f2+f2` half | 1.098 | **0.15** | **0.51** | 44.66 | **0.18** | **0.55** |
| `abs(a)` f64 (unary only) | 0.025 | 1.75 | **0.65** | 3.53 | **0.87** | 1.59 |
| strided `a[::2]*b[::2]+c[::2]` | 0.047 | 1.28 | 1.08 | 4.11 | 1.50 | 1.94 |
| `sum(a*b)` f64 | 0.022 | 2.85 | 2.16 | 2.68 | 2.02 | **3.27** |
| `sum(af*bf)` f32 | 0.027 | **0.60** | **0.84** | 1.49 | 1.72 | 2.80 |
| `mean((a-b)²)` f64 | 0.023 | 5.94 | 17¹ | 2.88 | 9.28 | **7.91** |
| `max(a*b)` f64 | 0.087 | **0.67** | **0.44** | 4.14 | 1.24 | 1.98 |
| `sum(a2*b2, axis=0)` | 0.040 | 2.09 | 0.98 | 3.29 | 1.54 | 2.35 |
| `sum(a2*b2, axis=1)` | 0.024 | 2.71 | 1.86 | 2.92 | 2.41 | 3.03 |

¹ NumPy's 100K cells for `(a-b)/(a+b)` (0.48 ms), `sqrt(a*a+b*b)` (0.66 ms), leaky relu (0.34 ms)
and `mean((a-b)²)` (0.40 ms) are anomalously slow in this run (its `a*b+c` at the same size is
0.072 ms); re-measure before quoting those four ratios. The 4M column is consistent.

Reading: fusion wins decisively wherever the tree is homogeneous float arithmetic (2.3–4.5× NumPy,
2–9× NumSharp's own chain). It **loses** — to NumPy and often to the unfused chain — wherever the
scalar path is forced: comparisons (`a>0.5` 0.16× at 100K), `Where`/`Min/Max` (0.46–0.89×), Half
(0.15× unfused), unary-only trees (0.65–0.87×), and reductions that fold in scalar code
(`max(a*b)` 0.44×, `sum f32` 0.60× the SIMD pairwise `np.sum`).

### 2.4 Fixed cost per call, n = 8 (section D; twin section D)

| Call | NumSharp ns | managed B/call | NumPy ns |
|---|---:|---:|---:|
| `np.evaluate((NDExpr)a*b+c)` (tree rebuilt per call) | 1008 | 3048 | — |
| `np.evaluate(expr)` (prebuilt tree) | 1006 | 2896 | — |
| `np.evaluate(expr, out=)` | 831 | 2432 | — |
| unfused `a*b+c` | 359 | 928 | 524 |
| unfused `np.multiply/np.add(out=)` two passes | 345 | 400 | 515 |
| `np.evaluate(Sum(a*b))` | 775 | 2648 | 1717 (`np.sum(a*b)`) |
| unfused `np.sum(a*b)` | 363 | 1048 | |
| `np.evaluate(Where(a>b,a,b))` | 935 | 2968 | 934 (`np.where`) |
| unfused `np.where(a>b,a,b)` | 427 | 1040 | |

Rebuilding the tree costs nothing measurable; the ~1.0 µs is **binding + typing + string cache-key +
broadcast + iterator + result allocation**, with ~3 KB of garbage (bound-tree clone, `Dictionary`
side table, `StringBuilder` key, `long[]`/flag arrays, two `Shape`s). First compile of a never-seen
tree: **1.2 ms** (DynamicMethod JIT; 2.4 ms for a 25-node Horner polynomial); every distinct literal
value compiles a new kernel (section E: `a*b+2.0` and `a*b+3.0` are two kernels).

### 2.5 2-D layouts, `a*b+c` 2000×2000 f64 (section F; NumPy from `fusion_results.md`)

| Layout | fused ms | unfused ms | unfused/fused | NumPy ms | NPY/NS |
|---|---:|---:|---:|---:|---:|
| C | 4.61 | 8.82 | 1.91 | 12.83 | 2.78 |
| F (all inputs F, result allocated F) | 4.75 | 10.07 | 2.12 | 12.50 | 2.63 |
| T (transposed views, result F) | 3.91 | 11.29 | 2.89 | 12.59 | 3.22 |
| `[:, ::2]` (inner stride 2 → gather) | 3.83 | 6.30 | 1.65 | 7.87 | 2.06 |
| `[::2, :]` | 1.85 | 4.75 | 2.57 | — | — |
| broadcast rows (stride 0) | 2.31 | 7.17 | 3.10 | 12.19 | 5.27 |

The F/T "cliff" the committed `fusion_results.md` shows (11.8 ms) does **not** reproduce on a pinned
core: the F-aware allocation/iteration (`AreAllInputsStrictFContig`) works, and the result layout
matches NumPy's K-order rule (F inputs → F result) on both fused and unfused paths.

---

## 3. Gap analysis

### 3.1 Correctness and parity (ranked)

| # | Gap | Where | Fix shape |
|---|---|---|---|
| G1 | `SByte` missing from `EmitPushZeroPublic` → `Where`/`LogicalNot`/zero-test over int8 throws | `NDExpr.cs:948` | add SByte (and audit every dtype switch in NDExpr for the 15-type rule); delete the `MisalignedRegistry` W1-E excuse so the fuzz gate verifies it |
| G2 | `Abs(complex)` → must be **float64** (`np.absolute` `D->d` loop) | `NDExpr.Typing.cs:493` (`UnaryNode.InferType`), `NDExpr.cs:670` (emit) | Abs on Complex resolves to Double; emit `Complex.Abs` (= `hypot`, the NumPy-exact `npy_cabs`) |
| G3 | fused float reductions not NumPy-exact: 4-acc fold order; f16/f32 `Sum/Prod/Mean` accumulate in f64; `Prod` folded in 4 lanes where NumPy multiplies sequentially; broadcast/strided chunks change the order | `NDExpr.Evaluate.cs:499` | Phase 2 — pairwise schedule for `add`, sequential for `multiply`, result-dtype accumulation (f16 mean via f32 per `_methods.py:128`) |
| G4 | `Min/Max(complex)` reductions throw | `ResolveReduceResultType` | lexicographic complex comparator (the engine's `np.min(complex)` policy) |
| G5 | negative integer exponent in an exponent ARRAY silently wraps | `BinaryNode.InferType` checks literals only | kernel-side sign scan of the exponent operand (OR into an aux flag; host raises NumPy's text after the pass) — cheap, and closes the documented divergence |
| G6 | literal spellings: no `bool/uint/ulong/Half/decimal/Complex` `Const`; `ulong` binds to `double` | `NDExpr.cs:174`, `NDExpr.Evaluate.cs:77` | typed `ConstNode` carrying the CLR value + weak/strong kind per the house NEP50 mapping (`np.r_`: bool/ints/float/double/Complex weak; char/Half/decimal strong) |
| G7 | `int64` vs `uint64` comparison promotes to f64 → inexact past 2⁵³ (NumPy `qQ->?`) | `NDExprTypeRules.PromoteStrong` for comparisons **and** the engine's comparison ufuncs | shared fix: comparison nodes compare exactly for the (q,Q) pair (sign test + unsigned compare) |
| G8 | `np.evaluate` resolves `BackendFactory.GetEngine()`, not the operands' engine | `np.evaluate.cs:43` | dispatch on `operands[0].TensorEngine` (ARCHITECTURE.md P2) |
| G9 | legacy `Compile` (all-nodes-at-output-dtype) still public behind `ExecuteExpression` | `NDExpr.cs:93` | keep for Tier-3C users; document as "advanced, not NumPy-typed" |
| G10 | `Call` rejects SByte/Half/Complex signatures | `CallNode.IsSupported` | extend the 12→15 dtype map (Half/Complex via the existing convert emitters) |

### 3.2 Coverage vs the np.\* surface NumSharp already ships

Elementwise ufuncs NumSharp implements (CLAUDE.md "Math — Arithmetic / Bitwise / Comparison &
Logic") that have **no NDExpr node**:

| Family | Missing nodes | Notes |
|---|---|---|
| Unary | `Positive`, `Conjugate`/`Conj`, `Real`, `Imag`, `Angle`, `IsPosInf`, `IsNegInf`, `Rint` (alias of Round), `Round(decimals)` | `UnaryOp` already has Positive/Conjugate/IsPosInf/IsNegInf and vector emits for Positive |
| Unary | `Cast(dtype)` / `AsType` | the single most requested composition (`(a*b).astype(f4) + c`); also the only way to pin a loop dtype like the ufunc `dtype=` keyword |
| Binary | `Maximum/Minimum` aliases, `FMax/FMin`, `CopySign`, `NextAfter`, `LogAddExp`, `LogAddExp2`, `LeftShift`, `RightShift` | all exist as `BinaryOp` with scalar emitters (`EmitScalarOperation`) and, for max/min, vector emitters |
| Logical | `LogicalAnd/Or/Xor` with NumPy's nonzero-test semantics on non-bool inputs (today `&` is bitwise) | bool result; `(a!=0)&(b!=0)` |
| N-ary | `Select(condlist, choicelist, default)`, `Clip(x, lo?, hi?)` with `None` bounds | `Clamp` exists for the two-sided case |
| Reductions | `Any`, `All`, `ArgMax`, `ArgMin`, `Std`, `Var` (ddof), `NanSum/NanProd/NanMin/NanMax/NanMean`, `CountNonzero`, `Ptp`, weighted `Average` (two accumulators) | plus tuple `axis`, `axis=None keepdims`, `out=` on axis reductions without the copyto detour |
| Keywords | `where=` (mask), `casting=`, `order=`, `dtype=` (the loop dtype) | `where=` composes for free once evaluate uses the masked `ExecuteElementWise` entry |
| Operators | `< > <= >=` returning `NDExpr` (legal C#; only `==`/`!=` are hazardous) ; unary `+` | today `NDExpr.Greater(a, 2.0)` |
| Literals | see G6 | |

### 3.3 Performance (ranked by measured loss)

| # | Regime | Evidence (§2.3/§2.4) | Root cause |
|---|---|---|---|
| P1 | any tree with a comparison / `Where` / `Min/Max` / logical / predicate node | `a>0.5` 0.16× NumPy @100K, 0.71× @4M; `maximum` 0.46×/0.69× unfused; `where` 0.82× unfused @100K | Boolean-typed nodes break the "one dtype" SIMD gate → whole tree scalar |
| P2 | Half operands | `f2*f2+f2` 0.15–0.18× unfused, 0.51–0.55× NumPy | `CanUseSimd(Half)` false → scalar `Half` operator calls per node; the unfused chain has widen-compute-narrow SIMD |
| P3 | fixed cost | 1.0 µs + 3 KB vs 0.36 µs + 0.9 KB unfused, 0.52 µs NumPy | bind clone + `Dictionary` typing + string key + broadcast + iterator per call |
| P4 | fused reductions | `max(a*b)` 0.44× NumPy @100K; `sum f32` 0.60× unfused; f64 sum 4 ULP off | scalar 4-accumulator fold; no SIMD lanes; not NumPy's pairwise |
| P5 | mixed dtypes | `i4*2+f8` 2.5× NumPy but scalar (1.25× unfused @4M) | one-dtype SIMD gate |
| P6 | unary-only trees | `abs(a)` 0.87× unfused @4M, 0.65× NumPy @100K | iterator + generic shell vs the direct unary kernel; no fusion benefit to amortize |
| P7 | f64 transcendental chains | `exp(a)*b` 1.66×, `sin*cos` 1.53× — parity-class | CRT-bound on both sides (the NumSharp float64 loops are the same `ucrtbase` calls); only the f32 ports vectorize |
| P8 | no CSE / constant folding | `exp(x)/(1+exp(x))` evaluates `exp` twice | tree is emitted verbatim |
| P9 | single-threaded | numexpr's headline lever is absent | house rule: threading is opt-in only (`np.multithreading`, `TensorEngine.Threading`) |

### 3.4 API and ergonomics

No comparison operators; no `Cast`; literals limited; reductions root-only (numexpr has the same
rule — keep it, but make the two-call idiom cheap); no compiled-expression object (numexpr's
`NumExpr(...)`/`re_evaluate`); no string front-end; no `where=`; no way to inspect the plan (which
nodes vectorized, resolved dtypes, operand dedup, iteration shape); literals baked into the kernel
(a parameter sweep compiles one kernel per value).

---

## 4. The plan

Order of execution and why: **P0 → P1 → P3 → P2 → P4 → P5 → P6**. Correctness and the oracle gate
first (everything after is measured against it); the mask/select SIMD is the largest measured loss
and unlocks the numexpr-style workloads (`where`, relu, masks); the fixed cost is cheap to fix and
decides every small-array call; reduction parity is deep but bounded; node coverage is broad and
mechanical once the typed vector emission exists; Half/mixed-width SIMD reuse the cast-kernel
converters; the optimizer passes, threading and the string front-end are independent add-ons.

### Phase 0 — Correctness, parity contract, and the differential gate

0.1 **G1** — `EmitPushZeroPublic` gains `SByte`; sweep every `switch (NPTypeCode)` in
`NDExpr*.cs` against the 15-type rule (`ConstNode.EmitLoadTyped`, `WriteMinMaxIdentity`, `EmitFold`,
`WriteOne`, `WriteMeanOfEmpty`). Remove the `MisalignedRegistry` W1-E excuse. Gate: the §2.2 sweep
as a unit test (15 dtypes × every node kind) — a "no cell throws unless NumPy has no loop" pin.

0.2 **G2** — `Abs(Complex)` types to `Double` and emits `Complex.Abs`; audit the other complex
unary results against NumPy's loop table (`sign` complex→complex ✓, `isnan` → bool ✓, `floor`/`mod`
no-loop ✓).

0.3 **G5** — exponent-array sign check: the integer `Power` kernel ORs `(exp < 0)` into an aux flag
per element; the host throws NumPy's verbatim `ValueError` text after the pass (NumPy checks per
element inside its loop; a pre-pass over the exponent operand alone is an alternative when it is a
distinct operand).

0.4 **G4** — complex `Min/Max` reductions via the engine's lexicographic complex comparator (same
NaN policy as `np.min(complex)`).

0.5 **G6** — literal spellings: `ConstNode` stores the boxed CLR value + `NDExprWeak` kind;
overloads/implicit conversions for `bool, byte, sbyte, short, ushort, int, uint, long, ulong, float,
double, Complex` (weak per the `np.r_` mapping) and `Half, decimal, char` (strong). A `ulong` literal
above `long.MaxValue` adopts `uint64` like NumPy. `CheckIntLiteralFits` stays the overflow gate.

0.6 **G7** — exact `int64`/`uint64` comparison (sign test + unsigned compare) in `ComparisonNode`,
and the same in the engine's comparison ufuncs (shared fix; NumPy `generate_umath.py:537-548`).

0.7 **G8** — `np.evaluate` dispatches on the first operand's engine.

0.8 **The `evaluate.jsonl` differential tier** (the quality backbone for every later phase).
NumPy has no fusion, so the oracle is the **unfused NumPy chain evaluated node by node** — exactly
the contract `np.evaluate` claims. Encode the tree in `params`, e.g.
`{"expr": "add(mul(in0,in1),lit_i:2)"}` (prefix grammar over the node catalog; `lit_i`/`lit_f`
mark weak literals), operands as usual from `layout_catalog.py` (26 single + 9 pair layouts), all
dtypes, weak literals, comparisons/where/minmax, reductions (flat + axis + keepdims), `out=` (with
the whole `out` base buffer recorded, as `out_where.jsonl` does), `where=` once it exists. Generator
side: a tiny evaluator that maps each node to its ufunc (`np.add`, `np.where`, `np.maximum`,
`np.add.reduce(..., dtype=<NumPy's reduce dtype>)`); C# side: `OpRegistry` builds the `NDExpr`.
Excusals: float `Sum/Prod/Mean` until Phase 2 lands (then the excuse is deleted, not widened).
Also: a **metamorphic self-consistency sweep** (no Python) — random trees × dtypes × layouts,
fused vs the unfused NumSharp chain, bit-exact for non-reduction trees (the §2.2 sweep generalized),
in the style of the 16,831-check 2-D-kernel probe.

Acceptance: A1/A2/A3/A6c/A5 in the probe read OK; `evaluate.jsonl` green in `FuzzMatrix`;
`OracleSurfaceCoverageTests` moves `evaluate` out of `SiblingOwned`.

### Phase 1 — Typed vector emission ("vector v2"): masks, select, min/max, logical, bool I/O

Design. Each node emits a vector of **its own** resolved type; a `Boolean`-typed node emits a
**lane mask** of the width of the type it was compared at (all-ones/zero lanes), never bytes.
Conversions live at node edges exactly as in the scalar path:

- `ComparisonNode`: `Vector256.Equals/LessThan/GreaterThan/…` on the common type; ordered float
  compares give NumPy's NaN semantics (NaN → false; `NotEqual = ~Equals` → NaN → true); unsigned
  integer compares through the existing unsigned-aware helpers.
- `WhereNode`: `ConditionalSelect(mask, a, b)`; a non-bool condition (int/float array) becomes a
  mask via `!= Zero`.
- `MinMaxNode`: `EmitVectorMinOrMax(propagateNaN: true)` (`DirectILKernelGenerator.cs`, the
  fuzz-validated `np.maximum` body) — nothing new to prove.
- Logical/bitwise on masks: `And/Or/Xor/OnesComplement`; `LogicalNot` = `Equals(x, Zero)`.
- Mask feeding arithmetic (`a * (a > 2)`): `mask & One` (exact 1.0/0.0 lanes — NOT a multiply by a
  0/1 float, which would differ at inf/NaN; the scalar path converts I4 0/1 → dtype, identical).
- Bool **inputs** (bool arrays in the tree): byte lanes widened to the compared width by the
  existing `EmitInlineMaskCreation` (the `np.where`/`np.select` kernels' bool→lane expansion).
- Bool **output** (root is a comparison/predicate): pack lanes to bytes with the comparison kernel's
  existing mask→byte pack (`DirectILKernelGenerator.Comparison.cs`); the bool store is 1 byte/lane.
- Predicates `IsNaN/IsInf/IsFinite`: `Vector.IsNaN` = `!Equals(x,x)`; `IsInf` = `Abs(x) == +inf`;
  `IsFinite` = the complement — float lanes only (ints are constant masks).

Shell change: `CompileInnerLoop`'s gate becomes "all non-bool operands share one SIMD dtype W and
every node type ∈ {W, Boolean}"; bool operands are loaded as bytes → lanes, a bool output is stored
as packed bytes. That is one new operand-load/store kind in `EmitSimdContigLoop`/gather/SL-SR, with
the existing scalar tail unchanged (the scalar path already handles everything).

Acceptance (probe §2.3 rows): `a>0.5` ≥ 1.0× NumPy at 100K and 4M (target ≥ 1.5×); `where(a>b,a,b)`
≥ 1.5× unfused; `maximum(a,b)` ≥ 1.0× `np.maximum`; `(a>0.2)&(b<0.8)` ≥ 1.5× NumPy; leaky relu ≥ 5×
NumPy; bit-exact against the Phase-0 oracle across all layouts.

### Phase 2 — Reductions: NumPy-exact, SIMD, and the missing kinds

2.1 **Parity contract**: `evaluate(Sum(expr)) == np.sum(materialized expr)` **bit-for-bit** for
f32/f64/complex128. NumPy's `add.reduce` over a contiguous 1-D temp is ONE `pairwise_sum(n)`
(`loops_utils.h.src:81`, `PW_BLOCKSIZE 128`: blocks of 128 with eight interleaved accumulators
combined as `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`, recursion splitting at `n/2` rounded down to a
multiple of 8). Two implementation options, both exact:
   - **Streaming pairwise schedule** (recommended): the recursion tree depends on `n` only;
     precompute leaf boundaries (O(n/128) entries) and fold the value stream post-order with a
     small stack, so chunk boundaries (strided/broadcast inputs → many EXTERNAL_LOOP chunks) do not
     change the result. The eight accumulators are exactly eight lanes (`Vector256<float>`, or two
     `Vector256<double>`) — the SIMD form falls out of the algorithm, as NumSharp's own
     `ILKernelGenerator.Reduction.Pairwise.cs` (`TryEmitPairwiseSumKernel`) already proves for the
     axis path.
   - **Scratch materialization** (first milestone / fallback): for `n ≤ 64K` evaluate the expression
     into a pooled contiguous scratch and call the existing pairwise kernel — exact and tiny.
   `Prod`: NumPy's `multiply.reduce` is a sequential `io1 *= in2` loop — the fused fold must be
   sequential (today's 4-lane fold is wrong for floats; correct for ints/bool/decimal by
   associativity). `Mean`: accumulate at the **result** dtype (f32 stays f32; f16 via f32 per
   `_methods.py:128-139`), divide once. `Min/Max`: any order for finite values, NaN propagates, ±0
   tie policy of the engine's `np.max` (first operand wins). Delete the Phase-0 excusal.
   Shared finding to hand over (out of scope here): NumSharp's **flat** `np.sum` f64 drifts from
   NumPy at large N (1 ULP @100K, 16 ULP @4M — Appendix B) because the flat contiguous reduction is
   a multi-accumulator SIMD fold, not pairwise; the streaming pairwise kernel could serve it too.

2.2 **SIMD folds**: pairwise lanes for `Sum/Mean`; `Vector.Min/Max` with NaN blend for `Min/Max`;
sequential-with-4-independent-streams is NOT allowed for float `Prod` (see above), but int/bool
prod may vectorize.

2.3 **Missing reduction kinds**: `Any/All` (early-exit), `ArgMax/ArgMin` (index tracking, NaN-first
like `np.argmax`), `Std/Var(ddof)` as a fused two-pass (`mean`, then `sum((x-m)²)/(n-ddof)`; NumPy's
`_var` does exactly `x - mean` at the loop dtype then `um.multiply(x, x)` then `umr_sum` — reproduce
op for op), `NanSum/NanProd/NanMin/NanMax/NanMean` (NumPy's `_replace_nan` composition ≡
`Where(IsNaN(x), identity, x)` at the tree level, so they are node rewrites, not kernels),
`CountNonzero`, `Ptp` (`max - min` in one pass, two accumulators), weighted `Average`
(`Sum(a*w)`/`Sum(w)` two accumulators, one pass — the `WeightedSum.cs` shape).

2.4 **Axis forms**: tuple `axis`, `axis=None` with `keepdims`, `out=` written directly (no copyto).
SIMD in the slab path (vector over the kept axis) and pinned path (pairwise lanes).

Acceptance: `Sum/Mean/Prod` bit-exact vs NumPy in `evaluate.jsonl` at N ∈ {7, 8, 127, 128, 129,
1000, 100003, 4M} and across layouts; `sum(af*bf)` ≥ 1.2× unfused at 100K; `max(a*b)` ≥ 1.0× NumPy
at 100K (target 2×); `mean((a-b)²)` keeps ≥ 7× NumPy at 4M.

### Phase 3 — Fixed cost: compiled expressions

3.1 `CompiledExpression` (`NDExpr.Compile()` → object with `Evaluate(params NDArray[])` and
`Evaluate(NDArray[] ops, NDArray out)`), and the same cache used **implicitly** by `np.evaluate`:
keyed per tree instance by the operand dtype signature (`NPTypeCode[]` packed to a `ulong` for ≤ 8
operands — the `InnerLoopKernelKey` idea), holding the bound tree, node types (array-indexed slots
assigned at bind time instead of a `Dictionary`), the kernel, the resolved output dtype and the
flags arrays. Second call of the same tree: no clone, no typing, no string.

3.2 Host path: identical-dims fast path (skip `Broadcast` + `ResolveUfuncIterationShape`, as the
`*UfuncInto` routes do); `Shape` reuse; result allocated `fillZeros:false` (already); iterator
construction is already ~93 ns after the block cache.

3.3 Literal-as-parameter: `NDExpr.Param("k")` (or a 0-d `NDArray` leaf) so a parameter sweep
evaluates one kernel with a broadcast scalar operand instead of compiling one kernel per constant
(section E).

Acceptance (probe §2.4): `np.evaluate(expr)` ≤ 450 ns and ≤ 600 B at n=8 (unfused chain 359 ns /
928 B; NumPy 524 ns); every 1K row in §2.3 fused ≥ unfused.

### Phase 4 — Coverage: nodes, keywords, operators

4.1 Unary nodes: `Positive`, `Conjugate`, `Real`, `Imag`, `Angle`, `IsPosInf`, `IsNegInf`, `Rint`,
`Round(decimals)` (NumPy's `multiply→rint→divide` composition, cast-error text naming `multiply`),
**`Cast(dtype)`** (unsafe `astype` semantics through the existing `EmitConvertTo`; in vector mode only
the exact widenings Phase 5 provides; otherwise scalar).
4.2 Binary nodes: `Maximum/Minimum` (aliases), `FMax/FMin`, `CopySign`, `NextAfter`, `LogAddExp`,
`LogAddExp2`, `LeftShift/RightShift`, `LogicalAnd/Or/Xor` (nonzero-test, bool result).
4.3 N-ary: `Select(conds, choices, default)` (lowered to a reverse `Where` chain — the fused
`np.select` kernel's semantics), `Clip(x, lo?, hi?)`.
4.4 Operators: `<  >  <=  >=` on `NDExpr` (return `NDExpr`; `NDExpr`×`NDArray`×scalar overload
triangle like the arithmetic ones); unary `+`; keep `Equal/NotEqual` as methods (overloading `==`
would hijack null checks).
4.5 Keywords on `np.evaluate`: `where=` (append the mask as the trailing ARRAYMASK operand and route
through the packed-key `ExecuteElementWise`, which also brings the masked 2-D block kernel);
`casting=` (validate the `out` cast rule, default same_kind); `order=` for the fresh result
(`K` default = today's F preservation; `C`/`F`/`A`); `dtype=` (pins the root loop dtype = an implicit
root `Cast` with the ufunc `dtype=` semantics).
4.6 Route `np.evaluate` through `ExecuteElementWise` (packed key) so narrow strided rows and masks
get the 2-D block kernels (§1.4 point 2).
4.7 Docs: rewrite the NDIter.md Tier 3C catalog (SIMD column per node per dtype) + the CLAUDE.md
"Fused Expressions" section from the probe tables; API reference pages for the new nodes.

Acceptance: every NumSharp elementwise ufunc has a node; `evaluate.jsonl` covers each new node across
dtypes/layouts; `where=` cases mirror `out_where.jsonl`'s base-buffer check.

### Phase 5 — Half and mixed-width SIMD

5.1 Half: widen-compute-narrow **per node** (`(Half)((float)a op (float)b)` is NumPy's npy_half
model, so per-node rounding is required and already what the scalar path does); the Giesen f16↔f32
converters from `Cast.Half.cs`/`Cast.ToHalf.cs` give bit-exact lanes. Target: `f2*f2+f2` ≥ 1.5×
unfused (today 0.15×), ≥ 2× NumPy.
5.2 Mixed widths ("lane groups"): the vector element count is `V / maxWidth`; narrower operands load
a partial vector and widen exactly (`int32/float32 → double` via `ConvertToVector256Double`,
`int8/16 → int32` via `Widen`, `float → double`); narrowing at edges only where exact (`double →
float` is a rounding NumPy also performs, so it is admitted; float→int stays scalar unless the cast
kernels' NumPy-exact vector converters are reused). Target: `i4*2+f8` ≥ 4× NumPy.

### Phase 6 — Optimizer passes and bigger levers

6.1 **CSE**: structural hash of subtrees (post-bind), identical subtrees emitted once into a local
(scalar and vector); `Where` branches hoisted only when both sides are cheap or shared (cost model =
node count). 6.2 **Constant folding** at typing time with weak kinds preserved. 6.3 **Unary-only
trees** delegate to the direct unary kernel route (`abs(a)` should equal `np.abs(a)`). 6.4 **Opt-in
threading** behind `np.multithreading` / `TensorEngine.Threading["NumSharp"]`: split the outer
iteration into slabs (operand slicing — `ResetToIterIndexRange` + EXTERNAL_LOOP does not clamp, a
documented `NDIterRef` defect), reductions merged per slab in a fixed order so results stay
deterministic (pairwise schedule per slab + fixed merge). 6.5 **String front-end**
`np.evaluate("a*b + sin(c)", ("a", a), …)`: a small Pratt parser to `NDExpr` with NumPy/numexpr
function names — numexpr-porting convenience, self-contained. 6.6 **Diagnostics**:
`NDExpr.Explain(inputs)` → resolved dtype per node, vectorized-or-scalar per node with the reason,
operand dedup, cache key, iteration shape/chunking; `GeneratedDelegates` hooks already count kernels.

---

## 5. Testing and gates per phase

- **Unit pins** in `NDEvaluateTests` for every semantic decision (NumPy-probed values, verbatim
  error texts), plus the §2.2 sweep as a dtype × node non-crash pin.
- **`evaluate.jsonl`** (Phase 0.8) — bit-exact vs NumPy's node-by-node chain; grows with each phase;
  excusals only in `MisalignedRegistry` with a phase-tagged reason and a deletion date.
- **Metamorphic sweep** (no Python): random trees, fused vs unfused NumSharp, all dtypes × layouts.
- **Benchmark**: promote `expr_probe.cs` sections C/D/F into the `benchmark/fusion` sheet (the
  current `evaluate_bench.cs` covers 6 expressions; the probe's 22 + fixed cost + layouts is the
  regression surface), with NPY/NS columns from `numpy_twins.py`; every phase's acceptance line is a
  row in that sheet.
- **Docs**: NDIter.md Tier 3C + fused-kernels sections and CLAUDE.md regenerated from the same
  tables (the current NDIter.md still says `Round/Truncate` are scalar-only and comparisons are
  "0/1 at output dtype" — both stale).

## 6. Non-goals and risks

- Nested reductions stay rejected (numexpr rule; two `np.evaluate` calls). Reductions as scalar
  operands of an outer tree are Phase-6-optional at most.
- No object/string dtypes; no FP-status flags/warnings (NumSharp models none).
- Legacy `Compile` keeps its semantics for Tier-3C callers.
- The pairwise schedule must be verified against NumPy at the split boundaries (n = 8·k ± 1, 128,
  129, 256+7, …) — the same size grid Appendix B uses; getting `n2 -= n2 % 8` wrong is invisible at
  small N.
- Vector masks must never be converted to numeric by multiply (inf·0 = NaN) — `mask & One` only.
- Threading is opt-in and must keep results bit-identical to single-threaded (fixed merge order).

---

## Appendix A — raw probe output (2026-09-07, i9-13900K P-core, V256, .NET 10, NumPy 2.4.2)

Sections A–F of `expr_probe.cs` and C/D of `numpy_twins.py`; every table in §2 is derived from
these rows.

```
[A1] Where(int8 cond, f8, f8)      -> THROWS NotSupportedException: Zero-push unsupported for SByte
[A2] LogicalNot(int8)              -> THROWS NotSupportedException: Zero-push unsupported for SByte
[A3] Abs(complex128)               -> OK dtype=Complex r[0]=<5; 0>        (NumPy: float64 5.0)
[A5] Power(i4, i4 with negatives)  -> OK(silently) dtype=Int32 r[0]=1     (NumPy: ValueError)
[A6c] Min(complex128)              -> THROWS NotSupportedException          (NumPy: (nan+0j))
[A8] Sum(af*bf) f32 N=100003       -> fused=0x47429D72 unfused=0x47429D75 ulp=3   (NumPy 0x47429D72)
[A9] Mean(af*bf)                   -> fused=0x3EFF1401 unfused=0x3EFF1405 ulp=4   (NumPy 0x3EFF1401)
[A9b] Sum(a*b) f64 N=100003        -> fused=...D780 unfused=...D783             (NumPy ...D784)
[A11] Greater(u8[2^63+1], i8[2^63-1]) -> fused=False unfused=False           (NumPy: True)
[A12] u8 + 18446744073709551615UL  -> dtype=Double                           (NumPy: uint64 [0])
[A21] Sum(broadcast_to(x,(1000,64)) * b) -> fused vs unfused ulp=429

D. fixed cost n=8 (ns / B per call):  evaluate 1008/3048 · prebuilt 1006/2896 · out= 831/2432 ·
   unfused a*b+c 359/928 · two-pass out= 345/400 · evaluate(Sum) 775/2648 · np.sum 363/1048 ·
   evaluate(Where) 935/2968 · np.where 427/1040.  NumPy: a*b+c 524 · two-pass 515 · sum 1717 · where 934.
   kernel compile: ~1.2 ms per new tree; horner deg-12 2.4 ms.
E. kernels 310 → a*b+c 310 (cached from tests) → other arrays same shape 310 → f4 operand 311 → two literal variants 313.
```

## Appendix B — `np.sum(a*b)` raw bits by N (NumPy vs NumSharp unfused vs fused), same data

```
n=  100003  NumPy f64 0x40E853AE39BDD784  unfused 0x…D783 (1 ULP)  fused 0x…D780 (4 ULP)   f32: NumPy 47429D72 unfused 47429D75 fused 47429D72
n=  100000  NumPy     0x40E85390C1A7C716  unfused 0x…C715 (1 ULP)  fused 0x…C712 (4 ULP)   f32: NumPy 47429C86 unfused 47429C89 fused 47429C86
n= 4000000  NumPy     0x413E7842AC5FDFF0  unfused 0x…E000 (16 ULP) fused 0x…E020 (48 ULP)  f32: NumPy 49F3C215 unfused 49F3C216 fused 49F3C215
n=    1000  NumPy     0x407ED90684FBAE96  unfused = NumPy           fused 0x…AE97 (1 ULP)   f32 all equal
n=     129  NumPy     0x4022AD5CB92AFE24  unfused = NumPy           fused 0x…FE23 (1 ULP)   f32 all equal
n=     128  NumPy     0x40224F2CA1AAE204  unfused = NumPy           fused = NumPy           f32 all equal
n=       8  NumPy     0x3F8F6B8040226706  unfused = NumPy           fused 0x…6705 (1 ULP)
n=       7  NumPy     0x3F875B8EEF4162E8  unfused = NumPy           fused 0x…62E7 (1 ULP)
```

## Appendix C — pointers

- Probe: `benchmark/fusion/probes/expr_probe.cs` (`[ABCDEF]` sections) + `numpy_twins.py` (`[ACD]`).
- NumPy references: `refs/numpy/numpy/_core/src/umath/loops_utils.h.src:63,81` (pairwise sum),
  `refs/numpy/numpy/_core/code_generators/generate_umath.py:537-548` (`qQ`/`Qq` comparison loops),
  `refs/numpy/numpy/_core/_methods.py:118-139` (`_mean` float16 → float32 intermediate).
- Reusable NumSharp pieces: `EmitVectorMinOrMax` / `EmitVectorFMinOrFMax` / `EmitBoolVectorLogicalOp`
  (`DirectILKernelGenerator.cs`), the comparison kernel's mask pack (`Comparison.cs`),
  `EmitInlineMaskCreation` (`Where.cs`/`Select.cs`), `TryEmitPairwiseSumKernel`
  (`ILKernelGenerator.Reduction.Pairwise.cs`), the Giesen Half converters (`Cast.Half.cs`,
  `Cast.ToHalf.cs`), `Is2DElementwiseShape`/`TryExecute2DMasked` (`NDIter.Execution.Custom.cs`),
  `TensorEngine.Threading` (`TensorEngine.Threading.cs`).
