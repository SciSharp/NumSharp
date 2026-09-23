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

> **Beyond this plan:** the capabilities NDExpr still cannot express — multi-output passes,
> reductions inside trees, scans/recurrences/shifts, gather, vectorizable `Call`, macro nodes,
> fixed-cost dispatch, cache management, AOT, the public tree API and an ONNX export — are planned
> separately in `docs/plans/ndexpr-capabilities.md` (C1–C14, ordered by value ÷ effort).

### 0.1 Landed status (2026-09-14, branch `exprs`; P0 → P1 → P3 → P2 → P4 done, **P5 done (5.1 + 5.2)**, **P6.3 done**; the rest of P6 open)

| Phase | Status | Commit | What landed / evidence |
|---|---|---|---|
| **P0** correctness + oracle | landed | `53602da9` | G1–G8 + G10 fixed (int8 zero-push, `Abs(complex)→float64`, exponent-array sign check, complex `Min/Max`, typed literals, exact `int64`/`uint64` compares, engine dispatch, `Call` dtype map). **`evaluate.jsonl`** tier: 14,702 cases, floor 11,800, green; `MisalignedRegistry` E1–E5. **E1** (float `Sum/Prod/Mean` 4-accumulator fold ≤16 ULP, complex ≤64 ULP of magnitude) and **E5** (complex min/max NaN identity on negative-stride views — NumPy reduces in logical order, the KEEPORDER iterator in memory order) are *pending Phase 2* and must be deleted with it, not widened. |
| **P1** typed vector emission | landed | `939d0636` | "Vector v2": one lane dtype W per kernel, Boolean-typed nodes as lane masks (`NDExpr.Vector.cs`), the fused inner-loop shell with bool operand expansion / packed bool output / hoisted stride-0 operands / AVX2 gather (`DirectILKernelGenerator.InnerLoop.Fused.cs`), `NDExpr.ForceScalar` test hook + the 47-tree × 8-lane × 8-layout vector-vs-scalar metamorphic sweep (`NDEvaluateVectorTests`, 11). Measured NPY/NS at 100K / 4M: `a>0.5` 0.65 / 1.26–1.45 (was 0.16 / 0.71), `where(a>b,a,b)` ~1.0 / 2.6, `maximum(a,b)` 0.9 / 1.8, `(a>0.2)&(b<0.8)` 1.26 / 1.85, leaky relu 17.6 / 5.2, f32 `af>0.5` 8× the unfused engine compare. |
| **P3** fixed cost | landed | `feat(evaluate): NDExpr Phase 3 — per-root compiled program, CompiledExpression handle, N-ary input broadcast` | `NDExprProgram` per-root cache (`NDExpr.Program.cs`), identical-dims fast path, single-allocation N-ary input broadcast with NumPy's every-operand error text, `NDExpr.Compile()` / `Compile(params NPTypeCode[])` → `CompiledExpression` (strict signature), the 0-d-operand parameter form pinned (`NDEvaluateProgramTests`, 14). Numbers below. |
| **6.1** structural program cache | landed | (2026-09-13 landing) | `NDExprProgramCache` + `NDExpr.Structure.cs`: a tree rebuilt per call finds its program by structural hash + node-by-node verification — rebuilt `a*b+c` 995 → **464–508 ns / 864 B** at n = 8, fused ≥ unfused on **13/15** rows at 1K (was 5/15). `Call` slot identity fixed. Numbers under "Both residuals closed" below. |
| **3.3** parameters (0-d inputs hoisted into the kernel aux block) | landed | (2026-09-13 landing) | `NDExpr.Params.cs`: a 0-d input is a kernel parameter, not a stride-0 operand — `a*b+k` 681 → **388 ns**, zero per-element cost, one kernel per structure (the windows compile 0 kernels over 20 distinct M). `NDEvaluateParamTests` (12). |
| P2 reductions (NumPy-exact pairwise, SIMD folds, Any/All/Arg/Std/Var/Nan*/Ptp/Average, tuple axis) | **M1 + M2 landed**; **M4a landed** (Any/All/CountNonzero/NanSum/NanProd); **M4b landed** (Ptp/NanMin/NanMax); **M4c-index landed** (ArgMax/ArgMin); **M4c-summation landed** (NanMean/Var/Std); **M4c-average landed** (weighted Average); **M5 reachable scope landed** (`axis=None`+`keepdims` + direct `out=`; tuple axis deferred); M3 open — **the reduction KINDS + axis-forms surface is now complete** | M1 `7ac2c7b5`, M2 `bb02d212`, M4a `9d04b1a4`, M4b `417a448a`, M4c-index `5862547c`, M4c-summation `03ae07c2`, M4c-average `cb6ab165`, M5 (this landing) | **M1** — flat float32/float64/complex Sum·Mean + float Prod bit-exact (materialize the child, reduce with the pairwise/sequential kernel np.sum·np.prod use). **M2** — the same for **axis** Sum·Mean·Prod when the materialized child is **C-contiguous** (`EvaluateAxisReduce` diverts to `ExactAxisSum`, which rides the engine's own IL pairwise axis add.reduce, and `SequentialAxisProd`, a coordinate-order fold = np.multiply.reduce). E1 now excuses only Half (no f16 pairwise), complex Prod (npy_cmul FMA gap #12) and the strict-F-contiguous axis corner. **M4a** — the M4 kinds that ride the existing exact reductions with no new kernel: `Any`/`All` are two new bool ReduceKinds (logical OR/AND fold — associative+idempotent so bit-exact through the 4-accumulator unroll — with `False`/`True` identities, so an empty input is `False`/`True` where max/min would raise), and `CountNonzero`/`NanSum`/`NanProd` are pure factory rewrites (`Sum(x!=0)` / `Sum(Where(IsNaN(x),0,x))` / `Prod(Where(IsNaN(x),1,x))`), so they inherit flat+axis+M1/M2-bit-exactness for free. **M4b** — the ORDER-INDEPENDENT range / NaN-aware min-max kinds `Ptp`/`NanMin`/`NanMax`: three new ReduceKinds the host DELEGATES (`EvaluateDelegatingReduce` materializes the child once — the M1/M2 route — then reduces it through the already-NumPy-exact engine reductions `np.ptp`/`np.nanmin`/`np.nanmax`). Because a min, a max and their difference are the same value in ANY visiting order, they are bit-exact at every dtype with **NO E1 excuse** (a regression is red), carry NumPy's type-awareness for free (int child → plain min/max; float child → NaN-skip / all-NaN→NaN; ptp preserves dtype so integer overflow WRAPS) and reach the fold kernel never (`EmitFold` untouched). **M4c-index** — the int64 INDEX kinds `ArgMax`/`ArgMin`: two more ReduceKinds on the SAME `EvaluateDelegatingReduce` seam (materialize the child, then `np.argmax`/`np.argmin`), returning **int64** (an index, dtype-independent of the child) rather than a child-typed value. Their index is not value-order-independent — it depends on the C-order tie/NaN rule (first maximum / first NaN wins) — but the materialized child is exactly the fresh C-contiguous buffer NumPy's own argmax reduces, so they too are bit-exact with **NO E1 excuse**. **M4c-summation** — the SUMMATION kinds `NanMean`/`Var`/`Std`: three new ReduceKinds host-computed (new `EvaluateStatReduce`) over the materialized child with the SAME NumPy-exact pairwise sum the M1/M2 diverts use. They CANNOT delegate (the engine's own np.nanmean/np.var drift through its multi-accumulator flat sum) and are NOT tree rewrites (nanmean's divisor is a per-slab non-NaN count; var/std are two-pass with a ddof divisor), so the host reproduces NumPy's `nanmean`/`_var` op for op: `NanMean = pairwise_sum(NaN→0)/count`, `Var = pairwise_sum((x−mean)²)/max(N−ddof,0)`, `Std = sqrt(Var)`. A complex128 child yields a REAL float64 var/std (`|x−mean|²`, computed as `re²+im²` — NOT an FMA, verified bit-exact over 20K adversarial cases). FLAT folds the child buffer in MEMORY order (bit-exact for a C- or F-contiguous child); AXIS rides the engine's C-contiguous axis add.reduce (the strict-F multi-D axis corner is inherited from M2). **BIT-EXACT vs NumPy 2.4.2 for the 13 supported dtypes with NO E1 excuse**; Half + Decimal are rejected with a directed `NotSupportedException` (a bit-exact float16 variance needs a float16 pairwise kernel that dtype lacks — the "Half not diverted" gap M1/M2 share). **M4c-average** — the weighted `Average` = `Σ(values·weights) / Σ(weights)`, the FIRST reduction over TWO operand trees (`WeightedAverageNode`, not a single-child `ReduceNode`). Host-computed (`EvaluateWeightedAverage`) over BOTH materialized children, each cast to the ONE result dtype (NumPy forces the product AND both sums to `result_dtype`, so an int·int product is float64, never a wrapping int64 one — dtype rule keyed on the VALUES dtype: int/bool → `result_type(values, weights, float64)`, else `result_type(values, weights)`), reducing the product and the weights with the SAME NumPy-exact pairwise sum (`ExactSumArray`) — **NOT** the drifting multi-accumulator engine sum the library `np.average` itself uses, so the FUSED Average is **bit-exact with NumPy where the library `np.average` is only allclose at large N**. Zero total weight (empty included) raises `DivideByZeroException("Weights sum to zero, can't be normalized")` before the divide (a NaN weight → NaN, not a raise, since NaN ≠ 0); Half/Decimal result dtypes rejected. **BIT-EXACT for Single/Double/Complex, NO E1 excuse.** **M5 (reachable scope)** — `axis=None`+`keepdims` for EVERY reduction kind + direct `out=` validation, host-reshaped `(1,)*childNdim` (a free reshape of the 0-d result, so bit-exactness is inherited with no new excuse; a 0-d child stays 0-d). **Tuple `axis` is DEFERRED** — every NumSharp reduction is single-axis (`int?`), a library-wide gap, not an evaluate one. See the milestone ledger below. **Open:** M3 (streaming pairwise — no temp, restore perf). `sum(af*bf)` f32 0.60× unfused and `max(a*b)` 0.44× NumPy at 100K are M3's cells; the M4c-summation + M4c-average reductions materialize an intermediate (like M1/M2), so M3 restores their perf too. |
| P4 coverage (Cast, Positive/Conjugate/Real/Imag/…, FMax/FMin/CopySign/NextAfter/LogAddExp/shifts, logical and/or/xor, Select/Clip, `< > <= >=` operators, `where=`/`casting=`/`order=`/`dtype=`) | **binary nodes landed** (4.2); **mechanical unary nodes landed** (4.1a); **4.1b's `Rint` landed** `6e5d7e51`; **4.1b's `Real`/`Imag`/`Angle` landed** `7578ab31`; **4.1b's `Cast(dtype)` landed** `6a31bcf3`; **4.1b's `Round(decimals)` landed** `e55d1fd6`; **4.3 (logical + Select/Clip) + 4.4 (operators) landed** `4c3f74a5`; **4.5's `dtype=`/`casting=`/`order=` landed** `402d8adc`; **4.5's `where=` landed** (this) — **4.1b/4.2/4.3/4.4/4.5 COMPLETE**; 4.6 routing (perf-only) / 4.7 docs open | 4.2 `9abf25f3`, 4.1a `46f96507`, 4.1b-rint `6e5d7e51`, 4.1b-components `7578ab31`, 4.1b-cast `6a31bcf3`, 4.1b-round `e55d1fd6`, 4.3+4.4 `4c3f74a5`, 4.5-keywords `402d8adc`, 4.5-where (this) | **4.3 logical + Select/Clip, 4.4 operators** — `LogicalAnd`/`LogicalOr`/`LogicalXor` are the ONLY new node (a `LogicalNode`, `NDExprNodeTag.Logical = 13`): result ALWAYS Boolean via a per-operand NONZERO-TEST at each child's OWN dtype (`(a != 0) op (b != 0)`), so every dtype is accepted (complex `z != 0+0i` via `EmitComparisonOperation`), NaN truthy, ±0 falsy — NOT the bitwise `&`/`|`/`^`. It VECTORIZES (each child → a truthiness lane mask combined by `NDExprVec.EmitAnd/Or/Xor`; byte mode uses the engine's canonical-bool logical op), verified vector==scalar. `Select(condlist, choicelist, default=null)` is a pure LOWERING to a reverse `Where` chain (FIRST true wins; default null ≙ weak-int 0; result dtype = `result_type(*choices, default)`; NumPy-verbatim length-mismatch/empty errors). `Clip(x, lo?, hi?)` is a pure LOWERING: one-sided → `Max`/`Min` (= np.maximum/np.minimum), two-sided → `Min(Max(x, lo), hi)` (NumPy's general clip ufunc `_NPY_MIN(_NPY_MAX)` from `clip.cpp`; bit-exact with ARRAY bounds — the ONLY divergence is the signed-zero-at-a-finite-SCALAR-bound const-fast-path corner + tokenized NaN-bound signs; both-null raises "One of max or min must be given"). 4.4 operators `< > <= >=` build a Boolean `ComparisonNode` and RETURN an `NDExpr` (composable; NDArray/scalar convert implicitly); unary `+` is `Positive`; `==`/`!=` are NOT overloaded (they would hijack `expr == null` reference checks). Select/Clip/operators add NO new node — they ride the already-gated Where/MinMax/Comparison/Positive tiers. Oracle: `land`/`lor`/`lxor` in `_EV_BINARY` + block A3 (all pair layouts × dtypes + 2 composites); `evaluate.jsonl` 26,040 → **26,742** (+702); `NDEvaluateTests.P43_*` (5 — logical all-dtypes/complex/NaN, logical vector==scalar, select first-true-wins + errors, clip one/two-sided + both-null-raise, comparison operators + unary `+` + the `== null` reference check). 351 NDEvaluate/NDExpr units green net8+net10; evaluate FuzzMatrix tier + surface guard green. **4.1b `Round(decimals)`** — `np.round(x, decimals)`, the LAST member of 4.1b (completing it): a `RoundNode` carrying `_decimals` (the factory routes `decimals == 0` to the plain `Round` node), a dtype-DEPENDENT port of `PyArray_Round`. float/complex → `op2(rint(op1(x, 10^\|d\|)), 10^\|d\|)` (mul→div positive, div→mul negative) at the child's OWN float dtype (complex per-lane, dtype PRESERVED); integer d≥0 → IDENTITY (no float round-trip — huge int64 untouched); integer d<0 → float64 round CAST BACK to the int dtype (WRAPS, e.g. `round(uint8 255,-1)`=4); bool d≠0 → THROWS NumPy's UFuncTypeError (multiply for +, divide for −). `power_of_ten` ported exactly (table + repeated `*10.`, not Math.Pow → `f` bit-identical); emit reuses the engine's own per-element multiply/divide + rint. Scalar-only. **The ENGINE's `np.around`/`np.round_` is BUGGY** (Math.Round rejects negative decimals → throws; complex not rounded), so the fused RoundNode is the CORRECT port, validated against NumPy DIRECTLY (the oracle), never the engine (an engine-np.around fix is a separate issue). `evaluate.jsonl` 25,706 → 26,040 (+334); `NDEvaluateTests.P41bRound_*` (+4). **4.1b `Cast(dtype)`** — `expr.astype(dtype)`, the FIRST NEW NODE TYPE in P4 (a `CastNode` carrying the target dtype, which a payload-less `UnaryOp` enum cannot): full node contract (`InferType`/`EmitScalar`/`BindArrays`/`HashStructure`/`StructureEquals`/`CanEmitVectorV2`/`AppendSignature`). Result IS the target (a WEAK child resolves to its OWN default first — `astype(np.array(5),f4)` casts int64, not adopted-float); per-element `EmitConvertTo` = `np.evaluate(child).astype(target)`, bit-exact incl. the C-undefined float→int edges (both sides share the NumPy-faithful `Converts.*`). Two edge-only `EmitConvertTo` bugs Cast was the first to hit are special-cased in `CastNode.EmitScalar`: `→bool` is a type-correct `child != 0` (its `to==Boolean` did a double-vs-int stack compare), `complex→real` extracts the real lane (it mis-computed complex→int and ACCESS-VIOLATED on complex→Half). SCALAR-ONLY (`CanEmitVectorV2` false; SIMD widening is Phase 5). Verified fused == the engine's NumPy-gated `astype` over the FULL 15×15 matrix incl. every C-undefined edge (239 probe cells, 0 mismatch). Oracle: a `cast_<dtype>(child)` token + a src×target×layouts block over a SAFE non-negative pool (portable bit-exact, no host-dependent cell); `evaluate.jsonl` 24,987 → 25,706 (+719); `NDEvaluateTests.P41bCast_*` (+4). **4.1b `Real`/`Imag`/`Angle`** — the complex→real component extractors, the first genuinely-new emit path in 4.1b (NOT ufuncs, so no engine `UnaryOp` kernel): three new `UnaryOp` enum values handled ENTIRELY in `UnaryNode.EmitScalar` (before the generic tail, which throws for them). A COMPLEX child extracts a float64 lane (`Real`→`z.Real`, `Imag`→`z.Imaginary`, `Angle`→`atan2(im,re)`) via three static `NDComplexMath` helpers (`RealPart`/`ImagPart`/`Angle`, the `s_complexAbs` pass-by-value pattern); a REAL child degenerates — `Real` identity (dtype PRESERVED), `Imag` zeros (dtype PRESERVED, child value UNREAD = `zeros_like`), `Angle` `atan2(0,x)` (0 / pi / NaN, `-0.0`→pi) narrowed to `np.AngleRealTier` (made `internal`, reused so the fused node and `np.angle` share ONE table). Typing (probed 2.4.2): `Real`/`Imag` dtype-PRESERVING (`Complex?Double:childType`), `Angle` `Complex?Double:AngleRealTier`. Scalar-only (`CanEmitVectorV2` false — not in `IsSimdUnary`). `Real`/`Imag` BIT-EXACT (lane extract / identity / zeros); `Angle` rides host `atan2`, BIT-EXACT within the host-pinned tier (win-amd64 shares MSVC `ucrtbase`, the arctanh class), NO ULP excuse. Verified fused == the engine's own NumPy-gated `np.real`/`np.imag`/`np.angle` across dtypes × layouts × composites × NaN/inf/-0.0. `evaluate.jsonl` 24,415 → 24,987 (+572); `NDEvaluateTests.P41b_*` (+4). **4.1b `Rint`** — the mechanical member of the dtype-CHANGING set: `np.rint` (round-half-to-even) has the SAME value as `Round`, so the new `UnaryOp.Rint` ALIASES `Round` at every kernel emit site (the `Fabs`/`Abs` precedent — no new emit path) and differs only in typing (`IsFloatPromoting` → `UnaryFloatResult`: bool/i8/u8→f16, i16/u16→f32, i32+→f64; float/complex/decimal preserved), where `Round` preserves the input dtype. Vectorizes at f32/f64 like `Round` (`IsRoundingOp` routes `IsSimdUnaryAt`→`RoundingVectorSimdAvailable`). ALL cells BIT-EXACT vs NumPy 2.4.2, NO ULP excuse (`Math.Round` portable; complex rounds both lanes). `evaluate.jsonl` 24,168 → 24,415 (+247); `NDEvaluateTests.P41b_*` (2). **4.2** — every engine binary ufunc that lacked a node now has one: `Maximum`/`Minimum` (NaN-propagating aliases of Max/Min), `FMax`/`FMin` (NaN-ignoring), `Fmod`, the float-tier family `CopySign`/`NextAfter`/`LogAddExp`/`LogAddExp2`/`Hypot`/`Heaviside`, integer-only `Gcd`/`Lcm`, and `LeftShift`/`RightShift`. Mechanical by construction — the shared scalar emitter `EmitScalarOperation` already dispatches the whole family at every result dtype (incl. float16), so only factory + typing (BinaryNode.InferType promotion + no-loop rejection) were added; the emit is byte-for-byte the unfused chain. BIT-EXACT vs NumPy 2.4.2 for copysign/nextafter/heaviside/fmod/fmax/fmin/gcd/lcm/shifts; logaddexp/logaddexp2 (≤2 ULP) and hypot@f64 (≤1 ULP) ride the `EvaluateLibmOps` ~ULP excuse (the same host-libm envelope the engine documents). `evaluate.jsonl` 21,280 → 23,224 (+1,944); `NDEvaluateTests.P4_*` (9). **4.1a** — the same mechanical route for the engine UNARY ufuncs that lacked a node: `Positive` (identity, dtype-preserving, no bool loop), `Conjugate`/`Conj` (complex flip; real identity; **bool→int8** quirk), `Fabs` (float-only |x|, int→float promote, no complex loop), `Spacing` (one-ULP, float-only), `SignBit`/`IsPosInf`/`IsNegInf` (bool predicates; complex rejected — signbit with the ufunc no-loop message, isposinf/isneginf with the DISTINCT "ambiguous for complex128" message since they are functions, not ufuncs), and `BitwiseCount` (popcount→uint8, integer/bool/char only). `UnaryNode.EmitUnaryScalarOperation` already handles all eight, so only factory + typing (InferType) + routing the bool/byte-result ops through the existing "emit-at-child-dtype, convert" path were added. **ALL cells BIT-EXACT — NO ULP excuse** (identity / sign-bit clear / popcount / pure-bit-increment spacing, no host-libm). Positive/Fabs vectorize (added to `IsSimdUnary`, verified vector==scalar bit-for-bit); Spacing stays SCALAR-only (its vector body and scalar body differ on the NaN payload at an inf/NaN input — a fused-vs-scalar divergence not worth the marginal SIMD win). `evaluate.jsonl` 23,224 → 24,168 (+944); `NDEvaluateTests.P41_*` (8). **4.5's `dtype=` / `casting=` / `order=`** (this) — the three RESULT-shaping keywords, threaded from the API as ONE resolved public `NDEvaluateOptions` struct built so `default` reproduces pre-4.5 behaviour exactly. `dtype=` (a `DType`) is an **implicit root cast** on the elementwise path — compute at the natural NEP50 type, cast the RESULT (== `np.evaluate(expr).astype(X)`) in ONE fused pass via the out= buffered-cast machinery (distinct from the Cast NODE's per-element `EmitConvertTo`); rejected on a reduction tree (a reduction fixes its accumulator dtype) and mutually exclusive with `out=`. `casting=` (default `same_kind`) governs the `out=` cast rule, threaded to every path incl. the reduce helpers' now-parameterised `ValidateOutCast` (`'unsafe'` admits float→int out, `'no'` rejects any real cast). `order=` (default `'K'`) lays out the FRESH result — `'C'`/`'F'` force, `'A'`/`'K'` keep today's strict-F heuristic; a caller `out=` keeps its own layout, and a non-`'K'` order is rejected on a reduction tree. Oracle: `evaluate.jsonl` **B5** block (+281 — `dtype=X` over a fused tree × safe pool × real+complex targets × 5 layouts, `<expr>.astype(X)` reference — gating the fused buffered-cast bit-exact vs NumPy independently of the Cast node); 26,742 → **27,023**. `NDEvaluateTests.P45_*` (9) + a 63-cell layout differential (dtype= == the NumPy-gated `.astype()` on broadcast/transposed/negstride/strided/offset). **4.5's `where=`** (this) — the masked-WRITE keyword, the LAST P4 correctness item. It rides the SAME options bundle (`NDEvaluateOptions.Where`, so the seam is unchanged and `default` is unmasked) and reuses the ufunc where= machinery: the mask is validated bool (`ValidateWhereMask`), joins the broadcast (`ResolveUfuncIterationShape` — never stretching a provided `out=`), and the fused pass appends it as a trailing **ARRAYMASK** operand with the output flipped to **WRITEMASKED** (`EvalElementwiseMaskedFlags`). **It needs NO 4.6 re-routing:** `NDIterRef.ForEach(kernel, aux)` — which np.evaluate already calls with its pre-compiled fused `NDInnerLoopFunc` — resolves the mask itself (`ResolveForEachMaskOp`) and runs the kernel per mask-true run, so masked-off destination slots keep their prior contents (the SIMD 32-byte block run-scan drives dense/sparse masks). Composes with `dtype=`-cast (the WRITEMASKED buffered flush also masks) and 0-d parameters (the aux block is invariant across runs); **rejected on a reduction tree** (a masked reduction is a different operation — the message points at masking the inputs first). Oracle: `evaluate.jsonl` **D2** block (+280 — masked writes into out= over checker/alt/all-true/all-false/strided/negstride/row/col masks × c/strided out × float64/float32/int32/int64 × two fused trees; each records BOTH the returned view AND the whole out base, so a write-everywhere kernel turns it red), 27,023 → **27,303**, bit-exact vs NumPy 2.4.2 (host-pinned). `NDEvaluateTests.P45Where_*` (9 — prior-preserved masked write, no-out, broadcast mask, all-true/all-false, cast+mask, 0-d param + mask, reduction rejection, non-bool-mask throw, positional overload) + a Release run-scan stress probe (N 31…4097 × 6 mask regimes). **Open in P4 (4.1/4.2/4.3/4.4/4.5 COMPLETE):** 4.6 (route through `ExecuteElementWise` so narrow strided rows AND masks reach the 2-D block kernels — PERF ONLY, and it needs a 2-D-block variant of the fused kernel since np.evaluate compiles an `NDInnerLoopFunc`, not the scalar/vector emit bodies `ExecuteElementWise` consumes; where= correctness is already complete via ForEach's masked driver), 4.7 docs. |
| P5.1 Half arithmetic SIMD (+ the double→float32 fix it forced) | **landed** | (2026-09-14 landing) | **A latent correctness fix AND the flagship f16 perf win.** The fused SCALAR Half `add/sub/mul/div` computed in **double** (the old `EmitHalfOperation` bridge) where NumPy's HALF loop — and the engine's own `HalfArithFull` kernel — compute in **float32** (`astype 'e'->'f'`), a real 1-ULP-plus divergence on exponent-gap sums (and it BCL-quieted sNaN); it now routes through `HalfArithScalarStruct` (== `HalfArithBits`, float32 + RTNE-narrow + the operand-order NaN pin), so fused Half arithmetic is **bit-for-bit the NumPy-gated engine kernel** (probe: diff 0/35 over an adversarial NaN/inf/exponent-gap pool, both scalar AND vector). The fused VECTOR path then rides a `Vector256<ushort>` lane (16 raw f16 patterns): each arithmetic node widens to two `Vector256<float>`, does one `Avx.Add/Sub/Mul/Div`, and narrows RTNE with the NaN fixup — reusing the engine's proven 8-lane pieces (`HalfArithVec256`), so **vector == scalar by construction** (metamorphic sweep green with Half added, all layouts). **Scope:** pure f16 ARITHMETIC on a 256-bit AVX2 host; any comparison / where / min-max / transcendental / unary in a Half tree keeps the whole tree scalar (still correct), and a non-256-bit host stays scalar. **Perf (best-of-15, Release, `f2*f2+f2`):** fused VECTOR is **4.3–5.6× the old scalar fused path** (100K 4.30×, 1M 4.82×, 6-op chain 5.56×), which — NumPy's Half loop being scalar — flips fused Half from ~0.3–0.5× NumPy (losing) to beating NumPy (the engine's `HalfArithFull` is 2–3.6× NumPy and the fused is ~0.6–0.77× of that). It does NOT beat the engine's specialized unfused chain (0.6–0.77×): the Giesen f16↔f32 conversion is the shared ceiling (no F16C in the BCL) and the general fused shell carries per-node widen/narrow overhead the hand-tuned `HalfArithFull` avoids — so the plan's ≥1.5×-unfused target is unmet for Half and would need a specialized fused Half kernel (deferred). Files: `HalfArithScalarStruct`/`HalfArithVec256` (`DirectILKernelGenerator.Binary.Arith.Half.cs`), `EmitHalfOperation` add/sub/mul/div intercept + `GetSimdLaneType(Half)→ushort` (`DirectILKernelGenerator.cs`), the shell's Half lane gate + broadcast (`InnerLoop.Fused.cs`), the plan/emit (`NDExpr.Vector.cs`, `NDExpr.cs`, `NDExpr.Params.cs`). Gates: `NDEvaluateVectorTests` (Half added to the lane sweep, 10/10) + `NDEvaluateTests.P51_*` (4) + the `evaluate.jsonl` tier (green, its float16 cases now bit-exact). |
| P5.2 mixed-width SIMD ("lane groups") | **landed** | (2026-09-14 landing) | Lane W = the ROOT dtype (the widest — NEP50 promotion never narrows along a parent edge), elemCount = one full `Vector256<W>`; every node computes at its OWN dtype in a container of elemCount·sizeof(dtype) bytes — admitted only at 128/256 bits, so half-lane dtypes ride partial `Vector128`s and a partial load is exactly 16 bytes, never an over-read — and every child edge is one EXACT V128→V256 widen (`Avx/Avx2.ConvertToVector256*`; `u4→f8` — which AVX2 lacks — via the exact sign-bias trick `cvt((int)(u^0x80000000)) + 2^31`). Per-node dtypes are the load-bearing decision: `i4*2+f8`'s multiply WRAPS at int32 BEFORE the edge widens, bit-compatible with NumPy's unfused sequence (probed 2.4.2). Scope mirrors P5.1: leaves + binary arithmetic/bitwise on a 256-bit AVX2 host (`FusedMixedWidthAvailable`); bool-carrying / Half / unary / min-max / comparison-root mixed trees — and any tree with a same-size INEXACT edge (`i8→f8` rounds; no AVX2 vector form) — stay whole-tree scalar (pre-P5.2 behavior). Also serves UNIFORM-input trees lifted past their inputs by a weak literal (`i4 + 2.5` → f8) and int true-divide (`i4/i4` → f8). **Perf (best-of-15, Release, pinned core; NPY/NS):** `i4*2+f8` **14.2× @100K / 5.6× @4M** (the ≥4× target — met; 1.35× @1K, the per-call floor), `f4*f8+f4` 22.2× / 7.7×, `i4/i4` 6.0× / 4.1×; fused beats NumSharp's own unfused chains 1.5–13×. Gates: `evaluate.jsonl` 27,303 → **28,311** (+1,008: block **A4** — 14 ratio-2 dtype pairs × 8 exprs × the pair layouts, incl. the `u4→f8` bias edge, the both-widen promotions `i4+u4→i8`/`i2+u2→i4`/`i1+u1→i2`, and the wrap composite `add(mul(in0,li:2),in1)`), with the 3,344 PRE-EXISTING mixed-dtype cases now riding the vector path — all bit-exact vs NumPy 2.4.2; `NDEvaluateVectorTests.MixedWidth_*` (3: the 15-pair × 11-tree × 4-layout + 0-d-params vec==scalar sweep, the NumPy-probed value pins incl. the int32-wrap pin, out=/where=/dtype= composition) + the flipped `VectorPlan` pins (i4+f8 / i4-true-divide / f4+f8 / i1*u1 now plan MIXED; i8+f8, ratio-4 i1+i4, bool-mixed, comparison-root, unary-mixed stay scalar) + a 2,234-check self-consistency probe (18 pairs × 6 layouts × 10 boundary sizes, 0 fails). Files: `NDExpr.Vector.cs` (`TryPlanMixed` + per-node `CanEmitVectorMixed` + `NDExprVec.HasWidenEdge`/`EmitWidenEdge`), `NDExpr.cs` (ctx `MixedWidth`/`ContainerBits`, the widen edge in `EmitVectorChildAs`, width-explicit Const/Binary emission), `NDExpr.Typing.cs` (per-operand vector locals + plan threading), `NDExpr.Params.cs` (per-dtype parameter broadcasts), `DirectILKernelGenerator.cs` (`EmitVectorLoadAt`/`EmitVectorCreateAt`/`EmitVectorOperationAt`), `InnerLoop.Fused.cs` (`FusedMixedWidthAvailable`, `FusedSimdViable(mixed)`, per-operand loads/broadcasts/contiguity, gather tier skipped). |
| **6.3** single-op trees delegate to the direct kernel | **landed** | (2026-09-14 landing) | A one-op-over-array-leaves tree — `abs(a)`, `maximum(a, b)`, `a + b` — has nothing to fuse, so the fused NDIter pass only ADDS fixed cost over the engine's own whole-array kernel (measured: it LOST at 1K — `maximum(a,b)` 0.59×, `abs(a)` 0.63×, §0.1 Section C's last two sub-1.0× rows). Now, on a PLAIN call (no `out=`/`where=`/`dtype=`/non-`'K'` order) over C-contiguous array leaves, `np.evaluate` recognises the shape at PROGRAM BUILD time (`NDExprProgram.DirectOp`, from `NDExpr.AsDirectSingleOp`) and delegates to `ExecuteUnaryOp`/`ExecuteBinaryOp` (`DefaultEngine.TryDelegateDirectSingleOp`). It is bit-exact BY CONSTRUCTION — both paths are already gated bit-exact to NumPy by `evaluate.jsonl` and the unary/binary corpus tiers, so two things equal to NumPy are equal to each other — with three gate conditions making that airtight: **array leaves** (a weak-literal `ConstNode` would promote differently under NEP50), **C-contiguous inputs** (so the result is C-contiguous on BOTH paths — the fused strict-F heuristic never fires for C inputs, so the delegated bytes match), and the **plain call**. Two promotion-engine hazards forced NARROW positive allow-lists (`NDExpr.IsDelegatable{Unary,Binary}`, validated by a delegated-vs-fused sweep over every op × dtype): a forced `typeCode` on some unary ops reaches a trivial-contiguous engine path that FATAL-CRASHES the CLR (`logical_not(complex)` — a latent engine bug the general fused pass never hits), and `Reciprocal(int)` where the engine and fused paths simply disagree — so the unary list is the "real math" ufuncs only (predicates / `LogicalNot` / `SignBit` / `BitwiseNot` / `BitwiseCount` / `Conjugate` / `Positive` / `Spacing` / `Rint` / `Real`/`Imag`/`Angle` stay fused); the binary path forces NO dtype and excludes 0-d operands (whose value-based `_FindCommonType` promotion diverges from the fused STRONG typing) plus the float-tier promoting ops (`ATan2`/`CopySign`/`NextAfter`/`LogAddExp`/`LogAddExp2`/`Hypot`/`Heaviside`, which fused-promote int→float where `_FindCommonType` keeps int → "no loop") and `Power` (whose negative-int-exponent guard a forced dtype bypassed). The excluded ops are all rare single-op forms; leaving them fused costs nothing measurable. Gates: `NDEvaluateTests.P63_*` (5 — the delegated-vs-fused unary AND binary allow-list sweeps over every op × 15 dtypes, the excluded-envelope pins {non-contiguous / weak literal / 0-d operand / multi-op / `out=` / `dtype=`}, `NDExpr.DisableDirectOp`/`DirectOpDelegations` proving the fast path engages); `evaluate.jsonl` tier unchanged/green (its C-contiguous single-op cells now delegate and still match NumPy). Files: `NDExpr.DirectOp.cs` (the descriptor + `AsDirectSingleOp` overrides + the two allow-lists + the `DisableDirectOp`/`DirectOpDelegations` hooks), `NDExpr.Program.cs` (`DirectOp` field, computed in `Build`), `DefaultEngine.Evaluate.cs` (the `EvaluateCore` gate + `TryDelegateDirectSingleOp`). |
| P6 optimizer (CSE / constant folding / opt-in threading / string front-end / `Explain`) | **6.3 landed** (above); 6.1 landed (row above); 6.2/6.4/6.5/6.6 open | | the structural hash (6.1) is also what would make the inline-rebuilt spelling below cheap. |
| **C6** combinators — oracle-grammar coverage | **landed** (this) | (2026-09-14) | The 39 C6 macro/decision combinators (`NDExpr.Combinators.cs`, brought over from master's `0c47162e`) gained differential-fuzz oracle coverage — the documented "open follow-up" the C6 landing left (its gate had been the in-process metamorphic `NDExprCombinatorTests` only, not the Python-oracle grammar). `gen_oracle.gen_evaluate` gained `_EV_COMBINATOR` (each token → the SAME NumPy composition the factory builds, honoring the WEAK Python scalars it bakes in — `Const(0)` a weak int, `Const(0.0)` a weak float — so NEP50 promotion matches the fused per-node typing) + block **E** (E1 single-array roots × single layouts, E2 two-operand roots × pair layouts, E3 COMBINATIONS — combinators nested in primitives and in each other), and `OpRegistry.Evaluate.cs` gained the matching `BuildNode` cases (the variadic `switch`/`mux`/`bucketize` reconstructed from the flat arg list). Dtype policy per group: transcendental activations (sigmoid/swish/elu/softplus/gelu) stay float32/float64 (bit-exact ports on the host-pinned tier; f16's exp edge held out), the bitwise boolean-logic family (nand/nor/xnor/implies/majority3) is int/bool ONLY (a float operand is a NumPy no-loop), complex is excluded (Max/Min/comparison/floor have no complex loop). **On this branch `Heaviside` is the Phase-4.2 binary NODE, not a combinator** — the C6 `Heaviside(x,h0)` composition was dropped (a CS0111 duplicate of the node, and the node is the real `np.heaviside` that propagates NaN where the composition would return `h0`), so the token set excludes it (`NDExpr.Heaviside` still resolves — to the node). `evaluate.jsonl` 28,311 → **29,187** (+876), ALL bit-exact vs NumPy 2.4.2 (host-pinned) with NO new `MisalignedRegistry` excuse. The one divergence surfaced was a REFERENCE bug, not a kernel one: float32 `steptoward` resolved to float64 in the fused tree because `-delta` is a `Negate` NODE (a ufunc, which strong-ifies a weak scalar exactly as `np.negative` does), while the reference used Python's folded `-d` literal (weak → float32); fixed by spelling the reference `np.negative(d)`, so both sides are float64. Metamorphic gate `NDExprCombinatorTests` (36) stays green (its `Heaviside_WithValueAtZero` now binds to the P4.2 node, same values). Gate: `evaluate.jsonl` tier green net8+net10; full FuzzMatrix 100/3/1 (the 1 the pre-existing piecewise/logspace/geomspace scope-audit red, no combinator family). |

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

### 0.2 Phase 2 milestone ledger (reductions)

Phase 2 is split into five milestones, executed in order. The parity approach is uniform — **materialize
the child through the already-bit-exact elementwise engine, then reduce that fresh buffer with NumPy's OWN
schedule** (the fold that `FlatReduceKernel` / `AxisReduceKernel` emit is the E1 order-of-summation drift).
Materialization regresses reduction PERF (an intermediate is allocated); M3's streaming pairwise (the plan's
"recommended" form, no temp) restores it. The FLAT `np.sum` / axis pairwise the engine already runs is
bit-exact (`ILKernelGenerator.Reduction.Pairwise.cs`); only the fused reduce paths folded.

| M | Scope | Status | How / evidence |
|---|---|---|---|
| **M1** | FLAT float32/float64/complex Sum·Mean + float32/float64 Prod, bit-exact | landed `7ac2c7b5` | `DefaultEngine.EvaluateReduce` flat path: `PairwiseSumInto` (reuses `TryEmitPairwiseSumKernel` via its PINNED stride-0 route = np.add.reduce) / `SequentialProductInto` (np.multiply.reduce). Oracle block C2 (56 flat bigN cases at N∈{7,8,127,128,129,257,1000}, wide-magnitude). |
| **M2** | AXIS float32/float64/complex Sum·Mean + float32/float64 Prod, bit-exact, when the materialized child is **C-contiguous** | landed (this) | `EvaluateAxisReduce` diverts: `ExactAxisSum` (1-D → `PairwiseSumInto`; multi-D → the engine's own `ReduceAdd` on the C-contiguous child, whose IL pairwise kernel is bit-for-bit np.add.reduce — PINNED for a contiguous reduced axis, SLAB-sequential for an outer one) and `SequentialAxisProd` (a C-order coordinate-order fold = np.multiply.reduce, layout-independent for a sequential product). Gated to `!AreAllInputsStrictFContig` (a fresh child is C-contig unless every input is strictly F-contig multi-D). Oracle block C3 (56 axis bigN cases, 2-D `(3,200)` / `(200,3)` + 3-D `(2,3,60)`, axes 0/1/2). **Teeth:** divert OFF → 18 wide-magnitude cases hard-diverge (>16 ULP) + the ≤16-ULP ones fall to E1; divert ON → 0. |
| **M3** | Streaming pairwise — no temp; restore reduction perf | **landed (2026-09-23)** for the M1/M2 divert set over contiguous inputs; SIMD Min/Max/int folds still open (see "Perf review 2026-09-23" below) | `DefaultEngine.Evaluate.Stream.cs`. The host drives NumPy's OWN pairwise recursion (`left = n/2 − (n/2)%8`; complex `left = (n − n%8)/2`) and, at each sub-range ≤ an 8 KB L1 scratch block, evaluates the child's fused kernel into scratch and folds it with the SAME cached pairwise fold every `np.add.reduce` kernel calls (typed accessors `ILKernelGenerator.TryGetPairwiseFold{Double,Single,Complex}`) — bit-identical to materialize-then-reduce **by construction**; the sequential product streams block by block. Axis: whole-row PINNED pairwise (many short rows per block) and increasing-index SLAB accumulation, size-1 axis = copy (the engine's trivial-axis behavior). Gate: `CanStreamChild` — identical dims, no broadcast, one shared contiguous order (all C, or flat-only all F); everything else keeps materialize. **Verified:** 3,614-case in-process raw-bit differential vs the materialize path (NaN tokenized — see caveat), all engaged/fallback as expected; a NumPy-2.4.2 `.npy` oracle on long rows (20K/100K), 50K×3 short rows and 3-D all axes (0 diffs, streamed); `NDEvaluateStreamingTests` (6, uses the `NDExpr.DisableStreamingReduce` / `StreamingReductions` hooks); 423 NDEvaluate/NDExpr units; FuzzMatrix 100/3/1 (the 1 = pre-existing 38-family leak-audit red). **Caveat:** which of two DISTINCT NaN payloads survives may differ (RyuJIT commutes a C# float add; non-contractual — the oracle tokenizes NaN). **Perf (stream vs the old materialize path, pinned P-core, best-of):** `sum(a*b)` f64 1.40× @1K / 1.61× @100K / 2.45× @1M / 2.47× @4M; `mean((a-b)²)` 2.00/1.77/2.59/1.90×; `sum(af*bf)` f32 1.75/1.18/1.58/2.11×; complex sum 1.08–1.82×; `prod` 1.05–1.44× (the exact sequential multiply chain is latency-bound — NumPy's `multiply.reduce` is sequential too); axis sums 1.45× @1K, 1.66–1.93× @100K, 1.76–2.04× @4M (2000²), 3.5–5.2× on 50000×3, 2.0–2.3× on 3×1M. |

**Perf review 2026-09-23 — measured findings and the next levers (start here next session).** Probes:
`benchmark/fusion/probes/expr_probe.cs` (sections C/D) + `numpy_twins.py`; the M3 probes lived in the session scratchpad
(stream prototype, `.npy` NumPy oracle, raw-bit differential, reduction perf) — rebuild from the description above.
1. **Fixed cost, n = 8 (ns / B):** prebuilt `np.evaluate(a*b+c)` 512 / 592 — of which `new NDArray(result)` ≈ 260 / 472
   (shared by every op), `MultiNew+ForEach+Dispose` ≈ 136, program lookup 3, and ≈ 150 of host bookkeeping beyond what
   `np.add(out=)` (156 total) pays. Levers: (a) **direct kernel dispatch** when every iterator operand and the target are
   contiguous in one shared order with identical dims (no `where=`, no out-cast, no partial overlap with a provided out) —
   call `program.Kernel` once with element strides, skipping NDIter (~123 ns; the engine's trivial-loop routes already do
   this); (b) skip `ResolveUfuncIterationShape(...).Clean()` when `out=` has the inputs' dims (the `*UfuncInto` routes skip
   it); (c) `CanonicalResultShape` instead of `Shape.Clean()` for the fresh result (no dims/strides clone).
   **LANDED (a)+(b)+(c) as NumPy's trivial loop** — `DefaultEngine.Evaluate.Trivial.cs`, a port of
   `try_trivial_single_output_loop` (`ufunc_object.c`): every non-0-d operand (inputs + out) has the operation's exact
   dims and is 1-D (any stride, incl. negative / 0) or contiguous in ONE shared order (C or F; both-contiguous operands fix
   none), 0-d iterator operands ride a zero stride, the fresh result takes exactly the layout the NDIter pass allocates
   (`ResolveEvalOrder` over the strict-F heuristic) through `CanonicalResultShape` (no dims/strides clone), and a
   provided out may overlap an input ONLY exactly (same first byte + byte stride + element size) — any other overlap,
   an out-cast, `dtype=`, `where=`, a read-only out or a genuine broadcast DECLINES (the gate never throws, so every
   error path is the iterator's, unchanged). The kernel call is the one NDIter makes after coalescing (same pointers,
   count, byte strides), so the result is byte-identical by construction; the 1-D-strided / 0-d-stride shapes rely on the
   vector==scalar contract for any chunking difference. (b) rides along on the NDIter path for a same-dims out.
   Hooks `NDExpr.DisableTrivialLoop` / `TrivialLoopRuns`. **Measured (pinned P-core, ON vs OFF interleaved in one
   process, best-of):** n = 8 prebuilt 527 → 255 ns (592 → 472 B), rebuilt 655 → 347, `out=` 332 → **75** ns (184 → 0 B),
   0-d-param tree 540 → 262, f32 561 → 284, 1-D stride-2 364 → 256; n = 1K `out=` 2.98–3.25×, fresh 1.1–1.8× (the
   allocator floor dominates), stride-2 1.3–1.8×; 100K–1M 1.00–1.04× (as it must be — only fixed cost moved; an
   earlier 0.92× reading was ON-first allocator warm-up bias and vanished with alternating order). The fused `a*b+c` at
   n = 8 is now ~2× NumPy's two-ufunc chain (526 ns) and beats NumSharp's own unfused chain (373 ns). Gates:
   `NDEvaluateTrivialLoopTests` (9: every layout class engages/declines as specified and is byte-, dtype-, shape- and
   C/F-flag-identical to the NDIter pass; a MUTANT with the overlap guard removed corrupts the output-ahead self-overlap
   case, so the gate has teeth), 437 NDEvaluate/NDExpr/Evaluate units, FuzzMatrix 100/3/1 (the 1 = the pre-existing
   38-family leak red, unchanged). What remains at n = 8 is the fresh-result `new NDArray` (≈ 170 ns), shared with every
   op — the allocator floor, not an evaluate cost.
2. **Boxed per-element helpers on the axis path:** `ILKernelGenerator.SeedReduceIdentity` / `MeanDivideByCount` and
   `EvaluateAxisReduce`'s complex-mean loop use `SetAtIndex(object)`/`GetAtIndex` per output — `mean(a*b, axis=0)` on
   3×1M spends ≈ 4 ms of its 6.9 ms there (and the engine's `np.mean`/`np.sum(axis)` pay the same). Typed contiguous fast
   paths with the identical arithmetic (`x / (double)n`, `x / (float)n`, `ComplexDivideByCountLikeNumPy`).
   **LANDED** (`ILKernelGenerator.Reduction.cs` + `EvaluateAxisReduce`): a writeable, non-broadcast, C- OR F-contiguous
   output is a DENSE BLOCK at `Address + offset·itemsize` (a contiguous row slice re-seats the address with offset 0; an
   F column block of an F array keeps a non-zero offset — both covered). `SeedReduceIdentity` writes logical element 0
   through the boxed setter (its exact unboxing conversion + read-only check are the contract) and replicates those bytes
   across the block with a width-typed `Span.Fill` (1/2/4/8/16-byte carriers); `MeanDivideByCount` divides the block
   through typed pointers — `Vector<T>` lanes for f64/f32 (IEEE division is correctly rounded per lane, so bit-identical
   to the scalar `x / d`, NaN payloads and signed zeros included), scalar complex (component-wise) and decimal; the fused
   complex mean divides its fresh C accumulator through a typed pointer with `ComplexDivideByCountLikeNumPy`. Strided /
   read-only outputs keep the boxed loops verbatim. **Measured (pinned P-core, best-of; bits: every result checksum
   identical before/after):** engine `np.mean(a,0)` 3×1M f64 6.75 → **2.07 ms** (NumPy 3.11 → NPY/NS 0.46 → **1.50**),
   f32 7.04 → **0.77 ms** (NumPy 2.03 → 0.29 → **2.65**), `np.sum(a,0)` 3×1M 2.61 → 1.72 (NPY/NS 1.07 → 1.62),
   `np.mean(a,1)` 1M×3 12.5 → 8.1 ms (NumPy 7.18 → 0.58 → 0.89 — the rest is per-3-element-row iterator overhead, the
   narrow-row gap, not this pass); fused `Mean(a*b,0)` 3×1M 6.14 → 2.48 ms, `Max(a*b,0)` 3×1M 3.9 → 2.9. Gates:
   `ReductionPostPassTests` (5: every identity × dtype × layout — C, F, C row block, F offset block, strided fallback —
   writes only the output's elements; the division is bit-exact `x / (T)count` over a NaN-payload/±0/±inf/subnormal pool
   for counts 1/3/7/1,000,003; a read-only out still raises and stays unchanged; engine + evaluate axis means equal
   sum/n bit-for-bit — MUTANTS ignoring the view offset (3 tests) or dividing by a reciprocal (2 tests) go red), full
   NumSharp.Tests 15,552 green (the 25 failures are the OpenBLAS-not-staged Examples demos), FuzzMatrix 100/3/1 unchanged.
   Found on the way, NOT fixed (engine, flat `np.var`/`np.std`): `ddof ≥ n` on non-constant data returns NaN where NumPy
   returns +inf — already pinned `[OpenBugs]` (`AuditV2 T1_37a/b`); the fused `NDExpr.Var`/`Std` get it right.
3. **Flat-reduce fold is SCALAR** (`CompileReduceKernel`: 4 scalar accumulators) — Min/Max, integer Sum/Prod, Any/All,
   CountNonzero ride it: `max(a*b)` f64 is 0.91× NumPy @100K. Vectorize (order-independent for values; keep the ±0 tie and
   first-NaN semantics of the scalar `np.maximum` clamp).
   **Bool folds STREAMED (LANDED):** a flat `Any` / `All` / `Sum`-of-a-bool-child (`CountNonzero`, int64 accumulator) no
   longer rides the scalar fold — `DefaultEngine.TryStreamBoolFold` (`DefaultEngine.Evaluate.Stream.cs`, hooked into
   `EvaluateReduce`'s flat path) evaluates the bool child into the 8 KB L1 scratch block by block with the child's own
   kernel and folds each block with the BCL's vectorized span scans (`IndexOfAnyExcept(0)` / `IndexOf(0)` /
   `Count(0)`); Any / All STOP at the first deciding block. Exact BY CONSTRUCTION — OR / AND / an integer count do not
   depend on order — so any shared contiguous order streams (all C or all F; `CanStreamChild(allowF: true)`) and the
   early exit is safe. Two refinements carry the speed: (1) the factories spell the child `x != 0`, and for a Boolean
   `x` NEP50 types that as an **int64** comparison (a mixed-width tree the vector plan declines → a SCALAR child kernel,
   so `count_nonzero(a > b)` first gained nothing) — `NDExprProgram.NonzeroBoolOperandProgram` recognizes exactly the
   factories' spelling (`ComparisonNode.NonzeroTestOperand`: `NotEqual`, right operand the INTEGER literal 0) over a
   Boolean `x` (decided by the typing pass alone, so a numeric `x` never JITs a probe kernel; the null answer is cached
   behind its own resolved flag) and streams `x` itself — the same bools through `x`'s SIMD kernel; (2) a bare bool
   LEAF (`any(mask)`, `count_nonzero(mask)`) is folded IN PLACE over its dense block at `Address + offset` — no identity
   kernel, no scratch copy (the copy had it at 0.65–0.83× NumPy). **Measured (pinned P-core, alternating ON/OFF order,
   best-of; stream vs the old fold, then NPY/NS against a pinned NumPy 2.4.2 twin):** `any(a>b)` 1.9× / 34× / 2242×
   at 1K / 100K / 4M (NPY/NS 3.8 / 7.7 / 1870 — the first true is early; NumPy materializes the whole comparison),
   `any(z)` full scan 2.4 / 6.7 / 3.0× (NPY/NS 5.0 / 4.2 / 2.1–2.3), `all(a>0)` 1.8 / 4.2 / 2.2× (NPY/NS 4.0 / 0.94 /
   1.7–1.9), `count_nonzero(a>b)` 1.8 / 3.3–3.7 / 1.3× (NPY/NS 1.4 / 0.86–0.95 / 1.2 — 100K sits at the read-bandwidth
   ceiling of the two f64 operands on both sides), `count_nonzero(mask)` 2.5 / 28 / 29× (NPY/NS ≈1.0 / 1.43 / 1.22),
   `all(mask)` early exit ≈ 0.3 µs (NPY/NS 4.7). Gates: `NDEvaluateStreamingTests` 11 (+2 — byte equality vs the fold
   kernel at every deciding-element position (first/last/block boundaries/none) × C/F/offset/bool-leaf layouts × 0-d
   params, strided/broadcast declines, and the resolution contract; MUTANTS: widening the literal match so
   `sum(flags != 1)` streams `flags`, and dropping the view offset from the in-place scan (an F column block at offset
   42), both go red), 424 NDEvaluate/NDExpr/Evaluate/post-pass units.
   **Flat `Min`/`Max` NumPy-EXACT (LANDED):** pinned first — a Python simulation of NumPy 2.4.2's
   `loops_minmax.dispatch.c.src::simd_reduce_c_{max,min}` with the AVX2 `npyv` intrinsics matched `np.max`/`np.min`
   BIT-FOR-BIT on 2,056 tie-heavy (±0) and NaN-laced cases (NaN payloads included), while a per-8192-chunk restart
   missed 48 and a sequential scan (either tie rule) 152–304. The schedule: ONE call over the n−1 elements after the
   copied `x[0]`, `acc = splat(x[0])`, 8-vector groups `acc = N(acc, N(N(N(v0,v1),N(v2,v3)), N(N(v4,v5),N(v6,v7))))`,
   then single vectors, then `npyv_reduce_{max,min}n` (ANY NaN lane → the canonical +NaN `0x7ff8…`/`0x7fc00000`, else
   halves folded lo-vs-hi down to lane 0), then the SSE scalar tail; `N(a,b) = isnan(a) ? a : (a > b ? a : b)` — the
   SECOND operand wins a tie (on x86 NumPy overrides the scalar C macro with this same SSE rule). So the ±0 sign and the
   NaN payload are schedule-dependent: the fused 4-accumulator fold matched the VALUE only (block C folded signed zeros
   into +0 for exactly that reason). `DefaultEngine.Evaluate.MinMax.cs`: `NumPyMinMaxReduce` is a lane-exact port shaped
   as `Vector256<T>` on EVERY host (the lane structure IS the contract; the portable path reproduces it without AVX),
   the float lane op NumPy's own `vmaxp*`+`vcmpordp*`+`vblendvp*` under `Avx.IsSupported` (RyuJIT never swaps the
   operands of a floating-point max/min intrinsic; the tie cases would show a swap) — the portable compare+select
   form cost ~7 µops a vector vs 3 (bare-leaf max @100K 11.1 → 8.7 µs) — and integers `Vector256.Max/Min`
   (order-free). `TryExactFlatMinMax` runs it over the buffer NumPy reduces: a bare dense leaf IN PLACE (memory order),
   a streamable computed child block by block (8 KB blocks = a whole number of groups, so only the last is partial),
   a non-streamable float child MATERIALIZED (as NumPy materializes it); a non-dense bare leaf declines (NumPy's strided
   reduce is its 8-accumulator SCALAR unroll — not ported, the fold stays) and so does a non-streamable integer child
   (order-free — the fold is exact). **Verified:** a NumPy 2.4.2 replay oracle (1,568 cases: f64/f32/i32/i64/u8/i16 ×
   max/min × {all-±0, sparse lane-structured ±0, NaN-laced with payloads, tail-only NaN, random} × {C 1-D, C 2-D, F 2-D,
   strided, broadcast, mixed C×F} × leaf/product, n 1…65,543) — 0 misses on every exact route (the 19 misses are the
   declined strided leaves, identical to the old fold's); the old fold missed 360. `NDEvaluateMinMaxTests` (5 — NumPy-
   probed literals for lane-structured ties, cross-stream-block ties, canonical vs payload NaN, lone element; every
   route vs an INDEPENDENT test-side lane model; integers at every width with extremes; declines; keepdims/out/F-offset;
   MUTANTS: AVX operand swap, no canonical NaN, per-block restart, dropped view offset, reversed horizontal pairing — all
   red). The evaluate corpus gained block **C9** (376 cases, +1.9 MB — lane-structured ±0 pools across the 32/64-group
   and 1024/2048 stream-block edges, both sign assignments, leaf/product/strided/negstride/F 2-D, NaN pools; signed
   zeros KEPT), appended LAST so no existing case id moves. **Measured (pinned, alternating order, best-of; NPY/NS vs a
   pinned NumPy twin; "fold" = the old path):** `max(a*b)` f64 1K/100K/4M 4.4 / **1.55** / 3.5 (fold 100K was 0.87),
   `min(a*b)` f64 100K 1.53 / 4M 3.4, `max(a*b)` f32 100K 1.15 / 4M 4.5, `max(a*b)` i32 4.0 / 5.3, bare `max(a)` f64
   1.03 / ~parity at 4M (both sides read-bandwidth bound); a strided float product (materialized) 1.27 / 1.81 — 0.73× the
   old fold at 4M, the temp NumPy pays too. The two parity fixes below are speed-ups as well: a transposed-3-D flat sum
   streams in memory order at NPY/NS 1.27 / 2.06 (100K / 4M) — 2.8× / 11.2× the old logical-order materialize route
   (10.4 → 0.93 ms at 4M, the strided transposed copy gone) — and the chunked integer mean is 4.7 / 1.10 / 1.16.
   **Three parity bugs it exposed, all fixed:** the C9 teeth check (force the fold → must go red) PASSED at first,
   because `MisalignedRegistry`'s generic "(5) unary ~ULP" branch — keyed only on one operand — gave EVERY
   single-operand fused tree a blanket 2-ULP excuse (+0/−0 are "within 2 ULP"); scoping it away from `evaluate` (which
   has its own E1–E5 policy) surfaced 18 hidden real divergences: (1) a TRANSPOSED 3-D flat Sum/Mean/Prod was
   materialized and reduced in LOGICAL C order, while NumPy's K-order iterator coalesces a dense block into one loop over
   MEMORY order (probed 0/400 misses vs 231/400) — `CanStreamChild(allowPermuted: true)` +
   `IsSharedDensePermutation` now stream any shared dense axis permutation in memory order for every FLAT caller (sum /
   mean / prod / bool folds / min-max / weighted average's flat sums; the axis streams keep C/F); (2) a flat `Mean` of
   an INTEGER / bool child rode the 4-accumulator fold, while `np.mean(int)` is `add.reduce(child, dtype=float64)` —
   a BUFFERED cast, so `+0.0 + Σ pairwise(np.getbufsize()-element chunk)` (probed: whole-array pairwise missed 11/60;
   and `np.setbufsize` changes 11/20 NumPy results, so the chunk follows `np.getbufsize()` — the `np.bufsize` docs'
   "never changes a result" claim was corrected) — `ExactIntegerMeanSumInto` streams
   `NDExprProgram.ChildAsFloat64Program` chunk by chunk (materializes it for non-streamable layouts). With the excuse
   scoped, the forced-fold build fails 52/29,563 cases — C9 has teeth. Gates: `NDEvaluateStreamingTests` 13 (+2:
   permuted-dense flat reductions == the base's bits, with a teeth check that logical order differs; NumPy-probed
   integer means at bufsize 8192 AND 4096 on both routes; MUTANTS no-permuted / whole-array pairwise / dense-check-
   always-true all red). Still on the scalar fold: integer `Sum`/`Prod` (flat and axis — flat already 3.4× NumPy); the
   axis `Min`/`Max` and the non-dense bare leaf's min/max went NumPy-exact in "Axis + flat Min/Max everywhere" below
   (the float axis `Sum`/`Mean`/`Prod` stream since M3, the axis bool folds since the paragraph below).
   **Axis bool folds STREAMED (LANDED):** the axis `Any` / `All` / `Sum`-of-a-bool-child (`CountNonzero`) was the worst
   remaining gap-map cell — `any(a>b, axis=0)` NPY/NS 0.37 @100K, 0.23× NumSharp's own unfused `np.any`, the seeded
   per-output scalar fold paying one iterator pass for nothing an order-free reduction needs. `DefaultEngine.
   TryStreamAxisBoolFold` (`DefaultEngine.Evaluate.Stream.cs`, hooked into `EvaluateAxisReduce`'s non-divert branch;
   null = nothing allocated, the fold runs as before) streams the same bool child the flat fold streams
   (`NonzeroBoolOperandProgram ?? ChildElementwiseProgram`; a bare bool leaf read IN PLACE at `Address + offset`) over
   all-C or all-F operands (`CanStreamChild(allowF: true)`; an F walk is the C walk of the reversed dims — reducing axis
   k is `outer = Π dims[k+1:]`, `inner = Π dims[:k]`, and a multi-D result is allocated F-contiguous, NumPy's K-order
   output), in two shapes picked by `inner`: **rows** (`inner == 1`, the reduced axis contiguous) — `8192 / axisSize`
   rows per block, each row folded by the BCL span scans (`IndexOfAnyExcept(0)` / `IndexOf(0)` / `axisSize −
   Count(0)`), rows longer than a block chunked with a per-row early exit; **slabs** (`inner > 1`) — the axis-k slab of
   every output is one contiguous `inner`-byte run, so an accumulator row takes `acc |= nz` / `acc &= nz` / `cnt += nz`
   per slab with `nz = ~(v == 0) & 1` (any nonzero byte normalized, a vector at a time), Any / All stop once a block
   leaves every output of the current outer index decided (`SlabDecided`), and the count runs in BYTE counters flushed
   into the int64 result every 255 slabs (a byte holds 255 additions; the result's slab row is CLEARED first — the
   allocation is uninitialized, and the first draft read garbage there). Exact BY CONSTRUCTION (OR / AND / an integer
   count are order-free), so the early exits and any walk order are legal. Declines: other kinds, an empty reduced axis
   or empty output (the fold owns the identities), strided / broadcast / mixed-order operands, `inner > int.MaxValue`.
   **Measured (pinned P-core, both ON/OFF orders, best-of; NPY/NS vs a pinned NumPy 2.4.2 twin, then the speed-up over
   the old fold):** `any(a>b, axis=0)` **12.4** @100K (33×) and 1693 @4M (every column decides within 2 rows);
   never-deciding `any(a>1e9, axis=0)` 1.36 / 1.58 (4.2× / 1.9×); `all(a>b, axis=0)` 12.2 / 1.25; `count_nonzero(a>b,
   axis=0)` 4.8 / 2.14 (3.7× / 1.34×); `any(a>b, axis=1)` 1.50 / 1.30 (4.7× / 1.5×); `count_nonzero(a>b, axis=1)`
   5.1 / 1.98; bool leaf `any(none, axis=0)` 2.5 / 1.04 (11× / 20×), `count_nonzero(mask, axis=0/1)` 10.6 / 18 @100K
   and 50 / 53 @4M (≈19× the fold); F operands `any(fa>fb, axis=0)` 1.08 / 3.5 and `count_nonzero(fa>fb, axis=1)`
   4.2 / 5.2 (the fold took 28 ms at 4M for F — **14×**). No cell below NumPy. Gates: `NDEvaluateStreamingTests` 16
   (+3: stream == fold byte-for-byte over rows / long rows / slabs / slabs longer than a block, every axis of 3-D
   blocks, C and F walks, computed children with a 0-d parameter and bare bool leaves, DENSE all-true counts at the
   255 / 256 / 511-slab flush edges, the deciding element at every row position incl. both sides of a block boundary,
   the slab early exit both taken and refused, keepdims / `out=` / an F column block at offset 42 / declines; 15/15
   MUTANTS red — count row not cleared, flush at 256, flush dropped, both slab early exits loosened, view offset
   dropped, F result allocated C, F walk treated as C, long-row single chunk, inverted row / chunk counts, Any folded
   with AND, three wrong block/chunk offsets), 434 NDEvaluate/NDExpr/Evaluate/post-pass units on net10.0 AND net8.0,
   FuzzMatrix 100/3/1 unchanged (the leak audit's 38 pre-existing logspace/geomspace/piecewise families, none new),
   full NumSharp.Tests 15,572 green (the 25 failures are the OpenBLAS-not-staged Examples demos). Two follow-ups left on
   the table: rows-mode Any/All exits per BLOCK, so an early-deciding row still pays its whole compare (`any(a>b,
   axis=1)` @4M is at parity with the unfused chain, 1.3× NumPy) — producing a short prefix of each row first would
   close it; and the leaf slab combine re-reads/re-writes the accumulator per slab, where OR-ing two slabs before the
   accumulator touch would cut the `any(none, axis=0)` 4M cell's traffic by a quarter.
   **Axis + flat `Min`/`Max` everywhere — engine AND np.evaluate — NumPy-EXACT (LANDED):** after the contiguous flat
   schedule above, the rest of the min/max surface still matched NumPy's VALUE only (the ±0 sign and the NaN payload
   were schedule-dependent): every AXIS reduction (engine `np.max`/`np.min`/`np.ptp` and the fused `Max(x, axis)`), the
   engine's FLAT `np.max`/`np.min`/`np.ptp`, and the fused form's non-dense bare leaf. All now run NumPy's own
   schedule (`Backends/Default/Math/Reduction/Default.Reduction.MinMax.Exact.cs`, `NumPyMinMaxReduce`), established
   first by reading `nditer_constr.c`/`reduction.c`/`loops_minmax.dispatch.c.src` and then pinned by probes:
   - **Axis:** NumPy copies each output's first reduced element (`PyArray_CopyInitialReduceValues`), then walks a
     K-order `NpyIter` built with `NPY_ITER_DONT_NEGATE_STRIDES` (logical order along every axis). Its innermost axis is
     the extent > 1 axis with the smallest |stride| of the input (the output's stride-0 axis never votes), a TIE going
     to the LATER axis (the stable insertion sort starts from reversed C order). If that is the reduced axis: **ROW
     mode** — every output is one inner-loop call over its row after the copied element, `simd_reduce_c` for a
     contiguous row (the lane-exact schedule above, `ReduceRow`), NumPy's scalar 8-accumulator unroll for a strided one
     (`ReduceStridedRow`). Otherwise **SLAB mode** — the first slab is copied into the output run, and every further
     reduced index folds in elementwise (`CombineRun`: `o = N(o, x)`, the per-element sequential fold). Outer axes that
     continue the run contiguously in BOTH operands merge into it (NumPy's coalescing), and the outer walk goes in memory
     order for locality — each output's operation sequence is fixed by the mode, not by the walk.
   - **Flat:** the 0-d output makes every axis a reduce axis and the iterator coalesces axes SIGNED (`stride·extent ==
     next stride`, so a reversed block is one run of stride −1 and a stepped one a run of stride 2 — `FlatIteratorAxes`).
     One run → one call (`simd_reduce_c` contiguous, the 8-accumulator unroll strided). Two or more runs →
     `npyiter_find_buffering_setup`'s cost model (`FlatBufferingDim`, ported line for line — including its
     `size >= maximum_size` stop): best dim 0 (the innermost run is longer than half the buffer) → UNBUFFERED, one call
     per run, the first skipping the copied element (`ReduceFlatRows`); otherwise BUFFERED — the elements are copied in
     iteration order into fills of `coresize · ⌊bufsize/coresize⌋` elements (or the whole best block when it fits),
     never crossing the end of the outer block, and EACH FILL IS ONE CONTIGUOUS `simd_reduce_c` CALL
     (`ReduceFlatBuffered`) — so a NaN met anywhere in a fill comes back canonical, a running NaN is re-canonicalized by
     the next fill however short, and a ±0 tie can be decided by where a fill boundary falls. The fill size follows
     `np.getbufsize()` (thread-local; `np.setbufsize(4096)` changes NumPy's answer too, probed). Scratch: a 2 KB
     `stackalloc`, else `NativeMemory` freed in `finally`.
   - **The strided unroll's lane order** (m0..m7 → `(0,1)(2,3)(4,5)(6,7)` → `(01,23)(45,67)` → `(0123,4567)`, then
     `r = N(r, m)`, then the scalar tail): NumPy's lane op `N(a, b) = isnan(a) ? a : (a > b ? a : b)` is ASSOCIATIVE as a
     selection (first NaN wins, else the LAST max-equal element), so the whole schedule is "the latest position in the
     order x[0], lane 0 … lane 7, tail wins a ±0 tie" — probed on 1,056 cases (every lane pair × both zero orders ×
     strides −1/2/−3 × max/min × f64/f32 + x[0]-vs-lane + lane-vs-tail): 0 disagreements. Consequence: the eight
     accumulators can be VECTOR LANES (`ChainStrided` — one `Vector256` of 4-byte lanes, a lo/hi pair of 8-byte lanes)
     fed by `GatherRun` (a stride of −1 is ONE contiguous load + a lane reverse — `Permute4x64` / `PermuteVar8x32`;
     other strides a scalar-load gather), exact at any width because the lanes never mix before the combine.
   **Routes.** Engine: `Default.Reduction.AMax/AMin` try `TryExactAxisMinMax` / `TryExactFlatMinMaxScalar` first (flat
   only when no `dtype=` cast is requested), falling back to the old IL kernels on decline; `np.ptp` composes them and
   inherits. np.evaluate: `TryExactFlatMinMax` — a dense bare leaf (C/F/any transpose) keeps the in-place contiguous
   schedule, any other non-broadcast bare leaf goes through `TryExactFlatMinMaxArray` (the flat walker above); axis —
   `TryExactAxisMinMaxEval` (`DefaultEngine.Evaluate.MinMaxAxis.cs`): a bare leaf reduces IN PLACE through
   `ReduceAxis`; a computed child over all-C / all-F operands STREAMS (`StreamExactAxisMinMax` — rows reduced per
   produced block, a row longer than a block streamed as ONE unbroken `simd_reduce_c` via `StreamMinMax(start)`, slabs
   folded as produced with the output block kept hot); any other computed child is MATERIALIZED in the layout NumPy's
   own ufunc would allocate (`MaterializeChildNumPyLayout` — `npyiter_new_temp_array`'s K-order permutation, never a
   negated stride) and then reduced by `ReduceAxis`; the flat materialize route uses the same layout. Declines (the old
   paths, value-exact): a broadcast operand (NumPy's stride-0 handling is not ported), Half / Decimal / Complex / Bool /
   Char (not served), rank > 64, empty. `NDExpr.DisableExactMinMax` (thread-static) turns every exact route off;
   `NDExpr.ExactMinMaxRuns` counts engagements.
   **Making the exact paths fast** (exactness first cost 0.6–0.7× the old kernels on strided/reversed layouts): the
   vectorized strided unroll and gather/reverse copy above; a **NaN-free fast fold** in `FoldGroups` — NumPy's blended
   `N` equals the plain `vmaxp`/`vminp` on every lane whose FIRST operand is not NaN, so a group whose eight vectors are
   proven NaN-free by four `vcmpunordp` (one per vector pair) folds with one instruction per op instead of three, and the
   first group holding a NaN breaks out BEFORE touching the accumulator and continues on NumPy's blended rule (only
   entered with a NaN-free seed); the horizontal step (`ReduceLanes`) runs the no-NaN cascade as an SSE `P128` ladder
   after its any-NaN check; one-group-ahead `Prefetch0` in the group loop.
   **Verified:** three NumPy 2.4.2 replay oracles — axis `.npy` (600 cases: f64/f32 lane-structured ±0 / NaN-payload
   pools × 11 shapes × C/F × every axis × max/min — engine, fused leaf, fused child: 0 misses), axis round 2 (5,764
   cases: reversed / stepped / sliced / transposed-permutation / rows past the buffer + computed children over mixed
   layouts; every exact route 0 misses — the only misses are the UNFUSED eager route `np.max(x * 1, axis)` on
   permuted / `F*negcol` products: 81 + 37 + 3, NumSharp's eager elementwise writes its result in C order where NumPy's
   ufunc keeps the K-order layout, so the reduction then walks different memory — an elementwise-layout gap, not this
   lever), flat (3,136 cases across every route incl. the engine: new 0 misses, the old paths 360). `MinMaxExactScheduleTests`
   (25 — every literal NumPy-probed with the same base+view recipe: row-strided lane ties + payload, contiguous rows with
   the fast fold then a NaN group, the slab sequential fold, reversed / stepped runs, buffered rows canonicalizing a
   NaN, rows longer than half the buffer staying unbuffered, a tie decided by a fill boundary, the next fill
   re-canonicalizing a running NaN, fills following `np.setbufsize`, F-order dense = one memory-order call, a lone
   element's bits, integers at every layout, keepdims / ptp, the copied-element skip in both modes, fills as whole
   cores, fills stopping at the outer block end, NaN in each vector of a fast-fold group, a NaN seed, short calls
   canonicalizing the seed, the disable hook, declines, the axis-order / coalescing and cost-model ports incl. the
   `size == bufsize` boundary, and the every-lane-pair tie test) + `NDEvaluateMinMaxTests`' decline test updated (a bare
   strided leaf now ENGAGES the walker; a bare broadcast leaf still declines). **Mutation-tested behind a green-baseline
   gate:** 22 targeted mutants (axis-order tie `>=`, unsigned coalescing, cost `<` vs `<=`, uncapped bufsize, the stop
   test `>` vs `>=`, fills not whole cores, the copied-element skip dropped in rows / buffered / one-run modes, no
   block-end flush, 4-/8-byte lane extraction swaps, the reversed load off by one, both reverse permutes, a permuted
   buffer copy, broadcast accepted, short calls not canonicalizing, the fast fold's NaN detector inverted / missing a
   pair, an unguarded NaN seed, swapped `P256` operands) — all 22 killed, each by the test aimed at it. **The gate
   caught a false green on the way:** the first run reported 22/22 killed, but a stale assertion in the decline test
   (written when a bare strided leaf still declined) failed on EVERY run, so every mutant "died" of it; with a baseline
   that must pass unmutated, five survived (the lane-pair swaps and both reverse permutes, invisible to ties that sat in
   one lane or in lanes 2/3 only, plus the cost-model stop boundary, which fills the same buffers either way) — hence the
   every-lane-pair test and the unit pin. Full NumSharp.Tests: net10.0 15,607 / net8.0 15,609 green (the 25 failures are
   the OpenBLAS-not-staged Examples demos, unchanged).
   **Measured** (pinned P-core, best-of, two rounds; NPY/NS vs a pinned NumPy twin; "old" = `DisableExactMinMax`, the
   value-exact previous kernels): FLAT @100K — C f64 max/min 1.61 / 1.64 (old 1.41 / 1.45), f32 1.49, i32 1.45, i64
   1.14, u8 2.30, F 1.85 (0.63), transposed 1.61 (0.61), 3-D permutation 1.82 (0.63), stepped 1.42 (1.16), reversed
   columns 1.91, reversed 1-D 1.45, ptp C / reversed columns 1.52 / 1.82. AXIS @100K — f64 ax0/ax1 1.61 / 2.16 (old
   1.01 / 0.88), f32 2.23 / 1.97, i32 2.11 / 2.15, i64 ax0 1.67, u8 ax1 2.23, F 1.87 / 1.39, transposed 2.17, stepped
   ax1/ax0 2.13 / 1.72 (0.72 / 0.52), reversed rows 1.68, permutation 2.23 (0.64), ptp 2.09, fused `max(a*b, 0/1)`
   1.97 / 2.50 (0.58 / 0.85), `min(i*i, 0)` 5.20, `max(a, 0)` 1.55 (0.28), `max(aF*b, 0)` 1.28 — no 100K cell below
   1.14. Where the old kernel was faster (reversed / reversed-column flat cells, reversed axis rows: it read memory
   forward, which exactness forbids) the new one still beats NumPy 1.45–1.91×. @4M (L3/DRAM-bound, on a host carrying
   other load — ±25 % run to run) the fused cells stay far ahead (`max(a*b, 0/1)` 3.26 / 3.29, `min(i*i, 0)` 7.70) and
   the engine cells sit 0.70–1.19: the ones below NumPy are almost all at or above the old kernels' own ratio (flat C
   i32 0.77 / u8 0.70 / f32 0.87 vs old 0.76 / 0.70 / 0.88; axis i32 ax0 0.70 vs 0.73; the F / step / permutation cells
   up from 0.45–0.65), i.e. pre-existing bandwidth gaps — follow-ups below.
   **Follow-ups:** (1) SLAB mode re-reads and re-writes the output run once per reduced index; folding SEVERAL reduced
   indices per pass (`o = N(N(N(o, x_k), x_k+1), …)`) keeps every element's operation order, so it stays exact, and a
   raw-loop probe measured 8 per pass at 1.8–2.4× the per-slab loop on DRAM-bound f64 (`max(f64, axis=0)`,
   `max(F, axis=1)`) — the next lever. (2) The L3-bound 4M flat / row cells (4-byte and 1-byte lanes) trail NumPy's
   identical schedule by 0.70–0.87 (NumSharp's buffers are 16-byte aligned, so every other 32-byte load splits a line;
   the one-group-ahead prefetch is unmeasured there). (3) The eager elementwise output-layout gap the oracle exposed.
   (4) Broadcast operands still decline.
4. The stat / delegating / average reductions (`NanMean`/`Var`/`Std`, `Ptp`/`NanMin`/`NanMax`/`ArgMax`/`ArgMin`,
   weighted `Average`) still materialize their child(ren) — streaming candidates (Var is two-pass: recompute vs store).
   **A 2026-09-23 gap map** (fused vs NumPy, pinned, best-of; probe `red_gap2.cs` + twin) ranked them: weighted
   `Average` was the worst fused reduction by far — 0.47× NumPy @100K / 0.66× @4M and 0.06–0.19× NumSharp's own
   unfused `np.average`; `Any`/`CountNonzero` 0.25× @100K (lever 3's scalar bool fold); `Min`/`Max` 0.72–0.80× @100K;
   `ArgMax`/`Ptp` ≈ 0.95× @100K (1.7–1.9× @4M); `Var` 5.8× / int `Sum` 3.4× already ahead.
   **Weighted `Average` STREAMED (LANDED):** Σ(v·w) and Σ(w) run as two M3 Sum streams over the children
   `NDExprProgram.AvgNumeratorProgram` (`Cast(v,rt)·Cast(w,rt)` — a side already at `rt` stays uncast so the kernel keeps
   its SIMD body; NumPy's `np.multiply(a, wgt, dtype=result_dtype)`) and `AvgDenominatorProgram` (`Cast(w,rt)`) —
   `DefaultEngine.TryStreamWeightedAverage`; no product, no weights copy, no materialized values. Bit-identical to the
   materialize route for C operands (flat + every axis + keepdims, incl. int·int multiplied at f64). **It also FIXED a
   parity bug:** the materialize route copied F inputs to C (`ToResultContig`) and summed in C order, while NumPy's
   `np.average` keeps the product in K order and sums MEMORY order — a NumPy 2.4.2 `.npy` oracle (C/F × flat/axis 0/1
   over 37×129 / 300×7 / 5×2000, wide-magnitude pools) had 6 of 9 F cases diverging (up to 98/129 elements); now
   18/18 bit-exact, all streamed. **Measured (stream vs old, pinned, alternating order):** flat 2.2× @1K / 3.6×
   @100K / 3.9× @4M (17.8 → 4.5 ms; NumPy 11.7 → NPY/NS 0.66 → 2.6), C axis 3.2–4.2×, F axis 5.0–7.7×. Follow-up:
   the two streams read `w` twice — one joint recursion (both sums per leaf) would cut the 4M call to ~3 ms.
   **The F-aware axis stream (LANDED with it):** `TryStreamAxisReduce` accepts ALL-F operands — an F-contiguous array
   is the C-contiguous array of its reversed dims, so reducing axis k walks memory as the C reduction of axis nd-1-k
   (`outer = Π dims[k+1:]`, `inner = Π dims[:k]`, F-ordered result = NumPy's K-order output), exactly the schedule
   NumPy's reduce iterator runs over the F-contiguous child `np.sum(a*b, axis)` materializes. `EvaluateAxisReduce` routes
   the strict-F corner there first (child-dtype accumulator when it streams; the fold otherwise, as before) — closing
   the M2 strict-F E1 corner: 80/80 strict-F Sum/Mean/Prod cases (f64/f32/c128, 2-D + 3-D, every axis incl. size-1)
   bit-exact vs a NumPy 2.4.2 `.npy` oracle, where the old fold diverged on 61; and 4–11× faster (4M F `sum(axis=0)`
   24.5 → 2.2 ms). `MisalignedRegistry`'s E1 predicate drops its strict-F exception (+ the now-dead
   `IsStrictFContiguousMultiD`), so the corpus's 84 f32/f64/c128 strict-F axis Sum/Mean/Prod cases are now ENFORCED
   bit-exact (they pass). Gates: `NDEvaluateStreamingTests` 9 (+3: F axis == transposed-C reduction bit-for-bit over
   f64/f32/c128 × 2-D/3-D × every axis incl. size-1 × Sum/Mean/Prod; mixed-order fallback; weighted Average C
   streams == materialize; F Average == transposed-C Average), 445 NDEvaluate/NDExpr/Evaluate/post-pass units,
   FuzzMatrix 100/3/1 unchanged (evaluate tier green with the tightened excuse).
   **`ArgMax` / `ArgMin` STREAMED, over a new ENGINE axis argmax (LANDED):** the fused form materialized its child
   and handed it to `np.argmax(child, axis)`, whose axis path was a per-output IL kernel that recomputed every
   output's base offset with a div/mod chain and walked the axis with a scalar strided loop — so axis 0 of a C block
   touched one element per cache line per step and a contiguous axis never used the SIMD row kernels the FLAT argmax
   has (engine `np.argmax(a, 1)` f32 0.26× NumPy @100K; NumPy itself transposes the axis last and copies the operand
   C-contiguous before one SIMD argmax per row, so its axis-0 cells pay a full copy). Three pieces:
   (1) **`TryExecuteAxisArgFast`** (`Backends/Default/Math/Reduction/Default.Reduction.ArgAxis.Fast.cs`, hooked into
   `ExecuteAxisArgReduction` — one hook serves argmax AND argmin; null = declined, nothing allocated, the IL kernel
   runs as before) folds the axis IN PLACE with the loop order the STRIDES pick, never the output order: the non-axis
   dims coalesce (an outer run merges into the next when it continues it in memory), then **Zero** (axis stride 0 —
   every answer 0, the result allocated zeroed and nothing read), **Rows** (axis contiguous — the flat SIMD row kernel
   per output, an inline scalar fold under 32 elements), **Slab** (a non-axis run at least as tight as the axis: chunks
   of ≤ 1024 lanes fold together row by row, each lane's running best in 8 KB of L1 scratch, one `Vector256` compare
   per row for a ±1 run — a −1 run is flipped first — ending on one OVERLAPPING vector instead of a scalar tail
   (re-folding a lane with the same row is idempotent), the row index written only for improved lanes and as whole
   vectors when all improve; a wider run folds scalar in memory order), or **AxisWalk** (the axis is the tightest
   non-contiguous stride — a scalar strided fold per output). The rule is NumPy's `@TYPE@_argmax` in LOGICAL axis
   order whatever the memory walk: first occurrence of the extreme under IEEE compare (−0 == +0, the earlier wins),
   the first NaN winning argmax AND argmin (`!(v <= best)` then stop), integers strict, bool argmax = first True /
   argmin = first False (0 when none — tracked as "undecided" lanes so a bool slab stops once every lane has met its
   deciding byte). Static-abstract rule structs (`IArgRule<T>`) specialize one fold per (dtype, rule). Serves Bool /
   the 8 integer widths / Char (as UInt16) / Single / Double; Half, Decimal, Complex and rank > 64 (the plan is
   stack-allocated per rank) decline. (2) **The 64-bit single-pass tournament** (`DirectILKernelGenerator.
   Reduction.Arg.cs`: `ArgMaxInt64Tournament` / `ArgMinInt64Tournament`, a port of NumPy's `simd_argmax_{s64,u64}`):
   the generic two-pass row kernel (a `Vector256.Max` pass, then an equality scan) ran `Vector256.Max<long>/<ulong>`,
   which has NO AVX2 instruction — the tournament is 2.0–2.6× faster at 1K–100K and 1.0–2.2× at 4M wherever the
   maximum sits (pinned A/B), and lifts every int64 Rows fold with it. A 32-bit tournament was implemented and measured
   SLOWER in cache (0.30–0.85× the two-pass — `Vector256.Max<int>` is one instruction and the equality scan exits
   early), so 8/16/32-bit lanes keep the two-pass; the routing is gated on `Vector256.IsHardwareAccelerated` (an ARM64
   host keeps its Vector128 two-pass). (3) **`TryStreamArgReduce`** (`DefaultEngine.Evaluate.ArgStream.cs`, tried first
   in `EvaluateDelegatingReduce`; the materialize block is unchanged behind it): a BARE leaf is the operand itself, so
   the engine argmax runs on it directly (every layout, no copy); a COMPUTED child over all-C operands
   (`CanStreamChild(allowF: false)` — only there is memory index k the logical C index k NumPy's argmax visits) is
   produced block by block into the 8 KB scratch and folded as it arrives — **rows** (the reduced axis innermost, or
   flat = one row of everything): many short rows per block through the row kernel, a row longer than a block chunk by
   chunk with the chunk winners combined in increasing order by the SAME strict rule (a tie in a later chunk never
   replaces; the row stops once decided); **slabs** (an outer axis): the engine's own `SeedSlabChunk` /
   `FoldSlabRows` fed with the produced rows, wide slabs one 1024-lane chunk at a time. F / strided / broadcast /
   mixed-order children, Half / Decimal / Complex children, empty children and an out-of-range axis decline (the
   materialize route keeps NumPy's error). **Verified:** engine 137,268 axis + 61,194 flat checks bit-identical to an
   independent reference AND the legacy kernel (12 dtypes × C/F/transposed/reversed/strided/offset/broadcast/3-D ×
   every axis × ties/NaN/±0/extremes/monotone); stream 211,104 checks bit-identical to the materialize route and to
   the engine over the materialized child (8 dtypes × 14 shapes incl. rows/slabs crossing every block and chunk edge ×
   C/F/CF/strided/T/offset layouts × leaf / `a−b` / `a>b` / 0-d-param children × every axis × keepdims × `out=` int64
   AND float64), every route choice as expected (F/strided/mixed decline, C and every leaf stream), 24 edge cases
   (empties, 0-d, bad axes, broadcast operands, a `Call` node throwing mid-stream) returning the same value or the
   same exception with streaming on and off. **Measured (pinned P-core, best-of, both processes' two rounds; NPY/NS
   vs a pinned NumPy 2.4.2 twin; `argeval_bench.cs` + `.py`):** fused `argmax(a*b)` f64 flat 2.60 / **2.99** (@100K /
   4M), ax0 8.81 / **7.51**, ax1 2.47 / 3.10; f32 flat 1.18 / 3.98, ax0 6.17 / 8.60, ax1 1.34 / 3.59; i32 flat 3.57 /
   4.75, ax0 8.17 / **9.04**, ax1 2.17 / 4.22; i64 flat 2.08 / 2.88, ax0 **13.8** / 6.50, ax1 2.24 / 3.01; bool child
   `argmax(a>b, axis=1)` 1.21–2.03 — **no cell below NumPy**, where the pre-lever route (materialize + legacy axis
   kernel) had 16 of 48 under parity (down to 0.20× — `argmax(a>b, axis=1)` f32 @100K); the stream is 1.1–9.3× the old
   route and 1.1–2.7× NumSharp's own unfused `np.argmax(a*b, axis)`. The engine alone (`np.argmax(a, axis)` on a
   precomputed array, `argaxis_bench.cs`): ax0 f64 4.4 / 8.6–8.8, f32 ax0 @4M 16.4, i32 16.4, i64 10.5–11.8,
   8/16-bit 10.6–16, bool 95–106, F ax1 13.5–20, strided 2.1–3.5, 3-D middle axis 6.1–6.8 (1.0–137× the legacy
   kernel). Still at ~parity on the engine alone: Rows over a MEMORY-BOUND 4M buffer (f64/f32 ax1 0.84–1.04, i32 ax1
   0.77–0.80 — the two-pass row kernel's second pass costs ~60 µs / 4M), the same per-row kernel NumPy runs.
   **Gates:** `AxisArgFastPathTests` (8 — every served dtype through every walk incl. a padded 3-D block whose runs must
   NOT coalesce, 1024-lane chunks and overlapping vector tails, monotone rows, deciding values at every chunk / vector
   boundary, NumPy-probed literals, broadcast axes, declines, tie-heavy rows through every flat row kernel, the int64 /
   uint64 tournament's ties, extremes and unsigned compare) + `NDEvaluateStreamingTests` `ArgReduce_*` (4 — computed
   children of 7 dtype families × rows / long rows / narrow and wide slabs × every axis × keepdims vs the materialize
   route and the engine, the long-row chunk combine's first occurrence, leaves / declines / `out=` / pinned values, a
   throwing child); **33/33 MUTANTS red** (14 engine — flip base, float/int vector ties, both NaN guards, seed not
   cleared, final vector dropped, short-row / axis-walk strides, over-eager coalescing, both bool early exits, chunk
   and row indices; 11 stream — chunk combine non-strict, chunk offset, decided break inverted, rows produce, both slab
   row indices, the wide-slab lane offset, leaf argmax/argmin swapped (axis + flat), F children streaming, keepdims
   dropped; 8 tournament / row kernel — tie rule, u64 signed compare, accumulator move non-strict in the f32 / f64
   helpers and the int64 tournament, argmin tail). The first sweep's T3 SURVIVED — its `sed` hit the pre-existing f32
   row helper, whose cross-block ties no test reached (the pools' float max was a lone ±inf / NaN) — which is what
   added the tie-heavy row-kernel test. Full NumSharp.Tests net10.0 15,584 green (the 25 failures are the
   OpenBLAS-not-staged Examples demos, unchanged), the affected classes (832) green on net8.0, FuzzMatrix 100/3/1 (the
   1 = the leak audit's same 38 pre-existing logspace/geomspace/piecewise/loadtxt/frombuffer/fromfile families).
   **Follow-ups:** a bool child's rows produce whole rows before the first-True scan — `argmax(a>b, axis=1)` @4M streams
   at 2.17 ms vs the unfused chain's 2.00 (still 1.21× NumPy); a prefix-first produce (the same follow-up as the axis
   bool folds' rows) would let an early True skip the rest of the row. Pre-existing, NOT fixed here: the engine's
   `ReduceArgMax` wraps an out-of-range negative axis (`while (axis < 0) axis += ndim` — `np.argmax(a2d, -3)` answers
   instead of NumPy's AxisError) and reports a positive overshoot as a bare `ArgumentOutOfRangeException`; the fused
   form inherits both through the materialize route (the stream declines any out-of-range axis).
5. Known, unchanged: fused Half arithmetic 0.66–0.74× the engine's tuned unfused chain (P5.1 ceiling note).
6. **Engine divergence found (not fixed, not evaluate-specific):** `np.sum` over a SIZE-1 axis of `-0.0` returns `-0.0`
   (`ReduceAdd` → `HandleTrivialAxisReduction` copies) where NumPy seeds the identity and returns `+0.0`; np.evaluate's
   axis path inherits it (the streaming path copies too, to stay byte-identical with the materialize path).
| **M4a** | `Any` · `All` · `CountNonzero` · `NanSum` · `NanProd` — the M4 kinds that ride the existing exact reductions, no new kernel | landed (this) | `Any`/`All` are two new bool ReduceKinds: result Boolean, per-element fold `OR`/`AND` (a bitwise op on a bool accumulator — **associative AND idempotent**, so the 4-accumulator unroll reorders it with zero drift, bit-exact by construction), identities `False`/`True` (so `Any([])==False`, `All([])==True`, where max/min raise "no identity"); the child is the nonzero-test `NotEqual(x,0)`. `CountNonzero`/`NanSum`/`NanProd` are **pure factory rewrites** — `Sum(x!=0)` (→int64 count) / `Sum(Where(IsNaN(x),0,x))` / `Prod(Where(IsNaN(x),1,x))` — so they inherit flat+axis+the M1/M2 divert's bit-exactness and correct empty identities for free, and an integer/bool child (no NaN; `IsNaN` all-false) passes through as the plain Sum/Prod (int64), matching NumPy. Wiring: enum + factories + `ResolveReduceResultType`(Any/All→bool) + `EmitFold`(Any→Or, All→And) + the per-chunk seed (Any→0, All→1) in both the flat and axis-PINNED kernels; host `ReduceUfuncName`/empty-seed/`seedOp` (Any→`ReductionOp.Any`, All→`ReductionOp.All`; `GetIdentity` already seeds false/true). Oracle: `_EV_REDUCE` + block C sweep (any/all/count_nonzero) + block C4 (a mixed zero/nonzero/NaN pool exercising the nan-skip, flat+axis, over the divert-exact dtypes; f16 + complex-prod absent — folded/E1) + block C4b (WIDE-magnitude NaN pools at N∈{129,1000} + a (3,200) axis case — where pairwise≠fold, so they match NumPy ONLY if NanSum/NanProd take the M1/M2 divert, not the fold; verified the pool discriminates by 0x…78000 vs 0x…77500). Gate: `evaluate.jsonl` 17,849 cases green (host-pinned) + `NDEvaluateTests.M4_*` (5 — values, dtypes, NaN-is-truthy, **empty identities** the corpus can't reach, axis). |
| **M4b** | `Ptp` · `NanMin` · `NanMax` — the ORDER-INDEPENDENT range / NaN-aware min-max kinds, host-delegated | landed (this) | Three new `NDExprReduceKind`s whose child is the RAW expression (not a `Where`-rewrite: `Ptp` is two reductions and cannot be one root reduce; `NanMin`/`NanMax` must be TYPE-AWARE, which a `Where(IsNaN(x),±inf,x)` rewrite could not be — it would promote an int child to float64). Because a min, a max and their difference are order-independent, `DefaultEngine.EvaluateReduce` intercepts them at its top (before the axis dispatch and before ANY fold kernel is compiled — `EmitFold` is untouched) and routes to **`EvaluateDelegatingReduce`**: materialize the child once (the M1/M2 route — `EvaluateCore(program.ChildElementwiseProgram, …)`), then reduce that buffer through the already-NumPy-exact engine reduction `np.ptp` / `np.nanmin` / `np.nanmax` (which carry the dtype-preservation + wrapping, the NaN-skip / all-NaN→NaN, and the type-awareness for free). `out=` follows evaluate's contract (validate the same_kind cast, reject a non-0-d out for a flat reduce, `np.copyto` the result). BIT-EXACT at EVERY dtype with **no E1 excuse** (order-independence, so a regression is red — the whole point of adding them here rather than under the summation-drift bucket). Wiring: enum (+3) + factories (flat/axis/keepdims) + `ResolveReduceResultType`(→child) + `ResolveAccType`(→result) + `ReduceUfuncName`(+3) + the host intercept + `EvaluateDelegatingReduce`; Structure.cs already hashes `_kind`, Vector.cs already returns false. Oracle: `_EV_REDUCE` +3 + `OpRegistry.Evaluate.cs` +6 cases + **block C5** (every layout {C-1d/2d, F-2d, negstride} × axis/keepdims × int/uint/float pools, NaN scattered for nanmin/nanmax and a NaN-propagation ptp case; int→int pools so uint64's wrap is well-defined parity) + a wide-N flat ptp. Gate: `evaluate.jsonl` **18,093** cases green (host-pinned, +244) + `NDEvaluateTests.M4Tail_*` (7 — dtype-preserving WRAP at int8/uint8, NaN propagation, NaN-skip + int passthrough, all-NaN→NaN, axis/keepdims, `out=`, and the **empty edge** the corpus can't reach: `Ptp([])` raises like `np.ptp([])`, `NanMin`/`NanMax([])` return the engine's 0-d NaN — a pre-existing `np.nanmin` divergence from NumPy's raise that evaluate faithfully mirrors rather than papers over). |
| **M4c-index** | `ArgMax` · `ArgMin` — the int64 INDEX kinds | landed (this) | Two new `NDExprReduceKind`s on the SAME host seam as M4b — `EvaluateReduce` intercepts them (before the axis dispatch / any fold kernel) and `EvaluateDelegatingReduce` materializes the child once, then reduces it through the engine's `np.argmax` / `np.argmin` (flat: the scalar-`long` overload wrapped into a 0-d int64 array). Two things set them apart from the M4b value kinds and both are handled: the result dtype is **int64** (`ResolveReduceResultType` returns `Int64`, not the child dtype), and the answer is not value-order-independent — it depends on the C-order tie/NaN rule (FIRST maximum / FIRST NaN wins). It is nonetheless bit-exact with **NO E1 excuse**, because the materialized child is exactly the fresh C-contiguous buffer NumPy's own argmax builds and reduces, so `np.argmax(materialized)` walks the same logical C-order and returns the same tie/NaN index (verified over C/F/transposed/negstride/strided layouts × ties × NaN). Empty raises (no identity, like `np.argmax([])`). Oracle: `_EV_REDUCE` +2, `OpRegistry.Evaluate.cs` +4, block **C6** (every layout × axis/keepdims × int (int64 result proves dtype-independence) / float / NaN pool for first-NaN-wins, + a fused child + a wide-N flat case). Gate: `evaluate.jsonl` **18,321** cases green (host-pinned, +228) + `NDEvaluateTests.M4c_*` (5 — int64 result & first-tie, first-NaN-wins, axis/keepdims, `out=`, empty raises). **Since 2026-09-23 they STREAM** (a computed child over all-C operands is folded block by block, a bare leaf goes straight to the engine argmax) over a strides-driven engine axis argmax — see "Perf review 2026-09-23" item 4; the materialize route above remains the fallback for every other layout. |
| **M4c-summation** | `NanMean` · `Var`/`Std(ddof)` — the summation kinds, host-computed over the exact pairwise sum | landed (this) | Three new `NDExprReduceKind`s host-computed by **`EvaluateStatReduce`** (intercepted at the top of `EvaluateReduce`, before the axis dispatch). They can NOT delegate to the engine's `np.nanmean`/`np.var` (whose flat sum is a drifting multi-accumulator fold) and are NOT tree rewrites (nanmean's divisor is a per-slab non-NaN count; var/std are a two-pass with a ddof divisor), so the host materializes the child once and reproduces NumPy's own algorithm op for op over the SAME pairwise (`PairwiseSumInto`) / axis (`ExactAxisSum`) primitives M1/M2 use: `NanMean = pairwise_sum(NaN→0)/count_of_non_NaN`, `Var = pairwise_sum((x−mean)²)/max(N−ddof,0)` with `mean = pairwise_sum(x)/N`, `Std = sqrt(Var)`. **Result dtypes:** NanMean follows Mean (int/bool/char→f64, f32/f64 preserved, complex128 preserved); Var/Std likewise EXCEPT a complex128 child yields a **REAL float64** (the deviations reduce as `\|x−mean\|²`). **`ddof` is carried on `ReduceNode` and is part of the program identity** (structural hash + `StructureEquals` + the legacy string key), so two trees differing only in ddof are two programs. **FLAT** folds the child buffer in MEMORY order (bit-exact for a C- OR F-contiguous child — `np.add.reduce` over a contiguous buffer iterates memory order); **AXIS** rides the engine's C-contiguous axis add.reduce, so the strict-F multi-D axis corner is inherited from M2 (excluded from the corpus). BIT-EXACT vs NumPy 2.4.2 for the **13 supported dtypes** with **NO E1 excuse**. **Half + Decimal are rejected** with a directed `NotSupportedException` — NumPy accumulates a float16 variance IN float16 (a widen-to-f32 computation is ~30 % off), which needs a float16 pairwise kernel that dtype lacks (the "Half not diverted" gap M1/M2 share); Decimal has no NumPy analog / pairwise kernel here. Oracle: `_EV_REDUCE` +3 (var/std carry ddof) + `OpRegistry.Evaluate.cs` +6 cases + **block C7** (every non-strict-F-axis layout × axis/keepdims × int/uint/float/complex pools + a NaN-skip pool for nanmean + ddof{0,1} for var/std + a FUSED child + WIDE-N flat teeth crossing PW_BLOCKSIZE for var/std/nanmean — a naive/multi-accumulator sum would diverge there). Gate: `evaluate.jsonl` **18,780** cases green (host-pinned, +459) + `NDEvaluateTests.M4cSum_*` (7 — the corpus-unreachable edges: empty→NaN, all-NaN→NaN, ddof≥N→+inf, complex-var-is-real, dtype tiers, axis/keepdims, Half/Decimal→NotSupportedException). |
| **M4c-average** | weighted `Average` = `Σ(values·weights) / Σ(weights)` over TWO operand trees | landed (this) | A NEW root node `WeightedAverageNode` (not a `ReduceNode`) — the first reduction over TWO sub-trees, which the single-child `ReduceNode` cannot carry. Host-computed by **`EvaluateWeightedAverage`**: materialize BOTH children once (`AvgValuesProgram` / `AvgWeightsProgram`, the M1/M2 route), cast each to the ONE result dtype (NumPy forces the product AND both sums to `result_dtype` — so an int·int product is a float64 product, never a wrapping int64 one), form the product, and reduce the product and the weights with the SAME NumPy-exact pairwise sum (`ExactSumArray`) — **NOT** the drifting multi-accumulator engine sum the library `np.average` itself uses, which is exactly why the fused `Average` is BIT-EXACT with NumPy where `np.average` is only allclose at large N — then divide. **dtype rule** (probed, keyed on the VALUES dtype): int/bool values → `result_type(values, weights, float64)`, else `result_type(values, weights)` — always float/complex. Zero total weight (empty input included) raises `DivideByZeroException("Weights sum to zero, can't be normalized")` before the divide (checked per slab for the axis form); a NaN weight → NaN (NaN ≠ 0, not a raise). BIT-EXACT vs NumPy 2.4.2 for Single/Double/Complex with **NO E1 excuse**; Half + Decimal result dtypes rejected (the "no float16 pairwise kernel" gap M1/M2/M4c-summation share). Wiring: `NDExpr.Average(values, weights[, axis, keepdims])` factories + the `WeightedAverageNode` (`NDExpr.Evaluate.cs`); tag `WeightedAverage` + hash/equals (`NDExpr.Structure.cs`); the `Average` field + `AvgValuesProgram`/`AvgWeightsProgram` + `Build` branch (`NDExpr.Program.cs`); the host `EvaluateWeightedAverage` (`DefaultEngine.Evaluate.cs`); `OpRegistry.Evaluate.cs` (a `weights` sub-expr in the `reduce` dict); `gen_oracle.py` block **C8** (14 dtype pairs × layouts × axis/keepdims, positive weights, + wide-N pairwise teeth + a fused-child case). Gate: `evaluate.jsonl` **18,936** cases green (host-pinned, +156) + `NDEvaluateTests.M4cAvg_*` (9 — value/dtype tiers, unit-weights == mean, complex, axis/keepdims, fused children, zero-weight + empty + axis-slab raise, NaN propagation, `out=`, Half/Decimal reject). |
| **M5** | Axis forms — `axis=None` + `keepdims`, direct `out=`; tuple `axis` deferred | **reachable scope landed** (this) | **`axis=None` + `keepdims`** landed for EVERY reduction kind (the five base + `Any`/`All`/`CountNonzero`/`NanSum`/`NanProd` + `Ptp`/`NanMin`/`NanMax` + `ArgMax`/`ArgMin` + `NanMean`/`Var`/`Std` + weighted `Average`). NumPy's `np.sum(a, axis=None, keepdims=True)` returns shape `(1,)*a.ndim`; the flat factories gained a `keepdims`-taking overload (a REQUIRED `bool`, so `Sum(x)` stays the 0-d form), and the four host reduce paths (fold / delegating / stat / average) reshape their 0-d scalar result to `(1,)*childNdim` via one shared `KeepdimsFlat`/`FlatReduceShape`/`ValidateFlatReduceOut` triple. The VALUE is identical to the 0-d form (a free reshape of the one element), so it inherits M1/M2/M4*'s bit-exactness with NO new excuse; a 0-d child stays 0-d (matching `np.sum(scalar, keepdims=True)`). Direct `out=` is validated for the keepdims shape (0-d without keepdims, exactly `(1,)*childNdim` with it). **Tuple `axis` is DEFERRED**: every NumSharp reduction is single-axis (`int?`) — multi-axis is a library-wide gap, not an evaluate-specific one, so offering it in `np.evaluate` alone would be inconsistent (a separate, larger feature). Oracle: `(None, True)` added to the flat combos of blocks C / C4 / C5 / C6 / C7 / C8. Gate: `evaluate.jsonl` **21,280** cases green (host-pinned, +2,344) net8+net10 + `NDEvaluateTests.M5_*` (7 — shape/value across every host path, rank-per-input 1-D/3-D, int64 argmax index, weighted average, var/std ddof, `out=` write-through + wrong-rank raise, 0-d child stays 0-d + keepdims=false unchanged). |

**Remaining E1 fold surface (still excused, `MisalignedRegistry`):** Half at any axis (no float16 pairwise
kernel — M-Half) and complex Prod (a complex-multiply chain, npy_cmul FMA-contracted on NumPy's win-amd64 build
but not on .NET's — the documented multiply gap #12, which an order fix cannot close). The **strict-F-contiguous
axis corner** is CLOSED (2026-09-23): the F-aware axis stream reduces it in NumPy's memory order (see "Perf review"
item 4), and the excuse no longer exempts it — a strict-F case that cannot stream (a broadcast size-1 operand) still
folds and would now be reported red (the corpus has none: all 84 of its f32/f64/c128 strict-F axis cases stream and
pass). E5 (complex min/max NaN identity) is a separate min/max milestone. The
excuse's `matExactReduce` predicate un-excuses everything M1+M2+the F stream fix, so a regression there turns the
gate red.

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
| Binary arithmetic | `Add Subtract Multiply Divide` | ✓ | bool add/multiply are logical or/and; divide of ints → f64; **Half via the widen-compute-narrow `Vector256<ushort>` lane (P5.1)** — the only vectorizing Half op family |
| Binary arithmetic | `Mod Power FloorDivide ATan2` | — | NumPy floored mod, `NDIntegerPower` wrapping, `npy_floor_divide` port, `Math.Atan2` |
| Bitwise | `BitwiseAnd BitwiseOr BitwiseXor` | ✓ | float/complex/decimal → NumPy no-loop TypeError text |
| Select | `Min(a,b) Max(a,b) Clamp Where` | — | branchy scalar; NaN-propagating like `np.maximum` |
| Binary (P4.2 `9abf25f3`) | `Maximum Minimum` (aliases of Max/Min), `FMax FMin Fmod CopySign NextAfter LogAddExp LogAddExp2 Hypot Heaviside Gcd Lcm LeftShift RightShift` | — | scalar via the shared `EmitScalarOperation` (byte-for-byte the engine ufunc). fmax/fmin NaN-ignore; fmod = dividend-sign; copysign-family = float-tier (arctan2 rule), complex no-loop; gcd/lcm integer-only (bool/bool no-loop); shifts integer (bool→int8). BIT-EXACT except logaddexp/logaddexp2 (≤2 ULP) + hypot@f64 (≤1 ULP), which ride `EvaluateLibmOps` |
| Unary arithmetic | `Negate Abs Square Reciprocal Deg2Rad Rad2Deg BitwiseNot` | ✓ | |
| Unary arithmetic | `Sqrt` | ✓ | |
| Unary transcendental | `Exp Exp2 Log Sin Cos Tanh` | f32 only | the NumPy-ported bit-exact float32 kernels (`NDFloatMath`) |
| Unary transcendental | `Expm1 Log2 Log10 Log1p Tan Sinh Cosh ASin ACos ATan Asinh Acosh Atanh Cbrt Sign` | — | CRT calls |
| Rounding | `Floor Ceil Round Truncate` | f32/f64 when the BCL has the method | identity on ints (NumPy loops) |
| Predicates | `IsNaN IsFinite IsInf LogicalNot` | — | bool result, tested at the child's dtype |
| Unary (P4.1a) | `Positive` (SIMD), `Conjugate`/`Conj`, `Fabs` (SIMD), `Spacing`, `SignBit`, `IsPosInf`, `IsNegInf`, `BitwiseCount` | Positive/Fabs | via the shared `EmitUnaryScalarOperation` (byte-for-byte the engine ufunc). Positive identity (dtype-preserving, bool no-loop); conjugate complex-flip/real-identity (**bool→int8**); fabs/spacing float-only (int→float promote, complex no-loop); signbit/isposinf/isneginf bool predicates (complex rejected — two distinct messages); bitwise_count int/bool/char→uint8. ALL BIT-EXACT, no ULP excuse |
| Unary (P4.1b) | `Rint` | f32/f64 when the BCL has the method | `np.rint` = round-half-to-even. ALIASES `Round`'s kernel (same value) but float-TIER typing (`UnaryFloatResult`: bool/i8/u8→f16, i16/u16→f32, i32+→f64; float/complex/decimal preserved) where `Round` preserves the input dtype. Complex rounds both lanes. BIT-EXACT, no ULP excuse |
| Unary (P4.1b) | `Real`, `Imag`, `Angle` | — | complex→real component extractors — NOT ufuncs, no engine kernel (handled in `UnaryNode.EmitScalar`). complex128→float64 (real lane / imag lane / `atan2(im,re)`); a REAL child: `Real`=identity (dtype PRESERVED), `Imag`=zeros (dtype PRESERVED, value unread), `Angle`=`atan2(0,x)` at `AngleRealTier`. `Real`/`Imag` bit-exact; `Angle` host-`atan2` (bit-exact within the host-pinned tier). Scalar-only. Radians (deg is a caller composition) |
| Cast (P4.1b) | `Cast(x, dtype)` (`NDExpr.Cast`) | — | `expr.astype(dtype)` — the first NEW NODE TYPE in P4 (a `CastNode` carrying the target dtype). Result IS the target; per-element `EmitConvertTo` (= `np.evaluate(child).astype`). `→bool` is `child != 0`, `complex→real` extracts the real lane (both special-cased around EmitConvertTo's edge-only bugs). casting='unsafe' (any conversion). Scalar-only (SIMD widening is Phase 5) |
| Round-decimals (P4.1b, this) | `Round(x, decimals)` (`NDExpr.Round`, decimals ≠ 0) | — | `np.round(x, decimals)` — a `RoundNode` carrying `_decimals`, dtype-PRESERVING port of `PyArray_Round`. float/complex: `op2(rint(op1(x, 10^\|d\|)), 10^\|d\|)` at the input float dtype (complex per-lane); integer d≥0 = identity; integer d<0 = float64 round → cast back (WRAPS); bool d≠0 = raises (multiply/divide cast error). `decimals == 0` routes to the plain `Round` node. Scalar-only. The engine's own `np.around` is buggy here — this is the correct port |
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
   `InnerLoop.cs:456`; `CompileNumPy`'s `homogeneous` check, `NDExpr.Typing.cs:328`). Phase 1 lifted
   the comparison / `Where` / predicate / logical restriction (Boolean nodes ride as lane masks), and
   **P5.1 lifted Half for pure arithmetic** (a `Vector256<ushort>` widen-compute-narrow lane, `add/sub/
   mul/div` only — a Half tree with any other node stays scalar). Still scalar-only: every mixed-dtype
   tree (`i4*2+f8`, P5.2) and every Decimal/Complex tree.
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
| Unary | ✅ **4.1a landed**: `Positive`, `Conjugate`/`Conj`, `Fabs`, `Spacing`, `SignBit`, `IsPosInf`, `IsNegInf`, `BitwiseCount`. ✅ **4.1b `Rint` landed** (aliases `Round`'s kernel, float-tier typing). **Still open (4.1b):** `Real`, `Imag`, `Angle`, `Round(decimals)` | the landed set rode `UnaryOp` + `EmitUnaryScalarOperation`; the rest of 4.1b changes dtype (complex→real) or is parameterized and needs new emit |
| Unary | `Cast(dtype)` / `AsType` (4.1b) | the single most requested composition (`(a*b).astype(f4) + c`); also the only way to pin a loop dtype like the ufunc `dtype=` keyword |
| Binary | ✅ **4.2 landed**: `Maximum/Minimum` aliases, `FMax/FMin`, `Fmod`, `CopySign`, `NextAfter`, `LogAddExp/LogAddExp2`, `Hypot`, `Heaviside`, `Gcd/Lcm`, `LeftShift`, `RightShift` | rode `BinaryOp` + `EmitScalarOperation` (factory + typing only) |
| Logical | `LogicalAnd/Or/Xor` with NumPy's nonzero-test semantics on non-bool inputs (today `&` is bitwise) — 4.3 | bool result; `(a!=0)&(b!=0)`; a dedicated node, not the mechanical route |
| N-ary | `Select(condlist, choicelist, default)`, `Clip(x, lo?, hi?)` with `None` bounds | `Clamp` exists for the two-sided case |
| Reductions | `Any`, `All`, `ArgMax`, `ArgMin`, `Std`, `Var` (ddof), `NanSum/NanProd/NanMin/NanMax/NanMean`, `CountNonzero`, `Ptp`, weighted `Average` (two accumulators) | plus tuple `axis`, `axis=None keepdims`, `out=` on axis reductions without the copyto detour |
| Keywords | `where=` (mask), `casting=`, `order=`, `dtype=` (the loop dtype) | `where=` composes for free once evaluate uses the masked `ExecuteElementWise` entry |
| Operators | `< > <= >=` returning `NDExpr` (legal C#; only `==`/`!=` are hazardous) ; unary `+` | today `NDExpr.Greater(a, 2.0)` |
| Literals | see G6 | |

### 3.3 Performance (ranked by measured loss)

| # | Regime | Evidence (§2.3/§2.4) | Root cause |
|---|---|---|---|
| P1 | any tree with a comparison / `Where` / `Min/Max` / logical / predicate node | `a>0.5` 0.16× NumPy @100K, 0.71× @4M; `maximum` 0.46×/0.69× unfused; `where` 0.82× unfused @100K | Boolean-typed nodes break the "one dtype" SIMD gate → whole tree scalar |
| P2 | Half operands | `f2*f2+f2` 0.15–0.18× unfused, 0.51–0.55× NumPy | ✅ **P5.1 landed**: Half arithmetic now vectorizes (`Vector256<ushort>` widen-compute-narrow), 4.3–5.6× the old scalar fused path, and the fused scalar Half was fixed from a divergent double bridge to float32. Now ~0.6–0.77× the engine's tuned unfused chain (the Giesen-conversion ceiling); a non-arithmetic Half node still forces the tree scalar. |
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

> **Status (see §0.2 for the milestone ledger):** M1 (flat) landed `7ac2c7b5`; M2 (axis, C-contiguous
> child) landed `bb02d212`; M4a (`Any`/`All`/`CountNonzero`/`NanSum`/`NanProd`) landed `9d04b1a4`; M4b
> (`Ptp`/`NanMin`/`NanMax`) landed `417a448a`; M4c-index (`ArgMax`/`ArgMin`) landed `5862547c`;
> M4c-summation (`NanMean`/`Var`/`Std`) landed `03ae07c2`; **M4c-average (weighted `Average`)
> landed**. M1/M2 take the "scratch materialization" route in
> 2.1 below — materialize the child, then reduce it with the exact schedule; M4b and M4c-index reuse that
> materialize step and delegate to the engine reductions (`np.ptp`/`np.nanmin`/`np.nanmax`;
> `np.argmax`/`np.argmin`), bit-exact by construction — the value kinds because a min/max is
> order-independent, the index kinds because the materialized child is the same C-order buffer NumPy's own
> argmax reduces. M4c-summation reuses the materialize step but computes the answer itself
> (`EvaluateStatReduce`), reproducing NumPy's `nanmean`/`_var` op for op over the M1/M2 pairwise sum
> (Half/Decimal rejected — no float16 pairwise kernel). M4c-average is the FIRST reduction over TWO
> operand trees (`WeightedAverageNode` — 2.3): the host (`EvaluateWeightedAverage`) materializes both
> children and computes `Σ(v·w)/Σ(w)` over the SAME pairwise sum, bit-exact where the library
> `np.average` (which sums through the drifting engine fold) is only allclose at large N. **M5's
> reachable scope** — `axis=None`+`keepdims` for every kind + direct `out=` — landed too (the flat 0-d
> result reshaped to `(1,)*childNdim`; a free reshape, so bit-exact with no new excuse); **tuple `axis`
> is deferred** (single-axis is a library-wide gap). **The reduction KINDS + axis-forms surface is now
> complete.** Only M3 (streaming pairwise, no temp — 2.1/2.2) is open.

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

2.3 **Missing reduction kinds** (status): `Any/All` ✅ M4a (bool ReduceKinds, logical OR/AND fold),
`CountNonzero`/`NanSum`/`NanProd` ✅ M4a (factory rewrites `Sum(x!=0)` / `Sum(Where(IsNaN(x),0,x))` /
`Prod(Where(IsNaN(x),1,x))` — the `_replace_nan` composition at the tree level), `Ptp`/`NanMin`/`NanMax`
✅ M4b (order-independent → host-delegated to `np.ptp`/`np.nanmin`/`np.nanmax` over the materialized
child — `Ptp` = `max−min` and the type-aware NaN-skip live in those engine reductions, not a tree
rewrite). `ArgMax/ArgMin` ✅ **M4c-index** (host-delegated over the same `EvaluateDelegatingReduce`
shape via `np.argmax`/`np.argmin`; an **int64** index result, with the C-order first-tie / first-NaN
rule bit-exact because the materialized child is NumPy's own C-order buffer). `NanMean`/`Var`/`Std`
✅ **M4c-summation** (host-computed by `EvaluateStatReduce` over the materialized child, reproducing
NumPy's `nanmean` / `_var` op for op over the M1/M2 pairwise sum: `NanMean = pairwise_sum(NaN→0)/count`,
`Var = pairwise_sum((x-mean)²)/max(N-ddof,0)`, `Std = sqrt(Var)`; a complex128 child → REAL float64 var/std;
bit-exact for 13 dtypes, Half/Decimal rejected). weighted `Average`
✅ **M4c-average** (`Σ(values·weights)/Σ(weights)` — the FIRST reduction over TWO operand trees, so its own
root `WeightedAverageNode`, host-computed by `EvaluateWeightedAverage` over the two materialized children
with the SAME pairwise sum — bit-exact where the library `np.average`, which sums through the drifting engine
fold, is only allclose at large N; zero total weight → `DivideByZeroException`; Half/Decimal rejected).
**The reduction KINDS surface is now complete** — every kind in §3.2's reductions row is implemented.
`axis=None` + `keepdims` (M5's reachable scope, 2.4) is also implemented for every kind; only M3 (perf)
and M5's **tuple axis** (deferred — a library-wide gap) remain open.

2.4 **Axis forms**: `axis=None` with `keepdims` ✅ **M5** (implemented for every reduction kind — the flat
0-d result is reshaped to `(1,)*childNdim`, a free reshape that inherits M1/M2/M4*'s bit-exactness; direct
`out=` validated for the keepdims shape). **Tuple `axis`** is DEFERRED: every NumSharp reduction is
single-axis (`int?`), so multi-axis is a library-wide gap, not an evaluate-specific one — offering it in
`np.evaluate` alone would be inconsistent, and closing it is a separate, larger feature. Direct axis `out=`
(no copyto) and SIMD in the slab/pinned paths remain the M3-adjacent perf work.

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

> **Status: 4.1b, 4.2, 4.3, 4.4 and ALL of 4.5 (`where=`/`dtype=`/`casting=`/`order=`) are COMPLETE.**
> **4.2 (binary nodes) LANDED `9abf25f3`**, **4.1a (mechanical unary nodes) LANDED `46f96507`**, **4.1b's
> `Rint` LANDED `6e5d7e51`**, **4.1b's `Real`/`Imag`/`Angle` LANDED `7578ab31`**, **4.1b's `Cast(dtype)`
> LANDED `6a31bcf3`**, **4.1b's `Round(decimals)` LANDED `e55d1fd6`**, **4.3
> (`LogicalAnd`/`LogicalOr`/`LogicalXor` + `Select` + `Clip`) + 4.4 (operators `< > <= >=`, unary `+`)
> LANDED `4c3f74a5`**, **4.5's `dtype=` / `casting=` / `order=` keywords LANDED `402d8adc`**, and **4.5's
> `where=` LANDED (this)** — see §0.1's P4 row. **`where=` did NOT need 4.6:** the mask rides as a
> trailing ARRAYMASK operand with a WRITEMASKED output, and `NDIterRef.ForEach` (which np.evaluate
> already calls with its pre-compiled fused kernel) resolves and drives the mask itself, so masked-off
> `out` slots keep their prior contents without any routing change. Next: 4.6 (route through
> `ExecuteElementWise` so narrow strided rows AND masks reach the 2-D block kernels — **PERF ONLY**, and
> it needs a 2-D-block variant of the fused kernel, since np.evaluate compiles an `NDInnerLoopFunc` not
> the scalar/vector emit bodies `ExecuteElementWise` consumes), 4.7 (docs).
> **4.2's `LogicalAnd/Or/Xor` slid to 4.3** — they are NOT plain BinaryOps (bool result via a
> nonzero-test on non-bool inputs, no engine `BinaryOp`), so they landed as a dedicated `LogicalNode`
> like `Where`, not the mechanical factory+typing route. **`Rint` was the mechanical member of 4.1b**
> (aliases `Round`'s kernel, float-tier typing only); **`Real`/`Imag`/`Angle` were the first
> genuinely-new emit path** (NOT ufuncs, handled in `UnaryNode.EmitScalar`); **`Cast` is the first NEW
> NODE TYPE in P4** (a `CastNode` carrying the target dtype); **`Round(decimals)` is a NODE carrying
> `_decimals`** (a dtype-dependent port of `PyArray_Round` — and the fused RoundNode is the CORRECT
> reference, because the ENGINE's `np.around`/`np.round_` is itself BUGGY for negative decimals and
> complex, see §0.1). **`Select`/`Clip` are pure LOWERINGS** (Select → a reverse `Where` chain, Clip →
> `Max`/`Min`/`Min(Max())`), so they introduce NO new node and inherit those tiers' parity.

4.1a Unary nodes ✅ **LANDED (this)**: `Positive`, `Conjugate`/`Conj`, `Fabs`, `Spacing`, `SignBit`,
`IsPosInf`, `IsNegInf`, `BitwiseCount` — the mechanical engine-unary-ufunc set (the direct analog of
4.2's binary route). `UnaryNode.EmitUnaryScalarOperation` already dispatches all eight, so only the
factory + typing (`UnaryNode.InferType`) + routing the bool/byte-result ops through the existing
"emit-at-child-dtype, convert" path were added; the emit is byte-for-byte the unfused chain. Typing
(probed 2.4.2): positive dtype-preserving with NO bool loop; conjugate real-identity / complex-flip /
**bool→int8**; fabs & spacing FLOAT-only (UnaryFloatResult int→float tier, complex has no loop);
signbit/isposinf/isneginf → bool (complex rejected — signbit with the ufunc no-loop text, isposinf/
isneginf with the DISTINCT "…not supported for complex128 values because it would be ambiguous"
message, since they are functions not ufuncs); bitwise_count → uint8 (integer/bool/char only).
**ALL cells BIT-EXACT — no ULP excuse.** Positive/Fabs vectorize (in `IsSimdUnary`, vector==scalar
verified); Spacing stays scalar-only (vector/scalar NaN-payload divergence at inf/NaN, not worth the
marginal win). `evaluate.jsonl` 23,224 → 24,168 (+944); `NDEvaluateTests.P41_*` (8).
**4.1b `Rint` ✅ LANDED (this):** the mechanical member of the dtype-CHANGING set. `np.rint` — the TRUE
ufunc form of round-half-to-even — has the SAME value as `Round` (both `Math.Round`'s banker's rounding),
so the new `UnaryOp.Rint` **aliases `Round` at every kernel emit site** (the `Fabs`/`Abs` precedent: main
scalar switch, the half/complex/decimal emitters, the vector emitter's name map + gate, and
`VectorRoundingMethodName`), needing NO new emit path. It differs from `Round` only in DTYPE: it PROMOTES
to a float tier (`UnaryFloatResult` — bool/int8/uint8→float16, int16/uint16→float32, int32+→float64;
float/complex/decimal preserved, probed 2.4.2) where `Round` preserves the input dtype — so `Rint(int32)`
is a float64 while `Round(int32)` is the int32 identity. Wired in NDExpr as `IsFloatPromoting` (typing) +
`IsRoundingOp` (so `IsSimdUnaryAt` routes it to `RoundingVectorSimdAvailable` — it vectorizes at
float32/float64 exactly like `Round`) + the `Rint` factory; the structural hash / signature self-wire off
`(long)_op`. **ALL cells BIT-EXACT vs NumPy 2.4.2, NO ULP excuse** (`Math.Round` is portable; complex
rounds both lanes) — verified fused == the engine's own `np.rint` (itself oracle-gated to NumPy) across all
15 dtypes × tiers × half-value ties × layouts × SIMD-vs-scalar × compositions. Oracle: `rint` added to
`_EV_UNARY` (has a loop for every dtype, no rejection cell) + a sub-tree composite
`add(rint(mul(in0,1.5)),1.0)`; `evaluate.jsonl` 24,168 → 24,415 (+247); `NDEvaluateTests.P41b_*` (2 —
float-tier + half-to-even + complex both lanes + the `Round`-preserves-int contrast; vector==scalar +
sub-tree compose).
**4.1b `Real`/`Imag`/`Angle` ✅ LANDED (this) — the complex→real component extractors.** NOT ufuncs, so
there is NO engine `UnaryOp` kernel: they are three new `UnaryOp` enum values handled ENTIRELY in
`UnaryNode.EmitScalar` (before the generic `EmitUnaryScalarOperation` tail, which would throw for them),
never reaching the engine. For a COMPLEX child the value is extracted as float64 — `Real` → `z.Real`,
`Imag` → `z.Imaginary`, `Angle` → `atan2(im, re)` — via three static pass-by-value helpers on
`NDComplexMath` (`RealPart`/`ImagPart`/`Angle`, the `s_complexAbs` pattern: a `Complex` on the IL stack
consumed with one `call`). For a REAL child each degenerates: `Real` is the IDENTITY with its dtype
PRESERVED (`real(int32)` is int32, not a float), `Imag` is a constant ZERO of the preserved dtype (the
child value is NOT emitted — `imag(real)` is `zeros_like`), and `Angle` is `atan2(0, x)` (0 for x ≥ 0,
pi for x &lt; 0, `-0.0` → pi, NaN → NaN) computed in double and narrowed to NumPy's per-dtype float
tier (`np.AngleRealTier`, made `internal` and reused so the fused node and `np.angle` resolve the dtype
through ONE table). Typing (probed 2.4.2): `Real`/`Imag` → `childType == Complex ? Double : childType`
(dtype-PRESERVING, NOT float-promoting); `Angle` → `Complex ? Double : AngleRealTier(childType)`
(bool/int32+/f64 → f64, int8/uint8/f16 → f16, int16/uint16/char/f32 → f32). Scalar-only in the fused
kernel (`CanEmitVectorV2` returns false — none is in `IsSimdUnary`), so a tree containing one runs scalar
end-to-end. `Real`/`Imag` are BIT-EXACT (pure lane extract / identity / zeros — no host-libm); `Angle`
rides `atan2` (host-libm), BIT-EXACT within the host-pinned evaluate tier (win-amd64 shares MSVC
`ucrtbase` between `Math.Atan2` and `npy_atan2` — the arctanh class), so NO ULP excuse. Verified fused ==
the engine's own `np.real`/`np.imag`/`np.angle` (themselves NumPy-gated) across all dtypes × layouts ×
composites × NaN/inf/-0.0 edges. `evaluate.jsonl` 24,415 → **24,987** (+572); `NDEvaluateTests.P41b_*`
(+4). **4.1b `Cast(dtype)` ✅ LANDED (this) — the first NEW NODE TYPE in P4** (`expr.astype(dtype)`, the
foundational primitive 4.5's `dtype=` and `Round(decimals)`'s integer path both need). A `CastNode`
wraps a child and carries the TARGET dtype (a payload-less `UnaryOp` enum cannot), so it is a distinct
node implementing the full node contract (`InferType`/`EmitScalar`/`BindArrays`/`HashStructure`/
`StructureEquals`/`CanEmitVectorV2`/`AppendSignature`). Typing: result IS the target, independent of the
child (a WEAK child resolves to its OWN default first — `astype(np.array(5), f4)` casts an int64 array,
not an adopted-to-float one). Emit: compute the child, convert per element with the SAME
`EmitConvertTo` every node edge uses — so a `Cast` is byte-for-byte `np.evaluate(child).astype(target)`
including the C-undefined float→int edges (NaN/±inf/out-of-range go through the NumPy-faithful
`Converts.*` table both sides share). **Two conversions `EmitConvertTo` mis-handled** (this Cast was its
first caller for them, so latent bugs surfaced) are special-cased in `CastNode.EmitScalar`: `→bool` is
a type-correct `child != 0` (EmitConvertTo's `to==Boolean` did a double-vs-int stack compare, only ever
reached from a 0/1 Int32 at edges), and `complex→real` extracts the real lane first
(`NDComplexMath.RealPart`) — EmitConvertTo mis-computed `complex→int` and ACCESS-VIOLATED on
`complex→Half`. **SCALAR-ONLY** (`CanEmitVectorV2` false; the SIMD-widening cast lanes are Phase 5).
Verified fused == the engine's own NumPy-gated `astype` across the FULL 15×15 matrix incl. every
C-undefined edge (225 safe + 14 edge probe cells, 0 mismatch). Oracle: a `cast_<dtype>(child)` token
(`_ev_eval`/`OpRegistry` prefix-dispatch) + a src×target×layouts block over a SAFE non-negative pool
(portable bit-exact parity, no host-dependent cell); `evaluate.jsonl` 24,987 → 25,706 (+719);
`NDEvaluateTests.P41bCast_*` (+4 — the float→bool fix, complex→real/bool, the C-undefined edges ≡ engine,
char/decimal/wrap/identity/compose). **4.1b `Round(decimals)` ✅ LANDED (this) — the LAST member of 4.1b,
completing it.** A `RoundNode` carrying `_decimals` (nonzero — the factory routes `decimals == 0` to the
plain `UnaryOp.Round` node), a dtype-DEPENDENT port of NumPy's `PyArray_Round`: for a FLOAT/complex child
it composes `op2(rint(op1(x, 10^|d|)), 10^|d|)` (mul→div for positive decimals, div→mul for negative) AT
THE CHILD'S OWN float dtype (complex rounds each float64 lane and reassembles), dtype PRESERVED; an INTEGER
child with decimals ≥ 0 is the IDENTITY (no fractional part, no float round-trip — a huge int64 is
untouched); an INTEGER child with decimals < 0 runs the round in float64 and CASTS BACK to the integer
dtype, so an out-of-range result WRAPS (`round(uint8 255, -1)` → 4); a BOOL child with decimals ≠ 0 THROWS
NumPy's `UFuncTypeError` (multiply for +, divide for −). `power_of_ten` is ported exactly (table + repeated
`*10.`, not `Math.Pow`, so `f` is bit-identical). The emit reuses the SAME per-element multiply/divide
(`EmitScalarOperation`) and rint (`UnaryOp.Round`) the engine's ufuncs use. Scalar-only (`CanEmitVectorV2`
false). **The ENGINE's `np.around`/`np.round_` is BUGGY** — it uses `Math.Round(x, decimals)` which THROWS
for negative decimals, and it does not round complex — so the fused RoundNode is the CORRECT port and is
validated against NumPy DIRECTLY (the oracle), never the engine (a separate engine-`np.around` fix is out
of scope, worth its own issue). Verified fused == NumPy 2.4.2 across dtypes × {2,1,-1,-2,3,-3} × layouts
(the sanity probe + the oracle). Oracle: a `round_<d>(child)` token (`m<n>` = −n) + a src×decimals×layouts
block over a MODERATE controlled pool (portable bit-exact, no C-undefined int round-trip); `evaluate.jsonl`
25,706 → **26,040** (+334); `NDEvaluateTests.P41bRound_*` (+4 — float at input precision, integer
identity/wrap, complex per-lane + bool raise, decimals=0 delegation + compose + decimal). **4.1b is now
COMPLETE** (Rint · Real/Imag/Angle · Cast · Round(decimals)).
4.2 Binary nodes ✅ **LANDED `9abf25f3`**: `Maximum/Minimum` (aliases), `FMax/FMin`, `Fmod`, `CopySign`,
`NextAfter`, `LogAddExp/LogAddExp2`, `Hypot`, `Heaviside`, `Gcd/Lcm`, `LeftShift/RightShift`. Each is a
`BinaryNode` over the shared scalar emitter (`EmitScalarOperation`); only the factory + BinaryNode.InferType
promotion (and the no-loop rejections) were added. `LogicalAnd/Or/Xor` (nonzero-test on non-bool inputs,
bool result) stay OPEN — they are not plain `BinaryOp`s, so they need a dedicated node (4.1/4.3), not this
mechanical route.
4.3 ✅ **LANDED (this):** `LogicalAnd`/`LogicalOr`/`LogicalXor` (dedicated `LogicalNode`), `Select`,
`Clip`.
- **`LogicalNode`** — the ONLY new node in 4.3. Result is ALWAYS Boolean via a NONZERO-TEST of each
  operand at its OWN dtype (`(a != 0) op (b != 0)`), which no engine `BinaryOp` expresses (hence a
  dedicated node, not the 4.2 route). Every dtype is accepted, complex included (`z != 0+0i` via
  `EmitComparisonOperation`, which covers all 15 dtypes), NaN is truthy, ±0 falsy. Scalar emit: two
  `child != 0` I4 (0/1) results combined by the plain `And`/`Or`/`Xor` opcode. It VECTORIZES: each
  child becomes a truthiness lane mask (`EmitVectorChildAs(child, Boolean)` = the Where-cond path),
  combined by `NDExprVec.EmitAnd/Or/Xor` (W-mode canonical masks) or the engine's canonical-bool
  logical op in byte mode (xor → not_equal). Typing resolves each child at its own dtype (no
  promotion between the two, like Where's condition) and returns Boolean. **ALL cells BIT-EXACT vs
  NumPy 2.4.2, no ULP excuse** (an integer/mask combination, no host-libm) — verified vector==scalar.
  Oracle: `land`/`lor`/`lxor` in `_EV_BINARY` + a new block A3 (the three ops over all pair layouts ×
  dtypes, plus two composites); `NDExprNodeTag.Logical = 13`.
- **`Select(condlist, choicelist, default=null)`** — a pure LOWERING to a reverse `Where` chain
  (`where(c0, v0, where(c1, v1, … where(cN, vN, default)))`, so condlist[0] is outermost and the FIRST
  true condition wins). `default` null ≙ NumPy's weak-int `0`. Result dtype = `result_type(*choices,
  default)` (the nested where nodes compose the same promotion), and the conditions are nonzero-tested
  at their own dtype (in practice bool comparisons, matching np.select's bool condlist). Factory-time
  validation is NumPy's verbatim ORDER: length mismatch ("list of cases must be same length as list of
  conditions") BEFORE emptiness ("select with an empty condition list is not possible"). No new node —
  rides the already-gated WhereNode tier.
- **`Clip(x, lo?, hi?)`** — a pure LOWERING: one-sided → the NaN-propagating `Max`/`Min`
  (`np.clip(x, lo, None) ≡ np.maximum(x, lo)`, `np.clip(x, None, hi) ≡ np.minimum(x, hi)`), two-sided
  → `Min(Max(x, lo), hi)` (= the existing `Clamp`), which is NumPy's exact general clip ufunc
  `_NPY_MIN(_NPY_MAX(x, lo), hi)` (`clip.cpp`). Bit-exact with NumPy 2.4.2 for one-sided and for
  two-sided with ARRAY bounds; the ONE divergence is the pathological signed-zero-exactly-at-a-finite
  SCALAR-bound corner (`clip(-0.0, -inf, +0.0)`: NumPy's const-scalar FAST path returns `-0.0`, the
  general `min(max)` path — which NumSharp always takes — returns `+0.0`; NaN-sign-of-bound corners
  are float-NaN differences the oracle tokenizes). Both-null raises NumPy's "One of max or min must be
  given". No new node — rides the MinMax tier.
- Gate: `NDEvaluateTests.P43_*` (5 — logical all-dtypes/complex/NaN, logical vector==scalar,
  select first-true-wins + default + errors, clip one/two-sided + dtype + both-null-raise, the
  comparison operators + unary `+` + the `== null` reference-check); the `evaluate.jsonl` tier's block
  A3 (logical nodes) grew it 26,040 → 26,742 (+702).
4.4 ✅ **LANDED (this):** `<  >  <=  >=` on `NDExpr` (build a Boolean `ComparisonNode`, RETURN an
`NDExpr` — composable; an `NDArray`/scalar operand converts implicitly, so `expr < 0.5` and
`arr < expr` both bind); unary `+` is `Positive` (identity copy). `==`/`!=` are deliberately NOT
overloaded (they would hijack `expr == null` reference checks — pinned by a test) — use
`Equal`/`NotEqual`. No new node / no oracle grammar (the comparisons ride `lt`/`le`/`gt`/`ge`, unary
`+` rides `positive`); pinned by `NDEvaluateTests.P43_ComparisonOperators_And_UnaryPlus`.
4.5 Keywords on `np.evaluate` — **`where=` / `dtype=` / `casting=` / `order=` ✅ ALL LANDED
(`dtype=`/`casting=`/`order=` `402d8adc`, `where=` this).** All four keywords are threaded from the API
boundary to the engine as ONE resolved `NDEvaluateOptions` struct (`Backends/Iterators/NDEvaluateOptions.cs`,
public because the public `TensorEngine.Evaluate` seam carries it), built so `default(NDEvaluateOptions)`
reproduces the pre-4.5 behaviour EXACTLY — every internal `EvaluateCore` sub-tree materialisation passes
`default` and MUST see the natural dtype/layout and NO mask, so a nullable casting, a NUL order char and a
null `Where` reference stand in for "unset".
- **`dtype=`** (a `DType`, implicit conversions from `NPTypeCode`/`Type`/string): an **implicit root
  cast** on the elementwise path — the tree still COMPUTES at its natural NEP50 result type and the RESULT
  is cast to `dtype`, so `np.evaluate(expr, dtype: X)` is bit-identical to `np.evaluate(expr).astype(X)`
  **in ONE fused pass** (it allocates the fresh result at `dtype` and rides the exact out= buffered-cast
  path — the same NDIter cast the tested out= path uses, distinct from the Cast NODE's per-element
  `EmitConvertTo`). It is a RESULT cast, NOT an accumulator/loop dtype, so it is **rejected on a reduction
  tree** with `NotSupportedException` (a reduction fixes its own accumulator dtype — cast the result
  instead). **Mutually exclusive with `out=`** (both fix the result dtype — the API rejects the pair with
  `ArgumentException`).
- **`casting=`** (a string, default null ≙ `same_kind`): the cast rule enforced when a caller `out=` has a
  different dtype than the resolved result — parsed at the API via `DTypeCasting.ParseCasting` (NumPy's
  verbatim "casting must be one of …" for a bad string), threaded to **every** path incl. the reduce
  helpers' `ValidateOutCast` (now casting-parameterised, naming the failed rule via `CastingToString`). So
  `casting='unsafe'` admits a float→int out that `same_kind` rejects; `'no'` rejects any real cast.
- **`order=`** (a char, default `'K'`): the FRESH result's memory layout — `'C'` forces row-major, `'F'`
  column-major (even from C inputs), `'A'`/`'K'` keep today's heuristic (F only when every input is strictly
  F-contiguous). A caller `out=` keeps its own layout; the iteration order follows the result buffer's own
  contiguity (elementwise is order-independent, so this changes traversal, never values). A non-`'K'` order
  is rejected on a reduction tree (fixed-layout result); a bad char is a `ValueError` (validated at the API
  even for a reduction tree). `ResolveEvalOrder` in `DefaultEngine.Evaluate.cs`.
- Gate: `NDEvaluateTests.P45_*` (9 — dtype implicit-root-cast across float/int/widen/narrow/real→complex ×
  C/F layouts, dtype truncation, dtype⊕out mutual exclusion, order layout + values-unchanged + bad-char,
  casting out-rule + bad-string, the reduction rejection + casting-threads-into-reduce-out, the
  positional-operand overload); a new `evaluate.jsonl` **B5** block (281 cases — `dtype=X` over a fused tree
  × safe non-negative pool × real+complex targets × C/F/negstride/strided/2-D layouts, referenced as
  `<expr>.astype(X)`, gating the fused BUFFERED-cast path bit-exact vs NumPy independently of the Cast
  node's per-element path). `evaluate.jsonl` 26,742 → **27,023**. A 63-cell layout differential
  (broadcast/transposed/negstride/strided/offset/F/C × 8 targets + order) confirmed `dtype=` == the
  NumPy-gated `.astype()` on every layout the corpus pool does not reach.
- **`where=`** (a boolean write mask, ✅ LANDED this) — the fused kernel writes the result only where the
  mask is True, leaving masked-off `out=` slots at their PRIOR contents (NumPy's ufunc where= convention).
  **It did NOT need 4.6's `ExecuteElementWise` re-routing** (the plan's original guess): the mask is
  appended as a trailing **ARRAYMASK** operand and the output flipped to **WRITEMASKED**
  (`EvalElementwiseMaskedFlags`), and `NDIterRef.ForEach(kernel, aux)` — which np.evaluate already calls
  with its pre-compiled fused `NDInnerLoopFunc` — resolves the mask itself (`ResolveForEachMaskOp`) and runs
  the kernel per mask-true run, with the SIMD 32-byte block run-scan driving dense/sparse masks. So the
  masked WRITE (correctness) is decoupled from the masked 2-D BLOCK kernel (perf, 4.6). It is validated bool
  (`ValidateWhereMask` — verbatim cast TypeError), joins the broadcast without stretching a provided `out`
  (`ResolveUfuncIterationShape`), composes with `dtype=`-cast (the WRITEMASKED buffered flush also masks) and
  0-d parameters (the aux block is invariant across mask-true runs), and is **rejected on a reduction tree**
  (a masked reduction is a different operation — the message points at masking the inputs first, alongside
  the dtype=/order= reduction guards). Oracle: `evaluate.jsonl` **D2** block (+280); `NDEvaluateTests.P45Where_*`
  (9) + a Release run-scan stress probe.
4.6 Route `np.evaluate` through `ExecuteElementWise` (packed key) so narrow strided rows and masks
get the 2-D block kernels (§1.4 point 2) — **PERF ONLY** now that `where=` correctness is complete via
ForEach's masked driver. This needs a 2-D-block VARIANT of the fused kernel: np.evaluate compiles an
`NDInnerLoopFunc` (`program.Kernel`), while the 2-D block route (`TryExecute2DElementwise` /
`TryExecute2DMasked`) is built around the scalar/vector emit-body pair the ufunc packed-key route hands
`ExecuteElementWise`. So the fused compiler would emit a second, 2-D-block-shaped kernel per program (the
narrow-strided-row lever from the ufunc side, `docs/NDITER_2D_BLOCK_KERNEL.md`).
4.7 Docs: rewrite the NDIter.md Tier 3C catalog (SIMD column per node per dtype) + the CLAUDE.md
"Fused Expressions" section from the probe tables; API reference pages for the new nodes.

Acceptance: every NumSharp elementwise ufunc has a node; `evaluate.jsonl` covers each new node across
dtypes/layouts; `where=` cases mirror `out_where.jsonl`'s base-buffer check.

### Phase 5 — Half and mixed-width SIMD

5.1 Half ✅ **LANDED (this)** — widen-compute-narrow **per node** (`(Half)((float)a op (float)b)` is
NumPy's npy_half model, so per-node rounding is required), reusing the Giesen f16↔f32 converters the
engine's own Half kernel (`DirectILKernelGenerator.Binary.Arith.Half.cs`) already proves bit-exact.
**It began with a correctness fix that the vector==scalar contract forced:** the fused SCALAR Half
`add/sub/mul/div` (`EmitHalfOperation`) computed in **double** where NumPy's HALF loop computes in
**float32** — a real 1-ULP-class divergence on exponent-gap sums, and a BCL sNaN-quiet — so it now
routes those four ops through `HalfArithScalarStruct` (== `HalfArithBits`), making the fused scalar
path bit-for-bit the NumPy-gated engine kernel. The vector path then carries a Half tree's lane as
`Vector256<ushort>` (16 raw f16 patterns): the shell's `GetSimdLaneType(Half)→ushort` makes the load/
store/broadcast work as ushort, and each Half arithmetic node emits a `call HalfArithVec256` (widen
both operands to two `Vector256<float>`, one `Avx.Add/Sub/Mul/Div`, narrow RTNE + the operand-order
NaN fixup — the engine's proven 8-lane pieces composed for 16 lanes), so the vector body is
byte-for-byte the scalar body. **Only pure arithmetic vectorizes:** `CanEmitVectorV2` returns false
for every non-arithmetic node at a Half lane (comparison / where / min-max / logical / unary /
transcendental), so such a tree stays whole-tree scalar (as does a non-256-bit or non-AVX2 host —
`FusedHalfArithAvailable`). **Measured** (best-of-15, Release, `f2*f2+f2`): vector is **4.3–5.6× the
old scalar fused path** (the win), which — NumPy's Half loop being scalar — moves fused Half from
losing to NumPy to beating it. It does NOT reach the target ≥ 1.5× unfused (it is **0.6–0.77× the
engine's hand-tuned `HalfArithFull` chain**): the Giesen conversion is the shared ceiling (no F16C in
the BCL) and the general fused shell's per-node widen/narrow overhead is the gap — closing it needs a
specialized fused Half kernel (deferred; the correctness fix + the 4–5.6× vectorization is the
increment). Gates: `NDEvaluateVectorTests` (Half in the lane sweep) + `NDEvaluateTests.P51_*` (4).
5.2 Mixed widths ("lane groups") ✅ **LANDED (2026-09-14)** — as planned: the vector element
count is `V / maxWidth` (lane W = the ROOT dtype, which IS the widest — NEP50 promotion never
narrows along a parent edge), and narrower operands load a partial vector and widen exactly
(`int32/float32 → double` via `Avx.ConvertToVector256Double`, the int widens via
`Avx2.ConvertToVector256Int16/32/64` sign/zero-extends, `float → double`; `uint32 → double` —
absent from AVX2 — via the exact sign-bias trick `cvt((int)(u^0x80000000)) + 2^31`, equal to the
scalar `conv.r.un` bit for bit). What the sketch left open, resolved during implementation:
(1) widening happens at NODE EDGES, not only operand loads — each node computes at its OWN dtype in
its own container (a half-lane node at `Vector128`), because `(i4*2)+f8`'s multiply must WRAP at
int32 before promoting, exactly like NumPy's unfused sequence (probed 2.4.2; computing at the lane
would silently diverge on overflow); (2) the container rule — a node's `elemCount·sizeof(dtype)`
must be 128 or 256 bits — makes every admitted narrow dtype exactly HALF the lane, so every edge is
ONE V128→V256 widen and a partial load is exactly 16 bytes (never an over-read at the buffer end);
(3) a same-size inexact edge (`int64 → float64` ROUNDS, and AVX2 has no vector form) is simply
absent from the edge map (`NDExprVec.HasWidenEdge`), declining such trees to scalar; (4) the
`double → float` narrowing edge the sketch admitted turned out UNREACHABLE in the vectorizable node
set (promotion never narrows; `Cast` is scalar-only), so none was implemented — float→int stays
scalar as planned. Scope mirrors 5.1's "arithmetic only" shape: leaves + binary arithmetic/bitwise
(`CanEmitVectorMixed`, default false — bool-typed nodes, Half, unary, min-max, where, logical keep
the whole tree scalar), on a 256-bit AVX2 host (`FusedMixedWidthAvailable`); the plan also serves
UNIFORM-input trees lifted by a weak literal (`i4 + 2.5` → f8) and int true-divide (`i4/i4` → f8).
0-d parameters broadcast at their own dtype's container, so a 0-d and a streamed operand share one
representation. **Target met:** `i4*2+f8` measured **14.2× NumPy @100K / 5.6× @4M** (≥ 4×; 1K sits
at the per-call fixed-cost floor, 1.35×); `f4*f8+f4` 22.2×/7.7×, `i4/i4` 6.0×/4.1×. Gates:
`evaluate.jsonl` 28,311 (+1,008 block A4 over 14 ratio-2 pairs; the 3,344 pre-existing mixed-dtype
cases now take the vector path — all bit-exact), `NDEvaluateVectorTests.MixedWidth_*` (3) + the
flipped plan pins, a 2,234-check vec==scalar==unfused probe (0 fails). Follow-ups deferred: unary/
min-max/comparison nodes in mixed trees (need width-explicit mask + unary emitters), bool operands
(per-width mask reconciliation), the strided-operand gather tier (mixed kernels fall to the scalar
strided loop), V512 hosts.

### Phase 6 — Optimizer passes and bigger levers

6.1 **CSE**: structural hash of subtrees (post-bind), identical subtrees emitted once into a local
(scalar and vector); `Where` branches hoisted only when both sides are cheap or shared (cost model =
node count). 6.2 **Constant folding** at typing time with weak kinds preserved. 6.3 **Single-op
trees** delegate to the direct kernel — ✅ **LANDED** (§0.1): a one-op-over-array-leaves tree
(`abs(a)`, `maximum(a, b)`, `a + b`) has nothing to fuse, so on a plain call over C-contiguous inputs
`np.evaluate` hands it to `ExecuteUnaryOp`/`ExecuteBinaryOp` (`abs(a)` ≡ `np.abs(a)`) instead of paying
the NDIter pass's fixed cost — bit-exact by construction (both paths are oracle-gated to NumPy), gated
to narrow crash-free allow-lists by a delegated-vs-fused sweep. 6.4 **Opt-in
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
