# NDExpr / `np.evaluate` — capabilities beyond the current plan

> Written 2026-09-13, right after the `exprs` branch (P0/P1/P3) and the structural-program-cache +
> hoisted-parameter work landed on master (`45c68405`, `670afdd1`). This document is the SECOND plan
> for the fused-expression engine: everything in it is OUTSIDE `docs/plans/ndexpr-evaluate.md`
> (whose open phases — P2 NumPy-exact reductions, P4 node/keyword coverage, P5 Half + mixed-width
> SIMD, P6 optimizer/threading/string front-end/`Explain` — are unchanged and referenced where an
> item here touches them). Every number quoted is from `benchmark/fusion/probes/fc_probe.cs` and the
> session's parameter probe (Release, one pinned P-core, `DOTNET_TC_CallCountingDelayMs=0`, n = 8
> for fixed cost; NPY/NS = NumPy time ÷ NumSharp time, higher = NumSharp faster). File:line
> citations are into master at `670afdd1`.

---

## 0. Bottom lines and the order to do them in

What `np.evaluate` is today: an immutable `NDExpr` tree, typed per node with NumPy 2.x `result_type`
(NEP 50 weak literals included), compiled ONCE per (structure, dtype signature, 0-d mask) into a
fused NDIter inner loop (4× unrolled SIMD with lane masks for Boolean nodes, AVX2 gather for strided
operands, scalar fallback), driven by ONE iterator pass, with root-only reductions, `out=` with
ufunc overlap rules, and 0-d inputs hoisted into the kernel's aux block as parameters. A tree rebuilt
per call now costs ~0.46–0.51 µs / 0.86 KB at n = 8 (a hoisted tree ~0.41 µs; the unfused
`a*b+c` chain 0.35–0.37 µs; NumPy 0.52 µs), and fused beats the unfused chain on 13 of 15 probe
rows at 1K.

What it still cannot express — and every one of these is a whole class of numeric code, not a
missing ufunc:

| # | Capability | What it unlocks | Effort | Depends on |
|---|---|---|---|---|
| C1 | **Multi-output evaluation** (several roots, one pass, reference-CSE across roots) | optimizer steps in ONE pass (AdamW is 3 passes today, 1.6–1.7× headroom measured), sum+sum-of-squares / min+max in one sweep, `divmod`/`modf`/`frexp`-style pairs | 4–5 d | shell K-output contract |
| C2 | **Reductions inside a tree** (phased reduce → broadcast) | softmax, layernorm, standardization `(x-mean)/std`, `x/sum(x)`, `where(x > mean(x), …)` in one call, no user temporaries | 5–6 d | C1 (for materialize), P2 for exact float folds |
| C3 | **Scan / recurrence / shift nodes** | `cumsum(a*b)`, EMA / IIR recurrences, `diff`, central-difference stencils | 4–6 d | — |
| C4 | **Gather (`Take`) nodes** | lookup tables, embeddings, piecewise-linear interpolation | 3 d | P4 `Cast` (float index rejection text) |
| C5 | **Vectorizable `Call` + partial vectorization** | a user special function no longer scalarizes the whole tree; integer `Mod`/`Power` subtrees stop forcing the scalar path | 3–4 d | — |
| C6 | **Macro / decision combinators** (activations, selectors, boolean logic, predicates, directional, multi-way) — ✅ **LANDED 2026-09-14** (`NDExpr.Combinators.cs`); special-function hooks (`erf`/`gamma`/`i0`) still pending the Cephes ufuncs | readable NN / decision trees with parity by construction | 1–2 d (+ scipy-plan ports) | `docs/plans/scipy.md` §5 |
| C7 | **Complex construction** `Complex(re, im)` | FFT pre/post-processing trees | 0.5 d | P4 `Real/Imag/Conj/Angle` |
| C8 | **Ergonomics**: `np.*` overloads over `NDExpr`, explicit `Eval()`; NO implicit `NDExpr→NDArray` (reason inside) | trees that read like NumPy code | 1–2 d | — |
| C9 | **Fixed cost**: direct kernel dispatch (skip NDIter for identical contiguous operands), the same for all-0-d trees | every small-array call: prebuilt ~0.41 → ≤ 0.30 µs | 1–2 d | — |
| C10 | **Kernel-cache management** (cap/LRU, clear, counters), bit-exact literal keys, analyzer rule NDW019 | no unbounded `DynamicMethod` growth from literal-varying loops; users steered to the parameter form | 2 d | analyzer project |
| C11 | **Broadcast-pattern-specialized SIMD loops** + running pointers in the strided scalar fallback | genuinely broadcast operands (a `(N,1)` column) stop paying the per-load branch (8–10 %); strided scalar +10–15 % | 2–3 d | — |
| C12 | **AOT**: an unfused-chain fallback when dynamic code is unavailable; optional source-generated kernels | NativeAOT correctness now; zero first-call JIT later | 2–3 d (+2–4 w optional) | — |
| C13 | **Public tree API** (`ToString`, visitor, structural equality/hash) and **backends** (ONNX export via the Interop package) | debuggable trees; a fused tree on any ORT execution provider | 3–5 d | — |
| C14 | **Gates**: oracle grammar for every new node, fusion benchmark rows for fixed cost / parameter form / 1K, docs | nothing above ships unverified | alongside each | — |

Recommended order (value ÷ effort, and dependencies): **C9 → C10 → C1 → C2 → C3 → C4 → C5 → C8 →
C6/C7 → C11 → C13 → C12**, with C14 folded into each item's definition of done. C9 and C10 are
one-week-or-less wins that make everything after cheaper to measure; C1 and C2 are the two items
that turn `np.evaluate` from "fuse an elementwise chain" into "fuse a numeric kernel" (an optimizer
step, a normalization, a softmax), which is what the NN / signal-processing consumers actually
write.

---

## 1. The substrate every item builds on (as of `670afdd1`)

Read this before designing any item below; each design is expressed in these terms.

- **Kernel contract** (`NDInnerLoopFunc`): `(void** dataptrs, long* strides, long count, void* aux)`
  over ONE inner chunk; NDIter drives the outer loop (`EXTERNAL_LOOP`). The fused shell
  (`DirectILKernelGenerator.InnerLoop.Fused.cs::CompileFusedInnerLoop(operandTypes, lane,
  scalarBody, vectorBody, cacheKey, prologue)`) wraps a tree's `(scalarBody, vectorBody)` pair in:
  the operand-pointer/stride prologue, an optional user `prologue` (parameters), then the runtime
  dispatch — all-contiguous 4×-unrolled SIMD → contiguous-or-broadcast SIMD (hoisted stride-0
  vectors, selected per load by a branch) → AVX2 gather for strided 32/64-bit lanes → scalar
  contiguous → scalar strided. `operandTypes` is `[inputs..., output]` — exactly ONE output.
- **Aux block**: 16-byte slots. Elementwise kernels: parameters at slot 0..; reduce kernels: slot 0 is
  the flat accumulator, parameters from slot 1 (`NDExprParamPlan.ReduceParamOffset`). The host packs
  it per call with `stackalloc` (`DefaultEngine.Evaluate.cs`).
- **Program** (`NDExprProgram`, `NDExpr.Program.cs`): bound tree, `InputTypes`, `IsParam` mask,
  `Kernel` or `Reduce` + lazily compiled flat/axis reduce kernels, `ResultType`, `ForcedScalar`.
  Immutable, shared. **Resolution**: per-root slot (`NDExprResolution`) → global structural cache
  (`NDExprProgramCache`: 64-bit hash over structure + dtypes + mask + mode, verified node by node
  with `StructureEquals`, 4096-entry cap) → `Build`.
- **Typing** (`NDExpr.Typing.cs`): `InferType` per node into a reference-keyed table; `CompileNumPy`
  plans the vector lane (`NDExprVectorPlan.TryPlan`: one non-bool dtype W, every node W or Boolean,
  every node with a vector emit at W).
- **Host** (`DefaultEngine.Evaluate.cs`): `EvaluateCore` (identical-dims fast path or N-ary
  broadcast, F-order preservation, `out=` cast via a buffered iterator, `COPY_IF_OVERLAP` +
  `OVERLAP_ASSUME_ELEMENTWISE_PER_OP`), `EvaluateReduce` (flat: 16-byte slot, 4-accumulator fold),
  `EvaluateAxisReduce` (`REDUCE_OK`, pinned vs slab kernel).
- **Fixed cost at n = 8** (prebuilt tree, ~0.41 µs / 592 B): NDIter `MultiNew` + dispose ~100–115 ns,
  the result `NDArray` + `Shape` ~100–150 ns, flags/dims ~50 ns, the kernel call itself ~30 ns, the
  rest is the resolution fast path and `FirstArray()` engine dispatch.
- **Gates**: `evaluate.jsonl` (14,702 cases; prefix grammar in `gen_oracle.gen_evaluate` ↔
  `OpRegistry.Evaluate.cs`), `NDEvaluate{Tests,ParityTests,VectorTests,ProgramTests,ParamTests}`,
  the metamorphic vector-vs-scalar sweep (`NDExpr.ForceScalar`), and `benchmark/fusion`.

Two limits to keep in mind, because several items lift them: **one output per kernel** and
**reductions are root-only** (`NDExprProgram.Build` rejects a `ReduceNode` below the root).

---

## 2. C1 — Multi-output evaluation

### Why

`np.evaluate` returns ONE array, so a computation with several results is several passes over the
same inputs. The measured case is the AdamW step (memory `ndexpr-adamw-optimizer-perf`): the three
updates `m`, `v`, `p` are three `np.evaluate` calls because `p` needs the NEW `m` and `v` — 10N
bytes of traffic in 3 passes where a hand-written single-pass SIMD kernel moves 7N in one, and the
hand kernel was **1.6–1.7× faster** at 100K–10M. The same shape recurs everywhere a statistic needs
two folds (`sum` and `sum(x²)` for a variance, `min` and `max` for `ptp`), and in the two-output
ufuncs (`divmod`, `modf`, `frexp`) that NumSharp fuses by hand today.

### API

```csharp
// K roots, one pass; outs optional per root (null = allocate); returns the K results in order.
NDArray[] np.evaluate(NDExpr[] roots, NDArray[] outs = null);

// Tuple sugar for the common arities.
(NDArray, NDArray)          np.evaluate(NDExpr r0, NDExpr r1, NDArray out0 = null, NDArray out1 = null);
(NDArray, NDArray, NDArray) np.evaluate(NDExpr r0, NDExpr r1, NDExpr r2, NDArray out0 = null, NDArray out1 = null, NDArray out2 = null);

// AdamW (PyTorch decoupled weight decay), ONE pass — the m/v subtrees are shared by reference, so
// they are emitted once (reference-CSE, below) and `p` reads the NEW moments per element.
NDExpr mNew = b1 * (NDExpr)m + (1 - b1) * g;
NDExpr vNew = b2 * (NDExpr)v + (1 - b2) * (NDExpr)g * g;
NDExpr pNew = (NDExpr)p * (1 - lr * wd) - step * (mNew / (NDExpr.Sqrt(vNew) / sqrtBc2 + eps));
np.evaluate(new[] { mNew, vNew, pNew }, outs: new[] { m, v, p });

// Two folds in one sweep (each root a reduction): variance's two sums, or min and max for ptp.
var (s, ss) = np.evaluate(NDExpr.Sum((NDExpr)x), NDExpr.Sum((NDExpr)x * x));
var (lo, hi) = np.evaluate(NDExpr.Min((NDExpr)x), NDExpr.Max((NDExpr)x));

// The two-output ufuncs as forests (helpers that return the pair of roots).
var (q, r) = np.evaluate(NDExpr.DivmodRoots((NDExpr)a, b));   // == np.divmod(a, b), one pass
```

### Semantics (the contract the oracle checks)

- Every root is the unfused NumPy chain over the ORIGINAL inputs: per element, all operand values
  are loaded before any root's store, so `np.evaluate(new[]{ a + 1, a * 2 }, outs: new[]{ a, b })`
  gives `b == old_a * 2`, exactly what two separate ufunc calls with `out=` would give when reading
  `a` first. This falls out of the shell: operand values land in locals before the bodies run.
- The iteration shape is the broadcast of the UNION of the roots' inputs; each root has its own
  `result_type`; each `out` follows the ufunc `out=` rules independently (same_kind cast, never
  stretched, may alias an input — `COPY_IF_OVERLAP` covers cross-element overlap as today).
- All roots must be elementwise, OR all roots must be flat reductions, OR all roots must be axis
  reductions over the same axis/keepdims (v1). A mixed forest (an elementwise root beside a
  reduction) is rejected with a clear message until C2's materialize phase needs it.
- Repeated NDArray references dedup across roots (one operand per array), as within a tree.

### Design

- **Shell**: `CompileFusedInnerLoop` gains `outputCount` (`operandTypes = [inputs..., outs...]`). The
  scalar body leaves K values on the stack (root order); `EmitScalarElement` stores them in reverse.
  The vector body leaves K vectors; `EmitFusedStore` runs per output (bool packing per output; the
  4-block packed-mask store stays a K == 1 optimization). The runtime dispatch checks every output's
  stride for contiguity. Gather path: K stores.
- **Reference-CSE** (the minimum needed for the AdamW shape): the compile context memoizes emitted
  subtrees BY REFERENCE (`Dictionary<NDExpr, LocalBuilder>` per body), so a subtree the user shares
  between roots (`mNew` above) is computed once and read from a local. Structural CSE across
  non-identical instances stays P6.1.
- **Reduce forests**: the flat reduce kernel becomes K-accumulator (aux slots 0..K-1, parameters after
  them); the axis kernel writes K output operands (K `REDUCE_OK` operands, each pre-seeded).
- **Program**: `Kernel` + `ResultTypes[]`; the hash folds every root (`NDExprStructureHasher.Add(root
  count)` then each root's structure); `StructureEquals` walks root by root. Reference-CSE does not
  change the key — two roots sharing a subtree by reference vs by copy compute the same thing (only
  the emitted code differs), so the key must NOT distinguish them; the memo is keyed per compile.
- **Host**: `EvaluateCore` allocates K targets, K `out` validations; `EvaluateReduce` reads K slots.

### Edge cases to pin

Aliasing between outs and inputs (elementwise safe; shifted alias → COPY_IF_OVERLAP copy); a root
that is a bare leaf (`np.evaluate(new[]{ (NDExpr)a, (NDExpr)a * 2 })` → a copy plus a product);
bool outputs beside numeric ones; different result dtypes per root under one lane (the vector plan's
"every node W or Boolean" holds per root; a root typed differently than W is a scalar-path forest);
an empty forest (`ArgumentException`); K outs with one null.

### Gates and acceptance

- `evaluate.jsonl`: a `multi(<root>,<root>,…)` grammar producing a `tuple` expected value (the
  harness already asserts tuple arity); reduce forests too.
- Unit: the AdamW forest bit-equal to the three-pass version; the aliasing read-before-write pin;
  every arity sugar.
- Perf: AdamW at 1M/10M ≥ 1.5× the three-pass fused version; `sum+sumsq` one pass ≥ 1.8× two
  passes at 1M.

---

## 3. C2 — Reductions inside a tree (phased reduce → broadcast)

### Why

`NDExprProgram.Build` rejects any reduction below the root ("elementwise use of a reduced value
needs two np.evaluate calls"). That is numexpr's rule, and it keeps the most common fused kernels
out of reach: a softmax, a layer norm, a standardization, `x / sum(x)`, `where(x > mean(x), …)`.
Users write the two calls by hand and materialize the reduction themselves; the engine can do it
without the temporary, and with the reduction's chunking hidden.

### API

No new factories — the existing `Sum/Prod/Min/Max/Mean(expr[, axis, keepdims])` (and P2's
`Std/Var/Any/All/…`) become legal anywhere in a tree:

```csharp
// standardize, one call (two internal phases, no user temporaries)
var z = np.evaluate(((NDExpr)x - NDExpr.Mean(x)) / NDExpr.Std(x));

// softmax over axis 1 — three phases: max → sum(exp(x - max)) → exp(x - max)/sum
NDExpr xm = (NDExpr)x - NDExpr.Max(x, axis: 1, keepdims: true);
var sm = np.evaluate(NDExpr.Exp(xm) / NDExpr.Sum(NDExpr.Exp(xm), axis: 1, keepdims: true));

// layer norm, NumPy's two-pass variance kept (P2's Var is `mean((x - mean)²)`)
NDExpr mu = NDExpr.Mean(x, axis: -1, keepdims: true);
var ln = np.evaluate(((NDExpr)x - mu) / NDExpr.Sqrt(NDExpr.Var(x, axis: -1, keepdims: true) + eps) * gamma + beta);
```

### Semantics

Exactly the NumPy chain with the reduction materialized: the reduction runs over its subtree's full
broadcast input; its result — a 0-d array (flat) or the reduced-shape array (axis; `keepdims`
decides the rank) — takes part in the outer expression by ordinary broadcasting, and as a STRONG
value (NumPy returns an ndarray from a reduction, so `x.mean()` is `np.float64`, never a weak
literal; the parameter machinery already treats a 0-d input that way). A non-keepdims axis result
that does not broadcast against the outer operands fails with NumPy's broadcast error text (e.g.
`x - x.mean(axis=1)` on a `(3,4)` input), because that is what NumPy does.

### Design

- **Planner** (`NDExprPhasePlan`, built at `Build`): post-order walk; a `ReduceNode` whose subtree
  holds no unresolved reduction is assigned the earliest phase; its node is replaced in the parent by
  a `PhaseValueNode(slot)`; repeat until the root (an elementwise tree, or a final reduction) is the
  last phase. Each phase is an ordinary program over inputs + phase values: a flat phase value is a
  hoisted PARAMETER (aux slot — the parameter path built for 0-d inputs is reused as is); an axis
  phase value is a materialized array operand (keepdims shape, or the reduced shape) with stride 0
  on the reduced axes — the broadcast SIMD loop handles it.
- **Recompute vs materialize**: a subtree under a reduction is recomputed in later phases (softmax
  computes `exp(x - m)` twice — numexpr's own rule, and the cheap default). A `Materialize(expr)`
  hint node stores a subtree once during the phase that first needs it; that is a mixed
  elementwise + reduction phase, i.e. C1's "mixed forest", which is why C2's materialize is
  sequenced after C1. Measured guidance goes in the docs: recompute wins for arithmetic, materialize
  wins for `exp`/`log`/transcendental subtrees (CRT-bound at f64).
- **Program**: `NDExprPlan = Phase[]`, each phase a `NDExprProgram` + the slot map; the structural
  hash covers the whole tree (it already folds `ReduceNode` kind/axis/keepdims), so the plan is
  cached like a program. Phase intermediates come from the pooled allocator with `fillZeros:false`
  and are disposed at the end of the call (`[NDScoped]` on the host method).
- **Host**: run phases in order; a flat result is written into the next phase's aux block; an axis
  result is an input operand of the next phase; the last phase writes `out`.

### Edge cases to pin

Empty reduction axis (NumPy: identity for sum/prod, `ValueError` for min/max — the flat host
already raises the verbatim text); a reduction over a subtree that itself contains a parameter;
two reductions over the same subtree (they share the phase — an instance of C1's reduce forest);
`out=` aliasing an input that a phase-one reduction reads (the reduction must run BEFORE any
write — phases are sequential, so it does); dtype: `Mean(int32)` is float64 and promotes the outer
tree; a reduction result at f16/f32 (P2 makes the fold exact at that dtype; until then E1's excuse
applies inside trees too — do not widen it, delete it with P2).

### Gates and acceptance

- Oracle: nested `sum(...)`/`mean(...)`/… tokens with `axis`/`keepdims` attributes inside `expr`
  (the generator's chain evaluator materializes `np.sum(...)` where it meets one); the four
  reference trees (standardize, softmax, layernorm, normalize) at every dtype × layout.
- Unit: phase planning (which reduction lands in which phase), the strong-scalar promotion, the
  broadcast-error text, materialize vs recompute bit-equality.
- Perf: softmax(1M, axis 1) ≥ 1.5× the unfused NumSharp chain and ≥ 2× NumPy; layernorm ≥ 1.5× /
  ≥ 2×; standardize at 100K ≥ 2× unfused.

---

## 4. C3 — Scan, recurrence and shift nodes

### Why

Nothing in NDExpr carries state along an axis, so `cumsum(a*b)` materializes `a*b`, a running
average or an IIR filter is a scalar loop in user code, and a finite difference (`np.diff`, the
`gradient` interior formula) is two sliced operands plus a subtraction the user writes by hand.
`scipy.signal` (`lfilter`, EMA, cumulative statistics) is exactly this class — the SciPy plan
(`docs/plans/scipy.md` §0, "Pattern 2 — sequential IIR recurrences") names it as the work that does
not vectorize and that NumSharp writes kernels for.

### API

```csharp
// Scans (NumPy cumsum/cumprod dtype rules: bool/int → int64, uint → uint64, floats preserved; sequential = exact)
NDExpr.CumSum(expr[, axis]);  NDExpr.CumProd(expr[, axis]);
var c = np.evaluate(NDExpr.CumSum((NDExpr)a * b, axis: 0));            // == np.cumsum(a*b, axis=0), no temp

// Recurrence: y[i] = body(Prev = y[i-1], inputs[i]), Prev = init at i = 0 (NumSharp extension)
NDExpr.Scan(NDExpr body, NDExpr init[, axis]);   NDExpr.Prev   // the carried value, legal only inside a Scan body
var ema = np.evaluate(NDExpr.Scan(alpha * (NDExpr)x + (1 - alpha) * NDExpr.Prev, init: x0));
var iir1 = np.evaluate(NDExpr.Scan(b0 * (NDExpr)x - a1 * NDExpr.Prev, init: 0.0));   // first-order IIR

// Shifted access: Shift(x, k) reads x[i + k]; mode "valid" trims the output to the range every tap
// covers, "same" keeps the length and fills the edge with `fill` (NumSharp extension over slicing).
NDExpr.Shift(NDExpr x, long k, ShiftMode mode = ShiftMode.Valid, double fill = 0);
NDExpr.Diff(NDExpr x, int n = 1, int axis = -1);                          // == np.diff
var d = np.evaluate(NDExpr.Diff((NDExpr)a * b));                          // (N-1,), no temp
var central = np.evaluate((NDExpr.Shift(x, +1) - NDExpr.Shift(x, -1)) / (2 * h));   // (N-2,), the gradient interior formula
```

### Semantics

- `CumSum/CumProd`: `np.cumsum(materialized expr, axis)` bit for bit — NumPy's cumulative loops are
  sequential (`io1 += in2`), so a sequential fused fold IS the exact answer; no pairwise question.
- `Scan`: no NumPy analog; the contract is the scalar loop `y[0] = body(init, in[0]); y[i] =
  body(y[i-1], in[i])`, typed as follows: `Prev` takes the dtype of `init` (a weak literal adopts the
  body's other operands' dtype, a 0-d array is strong), the body is typed, and the RESULT dtype must
  equal `Prev`'s (the carried value must round-trip its own slot) — a body whose result promotes past
  `Prev` is an error naming both dtypes, never a silent narrowing.
- `Shift`/`Diff`: pure view arithmetic — `Diff(x)` is `x[1:] - x[:-1]` (NumPy: `np.diff`); `Shift(x,
  k)` in `Valid` mode is a slice of every operand in the tree so that all taps line up (the tree's
  output is the common valid range), in `Same` mode the valid core plus edge fills. Because it is
  slicing, the kernel path is the ordinary fused loop and every tap is a contiguous view → SIMD.

### Design

- **`ScanNode`** (root only, like `ReduceNode`; a scan below the root is a phase in C2's planner —
  its result is an array operand): a raw inner loop (the reduce kernel's shape) with the carried
  value in an aux slot across chunks, driven in C order along the scan axis. Flat scans over the
  raveled C-order input use `NPY_CORDER` with EXTERNAL_LOOP (the carry lives in aux, so chunking is
  invisible). Axis scans drive one lane at a time with the `IterAllButAxis` driver
  (`AxisSort.DriveAllButAxis`, the partition/sort driver) — a strided lane is the general case, a
  contiguous lane the fast one; `DirectILKernelGenerator.Scan.cs` (`AxisCumSumHelper`) is the
  non-fused template.
- **`Scan(body, init)`**: the same kernel with the body's `EmitScalar` in place of the fold and `Prev`
  reading the carry local. No vector body (a true recurrence does not vectorize; SIMD across
  independent lanes of an axis scan — the slab form — does, and is the perf lever: for axis scans
  where the scanned axis is NOT the inner one, `count` consecutive elements belong to `count`
  different lanes, so a vector of carries advances together — the slab path of the axis-reduce
  kernel is the model).
- **`Shift`/`Diff`**: a BIND-TIME rewrite, no kernel: the binder collects every `Shift` tap on a
  leaf, computes the common valid window `[max(-k), N - max(k))` per axis, replaces each tap by the
  correspondingly sliced view of the array (a new operand; two views of one base do not dedup —
  correct, they read different windows) and slices every other operand in the tree to the same
  window. `Same` mode = evaluate the valid core into the middle of a full-length output and fill the
  edges (host-side, two tiny writes). `Diff(x, n)` = the n-fold `Shift` composition NumPy's
  recursion spells (`np.diff` applies `a[1:] - a[:-1]` n times) — for n > 1 recompute, don't
  materialize, it is cheap.
- An `IndexNode(axis)` (the running coordinate, NumPy's broadcast `arange`) is a natural companion
  (ramps, `Where(Index(0) < n, …)`); v1 materializes it as an `arange` operand on demand (the kernel
  gets no coordinates from EXTERNAL_LOOP chunks), which is one temp of the output's length — fine for
  1-D, revisit if it shows up in profiles.

### Gates and acceptance

- Oracle: `cumsum(...)`/`cumprod(...)` (with axis) and `diff(...)` tokens (NumPy chain: `np.cumsum`,
  `np.diff`); `Shift` in `Valid` mode as slicing in the generator.
- Metamorphic (no NumPy analog): `Scan` vs a C# scalar loop over the materialized inputs, every
  dtype, flat and axis, strided/negative-stride/broadcast lanes; `Same`-mode shifts vs the sliced
  spelling plus fills.
- Perf: `cumsum(a*b)` at 1M ≥ 1.5× `np.cumsum(a * b)` (it saves the temp; the scalar sequential
  fold is what NumPy runs too); EMA at 1M ≥ 5× a managed scalar loop over `NDArray` element
  accessors (the honest comparison — NumPy has no EMA; SciPy's `lfilter` is the C reference).

---

## 5. C4 — Gather (`Take`) nodes

### Why

A per-element table lookup inside a fused tree — tone-mapping LUTs, embedding rows, piecewise-linear
interpolation from precomputed bins, categorical parameters — is today a `np.take` pass plus a
second pass for the arithmetic. The take/put kernels (`DirectILKernelGenerator.{Take,GatherFlat}.cs`)
and the shell's AVX2 gather support (`TryGetGatherSupport`) already contain the pieces.

### API

```csharp
NDExpr.Take(NDArray table, NDExpr index, TakeMode mode = TakeMode.Raise);   // table: any array, read flat (np.take axis=None)

// LUT tone mapping: one pass, the index computed in-tree
var mapped = np.evaluate(NDExpr.Take(lut, NDExpr.Floor((NDExpr)x * 255.0)));   // index must be integer-typed → wrap it in P4's Cast

// piecewise-linear interpolation with precomputed bin index `i` (int64 array)
var y = np.evaluate(NDExpr.Take(ys, i) + ((NDExpr)x - NDExpr.Take(xs, i)) * NDExpr.Take(slopes, i));

// embedding lookup + arithmetic
var e = np.evaluate(NDExpr.Take(E, ids) * scale + bias);
```

### Semantics

`np.take(table, index_array, mode=…)` with `axis=None` — the table is raveled in C order, the index
array is the evaluated `index` subtree, which must be integer-typed (NumPy's `same_kind` cast rule
for `take`: bool and every integer width pass, floats raise `TypeError: Cannot cast array data from
dtype('float64') to dtype('int64') according to the rule 'same_kind'` — reproduced at typing time
with the verbatim text). Modes: `raise` (index normalized once, `IndexError` with NumPy's text
carrying the original value), `wrap` (Python modulo), `clip`. Result dtype = the table's dtype.

### Design

- The table is a **table operand**: not iterated — its base pointer and length go into an aux slot
  (16 bytes: pointer + count), the way parameters travel; it must be contiguous (the host `ravel`s a
  non-contiguous table once per call — a documented copy). The structural hash folds "table at
  input i" as a distinct leaf kind; the dtype signature carries its dtype.
- Scalar emit: index → `long`, mode handling, `ldind` from `base + idx*size`. Vector emit (32/64-bit
  tables and lanes): `Avx2.GatherVector256` with the index vector converted to `int32`/`int64`
  lanes; `wrap`/`clip` as vector ops; `raise` runs a vector bounds compare and takes the scalar path
  for the group when any lane is out of range (the out-of-range lane then raises with the right
  value). Without AVX2 the node is scalar-only (`CanEmitVectorV2` false).
- Aliasing: a table that is also an iterated operand (`Take(x, idx) + x`) is fine — read-only both
  ways; a table aliasing `out` is rejected up front (the gather would read written data) with a clear
  message — NumPy's `np.take(a, idx, out=a)` is likewise undefined.

### Edge cases and gates

Empty table with any index (`raise` → IndexError; `wrap` → NumPy's `ZeroDivisionError` text —
reproduce); negative indices per mode; bool index arrays (legal: cast to int); 16-byte dtypes
(scalar only); a 2-D table (flat semantics — document that `axis=` is P4-adjacent work).
Oracle: `take(inK, <expr>, mode)` tokens (NumPy chain: `np.take(table, idx, mode=…)`); unit tests per
mode × dtype; perf: LUT map at 1M ≥ 1.5× `np.take` + chain, embedding lookup ≥ 1.5×.

`SearchSorted(xs, e)` (a binary search per element, scalar, `O(log n)`) is the natural second node
for interpolation without precomputed bins; it is C4.2, optional, and gated by `np.searchsorted`.

---

## 6. C5 — Vectorizable `Call` and partial vectorization

### Why

`CallNode.SupportsSimd => false` scalarizes the WHOLE tree. `np.kaiser` is the in-repo example:
`i0(beta·sqrt(1 - ((n-α)/α)²)) / i0(β)` runs at ~44 ns/element (46 µs at 1K) because the Bessel
call drags the vectorizable `sqrt`/arithmetic subtree onto the scalar path; the same happens to any
tree with an integer `Mod`/`Power`/`FloorDivide` node (scalar-only ops today) or a user function.

### API

```csharp
// (a) a delegate pair: the vector twin runs per group, the scalar one in the tail and the scalar fallback
NDExpr.Call(Func<double, double> scalar, Func<Vector256<double>, Vector256<double>> vector, NDExpr x);

// (b) a static class + name: binds the scalar overload and any Vector128/256/512<T> or System.Numerics.Vector<T>
//     overload it finds; picks the host width, falls back to the scalar overload where none matches
NDExpr.Call(typeof(Special), nameof(Special.Bessel0), arg);

public static class Special
{
    public static double Bessel0(double x) { … }                       // the scalar contract
    public static Vector256<double> Bessel0(Vector256<double> x) { … }  // the twin: MUST be lane-for-lane bit-equal
}
```

### Design

- `CallNode` gains an optional vector `MethodInfo` per host width; `CanEmitVectorV2` is true when a
  twin matches the lane type (`Vector<T>` twins are accepted when `Vector<T>.Count * sizeof(T) * 8
  == VectorBits`, reinterpreted with `AsVector256()`/`AsVector()`; otherwise ignored). The vector emit
  is a direct `call` for a static twin, a slot lookup + `callvirt` for a delegate — one managed call
  per group of W lanes, the same shape as today's scalar call per element.
- **Partial vectorization** (automatic, no API): when a node has no vector emit but its children do,
  the emitter extracts lanes: child vector → for each lane `GetElement`, the node's scalar emit,
  `WithElement` into the result vector. The cost model that turns it on: only when the node's scalar
  emit is "expensive" (a `Call`, a CRT transcendental, integer `Mod`/`Power`/`FloorDivide`) — for
  cheap scalar-only nodes the extraction overhead outweighs the vectorized remainder. Measured
  expectation, honestly: kaiser gains little (the Bessel call is ~90 % of the element), a tree like
  `erf(a*b + c) * d` or `(a % 7) * b + c` gains 1.5–2× (the vectorizable share dominates).
- **Contract check** (opt-in `verify: true` on the `Call` factory): at first compile, run the scalar
  and vector twins over a probe vector and require bit-equality — a wrong twin then fails loudly at
  the call site instead of diverging silently at scale.

### Gates and acceptance

Metamorphic: every `Call` tree bit-equal under `ForceScalar`; a deliberately divergent twin fails
`verify`. Perf: `erf(a*b+c)*d` (a cephes `erf` twin) ≥ 1.5× the scalar-path version at 100K; `(a %
7) * b + c` ≥ 1.5× via partial vectorization.

---

## 7. C6 — Macro nodes and special-function hooks

### Why

NN trees are written from a small vocabulary — sigmoid, softplus, gelu, swish, relu variants — that
NumPy does not ship as ufuncs; users spell them as compositions every time, and the DSL has no
canonical (and therefore no vectorized-by-construction) form for them.

### API and semantics

Macro nodes are pure compositions of existing nodes, so their parity contract is the composition's
unfused NumPy chain — bit-exact by construction, and the oracle's chain evaluator spells the same
composition:

```csharp
NDExpr.Sigmoid(x)     // 1 / (1 + Exp(-x))
NDExpr.Softplus(x)    // LogAddExp(0, x)   (P4's LogAddExp; NumPy: np.logaddexp(0, x) — the numerically stable form)
NDExpr.Relu(x)        // Max(x, 0)                                (NaN-propagating like np.maximum)
NDExpr.LeakyRelu(x, s) // Where(x > 0, x, s * x)
NDExpr.Elu(x, a)      // Where(x > 0, x, a * Expm1(x))
NDExpr.Swish(x)       // x * Sigmoid(x)
NDExpr.Gelu(x)        // 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x³)))   — the tanh form; the erf form waits for Erf
NDExpr.HardSigmoid(x) // Clip(x / 6 + 0.5, 0, 1)
```

Special functions (`erf`, `erfc`, `gamma`, `lgamma`, `i0`, `expit`) are NOT NumPy ufuncs; NumPy
itself has none of them (SciPy does, via Cephes). The SciPy plan (`docs/plans/scipy.md` §5) lands the
Cephes ports in NumSharp as ordinary `UnaryOp` kernels with a `scipy==1.16.3` oracle. The hook here
is one line per function once those exist: a `UnaryNode` factory + the float-tier typing rule
(`UnaryFloatResult`) + the vector emit when the port ships a vector body. `np.kaiser`'s private
`BesselI0` becomes the first consumer (its `Call` disappears, and with it the scalar path).

### Gates

Oracle: each macro as its composition in the grammar (e.g. `sigmoid(in0)` expands generator-side to
`1/(1+exp(-x))`); unit tests pin the exact composition (so a future "faster sigmoid" that changes
bits is a deliberate, tested decision, not a drift).

### Landed (2026-09-14, `NDExpr.Combinators.cs`)

The macro/decision half of C6 is complete — 39 static factories on `NDExpr`, every one a pure
composition of the primitive nodes (no new node type, no new IL emitter, no typing/structure/vector
change), so each fuses into the same `np.evaluate` pass, is correct by construction, and inherits
whatever SIMD path its underlying `Where`/`Min`/`Max`/arithmetic nodes have. Verified by
disassembling each scalar body (persisted-assembly round-trip): **22 compile to PURE inline IL**
(the selection family via `brfalse`/`br`, boolean logic via bitwise ops, comparisons/predicates via
`ceq`/`clt`/`cgt`, arithmetic like `Lerp`), and **17 emit exactly one shared helper `call`** — the
min/max family to `Double{Max,Min}NaN` (the np.maximum/minimum body, so fused == unfused == NumPy),
the transcendental activations to `Math.Exp`/`NDFloatMath.Tanh`/`Math.Log`, and the predicates that
need one primitive to `Double.IsNaN`/`IsFinite`/`Math.Floor`/`Math.Abs`. The call is the same
per-element work the equivalent `np.*` does; fusion still removes the intermediate arrays around it:

- **Selection & masking**: `If`, `IfNot`, `When`, `Unless`, `Switch` (multi-way, first-match-wins),
  `Mux` (integer-indexed).
- **Clamp/saturate**: `ClampMin`, `ClampMax`, `Saturate`.
- **Robustness**: `NanTo`, `Coalesce` (first-finite).
- **Activations**: `Relu`, `LeakyRelu`, `Elu`, `Sigmoid`, `Swish`, `Softplus` (stable
  `max(x,0)+log1p(exp(-|x|))`), `Gelu` (tanh form), `HardSigmoid`, `Step`.
- **Boolean logic**: `Nand`, `Nor`, `Xnor`, `Implies`, `Majority3`.
- **Predicates**: `IsPositive`, `IsNegative`, `IsInteger`, `IsClose` (np.isclose relation, equal_nan=False),
  `SameSign`, `Between`.
- **Directional/sign**: `Cmp` (3-way), `Heaviside`, `StepToward`, `MaxMagnitude`.
- **Multi-way/interp**: `Bucketize` (np.digitize; sums INTEGER `Where(…,1,0)`, since `bool+bool` is
  logical OR, not a count), `Median3` (branch-free), `Threshold`, `Lerp`.

**ML hot path (float32) is fully vectorized** (verified by disassembling the vector body). On a
float32 array `Relu`/`LeakyRelu`/`HardSigmoid`/`Step` are PURE hardware intrinsics (`Avx.Max`/`Min`/
`Add`/`Mul` + `Vector256.ConditionalSelect`/`GreaterThan` — one machine instruction each, no real
call), and `Sigmoid`/`Swish`/`Elu`/`Softplus`/`Gelu` are hardware intrinsics plus ONE call per 8-lane
group into the NumPy-ported `NDFloatMath.Exp`/`Log`/`Tanh` SIMD kernel (the same per-group approach
NumPy uses). To keep it that way, `Elu` is composed over `exp(x)-1` (NOT `expm1`) and `Softplus` over
`log(1+…)` (NOT `log1p`): `expm1`/`log1p` have no SIMD emit, so a single such node would force the
whole activation onto the per-ELEMENT scalar path (bounded, documented precision give-up per method).
The remaining `call` — a scalar `Math.Exp`/`Log`/`Tanh` on a **float64** activation — is a dtype
limit, not this item's: no bit-exact vector f64 transcendental exists, and NumPy is scalar there too.

**Deliberately deferred to P4**, NOT composed here so they never collide when P4 adds them as
first-class (vectorizable) ufunc nodes: `fmax`/`fmin`, `copysign`, `nextafter`, `logaddexp`, the
shifts, and the `np.select`/`np.clip` nodes (which is why the multi-way selector is named `Switch`
and one-sided clamps ride `Min`/`Max`). Special functions (`erf`/`gamma`/`i0`/`expit`) still wait on
the Cephes ports (`docs/plans/scipy.md` §5); each is then one `UnaryNode` factory line.

**Gate**: `NDExprCombinatorTests` (36) — the metamorphic gate C14 allows for composition nodes:
each fused combinator compared to its eager `np.*` reference (or a hand-verified NumPy 2.4.2 output),
discrete results bit-exact, transcendental activations within allclose; plus fusion-parity (a
combinator inside a chain equals the unfused `np.*` chain) and cross-combinator composition. The
oracle-grammar tokens (`sigmoid(in0)` → its composition in `gen_evaluate`) remain the one open
follow-up for this item; the metamorphic gate is airtight for compositions in the meantime.

---

## 8. C7 — Complex construction

`NDExpr.Complex(re, im)` builds `complex128` lanes from two real subtrees (FFT input assembly,
polar-to-rectangular `Complex(r * Cos(θ), r * Sin(θ))`). It is a NumSharp extension: NumPy's
idiom `re + 1j*im` goes through a complex multiply and gives `nan+nanj` for an infinite `im`
(`0*inf`), while direct construction gives `(re, inf)`; the node documents this and the test pins it.
P4 already lists `Real/Imag/Conjugate/Angle`; together they close the complex round-trip. Vector
emit: the pair interleave the complex cast kernels use (`Cast.Complex.cs`); scalar emit: `newobj
Complex(double, double)` (`ConstNode` already emits it).

---

## 9. C8 — Ergonomics: `np.*` over `NDExpr`, explicit `Eval`, and why NOT an implicit conversion

### `np.*` overloads returning `NDExpr`

Every elementwise `np.*` NumSharp ships gains an `NDExpr` overload that returns the corresponding
node, so a tree can be written in the vocabulary users already know:

```csharp
NDExpr x = a;                                             // one cast at the head
var r = np.evaluate(np.sqrt(np.square(x) + np.square(b)) * np.where(x > 0, x, 0.01 * x));
```

No overload-resolution hazard: `np.sqrt(NDArray)` is an exact match for an array argument, the
`NDExpr` overload is an exact match for a tree, and the `NDArray → NDExpr` implicit conversion is
never preferred over an exact match. Comparison operators `< > <= >=` on `NDExpr` are P4.4.

### `expr.Eval(out)` and `expr.Eval(operands, out)`

Instance sugar over `np.evaluate` for fluent code (`(x * y).Eval()`); zero semantics of its own.

### Why there must be NO implicit `NDExpr → NDArray` conversion

It looks like the biggest usability win (`NDArray y = (NDExpr)a * b + c;` would evaluate on
assignment) and it is a trap the DType work already fell into once (memory `dtype-system-stage-a`):
with implicit conversions in BOTH directions, `expr == null` would bind `NDArray.operator ==(NDArray,
NDArray)` through the conversion — every null check would silently EVALUATE the tree and compare
arrays — and `expr == ndarray` would do the same. C# picks the user-defined operator over reference
equality whenever a conversion makes it applicable. So: explicit `Eval()`/`np.evaluate` only, and a
unit test that asserts `NDExpr` declares no `==`/`!=` and no implicit conversion to `NDArray`.

### A lazy scope (`using (np.lazy()) { … }`) — not planned

Making plain `NDArray` operators build trees inside a scope needs every operator (and every
`np.*` entry) to check a thread-static mode and return a deferred subclass, and every consumer of an
`NDArray` to materialize it on data access. That is a whole-library change for a convenience the
`np.*` overloads above already deliver with one cast; recorded here so the idea is not re-opened
without that cost in view.

---

## 10. C9 — Fixed cost: direct kernel dispatch and the scalar fast path

### Why

At n = 8 a prebuilt tree costs ~0.41 µs, of which NDIter construction + disposal is ~100–115 ns and
the result allocation ~100–150 ns; the kernel itself is ~30 ns. For the overwhelmingly common
call — every operand C-contiguous with identical dims, no cast — the iterator does nothing the
kernel cannot do in one call.

### Design

`EvaluateCore` fast path: when every iterator operand is C-contiguous (or every one F-contiguous
— the result is allocated in that order anyway), dims identical, `out` null or an exact-shape
contiguous array of the result dtype, and no operand overlaps `out` except as an EXACT alias (same
base, same offset — elementwise-safe), call the kernel directly:
`kernel(ptrs, strides = itemsizes, count = size, aux)`. Overlap is decided with the existing bounds
solver (`NDMemOverlap.SolveMayShareMemory(maxWork: 0)`, the `putmask` guard); anything else takes
the NDIter path unchanged. The same direct call serves the all-0-d case (`count = 1`), which today
also builds an iterator for one element.

Expected: prebuilt `a*b+c` ≤ 0.30 µs (from 0.41), rebuilt ≤ 0.38 µs, and every 1K row moves by the
same ~110 ns — the two single-op rows (`maximum`, `abs`) that still lose to the direct kernels
(0.59×, 0.63×) come within reach without P6.3's delegation.

### Gates

Bit-equality of the direct path vs the iterator path across the layout sweep (the metamorphic
harness already toggles `ForceScalar`; add a `ForceIterator` hook the same way); the aliasing
matrix (exact alias, shifted view of the same base, disjoint views) pinned; the fixed-cost rows in
the benchmark sheet (C14).

---

## 11. C10 — Kernel-cache management, bit-exact literal keys, analyzer NDW019

### The leak

Literal values are baked into the kernel's IL and its cache key, so `for t in …: np.evaluate(a *
t)` compiles one `DynamicMethod` per value and stores it in `DirectILKernelGenerator._innerLoopCache`
forever — the structural program cache is capped (4096, clears) but the kernel cache under it is
unbounded, and a `DynamicMethod` held by a live delegate is never collected. The parameter form
(`NDArray.Scalar(t)`) is the remedy, and the docs now say so; the engine should also defend itself.

### Design

- `_innerLoopCache` becomes a bounded, generation-evicting cache (cap 8192; on overflow drop the
  oldest generation — an LRU is overkill for a cache whose hit rate is ~100 % in steady state);
  `DirectILKernelGenerator.ClearInnerLoopCache()` for tests/long-running hosts; `GeneratedDelegates`
  gains hit/miss/evict counters like `NDExprProgramCache` has.
- The literal part of the kernel key uses the value's BITS (`DoubleToInt64Bits` in hex, the decimal's
  four `GetBits` words), not `"R"` formatting: today every NaN payload and sign formats as `NaN`, so
  two literals with different NaN bits share one kernel and the second silently gets the first's
  payload. The structural hash already keys by bits; the kernel key must agree.
- **NDW019** (Roslyn analyzer, the `NumSharp.Analyzers` project — id-FIRST in
  `AnalyzerReleases.Unshipped.md`): inside an `np.evaluate(...)`/`NDExpr` operator expression, a
  numeric operand that is a local, parameter, field or method call (anything but a compile-time
  constant) is reported: "`t` is baked into the fused kernel as a literal — one kernel per distinct
  value; pass `NDArray.Scalar(t)` (a hoisted parameter, one kernel for every value)". A code fix
  rewrites it.

---

## 12. C11 — Broadcast-pattern-specialized SIMD loops and running pointers

### Why (measured)

Before hoisting, a stride-0 operand cost 8–10 % per element on BOTH kernel paths: the SIMD path's
`EmitFusedLoad` selects "load or hoisted vector" with a branch per operand per group (loop-invariant,
but RyuJIT does not unswitch loops), and the strided scalar fallback computes `ptr + i * stride` per
operand per element. Parameters remove it for 0-d inputs only; a genuinely broadcast operand — a
`(N,1)` column against a `(N,M)` matrix (inner stride 0), or `np.broadcast_to` views — still pays it,
and every strided scalar loop still multiplies.

### Design

- `EmitFusedSimdLoop(allowBroadcast: true)`: emit one loop copy per SINGLE-broadcast-operand pattern
  for `nIn ≤ 4` (the copies select on the runtime mask once, before the loop; each copy has branch-free
  loads), keep the generic branchy loop for multi-broadcast masks. Code size grows by `nIn` copies of
  the unrolled body — bounded by the operand cap; trees with more inputs use the generic loop.
- `EmitScalarStridedLoop`: running pointers (`ptr_op += stride_op` per element, the reduce kernels'
  shape) instead of the multiply; the contiguous scalar loop already folds constant strides.

Expected: `(N,1)`-column rows +8–10 %, strided scalar rows +10–15 %; gated by the layout sweep in
`NDEvaluateVectorTests` (bit-equality is unaffected — only addressing changes).

---

## 13. C12 — AOT: an unfused-chain fallback now, source-generated kernels later

### Tier A — correctness under NativeAOT (2–3 days)

`CompileFusedInnerLoop` is `DynamicMethod`-based; under NativeAOT (`RuntimeFeature.IsDynamicCodeSupported
== false`, the "PublishAot disables kernels" trap) `np.evaluate` cannot run at all. The cheapest
correct fallback is the oracle's own definition: walk the bound tree and evaluate it as the UNFUSED
`np.*` chain (each `BinaryNode` → the engine's binary ufunc at the node's resolved dtype, each
`UnaryNode` → the unary ufunc, `Where` → `np.where`, reductions → the engine's reductions),
materializing intermediates. It is bit-identical by construction (the chain IS the contract), slower
(no fusion), and reuses the engine's AOT-safe managed kernels. `NDExpr.Program` picks it when dynamic
code is unavailable (or on an explicit `NDExpr.ForceUnfused` test hook, which doubles as a
self-consistency oracle for every other item in this document — a no-Python differential gate).

### Tier B — source-generated kernels (optional, 2–4 weeks)

A Roslyn source generator that finds `np.evaluate(<tree>)` call sites whose tree is statically known
(leaves, ops, literals) and emits, at build time, a C# kernel per (structure, dtype signature) with
the same scalar/vector bodies the IL emitters produce — registered under the same cache key by a
`[ModuleInitializer]` so `CompileNumPy` finds it before reaching for `DynamicMethod`. Value: NativeAOT
at full speed and zero first-call JIT (~0.2–1 ms per new tree today). Cost: a second emitter backend
(C# text) mirroring every IL emitter — which is why it is optional and last; the emitters would have
to be refactored to a small "emit operation" abstraction with IL and C#-text implementations.

---

## 14. C13 — Public tree API and backends

### Tree API (1–2 days)

- `NDExpr.ToString()`: a readable rendering (`(in0 * in1) + 2.5`, `Sum(exp(in0 - Max(in0, axis=1, keepdims)), axis=1)`),
  and `ToString("prefix")` for the oracle grammar's spelling (`add(mul(in0,in1),lf:2.5)`), so a failing
  tree can be pasted into `gen_oracle.py` as a case.
- `NDExpr.Accept<T>(INDExprVisitor<T>)` — a public visitor over the node kinds (the shape
  `BindArrays`/`HashStructure` already follow), so tools and backends can walk trees without
  reflection.
- `NDExpr.StructurallyEquals(NDExpr)` and `GetStructuralHash()` — the identity the program cache uses,
  exposed (tests, dedup in user code, memoization).

### Backends (3–5 days for the ONNX export)

The tree is engine-neutral and `TensorEngine.Evaluate(NDExpr, …)` is virtual: an alternative engine
overrides it and translates the tree with the visitor. The concrete, cheap, high-value backend is an
**ONNX export** in `NumSharp.Interop.OnnxRuntime`: `expr.ToOnnxModel(inputNames)` maps nodes to ONNX
ops (`Add/Sub/Mul/Div/Pow/Sqrt/Exp/Log/Sin/…/Where/Max/Min/Less/…/ReduceSum/ReduceMax/…`) with the
dtype map the package already has, and `session.Run(NDArray…)` executes it on ANY ORT execution
provider — a fused NumSharp tree on a GPU without a GPU engine in NumSharp. Parity is ORT's, not
NumPy's (documented; the export is for throughput, not for the oracle).

---

## 15. C14 — Gates that ship with every item

- **Oracle grammar** (`gen_oracle.gen_evaluate` ↔ `OpRegistry.Evaluate.cs`, same tokens both sides):
  `multi(r0,r1,…)` (tuple expected, C1); nested reductions with `axis`/`keepdims` attributes (C2);
  `cumsum/cumprod/diff` (C3); `take(inK, <expr>, mode)` (C4); macro names expanding to their
  compositions generator-side (C6). NumSharp-only nodes (`Scan`, `Shift(Same)`, `Complex`, the
  `Call` twins) are gated by the metamorphic harness (fused vs `ForceScalar` vs C12's
  `ForceUnfused`) and classified `SiblingOwned` in `OracleSurfaceCoverageTests` with the reason.
- **Benchmark sheet** (`benchmark/fusion`): promote `fc_probe.cs`'s sections into
  `evaluate_bench.cs`/`fusion_sheet.py` — the fixed-cost rows (rebuilt vs prebuilt vs parameter
  form), the 1K rows, the consumer trees — so the regression the P3 landing would have shipped
  (1.58 µs rebuilt) is visible in `run_benchmark.py`; then one row per item above (AdamW 1-pass vs
  3-pass, softmax, layernorm, `cumsum(a*b)`, EMA, LUT map, `erf` twin).
- **Docs**: `docs/website-src/docs/NDIter.md` Tier 3C catalog (one row per node with its SIMD
  status), `.claude/CLAUDE.md` "Fused Expressions", and this document's status table (§0) updated as
  items land — the plan doc is the ledger, keep it current the way `ndexpr-evaluate.md` §0.1 is.

---

## 16. Non-goals and risks

- **Random numbers inside kernels** (dropout masks): feeding a pre-generated random array is the
  contract — a NumPy-parity bit generator per lane inside a fused loop is not worth its complexity.
- **Threading**: stays P6.4 (opt-in, deterministic merge); nothing here assumes it.
- **A GPU engine**: only the ONNX export (C13); a native GPU `TensorEngine` is a separate plan.
- **Full numexpr syntax**: P6.5's string front-end; C8's `np.*` overloads are the C#-native answer.
- **Risk — semantics drift**: every extension node must state its NumPy chain (or, for
  NumSharp-only nodes, its scalar-loop contract) before it gets a kernel, and the oracle/metamorphic
  gate must exist in the same commit. The one-line rule from the parity work applies unchanged:
  bit-exact to the stated chain, or a documented `[Misaligned]` with a deletion condition.
- **Risk — kernel-count growth**: C1 (forests), C2 (phases), C3 (scan kinds) and C4 (modes) each add
  kernel variants; C10's bounded cache and the parameter rule keep it finite, and the structural hash
  keys every variant explicitly so nothing is ever shared by accident.
- **Risk — the shell's one-output assumption** is baked into `EmitScalarElement`, `EmitGroup`, the
  packed bool store and the gather/scatter dispatch; C1 touches all of them and must ship with the
  16,831-check-style self-consistency sweep the 2-D kernel work used (`NDEvaluateVectorTests` is the
  template).

---

## Appendix A — the numbers this plan rests on (2026-09-13, `670afdd1`)

| Measurement | Value |
|---|---|
| `np.evaluate((NDExpr)a*b+c)` rebuilt per call, n = 8 | 464–508 ns / 864 B (old NDExpr 995 / 3056; P3-only 1580 / 3136) |
| same tree hoisted | 397–414 ns / 592 B |
| `evaluate(expr, out=)` rebuilt | 353–363 ns / 456 B |
| `Sum(a*b)` / `Where(a>b,a,b)` rebuilt | 368–388 / 448–456 ns |
| `a*b+k`, 0-d `k` hoisted | 388 ns prebuilt, 481 rebuilt (was 681 / 1255–1304) |
| stride-0 operand per element (before hoisting) | +8–10 % on SIMD and scalar paths; hanning tree 408 vs 372 µs at 100K |
| `np.hanning(M)` over 20 distinct M | 0 kernels (was 20; 0.46 ms vs 3.64 ms) |
| fused ≥ unfused at 1K, rebuilt spelling | 13 of 15 rows (losses: `maximum(a,b)` 0.59×, `abs(a)` 0.63×) |
| `np.kaiser(1000, 5)` | 46 µs (~44 ns/element, `Call`-forced scalar path) |
| AdamW fused (3 passes) vs hand single-pass SIMD | hand kernel 1.6–1.7× faster at 100K–10M (memory `ndexpr-adamw-optimizer-perf`) |
| NDIter construction + disposal per call | ~100–115 ns; result allocation ~100–150 ns |
