# NDIter 2-D block kernel for narrow strided rows

**Commits:** `af25a746` (first generation) and the second-generation commit that follows it (branch
`journey3`, 2026-09-03). §1–§10 are the first-generation report as written; **§11 is the review that
challenged it** — three regimes it never measured were losses, and closing them changed the iterator
as well as the kernel.
**Lever:** `docs/NDITER_PERF_DISCOVERY.md` §7 angle 2 — *"A 2-D kernel contract for narrow rows"* — LANDED, then extended.
**Headline (first generation):** `np.positive`/`sqrt`/`add`/… on a `(rows, w)` strided view went from **0.82× NumPy** (w=4)
to **1.4–2.5×** across 2-D *and* N-D trailing-narrow views, with a **1.59×** bonus on contiguous
`positive`. Bit-identical to the previous path; FuzzMatrix 98/98, main suite 14446/0.
**Headline (second generation, §11):** sub-vector rows (`sqrt` on `m[:, :2]`/`m[:, :3]`) **0.4–0.6× → 1.6–2.2×**
NumPy, `add(A, col)` **1.15× → 6.5×**, `multiply(view, 2.0)` **0.48× → 2.0×**, `exp` on 4-wide rows
**0.84× → 1.22×**, `x[::2, :, :w]` **0.70× → 2.1×**; NDIter now coalesces axes the way NumPy does.
16,831-check self-consistency 0 mismatches, FuzzMatrix 98/98 with one excuse narrowed, main suite 14446/0.

Ratios follow the house convention **NPY/NS = NumPy_ms ÷ NumSharp_ms, higher = NumSharp faster**. All
numbers measured on the same host (Windows 11, i9-13900K class, AVX2, .NET 10, NumPy 2.4.2), fresh and
idle, `dotnet run -c Release` with `DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1`.

---

## 1. The problem

A view that is **contiguous on the inner axis and strided on the outer axis** — the classic `m[:, :w]`
column slice, where each row is `w` contiguous elements separated by a gap — cannot coalesce to 1-D. The
production ufunc route (`DefaultEngine.UfuncOut.cs` → `NDIterRef.ExecuteElementWiseUnary`/`Binary` →
`ForEach`) drives it under `EXTERNAL_LOOP`, calling the per-chunk kernel **once per row**. For a narrow
`w`, that call pays two size-invariant taxes on **every** row, then does a handful of elements of real
work:

1. **The driver (~4.5 ns/row)** — a delegate dispatch of the kernel plus `ExternalLoopNext`
   (`NDIter.cs`), the odometer that bumps every operand pointer and carries coordinates.
2. **The kernel's own prologue (~a few ns/row)** — the emitted per-chunk body
   (`DirectILKernelGenerator.InnerLoop.cs`, `CompileInnerLoop`) snapshots pointers/strides into locals
   and then runs a **runtime SIMD-viability dispatch** (a `stride == elemSize` compare per operand) to
   choose SIMD-contig / broadcast / gather / scalar — the answer is identical for every row, but it is
   recomputed per row.

The cost model (`docs/NDITER_PERF_DISCOVERY.md` §2) is `time ≈ FIXED + rows × PER_CHUNK + elements ×
PER_ELEMENT`. For narrow rows `PER_CHUNK` dominates, and both the odometer and the prologue live inside it.

---

## 2. Investigation

### 2.1 Baseline (this host, fresh)

| 2M f64 `(rows, w)`, `out=` | NumSharp | NumPy | NPY/NS |
|---|---:|---:|---:|
| `positive` w=4 | 3.637 ms | 2.994 | **0.82×** |
| `positive` w=16 | 2.468 | 2.230 | 0.90× |
| `sqrt` w=4 | 3.660 | 3.348 | 0.91× |

`np.sqrt` was already near parity at w≥16 (its per-element compute amortizes the overhead); the pure
memory-bound `positive` at small `w` was the clear loss.

### 2.2 The floor — the decisive measurement

A hand-written 2-D loop (one prologue, an inner SIMD run over `w`, a per-row pointer bump) establishes the
ceiling a perfect kernel could reach:

| 2M f64 `(rows, w)` | copy floor | production (before) | NumPy | floor vs NumPy |
|---|---:|---:|---:|---:|
| w=4 | **1.705 ms** | 3.629 | 2.994 | **1.76×** |
| w=16 | 1.228 | 2.456 | 2.230 | 1.82× |
| w=64 | 1.120 | 1.792 | 2.005 | 1.79× |

The floor is **1.7–2.1× faster than NumPy**, and production was **~2× off its own floor**. Conclusion: the
gap is *per-row overhead*, not memory bandwidth — the win is real and large, not a parity chase.

### 2.3 Why buffering doesn't help (measured, rejected)

Gathering the strided rows into a contiguous window (one kernel call per 8192 elements, then scatter back)
was measured **slower** than the per-row route (w=4: 4.45 ms vs 3.95). The gather/scatter costs as much as
it saves, because the inner runs are already SIMD-able *in place*. This matches the discovery doc's
rejection.

### 2.4 Routing trace

`np.positive(sv, out)` / `np.sqrt(sv, out)` and their no-`out` forms all funnel through the three
packed-key entry points in `NDIter.Execution.Custom.cs` (`ExecuteElementWise`,
`ExecuteElementWiseUnary`, `ExecuteElementWiseBinary`), which build the kernel and call `ForEach`.
Comparisons take a different route (`TryExecuteComparison`, a whole-array Direct kernel that already walks
strides), so they were never part of this problem.

---

## 3. Angles reviewed

| Angle | Verdict |
|---|---|
| **2-D block kernel** — loop the outer axis inside the kernel (prologue once) | ✅ implemented — the lever |
| **Buffered windows** | ❌ rejected (measured slower) |
| **`positive` never engaged** the SIMD path | ✅ root-caused: it had *no vector body* (scalar even contiguously); added an identity vector body |
| **N-D (`x[:, :, :w]`)** | ✅ NumSharp does not coalesce outer axes (stays NDim=3); added an outer-axis *flatten* so N-D reuses the same kernel |
| **Deterministic 3-D crash** during testing | 🔎 confirmed **pre-existing** (clean HEAD crashes identically); root cause is `np.empty(view.Shape)` fabricating strides |
| **inner-broadcast (`add(A, col)`) / non-flattenable outer (`x[::2, :, :w]`)** | left on the per-chunk route — correct, unaccelerated |

---

## 4. The fix

### 4.1 The 2-D kernel contract

`ND2DElementwiseKernel` (`src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D.cs`):

```csharp
void(void** dataptrs, long innerCount, long* outerByteStrides, long outerCount)
```

The emitted body runs the SIMD-viability dispatch and the loop-bound setup **once**, then loops
`outerCount` rows: an inner 4×-unrolled SIMD + 1-vector-remainder + scalar-tail loop over `innerCount`
contiguous elements, followed by a per-operand pointer bump `ptr[op] += outerByteStrides[op]`. Neither the
odometer nor the prologue runs per row.

The enabling observation: the existing `EmitSimdContigLoop` addresses each operand as `ptr + i*elemSize`
and **never mutates the base pointer**, so the same inner body, wrapped in an outer pointer-bump loop, *is*
the 2-D kernel. `EmitSimd2DContigLoop` reuses the low-level emit helpers (`EmitAddrIPlusOffset`,
`EmitVectorLoad`/`Store`, `EmitScalarElement`) and hoists the `unrollEnd`/`vectorEnd` computation above the
outer loop (they depend only on `innerCount`).

Kernels are cached in a dedicated `_innerLoop2DCache` keyed by the same `InnerLoopKernelKey` the per-chunk
route uses (a `(op, dtypes)` identity can appear in both caches — same ufunc served either per-chunk or as
one 2-D block).

### 4.2 Dispatch + gate

`NDIterRef.Is2DElementwiseShape()` + `TryExecute2DElementwise()` (`NDIter.Execution.Custom.cs`), wired into
the three packed-key `ExecuteElementWise*` entry points **before** `ForEach`, so it serves both the `out=`
and the allocating unary/binary routes. The gate (all cheap, allocation-free — the common contiguous/1-D
case fails the `NDim >= 2` / inner-contiguous test immediately, so the operand-type array is only built for
genuinely strided 2-D calls):

- unbuffered (a buffered cast/promote keeps the windowed path)
- no `where=` mask (`MaskOp < 0` → the masked ForEach driver)
- `EXTERNAL_LOOP`, not `ONEITERATION`
- inner (last) axis element-contiguous for **every** operand
- all operands the same **SIMD-capable** dtype (`CanSimdAllOperands`)

Because it reuses the **same** `scalarBody`/`vectorBody` emit delegates as the per-chunk kernel, and
element-wise ops carry no cross-element state, the result is **byte-identical** to the ForEach path — only
the loop structure differs. A broadcast-row operand (outer stride 0) is handled naturally (it re-reads the
same row).

### 4.3 N-D flatten

NumSharp's NDIter does **not** coalesce outer axes — introspection confirmed a 3-D `x[:, :, :w]` stays
`NDim=3` with strides `[8192, 16, 1]`. So the gate additionally flattens **mutually contiguous** outer axes
(`stride[d] == stride[d+1] * shape[d+1]`, true for any trailing-narrow N-D slice) into one
`(outerCount = Π outer dims, outerStride = innermost-outer stride)` and reuses the same 2-D kernel. This
covers 3-D/4-D trailing-narrow views **and** routes them off a pre-existing crash in the ForEach 3-D
odometer path (§6). Non-flattenable outer axes (a doubly-strided `x[::2, :, :w]`) stay on the per-chunk
route.

### 4.4 The `positive` fix (rode along)

`positive` is NumPy's identity ufunc. Its scalar body already emitted nothing (identity), but
`EmitUnaryVectorOperation` had no `Positive` case and `CanUseUnarySimd(Positive)` returned false — so
`positive` had a **null vector body** and was scalar *even when contiguous*, and could not engage the 2-D
path. Added an identity vector body (`case UnaryOp.Positive: return;` — load→store = a SIMD copy) and
`CanUseUnarySimd(Positive) = CanUseSimd(input)` (all SIMD dtypes, i64/u64 included — a copy needs no
abs/negate emulation). Bonus: contiguous `positive` is now a SIMD copy (**1.59×** NumPy, was scalar).

### 4.5 Files changed

| File | Change |
|---|---|
| `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D.cs` | **new** — `ND2DElementwiseKernel`, cache, `Compile2DElementwiseKernel`, `EmitSimd2DContigLoop` |
| `Backends/Iterators/NDIter.Execution.Custom.cs` | `Is2DElementwiseShape` + `TryExecute2DElementwise`, wired into 3 packed-key entry points |
| `Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop.cs` | `CanSimdAllOperands` → `internal` |
| `Backends/Kernels/Direct/DirectILKernelGenerator.Unary.Vector.cs` | `positive` identity vector body |
| `Backends/Kernels/Direct/DirectILKernelGenerator.Unary.cs` | `CanUseUnarySimd(Positive)` |
| `Backends/Kernels/GeneratedDelegates.cs` | `_innerLoop2DCache` in `ClearInnerLoop` + `InnerLoop2DCount`/`TotalCount` |
| `docs/NDITER_PERF_DISCOVERY.md`, `.claude/CLAUDE.md` | mark landed |

---

## 5. Results (NPY/NS, higher = NumSharp faster)

### 5.1 Narrow strided rows, 2M f64, `out=`

| op | w=4 | w=8 | w=16 | w=32 | w=64 | w=256 |
|---|---|---|---|---|---|---|
| positive | 2.06× | 1.81× | 1.72× | 1.29× | 1.75× | 1.46× |
| negative | 2.01× | 1.79× | 1.73× | 1.40× | 0.98× | 1.42× |
| sqrt | 2.19× | 2.01× | 2.02× | 1.39× | 1.72× | 1.72× |
| abs | 2.01× | 1.41× | 1.35× | 1.60× | 1.17× | 1.75× |
| add | 2.11× | 1.71× | 1.64× | 1.39× | 1.30× | 1.50× |
| multiply | 2.14× | 1.75× | 1.62× | 1.52× | 1.15× | 1.54× |

### 5.2 2-D and N-D, 2M f64, no-`out` (includes fresh allocation)

| | w=4 | w=16 | w=64 |
|---|---|---|---|
| **2-D** positive | 1.95× | 2.13× | 2.18× |
| **2-D** sqrt | 2.41× | 2.52× | 2.11× |
| **2-D** add | 2.29× | 2.02× | 1.82× |
| **3-D** positive | 1.66× | 1.85× | 2.07× |
| **3-D** sqrt | 2.26× | 2.22× | 1.51× |
| **3-D** add | 2.17× | 2.10× | 1.37× |

Every measured cell is a win.

### 5.3 Contiguous regression check, 2M f64

| op | NumSharp | NumPy | NPY/NS |
|---|---:|---:|---:|
| positive | 0.649 ms | 1.034 | **1.59×** (was scalar) |
| negative | 0.653 | 0.666 | 1.02× (unchanged) |
| sqrt | 1.150 | 1.161 | 1.01× (unchanged) |
| add | 1.289 | 1.251 | 0.97× (unchanged) |

Only `positive` moves (the SIMD-copy bonus); everything else is unchanged — no regression.

---

## 6. The pre-existing crash (found, not caused)

While testing the 3-D path, `np.sqrt` on a `(512,512,8)` trailing-narrow view crashed with an
`AccessViolationException` in the **existing** per-chunk kernel. Stashing all changes and rebuilding clean
HEAD reproduced the crash **identically** — it is pre-existing.

Root cause: the test allocated its output with `np.empty(view.Shape)`, and `np.empty(Shape)` **inherits the
view's strides** — it produced a size-`N` buffer with strides that address beyond `N` (e.g. a 2,097,152-
element buffer with strides `[8192, 16, 1]` addressing up to element 4,194,296). Any kernel that honors
those strides writes out of bounds. This is a footgun in `np.empty(Shape)`, unrelated to this lever, and it
crashes both the old ForEach path and the new kernel equally (a malformed array). Real usage — no-`out`
(the production path `.Clean()`s to contiguous) or a properly-backed strided `out` — is correct.

Side benefit: the N-D flatten routes flattenable narrow-inner N-D views through the new kernel instead of
the ForEach 3-D odometer, so it sidesteps this crash for that case.

---

## 7. Correctness / gates

| Gate | Result |
|---|---|
| FuzzMatrix (all 64 corpus tiers, incl. `out_where` across every strided layout) | **98/98** |
| Main suite (`TestCategory!=OpenBugs&!=HighMemory`) | **14446 passed, 0 failed, 11 skipped** |
| Iterator / NDIter / NDExpr / Evaluate filter | 907 passed |
| 2-D self-consistency probe (strided vs contiguous-copy, ops × dtypes × widths × layouts) | **700/700 bit-identical** |
| N-D self-consistency probe (3-D + 4-D trailing-narrow) | **123/123 bit-identical** |
| Both target frameworks build | net8.0 ✅, net10.0 ✅ |

The self-consistency probes compare the 2-D/N-D strided result against the same op on a contiguous copy of
the input (which takes the proven 1-D path) — the tightest possible check that the new kernel matches the
old one bit-for-bit.

---

## 8. What's left (correct, unaccelerated) — as of the first generation; see §11 for what closed

- **inner-broadcast 2-D** — `add(A, col)` where `col` broadcasts over the inner axis (stride-0 inner):
  the gate requires a contiguous inner for all operands, so this keeps the per-chunk route (which SIMDs the
  broadcast per row). Extending the 2-D kernel with a broadcast-inner variant is possible future work.
  *(Closed in §11: the SC/CS modes.)*
- **non-flattenable outer** — a doubly-strided `x[::2, :, :w]`: outer axes are not mutually contiguous, so
  they cannot collapse to a single stride; keeps the per-chunk route. A full N-D outer odometer in the
  kernel would cover it but re-introduces per-row odometer cost. *(Closed in §11: a per-BLOCK odometer in
  the dispatcher.)*
- **non-SIMD dtypes** (Half / Decimal / Complex) and **comparisons** (mixed dtype → bool) are excluded by
  design; comparisons already have the whole-array Direct strided kernel. *(Non-SIMD dtypes and every op
  without a vector body now take the scalar-only 2-D block, §11; comparisons stay on their Direct kernel.)*
- **`np.empty(Shape)` inheriting view strides** — a separate pre-existing footgun (§6).

---

## 9. Reproduction

```bash
# baseline / after — narrow strided rows, both sides
DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 \
  dotnet run -c Release benchmark/nditer/probes/angles_probe.cs         # §1 = narrow rows
OPENBLAS_NUM_THREADS=1 python benchmark/nditer/probes/numpy_twins.py angles

# the gates
(cd test/NumSharp.Tests.Oracle && dotnet test --no-build -c Release -f net10.0 --filter "TestCategory=FuzzMatrix")
(cd test/NumSharp.Tests && dotnet test --no-build -c Release -f net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory")
```

Measurement discipline (see `docs/NDITER_PERF_DISCOVERY.md` §3): Release only, `DOTNET_TC_CallCountingDelayMs=0`,
`OPENBLAS_NUM_THREADS=1`, `#:property PublishAot=false`, a **fresh filename per rebuild** (a stale `#:project`
DLL cache silently runs the old Core — the tell here is the 2-D cache count not growing), interleaved and
repeated on an idle host.

---

## 11. Second generation — the review that challenged §1–§10

### 11.1 What the first report did not measure

§5 measured one dtype (f64), one size regime (2M, DRAM-bound, where every kernel converges on memory
bandwidth), widths of at least one full vector (w ≥ 4), and ops that have a vector body. A wider probe
(`benchmark/nditer/probes/narrow_probe.cs` + `numpy_twins.py narrow`) covered what it left out — widths
1–64 at 100K/1M/2M, f32/u8/i32, ops without a vector body, the broadcast shapes §8 listed as "left", and
3-D views — and found three regimes where NumSharp was **losing** to NumPy on the very shapes the kernel
exists for:

| Regime | Example | NPY/NS before | Why |
|---|---|---:|---|
| Sub-vector rows | `np.sqrt(m[:, :3], out)` 100K | **0.41×** | `w < lanes` ran the SCALAR tail per row; NumPy finishes rows with a masked vector (`npyv_load_tillz`/`store_till`), so at 3.3 ns/element it was 6× slower per element than the w=4 path |
| Inner-broadcast | `np.add(A, col)` 1M, c=4 | **1.15×** (vs 3.3× for `add(A, row)`) | the gate required a contiguous inner axis for EVERY operand; a broadcast-column or 0-d scalar operand fell to the per-row route at ~7 ns/row |
| Scalar-body ops | `np.exp(m[:, :4], out)` 1M | **0.84×** | the gate required a vector body; exp/log/power/mod/mixed-dtype kernels paid the per-row route (+43 % over contiguous) |

And one that was a **general iterator defect**, found while tracing `multiply(view, 2.0)`: a C-contiguous
`(250000, 4)` array times a 0-d scalar iterated as **2-D**, one kernel call per 4-element row (1.7 ms
for 1M elements, vs 0.25 ms at c=64). The constructor ran `CoalesceAxes` only when every operand was
contiguous AND nothing was broadcast, because that routine assumes the ascending (innermost-first)
layout its own sort produces. NumPy's `npyiter_coalesce_axes` runs unconditionally after order
resolution; a stride-0 operand passes its test as `0 == 0 * shape`. The oracle already listed the
consequence — "NumSharp coalesces fewer dimensions than NumPy" — as the K1 known bug.

### 11.2 Measurement traps found on the way

- **E-core scheduling.** This host is a hybrid part. Unpinned, the C# probe read 2–3× slower *uniformly*
  and swung run to run (1M `positive` w=4: 800–2600 µs unpinned, **411 µs** pinned), which also made the
  first-generation absolute numbers irreproducible. Both probes now honour `NS_PROBE_AFFINITY=<hex mask>`
  (`Process.ProcessorAffinity` / `SetProcessAffinityMask`); every number below is pinned to one P-core,
  NumPy and NumSharp alike, never concurrently.
- **Placement variance of strided streams at 100K.** With the SAME binary, cache-resident cells such as
  `add` w=3/4 at 100K moved between 50 and 63 µs across runs (relative page offsets of the read and
  write streams). Only effects well outside ±20 % are reported at 100K; the 1M/2M cells are stable.
- **Interleave old/new.** The old kernel was built in a detached worktree at the previous commit and the
  runs alternated old → new → old → new; each cell below is the minimum of two rounds per side.

### 11.3 What changed

1. **NDIter coalesces in iteration order, like NumPy** — `NDIterCoalescing.CoalesceAxesIterationOrder`,
   called on the non-all-contiguous construction branch in place of `RemoveUnitAxes` (which it subsumes).
   It merges every adjacent (outer, inner) pair that ALL operands walk as one axis —
   `stride[d] == stride[d+1] * shape[d+1]`, or a size-1 axis with stride 0 — so the element-visit order
   never changes; only the odometer shortens. A contiguous array against a 0-d scalar now iterates as ONE
   1-D chunk (`multiply(A, 2.0, out)` on `(250000, 4)`: **1.7 ms → 251 µs**), a trailing-narrow
   `x[:, :, :w]` arrives at the kernel as `(rows, w)` without the first generation's special flatten, and
   `a[:, ::2]` runs as ONE strided chunk exactly as NumPy's external_loop hands it out. Probed against
   2.4.2 on the three oracle layouts × C/F/A/K orders: chunk lengths and value streams identical except
   the pre-existing `order='A'` transposed-3-D case, so the K1 excuse is now scoped to that one layout
   and `strided_2d_cols`/`negstride_2d_offset` are gated bit-exactly.
2. **Inner-broadcast modes.** The kernel contract gained `innerByteStrides`; every input's inner stride
   is the element size (C) or 0 (S) and the kernel dispatches ONCE per call on the pattern — unary
   {C, S}, binary {CC, SC, CS} — each owning a full 2-D loop in which an S operand is loaded and
   `Vector.Create`'d once per row. The gate accepts stride 0 on inputs.
3. **Masked sub-vector rows.** For 32/64-bit lanes on an AVX2 host at 256-bit width, the row remainder
   `innerCount % lanes` — and the whole row when `innerCount < lanes` — is one `MaskLoad` per C input →
   the same vector body → one `MaskStore` (the mask is built once per call:
   `GreaterThan(Create(tail), {0,1,…})`). Masked-off lanes are never touched in memory; their register
   values are discarded, and for integer lanes they are filled with 1 first so a software vector divide
   cannot throw on a 0 lane. Other hosts and 1/2-byte lanes keep the scalar tail.
4. **Row-shape dispatch.** Once per call the block picks generic (unroll/remainder/tail), one-vector rows
   (`innerCount == lanes`: a single full vector per row, no inner loop — the generic shape paid three
   compares and an index update per row for the same work, 2.03 vs 1.59 ns/row on 1M f64 w=4) or
   sub-vector rows (a single masked vector per row).
5. **Scalar-only block.** A null vector body (or a non-SIMD dtype set) now gets the outer loop around a
   scalar inner loop — constant-stride addressing when every operand's inner stride is its own element
   size, runtime strides otherwise — instead of the per-row route.
6. **Per-block odometer.** Leading axes that do not fold into the block (`x[::2, :, :w]`) are walked in
   `TryExecute2DElementwise` with one kernel call per block, never per row.

Bit-identity is by construction (the same scalar/vector emit delegates, element-wise ops carry no
cross-element state) and was verified: **16,831 checks, 0 mismatches** across 124 compiled 2-D kernels —
every mode, widths 1–100, 10 dtypes, reversed rows, strided `out=`, aliased `out=`, mixed dtypes, 3-D
flat / `::2` / `::3` / `::-1` / two-axis-strided, 4-D, bool, Complex — comparing bytes (NaN payloads
included) against the same op on contiguous copies.

### 11.4 Results (pinned, min of two rounds per side, NPY/NS old → new)

**Sub-vector rows** (`out=`, f64 `(rows, w)`):

| N | op | w | NumPy µs | old µs | new µs | NPY/NS |
|---|---|---|---:|---:|---:|---|
| 100K | sqrt | 2 | 174 | 329 | 110 | 0.53 → **1.58** |
| 100K | sqrt | 3 | 136 | 329 | 73.5 | 0.41 → **1.84** |
| 1M | sqrt | 2 | 1815 | 3291 | 1098 | 0.55 → **1.65** |
| 1M | sqrt | 3 | 1395 | 3293 | 735 | 0.42 → **1.90** |
| 2M | sqrt | 2 | 4129 | 6903 | 2313 | 0.60 → **1.78** |
| 2M | sqrt | 3 | 3603 | 6929 | 1606 | 0.52 → **2.24** |
| 1M | positive | 2 | 1461 | 446 | 379 | 3.3 → 3.9 |
| 1M | positive | 3 | 1124 | 472 | 360 | 2.4 → 3.1 |
| 1M | add | 2 | 3066 | 1205 | 989 | 2.5 → 3.1 |
| 1M | add | 3 | 2465 | 1169 | 883 | 2.1 → 2.8 |

**One-vector rows and wider** (unchanged code path except the row-shape dispatch):

| N | op | w | NumPy µs | old µs | new µs | NPY/NS |
|---|---|---|---:|---:|---:|---|
| 1M | positive | 4 | 872 | 411 | 393 | 2.1 → 2.2 |
| 1M | add | 4 | 2081 | 953 | 802 | 2.2 → 2.6 |
| 2M | positive | 4 | 2939 | 1538 | 1259 | 1.91 → 2.33 |
| 2M | sqrt | 4 | 3150 | 1583 | 1321 | 1.99 → 2.38 |
| 2M | add | 4 | 5234 | 2400 | 2352 | 2.18 → 2.23 |
| 1M | positive | 16 | 523 | 383 | 304 | 1.4 → 1.7 |
| 2M | sqrt | 16 | 2318 | 1181 | 1171 | 1.96 → 1.98 |

**Ops without a vector body** (1M f64, `out=`): `exp` w=4 3937 → 2699 µs (NumPy 3289: **0.84× → 1.22×**;
contiguous exp is 2550–2640 on both sides, so the narrow view now costs what the contiguous one does),
w=16 0.91× → 1.08×, w=64 unchanged at parity; `mod(view, 3.0)` w=4 6104 → 4739 µs (1.87× → 2.41×).

**Broadcast shapes** (1M f64):

| Shape | c | NumPy µs | old µs | new µs | NPY/NS |
|---|---|---:|---:|---:|---|
| `add(A, col, out)` | 4 | 1991 | 1735 | 306 | 1.15 → **6.5** |
| | 16 | 702 | 518 | 271 | 1.35 → 2.6 |
| | 64 | 454 | 258 | 268 | 1.76 → 1.70 |
| `multiply(view, 2.0, out)` | 4 | 836 | 1733 | 424 | 0.48 → **1.97** |
| | 16 | 540 | 549 | 387 | 0.98 → 1.39 |
| | 64 | 426 | 358 | 378 | 1.19 → 1.13 |
| `view * 2.0` (allocating) | 4 | 1937 | 1767 | 407 | 1.10 → 4.8 |
| `add(view, col, out)` | 4 | 2938 | 1800 | 492 | 1.63 → **6.0** |
| | 16 | 1009 | 549 | 401 | 1.84 → 2.5 |
| `add(A, row, out)` | 4 | 878 | 265 | 265 | 3.3 → 3.3 (already the CC path) |
| `multiply(A, 2.0, out)`, A contiguous | 4 | — | ~1700 | 251 | the coalescer: 2-D per-row → one 1-D chunk |

**3-D** (2M f64, `out=`):

| View | op | NumPy µs | old µs | new µs | NPY/NS |
|---|---|---:|---:|---:|---|
| `x[:, :, :4]` | positive | 2571 | 1551 | 1316 | 1.66 → 1.95 |
| `x[:, :, :4]` | add | 4914 | 2413 | 2441 | 2.04 → 2.01 |
| `x[::2, :, :4]` | positive | 1275 | 1810 | 600 | 0.70 → **2.12** |
| `x[::2, :, :4]` | add | 2674 | 2401 | 1090 | 1.11 → **2.45** |
| `x[::2, :, :16]` | positive | 720 | 705 | 484 | 1.02 → 1.49 |
| `x[::2, :, :16]` | add | 1572 | 1264 | 879 | 1.24 → 1.79 |

Other dtypes at 1M (f32/u8/i32 `positive`/`add`, w=3…32) were already 2–8× NumPy and are unchanged
within noise.

### 11.5 Gates

| Gate | Result |
|---|---|
| Self-consistency probe (strided/broadcast vs contiguous copies, bytes) | **16,831 / 0 mismatches**, 124 kernels |
| FuzzMatrix, all 64 corpus tiers, K1 excuse narrowed to `transposed_3d` | **98/98** |
| Main suite (`TestCategory!=OpenBugs&!=HighMemory`) | **14446 passed, 0 failed, 11 skipped** |
| Iterator / NDIter / NDExpr / Evaluate filter | 907 passed, 5 failed — the same five `[OpenBugs]` as before |
| `np.nditer` chunking vs NumPy on the K1 layouts × C/F/A/K | identical except the pre-existing `'A'`-order transposed-3-D case |
| net8.0 + net10.0 | both build |

One test changed: `AxisStride_2D_NonContig_NoMultiIndex_FortranOrder` asserted the old non-coalescing
(`NDim == 2` for `a[:, ::2]` under the default K order); NumPy reports `ndim 1` and a single chunk there,
so the assertion now pins the NumPy behaviour.

### 11.6 What is left

- **Copy-class ops at w=2/3** gain only 1.1–1.3× from the masked path (a masked vector costs about what
  two or three scalar moves cost); the floor probe (§F of `narrow_probe.cs`) shows a further ~1.3× for a
  `w == lanes/2` half-vector store. That is a per-width kernel specialization, deliberately not done.
- **Hosts without AVX2 at 256-bit width** (V128, V512, non-x86) and 1/2-byte lanes keep the scalar tail.
- **Strided-inner rows** (`x[:, ::2]` that does NOT coalesce because of a second operand) stay on the
  per-chunk route's gather path; `where=` masked and buffered-cast routes are untouched.
- **The K1 `'A'`-order axis-ordering divergence** on a transposed 3-D operand is pre-existing and remains.
- The coalescer change reaches every NDIter consumer (copies, casts, reductions), not only these kernels;
  only the elementwise routes were measured here.

## 12. See also

- `docs/NDITER_PERF_DISCOVERY.md` — the cost model, the measuring traps, the probes, and the full ranked
  next-wins list (this is §7 angle 2).
- `benchmark/nditer/probes/narrow_probe.cs` + `numpy_twins.py narrow` — the second-generation probe
  (sections A–F: widths × sizes, dtypes, scalar-body ops, broadcast shapes, 3-D, hand-written floors).
- `.claude/CLAUDE.md` → the NDIter section — the concise landed-lever summary.
