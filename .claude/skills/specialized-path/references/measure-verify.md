# Measure honestly & verify parity

## The measurement harness (copy-paste)

**Prefer the checked-in probes** — `benchmark/nditer/probes/*_probe.cs` + `numpy_twins.py <twin>`
(`fixed`, `ab`, `angles`, `narrow`, `fancy_where`, `neighbours`) + `numpy_twins.py join before.tsv
after.tsv numpy.tsv`. They already encode every rule below (Debug assert, affinity pin, pilot-sized
best-of-rounds, keyed TSV output). Copy one as your template. The full protocol with the worktree A/B:
`docs/stale-docs/NDITER_PERF_CONTINUATION.md` §2.

Ad-hoc `dotnet_run` scripts. **Every one MUST be `-c Release`** (Debug taints hand loops ~2×), and
**every one needs a FRESH filename** when you rebuild `NumSharp.Core` (see traps).

```csharp
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property AllowUnsafeBlocks=true
#:property PublishAot=false        // MANDATORY: without it DynamicMethod is disabled → every IL
                                   // kernel returns null → 10–17× slower scalar fallback (see traps)
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
using System; using System.Diagnostics; using NumSharp;

// refuse a Debug Core (the #:project is built Debug unless you ran `dotnet run -c Release`)
var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core"); return; }
// pin to one P-core on a hybrid host (same mask on the NumPy side; never run both at once)
if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } aff)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(aff, 16);

static double BestUs(Action body)               // warm, pilot-size the rounds, best-of-rounds (min)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(5, (int)Math.Ceiling(250.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e3 / it); }
    GC.Collect(); GC.WaitForPendingFinalizers();      // between-case reset
    return best;
}
static void Row(string key, double us) => Console.WriteLine($"{key}\t{us:G17}");   // keyed TSV → join
// warm the kernel, VERIFY CORRECTNESS, then time. Timed bodies that are hand loops go in a
// `static class` method with [MethodImpl(AggressiveOptimization)] — script local functions and
// capturing lambdas run tier-0 (56–70 GB/s vs 186 for the same loop).
```

Run it with the full environment line — every variable matters (traps 2–5, 7):

```bash
NS_PROBE_AFFINITY=0x4 DOTNET_TC_CallCountingDelayMs=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 dotnet run -c Release <scratch>/probe_v3.cs > after.tsv
```

NumPy twin — **always thread-pinned, same core, never concurrently** (env MUST precede `import numpy`):

```bash
NS_PROBE_AFFINITY=0x4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE python <<'EOF'
import os, sys, numpy as np, time
def pin_affinity():                     # numpy_twins.py's pin — same mask as the C# probe
    aff = os.environ.get("NS_PROBE_AFFINITY")
    if aff and sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(int(aff, 16)))
pin_affinity()
def best_us(body):                      # the probes' best_ns/best_ms: warm, pilot, best-of-rounds
    t = time.perf_counter(); body()
    while time.perf_counter() - t < 0.06: body()
    t0 = time.perf_counter(); pc = 0
    while time.perf_counter() - t0 < 0.02: body(); pc += 1
    per = (time.perf_counter() - t0) / pc
    it = max(1, min(1_000_000, int(round(0.001 / max(per, 1e-9)))))
    rds = max(3, int(0.15 / max(it * per, 1e-9)) + 1)
    best = float("inf")
    for _ in range(rds):
        t0 = time.perf_counter()
        for _ in range(it): body()
        best = min(best, (time.perf_counter() - t0) * 1e6 / it)
    return best
EOF
```

## THE traps (each cost real time)

1. **Debug taint ~2×.** `dotnet run file.cs` compiles Core in Debug even with `#:property Optimize=true`
   (that only fixes the script). Use the CLI `-c Release`. IL-emitted kernels look normal in Debug while
   hand loops inflate — so a Debug run silently mis-ranks two candidates. The probes assert
   `IsJITOptimizerDisabled == false` and refuse to run.
2. **Stale runfile cache.** The cache keys on filename+content. After `git stash pop` + `dotnet build`,
   re-running the SAME `bench.cs` reused the OLD stashed DLL — NEW read byte-identical to OLD until I
   copied to a fresh filename. **New build ⇒ new filename** (`cp bench.cs bench_v2.cs`). The "2× off
   floor" reading of a kernel that had never engaged was this trap; an engagement counter catches it.
3. **Multi-threading skews BOTH sides (see the dedicated section below).** Unpinned NumPy BLAS (`A@x` at
   2048 hit 171 GB/s — impossible on one core) makes your kernel look ~5× worse; the NumSharp OpenBLAS
   backend threads too, and its result BITS change with the count, so >1 also reddens byte-parity gates.
   Pin every backend to 1 on both sides. Single-thread is the fair baseline (the official harness does it).
4. **`PublishAot=false` is mandatory in kernel-touching scripts.** A `dotnet run` file-based app without
   it disables `DynamicMethod` (`PlatformNotSupportedException: Dynamic code generation is not supported`),
   so EVERY IL kernel (`GetPutKernel`/`GetTakeKernel`/NDIter.Copy/casts/…) returns null and the code
   falls back to a **10–17× slower** scalar/NDIter path. This manufactured a fake "5 µs SetData
   bottleneck" that was 0.31 µs with the flag — every kernel-vs-scalar number from an AOT script is
   invalid. Diagnose: reflection-call the private `Generate*KernelIL` and it surfaces the exception; or
   check `DirectILKernelGenerator.GetPutKernel(8) == null`.
5. **`DOTNET_TC_CallCountingDelayMs=0`.** Tiered compilation promotes a method to tier-1 only after a
   100 ms window in which no NEW method was jitted at tier-0. A bench keeps jitting DynamicMethod
   kernels for most of a 200 ms row, so the FIRST rows of a process are timed at tier-0: `np.less(out=)`
   n=1 read 484 ns by default, 124 with the flag, 130 as the fifth row of the same process; `astype@1`
   755 → 346. NumPy has no JIT, so this only ever inflates the NumSharp side. `nditer_sheet.py` sets it;
   set it for any hand-run timing below ~10K elements (or time the row twice and keep the second).
6. **Pin to one P-core on a hybrid part, BOTH sides, never concurrently.** An unpinned benchmark thread
   on this 8P+16E host reads 2–3× slower UNIFORMLY and swings run to run (1M f64 `positive` on 4-wide
   rows: 800–2600 µs unpinned, 411 µs pinned — the first-generation 2-D-kernel numbers did not
   reproduce). Uniform slowness across every row of a probe is the tell; a "before" and an "after" taken
   in different scheduling states compare nothing. `NS_PROBE_AFFINITY=<hex mask>` (`0x4` here); the
   official `bench_common.py` does NOT pin yet. Cache-resident strided-stream cells (100K) additionally
   move ±20% with the SAME binary from stream placement — believe only effects well outside that band
   there, min of ≥ 2 rounds per side.
7. **Never A/B the live tree.** A `dotnet run` bench compiles the tree AS IT IS when the run starts;
   editing a source file while a chain of "post" runs is in flight hands later runs a half-edited tree
   (one run built with a compile error and reported partial rows). Both sides of an A/B belong in
   detached worktrees (below) and are INTERLEAVED old→new→old→new.
8. **Long benches contaminate.** Regime pressure accumulates across a 26-case run; a gevm group measured
   after 15 gemv cases read ~2× slow. Four dtype×op variants at 10M in ONE loop allocated 120 MB and
   GC-stormed a phantom 4223 µs argmin (1766 isolated). Keep benches SHORT and grouped, `GC.Collect()`
   between cases, or isolate one op-family per run. When numbers disagree between a short and a long
   run, trust the short.
9. **Cross-run drift.** The machine's memory regime swings hourly — allocator-heavy cells 1.7–4× (the
   SAME 12-case BDN filter: geomean 81 µs at 07:00, 47 µs at 09:30, three consecutive runs each), and
   every "dirty" committed snapshot predates the pool pacing fix `160ecbba` (full gen2 every ~24 calls).
   The ONLY reliable improvement number is a same-session before/after (below); cross-run/cross-machine
   ratios lie. Check the snapshot's raw `Statistics.Min` before blaming code: if min matches today, the
   mean was tail. (The `benchmark` skill + `benchmark/CLAUDE.md` document the same volatility.)
10. **Min-of-N on BOTH sides, same session, same load.** A single standalone NumPy reading can be a
    fluke: `left_shift(bool)` 100K read 0.2025 ms once (a bad-page-supply moment) and 0.0425 robust —
    which turned a real 0.61× LOSS into a fake 3.13× "win". A stale NumPy reference from a quieter run
    faked a 0.61× Half SLAB regression that was 2.3× in NumSharp's favour (6.5 ms phantom vs ~29 ms real).
    Cross-process ratios MUST be measured simultaneously under the same load; `merge-results.py`
    compares per-case MINS for the same reason (mean/mean 0.864 vs min/min 0.994 on the same run).
11. **Interleave and repeat before believing a placement effect.** A one-shot probe read the same V256 add
    at 0.460 ms with pooled buffers at their natural page offsets and 0.355 ms staggered by 64/128 bytes
    — a "1.30× 4K-aliasing win". Alternated five times, best-of-15, idle host: 0.388–0.411 natural vs
    0.383–0.454 staggered at 1M, 3.08–3.63 vs 2.89–3.31 at 4M — nothing outside the noise band. The
    first measurement ran cold and first. Any alignment / prefetch / placement claim needs the
    interleaved repeated form (`angles_probe.cs` section 5).
12. **NumPy's 1M allocating cells flatter NumSharp.** NumPy page-faults a fresh 8 MB result per call
    while our pool hands back a warm buffer. Compare KERNELS on `out=` cells or at 100K; and do compare
    both the `+dispose` and the dropped regime at 100K (the leaked-output asymmetry runs the other way).
13. **Probe-script traps.** `/` on an integer NDArray is NumPy TRUE division — `(ar / 64 % 2) == 0` is a
    SPARSE mask (one true per 128), not 64-runs; build masks with `np.floor_divide` (the first `where=`
    numbers compared different masks on the two sides — timings that differ in SHAPE across run lengths
    are the tell). `"label " + nd` binds to `NDArray.op_Addition` via the implicit string→char-array
    conversion (a broadcast error whose left shape equals the label's length — not a stale cache).
    `GetDouble(i)` on a float32 array reinterprets bytes — use `GetSingle`. `np.empty(view.Shape)`
    inherits the view's strides and writes out of bounds.
14. **The route you built may not be the route that ran.** A gate wired into the string-key
    `ExecuteElementWise` overload while the masked ufunc routes call the PACKED-key one; a Tier 3B path
    that intercepts `np.add(arr, 0-d)` before your MixedType gate; a `CanUseReductionSimd` that rejects
    the dtype your helper serves. Prove engagement with a counter or a kernel-cache count before
    reporting any number.

## Single-thread pinning — make it fair AND deterministic (BOTH sides)

NumSharp.Core is single-threaded by design ("parallelization is minimal"), so a fair comparison needs
the *other* side — and the optional NumSharp OpenBLAS backend — pinned to one thread too. Two reasons,
not one: **fairness** (unpinned NumPy BLAS multi-threads `@`/`dot`/`solve`/… and beats a single-core
kernel ~5×) AND **determinism** (a BLAS's result BITS depend on the thread count — CLAUDE.md: "1/2/4/24
threads give four different answers" — so >1 also reddens the byte-parity gates you run in VERIFY).

**NumPy / Python side — export in the SHELL, before `import numpy`.** Backends read their thread count
once at load, so `os.environ[...] =` set *after* the import (or a shell var set after the interpreter
starts) is too late for some of them. The full set (each targets a different backend a given numpy build
might use):

```bash
OPENBLAS_NUM_THREADS=1      # OpenBLAS — numpy 2.x's default (scipy-openblas) wheel
OMP_NUM_THREADS=1           # OpenMP — the threading layer inside OpenBLAS / MKL / BLIS
MKL_NUM_THREADS=1           # Intel MKL — conda / intel-numpy builds
BLIS_NUM_THREADS=1          # BLIS — some source/conda builds
NUMEXPR_NUM_THREADS=1       # numexpr — if pandas/numexpr is import­ed alongside
VECLIB_MAXIMUM_THREADS=1    # Apple Accelerate / vecLib — macOS
OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE   # stop the runtime re-growing the pool
```

Belt-and-suspenders (pins pools that are already loaded, and confirms what numpy actually linked):

```python
import numpy as np
np.show_config()                                   # which BLAS did this wheel pick?
from threadpoolctl import threadpool_limits, threadpool_info
threadpool_limits(1)                               # limit every detected BLAS/OpenMP pool at runtime
print(threadpool_info())                           # verify num_threads == 1 everywhere
```

**NumSharp / .NET side.** Core's managed kernels are single-threaded unless you opted in, and the
optional BLAS backend threads:
- **Leave `np.multithreading` OFF** (the default). If a harness enabled the opt-in threaded kernels,
  turn it off for the comparison: `np.multithreading(false)` (or simply don't call it).
- **If the OpenBLAS backend is active** (`NumSharp.Interop.OpenBLAS` referenced), pin it explicitly:
  `OpenBlasEngine.Enable(threads: 1)` — this calls the native `openblas_set_num_threads` directly, which
  is the reliable path.
- **The CRT trap (CLAUDE.md):** `Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS","1")` does
  **NOT** reach the native `getenv` — .NET keeps its own env table, and Windows `SetEnvironmentVariableW`
  doesn't reach it either. So for a bundled/native BLAS either (a) set the var in the SHELL before
  launching the `dotnet` process, or (b) use `OpenBlasEngine.Enable(threads: 1)`. Setting it in C# reads
  back correctly and changes nothing.

**Verify the pin took** rather than trusting it: a single-core `dgemm`/`dgemv` cannot exceed ~30–60 GB/s
on a desktop core, so if `2·M·N·K / seconds` implies >~80 GB/s the pin silently failed (an unpinned
2048² `A@x` read 171 GB/s). This is the same guardrail the memory note `nn-byte-parity-vs-numpy` uses:
byte-parity with NumPy needs threads=1 on BOTH sides.

## The reliable before/after: detached worktrees (or git-stash) in one session

**Worktrees — the safe form when other sessions may be editing the checkout:**

```bash
git worktree add --detach <scratch>/pre  <before-sha>
git worktree add --detach <scratch>/post <after-sha>
# rewrite the probe's `#:project` line to <scratch>/pre/src/NumSharp.Core/NumSharp.Core.csproj,
# give it a fresh filename (the runfile cache keys on the file), run; then the same for post.
ENV=... dotnet run -c Release <scratch>/probe_pre.cs  > before.tsv
ENV=... dotnet run -c Release <scratch>/probe_post.cs > after.tsv     # interleave: pre, post, pre, post
ENV=... python benchmark/nditer/probes/numpy_twins.py <twin> > numpy.tsv
python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv
git worktree remove --force <scratch>/pre
```

**git-stash — fine when the checkout is yours alone:**

```bash
git stash push -- <your changed files>          # revert to OLD
dotnet build src/NumSharp.Core -c Release -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
dotnet run -c Release ba.cs        > old.txt     # measure OLD
git stash pop                                    # restore NEW
dotnet build src/NumSharp.Core -c Release ...
dotnet run -c Release ba_fresh.cs  > new.txt     # FRESH FILENAME — measure NEW
```

This gave the rock-solid `gemv 188→21.8µs (8.6×)`, `gevm 359→21.2µs (16.9×)` numbers. It is the number
to put in a commit message — not a cross-run or cross-machine comparison.

## Report BOTH baselines — never conflate them

State the improvement against the **old managed path** AND against **single-thread NumPy**, separately:

> gemv `dot(m,v)` 1000²: **7.1× over the old managed path** (728→102µs); **~0.9× vs single-thread
> NumPy** (near parity — memory-bound, cblas dgemv at the floor).

"6-7× faster" with no baseline is the mistake — it was 6-7× over the *old managed path*, but *parity*
with NumPy. Both are true and both matter; say which.

Also state, in the report: the regimes you did NOT measure (widths, dtypes, sizes, ops without a
vector body, broadcast shapes — §7 of `discover.md`); that both sides were pinned, min-of-N, same
session; whether the cells were `out=` or allocating, disposed or dropped; and the probe file + twin
that reproduce every number (the discovery docs put them in `benchmark/nditer/probes/`). A report
whose ceiling is narrower than its probe is how the first 2-D kernel report shipped with three losses.

## Non-compute kinds don't get timed — they get semantic checks

A **view / early-exit** path has no throughput to compare; it's validated by asymptotics + semantics:

- **view vs copy:** confirm it's actually O(1) (shares storage, no data movement — `nd.Storage` /
  `shares_memory`), and that it carries the SAME metadata the copy path would: writeable, broadcast,
  owns-data flags, dtype, shape. A `flip` of a read-only broadcast MUST stay read-only; a materializing
  bug shows up as a wrongly-writeable result, not a wrong value. `benchmark/O1_EXCLUSIONS.md` keeps these
  out of every geomean (their benchmark tracks dispatch/alloc overhead, not throughput).
- **early exit / memory strategy:** confirm the short-circuit/threshold branch returns the identical
  result the full/other-strategy branch would, at the boundary (e.g. `size==0`, exactly at
  `WriteOnceMaxBytes`). A micro-benchmark only confirms it's cheaper — the oracle confirms it's correct.
  A lazy short-circuit (`bool >> s≠0` ~1.5 µs at any size) must be disposed per invocation in a BDN
  bench or it commit-OOMs under batching at 10M.

## Verify parity — the checklist (values AND metadata)

Bit-identical *values* and matching *metadata* (dtype, shape, view flags, NaN sign/payload/-0 handling,
error type and text, partial-write behaviour) to the general path on the subset.

1. **Know what the oracle pins.** Grep the corpus for the op:
   ```bash
   grep -h '"op": *"matmul"' test/NumSharp.Tests.Oracle/Fuzz/corpus/matmul.jsonl \
     | grep '"dtype":"float64"' | grep -o '"shape": *\[[0-9, ]*\]' | sort | uniq -c
   ```
   Matmul shapes are dims ≤5 → **small-K byte-exact**; large-K is not pinned (managed float GEMM was
   never byte-exact with cblas there). cov's `products.jsonl` cases are all K≤4. The corpus does NOT
   enumerate widths 1–40, a ≥ 2^17-distinct bailout, or your threshold's boundary — those are yours.
2. **Design so the pinned regime is byte-identical.** A 4-accumulator SIMD dot falls entirely into its
   *sequential scalar tail* below one unrolled step (K<16), which IS a left-to-right dot — matching the
   general path bit-for-bit for the small K the corpus pins. That's the whole reason gemv/Gram stay
   green while reordering large-K accumulation. A 64-bit-int → double sum CAN round, so the widening axis
   kernel keeps innermost/pinned cases on the exact scalar helper (`wideSeqOnly`) and streams only the
   leading-axis case.
3. **Reuse the general kernel for the pinned regime** where possible (gevm routes small `M==1` to the
   unchanged `SimpleStrided` → guaranteed identical bytes; only the unpinned large path changed; the 2-D
   block kernel wraps the per-chunk kernel's own emit bodies).
4. **Self-consistency probe for any kernel change** (before the differential gate): every strided result
   byte-equal to the same call on CONTIGUOUS COPIES and to an INDEPENDENT composition — `np.where` for a
   mask, `np.take` for a gather — across ops × dtypes × widths 1–100 × layouts (reversed, strided-out,
   aliased, mixed-dtype, 3-D/4-D, bool, Complex). The 2-D kernel's probe: 16,831 checks, 0 mismatches;
   the masked kernel's: 8,358. When the domain is small, sweep it EXHAUSTIVELY: all 65,536 f16 patterns
   × 4 roundings against live NumPy; all 2³² float32 inputs through both the SIMD and the scalar entry of
   each ported kernel (chunked checksums).
5. **Pin the gate itself, in unit tests:** a case on each side of the threshold
   (`Tril_AboveTheWriteOnceThreshold_StillZeroesTheDroppedTriangle` crosses 64 MiB; the bool shift is
   bit-exact across 262144), the bailout regime (`Unique_HighCardinality_RadixBailout`, 300K values,
   ~297K distinct), and the FALLBACK (an env knob forcing every size through the direct route, or a
   deliberately non-eligible layout). Pin route semantics beyond values — `Indexing.KernelRoute.Tests`
   holds the verbatim IndexError, the two value-broadcast ValueError texts, last-write-wins on
   duplicates, no partial write on a bad index (pre-scan before the first store, as NumPy's
   `mapiter_trivial_set`), and the view-offset conventions.
6. **NaN is a contract.** The oracle TOKENIZES float NaN, so a NaN sign or payload regression hides in a
   green FuzzMatrix — the `nan` tier (`56a55712`, host-gated) and per-family kernel tests pin it:
   first-NaN-wins with payload+sign VERBATIM for f16/complex min/max, the canonical `0x7fc00000` /
   `0x7ff8…` where NumPy canonicalizes (the ported f32 kernels, all-NaN nanmax), ±0 tie sign, and the
   NaN-GUARDED `npy_half_ge/le` semantics (a reference on raw compares mispredicts every `(finite, −NaN)`
   maximum). Never turn a portable cell host-dependent: MSVC FMA contraction, MSVC operand order and the
   CRT libm are PINNED classes (`test/NumSharp.Tests.Oracle/Fuzz/README.md` → "Host-dependent values").
7. **Run the gates:**
   ```bash
   dotnet test --filter "TestCategory=FuzzMatrix"          # byte-parity (matmul 769, products 326, out_where 3,727, …)
   dotnet test --filter "ClassName~<Op>BattleTests"        # the op's unit/battle tests
   dotnet test --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"   # the whole suite, CI filter
   dotnet build src/NumSharp.Core -c Release -f net8.0     # both targets; the NDW analyzer runs here
   ```
   plus a differential sweep vs NumPy 2.4.2 across shapes × dtypes × layouts (contiguous, F-contig,
   transposed, negative-stride, strided, and the FALLBACK path). Small-K should show `err=0.0`
   (byte-exact); larger machine-eps (tolerance). See the **`oracle`** skill for the corpus + OpRegistry.
8. **A reorder is only safe if unpinned — VERIFY, don't assume.** If FuzzMatrix goes red, the pinned
   regime diverged: match the general kernel's exact op order (FMA-or-not, k-order) for that regime.
9. **Commit only your paths** (`git commit --only <paths>` or a private `GIT_INDEX_FILE`) — concurrent
   sessions share this checkout, and a bare `git add` + commit sweeps their staged files.
