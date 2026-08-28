# Measure honestly & verify parity

## The measurement harness (copy-paste)

Ad-hoc `dotnet_run` scripts. **Every one MUST be `-c Release`** (Debug taints hand loops ~2×), and
**every one needs a FRESH filename** when you rebuild `NumSharp.Core` (see traps).

```csharp
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property AllowUnsafeBlocks=true
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
using System; using System.Diagnostics; using NumSharp;
static void B(string n, Action f, int warm=12, int it=100){
    for(int i=0;i<warm;i++) f();
    GC.Collect(); GC.WaitForPendingFinalizers();          // between-case reset
    double best=1e9;
    for(int r=0;r<13;r++){ var s=Stopwatch.StartNew(); for(int i=0;i<it;i++) f();
        best=Math.Min(best, s.Elapsed.TotalMilliseconds/it*1000); }
    Console.WriteLine($"NS|{n}|{best:F2}"); GC.Collect();
}
// warm the kernel, verify correctness, THEN time. best-of-13 (min), µs.
```

NumPy twin — **always thread-pinned**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python <<'EOF'
import numpy as np, time
def bench(f, warm=12, it=100):
    for _ in range(warm): f()
    return min((lambda: (lambda t: (sum(1 for _ in range(it) if f() is None or True), (time.perf_counter()-t)/it)[1])(time.perf_counter()))() for _ in range(13))*1e6
EOF
```

## THE traps (each cost real time this session)

1. **Debug taint ~2×.** `dotnet run file.cs` compiles Core in Debug even with `#:property Optimize=true`
   (that only fixes the script). Use the CLI `-c Release`. IL-emitted kernels look normal in Debug while
   hand loops inflate — so a Debug run silently mis-ranks two candidates.
2. **Stale runfile cache.** The cache keys on filename+content. After `git stash pop` + `dotnet build`,
   re-running the SAME `bench.cs` reused the OLD stashed DLL — NEW read byte-identical to OLD until I
   copied to a fresh filename. **New build ⇒ new filename** (`cp bench.cs bench_v2.cs`).
3. **NumPy multi-threads BLAS by default.** An unpinned `A@x` at 2048 hit 171 GB/s — impossible on one
   core. Pin `OPENBLAS/OMP/MKL_NUM_THREADS=1`; the official harness does, so single-thread is the fair
   baseline. Unpinned NumPy makes your kernel look ~5× worse than it is.
4. **Long benches contaminate.** Regime pressure accumulates across a 26-case run; a gevm group measured
   after 15 gemv cases read ~2× slow. Keep benches SHORT and grouped, `GC.Collect()` between cases, or
   isolate one op-family per run. When numbers disagree between a short and a long run, trust the short.
5. **Cross-run drift.** The machine's memory regime swings hourly. The ONLY reliable improvement number
   is **git-stash before/after in the same session** (below); cross-run/cross-machine ratios lie. (The
   `benchmark` skill + `benchmark/CLAUDE.md` document the same volatility for the official harness.)

## The reliable before/after: git-stash in one session

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

## Non-compute kinds don't get timed — they get semantic checks

A **view / early-exit** path has no throughput to compare; it's validated by asymptotics + semantics:

- **view vs copy:** confirm it's actually O(1) (shares storage, no data movement — `nd.Storage` /
  `shares_memory`), and that it carries the SAME metadata the copy path would: writeable, broadcast,
  owns-data flags, dtype, shape. A `flip` of a read-only broadcast MUST stay read-only; a materializing
  bug shows up as a wrongly-writeable result, not a wrong value.
- **early exit / memory strategy:** confirm the short-circuit/threshold branch returns the identical
  result the full/other-strategy branch would, at the boundary (e.g. `size==0`, exactly at
  `WriteOnceMaxBytes`). A micro-benchmark only confirms it's cheaper — the oracle confirms it's correct.

## Verify parity — the checklist (values AND metadata)

Bit-identical *values* and matching *metadata* (dtype, shape, view flags, NaN/-0 handling) to the
general path on the subset.

1. **Know what the oracle pins.** Grep the corpus for the op:
   ```bash
   grep -h '"op": *"matmul"' test/NumSharp.Tests.Oracle/Fuzz/corpus/matmul.jsonl \
     | grep '"dtype":"float64"' | grep -o '"shape": *\[[0-9, ]*\]' | sort | uniq -c
   ```
   Matmul shapes are dims ≤5 → **small-K byte-exact**; large-K is not pinned (managed float GEMM was
   never byte-exact with cblas there). cov's `products.jsonl` cases are all K≤4.
2. **Design so the pinned regime is byte-identical.** A 4-accumulator SIMD dot falls entirely into its
   *sequential scalar tail* below one unrolled step (K<16), which IS a left-to-right dot — matching the
   general path bit-for-bit for the small K the corpus pins. That's the whole reason gemv/Gram stay
   green while reordering large-K accumulation.
3. **Reuse the general kernel for the pinned regime** where possible (gevm routes small `M==1` to the
   unchanged `SimpleStrided` → guaranteed identical bytes; only the unpinned large path changed).
4. **Run the gates:**
   ```bash
   dotnet test --filter "TestCategory=FuzzMatrix"          # byte-parity (matmul 769, products 326, …)
   dotnet test --filter "ClassName~<Op>BattleTests"        # the op's unit/battle tests
   ```
   plus a differential sweep vs NumPy 2.4.2 across shapes × dtypes × layouts (contiguous, F-contig,
   transposed, negative-stride, strided, and the FALLBACK path). Small-K should show `err=0.0`
   (byte-exact); larger machine-eps (tolerance). See the **`oracle`** skill for the corpus + OpRegistry.
5. **A reorder is only safe if unpinned — VERIFY, don't assume.** If FuzzMatrix goes red, the pinned
   regime diverged: match the general kernel's exact op order (FMA-or-not, k-order) for that regime.
