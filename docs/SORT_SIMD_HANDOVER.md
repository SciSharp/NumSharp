# Handover — High-Perf Sort: IL-generated radix (#1) + AVX-512 vectorized quicksort (#2)

Status date: 2026-06-26 · Branch: `nditer` · Scope: `np.sort` / `np.argsort` / `ndarray.sort`
(`Backends/Default/Sorting/`)

This handover specifies **two** follow-up tracks for the along-axis sort, both requested
explicitly ("NDIter, IL, unrolling, simd/cpu acceleration, no loop per type/size, lowlevel
highperf"):

- **#1 — IL-generated per-dtype radix kernel.** Re-house the sort in the
  `DirectILKernelGenerator` pattern (emitted per-dtype kernels, cache-keyed, NDIter-driven,
  SIMD transform, unrolled histogram, branchless scatter). Removes the 15-way dtype switch and
  the per-size generic methods. **Perf is parity** — this buys the *architecture*, not speed
  (the scatter is the memory floor; proven below). Runs on all hardware.
- **#2 — AVX-512-gated vectorized quicksort.** A branchless `Avx512F.CompressStore` (true
  `vpcompressd`, 1× write) in-place quicksort for the **value-sort path only**, active only when
  `Avx512F.IsSupported`, radix fallback otherwise. The *real* SIMD win (~NumPy speed) **but only
  on AVX-512 silicon**. A correct, exhaustively-tested AVX2 prototype of the partition exists
  (embedded below); the AVX-512 version swaps the masked-store for `CompressStore`.

**Why two tracks, not one fast SIMD answer:** on AVX2 (the dev CPU — i9-13900K has AVX-512 fused
off) a vectorized sort is **slower** than the radix, and **NumPy itself falls back to scalar sort
on AVX2** (proven below). SIMD sort wins *only* on AVX-512. So #1 is the portable-architecture
play; #2 is the speed play that needs AVX-512 hardware to matter (and to benchmark).

---

## 0. Already shipped on this branch (context — DO NOT redo)

Three commits already improved the radix; this handover builds on top of them.

| Commit | What | Result |
|--------|------|--------|
| `5c91fb33` | **insertion-sort fast path** for short lines + **dtype-aware scratch** | (1e6,10) int32 sort **624→72ms (8.7×)**; killed the short-line radix catastrophe |
| `4319fb0e` | **single-pass multi-histogram** radix (read keys once, build all byte-histograms) | u32 ≈1.08×, u64 ≈1.17× raw; full np.sort int64 1.11→1.38× |
| (in `5c91fb33`) | width-specific insertion thresholds (4-byte n≤80, 8-byte n≤120) | from measured crossovers |

Tests green throughout: **55 sort tests** (`NpSortTests`, `SortBoundaryTests`,
`ArgsortInt64/NaN`, `argsort.Test`) + the **FuzzMatrix gate** (`sort.jsonl`, bit-exact vs
NumPy 2.4.2). Any change here MUST keep both green.

---

## 1. Current radix architecture (the baseline both tracks start from)

```
np.sort / np.argsort / ndarray.sort
  └─ AxisSort  (Backends/Default/Sorting/AxisSort.cs)
        • NDIter "all-but-axis" drive: one contiguous 1-D line per kernel call
          (DriveAllButAxis → NDIterRef.AdvancedNew(op_axes drops the sort axis) → ForEach)
        • 1-D / axis=None promoted to (1,N) via expand_dims so exactly one line-call happens
        • per line: monotonic transform (value→key) → RadixSort core → un-transform
        • dtype dispatch: GetSortKernel(tc) / GetArgSortKernel(tc)  ← the 15-way switch #1 removes
        • scratch alloc keyed by KeyWidth(tc): u32-path / u64-path / scalar(Half,Complex,Decimal)
  └─ RadixSort (Backends/Default/Sorting/RadixSort.cs)
        • 15 dtypes → 2 key widths via monotonic UNSIGNED transform (unsigned-asc == value order):
            KI8/KI16/KI32/KU8/KU16/KChar/KBool/KFloat32  → u32 keys  (SortU32, nbytes∈{1,2,4})
            KI64/KU64/KFloat64                            → u64 keys  (SortU64, 8 passes)
          Half/Complex/Decimal: scalar BCL introsort with NumPy comparators (no radix)
        • SortU32/SortU64/ArgSortU32/ArgSortU64 entry points:
            n ≤ InsertionThreshold{32=80,64=120} → stable binary insertion (short-line fast path)
            else: BuildHist32/64 (single read pass, all histograms) → per-pass prefix + scatter
        • argsort = same, co-scattering an int64 index column (stable → beats NumPy 2–9×)
```

**Key invariant (load-bearing):** the monotonic transform makes everything unsigned u32/u64
keys, so the sort core is **dtype-agnostic** and the per-byte histograms are **order-invariant**
(reorder by lower bytes never changes a higher byte's distribution → single-pass histogram is
correct). Floats: NaN partitioned out by the caller (NaN sorts last), the rest mapped monotonically.

**Where it stands (10M, NPY/NS, >1 = NumSharp faster):**

| | int16 | int32 | int64 | float32 | float64 | int8/uint8 |
|---|---|---|---|---|---|---|
| sort | 9.4× | 0.70× | 1.38× | 0.61× | 0.57× | 6–9× |
| argsort | 7.6× | 3.6× | 3.3× | 4.0× | 2.2× | ~5× |

The only sub-1× cells are 4/8-byte float **value-sort** — the target of #2.

---

## 2. The hard constraint: argsort MUST stay stable

`NpSortTests.Argsort_Stable_Ties` and the boundary tests assert **stable** argsort (ties in
ascending index order — NumPy `kind='stable'` semantics, which NumSharp pins as its contract).
Radix is stable; **quicksort is not**. Therefore:

- **#2 (vectorized quicksort) targets the VALUE-sort path ONLY.** `np.sort` doesn't distinguish
  equal elements, so instability is invisible there.
- **argsort always keeps the stable radix** (which already *wins* 2–9×, so there's no reason to
  touch it).

This split is clean and non-negotiable. Do not route argsort through quicksort.

---

## 3. Definitive findings — DO NOT re-run these dead ends

All measured on the i9-13900K (AVX2, **no AVX-512**), 10M elements, `dotnet run -c Release`,
best-of, warmup excluded. NumPy 2.4.2.

| approach | result | why it lost |
|---|---|---|
| **radix (current)** | **58ms raw / 65ms full** | branchless — the winner on AVX2 |
| NumPy `np.sort` int32 | 38ms (in-place) / 46ms (copy) | **scalar introsort** — see below |
| .NET `Array.Sort` / `Span.Sort` (int32) | **532ms** | generic introsort, terrible — validates the radix-over-BCL choice |
| hand-tuned scalar Hoare quicksort | **568ms** | **branch mispredict** (~50% on random data) |
| vectorized quicksort, **out-of-place** (correct 138/138) | 160ms | 2× write (two compaction streams) |
| vectorized quicksort, **in-place full-store** (correct 175/175) | 200ms | 2× write (`Avx.Store` both ends) |
| vectorized quicksort, **in-place masked compress-store** (correct 161/161, Intel's AVX2 method) | **230ms** | `Avx2.MaskStore` is **microcoded/slow** on this µarch |
| 11-bit radix digits (3 passes not 4) | 0.95× (slower) | 2KB histogram costs more cache than the saved pass |
| software write-combining scatter buffers | 0.56× (slower) | bookkeeping > cache-line benefit; modern store buffers already coalesce |
| SIMD transform (`Vector256.Xor`) | saves **2.4ms** only | (un)transform is BW-bound; 4.3ms scalar → 3.1ms SIMD |

**The anchor proof — NumPy does NOT use SIMD for sort on AVX2:**
```
NPY_DISABLE_CPU_FEATURES=''      → 38.10 ms
NPY_DISABLE_CPU_FEATURES='AVX2'  → 37.47 ms   ← identical → sort is scalar introsort here
```
x86-simd-sort *has* an AVX2 path (`avx2-32bit-qsort.hpp`, `avx2_emu_mask_compressstoreu32` =
permute-LUT + `_mm256_maskstore`), but on AVX2-only hardware NumPy's dispatch doesn't pick it
(it's only a win on AVX-512). **Conclusion: the radix scatter is a random-access write — the
memory-bandwidth floor — which no AVX2 vector op moves. SIMD sort wins ONLY on AVX-512
(`vpcompressd`, 1× write).** This is hardware reality, not a tuning gap.

**.NET intrinsic availability (verified):**
- `Avx512F.CompressStore` / `Avx512F.Compress` / `.VL` variants: **EXIST** (this is #2's enabler).
- `Avx512F.IsSupported` on this dev CPU: **False** (can verify correctness, cannot benchmark #2 here).
- `Avx2.PermuteVar8x32`, `Avx2.MaskStore`, `Avx2.CompareGreaterThan(i32/i64)`: exist (`MaskStore` slow).

---

## 4. Benchmark + correctness harness (shared by both tracks)

**Always Release** (`benchmark/CLAUDE.md`: Debug taints hand-written kernels ~2×).
Concurrent agents on `nditer` intermittently break the whole-assembly build — if so, work in an
isolated worktree: `git worktree add --detach .claude/worktrees/sortwork HEAD`, copy changed
files in, `MUTEX_SCOPE=<wt> mutex-capture build -- dotnet ...`.

**Standalone SIMD prototype** (no project ref needed — sort raw `uint[]`/`ulong[]`, compare to
`Array.Sort`): see §6 embedded code. Exhaustive correctness pattern that MUST pass before trusting
any partition (sizes bracketing the reserve + adversarial inputs):
```
foreach n in {0,1,2,7,8,16,31,32,40,41,48,63,64,80,127,128,200,257,512,1000,4099,65537,1000000}
  foreach trial in {random, random%4 (ties), sorted, reverse, narrow&0xFF, 0/MAX alternating, all-5}
    sort; assert == Array.Sort(clone)
```

**Full np.* benchmark** (NumSharp vs NumPy, all dtypes × sizes × layouts): the scripts used for
the tables above live in the session scratch; re-derive from this template:
```csharp
#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// best-of timing of np.sort(a,axis)/np.argsort(a,axis) across dtypes {i16,i32,i64,f32,f64},
// sizes {1k,100k,1M,10M}, layouts {1D, (1000,1000) ax=-1/0, (1e6,10), (10,1e6), strided ::2}.
```
NumPy side: `w=a.copy(); t=time.perf_counter(); w.sort(); …` (in-place avoids the 40MB copy).

**Gates (must stay green):**
```bash
dotnet test test/NumSharp.UnitTest/NumSharp.UnitTest.csproj -c Release --framework net8.0 \
  --filter "FullyQualifiedName~NpSortTests|FullyQualifiedName~SortBoundaryTests|FullyQualifiedName~ArgsortInt64|FullyQualifiedName~ArgsortNaN|FullyQualifiedName~argsort.Test|TestCategory=FuzzMatrix"
```

---

## 5. OPTION #1 — IL-generated per-dtype radix kernel (architecture; parity perf)

**Goal:** replace the hand-written 15-way dtype switch (`GetSortKernel`/`GetArgSortKernel`) and the
generic `SortLineXX<T,K>` line kernels with `DirectILKernelGenerator`-emitted per-dtype kernels,
matching the house pattern. Removes "loop per type/size" from the dispatch; transform becomes
SIMD; histogram unrolled; scatter stays branchless. **NDIter still drives the all-but-axis loop.**

**Honest expectation:** **parity perf.** The house IL kernels self-describe as "~10–15% over C#"
for *compute-bound elementwise* ops; a random-scatter sort gets ~0% from IL (proven §3). The win
is architectural uniformity + removing the switch, not speed. State this in the commit so nobody
expects a number.

### Design

New partial: `Backends/Kernels/Direct/DirectILKernelGenerator.Sort.cs`.

- **Cache key:** `struct SortKernelKey { NPTypeCode Dtype; bool IsArgsort; }` →
  `ConcurrentDictionary<SortKernelKey, NDInnerLoopFunc>` (same `GetOrAdd` pattern as the other
  partials; see `_mixedTypeCache` etc.). Register `ClearSort()` in `ClearAll()` (MixedType.cs).
- **Emitted delegate shape:** reuse `NDInnerLoopFunc(void** dataptrs, long* strides, long count,
  void* auxdata)` so `AxisSort.DriveAllButAxis` is unchanged — `auxdata` stays the `LineCtx*`.
  The emitted body reads `LineCtx` fields (n, inStride, outStride, k32/t32/k64/t64/idx/it/count)
  via emitted field-offset loads (or pass a flattened arg struct).
- **Body the emitter generates, per dtype:**
  1. **Transform** value→key. Contiguous line (inStride == elsize) → emit a `Vector256` loop
     (load `Vector256<T>`, apply the monotonic transform, store key) + scalar tail; strided →
     scalar. The transform IL is the only per-dtype-family branch in the *emitter* (cold path):
       - unsigned int (byte/ushort/uint/char/bool): identity → `cpblk` / `Vector256` copy.
       - signed int (sbyte/i16/i32/i64): `Vector256.Xor` with the sign-bit vector.
       - float32/float64: `b ^ ((b>>(bits-1)) | signbit)` in vector form (see `FKey32/FKey64` in
         AxisSort.cs lines 261-264 for the scalar form to vectorize).
     Widening cases (i8/i16 → u32 key) need `Vector256.Widen`; or keep those scalar (they already
     win 6–9×, so SIMD-transforming them is wasted risk).
  2. **Sort** the keys: emit a `call` to the existing `RadixSort.SortU32/SortU64` (branchless,
     proven, keep as C#) — OR inline the single-pass histogram + scatter in IL with unrolled
     counters. Recommend **call** first (lower risk; IL-inlining the scatter buys nothing). The
     u32/u64 choice is baked into the emitted kernel (no runtime branch) — this is how "no per-size
     loop" is satisfied: the *size* is a compile-time constant of the emitted method.
  3. **Un-transform** key→value back into the line (SIMD, mirror of step 1). For argsort, write the
     int64 indices instead (no un-transform).
- **Insertion fast path** (n ≤ threshold): the emitted kernel still calls
  `RadixSort.InsertionU32/64` (already there). Keep the threshold logic.
- **AxisSort change:** `GetSortKernel(tc)` / `GetArgSortKernel(tc)` → `DirectILKernelGenerator
  .GetSortLineKernel(new SortKernelKey(tc, isArgsort))`. Delete the two switch methods + the
  `SortLineXX`/`ArgLineXX` generic methods + the `KBool…KU64` adapter structs (their transform
  logic moves into the emitter). Half/Complex/Decimal: the emitter returns the existing scalar
  `SortLineScalar`/`ArgLineScalar` delegate (no IL — graceful, like the other `TryGet*`→null fallbacks).

### Reference emitter helpers (already in DirectILKernelGenerator.cs)
`GetVectorContainerType()`, `GetVectorType()`, `EmitVectorLoad/Store/Create/Operation()`,
`EmitLoadIndirect/EmitStoreIndirect()`, `GetTypeSize/GetClrType/CanUseSimd/IsUnsigned()`,
`VectorBits/VectorBytes`. Read `DirectILKernelGenerator.Unary.Vector.cs` for a worked Vector256
emit loop; `.Cast.*.cs` for widening (`Vector256.Widen`) emit patterns (e.g. i16→u32).

### Definition of Done (#1)
- [ ] `GetSortKernel`/`GetArgSortKernel` switches + `SortLineXX`/`ArgLineXX` generics deleted; sort
      dispatch is a cache lookup over emitted kernels.
- [ ] Transform/un-transform SIMD on contiguous lines (Vector256), scalar tail + strided fallback.
- [ ] Half/Complex/Decimal still correct via the scalar fallback path.
- [ ] 55 sort tests + FuzzMatrix `sort.jsonl` green (bit-exact, stable argsort).
- [ ] Bench: confirm parity (no regression) on 1D + (1000,1000) + (1e6,10) + strided, all dtypes.
- [ ] Commit message states "architecture/parity, not a speedup" with the §3 reason.

### Risks
- Emitting widening transforms (i8/i16→u32) is fiddly — keep them scalar if it saves a day.
- The `LineCtx*` field access from emitted IL needs correct field offsets — prefer a stable
  explicit-layout struct or pass pointers as separate args.
- First-call IL-emit latency: negligible (cached), but the tiny-array bench warms it first.

---

## 6. OPTION #2 — AVX-512-gated vectorized quicksort (real SIMD win; value-sort only)

**Goal:** on `Avx512F.IsSupported` hardware, sort the value-sort path with a branchless
vectorized quicksort whose partition uses **`Avx512F.CompressStore`** (true `vpcompressd`/`q`,
1× write — the thing AVX2 can't do at speed). Expected ~NumPy AVX-512 speed (well below radix).
On non-AVX-512 hardware, **fall back to radix** (the current path). argsort untouched (stable radix).

**Critical:** only worth shipping if you have/target AVX-512 silicon. On AVX2 it's a 4× regression
(§3). It can be **correctness-verified anywhere** (the algorithm is identical; only the store op
differs) but **benchmarked only on AVX-512**.

### Where it plugs in
`AxisSort.SortInPlace` value-sort path, after the transform produces keys, *instead of*
`RadixSort.SortU32/U64`, when `Avx512F.IsSupported`. It sorts the **u32/u64 key buffer** in place
(the keys are already the monotonic unsigned image; sorting them ascending == value order; then
the existing un-transform writes values back). So it's a drop-in `uint*`/`ulong*` key sorter:
`VQSort.Sort(uint* keys, int n)` replacing `RadixSort.SortU32` for n > threshold.

### The partition algorithm (PROVEN correct on AVX2; AVX-512 swaps the store)

Two facts that took the whole investigation to nail — bake them in:
1. **2-vector reserve per end (32-slot slack), read-the-smaller-slack side.** With 1 vector/end
   (16 slack) the slack drifts below the 8-lane store width and corrupts unread data (the bug in
   the first attempt). 32 slack + "read whichever side has less slack" keeps both ≥ 8 — proven.
2. **Compress-store, not full-store.** Store ONLY the valid lanes. AVX2: `Avx2.MaskStore` (works,
   but microcoded → slow, hence radix wins on AVX2). **AVX-512: `Avx512F.CompressStore(ptr, mask,
   vec)`** — hardware compress, 1× write, fast. This is the only line that changes between the two.

**Embedded prototype — AVX2 masked-store, in-place, exhaustively tested (161/161 correct, 230ms
on AVX2).** This is the foundation; for #2, replace the two `Place` stores with `CompressStore`
(notes after the code).

```csharp
// Standalone: sorts uint32 keys in place. dotnet run -c Release. AVX2 version (slow); see notes
// for the AVX-512 CompressStore swap. Exhaustive correctness harness in §4.
using System; using System.Runtime.Intrinsics; using System.Runtime.Intrinsics.X86;
static unsafe class VQ {
    public static int INS = 32; const int N = 8;
    static readonly int* Perm;   // [256][8] compact-LUT: gather set-bit lanes to the front
    static readonly int* SMask;  // [9][8] store-mask: first k lanes' high bit set (for MaskStore)
    static VQ() {
        Perm = (int*)System.Runtime.InteropServices.NativeMemory.Alloc(256u*8u*4u);
        for (int m=0;m<256;m++){ int w=0;
            for(int i=0;i<8;i++) if((m&(1<<i))!=0) Perm[m*8+(w++)]=i;   // selected lanes first
            for(int i=0;i<8;i++) if((m&(1<<i))==0) Perm[m*8+(w++)]=i; }
        SMask = (int*)System.Runtime.InteropServices.NativeMemory.Alloc(9u*8u*4u);
        for(int k=0;k<=8;k++) for(int i=0;i<8;i++) SMask[k*8+i] = (i<k)? unchecked((int)0x80000000):0;
    }
    static readonly Vector256<int> Sign = Vector256.Create(unchecked((int)0x80000000));
    static void Ins(uint* a,int lo,int hi){ for(int i=lo+1;i<hi;i++){uint v=a[i];int j=i-1;
        while(j>=lo&&a[j]>v){a[j+1]=a[j];j--;} a[j+1]=v;} }

    // partition vector v: <=pivot to the left write cursor, >pivot to the right — each a
    // masked compress-store of exactly its valid lanes (1× write, no overshoot).
    static void Place(uint* a, Vector256<uint> v, Vector256<int> P, ref int wL, ref int wR){
        // unsigned compare via sign-flip: v<=pivot  ==  !(v_signed > P_signed)
        var vs = v.AsInt32() ^ Sign;
        int mle = Avx.MoveMask((Avx2.CompareGreaterThan(P, vs) | Avx2.CompareEqual(P, vs)).AsSingle());
        int nLe = System.Numerics.BitOperations.PopCount((uint)mle), nGt = N-nLe, mgt = (~mle)&0xFF;
        var cLe = Avx2.PermuteVar8x32(v.AsInt32(), Vector256.Load(Perm+mle*8));
        Avx2.MaskStore((int*)(a+wL), Vector256.Load(SMask+nLe*8), cLe); wL += nLe;     // <-- AVX-512: CompressStore
        var cGt = Avx2.PermuteVar8x32(v.AsInt32(), Vector256.Load(Perm+mgt*8));
        Avx2.MaskStore((int*)(a+wR-nGt), Vector256.Load(SMask+nGt*8), cGt); wR -= nGt; // <-- AVX-512: CompressStore
    }
    static int Partition(uint* a,int lo,int hi,uint pivot){
        var P = Vector256.Create((int)(pivot ^ 0x80000000u));
        var vL0=Avx.LoadVector256(a+lo);     var vL1=Avx.LoadVector256(a+lo+N);    // 2-vec reserve / end
        var vR0=Avx.LoadVector256(a+hi-2*N); var vR1=Avx.LoadVector256(a+hi-N);
        int wL=lo, wR=hi, readL=lo+2*N, readR=hi-2*N;
        while(readR-readL >= N){ Vector256<uint> v;
            if(readL-wL <= wR-readR){ v=Avx.LoadVector256(a+readL); readL+=N; }     // read smaller-slack side
            else { readR-=N; v=Avx.LoadVector256(a+readR); }
            Place(a,v,P,ref wL,ref wR); }
        // endgame: 4 held vectors + <8 middle leftovers → scratch → scalar partition into the gap
        uint* s=stackalloc uint[5*N]; Avx.Store(s,vL0);Avx.Store(s+N,vL1);Avx.Store(s+2*N,vR0);Avx.Store(s+3*N,vR1);
        int cnt=4*N; for(int i=readL;i<readR;i++) s[cnt++]=a[i];
        for(int i=0;i<cnt;i++){ uint x=s[i]; if(x<=pivot) a[wL++]=x; else a[--wR]=x; }
        return wL;
    }
    static uint Med3(uint* a,int lo,int hi){ int m=lo+((hi-lo)>>1); uint x=a[lo],y=a[m],z=a[hi-1];
        return x<y ? (y<z?y:(x<z?z:x)) : (x<z?x:(y<z?z:y)); }
    public static void Sort(uint* a,int lo,int hi){
        while(hi-lo > INS){
            if(hi-lo < 5*N){ Ins(a,lo,hi); return; }              // need 2-vec/end + 1 middle
            uint pivot = Med3(a,lo,hi);
            int sp = Partition(a,lo,hi,pivot);
            if(sp==lo || sp==hi){ Ins(a,lo,hi); return; }         // degenerate pivot guard (see TODO)
            if(sp-lo < hi-sp){ Sort(a,lo,sp); lo=sp; } else { Sort(a,sp,hi); hi=sp; }
        }
        Ins(a,lo,hi);
    }
}
```

### AVX-512 adaptation (the actual #2 deliverable)
- **Lanes:** `Vector512<uint>` = 16 lanes (u32) / `Vector512<ulong>` = 8 lanes (u64). Reserve
  2 vectors/end = 64-slot (u32) / 32-slot slack. `Perm`/`SMask` LUTs grow to 16/8 lanes — or
  drop them entirely: AVX-512 compress doesn't need the permute LUT.
- **Compare → mask:** `Avx512F.CompareLessThanOrEqual(v, P)` returns a `Vector512<uint>` mask;
  convert to a `byte`/`ushort` opmask. (AVX-512 compares yield k-mask registers natively in HW;
  in .NET they surface as vector masks — use `Avx512F.MoveMask`-equivalent / the mask-store APIs.)
- **The store:** `Avx512F.CompressStore(uint* ptr, mask, vec)` writes only the masked lanes,
  contiguously, in one op — replaces both `PermuteVar8x32`+`MaskStore` lines in `Place`. Left:
  `CompressStore(a+wL, leMask, v); wL += popcount(leMask)`. Right: compress the >pivot lanes
  similarly to `a+wR-nGt`. **1× write, no LUT, no microcode penalty.**
- **u64 twin:** identical with `Vector512<ulong>`, `Avx512F.CompressStore(ulong*…)`, 8 lanes.

### TODO before production (#2)
- [ ] **Introsort depth guard.** Median-of-3 + the `sp==lo||sp==hi` insertion guard prevents the
      worst O(n²) on the test patterns, but a real impl needs a depth counter → fall back to
      `RadixSort` (guaranteed O(n)) for that sub-range when recursion exceeds ~2·log2(n). Radix as
      the introsort fallback is elegant (already present, branchless).
- [ ] **u64 path** (`Vector512<ulong>`), and wire both into `AxisSort` behind `Avx512F.IsSupported`.
- [ ] **Pivot quality.** Median-of-3 is fine for random; consider median-of-medians / ninther for
      adversarial. NumPy uses a network-based pivot (`xss-pivot-selection.hpp`) — optional.
- [ ] **SIMD bitonic base case** (optional, perf): replacing the scalar `Ins` base with an
      AVX-512 sorting-network base (sort ≤64–128 in-register) is where x86-simd-sort gets a chunk
      of its speed — but measure first; the partition is the bulk. Reference:
      `src/numpy/numpy/_core/src/npysort/x86-simd-sort/src/xss-network-qsort.hpp` and
      `avx512-32bit-qsort.hpp`.
- [ ] **Gating + validation.** Verify correctness via the §4 exhaustive harness AND the FuzzMatrix
      `sort.jsonl` ON AN AVX-512 BOX. The AVX2 fallback must be byte-identical (it's the same
      radix). Add a `[TestCategory]` or runtime-skip so CI on AVX2 runners exercises the fallback.

### Definition of Done (#2)
- [ ] `Avx512F.CompressStore` u32 + u64 in-place quicksort, value-sort path only.
- [ ] Gated `if (Avx512F.IsSupported)` in `AxisSort` value-sort; radix fallback otherwise; argsort
      untouched (stable radix).
- [ ] Exhaustive correctness (§4 harness) + FuzzMatrix green **on AVX-512 hardware**; AVX2 fallback
      green on this box.
- [ ] Benchmarked on AVX-512: target ≥ radix, ideally ~NumPy. If it doesn't beat radix even on
      AVX-512, **don't ship it** — keep radix (document the number).

### Reference (NumPy's own AVX2/AVX-512 sort, vendored)
`src/numpy/numpy/_core/src/npysort/x86-simd-sort/src/` — `avx2-emu-funcs.hpp`
(`avx2_emu_mask_compressstoreu32` = the permute-LUT+maskstore this prototype mirrors),
`avx512-32bit-qsort.hpp` / `avx512-64bit-qsort.hpp` (the fast AVX-512 path #2 mirrors),
`xss-common-qsort.h` (driver + introsort guard), `xss-network-qsort.hpp` (bitonic base).

---

## 7. Recommendation

- If the deployment target is **AVX2 / mixed consumer hardware**: do **#1** (architecture, parity)
  and **stop** — the radix is already the high-perf answer there; #2 regresses on AVX2.
- If the target includes **AVX-512 servers** (Xeon Scalable / EPYC / Sapphire Rapids+): do **#2**
  gated, keep radix as the AVX2 fallback. Optionally #1 underneath for the fallback's architecture.
- Both keep **argsort on the stable radix** (it wins 2–9× — leave it).
