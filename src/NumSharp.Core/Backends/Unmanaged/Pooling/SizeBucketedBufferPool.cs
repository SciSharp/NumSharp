using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace NumSharp.Backends.Unmanaged.Pooling
{
    /// <summary>
    ///     Thread-safe pool of recently-freed unmanaged buffers, bucketed by
    ///     exact byte size. Acts as a tcache-like front for
    ///     <see cref="NativeMemory.Alloc(nuint)"/> /
    ///     <see cref="NativeMemory.Free(void*)"/>: a successful Take is just
    ///     a pop from a per-size <see cref="ConcurrentStack{T}"/>; a failed
    ///     Take falls through to NativeMemory.Alloc.
    ///
    ///     WHY THIS EXISTS
    ///     ---------------
    ///     Profiling NumSharp's binary-op pipeline shows ~500 µs of every
    ///     1024×1024 float32 `a + b` is spent on first-touch overhead of the
    ///     fresh output buffer — page-faulting each cache line on first write
    ///     plus the kernel-mode cost of <see cref="NativeMemory.Alloc"/>
    ///     reaching out to the OS for a fresh chunk. NumPy hides the same
    ///     cost via glibc tcache reuse: a buffer freed by the previous op is
    ///     handed back warm to the next call. This pool replicates that
    ///     behaviour at the NumSharp layer.
    ///
    ///     GC PACING — the promptness gap vs NumPy (Wave 2.5)
    ///     --------------------------------------------------
    ///     NumPy frees a dropped temporary the INSTANT its refcount hits zero,
    ///     so the very next allocation reuses the same warm block. NumSharp's
    ///     dropped-but-undisposed <c>NDArray</c> returns its buffer only via
    ///     the block Disposer's FINALIZER, which runs only after a GC collects
    ///     the wrapper graph. In a tight allocating loop the managed garbage
    ///     per op is tiny (~a few hundred bytes), so gen-0 collections are
    ///     thousands of calls apart — the pool starves and every call pays a
    ///     cold <c>NativeMemory.Alloc</c> plus first-touch page faults
    ///     (measured: 1K float32 unary 16% hit-rate; 100K ~35 µs/op of soft
    ///     faults — 7× the kernel's own time).
    ///
    ///     The pool therefore PACES the GC: when a Take MISSES (or a bucket is
    ///     about to run dry) after at least <see cref="PaceThresholdBytes"/>
    ///     of buffers were handed out since the last paced collection, it
    ///     requests ONE forced, blocking, non-compacting <b>gen-0</b>
    ///     collection (~4-5 µs on a small heap). That collects the dropped
    ///     wrapper graphs (allocated since the last pace → still gen 0), the
    ///     finalizer thread drains, and the returns refill the buckets — the
    ///     .NET stand-in for CPython's refcount-prompt free, amortized to at
    ///     most one gen-0 pause per <see cref="PaceThresholdBytes"/> of
    ///     allocation. Deterministically-disposing workloads (Dispose /
    ///     NDScope) keep their buckets stocked and never trip the trigger.
    ///     Opt-out: <c>NUMSHARP_POOL_GC_PACING=0</c>; window:
    ///     <c>NUMSHARP_POOL_GC_PACING_MB</c> (default 16).
    ///
    ///     GC PRESSURE — coarse chunks, not per-call (Wave 2.5)
    ///     ----------------------------------------------------
    ///     Wave 2.4 registered <see cref="GC.AddMemoryPressure"/> on every
    ///     Take and removed it on every Return. Two measured pathologies:
    ///     the Add/Remove pair costs ~55 ns per op, and the churn drove the
    ///     GC into CONSTANT full gen-2 collections in allocating loops
    ///     (100K float32 unary: a full GC every ~24 calls — ~12 µs/op of GC
    ///     time; every collection observed was gen 2). Pressure now tracks
    ///     the pool's OUTSTANDING bytes (taken − returned, i.e. what user
    ///     code actually holds) in <see cref="PressureChunkBytes"/> chunks:
    ///     the hot path pays two interlocked adds, and the GC only hears
    ///     about net growth/shrink at 64 MiB granularity — enough for the
    ///     issue #501 protection (a process holding gigabytes of native
    ///     arrays still pushes the GC to collect) without per-op churn.
    ///     Reclamation promptness inside the window is the pacing trigger's
    ///     job, not pressure's.
    ///
    ///     SIZING POLICY
    ///     -------------
    ///     • The window is <see cref="MinPoolableBytes"/> (1 B) to
    ///       <see cref="MaxPoolableBytes"/> (64 MiB) — Wave 2.4 opened both
    ///       ends: the 1000-element float32 result (4000 B) missed the old
    ///       4 KiB floor by 96 bytes, and every 4M-element output (16–32 MiB)
    ///       missed the old 1 MiB cap, paying ~2× in demand-zero page faults
    ///       per call (in-place toggle-verified: P1 contig add 4M 3.37→1.74 ms).
    ///     • Above the cap: no pooling. Huge buffers are rare and the memory
    ///       cost of keeping them around dwarfs the alloc-cost savings.
    ///     • Per-bucket caps are DYNAMIC (Wave 2.5): finalizer-driven returns
    ///       arrive in one burst per collection — up to a whole pacing window
    ///       of buffers at once. The old flat cap of 8 discarded ~95% of each
    ///       burst (measured: 20K-call 1K loop → 3.3K pooled, 16K freed) and
    ///       re-starved the loop. A bucket now holds up to one pacing window
    ///       of bytes: <c>cap = PaceThresholdBytes / size</c>, clamped to
    ///       [<see cref="MaxBuffersPerBucket"/> .. <see cref="MaxBuffersPerDynamicBucket"/>]
    ///       below <see cref="LargeBucketThreshold"/> and to
    ///       [<see cref="MaxBuffersPerLargeBucket"/> .. window] at/above it —
    ///       so per-bucket resident stays ≈ the pacing window (16 MiB default)
    ///       for mid sizes and keeps the old floor of 2 for huge buffers.
    ///     • Bucket key is the EXACT byte count requested (no rounding).
    ///       Same-size repeated allocs are the dominant pattern in element-
    ///       wise ops; rounding to power-of-2 would waste memory and break
    ///       exact-fit reuse for typical workloads (e.g. 4 MiB float32 1K×1K).
    ///
    ///     CORRECTNESS
    ///     -----------
    ///     • Stored buffers are NOT zero-filled. Callers that need zeroed
    ///       memory must zero on Take (the same contract NativeMemory.Alloc
    ///       has).
    ///     • Buffer ownership transfers fully on Take: the pool no longer
    ///       references the pointer, so subsequent Return calls aren't
    ///       at risk of double-pop.
    ///     • Return is best-effort: when the bucket is full or the size
    ///       falls outside the pool's window the pointer is freed
    ///       immediately via <see cref="NativeMemory.Free"/>.
    /// </summary>
    public static unsafe class SizeBucketedBufferPool
    {
        /// <summary>
        ///     Minimum allocation size to pool (bytes). Wave 2.4 lowered this
        ///     from 4096 to 1: the small-N hot path (e.g. a 1000-element
        ///     float32 ufunc result = 4000 bytes) sat just under the old
        ///     threshold and paid a fresh NativeMemory.Alloc + GC memory
        ///     pressure pair on EVERY call. Tiny buckets cost almost nothing
        ///     resident and a pool hit skips the cold alloc entirely.
        /// </summary>
        public const long MinPoolableBytes = 1;

        /// <summary>
        ///     Maximum allocation size to pool (bytes). Wave 2.4 raised this
        ///     from 1 MiB to 64 MiB: the dominant benchmark/e2e shapes (4M
        ///     elements = 16 MiB float32 / 32 MiB float64 outputs) all missed
        ///     the old cap and paid ~0.3–0.4 ms of first-touch page faults per
        ///     call — the "allocator tax" residual on every measured e2e
        ///     strided row. NumPy gets the same reuse for free from glibc's
        ///     arena caching. Resident growth is bounded by the per-bucket cap.
        /// </summary>
        public const long MaxPoolableBytes = 64L * 1024 * 1024;

        /// <summary>
        ///     FLOOR of the dynamic per-bucket cap below <see cref="LargeBucketThreshold"/>
        ///     (the pre-Wave-2.5 flat cap). The effective cap is
        ///     <c>clamp(PaceThresholdBytes / size, MaxBuffersPerBucket, MaxBuffersPerDynamicBucket)</c>
        ///     — see the SIZING POLICY note on burst-sized buckets.
        /// </summary>
        public const int MaxBuffersPerBucket = 8;

        /// <summary>Ceiling of the dynamic per-bucket cap — bounds tiny-size buckets (entries, not bytes).</summary>
        public const int MaxBuffersPerDynamicBucket = 4096;

        /// <summary>Bucket sizes at/above this use <see cref="MaxBuffersPerLargeBucket"/> as their cap FLOOR.</summary>
        public const long LargeBucketThreshold = 1024L * 1024;

        /// <summary>Per-bucket cap floor for large (≥ 1 MiB) buckets — bounds peak resident memory.</summary>
        public const int MaxBuffersPerLargeBucket = 2;

        /// <summary>
        ///     Granularity of the pool's <see cref="GC.AddMemoryPressure"/> accounting. The pool tells the
        ///     GC about net OUTSTANDING native bytes (taken − returned) in whole chunks of this size, so
        ///     the per-op hot path never calls the pressure APIs (see the GC PRESSURE header note).
        /// </summary>
        public const long PressureChunkBytes = 64L * 1024 * 1024;

        // (Low-water pacing is per-bucket: each Bucket carries its LowWater —
        //  the slack above one pacing window — computed in CapFor.)

        /// <summary>Whether pool-initiated gen-0 pacing is enabled (env <c>NUMSHARP_POOL_GC_PACING</c>, default on).</summary>
        public static readonly bool GcPacingEnabled = EnvVars.PoolGcPacing;

        /// <summary>
        ///     The pacing window in bytes (env <c>NUMSHARP_POOL_GC_PACING_MB</c>, default 16 MiB, clamped
        ///     1..1024 MiB): at most one pool-initiated gen-0 collection per this many bytes handed out,
        ///     and the per-bucket warm-set byte budget.
        /// </summary>
        public static readonly long PaceThresholdBytes =
            Math.Clamp(EnvVars.PoolGcPacingMB, 1, 1024) * (1024L * 1024);

        /// <summary>
        ///     Opt-in diagnostic page-heap mode (env <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c>, Windows only).
        ///     When on, every <see cref="Take"/>/<see cref="TakeZeroed"/> hands back a buffer whose
        ///     last byte abuts an inaccessible guard page (<see cref="OsVirtualMemory.AllocGuarded"/>),
        ///     pooling is bypassed, and any one-past-the-end write faults INSTANTLY at the offending
        ///     access — used to localise an indexing out-of-bounds write to the exact case/site.
        ///     Default OFF (the field is read once at startup) so production paths are untouched.
        /// </summary>
        public static readonly bool GuardPagesEnabled =
            OsVirtualMemory.IsSupported && EnvVars.DebugGuardPages;

        // Guard-mode bookkeeping: usable pointer -> reserved region base (for FreeGuarded).
        private static readonly ConcurrentDictionary<IntPtr, IntPtr> _guardRegions = new();

        /// <summary>
        ///     One size-class: the buffer stack, its depth (ConcurrentStack lacks an O(1) thread-safe
        ///     Count), the size-derived cap and the low-water mark — one dictionary probe on the hot
        ///     path instead of the former two (separate bucket + depth maps).
        ///
        ///     The cap is one pacing WINDOW of buffers plus ~25% SLACK; <see cref="LowWater"/> is that
        ///     slack. A full bucket drains to exactly the slack right as the window's byte threshold
        ///     trips, so the hit-path pacing check (<c>depth &lt;= LowWater</c> AND threshold reached)
        ///     fires while <see cref="LowWater"/> warm buffers still cover the asynchronous
        ///     finalizer-drain latency — the steady state pays ~zero cold misses. A deterministically
        ///     disposing workload keeps the bucket near its cap, so the depth gate keeps pacing
        ///     entirely off there.
        /// </summary>
        private sealed class Bucket
        {
            public readonly ConcurrentStack<IntPtr> Stack = new();
            public readonly int Cap;
            public readonly int LowWater;
            public int Depth;

            public Bucket(long bytes)
            {
                long window = PaceThresholdBytes / Math.Max(1, bytes);
                int slack = (int)Math.Min(Math.Max(2, window / 4), MaxBuffersPerDynamicBucket / 4);
                Cap = CapFor(bytes, window, slack);
                LowWater = slack;
            }
        }

        // Bucket map keyed on exact byte size. ConcurrentStack gives lock-
        // free Push/TryPop, which is the entire fast-path here.
        private static readonly ConcurrentDictionary<long, Bucket> _buckets = new();

        // ---------------------------------------------------------------
        // Pacing + coarse-pressure state (see the class header notes).
        // ---------------------------------------------------------------
        private static long _takenBytesSincePace;   // bytes handed out since the last paced collection
        private static int _paceGate;               // 1 while a paced collection is in flight
        private static long _outstandingBytes;      // taken − returned (≈ native bytes user code holds)
        private static long _registeredPressure;    // what we've told GC.AddMemoryPressure, in whole chunks

        // Diagnostic counters — useful for telling whether the pool is
        // doing real work or just adding overhead. Not on a hot path so
        // Interlocked is fine.
        private static long _hits;
        private static long _misses;
        private static long _returns;
        private static long _returnsFreed;
        private static long _zeroedAllocs;
        private static long _pacedCollections;

        /// <summary>How many Take calls served from the pool.</summary>
        public static long Hits => Interlocked.Read(ref _hits);

        /// <summary>How many Take calls fell through to NativeMemory.Alloc.</summary>
        public static long Misses => Interlocked.Read(ref _misses);

        /// <summary>How many Return calls accepted the buffer into the pool.</summary>
        public static long Returns => Interlocked.Read(ref _returns);

        /// <summary>How many Return calls freed the buffer (bucket full / out-of-range).</summary>
        public static long ReturnsFreed => Interlocked.Read(ref _returnsFreed);

        /// <summary>How many TakeZeroed calls went straight to calloc (the np.zeros fast path).</summary>
        public static long ZeroedAllocs => Interlocked.Read(ref _zeroedAllocs);

        /// <summary>How many gen-0 collections the pool has requested (pacing; see the class header).</summary>
        public static long PacedCollections => Interlocked.Read(ref _pacedCollections);

        /// <summary>Reset all counters. Diagnostic only.</summary>
        public static void ResetCounters()
        {
            Interlocked.Exchange(ref _hits, 0);
            Interlocked.Exchange(ref _misses, 0);
            Interlocked.Exchange(ref _returns, 0);
            Interlocked.Exchange(ref _returnsFreed, 0);
            Interlocked.Exchange(ref _zeroedAllocs, 0);
            Interlocked.Exchange(ref _pacedCollections, 0);
        }

        /// <summary>
        ///     Dynamic per-bucket cap: one pacing window of bytes per bucket (so a whole finalizer-burst
        ///     of returns fits) plus the low-water slack, clamped by the entry floors/ceilings — see the
        ///     SIZING POLICY note and <see cref="Bucket"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CapFor(long bytes, long window, int slack)
        {
            return bytes >= LargeBucketThreshold
                ? (int)Math.Max(MaxBuffersPerLargeBucket, Math.Min(window + slack, MaxBuffersPerDynamicBucket))
                : (int)Math.Clamp(window + slack, MaxBuffersPerBucket, MaxBuffersPerDynamicBucket);
        }

        /// <summary>
        ///     Take a buffer of the given byte size. Returns either a reused
        ///     warm buffer or a fresh allocation; either way the caller owns
        ///     it and must eventually Return or Free it. The memory is NOT
        ///     zeroed.
        /// </summary>
        /// <param name="bytes">Byte size of the buffer. Must be &gt; 0.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static IntPtr Take(long bytes)
        {
            if (bytes > 0)
                AccountTake(bytes);

            if (GuardPagesEnabled && bytes > 0)
                return TakeGuarded(bytes);

            if (bytes < MinPoolableBytes || bytes >= MaxPoolableBytes)
            {
                Interlocked.Increment(ref _misses);
                PaceIfStarved();
                return (IntPtr)NativeMemory.Alloc((nuint)bytes);
            }

            if (_buckets.TryGetValue(bytes, out var bucket) && bucket.Stack.TryPop(out var ptr))
            {
                // Lock-free decrement matches the lock-free pop above.
                int depth = Interlocked.Decrement(ref bucket.Depth);
                Interlocked.Increment(ref _hits);
                // Low-water refill: request the paced collection BEFORE the
                // bucket runs dry so the finalizer-burst returns land while
                // LowWater warm buffers still cover the drain latency.
                if (depth <= bucket.LowWater)
                    PaceIfStarved();
                return ptr;
            }

            Interlocked.Increment(ref _misses);
            PaceIfStarved();
            return (IntPtr)NativeMemory.Alloc((nuint)bytes);
        }

        /// <summary>
        ///     Take a ZERO-INITIALIZED buffer of the given byte size. This is
        ///     the np.zeros / fill-with-default fast path and the analogue of
        ///     NumPy's <c>npy_alloc_cache_zero</c> / <c>PyDataMem_NEW_ZEROED</c>.
        ///
        ///     WHY calloc INSTEAD OF Take + memset
        ///     -----------------------------------
        ///     <see cref="NativeMemory.AllocZeroed"/> calls the CRT
        ///     <c>calloc</c>. For any non-trivial size the CRT/OS serves the
        ///     request from fresh, copy-on-write zero pages (Windows
        ///     <c>VirtualAlloc</c> / Linux <c>mmap(MAP_ANONYMOUS)</c>): the
        ///     pages are only physically committed and zeroed by the kernel
        ///     lazily, on first write. So zeroing a 10M-element (80 MB) block
        ///     costs ~0.01 ms instead of the ~14 ms an explicit element fill —
        ///     or even a <see cref="NativeMemory.Clear"/> memset (~21 ms) —
        ///     pays to touch every page up front.
        ///
        ///     The dirty same-size bucket cache used by <see cref="Take"/> is
        ///     intentionally NOT consulted: a recycled buffer is dirty and
        ///     would force a full memset, touching every page and throwing
        ///     away the entire lazy-zero advantage for exactly the large sizes
        ///     that matter. NumPy makes the same call — its zero cache only
        ///     engages below 1 KiB, a regime where the CRT's own low-
        ///     fragmentation heap already supplies that small-block reuse for
        ///     our calloc.
        ///
        ///     OWNERSHIP &amp; PRESSURE
        ///     ---------------------
        ///     The caller owns the returned pointer and must eventually
        ///     <see cref="Return"/> or free it; a calloc'd pointer is a normal
        ///     <see cref="NativeMemory"/> allocation, so Return may pool it for
        ///     later (non-zero) reuse by <see cref="Take"/> or free it via
        ///     <see cref="NativeMemory.Free"/>. Outstanding-byte accounting is
        ///     registered here exactly as <see cref="Take"/> does, so the
        ///     paired <see cref="Return"/> balances it, and the pacing trigger
        ///     keeps dropped np.zeros graphs collected promptly.
        /// </summary>
        /// <param name="bytes">Byte size of the buffer. Must be &gt;= 0.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static IntPtr TakeZeroed(long bytes)
        {
            if (bytes > 0)
                AccountTake(bytes);

            // Go straight to calloc — no dirty-bucket reuse. A recycled buffer
            // would need a full NativeMemory.Clear, which for large sizes throws
            // away the lazy demand-zero advantage and for small sizes is no
            // cheaper than calloc itself (while adding a ConcurrentDictionary
            // probe on every miss). Large sizes don't even reach here: the
            // caller (UnmanagedMemoryBlock.AllocateZeroed) routes them to
            // VirtualAlloc on Windows, and on other platforms calloc already
            // mmaps large blocks lazily.
            if (GuardPagesEnabled && bytes > 0)
            {
                // Guard pages from VirtualAlloc(MEM_COMMIT) are demand-zero already.
                Interlocked.Increment(ref _zeroedAllocs);
                return TakeGuarded(bytes);
            }

            Interlocked.Increment(ref _zeroedAllocs);
            PaceIfStarved();
            return (IntPtr)NativeMemory.AllocZeroed((nuint)bytes);
        }

        /// <summary>Guard-mode allocation: a page-protected buffer, tracked for <see cref="Return"/>.</summary>
        private static IntPtr TakeGuarded(long bytes)
        {
            var p = OsVirtualMemory.AllocGuarded(bytes, out var region);
            if (p == IntPtr.Zero)   // VirtualAlloc exhausted: fall back so the run continues
                return (IntPtr)NativeMemory.Alloc((nuint)bytes);
            _guardRegions[p] = region;
            Interlocked.Increment(ref _misses);
            return p;
        }

        /// <summary>
        ///     Return a buffer to the pool. Caller transfers ownership; do
        ///     NOT touch the pointer after the call.
        ///
        ///     If the size falls outside the pool window or the bucket is
        ///     already at capacity, the buffer is freed via
        ///     <see cref="NativeMemory.Free"/> instead of being kept.
        /// </summary>
        /// <param name="ptr">Pointer obtained from <see cref="Take"/> or a paired NativeMemory.Alloc.</param>
        /// <param name="bytes">Size in bytes originally requested.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static void Return(IntPtr ptr, long bytes)
        {
            if (ptr == IntPtr.Zero) return;

            // Outstanding-byte accounting: the buffer leaves user hands
            // here regardless of whether it parks in the pool or frees.
            if (bytes > 0)
                AccountReturn(bytes);

            if (GuardPagesEnabled && _guardRegions.TryRemove(ptr, out var region))
            {
                OsVirtualMemory.FreeGuarded(region);
                Interlocked.Increment(ref _returnsFreed);
                return;
            }

            if (bytes < MinPoolableBytes || bytes >= MaxPoolableBytes)
            {
                NativeMemory.Free((void*)ptr);
                Interlocked.Increment(ref _returnsFreed);
                return;
            }

            var bucket = _buckets.GetOrAdd(bytes, static s => new Bucket(s));
            // Increment first; if we go over the cap, undo and free. This is
            // a benign race — slightly more buffers can be in flight than
            // the cap suggests at any moment, but never permanently.
            int newDepth = Interlocked.Increment(ref bucket.Depth);
            if (newDepth > bucket.Cap)
            {
                Interlocked.Decrement(ref bucket.Depth);
                NativeMemory.Free((void*)ptr);
                Interlocked.Increment(ref _returnsFreed);
                return;
            }

            bucket.Stack.Push(ptr);
            Interlocked.Increment(ref _returns);
        }

        /// <summary>
        ///     Drain every pooled buffer immediately (testing / memory pressure).
        ///     Calls <see cref="NativeMemory.Free"/> on each. No pressure
        ///     adjustment — pooled buffers carry none (they are not outstanding).
        /// </summary>
        public static void Clear()
        {
            foreach (var kv in _buckets)
            {
                while (kv.Value.Stack.TryPop(out var ptr))
                {
                    NativeMemory.Free((void*)ptr);
                    Interlocked.Decrement(ref kv.Value.Depth);
                }
            }
        }

        // ---------------------------------------------------------------
        // Pacing + coarse-pressure internals.
        // ---------------------------------------------------------------

        /// <summary>
        ///     Hot-path bookkeeping for a hand-out: bumps the pacing window
        ///     counter and the outstanding total, and registers a coarse
        ///     pressure chunk when net outstanding grew past the registered
        ///     amount by a whole chunk (rare → cold helper).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AccountTake(long bytes)
        {
            Interlocked.Add(ref _takenBytesSincePace, bytes);
            long outstanding = Interlocked.Add(ref _outstandingBytes, bytes);
            long registered = Volatile.Read(ref _registeredPressure);
            if (outstanding - registered >= PressureChunkBytes)
                AddPressureChunk(registered);
        }

        /// <summary>Mirror of <see cref="AccountTake"/> for the give-back side (Return / free).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AccountReturn(long bytes)
        {
            long outstanding = Interlocked.Add(ref _outstandingBytes, -bytes);
            long registered = Volatile.Read(ref _registeredPressure);
            // Hysteresis: keep one chunk of slack so a workload oscillating
            // around a chunk boundary doesn't churn Add/Remove pairs.
            if (registered - outstanding >= 2 * PressureChunkBytes)
                RemovePressureChunk(registered);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void AddPressureChunk(long observedRegistered)
        {
            // CAS so exactly one thread registers each chunk transition.
            if (Interlocked.CompareExchange(ref _registeredPressure,
                    observedRegistered + PressureChunkBytes, observedRegistered) == observedRegistered)
                GC.AddMemoryPressure(PressureChunkBytes);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void RemovePressureChunk(long observedRegistered)
        {
            if (observedRegistered >= PressureChunkBytes &&
                Interlocked.CompareExchange(ref _registeredPressure,
                    observedRegistered - PressureChunkBytes, observedRegistered) == observedRegistered)
                GC.RemoveMemoryPressure(PressureChunkBytes);
        }

        /// <summary>
        ///     The pacing trigger (see the GC PACING header note): when at least
        ///     <see cref="PaceThresholdBytes"/> were handed out since the last
        ///     paced collection, request ONE forced, blocking, non-compacting
        ///     gen-0 collection so the finalizer thread can return the dropped
        ///     buffers. Gated so concurrent triggers collapse into one; the
        ///     fast path (below threshold) is a volatile read + compare.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void PaceIfStarved()
        {
            if (!GcPacingEnabled)
                return;
            if (Volatile.Read(ref _takenBytesSincePace) < PaceThresholdBytes)
                return;
            PaceNow();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void PaceNow()
        {
            if (Interlocked.Exchange(ref _paceGate, 1) != 0)
                return;   // a paced collection is already in flight
            try
            {
                // Reset BEFORE collecting so takes that race with the
                // collection start a fresh window instead of re-triggering.
                Interlocked.Exchange(ref _takenBytesSincePace, 0);
                Interlocked.Increment(ref _pacedCollections);
                // Gen 0 suffices: the dropped wrapper graphs were allocated
                // inside the current window, so they are still gen 0. Non-
                // compacting keeps the pause at the ~4-5 µs floor.
                GC.Collect(0, GCCollectionMode.Forced, blocking: true, compacting: false);
            }
            finally
            {
                Volatile.Write(ref _paceGate, 0);
            }
        }
    }
}
