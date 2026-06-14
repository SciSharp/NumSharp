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
    ///     • Per-bucket cap of <see cref="MaxBuffersPerBucket"/> entries
    ///       (<see cref="MaxBuffersPerLargeBucket"/> at ≥ 1 MiB) to bound
    ///       peak resident memory.
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
        ///     resident (8 × size) and a pool hit skips the pressure churn
        ///     entirely (see the pool-owned pressure accounting below).
        /// </summary>
        public const long MinPoolableBytes = 1;

        /// <summary>
        ///     Maximum allocation size to pool (bytes). Wave 2.4 raised this
        ///     from 1 MiB to 64 MiB: the dominant benchmark/e2e shapes (4M
        ///     elements = 16 MiB float32 / 32 MiB float64 outputs) all missed
        ///     the old cap and paid ~0.3–0.4 ms of first-touch page faults per
        ///     call — the "allocator tax" residual on every measured e2e
        ///     strided row. NumPy gets the same reuse for free from glibc's
        ///     arena caching. Resident growth is bounded by the per-bucket cap,
        ///     which drops to <see cref="MaxBuffersPerLargeBucket"/> at
        ///     <see cref="LargeBucketThreshold"/> (realistic workloads keep one
        ///     or two hot output shapes — exactly the tcache pattern).
        /// </summary>
        public const long MaxPoolableBytes = 64L * 1024 * 1024;

        /// <summary>Maximum number of buffers kept per exact-size bucket (below <see cref="LargeBucketThreshold"/>).</summary>
        public const int MaxBuffersPerBucket = 8;

        /// <summary>Bucket sizes at/above this hold at most <see cref="MaxBuffersPerLargeBucket"/> buffers.</summary>
        public const long LargeBucketThreshold = 1024L * 1024;

        /// <summary>Per-bucket cap for large (≥ 1 MiB) buckets — bounds peak resident memory.</summary>
        public const int MaxBuffersPerLargeBucket = 2;

        // Bucket map keyed on exact byte size. ConcurrentStack gives lock-
        // free Push/TryPop, which is the entire fast-path here.
        private static readonly ConcurrentDictionary<long, ConcurrentStack<IntPtr>> _buckets = new();

        // Track depth per bucket so we can cap without locking the stack.
        // ConcurrentStack lacks a thread-safe Count that doesn't walk the
        // list, so a separate counter is cheaper.
        private static readonly ConcurrentDictionary<long, StrongBox<int>> _bucketDepth = new();

        // Diagnostic counters — useful for telling whether the pool is
        // doing real work or just adding overhead. Not on a hot path so
        // Interlocked is fine.
        private static long _hits;
        private static long _misses;
        private static long _returns;
        private static long _returnsFreed;
        private static long _zeroedAllocs;

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

        /// <summary>Reset all counters. Diagnostic only.</summary>
        public static void ResetCounters()
        {
            Interlocked.Exchange(ref _hits, 0);
            Interlocked.Exchange(ref _misses, 0);
            Interlocked.Exchange(ref _returns, 0);
            Interlocked.Exchange(ref _returnsFreed, 0);
            Interlocked.Exchange(ref _zeroedAllocs, 0);
        }

        /// <summary>
        ///     Take a buffer of the given byte size. Returns either a reused
        ///     warm buffer or a fresh allocation; either way the caller owns
        ///     it and must eventually Return or Free it. The memory is NOT
        ///     zeroed.
        /// </summary>
        /// <param name="bytes">Byte size of the buffer. Must be &gt; 0.</param>
        // -----------------------------------------------------------------
        // GC memory pressure accounting (Wave 2.4) — POOL-OWNED, tracking
        // the buffer's LIVE state (checked out to a caller), not its native
        // residency. Pressure exists so the GC collects often enough for
        // finalizer-driven NDArray reclamation to keep up (issue #501) —
        // pressure must therefore follow what user code holds. Registering
        // pressure for IDLE pooled buffers was measured to inflate the GC's
        // view of memory tightness by the pool's whole resident set
        // (~100-200 MB of warm large buffers) and drove constant gen2
        // collections: every probe row degraded 30-50%. So: Add on every
        // Take (fresh or hit), Remove on every Return (pooled or freed) —
        // the same net semantics the per-block Disposer accounting had,
        // owned by the pool so the Disposer stays pressure-free.
        // -----------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IntPtr Take(long bytes)
        {
            if (bytes > 0)
                GC.AddMemoryPressure(bytes);

            if (bytes < MinPoolableBytes || bytes >= MaxPoolableBytes)
            {
                Interlocked.Increment(ref _misses);
                return (IntPtr)NativeMemory.Alloc((nuint)bytes);
            }

            if (_buckets.TryGetValue(bytes, out var stack) && stack.TryPop(out var ptr))
            {
                // Lock-free decrement matches the lock-free pop above.
                if (_bucketDepth.TryGetValue(bytes, out var depth))
                    Interlocked.Decrement(ref depth.Value);
                Interlocked.Increment(ref _hits);
                return ptr;
            }

            Interlocked.Increment(ref _misses);
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
        ///     <see cref="NativeMemory.Free"/>. GC memory pressure is registered
        ///     here exactly as <see cref="Take"/> does, so the paired
        ///     <see cref="Return"/> balances it (live-state accounting).
        /// </summary>
        /// <param name="bytes">Byte size of the buffer. Must be &gt;= 0.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IntPtr TakeZeroed(long bytes)
        {
            if (bytes > 0)
                GC.AddMemoryPressure(bytes);

            // Go straight to calloc — no dirty-bucket reuse. A recycled buffer
            // would need a full NativeMemory.Clear, which for large sizes throws
            // away the lazy demand-zero advantage and for small sizes is no
            // cheaper than calloc itself (while adding a ConcurrentDictionary
            // probe on every miss). Large sizes don't even reach here: the
            // caller (UnmanagedMemoryBlock.AllocateZeroed) routes them to
            // VirtualAlloc on Windows, and on other platforms calloc already
            // mmaps large blocks lazily.
            Interlocked.Increment(ref _zeroedAllocs);
            return (IntPtr)NativeMemory.AllocZeroed((nuint)bytes);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Return(IntPtr ptr, long bytes)
        {
            if (ptr == IntPtr.Zero) return;

            // Live-state pressure accounting: the buffer leaves user hands
            // here regardless of whether it parks in the pool or frees.
            if (bytes > 0)
                GC.RemoveMemoryPressure(bytes);

            if (bytes < MinPoolableBytes || bytes >= MaxPoolableBytes)
            {
                NativeMemory.Free((void*)ptr);
                Interlocked.Increment(ref _returnsFreed);
                return;
            }

            var depth = _bucketDepth.GetOrAdd(bytes, _ => new StrongBox<int>(0));
            // Increment first; if we go over the cap, undo and free. This is
            // a benign race — slightly more buffers can be in flight than
            // the cap suggests at any moment, but never permanently.
            // Large buckets hold fewer buffers to bound resident memory.
            int cap = bytes >= LargeBucketThreshold ? MaxBuffersPerLargeBucket : MaxBuffersPerBucket;
            int newDepth = Interlocked.Increment(ref depth.Value);
            if (newDepth > cap)
            {
                Interlocked.Decrement(ref depth.Value);
                NativeMemory.Free((void*)ptr);
                Interlocked.Increment(ref _returnsFreed);
                return;
            }

            var stack = _buckets.GetOrAdd(bytes, _ => new ConcurrentStack<IntPtr>());
            stack.Push(ptr);
            Interlocked.Increment(ref _returns);
        }

        /// <summary>
        ///     Drain every pooled buffer immediately (testing / memory pressure).
        ///     Calls <see cref="NativeMemory.Free"/> on each. No pressure
        ///     adjustment — pooled buffers carry none (live-state accounting).
        /// </summary>
        public static void Clear()
        {
            foreach (var kv in _buckets)
            {
                while (kv.Value.TryPop(out var ptr))
                {
                    NativeMemory.Free((void*)ptr);
                    if (_bucketDepth.TryGetValue(kv.Key, out var depth))
                        Interlocked.Decrement(ref depth.Value);
                }
            }
        }
    }
}
