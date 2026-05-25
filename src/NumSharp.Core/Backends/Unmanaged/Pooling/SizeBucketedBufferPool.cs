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
    ///     • Below <see cref="MinPoolableBytes"/> (4 KiB): no pooling. The
    ///       existing <c>StackedMemoryPool</c> already serves scalar / tiny
    ///       allocations; adding them here just doubles the work.
    ///     • Above <see cref="MaxPoolableBytes"/> (default 64 MiB): no
    ///       pooling. Huge buffers are rare and the memory cost of keeping
    ///       them around dwarfs the alloc-cost savings.
    ///     • Per-bucket cap of <see cref="MaxBuffersPerBucket"/> entries to
    ///       bound peak resident memory.
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
        /// <summary>Minimum allocation size to pool (bytes). Smaller allocations skip the pool entirely.</summary>
        public const long MinPoolableBytes = 4096;

        /// <summary>
        ///     Maximum allocation size to pool (bytes). Capped at &lt; 1 MiB
        ///     so the pool can't accumulate large resident buffers — a single
        ///     workload pattern with many 4 MiB+ allocations could otherwise
        ///     keep tens of MiB pinned in pool indefinitely. Allocations at or
        ///     above this cap go straight to <see cref="NativeMemory.Alloc"/>
        ///     and are freed straight back via <see cref="NativeMemory.Free"/>
        ///     on release; no pool involvement either way.
        /// </summary>
        public const long MaxPoolableBytes = 1024L * 1024;

        /// <summary>Maximum number of buffers kept per exact-size bucket.</summary>
        public const int MaxBuffersPerBucket = 8;

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

        /// <summary>How many Take calls served from the pool.</summary>
        public static long Hits => Interlocked.Read(ref _hits);

        /// <summary>How many Take calls fell through to NativeMemory.Alloc.</summary>
        public static long Misses => Interlocked.Read(ref _misses);

        /// <summary>How many Return calls accepted the buffer into the pool.</summary>
        public static long Returns => Interlocked.Read(ref _returns);

        /// <summary>How many Return calls freed the buffer (bucket full / out-of-range).</summary>
        public static long ReturnsFreed => Interlocked.Read(ref _returnsFreed);

        /// <summary>Reset all counters. Diagnostic only.</summary>
        public static void ResetCounters()
        {
            Interlocked.Exchange(ref _hits, 0);
            Interlocked.Exchange(ref _misses, 0);
            Interlocked.Exchange(ref _returns, 0);
            Interlocked.Exchange(ref _returnsFreed, 0);
        }

        /// <summary>
        ///     Take a buffer of the given byte size. Returns either a reused
        ///     warm buffer or a fresh allocation; either way the caller owns
        ///     it and must eventually Return or Free it. The memory is NOT
        ///     zeroed.
        /// </summary>
        /// <param name="bytes">Byte size of the buffer. Must be &gt; 0.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IntPtr Take(long bytes)
        {
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
            int newDepth = Interlocked.Increment(ref depth.Value);
            if (newDepth > MaxBuffersPerBucket)
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
        ///     Calls <see cref="NativeMemory.Free"/> on each.
        /// </summary>
        public static void Clear()
        {
            foreach (var kv in _buckets)
            {
                while (kv.Value.TryPop(out var ptr))
                    NativeMemory.Free((void*)ptr);
                if (_bucketDepth.TryGetValue(kv.Key, out var depth))
                    Interlocked.Exchange(ref depth.Value, 0);
            }
        }
    }
}
