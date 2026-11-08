using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Fuzz.Ownership
{
    /// <summary>
    ///     Looks for parked pool buffers inside memory a caller still holds: the buffer pool must never own memory that a
    ///     live array reads, whoever allocated it.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>What it catches.</b> A buffer the pool parks is handed to the next same-size <c>Take</c>. If that buffer lies
    ///     inside a caller's array — the caller's own buffer returned while the caller holds it, or memory NumSharp never
    ///     allocated (a pinned managed array, a caller's unmanaged block) wrapped by a NEW pooled block and returned when that
    ///     block is disposed — two owners write the same bytes. The ARC state of the caller's own block can look intact in
    ///     the second case (a different block did the returning), so the ownership ledger scans the pool too.
    ///     </para>
    ///     <para>
    ///     <b>Which buckets.</b> The pool keys its buckets by exact byte size, and a long replay leaves thousands of them; a
    ///     scan of every bucket after every case made the corpus sweep 25 times slower. A buffer returned in a caller's place
    ///     is the size of the block that held it: the caller's own block (its slice's byte length) or a block laid over the
    ///     caller's view (the view's byte length). The scan therefore reads only the buckets of the sizes it is given. An
    ///     adopting block of any other size is outside its reach.
    ///     </para>
    ///     <para>
    ///     <b>What it cannot catch.</b> A returned buffer the pool FREES instead of parking (a full bucket, a size outside the
    ///     pool's window) leaves nothing to find; freeing memory NumSharp does not own then corrupts the native heap, which
    ///     ends the process instead.
    ///     </para>
    ///     <para>
    ///     The pool's buckets are private: they are read through reflection, and a missing field fails loudly rather than
    ///     reading as "nothing parked".
    ///     </para>
    /// </remarks>
    internal static class PoolScan
    {
        /// <summary>The pool's bucket map (<c>ConcurrentDictionary&lt;long, Bucket&gt;</c>, Bucket private).</summary>
        private static readonly FieldInfo BucketsField =
            typeof(SizeBucketedBufferPool).GetField("_buckets", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("SizeBucketedBufferPool has no _buckets field: update PoolScan");

        /// <summary>The <c>Stack</c> field of the private Bucket class (resolved from the first bucket seen).</summary>
        private static FieldInfo _stackField;

        /// <summary>
        ///     How many of the most recently parked buffers of a bucket a scan reads. A buffer parked BEFORE the caller's
        ///     arrays were allocated cannot lie inside them (two live allocations never overlap, and a parked buffer is live
        ///     pool memory), so only buffers parked since the case began can — and the pool's stacks are last-in first-out,
        ///     so those sit on top. A case returns a handful of buffers per size; 64 leaves a wide margin, and reading whole
        ///     buckets (thousands of small buffers each, after a long replay) doubled the corpus sweep's wall time.
        /// </summary>
        public const int RecentPerBucket = 64;

        /// <summary>
        ///     The first recently parked buffer of one of the given sizes whose start lies inside one of
        ///     <paramref name="ranges"/>, or <see cref="IntPtr.Zero"/>.
        /// </summary>
        /// <param name="ranges">Half-open byte ranges <c>[lo, hi)</c> of memory callers hold.</param>
        /// <param name="sizes">The bucket sizes to read (see the class remarks for why not every bucket).</param>
        /// <returns>The offending pooled pointer, or zero when none is parked inside the ranges.</returns>
        /// <exception cref="InvalidOperationException">The pool's internals moved (update the scan).</exception>
        /// <remarks>
        ///     Enumerating a <see cref="ConcurrentStack{T}"/> walks a moment-in-time snapshot from its top without copying it,
        ///     so a scan costs one enumerator per bucket read and reads at most <see cref="RecentPerBucket"/> buffers of each.
        /// </remarks>
        public static IntPtr FirstInside(IReadOnlyList<(ulong lo, ulong hi)> ranges, IEnumerable<long> sizes)
        {
            if (ranges.Count == 0)
                return IntPtr.Zero;
            if (BucketsField.GetValue(null) is not IDictionary buckets)
                throw new InvalidOperationException("SizeBucketedBufferPool._buckets is not a dictionary: update PoolScan");

            foreach (long size in sizes)
            {
                // The non-generic indexer: the value type (Bucket) is private, so the typed dictionary cannot be named.
                if (size <= 0 || !buckets.Contains(size))
                    continue;
                var bucket = buckets[size];
                if (bucket is null)
                    continue;
                var stackField = _stackField ??= bucket.GetType().GetField("Stack", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)
                                                 ?? throw new InvalidOperationException("the pool's Bucket has no Stack field: update PoolScan");
                if (stackField.GetValue(bucket) is not ConcurrentStack<IntPtr> stack)
                    continue;
                int read = 0;
                foreach (var p in stack)
                {
                    if (++read > RecentPerBucket)
                        break;
                    ulong a = (ulong)p.ToInt64();
                    for (int i = 0; i < ranges.Count; i++)
                        if (a >= ranges[i].lo && a < ranges[i].hi)
                            return p;
                }
            }
            return IntPtr.Zero;
        }
    }
}
