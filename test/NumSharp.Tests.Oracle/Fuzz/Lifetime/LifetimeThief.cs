using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Fuzz.Lifetime
{
    /// <summary>
    ///     The stress thread of the lifetime replay: while a call is open it forces full collections, drains the finalizer
    ///     queue and overwrites every buffer the finalizers just returned to the pool — so a kernel that still reads or
    ///     writes an operand whose last reference died meets the poison (a wrong value, a spurious IndexError) or freed
    ///     memory (an access violation in the child process).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the collections are confined to open calls.</b> The thread waits on <see cref="Enter"/>; collecting
    ///     between calls only promotes the next call's temporaries before the call starts and spends the time the
    ///     calls need. Three of four collections are gen-0 (cheap, so even a window of tens of microseconds is likely
    ///     to contain one; a temporary built for the call is young), the fourth is full: an operand promoted by an
    ///     earlier collection of the same call is found unreachable only by a collection of its generation.
    ///     </para>
    ///     <para>
    ///     <b>Why poison, not only collect.</b> A freed operand's buffer goes back to <see cref="SizeBucketedBufferPool"/>
    ///     (its block's <c>~Disposer</c>, a pool return), where it keeps its old bytes until someone takes it — a kernel
    ///     reading it would still compute the right answer. The thread therefore takes the parked buffers of every
    ///     bucket and fills them with a pattern, from the END toward the start (a forward-reading kernel meets the
    ///     poison soonest), then returns them. The pattern rotates <c>0x00</c> / <c>0xFF</c> / <c>0x5A</c>: zero makes an
    ///     index array silently in range, the other two make it raise or run off the buffer.
    ///     </para>
    ///     <para>
    ///     <b>The bucket list</b> comes from the pool's private bucket table (reflection; test-only): the operand sizes of
    ///     the current case on every cycle, every other bucket on every eighth (a lost anchor on an internal temporary
    ///     has some other size). Buckets above <see cref="MaxPoisonBytes"/> are skipped (the pool does not park them).
    ///     </para>
    ///     <para>
    ///     The thread never touches a buffer anyone holds: <see cref="SizeBucketedBufferPool.Take"/> hands out only
    ///     parked (free) buffers or fresh ones.
    ///     </para>
    /// </remarks>
    internal sealed class LifetimeThief : IDisposable
    {
        /// <summary>The largest bucket the thread poisons (the pool's own parking limit).</summary>
        public const long MaxPoisonBytes = SizeBucketedBufferPool.MaxPoolableBytes;

        /// <summary>
        ///     How many parked buffers of one bucket the thread takes per cycle: ONE, the top of the bucket's stack —
        ///     the buffer a finalizer returned last, which is the freed operand. A shorter cycle puts more collections
        ///     inside a short window; that matters more than poisoning older parked buffers.
        /// </summary>
        private const int TakesPerBucket = 1;

        /// <summary>The stress thread (a background thread: a test host that exits never waits for it).</summary>
        private readonly Thread _thread;

        /// <summary>
        ///     Set while a call is open. A kernel wait (spin count 1), not a spin loop: an idle thread must not compete with
        ///     the replay for a core, and a <c>Thread.SpinWait</c> loop on .NET 8 is an FCALL the GC cannot suspend.
        /// </summary>
        private readonly ManualResetEventSlim _open = new(false, spinCount: 1);

        /// <summary>Set by <see cref="Dispose"/>: the loop exits at its next check.</summary>
        private volatile bool _stop;

        /// <summary>The current case's operand bucket sizes (replaced as a whole array, so the loop reads a consistent set).</summary>
        private volatile long[] _caseSizes = Array.Empty<long>();

        /// <summary>The collections forced so far (read by the replay for its report).</summary>
        private int _collections;

        /// <summary>The poison byte of the current cycle (rotates 0x00 / 0xFF / 0x5A).</summary>
        private byte _pattern;

        /// <summary>The pool's private bucket table (<c>ConcurrentDictionary&lt;long, Bucket&gt;</c>).</summary>
        private static readonly FieldInfo BucketsField =
            typeof(SizeBucketedBufferPool).GetField("_buckets", BindingFlags.NonPublic | BindingFlags.Static);

        /// <summary>A bucket's parked-buffer count (<c>Bucket.Depth</c>).</summary>
        private static readonly FieldInfo DepthField =
            typeof(SizeBucketedBufferPool).GetNestedType("Bucket", BindingFlags.NonPublic)?.GetField("Depth");

        /// <summary>Starts the (idle) stress thread.</summary>
        public LifetimeThief()
        {
            _thread = new Thread(Loop) { IsBackground = true, Name = "lifetime-thief" };
            _thread.Start();
        }

        /// <summary>The number of collections forced so far.</summary>
        public int Collections => Volatile.Read(ref _collections);

        /// <summary>Sets the bucket sizes of the current case's operands (poisoned on every cycle).</summary>
        /// <param name="sizes">The byte sizes of the operands the case builds afresh.</param>
        public void SetCaseSizes(IEnumerable<long> sizes)
            => _caseSizes = sizes.Where(s => s > 0 && s < MaxPoisonBytes).Distinct().ToArray();

        /// <summary>Opens a call: the thread starts collecting and poisoning.</summary>
        public void Enter() => _open.Set();

        /// <summary>Closes the call: the thread finishes its current cycle and waits for the next call.</summary>
        public void Exit() => _open.Reset();

        /// <summary>Stops the thread and waits for it.</summary>
        public void Dispose()
        {
            _stop = true;
            _open.Set();
            _thread.Join();
            _open.Dispose();
        }

        /// <summary>
        ///     The stress loop: wait for an open call, collect, drain finalizers, poison, repeat until <see cref="Dispose"/>.
        /// </summary>
        private void Loop()
        {
            int cycle = 0;
            while (!_stop)
            {
                _open.Wait();
                if (_stop)
                    return;
                // Mostly gen-0 collections (tens of microseconds on the child's small heap, so a short window — a mask
                // scan, a cast — is likely to contain one), with a full collection every fourth cycle: an operand an
                // earlier collection of the same call promoted (it was still referenced then) is found only by a
                // collection of its own generation or above.
                GC.Collect(cycle % 4 == 3 ? 2 : 0, GCCollectionMode.Forced, blocking: true, compacting: false);
                GC.WaitForPendingFinalizers();
                Interlocked.Increment(ref _collections);
                _pattern = _pattern == 0x00 ? (byte)0xFF : _pattern == 0xFF ? (byte)0x5A : (byte)0x00;
                Poison(_caseSizes);
                if (++cycle % 8 == 0)
                    Poison(AllBucketSizes());   // a lost anchor on an internal temporary has some other size
            }
        }

        /// <summary>
        ///     Takes the parked buffers of each size (at most <see cref="TakesPerBucket"/>, never more than the bucket
        ///     holds — taking from an empty bucket would allocate a fresh buffer, which holds nothing to poison), fills
        ///     them from the end, and returns them.
        /// </summary>
        /// <param name="sizes">The bucket sizes to poison.</param>
        /// <remarks>
        ///     The thread owns every buffer it fills: it fills only what <see cref="SizeBucketedBufferPool.Take"/> handed it.
        ///     Filling the parked buffers where they lie would race the main thread taking one of them for a live
        ///     result, and the poison in that result would read as a lifetime bug that is not there.
        /// </remarks>
        private unsafe void Poison(long[] sizes)
        {
            var buckets = BucketsField?.GetValue(null) as IDictionary;
            var taken = new List<(IntPtr p, long bytes)>();
            foreach (var bytes in sizes)
            {
                int parked = Math.Min(TakesPerBucket, BucketDepth(buckets, bytes));
                for (int k = 0; k < parked; k++)
                {
                    var p = SizeBucketedBufferPool.Take(bytes);
                    FillFromEnd((byte*)p, bytes, _pattern);
                    taken.Add((p, bytes));
                }
            }
            foreach (var (p, bytes) in taken)
                SizeBucketedBufferPool.Return(p, bytes);
        }

        /// <summary>The number of buffers parked in one bucket (0 when the bucket does not exist).</summary>
        /// <param name="buckets">The pool's bucket table (null when the reflection lookup failed).</param>
        /// <param name="bytes">The bucket size.</param>
        /// <returns>The bucket's depth; a racy snapshot, which is all a poisoning heuristic needs.</returns>
        private static int BucketDepth(IDictionary buckets, long bytes)
        {
            if (buckets == null || buckets[bytes] is not object bucket)
                return 0;
            return DepthField?.GetValue(bucket) is int depth ? depth : 0;
        }

        /// <summary>Fills [p, p + bytes) with a pattern in 64 KiB chunks, the last chunk first.</summary>
        /// <param name="p">A buffer the thread owns (taken from the pool).</param>
        /// <param name="bytes">Its size.</param>
        /// <param name="pattern">The poison byte.</param>
        private static unsafe void FillFromEnd(byte* p, long bytes, byte pattern)
        {
            const long Chunk = 64 * 1024;
            for (long end = bytes; end > 0; end -= Chunk)
            {
                long start = Math.Max(0, end - Chunk);
                new Span<byte>(p + start, (int)(end - start)).Fill(pattern);
            }
        }

        /// <summary>Every bucket size the pool currently has, below <see cref="MaxPoisonBytes"/>.</summary>
        /// <returns>The sizes (empty when the reflection lookup failed: the case sizes are still poisoned).</returns>
        private static long[] AllBucketSizes()
        {
            if (BucketsField?.GetValue(null) is not IDictionary buckets)
                return Array.Empty<long>();
            var sizes = new List<long>();
            foreach (var key in buckets.Keys)
                if (key is long s && s > 0 && s < MaxPoisonBytes)
                    sizes.Add(s);
            return sizes.ToArray();
        }
    }
}
