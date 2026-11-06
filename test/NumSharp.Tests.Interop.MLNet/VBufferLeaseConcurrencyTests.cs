using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     The <see cref="VBuffer{T}"/> import lease's booking under concurrency (P8 Q3; the owner's request of 2026-11-06 to
    ///     prove multithreading, memory safety and GC load): views of fresh buffers made on eight threads, kept alive for a
    ///     while, re-checked, then disposed on their own thread, on ANOTHER thread or dropped for the finalizer, while
    ///     another thread collects. Every view reads its own buffer for as long as it lives, and every booking comes back.
    /// </summary>
    /// <remarks>
    ///     The suite runs without parallelism (<c>[assembly: DoNotParallelize]</c>), so the process-wide booking counters
    ///     move only with this test's threads. The base class fails a test that leaves a live import behind.
    /// </remarks>
    [TestClass]
    public class VBufferLeaseConcurrencyTests : MLNetTestBase
    {
        /// <summary>Worker threads.</summary>
        private const int Threads = 8;

        /// <summary>
        ///     Eight threads each view 200 fresh 16 KB buffers holding the thread's own values, keep their last eight views
        ///     alive, and check a view's values again when they let go of it - after the other threads' collections - before
        ///     disposing it here, handing it to another thread, or dropping it. Every view reads its own values to the end,
        ///     and once the dropped views are collected the foreign balance is back to the byte.
        /// </summary>
        [TestMethod]
        public void Views_FromManyThreads_ReadTheirOwnBuffers_AndBookBackToBaseline()
        {
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            int wrong = 0;
            var crossThread = new ConcurrentQueue<NDArray>();
            using var stop = new CancellationTokenSource();
            Task collector = Task.Run(() =>
            {
                // Young collections while the workers run: dropped views are finalized, and their leases released,
                // concurrently with new views being made and booked.
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            });
            try
            {
                Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
                {
                    var window = new Queue<(NDArray View, float Value)>();
                    for (int i = 0; i < 200; i++)
                    {
                        float value = t * 1000 + i;
                        var values = new float[4096];
                        Array.Fill(values, value);
                        NDArray view = new VBuffer<float>(values.Length, values).AsNDArray();
                        if (!Holds(view, value))
                            Interlocked.Increment(ref wrong);
                        window.Enqueue((view, value));
                        if (window.Count > 8)
                            LetGo(window.Dequeue(), i, crossThread, ref wrong);
                    }
                    while (window.Count > 0)
                        LetGo(window.Dequeue(), 0, crossThread, ref wrong);
                });
            }
            finally
            {
                stop.Cancel();
                collector.Wait();
            }
            while (crossThread.TryDequeue(out NDArray rest))
                rest.Dispose();

            wrong.Should().Be(0, "every view reads its own buffer for as long as it lives");
            WaitFor(() => SizeBucketedBufferPool.OutstandingForeignBytes == foreignBefore, 10_000)
                .Should().BeTrue($"every lease gives its booking back (foreign {SizeBucketedBufferPool.OutstandingForeignBytes} vs {foreignBefore})");
        }

        /// <summary>Does <paramref name="view"/> still hold <paramref name="value"/> at its first, middle and last element?</summary>
        /// <param name="view">A 4,096-element float32 view.</param>
        /// <param name="value">The value its buffer was filled with.</param>
        /// <returns><c>true</c> when all three elements hold the value.</returns>
        private static bool Holds(NDArray view, float value) =>
            view.GetSingle(0) == value && view.GetSingle(2048) == value && view.GetSingle(4095) == value;

        /// <summary>
        ///     Checks a view's values one last time, then disposes it here, hands it to another thread to dispose, or drops it
        ///     (the third way leaves it to the finalizer).
        /// </summary>
        /// <param name="entry">The view and the value its buffer was filled with.</param>
        /// <param name="i">The iteration, which picks the way.</param>
        /// <param name="crossThread">The queue other threads dispose from.</param>
        /// <param name="wrong">The count of views that read wrong values.</param>
        private static void LetGo((NDArray View, float Value) entry, int i, ConcurrentQueue<NDArray> crossThread, ref int wrong)
        {
            if (!Holds(entry.View, entry.Value))
                Interlocked.Increment(ref wrong);
            switch (i % 3)
            {
                case 0:
                    entry.View.Dispose();
                    break;
                case 1:
                    crossThread.Enqueue(entry.View);
                    if (crossThread.TryDequeue(out NDArray other))
                        other.Dispose();
                    break;
                default:
                    break;   // dropped: the finalizer releases the pin and gives the booking back
            }
        }
    }
}
