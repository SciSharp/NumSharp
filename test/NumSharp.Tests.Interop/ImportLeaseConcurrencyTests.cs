using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged.Pooling;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The pythonnet import lease's booking under concurrency (P8 Q2; the owner's request of 2026-11-06 to prove
    ///     multithreading, memory safety and GC load): zero-copy views of numpy arrays made on four threads that contend for
    ///     the GIL, kept alive for a while, re-checked, then disposed on their own thread, on ANOTHER thread, or dropped for
    ///     the finalizer, while another thread collects. A lease's Python side is released under the GIL by a queued drain,
    ///     so every path ends in that drain. Every view reads its own array while it lives, and every booking comes back.
    /// </summary>
    [TestClass]
    public class ImportLeaseConcurrencyTests : InteropTestBase
    {
        /// <summary>Worker threads (each takes the GIL for every view it makes).</summary>
        private const int Threads = 4;

        /// <summary>
        ///     Four threads each take 200 views of their own 16 KB numpy array of the thread's value, keep their last eight
        ///     views alive and check a view again when they let go of it - after the other threads' views and collections -
        ///     before disposing it here, handing it to another thread, or dropping it. Every view reads its own array to the
        ///     end, no thread deadlocks, and once the drops are collected and the queued releases drained the foreign balance
        ///     is back to the byte.
        /// </summary>
        [TestMethod]
        public void Views_FromManyThreads_ReadTheirOwnArrays_AndBookBackToBaseline()
        {
            for (int t = 0; t < Threads; t++)
                PyExec($"lease_conc_{t} = np.full(4096, {t + 1}, dtype=np.float32)");
            long foreignBefore = SizeBucketedBufferPool.OutstandingForeignBytes;
            int wrong = 0;
            var failures = new ConcurrentQueue<Exception>();
            var crossThread = new ConcurrentQueue<NDArray>();
            using var stop = new CancellationTokenSource();
            var collector = new Thread(() =>
            {
                // Young collections while the workers run: dropped views are finalized and their releases queued
                // concurrently with new views being made, booked and drained.
                while (!stop.IsCancellationRequested)
                {
                    GC.Collect(0);
                    Thread.Sleep(1);
                }
            }) { IsBackground = true };
            collector.Start();

            void Worker(int t)
            {
                try
                {
                    var window = new Queue<NDArray>();
                    for (int i = 0; i < 200; i++)
                    {
                        NDArray view = ViewOf($"lease_conc_{t}");
                        if (!Holds(view, t + 1))
                            Interlocked.Increment(ref wrong);
                        window.Enqueue(view);
                        if (window.Count > 8)
                            LetGo(window.Dequeue(), t + 1, i, crossThread, ref wrong);
                    }
                    while (window.Count > 0)
                        LetGo(window.Dequeue(), t + 1, 0, crossThread, ref wrong);
                }
                catch (Exception e)
                {
                    failures.Enqueue(e);
                }
            }

            var workers = new Thread[Threads];
            for (int t = 0; t < Threads; t++)
            {
                int id = t;
                workers[t] = new Thread(() => Worker(id)) { IsBackground = true };
                workers[t].Start();
            }
            foreach (Thread worker in workers)
                worker.Join(120_000).Should().BeTrue("workers contending for the GIL must not deadlock");
            stop.Cancel();
            collector.Join();
            while (crossThread.TryDequeue(out NDArray rest))
                rest.Dispose();

            failures.Should().BeEmpty(string.Join(" | ", failures));
            wrong.Should().Be(0, "every view reads its own array for as long as it lives");
            PyExec(string.Join("; ", Enumerable.Range(0, Threads).Select(t => $"del lease_conc_{t}")));
            WaitFor(() => SizeBucketedBufferPool.OutstandingForeignBytes == foreignBefore, 20_000)
                .Should().BeTrue($"every lease gives its booking back (foreign {SizeBucketedBufferPool.OutstandingForeignBytes} vs {foreignBefore})");
        }

        /// <summary>Does <paramref name="view"/> hold <paramref name="value"/> at its first, middle and last element?</summary>
        /// <param name="view">A 4,096-element float32 view.</param>
        /// <param name="value">The value its numpy array holds.</param>
        /// <returns><c>true</c> when all three elements hold the value.</returns>
        private static bool Holds(NDArray view, float value) =>
            view.GetSingle(0) == value && view.GetSingle(2048) == value && view.GetSingle(4095) == value;

        /// <summary>
        ///     Checks a view's values one last time, then disposes it here, hands it to another thread to dispose, or drops it
        ///     (the third way leaves it to the finalizer; every way ends in the lease's queued release).
        /// </summary>
        /// <param name="view">The view.</param>
        /// <param name="value">The value its numpy array holds.</param>
        /// <param name="i">The iteration, which picks the way.</param>
        /// <param name="crossThread">The queue other threads dispose from.</param>
        /// <param name="wrong">The count of views that read wrong values.</param>
        private static void LetGo(NDArray view, float value, int i, ConcurrentQueue<NDArray> crossThread, ref int wrong)
        {
            if (!Holds(view, value))
                Interlocked.Increment(ref wrong);
            switch (i % 3)
            {
                case 0:
                    view.Dispose();
                    break;
                case 1:
                    crossThread.Enqueue(view);
                    if (crossThread.TryDequeue(out NDArray other))
                        other.Dispose();
                    break;
                default:
                    break;   // dropped: the finalizer queues the lease's release, which a later drain runs under the GIL
            }
        }
    }
}
