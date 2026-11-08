using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics.Tensors;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Concurrency stress. The bridge's lifetime bookkeeping is lock-free by design — the export handle's dispose
    ///     guard, the import lease's release guard, the <see cref="NDArrayTensorsInterop.LiveExports"/> /
    ///     <see cref="NDArrayTensorsInterop.LiveImports"/> counters and NumSharp's ARC reference count are all
    ///     <c>Interlocked</c> — so these tests race them from many threads and check every count lands EXACTLY: a lost
    ///     release leaks a pinned buffer, a doubled one frees memory still in use.
    /// </summary>
    [TestClass]
    public class ConcurrencyTests : TensorsTestBase
    {
        /// <summary>Worker threads per race (more than most CI runners have cores, so the scheduler interleaves them).</summary>
        private const int Workers = 8;

        /// <summary>
        ///     Eight workers each export, copy and import shared data hundreds of times concurrently — a transposed
        ///     view of one shared owner through AsTensorSpan, the owner through ToTensor, a shared tensor through
        ///     AsNDArray with a derived slice — reading known elements every time. Every read is right, and afterwards
        ///     both counters are back at their baseline and the owner's buffer is referenced by the owner alone.
        /// </summary>
        [TestMethod]
        public void ParallelExportsCopiesAndImports_ReadTheData_AndEveryCountSettles()
        {
            const int iterations = 300;
            using NDArray owner = np.arange(24.0);
            Tensor<double> shared = Tensor.Create(Enumerable.Range(0, 24).Select(i => (double)i).ToArray(), new nint[] { 4, 6 });
            int exports = NDArrayTensorsInterop.LiveExports;
            int imports = NDArrayTensorsInterop.LiveImports;
            var errors = new ConcurrentQueue<string>();

            Parallel.For(0, Workers, new ParallelOptions { MaxDegreeOfParallelism = Workers }, worker =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    using (NDArray m = owner.reshape(4, 6))
                    using (NDArray transposed = m.T)
                    using (var h = transposed.AsTensorSpan<double>())
                    {
                        double seen = h.ReadOnlySpan[new nint[] { 5, 3 }];
                        if (seen != 3 * 6 + 5)
                            errors.Enqueue($"worker {worker}: transposed span[5,3] = {seen}");
                    }

                    Tensor<double> copy = owner.ToTensor<double>();
                    if (copy[new nint[] { 23 }] != 23)
                        errors.Enqueue($"worker {worker}: ToTensor()[23] = {copy[new nint[] { 23 }]}");

                    using (NDArray view = shared.AsNDArray())
                    using (NDArray rows = view["1:3"])
                    {
                        double seen = rows.GetDouble(1, 5);
                        if (seen != 2 * 6 + 5)
                            errors.Enqueue($"worker {worker}: imported rows[1,5] = {seen}");
                    }
                }
            });

            errors.Should().BeEmpty();
            NDArrayTensorsInterop.LiveExports.Should().Be(exports, "every handle was disposed exactly once");
            NDArrayTensorsInterop.LiveImports.Should().Be(imports, "every lease was released exactly once");
            owner.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("every export reference and view reference was released");
        }

        /// <summary>
        ///     Many threads dispose the SAME export handle at the same instant (released by a barrier), over and over:
        ///     the handle releases its buffer reference exactly once — the live-export count drops by one, the owner is
        ///     referenced by itself alone again, and its buffer was not freed underneath it (its values still read).
        /// </summary>
        [TestMethod]
        public void ConcurrentDispose_OfOneHandle_ReleasesExactlyOnce()
        {
            using NDArray owner = np.arange(8.0);
            for (int round = 0; round < 64; round++)
            {
                int before = NDArrayTensorsInterop.LiveExports;
                var h = owner.AsTensorSpan<double>();
                RaceAll(Workers, h.Dispose);

                h.IsDisposed.Should().BeTrue();
                NDArrayTensorsInterop.LiveExports.Should().Be(before, $"round {round}: the reference is released once, not once per thread");
                owner.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue($"round {round}");
                owner.GetDouble(7).Should().Be(7.0, $"round {round}: the buffer must not have been freed by an extra release");
            }
        }

        /// <summary>
        ///     Every NumSharp view sharing ONE import lease — the imported view and three views derived from it — is
        ///     disposed from its own thread at the same instant: the lease is released exactly once, when the last of
        ///     them goes (the live-import count returns to its baseline, not below).
        /// </summary>
        [TestMethod]
        public void ConcurrentDisposal_OfViewsSharingOneLease_ReleasesItOnce()
        {
            for (int round = 0; round < 64; round++)
            {
                int before = NDArrayTensorsInterop.LiveImports;
                Tensor<double> t = Tensor.Create(Enumerable.Range(0, 12).Select(i => (double)i).ToArray(), new nint[] { 3, 4 });
                NDArray view = t.AsNDArray();
                NDArray[] sharing = { view, view["1:"], view.T, view["::2, 1::2"] };
                NDArrayTensorsInterop.LiveImports.Should().Be(before + 1, "derived views share the one lease");

                int next = -1;
                RaceAll(sharing.Length, () => sharing[Interlocked.Increment(ref next)].Dispose());

                NDArrayTensorsInterop.LiveImports.Should().Be(before, $"round {round}: the lease is released exactly once");
            }
        }

        /// <summary>
        ///     The import lease's release runs its action exactly once however many times, and from however many threads
        ///     at once, it is called; the live-import count drops by exactly one.
        /// </summary>
        /// <remarks>
        ///     Normal operation releases a lease once (NumSharp's memory block runs its disposer once), so no end-to-end
        ///     test reaches a second release — mutation testing showed the once-only guard could be deleted with every
        ///     other test still green. A FAULT path does release twice: if anything throws after the tensor's memory was
        ///     wrapped, <c>AsNDArray</c>'s catch releases the lease, and the orphaned memory block's finalizer releases it
        ///     again. Without the guard that frees the pinned <c>GCHandle</c> twice — an exception on the finalizer
        ///     thread, which ends the process — and drives the live-import count negative. Hence this direct test of the
        ///     lease's contract.
        /// </remarks>
        [TestMethod]
        public void ImportLease_Release_RunsItsActionExactlyOnce_EvenWhenRaced()
        {
            for (int round = 0; round < 32; round++)
            {
                int before = NDArrayTensorsInterop.LiveImports;
                int runs = 0;
                var lease = new ImportLease(() => Interlocked.Increment(ref runs), 64);
                NDArrayTensorsInterop.LiveImports.Should().Be(before + 1);

                RaceAll(Workers, lease.Release);
                lease.Release();

                runs.Should().Be(1, $"round {round}: the release action must run exactly once");
                lease.IsReleased.Should().BeTrue();
                NDArrayTensorsInterop.LiveImports.Should().Be(before, $"round {round}: the count drops by exactly one");
            }
        }

        /// <summary>
        ///     Runs <paramref name="action"/> on <paramref name="threads"/> dedicated threads released together by a
        ///     barrier, so the calls overlap as closely as the scheduler allows, and waits for all of them.
        /// </summary>
        /// <param name="threads">How many threads (and calls).</param>
        /// <param name="action">The call each thread makes once.</param>
        /// <exception cref="AggregateException">One or more calls threw; their exceptions are inside.</exception>
        private static void RaceAll(int threads, Action action)
        {
            using var barrier = new Barrier(threads);
            var thrown = new ConcurrentQueue<Exception>();
            Thread[] workers = Enumerable.Range(0, threads).Select(_ => new Thread(() =>
            {
                barrier.SignalAndWait();
                try
                {
                    action();
                }
                catch (Exception e)
                {
                    thrown.Enqueue(e);
                }
            })).ToArray();
            foreach (Thread w in workers)
                w.Start();
            foreach (Thread w in workers)
                w.Join();
            if (!thrown.IsEmpty)
                throw new AggregateException(thrown);
        }
    }
}
