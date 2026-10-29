using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     Concurrency stress on the reference counting the bridge relies on: handles disposed by many threads at once, pins
    ///     racing the handle's dispose, one session run from many threads over one shared input, and many views of one
    ///     zero-copy import disposed at once. Every race must end with every reference released exactly once — the source
    ///     uniquely referenced, never released early — and the live counters back at their baseline.
    /// </summary>
    [TestClass]
    public class ConcurrencyStressTests : OnnxTestBase
    {
        /// <summary>Rounds per race.</summary>
        private const int Rounds = 64;

        /// <summary>Threads per race.</summary>
        private const int Threads = 8;

        /// <summary>
        ///     One export handle disposed by eight threads at once, 64 rounds per verb: the handle's reference is released
        ///     exactly once (a second release would free the buffer under its owner) and the source ends uniquely referenced.
        /// </summary>
        [TestMethod]
        public void OneHandle_DisposedByManyThreadsAtOnce_ReleasesOnce()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 16);
            IArraySlice slice = nd.Storage.InternalArray;
            for (int round = 0; round < Rounds; round++)
            {
                IDisposable h = round % 2 == 0 ? nd.AsOrtValue() : nd.AsDenseTensor<float>();
                using var start = new Barrier(Threads);
                Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, _ =>
                {
                    start.SignalAndWait();
                    h.Dispose();
                });
                Assert.IsTrue(slice.IsUniquelyReferenced, $"round {round}: the reference was not released");
                Assert.IsFalse(slice.IsReleased, $"round {round}: the owner's own reference was released too");
            }
        }

        /// <summary>
        ///     Pins of an exported <see cref="Memory{T}"/> race the handle's dispose: each pin either succeeds — and then reads
        ///     the right data for as long as it lives, whatever the handle does meanwhile — or is refused with
        ///     <see cref="ObjectDisposedException"/>; once every pin is released the source is uniquely referenced again.
        /// </summary>
        [TestMethod]
        public void MemoryPins_RacingTheHandlesDispose_StayValidAndBalance()
        {
            using NDArray nd = Arange(NPTypeCode.Double, 64);
            IArraySlice slice = nd.Storage.InternalArray;
            for (int round = 0; round < Rounds; round++)
            {
                OrtTensor<double> h = nd.AsDenseTensor<double>();
                Memory<double> memory = h.Memory;
                var pins = new ConcurrentBag<MemoryHandle>();
                using var start = new Barrier(Threads);
                Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, t =>
                {
                    start.SignalAndWait();
                    if (t == 0)
                    {
                        h.Dispose();
                        return;
                    }
                    try
                    {
                        pins.Add(memory.Pin());
                    }
                    catch (ObjectDisposedException)
                    {
                        // refused: the handle won the race
                    }
                });
                foreach (MemoryHandle pin in pins)
                {
                    unsafe
                    {
                        Assert.AreEqual(63.0, ((double*)pin.Pointer)[63], $"round {round}: a pin read the wrong data");
                    }
                    pin.Dispose();
                }
                Assert.IsTrue(slice.IsUniquelyReferenced, $"round {round}: a pin's reference was not released");
                Assert.IsFalse(slice.IsReleased, $"round {round}: the owner's own reference was released");
            }
        }

        /// <summary>
        ///     One session run from eight threads at once over ONE shared strided input (each run densifies it into its own
        ///     temporary) gives every thread the right output, and every export and temporary is released.
        /// </summary>
        [TestMethod]
        public void OneSession_RunFromManyThreads_OverOneSharedInput()
        {
            using InferenceSession session = OpenSession("identity_float64");
            using NDArray input = Arange(NPTypeCode.Double, 40, 30)["::2, ::-3"];
            using NDArray want = input.copy();
            var failures = new ConcurrentQueue<string>();
            Parallel.For(0, Threads * 8, new ParallelOptions { MaxDegreeOfParallelism = Threads }, i =>
            {
                using NDArray y = session.Run(input);
                if (!np.array_equal(y, want))
                    failures.Enqueue($"run {i} returned the wrong output");
            });
            Assert.IsTrue(failures.IsEmpty, string.Join("; ", failures));
            Assert.IsTrue(input.Storage.InternalArray.IsUniquelyReferenced || !input.Storage.InternalArray.IsReleased, "the shared input was released");
        }

        /// <summary>
        ///     Many NumSharp views of one owning zero-copy import, disposed by many threads at once: the import's lease is
        ///     released exactly once (the OrtValue disposed once — a second dispose of an ORT value is a native fault), after
        ///     the LAST view, and the live-import counter returns to its baseline.
        /// </summary>
        [TestMethod]
        public void ViewsOfOneImport_DisposedByManyThreadsAtOnce_ReleaseTheLeaseOnce()
        {
            int baseline = NDArrayOnnxInterop.LiveImports;
            for (int round = 0; round < Rounds; round++)
            {
                OrtValue value = OrtValue.CreateTensorValueFromMemory(Enumerable.Range(0, 32).Select(i => (float)i).ToArray(), new long[] { 4, 8 });
                NDArray root = value.AsNDArray(ownsValue: true);
                NDArray[] views = Enumerable.Range(0, Threads).Select(i => root[$"{i % 4}"]).ToArray();
                using var start = new Barrier(Threads + 1);
                var tasks = views.Select(v => Task.Run(() =>
                {
                    start.SignalAndWait();
                    v.Dispose();
                })).ToArray();
                start.SignalAndWait();
                root.Dispose();
                Task.WaitAll(tasks);
                Assert.IsTrue(WaitFor(() => NDArrayOnnxInterop.LiveImports <= baseline, 10_000), $"round {round}: the lease was not released");
            }
        }
    }
}
