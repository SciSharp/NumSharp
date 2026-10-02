using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Concurrency stress on the reference counting the data view relies on: one view disposed by many threads at once,
    ///     cursors still reading on other threads while their view is disposed, and one view imported from many threads.
    ///     Every race must end with every reference released exactly once and the live counter back at its baseline.
    /// </summary>
    [TestClass]
    public class ConcurrencyStressTests : MLNetTestBase
    {
        /// <summary>Rounds per race.</summary>
        private const int Rounds = 64;

        /// <summary>Threads per race.</summary>
        private const int Threads = 8;

        /// <summary>
        ///     One view (zero-copy, converted and snapshot) disposed by eight threads at once, 64 rounds: its reference is
        ///     released exactly once and the source ends uniquely referenced, never released.
        /// </summary>
        [TestMethod]
        public void OneView_DisposedByManyThreadsAtOnce_ReleasesOnce()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 6, 4);
            IArraySlice slice = nd.Storage.InternalArray;
            for (int round = 0; round < Rounds; round++)
            {
                NDArrayDataView view = (round % 3) switch
                {
                    0 => nd.AsDataView("F"),
                    1 => nd.AsDataView(new[] { "a", "b", "c", "d" }),
                    _ => nd.ToDataView("F"),
                };
                using var start = new Barrier(Threads);
                Parallel.For(0, Threads, new ParallelOptions { MaxDegreeOfParallelism = Threads }, _ =>
                {
                    start.SignalAndWait();
                    view.Dispose();
                });
                Assert.IsTrue(slice.IsUniquelyReferenced, $"round {round}: the view's reference was not released");
                Assert.IsFalse(slice.IsReleased, $"round {round}: the owner's own reference was released too");
            }
        }

        /// <summary>
        ///     Eight cursors read the same view on eight threads while the main thread disposes the view (and the source) in the
        ///     middle: every cursor still reads every row correctly — its own pin keeps the buffer alive — and once the cursors
        ///     are disposed every reference is gone.
        /// </summary>
        [TestMethod]
        public void Cursors_KeepReadingOnOtherThreads_WhileTheViewIsDisposed()
        {
            for (int round = 0; round < 16; round++)
            {
                NDArray nd = Arange(NPTypeCode.Double, 200, 5);
                double[] want = Oracle.Elements<double>(nd);
                NDArrayDataView view = nd.AsDataView("F");
                DataViewRowCursor[] cursors = Enumerable.Range(0, Threads).Select(_ => view.GetRowCursor(view.Schema)).ToArray();
                var failures = new ConcurrentQueue<string>();
                using var halfway = new CountdownEvent(Threads);
                Task[] readers = cursors.Select((cursor, t) => Task.Run(() =>
                {
                    using (cursor)
                    {
                        ValueGetter<VBuffer<double>> g = cursor.GetGetter<VBuffer<double>>(cursor.Schema["F"]);
                        var buffer = default(VBuffer<double>);
                        long row = 0;
                        while (cursor.MoveNext())
                        {
                            if (row == 100)
                                halfway.Signal();
                            g(ref buffer);
                            ReadOnlySpan<double> values = buffer.GetValues();
                            for (int j = 0; j < 5; j++)
                                if (values[j] != want[row * 5 + j])
                                    failures.Enqueue($"thread {t}: row {row} column {j} read {values[j]}");
                            row++;
                        }
                        if (row != 200)
                            failures.Enqueue($"thread {t}: read {row} rows");
                    }
                })).ToArray();
                halfway.Wait();
                view.Dispose();
                nd.Dispose();
                Task.WaitAll(readers);
                Assert.IsTrue(failures.IsEmpty, $"round {round}: {string.Join("; ", failures.Take(5))}");
                Assert.IsTrue(nd.Storage.InternalArray.IsReleased, $"round {round}: a reference outlived every cursor");
            }
        }

        /// <summary>One view imported (materialized) from eight threads at once, each with its own cursor: every result is right.</summary>
        [TestMethod]
        public void OneView_ImportedFromManyThreads_GivesEveryThreadTheRightArray()
        {
            using NDArray nd = Arange(NPTypeCode.Int32, 50, 9)["::-1, 1::2"];
            using NDArray want = nd.copy();
            using NDArrayDataView vectors = nd.AsDataView("F");
            using NDArrayDataView scalars = nd.AsDataView(new[] { "a", "b", "c", "d" });
            var failures = new ConcurrentQueue<string>();
            Parallel.For(0, Threads * 8, new ParallelOptions { MaxDegreeOfParallelism = Threads }, i =>
            {
                using NDArray v = vectors.ToNDArray("F");
                if (!np.array_equal(v, want))
                    failures.Enqueue($"import {i}: the vector column differs");
                using NDArray s = scalars.ToNDArray("c");
                using NDArray column = want[":, 2"];
                if (!np.array_equal(s, column))
                    failures.Enqueue($"import {i}: the scalar column differs");
            });
            Assert.IsTrue(failures.IsEmpty, string.Join("; ", failures.Take(5)));
        }
    }
}
