using System;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Thread-safety of the lazy view: the pin bookkeeping is lock-free, and a read-only shared buffer is safe
    ///     for any number of concurrent cursors. Every test drives real parallelism and asserts the LiveExports
    ///     counter returns to its baseline (the leak gate under contention).
    /// </summary>
    [TestClass]
    public class ConcurrencyTests : MLNetTestBase
    {
        [TestMethod]
        public void ParallelViewsOnOneSharedBuffer_EachHoldsItsOwnPin_AllReleased()
        {
            using NDArray a = Arange(NPTypeCode.Double, 100, 8);
            int baseline = NDArrayMLNetInterop.LiveExports;
            Parallel.For(0, 128, _ =>
            {
                using NDArrayDataView dv = a.AsDataView("F");
                using NDArray back = dv.ToNDArray("F");
                Assert.AreEqual(800L, back.size);
                Assert.IsTrue(np.array_equal(a, back));
            });
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveExports <= baseline, 10_000),
                $"pins leaked: {baseline} -> {NDArrayMLNetInterop.LiveExports}");
        }

        [TestMethod]
        public void ParallelViewsOnDistinctBuffers_SettleToZero()
        {
            int baseline = NDArrayMLNetInterop.LiveExports;
            Parallel.For(0, 128, i =>
            {
                using NDArray a = Arange(NPTypeCode.Single, 3, 4);
                using NDArrayDataView dv = a.AsDataView(new[] { "a", "b", "c", "d" });
                using NDArray col = dv.ToNDArray("c");
                Assert.AreEqual(3L, col.size);
            });
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveExports <= baseline, 10_000));
        }

        [TestMethod]
        public void ManyConcurrentCursors_OverOneView_ReadTheSameValues()
        {
            using NDArray a = np.arange(64).astype(NPTypeCode.Int32).reshape(64, 1);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            DataViewSchema.Column col = dv.Schema["v"];

            Parallel.For(0, 64, _ =>
            {
                using DataViewRowCursor cur = dv.GetRowCursor(new[] { col });
                ValueGetter<int> g = cur.GetGetter<int>(col);
                long sum = 0;
                int v = 0;
                while (cur.MoveNext())
                {
                    g(ref v);
                    sum += v;
                }
                Assert.AreEqual(64L * 63L / 2L, sum, "each thread sees the full 0..63 sequence");
            });
        }

        [TestMethod]
        public void ParallelMixedExportsAndMaterializations_SettleToZero()
        {
            using NDArray shared = Arange(NPTypeCode.Double, 50, 4);
            int baseline = NDArrayMLNetInterop.LiveExports;
            Parallel.For(0, 96, i =>
            {
                switch (i % 3)
                {
                    case 0:
                        using (NDArrayDataView dv = shared.AsDataView("F"))
                        using (NDArray _ = dv.ToNDArray("F")) { }
                        break;
                    case 1:
                        using (NDArrayDataView dv = shared.ToDataView(new[] { "a", "b", "c", "d" }))
                        using (NDArray _ = dv.ToNDArray("b")) { }
                        break;
                    default:
                        VBuffer<double> vb = shared["0"].ToVBuffer<double>();   // a row -> VBuffer (copy)
                        using (NDArray _ = vb.ToNDArray()) { }
                        break;
                }
            });
            Assert.IsTrue(WaitFor(() => NDArrayMLNetInterop.LiveExports <= baseline, 10_000),
                $"mixed workload leaked: {baseline} -> {NDArrayMLNetInterop.LiveExports}");
        }
    }
}
