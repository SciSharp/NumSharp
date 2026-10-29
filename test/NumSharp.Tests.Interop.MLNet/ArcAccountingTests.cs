using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>
    ///     Reference accounting and fault injection: every path through the bridge — success, refusal, failure half-way,
    ///     an abandoned cursor — must leave NumSharp's buffer references and its buffer pool exactly balanced. Each check
    ///     reads the counted state directly (<see cref="IArraySlice.IsUniquelyReferenced"/>) or the pool's take/return
    ///     balance (<see cref="Pool.Balance"/>).
    /// </summary>
    [TestClass]
    public class ArcAccountingTests : MLNetTestBase
    {
        /// <summary>A POCO row with a variable-length vector, the input of the import fault-injection test.</summary>
        private sealed class RaggedRow
        {
            /// <summary>A vector whose length differs between rows.</summary>
            [VectorType]
            public float[] V { get; set; }
        }

        /// <summary>
        ///     Every refusal leaves the source exactly as referenced as before and gives back every buffer it took: the
        ///     refusals the shape, dtype, names and size decide come before any conversion or snapshot, so none of them may
        ///     allocate (a 2 GB snapshot taken only to be refused was the old behaviour of <c>ToDataView</c>).
        /// </summary>
        [TestMethod]
        public void Refusals_LeaveTheSourceUniquelyReferenced_AndThePoolBalanced()
        {
            var cases = new (string Name, NPTypeCode Code, int[] Shape, Func<NDArray, NDArray> MakeView, Action<NDArray> Refuse)[]
            {
                ("AsDataView(column) of a 3-D view", NPTypeCode.Single, new[] { 2, 3, 4 }, b => b.T, v => v.AsDataView("F")),
                ("AsDataView(column) of a 3-D Half (no conversion)", NPTypeCode.Half, new[] { 2, 3, 4 }, b => b.T, v => v.AsDataView("F")),
                ("ToDataView(column) of a 3-D Decimal (no snapshot)", NPTypeCode.Decimal, new[] { 2, 3, 4 }, b => b.T, v => v.ToDataView("F")),
                ("AsDataView(columns) with the wrong name count", NPTypeCode.Double, new[] { 4, 3 }, b => b.T, v => v.AsDataView(new[] { "a" })),
                ("ToDataView(columns) with an empty name", NPTypeCode.Half, new[] { 4, 2 }, b => b.T, v => v.ToDataView(new[] { "a", "" })),
                ("ToDataView of Complex", NPTypeCode.Complex, new[] { 4, 3 }, b => b["::2"], v => v.ToDataView("F")),
                ("ToVBuffer with the wrong T", NPTypeCode.Half, new[] { 4, 3 }, b => b.T, v => v.ToVBuffer<Half>()),
                ("ToVBuffer of Complex", NPTypeCode.Complex, new[] { 4, 3 }, b => b.T, v => v.ToVBuffer<double>()),
            };
            foreach (var (name, code, shape, makeView, refuse) in cases)
            {
                long balance = Pool.Balance(() =>
                {
                    using NDArray b = Arange(code, shape);
                    NDArray view = makeView(b);
                    bool threw = false;
                    try
                    {
                        refuse(view);
                    }
                    catch (Exception)
                    {
                        threw = true;
                    }
                    Assert.IsTrue(threw, $"{name}: expected a refusal");
                    view.Dispose();
                    Assert.IsTrue(b.Storage.InternalArray.IsUniquelyReferenced, $"{name}: the refusal kept a reference on the source");
                });
                Assert.AreEqual(0L, balance, $"{name}: {balance} buffer(s) taken and never returned");
            }
        }

        /// <summary>
        ///     A view holds exactly one reference, each cursor one more, and each is released exactly once — disposing twice
        ///     changes nothing, and once view and cursors are gone the source is uniquely referenced again.
        /// </summary>
        [TestMethod]
        public void ViewsAndCursors_HoldOneReferenceEach_ReleasedExactlyOnce()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 3, 2);
            IArraySlice slice = nd.Storage.InternalArray;
            NDArrayDataView view = nd.AsDataView("F");
            Assert.IsFalse(slice.IsUniquelyReferenced, "the view holds no reference");
            DataViewRowCursor first = view.GetRowCursor(view.Schema);
            DataViewRowCursor second = view.GetRowCursor(view.Schema);
            view.Dispose();
            view.Dispose();
            Assert.IsFalse(slice.IsUniquelyReferenced, "the cursors hold no reference of their own");
            first.Dispose();
            first.Dispose();
            Assert.IsFalse(slice.IsUniquelyReferenced, "the second cursor's reference went with the first");
            second.Dispose();
            Assert.IsTrue(slice.IsUniquelyReferenced, "a reference was never released");
            Assert.IsFalse(slice.IsReleased, "a double dispose released the owner's own reference");
        }

        /// <summary>
        ///     The conversions and snapshots a view owns (Half → Single, Decimal → Double, <c>ToDataView</c>'s copy) are
        ///     released with the view: each export, once the view is disposed, gives back every buffer it took.
        /// </summary>
        [TestMethod]
        public void OwnedTemporaries_AreReleasedWithTheView()
        {
            var exports = new (string Name, NPTypeCode Code, Func<NDArray, NDArrayDataView> Export)[]
            {
                ("AsDataView of Half (converted)", NPTypeCode.Half, nd => nd.AsDataView("F")),
                ("AsDataView of Decimal (converted)", NPTypeCode.Decimal, nd => nd.AsDataView(new[] { "a", "b", "c" })),
                ("ToDataView of float32 (snapshot)", NPTypeCode.Single, nd => nd.ToDataView("F")),
                ("ToDataView of Half (converted snapshot)", NPTypeCode.Half, nd => nd.ToDataView(new[] { "a", "b", "c" })),
            };
            foreach (var (name, code, export) in exports)
            {
                long balance = Pool.Balance(() =>
                {
                    using NDArray nd = Arange(code, 4, 3);
                    using NDArrayDataView view = export(nd);
                    Assert.AreEqual(4L, view.GetRowCount(), name);
                });
                Assert.AreEqual(0L, balance, $"{name}: {balance} buffer(s) never returned");
            }
        }

        /// <summary>
        ///     The import stages rows in NumSharp memory and gives back everything but its result: through a view that reports
        ///     its row count (one chunk that becomes the result), through one that does not (chunks copied together), and —
        ///     fault injection — through a column whose rows disagree on length, where the import fails half-way and must free
        ///     what it staged.
        /// </summary>
        [TestMethod]
        public void Import_GivesBackEverythingButItsResult_EvenWhenItFailsHalfWay()
        {
            Assert.AreEqual(0L, Pool.Balance(() =>
            {
                using NDArray nd = Arange(NPTypeCode.Double, 5000, 3);
                using NDArrayDataView view = nd.AsDataView("F");
                using NDArray back = view.ToNDArray("F");
                Assert.AreEqual(15000L, back.size);
            }), "known row count");

            Assert.AreEqual(0L, Pool.Balance(() =>
            {
                using NDArray values = np.arange(9000.0);
                using NDArrayDataView view = values.AsDataView(new[] { "x" });
                IDataView filtered = Ml.Data.FilterRowsByColumn(view, "x", 0, 1e9);
                using NDArray back = filtered.ToNDArray("x");
                Assert.AreEqual(9000L, back.size);
            }), "unknown row count (chunks)");

            var ragged = new List<RaggedRow>();
            for (int r = 0; r < 6000; r++)
                ragged.Add(new RaggedRow { V = new float[r < 5000 ? 3 : 4] });
            IDataView raggedView = Ml.Data.LoadFromEnumerable(ragged);
            Assert.AreEqual(0L, Pool.Balance(() =>
                Assert.ThrowsException<ArgumentException>(() => raggedView.ToNDArray("V"))), "the failed import kept its staged chunks");
        }

        /// <summary>
        ///     A cursor nobody disposes releases its reference when it is collected. ML.NET's own <c>ShuffleRows</c> abandons
        ///     the cursor it opens on its input, and before the cursor had a finalizer that one shuffle kept the NumSharp
        ///     buffer referenced for the life of the process (and <c>resize(refcheck)</c> refusing). Checked directly (a
        ///     cursor dropped undisposed) and through the real shuffle.
        /// </summary>
        [TestMethod]
        public void AbandonedCursors_AreReleasedByTheFinalizer()
        {
            using NDArray nd = Arange(NPTypeCode.Single, 6, 4);
            IArraySlice slice = nd.Storage.InternalArray;

            AbandonACursor(nd);
            Assert.IsTrue(WaitFor(() => slice.IsUniquelyReferenced, 10_000), "an abandoned cursor kept its reference");

            ShuffleAndRead(nd);
            Assert.IsTrue(WaitFor(() => slice.IsUniquelyReferenced, 10_000), "ShuffleRows' abandoned input cursor kept its reference");
            nd.resize(48);   // refcheck: refuses while anything still references the buffer
            Assert.AreEqual(48L, nd.size);
        }

        /// <summary>Opens a cursor over a view of <paramref name="nd"/>, reads one row and drops the cursor undisposed (the view is disposed).</summary>
        /// <param name="nd">The array to view.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void AbandonACursor(NDArray nd)
        {
            using NDArrayDataView view = nd.AsDataView("F");
            DataViewRowCursor cursor = view.GetRowCursor(view.Schema);
            Assert.IsTrue(cursor.MoveNext());
        }

        /// <summary>Shuffles a view of <paramref name="nd"/> with ML.NET and reads every row (disposing everything this code opened).</summary>
        /// <param name="nd">The array to view.</param>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ShuffleAndRead(NDArray nd)
        {
            using NDArrayDataView view = nd.AsDataView("F");
            IDataView shuffled = Ml.Data.ShuffleRows(view, seed: 7);
            Assert.AreEqual(6, Oracle.VectorRows<float>(shuffled, "F").Count);
        }
    }
}
