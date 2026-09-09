using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>The <see cref="DataViewRowCursor"/> contract ML.NET relies on when it walks the view.</summary>
    [TestClass]
    public class CursorContractTests : MLNetTestBase
    {
        [TestMethod]
        public void Position_Batch_And_MoveNext()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["x"] });

            Assert.AreEqual(-1L, cur.Position, "before the first MoveNext");
            Assert.AreEqual(0L, cur.Batch);
            long expected = 0;
            while (cur.MoveNext())
                Assert.AreEqual(expected++, cur.Position);
            Assert.AreEqual(3L, cur.Position, "parked at rowCount past the end");
            Assert.IsFalse(cur.MoveNext(), "MoveNext stays false");
            Assert.IsFalse(cur.MoveNext());
        }

        [TestMethod]
        public void GetIdGetter_IsSequential()
        {
            using NDArray a = Arange(NPTypeCode.Single, 4);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["v"] });
            ValueGetter<DataViewRowId> idg = cur.GetIdGetter();
            long i = 0;
            DataViewRowId id = default;
            while (cur.MoveNext())
            {
                idg(ref id);
                Assert.AreEqual(new DataViewRowId((ulong)i, 0), id, $"id at row {i}");
                i++;
            }
        }

        [TestMethod]
        public void IsColumnActive_OnlyRequestedColumns()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y", "z" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["y"] });
            Assert.IsFalse(cur.IsColumnActive(dv.Schema["x"]));
            Assert.IsTrue(cur.IsColumnActive(dv.Schema["y"]));
            Assert.IsFalse(cur.IsColumnActive(dv.Schema["z"]));
        }

        [TestMethod]
        public void InactiveColumnGetter_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["x"] });
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => cur.GetGetter<float>(dv.Schema["y"]));
        }

        [TestMethod]
        public void NullColumnsNeeded_AllInactive()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using DataViewRowCursor cur = dv.GetRowCursor(null);
            Assert.IsFalse(cur.IsColumnActive(dv.Schema["x"]));
            Assert.IsFalse(cur.IsColumnActive(dv.Schema["y"]));
        }

        [TestMethod]
        public void GetGetter_WrongT_Scalar_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["x"] });
            Assert.ThrowsException<InvalidOperationException>(() => cur.GetGetter<double>(dv.Schema["x"]));
        }

        [TestMethod]
        public void GetGetter_WrongT_Vector_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArrayDataView dv = a.AsDataView("F");
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["F"] });
            Assert.ThrowsException<InvalidOperationException>(() => cur.GetGetter<VBuffer<double>>(dv.Schema["F"]));
        }

        [TestMethod]
        public void CursorSchema_MatchesView()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["x"] });
            Assert.AreEqual(dv.Schema.Count, cur.Schema.Count);
            Assert.AreEqual("x", cur.Schema[0].Name);
        }

        [TestMethod]
        public void GetRowCursorSet_ReturnsWorkingCursors()
        {
            using NDArray a = Arange(NPTypeCode.Single, 4);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            DataViewRowCursor[] set = dv.GetRowCursorSet(new[] { dv.Schema["v"] }, 3);
            Assert.IsTrue(set.Length >= 1);
            foreach (DataViewRowCursor c in set)
            {
                int n = 0;
                while (c.MoveNext()) n++;
                Assert.AreEqual(4, n);
                c.Dispose();
            }
        }

        [TestMethod]
        public void TwoConcurrentCursors_ReadIndependently()
        {
            using NDArray a = Arange(NPTypeCode.Int32, 5);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            DataViewSchema.Column col = dv.Schema["v"];
            using DataViewRowCursor c1 = dv.GetRowCursor(new[] { col });
            using DataViewRowCursor c2 = dv.GetRowCursor(new[] { col });
            ValueGetter<int> g1 = c1.GetGetter<int>(col);
            ValueGetter<int> g2 = c2.GetGetter<int>(col);

            c1.MoveNext(); c1.MoveNext();   // c1 at row 1
            c2.MoveNext();                  // c2 at row 0
            int v1 = 0, v2 = 0;
            g1(ref v1); g2(ref v2);
            Assert.AreEqual(1, v1);
            Assert.AreEqual(0, v2);
        }

        [TestMethod]
        public void VectorGetter_ReusesBuffer_AcrossRows()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 2);   // rows [0,1],[2,3],[4,5]
            using NDArrayDataView dv = a.AsDataView("F");
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { dv.Schema["F"] });
            ValueGetter<VBuffer<float>> g = cur.GetGetter<VBuffer<float>>(dv.Schema["F"]);
            var buf = default(VBuffer<float>);
            cur.MoveNext(); g(ref buf);
            CollectionAssert.AreEqual(new float[] { 0, 1 }, buf.DenseValues().ToArray());
            cur.MoveNext(); g(ref buf);   // same buffer reused
            CollectionAssert.AreEqual(new float[] { 2, 3 }, buf.DenseValues().ToArray());
            cur.MoveNext(); g(ref buf);
            CollectionAssert.AreEqual(new float[] { 4, 5 }, buf.DenseValues().ToArray());
        }
    }
}
