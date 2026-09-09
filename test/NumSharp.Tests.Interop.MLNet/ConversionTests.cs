using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class ConversionTests : MLNetTestBase
    {
        // ---- AsDataView: vector mode ----------------------------------------------------------------

        [TestMethod]
        public void AsDataView_2D_IsOneVectorColumn_RowsAlongAxis0()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 4);
            using NDArrayDataView dv = a.AsDataView("Features");

            Assert.AreEqual(1, dv.Schema.Count);
            DataViewSchema.Column col = dv.Schema["Features"];
            Assert.IsInstanceOfType(col.Type, typeof(VectorDataViewType));
            Assert.AreEqual(4, ((VectorDataViewType)col.Type).Size);
            Assert.AreEqual(3L, dv.GetRowCount());

            using NDArray back = dv.ToNDArray("Features");
            AssertShape(back, 3, 4);
            Assert.IsTrue(np.array_equal(a, back));
        }

        [TestMethod]
        public void AsDataView_1D_IsSingleRowVector()
        {
            using NDArray a = Arange(NPTypeCode.Double, 5);
            using NDArrayDataView dv = a.AsDataView();  // default name "Features"
            Assert.AreEqual(1L, dv.GetRowCount());
            Assert.AreEqual(5, ((VectorDataViewType)dv.Schema["Features"].Type).Size);

            using NDArray back = dv.ToNDArray("Features");
            AssertShape(back, 1, 5);
            using NDArray flat = back.reshape(5);
            Assert.IsTrue(np.array_equal(a, flat));
        }

        // ---- AsDataView: scalar-columns mode --------------------------------------------------------

        [TestMethod]
        public void AsDataView_ScalarColumns_2D_OneColumnPerFeature()
        {
            using NDArray a = Arange(NPTypeCode.Int32, 3, 2);   // [[0,1],[2,3],[4,5]]
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y" });

            Assert.AreEqual(2, dv.Schema.Count);
            Assert.AreEqual(NumberDataViewType.Int32, dv.Schema["x"].Type);
            Assert.AreEqual(3L, dv.GetRowCount());

            using NDArray x = dv.ToNDArray("x");
            using NDArray y = dv.ToNDArray("y");
            AssertShape(x, 3);
            Assert.AreEqual(0, (int)x[0]); Assert.AreEqual(2, (int)x[1]); Assert.AreEqual(4, (int)x[2]);
            Assert.AreEqual(1, (int)y[0]); Assert.AreEqual(3, (int)y[1]); Assert.AreEqual(5, (int)y[2]);
        }

        [TestMethod]
        public void AsDataView_ScalarColumns_1D_OneColumnManyRows()
        {
            using NDArray a = Arange(NPTypeCode.Int64, 4);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            Assert.AreEqual(1, dv.Schema.Count);
            Assert.AreEqual(4L, dv.GetRowCount());
            using NDArray back = dv.ToNDArray("v");
            AssertShape(back, 4);
            Assert.IsTrue(np.array_equal(a, back));
        }

        // ---- lazy / shared vs snapshot --------------------------------------------------------------

        [TestMethod]
        public void AsDataView_SharesBuffer_SeesLaterMutation()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            a[0, 0] = 999f;   // mutate the shared source AFTER building the view
            using NDArray back = dv.ToNDArray("Features");
            Assert.AreEqual(999f, (float)back[0, 0], "the lazy view reads the live buffer");
        }

        [TestMethod]
        public void ToDataView_IsSnapshot_IndependentOfSource()
        {
            NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.ToDataView("Features");
            a[0, 0] = 999f;   // mutate then dispose the source
            a.Dispose();
            using NDArray back = dv.ToNDArray("Features");
            Assert.AreEqual(0f, (float)back[0, 0], "the snapshot is decoupled from the source");
        }

        // ---- layouts: the any-layout strided read ---------------------------------------------------

        [TestMethod]
        public void AsDataView_ReadsEveryLayout_ThroughStrides()
        {
            using NDArray baseArr = Arange(NPTypeCode.Double, 4, 3);

            AssertLayoutRoundTrips(baseArr.T, "transpose");                 // (3,4), non-contiguous
            AssertLayoutRoundTrips(baseArr["::-1"], "reversed rows");       // negative stride axis 0
            AssertLayoutRoundTrips(baseArr[":, ::2"], "strided cols");      // step 2 on axis 1
            AssertLayoutRoundTrips(baseArr["1:3"], "sliced offset window"); // non-zero offset
            AssertLayoutRoundTrips(np.asfortranarray(baseArr), "fortran");  // F-contiguous
            AssertLayoutRoundTrips(np.broadcast_to(Arange(NPTypeCode.Double, 1, 3), new Shape(4, 3)), "broadcast");
        }

        private static void AssertLayoutRoundTrips(NDArray view, string label)
        {
            using (view)
            {
                using NDArrayDataView dv = view.AsDataView("Features");
                using NDArray back = dv.ToNDArray("Features");
                Assert.IsTrue(np.array_equal(view, back), $"{label}: strided read did not reproduce the logical array");
            }
        }

        // ---- dtypes ---------------------------------------------------------------------------------

        [TestMethod]
        public void AsDataView_RoundTrips_EveryDirectDtype()
        {
            foreach (NPTypeCode code in DirectDtypes)
            {
                using NDArray a = Arange(code, 2, 3);
                using NDArrayDataView dv = a.AsDataView("Features");
                using NDArray back = dv.ToNDArray("Features");
                Assert.AreEqual(code, back.typecode, $"{code} column dtype");
                Assert.IsTrue(np.array_equal(a, back), $"{code} round-trip");
            }
        }

        [TestMethod]
        public void AsDataView_Char_CrossesAsUInt16()
        {
            using NDArray a = np.array(new[] { 'A', 'B', 'C', 'D' }).reshape(2, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            Assert.AreEqual(NumberDataViewType.UInt16, ((VectorDataViewType)dv.Schema["Features"].Type).ItemType);
            using NDArray back = dv.ToNDArray("Features");
            Assert.AreEqual(NPTypeCode.UInt16, back.typecode);   // directional: UInt16, never Char
            Assert.AreEqual((int)'A', (int)back[0, 0]);
            Assert.AreEqual((int)'D', (int)back[1, 1]);
        }

        [TestMethod]
        public void AsDataView_Half_ConvertsToSingle_Lossless()
        {
            using NDArray a = Arange(NPTypeCode.Half, 2, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            Assert.AreEqual(NumberDataViewType.Single, ((VectorDataViewType)dv.Schema["Features"].Type).ItemType);
            using NDArray back = dv.ToNDArray("Features");
            using NDArray expected = a.astype(NPTypeCode.Single);
            Assert.IsTrue(np.array_equal(expected, back));
        }

        [TestMethod]
        public void AsDataView_Complex_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Double, 4).astype(NPTypeCode.Complex);
            Assert.ThrowsException<NotSupportedException>(() => a.AsDataView("Features"));
        }

        // ---- VBuffer <-> NDArray --------------------------------------------------------------------

        [TestMethod]
        public void ToVBuffer_Then_ToNDArray_RoundTrips()
        {
            using NDArray a = Arange(NPTypeCode.Single, 6);
            VBuffer<float> vb = a.ToVBuffer<float>();
            Assert.AreEqual(6, vb.Length);
            Assert.IsTrue(vb.IsDense);
            using NDArray back = vb.ToNDArray();
            Assert.IsTrue(np.array_equal(a, back));
        }

        [TestMethod]
        public void ToVBuffer_ReadsAnyLayoutInLogicalOrder()
        {
            using NDArray a = Arange(NPTypeCode.Int32, 3, 4);
            using NDArray t = a.T;   // non-contiguous
            VBuffer<int> vb = t.ToVBuffer<int>();
            Assert.AreEqual(12, vb.Length);
            using NDArray back = vb.ToNDArray().reshape(4, 3);   // t is (4,3)
            Assert.IsTrue(np.array_equal(t, back), "logical C-order flatten of the transpose");
        }

        [TestMethod]
        public void ToVBuffer_WrongT_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 4);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => a.ToVBuffer<double>());
            StringAssert.Contains(ex.Message, "Single");
        }

        // ---- errors ---------------------------------------------------------------------------------

        [TestMethod]
        public void AsDataView_3D_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2, 2);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => a.AsDataView("Features"));
            StringAssert.Contains(ex.Message, "1-D or 2-D");
        }

        [TestMethod]
        public void AsDataView_ScalarColumns_NameCountMismatch_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 4);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "only", "three", "names" }));
            StringAssert.Contains(ex.Message, "4");
        }

        [TestMethod]
        public void ToNDArray_UnknownColumn_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 2);
            using NDArrayDataView dv = a.AsDataView("Features");
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => dv.ToNDArray("Nope"));
            StringAssert.Contains(ex.Message, "Features");
        }

        [TestMethod]
        public void GetGetter_WrongType_Throws()
        {
            using NDArray a = Arange(NPTypeCode.Single, 2, 3);
            using NDArrayDataView dv = a.AsDataView(new[] { "x", "y", "z" });
            DataViewSchema.Column col = dv.Schema["x"];
            using DataViewRowCursor cur = dv.GetRowCursor(new[] { col });
            Assert.ThrowsException<InvalidOperationException>(() => cur.GetGetter<double>(col));
        }
    }
}
