using System;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>Empty axes, 0-d, degenerate and unit dimensions — in both the vector and scalar-column shapes.</summary>
    [TestClass]
    public class ShapeEdgeTests : MLNetTestBase
    {
        private static NDArray Empty(NPTypeCode code, params int[] shape)
        {
            NDArray flat = np.arange(0).astype(code, copy: true);   // owning (0,); NOT disposed here — the caller owns it
            return shape.Length <= 1 ? flat : flat.reshape(shape);
        }

        // ---- empty vectors --------------------------------------------------------------------------

        [TestMethod]
        public void Empty1D_Vector_IsOneRowOfWidthZero()
        {
            using NDArray a = Empty(NPTypeCode.Single);          // (0,)
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.AreEqual(1L, dv.GetRowCount());
            Assert.AreEqual(0, ((VectorDataViewType)dv.Schema["F"].Type).Size);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 1, 0);
        }

        [TestMethod]
        public void EmptyRows_Vector_IsZeroRows()
        {
            using NDArray a = Empty(NPTypeCode.Single, 0, 3);    // (0,3)
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.AreEqual(0L, dv.GetRowCount());
            Assert.AreEqual(3, ((VectorDataViewType)dv.Schema["F"].Type).Size);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 0, 3);
        }

        [TestMethod]
        public void EmptyFeatures_Vector_IsWidthZeroColumn()
        {
            using NDArray a = Empty(NPTypeCode.Single, 3, 0);    // (3,0)
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.AreEqual(3L, dv.GetRowCount());
            Assert.AreEqual(0, ((VectorDataViewType)dv.Schema["F"].Type).Size);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 3, 0);
        }

        [TestMethod]
        public void EmptyBoth_Vector()
        {
            using NDArray a = Empty(NPTypeCode.Double, 0, 0);
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 0, 0);
        }

        // ---- empty scalar columns -------------------------------------------------------------------

        [TestMethod]
        public void Empty1D_ScalarColumn_IsZeroRows()
        {
            using NDArray a = Empty(NPTypeCode.Int32);           // (0,)
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            Assert.AreEqual(0L, dv.GetRowCount());
            using NDArray back = dv.ToNDArray("v");
            AssertShape(back, 0);
        }

        [TestMethod]
        public void EmptyRows_ScalarColumns_EachColumnIsZeroLength()
        {
            using NDArray a = Empty(NPTypeCode.Single, 0, 3);
            using NDArrayDataView dv = a.AsDataView(new[] { "a", "b", "c" });
            Assert.AreEqual(3, dv.Schema.Count);
            using NDArray b = dv.ToNDArray("b");
            AssertShape(b, 0);
        }

        // ---- 0-d rejected ---------------------------------------------------------------------------

        [TestMethod]
        public void Scalar0D_Vector_Throws()
        {
            using NDArray a = np.array(5f);
            Assert.AreEqual(0, a.ndim);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => a.AsDataView("F"));
            StringAssert.Contains(ex.Message, "1-D or 2-D");
        }

        [TestMethod]
        public void Scalar0D_ScalarColumns_Throws()
        {
            using NDArray a = np.array(5f);
            Assert.ThrowsException<ArgumentException>(() => a.AsDataView(new[] { "v" }));
        }

        // ---- rank > 2 rejected ----------------------------------------------------------------------

        [TestMethod]
        public void Rank3And4_BothModes_Throw()
        {
            using NDArray a3 = Arange(NPTypeCode.Single, 2, 2, 2);
            using NDArray a4 = Arange(NPTypeCode.Single, 2, 2, 2, 2);
            Assert.ThrowsException<ArgumentException>(() => a3.AsDataView("F"));
            Assert.ThrowsException<ArgumentException>(() => a3.AsDataView(new[] { "a", "b" }));
            Assert.ThrowsException<ArgumentException>(() => a4.AsDataView("F"));
            Assert.ThrowsException<ArgumentException>(() => a4.ToDataView("F"));
        }

        // ---- unit / degenerate dims -----------------------------------------------------------------

        [TestMethod]
        public void Unit_1x1_Vector()
        {
            using NDArray a = np.array(5f).reshape(1, 1);
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 1, 1);
            Assert.AreEqual(5f, (float)back[0, 0]);
        }

        [TestMethod]
        public void ColumnVector_Nx1_Vector_IsNRowsOfWidth1()
        {
            using NDArray a = Arange(NPTypeCode.Single, 5, 1);
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.AreEqual(5L, dv.GetRowCount());
            Assert.AreEqual(1, ((VectorDataViewType)dv.Schema["F"].Type).Size);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 5, 1);
        }

        [TestMethod]
        public void RowVector_1xN_Vector_IsOneRowOfWidthN()
        {
            using NDArray a = Arange(NPTypeCode.Single, 1, 5);
            using NDArrayDataView dv = a.AsDataView("F");
            Assert.AreEqual(1L, dv.GetRowCount());
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 1, 5);
        }
    }
}
