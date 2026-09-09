using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>Materializing columns out of REAL ML.NET data views (not just NumSharp-backed ones).</summary>
    [TestClass]
    public class ImportEdgeTests : MLNetTestBase
    {
        // ---- reading a non-NumSharp IDataView (POCO loader) -----------------------------------------

        [TestMethod]
        public void ScalarColumns_FromPocoLoader_CrossCheck()
        {
            var rows = new[]
            {
                new ScalarRow { A = 1f, B = 10f, Flag = true },
                new ScalarRow { A = 2f, B = 20f, Flag = false },
                new ScalarRow { A = 3f, B = 30f, Flag = true },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);

            using NDArray a = dv.ToNDArray("A");
            using NDArray b = dv.ToNDArray("B");
            using NDArray flag = dv.ToNDArray("Flag");
            AssertShape(a, 3);
            Assert.AreEqual(NPTypeCode.Single, a.typecode);
            Assert.AreEqual(NPTypeCode.Boolean, flag.typecode);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3 }, a.ToArray<float>());
            CollectionAssert.AreEqual(new float[] { 10, 20, 30 }, b.ToArray<float>());
            CollectionAssert.AreEqual(new[] { true, false, true }, flag.ToArray<bool>());
        }

        [TestMethod]
        public void VectorColumn_FixedSize_FromPocoLoader_Is2D()
        {
            var rows = new[]
            {
                new VecRow { F = new float[] { 1, 2, 3 } },
                new VecRow { F = new float[] { 4, 5, 6 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 2, 3);
            Assert.AreEqual(6f, (float)back[1, 2]);
        }

        [TestMethod]
        public void VectorColumn_UnsizedButConsistent_Materializes()
        {
            var rows = new[]
            {
                new UnsizedVecRow { F = new float[] { 1, 2 } },
                new UnsizedVecRow { F = new float[] { 3, 4 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 2, 2);
        }

        [TestMethod]
        public void VectorColumn_ActuallyVariableLength_Throws()
        {
            var rows = new[]
            {
                new UnsizedVecRow { F = new float[] { 1, 2 } },
                new UnsizedVecRow { F = new float[] { 1, 2, 3 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => dv.ToNDArray("F"));
            StringAssert.Contains(ex.Message, "variable-length");
        }

        [TestMethod]
        public void TextColumn_Throws()
        {
            var rows = new[] { new TextRow { T = "hello" }, new TextRow { T = "world" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => dv.ToNDArray("T"));
            StringAssert.Contains(ex.Message, "text");
        }

        // ---- key column -----------------------------------------------------------------------------

        [TestMethod]
        public void KeyColumn_ReadsAsUInt32()
        {
            var rows = new[] { new TextRow { T = "a" }, new TextRow { T = "b" }, new TextRow { T = "a" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ITransformer model = Ml.Transforms.Conversion.MapValueToKey("K", "T").Fit(dv);
            IDataView keyed = model.Transform(dv);

            Assert.IsInstanceOfType(keyed.Schema["K"].Type, typeof(KeyDataViewType));
            using NDArray k = keyed.ToNDArray("K");
            Assert.AreEqual(NPTypeCode.UInt32, k.typecode);
            // keys are 1-based; "a"->1, "b"->2, "a"->1
            CollectionAssert.AreEqual(new uint[] { 1, 2, 1 }, k.ToArray<uint>());
        }

        // ---- empty / lazy views ---------------------------------------------------------------------

        [TestMethod]
        public void EmptyView_ScalarColumn_IsZeroRows()
        {
            using NDArray a = np.arange(10).astype(NPTypeCode.Single).reshape(10, 1);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            IDataView empty = Ml.Data.FilterRowsByColumn(dv, "v", lowerBound: 100, upperBound: 200);   // removes all
            using NDArray back = empty.ToNDArray("v");
            AssertShape(back, 0);
        }

        [TestMethod]
        public void LazyView_UnknownRowCount_Materializes()
        {
            using NDArray a = np.arange(10).astype(NPTypeCode.Single).reshape(10, 1);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            IDataView filtered = Ml.Data.FilterRowsByColumn(dv, "v", lowerBound: 3, upperBound: 8);   // keeps 3..7
            Assert.IsNull(filtered.GetRowCount(), "a filtered view reports an unknown row count");
            using NDArray back = filtered.ToNDArray("v");
            CollectionAssert.AreEqual(new float[] { 3, 4, 5, 6, 7 }, back.ToArray<float>());
        }

        // ---- overloads & multiple columns -----------------------------------------------------------

        [TestMethod]
        public void ToNDArray_ByColumnStruct_Overload()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 4);
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray(dv.Schema["F"]);
            AssertShape(back, 3, 4);
        }

        [TestMethod]
        public void MultipleColumns_MaterializeIndependently()
        {
            var rows = new[]
            {
                new ScalarRow { A = 1f, B = 10f, Flag = true },
                new ScalarRow { A = 2f, B = 20f, Flag = false },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray a1 = dv.ToNDArray("A");
            using NDArray a2 = dv.ToNDArray("A");   // re-cursoring the same view
            using NDArray b = dv.ToNDArray("B");
            Assert.IsTrue(np.array_equal(a1, a2));
            CollectionAssert.AreEqual(new float[] { 10, 20 }, b.ToArray<float>());
        }

        private class ScalarRow
        {
            public float A { get; set; }
            public float B { get; set; }
            public bool Flag { get; set; }
        }

        private class VecRow
        {
            [VectorType(3)] public float[] F { get; set; }
        }

        private class UnsizedVecRow
        {
            [VectorType] public float[] F { get; set; }
        }

        private class TextRow
        {
            public string T { get; set; }
        }
    }
}
