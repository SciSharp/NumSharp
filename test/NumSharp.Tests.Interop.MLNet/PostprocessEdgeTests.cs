using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>The Postprocess edges — the analog of the ONNX bridge's PostprocessEdgeTests (pure np compositions).</summary>
    [TestClass]
    public class PostprocessEdgeTests : MLNetTestBase
    {
        [TestMethod]
        public void NullInputs_ThrowArgumentNull()
        {
            Assert.ThrowsException<ArgumentNullException>(() => Postprocess.Softmax(null));
            Assert.ThrowsException<ArgumentNullException>(() => Postprocess.LogSoftmax(null));
            Assert.ThrowsException<ArgumentNullException>(() => Postprocess.Sigmoid(null));
            Assert.ThrowsException<ArgumentNullException>(() => Postprocess.Argmax(null));
            Assert.ThrowsException<ArgumentNullException>(() => Postprocess.TopK(null, 1));
        }

        [TestMethod]
        public void Softmax_AlongAxis0_NormalizesColumns_ShapePreserved()
        {
            using NDArray x = np.array(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }).reshape(3, 2);
            using NDArray s = Postprocess.Softmax(x, axis: 0);
            AssertShape(s, 3, 2);
            using NDArray colSums = np.sum(s, axis: 0);
            Assert.AreEqual(1f, (float)colSums[0], 1e-5f);
            Assert.AreEqual(1f, (float)colSums[1], 1e-5f);
        }

        [TestMethod]
        public void Softmax_PreservesFloatPrecision_AndPromotesIntegersToTheFloatTier()
        {
            using NDArray f32 = np.array(new[] { 1f, 2f, 3f });
            using NDArray f64 = np.array(new[] { 1.0, 2.0, 3.0 });
            using NDArray i32 = np.array(new[] { 1, 2, 3 });
            using NDArray s32 = Postprocess.Softmax(f32);
            using NDArray s64 = Postprocess.Softmax(f64);
            using NDArray si = Postprocess.Softmax(i32);
            Assert.AreEqual(NPTypeCode.Single, s32.typecode);
            Assert.AreEqual(NPTypeCode.Double, s64.typecode);
            Assert.AreEqual(NPTypeCode.Double, si.typecode, "int32 lands on NumPy's float64 tier for exp");
        }

        [TestMethod]
        public void Softmax_OnANonContiguousView_MatchesTheContiguousResult()
        {
            using NDArray full = np.array(new float[] { 1f, 2f, 3f, 4f, 5f, 6f }).reshape(2, 3);
            using NDArray view = full["::-1"];   // reversed rows, non-contiguous
            Assert.IsFalse(view.Shape.IsContiguous);
            using NDArray sView = Postprocess.Softmax(view);
            using NDArray sCopy = Postprocess.Softmax(view.copy());
            Assert.IsTrue((bool)np.allclose(sView, sCopy));
            using NDArray rowSums = np.sum(sView, axis: -1);
            using NDArray ones = np.ones_like(rowSums);
            Assert.IsTrue((bool)np.allclose(rowSums, ones));
        }

        [TestMethod]
        public void LogSoftmax_IsStableForHugeLogits_AndEqualsLogOfSoftmax()
        {
            using NDArray huge = np.array(new[] { 1000f, 1001f, 1002f });
            using NDArray ls = Postprocess.LogSoftmax(huge);
            Assert.IsTrue((bool)np.all(np.isfinite(ls)), "no overflow for huge logits");
            using NDArray sm = Postprocess.Softmax(huge);
            using NDArray logSm = np.log(sm);
            Assert.IsTrue((bool)np.allclose(ls, logSm, atol: 1e-5));
        }

        [TestMethod]
        public void Sigmoid_IsSymmetric_AndHandlesExtremes()
        {
            using NDArray x = np.array(new[] { -3f, -1f, 0f, 1f, 3f });
            using NDArray s = Postprocess.Sigmoid(x);
            using NDArray negx = -x;
            using NDArray sNeg = Postprocess.Sigmoid(negx);
            using NDArray oneMinus = 1f - s;
            Assert.IsTrue((bool)np.allclose(sNeg, oneMinus, atol: 1e-6), "sigmoid(-x) == 1 - sigmoid(x)");

            using NDArray ext = np.array(new[] { -100f, 100f });
            using NDArray se = Postprocess.Sigmoid(ext);
            Assert.AreEqual(0f, (float)se[0], 1e-6f);
            Assert.AreEqual(1f, (float)se[1], 1e-6f);
        }

        [TestMethod]
        public void Argmax_Keepdims_And_NegativeAxis()
        {
            using NDArray a = np.array(new[] { 1f, 5f, 2f, 8f, 3f, 1f }).reshape(2, 3);
            using NDArray keep = Postprocess.Argmax(a, axis: -1, keepdims: true);
            AssertShape(keep, 2, 1);
            Assert.AreEqual(1L, (long)keep[0, 0]);   // row 0 max at index 1
            Assert.AreEqual(0L, (long)keep[1, 0]);   // row 1 max at index 0

            using NDArray flat = Postprocess.Argmax(a, axis: -1, keepdims: false);
            AssertShape(flat, 2);
        }

        [TestMethod]
        public void TopK_1D_ReturnsA1DResult()
        {
            using NDArray x = np.array(new[] { 3f, 1f, 4f, 1f, 5f, 9f, 2f });
            (NDArray values, NDArray indices) = Postprocess.TopK(x, k: 3);
            using (values)
            using (indices)
            {
                AssertShape(values, 3);
                AssertShape(indices, 3);
                Assert.AreEqual(9f, (float)values[0]);
                Assert.AreEqual(5f, (float)values[1]);
                Assert.AreEqual(4f, (float)values[2]);
            }
        }

        [TestMethod]
        public void TopK_NegativeAxis_RanksAlongTheResolvedAxis()
        {
            using NDArray x = np.array(new[] { 1f, 9f, 3f, 8f, 2f, 7f }).reshape(2, 3);
            (NDArray v1, NDArray i1) = Postprocess.TopK(x, k: 2, axis: -1);
            (NDArray v2, NDArray i2) = Postprocess.TopK(x, k: 2, axis: 1);
            using (v1)
            using (i1)
            using (v2)
            using (i2)
            {
                Assert.IsTrue(np.array_equal(v1, v2), "axis -1 == axis 1 for a 2-D array");
                Assert.IsTrue(np.array_equal(i1, i2));
            }
        }

        [TestMethod]
        public void TopK_SortedFlagIsIgnored_ResultIsAlwaysSorted()
        {
            using NDArray x = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
            (NDArray vT, NDArray iT) = Postprocess.TopK(x, k: 3, sorted: true);
            (NDArray vF, NDArray iF) = Postprocess.TopK(x, k: 3, sorted: false);
            using (vT)
            using (iT)
            using (vF)
            using (iF)
            {
                Assert.IsTrue(np.array_equal(vT, vF), "the result is always sorted regardless of the flag");
                Assert.AreEqual(5f, (float)vT[0]);
                Assert.AreEqual(4f, (float)vT[1]);
                Assert.AreEqual(3f, (float)vT[2]);
            }
        }
    }
}
