using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    [TestClass]
    public class PostprocessTests : MLNetTestBase
    {
        [TestMethod]
        public void Softmax_RowsSumToOne()
        {
            using NDArray logits = np.array(new float[] { 1f, 2f, 3f, 4f }).reshape(2, 2);
            using NDArray sm = Postprocess.Softmax(logits);
            using NDArray rowSums = np.sum(sm, axis: -1);
            Assert.AreEqual(1f, (float)rowSums[0], 1e-5f);
            Assert.AreEqual(1f, (float)rowSums[1], 1e-5f);
        }

        [TestMethod]
        public void Softmax_MatchesManualStableForm()
        {
            using NDArray x = np.array(new float[] { 1f, 2f, 3f });
            using NDArray sm = Postprocess.Softmax(x);
            // e^0 : e^1 : e^2 normalized (shifted by max=3)
            double e0 = Math.Exp(-2), e1 = Math.Exp(-1), e2 = Math.Exp(0);
            double denom = e0 + e1 + e2;
            Assert.AreEqual((float)(e0 / denom), (float)sm[0], 1e-5f);
            Assert.AreEqual((float)(e2 / denom), (float)sm[2], 1e-5f);
        }

        [TestMethod]
        public void LogSoftmax_EqualsLogOfSoftmax()
        {
            using NDArray x = np.array(new float[] { 0.5f, 1.5f, -1f, 2f }).reshape(2, 2);
            using NDArray ls = Postprocess.LogSoftmax(x);
            using NDArray sm = Postprocess.Softmax(x);
            using NDArray logSm = np.log(sm);
            Assert.IsTrue((bool)np.allclose(ls, logSm, atol: 1e-5));
        }

        [TestMethod]
        public void Sigmoid_KnownValues()
        {
            using NDArray x = np.array(new float[] { 0f, 100f, -100f });
            using NDArray s = Postprocess.Sigmoid(x);
            Assert.AreEqual(0.5f, (float)s[0], 1e-6f);
            Assert.AreEqual(1f, (float)s[1], 1e-5f);
            Assert.AreEqual(0f, (float)s[2], 1e-5f);
        }

        [TestMethod]
        public void Argmax_FirstOccurrenceOnTies()
        {
            using NDArray x = np.array(new float[] { 1f, 3f, 3f, 2f });
            using NDArray idx = Postprocess.Argmax(x);
            Assert.AreEqual(1L, (long)idx);
        }

        [TestMethod]
        public void TopK_Largest_SortedDescending_TiesLowerIndexFirst()
        {
            using NDArray scores = np.array(new float[] { 10f, 30f, 30f, 20f, 5f }).reshape(1, 5);
            (NDArray values, NDArray indices) = Postprocess.TopK(scores, k: 3, axis: -1, largest: true);
            using (values)
            using (indices)
            {
                AssertShape(values, 1, 3);
                // descending values 30,30,20; ties (the two 30s) keep the lower index (1) before (2)
                Assert.AreEqual(30f, (float)values[0, 0]);
                Assert.AreEqual(30f, (float)values[0, 1]);
                Assert.AreEqual(20f, (float)values[0, 2]);
                Assert.AreEqual(1L, (long)indices[0, 0]);
                Assert.AreEqual(2L, (long)indices[0, 1]);
                Assert.AreEqual(3L, (long)indices[0, 2]);
            }
        }

        [TestMethod]
        public void TopK_Smallest_SortedAscending()
        {
            using NDArray scores = np.array(new float[] { 10f, 30f, 5f, 20f });
            (NDArray values, NDArray indices) = Postprocess.TopK(scores, k: 2, axis: -1, largest: false);
            using (values)
            using (indices)
            {
                Assert.AreEqual(5f, (float)values[0]);
                Assert.AreEqual(10f, (float)values[1]);
                Assert.AreEqual(2L, (long)indices[0]);
                Assert.AreEqual(0L, (long)indices[1]);
            }
        }

        [TestMethod]
        public void TopK_InvalidK_Throws()
        {
            using NDArray scores = np.array(new float[] { 1f, 2f, 3f });
            Assert.ThrowsException<ArgumentOutOfRangeException>(() => Postprocess.TopK(scores, k: 5));
        }
    }
}
