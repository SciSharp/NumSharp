using System;
using System.Collections.Generic;
using AwesomeAssertions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     <see cref="Postprocess"/> against the ORACLE the package replaces: ORT's own Softmax / ArgMax / TopK
    ///     operators run on the same data through the same session API.
    /// </summary>
    [TestClass]
    public class PostprocessTests : OnnxTestBase
    {
        private static NDArray RandomLogits(int rows, int cols, int seed)
        {
            np.random.seed(seed);
            using NDArray g = np.random.randn(rows, cols);
            using NDArray scaled = g * 3.0;
            return scaled.astype(NPTypeCode.Single);
        }

        [TestMethod]
        public void Softmax_And_Argmax_MatchOrtsOperators()
        {
            using InferenceSession session = OpenSession("postprocess_f32");
            using NDArray x = RandomLogits(4, 7, seed: 3);
            IReadOnlyDictionary<string, NDArray> ort = session.Run(new Dictionary<string, NDArray> { { "X", x } });
            using NDArray ortProbs = ort["probs"];
            using NDArray ortArgmax = ort["argmax"];

            using NDArray probs = Postprocess.Softmax(x);
            probs.typecode.Should().Be(NPTypeCode.Single, "float32 in, float32 out");
            probs.shape.Should().Equal(4, 7);
            np.allclose(probs, ortProbs, 1e-5, 1e-6).Should().BeTrue("softmax agrees with ORT's Softmax to float tolerance");
            using NDArray rowSums = np.sum(probs, 1);
            using NDArray ones = np.ones(new Shape(4), typeof(float));
            np.allclose(rowSums, ones, 1e-5, 1e-6).Should().BeTrue("rows sum to 1");

            using NDArray argmax = Postprocess.Argmax(x);
            argmax.typecode.Should().Be(NPTypeCode.Int64);
            np.array_equal(argmax, ortArgmax).Should().BeTrue("argmax agrees exactly with ORT's ArgMax");
        }

        [TestMethod]
        public void Softmax_IsStable_ForHugeLogits()
        {
            using NDArray x = np.array(new[] { 1000f, 1001f, 1002f });
            using NDArray p = Postprocess.Softmax(x);
            using NDArray direct = np.exp(x);
            np.all(np.isfinite(p)).Should().BeTrue("the max-shift keeps exp in range where the naive form overflows");
            np.isfinite(direct).GetBoolean(2).Should().BeFalse("proof the naive form overflows");
            p.GetSingle(2).Should().BeApproximately(0.66524f, 1e-4f);
        }

        [TestMethod]
        public void LogSoftmax_And_Sigmoid()
        {
            using NDArray x = RandomLogits(2, 5, seed: 11);
            using NDArray lsm = Postprocess.LogSoftmax(x);
            using NDArray sm = Postprocess.Softmax(x);
            using NDArray logOfSm = np.log(sm);
            np.allclose(lsm, logOfSm, 1e-4, 1e-5).Should().BeTrue();

            using NDArray z = np.array(new[] { -1000.0, 0.0, 1000.0 });
            using NDArray s = Postprocess.Sigmoid(z);
            s.GetDouble(0).Should().BeApproximately(0.0, 1e-12);
            s.GetDouble(1).Should().Be(0.5);
            s.GetDouble(2).Should().Be(1.0);
        }

        [TestMethod]
        public void TopK_MatchesOrtsTopK_ValuesAndIndices()
        {
            using InferenceSession session = OpenSession("topk_f32");
            using NDArray x = RandomLogits(3, 8, seed: 5);
            foreach (int k in new[] { 1, 3, 8 })
            {
                using NDArray kTensor = np.array(new[] { (long)k });
                IReadOnlyDictionary<string, NDArray> ort = session.Run(new Dictionary<string, NDArray> { { "X", x }, { "K", kTensor } });
                using NDArray ortValues = ort["Values"];
                using NDArray ortIndices = ort["Indices"];

                (NDArray values, NDArray indices) = Postprocess.TopK(x, k);
                using (values)
                using (indices)
                {
                    values.shape.Should().Equal(3, k);
                    indices.typecode.Should().Be(NPTypeCode.Int64);
                    np.array_equal(indices, ortIndices).Should().BeTrue($"k={k}: indices");
                    BytesOf(values).Should().Equal(BytesOf(ortValues), $"k={k}: values are gathered, so bit-exact");
                }
            }
        }

        [TestMethod]
        public void TopK_TieBreak_LowerIndexFirst_LikeOrt()
        {
            using InferenceSession session = OpenSession("topk_f32");
            using NDArray x = np.array(new[,] { { 1f, 3f, 3f, 2f, 3f }, { 5f, 5f, 1f, 5f, 0f } });
            using NDArray k2 = np.array(new[] { 3L });
            IReadOnlyDictionary<string, NDArray> ort = session.Run(new Dictionary<string, NDArray> { { "X", x }, { "K", k2 } });
            using NDArray ortIndices = ort["Indices"];
            using NDArray ortValues = ort["Values"];

            (NDArray values, NDArray indices) = Postprocess.TopK(x, 3);
            using (values)
            using (indices)
            {
                indices.GetInt64(0, 0).Should().Be(1L);
                indices.GetInt64(0, 1).Should().Be(2L);
                indices.GetInt64(0, 2).Should().Be(4L);
                indices.GetInt64(1, 0).Should().Be(0L);
                indices.GetInt64(1, 1).Should().Be(1L);
                indices.GetInt64(1, 2).Should().Be(3L);
                np.array_equal(indices, ortIndices).Should().BeTrue("ORT's TopK breaks ties by lower index too");
                np.array_equal(values, ortValues).Should().BeTrue();
            }

            (NDArray smallest, NDArray smallestIdx) = Postprocess.TopK(x, 2, largest: false);
            using (smallest)
            using (smallestIdx)
            {
                smallestIdx.GetInt64(0, 0).Should().Be(0L);
                smallestIdx.GetInt64(0, 1).Should().Be(3L);
                smallest.GetSingle(1, 0).Should().Be(0f);
                smallest.GetSingle(1, 1).Should().Be(1f);
            }
        }

        [TestMethod]
        public void TopK_OtherAxis_AndIntegerDtype()
        {
            using NDArray x = np.array(new[,] { { 9, 1 }, { 4, 8 }, { 7, 7 } });
            (NDArray values, NDArray indices) = Postprocess.TopK(x, 2, axis: 0);
            using (values)
            using (indices)
            {
                values.shape.Should().Equal(2, 2);
                values.typecode.Should().Be(NPTypeCode.Int32, "values keep the input dtype");
                indices.GetInt64(0, 0).Should().Be(0L);
                indices.GetInt64(1, 0).Should().Be(2L);
                indices.GetInt64(0, 1).Should().Be(1L);
                indices.GetInt64(1, 1).Should().Be(2L);
                values.GetInt32(0, 0).Should().Be(9);
                values.GetInt32(1, 1).Should().Be(7);
            }
        }

        [TestMethod]
        public void TopK_Validation()
        {
            using NDArray x = np.array(new[] { 1f, 2f, 3f });
            new Action(() => Postprocess.TopK(x, 0)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => Postprocess.TopK(x, 4)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => Postprocess.TopK(x, 1, axis: 2)).Should().Throw<ArgumentOutOfRangeException>();
            using NDArray scalar = NDArray.Scalar(1f);
            new Action(() => Postprocess.TopK(scalar, 1)).Should().Throw<ArgumentException>();
        }
    }
}
