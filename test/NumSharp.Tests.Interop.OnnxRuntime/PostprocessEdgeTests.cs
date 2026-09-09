using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     <see cref="Postprocess"/> corners <see cref="PostprocessTests"/> leaves: null guards, non-default and
    ///     negative axes, <c>keepdims</c>, integer-input dtype promotion, non-contiguous inputs, 1-D TopK, the
    ///     ignored <c>sorted</c> flag, and the numerical-stability formulations of LogSoftmax / Sigmoid.
    /// </summary>
    [TestClass]
    public class PostprocessEdgeTests : OnnxTestBase
    {
        [TestMethod]
        public void NullInputs_ThrowArgumentNull()
        {
            new Action(() => Postprocess.Softmax(null)).Should().Throw<ArgumentNullException>();
            new Action(() => Postprocess.LogSoftmax(null)).Should().Throw<ArgumentNullException>();
            new Action(() => Postprocess.Sigmoid(null)).Should().Throw<ArgumentNullException>();
            new Action(() => Postprocess.Argmax(null)).Should().Throw<ArgumentNullException>();
            new Action(() => Postprocess.TopK(null, 1)).Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void Softmax_AlongAxis0_NormalizesColumns_ShapePreserved()
        {
            using NDArray x = np.array(new[,] { { 1f, 5f }, { 2f, 3f }, { 0f, 1f } });
            using NDArray p = Postprocess.Softmax(x, axis: 0);
            p.shape.Should().Equal(3, 2);
            using NDArray colSums = np.sum(p, 0);
            using NDArray ones = np.ones(new Shape(2), typeof(float));
            np.allclose(colSums, ones, 1e-5, 1e-6).Should().BeTrue("columns sum to 1 when axis=0");
        }

        [TestMethod]
        public void Softmax_PreservesFloat64_AndPromotesIntegerInputsToTheirFloatTier()
        {
            using NDArray d = np.array(new[] { 1.0, 2.0, 3.0 });
            using NDArray pd = Postprocess.Softmax(d);
            pd.typecode.Should().Be(NPTypeCode.Double, "float64 in, float64 out");

            // integer inputs land on NumPy's float tier for exp (bool would fail the subtract, so it is excluded)
            using (NDArray i8 = Arange(NPTypeCode.SByte, 3))
            using (NDArray p8 = Postprocess.Softmax(i8))
                p8.typecode.Should().Be(NPTypeCode.Half, "int8 -> float16 tier");
            using (NDArray i16 = Arange(NPTypeCode.Int16, 3))
            using (NDArray p16 = Postprocess.Softmax(i16))
                p16.typecode.Should().Be(NPTypeCode.Single, "int16 -> float32 tier");
            using (NDArray i32 = Arange(NPTypeCode.Int32, 3))
            using (NDArray p32 = Postprocess.Softmax(i32))
                p32.typecode.Should().Be(NPTypeCode.Double, "int32+ -> float64 tier");
        }

        [TestMethod]
        public void Softmax_OnANonContiguousView_MatchesTheContiguousResult()
        {
            using NDArray m = Arange(NPTypeCode.Single, 3, 4);
            using NDArray view = m.T;                 // (4,3), non-contiguous
            view.Shape.IsContiguous.Should().BeFalse();
            using NDArray dense = view.copy();

            using NDArray a = Postprocess.Softmax(view);
            using NDArray b = Postprocess.Softmax(dense);
            a.shape.Should().Equal(4, 3);
            np.allclose(a, b, 1e-6, 1e-7).Should().BeTrue("softmax reads a view in logical order, same result as the copy");
        }

        [TestMethod]
        public void LogSoftmax_IsStableForHugeLogits_AndEqualsLogOfSoftmax()
        {
            using NDArray x = np.array(new[] { 1000f, 1001f, 1002f });
            using NDArray lsm = Postprocess.LogSoftmax(x);
            np.all(np.isfinite(lsm)).Should().BeTrue("the max-shift keeps log(sum(exp)) finite where log(softmax) would be -inf/nan");
            lsm.GetSingle(2).Should().BeApproximately(-0.40760595f, 1e-4f, "log-softmax of the max element");

            using NDArray normal = np.array(new[] { 0.5, -1.0, 2.0, 0.0 });
            using NDArray lsmN = Postprocess.LogSoftmax(normal);
            using NDArray logOfSm = np.log(Postprocess.Softmax(normal));
            np.allclose(lsmN, logOfSm, 1e-9, 1e-10).Should().BeTrue();
        }

        [TestMethod]
        public void Sigmoid_IsSymmetric_AndHandlesExtremes()
        {
            using NDArray x = np.array(new[] { -8.0, -2.0, 0.0, 3.0, 10.0 });
            using NDArray s = Postprocess.Sigmoid(x);
            using NDArray negX = -x;
            using NDArray sNeg = Postprocess.Sigmoid(negX);
            using NDArray oneMinus = 1.0 - s;
            np.allclose(sNeg, oneMinus, 1e-12, 1e-12).Should().BeTrue("sigmoid(-x) == 1 - sigmoid(x)");
            s.GetDouble(2).Should().Be(0.5);
            np.all(np.isfinite(s)).Should().BeTrue();
        }

        [TestMethod]
        public void Argmax_Keepdims_And_NegativeAxis()
        {
            using NDArray x = np.array(new[,] { { 1f, 5f, 2f }, { 9f, 0f, 3f } });

            using NDArray kd = Postprocess.Argmax(x, axis: -1, keepdims: true);
            kd.shape.Should().Equal(2, 1);
            kd.GetInt64(0, 0).Should().Be(1L);
            kd.GetInt64(1, 0).Should().Be(0L);

            using NDArray a0 = Postprocess.Argmax(x, axis: 0);
            a0.shape.Should().Equal(3);
            a0.GetInt64(0).Should().Be(1L);   // 9 > 1
            using NDArray aNeg = Postprocess.Argmax(x, axis: -2);
            np.array_equal(a0, aNeg).Should().BeTrue("axis -2 == axis 0 on a 2-D array");
        }

        [TestMethod]
        public void TopK_1D_ReturnsA1DResult()
        {
            using NDArray x = np.array(new[] { 3f, 1f, 4f, 1f, 5f });
            (NDArray values, NDArray indices) = Postprocess.TopK(x, 2);
            using (values)
            using (indices)
            {
                values.shape.Should().Equal(2);
                values.GetSingle(0).Should().Be(5f);
                values.GetSingle(1).Should().Be(4f);
                indices.GetInt64(0).Should().Be(4L);
                indices.GetInt64(1).Should().Be(2L);
                indices.Storage.IsView.Should().BeFalse("indices is an owning C-contiguous copy, not a window into the argsort");
            }

            (NDArray sv, NDArray si) = Postprocess.TopK(x, 2, largest: false);
            using (sv)
            using (si)
            {
                sv.GetSingle(0).Should().Be(1f);
                si.GetInt64(0).Should().Be(1L, "ascending, ties keep the lower index first");
            }
        }

        [TestMethod]
        public void TopK_NegativeAxis_RanksAlongTheResolvedAxis()
        {
            using NDArray x = np.array(new[,] { { 9f, 1f }, { 4f, 8f }, { 7f, 7f } });
            (NDArray values, NDArray indices) = Postprocess.TopK(x, 2, axis: -2);   // == axis 0
            using (values)
            using (indices)
            {
                values.shape.Should().Equal(2, 2);
                values.GetSingle(0, 0).Should().Be(9f);
                indices.GetInt64(0, 0).Should().Be(0L);
                indices.GetInt64(1, 0).Should().Be(2L, "second-largest of column 0 [9,4,7] is 7 at index 2");
            }
        }

        [TestMethod]
        public void TopK_SortedFlagIsIgnored_ResultIsAlwaysSorted()
        {
            using NDArray x = np.array(new[,] { { 1f, 3f, 2f, 5f, 4f } });
            (NDArray vSorted, NDArray iSorted) = Postprocess.TopK(x, 3, sorted: true);
            (NDArray vUnsorted, NDArray iUnsorted) = Postprocess.TopK(x, 3, sorted: false);
            using (vSorted)
            using (iSorted)
            using (vUnsorted)
            using (iUnsorted)
            {
                np.array_equal(vSorted, vUnsorted).Should().BeTrue("the result is always sorted; the flag is accepted for ONNX parity only");
                np.array_equal(iSorted, iUnsorted).Should().BeTrue();
                vSorted.GetSingle(0, 0).Should().Be(5f);
            }
        }
    }
}
