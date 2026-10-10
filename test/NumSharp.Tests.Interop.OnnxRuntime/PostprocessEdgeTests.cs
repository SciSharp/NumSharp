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

            // integer inputs land on NumPy's float tier for exp (the values are checked in
            // IntegerAndBoolInputs_GiveTheMathematicalResult_NotTheWrappedOne)
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

        /// <summary>
        ///     Every bool and integer dtype through the three exponential-based methods gives the value that exact
        ///     float64 math gives, to the precision of the result dtype. The inputs are the ones that used to wrap:
        ///     an unsigned element below the maximum (<c>1 - 3</c> is 254 in uint8, and Softmax returned
        ///     <c>[NaN, 0]</c>), a signed range wider than the dtype (<c>-128 - 127</c> wraps in int8), every non-zero
        ///     unsigned element under negation, and the most negative signed value.
        /// </summary>
        [TestMethod]
        public void IntegerAndBoolInputs_GiveTheMathematicalResult_NotTheWrappedOne()
        {
            var cases = new (NPTypeCode Code, NPTypeCode Result, double[] Values)[]
            {
                (NPTypeCode.Boolean, NPTypeCode.Half, new double[] { 1, 0, 1 }),
                (NPTypeCode.Byte, NPTypeCode.Half, new double[] { 1, 3 }),
                (NPTypeCode.Byte, NPTypeCode.Half, new double[] { 200, 100, 250 }),
                (NPTypeCode.SByte, NPTypeCode.Half, new double[] { -128, 0, 127 }),
                (NPTypeCode.SByte, NPTypeCode.Half, new double[] { -2, 0, 5 }),
                (NPTypeCode.UInt16, NPTypeCode.Single, new double[] { 0, 1, 2 }),
                (NPTypeCode.Int16, NPTypeCode.Single, new double[] { -32768, 0, 32767 }),
                (NPTypeCode.Char, NPTypeCode.Single, new double[] { 65, 66, 67 }),
                (NPTypeCode.UInt32, NPTypeCode.Double, new double[] { 1, 3 }),
                (NPTypeCode.Int32, NPTypeCode.Double, new double[] { int.MinValue, 0, int.MaxValue }),
                (NPTypeCode.UInt64, NPTypeCode.Double, new double[] { 0, 1, 2 }),
                (NPTypeCode.Int64, NPTypeCode.Double, new double[] { -7, 7, 9 }),
            };
            foreach (var (code, result, values) in cases)
            {
                using NDArray asDouble = np.array(values);
                using NDArray x = asDouble.astype(code);
                string cell = $"{code} [{string.Join(", ", values)}]";

                using (NDArray p = Postprocess.Softmax(x))
                {
                    p.typecode.Should().Be(result, $"{cell}: softmax dtype");
                    AssertClose(p, SoftmaxOf(values), $"{cell}: softmax");
                }
                using (NDArray l = Postprocess.LogSoftmax(x))
                {
                    l.typecode.Should().Be(result, $"{cell}: log-softmax dtype");
                    AssertClose(l, LogSoftmaxOf(values), $"{cell}: log-softmax");
                }
                using (NDArray s = Postprocess.Sigmoid(x))
                {
                    s.typecode.Should().Be(result, $"{cell}: sigmoid dtype");
                    AssertClose(s, Array.ConvertAll(values, v => 1.0 / (1.0 + Math.Exp(-v))), $"{cell}: sigmoid");
                }
            }
        }

        /// <summary>
        ///     A slice whose maximum is not finite follows <c>scipy.special.log_softmax</c> 1.16.3, which replaces that
        ///     maximum by 0: <c>[inf, 1]</c> is <c>[nan, -inf]</c> (the plain formula gave NaN for both), an all
        ///     <c>-inf</c> slice and a slice with NaN are all NaN, and a finite slice beside them is not touched.
        /// </summary>
        [TestMethod]
        public void LogSoftmax_NonFiniteMaximum_FollowsSciPy()
        {
            using NDArray x = np.array(new[,]
            {
                { double.PositiveInfinity, 1.0 },
                { double.NegativeInfinity, double.NegativeInfinity },
                { double.NaN, 1.0 },
                { 0.0, 1.0 },
            });
            using NDArray l = Postprocess.LogSoftmax(x);
            double.IsNaN(l.GetDouble(0, 0)).Should().BeTrue("inf - inf");
            l.GetDouble(0, 1).Should().Be(double.NegativeInfinity, "a finite element beside +inf has log-probability -inf");
            double.IsNaN(l.GetDouble(1, 0)).Should().BeTrue("an all -inf slice has no distribution");
            double.IsNaN(l.GetDouble(1, 1)).Should().BeTrue();
            double.IsNaN(l.GetDouble(2, 0)).Should().BeTrue("NaN reaches every element of its slice");
            double.IsNaN(l.GetDouble(2, 1)).Should().BeTrue();
            double[] finite = LogSoftmaxOf(new[] { 0.0, 1.0 });
            l.GetDouble(3, 0).Should().BeApproximately(finite[0], 1e-12, "the finite slice is computed as before");
            l.GetDouble(3, 1).Should().BeApproximately(finite[1], 1e-12);

            // float32 takes the same route (the replacement keeps the dtype)
            using NDArray f = np.array(new[] { float.PositiveInfinity, 1f });
            using NDArray lf = Postprocess.LogSoftmax(f);
            lf.typecode.Should().Be(NPTypeCode.Single);
            float.IsNaN(lf.GetSingle(0)).Should().BeTrue();
            lf.GetSingle(1).Should().Be(float.NegativeInfinity);
        }

        /// <summary>
        ///     <c>Argmax</c> refuses an axis outside <c>[-ndim, ndim)</c> itself, with NumPy's AxisError text; it does
        ///     not depend on what <c>np.argmax</c> does with a too-negative axis (an engine that wraps <c>-3</c> to
        ///     axis 1 on a 2-D array returns indices along the wrong axis with no error).
        /// </summary>
        [TestMethod]
        public void Argmax_AxisOutOfRange_Throws_ForBothSigns()
        {
            using NDArray x = np.array(new[,] { { 1f, 5f, 2f }, { 9f, 0f, 3f } });
            new Action(() => Postprocess.Argmax(x, axis: -3)).Should().Throw<AxisError>()
                .WithMessage("axis -3 is out of bounds for array of dimension 2*");
            new Action(() => Postprocess.Argmax(x, axis: 2)).Should().Throw<AxisError>()
                .WithMessage("axis 2 is out of bounds for array of dimension 2*");

            using NDArray last = Postprocess.Argmax(x, axis: -2);   // the most negative legal axis
            last.shape.Should().Equal(3);
        }

        /// <summary>The exact softmax of <paramref name="x"/> in float64, from the values (no integer arithmetic).</summary>
        /// <param name="x">The logits.</param>
        /// <returns>The probabilities.</returns>
        private static double[] SoftmaxOf(double[] x)
        {
            double max = double.NegativeInfinity;
            foreach (double v in x)
                max = Math.Max(max, v);
            double[] e = Array.ConvertAll(x, v => Math.Exp(v - max));
            double sum = 0;
            foreach (double v in e)
                sum += v;
            return Array.ConvertAll(e, v => v / sum);
        }

        /// <summary>The exact log-softmax of <paramref name="x"/> in float64: <c>x - max - log(sum(exp(x - max)))</c>.</summary>
        /// <param name="x">The logits (finite).</param>
        /// <returns>The log-probabilities.</returns>
        private static double[] LogSoftmaxOf(double[] x)
        {
            double max = double.NegativeInfinity;
            foreach (double v in x)
                max = Math.Max(max, v);
            double sum = 0;
            foreach (double v in x)
                sum += Math.Exp(v - max);
            double logSum = Math.Log(sum);
            return Array.ConvertAll(x, v => v - max - logSum);
        }

        /// <summary>
        ///     Asserts a 1-D result equals the expected values to the precision of its own dtype: float16 holds about
        ///     three decimal digits, float32 about seven. A wrapped computation misses by far more than that (NaN,
        ///     0 for 0.73, 0.73 for 0).
        /// </summary>
        /// <param name="actual">The result (float16, float32 or float64).</param>
        /// <param name="expected">The exact values.</param>
        /// <param name="what">The cell, for the message.</param>
        private static void AssertClose(NDArray actual, double[] expected, string what)
        {
            double tolerance = actual.typecode switch
            {
                NPTypeCode.Half => 2e-3,
                NPTypeCode.Single => 1e-6,
                _ => 1e-12,
            };
            actual.size.Should().Be(expected.Length, what);
            using NDArray wide = actual.astype(NPTypeCode.Double);
            for (int i = 0; i < expected.Length; i++)
            {
                double got = wide.GetDouble(i);
                // relative for large magnitudes (log-probabilities reach -4e9), absolute near zero
                double allowed = tolerance * Math.Max(1.0, Math.Abs(expected[i]));
                Math.Abs(got - expected[i]).Should().BeLessThanOrEqualTo(allowed, $"{what}: element {i} is {got}, exact {expected[i]}");
            }
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
