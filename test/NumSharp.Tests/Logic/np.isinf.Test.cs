using System.Linq;

namespace NumSharp.Tests.Logic
{
    /// <summary>
    /// Tests for np.isinf - test element-wise for positive or negative infinity.
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
    /// </summary>
    /// <remarks>
    /// Written when DefaultEngine.IsInfinite returned null, and marked OpenBugs until np.isinf was implemented.
    /// np.isinf has been implemented since (a fused predicate kernel, see "Comparison &amp; Logic" in CLAUDE.md), but the
    /// class had no [TestClass] from the TUnit → MSTest migration (ac020336, 2026-04-15) until 2026-10-05, so MSTest never
    /// ran these tests. The 2026-10-05 run of the whole OpenBugs category added the attribute, found all nine passing and
    /// removed their [OpenBugs] tags.
    /// </remarks>
    [TestClass]
    public class np_isinf_Test
    {
        /// <summary>
        /// np.isinf marks exactly the two infinities of a float64 vector: π, NaN and 0 are not infinite.
        /// </summary>
        [TestMethod]
        public void np_isinf_1D()
        {
            var arr = new NDArray(new[] { 1.0, System.Math.PI, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0.0 });
            var result = np.isinf(arr);

            Assert.AreEqual(6, result.size);
            Assert.AreEqual(1, result.ndim);
            // Only infinity values should return true
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, false, true, true, false, false }));
        }

        /// <summary>
        /// np.isinf keeps the input's 2-D shape and marks ±inf only: double.MaxValue and NaN are not infinite.
        /// </summary>
        [TestMethod]
        public void np_isinf_2D()
        {
            var arr = new NDArray(new[] {
                double.PositiveInfinity, 1.0, double.NaN,
                double.NegativeInfinity, 2.0, double.MaxValue
            }, new Shape(2, 3));
            var result = np.isinf(arr);

            Assert.AreEqual(6, result.size);
            Assert.AreEqual(2, result.ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { true, false, false, true, false, false }));
        }

        /// <summary>
        /// np.isinf of float32 returns a bool array that marks ±inf only; NaN and both signed zeros are not infinite.
        /// </summary>
        [TestMethod]
        public void np_isinf_Float32()
        {
            // Test with float32 array
            var arr = np.array(new float[] { 1.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f, -0.0f });
            var result = np.isinf(arr);

            Assert.AreEqual(typeof(bool), result.dtype);
            // Only +Inf and -Inf should be true
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, true, true, false, false, false }));
        }

        /// <summary>
        /// Integer dtypes have no infinity: np.isinf is all False for int32, uint8, int64 and int16, their extreme values
        /// included (an extreme integer must not be mistaken for an overflowed float).
        /// </summary>
        [TestMethod]
        public void np_isinf_IntegerTypes_AlwaysFalse()
        {
            // Integer types cannot be infinity
            Assert.IsFalse(np.any(np.isinf(np.array(new int[] { 0, 1, -1, int.MaxValue, int.MinValue }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new byte[] { 0, 1, 255 }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new long[] { 0, long.MaxValue, long.MinValue }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new short[] { 0, short.MaxValue, short.MinValue }))));
        }

        /// <summary>
        /// np.isinf of 0-d arrays: True for ±inf, False for a finite value and for NaN.
        /// </summary>
        [TestMethod]
        public void np_isinf_Scalar()
        {
            Assert.IsFalse(np.isinf(np.array(1.0)).GetBoolean());
            Assert.IsFalse(np.isinf(np.array(double.NaN)).GetBoolean());
            Assert.IsTrue(np.isinf(np.array(double.PositiveInfinity)).GetBoolean());
            Assert.IsTrue(np.isinf(np.array(double.NegativeInfinity)).GetBoolean());
        }

        /// <summary>
        /// np.isinf of an empty float64 array is an empty bool array, not an error.
        /// </summary>
        [TestMethod]
        public void np_isinf_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.isinf(empty);
            Assert.AreEqual(0, result.size);
            Assert.AreEqual(typeof(bool), result.dtype);
        }

        /// <summary>
        /// np.isinf of a stepped view (<c>[::2]</c>) reads only the elements the view selects: the infinities and the NaN
        /// sit between them, so every result is False.
        /// </summary>
        [TestMethod]
        public void np_isinf_SlicedArray()
        {
            // Test with non-contiguous (sliced) array
            var arr = np.array(new double[] { 1.0, double.PositiveInfinity, 2.0, double.NegativeInfinity, 3.0, double.NaN });
            var sliced = arr["::2"];  // [1.0, 2.0, 3.0]
            var result = np.isinf(sliced);

            // All should be false since the sliced values are all finite numbers
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, false, false }));
        }

        /// <summary>
        /// NaN is not an infinity: np.isinf(NaN) is False (both are non-finite, so a predicate built on isfinite would
        /// get this wrong).
        /// </summary>
        [TestMethod]
        public void np_isinf_NaNIsNotInfinity()
        {
            // Important: NaN is NOT infinity
            var arr = np.array(new double[] { double.NaN });
            var result = np.isinf(arr);
            Assert.IsFalse(result.GetBoolean(0));
        }

        /// <summary>
        /// The largest finite doubles are not infinities: np.isinf(double.MaxValue) and np.isinf(double.MinValue) are
        /// False.
        /// </summary>
        [TestMethod]
        public void np_isinf_MaxValueIsNotInfinity()
        {
            // Important: MaxValue/MinValue are NOT infinity
            var arr = np.array(new double[] { double.MaxValue, double.MinValue });
            var result = np.isinf(arr);
            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
        }
    }
}
