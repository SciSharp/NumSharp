using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Tests for np.isinf - test element-wise for positive or negative infinity.
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.isinf.html
    /// </summary>
    public class np_isinf_Test
    {
        [Test]
        public void np_isinf_1D()
        {
            var arr = new NDArray(new[] { 1.0, Math.PI, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0.0 });
            var result = np.isinf(arr);

            Assert.AreEqual(6, result.size);
            Assert.AreEqual(1, result.ndim);
            // Only infinity values should return true
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, false, true, true, false, false }));
        }

        [Test]
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

        [Test]
        public void np_isinf_Float32()
        {
            // Test with float32 array
            var arr = np.array(new float[] { 1.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f, -0.0f });
            var result = np.isinf(arr);

            Assert.AreEqual(typeof(bool), result.dtype);
            // Only +Inf and -Inf should be true
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, true, true, false, false, false }));
        }

        [Test]
        public void np_isinf_IntegerTypes_AlwaysFalse()
        {
            // Integer types cannot be infinity
            Assert.IsFalse(np.any(np.isinf(np.array(new int[] { 0, 1, -1, int.MaxValue, int.MinValue }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new byte[] { 0, 1, 255 }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new long[] { 0, long.MaxValue, long.MinValue }))));
            Assert.IsFalse(np.any(np.isinf(np.array(new short[] { 0, short.MaxValue, short.MinValue }))));
        }

        [Test]
        public void np_isinf_Scalar()
        {
            Assert.IsFalse(np.isinf(np.array(1.0)).GetBoolean());
            Assert.IsFalse(np.isinf(np.array(double.NaN)).GetBoolean());
            Assert.IsTrue(np.isinf(np.array(double.PositiveInfinity)).GetBoolean());
            Assert.IsTrue(np.isinf(np.array(double.NegativeInfinity)).GetBoolean());
        }

        [Test]
        public void np_isinf_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.isinf(empty);
            Assert.AreEqual(0, result.size);
            Assert.AreEqual(typeof(bool), result.dtype);
        }

        [Test]
        public void np_isinf_SlicedArray()
        {
            // Test with non-contiguous (sliced) array
            var arr = np.array(new double[] { 1.0, double.PositiveInfinity, 2.0, double.NegativeInfinity, 3.0, double.NaN });
            var sliced = arr["::2"];  // [1.0, 2.0, 3.0]
            var result = np.isinf(sliced);

            // All should be false since the sliced values are all finite numbers
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] { false, false, false }));
        }

        [Test]
        public void np_isinf_NaNIsNotInfinity()
        {
            // Important: NaN is NOT infinity
            var arr = np.array(new double[] { double.NaN });
            var result = np.isinf(arr);
            Assert.IsFalse(result.GetBoolean(0));
        }

        [Test]
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
