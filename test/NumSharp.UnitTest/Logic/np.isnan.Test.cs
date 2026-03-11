using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Tests for np.isnan - test element-wise for NaN.
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
    /// </summary>
    public class np_isnan_Test
    {
        [Test]
        public void np_isnan_1D()
        {
            var np1 = new NDArray(new[] {1.0, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np2 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np3 = new NDArray(new int[] {333, 444});

            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np1).Data<bool>(), new[] {false, false, false, false, false, false, true}));
            Assert.AreEqual(7, np.isnan(np1).size);
            Assert.AreEqual(1, np.isnan(np1).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np2).Data<bool>(), new[] {false, false, false, false, false, false, false, true}));
            Assert.AreEqual(8, np.isnan(np2).size);
            Assert.AreEqual(1, np.isnan(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np3).Data<bool>(), new[] {false, false}));
            Assert.AreEqual(2, np.isnan(np3).size);
            Assert.AreEqual(1, np.isnan(np3).ndim);
        }

        [Test]
        public void np_isnan_2D()
        {
            var np1 = new NDArray(new[] {Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 3));
            var np2 = new NDArray(typeof(double), new Shape(39, 17));
            var np3 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 4));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np1).Data<bool>(), new[] {false, false, false, false, false, true}));
            Assert.AreEqual(6, np.isnan(np1).size);
            Assert.AreEqual(2, np.isnan(np1).ndim);
            Assert.IsFalse(np.all(np.isnan(np2)));
            Assert.AreEqual(39 * 17, np.isnan(np2).size);
            Assert.AreEqual(2, np.isnan(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isnan(np3).Data<bool>(), new[] {false, false, false, false, false, false, false, true}));
            Assert.AreEqual(8, np.isnan(np3).size);
            Assert.AreEqual(2, np.isnan(np3).ndim);
        }

        [Test]
        public void np_isnan_Float32()
        {
            // Test with float32 array
            var arr = np.array(new float[] {1.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f, -0.0f});
            var result = np.isnan(arr);

            Assert.AreEqual(typeof(bool), result.dtype);
            // Only NaN should be true, infinity is NOT NaN
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] {false, false, false, true, false, false}));
        }

        [Test]
        public void np_isnan_IntegerTypes_AlwaysFalse()
        {
            // Integer types cannot be NaN
            Assert.IsFalse(np.any(np.isnan(np.array(new int[] {0, 1, -1, 100, -100}))));
            Assert.IsFalse(np.any(np.isnan(np.array(new byte[] {0, 1, 255}))));
            Assert.IsFalse(np.any(np.isnan(np.array(new long[] {0, long.MaxValue, long.MinValue}))));
            Assert.IsFalse(np.any(np.isnan(np.array(new short[] {0, short.MaxValue, short.MinValue}))));
        }

        [Test]
        public void np_isnan_Scalar()
        {
            Assert.IsFalse(np.isnan(np.array(1.0)).GetBoolean());
            Assert.IsTrue(np.isnan(np.array(double.NaN)).GetBoolean());
            // Infinity is NOT NaN
            Assert.IsFalse(np.isnan(np.array(double.PositiveInfinity)).GetBoolean());
            Assert.IsFalse(np.isnan(np.array(double.NegativeInfinity)).GetBoolean());
        }

        [Test]
        public void np_isnan_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.isnan(empty);
            Assert.AreEqual(0, result.size);
            Assert.AreEqual(typeof(bool), result.dtype);
        }

        [Test]
        public void np_isnan_SlicedArray()
        {
            // Test with non-contiguous (sliced) array
            var arr = np.array(new double[] {1.0, double.NaN, 2.0, double.PositiveInfinity, 3.0, double.NaN});
            var sliced = arr["::2"];  // [1.0, 2.0, 3.0]
            var result = np.isnan(sliced);

            // All should be false since the sliced values are all finite numbers
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] {false, false, false}));
        }

        [Test]
        public void np_isnan_InfinityIsNotNaN()
        {
            // Important: Infinity is NOT NaN
            var arr = np.array(new double[] {double.PositiveInfinity, double.NegativeInfinity});
            var result = np.isnan(arr);
            Assert.IsFalse(result.GetBoolean(0));
            Assert.IsFalse(result.GetBoolean(1));
        }
    }
}
