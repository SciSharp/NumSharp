using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest;


namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Tests for np.isfinite - test element-wise for finiteness (not infinity and not NaN).
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
    /// </summary>
    public class np_isfinite_Test
    {
        [TestMethod]
        public void np_isfinite_1D()
        {
            var np1 = new NDArray(new[] {1.0, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np2 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN});
            var np3 = new NDArray(new int[] {333, 444});

            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np1).Data<bool>(), new[] {true, true, true, true, true, true, false}));
            Assert.AreEqual(7, np.isfinite(np1).size);
            Assert.AreEqual(1, np.isfinite(np1).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np2).Data<bool>(), new[] {false, false, true, true, true, true, true, false}));
            Assert.AreEqual(8, np.isfinite(np2).size);
            Assert.AreEqual(1, np.isfinite(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np3).Data<bool>(), new[] {true, true}));
            Assert.AreEqual(2, np.isfinite(np3).size);
            Assert.AreEqual(1, np.isfinite(np3).ndim);
        }

        [TestMethod]
        public void np_isfinite_2D()
        {
            var np1 = new NDArray(new[] {Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 3));
            var np2 = new NDArray(typeof(double), new Shape(39, 17));
            var np3 = new NDArray(new[] {double.NegativeInfinity, double.PositiveInfinity, Math.PI, Math.E, 42, double.MaxValue, double.MinValue, double.NaN}, new Shape(2, 4));
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np1).Data<bool>(), new[] {true, true, true, true, true, false}));
            Assert.AreEqual(6, np.isfinite(np1).size);
            Assert.AreEqual(2, np.isfinite(np1).ndim);
            Assert.IsTrue(np.all(np.isfinite(np2)));
            Assert.AreEqual(39 * 17, np.isfinite(np2).size);
            Assert.AreEqual(2, np.isfinite(np2).ndim);
            Assert.IsTrue(Enumerable.SequenceEqual(np.isfinite(np3).Data<bool>(), new[] {false, false, true, true, true, true, true, false}));
            Assert.AreEqual(8, np.isfinite(np3).size);
            Assert.AreEqual(2, np.isfinite(np3).ndim);
        }

        [TestMethod]
        public void np_isfinite_Float32()
        {
            // Test with float32 array
            var arr = np.array(new float[] {1.0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, 0.0f, -0.0f});
            var result = np.isfinite(arr);

            Assert.AreEqual(typeof(bool), result.dtype);
            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] {true, false, false, false, true, true}));
        }

        [TestMethod]
        public void np_isfinite_IntegerTypes_AlwaysTrue()
        {
            // All integer types are always finite
            Assert.IsTrue(np.all(np.isfinite(np.array(new int[] {0, 1, -1, 100, -100}))));
            Assert.IsTrue(np.all(np.isfinite(np.array(new byte[] {0, 1, 255}))));
            Assert.IsTrue(np.all(np.isfinite(np.array(new long[] {0, long.MaxValue, long.MinValue}))));
            Assert.IsTrue(np.all(np.isfinite(np.array(new short[] {0, short.MaxValue, short.MinValue}))));
        }

        [TestMethod]
        public void np_isfinite_Scalar()
        {
            Assert.IsTrue(np.isfinite(np.array(1.0)).GetBoolean());
            Assert.IsFalse(np.isfinite(np.array(double.NaN)).GetBoolean());
            Assert.IsFalse(np.isfinite(np.array(double.PositiveInfinity)).GetBoolean());
            Assert.IsFalse(np.isfinite(np.array(double.NegativeInfinity)).GetBoolean());
        }

        [TestMethod]
        public void np_isfinite_EmptyArray()
        {
            var empty = np.array(new double[0]);
            var result = np.isfinite(empty);
            Assert.AreEqual(0, result.size);
            Assert.AreEqual(typeof(bool), result.dtype);
        }

        [TestMethod]
        public void np_isfinite_SlicedArray()
        {
            // Test with non-contiguous (sliced) array
            var arr = np.array(new double[] {1.0, double.PositiveInfinity, 2.0, double.NaN, 3.0, double.NegativeInfinity});
            var sliced = arr["::2"];  // [1.0, 2.0, 3.0]
            var result = np.isfinite(sliced);

            Assert.IsTrue(Enumerable.SequenceEqual(result.Data<bool>(), new[] {true, true, true}));
        }
    }
}
