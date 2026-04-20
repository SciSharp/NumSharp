using System;
using System.Diagnostics;
using NumSharp.Backends.Kernels;
using NumSharp.UnitTest.Utilities;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Backends.Kernels
{
    /// <summary>
    /// Tests for SIMD-optimized np.where implementation.
    /// Verifies correctness of the SIMD path for all supported dtypes.
    /// </summary>
    [TestClass]
    public class WhereSimdTests
    {
        #region SIMD Correctness

        [TestMethod]
        public void Where_Simd_Float32_Correctness()
        {
            var rng = np.random.RandomState(42);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = rng.rand(size).astype(NPTypeCode.Single);
            var y = rng.rand(size).astype(NPTypeCode.Single);

            var result = np.where(cond, x, y);

            // Verify correctness manually
            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (float)x[i] : (float)y[i];
                Assert.AreEqual(expected, (float)result[i], 1e-6f, $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Float64_Correctness()
        {
            var rng = np.random.RandomState(43);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = rng.rand(size);
            var y = rng.rand(size);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (double)x[i] : (double)y[i];
                Assert.AreEqual(expected, (double)result[i], 1e-10, $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Int32_Correctness()
        {
            var rng = np.random.RandomState(44);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = rng.randint(0, 1000, new[] { size });
            var y = rng.randint(0, 1000, new[] { size });

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (int)x[i] : (int)y[i];
                Assert.AreEqual(expected, (int)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Int64_Correctness()
        {
            var rng = np.random.RandomState(45);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = np.arange(size).astype(NPTypeCode.Int64);
            var y = np.arange(size, size * 2).astype(NPTypeCode.Int64);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (long)x[i] : (long)y[i];
                Assert.AreEqual(expected, (long)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Byte_Correctness()
        {
            var rng = np.random.RandomState(46);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = (rng.rand(size) * 255).astype(NPTypeCode.Byte);
            var y = (rng.rand(size) * 255).astype(NPTypeCode.Byte);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (byte)x[i] : (byte)y[i];
                Assert.AreEqual(expected, (byte)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Int16_Correctness()
        {
            var rng = np.random.RandomState(47);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = np.arange(size).astype(NPTypeCode.Int16);
            var y = np.arange(size, size * 2).astype(NPTypeCode.Int16);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (short)x[i] : (short)y[i];
                Assert.AreEqual(expected, (short)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_UInt16_Correctness()
        {
            var rng = np.random.RandomState(48);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = np.arange(size).astype(NPTypeCode.UInt16);
            var y = np.arange(size, size * 2).astype(NPTypeCode.UInt16);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (ushort)x[i] : (ushort)y[i];
                Assert.AreEqual(expected, (ushort)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_UInt32_Correctness()
        {
            var rng = np.random.RandomState(49);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = np.arange(size).astype(NPTypeCode.UInt32);
            var y = np.arange(size, size * 2).astype(NPTypeCode.UInt32);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (uint)x[i] : (uint)y[i];
                Assert.AreEqual(expected, (uint)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_UInt64_Correctness()
        {
            var rng = np.random.RandomState(50);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = np.arange(size).astype(NPTypeCode.UInt64);
            var y = np.arange(size, size * 2).astype(NPTypeCode.UInt64);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (ulong)x[i] : (ulong)y[i];
                Assert.AreEqual(expected, (ulong)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Boolean_Correctness()
        {
            var rng = np.random.RandomState(51);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var x = rng.rand(size) > 0.3;
            var y = rng.rand(size) > 0.7;

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (bool)x[i] : (bool)y[i];
                Assert.AreEqual(expected, (bool)result[i], $"Mismatch at index {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_Char_Correctness()
        {
            var rng = np.random.RandomState(52);
            var size = 1000;
            var cond = rng.rand(size) > 0.5;
            var xData = new char[size];
            var yData = new char[size];
            for (int i = 0; i < size; i++)
            {
                xData[i] = (char)('A' + (i % 26));
                yData[i] = (char)('a' + (i % 26));
            }
            var x = np.array(xData);
            var y = np.array(yData);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (char)x[i] : (char)y[i];
                Assert.AreEqual(expected, (char)result[i], $"Mismatch at index {i}");
            }
        }

        #endregion

        #region Path Selection

        [TestMethod]
        public void Where_NonContiguous_Works()
        {
            // Sliced arrays are non-contiguous, should work correctly
            var baseArr = np.arange(20);
            var cond = (baseArr % 2 == 0)["::2"];  // Sliced: [true, true, true, true, true, true, true, true, true, true]
            var x = np.ones(10, NPTypeCode.Int32);
            var y = np.zeros(10, NPTypeCode.Int32);

            var result = np.where(cond, x, y);

            Assert.AreEqual(10, result.size);
            // All true -> all from x
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(1, (int)result[i]);
            }
        }

        [TestMethod]
        public void Where_Broadcast_Works()
        {
            // Broadcasted arrays
            // cond shape (3,) broadcasts to (3,3): [[T,F,T],[T,F,T],[T,F,T]]
            // x shape (3,1) broadcasts to (3,3): [[1,1,1],[2,2,2],[3,3,3]]
            // y shape (1,3) broadcasts to (3,3): [[10,20,30],[10,20,30],[10,20,30]]
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var y = np.array(new int[,] { { 10, 20, 30 } });
            var result = np.where(cond, x, y);

            result.Should().BeShaped(3, 3);
            // Verify values: result[i,j] = cond[j] ? x[i,0] : y[0,j]
            Assert.AreEqual(1, (int)result[0, 0]);   // cond[0]=true -> x=1
            Assert.AreEqual(20, (int)result[0, 1]);  // cond[1]=false -> y=20
            Assert.AreEqual(1, (int)result[0, 2]);   // cond[2]=true -> x=1
            Assert.AreEqual(2, (int)result[1, 0]);   // cond[0]=true -> x=2
            Assert.AreEqual(20, (int)result[1, 1]);  // cond[1]=false -> y=20
        }

        [TestMethod]
        public void Where_Decimal_Works()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new decimal[] { 1.1m, 2.2m, 3.3m });
            var y = np.array(new decimal[] { 10.1m, 20.2m, 30.3m });

            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(decimal), result.dtype);
            Assert.AreEqual(1.1m, (decimal)result[0]);
            Assert.AreEqual(20.2m, (decimal)result[1]);
            Assert.AreEqual(3.3m, (decimal)result[2]);
        }

        [TestMethod]
        public void Where_NonBoolCondition_Works()
        {
            // Non-bool condition requires truthiness check
            var cond = np.array(new[] { 0, 1, 2, 0 });  // int condition
            var result = np.where(cond, 100, -100);

            result.Should().BeOfValues(-100, 100, 100, -100);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void Where_Simd_SmallArray()
        {
            // Array smaller than vector width
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });

            var result = np.where(cond, x, y);

            result.Should().BeOfValues(1, 20, 3);
        }

        [TestMethod]
        public void Where_Simd_VectorAlignedSize()
        {
            var rng = np.random.RandomState(53);
            // Size exactly matches vector width (no scalar tail)
            var size = 32;  // V256 byte count
            var cond = rng.rand(size) > 0.5;
            var x = np.ones(size, NPTypeCode.Byte);
            var y = np.zeros(size, NPTypeCode.Byte);

            var result = np.where(cond, x, y);

            Assert.AreEqual(size, result.size);
            for (int i = 0; i < size; i++)
            {
                var expected = (bool)cond[i] ? (byte)1 : (byte)0;
                Assert.AreEqual(expected, (byte)result[i]);
            }
        }

        [TestMethod]
        public void Where_Simd_WithScalarTail()
        {
            // Size that requires scalar tail processing
            var size = 35;  // 32 + 3 tail for bytes
            var cond = np.ones(size, NPTypeCode.Boolean);
            var x = np.full(size, (byte)255);
            var y = np.zeros(size, NPTypeCode.Byte);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual((byte)255, (byte)result[i], $"Mismatch at {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_AllTrue()
        {
            var size = 100;
            var cond = np.ones(size, NPTypeCode.Boolean);
            var x = np.arange(size);
            var y = np.full(size, -1L);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual((long)i, (long)result[i]);
            }
        }

        [TestMethod]
        public void Where_Simd_AllFalse()
        {
            var size = 100;
            var cond = np.zeros(size, NPTypeCode.Boolean);
            var x = np.arange(size);
            var y = np.full(size, -1L);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual(-1L, (long)result[i]);
            }
        }

        [TestMethod]
        public void Where_Simd_Alternating()
        {
            var size = 100;
            var condData = new bool[size];
            for (int i = 0; i < size; i++)
                condData[i] = i % 2 == 0;
            var cond = np.array(condData);
            var x = np.ones(size, NPTypeCode.Int32);
            var y = np.zeros(size, NPTypeCode.Int32);

            var result = np.where(cond, x, y);

            for (int i = 0; i < size; i++)
            {
                Assert.AreEqual(i % 2 == 0 ? 1 : 0, (int)result[i], $"Mismatch at {i}");
            }
        }

        [TestMethod]
        public void Where_Simd_NaN_Propagates()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new[] { double.NaN, 1.0, 2.0 });
            var y = np.array(new[] { 0.0, double.NaN, 0.0 });

            var result = np.where(cond, x, y);

            Assert.IsTrue(double.IsNaN((double)result[0]));  // NaN from x
            Assert.IsTrue(double.IsNaN((double)result[1]));  // NaN from y
            Assert.AreEqual(2.0, (double)result[2], 1e-10);
        }

        [TestMethod]
        public void Where_Simd_Infinity()
        {
            var cond = np.array(new[] { true, false, true, false });
            var x = np.array(new[] { double.PositiveInfinity, 0.0, double.NegativeInfinity, 0.0 });
            var y = np.array(new[] { 0.0, double.PositiveInfinity, 0.0, double.NegativeInfinity });

            var result = np.where(cond, x, y);

            Assert.AreEqual(double.PositiveInfinity, (double)result[0]);
            Assert.AreEqual(double.PositiveInfinity, (double)result[1]);
            Assert.AreEqual(double.NegativeInfinity, (double)result[2]);
            Assert.AreEqual(double.NegativeInfinity, (double)result[3]);
        }

        #endregion

        #region Performance Sanity Check

        [TestMethod]
        public void Where_Simd_LargeArray_Correctness()
        {
            var rng = np.random.RandomState(54);
            var size = 100_000;
            var cond = rng.rand(size) > 0.5;
            var x = np.ones(size, NPTypeCode.Double);
            var y = np.zeros(size, NPTypeCode.Double);

            var result = np.where(cond, x, y);

            // Spot check
            for (int i = 0; i < 100; i++)
            {
                var expected = (bool)cond[i] ? 1.0 : 0.0;
                Assert.AreEqual(expected, (double)result[i], 1e-10);
            }

            // Check last few elements (scalar tail)
            for (int i = size - 10; i < size; i++)
            {
                var expected = (bool)cond[i] ? 1.0 : 0.0;
                Assert.AreEqual(expected, (double)result[i], 1e-10);
            }
        }

        #endregion

        #region 2D/Multi-dimensional

        [TestMethod]
        public void Where_Simd_2D_Contiguous()
        {
            var rng = np.random.RandomState(55);
            // 2D contiguous array should use SIMD
            var shape = new[] { 100, 100 };
            var cond = rng.rand(shape) > 0.5;
            var x = np.ones(shape, NPTypeCode.Int32);
            var y = np.zeros(shape, NPTypeCode.Int32);

            var result = np.where(cond, x, y);

            result.Should().BeShaped(100, 100);

            // Spot check
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    var expected = (bool)cond[i, j] ? 1 : 0;
                    Assert.AreEqual(expected, (int)result[i, j]);
                }
            }
        }

        [TestMethod]
        public void Where_Simd_3D_Contiguous()
        {
            var rng = np.random.RandomState(56);
            var shape = new[] { 10, 20, 30 };
            var cond = rng.rand(shape) > 0.5;
            var x = np.ones(shape, NPTypeCode.Single);
            var y = np.zeros(shape, NPTypeCode.Single);

            var result = np.where(cond, x, y);

            result.Should().BeShaped(10, 20, 30);

            // Spot check
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    for (int k = 0; k < 5; k++)
                    {
                        var expected = (bool)cond[i, j, k] ? 1.0f : 0.0f;
                        Assert.AreEqual(expected, (float)result[i, j, k], 1e-6f);
                    }
                }
            }
        }

        #endregion
    }
}
