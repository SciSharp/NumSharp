using System;
using System.Linq;
using TUnit.Core;
using NumSharp.UnitTest.Utilities;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Battle tests for np.where - edge cases, strided arrays, views, etc.
    /// </summary>
    public class np_where_BattleTest
    {
        #region Strided/Sliced Arrays

        [Test]
        public void Where_SlicedCondition()
        {
            // Sliced condition array
            var arr = np.arange(10);
            var cond = (arr % 2 == 0)["::2"];  // Every other even check
            var x = np.ones(5, NPTypeCode.Int32);
            var y = np.zeros(5, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            // Should work with sliced condition
            Assert.AreEqual(5, result.size);
        }

        [Test]
        public void Where_SlicedXY()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.arange(6)["::2"];  // [0, 2, 4]
            var y = np.arange(6)["1::2"]; // [1, 3, 5]
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(0L, 3L, 4L);
        }

        [Test]
        public void Where_TransposedArrays()
        {
            var cond = np.array(new bool[,] { { true, false }, { false, true } }).T;
            var x = np.array(new int[,] { { 1, 2 }, { 3, 4 } }).T;
            var y = np.array(new int[,] { { 10, 20 }, { 30, 40 } }).T;
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 2);
            // After transpose: cond[0,0]=T, cond[0,1]=F, cond[1,0]=F, cond[1,1]=T
            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(30, (int)result[0, 1]);
            Assert.AreEqual(20, (int)result[1, 0]);
            Assert.AreEqual(4, (int)result[1, 1]);
        }

        [Test]
        public void Where_ReversedSlice()
        {
            var cond = np.array(new[] { true, false, true, false, true });
            var x = np.arange(5)["::-1"];  // [4, 3, 2, 1, 0]
            var y = np.zeros(5, NPTypeCode.Int64);
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(4L, 0L, 2L, 0L, 0L);
        }

        #endregion

        #region Complex Broadcasting

        [Test]
        public void Where_3Way_Broadcasting()
        {
            // cond: (2,1,1), x: (1,3,1), y: (1,1,4) -> result: (2,3,4)
            var cond = np.array(new bool[,,] { { { true } }, { { false } } });
            var x = np.arange(3).reshape(1, 3, 1);
            var y = (np.arange(4) * 10).reshape(1, 1, 4);
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 3, 4);
            // First "page" (cond=True): values from x broadcast
            Assert.AreEqual(0, (long)result[0, 0, 0]);
            Assert.AreEqual(0, (long)result[0, 0, 3]);
            Assert.AreEqual(2, (long)result[0, 2, 0]);
            // Second "page" (cond=False): values from y broadcast
            Assert.AreEqual(0, (long)result[1, 0, 0]);
            Assert.AreEqual(30, (long)result[1, 0, 3]);
            Assert.AreEqual(30, (long)result[1, 2, 3]);
        }

        [Test]
        public void Where_RowVector_ColVector_Broadcast()
        {
            // cond: (1,4), x: (3,1), y: scalar -> result: (3,4)
            var cond = np.array(new bool[,] { { true, false, true, false } });
            var x = np.array(new int[,] { { 1 }, { 2 }, { 3 } });
            var result = np.where(cond, x, 0);

            result.Should().BeShaped(3, 4);
            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(0, (int)result[0, 1]);
            Assert.AreEqual(2, (int)result[1, 0]);
            Assert.AreEqual(0, (int)result[1, 1]);
        }

        #endregion

        #region Numeric Edge Cases

        [Test]
        public void Where_NaN_Values()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new[] { double.NaN, 1.0, double.NaN });
            var y = np.array(new[] { 0.0, double.NaN, 0.0 });
            var result = np.where(cond, x, y);

            Assert.IsTrue(double.IsNaN((double)result[0]));  // from x
            Assert.IsTrue(double.IsNaN((double)result[1]));  // from y
            Assert.IsTrue(double.IsNaN((double)result[2]));  // from x
        }

        [Test]
        public void Where_Infinity_Values()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new[] { double.PositiveInfinity, 1.0 });
            var y = np.array(new[] { 0.0, double.NegativeInfinity });
            var result = np.where(cond, x, y);

            Assert.AreEqual(double.PositiveInfinity, (double)result[0]);
            Assert.AreEqual(double.NegativeInfinity, (double)result[1]);
        }

        [Test]
        public void Where_MaxMin_Values()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new[] { long.MaxValue, 0L });
            var y = np.array(new[] { 0L, long.MinValue });
            var result = np.where(cond, x, y);

            Assert.AreEqual(long.MaxValue, (long)result[0]);
            Assert.AreEqual(long.MinValue, (long)result[1]);
        }

        #endregion

        #region Single Arg Edge Cases

        [Test]
        public void Where_SingleArg_Float_Truthy()
        {
            // 0.0 is falsy, anything else (including -0.0, NaN, Inf) is truthy
            var arr = np.array(new[] { 0.0, 1.0, -1.0, 0.5, -0.0 });
            var result = np.where(arr);

            // Note: -0.0 == 0.0 in IEEE 754, so it's falsy
            result[0].Should().BeOfValues(1L, 2L, 3L);
        }

        [Test]
        public void Where_SingleArg_NaN_IsTruthy()
        {
            // NaN is non-zero, so it's truthy
            var arr = np.array(new[] { 0.0, double.NaN, 0.0 });
            var result = np.where(arr);

            result[0].Should().BeOfValues(1L);
        }

        [Test]
        public void Where_SingleArg_4D()
        {
            var arr = np.zeros(new[] { 2, 2, 2, 2 }, NPTypeCode.Int32);
            arr[0, 1, 0, 1] = 1;
            arr[1, 0, 1, 0] = 1;
            var result = np.where(arr);

            Assert.AreEqual(4, result.Length);  // 4 dimensions
            Assert.AreEqual(2, result[0].size); // 2 non-zero elements
        }

        #endregion

        #region Performance/Stress Tests

        [Test]
        public void Where_LargeArray_Performance()
        {
            var size = 1_000_000;
            var cond = np.random.rand(size) > 0.5;
            var x = np.ones(size, NPTypeCode.Double);
            var y = np.zeros(size, NPTypeCode.Double);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            var result = np.where(cond, x, y);
            sw.Stop();

            Assert.AreEqual(size, result.size);
            // Should complete in reasonable time (< 1 second for 1M elements)
            Assert.IsTrue(sw.ElapsedMilliseconds < 1000, $"Took {sw.ElapsedMilliseconds}ms");
        }

        [Test]
        public void Where_ManyDimensions()
        {
            // 6D array
            var shape = new[] { 2, 3, 2, 2, 2, 3 };
            var cond = np.ones(shape, NPTypeCode.Boolean);
            var x = np.ones(shape, NPTypeCode.Int32);
            var y = np.zeros(shape, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 3, 2, 2, 2, 3);
            Assert.AreEqual(1, (int)result[0, 0, 0, 0, 0, 0]);
        }

        #endregion

        #region Type Conversion Edge Cases

        [Test]
        public void Where_UnsignedOverflow()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new byte[] { 255, 0 });
            var y = np.array(new byte[] { 0, 255 });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(byte), result.dtype);
            Assert.AreEqual((byte)255, (byte)result[0]);
            Assert.AreEqual((byte)255, (byte)result[1]);
        }

        [Test]
        public void Where_Decimal()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new decimal[] { 1.23456789m, 0m });
            var y = np.array(new decimal[] { 0m, 9.87654321m });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(decimal), result.dtype);
            Assert.AreEqual(1.23456789m, (decimal)result[0]);
            Assert.AreEqual(9.87654321m, (decimal)result[1]);
        }

        [Test]
        public void Where_Char()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.array(new char[] { 'A', 'B', 'C' });
            var y = np.array(new char[] { 'X', 'Y', 'Z' });
            var result = np.where(cond, x, y);

            Assert.AreEqual(typeof(char), result.dtype);
            Assert.AreEqual('A', (char)result[0]);
            Assert.AreEqual('Y', (char)result[1]);
            Assert.AreEqual('C', (char)result[2]);
        }

        #endregion

        #region View Behavior

        [Test]
        public void Where_ResultIsNewArray_NotView()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new[] { 1, 2 });
            var y = np.array(new[] { 10, 20 });
            var result = np.where(cond, x, y);

            // Modify original, result should not change
            x[0] = 999;
            Assert.AreEqual(1, (int)result[0], "Result should be independent of x");

            y[1] = 999;
            Assert.AreEqual(20, (int)result[1], "Result should be independent of y");
        }

        [Test]
        public void Where_ModifyResult_DoesNotAffectInputs()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new[] { 1, 2 });
            var y = np.array(new[] { 10, 20 });
            var result = np.where(cond, x, y);

            result[0] = 999;
            Assert.AreEqual(1, (int)x[0], "x should not be modified");
            Assert.AreEqual(10, (int)y[0], "y should not be modified");
        }

        #endregion

        #region Alternating Patterns

        [Test]
        public void Where_Checkerboard_Pattern()
        {
            // Create checkerboard condition
            var cond = np.zeros(new[] { 4, 4 }, NPTypeCode.Boolean);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    cond[i, j] = (i + j) % 2 == 0;

            var x = np.ones(new[] { 4, 4 }, NPTypeCode.Int32);
            var y = np.zeros(new[] { 4, 4 }, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            // Verify checkerboard pattern
            Assert.AreEqual(1, (int)result[0, 0]);
            Assert.AreEqual(0, (int)result[0, 1]);
            Assert.AreEqual(0, (int)result[1, 0]);
            Assert.AreEqual(1, (int)result[1, 1]);
        }

        [Test]
        public void Where_StripedPattern()
        {
            // Every row alternates between all True and all False
            var cond = np.zeros(new[] { 4, 4 }, NPTypeCode.Boolean);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    cond[i, j] = i % 2 == 0;

            var x = np.full(new[] { 4, 4 }, 1);
            var y = np.full(new[] { 4, 4 }, 0);
            var result = np.where(cond, x, y);

            // Rows 0, 2 should be 1; rows 1, 3 should be 0
            for (int j = 0; j < 4; j++)
            {
                Assert.AreEqual(1, (int)result[0, j]);
                Assert.AreEqual(0, (int)result[1, j]);
                Assert.AreEqual(1, (int)result[2, j]);
                Assert.AreEqual(0, (int)result[3, j]);
            }
        }

        #endregion
    }
}
