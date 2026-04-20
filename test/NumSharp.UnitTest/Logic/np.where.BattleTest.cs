using System;
using System.Linq;
using NumSharp.UnitTest.Utilities;
using Assert = Microsoft.VisualStudio.TestTools.UnitTesting.Assert;

namespace NumSharp.UnitTest.Logic
{
    /// <summary>
    /// Battle tests for np.where - edge cases, strided arrays, views, etc.
    ///
    /// These tests verify NumSharp behavior against NumPy 2.4.2.
    ///
    /// KNOWN DIFFERENCES FROM NUMPY 2.x:
    ///
    /// 1. Scalar Type Promotion (NEP50):
    ///    NumPy 2.x uses "weak scalar" semantics where Python scalars adopt array dtype.
    ///    NumSharp uses C# semantics where literals have fixed types (int=int32, etc).
    ///
    ///    Example: np.where(cond, 1, uint8_array)
    ///    - NumPy 2.x: returns uint8 (weak scalar rule)
    ///    - NumSharp: returns int32 (C# int literal is int32)
    ///
    /// 2. Python int Scalar Default:
    ///    - NumPy: Python int → int64 (platform default)
    ///    - NumSharp: C# int literal → int32
    ///
    /// 3. Missing sbyte (int8) support:
    ///    NumSharp does not support sbyte arrays (throws NotSupportedException).
    /// </summary>
    [TestClass]
    public class np_where_BattleTest
    {
        #region Strided/Sliced Arrays

        [TestMethod]
        public void Where_SlicedCondition()
        {
            // Sliced condition array (non-contiguous)
            var arr = np.arange(10);
            var cond = (arr % 2 == 0)["::2"];  // Every other even check: [T,T,T,T,T]
            var x = np.ones(5, NPTypeCode.Int32);
            var y = np.zeros(5, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            Assert.AreEqual(5, result.size);
            result.Should().BeOfValues(1, 1, 1, 1, 1);
        }

        [TestMethod]
        public void Where_SlicedXY()
        {
            var cond = np.array(new[] { true, false, true });
            var x = np.arange(6)["::2"];  // [0, 2, 4]
            var y = np.arange(6)["1::2"]; // [1, 3, 5]
            var result = np.where(cond, x, y);

            result.Should().BeOfValues(0L, 3L, 4L);
        }

        [TestMethod]
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

        [TestMethod]
        public void Where_ReversedSlice()
        {
            var cond = np.array(new[] { true, false, true, false, true });
            var x = np.arange(5)["::-1"];  // [4, 3, 2, 1, 0]
            var y = np.zeros(5, NPTypeCode.Int64);
            var result = np.where(cond, x, y);

            // NumPy: [4, 0, 2, 0, 0]
            result.Should().BeOfValues(4L, 0L, 2L, 0L, 0L);
        }

        #endregion

        #region Complex Broadcasting

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
        public void Where_ScalarCondition_True()
        {
            // NumPy: np.where(True, [1,2,3], [4,5,6]) -> [1,2,3]
            var result = np.where(np.array(true), np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }));
            result.Should().BeOfValues(1, 2, 3);
        }

        [TestMethod]
        public void Where_ScalarCondition_False()
        {
            // NumPy: np.where(False, [1,2,3], [4,5,6]) -> [4,5,6]
            var result = np.where(np.array(false), np.array(new[] { 1, 2, 3 }), np.array(new[] { 4, 5, 6 }));
            result.Should().BeOfValues(4, 5, 6);
        }

        #endregion

        #region Non-Boolean Conditions (Truthy/Falsy)

        [TestMethod]
        public void Where_IntegerCondition_ZeroIsFalsy()
        {
            // NumPy: 0 is falsy, non-zero is truthy
            var cond = np.array(new[] { 0, 1, 2, -1, 0 });
            var x = np.ones(5);
            var y = np.zeros(5);
            var result = np.where(cond, x, y);

            // NumPy: [0, 1, 1, 1, 0]
            result.Should().BeOfValues(0.0, 1.0, 1.0, 1.0, 0.0);
        }

        [TestMethod]
        public void Where_FloatCondition_ZeroIsFalsy()
        {
            // NumPy: 0.0 is falsy
            var cond = np.array(new[] { 0.0, 0.5, 1.0, -0.1, 0.0 });
            var x = np.ones(5);
            var y = np.zeros(5);
            var result = np.where(cond, x, y);

            // NumPy: [0, 1, 1, 1, 0]
            result.Should().BeOfValues(0.0, 1.0, 1.0, 1.0, 0.0);
        }

        [TestMethod]
        public void Where_NaN_IsTruthy()
        {
            // NumPy: NaN is truthy (non-zero)
            var cond = np.array(new[] { 0.0, double.NaN, 1.0 });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            // NumPy: [10, 2, 3] (NaN is truthy)
            result.Should().BeOfValues(10, 2, 3);
        }

        [TestMethod]
        public void Where_Infinity_IsTruthy()
        {
            // NumPy: Inf and -Inf are truthy
            var cond = np.array(new[] { 0.0, double.PositiveInfinity, double.NegativeInfinity });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            // NumPy: [10, 2, 3]
            result.Should().BeOfValues(10, 2, 3);
        }

        [TestMethod]
        public void Where_NegativeZero_IsFalsy()
        {
            // NumPy: -0.0 == 0.0 in IEEE 754, so it's falsy
            var cond = np.array(new[] { 0.0, -0.0, 1.0 });
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            // NumPy: [10, 20, 3] (both 0.0 and -0.0 are falsy)
            result.Should().BeOfValues(10, 20, 3);
        }

        #endregion

        #region Numeric Edge Cases

        [TestMethod]
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

        [TestMethod]
        public void Where_Infinity_Values()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new[] { double.PositiveInfinity, 1.0 });
            var y = np.array(new[] { 0.0, double.NegativeInfinity });
            var result = np.where(cond, x, y);

            Assert.AreEqual(double.PositiveInfinity, (double)result[0]);
            Assert.AreEqual(double.NegativeInfinity, (double)result[1]);
        }

        [TestMethod]
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

        [TestMethod]
        public void Where_SingleArg_Float_Truthy()
        {
            // 0.0 is falsy, anything else (including -0.0, NaN, Inf) is truthy
            // Note: -0.0 == 0.0 in IEEE 754, so it's falsy
            var arr = np.array(new[] { 0.0, 1.0, -1.0, 0.5, -0.0 });
            var result = np.where(arr);

            // NumPy: indices [1, 2, 3] (-0.0 is falsy)
            result[0].Should().BeOfValues(1L, 2L, 3L);
        }

        [TestMethod]
        public void Where_SingleArg_NaN_IsTruthy()
        {
            // NaN is non-zero, so it's truthy
            var arr = np.array(new[] { 0.0, double.NaN, 0.0 });
            var result = np.where(arr);

            result[0].Should().BeOfValues(1L);
        }

        [TestMethod]
        public void Where_SingleArg_Infinity_IsTruthy()
        {
            // Inf values are truthy
            var arr = np.array(new[] { 0.0, double.PositiveInfinity, double.NegativeInfinity, 0.0 });
            var result = np.where(arr);

            result[0].Should().BeOfValues(1L, 2L);
        }

        [TestMethod]
        public void Where_SingleArg_4D()
        {
            var arr = np.zeros(new[] { 2, 2, 2, 2 }, NPTypeCode.Int32);
            arr[0, 1, 0, 1] = 1;
            arr[1, 0, 1, 0] = 1;
            var result = np.where(arr);

            Assert.AreEqual(4, result.Length);  // 4 dimensions
            Assert.AreEqual(2, result[0].size); // 2 non-zero elements
        }

        [TestMethod]
        public void Where_SingleArg_ReturnsInt64Indices()
        {
            // NumPy returns int64 for indices
            var arr = np.array(new[] { 0, 1, 0, 2 });
            var result = np.where(arr);

            Assert.AreEqual(typeof(long), result[0].dtype);
        }

        #endregion

        #region 0D Scalar Arrays

        [TestMethod]
        public void Where_0D_AllScalars_Returns0D()
        {
            // NumPy: when all inputs are 0D, result is 0D
            var cond = np.array(true).reshape();  // 0D
            var x = np.array(42).reshape();       // 0D
            var y = np.array(99).reshape();       // 0D
            var result = np.where(cond, x, y);

            Assert.AreEqual(0, result.ndim);
            Assert.AreEqual(42, (int)result.GetValue(0));
        }

        [TestMethod]
        public void Where_0D_Cond_With_1D_Arrays()
        {
            // 0D condition broadcasts to match x/y shape
            var cond = np.array(true).reshape();  // 0D
            var x = np.array(new[] { 1, 2, 3 });
            var y = np.array(new[] { 10, 20, 30 });
            var result = np.where(cond, x, y);

            result.Should().BeShaped(3);
            result.Should().BeOfValues(1, 2, 3);
        }

        #endregion

        #region Type Promotion (Array-to-Array)

        [TestMethod]
        public void Where_TypePromotion_Bool_Int16()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new bool[] { true, false });
            var y = np.array(new short[] { 10, 20 });
            var result = np.where(cond, x, y);

            // NumPy: int16
            Assert.AreEqual(typeof(short), result.dtype);
        }

        [TestMethod]
        public void Where_TwoScalars_Byte_StaysByte()
        {
            // C# byte (like np.uint8) stays byte, not widened to int64
            var cond = np.array(new[] { true, false });
            var result = np.where(cond, (byte)1, (byte)0);

            Assert.AreEqual(typeof(byte), result.dtype);
            Assert.AreEqual((byte)1, (byte)result[0]);
            Assert.AreEqual((byte)0, (byte)result[1]);
        }

        [TestMethod]
        public void Where_TwoScalars_Short_StaysShort()
        {
            // C# short (like np.int16) stays short
            var cond = np.array(new[] { true, false });
            var result = np.where(cond, (short)100, (short)200);

            Assert.AreEqual(typeof(short), result.dtype);
        }

        [TestMethod]
        public void Where_TypePromotion_Int32_UInt32()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new int[] { 1, 2 });
            var y = np.array(new uint[] { 10, 20 });
            var result = np.where(cond, x, y);

            // NumPy: int64 (to accommodate both signed and unsigned 32-bit range)
            Assert.AreEqual(typeof(long), result.dtype);
        }

        [TestMethod]
        public void Where_TypePromotion_Int64_UInt64()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new long[] { 1, 2 });
            var y = np.array(new ulong[] { 10, 20 });
            var result = np.where(cond, x, y);

            // NumPy: float64 (no integer type can hold both int64 and uint64 full range)
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [TestMethod]
        public void Where_TypePromotion_UInt8_Float32()
        {
            var cond = np.array(new[] { true, false });
            var x = np.array(new byte[] { 1, 2 });
            var y = np.array(new float[] { 10.5f, 20.5f });
            var result = np.where(cond, x, y);

            // NumPy: float32
            Assert.AreEqual(typeof(float), result.dtype);
        }

        #endregion

        #region Performance/Stress Tests

        [TestMethod]
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

        [TestMethod]
        public void Where_ManyDimensions()
        {
            // 6D array
            var shape = new[] { 2, 3, 2, 2, 2, 3 };
            var cond = np.ones(shape, NPTypeCode.Boolean);
            var x = np.ones(shape, NPTypeCode.Int32);
            var y = np.zeros(shape, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 3, 2, 2, 2, 3);
            Assert.AreEqual(144, result.size);
            Assert.AreEqual(144, (long)np.sum(result));  // All 1s
        }

        [TestMethod]
        public void Where_AllTrue_LargeArray()
        {
            var size = 10000;
            var cond = np.ones(size, NPTypeCode.Boolean);
            var x = np.arange(size);
            var y = np.zeros(size, NPTypeCode.Int64);
            var result = np.where(cond, x, y);

            // Sum of 0 to 9999 = 49995000
            Assert.AreEqual(49995000L, (long)np.sum(result));
        }

        [TestMethod]
        public void Where_AllFalse_LargeArray()
        {
            var size = 10000;
            var cond = np.zeros(size, NPTypeCode.Boolean);
            var x = np.arange(size);
            var y = np.zeros(size, NPTypeCode.Int64);
            var result = np.where(cond, x, y);

            Assert.AreEqual(0L, (long)np.sum(result));
        }

        [TestMethod]
        public void Where_Alternating_LargeArray()
        {
            var size = 10000;
            var cond = np.zeros(size, NPTypeCode.Boolean);
            for (int i = 0; i < size; i += 2)
                cond[i] = true;

            var x = np.arange(size);
            var y = np.zeros(size, NPTypeCode.Int64);
            var result = np.where(cond, x, y);

            // Sum of even indices: 0+2+4+...+9998 = 24995000
            Assert.AreEqual(24995000L, (long)np.sum(result));
        }

        #endregion

        #region Type Conversion Edge Cases

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
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

        [TestMethod]
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

        #region Empty Array Edge Cases

        [TestMethod]
        public void Where_Empty2D()
        {
            // Empty (0,3) shape
            var cond = np.zeros(new[] { 0, 3 }, NPTypeCode.Boolean);
            var x = np.zeros(new[] { 0, 3 }, NPTypeCode.Double);
            var y = np.zeros(new[] { 0, 3 }, NPTypeCode.Double);
            var result = np.where(cond, x, y);

            result.Should().BeShaped(0, 3);
            Assert.AreEqual(typeof(double), result.dtype);
        }

        [TestMethod]
        public void Where_Empty3D()
        {
            // Empty (2,0,3) shape
            var cond = np.zeros(new[] { 2, 0, 3 }, NPTypeCode.Boolean);
            var x = np.zeros(new[] { 2, 0, 3 }, NPTypeCode.Int32);
            var y = np.zeros(new[] { 2, 0, 3 }, NPTypeCode.Int32);
            var result = np.where(cond, x, y);

            result.Should().BeShaped(2, 0, 3);
            Assert.AreEqual(typeof(int), result.dtype);
        }

        [TestMethod]
        public void Where_SingleArg_Empty2D()
        {
            var arr = np.zeros(new[] { 0, 3 }, NPTypeCode.Int32);
            var result = np.where(arr);

            Assert.AreEqual(2, result.Length);  // 2 dimensions
            Assert.AreEqual(0, result[0].size);
            Assert.AreEqual(0, result[1].size);
        }

        #endregion

        #region Error Conditions

        [TestMethod]
        public void Where_IncompatibleShapes_ThrowsException()
        {
            // Shapes (2,) and (3,) cannot be broadcast together
            var cond = np.array(new[] { true, false });  // (2,)
            var x = np.array(new[] { 1, 2, 3 });  // (3,)
            var y = np.array(new[] { 4, 5, 6 });  // (3,)

            Assert.ThrowsException<IncorrectShapeException>(() => np.where(cond, x, y));
        }

        #endregion

        #region NEP50 Type Promotion (NumPy 2.x Parity)

        /// <summary>
        /// Verifies NEP50 weak scalar semantics: when a scalar is combined with an array,
        /// the array's dtype wins for same-kind operations.
        /// </summary>
        [TestMethod]
        public void Where_ScalarTypePromotion_NEP50_WeakScalar()
        {
            // NumPy 2.x: np.where(cond, 1, uint8_array) -> uint8 (weak scalar)
            var cond = np.array(new[] { true, false });
            var yUint8 = np.array(new byte[] { 10, 20 });
            var result = np.where(cond, 1, yUint8);

            // Array dtype wins - matches NumPy 2.x NEP50
            Assert.AreEqual(typeof(byte), result.dtype);
            Assert.AreEqual((byte)1, (byte)result[0]);
            Assert.AreEqual((byte)20, (byte)result[1]);
        }

        /// <summary>
        /// Two same-type scalars preserve their type.
        /// Note: NumPy would return int64 for Python int literals, but C# int32 scalars
        /// cannot be distinguished from explicit np.array(1, dtype=int32), so we preserve.
        /// </summary>
        [TestMethod]
        public void Where_TwoScalars_SameType_Preserved()
        {
            var cond = np.array(new[] { true, false });

            // int + int → int (preserved)
            var result = np.where(cond, 1, 0);
            Assert.AreEqual(typeof(int), result.dtype);
            Assert.AreEqual(1, (int)result[0]);
            Assert.AreEqual(0, (int)result[1]);

            // long + long → long (preserved)
            result = np.where(cond, 1L, 0L);
            Assert.AreEqual(typeof(long), result.dtype);
        }

        /// <summary>
        /// Verifies C# float scalars stay float32 (like np.float32, not Python float).
        /// </summary>
        [TestMethod]
        public void Where_TwoScalars_Float32_StaysFloat32()
        {
            // C# float (1.0f) is like np.float32, not Python's float (which is float64)
            // np.where(cond, np.float32(1.0), np.float32(0.0)) -> float32
            var cond = np.array(new[] { true, false });
            var result = np.where(cond, 1.0f, 0.0f);

            Assert.AreEqual(typeof(float), result.dtype);
        }

        /// <summary>
        /// Verifies NEP50: int scalar + float32 array -> float32 (same-kind, array wins).
        /// </summary>
        [TestMethod]
        public void Where_IntScalar_Float32Array_ReturnsFloat32()
        {
            var cond = np.array(new[] { true, false });
            var yFloat32 = np.array(new float[] { 10.5f, 20.5f });
            var result = np.where(cond, 1, yFloat32);

            // Array dtype wins for same-kind (int->float conversion)
            Assert.AreEqual(typeof(float), result.dtype);
        }

        /// <summary>
        /// Verifies NEP50: float scalar + int32 array -> float64 (cross-kind promotion).
        /// </summary>
        [TestMethod]
        public void Where_FloatScalar_Int32Array_ReturnsFloat64()
        {
            var cond = np.array(new[] { true, false });
            var yInt32 = np.array(new int[] { 10, 20 });
            var result = np.where(cond, 1.5, yInt32);

            // Cross-kind: float scalar forces float64
            Assert.AreEqual(typeof(double), result.dtype);
        }

        #endregion
    }
}
