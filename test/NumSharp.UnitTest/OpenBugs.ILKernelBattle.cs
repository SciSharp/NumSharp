using System;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using TUnit.Core;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Bugs 82-84: IL Kernel migration battle test bugs.
    ///
    ///     These bugs were discovered during comprehensive battle testing of
    ///     recent IL kernel changes against NumPy 2.x (v2.4.2) as ground truth.
    ///
    ///     Tested operations:
    ///       - Std/Var axis SIMD with integer types (PASS)
    ///       - CumSum axis optimization (PASS)
    ///       - LeftShift/RightShift IL kernel (FIXED - Bug 81)
    ///       - Dot.NDMD IL kernel (FAIL - transposed arrays)
    ///       - Bool support for Prod/CumSum (PASS)
    ///
    ///     Bug summary:
    ///       Bug 81: FIXED - Shift by full bit width now returns 0 (see ILKernelPassingTests)
    ///       Bug 82: Dot product with transposed arrays produces wrong results
    ///       Bug 83: 1D x 2D dot throws instead of computing (missing API)
    ///       Bug 84: Empty bool prod returns bool dtype instead of int64
    /// </summary>
    [OpenBugs]
    public class ILKernelBattleTests
    {
        // ================================================================
        //
        //  BUG 82: Dot product with transposed arrays produces wrong results
        //
        //  STATUS: FIXED in 0.41.x
        //  Tests moved to ILKernelPassingTests
        //
        // ================================================================

        // ================================================================
        //  BUG 83: FIXED — 1D x 2D dot now works (Default.Dot.cs)
        //  Test moved to LinearAlgebraTests.Dot_1D_2D
        // ================================================================

        // ================================================================
        //
        //  BUG 84: Empty bool prod returns bool dtype instead of int64
        //
        //  SEVERITY: Low — minor type difference, values are equivalent.
        //
        //  NumPy: prod([]) for bool returns int64(1)
        //  NumSharp: Returns bool(True)
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.prod(np.array([], dtype=bool))
        //    1
        //    >>> np.prod(np.array([], dtype=bool)).dtype
        //    dtype('int64')
        //
        // ================================================================

        /// <summary>
        ///     BUG 84: prod(empty bool) should return int64, not bool.
        /// </summary>
        [Test]
        public void Bug84_Prod_EmptyBool_ReturnsInt64()
        {
            var a = np.array(new bool[0]);
            var result = np.prod(a);

            // NumPy returns int64 with value 1
            result.dtype.Should().Be(typeof(long),
                "NumPy: prod(empty bool).dtype = int64");

            // The value should be 1 (identity for multiplication)
            result.GetInt64(0).Should().Be(1L,
                "NumPy: prod(empty bool) = 1 (identity)");
        }
    }

    // ================================================================
    //  PASSING TESTS - Verify IL Kernel migrations work correctly
    // ================================================================

    /// <summary>
    ///     Passing tests that verify IL kernel migrations work correctly.
    ///     These are NOT OpenBugs - they should always pass.
    /// </summary>
    public class ILKernelPassingTests
    {
        // ----- STD/VAR TESTS -----

        [Test]
        public void StdVar_BasicDtypes_MatchNumPy()
        {
            var a_int32 = np.array(new int[] { 1, 2, 3, 4, 5 });
            var a_int64 = np.array(new long[] { 1, 2, 3, 4, 5 });
            var a_float32 = np.array(new float[] { 1, 2, 3, 4, 5 });
            var a_float64 = np.array(new double[] { 1, 2, 3, 4, 5 });

            // NumPy std = 1.4142135623730951
            np.std(a_int32).GetDouble(0).Should().BeApproximately(1.4142135623730951, 1e-10);
            np.std(a_int64).GetDouble(0).Should().BeApproximately(1.4142135623730951, 1e-10);
            np.std(a_float64).GetDouble(0).Should().BeApproximately(1.4142135623730951, 1e-10);

            // NumPy var = 2.0
            np.var(a_int32).GetDouble(0).Should().BeApproximately(2.0, 1e-10);
            np.var(a_int64).GetDouble(0).Should().BeApproximately(2.0, 1e-10);
            np.var(a_float64).GetDouble(0).Should().BeApproximately(2.0, 1e-10);
        }

        [Test]
        public void StdVar_WithAxis_MatchNumPy()
        {
            var a = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });

            // std(axis=0) = [1.5, 1.5, 1.5]
            var std_axis0 = np.std(a, axis: 0);
            std_axis0.GetDouble(0).Should().BeApproximately(1.5, 1e-10);
            std_axis0.GetDouble(1).Should().BeApproximately(1.5, 1e-10);
            std_axis0.GetDouble(2).Should().BeApproximately(1.5, 1e-10);

            // std(axis=1) = [0.81649658, 0.81649658]
            var std_axis1 = np.std(a, axis: 1);
            std_axis1.GetDouble(0).Should().BeApproximately(0.81649658, 1e-6);
            std_axis1.GetDouble(1).Should().BeApproximately(0.81649658, 1e-6);
        }

        [Test]
        public void StdVar_EmptyArray_ReturnsNaN()
        {
            var empty = np.array(new double[0]);
            double.IsNaN(np.std(empty).GetDouble(0)).Should().BeTrue("std(empty) = NaN");
            double.IsNaN(np.var(empty).GetDouble(0)).Should().BeTrue("var(empty) = NaN");
        }

        [Test]
        public void StdVar_SingleElement_Ddof1_ReturnsNaN()
        {
            var single = np.array(new double[] { 5.0 });
            double.IsNaN(np.std(single, ddof: 1).GetDouble(0)).Should().BeTrue("std(single, ddof=1) = NaN");
            double.IsNaN(np.var(single, ddof: 1).GetDouble(0)).Should().BeTrue("var(single, ddof=1) = NaN");
        }

        // ----- CUMSUM TESTS -----

        [Test]
        public void CumSum_TypePromotion_MatchNumPy()
        {
            // int32 -> int64
            var a_int32 = np.array(new int[] { 1, 2, 3, 4, 5 });
            var cs_int32 = np.cumsum(a_int32);
            cs_int32.dtype.Should().Be(typeof(long), "cumsum(int32) -> int64");

            // uint8 -> uint64
            var a_byte = np.array(new byte[] { 1, 2, 3 });
            var cs_byte = np.cumsum(a_byte);
            cs_byte.dtype.Should().Be(typeof(ulong), "cumsum(uint8) -> uint64");

            // float32 stays float32
            var a_float32 = np.array(new float[] { 1, 2, 3 });
            var cs_float32 = np.cumsum(a_float32);
            cs_float32.dtype.Should().Be(typeof(float), "cumsum(float32) -> float32");
        }

        [Test]
        public void CumSum_WithAxis_MatchNumPy()
        {
            var a = np.array(new int[,] { { 1, 2, 3 }, { 4, 5, 6 } });

            // cumsum(axis=0) = [[1, 2, 3], [5, 7, 9]]
            var cs_axis0 = np.cumsum(a, axis: 0);
            cs_axis0.GetInt64(0, 0).Should().Be(1L);
            cs_axis0.GetInt64(0, 2).Should().Be(3L);
            cs_axis0.GetInt64(1, 0).Should().Be(5L);
            cs_axis0.GetInt64(1, 2).Should().Be(9L);

            // cumsum(axis=1) = [[1, 3, 6], [4, 9, 15]]
            var cs_axis1 = np.cumsum(a, axis: 1);
            cs_axis1.GetInt64(0, 2).Should().Be(6L);
            cs_axis1.GetInt64(1, 2).Should().Be(15L);
        }

        [Test]
        public void CumSum_BoolArray_MatchNumPy()
        {
            var a = np.array(new bool[] { true, false, true, true, false });
            var cs = np.cumsum(a);

            // NumPy: [1, 1, 2, 3, 3]
            cs.dtype.Should().Be(typeof(long), "cumsum(bool) -> int64");
            cs.GetInt64(0).Should().Be(1L);
            cs.GetInt64(1).Should().Be(1L);
            cs.GetInt64(2).Should().Be(2L);
            cs.GetInt64(3).Should().Be(3L);
            cs.GetInt64(4).Should().Be(3L);
        }

        // ----- SHIFT TESTS -----

        [Test]
        public void Shift_BasicOperations_MatchNumPy()
        {
            var a = np.array(new int[] { 1, 2, 4, 8, 16 });

            // left_shift by 1
            var ls = np.left_shift(a, 1);
            ls.GetInt32(0).Should().Be(2);
            ls.GetInt32(4).Should().Be(32);

            // right_shift by 1
            var rs = np.right_shift(a, 1);
            rs.GetInt32(0).Should().Be(0);
            rs.GetInt32(4).Should().Be(8);
        }

        [Test]
        public void Shift_SignedRightShift_IsArithmetic()
        {
            // NumPy uses arithmetic right shift for signed types
            var a = np.array(new int[] { -1, -2, -4, -8, -16 });
            var rs = np.right_shift(a, 1);

            // Arithmetic shift preserves sign bit
            rs.GetInt32(0).Should().Be(-1, "-1 >> 1 = -1 (arithmetic)");
            rs.GetInt32(1).Should().Be(-1, "-2 >> 1 = -1 (arithmetic)");
            rs.GetInt32(2).Should().Be(-2, "-4 >> 1 = -2 (arithmetic)");
            rs.GetInt32(3).Should().Be(-4, "-8 >> 1 = -4 (arithmetic)");
            rs.GetInt32(4).Should().Be(-8, "-16 >> 1 = -8 (arithmetic)");
        }

        // ----- BUG 81 FIX: Shift by >= bit width -----
        // These tests verify the fix for Bug 81 where shift by >= bit width
        // returned the original value instead of 0 (due to C# masking behavior).

        [Test]
        public void Shift_FullBitWidth_ReturnsZero()
        {
            // int32 << 32 should return 0 (not original value)
            var a = np.array(new int[] { 1, 2, 3 });
            var result = np.left_shift(a, 32);
            result.GetInt32(0).Should().Be(0, "NumPy: 1 << 32 = 0 for int32");
            result.GetInt32(1).Should().Be(0, "NumPy: 2 << 32 = 0 for int32");
            result.GetInt32(2).Should().Be(0, "NumPy: 3 << 32 = 0 for int32");

            // int32 >> 32 should return 0 for positive values
            result = np.right_shift(a, 32);
            result.GetInt32(0).Should().Be(0, "NumPy: 1 >> 32 = 0 for int32");
            result.GetInt32(1).Should().Be(0, "NumPy: 2 >> 32 = 0 for int32");
            result.GetInt32(2).Should().Be(0, "NumPy: 3 >> 32 = 0 for int32");

            // int32 << 64 should also return 0
            result = np.left_shift(a, 64);
            result.GetInt32(0).Should().Be(0, "NumPy: 1 << 64 = 0 for int32");
        }

        [Test]
        public void Shift_UInt32_FullBitWidth_ReturnsZero()
        {
            var a = np.array(new uint[] { 0xFFFFFFFF });
            var result = np.left_shift(a, 32);
            result.GetUInt32(0).Should().Be(0u, "NumPy: 0xFFFFFFFF << 32 = 0 for uint32");
        }

        [Test]
        public void Shift_NegativeRightShift_FullBitWidth_ReturnsMinusOne()
        {
            // Negative values right-shifted by >= bit width should return -1
            var neg = np.array(new int[] { -5, -100 });
            var result = np.right_shift(neg, 32);
            result.GetInt32(0).Should().Be(-1, "NumPy: -5 >> 32 = -1 for int32");
            result.GetInt32(1).Should().Be(-1, "NumPy: -100 >> 32 = -1 for int32");
        }

        [Test]
        public void Shift_ArrayShifts_WithOverflow()
        {
            // Test array shift amounts where some are >= bit width
            var arr = np.array(new int[] { 5, 10, 15 });
            var shifts = np.array(new int[] { 1, 32, 64 });
            var result = np.left_shift(arr, shifts);
            result.GetInt32(0).Should().Be(10, "5 << 1 = 10");
            result.GetInt32(1).Should().Be(0, "10 << 32 = 0");
            result.GetInt32(2).Should().Be(0, "15 << 64 = 0");

            result = np.right_shift(arr, shifts);
            result.GetInt32(0).Should().Be(2, "5 >> 1 = 2");
            result.GetInt32(1).Should().Be(0, "10 >> 32 = 0");
            result.GetInt32(2).Should().Be(0, "15 >> 64 = 0");
        }

        // ----- DOT TESTS (contiguous) -----

        [Test]
        public void Dot_1Dx1D_InnerProduct()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 4, 5, 6 });
            var result = np.dot(a, b);

            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            result.GetDouble(0).Should().Be(32.0);
        }

        [Test]
        public void Dot_2Dx2D_MatrixMultiply()
        {
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var result = np.dot(a, b);

            // [[19, 22], [43, 50]]
            result.GetDouble(0, 0).Should().Be(19.0);
            result.GetDouble(0, 1).Should().Be(22.0);
            result.GetDouble(1, 0).Should().Be(43.0);
            result.GetDouble(1, 1).Should().Be(50.0);
        }

        // ----- DOT TESTS (non-contiguous / transposed) - BUG 82 FIXED -----

        /// <summary>
        ///     BUG 82 (FIXED): Dot product with both operands transposed.
        ///     Previously produced wrong results due to incorrect stride handling.
        /// </summary>
        [Test]
        public void Dot_BothTransposed_MatchNumPy()
        {
            // Original arrays before transpose
            var arr1 = np.array(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });  // (3, 2)
            var arr2 = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } });      // (2, 3)

            // Transpose them
            var t1 = arr1.T;  // Now (2, 3): [[1,3,5], [2,4,6]]
            var t2 = arr2.T;  // Now (3, 2): [[1,4], [2,5], [3,6]]

            // Verify they are non-contiguous
            t1.Shape.IsContiguous.Should().BeFalse("Transposed array should be non-contiguous");
            t2.Shape.IsContiguous.Should().BeFalse("Transposed array should be non-contiguous");

            // Compute dot product
            var result = np.dot(t1, t2);

            // Expected from NumPy:
            // [[22, 49],
            //  [28, 64]]
            result.shape.Should().BeEquivalentTo(new[] { 2, 2 }, "Shape should be (2, 2)");

            result.GetDouble(0, 0).Should().BeApproximately(22.0, 1e-10,
                "NumPy: dot(t1, t2)[0,0] = 22");
            result.GetDouble(0, 1).Should().BeApproximately(49.0, 1e-10,
                "NumPy: dot(t1, t2)[0,1] = 49");
            result.GetDouble(1, 0).Should().BeApproximately(28.0, 1e-10,
                "NumPy: dot(t1, t2)[1,0] = 28");
            result.GetDouble(1, 1).Should().BeApproximately(64.0, 1e-10,
                "NumPy: dot(t1, t2)[1,1] = 64");
        }

        /// <summary>
        ///     BUG 82b (FIXED): Dot with one transposed operand.
        /// </summary>
        [Test]
        public void Dot_OneTransposed_MatchNumPy()
        {
            // One contiguous, one transposed
            var cont = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var noncont = np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }).T["0:2, 0:2"];

            cont.Shape.IsContiguous.Should().BeTrue();
            noncont.Shape.IsContiguous.Should().BeFalse();

            var result = np.dot(cont, noncont);

            // From NumPy:
            // noncont after transpose and slice:
            // [[1, 4],
            //  [2, 5]]
            //
            // cont @ noncont:
            // [[1*1 + 2*2, 1*4 + 2*5],   = [[5, 14],
            //  [3*1 + 4*2, 3*4 + 4*5]]     [11, 32]]

            result.GetDouble(0, 0).Should().BeApproximately(5.0, 1e-10,
                "NumPy: dot(cont, noncont)[0,0] = 5");
            result.GetDouble(0, 1).Should().BeApproximately(14.0, 1e-10,
                "NumPy: dot(cont, noncont)[0,1] = 14");
            result.GetDouble(1, 0).Should().BeApproximately(11.0, 1e-10,
                "NumPy: dot(cont, noncont)[1,0] = 11");
            result.GetDouble(1, 1).Should().BeApproximately(32.0, 1e-10,
                "NumPy: dot(cont, noncont)[1,1] = 32");
        }

        // ----- BOOL PROD TESTS -----

        [Test]
        public void Prod_BoolArray_MatchNumPy()
        {
            // prod([True, True, True]) = 1
            var all_true = np.array(new bool[] { true, true, true });
            np.prod(all_true).GetInt64(0).Should().Be(1L);

            // prod([True, True, False]) = 0
            var has_false = np.array(new bool[] { true, true, false });
            np.prod(has_false).GetInt64(0).Should().Be(0L);

            // prod([False, False, False]) = 0
            var all_false = np.array(new bool[] { false, false, false });
            np.prod(all_false).GetInt64(0).Should().Be(0L);
        }
    }
}
