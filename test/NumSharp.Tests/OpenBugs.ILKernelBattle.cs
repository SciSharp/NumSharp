namespace NumSharp.Tests
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
    ///       Bug 82: FIXED - Dot product with transposed arrays produced wrong results
    ///       Bug 83: FIXED - 1D x 2D dot threw instead of computing (missing API)
    ///       Bug 84: FIXED - Empty bool prod returned bool dtype instead of int64
    /// </summary>
    /// <remarks>
    ///     The class had no <c>[TestClass]</c> from the TUnit → MSTest migration (ac020336, 2026-04-15) until
    ///     2026-10-05, so MSTest never discovered its test. The 2026-10-05 run of the whole OpenBugs category added the
    ///     attribute, found Bug 84 passing and removed its <c>[OpenBugs]</c> tag.
    /// </remarks>
    [TestClass]
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
        //  STATUS: FIXED (found passing by the 2026-10-05 OpenBugs run)
        //  SEVERITY: Low — minor type difference, values are equivalent.
        //
        //  NumPy: prod([]) for bool returns int64(1)
        //  NumSharp (before the fix): returned bool(True)
        //
        //  PYTHON VERIFICATION (NumPy 2.4.2):
        //    >>> np.prod(np.array([], dtype=bool))
        //    1
        //    >>> np.prod(np.array([], dtype=bool)).dtype
        //    dtype('int64')
        //
        // ================================================================

        /// <summary>
        ///     BUG 84: prod(empty bool) returns int64 1, the multiplicative identity in NumPy's int64 accumulator, not
        ///     bool True.
        /// </summary>
        [TestMethod]
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
    /// <remarks>
    ///     All 18 tests carried <c>[OpenBugs]</c> from 60053e26 (2026-04-11), and the class had no <c>[TestClass]</c> from
    ///     the TUnit → MSTest migration (ac020336, 2026-04-15) until 2026-10-05, so none of them ran under MSTest. The
    ///     2026-10-05 run of the whole OpenBugs category added the attribute, found all 18 passing and removed the tags.
    /// </remarks>
    [TestClass]
    public class ILKernelPassingTests
    {
        // ----- STD/VAR TESTS -----

        /// <summary>
        ///     <c>np.std</c> / <c>np.var</c> of <c>[1..5]</c> are √2 / 2 for int32, int64 and float64 inputs: the integer
        ///     inputs reduce in float64 like NumPy, so the integer SIMD path must not truncate the result.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.std</c> along each axis of <c>[[1, 2, 3], [4, 5, 6]]</c>: <c>[1.5, 1.5, 1.5]</c> for axis 0 and
        ///     <c>[0.8165, 0.8165]</c> for axis 1 (NumPy, ddof=0) — one output per position of the kept axis.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.std</c> / <c>np.var</c> of an empty array are NaN (NumPy's <c>0 / 0</c>, with a RuntimeWarning NumSharp
        ///     does not model), never an exception or 0.
        /// </summary>
        [TestMethod]
        public void StdVar_EmptyArray_ReturnsNaN()
        {
            var empty = np.array(new double[0]);
            double.IsNaN(np.std(empty).GetDouble(0)).Should().BeTrue("std(empty) = NaN");
            double.IsNaN(np.var(empty).GetDouble(0)).Should().BeTrue("var(empty) = NaN");
        }

        /// <summary>
        ///     <c>ddof=1</c> on one element divides a zero sum of squares by <c>max(n - ddof, 0) = 0</c>: NumPy's
        ///     <c>0 / 0</c> is NaN for both <c>np.std</c> and <c>np.var</c>.
        /// </summary>
        [TestMethod]
        public void StdVar_SingleElement_Ddof1_ReturnsNaN()
        {
            var single = np.array(new double[] { 5.0 });
            double.IsNaN(np.std(single, ddof: 1).GetDouble(0)).Should().BeTrue("std(single, ddof=1) = NaN");
            double.IsNaN(np.var(single, ddof: 1).GetDouble(0)).Should().BeTrue("var(single, ddof=1) = NaN");
        }

        // ----- CUMSUM TESTS -----

        /// <summary>
        ///     <c>np.cumsum</c> widens small integers to the 64-bit integer of the same signedness (int32 → int64,
        ///     uint8 → uint64) and keeps float32, as NumPy 2.x does on 64-bit platforms; a narrower accumulator would
        ///     overflow on long inputs.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.cumsum</c> along axis 0 and axis 1 of an int32 <c>(2, 3)</c> array accumulates down the columns and
        ///     along the rows respectively, in int64, with NumPy's values.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.cumsum</c> of a bool array counts the Trues in int64: <c>[T, F, T, T, F]</c> → <c>[1, 1, 2, 3, 3]</c>
        ///     (NumPy), not a bool running OR.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.left_shift</c> / <c>np.right_shift</c> by 1 on int32 double and halve the values (the baseline the
        ///     full-bit-width tests below build on).
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.right_shift</c> of a signed integer is ARITHMETIC (sign-filling): <c>-1 &gt;&gt; 1</c> is -1 and
        ///     <c>-16 &gt;&gt; 1</c> is -8. A logical shift would turn small negatives into huge positives.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     Bug 81: shifting int32 by 32 or 64 bits gives 0 (NumPy), not the original value. C# masks a shift count to
        ///     the operand width (<c>x &lt;&lt; 32 == x</c> for an int), so a kernel that uses the C# operator directly
        ///     returns the input unchanged.
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     Bug 81, unsigned form: uint32 <c>0xFFFFFFFF &lt;&lt; 32</c> is 0.
        /// </summary>
        [TestMethod]
        public void Shift_UInt32_FullBitWidth_ReturnsZero()
        {
            var a = np.array(new uint[] { 0xFFFFFFFF });
            var result = np.left_shift(a, 32);
            result.GetUInt32(0).Should().Be(0u, "NumPy: 0xFFFFFFFF << 32 = 0 for uint32");
        }

        /// <summary>
        ///     Bug 81, negative form: a negative int32 right-shifted by 32 bits is -1 (every bit a copy of the sign), not
        ///     the C#-masked shift that returns the input unchanged.
        /// </summary>
        [TestMethod]
        public void Shift_NegativeRightShift_FullBitWidth_ReturnsMinusOne()
        {
            // Negative values right-shifted by >= bit width should return -1
            var neg = np.array(new int[] { -5, -100 });
            var result = np.right_shift(neg, 32);
            result.GetInt32(0).Should().Be(-1, "NumPy: -5 >> 32 = -1 for int32");
            result.GetInt32(1).Should().Be(-1, "NumPy: -100 >> 32 = -1 for int32");
        }

        /// <summary>
        ///     Per-element shift amounts below, at and past the bit width (1, 32, 64) apply Bug 81's rule element by
        ///     element: <c>[5, 10, 15] &lt;&lt; [1, 32, 64]</c> is <c>[10, 0, 0]</c> and <c>&gt;&gt;</c> gives
        ///     <c>[2, 0, 0]</c> (the array-shift kernel is a separate path from the scalar-shift one).
        /// </summary>
        [TestMethod]
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

        /// <summary>
        ///     <c>np.dot</c> of two contiguous 1-D float64 vectors is their inner product, a scalar:
        ///     <c>[1, 2, 3] · [4, 5, 6] = 32</c>.
        /// </summary>
        [TestMethod]
        public void Dot_1Dx1D_InnerProduct()
        {
            var a = np.array(new double[] { 1, 2, 3 });
            var b = np.array(new double[] { 4, 5, 6 });
            var result = np.dot(a, b);

            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            result.GetDouble(0).Should().Be(32.0);
        }

        /// <summary>
        ///     <c>np.dot</c> of two contiguous 2-D float64 matrices is the matrix product
        ///     <c>[[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]</c> (the contiguous baseline for the
        ///     transposed Bug 82 tests below).
        /// </summary>
        [TestMethod]
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
        [TestMethod]
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
            result.shape.Should().BeEquivalentTo(new long[] { 2, 2 }, "Shape should be (2, 2)");

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
        [TestMethod]
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

        /// <summary>
        ///     <c>np.prod</c> of a bool array multiplies in NumPy's int64 accumulator: 1 when every element is True, 0 as
        ///     soon as one is False.
        /// </summary>
        [TestMethod]
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
