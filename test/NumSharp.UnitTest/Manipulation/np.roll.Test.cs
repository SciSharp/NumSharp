using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends.Unmanaged;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Manipulation
{
    /// <summary>
    /// Comprehensive tests for np.roll / NDArray.roll, verified against NumPy 2.4.2 output.
    ///
    /// NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.roll.html
    /// NumPy test source: numpy/_core/tests/test_numeric.py, class TestRoll
    /// </summary>
    [TestClass]
    public class np_roll_Test
    {
        // ================================================================
        // 1D ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_1D_PositiveShift()
        {
            // np.roll(np.arange(10), 2) => [8 9 0 1 2 3 4 5 6 7]
            var x = np.arange(10);
            var result = np.roll(x, 2);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(8, 9, 0, 1, 2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Roll_1D_NegativeShift()
        {
            // np.roll(np.arange(10), -2) => [2 3 4 5 6 7 8 9 0 1]
            var x = np.arange(10);
            var result = np.roll(x, -2);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(2, 3, 4, 5, 6, 7, 8, 9, 0, 1);
        }

        [TestMethod]
        public void Roll_1D_ZeroShift()
        {
            // np.roll(np.arange(10), 0) => [0 1 2 3 4 5 6 7 8 9]
            var x = np.arange(10);
            var result = np.roll(x, 0);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Roll_1D_ShiftEqualsSize()
        {
            // np.roll(np.arange(10), 10) => [0 1 2 3 4 5 6 7 8 9]
            var x = np.arange(10);
            var result = np.roll(x, 10);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Roll_1D_ShiftEqualsNegativeSize()
        {
            // np.roll(np.arange(10), -10) => [0 1 2 3 4 5 6 7 8 9]
            var x = np.arange(10);
            var result = np.roll(x, -10);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Roll_1D_ShiftGreaterThanSize()
        {
            // np.roll(np.arange(10), 12) => [8 9 0 1 2 3 4 5 6 7]  (12 % 10 = 2)
            var x = np.arange(10);
            var result = np.roll(x, 12);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(8, 9, 0, 1, 2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Roll_1D_ShiftLessThanNegativeSize()
        {
            // np.roll(np.arange(10), -12) => [2 3 4 5 6 7 8 9 0 1]  (-12 % 10 = -2 => 8 effective)
            var x = np.arange(10);
            var result = np.roll(x, -12);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(2, 3, 4, 5, 6, 7, 8, 9, 0, 1);
        }

        [TestMethod]
        public void Roll_1D_ShiftMuchLarger()
        {
            // np.roll(np.arange(10), 25) => [5 6 7 8 9 0 1 2 3 4]  (25 % 10 = 5)
            var x = np.arange(10);
            var result = np.roll(x, 25);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_1D_VeryLargeShift()
        {
            // np.roll(np.arange(5), 1000000) => [0 1 2 3 4]  (1000000 % 5 = 0)
            var x = np.arange(5);
            var result = np.roll(x, 1000000);

            result.Should().BeShaped(5);
            result.Should().BeOfValues(0, 1, 2, 3, 4);
        }

        // ================================================================
        // 1D - NumPy test_roll1d (exact match)
        // ================================================================

        [TestMethod]
        public void Roll_NumPy_test_roll1d()
        {
            // Exact replication of NumPy TestRoll.test_roll1d
            var x = np.arange(10);
            var xr = np.roll(x, 2);
            xr.Should().Be(np.array(new int[] { 8, 9, 0, 1, 2, 3, 4, 5, 6, 7 }));
        }

        // ================================================================
        // 2D ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_2D_NoAxis_Flattens()
        {
            // np.roll(x2, 1) flattens, rolls, restores shape
            // x2 = [[0,1,2,3,4],[5,6,7,8,9]]
            // roll => [[9,0,1,2,3],[4,5,6,7,8]]
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(9, 0, 1, 2, 3, 4, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_2D_Axis0()
        {
            // np.roll(x2, 1, axis=0) => [[5,6,7,8,9],[0,1,2,3,4]]
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 1, 0);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_2D_Axis1()
        {
            // np.roll(x2, 1, axis=1) => [[4,0,1,2,3],[9,5,6,7,8]]
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_2D_NegativeAxis1()
        {
            // np.roll(x2, 1, axis=-1) == np.roll(x2, 1, axis=1)
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 1, -1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_2D_NegativeAxis2()
        {
            // np.roll(x2, 1, axis=-2) == np.roll(x2, 1, axis=0)
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 1, -2);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_2D_NegativeShift_Axis0()
        {
            // np.roll(x2, -1, axis=0) => [[5,6,7,8,9],[0,1,2,3,4]]
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, -1, 0);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_2D_NegativeShift_Axis1()
        {
            // np.roll(x2, -1, axis=1) => [[1,2,3,4,0],[6,7,8,9,5]]
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, -1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(1, 2, 3, 4, 0, 6, 7, 8, 9, 5);
        }

        [TestMethod]
        public void Roll_2D_ShiftGreaterThanDimSize_Axis1()
        {
            // np.roll(x2, 6, axis=1) => [[4,0,1,2,3],[9,5,6,7,8]]  (6 % 5 = 1)
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, 6, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_2D_NegativeShiftWraps_Axis1()
        {
            // np.roll(x2, -4, axis=1) => [[4,0,1,2,3],[9,5,6,7,8]]  (-4 % 5 = 1 effective)
            var x2 = np.arange(10).reshape(2, 5);
            var result = np.roll(x2, -4, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_2D_NoAxis_VariousShifts()
        {
            // x = [[0,1,2],[3,4,5]]
            var x = np.arange(6).reshape(2, 3);

            // roll(x, 1) => [[5,0,1],[2,3,4]]
            np.roll(x, 1).Should().BeOfValues(5, 0, 1, 2, 3, 4);
            np.roll(x, 1).Should().BeShaped(2, 3);

            // roll(x, 2) => [[4,5,0],[1,2,3]]
            np.roll(x, 2).Should().BeOfValues(4, 5, 0, 1, 2, 3);

            // roll(x, -1) => [[1,2,3],[4,5,0]]
            np.roll(x, -1).Should().BeOfValues(1, 2, 3, 4, 5, 0);

            // roll(x, -2) => [[2,3,4],[5,0,1]]
            np.roll(x, -2).Should().BeOfValues(2, 3, 4, 5, 0, 1);
        }

        // ================================================================
        // NumPy test_roll2d (exact match)
        // ================================================================

        [TestMethod]
        public void Roll_NumPy_test_roll2d_NoAxis()
        {
            var x2 = np.arange(10).reshape(2, 5);
            var x2r = np.roll(x2, 1);
            x2r.Should().Be(np.array(new int[,] { { 9, 0, 1, 2, 3 }, { 4, 5, 6, 7, 8 } }));
        }

        [TestMethod]
        public void Roll_NumPy_test_roll2d_Axis0()
        {
            var x2 = np.arange(10).reshape(2, 5);
            var x2r = np.roll(x2, 1, 0);
            x2r.Should().Be(np.array(new int[,] { { 5, 6, 7, 8, 9 }, { 0, 1, 2, 3, 4 } }));
        }

        [TestMethod]
        public void Roll_NumPy_test_roll2d_Axis1()
        {
            var x2 = np.arange(10).reshape(2, 5);
            var x2r = np.roll(x2, 1, 1);
            x2r.Should().Be(np.array(new int[,] { { 4, 0, 1, 2, 3 }, { 9, 5, 6, 7, 8 } }));
        }

        [TestMethod]
        public void Roll_NumPy_test_roll2d_MoreThanOneTurn_Positive()
        {
            var x2 = np.arange(10).reshape(2, 5);
            // roll(x2, 6, axis=1) same as roll(x2, 1, axis=1)
            var x2r = np.roll(x2, 6, 1);
            x2r.Should().Be(np.array(new int[,] { { 4, 0, 1, 2, 3 }, { 9, 5, 6, 7, 8 } }));
        }

        [TestMethod]
        public void Roll_NumPy_test_roll2d_MoreThanOneTurn_Negative()
        {
            var x2 = np.arange(10).reshape(2, 5);
            // roll(x2, -4, axis=1) same as roll(x2, 1, axis=1)
            var x2r = np.roll(x2, -4, 1);
            x2r.Should().Be(np.array(new int[,] { { 4, 0, 1, 2, 3 }, { 9, 5, 6, 7, 8 } }));
        }

        // ================================================================
        // 3D ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_3D_NoAxis()
        {
            // x3 = arange(24).reshape(2,3,4)
            // roll(x3, 1) => [[[23,0,1,2],[3,4,5,6],[7,8,9,10]],[[11,12,13,14],[15,16,17,18],[19,20,21,22]]]
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, 1);

            result.Should().BeShaped(2, 3, 4);
            result.Should().BeOfValues(
                23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22);
        }

        [TestMethod]
        public void Roll_3D_Axis0()
        {
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, 1, 0);

            result.Should().BeShaped(2, 3, 4);
            // [[12..23],[0..11]]
            result.Should().BeOfValues(
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Roll_3D_Axis1()
        {
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, 1, 1);

            result.Should().BeShaped(2, 3, 4);
            // [[[8,9,10,11],[0,1,2,3],[4,5,6,7]],[[20,21,22,23],[12,13,14,15],[16,17,18,19]]]
            result.Should().BeOfValues(
                8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7,
                20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19);
        }

        [TestMethod]
        public void Roll_3D_Axis2()
        {
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, 1, 2);

            result.Should().BeShaped(2, 3, 4);
            // [[[3,0,1,2],[7,4,5,6],[11,8,9,10]],[[15,12,13,14],[19,16,17,18],[23,20,21,22]]]
            result.Should().BeOfValues(
                3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10,
                15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22);
        }

        [TestMethod]
        public void Roll_3D_NegativeShift_Axis0()
        {
            // For 2-element dim, roll by -1 == roll by 1
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, -1, 0);

            result.Should().BeShaped(2, 3, 4);
            result.Should().BeOfValues(
                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Roll_3D_Shift2_Axis1()
        {
            var x3 = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x3, 2, 1);

            result.Should().BeShaped(2, 3, 4);
            // [[[4,5,6,7],[8,9,10,11],[0,1,2,3]],[[16,17,18,19],[20,21,22,23],[12,13,14,15]]]
            result.Should().BeOfValues(
                4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3,
                16, 17, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15);
        }

        [TestMethod]
        public void Roll_3D_NegativeAxis()
        {
            var x3 = np.arange(24).reshape(2, 3, 4);

            // axis=-1 == axis=2
            var r1 = np.roll(x3, 1, -1);
            var r2 = np.roll(x3, 1, 2);
            r1.Should().Be(r2);

            // axis=-2 == axis=1
            var r3 = np.roll(x3, 1, -2);
            var r4 = np.roll(x3, 1, 1);
            r3.Should().Be(r4);

            // axis=-3 == axis=0
            var r5 = np.roll(x3, 1, -3);
            var r6 = np.roll(x3, 1, 0);
            r5.Should().Be(r6);
        }

        // ================================================================
        // EMPTY ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_EmptyArray_NoAxis()
        {
            // np.roll([], 1) => []
            var x = np.array(new double[0]);
            var result = np.roll(x, 1);
            result.size.Should().Be(0);
            result.ndim.Should().Be(1);
        }

        [TestMethod]
        public void Roll_EmptyArray_ZeroShift()
        {
            var x = np.array(new double[0]);
            var result = np.roll(x, 0);
            result.size.Should().Be(0);
        }

        [TestMethod]
        public void Roll_EmptyArray_NegativeShift()
        {
            var x = np.array(new double[0]);
            var result = np.roll(x, -1);
            result.size.Should().Be(0);
        }

        [TestMethod]
        public void Roll_Empty2D_NoAxis()
        {
            var x = np.empty(new Shape(0, 3));
            var result = np.roll(x, 1);
            result.Shape.dimensions.Should().BeEquivalentTo(new[] { 0, 3 });
        }

        [TestMethod]
        public void Roll_Empty2D_Axis0()
        {
            var x = np.empty(new Shape(0, 3));
            var result = np.roll(x, 1, 0);
            result.Shape.dimensions.Should().BeEquivalentTo(new[] { 0, 3 });
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_Empty2D_Axis1()
        {
            // Bug: np.roll on empty 2D array (shape 0x3) with axis=1 throws
            // InvalidOperationException: "Can't construct NDIterator with an empty shape"
            // because the slice assignment result[dstBody] = a[srcBody] creates an empty
            // slice along axis=0 and NDIterator cannot handle empty shapes.
            // NumPy returns an empty array of shape (0, 3) with no error.
            var x = np.empty(new Shape(0, 3));
            var result = np.roll(x, 1, 1);
            result.Shape.dimensions.Should().BeEquivalentTo(new[] { 0, 3 });
        }

        // ================================================================
        // SCALAR (0-dim)
        // ================================================================

        [TestMethod]
        public void Roll_Scalar_ZeroShift()
        {
            // np.roll(np.array(42), 0) => array(42)
            var s = NDArray.Scalar(42);
            var result = np.roll(s, 0);

            result.shape.Should().BeEquivalentTo(Array.Empty<int>());
            result.GetInt32(0).Should().Be(42);
        }

        [TestMethod]
        public void Roll_Scalar_PositiveShift()
        {
            // np.roll(np.array(42), 1) => array(42)
            var s = NDArray.Scalar(42);
            var result = np.roll(s, 1);

            result.shape.Should().BeEquivalentTo(Array.Empty<int>());
            result.GetInt32(0).Should().Be(42);
        }

        [TestMethod]
        public void Roll_Scalar_NegativeShift()
        {
            // np.roll(np.array(42), -1) => array(42)
            var s = NDArray.Scalar(42);
            var result = np.roll(s, -1);

            result.shape.Should().BeEquivalentTo(Array.Empty<int>());
            result.GetInt32(0).Should().Be(42);
        }

        // ================================================================
        // OUT-OF-BOUNDS AXIS
        // ================================================================

        [TestMethod]
        public void Roll_OutOfBoundsAxis_Positive_Throws()
        {
            var x = np.arange(10);
            Action act = () => np.roll(x, 1, 2);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Roll_OutOfBoundsAxis_Negative_Throws()
        {
            var x = np.arange(10);
            Action act = () => np.roll(x, 1, -2);
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void Roll_Scalar_WithAxis_Throws()
        {
            // NumPy: axis 0 is out of bounds for array of dimension 0
            var s = NDArray.Scalar(42);
            Action act = () => np.roll(s, 1, 0);
            act.Should().Throw<ArgumentException>();
        }

        // ================================================================
        // ORIGINAL NOT MODIFIED (roll returns a copy)
        // ================================================================

        [TestMethod]
        public void Roll_1D_DoesNotModifyOriginal()
        {
            var orig = np.arange(5);
            var result = np.roll(orig, 2);

            // result = [3,4,0,1,2]
            result.Should().BeOfValues(3, 4, 0, 1, 2);

            // orig unchanged
            orig.Should().BeOfValues(0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_2D_WithAxis_DoesNotModifyOriginal()
        {
            var orig = np.arange(6).reshape(2, 3);
            var result = np.roll(orig, 1, 0);

            // Mutate result
            result.SetInt32(999, 0, 0);

            // orig unchanged
            orig.GetInt32(0, 0).Should().Be(0);
            orig.GetInt32(1, 0).Should().Be(3);
        }

        [TestMethod]
        public void Roll_1D_MutatingResultDoesNotAffectOriginal()
        {
            var orig = np.arange(5);
            var result = np.roll(orig, 2);

            result.SetInt32(999, 0);

            // orig[0] still 0
            orig.GetInt32(0).Should().Be(0);
        }

        // ================================================================
        // SHIFT=0 RETURNS A COPY (not a view)
        // ================================================================

        [TestMethod]
        public void Roll_ZeroShift_ReturnsCopy()
        {
            var a = np.arange(5);
            var r = np.roll(a, 0);

            r.Should().BeOfValues(0, 1, 2, 3, 4);

            // Modify result, original should be unaffected
            r.SetInt32(999, 0);
            a.GetInt32(0).Should().Be(0);
        }

        // ================================================================
        // SINGLE ELEMENT ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_SingleElement_AnyShift()
        {
            var a = np.array(new int[] { 42 });

            np.roll(a, 0).Should().BeOfValues(42);
            np.roll(a, 1).Should().BeOfValues(42);
            np.roll(a, -1).Should().BeOfValues(42);
            np.roll(a, 100).Should().BeOfValues(42);
        }

        // ================================================================
        // DTYPE PRESERVATION
        // ================================================================

        [TestMethod]
        public void Roll_PreservesDtype_Int32()
        {
            var a = np.array(new int[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Int32);
            r.Should().BeOfValues(4, 5, 1, 2, 3);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Double()
        {
            var a = np.array(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Double);
            r.Should().BeOfValues(4.0, 5.0, 1.0, 2.0, 3.0);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Single()
        {
            var a = np.array(new float[] { 1f, 2f, 3f, 4f, 5f });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Single);
            r.Should().BeOfValues(4f, 5f, 1f, 2f, 3f);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Int64()
        {
            var a = np.array(new long[] { 1L, 2L, 3L, 4L, 5L });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Int64);
            r.Should().BeOfValues(4L, 5L, 1L, 2L, 3L);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Byte()
        {
            var a = np.array(new byte[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Byte);
            r.Should().BeOfValues((byte)4, (byte)5, (byte)1, (byte)2, (byte)3);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Int16()
        {
            var a = np.array(new short[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Int16);
            r.Should().BeOfValues((short)4, (short)5, (short)1, (short)2, (short)3);
        }

        [TestMethod]
        public void Roll_PreservesDtype_UInt16()
        {
            var a = np.array(new ushort[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.UInt16);
            r.Should().BeOfValues((ushort)4, (ushort)5, (ushort)1, (ushort)2, (ushort)3);
        }

        [TestMethod]
        public void Roll_PreservesDtype_UInt32()
        {
            var a = np.array(new uint[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.UInt32);
            r.Should().BeOfValues(4u, 5u, 1u, 2u, 3u);
        }

        [TestMethod]
        public void Roll_PreservesDtype_UInt64()
        {
            var a = np.array(new ulong[] { 1, 2, 3, 4, 5 });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.UInt64);
            r.Should().BeOfValues(4ul, 5ul, 1ul, 2ul, 3ul);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Boolean()
        {
            var a = np.array(new bool[] { true, false, true, false, true });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Boolean);
            // NumPy: [False, True, True, False, True]
            r.Should().BeOfValues(false, true, true, false, true);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Boolean_Shift1()
        {
            var a = np.array(new bool[] { true, false, true, false, true });
            var r = np.roll(a, 1);
            r.typecode.Should().Be(NPTypeCode.Boolean);
            // NumPy: [True, True, False, True, False]
            r.Should().BeOfValues(true, true, false, true, false);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Boolean_NegativeShift()
        {
            var a = np.array(new bool[] { true, false, true, false, true });
            var r = np.roll(a, -1);
            r.typecode.Should().Be(NPTypeCode.Boolean);
            // NumPy: [False, True, False, True, True]
            r.Should().BeOfValues(false, true, false, true, true);
        }

        [TestMethod]
        public void Roll_PreservesDtype_Char()
        {
            var a = np.array(new char[] { 'a', 'b', 'c', 'd', 'e' });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Char);
            r.Should().BeOfValues('d', 'e', 'a', 'b', 'c');
        }

        [TestMethod]
        public void Roll_PreservesDtype_Decimal()
        {
            var a = np.array(new decimal[] { 1m, 2m, 3m, 4m, 5m });
            var r = np.roll(a, 2);
            r.typecode.Should().Be(NPTypeCode.Decimal);
            r.Should().BeOfValues(4m, 5m, 1m, 2m, 3m);
        }

        // ================================================================
        // SLICED ARRAYS (views)
        // ================================================================

        [TestMethod]
        public void Roll_SlicedArray_1D()
        {
            // orig = [0,1,2,3,4,5,6,7,8,9]
            // sliced = orig[2:7] = [2,3,4,5,6]
            // roll(sliced, 2) => [5,6,2,3,4]
            var orig = np.arange(10);
            var sliced = orig["2:7"];

            var result = np.roll(sliced, 2);

            result.Should().BeShaped(5);
            result.Should().BeOfValues(5, 6, 2, 3, 4);

            // Original not modified
            orig.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Roll_SlicedArray_2D_Axis0()
        {
            // orig = arange(20).reshape(4,5)
            // sliced = orig[1:3, :] = [[5,6,7,8,9],[10,11,12,13,14]]
            // roll(sliced, 1, axis=0) => [[10,11,12,13,14],[5,6,7,8,9]]
            var orig = np.arange(20).reshape(4, 5);
            var sliced = orig["1:3, :"];

            var result = np.roll(sliced, 1, 0);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(10, 11, 12, 13, 14, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        public void Roll_SlicedArray_2D_Axis1()
        {
            var orig = np.arange(20).reshape(4, 5);
            var sliced = orig["1:3, :"];

            var result = np.roll(sliced, 1, 1);

            result.Should().BeShaped(2, 5);
            // roll([[5,6,7,8,9],[10,11,12,13,14]], 1, axis=1)
            // => [[9,5,6,7,8],[14,10,11,12,13]]
            result.Should().BeOfValues(9, 5, 6, 7, 8, 14, 10, 11, 12, 13);
        }

        // ================================================================
        // BROADCAST ARRAYS
        // ================================================================

        [TestMethod]
        public void Roll_BroadcastArray_RowBroadcast_NoAxis()
        {
            // row = [[1,2,3,4,5]] broadcast to (3,5)
            // Flattened: [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
            // roll by 1: [5,1,2,3,4,5,1,2,3,4,5,1,2,3,4]
            // reshaped: [[5,1,2,3,4],[5,1,2,3,4],[5,1,2,3,4]]
            var row = np.array(new int[,] { { 1, 2, 3, 4, 5 } });
            var row_bc = np.broadcast_to(row, new Shape(3, 5));

            var result = np.roll(row_bc, 1);

            result.Should().BeShaped(3, 5);
            result.Should().BeOfValues(5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_BroadcastArray_RowBroadcast_Axis0()
        {
            // All rows identical, rolling along axis 0 just reorders identical rows
            // => [[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]
            var row = np.array(new int[,] { { 1, 2, 3, 4, 5 } });
            var row_bc = np.broadcast_to(row, new Shape(3, 5));

            var result = np.roll(row_bc, 1, 0);

            result.Should().BeShaped(3, 5);
            result.Should().BeOfValues(1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Roll_BroadcastArray_RowBroadcast_Axis1()
        {
            // Each row [1,2,3,4,5], rolled by 1 => [5,1,2,3,4]
            var row = np.array(new int[,] { { 1, 2, 3, 4, 5 } });
            var row_bc = np.broadcast_to(row, new Shape(3, 5));

            var result = np.roll(row_bc, 1, 1);

            result.Should().BeShaped(3, 5);
            result.Should().BeOfValues(5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_BroadcastArray_RowBroadcast_NegativeAxis1()
        {
            // Each row [1,2,3,4,5], rolled by -1 => [2,3,4,5,1]
            var row = np.array(new int[,] { { 1, 2, 3, 4, 5 } });
            var row_bc = np.broadcast_to(row, new Shape(3, 5));

            var result = np.roll(row_bc, -1, 1);

            result.Should().BeShaped(3, 5);
            result.Should().BeOfValues(2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1);
        }

        [TestMethod]
        public void Roll_BroadcastArray_ColumnBroadcast_NoAxis()
        {
            // col = [[10],[20],[30]] broadcast to (3,4)
            // Flattened: [10,10,10,10,20,20,20,20,30,30,30,30]
            // roll by 1: [30,10,10,10,10,20,20,20,20,30,30,30]
            // reshaped: [[30,10,10,10],[10,20,20,20],[20,30,30,30]]
            var col = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var col_bc = np.broadcast_to(col, new Shape(3, 4));

            var result = np.roll(col_bc, 1);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(30, 10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30);
        }

        [TestMethod]
        public void Roll_BroadcastArray_ColumnBroadcast_Axis0()
        {
            // col_bc: [[10,10,10,10],[20,20,20,20],[30,30,30,30]]
            // roll(axis=0, 1) => [[30,30,30,30],[10,10,10,10],[20,20,20,20]]
            var col = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var col_bc = np.broadcast_to(col, new Shape(3, 4));

            var result = np.roll(col_bc, 1, 0);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(30, 30, 30, 30, 10, 10, 10, 10, 20, 20, 20, 20);
        }

        [TestMethod]
        public void Roll_BroadcastArray_ColumnBroadcast_Axis1()
        {
            // col_bc: [[10,10,10,10],[20,20,20,20],[30,30,30,30]]
            // roll(axis=1, 1) => same (all values in each row are identical)
            var col = np.array(new int[,] { { 10 }, { 20 }, { 30 } });
            var col_bc = np.broadcast_to(col, new Shape(3, 4));

            var result = np.roll(col_bc, 1, 1);

            result.Should().BeShaped(3, 4);
            result.Should().BeOfValues(10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30);
        }

        // ================================================================
        // BOOLEAN 2D
        // ================================================================

        [TestMethod]
        public void Roll_Bool2D_NoAxis()
        {
            // [[True, False],[False, True]]
            // flattened: [T,F,F,T], roll 1 => [T,T,F,F]
            // reshaped: [[T,T],[F,F]]
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            var r = np.roll(a, 1);
            r.Should().BeShaped(2, 2);
            r.Should().BeOfValues(true, true, false, false);
        }

        [TestMethod]
        public void Roll_Bool2D_Axis0()
        {
            // [[True, False],[False, True]]
            // roll(axis=0, 1) => [[False,True],[True,False]]
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            var r = np.roll(a, 1, 0);
            r.Should().BeShaped(2, 2);
            r.Should().BeOfValues(false, true, true, false);
        }

        [TestMethod]
        public void Roll_Bool2D_Axis1()
        {
            // [[True, False],[False, True]]
            // roll(axis=1, 1) => [[False,True],[True,False]]
            var a = np.array(new bool[,] { { true, false }, { false, true } });
            var r = np.roll(a, 1, 1);
            r.Should().BeShaped(2, 2);
            r.Should().BeOfValues(false, true, true, false);
        }

        // ================================================================
        // INSTANCE METHODS: a.roll(shift, axis) and a.roll(shift)
        // ================================================================

        [TestMethod]
        public void Roll_InstanceMethod_NoAxis()
        {
            var x = np.arange(10);
            var result = x.roll(2);

            result.Should().BeShaped(10);
            result.Should().BeOfValues(8, 9, 0, 1, 2, 3, 4, 5, 6, 7);
        }

        [TestMethod]
        public void Roll_InstanceMethod_WithAxis()
        {
            var x = np.arange(10).reshape(2, 5);
            var result = x.roll(1, 0);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_InstanceMethod_WithAxis1()
        {
            var x = np.arange(10).reshape(2, 5);
            var result = x.roll(1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Roll_InstanceMethod_EquivalentToStaticMethod()
        {
            var x = np.arange(12).reshape(3, 4);

            // No axis
            np.array_equal(x.roll(2), np.roll(x, 2)).Should().BeTrue();

            // With axis
            np.array_equal(x.roll(1, 0), np.roll(x, 1, 0)).Should().BeTrue();
            np.array_equal(x.roll(1, 1), np.roll(x, 1, 1)).Should().BeTrue();
            np.array_equal(x.roll(-1, 0), np.roll(x, -1, 0)).Should().BeTrue();
        }

        // ================================================================
        // 1D ARRAY WITH axis=0
        // ================================================================

        [TestMethod]
        public void Roll_1D_WithAxis0()
        {
            // np.roll(np.arange(5), 2, axis=0) => [3 4 0 1 2]
            var x = np.arange(5);
            var result = np.roll(x, 2, 0);

            result.Should().BeShaped(5);
            result.Should().BeOfValues(3, 4, 0, 1, 2);
        }

        // ================================================================
        // NumPy test_roll_empty (exact match)
        // ================================================================

        [TestMethod]
        public void Roll_NumPy_test_roll_empty()
        {
            var x = np.array(new double[0]);
            var result = np.roll(x, 1);
            result.size.Should().Be(0);
        }

        // ================================================================
        // SHAPE PRESERVATION
        // ================================================================

        [TestMethod]
        public void Roll_2D_PreservesShape_NoAxis()
        {
            var x = np.arange(12).reshape(3, 4);
            var result = np.roll(x, 5);
            result.Should().BeShaped(3, 4);
        }

        [TestMethod]
        public void Roll_3D_PreservesShape_NoAxis()
        {
            var x = np.arange(24).reshape(2, 3, 4);
            var result = np.roll(x, 3);
            result.Should().BeShaped(2, 3, 4);
        }

        [TestMethod]
        public void Roll_2D_PreservesShape_WithAxis()
        {
            var x = np.arange(12).reshape(3, 4);
            np.roll(x, 1, 0).Should().BeShaped(3, 4);
            np.roll(x, 1, 1).Should().BeShaped(3, 4);
        }

        // ================================================================
        // DTYPE PRESERVATION FOR 2D WITH AXIS
        // ================================================================

        [TestMethod]
        public void Roll_2D_PreservesDtype_Float()
        {
            var a = np.array(new float[,] { { 1f, 2f, 3f }, { 4f, 5f, 6f } });
            var r = np.roll(a, 1, 0);
            r.typecode.Should().Be(NPTypeCode.Single);
        }

        [TestMethod]
        public void Roll_2D_PreservesDtype_Double()
        {
            var a = np.array(new double[,] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
            var r = np.roll(a, 1, 1);
            r.typecode.Should().Be(NPTypeCode.Double);
        }

        // ================================================================
        // MULTI-AXIS TUPLE SHIFT (NumPy feature not yet in NumSharp)
        // These tests document NumPy behavior and are marked OpenBugs
        // since NumSharp only supports single int shift / single int axis.
        // ================================================================

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_ScalarShift_TupleAxis()
        {
            // NumPy: np.roll(x2, 1, axis=(0,1))
            // Applies shift=1 to axis=0, then shift=1 to axis=1 sequentially
            // Result: [[9,5,6,7,8],[4,0,1,2,3]]
            //
            // NumSharp does not currently support tuple axis parameter.
            // This test documents the expected NumPy behavior.
            var x2 = np.arange(10).reshape(2, 5);

            // Simulate by chaining: roll by 1 on axis=0, then roll by 1 on axis=1
            var step1 = np.roll(x2, 1, 0);
            var result = np.roll(step1, 1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(9, 5, 6, 7, 8, 4, 0, 1, 2, 3);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_TupleShift_TupleAxis()
        {
            // NumPy: np.roll(x2, (1, 0), axis=(0, 1))
            // shift=1 on axis=0, shift=0 on axis=1
            // Result: [[5,6,7,8,9],[0,1,2,3,4]]
            //
            // NumSharp does not currently support tuple shift/axis parameters.
            var x2 = np.arange(10).reshape(2, 5);

            // Simulate: roll by 1 on axis 0, then roll by 0 on axis 1
            var step1 = np.roll(x2, 1, 0);
            var result = np.roll(step1, 0, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_NegativeShift_TupleAxis()
        {
            // NumPy: np.roll(x2, (-1, 0), axis=(0, 1))
            // For 2-row array, roll -1 on axis=0 same as roll 1
            // Result: [[5,6,7,8,9],[0,1,2,3,4]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, -1, 0);
            var result = np.roll(step1, 0, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_BothShift1()
        {
            // NumPy: np.roll(x2, (1, 1), axis=(0, 1))
            // Result: [[9,5,6,7,8],[4,0,1,2,3]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, 1, 0);
            var result = np.roll(step1, 1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(9, 5, 6, 7, 8, 4, 0, 1, 2, 3);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_BothNegative()
        {
            // NumPy: np.roll(x2, (-1, -1), axis=(0, 1))
            // Result: [[6,7,8,9,5],[1,2,3,4,0]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, -1, 0);
            var result = np.roll(step1, -1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(6, 7, 8, 9, 5, 1, 2, 3, 4, 0);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_SameAxis_Twice()
        {
            // NumPy: np.roll(x2, 1, axis=(0, 0))
            // Applies shift=1 twice on axis=0 => net effect: shift=2 on axis=0
            // For 2-row array: shift=2 % 2 = 0 => same as original
            // Result: [[0,1,2,3,4],[5,6,7,8,9]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, 1, 0);
            var result = np.roll(step1, 1, 0);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_SameAxis1_Twice()
        {
            // NumPy: np.roll(x2, 1, axis=(1, 1))
            // Applies shift=1 twice on axis=1 => net shift=2 on axis=1
            // Result: [[3,4,0,1,2],[8,9,5,6,7]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, 1, 1);
            var result = np.roll(step1, 1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(3, 4, 0, 1, 2, 8, 9, 5, 6, 7);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_ZeroAndOne()
        {
            // NumPy: np.roll(x2, (0, 1), axis=(0, 1))
            // shift=0 on axis=0 (no change), shift=1 on axis=1
            // Result: [[4,0,1,2,3],[9,5,6,7,8]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, 0, 0);
            var result = np.roll(step1, 1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        [TestMethod]
        [TestCategory("OpenBugs")]
        public void Roll_MultiAxis_ZeroAndNegative()
        {
            // NumPy: np.roll(x2, (0, -1), axis=(0, 1))
            // shift=0 on axis=0, shift=-1 on axis=1
            // Result: [[1,2,3,4,0],[6,7,8,9,5]]
            var x2 = np.arange(10).reshape(2, 5);

            var step1 = np.roll(x2, 0, 0);
            var result = np.roll(step1, -1, 1);

            result.Should().BeShaped(2, 5);
            result.Should().BeOfValues(1, 2, 3, 4, 0, 6, 7, 8, 9, 5);
        }

        // ================================================================
        // DOUBLE VALUES (not just integers)
        // ================================================================

        [TestMethod]
        public void Roll_1D_DoubleValues()
        {
            var a = np.array(new double[] { 1.5, 2.5, 3.5, 4.5, 5.5 });
            var r = np.roll(a, 2);
            r.Should().BeOfValues(4.5, 5.5, 1.5, 2.5, 3.5);
        }

        [TestMethod]
        public void Roll_2D_DoubleValues_WithAxis()
        {
            var a = np.array(new double[,] { { 1.1, 2.2, 3.3 }, { 4.4, 5.5, 6.6 } });
            var r = np.roll(a, 1, 1);
            r.Should().BeOfValues(3.3, 1.1, 2.2, 6.6, 4.4, 5.5);
        }

        // ================================================================
        // REGRESSION: Existing tests from NdArray.Roll.Test.cs (re-verified)
        // ================================================================

        [TestMethod]
        public void Roll_Regression_Base1DTest_Positive()
        {
            NDArray nd1 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var nd2 = nd1.roll(2);

            nd2.Should().BeOfValues(9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            // Original not modified
            nd1.Should().BeOfValues(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
        }

        [TestMethod]
        public void Roll_Regression_Base1DTest_Negative()
        {
            NDArray nd1 = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var nd2 = nd1.roll(-2);

            nd2.Should().BeOfValues(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0);
            nd1.Should().BeOfValues(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0);
        }

        [TestMethod]
        public void Roll_Regression_Base2DTest_ShapePreserved()
        {
            var nd1 = np.arange(10).reshape(2, 5);
            var nd2 = nd1.roll(2);
            nd2.Should().BeShaped(2, 5);

            var nd3 = nd1.roll(-2);
            nd3.Should().BeShaped(2, 5);
        }

        [TestMethod]
        public void Roll_Regression_RollWithAxis0()
        {
            var x2 = np.arange(10).reshape(2, 5);
            var x3 = x2.roll(1, 0);

            x3.Should().BeShaped(2, 5);
            x3.Should().BeOfValues(5, 6, 7, 8, 9, 0, 1, 2, 3, 4);
        }

        [TestMethod]
        public void Roll_Regression_RollWithAxis1()
        {
            var x2 = np.arange(10).reshape(2, 5);
            var x4 = x2.roll(1, 1);

            x4.Should().BeShaped(2, 5);
            x4.Should().BeOfValues(4, 0, 1, 2, 3, 9, 5, 6, 7, 8);
        }

        // ================================================================
        // EDGE CASES
        // ================================================================

        [TestMethod]
        public void Roll_LargeArray()
        {
            // Test with a larger array to ensure no off-by-one errors
            var a = np.arange(100);
            var r = np.roll(a, 37);

            // First element should be 100-37=63
            r.GetInt32(0).Should().Be(63);
            // Element at index 37 should be 0
            r.GetInt32(37).Should().Be(0);
            r.size.Should().Be(100);
        }

        [TestMethod]
        public void Roll_2D_LargeShift_Axis0()
        {
            // shift > dim size should wrap correctly
            var x = np.arange(12).reshape(3, 4);
            // roll by 7 on axis 0: 7 % 3 = 1
            var r = np.roll(x, 7, 0);
            var expected = np.roll(x, 1, 0);
            np.array_equal(r, expected).Should().BeTrue();
        }

        [TestMethod]
        public void Roll_2D_LargeNegativeShift_Axis1()
        {
            // shift < -dim_size should wrap correctly
            var x = np.arange(12).reshape(3, 4);
            // roll by -9 on axis 1: (-9 % 4 + 4) % 4 = (-1 + 4) % 4 = 3
            var r = np.roll(x, -9, 1);
            var expected = np.roll(x, -1, 1);
            np.array_equal(r, expected).Should().BeTrue();
        }

        [TestMethod]
        public void Roll_2D_ShiftEqualsAxisDim()
        {
            // shift == dim size => no change (same values)
            var x = np.arange(12).reshape(3, 4);
            var r = np.roll(x, 3, 0);  // 3 % 3 = 0
            np.array_equal(r, x).Should().BeTrue();
        }

        [TestMethod]
        public void Roll_3D_ShiftEqualsAxisDim()
        {
            var x = np.arange(24).reshape(2, 3, 4);
            var r = np.roll(x, 4, 2);  // 4 % 4 = 0
            np.array_equal(r, x).Should().BeTrue();
        }

        // ================================================================
        // ASYMMETRIC 2D SHAPES
        // ================================================================

        [TestMethod]
        public void Roll_2D_TallArray()
        {
            // 5x2 array
            var x = np.arange(10).reshape(5, 2);
            var r = np.roll(x, 2, 0);

            // Rows shift down by 2: rows [3,4] wrap to top
            // Original: [[0,1],[2,3],[4,5],[6,7],[8,9]]
            // After roll 2 on axis 0: [[6,7],[8,9],[0,1],[2,3],[4,5]]
            r.Should().BeShaped(5, 2);
            r.Should().BeOfValues(6, 7, 8, 9, 0, 1, 2, 3, 4, 5);
        }

        [TestMethod]
        public void Roll_2D_WideArray()
        {
            // 2x6 array
            var x = np.arange(12).reshape(2, 6);
            var r = np.roll(x, 3, 1);

            // Columns shift right by 3
            // [[0,1,2,3,4,5],[6,7,8,9,10,11]]
            // => [[3,4,5,0,1,2],[9,10,11,6,7,8]]
            r.Should().BeShaped(2, 6);
            r.Should().BeOfValues(3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8);
        }
    }
}
