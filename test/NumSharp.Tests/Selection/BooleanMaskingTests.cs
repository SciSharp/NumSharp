using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Selection
{
    /// <summary>
    /// Tests for boolean masking based on NumPy 2.4.2 behavior.
    /// </summary>
    [TestClass]
    public class BooleanMaskingTests
    {
        [TestMethod]
        public void ExplicitMask_1D_FiltersCorrectly()
        {
            // NumPy: a[mask] where mask = [True, False, True, False, True, False]
            // Result: [1, 3, 5]
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 3, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void Condition_1D_FiltersCorrectly()
        {
            // NumPy: a[a % 2 == 1] = [1, 3, 5]
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var result = a[a % 2 == 1];

            result.Should().BeOfValues(1, 3, 5).And.BeShaped(3);
        }

        [TestMethod]
        public void ExplicitMask_MatchesCondition()
        {
            // Explicit mask and condition should produce same result
            var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
            var mask = np.array(new[] { true, false, true, false, true, false }).MakeGeneric<bool>();

            var explicit_result = a[mask];
            var condition_result = a[a % 2 == 1];

            explicit_result.Should().BeOfValues(1, 3, 5);
            condition_result.Should().BeOfValues(1, 3, 5);
        }

        [TestMethod]
        public void AllTrue_ReturnsAll()
        {
            // NumPy: a[[T,T,T]] = [1, 2, 3]
            var a = np.array(new[] { 1, 2, 3 });
            var mask = np.array(new[] { true, true, true }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 2, 3).And.BeShaped(3);
        }

        [TestMethod]
        public void AllFalse_ReturnsEmpty()
        {
            // NumPy: a[[F,F,F]] = [], shape (0,)
            var a = np.array(new[] { 1, 2, 3 });
            var mask = np.array(new[] { false, false, false }).MakeGeneric<bool>();
            var result = a[mask];

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(1, result.ndim);  // 1D empty array
        }

        [TestMethod]
        public void TwoDimensional_RowSelection()
        {
            // NumPy: 2D array with 1D mask selects rows
            // a = [[1,2,3],[4,5,6],[7,8,9]]
            // mask = [True, False, True]
            // result = [[1,2,3],[7,8,9]], shape (2,3)
            var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } });
            var mask = np.array(new[] { true, false, true }).MakeGeneric<bool>();
            var result = a[mask];

            // First row should be [1,2,3], second row [7,8,9]
            Assert.AreEqual(2, result.shape[0]);  // 2 rows selected
            Assert.AreEqual(3, result.shape[1]);  // 3 columns
            result["0"].Should().BeOfValues(1, 2, 3);
            result["1"].Should().BeOfValues(7, 8, 9);
        }

        [TestMethod]
        public void TwoDimensional_ElementMask_Flattens()
        {
            // NumPy: 2D array with 2D mask of same shape selects elements, flattens
            // a = [[1,2],[3,4]]
            // mask = [[True,False],[False,True]]
            // result = [1, 4], shape (2,)
            var a = np.array(new[,] { { 1, 2 }, { 3, 4 } });
            var mask = np.array(new[,] { { true, false }, { false, true } }).MakeGeneric<bool>();
            var result = a[mask];

            result.Should().BeOfValues(1, 4).And.BeShaped(2);
        }

        [TestMethod]
        public void EmptyResult_PreservesDtype()
        {
            // NumPy: Empty result preserves dtype
            var a = np.array(new double[] { 1.0, 2.0, 3.0 });
            var mask = np.array(new[] { false, false, false }).MakeGeneric<bool>();
            var result = a[mask];

            Assert.AreEqual(0, result.size);
            Assert.AreEqual(NPTypeCode.Double, result.typecode);
        }

        // ================================================================
        // A leading mask over trailing extents of 1 selects rows (parity QA-23; NumPy 2.4.2's answers)
        // ================================================================

        /// <summary>
        ///     <c>a[mask, 0:1]</c> (a mask on axis 0, then a slice that keeps ONE column) is NumPy's <c>(k, 1)</c> selection:
        ///     <c>np.arange(6).reshape(2, 3)[[False, True], 0:1]</c> is <c>[[3]]</c> and <c>a[mask, -1:]</c> is <c>[[5]]</c>.
        ///     The getter slices first (a strided <c>(2, 1)</c> view) and used the leading mask as a FULL element mask
        ///     whenever the trailing block was one element, which raised <c>ValueError: non-broadcastable operand with shape
        ///     (2,1) doesn't match the broadcast shape (2,2)</c>.
        /// </summary>
        [TestMethod]
        public void LeadingMask_ThenSliceKeepingOneColumn_SelectsRows()
        {
            using var a = np.arange(6L).reshape(new Shape(2, 3));
            using var mask = np.array(new[] { false, true });

            using var first = a[mask, "0:1"];
            using var last = a[mask, "-1:"];

            first.Should().BeShaped(1, 1).And.BeOfValues(3L);
            last.Should().BeShaped(1, 1).And.BeOfValues(5L);
        }

        /// <summary>
        ///     A plain strided <c>(n, 1)</c> view masked by an <c>(n,)</c> mask keeps its trailing axis:
        ///     <c>np.arange(6).reshape(3, 2)[:, :1][[True, False, True]]</c> is <c>[[0], [4]]</c>, and a strided
        ///     <c>(3, 1, 1)</c> view masked on axis 0 keeps both unit axes (<c>[[[5]], [[9]]]</c>).
        /// </summary>
        [TestMethod]
        public void LeadingMask_OverStridedUnitTrailingView_KeepsTheTrailingAxes()
        {
            using var column = np.arange(6L).reshape(new Shape(3, 2))[":, :1"];
            using var deep = np.arange(12L).reshape(new Shape(3, 4, 1))[":, 1:2, :"];

            using var rows = column[np.array(new[] { true, false, true })];
            using var slabs = deep[np.array(new[] { false, true, true })];

            rows.Should().BeShaped(2, 1).And.BeOfValues(0L, 4L);
            slabs.Should().BeShaped(2, 1, 1).And.BeOfValues(5L, 9L);
        }

        /// <summary>
        ///     The same holds for a 3-D source and for a 2-D leading mask: <c>b3[[True, False], 2:3, 3:4]</c> of
        ///     <c>arange(24).reshape(2, 3, 4)</c> is <c>[[[11]]]</c>; <c>b3[m2, 0:1]</c> / <c>b3[m2, -1:]</c> with
        ///     <c>m2 = [[T, F, T], [F, T, F]]</c> are <c>(3, 1)</c> (<c>[0, 8, 16]</c> / <c>[3, 11, 19]</c>); a 2-D mask over
        ///     the reversed <c>arange(6).reshape(3, 2, 1)[::-1]</c> is <c>(4, 1)</c> <c>[4, 3, 0, 1]</c>.
        /// </summary>
        [TestMethod]
        public void LeadingMask_HigherRankAndTwoDimensionalMasks_SelectLikeNumPy()
        {
            using var b3 = np.arange(24L).reshape(new Shape(2, 3, 4));
            using var m2 = np.array(new[,] { { true, false, true }, { false, true, false } });
            using var reversed = np.arange(6L).reshape(new Shape(3, 2, 1))["::-1"];
            using var m32 = np.array(new[,] { { true, false }, { false, true }, { true, true } });

            using var corner = b3[np.array(new[] { true, false }), "2:3", "3:4"];
            using var firstColumn = b3[m2, "0:1"];
            using var lastColumn = b3[m2, "-1:"];
            using var fromReversed = reversed[m32];

            corner.Should().BeShaped(1, 1, 1).And.BeOfValues(11L);
            firstColumn.Should().BeShaped(3, 1).And.BeOfValues(0L, 8L, 16L);
            lastColumn.Should().BeShaped(3, 1).And.BeOfValues(3L, 11L, 19L);
            fromReversed.Should().BeShaped(4, 1).And.BeOfValues(4L, 3L, 0L, 1L);
        }

        /// <summary>
        ///     A mask of the array's FULL rank is still an element mask on the same strided view (the one case where the mask
        ///     is the gather mask itself): <c>np.arange(6).reshape(3, 2)[:, :1][[[True], [False], [True]]]</c> is the 1-D
        ///     <c>[0, 4]</c>, and the setter through a leading mask writes the rows (<c>[[-1, 1], [2, 3], [-1, 5]]</c>).
        /// </summary>
        [TestMethod]
        public void FullRankMask_OverStridedUnitTrailingView_StaysAnElementMask()
        {
            using var source = np.arange(6L).reshape(new Shape(3, 2));
            using var column = source[":, :1"];

            using var elements = column[np.array(new[,] { { true }, { false }, { true } })];
            column[np.array(new[] { true, false, true })] = -1L;

            elements.Should().BeShaped(2).And.BeOfValues(0L, 4L);
            source.Should().BeOfValues(-1L, 1L, 2L, 3L, -1L, 5L);
        }
    }
}
