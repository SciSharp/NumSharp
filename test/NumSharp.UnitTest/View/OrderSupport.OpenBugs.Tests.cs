using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.View
{
    /// <summary>
    ///     TDD coverage for NumPy memory order support (C/F/A/K) across the NumSharp API surface.
    ///     Each test uses NumPy 2.x's exact output as the expected value.
    ///     Failures are marked [OpenBugs] so CI continues to pass while tracking the gap.
    ///
    ///     Test organization:
    ///     - Section 1: Creation APIs (np.zeros, np.ones, np.empty, np.full)
    ///     - Section 2: Copy/conversion APIs (np.copy, NDArray.copy, np.array)
    ///     - Section 3: Manipulation (ravel, flatten, reshape)
    ///     - Section 4: Arithmetic output layout
    ///     - Section 5: Reductions on F-contig (math-equivalent)
    ///     - Section 6: Slicing contiguity preservation
    ///     - Section 7: Broadcasting output layout
    ///     - Section 8: Transpose behavior
    ///     - Section 9: Iteration order
    ///     - Section 10: Order property derivation
    /// </summary>
    [TestClass]
    public class OrderSupportOpenBugsTests
    {
        // ============================================================================
        // Section 1: Creation APIs — np.zeros, np.ones, np.empty, np.full
        // NumPy: only 'C' and 'F' accepted; 'A' and 'K' throw ValueError
        // ============================================================================

        [TestMethod]
        public void NpZeros_Default_IsCContig()
        {
            // NumPy: np.zeros((3,4)) -> C=True, F=False
            var arr = np.zeros(new Shape(3L, 4L), np.int32);
            arr.Shape.IsContiguous.Should().BeTrue();
            arr.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void NpZeros_FShape_PreservesFContig()
        {
            // Workaround: passing an F-contig Shape to np.zeros preserves the layout.
            // (np.zeros has no order parameter; this documents the functional workaround.)
            var shape = new Shape(new long[] { 3, 4 }, 'F');
            var arr = np.zeros(shape);
            arr.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpOnes_FShape_PreservesFContig()
        {
            var shape = new Shape(new long[] { 3, 4 }, 'F');
            var arr = np.ones(shape);
            arr.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpFull_FShape_PreservesFContig()
        {
            var shape = new Shape(new long[] { 3, 4 }, 'F');
            var arr = np.full(shape, 7);
            arr.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpEmpty_FOrder_IsFContig()
        {
            // NumPy: np.empty((3,4), order='F') -> C=False, F=True
            var arr = np.empty(new Shape(3L, 4L), order: 'F', dtype: typeof(int));
            arr.Shape.IsContiguous.Should().BeFalse();
            arr.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpEmpty_COrder_IsCContig()
        {
            var arr = np.empty(new Shape(3L, 4L), order: 'C', dtype: typeof(int));
            arr.Shape.IsContiguous.Should().BeTrue();
            arr.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void NpEmpty_AOrder_Throws()
        {
            // NumPy: np.empty((3,4), order='A') -> ValueError
            Action act = () => np.empty(new Shape(3L, 4L), order: 'A');
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void NpEmpty_KOrder_Throws()
        {
            // NumPy: np.empty((3,4), order='K') -> ValueError
            Action act = () => np.empty(new Shape(3L, 4L), order: 'K');
            act.Should().Throw<ArgumentException>();
        }

        // ============================================================================
        // Section 2: Copy/conversion APIs — np.copy, NDArray.copy
        // NumPy: all 4 orders accepted; A/K resolve based on source
        // ============================================================================

        [TestMethod]
        public void NpCopy_DefaultOrder_ProducesCContig()
        {
            // NumPy: np.copy(c_src) -> C=True (default)
            var src = np.arange(12).reshape(3, 4);
            var copy = np.copy(src);
            copy.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.copy ignores order parameter (see np.copy.cs:12 TODO)
        public void NpCopy_FOrder_ProducesFContig()
        {
            // NumPy: np.copy(c_src, order='F') -> F=True
            var src = np.arange(12).reshape(3, 4);
            var copy = np.copy(src, order: 'F');
            copy.Shape.IsFContiguous.Should().BeTrue();
            copy.Shape.IsContiguous.Should().BeFalse();
        }

        [TestMethod]
        [OpenBugs] // np.copy ignores order parameter
        public void NpCopy_AOrder_FSource_ProducesFContig()
        {
            // NumPy: np.copy(f_src, order='A') with F-contig src -> F=True
            var fSrc = np.arange(12).reshape(3, 4).T;  // F-contig via transpose
            var copy = np.copy(fSrc, order: 'A');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.copy ignores order parameter
        public void NpCopy_KOrder_FSource_ProducesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var copy = np.copy(fSrc, order: 'K');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpCopy_AOrder_CSource_ProducesCContig()
        {
            // Passes because current np.copy ignores order and always produces C-contig —
            // for 'A' with C-contig source, NumPy also expects C output.
            var src = np.arange(12).reshape(3, 4);
            var copy = np.copy(src, order: 'A');
            copy.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // NDArray.copy ignores order parameter (see NDArray.Copy.cs:11 TODO)
        public void NDArrayCopy_FOrder_ProducesFContig()
        {
            var src = np.arange(12).reshape(3, 4);
            var copy = src.copy(order: 'F');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // NDArray.copy ignores order parameter
        public void NDArrayCopy_AOrder_FSource_ProducesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var copy = fSrc.copy(order: 'A');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 3: Manipulation — flatten (ravel has no order overload)
        // NumPy:
        //   arr = np.arange(12).reshape(3,4), arr.flatten('C') = [0..11]
        //   arr.flatten('F') = [0,4,8,1,5,9,2,6,10,3,7,11]
        // ============================================================================

        [TestMethod]
        public void Flatten_CContig_COrder_MatchesNumPy()
        {
            // NumPy: arr.flatten('C') = [0..11]
            var arr = np.arange(12).reshape(3, 4);
            var r = arr.flatten(order: 'C');
            var expected = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            for (int i = 0; i < 12; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        [OpenBugs] // arr.flatten ignores order parameter
        public void Flatten_CContig_FOrder_MatchesNumPy()
        {
            // NumPy: arr.flatten('F') = [0,4,8,1,5,9,2,6,10,3,7,11]
            var arr = np.arange(12).reshape(3, 4);
            var r = arr.flatten(order: 'F');
            var expected = new int[] { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
            for (int i = 0; i < 12; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        public void Flatten_FContig_COrder_MatchesNumPy()
        {
            // Passes because flatten default is C-order logical traversal, which for
            // F-contig (4,3) with values [[0,4,8],[1,5,9],[2,6,10],[3,7,11]] gives
            // [0,4,8,1,5,9,2,6,10,3,7,11] — matches NumPy arrT.flatten('C').
            var arrT = np.arange(12).reshape(3, 4).T;  // F-contig
            var r = arrT.flatten(order: 'C');
            var expected = new int[] { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
            for (int i = 0; i < 12; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        [OpenBugs] // arr.flatten ignores order parameter
        public void Flatten_FContig_FOrder_MatchesNumPy()
        {
            // NumPy: arrT.flatten('F') = [0,1,2,3,4,5,6,7,8,9,10,11] (memory order for F-contig)
            var arrT = np.arange(12).reshape(3, 4).T;
            var r = arrT.flatten(order: 'F');
            var expected = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
            for (int i = 0; i < 12; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        [OpenBugs] // ravel has no order overload (np.ravel.cs / NDArray.ravel.cs)
        public void Ravel_FOrder_ApiGap()
        {
            // NumPy: arr.ravel('F') = [0,4,8,1,5,9,2,6,10,3,7,11]
            // NumSharp's NDArray.ravel() and np.ravel() have no order parameter.
            // This test documents the API gap; once an order-aware overload is added,
            // remove [OpenBugs] and assert the expected NumPy values.
            var arr = np.arange(12).reshape(3, 4);
            var r = arr.ravel();
            // Current (default) behavior is C-order; test fails if order='F' is wired.
            var expectedFOrder = new int[] { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
            for (int i = 0; i < 12; i++)
                ((int)r[i]).Should().Be(expectedFOrder[i]);
        }

        // ============================================================================
        // Section 4: Arithmetic output layout
        // NumPy:
        //   f * 2 -> preserves F-contig
        //   f + f (both F-contig) -> F-contig output
        // ============================================================================

        [TestMethod]
        [OpenBugs] // NumSharp element-wise ops always produce C-contig output
        public void Arithmetic_FContig_ScalarMul_PreservesFContig()
        {
            // NumPy: f_arr * 2 preserves F-contig output
            var fArr = np.arange(12).reshape(3, 4).T;  // F-contig
            var r = fArr * 2;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: scalar op on F-contig preserves F layout");
        }

        [TestMethod]
        public void Arithmetic_FContig_ScalarMul_ValuesCorrect()
        {
            // Math result must match regardless of layout
            var fArr = np.arange(12).reshape(3, 4).T;  // shape (4,3) values [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
            var r = fArr * 2;
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[0, 1]).Should().Be(8);
            ((int)r[3, 2]).Should().Be(22);
        }

        [TestMethod]
        [OpenBugs] // NumSharp element-wise on both F-contig produces C output
        public void Arithmetic_FPlusF_PreservesFContig()
        {
            // NumPy: when both operands F-contig, output is F-contig
            var a = np.arange(12).reshape(3, 4).T;  // F-contig
            var b = np.arange(12).reshape(3, 4).T;  // F-contig
            var r = a + b;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: F+F preserves F output layout");
        }

        // ============================================================================
        // Section 5: Reductions — math result must match regardless of layout
        // ============================================================================

        [TestMethod]
        public void Reduction_Sum_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.sum() = 66 (for arange(12))
            var fArr = np.arange(12).reshape(3, 4).T;
            ((long)np.sum(fArr)).Should().Be(66);
        }

        [TestMethod]
        public void Reduction_SumAxis0_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.sum(axis=0) = [6, 22, 38]
            var fArr = np.arange(12).reshape(3, 4).T;  // shape (4,3)
            var r = np.sum(fArr, axis: 0);
            ((long)r[0]).Should().Be(6);
            ((long)r[1]).Should().Be(22);
            ((long)r[2]).Should().Be(38);
        }

        [TestMethod]
        public void Reduction_SumAxis1_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.sum(axis=1) = [12, 15, 18, 21]
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.sum(fArr, axis: 1);
            ((long)r[0]).Should().Be(12);
            ((long)r[1]).Should().Be(15);
            ((long)r[2]).Should().Be(18);
            ((long)r[3]).Should().Be(21);
        }

        [TestMethod]
        public void Reduction_Mean_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.mean() = 5.5
            var fArr = np.arange(12).reshape(3, 4).T;
            ((double)np.mean(fArr)).Should().Be(5.5);
        }

        [TestMethod]
        public void Reduction_Min_FContig_MatchesNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            ((int)np.min(fArr)).Should().Be(0);
        }

        [TestMethod]
        public void Reduction_Max_FContig_MatchesNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            ((int)np.max(fArr)).Should().Be(11);
        }

        // ============================================================================
        // Section 6: Slicing contiguity preservation
        // NumPy:
        //   f_arr[1:3, :] shape (2,3) -> neither C nor F contig
        //   f_arr[:, 1:2] shape (4,1) -> both C and F contig (1-col)
        // ============================================================================

        [TestMethod]
        public void Slice_FContig_Rows_IsNotContig()
        {
            // NumPy: f_arr[1:3, :] -> neither C nor F contig
            var fArr = np.arange(12).reshape(3, 4).T;  // F-contig shape (4,3)
            var s = fArr["1:3, :"];
            s.Shape.IsContiguous.Should().BeFalse();
            s.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void Slice_FContig_SingleColumn_IsBothContig()
        {
            // NumPy: f_arr[:, 1:2] shape (4,1) is both C and F contig
            var fArr = np.arange(12).reshape(3, 4).T;
            var s = fArr[":, 1:2"];
            s.Shape.IsContiguous.Should().BeTrue(
                "1-column slice of F-contig has stride=1 so is C-contig");
            s.Shape.IsFContiguous.Should().BeTrue(
                "1-column slice is also F-contig (both flags set)");
        }

        // ============================================================================
        // Section 7: Broadcasting output layout
        // ============================================================================

        [TestMethod]
        [OpenBugs] // NumSharp broadcast ops always produce C-contig output
        public void Broadcast_FContig_PlusFCol_PreservesFContig()
        {
            // NumPy: F-contig (4,3) + F-contig (4,1) -> F-contig output
            var fArr = np.arange(12).reshape(3, 4).T;  // F-contig (4,3)
            var fCol = np.arange(4).reshape(4, 1);  // (4,1) both C and F contig
            var r = fArr + fCol;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: F+F broadcast produces F-contig output");
        }

        // ============================================================================
        // Section 8: Transpose behavior
        // NumPy:
        //   C-contig.T -> F-contig
        //   F-contig.T -> C-contig
        // ============================================================================

        [TestMethod]
        public void Transpose_CContig_ProducesFContig()
        {
            // NumPy: np.arange(6).reshape(2,3).T flags: C=False, F=True
            var c = np.arange(6).reshape(2, 3);
            c.Shape.IsContiguous.Should().BeTrue();
            var t = c.T;
            t.Shape.IsContiguous.Should().BeFalse();
            t.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Transpose_FContig_ProducesCContig()
        {
            // NumPy: F-contig.T -> C-contig
            var f = np.arange(6).reshape(2, 3).T;  // F-contig shape (3,2)
            f.Shape.IsFContiguous.Should().BeTrue();
            var tt = f.T;
            tt.Shape.IsContiguous.Should().BeTrue(
                "Transpose of F-contig produces C-contig");
        }

        [TestMethod]
        public void Transpose_RoundTrip_IsCContig()
        {
            // arr.T.T should have same layout as arr
            var c = np.arange(6).reshape(2, 3);
            var roundTrip = c.T.T;
            roundTrip.Shape.IsContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 9: Iteration order
        // NumPy:
        //   f_arr.flat: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11] (always C-order)
        // ============================================================================

        [TestMethod]
        public void Iteration_FContig_IndexingIsCOrder()
        {
            // NumPy: arr.flat always iterates C-order regardless of memory layout
            // NumSharp: indexing (shape.GetOffset) produces C-order logical traversal
            var fArr = np.arange(12).reshape(3, 4).T;  // F-contig (4,3)
            var expected = new int[] { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
            int idx = 0;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    ((int)fArr[i, j]).Should().Be(expected[idx++]);
        }

        // ============================================================================
        // Section 10: Order property derivation
        // ============================================================================

        [TestMethod]
        public void OrderProperty_FContigArray_ReportsF()
        {
            var fArr = np.arange(6).reshape(2, 3).T;
            fArr.Shape.Order.Should().Be('F');
        }

        [TestMethod]
        public void OrderProperty_CContigArray_ReportsC()
        {
            var cArr = np.arange(6).reshape(2, 3);
            cArr.Shape.Order.Should().Be('C');
        }

        [TestMethod]
        public void OrderProperty_NonContigSlice_ReportsC()
        {
            // Non-contiguous slice: Order defaults to 'C' as reference
            var arr = np.arange(12).reshape(3, 4);
            var sliced = arr["::2, ::2"];
            sliced.Shape.IsContiguous.Should().BeFalse();
            sliced.Shape.IsFContiguous.Should().BeFalse();
            sliced.Shape.Order.Should().Be('C');
        }
    }
}
