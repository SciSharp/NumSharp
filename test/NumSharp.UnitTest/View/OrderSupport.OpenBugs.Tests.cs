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

        // ============================================================================
        // Section 11: np.empty_like — default order='K' in NumPy
        // NumPy: preserves source layout by default.
        // NumSharp: has no order parameter (see np.empty_like.cs).
        //
        // Expected matrix:
        //   | source    | order=C | order=F | order=A | order=K |
        //   |-----------|---------|---------|---------|---------|
        //   | C-contig  | C       | F       | C       | C       |
        //   | F-contig  | C       | F       | F       | F       |
        // ============================================================================

        [TestMethod]
        public void EmptyLike_CSource_DefaultIsCContig()
        {
            // NumPy: np.empty_like(c_src) (order='K' default) -> C=True (preserves C)
            var src = np.arange(12).reshape(3, 4);
            var r = np.empty_like(src);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.empty_like doesn't preserve F-contig from source (K default)
        public void EmptyLike_FSource_KDefault_PreservesFContig()
        {
            // NumPy: np.empty_like(f_src) (order='K' default) -> F=True (preserves F)
            var fSrc = np.arange(12).reshape(3, 4).T;  // F-contig
            var r = np.empty_like(fSrc);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy order='K' (default) should preserve F-contig from F-contig source");
        }

        // ============================================================================
        // Section 12: np.zeros_like — default order='K' in NumPy
        // ============================================================================

        [TestMethod]
        public void ZerosLike_CSource_DefaultIsCContig()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.zeros_like(src);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.zeros_like doesn't preserve F-contig from source (K default)
        public void ZerosLike_FSource_KDefault_PreservesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var r = np.zeros_like(fSrc);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy order='K' should preserve F-contig from F-contig source");
        }

        [TestMethod]
        public void ZerosLike_ValuesAllZero()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.zeros_like(src);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)r[i, j]).Should().Be(0);
        }

        // ============================================================================
        // Section 13: np.ones_like — default order='K' in NumPy
        // ============================================================================

        [TestMethod]
        public void OnesLike_CSource_DefaultIsCContig()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.ones_like(src);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.ones_like doesn't preserve F-contig from source (K default)
        public void OnesLike_FSource_KDefault_PreservesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var r = np.ones_like(fSrc);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy order='K' should preserve F-contig from F-contig source");
        }

        [TestMethod]
        public void OnesLike_ValuesAllOne()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.ones_like(src);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)r[i, j]).Should().Be(1);
        }

        // ============================================================================
        // Section 14: np.full_like — default order='K' in NumPy
        // ============================================================================

        [TestMethod]
        public void FullLike_CSource_DefaultIsCContig()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.full_like(src, 7);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.full_like doesn't preserve F-contig from source (K default)
        public void FullLike_FSource_KDefault_PreservesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var r = np.full_like(fSrc, 7);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy order='K' should preserve F-contig from F-contig source");
        }

        [TestMethod]
        public void FullLike_ValuesAllFillValue()
        {
            var src = np.arange(12).reshape(3, 4);
            var r = np.full_like(src, 42);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((int)r[i, j]).Should().Be(42);
        }

        // ============================================================================
        // Section 15: np.eye — NumPy accepts order='C' (default) or 'F'
        // ============================================================================

        [TestMethod]
        public void Eye_Default_IsCContig()
        {
            // NumPy: np.eye(3) -> C=True
            var r = np.eye(3, dtype: typeof(int));
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Eye_Values_MatchIdentity()
        {
            // Identity matrix diagonal is 1, off-diagonal 0
            var r = np.eye(3, dtype: typeof(int));
            ((int)r[0, 0]).Should().Be(1);
            ((int)r[1, 1]).Should().Be(1);
            ((int)r[2, 2]).Should().Be(1);
            ((int)r[0, 1]).Should().Be(0);
            ((int)r[1, 2]).Should().Be(0);
        }

        [TestMethod]
        [OpenBugs] // np.eye has no order parameter (see np.eye.cs:30)
        public void Eye_FOrder_IsFContig_ApiGap()
        {
            // NumPy: np.eye(3, order='F') -> F=True with same identity values
            // NumSharp has no overload — this test documents the gap.
            // Until an overload is added, this test cannot express the F-order case.
            // Compile-time workaround: construct manually
            var manualFEye = np.empty(new Shape(3L, 3L), order: 'F', dtype: typeof(int));
            manualFEye.Shape.IsFContiguous.Should().BeTrue();
            // But there's no np.eye(N, order='F') public API
            false.Should().BeTrue("np.eye needs an order parameter to match NumPy");
        }

        // ============================================================================
        // Section 16: np.asarray / np.asanyarray
        // NumPy: accept order parameter; NumSharp versions don't (no NDArray overload either)
        // ============================================================================

        [TestMethod]
        [OpenBugs] // np.asarray has no NDArray overload accepting order
        public void Asarray_FOrder_ProducesFContig_ApiGap()
        {
            // NumPy: np.asarray(c_src, order='F') -> F=True
            // NumSharp's asarray only accepts struct/T[] types, not NDArray.
            // When asarray(NDArray, order) is added, this should match NumPy.
            false.Should().BeTrue("np.asarray needs NDArray+order overload");
        }

        [TestMethod]
        [OpenBugs] // np.asanyarray has TODO for order support (see np.asanyarray.cs:14)
        public void Asanyarray_FOrder_ProducesFContig_ApiGap()
        {
            // NumPy: np.asanyarray(src, order='F') -> F=True
            // NumSharp signature: asanyarray(in object a, Type dtype) — no order
            false.Should().BeTrue("np.asanyarray needs order parameter");
        }

        // ============================================================================
        // Section 17: astype — NumPy default order='K'
        // NumPy: ndarray.astype(dtype, order='K') preserves layout by default
        // NumSharp: astype(Type, bool copy) — no order parameter
        // ============================================================================

        [TestMethod]
        public void Astype_CSource_DefaultIsCContig()
        {
            // NumPy: c_src.astype(np.int64) (K default) -> C=True (preserves)
            var src = np.arange(12).reshape(3, 4);
            var r = src.astype(typeof(long));
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // astype has no order parameter; always produces C-contig
        public void Astype_FSource_KDefault_PreservesFContig()
        {
            // NumPy: f_src.astype(np.int64) (K default) -> F=True (preserves)
            var fSrc = np.arange(12).reshape(3, 4).T;
            var r = fSrc.astype(typeof(long));
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy astype order='K' default preserves F-contig");
        }

        [TestMethod]
        public void Astype_ValuesPreserved()
        {
            // Math result: same values regardless of layout
            var src = np.arange(12).reshape(3, 4);
            var r = src.astype(typeof(long));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    ((long)r[i, j]).Should().Be(i * 4 + j);
        }

        // ============================================================================
        // Section 18: np.reshape — NumPy accepts order='C'/'F'/'A'
        // NumPy: np.reshape(arange(12), (3,4), order='F') produces F-contig with
        //        values [[0,3,6,9],[1,4,7,10],[2,5,8,11]] (column-major fill)
        // NumSharp: np.reshape is not a static function (only NDArray.reshape method)
        // ============================================================================

        [TestMethod]
        public void Reshape_Default_COrderFill()
        {
            // NumPy: np.arange(12).reshape(3, 4) -> [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
            var r = np.arange(12).reshape(3, 4);
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[0, 3]).Should().Be(3);
            ((int)r[2, 3]).Should().Be(11);
        }

        [TestMethod]
        [OpenBugs] // NDArray.reshape has no order parameter
        public void Reshape_FOrder_FillColumnMajor()
        {
            // NumPy: np.arange(12).reshape((3,4), order='F')
            //   values: [[0,3,6,9],[1,4,7,10],[2,5,8,11]]
            //   flags: C=False, F=True
            // NumSharp: no order overload exists.
            false.Should().BeTrue("NDArray.reshape needs order parameter for F-order fill");
        }

        // ============================================================================
        // Section 19: np.ravel — NumPy accepts order='C'/'F'/'A'/'K'
        // NumPy on C-contig arr = arange(6).reshape(2,3):
        //   ravel('C') = [0,1,2,3,4,5]
        //   ravel('F') = [0,3,1,4,2,5]
        //   ravel('A') = [0,1,2,3,4,5]  (C-contig source -> C)
        //   ravel('K') = [0,1,2,3,4,5]  (memory order for C-contig)
        // NumPy on F-contig arrT:
        //   ravel('C') = [0,3,1,4,2,5]  (logical C-order traversal)
        //   ravel('F') = [0,1,2,3,4,5]  (logical F-order = memory for F-contig)
        //   ravel('A') = [0,1,2,3,4,5]  (F-contig source -> F = memory)
        //   ravel('K') = [0,1,2,3,4,5]  (memory order)
        // NumSharp: np.ravel(NDArray) has no order parameter.
        // ============================================================================

        [TestMethod]
        public void NpRavel_CContig_Default_COrder()
        {
            // NumPy: np.ravel(arr) default 'C' -> [0..5]
            var arr = np.arange(6).reshape(2, 3);
            var r = np.ravel(arr);
            var expected = new int[] { 0, 1, 2, 3, 4, 5 };
            for (int i = 0; i < 6; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        [OpenBugs] // np.ravel has no order parameter
        public void NpRavel_CContig_FOrder_MatchesNumPy_ApiGap()
        {
            // NumPy: np.ravel(arr, order='F') = [0,3,1,4,2,5]
            // NumSharp: no overload — documents the gap.
            false.Should().BeTrue("np.ravel needs order parameter");
        }

        [TestMethod]
        [OpenBugs] // np.ravel has no order parameter
        public void NpRavel_FContig_FOrder_MatchesNumPy_ApiGap()
        {
            // NumPy: np.ravel(arrT, order='F') = [0,1,2,3,4,5] (memory order for F)
            false.Should().BeTrue("np.ravel needs order parameter");
        }

        // ============================================================================
        // Section 20: np.array with order (Array input overload)
        // NumPy: np.array(list, order='F') produces F-contig from Python list
        // NumSharp: np.array(Array, dtype, ndmin, copy, order) accepts order but ignores it
        // ============================================================================

        [TestMethod]
        public void NpArray_FromManaged_DefaultCContig()
        {
            // NumPy: np.array([[1,2],[3,4]]) -> C-contig
            var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            arr.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // np.array(Array, ..., order='F') is accepted but ignored
        public void NpArray_FromManaged_FOrder_ProducesFContig()
        {
            // NumPy: np.array([[1,2],[3,4]], order='F') -> F-contig
            var arr = np.array(new int[,] { { 1, 2 }, { 3, 4 } }, order: 'F');
            arr.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: order='F' should produce F-contig output from list input");
        }

        // ============================================================================
        // Section 21: asfortranarray / ascontiguousarray (NumPy: no order param)
        // These are order-specific shortcuts. NumPy lacks these in NumSharp.
        // ============================================================================

        [TestMethod]
        [OpenBugs] // np.asfortranarray doesn't exist in NumSharp
        public void AsFortranArray_ProducesFContig_ApiGap()
        {
            // NumPy: np.asfortranarray(arr) always returns F-contig
            // NumSharp has no such function.
            false.Should().BeTrue("np.asfortranarray is not implemented");
        }

        [TestMethod]
        [OpenBugs] // np.ascontiguousarray doesn't exist in NumSharp
        public void AsContiguousArray_ProducesCContig_ApiGap()
        {
            // NumPy: np.ascontiguousarray(arr) always returns C-contig
            false.Should().BeTrue("np.ascontiguousarray is not implemented");
        }
    }
}
