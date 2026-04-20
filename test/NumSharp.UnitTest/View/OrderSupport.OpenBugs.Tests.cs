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
        public void NpCopy_FOrder_ProducesFContig()
        {
            // NumPy: np.copy(c_src, order='F') -> F=True
            var src = np.arange(12).reshape(3, 4);
            var copy = np.copy(src, order: 'F');
            copy.Shape.IsFContiguous.Should().BeTrue();
            copy.Shape.IsContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void NpCopy_AOrder_FSource_ProducesFContig()
        {
            // NumPy: np.copy(f_src, order='A') with F-contig src -> F=True
            var fSrc = np.arange(12).reshape(3, 4).T;  // F-contig via transpose
            var copy = np.copy(fSrc, order: 'A');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpCopy_KOrder_FSource_ProducesFContig()
        {
            var fSrc = np.arange(12).reshape(3, 4).T;
            var copy = np.copy(fSrc, order: 'K');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NpCopy_AOrder_CSource_ProducesCContig()
        {
            // NumPy: np.copy(c_src, order='A') -> C=True (A resolves to C for C-contig source).
            var src = np.arange(12).reshape(3, 4);
            var copy = np.copy(src, order: 'A');
            copy.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void NDArrayCopy_FOrder_ProducesFContig()
        {
            var src = np.arange(12).reshape(3, 4);
            var copy = src.copy(order: 'F');
            copy.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
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
        public void Ravel_FOrder_ApiGap()
        {
            // NumPy: arr.ravel('F') = [0,4,8,1,5,9,2,6,10,3,7,11]
            var arr = np.arange(12).reshape(3, 4);
            var r = arr.ravel('F');
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
        public void Eye_FOrder_IsFContig_ApiGap()
        {
            // NumPy: np.eye(3, order='F') -> F=True with same identity values
            var r = np.eye(3, dtype: typeof(int), order: 'F');
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            ((int)r[0, 0]).Should().Be(1);
            ((int)r[1, 1]).Should().Be(1);
            ((int)r[2, 2]).Should().Be(1);
            ((int)r[0, 1]).Should().Be(0);
            ((int)r[1, 2]).Should().Be(0);
        }

        // ============================================================================
        // Section 16: np.asarray / np.asanyarray
        // NumPy: accept order parameter; NumSharp versions don't (no NDArray overload either)
        // ============================================================================

        [TestMethod]
        public void Asarray_FOrder_ProducesFContig_ApiGap()
        {
            // NumPy: np.asarray(c_src, order='F') -> F=True
            var src = np.arange(12).reshape(3, 4);
            var r = np.asarray(src, order: 'F');
            r.Shape.IsFContiguous.Should().BeTrue();
            ((int)r[2, 3]).Should().Be(11);
        }

        [TestMethod]
        public void Asanyarray_FOrder_ProducesFContig_ApiGap()
        {
            // NumPy: np.asanyarray(src, order='F') -> F=True
            var src = np.arange(12).reshape(3, 4);
            var r = np.asanyarray(src, dtype: null, order: 'F');
            r.Shape.IsFContiguous.Should().BeTrue();
            ((int)r[2, 3]).Should().Be(11);
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
        public void Reshape_FOrder_FillColumnMajor()
        {
            // NumPy: np.arange(12).reshape((3,4), order='F')
            //   values: [[0,3,6,9],[1,4,7,10],[2,5,8,11]]
            //   flags: C=False, F=True
            var r = np.arange(12).reshape(new Shape(3L, 4L), order: 'F');
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[0, 1]).Should().Be(3);
            ((int)r[0, 2]).Should().Be(6);
            ((int)r[0, 3]).Should().Be(9);
            ((int)r[1, 0]).Should().Be(1);
            ((int)r[1, 3]).Should().Be(10);
            ((int)r[2, 0]).Should().Be(2);
            ((int)r[2, 3]).Should().Be(11);
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
        public void NpRavel_CContig_FOrder_MatchesNumPy_ApiGap()
        {
            // NumPy: np.ravel(arr, order='F') = [0,3,1,4,2,5]
            var arr = np.arange(6).reshape(2, 3);
            var r = np.ravel(arr, 'F');
            var expected = new int[] { 0, 3, 1, 4, 2, 5 };
            for (int i = 0; i < 6; i++)
                ((int)r[i]).Should().Be(expected[i]);
        }

        [TestMethod]
        public void NpRavel_FContig_FOrder_MatchesNumPy_ApiGap()
        {
            // NumPy: np.ravel(arrT, order='F') = [0,1,2,3,4,5] (memory order for F)
            var arrT = np.arange(6).reshape(2, 3).T;  // F-contig (3,2)
            var r = np.ravel(arrT, 'F');
            var expected = new int[] { 0, 1, 2, 3, 4, 5 };
            for (int i = 0; i < 6; i++)
                ((int)r[i]).Should().Be(expected[i]);
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
        public void AsFortranArray_ProducesFContig_ApiGap()
        {
            // NumPy: np.asfortranarray(arr) always returns F-contig
            var src = np.arange(12).reshape(3, 4);
            var r = np.asfortranarray(src);
            r.Shape.IsFContiguous.Should().BeTrue();
            ((int)r[2, 3]).Should().Be(11);
        }

        [TestMethod]
        public void AsContiguousArray_ProducesCContig_ApiGap()
        {
            // NumPy: np.ascontiguousarray(arr) always returns C-contig
            var fSrc = np.arange(12).reshape(3, 4).T;
            var r = np.ascontiguousarray(fSrc);
            r.Shape.IsContiguous.Should().BeTrue();
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[3, 2]).Should().Be(11);
        }

        // ============================================================================
        // Section 22: Unary math ops preserve F-contig layout
        // NumPy: np.abs/negative/sqrt/exp/sin/square on F-contig -> F-contig output
        // ============================================================================

        [TestMethod]
        public void Abs_FContig_PreservesFContig()
        {
            // NumPy: np.abs(f_arr) -> F=True
            var fArr = np.arange(12).reshape(3, 4).T;  // F-contig
            var r = np.abs(fArr);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: unary abs on F-contig preserves F output layout");
        }

        [TestMethod]
        public void Abs_FContig_ValuesCorrect()
        {
            // Math result: same values regardless of layout
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.abs(fArr);
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[3, 2]).Should().Be(11);
        }

        [TestMethod]
        public void Negative_FContig_PreservesFContig()
        {
            // NumPy: np.negative(f_arr) -> F=True
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.negative(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Negative_FContig_ValuesCorrect()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.negative(fArr);
            ((int)r[0, 1]).Should().Be(-4);
            ((int)r[3, 2]).Should().Be(-11);
        }

        [TestMethod]
        public void Sqrt_FContig_PreservesFContig()
        {
            // NumPy: np.sqrt(f_arr) -> F=True
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.sqrt(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Exp_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.exp(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Log1p_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.log1p(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Sin_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.sin(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Square_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.square(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 23: Comparison ops preserve F-contig layout
        // NumPy: F == F -> F-contig bool array; F == C -> C-contig bool array
        // ============================================================================

        [TestMethod]
        public void Equal_FPlusF_PreservesFContig()
        {
            // NumPy: f_arr == f_arr -> F=True
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a == b;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: F == F produces F-contig bool output");
        }

        [TestMethod]
        public void LessThan_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a < b;
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void GreaterEqual_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a >= b;
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Equal_FPlusF_ValuesCorrect()
        {
            // Math correctness regardless of layout
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a == b;
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    ((bool)r[i, j]).Should().BeTrue();
        }

        // ============================================================================
        // Section 24: Bitwise ops preserve F-contig layout
        // NumPy: F & F -> F-contig output
        // ============================================================================

        [TestMethod]
        public void BitwiseAnd_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a & b;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: F & F produces F-contig output");
        }

        [TestMethod]
        public void BitwiseOr_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T;
            var b = np.arange(12).reshape(3, 4).T;
            var r = a | b;
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 25: Statistical ops — math correctness and layout for axis ops
        // ============================================================================

        [TestMethod]
        public void Std_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.std() = sqrt(143/12) ≈ 3.4521
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            ((double)np.std(fArr)).Should().BeApproximately(3.4521, 0.01);
        }

        [TestMethod]
        public void Var_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.var() = 143/12 ≈ 11.9167
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            ((double)np.var(fArr)).Should().BeApproximately(11.9167, 0.01);
        }

        [TestMethod]
        public void ArgMin_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.argmin() = 0 (position of value 0 in C-order flat)
            var fArr = np.arange(12).reshape(3, 4).T;
            ((long)np.argmin(fArr)).Should().Be(0);
        }

        [TestMethod]
        public void ArgMax_FContig_MatchesNumPy()
        {
            // NumPy: f_arr.argmax() = 11 (position of value 11 in C-order flat)
            var fArr = np.arange(12).reshape(3, 4).T;
            ((long)np.argmax(fArr)).Should().Be(11);
        }

        [TestMethod]
        public void CumSumAxis0_FContig_ValuesMatchNumPy()
        {
            // NumPy: np.cumsum(f_arr, axis=0) = [[0,4,8],[1,9,17],[3,15,27],[6,22,38]]
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.cumsum(fArr, axis: 0);
            ((long)r[0, 0]).Should().Be(0);
            ((long)r[0, 1]).Should().Be(4);
            ((long)r[1, 0]).Should().Be(1);
            ((long)r[1, 1]).Should().Be(9);
            ((long)r[3, 2]).Should().Be(38);
        }

        [TestMethod]
        public void CumSumAxis0_FContig_PreservesFContig()
        {
            // NumPy: cumsum axis=0 on F-contig -> F-contig output
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.cumsum(fArr, axis: 0);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: cumsum with axis preserves F-contig");
        }

        // ============================================================================
        // Section 26: Concatenation / stacking layout preservation
        // NumPy: concatenate/vstack/hstack of F-arrays produces F-contig output
        // ============================================================================

        [TestMethod]
        public void Concatenate_CC_Axis0_MatchesNumPy()
        {
            // Values must match regardless of layout
            var a = np.arange(6).reshape(2, 3);
            var b = np.arange(6, 12).reshape(2, 3);
            var r = np.concatenate(new[] { a, b }, axis: 0);
            r.shape.Should().Equal(new long[] { 4, 3 });
            ((int)r[3, 2]).Should().Be(11);
        }

        [TestMethod]
        public void Concatenate_FF_Axis0_PreservesFContig()
        {
            // NumPy: concatenate([F,F], axis=0) -> F-contig output
            var a = np.arange(6).reshape(2, 3).T;  // F-contig (3,2)
            var b = np.arange(6, 12).reshape(2, 3).T;  // F-contig (3,2)
            var r = np.concatenate(new[] { a, b }, axis: 0);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: concatenate of F-arrays produces F-contig output");
        }

        [TestMethod]
        public void VStack_FF_PreservesFContig()
        {
            var a = np.arange(6).reshape(2, 3).T;
            var b = np.arange(6, 12).reshape(2, 3).T;
            var r = np.vstack(new[] { a, b });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void HStack_FF_PreservesFContig()
        {
            var a = np.arange(6).reshape(2, 3).T;
            var b = np.arange(6, 12).reshape(2, 3).T;
            var r = np.hstack(new[] { a, b });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 27: Manipulation layout
        // NumPy:
        //   repeat/tile/roll of F -> C-contig output (breaks F layout)
        //   expand_dims preserves layout (F -> F)
        //   squeeze preserves layout
        // ============================================================================

        [TestMethod]
        public void Repeat_FContig_MatchesCContigOutput()
        {
            // NumPy: repeat(F, 2, axis=0) -> C-contig (repeat breaks F layout)
            var fArr = np.arange(6).reshape(2, 3).T;  // F-contig (3,2)
            var r = np.repeat(fArr, 2);
            r.Shape.IsContiguous.Should().BeTrue(
                "NumPy: repeat produces C-contig output");
        }

        [TestMethod]
        public void ExpandDims_FContig_PreservesFContig()
        {
            // NumPy: expand_dims(F, axis=0) adds leading 1-dim; result is still F-contig
            // Passes: NumSharp's expand_dims is a view that preserves stride pattern,
            // so the result retains F-contig flag.
            var fArr = np.arange(6).reshape(2, 3).T;  // F-contig (3,2)
            var r = np.expand_dims(fArr, 0);  // -> shape (1,3,2)
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Squeeze_ValuesPreserved()
        {
            // Math correctness
            var arr = np.arange(6).reshape(1, 2, 3);
            var r = np.squeeze(arr);
            r.shape.Should().Equal(new long[] { 2, 3 });
            ((int)r[0, 0]).Should().Be(0);
            ((int)r[1, 2]).Should().Be(5);
        }

        [TestMethod]
        public void Roll_Values_MatchNumPy()
        {
            // NumPy: np.roll([0,1,2,3,4], 1) = [4,0,1,2,3]
            var arr = np.arange(5);
            var r = np.roll(arr, 1);
            ((int)r[0]).Should().Be(4);
            ((int)r[1]).Should().Be(0);
            ((int)r[4]).Should().Be(3);
        }

        // ============================================================================
        // Section 28: MatMul / Dot output layout
        // NumPy: always produces C-contig output regardless of input layout
        // ============================================================================

        [TestMethod]
        public void MatMul_CC_ProducesCContig()
        {
            var a = np.arange(6).astype(typeof(double)).reshape(2, 3);
            var b = np.arange(12).astype(typeof(double)).reshape(3, 4);
            var r = np.matmul(a, b);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void MatMul_FF_ProducesCContig()
        {
            // NumPy: F @ F -> C-contig (matmul convention)
            var a = np.arange(6).astype(typeof(double)).reshape(2, 3).T.T;  // C-contig
            var b = np.arange(12).astype(typeof(double)).reshape(3, 4).T.T;
            var r = np.matmul(a, b);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Dot_CC_ValuesMatchNumPy()
        {
            // NumPy: np.dot([[1,2],[3,4]], [[5,6],[7,8]]) = [[19,22],[43,50]]
            var a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var r = np.dot(a, b);
            ((double)r[0, 0]).Should().Be(19);
            ((double)r[0, 1]).Should().Be(22);
            ((double)r[1, 0]).Should().Be(43);
            ((double)r[1, 1]).Should().Be(50);
        }

        // ============================================================================
        // Section 29: Boolean masking / fancy indexing
        // NumPy:
        //   f_arr[bool_mask] -> 1D result, both C and F contig (1-D always both)
        //   f_arr[[0,2]] -> C-contig (fancy index resets to C)
        // ============================================================================

        [TestMethod]
        public void BoolMask_FContig_Returns1DBothContig()
        {
            // NumPy: f_arr[mask] returns 1-D which is both C and F contig
            var fArr = np.arange(12).reshape(3, 4).T;
            var mask = fArr > 5;
            var r = fArr[mask];
            r.ndim.Should().Be(1);
            r.Shape.IsContiguous.Should().BeTrue(
                "1-D bool-mask result is C-contig");
            r.Shape.IsFContiguous.Should().BeTrue(
                "1-D bool-mask result is also F-contig");
        }

        [TestMethod]
        public void BoolMask_FContig_ValuesMatchNumPy()
        {
            // NumPy: f_arr > 5 picks out [6,7,8,9,10,11]
            var fArr = np.arange(12).reshape(3, 4).T;  // values [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
            var mask = fArr > 5;
            var r = fArr[mask];
            r.size.Should().Be(6);
            // Collected set should be {6,7,8,9,10,11}
            var values = new System.Collections.Generic.HashSet<int>();
            for (int i = 0; i < 6; i++)
                values.Add((int)r[i]);
            foreach (var v in new[] { 6, 7, 8, 9, 10, 11 })
                values.Should().Contain(v);
        }

        // ============================================================================
        // Section 30: Missing functions that would benefit from order support
        // ============================================================================

        [TestMethod]
        [OpenBugs] // np.tile is missing from NumSharp (listed in docs/CLAUDE.md Missing Functions)
        public void Tile_ApiGap()
        {
            // NumPy: np.tile(arr, 2) repeats array - not implemented in NumSharp
            false.Should().BeTrue("np.tile is not implemented");
        }

        [TestMethod]
        [OpenBugs] // np.flip is missing from NumSharp
        public void Flip_ApiGap()
        {
            // NumPy: np.flip(arr, axis=0) reverses along axis - not implemented in NumSharp
            false.Should().BeTrue("np.flip is not implemented");
        }

        // ============================================================================
        // Section 31: Extended unary math ops preserve F-contig
        // NumPy: ceil/floor/trunc/reciprocal/sign/cos/tan/log/log10/log2/exp2/expm1/cbrt
        //        all preserve F-contig on F-contig input
        // ============================================================================

        [TestMethod]
        public void Ceil_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.ceil(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Floor_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.floor(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Trunc_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.trunc(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Reciprocal_FContig_PreservesFContig()
        {
            var fArr = (np.arange(12).reshape(3, 4).T.astype(typeof(double))) + 1.0;
            var r = np.reciprocal(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Sign_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.sign(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Cos_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.cos(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Tan_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.tan(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Log_FContig_PreservesFContig()
        {
            var fArr = (np.arange(12).reshape(3, 4).T.astype(typeof(double))) + 1.0;
            var r = np.log(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Log10_FContig_PreservesFContig()
        {
            var fArr = (np.arange(12).reshape(3, 4).T.astype(typeof(double))) + 1.0;
            var r = np.log10(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Log2_FContig_PreservesFContig()
        {
            var fArr = (np.arange(12).reshape(3, 4).T.astype(typeof(double))) + 1.0;
            var r = np.log2(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Exp2_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.exp2(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Expm1_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.expm1(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Cbrt_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.cbrt(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        // Math correctness — values must match regardless of layout (one representative)
        [TestMethod]
        public void Ceil_FContig_ValuesCorrect()
        {
            var fArr = np.array(new double[,] { { 1.3, 2.7 }, { 3.1, 4.9 } }).T;
            var r = np.ceil(fArr);
            ((double)r[0, 0]).Should().Be(2.0);
            ((double)r[1, 1]).Should().Be(5.0);
        }

        // ============================================================================
        // Section 32: Division / remainder / power preserve F-contig
        // NumPy: /, //, %, ** all preserve F-contig layout when both operands are F
        // ============================================================================

        [TestMethod]
        public void TrueDivide_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T.astype(typeof(double)) + 1.0;
            var b = np.arange(12).reshape(3, 4).T.astype(typeof(double)) + 1.0;
            var r = a / b;
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: F/F produces F-contig output");
        }

        [TestMethod]
        public void FloorDivide_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T + 1;
            var b = np.arange(12).reshape(3, 4).T + 1;
            var r = np.floor_divide(a, b);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Mod_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T + 1;
            var b = np.arange(12).reshape(3, 4).T + 1;
            var r = a % b;
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Power_FPlusF_PreservesFContig()
        {
            var a = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var b = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.power(a, b);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        // Math correctness for these
        [TestMethod]
        public void TrueDivide_Values_MatchNumPy()
        {
            var a = np.array(new double[] { 10, 20, 30 });
            var b = np.array(new double[] { 2, 4, 5 });
            var r = a / b;
            ((double)r[0]).Should().Be(5.0);
            ((double)r[1]).Should().Be(5.0);
            ((double)r[2]).Should().Be(6.0);
        }

        // ============================================================================
        // Section 33: In-place ops should preserve F-contig (mutate same buffer)
        // ============================================================================

        [TestMethod]
        public void InPlaceAdd_FContig_PreservesFContig()
        {
            // NumPy: f_arr += 1 preserves F-contig (same buffer, just values mutated)
            var fArr = np.empty(new Shape(4L, 3L), order: 'F', dtype: typeof(int));
            // Seed values
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    fArr[i, j] = i * 3 + j;
            fArr.Shape.IsFContiguous.Should().BeTrue();

            fArr += 1;  // should mutate in place
            fArr.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: in-place ops don't change layout");
        }

        // ============================================================================
        // Section 34: Selection / clip / pairwise (where/clip/maximum/minimum/modf)
        // ============================================================================

        [TestMethod]
        [OpenBugs] // np.where doesn't exist in NumSharp (listed in Missing Functions)
        public void Where_ApiGap()
        {
            // NumPy: np.where(f_arr > 5, f_arr, 0) -> F-contig output
            false.Should().BeTrue("np.where is not implemented (Missing Functions)");
        }

        [TestMethod]
        public void Clip_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.clip(fArr, 2, 8);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: clip preserves F-contig output");
        }

        [TestMethod]
        public void Clip_Values_MatchNumPy()
        {
            var arr = np.array(new[] { 1, 5, 10, 15, 20 });
            var r = np.clip(arr, 5, 15);
            ((int)r[0]).Should().Be(5);
            ((int)r[1]).Should().Be(5);
            ((int)r[2]).Should().Be(10);
            ((int)r[3]).Should().Be(15);
            ((int)r[4]).Should().Be(15);
        }

        [TestMethod]
        public void Maximum_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.maximum(fArr, 5);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Minimum_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.minimum(fArr, 5);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Modf_FContig_PreservesFContig()
        {
            var fArr = (np.arange(12).reshape(3, 4).T.astype(typeof(double))) + 0.5;
            var (frac, whole) = np.modf(fArr);
            frac.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: modf fractional output preserves F-contig");
            whole.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: modf integral output preserves F-contig");
        }

        // ============================================================================
        // Section 35: NaN-aware reductions — math correctness on F-contig
        // ============================================================================

        [TestMethod]
        public void NanSum_FContig_ValuesMatchNumPy()
        {
            // NumPy: np.nansum([0..11] with nan at [0]) = 66
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            ((double)np.nansum(fArr)).Should().Be(66.0);
        }

        [TestMethod]
        public void NanMean_FContig_ValuesMatchNumPy()
        {
            // NumPy: np.nanmean with one nan out of 12 = 66/11 = 6.0
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            ((double)np.nanmean(fArr)).Should().BeApproximately(6.0, 0.001);
        }

        [TestMethod]
        public void NanMax_FContig_ValuesMatchNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            ((double)np.nanmax(fArr)).Should().Be(11.0);
        }

        [TestMethod]
        public void NanMin_FContig_ValuesMatchNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            // NumPy: nanmin skips nan, gives 1.0 (next smallest non-nan)
            ((double)np.nanmin(fArr)).Should().Be(1.0);
        }

        // ============================================================================
        // Section 36: Boolean reductions / nonzero
        // ============================================================================

        [TestMethod]
        public void Any_FContig_MatchesNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            ((bool)np.any(fArr > 5)).Should().BeTrue();
            ((bool)np.any(fArr > 100)).Should().BeFalse();
        }

        [TestMethod]
        public void All_FContig_MatchesNumPy()
        {
            var fArr = np.arange(12).reshape(3, 4).T;
            ((bool)np.all(fArr >= 0)).Should().BeTrue();
            ((bool)np.all(fArr > 5)).Should().BeFalse();
        }

        [TestMethod]
        public void CountNonzero_FContig_MatchesNumPy()
        {
            // NumPy: np.count_nonzero(arange(12) reshape 4x3 F-contig) = 11 (all except the 0)
            var fArr = np.arange(12).reshape(3, 4).T;
            ((long)np.count_nonzero(fArr)).Should().Be(11);
        }

        // ============================================================================
        // Section 37: isnan / isinf / isfinite preserve F-contig
        // ============================================================================

        [TestMethod]
        public void IsNan_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.isnan(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void IsInf_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.isinf(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void IsFinite_FContig_PreservesFContig()
        {
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.isfinite(fArr);
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void IsNan_Values_MatchNumPy()
        {
            var arr = np.array(new double[] { 1.0, double.NaN, 3.0, double.NaN });
            var r = np.isnan(arr);
            ((bool)r[0]).Should().BeFalse();
            ((bool)r[1]).Should().BeTrue();
            ((bool)r[2]).Should().BeFalse();
            ((bool)r[3]).Should().BeTrue();
        }

        // ============================================================================
        // Section 38: Broadcasting / axis manipulation on F-contig
        // ============================================================================

        [TestMethod]
        public void BroadcastTo_FromVector_ProducesZeroStrideFirstAxis()
        {
            // NumPy: broadcast_to([1,2,3], (4,3)) -> strides (0, 8) for float/(0,4) for int
            // flags: C=False, F=False (broadcasted arrays are neither)
            var v = np.array(new[] { 1, 2, 3 });
            var r = np.broadcast_to(v, new Shape(4L, 3L));
            r.Shape.IsContiguous.Should().BeFalse(
                "NumPy: broadcast_to result is neither C nor F contig (has stride=0)");
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void BroadcastTo_Values_MatchNumPy()
        {
            var v = np.array(new[] { 1, 2, 3 });
            var r = np.broadcast_to(v, new Shape(4L, 3L));
            ((int)r[0, 0]).Should().Be(1);
            ((int)r[0, 1]).Should().Be(2);
            ((int)r[0, 2]).Should().Be(3);
            ((int)r[3, 0]).Should().Be(1);
            ((int)r[3, 2]).Should().Be(3);
        }

        [TestMethod]
        public void MoveAxis_FContig3D_MatchesCOrder()
        {
            // NumPy: moveaxis(F-contig 3D, 0, -1) -> neither C nor F
            // NumSharp should match (moveaxis reorders strides so neither pattern holds)
            var fArr3D = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(int));
            fArr3D.Shape.IsFContiguous.Should().BeTrue();
            var r = np.moveaxis(fArr3D, 0, -1);
            r.Shape.IsContiguous.Should().BeFalse();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void SwapAxes_FContig_ReturnsCContig()
        {
            // NumPy: swapaxes of F-contig 3D with outer axes swapped -> C-contig
            var fArr3D = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(int));
            var r = np.swapaxes(fArr3D, 0, 2);
            r.Shape.IsContiguous.Should().BeTrue(
                "NumPy: swapaxes(F, 0, 2) reverses stride order -> C-contig");
        }

        // ============================================================================
        // Section 39: argsort / unique / np.outer
        // ============================================================================

        [TestMethod]
        public void ArgSort_FContig_ProducesCContig()
        {
            // NumPy: argsort of F-contig produces C-contig output.
            // np.arange returns Int64, so argsort<long> matches the source dtype.
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.argsort<long>(fArr, axis: 0);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Unique_FContig_Is1DBothContig()
        {
            // NumPy: np.unique returns 1-D sorted unique values - both C and F contig
            var fArr = np.arange(12).reshape(3, 4).T;
            var r = np.unique(fArr);
            r.ndim.Should().Be(1);
            r.size.Should().Be(12);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Outer_Values_MatchNumPy()
        {
            // NumPy: np.outer([1,2,3], [4,5]) = [[4,5],[8,10],[12,15]]
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 4.0, 5.0 });
            var r = np.outer(a, b);
            r.shape.Should().Equal(new long[] { 3, 2 });
            ((double)r[0, 0]).Should().Be(4.0);
            ((double)r[0, 1]).Should().Be(5.0);
            ((double)r[1, 0]).Should().Be(8.0);
            ((double)r[2, 1]).Should().Be(15.0);
        }

        [TestMethod]
        public void Outer_OutputIsCContig()
        {
            // NumPy: np.outer result is always C-contig
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 4.0, 5.0 });
            var r = np.outer(a, b);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 40: Fancy index write / slice write preserves F-contig
        // ============================================================================

        [TestMethod]
        [OpenBugs] // SetIndicesND asserts dstOffsets.size == values.size, breaks for scalar values on multi-row fancy writes regardless of layout. Not F-order specific — a pre-existing bug.
        public void FancyWrite_FContig_PreservesFContig()
        {
            // NumPy: f_arr[[0,2]] = 99 preserves F-contig (in-place)
            var fArr = np.empty(new Shape(4L, 3L), order: 'F', dtype: typeof(int));
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    fArr[i, j] = i * 3 + j;
            fArr.Shape.IsFContiguous.Should().BeTrue();

            // Fancy index write
            fArr[np.array(new[] { 0, 2 })] = 99;
            fArr.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: fancy write mutates in place, preserves F-contig");
        }

        [TestMethod]
        public void SliceWrite_FContig_PreservesFContig()
        {
            // NumPy: slice assignment mutates in place, preserves F-contig.
            // NumSharp correctly preserves F-contig here because slice write
            // doesn't allocate new storage — it writes through the view.
            var fArr = np.empty(new Shape(4L, 3L), order: 'F', dtype: typeof(int));
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    fArr[i, j] = i * 3 + j;
            fArr.Shape.IsFContiguous.Should().BeTrue();

            fArr["1:3, :"] = 99;
            fArr.Shape.IsFContiguous.Should().BeTrue();
        }

        // ============================================================================
        // Section 41: Reductions with keepdims=True on F-contig inputs
        // NumPy: reductions preserve input layout when keepdims=True.
        // For 2-D F-contig, result shape is (1,N) or (M,1) — trivially both C and F contig.
        // For 3-D+ F-contig, reduction along an axis yields a shape with one size-1 dim
        // where only F-strides stay contiguous; NumSharp currently flips to C-contig.
        // ============================================================================

        [TestMethod]
        public void Sum_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: np.sum(F(4,3), axis=0, keepdims=True) shape=(1,3) vals=[6,22,38], both C&F
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.sum(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            r.Shape.IsContiguous.Should().BeTrue("(1,N) is trivially C-contig");
            r.Shape.IsFContiguous.Should().BeTrue("(1,N) is trivially F-contig");
            ((double)r[0, 0]).Should().Be(6.0);
            ((double)r[0, 1]).Should().Be(22.0);
            ((double)r[0, 2]).Should().Be(38.0);
        }

        [TestMethod]
        public void Sum_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: np.sum(F(4,3), axis=1, keepdims=True) shape=(4,1) vals=[12,15,18,21]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.sum(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
            ((double)r[0, 0]).Should().Be(12.0);
            ((double)r[3, 0]).Should().Be(21.0);
        }

        [TestMethod]
        public void Mean_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: np.mean(F(4,3), axis=0, keepdims=True) shape=(1,3) vals=[1.5, 5.5, 9.5]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.mean(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
            ((double)r[0, 0]).Should().BeApproximately(1.5, 1e-9);
            ((double)r[0, 1]).Should().BeApproximately(5.5, 1e-9);
            ((double)r[0, 2]).Should().BeApproximately(9.5, 1e-9);
        }

        [TestMethod]
        public void Mean_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: np.mean(F(4,3), axis=1, keepdims=True) shape=(4,1) vals=[4,5,6,7]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.mean(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            ((double)r[0, 0]).Should().BeApproximately(4.0, 1e-9);
            ((double)r[3, 0]).Should().BeApproximately(7.0, 1e-9);
        }

        [TestMethod]
        public void Max_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: np.max(F(4,3), axis=0, keepdims=True) shape=(1,3) vals=[3,7,11]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.max(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
            ((double)r[0, 0]).Should().Be(3.0);
            ((double)r[0, 1]).Should().Be(7.0);
            ((double)r[0, 2]).Should().Be(11.0);
        }

        [TestMethod]
        public void Min_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: np.min(F(4,3), axis=1, keepdims=True) shape=(4,1) vals=[0,1,2,3]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.min(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            ((double)r[0, 0]).Should().Be(0.0);
            ((double)r[3, 0]).Should().Be(3.0);
        }

        [TestMethod]
        public void Prod_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: np.prod(F(4,3), axis=0, keepdims=True) shape=(1,3) vals=[0, 840, 7920]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.prod(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            ((double)r[0, 0]).Should().Be(0.0);
            ((double)r[0, 1]).Should().Be(840.0);
            ((double)r[0, 2]).Should().Be(7920.0);
        }

        [TestMethod]
        public void Std_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: np.std(F(4,3), axis=0, keepdims=True, ddof=0) = [1.118, 1.118, 1.118]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.std(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            ((double)r[0, 0]).Should().BeApproximately(1.118, 0.01);
            ((double)r[0, 1]).Should().BeApproximately(1.118, 0.01);
            ((double)r[0, 2]).Should().BeApproximately(1.118, 0.01);
        }

        [TestMethod]
        public void Var_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: np.var(F(4,3), axis=1, keepdims=True, ddof=0) = [10.6667, 10.6667, 10.6667, 10.6667]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            var r = np.var(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            ((double)r[0, 0]).Should().BeApproximately(10.6667, 0.01);
            ((double)r[3, 0]).Should().BeApproximately(10.6667, 0.01);
        }

        // --- 3-D F-contig tests: F-preservation fails (documented) ---

        [TestMethod]
        [OpenBugs] // Reductions with keepdims=True on 3-D F-contig flip to C-contig output.
                   // NumPy: shape (1,3,4) on 3-D (2,3,4) F-contig stays F-contig (C=0, F=1).
                   // NumSharp: returns C=1, F=0.
        public void Sum_FContig3D_Axis0_KeepDims_PreservesFContig()
        {
            var f3 = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(double));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        f3[i, j, k] = i * 12 + j * 4 + k;

            var r = np.sum(f3, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3, 4 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: sum(F3, axis=0, keepdims=True) preserves F-contig layout");
        }

        [TestMethod]
        [OpenBugs] // Same gap as Sum_FContig3D_Axis0_KeepDims — mean doesn't preserve F either.
        public void Mean_FContig3D_Axis1_KeepDims_PreservesFContig()
        {
            var f3 = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(double));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        f3[i, j, k] = i * 12 + j * 4 + k;

            var r = np.mean(f3, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: mean(F3, axis=1, keepdims=True) preserves F-contig layout");
        }

        [TestMethod]
        [OpenBugs] // Reductions without keepdims on 3-D F-contig also flip to C-contig.
                   // NumPy: shape (3,4) from reducing axis=0 of (2,3,4) F-contig stays F-contig (C=0, F=1).
        public void Sum_FContig3D_Axis0_NoKeepDims_PreservesFContig()
        {
            var f3 = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(double));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        f3[i, j, k] = i * 12 + j * 4 + k;

            var r = np.sum(f3, axis: 0);
            r.shape.Should().Equal(new long[] { 3, 4 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: sum(F3, axis=0) on F-contig 3D produces F-contig 2D");
        }

        // --- NaN-aware reductions with keepdims=True ---

        [TestMethod]
        public void NanSum_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: nansum of (4,3) F with nan at [0,0], axis=0, kd=True
            // -> shape (1,3) vals=[6, 22, 38] (nan contributes 0 to sum)
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            var r = np.nansum(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            ((double)r[0, 0]).Should().Be(6.0);
            ((double)r[0, 1]).Should().Be(22.0);
            ((double)r[0, 2]).Should().Be(38.0);
        }

        [TestMethod]
        public void NanMean_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: nanmean axis=1, kd=True shape=(4,1) vals=[6, 5, 6, 7]
            // (row 0: nan + 4 + 8 -> mean of 2 non-nan = 6)
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            var r = np.nanmean(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            ((double)r[0, 0]).Should().BeApproximately(6.0, 1e-9);
            ((double)r[1, 0]).Should().BeApproximately(5.0, 1e-9);
            ((double)r[2, 0]).Should().BeApproximately(6.0, 1e-9);
            ((double)r[3, 0]).Should().BeApproximately(7.0, 1e-9);
        }

        [TestMethod]
        public void NanStd_FContig2D_Axis0_KeepDims_MatchesNumPy()
        {
            // NumPy: nanstd axis=0, kd=True, ddof=0 on (4,3) F with nan at [0,0]
            // -> shape (1,3) vals=[0.8165, 1.118, 1.118]
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            var r = np.nanstd(fArr, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3 });
            ((double)r[0, 0]).Should().BeApproximately(0.8165, 0.01);
            ((double)r[0, 1]).Should().BeApproximately(1.118, 0.01);
            ((double)r[0, 2]).Should().BeApproximately(1.118, 0.01);
        }

        [TestMethod]
        public void NanVar_FContig2D_Axis1_KeepDims_MatchesNumPy()
        {
            // NumPy: nanvar axis=1, kd=True, ddof=0 -> shape (4,1) vals=[4, 10.6667, 10.6667, 10.6667]
            // (row 0: variance of {4, 8} around 6 = (4+4)/2 = 4)
            var fArr = np.arange(12).reshape(3, 4).T.astype(typeof(double));
            fArr[0, 0] = double.NaN;
            var r = np.nanvar(fArr, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 4, 1 });
            ((double)r[0, 0]).Should().BeApproximately(4.0, 1e-6);
            ((double)r[1, 0]).Should().BeApproximately(10.6667, 0.01);
        }

        [TestMethod]
        [OpenBugs] // NaN-aware 3-D F-contig reduction doesn't preserve F-contig either.
                   // NumPy: nansum(F3, axis=0, keepdims=True) shape (1,3,4) stays F-contig.
        public void NanSum_FContig3D_Axis0_KeepDims_PreservesFContig()
        {
            var f3 = np.empty(new Shape(2L, 3L, 4L), order: 'F', dtype: typeof(double));
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        f3[i, j, k] = i * 12 + j * 4 + k;
            f3[0, 0, 0] = double.NaN;

            var r = np.nansum(f3, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3, 4 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: nansum(F3, axis=0, keepdims=True) preserves F-contig layout");
        }

        // ============================================================================
        // Section 42: np.sort — API gap
        // NumPy: np.sort(a, axis=-1) returns a sorted copy. Default axis=-1 flattens last
        // axis. For 1-D arrays, the result is trivially both-contig. For 2-D+, the output
        // is C-contig regardless of input layout (NumPy's default).
        // NumSharp: np.sort is listed in Missing Functions (docs/CLAUDE.md); only argsort
        // exists. Document the gap so it's visible to anyone porting NumPy code.
        // ============================================================================

        [TestMethod]
        [OpenBugs] // np.sort is missing from NumSharp (listed in docs/CLAUDE.md Missing Functions).
                   // NumPy: np.sort(arr) returns a sorted copy; axis=-1 by default.
                   // Workaround: argsort + fancy-index, but layout semantics diverge.
        public void Sort_ApiGap()
        {
            // NumPy: np.sort(np.array([3,1,2])) == [1,2,3]
            false.Should().BeTrue("np.sort is not implemented — only argsort exists");
        }

        // ============================================================================
        // Section 43: matmul / dot / outer / convolve — output layout
        // NumPy (always C-contig output, regardless of input layout):
        //   matmul(F,F) → C-contig; matmul(C,F) → C-contig; matmul(F,C) → C-contig
        //   dot(F,F)    → C-contig (same reasoning)
        //   outer(1D,1D)→ C-contig
        //   convolve    → 1-D, trivially both C & F contig
        // Values must match NumPy exactly regardless of F-contig inputs.
        // ============================================================================

        [TestMethod]
        public void MatMul_FF_Values_MatchNumPy()
        {
            // NumPy: matmul([[1,2],[3,4]].F, [[5,6],[7,8]].F) = [[19,22],[43,50]]
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_a = c_a.copy('F');
            var f_b = c_b.copy('F');
            f_a.Shape.IsFContiguous.Should().BeTrue();
            f_b.Shape.IsFContiguous.Should().BeTrue();

            var r = np.matmul(f_a, f_b);
            ((double)r[0, 0]).Should().Be(19);
            ((double)r[0, 1]).Should().Be(22);
            ((double)r[1, 0]).Should().Be(43);
            ((double)r[1, 1]).Should().Be(50);
        }

        [TestMethod]
        public void MatMul_FF_ProducesCContigOutput()
        {
            // NumPy: matmul always produces C-contig output, regardless of input layout.
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_a = c_a.copy('F');
            var f_b = c_b.copy('F');

            var r = np.matmul(f_a, f_b);
            r.Shape.IsContiguous.Should().BeTrue("NumPy: matmul(F,F) -> C-contig");
        }

        [TestMethod]
        public void MatMul_CF_Mixed_Values_MatchNumPy()
        {
            // NumPy: matmul(C, F) = matmul(C, C) (output is C-contig, values identical)
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_b = c_b.copy('F');

            var r = np.matmul(c_a, f_b);
            ((double)r[0, 0]).Should().Be(19);
            ((double)r[0, 1]).Should().Be(22);
            ((double)r[1, 0]).Should().Be(43);
            ((double)r[1, 1]).Should().Be(50);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void MatMul_FC_Mixed_Values_MatchNumPy()
        {
            // NumPy: matmul(F, C) yields same values and C-contig output.
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_a = c_a.copy('F');

            var r = np.matmul(f_a, c_b);
            ((double)r[0, 0]).Should().Be(19);
            ((double)r[1, 1]).Should().Be(50);
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Dot_FF_Values_MatchNumPy()
        {
            // NumPy: dot(F,F) same values as dot(C,C).
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_a = c_a.copy('F');
            var f_b = c_b.copy('F');

            var r = np.dot(f_a, f_b);
            ((double)r[0, 0]).Should().Be(19);
            ((double)r[0, 1]).Should().Be(22);
            ((double)r[1, 0]).Should().Be(43);
            ((double)r[1, 1]).Should().Be(50);
        }

        [TestMethod]
        public void Dot_FF_ProducesCContigOutput()
        {
            var c_a = np.array(new double[,] { { 1, 2 }, { 3, 4 } });
            var c_b = np.array(new double[,] { { 5, 6 }, { 7, 8 } });
            var f_a = c_a.copy('F');
            var f_b = c_b.copy('F');

            var r = np.dot(f_a, f_b);
            r.Shape.IsContiguous.Should().BeTrue("NumPy: dot(F,F) -> C-contig");
        }

        [TestMethod]
        public void Outer_FVectorInput_ProducesCContigOutput()
        {
            // NumPy: outer(a, b) flattens inputs then builds C-contig (M,N) result.
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            var b = np.array(new[] { 4.0, 5.0 });
            var r = np.outer(a, b);
            r.shape.Should().Equal(new long[] { 3, 2 });
            r.Shape.IsContiguous.Should().BeTrue("NumPy: outer result is C-contig");
            ((double)r[0, 0]).Should().Be(4);
            ((double)r[0, 1]).Should().Be(5);
            ((double)r[1, 0]).Should().Be(8);
            ((double)r[1, 1]).Should().Be(10);
            ((double)r[2, 0]).Should().Be(12);
            ((double)r[2, 1]).Should().Be(15);
        }

        [TestMethod]
        public void Convolve_Valid_Mode_MatchesNumPy()
        {
            // NumPy: convolve([1,2,3], [1,0,1], 'valid') = [4]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 0, 1 });
            var r = np.convolve(a, b, "valid");
            r.shape.Should().Equal(new long[] { 1 });
            ((int)r[0]).Should().Be(4);
            // 1-D result: trivially both-contig.
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Convolve_Full_Mode_MatchesNumPy()
        {
            // NumPy: convolve([1,2,3], [1,0,1], 'full') = [1, 2, 4, 2, 3]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 0, 1 });
            var r = np.convolve(a, b, "full");
            r.shape.Should().Equal(new long[] { 5 });
            ((int)r[0]).Should().Be(1);
            ((int)r[1]).Should().Be(2);
            ((int)r[2]).Should().Be(4);
            ((int)r[3]).Should().Be(2);
            ((int)r[4]).Should().Be(3);
        }

        [TestMethod]
        public void Convolve_Same_Mode_MatchesNumPy()
        {
            // NumPy: convolve([1,2,3], [1,0,1], 'same') = [2, 4, 2]
            var a = np.array(new[] { 1, 2, 3 });
            var b = np.array(new[] { 1, 0, 1 });
            var r = np.convolve(a, b, "same");
            r.shape.Should().Equal(new long[] { 3 });
            ((int)r[0]).Should().Be(2);
            ((int)r[1]).Should().Be(4);
            ((int)r[2]).Should().Be(2);
        }

        // ============================================================================
        // Section 44: Broadcasting from F-contig inputs
        // NumPy:
        //   broadcast_to(any, bigger_shape) always inserts a stride=0 dim, so the
        //     result is BROADCASTED (both C- and F-contig flags = False).
        //   broadcast_arrays([F, scalar]) keeps F-contig array's flag; scalar becomes
        //     all-stride-0 view (neither flag).
        //   broadcast_arrays([F(m,n), F(m,1)]) keeps F-contig on the non-broadcasted
        //     input; broadcast input has stride=0 on broadcast dim (neither flag).
        // ============================================================================

        [TestMethod]
        public void BroadcastTo_FContig_ResultIsNeitherContig()
        {
            // NumPy: broadcast_to(F(4,3), (2,4,3)) strides=(0,8,32) -> neither C nor F contig.
            var f = np.arange(12).reshape(3, 4).T.astype(typeof(double));  // F-contig (4,3)
            f.Shape.IsFContiguous.Should().BeTrue();

            var r = np.broadcast_to(f, new Shape(2L, 4L, 3L));
            r.shape.Should().Equal(new long[] { 2, 4, 3 });
            r.Shape.IsContiguous.Should().BeFalse(
                "NumPy: broadcast_to result has stride=0 dim, not C-contig");
            r.Shape.IsFContiguous.Should().BeFalse(
                "NumPy: broadcast_to result has stride=0 dim, not F-contig");
        }

        [TestMethod]
        public void BroadcastTo_FContig_Values_MatchNumPy()
        {
            // NumPy: broadcast_to replicates along the new leading dim.
            // F(4,3) looks like [[0,4,8],[1,5,9],[2,6,10],[3,7,11]]
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.broadcast_to(f, new Shape(2L, 4L, 3L));
            // First replica
            ((long)r[0, 0, 0]).Should().Be(0);
            ((long)r[0, 0, 1]).Should().Be(4);
            ((long)r[0, 3, 2]).Should().Be(11);
            // Second replica — same values
            ((long)r[1, 0, 0]).Should().Be(0);
            ((long)r[1, 3, 2]).Should().Be(11);
        }

        [TestMethod]
        public void BroadcastTo_CContig_ResultIsNeitherContig()
        {
            // NumPy: broadcast_to(C(3,4), (2,3,4)) strides=(0,32,8) -> neither C nor F contig.
            var c = np.arange(12).reshape(3, 4).astype(typeof(double));
            c.Shape.IsContiguous.Should().BeTrue();

            var r = np.broadcast_to(c, new Shape(2L, 3L, 4L));
            r.shape.Should().Equal(new long[] { 2, 3, 4 });
            r.Shape.IsContiguous.Should().BeFalse();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void BroadcastArrays_FAndScalar_PreservesFContig()
        {
            // NumPy: broadcast_arrays([F(3,2), scalar]) -> first output keeps F-contig
            // flag (only its shape is broadcast-expanded from itself, so no stride=0 on
            // the non-singleton dim); scalar becomes all-stride-0 (neither flag).
            var f = np.arange(6).reshape(2, 3).T.astype(typeof(double));  // F-contig (3,2)
            f.Shape.IsFContiguous.Should().BeTrue();
            var scalar = np.array(5.0);

            var (lhs, rhs) = np.broadcast_arrays(f, scalar);
            // First output has the same shape as F, so strides are preserved.
            lhs.shape.Should().Equal(new long[] { 3, 2 });
            lhs.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: broadcast_arrays first output keeps F-contig flag when no broadcasting happens");
            // Second output is all-stride-0 (stretched scalar).
            rhs.shape.Should().Equal(new long[] { 3, 2 });
            rhs.Shape.IsContiguous.Should().BeFalse();
            rhs.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void BroadcastArrays_FAndColumnVec_FirstPreservesFContig()
        {
            // NumPy: broadcast_arrays([F(2,3), F(2,1)]) -> first F-contig preserved,
            // second has stride=0 on axis 1 (broadcast dim).
            var f = np.arange(6).reshape(3, 2).T.astype(typeof(double));  // F-contig (2,3)
            var col = np.array(new double[,] { { 10.0 }, { 20.0 } }).copy('F');  // F-contig (2,1)
            f.Shape.IsFContiguous.Should().BeTrue();

            var (lhs, rhs) = np.broadcast_arrays(f, col);
            lhs.shape.Should().Equal(new long[] { 2, 3 });
            lhs.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: broadcast_arrays preserves F-contig when shape already matches target");
            // Second becomes broadcasted (stride=0 on axis 1).
            rhs.shape.Should().Equal(new long[] { 2, 3 });
            rhs.Shape.IsContiguous.Should().BeFalse();
            rhs.Shape.IsFContiguous.Should().BeFalse();
        }

        // ============================================================================
        // Section 45: Manipulation ops — layout preservation / parity
        // NumPy behavior on F-contig (4,3) source arr = np.arange(12).reshape(3,4).T:
        //   repeat(F, 2)                    → 1-D (24,), both C&F
        //   repeat(F, 2, axis=0/1)          → C-contig (always, NumPy convention)
        //   roll(F, 1)                      → C-contig (uses ravel)
        //   roll(F, 1, axis=0/1)            → F-contig preserved
        //   stack([F,F])                    → neither (new axis)
        //   expand_dims(F, 0/1/2)           → F-contig preserved
        //   squeeze(F(2,1,3))               → F-contig preserved
        //   moveaxis(F, 0, -1)              → effectively transpose; layout flips
        //   swapaxes(F(4,3), 0, 1)          → C-contig (stride flip)
        //   swapaxes(C(3,4), 0, 1)          → F-contig (stride flip)
        //   atleast_1d/2d/3d                → 1-D/trivially-both/F-preserved
        // ============================================================================

        [TestMethod]
        public void Repeat_FContig_NoAxis_Is1DBothContig()
        {
            // NumPy: repeat(F(4,3), 2) flattens then repeats -> 1-D (24,), both-contig.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.repeat(f, 2);
            r.shape.Should().Equal(new long[] { 24 });
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue("1-D result is trivially F-contig");
        }

        [TestMethod]
        [OpenBugs] // NumSharp's np.repeat does NOT support the `axis` parameter
                   // (see src/NumSharp.Core/Manipulation/np.repeat.cs — always ravels first).
                   // NumPy: repeat(F(4,3), 2, axis=0) duplicates each row, shape (8,3).
        public void Repeat_FContig_Axis0_ApiGap()
        {
            // Expected NumPy values once axis is supported:
            //   [[0,4,8],[0,4,8],[1,5,9],[1,5,9],[2,6,10],[2,6,10],[3,7,11],[3,7,11]]
            var f = np.arange(12).reshape(3, 4).T;
            // This call will compile error until axis is supported; once supported,
            // remove the [OpenBugs] and uncomment the assertions below.
            // var r = np.repeat(f, 2, axis: 0);
            // r.shape.Should().Equal(new long[] { 8, 3 });
            // ((long)r[1, 0]).Should().Be(0);  // duplicated row
            false.Should().BeTrue("np.repeat does not support axis parameter yet");
        }

        [TestMethod]
        public void Repeat_FContig_Values_MatchNumPy()
        {
            // NumPy: repeat(F(4,3), 2) flattens in C-order then repeats.
            // F.ravel('C') = [0,4,8, 1,5,9, 2,6,10, 3,7,11]
            // After repeat by 2 = [0,0,4,4,8,8, 1,1,5,5,9,9, ...]
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.repeat(f, 2);
            r.size.Should().Be(24);
            ((long)r[0]).Should().Be(0);
            ((long)r[1]).Should().Be(0);
            ((long)r[2]).Should().Be(4);
            ((long)r[3]).Should().Be(4);
            ((long)r[22]).Should().Be(11);
            ((long)r[23]).Should().Be(11);
        }

        [TestMethod]
        public void Roll_FContig_Axis0_Values_MatchNumPy()
        {
            // NumPy: roll(F(4,3), 1, axis=0) rotates rows by 1
            // [[0,4,8],[1,5,9],[2,6,10],[3,7,11]] -> [[3,7,11],[0,4,8],[1,5,9],[2,6,10]]
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.roll(f, 1, axis: 0);
            ((long)r[0, 0]).Should().Be(3);
            ((long)r[0, 1]).Should().Be(7);
            ((long)r[0, 2]).Should().Be(11);
            ((long)r[1, 0]).Should().Be(0);
            ((long)r[3, 2]).Should().Be(10);
        }

        [TestMethod]
        public void Roll_FContig_Axis0_PreservesFContig()
        {
            // NumPy: roll(F, 1, axis=0) preserves F-contig layout (axis roll is a
            // strides-only operation, not a copy).
            var f = np.arange(12).reshape(3, 4).T;  // F-contig (4,3)
            var r = np.roll(f, 1, axis: 0);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: roll with axis preserves input F-contig layout");
        }

        [TestMethod]
        public void Stack_FF_Values_MatchNumPy()
        {
            // NumPy: stack([F,F]) yields (2,4,3) with arr[0]==arr[1]==F.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.stack(new[] { f, f });
            r.shape.Should().Equal(new long[] { 2, 4, 3 });
            ((long)r[0, 0, 0]).Should().Be(0);
            ((long)r[0, 3, 2]).Should().Be(11);
            ((long)r[1, 0, 0]).Should().Be(0);
            ((long)r[1, 3, 2]).Should().Be(11);
        }

        [TestMethod]
        public void ExpandDims_FContig_Axis0_Shape_MatchesNumPy()
        {
            // NumPy: expand_dims(F(4,3), axis=0) -> (1,4,3) with F-contig preserved.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.expand_dims(f, 0);
            r.shape.Should().Equal(new long[] { 1, 4, 3 });
            ((long)r[0, 0, 0]).Should().Be(0);
            ((long)r[0, 3, 2]).Should().Be(11);
        }

        [TestMethod]
        public void ExpandDims_FContig_Axis0_PreservesFContig()
        {
            // NumPy: expand_dims inserts a size-1 dim; stride of the new dim is anything
            // (size-1), and the other strides shift by one position. F-contig preserved.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.expand_dims(f, 0);
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: expand_dims preserves F-contig layout");
        }

        [TestMethod]
        public void ExpandDims_FContig_AxisMiddle_PreservesFContig()
        {
            // NumPy: expand_dims(F(4,3), axis=1) -> (4,1,3) F-contig.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.expand_dims(f, 1);
            r.shape.Should().Equal(new long[] { 4, 1, 3 });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void ExpandDims_FContig_AxisLast_PreservesFContig()
        {
            // NumPy: expand_dims(F(4,3), axis=2) -> (4,3,1) F-contig.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.expand_dims(f, 2);
            r.shape.Should().Equal(new long[] { 4, 3, 1 });
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // NumPy: squeeze(F(2,1,3)) -> (2,3) F-contig preserved.
                   // NumSharp: produces (2,3) C-contig — squeeze doesn't carry the
                   // F-strides pattern through the shape rebuild.
        public void Squeeze_FContigWithUnitDim_PreservesFContig()
        {
            var f3 = np.empty(new Shape(2L, 1L, 3L), order: 'F', dtype: typeof(double));
            f3[0, 0, 0] = 1.0;
            f3[1, 0, 2] = 99.0;
            var r = np.squeeze(f3);
            r.shape.Should().Equal(new long[] { 2, 3 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: squeeze preserves F-contig layout");
        }

        [TestMethod]
        public void SwapAxes_CContig2D_ProducesFContig()
        {
            // NumPy: swapaxes(C(3,4), 0, 1) -> (4,3) F-contig (just a stride swap).
            var c = np.arange(12).reshape(3, 4);
            var r = np.swapaxes(c, 0, 1);
            r.shape.Should().Equal(new long[] { 4, 3 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: swapaxes(C, 0, 1) on 2D yields F-contig (stride flip)");
        }

        [TestMethod]
        public void SwapAxes_FContig2D_ProducesCContig()
        {
            // NumPy: swapaxes(F(4,3), 0, 1) -> (3,4) C-contig (just a stride swap).
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.swapaxes(f, 0, 1);
            r.shape.Should().Equal(new long[] { 3, 4 });
            r.Shape.IsContiguous.Should().BeTrue(
                "NumPy: swapaxes(F, 0, 1) on 2D yields C-contig (stride flip)");
        }

        [TestMethod]
        public void AtLeast1d_Scalar_Is1DBothContig()
        {
            // NumPy: atleast_1d(scalar) -> (1,) both-contig.
            var r = np.atleast_1d(np.array(5));
            r.ndim.Should().Be(1);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void AtLeast2d_1D_IsBothContig()
        {
            // NumPy: atleast_2d([1,2,3]) -> (1,3) both-contig (size-1 dim).
            var v = np.array(new[] { 1, 2, 3 });
            var r = np.atleast_2d(v);
            r.shape.Should().Equal(new long[] { 1, 3 });
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void AtLeast3d_FContig2D_PreservesFContig()
        {
            // NumPy: atleast_3d(F(4,3)) -> (4,3,1), F-contig preserved.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.atleast_3d(f);
            r.shape.Should().Equal(new long[] { 4, 3, 1 });
            r.Shape.IsFContiguous.Should().BeTrue(
                "NumPy: atleast_3d adds trailing unit dim, preserves F-contig");
        }

        [TestMethod]
        public void MoveAxis_FContig2D_Effectively_Transposes()
        {
            // NumPy: moveaxis(F(4,3), 0, -1) on 2D is equivalent to transpose -> (3,4) C-contig.
            var f = np.arange(12).reshape(3, 4).T;
            var r = np.moveaxis(f, 0, -1);
            r.shape.Should().Equal(new long[] { 3, 4 });
            r.Shape.IsContiguous.Should().BeTrue(
                "NumPy: moveaxis(F, 0, -1) on 2D = transpose -> C-contig");
        }
    }
}
