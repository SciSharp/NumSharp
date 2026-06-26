using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Combined boolean + advanced indexing — a boolean mask (or integer index array)
    /// mixed with scalar ints, slices, ellipsis and newaxis in one index tuple. Pinned
    /// to NumPy 2.4.2: each mask is replaced by its <c>nonzero()</c> integer index arrays
    /// (mapping.c prepare_index), then advanced indexing applies — slices keep their own
    /// output axes (outer product), advanced indices broadcast among themselves.
    /// </summary>
    [TestClass]
    public class CombinedIndexing_MatrixTests
    {
        private static NDArray<bool> M(params bool[] b) => np.array(b).MakeGeneric<bool>();
        private static NDArray A => np.arange(12).reshape(3, 4);
        private static NDArray B3 => np.arange(24).reshape(2, 3, 4);

        // ----------------------------- GET -----------------------------

        [TestMethod]
        public void Get_Mask_Int()
        {
            // a[[T,F,T], 2] -> column 2 of rows 0,2 -> [2,10]
            A[M(true, false, true), 2].Should().BeOfValues(2, 10).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_Int_Mask()
        {
            // a[1, [T,F,T,F]] -> row 1, cols 0,2 -> [4,6]
            A[1, M(true, false, true, false)].Should().BeOfValues(4, 6).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_Mask_Slice()
        {
            // a[[T,F,T], 1:3] -> rows 0,2 x cols 1,2 -> (2,2) [1,2,9,10]
            A[M(true, false, true), "1:3"].Should().BeOfValues(1, 2, 9, 10).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Get_Slice_Mask()
        {
            // a[:, [T,F,T,F]] -> all rows x cols 0,2 -> (3,2) [0,2,4,6,8,10]
            A[":", M(true, false, true, false)].Should().BeOfValues(0, 2, 4, 6, 8, 10).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Get_Mask_Colon()
        {
            // a[[T,F,T], :] -> rows 0,2 -> (2,4)
            A[M(true, false, true), ":"].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11).And.BeShaped(2, 4);
        }

        [TestMethod]
        public void Get_Mask_Mask()
        {
            // a[[T,F,T], [T,F,T,F]] -> nonzero pairs (0,0),(2,2) -> [0,10]
            A[M(true, false, true), M(true, false, true, false)].Should().BeOfValues(0, 10).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_MaskRow_IntArrayCol()
        {
            // a[[T,T,F], [0,3]] -> pairs (0,0),(1,3) -> [0,7]
            A[M(true, true, false), np.array(new long[] { 0, 3 })].Should().BeOfValues(0, 7).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_2DMask_Int()
        {
            // b(2,3,4)[2dmask(2,3), 1] -> nonzero (0,0),(0,2),(1,1) then [..,1] -> [1,9,17]
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            B3[mask, 1].Should().BeOfValues(1, 9, 17).And.BeShaped(3);
        }

        [TestMethod]
        public void Get_Ellipsis_Mask()
        {
            // b[..., [T,F,T,F]] -> mask on last axis -> (2,3,2)
            B3["...", M(true, false, true, false)]
                .Should().BeOfValues(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22).And.BeShaped(2, 3, 2);
        }

        [TestMethod]
        public void Get_Int_Mask_Int()
        {
            // b[0, [T,F,T], 2] -> row plane 0, mask axis1 (rows 0,2), col 2 -> [2,10]
            B3[0, M(true, false, true), 2].Should().BeOfValues(2, 10).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_Slice_Slice_Mask_3D()
        {
            // b[:, :, [T,F,T,F]] -> mask last axis -> (2,3,2)
            B3[":", ":", M(true, false, true, false)]
                .Should().BeOfValues(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22).And.BeShaped(2, 3, 2);
        }

        [TestMethod]
        public void Get_Mask_NewAxis()
        {
            // a[[T,F,T], np.newaxis] -> rows 0,2 with inserted axis -> (2,1,4)
            A[M(true, false, true), Slice.NewAxis].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11).And.BeShaped(2, 1, 4);
        }

        [TestMethod]
        public void Get_NewAxis_Mask()
        {
            // a[np.newaxis, [T,F,T]] -> leading inserted axis -> (1,2,4)
            A[Slice.NewAxis, M(true, false, true)].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11).And.BeShaped(1, 2, 4);
        }

        [TestMethod]
        public void Get_Slice_IntArray_BroaderFix()
        {
            // a[:, [0,2]] (integer-array form of slice+advanced) -> (3,2)
            A[":", np.array(new long[] { 0, 2 })].Should().BeOfValues(0, 2, 4, 6, 8, 10).And.BeShaped(3, 2);
        }

        // ----------------------------- SET -----------------------------

        [TestMethod]
        public void Set_Mask_Int()
        {
            var a = A;
            a[M(true, false, true), 2] = (NDArray)(-1L);     // a[0,2], a[2,2]
            a.Should().BeOfValues(0, 1, -1, 3, 4, 5, 6, 7, 8, 9, -1, 11);
        }

        [TestMethod]
        public void Set_Slice_Mask()
        {
            var a = A;
            a[":", M(true, false, true, false)] = (NDArray)(-1L);  // cols 0,2 all rows
            a.Should().BeOfValues(-1, 1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11);
        }

        [TestMethod]
        public void Set_Int_Mask()
        {
            var a = A;
            a[1, M(true, false, true, false)] = (NDArray)(-1L);   // row 1, cols 0,2
            a.Should().BeOfValues(0, 1, 2, 3, -1, 5, -1, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Set_Mask_Slice()
        {
            var a = A;
            a[M(true, false, true), "1:3"] = (NDArray)(-1L);      // rows 0,2 x cols 1,2
            a.Should().BeOfValues(0, -1, -1, 3, 4, 5, 6, 7, 8, -1, -1, 11);
        }

        [TestMethod]
        public void Set_Mask_Colon()
        {
            var a = A;
            a[M(true, false, true), ":"] = (NDArray)(-1L);        // rows 0,2 entirely
            a.Should().BeOfValues(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1);
        }

        [TestMethod]
        public void Set_Slice_IntArray_BroaderFix()
        {
            var a = A;
            a[":", np.array(new long[] { 0, 2 })] = (NDArray)(-1L); // cols 0,2 all rows
            a.Should().BeOfValues(-1, 1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11);
        }

        [TestMethod]
        public void Set_Slice_Slice_Mask_3D()
        {
            var b = B3;
            b[":", ":", M(true, false, true, false)] = (NDArray)(-99L);   // last axis cols 0,2
            b.Should().BeOfValues(-99, 1, -99, 3, -99, 5, -99, 7, -99, 9, -99, 11,
                                  -99, 13, -99, 15, -99, 17, -99, 19, -99, 21, -99, 23);
        }

        [TestMethod]
        public void Set_Mask_Int_VectorValue()
        {
            // value broadcasts over the selection: a[[T,F,T], 1:3] = [10,20]
            var a = A;
            a[M(true, false, true), "1:3"] = np.array(new long[] { 10, 20 }); // per-column broadcast to 2 rows
            a.Should().BeOfValues(0, 10, 20, 3, 4, 5, 6, 7, 8, 10, 20, 11);
        }

        // ------------------- GET: negative indices -------------------

        [TestMethod]
        public void Get_Mask_NegativeInt()
        {
            // a[[T,F,T], -1] -> last column of rows 0,2 -> [3,11]
            A[M(true, false, true), -1].Should().BeOfValues(3, 11).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_Mask_NegativeArray()
        {
            // a[[T,F,T], [-1,-2]] -> pairs (0,-1),(2,-2) -> [3,10]
            A[M(true, false, true), np.array(new long[] { -1, -2 })].Should().BeOfValues(3, 10).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_NegativeInt_Mask()
        {
            // a[-1, [T,F,T,F]] -> last row cols 0,2 -> [8,10]
            A[-1, M(true, false, true, false)].Should().BeOfValues(8, 10).And.BeShaped(2);
        }

        [TestMethod]
        public void Get_ReversedSlice_Mask()
        {
            // a[::-1, [T,F,T,F]] -> reversed rows x cols 0,2 -> (3,2) [8,10,4,6,0,2]
            A["::-1", M(true, false, true, false)].Should().BeOfValues(8, 10, 4, 6, 0, 2).And.BeShaped(3, 2);
        }

        // ------------------- GET: step slices -------------------

        [TestMethod]
        public void Get_Mask_StepSlice()
        {
            // a[[T,F,T], ::2] -> rows 0,2 x cols 0,2 -> (2,2) [0,2,8,10]
            A[M(true, false, true), "::2"].Should().BeOfValues(0, 2, 8, 10).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void Get_StepSlice_Mask()
        {
            // a[::2, [T,F,T,F]] -> rows 0,2 x cols 0,2 -> (2,2) [0,2,8,10]
            A["::2", M(true, false, true, false)].Should().BeOfValues(0, 2, 8, 10).And.BeShaped(2, 2);
        }

        // ------------------- GET: empty / single masks -------------------

        [TestMethod]
        public void Get_EmptyMask_Int()
        {
            var r = A[M(false, false, false), 2];
            r.size.Should().Be(0);
            r.Should().BeShaped(0);
        }

        [TestMethod]
        public void Get_EmptyMask_Slice()
        {
            var r = A[M(false, false, false), "1:3"];
            r.size.Should().Be(0);
            r.Should().BeShaped(0, 2);
        }

        [TestMethod]
        public void Get_SingleTrueMask_Int()
        {
            // a[[F,T,F], 2] -> row 1 col 2 -> [6]
            A[M(false, true, false), 2].Should().BeOfValues(6).And.BeShaped(1);
        }

        [TestMethod]
        public void Get_Slice_EmptyMask()
        {
            var r = A[":", M(false, false, false, false)];
            r.size.Should().Be(0);
            r.Should().BeShaped(3, 0);
        }

        // ------------------- GET: 3-D / 4-D, mask in middle -------------------

        [TestMethod]
        public void Get_Slice_Mask_Slice_3D()
        {
            // b[:, [T,F,T], :] -> mask middle axis -> (2,2,4)
            B3[":", M(true, false, true), ":"]
                .Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23).And.BeShaped(2, 2, 4);
        }

        [TestMethod]
        public void Get_Int_Mask_Slice_3D()
        {
            // b[1, [T,F,T], :] -> plane 1, mask axis1 (rows 0,2) -> (2,4)
            B3[1, M(true, false, true), ":"].Should().BeOfValues(12, 13, 14, 15, 20, 21, 22, 23).And.BeShaped(2, 4);
        }

        [TestMethod]
        public void Get_Mask_Slice_Int_3D()
        {
            // b[[T,F], 1:3, 2] -> plane 0, cols 1,2, depth 2 -> (1,2) [6,10]
            B3[M(true, false), "1:3", 2].Should().BeOfValues(6, 10).And.BeShaped(1, 2);
        }

        [TestMethod]
        public void Get_4D_Slice_Mask()
        {
            // b4[:, :, [T,F,T,F], :] -> mask axis2 (idx 0,2) -> (2,3,2,5)
            var b4 = np.arange(120).reshape(2, 3, 4, 5);
            var r = b4[":", ":", M(true, false, true, false), ":"];
            r.Should().BeShaped(2, 3, 2, 5);
            r.size.Should().Be(60);
            ((long)r.GetValue(0, 0, 0, 0)).Should().Be(0);
            ((long)r.GetValue(0, 0, 1, 0)).Should().Be(10);   // axis2 index 2 -> base 10
            ((long)r.GetValue(1, 2, 1, 4)).Should().Be(114);
        }

        [TestMethod]
        public void Get_Mask_Ellipsis()
        {
            // b[[T,F], ...] -> plane 0 -> (1,3,4)
            B3[M(true, false), "..."].Should().BeShaped(1, 3, 4)
                .And.BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        // ------------------- GET: k-D mask + basic (leading multi-dim mask) -------------------

        [TestMethod]
        public void Get_2DMask_Slice()
        {
            // b[2dmask(2,3), 1:3] -> nonzero (0,0),(0,2),(1,1) then cols 1,2 -> (3,2)
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            B3[mask, "1:3"].Should().BeOfValues(1, 2, 9, 10, 17, 18).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Get_2DMask_Colon()
        {
            // b[2dmask(2,3), :] -> 3 selected planes -> (3,4)
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            B3[mask, ":"].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19).And.BeShaped(3, 4);
        }

        // ------------------- GET: transposed / strided source -------------------

        [TestMethod]
        public void Get_Transposed_Mask_Int()
        {
            // a.T is (4,3); a.T[[T,F,T,F], 1] -> rows 0,2 of the transpose, col 1 -> [4,6]
            A.T[M(true, false, true, false), 1].Should().BeOfValues(4, 6).And.BeShaped(2);
        }

        // ------------------- SET: negative / step / empty / 3-D -------------------

        [TestMethod]
        public void Set_Mask_NegativeInt()
        {
            var a = A;
            a[M(true, false, true), -1] = (NDArray)(-1L);          // last col of rows 0,2
            a.Should().BeOfValues(0, 1, 2, -1, 4, 5, 6, 7, 8, 9, 10, -1);
        }

        [TestMethod]
        public void Set_Mask_StepSlice()
        {
            var a = A;
            a[M(true, false, true), "::2"] = (NDArray)(-1L);       // rows 0,2 x cols 0,2
            a.Should().BeOfValues(-1, 1, -1, 3, 4, 5, 6, 7, -1, 9, -1, 11);
        }

        [TestMethod]
        public void Set_EmptyMask_NoOp()
        {
            var a = A;
            a[M(false, false, false), 2] = (NDArray)(-1L);
            a.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Set_NegativeInt_Mask()
        {
            var a = A;
            a[-1, M(true, false, true, false)] = (NDArray)(-5L);   // last row, cols 0,2
            a.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, -5, 9, -5, 11);
        }

        [TestMethod]
        public void Set_Slice_Mask_GridValue()
        {
            // a[:, [T,F,T,F]] = [[1,2],[3,4],[5,6]] (full (3,2) grid)
            var a = A;
            a[":", M(true, false, true, false)] = np.array(new long[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            a.Should().BeOfValues(1, 1, 2, 3, 3, 5, 4, 7, 5, 9, 6, 11);
        }

        [TestMethod]
        public void Set_Slice_Mask_Slice_3D()
        {
            var b = B3;
            b[":", M(true, false, true), ":"] = (NDArray)(-7L);    // mask middle axis (rows 0,2)
            b.Should().BeOfValues(-7, -7, -7, -7, 4, 5, 6, 7, -7, -7, -7, -7,
                                  -7, -7, -7, -7, 16, 17, 18, 19, -7, -7, -7, -7);
        }

        [TestMethod]
        public void Set_2DMask_Slice()
        {
            // b[2dmask, 1:3] = -1 (leading k-D mask + slice, writes through the view)
            var b = B3;
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            b[mask, "1:3"] = (NDArray)(-1L);
            b.Should().BeOfValues(0, -1, -1, 3, 4, 5, 6, 7, 8, -1, -1, 11,
                                  12, 13, 14, 15, 16, -1, -1, 19, 20, 21, 22, 23);
        }

        // ------------------- error parity -------------------

        [TestMethod]
        public void Set_Mask_Slice_IncompatibleValue_Throws()
        {
            // np: a[[T,F,T], 1:3] (selection (2,2)) = [1,2,3,4,5] -> ValueError
            var a = A;
            Action act = () => a[M(true, false, true), "1:3"] = np.array(new long[] { 1, 2, 3, 4, 5 });
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void Set_Slice_Mask_VectorValueMismatch_Throws()
        {
            // np: a[:, [T,F,T,F]] (selection (3,2)) = [100,200,300] ((3,) !-> (3,2)) -> ValueError
            var a = A;
            Action act = () => a[":", M(true, false, true, false)] = np.array(new long[] { 100, 200, 300 });
            act.Should().Throw<IncorrectShapeException>();
        }

        // ------------------- documented gap (NumPy parity not yet implemented) -------------------

        [TestMethod]
        [OpenBugs]
        public void Get_TwoMasks_SeparatedBySlice_AdvancedAxesToFront_Unsupported()
        {
            // np: b[[T,F], :, [T,F,T,F]] -> two advanced indices separated by a slice.
            // NumPy broadcasts the two advanced indices (-> length-2) and, because they are
            // NON-contiguous, moves the advanced axis to the FRONT: result (2,3) [0,4,8,2,6,10].
            // NumSharp does not yet implement the advanced-axes-to-front rule for separated
            // advanced indices and raises instead.
            var r = B3[M(true, false), ":", M(true, false, true, false)];
            r.Should().BeShaped(2, 3).And.BeOfValues(0, 4, 8, 2, 6, 10);
        }
    }
}
