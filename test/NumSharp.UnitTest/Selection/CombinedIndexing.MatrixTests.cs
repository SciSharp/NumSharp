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
    }
}
