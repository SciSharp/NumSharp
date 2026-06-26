using System;
using System.Linq;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Generic;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Selection
{
    /// <summary>
    /// Exhaustive boolean-mask matrix (get + set) pinned to NumPy 2.4.2 output.
    /// Covers the full/axis-0/partial/0-D selection modes across C/F/strided/
    /// transposed/negative-stride/sliced/broadcast/empty/higher-rank layouts, the
    /// 15 NumSharp dtypes, value broadcasting/casting, and the error cases. Every
    /// expected value was produced by running the equivalent expression in NumPy.
    /// </summary>
    [TestClass]
    public class BooleanMasking_MatrixTests
    {
        private static NDArray<bool> M(params bool[] b) => np.array(b).MakeGeneric<bool>();
        private static NDArray Iota(int n) => np.arange(n);                  // 0..n-1 (Int64)
        private static NDArray Iota2(int r, int c) => np.arange(r * c).reshape(r, c);

        // =================================================================
        //  GET — full element mask (mask.ndim == arr.ndim)
        // =================================================================

        [TestMethod]
        public void Get_Full_1D()
        {
            // np.arange(6)[[T,F,T,F,T,F]] -> [0,2,4]
            Iota(6)[M(true, false, true, false, true, false)].Should().BeOfValues(0, 2, 4).And.BeShaped(3);
        }

        [TestMethod]
        public void Get_Full_2D_FlattensToCOrder()
        {
            // a=arange(12).reshape(3,4); a[a%2==0] -> [0,2,4,6,8,10] shape (6,)
            var a = Iota2(3, 4);
            a[(a % 2) == 0].Should().BeOfValues(0, 2, 4, 6, 8, 10).And.BeShaped(6);
        }

        [TestMethod]
        public void Get_Full_AllTrue_ReturnsFlatCopy()
        {
            var a = Iota2(3, 4);
            a[np.ones(new Shape(3, 4), np.bool_).MakeGeneric<bool>()]
                .Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11).And.BeShaped(12);
        }

        [TestMethod]
        public void Get_Full_AllFalse_EmptyShape()
        {
            // np: a[zeros((3,4),bool)] -> shape (0,)
            var a = Iota2(3, 4);
            var r = a[np.zeros(new Shape(3, 4), np.bool_).MakeGeneric<bool>()];
            r.size.Should().Be(0);
            r.Should().BeShaped(0);
        }

        [TestMethod]
        public void Get_Full_4D()
        {
            // arange(16).reshape(2,2,2,2)[a%3==0] -> [0,3,6,9,12,15]
            var a = np.arange(16).reshape(2, 2, 2, 2);
            a[(a % 3) == 0].Should().BeOfValues(0, 3, 6, 9, 12, 15).And.BeShaped(6);
        }

        // =================================================================
        //  GET — axis-0 / partial (mask.ndim < arr.ndim)
        // =================================================================

        [TestMethod]
        public void Get_Axis0_SelectsRows()
        {
            // a(3,4)[[T,F,T]] -> rows 0,2 -> (2,4)
            Iota2(3, 4)[M(true, false, true)]
                .Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11).And.BeShaped(2, 4);
        }

        [TestMethod]
        public void Get_Axis0_AllFalse_EmptyRows()
        {
            // a(3,4)[[F,F,F]] -> (0,4)
            var r = Iota2(3, 4)[M(false, false, false)];
            r.size.Should().Be(0);
            r.Should().BeShaped(0, 4);
        }

        [TestMethod]
        public void Get_Partial_2DMaskOn3D()
        {
            // a(2,3,4); mask(2,3)=[[T,F,T],[F,T,F]] -> (3,4)
            var a = np.arange(24).reshape(2, 3, 4);
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            a[mask].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19).And.BeShaped(3, 4);
        }

        [TestMethod]
        public void Get_Partial_OnFContiguousArray()
        {
            // Same partial selection, but the source is F-contiguous (non-C). C-order result identical.
            var a = np.asfortranarray(np.arange(24).reshape(2, 3, 4));
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            a[mask].Should().BeOfValues(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19).And.BeShaped(3, 4);
        }

        // =================================================================
        //  GET — 0-D mask (arr[True] / arr[False] add/empty an axis)
        // =================================================================

        [TestMethod]
        public void Get_ScalarTrue_AddsLeadingAxis()
        {
            // a(3,4)[np.array(True)] -> (1,3,4) with all data
            var a = Iota2(3, 4);
            var r = a[np.array(true).MakeGeneric<bool>()];
            r.Should().BeShaped(1, 3, 4);
            r.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Get_ScalarFalse_EmptyWithLeadingAxis()
        {
            // a(3,4)[np.array(False)] -> (0,3,4)
            var r = Iota2(3, 4)[np.array(false).MakeGeneric<bool>()];
            r.size.Should().Be(0);
            r.Should().BeShaped(0, 3, 4);
        }

        [TestMethod]
        public void Get_ScalarTrue_OnScalarArray()
        {
            // np.array(5.0)[np.array(True)] -> (1,) [5.0]
            var r = np.array(5.0)[np.array(true).MakeGeneric<bool>()];
            r.Should().BeShaped(1);
            ((double)r.GetValue(0)).Should().Be(5.0);
        }

        // =================================================================
        //  GET — layout variations (all gather in NumPy C-order)
        // =================================================================

        [TestMethod]
        public void Get_FContiguous_Full()
        {
            var a = np.asfortranarray(Iota2(3, 4));
            a[(a % 2) == 0].Should().BeOfValues(0, 2, 4, 6, 8, 10).And.BeShaped(6);
        }

        [TestMethod]
        public void Get_Transposed_Full()
        {
            // a.T is (4,3); C-order over the transposed view -> [0,4,8,2,6,10]
            var a = Iota2(3, 4).T;
            a[(a % 2) == 0].Should().BeOfValues(0, 4, 8, 2, 6, 10).And.BeShaped(6);
        }

        [TestMethod]
        public void Get_NegativeStride_Full()
        {
            // a[::-1] rows reversed; a%2==0 -> [8,10,4,6,0,2]
            var a = Iota2(3, 4)["::-1"];
            a[(a % 2) == 0].Should().BeOfValues(8, 10, 4, 6, 0, 2).And.BeShaped(6);
        }

        [TestMethod]
        public void Get_SlicedOffset_Full()
        {
            // arange(20)[5:][>10] -> [11..19]
            var a = Iota(20)["5:"];
            a[a > 10].Should().BeOfValues(11, 12, 13, 14, 15, 16, 17, 18, 19).And.BeShaped(9);
        }

        [TestMethod]
        public void Get_ColumnSlice_Strided()
        {
            // a(3,4)[:,1:3] is strided/non-contig; [%2==0] -> [2,6,10]
            var a = Iota2(3, 4)[":, 1:3"];
            a[(a % 2) == 0].Should().BeOfValues(2, 6, 10).And.BeShaped(3);
        }

        [TestMethod]
        public void Get_BroadcastArray_Full()
        {
            // broadcast_to(arange(4),(3,4)) read-only; [%2==0] -> [0,2,0,2,0,2]
            var a = np.broadcast_to(Iota(4), new Shape(3, 4));
            a[(a % 2) == 0].Should().BeOfValues(0, 2, 0, 2, 0, 2).And.BeShaped(6);
        }

        [TestMethod]
        public void Get_MaskNegativeStride()
        {
            // arange(5)[mask[::-1]] where mask=[T,F,T,F,T] reversed=[T,F,T,F,T] -> [0,2,4]
            var mask = M(true, false, true, false, true)["::-1"].MakeGeneric<bool>();
            Iota(5)[mask].Should().BeOfValues(0, 2, 4).And.BeShaped(3);
        }

        [TestMethod]
        public void Get_MaskFContiguous()
        {
            // a(3,4) C-contig, mask F-contig (same values) -> identical C-order result
            var a = Iota2(3, 4);
            var mask = np.asfortranarray((a % 2) == 0).MakeGeneric<bool>();
            a[mask].Should().BeOfValues(0, 2, 4, 6, 8, 10).And.BeShaped(6);
        }

        // =================================================================
        //  GET — regression: 1-D length-1 mask is a NORMAL mask, NOT 0-D.
        //  (Pre-unification this wrongly added an axis -> (1,1).)
        // =================================================================

        [TestMethod]
        public void Get_OneElement1DMask_IsNormalMask()
        {
            // np.array([7])[np.array([True])] -> (1,) [7]   (NOT (1,1))
            np.array(new long[] { 7 })[M(true)].Should().BeShaped(1).And.BeOfValues(7);
            // np.array([7])[np.array([False])] -> (0,)
            np.array(new long[] { 7 })[M(false)].size.Should().Be(0);
        }

        [TestMethod]
        public void Get_OneElement1DMask_On2D_SelectsRow()
        {
            // arange(4).reshape(1,4)[[True]] -> (1,4); [[False]] -> (0,4)
            np.arange(4).reshape(1, 4)[M(true)].Should().BeShaped(1, 4).And.BeOfValues(0, 1, 2, 3);
            np.arange(4).reshape(1, 4)[M(false)].Should().BeShaped(0, 4);
        }

        // =================================================================
        //  GET — error cases (mask must match a leading prefix)
        // =================================================================

        [TestMethod]
        public void Get_MaskShorterThanAxis_Throws()
        {
            // np.arange(5)[[T,F,T]] -> IndexError
            Action act = () => { var _ = Iota(5)[M(true, false, true)]; };
            act.Should().Throw<IndexOutOfRangeException>();
        }

        [TestMethod]
        public void Get_MaskLongerThanAxis_Throws()
        {
            Action act = () => { var _ = Iota(5)[M(true, false, true, false, true, true)]; };
            act.Should().Throw<IndexOutOfRangeException>();
        }

        [TestMethod]
        public void Get_MaskNdimGreaterThanArr_Throws()
        {
            // np.arange(3)[[[T,F,T]]] (mask 2-D on 1-D arr) -> IndexError
            var mask = np.array(new bool[,] { { true, false, true } }).MakeGeneric<bool>();
            Action act = () => { var _ = Iota(3)[mask]; };
            act.Should().Throw<IndexOutOfRangeException>();
        }

        // =================================================================
        //  GET — all 15 dtypes (proves the gather handles every element size)
        // =================================================================

        [TestMethod]
        public void Get_AllDtypes_FullAndBlock()
        {
            var mask = M(true, false, true, false, true, false);      // picks idx 0,2,4
            var rowMask = M(true, false, true);                       // picks rows 0,2
            foreach (NPTypeCode dt in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            })
            {
                var src = np.arange(1, 7).astype(dt);                 // 1..6
                var got = src[mask];
                got.Should().BeShaped(3);
                for (int i = 0; i < 3; i++)
                    Convert.ToString(got.GetValue(i)).Should().Be(Convert.ToString(src.GetValue(i * 2)),
                        $"value {i} of {dt} get");

                // (3,2) block gather: rows 0,2 -> values [1,2,5,6]
                var block = src.reshape(3, 2)[rowMask];
                block.Should().BeShaped(2, 2);
            }
        }

        // =================================================================
        //  SET — full element mask
        // =================================================================

        [TestMethod]
        public void Set_Full_Scalar()
        {
            var a = Iota2(3, 4);
            a[(a % 2) == 0] = (NDArray)(-1L);
            a.Should().BeOfValues(-1, 1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11);
        }

        [TestMethod]
        public void Set_Full_Array_COrder()
        {
            var a = Iota2(3, 4);
            a[(a % 3) == 0] = np.array(new long[] { 100, 200, 300, 400 });   // 4 trues at idx 0,3,6,9
            a.Should().BeOfValues(100, 1, 2, 200, 4, 5, 300, 7, 8, 400, 10, 11);
        }

        [TestMethod]
        public void Set_Full_CastFloatToInt_Truncates()
        {
            // np: int-array[mask] = 3.9 -> stores 3 (truncation, same_kind cast)
            var a = Iota2(3, 4);
            a[(a % 2) == 0] = (NDArray)3.9;
            a.Should().BeOfValues(3, 1, 3, 3, 3, 5, 3, 7, 3, 9, 3, 11);
        }

        [TestMethod]
        public void Set_Full_IncompatibleValue_Throws()
        {
            // np: a(3,4)[a%2==0] (6 trues) = [1,2,3] -> ValueError
            var a = Iota2(3, 4);
            Action act = () => a[(a % 2) == 0] = np.array(new long[] { 1, 2, 3 });
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void Set_Full_ValueTooLong_Throws()
        {
            var a = Iota2(3, 4);
            Action act = () => a[(a % 3) == 0] = np.array(new long[] { 1, 2, 3, 4, 5 }); // 4 trues, 5 vals
            act.Should().Throw<IncorrectShapeException>();
        }

        // =================================================================
        //  SET — axis-0 (value broadcasts to (count,) + arr.shape[1:])
        // =================================================================

        [TestMethod]
        public void Set_Axis0_Scalar()
        {
            var a = Iota2(3, 4);
            a[M(true, false, true)] = (NDArray)(-9L);
            a.Should().BeOfValues(-9, -9, -9, -9, 4, 5, 6, 7, -9, -9, -9, -9);
        }

        [TestMethod]
        public void Set_Axis0_RowBroadcast()
        {
            // value (4,) broadcasts across each selected row
            var a = Iota2(3, 4);
            a[M(true, false, true)] = np.array(new long[] { 10, 20, 30, 40 });
            a.Should().BeOfValues(10, 20, 30, 40, 4, 5, 6, 7, 10, 20, 30, 40);
        }

        [TestMethod]
        public void Set_Axis0_PerRow()
        {
            // value (2,4) assigns one row each to the 2 selected rows
            var a = Iota2(3, 4);
            a[M(true, false, true)] = np.array(new long[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
            a.Should().BeOfValues(1, 2, 3, 4, 4, 5, 6, 7, 5, 6, 7, 8);
        }

        [TestMethod]
        public void Set_Axis0_CountLengthValue_Throws()
        {
            // np: a(3,4)[rowmask picks 2] = [10,20]  -> ValueError ((2,) !-> (2,4))
            var a = Iota2(3, 4);
            Action act = () => a[M(true, false, true)] = np.array(new long[] { 10, 20 });
            act.Should().Throw<IncorrectShapeException>();
        }

        [TestMethod]
        public void Set_Axis0_CountByOneValue_OnePerRow()
        {
            // value (2,1) broadcasts across columns -> one value per selected row
            var a = Iota2(3, 4);
            a[M(true, false, true)] = np.array(new long[,] { { 10 }, { 20 } });
            a.Should().BeOfValues(10, 10, 10, 10, 4, 5, 6, 7, 20, 20, 20, 20);
        }

        // =================================================================
        //  SET — partial (2D mask on 3D)
        // =================================================================

        [TestMethod]
        public void Set_Partial_Scalar()
        {
            var a = np.arange(24).reshape(2, 3, 4);
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            a[mask] = (NDArray)(-7L);
            a.Should().BeOfValues(-7, -7, -7, -7, 4, 5, 6, 7, -7, -7, -7, -7,
                                  12, 13, 14, 15, -7, -7, -7, -7, 20, 21, 22, 23);
        }

        [TestMethod]
        public void Set_Partial_Block()
        {
            // value (3,4): one (4,) block per selected (i,j)
            var a = np.arange(24).reshape(2, 3, 4);
            var mask = np.array(new bool[,] { { true, false, true }, { false, true, false } }).MakeGeneric<bool>();
            a[mask] = np.array(new long[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
            a.Should().BeOfValues(1, 2, 3, 4, 4, 5, 6, 7, 5, 6, 7, 8,
                                  12, 13, 14, 15, 9, 10, 11, 12, 20, 21, 22, 23);
        }

        // =================================================================
        //  SET — 0-D mask, layout, edge cases
        // =================================================================

        [TestMethod]
        public void Set_ScalarTrue_AssignsAll()
        {
            // a[np.array(True)] = 0 -> whole array 0
            var a = Iota2(3, 4);
            a[np.array(true).MakeGeneric<bool>()] = (NDArray)0L;
            a.Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        }

        [TestMethod]
        public void Set_ScalarFalse_NoOp()
        {
            // a[np.array(False)] = -1 -> unchanged
            var a = Iota2(3, 4);
            a[np.array(false).MakeGeneric<bool>()] = (NDArray)(-1L);
            a.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Set_NonWriteable_Throws()
        {
            // assigning into a broadcast (read-only) view -> NumPy ValueError
            var a = np.broadcast_to(Iota(4), new Shape(3, 4));
            Action act = () => a[(a % 2) == 0] = (NDArray)(-1L);
            act.Should().Throw<NumSharpException>();
        }

        [TestMethod]
        public void Set_NegativeStrideView_WritesThrough()
        {
            // mutate a reversed view; underlying buffer reflects the writes
            var a = Iota2(3, 4);
            var v = a["::-1"];
            v[(v % 2) == 0] = (NDArray)(-1L);
            a.Should().BeOfValues(-1, 1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11);
        }

        [TestMethod]
        public void Set_EmptyMask_NoOp()
        {
            var a = Iota2(3, 4);
            a[a > 1000] = (NDArray)(-1L);
            a.Should().BeOfValues(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
        }

        [TestMethod]
        public void Set_ColumnSlice_Strided()
        {
            // assign into a strided column-slice view; writes land in the parent
            var a = Iota2(3, 4);
            var v = a[":, 1:3"];
            v[(v % 2) == 0] = (NDArray)(-5L);     // even values 2,6,10 in cols 1..2
            a.Should().BeOfValues(0, 1, -5, 3, 4, 5, -5, 7, 8, 9, -5, 11);
        }

        [TestMethod]
        public void Set_AllDtypes_ScalarSplat()
        {
            var rowMask = M(true, false, true);
            foreach (NPTypeCode dt in new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Char,
                NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Decimal, NPTypeCode.Complex
            })
            {
                var a = np.arange(1, 7).astype(dt).reshape(3, 2);     // rows [1,2],[3,4],[5,6]
                var fill = np.arange(1, 7).astype(dt).reshape(3, 2)[0, 0]; // dtype value "1"
                a[rowMask] = fill;                                    // rows 0,2 -> 1
                a.Should().BeShaped(3, 2);
                Convert.ToString(a.GetValue(0, 0)).Should().Be(Convert.ToString(fill.GetValue(0)), $"{dt} row0");
                Convert.ToString(a.GetValue(1, 0)).Should().Be(Convert.ToString(np.arange(1, 7).astype(dt).reshape(3, 2).GetValue(1, 0)), $"{dt} row1 untouched");
            }
        }

        // =================================================================
        //  Documented LIMITATIONS — combined boolean + other indexing.
        //  NumPy supports these; NumSharp's mask indexer does not. Marked
        //  [OpenBugs] so they surface when combined advanced indexing lands.
        // =================================================================

        [TestMethod]
        [OpenBugs]
        public void Get_CombinedMaskAndInteger_Unsupported()
        {
            // np: a(3,4)[ [T,F,T], 2 ] -> column 2 of rows 0,2 -> [2,10] shape (2,)
            // NumSharp currently ignores the integer index and returns (2,4).
            var a = Iota2(3, 4);
            var r = a[M(true, false, true), 2];
            r.Should().BeShaped(2).And.BeOfValues(2, 10);
        }

        [TestMethod]
        [OpenBugs]
        public void Get_SliceThenMask_NonLeadingAxis_Unsupported()
        {
            // np: a(3,4)[:, [T,F,T,F]] -> columns 0,2 -> (3,2) [0,2,4,6,8,10]
            // NumSharp applies the mask to axis 0 (length 3 != 4) and throws.
            var a = Iota2(3, 4);
            var r = a[":", M(true, false, true, false)];
            r.Should().BeShaped(3, 2).And.BeOfValues(0, 2, 4, 6, 8, 10);
        }
    }
}
