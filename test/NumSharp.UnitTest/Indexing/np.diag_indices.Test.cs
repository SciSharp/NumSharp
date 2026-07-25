using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Indexing
{
    /// <summary>
    ///     The index-tuple generators — np.diag_indices(_from), np.tril_indices(_from),
    ///     np.triu_indices(_from), np.mask_indices — plus in-place np.fill_diagonal.
    ///     Every expectation below is real NumPy 2.4.2 output.
    /// </summary>
    [TestClass]
    public class np_diag_indices_Test
    {
        // ---------------------------------------------------------------- diag_indices

        [TestMethod]
        public void DiagIndices_DefaultIsTwoDimensional()
        {
            var idx = np.diag_indices(3);
            idx.Length.Should().Be(2);
            idx[0].Should().BeOfValues(0, 1, 2);
            idx[1].Should().BeOfValues(0, 1, 2);
            idx[0].typecode.Should().Be(NPTypeCode.Int64, "NumPy's intp is int64 on 64-bit platforms");
        }

        [TestMethod]
        public void DiagIndices_ArityFollowsNdim()
        {
            np.diag_indices(3, 3).Length.Should().Be(3);
            np.diag_indices(4, 4).Length.Should().Be(4);
            np.diag_indices(3, 1).Length.Should().Be(1);
            // Python's (idx,) * 0 and (idx,) * -1 are both the empty tuple.
            np.diag_indices(3, 0).Length.Should().Be(0);
            np.diag_indices(3, -1).Length.Should().Be(0);
        }

        [TestMethod]
        public void DiagIndices_EntriesAliasOneAnother()
        {
            // NumPy returns `(idx,) * ndim` — literally the same array object in every slot,
            // so a write through one entry is visible through all of them.
            var idx = np.diag_indices(3, 3);
            idx[0][0] = 99L;
            idx[1].GetAtIndex<long>(0).Should().Be(99L);
            idx[2].GetAtIndex<long>(0).Should().Be(99L);
        }

        [TestMethod]
        public void DiagIndices_EmptyAndNegativeN()
        {
            np.diag_indices(0)[0].Should().BeShaped(0);
            np.diag_indices(-1)[0].Should().BeShaped(0);
        }

        [TestMethod]
        public void DiagIndicesFrom_UsesShapeAndNdim()
        {
            var two = np.diag_indices_from(np.zeros(new Shape(3, 3)));
            two.Length.Should().Be(2);
            two[0].Should().BeOfValues(0, 1, 2);

            var three = np.diag_indices_from(np.zeros(new Shape(2, 2, 2)));
            three.Length.Should().Be(3);
            three[0].Should().BeOfValues(0, 1);

            np.diag_indices_from(np.zeros(new Shape(0, 0)))[0].Should().BeShaped(0);
        }

        [TestMethod]
        public void DiagIndicesFrom_Errors_VerbatimText()
        {
            new Action(() => np.diag_indices_from(np.zeros(new Shape(3))))
                .Should().Throw<ArgumentException>().WithMessage("input array must be at least 2-d");
            new Action(() => np.diag_indices_from(NDArray.Scalar(1.0)))
                .Should().Throw<ArgumentException>().WithMessage("input array must be at least 2-d");
            new Action(() => np.diag_indices_from(np.zeros(new Shape(3, 4))))
                .Should().Throw<ArgumentException>()
                .WithMessage("All dimensions of input must be of equal length");
        }

        // ---------------------------------------------------------------- tril/triu_indices

        [TestMethod]
        public void TrilIndices_RowMajorOrder()
        {
            var t = np.tril_indices(3);
            t[0].Should().BeOfValues(0, 1, 1, 2, 2, 2);
            t[1].Should().BeOfValues(0, 0, 1, 0, 1, 2);
            t[0].typecode.Should().Be(NPTypeCode.Int64);
        }

        [TestMethod]
        public void TriuIndices_RowMajorOrder()
        {
            var t = np.triu_indices(3);
            t[0].Should().BeOfValues(0, 0, 0, 1, 1, 2);
            t[1].Should().BeOfValues(0, 1, 2, 1, 2, 2);
        }

        [TestMethod]
        public void TrilIndices_KOffsets()
        {
            var up = np.tril_indices(3, 1);
            up[0].Should().BeOfValues(0, 0, 1, 1, 1, 2, 2, 2);
            up[1].Should().BeOfValues(0, 1, 0, 1, 2, 0, 1, 2);

            var down = np.tril_indices(3, -1);
            down[0].Should().BeOfValues(1, 2, 2);
            down[1].Should().BeOfValues(0, 0, 1);
        }

        [TestMethod]
        public void TriuIndices_KOffsets()
        {
            var up = np.triu_indices(3, 1);
            up[0].Should().BeOfValues(0, 0, 1);
            up[1].Should().BeOfValues(1, 2, 2);

            var down = np.triu_indices(3, -1);
            down[0].Should().BeOfValues(0, 0, 0, 1, 1, 1, 2, 2);
            down[1].Should().BeOfValues(0, 1, 2, 0, 1, 2, 1, 2);
        }

        [TestMethod]
        public void TriIndices_RectangularM()
        {
            var wide = np.tril_indices(3, 0, 5);
            wide[0].Should().BeOfValues(0, 1, 1, 2, 2, 2);
            wide[1].Should().BeOfValues(0, 0, 1, 0, 1, 2);

            var narrow = np.tril_indices(3, 0, 2);
            narrow[0].Should().BeOfValues(0, 1, 1, 2, 2);
            narrow[1].Should().BeOfValues(0, 0, 1, 0, 1);

            var u = np.triu_indices(3, 0, 5);
            u[0].Should().BeOfValues(0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2);
            u[1].Should().BeOfValues(0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4);
        }

        [TestMethod]
        public void TriIndices_SaturatingKAndEmptyExtents()
        {
            np.tril_indices(3, 10)[0].Should().BeShaped(9);
            np.tril_indices(3, -10)[0].Should().BeShaped(0);
            np.triu_indices(3, 10)[0].Should().BeShaped(0);
            np.triu_indices(3, -10)[0].Should().BeShaped(9);

            np.tril_indices(0)[0].Should().BeShaped(0);
            np.tril_indices(3, 0, 0)[0].Should().BeShaped(0);
            np.tril_indices(-2)[0].Should().BeShaped(0);
            np.tril_indices(3, 0, -2)[0].Should().BeShaped(0);
            np.tril_indices(1)[0].Should().BeOfValues(0);
        }

        [TestMethod]
        public void TriIndicesFrom_TakesShapeFromArray()
        {
            var l = np.tril_indices_from(np.zeros(new Shape(3, 4)));
            l[0].Should().BeOfValues(0, 1, 1, 2, 2, 2);
            l[1].Should().BeOfValues(0, 0, 1, 0, 1, 2);

            var lk = np.tril_indices_from(np.zeros(new Shape(3, 4)), 1);
            lk[0].Should().BeOfValues(0, 0, 1, 1, 1, 2, 2, 2, 2);

            var u = np.triu_indices_from(np.zeros(new Shape(3, 4)));
            u[0].Should().BeOfValues(0, 0, 0, 0, 1, 1, 1, 2, 2);
            u[1].Should().BeOfValues(0, 1, 2, 3, 1, 2, 3, 2, 3);

            np.tril_indices_from(np.zeros(new Shape(0, 0)))[0].Should().BeShaped(0);
        }

        [TestMethod]
        public void TriIndicesFrom_RequiresExactlyTwoD()
        {
            new Action(() => np.tril_indices_from(np.zeros(new Shape(3))))
                .Should().Throw<ArgumentException>().WithMessage("input array must be 2-d");
            new Action(() => np.tril_indices_from(np.zeros(new Shape(2, 3, 4))))
                .Should().Throw<ArgumentException>().WithMessage("input array must be 2-d");
            new Action(() => np.triu_indices_from(np.zeros(new Shape(2, 3, 4))))
                .Should().Throw<ArgumentException>().WithMessage("input array must be 2-d");
        }

        [TestMethod]
        public void TriIndices_ActuallyIndexTheTriangle()
        {
            // Round-trip: gathering with the indices must equal the masked array's non-zeros.
            var m = np.arange(1, 17).reshape(4, 4);
            var idx = np.triu_indices(4, 1);
            var gathered = new long[idx[0].size];
            for (int i = 0; i < idx[0].size; i++)
                gathered[i] = m.GetValue<long>(idx[0].GetAtIndex<long>(i), idx[1].GetAtIndex<long>(i));

            gathered.Should().Equal(2L, 3, 4, 7, 8, 12);
        }

        // ---------------------------------------------------------------- mask_indices

        [TestMethod]
        public void MaskIndices_WithTriuAndTril()
        {
            var u = np.mask_indices(3, (a, k) => np.triu(a, k));
            u[0].Should().BeOfValues(0, 0, 0, 1, 1, 2);
            u[1].Should().BeOfValues(0, 1, 2, 1, 2, 2);

            var l = np.mask_indices(3, (a, k) => np.tril(a, k), 1);
            l[0].Should().BeOfValues(0, 0, 1, 1, 1, 2, 2, 2);
            l[1].Should().BeOfValues(0, 1, 0, 1, 2, 0, 1, 2);

            var lm = np.mask_indices(4, (a, k) => np.tril(a, k), -1);
            lm[0].Should().BeOfValues(1, 2, 2, 3, 3, 3);
            lm[1].Should().BeOfValues(0, 0, 1, 0, 1, 2);
        }

        [TestMethod]
        public void MaskIndices_ArityFollowsMaskFunctionRank()
        {
            // np.diag reduces the (n,n) probe to its 1-D diagonal, so nonzero yields ONE array.
            var d = np.mask_indices(3, (a, k) => np.diag(a, k));
            d.Length.Should().Be(1);
            d[0].Should().BeOfValues(0, 1, 2);
        }

        [TestMethod]
        public void MaskIndices_ZeroN()
        {
            var z = np.mask_indices(0, (a, k) => np.triu(a, k));
            z.Length.Should().Be(2);
            z[0].Should().BeShaped(0);
        }

        // ---------------------------------------------------------------- fill_diagonal

        [TestMethod]
        public void FillDiagonal_ScalarAndSequence()
        {
            var a = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(a, 5);
            a.Should().BeOfValues(5, 0, 0, 0, 5, 0, 0, 0, 5);

            var b = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(b, np.array(1, 2, 3));
            b.Should().BeOfValues(1, 0, 0, 0, 2, 0, 0, 0, 3);
        }

        [TestMethod]
        public void FillDiagonal_ValuesTileCyclically_AndTruncate()
        {
            // NumPy's `a.flat[:end:step] = val` repeats a short RHS rather than raising.
            var shortVal = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(shortVal, np.array(1, 2));
            np.diagonal(shortVal).Should().BeOfValues(1, 2, 1);

            var six = np.zeros(new Shape(6, 6), NPTypeCode.Int32);
            np.fill_diagonal(six, np.array(1, 2, 3, 4));
            np.diagonal(six).Should().BeOfValues(1, 2, 3, 4, 1, 2);

            var longVal = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(longVal, np.array(1, 2, 3, 4, 5));
            np.diagonal(longVal).Should().BeOfValues(1, 2, 3);

            // A 2-D value is raveled first.
            var twoD = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(twoD, np.array(1, 2, 3, 4).reshape(2, 2));
            np.diagonal(twoD).Should().BeOfValues(1, 2, 3);
        }

        [TestMethod]
        public void FillDiagonal_EmptyValueIsASilentNoOp()
        {
            var a = np.full(new Shape(3, 3), 9, NPTypeCode.Int32);
            np.fill_diagonal(a, np.array(new int[0]));
            a.Should().BeOfValues(9, 9, 9, 9, 9, 9, 9, 9, 9);
        }

        [TestMethod]
        public void FillDiagonal_WrapOnlyAffectsTallMatrices()
        {
            var tallNoWrap = np.zeros(new Shape(5, 3), NPTypeCode.Int32);
            np.fill_diagonal(tallNoWrap, 5);
            tallNoWrap.Should().BeOfValues(5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0);

            var tallWrap = np.zeros(new Shape(5, 3), NPTypeCode.Int32);
            np.fill_diagonal(tallWrap, 5, true);
            tallWrap.Should().BeOfValues(5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0);

            // Wide matrices are unaffected by wrap.
            var wideNoWrap = np.zeros(new Shape(3, 5), NPTypeCode.Int32);
            np.fill_diagonal(wideNoWrap, 5);
            var wideWrap = np.zeros(new Shape(3, 5), NPTypeCode.Int32);
            np.fill_diagonal(wideWrap, 5, true);
            wideWrap.Should().BeOfValues(5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0);
            wideNoWrap.Should().BeOfValues(5, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 5, 0, 0);
        }

        [TestMethod]
        public void FillDiagonal_WrapSpansMultipleBlocks()
        {
            var a = np.zeros(new Shape(7, 3), NPTypeCode.Int32);
            np.fill_diagonal(a, 9, true);
            a.Should().BeOfValues(9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9, 0, 0, 0, 9);

            var tiled = np.zeros(new Shape(7, 3), NPTypeCode.Int32);
            np.fill_diagonal(tiled, np.array(1, 2, 3, 4), true);
            tiled.Should().BeOfValues(1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 2);
        }

        [TestMethod]
        public void FillDiagonal_NDimensional_RequiresEqualDims()
        {
            var cube = np.zeros(new Shape(2, 2, 2), NPTypeCode.Int32);
            np.fill_diagonal(cube, 5);
            cube.Should().BeOfValues(5, 0, 0, 0, 0, 0, 0, 5);

            var hyper = np.zeros(new Shape(3, 3, 3, 3), NPTypeCode.Int32);
            np.fill_diagonal(hyper, 7);
            np.count_nonzero(hyper).Should().Be(3);

            new Action(() => np.fill_diagonal(np.zeros(new Shape(2, 3, 4), NPTypeCode.Int32), 5))
                .Should().Throw<ArgumentException>()
                .WithMessage("All dimensions of input must be of equal length");
        }

        [TestMethod]
        public void FillDiagonal_Errors_VerbatimText()
        {
            new Action(() => np.fill_diagonal(np.zeros(new Shape(3), NPTypeCode.Int32), 5))
                .Should().Throw<ArgumentException>().WithMessage("array must be at least 2-d");
            new Action(() => np.fill_diagonal(NDArray.Scalar(1), 5))
                .Should().Throw<ArgumentException>().WithMessage("array must be at least 2-d");

            var broadcast = np.broadcast_to(np.zeros(new Shape(3), NPTypeCode.Double), new Shape(3, 3));
            new Action(() => np.fill_diagonal(broadcast, 1))
                .Should().Throw<ArgumentException>().WithMessage("underlying array is read-only");
        }

        [TestMethod]
        public void FillDiagonal_EmptyShapesAreNoOps()
        {
            np.fill_diagonal(np.zeros(new Shape(0, 0), NPTypeCode.Int32), 5);
            np.fill_diagonal(np.zeros(new Shape(3, 0), NPTypeCode.Int32), 5);
            np.fill_diagonal(np.zeros(new Shape(0, 3), NPTypeCode.Int32), 5);
        }

        [TestMethod]
        public void FillDiagonal_WritesThroughNonContiguousViews()
        {
            // Column-stepped view
            var stepBase = np.zeros(new Shape(4, 6), NPTypeCode.Int32);
            np.fill_diagonal(stepBase[":, ::2"], 7);
            stepBase.Should().BeOfValues(7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0,
                                         0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0);

            // Transposed view
            var tBase = np.zeros(new Shape(5, 3), NPTypeCode.Int32);
            np.fill_diagonal(tBase.T, 3);
            tBase.Should().BeOfValues(3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0);

            // Negative-stride view
            var revBase = np.zeros(new Shape(3, 4), NPTypeCode.Int32);
            np.fill_diagonal(revBase[":, ::-1"], 4);
            revBase.Should().BeOfValues(0, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0);

            // Offset sub-view
            var subBase = np.zeros(new Shape(5, 5), NPTypeCode.Int32);
            np.fill_diagonal(subBase["1:4, 1:4"], 6);
            subBase.Should().BeOfValues(0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 6, 0, 0,
                                        0, 0, 0, 6, 0, 0, 0, 0, 0, 0);

            // F-contiguous
            var f = np.asfortranarray(np.zeros(new Shape(3, 4), NPTypeCode.Int32));
            np.fill_diagonal(f, 8);
            f.Should().BeOfValues(8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0);
        }

        [TestMethod]
        public void FillDiagonal_CastsValueToArrayDtype()
        {
            // NumPy truncates 2.7 -> 2 when the target is integral.
            var ints = np.zeros(new Shape(3, 3), NPTypeCode.Int32);
            np.fill_diagonal(ints, 2.7);
            np.diagonal(ints).Should().BeOfValues(2, 2, 2);

            var floats = np.zeros(new Shape(3, 3), NPTypeCode.Double);
            np.fill_diagonal(floats, double.NaN);
            double.IsNaN(floats.GetValue<double>(1, 1)).Should().BeTrue();

            var bools = np.zeros(new Shape(3, 3), NPTypeCode.Boolean);
            np.fill_diagonal(bools, true);
            bools.GetValue<bool>(2, 2).Should().BeTrue();
            bools.GetValue<bool>(0, 1).Should().BeFalse();
        }
    }
}
