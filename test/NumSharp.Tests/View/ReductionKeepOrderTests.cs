using System;
using System.Numerics;

namespace NumSharp.Tests.View
{
    /// <summary>
    ///     Edge-case + rare-scenario regression coverage for KEEPORDER reduction output layout
    ///     (issue #610): NumPy allocates a reduction's output operand in "keep order" and the reduce
    ///     loop writes straight into it, so an F-contiguous input yields an F-contiguous result with
    ///     NO copy. NumSharp mirrors this via DefaultEngine.AllocateReductionResult +
    ///     Storage.ExpandDimension. Every expected value/flag below is NumPy 2.4.2's exact output.
    ///
    ///     The core wins (sum/mean/nansum on 3-D F-contig, Decimal, 6-D) live in
    ///     OrderSupportOpenBugsTests (Section 41); this class pins the ops and shapes those do NOT
    ///     cover — every value op, every nan op, the size-1-reduced-axis edge, keepdims at each axis
    ///     position, the collapse-to-both-contig cases, dtype coverage across executor paths, empty
    ///     inputs, the C-stays-C and argmax-stays-C regression guards, and the documented
    ///     transposed-input residual.
    /// </summary>
    [TestClass]
    public class ReductionKeepOrderTests
    {
        // arr[i,j,k] = i*12 + j*4 + k (logical C-order values), laid out F-contiguous.
        private static NDArray F3(Type dt) => np.arange(24).astype(dt).reshape(2, 3, 4).copy('F');
        private static NDArray C3(Type dt) => np.arange(24).astype(dt).reshape(2, 3, 4).copy('C');
        private static NDArray F3d() => F3(typeof(double));
        private static NDArray C3d() => C3(typeof(double));

        private static void AssertFContig(NDArray r) => r.Shape.IsFContiguous.Should().BeTrue("NumPy preserves F-contig through the reduction");
        private static void AssertBothContig(NDArray r)
        {
            r.Shape.IsContiguous.Should().BeTrue("NumPy reports a <=1-non-unit-dim result as C-contig");
            r.Shape.IsFContiguous.Should().BeTrue("...and F-contig (both flags set)");
        }

        // ============================================================================
        // Group 1: every VALUE reduction op preserves F on a 3-D F-contig input
        //          (sum/mean covered in OrderSupport; prod/min/max/std/var pinned here).
        // ============================================================================

        [TestMethod]
        public void Prod_FContig3D_Axis0_PreservesF()
        {
            var r = np.prod(F3d(), axis: 0);
            r.shape.Should().Equal(new long[] { 3, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Min_FContig3D_Axis1_KeepDims_PreservesF()
        {
            var r = np.min(F3d(), axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Max_FContig3D_Axis2_PreservesF()
        {
            var r = np.max(F3d(), axis: 2);
            r.shape.Should().Equal(new long[] { 2, 3 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Std_FContig3D_Axis0_PreservesF()
        {
            var r = np.std(F3d(), axis: 0);
            r.shape.Should().Equal(new long[] { 3, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Var_FContig3D_Axis1_KeepDims_PreservesF()
        {
            var r = np.var(F3d(), axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        // ============================================================================
        // Group 2: every NaN-aware reduction op preserves F (nansum in OrderSupport;
        //          nanprod/nanmin/nanmax/nanmean/nanstd/nanvar pinned here). Single NaN at
        //          the origin so no output slice is all-NaN (results stay finite).
        // ============================================================================

        private static NDArray F3nan()
        {
            var f = F3d();
            f.SetData(double.NaN, new int[] { 0, 0, 0 });
            return f;
        }

        [TestMethod]
        public void NanProd_FContig3D_Axis0_PreservesF() => AssertFContig(np.nanprod(F3nan(), axis: 0));

        [TestMethod]
        public void NanMin_FContig3D_Axis2_KeepDims_PreservesF()
        {
            var r = np.nanmin(F3nan(), axis: 2, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 3, 1 });
            AssertFContig(r);
        }

        [TestMethod]
        public void NanMax_FContig3D_Axis0_PreservesF() => AssertFContig(np.nanmax(F3nan(), axis: 0));

        [TestMethod]
        public void NanMean_FContig3D_Axis1_KeepDims_PreservesF()
        {
            var r = np.nanmean(F3nan(), axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void NanStd_FContig3D_Axis0_PreservesF() => AssertFContig(np.nanstd(F3nan(), axis: 0));

        [TestMethod]
        public void NanVar_FContig3D_Axis2_KeepDims_PreservesF() => AssertFContig(np.nanvar(F3nan(), axis: 2, keepdims: true));

        [TestMethod]
        public void NanMean_Complex_FContig3D_Axis1_PreservesF()
        {
            // The complex nan np-layer path (SetComplex by coordinate) must land F too.
            var f = F3(typeof(Complex));
            var r = np.nanmean(f, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 4 });
            AssertFContig(r);
        }

        // ============================================================================
        // Group 3: C-contig input stays C — regression guard against over-applying F.
        // ============================================================================

        [TestMethod]
        public void Sum_CContig3D_Axis0_KeepDims_StaysC()
        {
            var r = np.sum(C3d(), axis: 0, keepdims: true);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void Std_CContig3D_Axis1_StaysC()
        {
            var r = np.std(C3d(), axis: 1);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void NanMean_CContig3D_Axis1_KeepDims_StaysC()
        {
            var c = C3d();
            c.SetData(double.NaN, new int[] { 0, 0, 0 });
            var r = np.nanmean(c, axis: 1, keepdims: true);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        // ============================================================================
        // Group 4: RARE — reducing a SIZE-1 axis of an F-contig array. NumPy still keeps
        //          F (the result has >=2 non-unit dims). Exercises HandleTrivialAxisReduction
        //          / AllocateReductionZeros, the last gap the KEEPORDER fix closed.
        // ============================================================================

        [TestMethod]
        public void Sum_FContig_Size1ReducedAxis_PreservesF()
        {
            // F(2,1,4) reduce axis 1 (size 1) -> (2,4), values unchanged (arr[i,0,k]=i*4+k).
            var f = np.arange(8).astype(typeof(double)).reshape(2, 1, 4).copy('F');
            var r = np.sum(f, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 4 });
            AssertFContig(r);
            ((double)r[0, 0]).Should().Be(0);
            ((double)r[1, 3]).Should().Be(7);
        }

        [TestMethod]
        public void Sum_FContig_Size1ReducedAxis_KeepDims_PreservesF()
        {
            var f = np.arange(8).astype(typeof(double)).reshape(2, 1, 4).copy('F');
            var r = np.sum(f, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Std_FContig_Size1ReducedAxis_PreservesF()
        {
            // std of a single element is 0 — the AllocateReductionZeros path.
            var f = np.arange(8).astype(typeof(double)).reshape(2, 1, 4).copy('F');
            var r = np.std(f, axis: 1);
            r.shape.Should().Equal(new long[] { 2, 4 });
            AssertFContig(r);
            ((double)r[0, 0]).Should().Be(0);
            ((double)r[1, 3]).Should().Be(0);
        }

        [TestMethod]
        public void Var_FContig_Size1ReducedAxis_KeepDims_PreservesF()
        {
            var f = np.arange(8).astype(typeof(double)).reshape(2, 1, 4).copy('F');
            var r = np.var(f, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        // ============================================================================
        // Group 5: keepdims at EACH axis position — the ExpandDimension re-insert must
        //          preserve F whether the reduced axis is first, middle, or last (the last
        //          case appends the size-1 axis).
        // ============================================================================

        [TestMethod]
        public void Sum_FContig3D_KeepDims_Axis0_PreservesF()
        {
            var r = np.sum(F3d(), axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 3, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Sum_FContig3D_KeepDims_Axis1_PreservesF()
        {
            var r = np.sum(F3d(), axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 1, 4 });
            AssertFContig(r);
        }

        [TestMethod]
        public void Sum_FContig3D_KeepDims_Axis2_PreservesF()
        {
            // Last-axis keepdims: ExpandDimension appends the size-1 axis; must stay F.
            var r = np.sum(F3d(), axis: 2, keepdims: true);
            r.shape.Should().Equal(new long[] { 2, 3, 1 });
            AssertFContig(r);
        }

        // ============================================================================
        // Group 6: results that collapse to <=1 non-unit dim are BOTH C- and F-contig
        //          (NumPy's flag rule), even from a strictly-F input.
        // ============================================================================

        [TestMethod]
        public void Sum_FContig_LeadingUnitDim_Axis1_KeepDims_IsBoth()
        {
            // F(1,3,4) reduce axis1 kd -> (1,1,4): one non-unit dim -> both contig.
            var f = np.arange(12).astype(typeof(double)).reshape(1, 3, 4).copy('F');
            var r = np.sum(f, axis: 1, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 1, 4 });
            AssertBothContig(r);
        }

        [TestMethod]
        public void Sum_FContig_TrailingUnitDim_Axis0_KeepDims_IsBoth()
        {
            // F(3,4,1) reduce axis0 kd -> (1,4,1): one non-unit dim -> both contig.
            var f = np.arange(12).astype(typeof(double)).reshape(3, 4, 1).copy('F');
            var r = np.sum(f, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 4, 1 });
            AssertBothContig(r);
        }

        [TestMethod]
        public void Sum_FContig_MiddleUnitDim_Axis0_KeepDims_IsBoth()
        {
            // F(3,1,4) reduce axis0 kd -> (1,1,4): one non-unit dim -> both contig.
            var f = np.arange(12).astype(typeof(double)).reshape(3, 1, 4).copy('F');
            var r = np.sum(f, axis: 0, keepdims: true);
            r.shape.Should().Equal(new long[] { 1, 1, 4 });
            AssertBothContig(r);
        }

        // ============================================================================
        // Group 7: a 1-D result is trivially both C- and F-contig.
        // ============================================================================

        [TestMethod]
        public void Sum_FContig2D_Axis0_OneDResult_IsBoth()
        {
            // F(3,4) reduce axis0 -> (4,): 1-D -> both contig, values [12,15,18,21].
            var f = np.arange(12).astype(typeof(double)).reshape(3, 4).copy('F');
            var r = np.sum(f, axis: 0);
            r.shape.Should().Equal(new long[] { 4 });
            AssertBothContig(r);
            ((double)r[0]).Should().Be(12);
            ((double)r[3]).Should().Be(21);
        }

        // ============================================================================
        // Group 8: dtype coverage — F preservation across the distinct executor paths
        //          (int widening -> Direct, float/double/Complex/Decimal -> NDIter, Half ->
        //          float32-shadow NDIter). Also pins the NEP50 sum output dtype.
        // ============================================================================

        [TestMethod]
        public void Sum_FContig_Int32_WidensToInt64_PreservesF()
        {
            var r = np.sum(F3(typeof(int)), axis: 0);
            AssertFContig(r);
            r.dtype.Should().Be(typeof(long)); // NEP50: sum(int32) -> int64, still F
        }

        [TestMethod]
        public void Sum_FContig_Single_PreservesF()
        {
            var r = np.sum(F3(typeof(float)), axis: 0);
            AssertFContig(r);
            r.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void Sum_FContig_Half_PreservesF()
        {
            var r = np.sum(F3(typeof(Half)), axis: 0);
            AssertFContig(r);
            r.dtype.Should().Be(typeof(Half));
        }

        [TestMethod]
        public void Sum_FContig_Complex_PreservesF()
        {
            var r = np.sum(F3(typeof(Complex)), axis: 0);
            AssertFContig(r);
            r.dtype.Should().Be(typeof(Complex));
        }

        [TestMethod]
        public void Sum_FContig_Decimal_PreservesF()
        {
            var r = np.sum(F3(typeof(decimal)), axis: 0);
            AssertFContig(r);
            r.dtype.Should().Be(typeof(decimal));
        }

        // ============================================================================
        // Group 9: values are correct despite the F layout — the kernels write logical
        //          values through the result's strides, so an F result equals the C result
        //          element-for-element, and matches NumPy's exact numbers.
        // ============================================================================

        [TestMethod]
        public void FResult_EqualsCResult_AcrossValueOps()
        {
            foreach (var op in new Func<NDArray, NDArray>[]
                     {
                         a => np.sum(a, axis: 1, keepdims: true),
                         a => np.prod(a, axis: 0),
                         a => np.min(a, axis: 2),
                         a => np.max(a, axis: 1),
                         a => np.mean(a, axis: 0, keepdims: true),
                         a => np.std(a, axis: 1),
                         a => np.var(a, axis: 2, keepdims: true),
                     })
            {
                var rF = op(F3d());
                var rC = op(C3d());
                np.array_equal(rF, rC).Should().BeTrue("F-layout result must hold the same logical values as the C result");
            }
        }

        [TestMethod]
        public void FResult_EqualsCResult_AcrossNanOps()
        {
            foreach (var op in new Func<NDArray, NDArray>[]
                     {
                         a => np.nansum(a, axis: 1, keepdims: true),
                         a => np.nanprod(a, axis: 0),
                         a => np.nanmin(a, axis: 2),
                         a => np.nanmax(a, axis: 1),
                         a => np.nanmean(a, axis: 0, keepdims: true),
                         a => np.nanstd(a, axis: 1),
                         a => np.nanvar(a, axis: 2, keepdims: true),
                     })
            {
                var rF = op(F3nan());
                var c = C3d(); c.SetData(double.NaN, new int[] { 0, 0, 0 });
                var rC = op(c);
                np.array_equal(rF, rC).Should().BeTrue("F-layout NaN-op result must match the C result logically");
            }
        }

        [TestMethod]
        public void Sum_FContig3D_Axis0_ExactValues_MatchNumPy()
        {
            // NumPy: result[j,k] = 12 + 2*(j*4+k)
            var r = np.sum(F3d(), axis: 0);
            ((double)r[0, 0]).Should().Be(12);
            ((double)r[0, 1]).Should().Be(14);
            ((double)r[2, 3]).Should().Be(34);
        }

        // ============================================================================
        // Group 10: empty F-contig reductions — any zero dim makes both flags true.
        // ============================================================================

        [TestMethod]
        public void Sum_EmptyFContig_Axis1_IsBoth()
        {
            var e = np.zeros(new Shape(new long[] { 0, 3 }, 'F'), typeof(double));
            var r = np.sum(e, axis: 1);
            r.shape.Should().Equal(new long[] { 0 });
            AssertBothContig(r);
        }

        // ============================================================================
        // Group 11: argmax/argmin are NOT affected — NumPy allocates their index output in
        //           C-order regardless of input layout (regression guard for the exclusion).
        // ============================================================================

        [TestMethod]
        public void ArgMax_FContig3D_Axis0_StaysC()
        {
            var r = np.argmax(F3d(), axis: 0);
            r.shape.Should().Equal(new long[] { 3, 4 });
            r.Shape.IsContiguous.Should().BeTrue("NumPy argmax output is C-order regardless of input layout");
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        [TestMethod]
        public void ArgMin_FContig3D_Axis1_StaysC()
        {
            var r = np.argmin(F3d(), axis: 1);
            r.Shape.IsContiguous.Should().BeTrue();
            r.Shape.IsFContiguous.Should().BeFalse();
        }

        // ============================================================================
        // Group 12: documented residual — a genuinely transposed (neither-C-nor-F) input.
        //           NumPy's full stride-sort KEEPORDER can pick F; NumSharp's C/F-only 'K'
        //           model (shared with cumsum / np.copy(order='K')) yields C. The VALUES are
        //           byte-correct either way, which is what actually matters (and all the
        //           oracle checks, since it compares ascontiguousarray on both sides).
        // ============================================================================

        [TestMethod]
        [Misaligned]
        public void Sum_TransposedInput_ValuesCorrect_LayoutIsCNotF()
        {
            // t[a,b,c] = C3[b,a,c] = b*12 + a*4 + c ; sum over axis2 -> 48*b + 16*a + 6.
            var t = np.transpose(C3d(), new int[] { 1, 0, 2 }); // (3,2,4), neither C nor F
            t.Shape.IsContiguous.Should().BeFalse();
            t.Shape.IsFContiguous.Should().BeFalse();

            var r = np.sum(t, axis: 2);
            r.shape.Should().Equal(new long[] { 3, 2 });
            // Values match NumPy exactly.
            ((double)r[0, 0]).Should().Be(6);
            ((double)r[2, 1]).Should().Be(86);
            // Layout: NumSharp resolves 'K' to C for a non-C/non-F input; NumPy 2.4.2 reports F here.
            r.Shape.IsContiguous.Should().BeTrue("NumSharp's C/F-only KEEPORDER yields C for a transposed input (NumPy: F)");
        }

        // ============================================================================
        // Group 13: std/var with ddof != 0 — the ddof post-pass scales the F buffer in place.
        // ============================================================================

        [TestMethod]
        public void Std_FContig3D_Axis0_Ddof1_PreservesF()
        {
            var r = np.std(F3d(), axis: 0, ddof: 1);
            AssertFContig(r);
            np.array_equal(r, np.std(C3d(), axis: 0, ddof: 1)).Should().BeTrue();
        }

        [TestMethod]
        public void Var_FContig3D_Axis1_Ddof1_KeepDims_PreservesF()
        {
            var r = np.var(F3d(), axis: 1, ddof: 1, keepdims: true);
            AssertFContig(r);
            np.array_equal(r, np.var(C3d(), axis: 1, ddof: 1, keepdims: true)).Should().BeTrue();
        }
    }
}
