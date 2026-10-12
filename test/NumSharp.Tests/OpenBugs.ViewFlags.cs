using System;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the array OBJECT a call returns: view vs copy, flags, and offset views read wrong
    // =====================================================================
    //
    // Found 2026-10-07 by the array-object oracle (test/oracle/gen_viewflags_oracle.py, replayed by
    // NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests.ViewFlags.cs): four tiers compare the object each API returns with
    // NumPy 2.4.2's — its values, its C / F / OWNDATA / WRITEABLE / ALIGNED flags, its byte strides, and whether it is
    // a view of, a copy of, or the very object of each operand — for returned arrays, out= arguments, shape arguments
    // and secondary array arguments. Every divergence the tiers found is carved into
    // NumSharp.Tests.Oracle/Fuzz/corpus/gaps/*_gaps.jsonl under a gap id (the GAPS table of the generator), replayed by
    // the *_KnownGaps tests there. This file reproduces the ones with consequences through PUBLIC calls:
    //
    //   * Offset views (VALUES). np.split / np.unstack / np.array_split children are C-contiguous views whose Shape
    //     carries a non-zero offset (a basic slice re-seats the storage address instead). Several kernels ignore that
    //     offset for a C-contiguous operand and read the parent from its start: np.searchsorted (v, sorter, the method
    //     too), np.where (condition, x, y), np.ldexp (x2), np.nansum / np.nanprod with axis=None, np.repeat and
    //     ndarray.repeat, np.unique and the set routines that call it, and a[:, idx] (which throws). The results are
    //     silently wrong.
    //   * Read-only arrays. A copy of a read-only array stays read-only, a basic index of a read-only broadcast view
    //     through the object[] indexer is writeable, and np.broadcast_arrays hands a read-only broadcast operand back
    //     writeable — a write through it changes every row the broadcast repeats.
    //   * Order and layout. ravel(order='K') of a permuted array returns the C-order values as a copy, astype(...,
    //     copy=False, order='C') of a stepped view returns the view itself, np.zeros(view.Shape) keeps the view's F
    //     layout, np.reshape(-1) of a broadcast view throws, and np.linalg.multi_dot writes into an out= that np.dot
    //     itself refuses.
    //
    // FIXED 2026-10-12, when branch strings was rebased onto journey4: np.repeat / ndarray.repeat and np.unique (with the
    // set routines) read an offset view right (VF39, VF40), ravel(order='K') of a permuted array walks memory (VF05), a
    // basic index of a read-only broadcast stays read-only (VF23), np.reshape(-1) of a broadcast view copies (VF24); the
    // tiers' replay found VF03, VF11, VF18, VF25, VF28, VF33, VF35 fixed too, and parts of VF01, VF08, VF12, VF26 and
    // VF32. The pins of the fixed gaps stay below as regression tests, untagged, each summary opening with "FIXED".
    //
    // Every expected value below is NumPy 2.4.2's answer for the same call. EXPECTED (when fixed): each test passes;
    // remove its [OpenBugs], move it next to its API's tests, and drop the gap's carve rule from the generator
    // (_gap_of) so the case returns to the gate.
    // =====================================================================
    public partial class OpenBugs
    {
        // ----- Offset views: silent wrong values -----

        /// <summary>
        ///     <c>np.searchsorted(a, v)</c> with <c>v</c> an <c>np.split</c> child (offset 3): NumPy returns <c>[0, 1, 5]</c>
        ///     for <c>v = [0.5, 2, 9]</c>; NumSharp searches the parent's first elements <c>[9, 9, 9]</c> (gap VA01).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VA01")]
        public void ViewFlags_Searchsorted_VAtAnOffset_ShouldReadTheView()
        {
            using var vbase = np.array(new double[] { 9, 9, 9, 0.5, 2, 9, 3, 4, 1 });
            var pieces = np.split(vbase, 3);
            using var a = np.array(new double[] { 1, 2, 3, 5, 8 });
            using var r = np.searchsorted(a, pieces[1], "left");

            r.ToArray<long>().Should().Equal(new long[] { 0, 1, 5 });
        }

        /// <summary>
        ///     <c>np.searchsorted(a, v, sorter=s)</c> with the sorter an <c>np.split</c> child (offset 2): NumPy returns
        ///     <c>[2, 4]</c> for <c>a = [5, 1, 3, 2, 8]</c>, <c>v = [2, 7.5]</c>, <c>s = [1, 3, 2, 0, 4]</c> (gap VA01).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VA01")]
        public void ViewFlags_Searchsorted_SorterAtAnOffset_ShouldReadTheView()
        {
            using var sbase = np.array(new long[] { 7, 7, 1, 3, 2, 0, 4 });
            var pieces = np.split(sbase, new[] { 2 });
            using var a = np.array(new double[] { 5, 1, 3, 2, 8 });
            using var v = np.array(new double[] { 2, 7.5 });
            using var r = np.searchsorted(a, v, "right", pieces[1]);

            r.ToArray<long>().Should().Equal(new long[] { 2, 4 });
        }

        /// <summary>
        ///     <c>np.where(cond, x, y)</c> with the condition an <c>np.split</c> child (offset 4, <c>[False, True, True,
        ///     False]</c>): NumPy returns <c>[-0, 1, 2, -3]</c> (gap VA03).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VA03")]
        public void ViewFlags_Where_ConditionAtAnOffset_ShouldReadTheView()
        {
            using var cbase = np.array(new[] { false, true, false, true, false, true, true, false, false, true, true, true });
            var pieces = np.split(cbase, 3);
            using var x = np.arange(4.0);
            using var y = -x;
            using var r = np.where(pieces[1], x, y);

            r.ToArray<double>().Should().Equal(new[] { -0.0, 1.0, 2.0, -3.0 });
        }

        /// <summary>
        ///     <c>np.where(cond, x, 0)</c> and <c>np.where(cond, 0, x)</c> with <c>x</c> an <c>np.split</c> child (offset 2,
        ///     <c>[1, 2, 3, 4]</c>): NumPy returns <c>[1, 0, 3, 0]</c> and <c>[0, 2, 0, 4]</c>; NumSharp reads the parent's
        ///     <c>[9, 9, …]</c> (gap VA03).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VA03")]
        public void ViewFlags_Where_XOrYAtAnOffset_ShouldReadTheView()
        {
            using var xbase = np.array(new double[] { 9, 9, 1, 2, 3, 4 });
            var pieces = np.split(xbase, new[] { 2 });
            using var cond = np.array(new[] { true, false, true, false });
            using var zeros = np.zeros(new Shape(4));
            using var rx = np.where(cond, pieces[1], zeros);
            using var ry = np.where(cond, zeros, pieces[1]);

            rx.ToArray<double>().Should().Equal(new[] { 1.0, 0, 3, 0 });
            ry.ToArray<double>().Should().Equal(new[] { 0.0, 2, 0, 4 });
        }

        /// <summary>
        ///     <c>np.ldexp(ones(4), e)</c> with the exponents an <c>np.split</c> child (offset 3, <c>[-1, 0, 1, -1]</c>):
        ///     NumPy returns <c>[0.5, 1, 2, 0.5]</c>; NumSharp scales by the parent's <c>[5, 5, 5, -1]</c> (gap VA05).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VA05")]
        public void ViewFlags_Ldexp_ExponentAtAnOffset_ShouldReadTheView()
        {
            using var ebase = np.array(new long[] { 5, 5, 5, -1, 0, 1, -1 });
            var pieces = np.split(ebase, new[] { 3 });
            using var ones = np.ones(new Shape(4));
            using var r = np.ldexp(ones, pieces[1]);

            r.ToArray<double>().Should().Equal(new[] { 0.5, 1, 2, 0.5 });
        }

        /// <summary>
        ///     <c>o[:, [0, 1]]</c> of an <c>np.split</c> child (rows 1-4 of <c>arange(20).reshape(5, 4)</c>): NumPy returns
        ///     <c>[[4, 5], [8, 9], [12, 13], [16, 17]]</c>; NumSharp throws <see cref="IndexOutOfRangeException"/> (gap VF22).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF22")]
        public void ViewFlags_SliceThenIndex_OnAnOffsetView_ShouldSelectTheColumns()
        {
            using var parent = np.arange(20).reshape(5, 4);
            var pieces = np.split(parent, new[] { 1 });
            using var idx = np.array(new long[] { 0, 1 });
            using var r = pieces[1][Slice.All, idx];

            r.shape.Should().Equal(new long[] { 4, 2 });
            r.ToArray<long>().Should().Equal(new long[] { 4, 5, 8, 9, 12, 13, 16, 17 });
        }

        /// <summary>
        ///     <c>np.nansum</c> / <c>np.nanprod</c> of the 0-d <c>np.unstack(arange(6.))[5]</c> (offset 5): NumPy returns
        ///     5.0 for both; NumSharp reads the parent's first element (gap VF30).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF30")]
        public void ViewFlags_NanSumNanProd_OfAZeroDimOffsetView_ShouldReadTheView()
        {
            using var parent = np.arange(6.0);
            var pieces = np.unstack(parent);
            using var s = np.nansum(pieces[5]);
            using var p = np.nanprod(pieces[5]);

            s.GetDouble().Should().Be(5.0);
            p.GetDouble().Should().Be(5.0);
        }

        /// <summary>
        ///     <c>np.nansum</c> of the 2-D <c>np.split(arange(20.), [4, 16])[1].reshape(3, 4)</c> (offset 4): NumPy returns
        ///     114.0 (4 + … + 15); <c>np.nanprod</c> of <c>np.split([9, 9, 1, 2, 3, 4], [2])[1]</c>: NumPy returns 24.0
        ///     (gap VF30).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF30")]
        public void ViewFlags_NanSumNanProd_FlatOfAnOffsetView_ShouldReadTheView()
        {
            using var parent = np.arange(20.0);
            var pieces = np.split(parent, new[] { 4, 16 });
            using var m = pieces[1].reshape(3, 4);
            using var s = np.nansum(m);
            using var pbase = np.array(new double[] { 9, 9, 1, 2, 3, 4 });
            var ppieces = np.split(pbase, new[] { 2 });
            using var p = np.nanprod(ppieces[1]);

            s.GetDouble().Should().Be(114.0);
            p.GetDouble().Should().Be(24.0);
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.repeat(v, 2)</c>, <c>v.repeat(2)</c> of <c>np.split(arange(6), [2])[1]</c> (offset 2, <c>[2, 3, 4, 5]</c>):
        ///     NumPy returns <c>[2, 2, 3, 3, 4, 4, 5, 5]</c>; and of the 0-d <c>np.unstack(arange(6))[5]</c>: <c>[5, 5]</c>
        ///     (gap VF39).
        /// </summary>
        [TestMethod]
        public void ViewFlags_Repeat_OfAnOffsetView_ShouldReadTheView()
        {
            using var parent = np.arange(6);
            var pieces = np.split(parent, new[] { 2 });
            var cells = np.unstack(parent);
            using var r1 = np.repeat(pieces[1], 2);
            using var r2 = pieces[1].repeat(2);
            using var r3 = np.repeat(cells[5], 2);

            r1.ToArray<long>().Should().Equal(new long[] { 2, 2, 3, 3, 4, 4, 5, 5 });
            r2.ToArray<long>().Should().Equal(new long[] { 2, 2, 3, 3, 4, 4, 5, 5 });
            r3.ToArray<long>().Should().Equal(new long[] { 5, 5 });
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.unique</c> of <c>np.split(big, 2)[1]</c> (offset 4, <c>[[0, 3], [7.5, 6]]</c>): NumPy returns
        ///     <c>[0, 3, 6, 7.5]</c>; NumSharp returns the parent's <c>[9]</c>. The set routines that call it on an
        ///     <c>ar2</c> inherit it: NumPy's <c>intersect1d</c> / <c>setdiff1d</c> / <c>setxor1d</c> with
        ///     <c>ar1 = [0, 1.5, 3, 4.5, 6]</c> are <c>[0, 3, 6]</c>, <c>[1.5, 4.5]</c>, <c>[1.5, 4.5, 7.5]</c> (gap VF40).
        /// </summary>
        [TestMethod]
        public void ViewFlags_UniqueAndSetRoutines_OfAnOffsetView_ShouldReadTheView()
        {
            using var big = np.array(new double[] { 9, 9, 9, 9, 0, 3, 7.5, 6 }).reshape(4, 2);
            var pieces = np.split(big, 2);
            using var ar1 = np.array(new double[] { 0, 1.5, 3, 4.5, 6 });

            np.unique(pieces[1]).values.ToArray<double>().Should().Equal(new[] { 0, 3, 6, 7.5 });
            np.intersect1d(ar1, pieces[1]).ToArray<double>().Should().Equal(new[] { 0.0, 3, 6 });
            np.setdiff1d(ar1, pieces[1]).ToArray<double>().Should().Equal(new[] { 1.5, 4.5 });
            np.setxor1d(ar1, pieces[1]).ToArray<double>().Should().Equal(new[] { 1.5, 4.5, 7.5 });
        }

        // ----- Read-only arrays -----

        /// <summary>
        ///     A copy of a read-only array is writeable in NumPy: <c>np.array(ro, copy=True)</c> and <c>np.positive(ro)</c>
        ///     of a <c>setflags(write=False)</c> owner both report <c>writeable True</c>; NumSharp's stay read-only, so a
        ///     write to the fresh copy throws (gap VF14).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF14")]
        public void ViewFlags_CopiesOfAReadOnlyArray_ShouldBeWriteable()
        {
            using var ro = np.arange(12).reshape(3, 4);
            ro.setflags(write: false);
            using var copy = np.array(ro, true);
            using var pos = np.positive(ro);

            copy.flags.writeable.Should().BeTrue();
            pos.flags.writeable.Should().BeTrue();
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>b[1]</c> of the read-only broadcast <c>np.broadcast_to(arange(4), (3, 4))</c>, spelled through the
        ///     <c>object[]</c> indexer: NumPy's view is read-only; NumSharp's is writeable, so a write through it changes the
        ///     one row every broadcast row shares (gap VF23).
        /// </summary>
        [TestMethod]
        public void ViewFlags_IndexOfAReadOnlyBroadcast_ThroughTheObjectIndexer_ShouldStayReadOnly()
        {
            using var row = np.arange(4);
            using var b = np.broadcast_to(row, new Shape(3, 4));
            using var r = b[new object[] { 1 }];

            r.flags.writeable.Should().BeFalse();
        }

        /// <summary>
        ///     <c>np.broadcast_arrays(a, b)</c> with <c>b</c> already the read-only broadcast of the result shape: NumPy
        ///     returns <c>b</c> itself, read-only; NumSharp returns a new WRITEABLE view of it (gap VF17).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF17")]
        public void ViewFlags_BroadcastArrays_OfAReadOnlyBroadcast_ShouldStayReadOnly()
        {
            using var a = np.arange(12).reshape(3, 4);
            using var row = np.arange(4);
            using var b = np.broadcast_to(row, new Shape(3, 4));
            var res = np.broadcast_arrays(new[] { a, b });

            res[1].flags.writeable.Should().BeFalse();
        }

        // ----- Order and layout -----

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.ravel(p, order='K')</c> of the permuted <c>arange(24).reshape(3, 4, 2).transpose(2, 0, 1)</c>: NumPy
        ///     walks memory, returns <c>[0, 1, …, 23]</c> as a VIEW of <c>p</c>; NumSharp returns the C-order values as a
        ///     copy (gap VF05).
        /// </summary>
        [TestMethod]
        public void ViewFlags_RavelOrderK_OfAPermutedArray_ShouldWalkMemory()
        {
            using var parent = np.arange(24).reshape(3, 4, 2);
            using var p = parent.transpose(new[] { 2, 0, 1 });
            using var r = np.ravel(p, 'K');

            r.ToArray<long>()[..8].Should().Equal(new long[] { 0, 1, 2, 3, 4, 5, 6, 7 });
            np.shares_memory(r, p).Should().BeTrue();
        }

        /// <summary>
        ///     <c>s.astype(int64, copy=False, order='C')</c> of the stepped <c>arange(12)[::2]</c>: NumPy copies to meet the
        ///     order (a new C-contiguous array, strides (8,)); NumSharp returns <c>s</c> itself (gap VF16).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VF16")]
        public void ViewFlags_AstypeCopyFalse_WithAnOrderTheViewLacks_ShouldCopy()
        {
            using var parent = np.arange(12);
            using var s = parent["::2"];
            using var r = s.astype(np.int64, false, 'C');

            ReferenceEquals(r, s).Should().BeFalse();
            r.flags.c_contiguous.Should().BeTrue();
        }

        /// <summary>
        ///     <c>np.zeros(t.Shape)</c> for the transposed <c>zeros((3, 2)).T</c>: NumPy's <c>np.zeros(t.shape)</c> is
        ///     C-ordered (strides (24, 8)); NumSharp allocates the view's F layout (gap VS01).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VS01")]
        public void ViewFlags_ZerosOfAViewShape_ShouldBeCOrdered()
        {
            using var t0 = np.zeros(new Shape(3, 2));
            using var t = t0.T;
            using var r = np.zeros(t.Shape);

            r.flags.c_contiguous.Should().BeTrue();
            r.strides.Should().Equal(new long[] { 24, 8 });
        }

        /// <summary>
        ///     FIXED by branch strings (rebased onto journey4 on 2026-10-12): NumSharp gives NumPy's answer now, and this test pins it.
        ///     Before the fix: <c>np.reshape(b, -1)</c> of the broadcast <c>np.broadcast_to(arange(4), (3, 4))</c>: NumPy copies and returns
        ///     <c>[0, 1, 2, 3] * 3</c>; NumSharp throws <see cref="NotSupportedException"/> (gap VF24).
        /// </summary>
        [TestMethod]
        public void ViewFlags_ReshapeMinusOne_OfABroadcast_ShouldCopy()
        {
            using var row = np.arange(4);
            using var b = np.broadcast_to(row, new Shape(3, 4));
            using var r = np.reshape(b, new long[] { -1 });

            r.ToArray<long>().Should().Equal(new long[] { 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 });
        }

        /// <summary>
        ///     <c>np.linalg.multi_dot([A, B, A], out=o)</c> with an F-ordered <c>o</c>: NumPy's final <c>np.dot</c> raises
        ///     <c>ValueError: output array is not acceptable (must have the right datatype, number of dimensions, and be a
        ///     C-Array)</c>, as NumSharp's own <c>np.dot</c> does; NumSharp's multi_dot writes into <c>o</c> (gap VO01).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "viewflags-VO01")]
        public void ViewFlags_MultiDot_IntoAnFOrderedOut_ShouldRefuseIt()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            using var b = np.ascontiguousarray(a.T);
            using var zeros = np.zeros(new Shape(3, 4));
            using var o = np.asfortranarray(zeros);

            Action act = () => np.linalg.multi_dot(new[] { a, b, a }, o);

            act.Should().Throw<Exception>().WithMessage("*output array is not acceptable*");
        }
    }
}
