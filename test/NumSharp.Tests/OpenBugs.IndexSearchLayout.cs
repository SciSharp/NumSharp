using System;

namespace NumSharp.Tests
{
    // =====================================================================
    // OPEN BUGS — the layout of np.argwhere / np.flatnonzero / np.unravel_index results
    // =====================================================================
    //
    // Found 2026-10-07 by the string oracle's shape-decided section (an empty string array's index searches run the
    // numeric path on a bool stand-in, so they inherit its result layout). NumPy 2.4.2 builds both results as VIEWS:
    //   - np.flatnonzero(a) is np.nonzero(np.ravel(a))[0], a column of nonzero's (k, 1) index buffer: C-contiguous,
    //     flags.owndata False;
    //   - np.argwhere(a) is np.transpose(np.nonzero(a)): the (ndim, k) array np.asarray makes of the index tuple, viewed
    //     transposed — shape (k, ndim), strides (8, 8k) bytes, F-contiguous, flags.owndata False (an empty result has
    //     all-zero strides).
    //   - np.unravel_index(indices, dims) returns column VIEWS of one interleaved indices.shape + (len(dims),) int64
    //     buffer: strides len(dims) * 8 bytes, flags.owndata False.
    // NumSharp's kernels write a fresh C-contiguous (k,) / (k, ndim) array (one per coordinate for unravel_index) that owns
    // its data. The VALUES agree; only the ownership flag and the strides differ. The string oracle excuses the layout of
    // these three calls (StringOracleTests.LayoutExcuse) until this is fixed.
    // EXPECTED (when fixed): each test passes; remove its [OpenBugs] and the string oracle's excuse.
    // =====================================================================
    public partial class OpenBugs
    {
        /// <summary>
        ///     <c>np.flatnonzero(np.array([[0, 1, 2], [3, 0, 5]]))</c> is <c>[1, 2, 3, 5]</c>, a C-contiguous VIEW
        ///     (<c>flags.owndata</c> False, strides <c>(8,)</c>); NumSharp's result owns its data.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "index-search-layout")]
        public void Flatnonzero_ShouldBeAViewOfNonzerosBuffer()
        {
            using var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 0, 5 } });
            using var r = np.flatnonzero(a);
            r.ToArray<long>().Should().Equal(1L, 2L, 3L, 5L);
            r.flags.c_contiguous.Should().BeTrue();
            r.flags.owndata.Should().BeFalse("NumPy's flatnonzero is a column view of nonzero's index buffer");
        }

        /// <summary>
        ///     <c>np.argwhere(np.array([[0, 1, 2], [3, 0, 5]]))</c> is <c>[[0, 1], [0, 2], [1, 0], [1, 2]]</c> laid out as
        ///     NumPy's transposed view: strides <c>(8, 32)</c> bytes, F-contiguous, not C-contiguous, <c>flags.owndata</c>
        ///     False. NumSharp's (4, 2) result is C-contiguous (strides <c>(16, 8)</c>) and owns its data.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "index-search-layout")]
        public void Argwhere_ShouldBeNumPysTransposedView()
        {
            using var a = np.array(new long[,] { { 0, 1, 2 }, { 3, 0, 5 } });
            using var r = np.argwhere(a);
            r.ToArray<long>().Should().Equal(0L, 1L, 0L, 2L, 1L, 0L, 1L, 2L);
            r.strides.Should().Equal(new long[] { 8, 32 }, "NumPy's argwhere transposes the (ndim, k) index array");
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
            r.flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     <c>np.unravel_index(np.array([1, 5, 4]), (2, 3))</c> is <c>([0, 1, 1], [1, 2, 1])</c>, each a VIEW of one
        ///     interleaved <c>(3, 2)</c> int64 buffer (column <c>j</c> is <c>buffer[..., j]</c>): strides <c>(16,)</c> bytes,
        ///     not C-contiguous, <c>flags.owndata</c> False. NumSharp's unravel kernel writes each coordinate into its own
        ///     C-contiguous owning array (found 2026-10-07 by the string oracle's section M, which excuses the layout).
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "index-search-layout")]
        public void UnravelIndex_ShouldReturnViewsOfOneInterleavedBuffer()
        {
            using var idx = np.array(new long[] { 1, 5, 4 });
            var r = np.unravel_index(idx, new[] { 2, 3 });
            r[0].ToArray<long>().Should().Equal(0L, 1L, 1L);
            r[1].ToArray<long>().Should().Equal(1L, 2L, 1L);
            r[0].strides.Should().Equal(new long[] { 16 }, "NumPy's coordinates are columns of a (3, 2) buffer");
            r[0].flags.c_contiguous.Should().BeFalse();
            r[0].flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     The empty forms keep the same ownership: <c>np.flatnonzero(np.zeros(0))</c> and
        ///     <c>np.argwhere(np.zeros((0, 3)))</c> report <c>flags.owndata</c> False in NumPy 2.4.2 — and so do the empty
        ///     string arrays whose index searches NumSharp answers from their shape.
        /// </summary>
        [TestMethod, OpenBugs(IssueUrl = "index-search-layout")]
        public void EmptyIndexSearches_ShouldNotOwnTheirData()
        {
            using var f = np.zeros(new Shape(0L));
            using var flat = np.flatnonzero(f);
            flat.flags.owndata.Should().BeFalse("np.flatnonzero(np.zeros(0)).flags.owndata is False");
            using var e = np.zeros(new Shape(0L, 3L));
            using var where = np.argwhere(e);
            where.shape.Should().Equal(new long[] { 0, 2 }); // (k, ndim)
            where.flags.owndata.Should().BeFalse("np.argwhere(np.zeros((0, 3))).flags.owndata is False");
            using var text = np.zeros(new Shape(0L), "U2");
            using var textFlat = np.flatnonzero(text);
            textFlat.flags.owndata.Should().BeFalse("the shape-decided string answer runs the numeric path");
        }
    }
}
