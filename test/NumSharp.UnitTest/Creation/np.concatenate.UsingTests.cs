using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Guards the <c>using</c>-scoped intermediates introduced into <see cref="np.concatenate"/>
    /// (the <c>dstSlice</c> view inside the general loop, plus the ravel'd workArrays
    /// when <c>axis=null</c>).
    /// </summary>
    /// <remarks>
    /// Two <c>TightLoop_DoesNotLeakWorkingSet</c> tests used to close this class. Both were removed
    /// — they measured process RSS, not concatenate. See <see cref="LeakGuards"/>.
    /// </remarks>
    [TestClass]
    public class np_concatenate_using_test : TestClass
    {
        // --------------------------- correctness ---------------------------

        /// <summary>
        /// Exercises the general path (NDIter.Copy). A transposed source is
        /// non-contiguous, which forces both fast paths (TryDirectMemcpyConcat,
        /// TryDirectCastConcat) to bail and the dstSlice loop to fire.
        /// </summary>
        [TestMethod]
        public void Concatenate_GeneralPath_TransposedSource_ProducesCorrectValues()
        {
            var a = np.arange(12).reshape(3, 4).astype(NPTypeCode.Int32);
            var b = np.arange(12, 24).reshape(3, 4).astype(NPTypeCode.Int32);

            // Transpose forces non-contig — fast paths must reject these.
            var aT = a.T;
            var bT = b.T;
            aT.Shape.IsContiguous.Should().BeFalse();
            bT.Shape.IsContiguous.Should().BeFalse();

            var c = np.concatenate(new[] { aT, bT }, axis: 1);

            c.shape.Should().ContainInOrder(4L, 6L);
            // Column 0 of aT == row 0 of a == [0, 4, 8].
            ((int)c[0, 0]).Should().Be(0);
            ((int)c[1, 0]).Should().Be(1);
            ((int)c[2, 0]).Should().Be(2);
            ((int)c[3, 0]).Should().Be(3);
            ((int)c[0, 3]).Should().Be(12);
            ((int)c[3, 5]).Should().Be(23);
        }

        /// <summary>
        /// axis=null path: ravel each input and concatenate. The fresh wrappers
        /// allocated in <c>disposableWorkArrays</c> are released in the finally;
        /// the caller's original arrays must remain untouched.
        /// </summary>
        [TestMethod]
        public void Concatenate_AxisNull_RavelsAndConcatenates_PreservingInputs()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 14).reshape(2, 4).astype(NPTypeCode.Int32);

            var c = np.concatenate(new[] { a, b }, axis: (int?)null);

            c.shape.Should().ContainInOrder(14L);
            for (int i = 0; i < 14; i++)
                ((int)c[i]).Should().Be(i);

            // Caller's arrays must still be alive and readable after the
            // concatenate returns (disposing ravel wrappers must not dispose
            // the inputs they alias).
            a.IsDisposed.Should().BeFalse();
            b.IsDisposed.Should().BeFalse();
            a.Storage.InternalArray.IsReleased.Should().BeFalse();
            b.Storage.InternalArray.IsReleased.Should().BeFalse();
            ((int)a[1, 2]).Should().Be(5);
            ((int)b[1, 3]).Should().Be(13);
        }

        /// <summary>
        /// Non-contig sources with cross-dtype: forces the general path AND
        /// makes dstSlice + NDIter.Copy do the casting work. Verifies the
        /// using on dstSlice doesn't cut the slice off mid-copy.
        /// </summary>
        [TestMethod]
        public void Concatenate_GeneralPath_CrossDtype_TransposedSource_CorrectValues()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 12).reshape(2, 3).astype(NPTypeCode.Double);

            var aT = a.T;
            var bT = b.T;

            var c = np.concatenate(new[] { aT, bT }, axis: 1);

            c.dtype.Should().Be(typeof(double));
            c.shape.Should().ContainInOrder(3L, 4L);
            // aT[0,0] == a[0,0] == 0; bT[0,0] == b[0,0] == 6.
            ((double)c[0, 0]).Should().Be(0.0);
            ((double)c[0, 2]).Should().Be(6.0);
            ((double)c[2, 3]).Should().Be(11.0);
        }

        // --------------------------- lifetime ---------------------------

        /// <summary>
        /// The general path's per-source <c>dstSlice</c> is a view INTO the freshly allocated
        /// result. Disposing it must not release the result's buffer.
        /// </summary>
        /// <remarks>
        /// This is the sharper of the two hazards: <c>dstSlice</c> is created and disposed once per
        /// source, so the last one to be released is the one that could take the whole output with
        /// it. The array would still be reachable and correctly shaped, just backed by freed pages.
        /// Transposed sources force the general path (both memcpy fast paths bail on non-contiguous
        /// input), and the input views must survive too.
        /// </remarks>
        [TestMethod]
        public void Concatenate_GeneralPath_ResultOutlivesItsDstSlices()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 12).reshape(2, 3).astype(NPTypeCode.Int32);
            var aT = a.T;
            var bT = b.T;

            var c = np.concatenate(new[] { aT, bT }, axis: 1);

            LeakGuards.StillUsable(c, " — the last dstSlice must not free the result it wrote into");
            LeakGuards.StillUsable(a, " — the sources must outlive the copy");
            LeakGuards.StillUsable(b);

            // Every element of the output is readable after the slices were released.
            c.shape.Should().ContainInOrder(3L, 4L);
            ((int)c[0, 0]).Should().Be(0);
            ((int)c[0, 2]).Should().Be(6);
            ((int)c[2, 3]).Should().Be(11);
        }

        /// <summary>
        /// The axis=null result must outlive the ravel'd work arrays it was built from, and stay
        /// readable once the inputs those wrappers aliased are disposed.
        /// </summary>
        /// <remarks>
        /// Note what this does and does not prove. It does NOT prove the output is a copy — a
        /// properly refcounted alias would survive this too. It proves the stronger thing worth
        /// guarding: nothing here is released EARLY. The ravel wrappers alias the inputs, so a
        /// missing ARC ref anywhere in that chain frees the buffer while the output still points
        /// at it, and the read-back below lands on freed pages.
        /// </remarks>
        [TestMethod]
        public void Concatenate_AxisNull_ResultSurvivesInputDispose()
        {
            var a = np.arange(6).reshape(2, 3).astype(NPTypeCode.Int32);
            var b = np.arange(6, 14).reshape(2, 4).astype(NPTypeCode.Int32);

            var c = np.concatenate(new[] { a, b }, axis: (int?)null);

            a.Dispose();
            b.Dispose();

            LeakGuards.StillUsable(c, " — the output must not be released with the ravels' sources");
            for (int i = 0; i < 14; i++)
                ((int)c[i]).Should().Be(i);
        }
    }
}
