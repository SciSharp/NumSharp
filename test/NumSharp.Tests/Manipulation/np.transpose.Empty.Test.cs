namespace NumSharp.Tests.Manipulation
{
    /// <summary>
    ///     Transposing an EMPTY array returns a VIEW of it, as NumPy's <c>PyArray_Transpose</c> does for every input:
    ///     <c>owndata</c> False and the source's writeability kept. Every expectation is NumPy 2.4.2's answer for the same call,
    ///     probed on 2026-10-05.
    /// </summary>
    /// <remarks>
    ///     Before the fix, <c>Default.Transpose</c> answered an empty input with a freshly allocated array: it owned its data
    ///     and was writeable even when the source was a read-only broadcast. Found through <c>Generator.permutation</c> of an
    ///     empty array along axis 1, whose NumPy result is a transpose view. (NumPy 2.x reports all-zero strides for any empty
    ///     array, and NumSharp's empty strides differ library-wide, so raw strides are not compared here.)
    /// </remarks>
    [TestClass]
    public class np_transpose_empty_Test
    {
        /// <summary><c>np.zeros((0, 2)).T</c> is a <c>(2, 0)</c> view; non-empty transposes were views already.</summary>
        [TestMethod]
        public void T_OfEmpty_IsAView()
        {
            using var e = np.zeros(new Shape(0L, 2L));
            using var t = e.T;
            t.shape.Should().Equal(new long[] { 2, 0 });
            t.flags.owndata.Should().BeFalse("NumPy's transpose is always a view");
            t.flags.writeable.Should().BeTrue();
            t.flags.c_contiguous.Should().BeTrue("an empty array is both C- and F-contiguous");
            t.flags.f_contiguous.Should().BeTrue();
        }

        /// <summary>
        ///     <c>np.transpose</c> with explicit axes, <c>np.swapaxes</c> and <c>np.moveaxis</c> of empty 2-D and 3-D arrays are
        ///     views with the permuted dimensions.
        /// </summary>
        [TestMethod]
        public void EveryTransposeFamilyCall_OfEmpty_IsAView()
        {
            using var e3 = np.zeros(new Shape(2L, 0L, 3L));
            using var tr = np.transpose(e3, new[] { 1, 2, 0 });
            tr.shape.Should().Equal(new long[] { 0, 3, 2 });
            tr.flags.owndata.Should().BeFalse();

            using var sw = np.swapaxes(e3, 0, 2);
            sw.shape.Should().Equal(new long[] { 3, 0, 2 });
            sw.flags.owndata.Should().BeFalse();

            using var e2 = np.zeros(new Shape(3L, 0L));
            using var mv = np.moveaxis(e2, 0, 1);
            mv.shape.Should().Equal(new long[] { 0, 3 });
            mv.flags.owndata.Should().BeFalse();
        }

        /// <summary>
        ///     A read-only empty source stays read-only through the transpose (<c>np.broadcast_to(np.zeros(0), (3, 0)).T</c> is
        ///     not writeable in NumPy); the former fresh allocation handed back a writeable array.
        /// </summary>
        [TestMethod]
        public void T_OfReadOnlyEmpty_StaysReadOnly()
        {
            using var z = np.zeros(new Shape(0L));
            using var b = np.broadcast_to(z, new Shape(3L, 0L));
            using var t = b.T;
            t.shape.Should().Equal(new long[] { 0, 3 });
            t.flags.owndata.Should().BeFalse();
            t.flags.writeable.Should().BeFalse("a transpose keeps its source's writeability");
        }
    }
}
