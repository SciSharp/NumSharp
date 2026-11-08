using System;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     <c>ndarray.getfield</c> / <c>setfield</c> and the in-place <c>ndarray.resize</c> on fixed-width string arrays,
    ///     against NumPy 2.4.2 (probed 2026-10-07). A string's item size need not divide the field's (an <c>S3</c> element
    ///     read as <c>int16</c>), so these views keep NumPy's BYTE strides, which NumSharp stores as whole elements: what
    ///     those can express is pinned bit for bit, what they cannot is a pinned refusal.
    /// </summary>
    [TestClass]
    public class GetFieldStringTests
    {
        /// <summary>The <c>[b'abc', b'def']</c> array the probes start from.</summary>
        /// <returns>A new <c>|S3</c> array.</returns>
        private static NDArray S3() => np.array(new[] { "abc", "def" }, dtype: "S3");

        /// <summary>The <c>['ab', 'cd']</c> array the probes start from.</summary>
        /// <returns>A new <c>&lt;U2</c> array.</returns>
        private static NDArray U2() => np.array(new[] { "ab", "cd" });

        /// <summary>
        ///     Guards byte-reinterpreting field views between string and numeric dtypes: values, NumPy's byte strides, and
        ///     a view that shares the array's memory (owndata false, writes go through).
        /// </summary>
        [TestMethod]
        public void GetField_ReadsTheFieldBytes()
        {
            var s3 = S3();
            using (var f = s3.getfield(np.dtype("S1"), 1))
            {
                f.ToString(true).Should().Be("array([b'b', b'e'], dtype='|S1')");
                f.strides.Should().Equal(3L);
                f.flags.owndata.Should().BeFalse();
                f.flags.c_contiguous.Should().BeFalse();
            }
            s3.getfield(np.uint8, 2).ToString(true).Should().Be("array([ 99, 102], dtype=uint8)");

            var u2 = U2();
            u2.getfield(np.dtype("S4"), 4).ToString(true).Should().Be("array([b'b', b'd'], dtype='|S4')");
            u2.getfield(np.dtype("U1"), 4).ToString(true).Should().Be("array(['b', 'd'], dtype='<U1')");
            u2.getfield(np.uint32, 4).ToString(true).Should().Be("array([ 98, 100], dtype=uint32)");

            var i4 = np.array(new[] { 0x64636261, 0x68676665 });
            using (var f = i4.getfield(np.dtype("S2"), 1))
            {
                f.ToString(true).Should().Be("array([b'bc', b'fg'], dtype='|S2')");
                f.strides.Should().Equal(4L);
            }
            np.array(new uint[] { 97, 98 }).getfield(np.dtype("U1"), 0).ToString(true).Should().Be("array(['a', 'b'], dtype='<U1')");
        }

        /// <summary>Guards the layouts: a transposed 2-D array, a 0-d array, a read-only broadcast, an empty array.</summary>
        [TestMethod]
        public void GetField_Layouts()
        {
            var grid = np.array(new[,] { { "abc", "def" }, { "ghi", "jkl" } }, dtype: "S3").T;
            using (var f = grid.getfield(np.dtype("S1"), 2))
            {
                f.ToString(true).Should().Be("array([[b'c', b'i'],\n       [b'f', b'l']], dtype='|S1')");
                f.strides.Should().Equal(3L, 6L);
            }

            np.array("abc", np.dtype("S3")).getfield(np.dtype("S2"), 1).ToString(true).Should().Be("array(b'bc', dtype='|S2')");

            var broadcast = np.broadcast_to(S3(), new Shape(2, 2));
            using (var f = broadcast.getfield(np.dtype("S1"), 0))
            {
                f.ToString(true).Should().Be("array([[b'a', b'd'],\n       [b'a', b'd']], dtype='|S1')");
                f.flags.writeable.Should().BeFalse("a field of a read-only array is read-only");
            }

            np.zeros(new Shape(0), np.dtype("S3")).getfield(np.dtype("S1"), 2).ToString(true).Should().Be("array([], dtype='|S1')");
        }

        /// <summary>Guards NumPy's three getfield ValueErrors on a string array, in its order.</summary>
        [TestMethod]
        public void GetField_ValueErrors_AreNumPys()
        {
            Action larger = () => S3().getfield(np.dtype("S4"), 0);
            larger.Should().Throw<ValueError>().WithMessage("new type is larger than original type");
            Action negative = () => S3().getfield(np.dtype("S1"), -1);
            negative.Should().Throw<ValueError>().WithMessage("offset is negative");
            Action past = () => S3().getfield(np.dtype("S2"), 2);
            past.Should().Throw<ValueError>().WithMessage("new type plus offset is larger than original type");
        }

        /// <summary>
        ///     Pins the views NumSharp cannot lay out: a byte stride that is not a whole number of field items (NumPy keeps
        ///     byte strides: <c>S3 → S2</c>, <c>S3 → int16</c>), and an unsized field (NumPy builds a zero-width <c>S0</c>
        ///     view). Each used to be silently wrong or a bare "type code" error; it is now a clear refusal.
        /// </summary>
        [TestMethod]
        [Misaligned]
        public void GetField_UnrepresentableViews_AreRefused()
        {
            Action splitS2 = () => S3().getfield(np.dtype("S2"), 1);
            splitS2.Should().Throw<NotSupportedException>().WithMessage("*byte stride 3*");
            Action splitInt16 = () => S3().getfield(np.int16, 0);
            splitInt16.Should().Throw<NotSupportedException>().WithMessage("*byte stride 3*");
            Action generic = () => S3().getfield<short>(0);
            generic.Should().Throw<NotSupportedException>().WithMessage("*byte stride 3*");
            Action unsized = () => S3().getfield(np.dtype("S"), 0);
            unsized.Should().Throw<NotSupportedException>().WithMessage("*zero-width*");
        }

        /// <summary>
        ///     Guards setfield into a string field: a <c>byte[]</c> is ONE bytes value (NumPy's <c>b'X'</c>), a string is
        ///     encoded, and the rest of each element is untouched.
        /// </summary>
        [TestMethod]
        public void SetField_WritesTheFieldOnly()
        {
            var a = S3();
            a.setfield(new[] { (byte)'X' }, np.dtype("S1"), 1);
            a.ToString(true).Should().Be("array([b'aXc', b'dXf'], dtype='|S3')");

            var b = S3();
            b.setfield("Z", np.dtype("S1"), 2);
            b.ToString(true).Should().Be("array([b'abZ', b'deZ'], dtype='|S3')");

            var u = U2();
            u.setfield("Q", np.dtype("U1"), 4);
            u.ToString(true).Should().Be("array(['aQ', 'cQ'], dtype='<U2')");
        }

        /// <summary>
        ///     Guards the in-place <c>ndarray.resize</c> of a string array: the buffer grows by the descriptor's width and the
        ///     new elements read <c>''</c> (NumPy zero-fills), a shrink keeps the prefix.
        /// </summary>
        [TestMethod]
        public void Resize_InPlace_GrowsWithEmptyStrings()
        {
            var u = np.array(new[] { "ab", "c", "dé" });
            u.resize(new Shape(5));
            u.ToString(true).Should().Be("array(['ab', 'c', 'dé', '', ''], dtype='<U2')");
            u.flags.owndata.Should().BeTrue();

            var s = np.array(new[] { "ab", "c", "def" }, dtype: "S3");
            s.resize(new Shape(2));
            s.ToString(true).Should().Be("array([b'ab', b'c'], dtype='|S3')");
        }
    }
}
