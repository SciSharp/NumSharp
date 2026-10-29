using System;

namespace NumSharp.Tests.Creation
{
    /// <summary>
    ///     np.frombuffer's dtype NAMES are read by np.dtype, as NumPy's array_frombuffer reads its dtype argument with
    ///     <c>PyArray_DescrConverter</c>: the string-spelled overloads had a code table of their own, which accepted names
    ///     NumPy refuses and reported the rest in its own words. Every expected value and text was probed against NumPy 2.4.2
    ///     (the np-layer dtype plan's batch 6 replay found the divergences).
    /// </summary>
    [TestClass]
    public class np_frombuffer_DtypeNames_Test
    {
        /// <summary>
        ///     A name nothing parses is np.dtype's refusal with NumPy's text ("data type 'bogus' not understood"; NumPy raises
        ///     TypeError, NumSharp's np.dtype NotSupportedException); the table's own text was "dtype string 'bogus' is not
        ///     supported".
        /// </summary>
        [TestMethod]
        public void UnknownName_IsNpDtypesRefusal()
        {
            var bytes = new byte[8];
            ((Action)(() => np.frombuffer(bytes, "bogus"))).Should().Throw<NotSupportedException>()
                .WithMessage("data type 'bogus' not understood");
            ((Action)(() => np.frombuffer(np.zeros(new Shape(1)).data, "bogus"))).Should().Throw<NotSupportedException>()
                .WithMessage("data type 'bogus' not understood");
        }

        /// <summary>
        ///     The empty name is refused (NumPy: "data type '' not understood"); the table read it as float64. A null name is
        ///     NumPy's None, the default float64.
        /// </summary>
        [TestMethod]
        public void EmptyName_IsRefused_NullName_IsTheDefault()
        {
            var bytes = BitConverter.GetBytes(1.5);
            ((Action)(() => np.frombuffer(bytes, ""))).Should().Throw<NotSupportedException>()
                .WithMessage("data type '' not understood");
            using var r = np.frombuffer(bytes, (string)null);
            r.dtype.Should().Be(np.float64);
            CollectionAssert.AreEqual(new[] { 1.5 }, r.ToArray<double>());
        }

        /// <summary>
        ///     '!' is no byte-order prefix in NumPy 2.4.2 ("data type '!i4' not understood"); the table read it as big-endian.
        /// </summary>
        [TestMethod]
        public void BangPrefix_IsNotAByteOrder()
        {
            ((Action)(() => np.frombuffer(new byte[4], "!i4"))).Should().Throw<NotSupportedException>()
                .WithMessage("data type '!i4' not understood");
        }

        /// <summary>The names NumPy reads keep their meaning: byte-order prefixes, one-letter codes, NumPy names, strings.</summary>
        [TestMethod]
        public void KnownNames_KeepTheirMeaning()
        {
            var bytes = new byte[] { 1, 2, 3, 4 };
            using var be = np.frombuffer(bytes, ">u2");                 // numpy: [258, 772]
            CollectionAssert.AreEqual(new ushort[] { 258, 772 }, be.ToArray<ushort>());
            using var le = np.frombuffer(bytes, "<u2");                 // numpy: [513, 1027]
            CollectionAssert.AreEqual(new ushort[] { 513, 1027 }, le.ToArray<ushort>());
            using var b = np.frombuffer(bytes, "b");                    // numpy: int8 [1, 2, 3, 4]
            b.dtype.Should().Be(np.int8);
            using var s = np.frombuffer(bytes, "S2");                   // numpy: [b'\x01\x02', b'\x03\x04']
            s.shape.Should().Equal(2L);
            s.dtype.Should().Be(np.dtype("S2"));
            ((Action)(() => np.frombuffer(bytes, "T"))).Should().Throw<ValueError>()
                .WithMessage("cannot create an OBJECT array from memory buffer");
        }
    }
}
