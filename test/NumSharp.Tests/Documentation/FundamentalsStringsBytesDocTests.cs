using System;
using System.IO;
using System.Linq;
using System.Text;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/strings-and-bytes.md</c> (the "Working with arrays of
    ///     strings and bytes" fundamentals article). Each test mirrors one documented snippet and
    ///     asserts the behaviour the page claims: NumPy's fixed-width <c>U</c> / <c>S</c> dtypes store, convert,
    ///     move and print like NumPy 2.4.2, and computing on text is refused (NumPy's text) or deferred to the
    ///     next phase.
    /// </summary>
    [TestClass]
    public class FundamentalsStringsBytesDocTests
    {
        private static NDArray Names() => np.array(new[] { "cat", "dog", "bird" });

        // ── The two string dtypes ─────────────────────────────────────────────────────────────────

        [TestMethod]
        public void TwoStringDtypes_WidthAndItemSize()
        {
            var names = Names();
            names.dtype.ToString(true).Should().Be("dtype('<U4')", "the longest value sets the width");
            names.dtype.itemsize.Should().Be(16, "4 characters x 4 bytes");
            np.dtype("S5").str.Should().Be("|S5");
            np.dtype("U5").name.Should().Be("str160", "NumPy's name counts bits");
            np.dtype("U5").itemsize.Should().Be(20);
            np.dtype("S5").itemsize.Should().Be(5);
            np.array(new[] { "\U0001F600" }).dtype.str.Should().Be("<U1", "an emoji is one UCS-4 code point");
        }

        [TestMethod]
        public void TwoStringDtypes_RawBytes()
        {
            np.array(new[] { "ab" }).tobytes().Should().Equal(0x61, 0, 0, 0, 0x62, 0, 0, 0);
            np.array(new[] { "ab" }).astype("S3").tobytes().Should().Equal(0x61, 0x62, 0);
        }

        // ── Creating string arrays ────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Creating_FromCSharpValues()
        {
            np.array(new[] { "cat", "dog" }).ToString(true).Should().Be("array(['cat', 'dog'], dtype='<U3')");
            np.array(new[] { "cat", "dog" }, np.dtype("S5")).ToString(true).Should().Be("array([b'cat', b'dog'], dtype='|S5')");
            np.zeros(new Shape(2), np.dtype("U3")).ToString(true).Should().Be("array(['', ''], dtype='<U3')");
            np.full(new Shape(2), "ab").ToString(true).Should().Be("array(['ab', 'ab'], dtype='<U2')");
            NDArray.Scalar("abc").ToString(true).Should().Be("array('abc', dtype='<U3')");
            np.frombuffer(Encoding.ASCII.GetBytes("abcdef"), "S3").ToString(true)
              .Should().Be("array([b'abc', b'def'], dtype='|S3')");
        }

        [TestMethod]
        public void Creating_BytesNeedObjectArray()
        {
            np.array(new object[] { Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("cde") }).ToString(true)
              .Should().Be("array([b'ab', b'cde'], dtype='|S3')");
            np.array(new object[] { Encoding.ASCII.GetBytes("hi") }, np.dtype("S2")).ToString(true)
              .Should().Be("array([b'hi'], dtype='|S2')");

            // The footgun: a jagged byte[][] is numeric data in C#.
            var jagged = np.array(new[] { Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("cd") });
            jagged.typecode.Should().Be(NPTypeCode.Byte);
            jagged.Should().BeShaped(2, 2);
        }

        // ── Reading and writing elements ──────────────────────────────────────────────────────────

        [TestMethod]
        public void Elements_ReadAndWrite()
        {
            var names = Names();
            names.item(1).Should().Be("dog");
            ((byte[])names.astype("S4").item(1)).Should().Equal((byte)'d', (byte)'o', (byte)'g');
            NDArray.AsStringArray(names).Should().Equal("cat", "dog", "bird");
            names[1].ToString(true).Should().Be("array('dog', dtype='<U4')");
            ((System.Collections.IEnumerable)names.tolist()).Cast<object>().Should().Equal("cat", "dog", "bird");

            names[1] = "zebras";
            names.ToString(true).Should().Be("array(['cat', 'zebr', 'bird'], dtype='<U4')", "a longer value is cut to the width");
        }

        [TestMethod]
        public void CharVectorApi_RefusesStringArrays()
        {
            var names = np.array(new[,] { { "ab", "c" } });
            Action get = () => names.GetString(0);
            get.Should().Throw<InvalidOperationException>().WithMessage("GetString works on a vector of chars*");
            Action set = () => names.SetString("x", 0);
            set.Should().Throw<InvalidOperationException>().WithMessage("SetString works on a vector of chars*");

            // The Char dtype keeps its char-vector API.
            var chars = NDArray.FromString("abc");
            chars.GetString().Should().Be("abc");
        }

        // ── Dtype integration ─────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Dtypes_PromotionAndCasting()
        {
            np.promote_types(np.dtype("S3"), np.dtype("U2")).ToString(true).Should().Be("dtype('<U3')");
            np.result_type(new[] { "U3", "S7" }).ToString(true).Should().Be("dtype('<U7')");
            np.can_cast(np.dtype("S3"), np.dtype("U3"), "safe").Should().BeTrue();
            np.can_cast(np.dtype("U3"), np.dtype("S3"), "same_kind").Should().BeFalse();
            np.can_cast(np.dtype("U5"), np.dtype("U3"), "safe").Should().BeFalse();
        }

        // ── Moving data ───────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Moving_ViewsCopiesAndCasts()
        {
            np.concatenate(new[] { np.array(new[] { "a" }), np.array(new[] { "long" }) }).ToString(true)
              .Should().Be("array(['a', 'long'], dtype='<U4')");
            Names().astype("U2").ToString(true).Should().Be("array(['ca', 'do', 'bi'], dtype='<U2')");
            np.array(new[,] { { "a", "b" }, { "c", "d" } }).T.ToString(true)
              .Should().Be("array([['a', 'c'],\n       ['b', 'd']], dtype='<U1')");
        }

        [TestMethod]
        public void Moving_NonAsciiToBytes_Throws()
        {
            Action act = () => np.array(new[] { "é" }).astype("S1");
            act.Should().Throw<UnicodeEncodeError>()
               .WithMessage("'ascii' codec can't encode character '\\xe9' in position 0: ordinal not in range(128)");
        }

        // ── Printing ──────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Printing_StrAndRepr()
        {
            var names = Names();
            names.ToString().Should().Be("['cat' 'dog' 'bird']");
            names.ToString(true).Should().Be("array(['cat', 'dog', 'bird'], dtype='<U4')");
            np.array2string(np.array(new[] { "a'b", "c\"d" })).Should().Be("[\"a'b\" 'c\"d']");
        }

        // ── Computing on text (the next phase) ────────────────────────────────────────────────────

        [TestMethod]
        public void Computing_NumPyRefusals()
        {
            var names = Names();
            Action sum = () => np.sum(names);
            sum.Should().Throw<TypeError>().WithMessage(
                "the resolved dtypes are not compatible with add.reduce. Resolved (dtype('<U4'), dtype('<U4'), dtype('<U8'))");
            Action sqrt = () => np.sqrt(names);
            sqrt.Should().Throw<TypeError>().WithMessage("ufunc 'sqrt' not supported for the input types, *");
        }

        [TestMethod]
        public void Computing_NextPhase()
        {
            var names = Names();
            Action compare = () => { _ = names == names; };
            compare.Should().Throw<StringComputationNotSupportedException>();
            Action concat = () => np.add(names, names);
            concat.Should().Throw<StringComputationNotSupportedException>();
            Action sort = () => np.sort(names);
            sort.Should().Throw<StringComputationNotSupportedException>();
            Action request = () => np.add(np.array(new[] { 1.5 }), np.array(new[] { 1.5 }), dtype: "U");
            request.Should().Throw<StringComputationNotSupportedException>();
            // StringComputationNotSupportedException is a NotSupportedException.
            sort.Should().Throw<NotSupportedException>();

            // The .NET route until then.
            NDArray.AsStringArray(names).Select(s => s.ToUpper()).Should().Equal("CAT", "DOG", "BIRD");
        }

        [TestMethod]
        public void Computing_NeverEqualToNumbers()
        {
            var eq = Names() == np.array(new[] { 1, 2, 3 });
            eq.ToString().Should().Be("[False False False]");
        }

        [TestMethod]
        public void Computing_ChecksInNumPysOrder()
        {
            Action axis = () => np.sum(Names(), 9);
            axis.Should().Throw<AxisError>().WithMessage("axis 9 is out of bounds for array of dimension 1*");
            Action noLoop = () => np.sqrt(np.array(new[] { 1.5 }), dtype: "U");
            noLoop.Should().Throw<IncorrectTypeException>()
                  .WithMessage("No loop matching the specified signature and casting was found for ufunc sqrt");
        }

        // ── .npy files ────────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Npy_RoundTrip()
        {
            var stream = new MemoryStream();
            np.save(stream, Names());
            stream.Position = 0;
            np.load_npy(stream).ToString(true).Should().Be("array(['cat', 'dog', 'bird'], dtype='<U4')");
        }

        [TestMethod]
        public void Npy_CharWritesAsU1()
        {
            var stream = new MemoryStream();
            np.save(stream, np.array(new[] { 'a', 'b' }));
            stream.Position = 0;
            // The <U1 file reads back as NumPy's own <U1 string array (breaking: it used to read as Char).
            np.load_npy(stream).ToString(true).Should().Be("array(['a', 'b'], dtype='<U1')");
        }

        // ── The Char dtype ────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Char_IsTwoByteUtf16CodeUnit()
        {
            np.array(new[] { 'a', 'b', 'c' }).typecode.Should().Be(NPTypeCode.Char);
            InfoOf<char>.Size.Should().Be(2, "actual memory footprint is 2 bytes (UTF-16 code unit)");
        }

        [TestMethod]
        public void UnsupportedDtypes_StillThrow()
        {
            ((Action)(() => np.dtype("V7"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("O"))).Should().Throw<NotSupportedException>();
        }

        // ── Byte streams via np.frombuffer ────────────────────────────────────────────────────────

        [TestMethod]
        public void Bytes_ViaFrombuffer()
        {
            byte[] raw = { 1, 2, 3 };
            var asBytes = np.frombuffer(raw, np.uint8);
            asBytes.typecode.Should().Be(NPTypeCode.Byte);
            asBytes.ToArray<byte>().Should().Equal((byte)1, (byte)2, (byte)3);

            // reinterpret 16 bytes as 4 float32
            byte[] fbytes = new byte[16];
            Buffer.BlockCopy(new[] { 1f, 2f, 3f, 4f }, 0, fbytes, 0, 16);
            np.frombuffer(fbytes, np.float32).ToArray<float>().Should().Equal(1f, 2f, 3f, 4f);
        }
    }
}
