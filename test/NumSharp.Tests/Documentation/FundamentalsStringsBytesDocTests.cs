using System;
using System.Linq;
using NumSharp;
using NumSharp.Utilities;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    ///     Executable coverage for every code example in
    ///     <c>docs/website-src/docs/fundamentals/strings-and-bytes.md</c> (the "Working with arrays of
    ///     strings and bytes" fundamentals article). Each test mirrors one documented snippet and
    ///     asserts the behaviour the page claims — including that NumPy's string/bytes dtypes throw.
    /// </summary>
    [TestClass]
    public class FundamentalsStringsBytesDocTests
    {
        // ── The Char dtype ─────────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Char_IsTwoByteUtf16CodeUnit()
        {
            np.array(new[] { 'a', 'b', 'c' }).typecode.Should().Be(NPTypeCode.Char);
            InfoOf<char>.Size.Should().Be(2, "actual memory footprint is 2 bytes (UTF-16 code unit)");
        }

        // ── NumPy string/bytes dtypes are not supported ────────────────────────────────────────────

        [TestMethod]
        public void UnsupportedStringDtypes_Throw()
        {
            ((Action)(() => np.dtype("U5"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("S10"))).Should().Throw<NotSupportedException>();
            ((Action)(() => np.dtype("V7"))).Should().Throw<NotSupportedException>();
        }

        // ── What to use instead ────────────────────────────────────────────────────────────────────

        [TestMethod]
        public void Text_StaysDotNet_LengthsViaLinq()
        {
            string[] labels = { "cat", "dog", "bird" };
            np.array(labels.Select(s => s.Length).ToArray()).ToArray<int>().Should().Equal(3, 3, 4);
        }

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
