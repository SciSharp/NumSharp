using System;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The element readers every text read goes through — <c>tolist()</c>, <c>item()</c>, <c>AsStringArray</c>:
    ///     <see cref="NDStringCodec.ReadUcs4"/> (NumPy's <c>UNICODE_getitem</c>), <see cref="NDStringCodec.ReadBytes"/>
    ///     (<c>STRING_getitem</c>) and <see cref="NDUtf8.DecodeTrusted"/> (a <c>StringDType</c> cell's text). Each runs its
    ///     steps a vector at a time (a trim from the end, a range test, a narrowing or widening copy whose last step overlaps
    ///     the one before), so the tests compare every width and length around the vector steps with a scalar reference
    ///     written from NumPy's rules, and read elements placed against inaccessible pages.
    /// </summary>
    [TestClass]
    public unsafe class TextElementReadTests
    {
        /// <summary>
        ///     The reference <c>UNICODE_getitem</c>: trailing zero units dropped, each remaining unit one UTF-16 char, a code
        ///     point past U+FFFF its surrogate pair, a unit past U+10FFFF CPython's <see cref="SystemError"/>.
        /// </summary>
        /// <param name="units">The element's units.</param>
        /// <returns>The value.</returns>
        /// <exception cref="SystemError">A unit past U+10FFFF.</exception>
        private static string ReferenceUcs4(uint[] units)
        {
            int len = units.Length;
            while (len > 0 && units[len - 1] == 0)
                len--;
            var sb = new StringBuilder();
            for (int i = 0; i < len; i++)
            {
                uint u = units[i];
                if (u > 0x10FFFF)
                    throw new SystemError("invalid maximum character passed to PyUnicode_New");
                if (u > 0xFFFF)
                    sb.Append(char.ConvertFromUtf32((int)u));
                else
                    sb.Append((char)u);
            }
            return sb.ToString();
        }

        /// <summary>The widths of the tests: every width to 70 (the 8- and 16-unit steps and their overlaps), then a few wide ones.</summary>
        private static readonly int[] Widths = BuildWidths();

        /// <summary>Builds <see cref="Widths"/>.</summary>
        /// <returns>1..70, 127, 128, 129, 255, 256, 257, 500.</returns>
        private static int[] BuildWidths()
        {
            var w = new System.Collections.Generic.List<int>();
            for (int i = 1; i <= 70; i++)
                w.Add(i);
            w.AddRange(new[] { 127, 128, 129, 255, 256, 257, 500 });
            return w.ToArray();
        }

        /// <summary>
        ///     A text of <paramref name="len"/> units then zero padding to <paramref name="width"/>: random BMP units (lone
        ///     surrogates, NULs inside and U+FFFF included), or with <paramref name="astral"/> one code point in sixteen past
        ///     U+FFFF. The last text unit is never zero, so the text is exactly <paramref name="len"/> units.
        /// </summary>
        /// <param name="rng">The random source.</param>
        /// <param name="width">The element width in units.</param>
        /// <param name="len">The text length in units.</param>
        /// <param name="astral">Whether code points past U+FFFF occur.</param>
        /// <returns>The element's units.</returns>
        private static uint[] Units(System.Random rng, int width, int len, bool astral)
        {
            var u = new uint[width];
            for (int i = 0; i < len; i++)
            {
                int r = rng.Next(16);
                u[i] = astral && r == 0 ? (uint)rng.Next(0x10000, 0x110000)
                    : r == 1 ? 0u                              // a NUL inside the text
                    : r == 2 ? (uint)rng.Next(0xD800, 0xE000)  // a lone surrogate unit
                    : r == 3 ? 0xFFFFu
                    : (uint)rng.Next(1, 0x10000);
            }
            if (len > 0 && u[len - 1] == 0)
                u[len - 1] = 'x';
            return u;
        }

        /// <summary>
        ///     <see cref="NDStringCodec.ReadUcs4"/> equals the scalar reference at every width and text length: BMP texts take
        ///     the vector trim, range test and narrowing copy (8- and 16-unit steps, the overlapping last step), texts with a
        ///     code point past U+FFFF the counting decode.
        /// </summary>
        [TestMethod]
        public void ReadUcs4_EqualsTheReference_AtEveryWidthAndLength()
        {
            var rng = new System.Random(4071);
            foreach (int w in Widths)
                foreach (bool astral in new[] { false, true })
                    for (int len = 0; len <= w; len += w > 70 ? 7 : 1)
                    {
                        uint[] units = Units(rng, w, len, astral);
                        fixed (uint* p = units)
                            NDStringCodec.ReadUcs4(p, w).Should().Be(ReferenceUcs4(units), $"width {w}, length {len}, astral {astral}");
                    }
        }

        /// <summary>
        ///     The code-point bounds, at each position of a 20-unit element: U+FFFF stays one char, U+10000 and U+10FFFF become
        ///     their surrogate pairs, and a unit at U+110000 or above raises CPython's SystemError wherever it is in the text —
        ///     a test one past the first astral code point would narrow U+10000 to a NUL.
        /// </summary>
        [TestMethod]
        public void ReadUcs4_CodePointBounds()
        {
            foreach (int pos in new[] { 0, 1, 7, 8, 15, 16, 19 })
            {
                foreach (var (unit, expected) in new (uint, string)[]
                {
                    (0xFFFF, "￿"), (0x10000, "𐀀"), (0x10FFFF, "􏿿"), (0xD800, "\uD800"), (0xDFFF, "\uDFFF"),
                })
                {
                    var units = new uint[20];
                    for (int i = 0; i < 20; i++)
                        units[i] = 'a';
                    units[pos] = unit;
                    string want = new string('a', pos) + expected + new string('a', 19 - pos);
                    fixed (uint* p = units)
                        NDStringCodec.ReadUcs4(p, 20).Should().Be(want, $"U+{unit:X} at {pos}");
                }
                foreach (uint bad in new[] { 0x110000u, 0x7FFFFFFFu, 0xFFFFFFFFu })
                {
                    var units = new uint[20];
                    for (int i = 0; i < 10; i++)
                        units[i] = 'a';
                    units[pos] = bad;
                    Action read = () =>
                    {
                        fixed (uint* p = units)
                            NDStringCodec.ReadUcs4(p, 20);
                    };
                    read.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New", $"0x{bad:X} at {pos}");
                }
            }
        }

        /// <summary>
        ///     An all-zero element and an element whose zeros are all inside (a text that ends in NUL padding only) read as the
        ///     empty string and as the text with its inner NULs, as NumPy's trim from the end gives.
        /// </summary>
        [TestMethod]
        public void ReadUcs4_TrimsOnlyTrailingZeros()
        {
            foreach (int w in new[] { 1, 7, 8, 9, 16, 17, 33 })
            {
                var zeros = new uint[w];
                fixed (uint* p = zeros)
                    NDStringCodec.ReadUcs4(p, w).Should().BeEmpty($"width {w}");
                var inner = new uint[w];
                inner[0] = 'x';
                fixed (uint* p = inner)
                    NDStringCodec.ReadUcs4(p, w).Should().Be("x", $"width {w}");
                if (w >= 3)
                {
                    var mid = new uint[w];
                    mid[0] = 'a';
                    mid[2] = 'b';
                    fixed (uint* p = mid)
                        NDStringCodec.ReadUcs4(p, w).Should().Be("a\0b", $"width {w}");
                }
            }
        }

        /// <summary>
        ///     An element read against an inaccessible page reads nothing outside it: elements of every width placed so that
        ///     their last unit is the last accessible one and so that their first unit is the first accessible one, holding a
        ///     text that fills the whole width (no zero to stop a search early) and short texts around the 8- and 16-unit steps
        ///     (the narrowing's overlapping last step must start inside the text: at the region's start, one unit before the
        ///     element faults).
        /// </summary>
        [TestMethod]
        public void ReadUcs4_ReadsOnlyItsElement()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            using var fence = new FencedMemory(pages: 1);
            var rng = new System.Random(911);
            foreach (int w in Widths)
                foreach (int len in new[] { w, 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33 })
                    foreach (bool atEnd in new[] { true, false })
                        foreach (bool astral in new[] { false, true })
                        {
                            if (len > w)
                                continue;
                            uint[] units = Units(rng, w, len, astral);
                            uint* p = (uint*)fence.Place(w * 4L, atEnd);
                            units.AsSpan().CopyTo(new Span<uint>(p, w));
                            NDStringCodec.ReadUcs4(p, w).Should().Be(ReferenceUcs4(units), $"width {w}, length {len}, at end {atEnd}");
                        }
        }

        /// <summary>
        ///     <see cref="NDStringCodec.ReadBytes"/> equals NumPy's trim at every width and length (NULs inside kept, trailing
        ///     NULs dropped, any byte value), also against inaccessible pages.
        /// </summary>
        [TestMethod]
        public void ReadBytes_EqualsTheReference_AtEveryWidthAndLength()
        {
            var rng = new System.Random(5093);
            FencedMemory fence = FencedMemory.IsSupported ? new FencedMemory(pages: 1) : null;
            try
            {
                foreach (int w in Widths)
                    for (int len = 0; len <= w; len += w > 70 ? 5 : 1)
                    {
                        var bytes = new byte[w];
                        for (int i = 0; i < len; i++)
                            bytes[i] = rng.Next(8) == 0 ? (byte)0 : (byte)rng.Next(1, 256);
                        if (len > 0 && bytes[len - 1] == 0)
                            bytes[len - 1] = 0xFF;
                        byte[] want = bytes.AsSpan(0, len).ToArray();
                        fixed (byte* p = bytes)
                            NDStringCodec.ReadBytes(p, w).Should().Equal(want, $"width {w}, length {len}");
                        if (fence != null && len == w)
                            foreach (bool atEnd in new[] { true, false })
                            {
                                byte* f = fence.Place(w, atEnd);
                                bytes.AsSpan().CopyTo(new Span<byte>(f, w));
                                NDStringCodec.ReadBytes(f, w).Should().Equal(want, $"fenced width {w}, at end {atEnd}");
                            }
                    }
            }
            finally
            {
                fence?.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="NDUtf8.DecodeTrusted"/> equals the BCL's UTF-8 decode for ASCII texts of every length (the vector
        ///     widening path) and for texts with a non-ASCII character at a random position (the decoder path), also against
        ///     inaccessible pages.
        /// </summary>
        [TestMethod]
        public void DecodeTrusted_EqualsTheUtf8Decode()
        {
            var rng = new System.Random(6151);
            FencedMemory fence = FencedMemory.IsSupported ? new FencedMemory(pages: 1) : null;
            try
            {
                foreach (int len in new[] { 0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 255, 256, 500 })
                    foreach (string extra in new[] { null, "é", "中", "😀", "\0" })
                    {
                        var sb = new StringBuilder();
                        for (int i = 0; i < len; i++)
                            sb.Append((char)rng.Next(1, 128));
                        if (extra != null)
                            sb.Insert(len == 0 ? 0 : rng.Next(len), extra);
                        byte[] utf8 = Encoding.UTF8.GetBytes(sb.ToString());
                        string want = Encoding.UTF8.GetString(utf8);
                        NDUtf8.DecodeTrusted(utf8).Should().Be(want, $"length {len}, extra {extra}");
                        if (fence != null && utf8.Length > 0)
                            foreach (bool atEnd in new[] { true, false })
                            {
                                byte* f = fence.Place(utf8.Length, atEnd);
                                utf8.AsSpan().CopyTo(new Span<byte>(f, utf8.Length));
                                NDUtf8.DecodeTrusted(new ReadOnlySpan<byte>(f, utf8.Length)).Should().Be(want, $"fenced length {len}, extra {extra}");
                            }
                    }
            }
            finally
            {
                fence?.Dispose();
            }
        }
    }
}
