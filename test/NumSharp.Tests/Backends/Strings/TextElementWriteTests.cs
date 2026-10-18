using System;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     The <c>U</c> element writers every store of a C# value goes through — <c>np.array</c>, setitem, fills:
    ///     <see cref="NDStringCodec.WriteUcs4"/> (NumPy's <c>UNICODE_setitem</c> for a <c>str</c>) and
    ///     <see cref="NDStringCodec.WriteUcs4FromAscii"/> (for <c>bytes</c>). Their widening copies run a vector at a time
    ///     with an overlapping last step, so the tests compare every element width and value length around the 8- and
    ///     16-unit steps with a reference written from NumPy's rules (truncation by code points, NUL padding), and write
    ///     elements placed against inaccessible pages.
    /// </summary>
    [TestClass]
    public unsafe class TextElementWriteTests
    {
        /// <summary>
        ///     The reference <c>UNICODE_setitem</c> of a <see cref="string"/>: its code points (a valid surrogate pair is one, a
        ///     lone surrogate one of its own value) truncated to <paramref name="chars"/>, then zero units.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="chars">The element width in units.</param>
        /// <returns>The element's units.</returns>
        private static uint[] ReferenceUcs4(string value, int chars)
        {
            var units = new uint[chars];
            int w = 0;
            for (int i = 0; i < value.Length && w < chars; i++)
            {
                char c = value[i];
                if (char.IsHighSurrogate(c) && i + 1 < value.Length && char.IsLowSurrogate(value[i + 1]))
                {
                    units[w++] = (uint)char.ConvertToUtf32(c, value[i + 1]);
                    i++;
                }
                else
                {
                    units[w++] = c;
                }
            }
            return units;
        }

        /// <summary>The widths of the tests: every width to 70, then a few wide ones.</summary>
        private static IEnumerable<int> Widths()
        {
            for (int i = 1; i <= 70; i++)
                yield return i;
            foreach (int w in new[] { 127, 128, 129, 255, 256, 257, 500 })
                yield return w;
        }

        /// <summary>A text of <paramref name="len"/> chars: BMP characters (NULs and U+FFFF among them), with
        ///     <paramref name="surrogates"/> also surrogate pairs and lone surrogates.</summary>
        /// <param name="rng">The random source.</param>
        /// <param name="len">The length in chars (a pair can make it one longer).</param>
        /// <param name="surrogates">Whether surrogate units occur.</param>
        /// <returns>The text.</returns>
        private static string Text(System.Random rng, int len, bool surrogates)
        {
            var sb = new StringBuilder();
            while (sb.Length < len)
            {
                int r = rng.Next(16);
                if (surrogates && r == 0)
                    sb.Append(char.ConvertFromUtf32(rng.Next(0x10000, 0x110000)));
                else if (surrogates && r == 1)
                    sb.Append((char)rng.Next(0xD800, 0xE000));
                else if (r == 2)
                    sb.Append('\0');
                else if (r == 3)
                    sb.Append('￿');
                else
                {
                    char c = (char)rng.Next(1, 0x10000);
                    sb.Append(char.IsSurrogate(c) ? 'x' : c);
                }
            }
            return sb.ToString();
        }

        /// <summary>
        ///     <see cref="NDStringCodec.WriteUcs4"/> writes the reference units at every width and value length — shorter
        ///     than the element (padded), as long, and longer (truncated) — for BMP-only values (the vector widening copy) and
        ///     values with surrogates (the pairing loop), over a destination that held other units before.
        /// </summary>
        [TestMethod]
        public void WriteUcs4_EqualsTheReference_AtEveryWidthAndLength()
        {
            var rng = new System.Random(7121);
            foreach (int w in Widths())
                foreach (bool surrogates in new[] { false, true })
                    for (int len = 0; len <= w + 20; len += w > 70 ? 9 : 1)
                    {
                        string value = Text(rng, len, surrogates);
                        var units = new uint[w];
                        units.AsSpan().Fill(0xAAAAu);
                        fixed (uint* d = units)
                            NDStringCodec.WriteUcs4(value, d, w);
                        units.Should().Equal(ReferenceUcs4(value, w), $"width {w}, length {len}, surrogates {surrogates}");
                    }
        }

        /// <summary>
        ///     A store into an element placed against an inaccessible page writes nothing outside it: elements of every width,
        ///     at both ends of the fenced region, with values as long as the element and longer (the copy's last step ends on
        ///     the element's last unit) and with short values around the 8- and 16-char steps in a wide element (the copy's
        ///     overlapping last step must start inside the value: at the region's start, one unit before the element faults).
        /// </summary>
        [TestMethod]
        public void WriteUcs4_WritesOnlyItsElement()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            using var fence = new FencedMemory(pages: 1);
            var rng = new System.Random(7122);
            foreach (int w in Widths())
                foreach (int len in new[] { w, w + 5, 0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33 })
                    foreach (bool atEnd in new[] { true, false })
                    {
                        if (len > w + 5)
                            continue;
                        string value = Text(rng, len, surrogates: false);
                        uint* d = (uint*)fence.Place(w * 4L, atEnd);
                        NDStringCodec.WriteUcs4(value, d, w);
                        new ReadOnlySpan<uint>(d, w).ToArray().Should().Equal(ReferenceUcs4(value, w), $"width {w}, length {len}, at end {atEnd}");
                    }
        }

        /// <summary>
        ///     <see cref="NDStringCodec.WriteUcs4FromAscii"/> widens ASCII bytes one unit per byte at every width and length
        ///     (padded, as long, truncated), reading a source and writing an element that both lie against inaccessible pages,
        ///     and still raises CPython's decode error for a non-ASCII byte past the kept width.
        /// </summary>
        [TestMethod]
        public void WriteUcs4FromAscii_WidensEveryByte()
        {
            var rng = new System.Random(7123);
            FencedMemory src = FencedMemory.IsSupported ? new FencedMemory(pages: 1) : null;
            FencedMemory dst = FencedMemory.IsSupported ? new FencedMemory(pages: 1) : null;
            try
            {
                foreach (int w in Widths())
                    for (int len = 0; len <= w + 20; len += w > 70 ? 9 : 1)
                    {
                        var bytes = new byte[len];
                        for (int i = 0; i < len; i++)
                            bytes[i] = (byte)rng.Next(0, 128);
                        var want = new uint[w];
                        for (int i = 0; i < System.Math.Min(len, w); i++)
                            want[i] = bytes[i];
                        var units = new uint[w];
                        units.AsSpan().Fill(0xAAAAu);
                        fixed (uint* d = units)
                            NDStringCodec.WriteUcs4FromAscii(bytes, d, w);
                        units.Should().Equal(want, $"width {w}, length {len}");
                        if (src != null && len > 0)
                            foreach (bool atEnd in new[] { true, false })
                            {
                                byte* s = src.Place(len, atEnd);
                                bytes.AsSpan().CopyTo(new Span<byte>(s, len));
                                uint* f = (uint*)dst.Place(w * 4L, atEnd);
                                NDStringCodec.WriteUcs4FromAscii(new ReadOnlySpan<byte>(s, len), f, w);
                                new ReadOnlySpan<uint>(f, w).ToArray().Should().Equal(want, $"fenced width {w}, length {len}, at end {atEnd}");
                            }
                    }
                var bad = new byte[40];
                bad.AsSpan().Fill((byte)'a');
                bad[35] = 0xE9;
                Action store = () =>
                {
                    var units = new uint[8];
                    fixed (uint* d = units)
                        NDStringCodec.WriteUcs4FromAscii(bad, d, 8);
                };
                store.Should().Throw<UnicodeDecodeError>().WithMessage("'ascii' codec can't decode byte 0xe9 in position 35: ordinal not in range(128)");
            }
            finally
            {
                src?.Dispose();
                dst?.Dispose();
            }
        }
    }
}
