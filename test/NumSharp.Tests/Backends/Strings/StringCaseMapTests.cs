using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NumPy's case maps <c>upper</c> / <c>lower</c> / <c>swapcase</c> / <c>capitalize</c> / <c>title</c> (strings phase 2,
    ///     P2.6 milestone 2): every element form against CPython's algorithm (<c>NDTextElements.CaseMapCodePoints</c> and the
    ///     bytes methods), the forms against inaccessible pages, the IL kernels against the managed ones, and the public
    ///     functions against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The string oracle (sections <c>strcase</c> / <c>tcase</c>) checks the public functions against NumPy over its
    ///     operands; these tests check what the operands do not reach: every <c>S</c> / <c>U</c> width across the one-chunk,
    ///     whole-chunk and overlapping-chunk forms, the escalation from the ASCII compares to the gather in the middle of a
    ///     call, each reason the gather path hands an element to the scalar loop (a unit past U+FFFF, a multi-code-point
    ///     mapping, U+03A3), every UTF-8 length of a <c>StringDType</c> text, and short cells whose bytes past the size are
    ///     junk.
    ///     </para>
    ///     <para>Every test is deterministic (fixed seeds) and touches no process-wide switch, so they run in parallel.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StringCaseMapTests
    {
        /// <summary>The maps, upper (0) to title (4).</summary>
        private static readonly int[] Maps =
        {
            NDTextElements.UpperMap, NDTextElements.LowerMap, NDTextElements.SwapcaseMap, NDTextElements.CapitalizeMap,
            NDTextElements.TitleMap,
        };

        /// <summary>
        ///     Code points the texts are drawn from: ASCII letters, the bytes next to the letter ranges ('@', '[', '`', '{':
        ///     a range test off by one maps them), digits, punctuation and NUL; Latin-1 letters ('ß' grows
        ///     to "SS", 'ª' is cased but no letter of either case); Greek with U+03A3 (the final sigma) and the case-ignorable
        ///     acute and ypogegrammeni; the titlecase digraphs; Cyrillic; the ligatures and 'ŉ' / 'ΐ' / 'ᾳ' / 'İ' (multi-code-
        ///     point mappings); 'ı' / 'ſ' / 'ⱥ' / 'ɐ' (mappings that change the UTF-8 size); Cherokee (far deltas); and
        ///     letters past U+FFFF.
        /// </summary>
        private static readonly uint[] Pool =
        {
            'a', 'z', 'A', 'Z', 'm', 'Q', '@', '[', '`', '{', '0', '9', ' ', '\'', '.', '_', 0,
            0xE9, 0xC9, 0xDF, 0x1E9E, 0xAA, 0x2B0, 0x3A3, 0x3C3, 0x3C2, 0x391, 0x3B1, 0x301, 0x345, 0x1C4, 0x1C5, 0x1C6,
            0x1C7, 0x1C8, 0x1C9, 0x416, 0x436, 0xFB03, 0xFB01, 0x149, 0x390, 0x1FB3, 0x130, 0x131, 0x17F, 0x2C65, 0x250,
            0xAB70, 0x13A0, 0x10400, 0x10428, 0x1D400,
        };

        /// <summary>Code units a <c>U</c> element can hold that are not text: lone surrogates and units past U+10FFFF.</summary>
        private static readonly uint[] UOnly = { 0xD800, 0xDFFF, 0x110000, 0xFFFF_FFFF };

        /// <summary>Skips a test on hosts without the vector forms.</summary>
        private static void RequireLanes()
        {
            if (!Avx2.IsSupported)
                Assert.Inconclusive("The vector forms need AVX2; this host runs the scalar bodies only.");
        }

        /// <summary>
        ///     A random text of <paramref name="n"/> code points in one of six styles (lowercase ASCII, uppercase ASCII,
        ///     title-cased words, mixed ASCII, Greek around U+03A3, any pool character), sometimes with an embedded NUL,
        ///     sometimes ending in NULs, and for <c>U</c> sometimes holding a unit that is not text.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The code points.</param>
        /// <param name="forU">Whether units that are not text may appear.</param>
        /// <param name="asciiOnly">Whether every code point is ASCII.</param>
        /// <returns>The code points.</returns>
        private static uint[] RandomText(System.Random rng, int n, bool forU, bool asciiOnly)
        {
            int style = rng.Next(6);
            var t = new uint[n];
            for (int i = 0; i < n; i++)
            {
                t[i] = style switch
                {
                    0 => (uint)('a' + rng.Next(26)),
                    1 => (uint)('A' + rng.Next(26)),
                    2 => i % 5 == 0 ? (uint)('A' + rng.Next(26)) : i % 5 == 4 ? ' ' : (uint)('a' + rng.Next(26)),
                    3 => new uint[] { 'a', 'B', 'c', ' ', '1', '\'', '.', 'Z', '@', '[', '`', '{' }[rng.Next(12)],
                    4 => new uint[] { 0x391, 0x3A3, 0x3B1, 0x301, 0x345, ' ', 'a', '.' }[rng.Next(8)],
                    _ => Pool[rng.Next(Pool.Length)],
                };
                if (asciiOnly)
                    t[i] &= 0x7F;
            }
            if (forU && !asciiOnly && n > 0 && rng.Next(40) == 0)
                t[rng.Next(n)] = UOnly[rng.Next(UOnly.Length)];
            if (n > 0 && rng.Next(8) == 0)
                t[rng.Next(n)] = 0;
            if (n > 0 && rng.Next(4) == 0)
                for (int k = 0, z = rng.Next(1, 4); k < z && k < n; k++)
                    t[n - 1 - k] = 0;
            return t;
        }

        /// <summary>Writes code points as an <c>S</c> (bytes; a code point past 0xFF becomes a byte at or past 0x80) or
        ///     <c>U</c> (units) element.</summary>
        /// <param name="text">The code points.</param>
        /// <param name="p">The element.</param>
        /// <param name="unit">1 or 4.</param>
        private static void Write(uint[] text, byte* p, int unit)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (unit == 4)
                    ((uint*)p)[i] = text[i];
                else
                    p[i] = text[i] < 0x100 ? (byte)text[i] : (byte)(0x80 | (text[i] & 0x7F));
            }
        }

        /// <summary>
        ///     The reference of one element: the bytes methods for <c>S</c>, CPython's algorithm for <c>U</c> (its status
        ///     returned for a unit past U+10FFFF).
        /// </summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="map">The map.</param>
        /// <param name="status">Receives the reference's status.</param>
        /// <returns>The output element's bytes.</returns>
        private static byte[] Reference(byte* src, int width, int unit, int map, out long status)
        {
            var dst = new byte[width];
            var state = NDTextElements.NewCaseState();
            fixed (byte* d = dst)
            {
                if (unit == 1)
                    NDTextElements.CaseMapBytesScalar(src, width, d, map);
                else
                    NDTextElements.CaseMapUnitsScalar((uint*)src, width / 4, (uint*)d, map, &state);
            }
            status = state.Status;
            return dst;
        }

        /// <summary>Runs the element body (the kernel's form) with a call state that starts ASCII or escalated.</summary>
        /// <param name="src">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="map">The map.</param>
        /// <param name="escalated">The state's escalation word before the call.</param>
        /// <param name="dst">The output element.</param>
        /// <returns>The state after the call.</returns>
        private static NDTextElements.CaseState RunBody(byte* src, int width, int unit, int map, long escalated, byte* dst)
        {
            var state = NDTextElements.NewCaseState();
            state.Escalated = escalated;
            if (unit == 1)
                NDTextElements.CaseMap<byte>(src, width, dst, map, &state);
            else
                NDTextElements.CaseMap<uint>(src, width, dst, map, &state);
            return state;
        }

        // =============================================================================================================
        // The element forms against CPython's algorithm
        // =============================================================================================================

        /// <summary>
        ///     Every <c>S</c> and <c>U</c> width from 1 to 72 units (one padded chunk, whole chunks, an overlapping last chunk)
        ///     and a few past it: the body equals the reference, from an ASCII call state and from an escalated one, the
        ///     escalation word goes up exactly when a <c>U</c> element is not ASCII, the error status equals the reference's,
        ///     and no byte outside the output element changes.
        /// </summary>
        [TestMethod]
        public void Elements_EqualCPythonsAlgorithm_EveryWidth()
        {
            RequireLanes();
            var rng = new System.Random(2621);
            foreach (int unit in new[] { 1, 4 })
            {
                foreach (int units in Enumerable.Range(1, 72).Concat(new[] { 96, 100, 129, 300 }))
                {
                    int width = units * unit;
                    var src = new byte[width];
                    // The output element sits between 32 guard bytes on each side.
                    var dst = new byte[width + 64];
                    for (int rep = 0; rep < 24; rep++)
                    {
                        bool ascii = rep % 3 == 0;
                        var text = RandomText(rng, units, forU: unit == 4, asciiOnly: ascii);
                        fixed (byte* s = src, d = dst)
                        {
                            Write(text, s, unit);
                            foreach (int map in Maps)
                            {
                                var want = Reference(s, width, unit, map, out long wantStatus);
                                foreach (long escalated in new long[] { 0, 1 })
                                {
                                    new Span<byte>(d, dst.Length).Fill(0xA5);
                                    var state = RunBody(s, width, unit, map, escalated, d + 32);
                                    state.Status.Should().Be(wantStatus, $"status, unit {unit}, units {units}, map {map}");
                                    if (wantStatus == 0)
                                        new Span<byte>(d + 32, width).ToArray().Should().Equal(want,
                                            $"unit {unit}, units {units}, map {map}, escalated {escalated}, text {Show(text)}");
                                    for (int g = 0; g < 32; g++)
                                    {
                                        d[g].Should().Be(0xA5, "a byte before the element");
                                        d[32 + width + g].Should().Be(0xA5, "a byte after the element");
                                    }
                                    if (unit == 4 && escalated == 0)
                                        state.Escalated.Should().Be(text.Any(c => c >= 0x80) ? 1 : 0, "the escalation word");
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     Each reason the gather path hands a <c>U</c> element to the scalar loop, at every chunk position of a 40-unit
        ///     element (the overlapping chunk included): a unit past U+FFFF, a multi-code-point mapping ('ß', 'ﬃ', 'İ'), U+03A3
        ///     for a map that lowers — and the maps that need none of them (U+03A3 under upper) stay on the vector path. The
        ///     results equal the reference.
        /// </summary>
        [TestMethod]
        public void GatherPath_HandsBackEveryReasonAtEveryPosition()
        {
            RequireLanes();
            const int units = 40;
            const int width = units * 4;
            var src = new byte[width];
            var dst = new byte[width];
            foreach (uint special in new uint[] { 0x10400, 0xDF, 0xFB03, 0x130, 0x3A3, 0x1C5, 0x345, 0x110000 })
            {
                for (int pos = 0; pos < units; pos++)
                {
                    var text = Enumerable.Range(0, units).Select(i => i % 7 == 3 ? 0x3C9u : (uint)('a' + i % 26)).ToArray();
                    text[pos] = special;
                    fixed (byte* s = src, d = dst)
                    {
                        Write(text, s, 4);
                        foreach (int map in Maps)
                        {
                            var want = Reference(s, width, 4, map, out long wantStatus);
                            var state = RunBody(s, width, 4, map, escalated: 1, d);
                            state.Status.Should().Be(wantStatus);
                            if (wantStatus == 0)
                                new Span<byte>(d, width).ToArray().Should().Equal(want, $"special U+{special:X} at {pos}, map {map}");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The <c>bytes</c> rules over runs of every length from 0 to 100 (the 16-byte vectors, the overlapping last one,
        ///     the stack copy below 16): the vector form equals the scalar body and writes nothing past the run.
        /// </summary>
        [TestMethod]
        public void ByteRuns_EqualTheBytesMethods_EveryLength()
        {
            RequireLanes();
            var rng = new System.Random(2622);
            for (int size = 0; size <= 100; size++)
            {
                for (int rep = 0; rep < 20; rep++)
                {
                    var text = RandomText(rng, size, forU: false, asciiOnly: true);
                    var src = text.Select(c => (byte)c).ToArray();
                    var want = new byte[size];
                    var got = new byte[size + 32];
                    fixed (byte* s = src, w = want, g = got)
                    {
                        foreach (int map in Maps)
                        {
                            NDTextElements.CaseMapBytesScalar(s, size, w, map);
                            new Span<byte>(g, got.Length).Fill(0x5A);
                            NDTextElements.CaseMapByteRun(s, size, g, map);
                            new Span<byte>(g, size).ToArray().Should().Equal(want, $"size {size}, map {map}");
                            for (int k = size; k < got.Length; k++)
                                g[k].Should().Be(0x5A, "a byte past the run");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     CPython's algorithm on probed texts (NumPy 2.4.2's <c>str</c> methods): the final sigma's contexts, the titlecase
        ///     digraphs, the multi-code-point mappings, the cased marks and title's "any uncased character starts a word".
        /// </summary>
        [TestMethod]
        public void CPythonsAlgorithm_MatchesPython()
        {
            // (text, upper, lower, swapcase, capitalize, title), each from CPython 3.12.
            var rows = new[]
            {
                ("ΌΣΟΣ", "ΌΣΟΣ", "όσος", "όσος", "Όσος", "Όσος"),
                ("aΣͅb", "AΣΙB", "aσͅb", "AσΙB", "Aσͅb", "Aσͅb"),
                ("ǆemal", "ǄEMAL", "ǆemal", "ǄEMAL", "ǅemal", "ǅemal"),
                ("ǅemal", "ǄEMAL", "ǆemal", "ǅEMAL", "ǅemal", "ǅemal"),
                ("ᾳa", "ΑΙA", "ᾳa", "ΑΙA", "ᾼa", "ᾼa"),
                ("ŉx", "ʼNX", "ŉx", "ʼNX", "ʼNx", "ʼNx"),
                ("İstanbul", "İSTANBUL", "i̇stanbul", "i̇STANBUL", "İstanbul", "İstanbul"),
                ("they're bill's", "THEY'RE BILL'S", "they're bill's", "THEY'RE BILL'S", "They're bill's", "They'Re Bill'S"),
                ("hello2world", "HELLO2WORLD", "hello2world", "HELLO2WORLD", "Hello2world", "Hello2World"),
                ("ΣΣ", "ΣΣ", "σς", "σς", "Σς", "Σς"),
                ("a\0Σ", "A\0Σ", "a\0σ", "A\0σ", "A\0σ", "A\0Σ"),
                // The final sigma looks back past a case-ignorable (U+0301) to a cased character, and ahead to a cased one.
                ("\u0391\u0301\u03A3", "\u0391\u0301\u03A3", "\u03B1\u0301\u03C2", "\u03B1\u0301\u03C2", "\u0391\u0301\u03C2", "\u0391\u0301\u03A3"),
                ("a\u0301\u03A3x", "A\u0301\u03A3X", "a\u0301\u03C3x", "A\u0301\u03C3X", "A\u0301\u03C3x", "A\u0301\u03A3x"),
                // The bytes next to the letter ranges are not letters.
                ("@[`{AZaz", "@[`{AZAZ", "@[`{azaz", "@[`{azAZ", "@[`{azaz", "@[`{Azaz"),
            };
            uint* mapped = stackalloc uint[3];
            foreach (var (text, up, low, swap, cap, tit) in rows)
            {
                var cps = CodePoints(text);
                var expected = new[] { up, low, swap, cap, tit };
                for (int m = 0; m < 5; m++)
                {
                    var dst = new uint[cps.Length * 3];
                    long k;
                    fixed (uint* s = cps, d = dst)
                        k = NDTextElements.CaseMapCodePoints(s, cps.Length, Maps[m], d, dst.Length, mapped);
                    FromCodePoints(dst, (int)k).Should().Be(expected[m], $"{text} map {m}");
                }
            }
        }

        /// <summary>The code points of a C# string (surrogate pairs combined).</summary>
        /// <param name="s">The string.</param>
        /// <returns>The code points.</returns>
        private static uint[] CodePoints(string s)
        {
            var list = new List<uint>();
            for (int i = 0; i < s.Length; i++)
            {
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    list.Add((uint)char.ConvertToUtf32(s[i], s[i + 1]));
                    i++;
                }
                else
                    list.Add(s[i]);
            }
            return list.ToArray();
        }

        /// <summary>A C# string of the first <paramref name="n"/> code points.</summary>
        /// <param name="cps">The code points.</param>
        /// <param name="n">Their number.</param>
        /// <returns>The string.</returns>
        private static string FromCodePoints(uint[] cps, int n)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++)
                sb.Append(char.ConvertFromUtf32((int)cps[i]));
            return sb.ToString();
        }

        /// <summary>A text's code points as hex, for failure messages.</summary>
        /// <param name="text">The code points.</param>
        /// <returns>The hex list.</returns>
        private static string Show(uint[] text) => string.Join(" ", text.Select(c => c.ToString("X")));

        // =============================================================================================================
        // StringDType texts
        // =============================================================================================================

        /// <summary>
        ///     <c>StringDType</c> texts of every UTF-8 length from 0 to 70 bytes (short cells, the 16-byte boundary, arena
        ///     texts), ASCII and not, mapped by the public functions: each result equals CPython's algorithm over the text's
        ///     code points (NULs kept), re-encoded.
        /// </summary>
        [TestMethod]
        public void StringDType_EqualsCPythonsAlgorithm_EveryLength()
        {
            var rng = new System.Random(2623);
            var texts = new List<string>();
            for (int chars = 0; chars <= 40; chars++)
                for (int rep = 0; rep < 6; rep++)
                    texts.Add(FromCodePoints(RandomText(rng, chars, forU: false, asciiOnly: rep % 2 == 0)
                                                 .Select(c => c is >= 0xD800 and <= 0xDFFF ? 'x' : c).ToArray(), chars));
            using var a = np.array(texts.ToArray(), np.dtypes.StringDType());
            uint* mapped = stackalloc uint[3];
            for (int m = 0; m < 5; m++)
            {
                using var r = m switch
                {
                    0 => np.strings.upper(a),
                    1 => np.strings.lower(a),
                    2 => np.strings.swapcase(a),
                    3 => np.strings.capitalize(a),
                    _ => np.strings.title(a),
                };
                for (int i = 0; i < texts.Count; i++)
                {
                    var cps = CodePoints(texts[i]);
                    var dst = new uint[System.Math.Max(1, cps.Length * 3)];
                    long k;
                    fixed (uint* s = cps, d = dst)
                        k = NDTextElements.CaseMapCodePoints(s, cps.Length, Maps[m], d, dst.Length, mapped);
                    ((string)r.item(i)).Should().Be(FromCodePoints(dst, (int)k), $"text {i} map {m}");
                }
            }
        }

        /// <summary>
        ///     Short cells whose bytes past the text are junk (an earlier, longer text of the same cell leaves them: the size
        ///     byte is all a shorter text rewrites) map as their text only, and the output cells keep zeros past the text.
        /// </summary>
        [TestMethod]
        public void ShortCells_IgnoreTheirJunkBytes()
        {
            using var a = np.array(new[] { "abcdefghijklmno", "xyzxyzxyzxyzxyz" }, np.dtypes.StringDType());
            // Shrink each short cell's size in its size byte (low nibble): the bytes past the new size stay as junk.
            byte* cells = (byte*)a.Storage.Address + a.Shape.offset * 16;
            cells[15] = (byte)((cells[15] & 0xF0) | 2);
            cells[16 + 15] = (byte)((cells[16 + 15] & 0xF0) | 1);
            using var r = np.strings.upper(a);
            ((string)r.item(0)).Should().Be("AB");
            ((string)r.item(1)).Should().Be("X");
            // The output cells hold the text, then zeros up to the size byte: what NumPy's load_new_string leaves in a fresh
            // (zeroed) cell. Mapped junk there would be invisible to item() but not to the cell bytes.
            byte* outCells = (byte*)r.Storage.Address + r.Shape.offset * 16;
            for (int b = 2; b < 15; b++)
                outCells[b].Should().Be(0, $"cell 0 byte {b}");
            for (int b = 1; b < 15; b++)
                outCells[16 + b].Should().Be(0, $"cell 1 byte {b}");
            using var t = np.strings.title(a);
            ((string)t.item(0)).Should().Be("Ab");
        }

        // =============================================================================================================
        // Inaccessible pages
        // =============================================================================================================

        /// <summary>
        ///     Every form over elements and byte runs that end at an inaccessible page or start right after one: an
        ///     out-of-bounds read or write ends the test host. The values are checked on the way.
        /// </summary>
        [TestMethod]
        public void Forms_StayInsideTheirElements()
        {
            RequireLanes();
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(2624);
            using var srcFence = new FencedMemory(pages: 1);
            using var dstFence = new FencedMemory(pages: 1);
            foreach (bool atEnd in new[] { true, false })
            {
                foreach (int unit in new[] { 1, 4 })
                {
                    for (int units = 1; units <= 72; units++)
                    {
                        int width = units * unit;
                        byte* s = srcFence.Place(width, atEnd);
                        byte* d = dstFence.Place(width, atEnd);
                        for (int rep = 0; rep < 8; rep++)
                        {
                            Write(RandomText(rng, units, forU: unit == 4, asciiOnly: rep % 2 == 0), s, unit);
                            foreach (int map in Maps)
                            {
                                var want = Reference(s, width, unit, map, out long wantStatus);
                                foreach (long escalated in new long[] { 0, 1 })
                                {
                                    var state = RunBody(s, width, unit, map, escalated, d);
                                    if (wantStatus == 0 && state.Status == 0)
                                        new Span<byte>(d, width).ToArray().Should().Equal(want);
                                }
                            }
                        }
                    }
                }
                for (int size = 0; size <= 70; size++)
                {
                    byte* s = srcFence.Place(System.Math.Max(size, 1), atEnd);
                    byte* d = dstFence.Place(System.Math.Max(size, 1), atEnd);
                    for (int i = 0; i < size; i++)
                        s[i] = (byte)rng.Next(0x20, 0x7F);
                    foreach (int map in Maps)
                        NDTextElements.CaseMapByteRun(s, size, d, map);
                }
            }
        }

        // =============================================================================================================
        // The IL kernels against the managed kernels
        // =============================================================================================================

        /// <summary>
        ///     The IL shell's case-map kernels (constant width and map, the body inlined) against the managed kernels of hosts
        ///     without dynamic code, over 300 elements whose first 120 are ASCII and the rest mixed: the call escalates in its
        ///     middle. The outputs, the final escalation words and the error statuses are equal.
        /// </summary>
        [TestMethod]
        public void ILKernels_EqualManagedKernels()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("No dynamic code on this host: the managed kernels are the only ones.");
            const int count = 300;
            foreach (int unit in new[] { 1, 4 })
            {
                foreach (int units in new[] { 1, 3, 8, 9, 17, 33, 64, 70 })
                {
                    int width = units * unit;
                    var rng = new System.Random(2625 + width);
                    var text = new byte[count * width];
                    fixed (byte* t = text)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            var cps = RandomText(rng, units, forU: false, asciiOnly: i < 120);
                            Write(cps, t + (long)i * width, unit);
                        }
                    }
                    foreach (int map in Maps)
                    {
                        var key = new TextMethodKey(TextMethodOp.CaseMap, unit, width, Extra: map);
                        var il = Run(ILKernelGenerator.GetTextMethodKernel(key), text, width, count, out var ilState);
                        var managed = Run(NDTextLoop.ManagedKernel(key, masked: false), text, width, count, out var managedState);
                        il.Should().Equal(managed, $"IL vs managed: {key}");
                        ilState.Escalated.Should().Be(managedState.Escalated, $"escalation word of {key}");
                        ilState.Status.Should().Be(managedState.Status, $"status of {key}");
                    }
                }
            }
        }

        /// <summary>Runs a case-map kernel over contiguous elements into a contiguous output, with a fresh call state.</summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="text">The elements.</param>
        /// <param name="width">Their width.</param>
        /// <param name="count">Their count.</param>
        /// <param name="state">Receives the call's final state.</param>
        /// <returns>The output bytes.</returns>
        private static byte[] Run(NDInnerLoopFunc kernel, byte[] text, int width, int count, out NDTextElements.CaseState state)
        {
            Assert.IsNotNull(kernel);
            var output = new byte[text.Length];
            var s = NDTextElements.NewCaseState();
            fixed (byte* t = text, o = output)
            {
                void** ptrs = stackalloc void*[2];
                long* strides = stackalloc long[2];
                ptrs[0] = t;
                ptrs[1] = o;
                strides[0] = width;
                strides[1] = width;
                kernel(ptrs, strides, count, &s);
            }
            state = s;
            return output;
        }

        // =============================================================================================================
        // The public functions against NumPy 2.4.2
        // =============================================================================================================

        /// <summary>The strings of a 1-D result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its values.</returns>
        private static string[] Strings(NDArray r) => Enumerable.Range(0, (int)r.size).Select(i => (string)r.item(i)).ToArray();

        /// <summary>
        ///     The five maps over a <c>U4</c> array (probed: NumPy 2.4.2): the mappings that grow are cut at the width
        ///     ('hello world' is 'hell' in U4, 'ß' upper is 'SS'), the final sigma, 'İ' lowered to two code points, the
        ///     result keeps the dtype.
        /// </summary>
        [TestMethod]
        public void U_MatchesNumPy()
        {
            // "i̇" is 'i' + COMBINING DOT ABOVE (two code points), "İ" the precomposed 'İ': CPython lowers the
            // second to the first.
            using var u = np.array(new[] { "abc", "ß", "ﬃx", "ǆa", "ΑΣ", "ΑΣ b", "aΣ́", "hello world", "ABC def", "a b", "i̇", "İ" },
                                   np.dtype("U4"));
            var expected = new[]
            {
                new[] { "ABC", "SS", "FFIX", "ǄA", "ΑΣ", "ΑΣ B", "AΣ́", "HELL", "ABC ", "A B", "İ", "İ" },
                new[] { "abc", "ß", "ﬃx", "ǆa", "ας", "ας b", "aς́", "hell", "abc ", "a b", "i̇", "i̇" },
                new[] { "ABC", "SS", "FFIX", "ǄA", "ας", "ας B", "Aς́", "HELL", "abc ", "A B", "İ", "i̇" },
                new[] { "Abc", "Ss", "Ffix", "ǅa", "Ας", "Ας b", "Aς́", "Hell", "Abc ", "A b", "İ", "İ" },
                new[] { "Abc", "Ss", "Ffix", "ǅa", "Ας", "Ας B", "Aς́", "Hell", "Abc ", "A B", "İ", "İ" },
            };
            var fns = new Func<NDArray, NDArray>[] { np.strings.upper, np.strings.lower, np.strings.swapcase, np.strings.capitalize, np.strings.title };
            for (int m = 0; m < 5; m++)
            {
                using var r = fns[m](u);
                r.dtype.Should().Be(np.dtype("U4"));
                Strings(r).Should().Equal(expected[m], $"map {m}");
            }
        }

        /// <summary>
        ///     The five maps over an <c>S8</c> array (probed: NumPy 2.4.2): ASCII rules only (0xE9 is no letter), NULs kept.
        /// </summary>
        [TestMethod]
        public void S_MatchesNumPy()
        {
            using var s = np.array(new object[]
            {
                Encoding.Latin1.GetBytes("abc"), Encoding.Latin1.GetBytes("aBc dEf"), new byte[] { 0xE9, (byte)'a' },
                new byte[] { (byte)'a', 0, (byte)'b' }, Encoding.Latin1.GetBytes("hello wo"), Array.Empty<byte>(),
            }, np.dtype("S8"));
            var expected = new[]
            {
                new[] { "ABC", "ABC DEF", "éA", "A\0B", "HELLO WO", "" },
                new[] { "abc", "abc def", "éa", "a\0b", "hello wo", "" },
                new[] { "ABC", "AbC DeF", "éA", "A\0B", "HELLO WO", "" },
                new[] { "Abc", "Abc def", "éa", "A\0b", "Hello wo", "" },
                new[] { "Abc", "Abc Def", "éA", "A\0B", "Hello Wo", "" },
            };
            var fns = new Func<NDArray, NDArray>[] { np.strings.upper, np.strings.lower, np.strings.swapcase, np.strings.capitalize, np.strings.title };
            for (int m = 0; m < 5; m++)
            {
                using var r = fns[m](s);
                r.dtype.Should().Be(np.dtype("S8"));
                for (int i = 0; i < 6; i++)
                    Encoding.Latin1.GetString((byte[])r.item(i)).Should().Be(expected[m][i], $"map {m}, element {i}");
            }
        }

        /// <summary>
        ///     <c>StringDType</c> (probed: NumPy 2.4.2): a NaN or None NA object is the str method's TypeError quoting its type;
        ///     under a string NA object an NA is computed as its text and any result equal to it is stored as the NA
        ///     ('abc'.upper() under na_object='ABC'); the result keeps the NA object.
        /// </summary>
        [TestMethod]
        public void StringDType_NA_FollowsNumPy()
        {
            using var nan = np.array(new object[] { "abc", double.NaN, "ABC" }, np.dtypes.StringDType(double.NaN));
            Action upperNan = () => np.strings.upper(nan);
            upperNan.Should().Throw<TypeError>().WithMessage("descriptor 'upper' for 'str' objects doesn't apply to a 'float' object");
            using var none = np.array(new object[] { "abc", null, "ABC" }, np.dtypes.StringDType(null));
            Action titleNone = () => np.strings.title(none);
            titleNone.Should().Throw<TypeError>().WithMessage("descriptor 'title' for 'str' objects doesn't apply to a 'NoneType' object");

            using var s = np.array(new object[] { "abc", "x", "ABC", "ABCD" }, np.dtypes.StringDType("ABC"));
            using var up = np.strings.upper(s);
            up.dtype.Should().Be(np.dtypes.StringDType("ABC"));
            // 'abc' -> 'ABC' and the NA ('ABC') -> 'ABC' are stored as the NA; NumPy's isnan is False for a string NA, so
            // the cells' NA flag is read.
            IsNA(up, 0).Should().BeTrue();
            IsNA(up, 1).Should().BeFalse();
            IsNA(up, 2).Should().BeTrue();
            IsNA(up, 3).Should().BeFalse();
            ((string)up.item(1)).Should().Be("X");
            ((string)up.item(3)).Should().Be("ABCD");
            using var low = np.strings.lower(s);
            // The NA lowers to 'abc', a text.
            IsNA(low, 2).Should().BeFalse();
            ((string)low.item(2)).Should().Be("abc");

            using var empty = np.array(new object[] { "", "a" }, np.dtypes.StringDType(""));
            using var ue = np.strings.upper(empty);
            IsNA(ue, 0).Should().BeTrue();
            IsNA(ue, 1).Should().BeFalse();
        }

        /// <summary>Whether a <c>StringDType</c> element is an NA (its cell's MISSING flag).</summary>
        /// <param name="a">A contiguous <c>StringDType</c> array.</param>
        /// <param name="i">The flat index.</param>
        /// <returns>True for an NA cell.</returns>
        private static bool IsNA(NDArray a, int i)
        {
            byte* cell = (byte*)a.Storage.Address + (a.Shape.offset + i) * 16;
            return (cell[15] & VStringAllocator.NPY_STRING_MISSING) == VStringAllocator.NPY_STRING_MISSING;
        }

        /// <summary>
        ///     The errors and the result objects (probed: NumPy 2.4.2): a unit past U+10FFFF is CPython's SystemError, a
        ///     non-text operand NumPy's TypeError; any layout gives a fresh C-ordered result; a chararray operand a plain array;
        ///     a C# string operand a 0-d <c>U</c> result.
        /// </summary>
        [TestMethod]
        public void ErrorsAndResultObjects_MatchNumPy()
        {
            using var big = np.zeros(new Shape(2), np.dtype("U2"));
            using (var units = big.view(np.uint32))
                units.SetUInt32(0x110000, 1);
            Action upperBig = () => np.strings.upper(big);
            upperBig.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");

            using var ints = np.array(new[] { 1, 2 });
            Action upperInts = () => np.strings.upper(ints);
            upperInts.Should().Throw<TypeError>().WithMessage("string operation on non-string array");

            using var m = np.array(new[] { "ab", "cd", "ef", "gh", "ij", "kl" }).reshape(2, 3);
            using var mt = m.T;
            using var r = np.strings.upper(mt);
            r.Shape.IsContiguous.Should().BeTrue();
            using var flat = r.flatten();
            Strings(flat).Should().Equal("AB", "GH", "CD", "IJ", "EF", "KL");

            using var c = np.@char.array(new[] { "ab", "c" });
            using var rc = np.strings.title(c);
            rc.GetType().Should().Be(typeof(NDArray));
            Strings(rc).Should().Equal("Ab", "C");

            using var scalar = np.strings.capitalize("hello WORLD");
            scalar.ndim.Should().Be(0);
            ((string)scalar.item()).Should().Be("Hello world");
        }
    }
}
