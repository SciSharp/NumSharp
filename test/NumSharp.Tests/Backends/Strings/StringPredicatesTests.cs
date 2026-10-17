using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Printing;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NumPy's string predicates <c>isalpha</c> ... <c>isnumeric</c> (strings phase 2, P2.6; strings plan D14) and the
    ///     Unicode tables they read (P2.A, <see cref="PyUnicodeData"/>): the tables against their pin and CPython's records,
    ///     every element form against NumPy's unit loops, the forms against inaccessible pages, the IL kernels against the
    ///     managed ones, and the public functions against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The string oracle (sections <c>strprop</c> / <c>tprop</c>) checks the public functions against NumPy over its
    ///     operands; these tests check that the element forms are NumPy's unit loops (<c>IsKindScalar</c>,
    ///     <c>IsKindUtf8Scalar</c>) on what the operands do not reach: every <c>S</c> / <c>U</c> width up to past the 64-unit
    ///     lane cap, every UTF-8 length across the 16-byte windows and the tail copy, the early exits after the first chunk,
    ///     the escalation from the ASCII compares to the gather in the middle of a call, code units past U+FFFF and past
    ///     U+10FFFF, embedded and trailing NULs, and short cells whose bytes past the size are junk.
    ///     </para>
    ///     <para>Every test is deterministic (fixed seeds) and touches no process-wide switch, so they run in parallel.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StringPredicatesTests
    {
        /// <summary>The predicate kinds, isalpha (0) to isnumeric (8).</summary>
        private static readonly int[] Kinds = Enumerable.Range(0, 9).ToArray();

        /// <summary>
        ///     Code points the elements are drawn from: ASCII letters, digits and whitespace (both rules), NUL, Latin-1,
        ///     Greek upper and lower (final sigma), titlecase digraphs, Cyrillic, numeric letters, CJK numerals, fullwidth
        ///     forms, superscript and fraction digits, Arabic-Indic digits, Unicode spaces, a combining lower mark, math
        ///     alphanumerics past U+FFFF (cased), an emoji and an astral digit.
        /// </summary>
        private static readonly uint[] Pool =
        {
            'a', 'z', 'A', 'Z', '0', '9', ' ', '\t', '\n', 0x0B, 0x0C, 0x0D, 0x1C, 0x1F, 0, '_', '\'',
            0xE9, 0xDF, 0xC9, 0xAA, 0xB2, 0xBD, 0x3A9, 0x3C9, 0x3A3, 0x3C2, 0x1C5, 0x1C4, 0x1C6, 0x1F88, 0x436, 0x416,
            0x2167, 0x2177, 0x3007, 0x4E00, 0x4E8C, 0xFF21, 0xFF41, 0xFF10, 0xA0, 0x2003, 0x3000, 0x85, 0x660, 0x966,
            0x345, 0x1D400, 0x1D41A, 0x1F600, 0x104A0, 0x1D7CE,
        };

        /// <summary>Code units a <c>U</c> element can hold that are not text: lone surrogates and units past U+10FFFF.</summary>
        private static readonly uint[] UOnly = { 0xD800, 0xDFFF, 0x110000, 0x7FFF_FFFF, 0xFFFF_FFFF, 0xFFFF, 0x10000, 0x10FFFF };

        /// <summary>Skips a test on hosts without the vector forms.</summary>
        private static void RequireLanes()
        {
            if (!Avx2.IsSupported || !Bmi2.X64.IsSupported)
                Assert.Inconclusive("The vector forms need AVX2 and BMI2; this host runs NumPy's unit loops only.");
        }

        /// <summary>
        ///     A random text of <paramref name="n"/> code points in one of ten styles (all lowercase / uppercase ASCII,
        ///     title-cased words, digits, whitespace, lowercase / uppercase / title-cased non-ASCII, numerics, or any unit),
        ///     sometimes with an embedded NUL, sometimes ending in NULs.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="n">The code points.</param>
        /// <param name="forU">Whether units that are not text (<see cref="UOnly"/>) may appear.</param>
        /// <returns>The code points.</returns>
        private static uint[] RandomText(System.Random rng, int n, bool forU)
        {
            int style = rng.Next(10);
            var t = new uint[n];
            for (int i = 0; i < n; i++)
            {
                t[i] = style switch
                {
                    0 => (uint)('a' + rng.Next(26)),
                    1 => (uint)('A' + rng.Next(26)),
                    2 => i % 5 == 0 ? (uint)('A' + rng.Next(26)) : i % 5 == 4 ? ' ' : (uint)('a' + rng.Next(26)),
                    3 => (uint)('0' + rng.Next(10)),
                    4 => new uint[] { ' ', '\t', 0x1C, 0x3000, 0xA0, 0x85 }[rng.Next(6)],
                    5 => new uint[] { 0xE9, 0xDF, 0x3C9, 0x436, 'a', 'b', 0x1D41A }[rng.Next(7)],
                    6 => new uint[] { 0x3A9, 0x416, 0xC9, 'A', 0x1D400 }[rng.Next(5)],
                    7 => i % 4 == 0 ? new uint[] { 0x3A9, 0x1C5, 'A', 0xC9 }[rng.Next(4)] : new uint[] { 0x3C9, 'a', 0xE9 }[rng.Next(3)],
                    8 => new uint[] { 0x2167, 0x4E00, 0x3007, 0x660, 0xFF10, 0xB2, 0xBD, '7' }[rng.Next(8)],
                    _ => forU && rng.Next(25) == 0 ? UOnly[rng.Next(UOnly.Length)] : Pool[rng.Next(Pool.Length)],
                };
            }
            if (n > 0 && rng.Next(8) == 0)
                t[rng.Next(n)] = 0;
            if (n > 0 && rng.Next(4) == 0)
                for (int k = 0, z = rng.Next(1, 4); k < z && k < n; k++)
                    t[n - 1 - k] = 0;
            return t;
        }

        /// <summary>Writes code points as an <c>S</c> (bytes; a code point past 0xFF becomes a byte at or past 0x80 or a
        ///     letter) or <c>U</c> (units) element.</summary>
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

        /// <summary>The UTF-8 bytes of code points (a lone surrogate or a unit past U+10FFFF becomes <c>'x'</c>).</summary>
        /// <param name="text">The code points.</param>
        /// <returns>The bytes.</returns>
        private static byte[] Utf8(uint[] text)
        {
            var bytes = new List<byte>(text.Length * 4);
            foreach (uint c0 in text)
            {
                uint c = c0 is >= 0xD800 and <= 0xDFFF || c0 > 0x10FFFF ? 'x' : c0;
                if (c < 0x80)
                {
                    bytes.Add((byte)c);
                }
                else if (c < 0x800)
                {
                    bytes.Add((byte)(0xC0 | (c >> 6)));
                    bytes.Add((byte)(0x80 | (c & 0x3F)));
                }
                else if (c < 0x10000)
                {
                    bytes.Add((byte)(0xE0 | (c >> 12)));
                    bytes.Add((byte)(0x80 | ((c >> 6) & 0x3F)));
                    bytes.Add((byte)(0x80 | (c & 0x3F)));
                }
                else
                {
                    bytes.Add((byte)(0xF0 | (c >> 18)));
                    bytes.Add((byte)(0x80 | ((c >> 12) & 0x3F)));
                    bytes.Add((byte)(0x80 | ((c >> 6) & 0x3F)));
                    bytes.Add((byte)(0x80 | (c & 0x3F)));
                }
            }
            return bytes.ToArray();
        }

        /// <summary>NumPy's UTF-8 predicate of a text in memory: False without code points (the trailing NULs dropped),
        ///     else the code-point loop.</summary>
        /// <param name="s">The text.</param>
        /// <param name="size">Its byte size.</param>
        /// <param name="kind">The predicate.</param>
        /// <returns>The predicate's value.</returns>
        private static bool Utf8Reference(byte* s, long size, int kind)
        {
            while (size > 0 && s[size - 1] == 0)
                size--;
            return size != 0 && NDTextElements.IsKindUtf8Scalar(s, size, kind);
        }

        // =============================================================================================================
        // The Unicode tables
        // =============================================================================================================

        /// <summary>
        ///     The tables are the generator's: Unicode 15.0.0 from CPython 3.12.12 (NumPy 2.4.2's oracle), and every code
        ///     point's flags, decimal value and full case mappings, decoded through the runtime lookups, hash to the SHA-256 the
        ///     generator recorded. A table edited by hand, or a lookup that decodes differently, fails here.
        /// </summary>
        [TestMethod]
        public void UnicodeTables_MatchTheirPin()
        {
            PyUnicodeData.UnicodeVersion.Should().Be("15.0.0");
            PyUnicodeData.CPythonVersion.Should().Be("3.12.12");
            PyUnicodeData.ComputeTableHash().Should().Be(PyUnicodeData.TableHash);
        }

        /// <summary>
        ///     Records of CPython 3.12 (probed with <c>str.isalpha()</c> ... and the case methods): the classes, the decimal
        ///     values, units past Unicode, and the full mappings that are longer than one code point.
        /// </summary>
        [TestMethod]
        public void UnicodeTables_HoldCPythonsRecords()
        {
            static bool Has(uint cp, int flag) => (PyUnicodeData.Flags(cp) & flag) != 0;
            // 'a', 'Ω' (U+03A9), 'ǅ' (U+01C5): lower, upper, title; all three alphabetic and cased.
            Has('a', PyUnicodeData.LowerMask).Should().BeTrue();
            Has('a', PyUnicodeData.UpperMask | PyUnicodeData.TitleMask).Should().BeFalse();
            Has(0x3A9, PyUnicodeData.UpperMask).Should().BeTrue();
            Has(0x3A9, PyUnicodeData.LowerMask | PyUnicodeData.TitleMask).Should().BeFalse();
            Has(0x1C5, PyUnicodeData.TitleMask).Should().BeTrue();
            Has(0x1C5, PyUnicodeData.UpperMask | PyUnicodeData.LowerMask).Should().BeFalse();
            foreach (uint cp in new uint[] { 'a', 0x3A9, 0x1C5 })
            {
                Has(cp, PyUnicodeData.AlphaMask).Should().BeTrue();
                Has(cp, PyUnicodeData.CasedMask).Should().BeTrue();
            }
            // '٣' (U+0663) decimal 3; '²' a digit, not decimal; '½' numeric only.
            PyUnicodeData.DecimalValue(0x663).Should().Be(3);
            Has(0x663, PyUnicodeData.DecimalMask | PyUnicodeData.DigitMask | PyUnicodeData.NumericMask).Should().BeTrue();
            Has(0x663, PyUnicodeData.AlphaMask).Should().BeFalse();
            PyUnicodeData.DecimalValue(0xB2).Should().Be(-1);
            Has(0xB2, PyUnicodeData.DigitMask).Should().BeTrue();
            Has(0xB2, PyUnicodeData.DecimalMask).Should().BeFalse();
            Has(0xBD, PyUnicodeData.NumericMask).Should().BeTrue();
            Has(0xBD, PyUnicodeData.DigitMask | PyUnicodeData.DecimalMask).Should().BeFalse();
            // Python's whitespace (U+001C, U+0085, U+3000), none printable; a lone surrogate has no class.
            foreach (uint cp in new uint[] { 0x1C, 0x85, 0x3000 })
            {
                PyUnicodeData.IsSpace(cp).Should().BeTrue();
                PyUnicodeData.IsPrintable(cp).Should().BeFalse();
            }
            PyUnicodeData.Flags(0xD800).Should().Be(0);
            // U+0345 (combining ypogegrammeni): lowercase and case-ignorable, not alphabetic; the apostrophe is case-ignorable.
            Has(0x345, PyUnicodeData.LowerMask | PyUnicodeData.CasedMask).Should().BeTrue();
            PyUnicodeData.IsCaseIgnorable(0x345).Should().BeTrue();
            Has(0x345, PyUnicodeData.AlphaMask).Should().BeFalse();
            PyUnicodeData.IsCaseIgnorable('\'').Should().BeTrue();
            PyUnicodeData.IsCased('\'').Should().BeFalse();
            // Past U+10FFFF: CPython's record 0.
            PyUnicodeData.Flags(0x110000).Should().Be(0);
            PyUnicodeData.Flags(0xFFFF_FFFF).Should().Be(0);
            PyUnicodeData.DecimalValue(0xFFFF_FFFF).Should().Be(-1);
            // Full mappings: 'ß'.upper() == 'SS', 'İ'.lower() == 'i̇', 'ǆ'.title() == 'ǅ', 'ﬃ'.upper() == 'FFI'.
            Mapping(PyUnicodeData.ToUpperFull, 0xDF).Should().Equal(0x53u, 0x53u);
            Mapping(PyUnicodeData.ToLowerFull, 0x130).Should().Equal(0x69u, 0x307u);
            Mapping(PyUnicodeData.ToTitleFull, 0x1C6).Should().Equal(0x1C5u);
            Mapping(PyUnicodeData.ToUpperFull, 0xFB03).Should().Equal(0x46u, 0x46u, 0x49u);
            Mapping(PyUnicodeData.ToUpperFull, 0x110000).Should().Equal(0x110000u);
        }

        /// <summary>A mapping function of <see cref="PyUnicodeData"/>.</summary>
        /// <param name="cp">The code unit.</param>
        /// <param name="dst">Room for three code points.</param>
        /// <returns>The mapping's length.</returns>
        private delegate int MappingFunction(uint cp, uint* dst);

        /// <summary>The full mapping of a code unit as an array.</summary>
        /// <param name="map">The mapping.</param>
        /// <param name="cp">The code unit.</param>
        /// <returns>The mapped code points.</returns>
        private static uint[] Mapping(MappingFunction map, uint cp)
        {
            uint* dst = stackalloc uint[PyUnicodeData.MaxMapping];
            int n = map(cp, dst);
            return new ReadOnlySpan<uint>(dst, n).ToArray();
        }

        /// <summary>
        ///     No code point is lowercase and uppercase or titlecase at once. The lane forms of <c>istitle</c> compute "bad"
        ///     as <c>(upper-or-title AND previous-cased) OR (lower AND NOT previous-cased)</c>, which equals NumPy's
        ///     if-else (the upper / title test first) only while no unit is in both classes.
        /// </summary>
        [TestMethod]
        public void UnicodeTables_NoUnitIsLowerAndUpperOrTitle()
        {
            int both = 0;
            for (uint cp = 0; cp < PyUnicodeData.UnicodeEnd; cp++)
            {
                int f = PyUnicodeData.Flags(cp);
                if ((f & PyUnicodeData.LowerMask) != 0 && (f & (PyUnicodeData.UpperMask | PyUnicodeData.TitleMask)) != 0)
                    both++;
            }
            both.Should().Be(0);
        }

        /// <summary>
        ///     The older readers of Unicode properties read the tables: <c>PyUnicode.IsSpace</c> / <c>DecimalValue</c> (number
        ///     parsing) and <c>PyRepr.IsPrintable</c> (repr) agree with <see cref="PyUnicodeData"/> on every code point,
        ///     whatever Unicode version the running .NET carries.
        /// </summary>
        [TestMethod]
        public void PyUnicodeHelpers_ReadTheTables()
        {
            for (uint cp = 0; cp < PyUnicodeData.UnicodeEnd; cp++)
            {
                if (PyUnicode.IsSpace((int)cp) != PyUnicodeData.IsSpace(cp))
                    Assert.Fail($"IsSpace U+{cp:X4}");
                if (PyUnicode.DecimalValue((int)cp) != PyUnicodeData.DecimalValue(cp))
                    Assert.Fail($"DecimalValue U+{cp:X4}");
                if (PyRepr.IsPrintable((int)cp) != PyUnicodeData.IsPrintable(cp))
                    Assert.Fail($"IsPrintable U+{cp:X4}");
            }
        }

        // =============================================================================================================
        // S / U elements: every form against NumPy's unit loops
        // =============================================================================================================

        /// <summary>
        ///     <c>IsKind</c> (lane masks up to 64 units with the first-chunk exit, the chunk stream past them) equals NumPy's
        ///     unit loop for every <c>S</c> width 1-70 and every <c>U</c> width 1-72 units, every predicate, from a call that
        ///     has not escalated and from one that has; an all-ASCII element never escalates the call.
        /// </summary>
        [TestMethod]
        public void Elements_EqualTheUnitLoops_EveryWidth()
        {
            RequireLanes();
            var rng = new System.Random(2606);
            var buffer = new byte[72 * 4];
            fixed (byte* p = buffer)
            {
                foreach (int unit in new[] { 1, 4 })
                {
                    int maxUnits = unit == 1 ? 70 : 72;
                    for (int units = 1; units <= maxUnits; units++)
                    {
                        int width = units * unit;
                        for (int rep = 0; rep < 150; rep++)
                        {
                            var text = RandomText(rng, units, forU: unit == 4);
                            Write(text, p, unit);
                            bool ascii = text.All(c => c < 0x80);
                            foreach (int kind in Kinds)
                            {
                                if (unit == 1 && kind >= NDTextElements.IsDecimalKind)
                                    continue;
                                bool want = unit == 1
                                    ? NDTextElements.IsKindScalar<byte>(p, width, kind)
                                    : NDTextElements.IsKindScalar<uint>(p, width, kind);
                                foreach (long escalated in new long[] { 0, 1 })
                                {
                                    var state = NDTextElements.NewKindState();
                                    state.Escalated = escalated;
                                    bool got = unit == 1
                                        ? NDTextElements.IsKind<byte>(p, width, kind, &state)
                                        : NDTextElements.IsKind<uint>(p, width, kind, &state);
                                    if (got != want)
                                        Assert.Fail($"{(unit == 1 ? "S" : "U")}{units} {NDTextElements.PredicateName(kind)} escalated={escalated}: " +
                                                    $"{got}, NumPy {want}; units [{string.Join(",", text.Select(c => c.ToString("X")))}]");
                                    if (unit == 4 && escalated == 0 && ascii && state.Escalated != 0)
                                        Assert.Fail($"U{units}: an all-ASCII element escalated the call");
                                }
                            }
                        }
                    }
                }
            }
        }

        // =============================================================================================================
        // UTF-8 text: every form against NumPy's code-point loop
        // =============================================================================================================

        /// <summary>
        ///     <c>IsKindUtf8</c> (one padded window up to 16 bytes, the two-half ASCII form up to 32, the ASCII stream, the
        ///     UTF-8 windows in place and through the tail copy, the scalar loop for a character past U+FFFF) equals NumPy's
        ///     code-point loop for texts of 0 to 130 bytes; <c>IsKindUtf8Window</c> (the short-cell form) too for texts of
        ///     up to 16 bytes.
        /// </summary>
        [TestMethod]
        public void Utf8_EqualsTheCodePointLoop_EveryLength()
        {
            RequireLanes();
            var rng = new System.Random(2607);
            var buffer = new byte[160];
            byte* window = stackalloc byte[16];
            uint* flags = PyUnicodeData.BmpTable;
            fixed (byte* s = buffer)
            {
                for (int chars = 0; chars <= 64; chars++)
                {
                    for (int rep = 0; rep < 250; rep++)
                    {
                        var bytes = Utf8(RandomText(rng, chars, forU: false));
                        if (bytes.Length > buffer.Length)
                            continue;
                        bytes.CopyTo(buffer, 0);
                        long stripped = bytes.Length;
                        while (stripped > 0 && s[stripped - 1] == 0)
                            stripped--;
                        foreach (int kind in Kinds)
                        {
                            bool want = Utf8Reference(s, bytes.Length, kind);
                            if (NDTextElements.IsKindUtf8(s, bytes.Length, kind, flags) != want)
                                Assert.Fail($"{NDTextElements.PredicateName(kind)} of {Convert.ToHexString(bytes)}: NumPy {want}");
                            if (bytes.Length <= 16)
                            {
                                for (int i = 0; i < 16; i++)
                                    window[i] = i < bytes.Length ? s[i] : (byte)0;
                                if (NDTextElements.IsKindUtf8Window(Vector128.Load(window), (int)stripped, kind, flags) != want)
                                    Assert.Fail($"window {NDTextElements.PredicateName(kind)} of {Convert.ToHexString(bytes)}: NumPy {want}");
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The <c>StringDType</c> element body over short cells (text in place, the bytes past its size junk, as a short
        ///     cell's padding is not guaranteed zero) equals NumPy's code-point loop over the text.
        /// </summary>
        [TestMethod]
        public void ShortCells_EqualTheCodePointLoop()
        {
            RequireLanes();
            var rng = new System.Random(2608);
            byte* cell = stackalloc byte[16];
            var aux = new VStringMethodAux { Flags = PyUnicodeData.BmpTable };
            for (int trial = 0; trial < 20000; trial++)
            {
                var bytes = Utf8(RandomText(rng, rng.Next(16), forU: false));
                if (bytes.Length > 15)
                    continue;
                for (int i = 0; i < 15; i++)
                    cell[i] = i < bytes.Length ? bytes[i] : (byte)rng.Next(1, 256);
                cell[15] = (byte)(VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA | bytes.Length);
                fixed (byte* text = bytes)
                {
                    foreach (int kind in Kinds)
                    {
                        bool want = Utf8Reference(text, bytes.Length, kind);
                        byte got = NDVStringMethods.VIsKind(cell, kind, &aux);
                        if (got != (want ? 1 : 0))
                            Assert.Fail($"{NDTextElements.PredicateName(kind)} of {Convert.ToHexString(bytes)}: {got}, NumPy {want}");
                    }
                }
            }
            aux.Status.Should().Be(0);
        }

        // =============================================================================================================
        // Fenced memory: no form reads outside its element or text
        // =============================================================================================================

        /// <summary>
        ///     Every form over elements and texts that end at an inaccessible page, or start right after one: an
        ///     out-of-bounds read ends the test host. The values are checked against NumPy's loops on the way.
        /// </summary>
        [TestMethod]
        public void Forms_StayInsideTheirElements()
        {
            RequireLanes();
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(2609);
            using var fence = new FencedMemory(pages: 1);
            uint* flags = PyUnicodeData.BmpTable;
            foreach (bool atEnd in new[] { true, false })
            {
                foreach (int unit in new[] { 1, 4 })
                {
                    for (int units = 1; units <= 72; units++)
                    {
                        int width = units * unit;
                        byte* p = fence.Place(width, atEnd);
                        for (int rep = 0; rep < 20; rep++)
                        {
                            Write(RandomText(rng, units, forU: unit == 4), p, unit);
                            foreach (int kind in Kinds)
                            {
                                var state = NDTextElements.NewKindState();
                                bool got = unit == 1
                                    ? NDTextElements.IsKind<byte>(p, width, kind, &state)
                                    : NDTextElements.IsKind<uint>(p, width, kind, &state);
                                bool want = unit == 1
                                    ? NDTextElements.IsKindScalar<byte>(p, width, kind)
                                    : NDTextElements.IsKindScalar<uint>(p, width, kind);
                                got.Should().Be(want);
                            }
                        }
                    }
                }
                for (int chars = 1; chars <= 40; chars++)
                {
                    for (int rep = 0; rep < 30; rep++)
                    {
                        var bytes = Utf8(RandomText(rng, chars, forU: false));
                        byte* s = fence.Place(bytes.Length, atEnd);
                        bytes.CopyTo(new Span<byte>(s, bytes.Length));
                        foreach (int kind in Kinds)
                            NDTextElements.IsKindUtf8(s, bytes.Length, kind, flags).Should().Be(Utf8Reference(s, bytes.Length, kind));
                    }
                }
            }
        }

        // =============================================================================================================
        // The IL kernels against the managed kernels
        // =============================================================================================================

        /// <summary>
        ///     The IL shell's predicate kernels (constant width and predicate, the body inlined) against the managed kernels of
        ///     hosts without dynamic code, over 400 elements whose first 150 are ASCII and the rest mixed: the call escalates in
        ///     its middle. The outputs and the final escalation words are equal.
        /// </summary>
        [TestMethod]
        public void ILKernels_EqualManagedKernels()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("No dynamic code on this host: the managed kernels are the only ones.");
            const int count = 400;
            foreach (int unit in new[] { 1, 4 })
            {
                foreach (int units in new[] { 1, 3, 8, 9, 17, 33, 64, 70 })
                {
                    int width = units * unit;
                    var rng = new System.Random(2610 + width);
                    var text = new byte[count * width];
                    fixed (byte* t = text)
                    {
                        for (int i = 0; i < count; i++)
                        {
                            var cps = RandomText(rng, units, forU: unit == 4);
                            if (i < 150)
                                for (int k = 0; k < cps.Length; k++)
                                    cps[k] &= 0x7F;
                            Write(cps, t + (long)i * width, unit);
                        }
                    }
                    foreach (int kind in Kinds)
                    {
                        if (unit == 1 && kind >= NDTextElements.IsDecimalKind)
                            continue;
                        var key = new TextMethodKey(TextMethodOp.IsKind, unit, width, Extra: kind);
                        var il = Run(ILKernelGenerator.GetTextMethodKernel(key), text, width, count, out long ilState);
                        var managed = Run(NDTextLoop.ManagedKernel(key, masked: false), text, width, count, out long managedState);
                        CollectionAssert.AreEqual(managed, il, $"IL vs managed: {key}");
                        ilState.Should().Be(managedState, $"escalation word of {key}");
                    }
                }
            }
        }

        /// <summary>Runs a predicate kernel over contiguous elements into a bool output, with a fresh call state.</summary>
        /// <param name="kernel">The kernel.</param>
        /// <param name="text">The elements.</param>
        /// <param name="width">Their width.</param>
        /// <param name="count">Their count.</param>
        /// <param name="escalated">Receives the call's final escalation word.</param>
        /// <returns>The output bytes.</returns>
        private static byte[] Run(NDInnerLoopFunc kernel, byte[] text, int width, int count, out long escalated)
        {
            Assert.IsNotNull(kernel);
            var output = new byte[count];
            var state = NDTextElements.NewKindState();
            fixed (byte* t = text, o = output)
            {
                void** ptrs = stackalloc void*[2];
                long* strides = stackalloc long[2];
                ptrs[0] = t;
                ptrs[1] = o;
                strides[0] = width;
                strides[1] = 1;
                kernel(ptrs, strides, count, &state);
            }
            escalated = state.Escalated;
            return output;
        }

        // =============================================================================================================
        // The public functions against NumPy 2.4.2
        // =============================================================================================================

        /// <summary>The bools of a 1-D result.</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its values.</returns>
        private static bool[] Bools(NDArray r) => Enumerable.Range(0, (int)r.size).Select(i => r.GetBoolean(i)).ToArray();

        /// <summary>
        ///     The nine predicates over a <c>U</c> array (probed: NumPy 2.4.2): an embedded NUL counts as a unit, trailing NULs
        ///     do not, U+001C is whitespace, '²' is a digit but not decimal, '½' numeric only, 'ǅ' titlecase, and the math
        ///     alphanumerics past U+FFFF are cased.
        /// </summary>
        [TestMethod]
        public void U_MatchesNumPy()
        {
            using var u = np.array(new[]
            {
                "abc", "a1", "", "éß", "ABC", "Abc Def", "abc\0", "a\0b", "\u001c", " \t", "²", "½", "٣", "ǅabc", "ǅ",
                "\U0001D400", "\U0001D41A",
            });
            Bools(np.strings.isalpha(u)).Should().Equal(true, false, false, true, true, false, true, false, false, false, false, false, false, true, true, true, true);
            Bools(np.strings.isdigit(u)).Should().Equal(false, false, false, false, false, false, false, false, false, false, true, false, true, false, false, false, false);
            Bools(np.strings.isspace(u)).Should().Equal(false, false, false, false, false, false, false, false, true, true, false, false, false, false, false, false, false);
            Bools(np.strings.isalnum(u)).Should().Equal(true, true, false, true, true, false, true, false, false, false, true, true, true, true, true, true, true);
            Bools(np.strings.islower(u)).Should().Equal(true, true, false, true, false, false, true, true, false, false, false, false, false, false, false, false, true);
            Bools(np.strings.isupper(u)).Should().Equal(false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, true, false);
            Bools(np.strings.istitle(u)).Should().Equal(false, false, false, false, false, true, false, false, false, false, false, false, false, true, true, true, false);
            Bools(np.strings.isdecimal(u)).Should().Equal(false, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false);
            Bools(np.strings.isnumeric(u)).Should().Equal(false, false, false, false, false, false, false, false, false, false, true, true, true, false, false, false, false);
        }

        /// <summary>
        ///     The seven <c>S</c> predicates (probed: NumPy 2.4.2's ASCII rules: a byte at or past 0x80 has no class, 0x1C is
        ///     not whitespace, \v is), and isdecimal / isnumeric, which have no <c>S</c> loop.
        /// </summary>
        [TestMethod]
        public void S_MatchesNumPy()
        {
            var items = new[] { "abc", "a1", "", "é", "ABC", "Abc Def", "abc\0", "a\0b", "\u001c", " \t", "\u000b", "12" }
                .Select(t => (object)t.Select(c => (byte)c).ToArray()).ToArray();
            using var s = np.array(items);
            s.dtype.str.Should().Be("|S7");
            Bools(np.strings.isalpha(s)).Should().Equal(true, false, false, false, true, false, true, false, false, false, false, false);
            Bools(np.strings.isdigit(s)).Should().Equal(false, false, false, false, false, false, false, false, false, false, false, true);
            Bools(np.strings.isspace(s)).Should().Equal(false, false, false, false, false, false, false, false, false, true, true, false);
            Bools(np.strings.isalnum(s)).Should().Equal(true, true, false, false, true, false, true, false, false, false, false, true);
            Bools(np.strings.islower(s)).Should().Equal(true, true, false, false, false, false, true, true, false, false, false, false);
            Bools(np.strings.isupper(s)).Should().Equal(false, false, false, false, true, false, false, false, false, false, false, false);
            Bools(np.strings.istitle(s)).Should().Equal(false, false, false, false, false, true, false, false, false, false, false, false);
            Action decimalS = () => np.strings.isdecimal(s);
            decimalS.Should().Throw<TypeError>().WithMessage(
                "ufunc 'isdecimal' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None");
            Action numericS = () => np.strings.isnumeric(s);
            numericS.Should().Throw<TypeError>().WithMessage(
                "ufunc 'isnumeric' did not contain a loop with signature matching types <class 'numpy.dtypes.BytesDType'> -> None");
        }

        /// <summary>
        ///     The predicates over <c>StringDType</c> (probed: NumPy 2.4.2): short and long texts, a trailing NUL, a character
        ///     past U+FFFF at the start of a title, a 41-byte text of 20 two-byte letters and an uppercase one.
        /// </summary>
        [TestMethod]
        public void StringDType_MatchesNumPy()
        {
            using var t = np.array(new[]
            {
                "abc", "a1", "", "éß", "ABC", "Abc Def", "abc\0", "\U0001D400bc", new string('a', 40), new string('é', 20) + "X",
            }, np.dtypes.StringDType());
            Bools(np.strings.isalpha(t)).Should().Equal(true, false, false, true, true, false, true, true, true, true);
            Bools(np.strings.islower(t)).Should().Equal(true, true, false, true, false, false, true, false, true, false);
            Bools(np.strings.isupper(t)).Should().Equal(false, false, false, false, true, false, false, false, false, false);
            Bools(np.strings.istitle(t)).Should().Equal(false, false, false, false, false, true, false, true, false, false);
            Bools(np.strings.isspace(t)).Should().Equal(false, false, false, false, false, false, false, false, false, false);
        }

        /// <summary>
        ///     NumPy's NA rule (probed: NumPy 2.4.2): a NaN-like NA is False for every predicate; a string NA reads as its
        ///     text; another NA object refuses with NumPy's ValueError.
        /// </summary>
        [TestMethod]
        public void StringDType_NA_FollowsTheDescriptorsRule()
        {
            using var nan = np.array(new object[] { "abc", double.NaN, "ABC" }, np.dtypes.StringDType(na_object: double.NaN));
            Bools(np.strings.isalpha(nan)).Should().Equal(true, false, true);
            Bools(np.strings.isupper(nan)).Should().Equal(false, false, true);
            using var str = np.array(new object[] { "abc", null }, np.dtypes.StringDType(na_object: "NA"));
            Bools(np.strings.isalpha(str)).Should().Equal(true, true);
            using var none = np.array(new object[] { "abc", null, "ABC" }, np.dtypes.StringDType(na_object: null));
            Action act = () => np.strings.isalpha(none);
            act.Should().Throw<ValueError>().WithMessage("Cannot use the isalpha function with a null that is not a nan-like value");
        }

        /// <summary>
        ///     The ufunc keywords (probed: NumPy 2.4.2): a 0-d operand gives a 0-d bool, <c>where=</c> keeps the masked-off
        ///     slots of <c>out=</c>, an int8 <c>out=</c> receives the bools through the output cast, <c>dtype=bool</c> is
        ///     the loop, and <c>dtype=int64</c> or a non-text operand has no loop.
        /// </summary>
        [TestMethod]
        public void Keywords_MatchNumPy()
        {
            using var zero = np.strings.isdigit(np.array("123"));
            zero.ndim.Should().Be(0);
            zero.GetBoolean().Should().BeTrue();
            using var x = np.array(new[] { "1", "a", "2" });
            using var o = np.ones(new Shape(3), np.bool_);
            using var mask = np.array(new[] { true, false, true });
            np.strings.isdigit(x, o, mask);
            Bools(o).Should().Equal(true, true, true);
            using var i8 = np.zeros(new Shape(2), np.int8);
            using var two = np.array(new[] { "1", "a" });
            np.strings.isdigit(two, i8);
            i8.GetSByte(0).Should().Be(1);
            i8.GetSByte(1).Should().Be(0);
            using var b = np.strings.isdigit(two, dtype: np.bool_);
            Bools(b).Should().Equal(true, false);
            Action wrongDtype = () => np.strings.isdigit(two, dtype: np.int64);
            wrongDtype.Should().Throw<TypeError>().WithMessage(
                "ufunc 'isdigit' did not contain a loop with signature matching types <class 'numpy.dtypes.StrDType'> -> <class 'numpy.dtypes.Int64DType'>");
            using var ints = np.array(new long[] { 1, 2 });
            Action noText = () => np.strings.isdigit(ints);
            noText.Should().Throw<TypeError>().WithMessage(
                "ufunc 'isdigit' did not contain a loop with signature matching types <class 'numpy.dtypes.Int64DType'> -> None");
        }
    }
}
