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
    ///     The element-at-once forms of the <c>numpy.strings</c> loops (strings phase 2, P2.5; strings plan D11 / D12 / D13)
    ///     against the general forms they shortcut: the <c>S</c> / <c>U</c> lane bodies (<c>NDTextElements.*Lanes</c>, AVX2)
    ///     against the scalar bodies (<c>*Scalar</c>) over every width class from 1 unit to past the 64-byte cap, the
    ///     <c>StringDType</c> short-cell forms (<c>NDVStringMethods.*Short</c>) against the general UTF-8 cores, and the IL
    ///     kernels against the managed fallback kernels.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The string oracle (sections <c>strfn</c> / <c>tfn</c>) checks the public functions against NumPy; these tests check
    ///     that the fast forms are the general forms byte for byte on inputs the oracle's operands do not reach: every width,
    ///     embedded and trailing NULs, whitespace of both classes, tabs, line breaks, non-ASCII units, every start / end /
    ///     step form, and short cells whose bytes past the size are garbage. Each output element is surrounded by sentinel
    ///     bytes, so a store past the element fails the test; the <see cref="FencedMemory"/> test places elements against
    ///     inaccessible pages, so a read past one ends the test host.
    ///     </para>
    ///     <para>Every test is deterministic (fixed seeds) and touches no process-wide switch, so they run in parallel.</para>
    /// </remarks>
    [TestClass]
    public unsafe class StringMethodsLanesTests
    {
        /// <summary>Code points the <c>U</c> elements are drawn from: letters, ASCII and non-ASCII whitespace, tabs and line
        ///     breaks, signs, a digit, NUL, a BMP letter and an astral one.</summary>
        private static readonly uint[] UPool =
        {
            'a', 'b', 'c', 'x', ' ', '\t', '\n', '\r', '+', '-', '0', 0, 0x0B, 0x1C, 0x85, 0xA0, 0x2003, 0x2028, 0x3000, 0xE9,
            0x1F600,
        };

        /// <summary>Bytes the <c>S</c> elements are drawn from (the ASCII whitespace class plus bytes it excludes).</summary>
        private static readonly byte[] SPool =
        {
            (byte)'a', (byte)'b', (byte)'c', (byte)'x', (byte)' ', (byte)'\t', (byte)'\n', (byte)'\r', (byte)'+', (byte)'-',
            (byte)'0', 0, 0x0B, 0x0C, 0x1C, 0x85, 0xA0, 0xE9,
        };

        /// <summary>The start / end / stop values every search and slice is tried with (CPython's offset rules).</summary>
        private static readonly long[] Offsets = { 0, 1, 2, 3, 5, 9, 17, 40, 70, -1, -2, -5, -40, -100, long.MaxValue, long.MinValue };

        /// <summary>The slice steps (1 and -1 take the shift without a multiply; a step past int's range selects one unit).</summary>
        private static readonly long[] Steps = { 1, -1, 2, -2, 3, -3, 7, 64, -65, 1L << 40, -(1L << 40), long.MaxValue, long.MinValue };

        /// <summary>Skips a test on hosts without the lane paths.</summary>
        private static void RequireLanes()
        {
            if (!Avx2.IsSupported || !Bmi2.X64.IsSupported)
                Assert.Inconclusive("The lane paths need AVX2 and BMI2; this host runs the scalar bodies only.");
        }

        /// <summary>The element widths (bytes) of one unit size: every width up to past the 64-byte cap.</summary>
        /// <param name="unit">1 (<c>S</c>) or 4 (<c>U</c>).</param>
        /// <returns>The widths.</returns>
        private static IEnumerable<int> Widths(int unit) => unit == 1 ? Enumerable.Range(1, 70) : Enumerable.Range(1, 18).Select(u => u * 4);

        /// <summary>Fills an element with random units: a random length of units from the pool (NULs included), then NULs.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="p">The element.</param>
        /// <param name="width">Its width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        private static void Fill(System.Random rng, byte* p, int width, int unit)
        {
            int units = width / unit;
            int len = rng.Next(units + 1);
            for (int i = 0; i < units; i++)
            {
                if (unit == 1)
                    p[i] = i < len ? SPool[rng.Next(SPool.Length)] : (byte)0;
                else
                    ((uint*)p)[i] = i < len ? UPool[rng.Next(UPool.Length)] : 0u;
            }
        }

        /// <summary>
        ///     Fills a needle: usually a run of the haystack's units (so matches occur), else random units; its own width is 1
        ///     to 6 units, with trailing NULs past a random length.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="hay">The haystack element.</param>
        /// <param name="hayWidth">Its width in bytes.</param>
        /// <param name="unit">1 or 4.</param>
        /// <param name="needle">Receives the needle bytes.</param>
        private static void Needle(System.Random rng, byte* hay, int hayWidth, int unit, out byte[] needle)
        {
            int units = 1 + rng.Next(6);
            needle = new byte[units * unit];
            int len = rng.Next(units + 1);
            int hayUnits = hayWidth / unit;
            int from = rng.Next(hayUnits);
            bool copy = rng.Next(3) != 0;
            for (int i = 0; i < len; i++)
            {
                uint c = copy && from + i < hayUnits
                    ? (unit == 1 ? hay[from + i] : ((uint*)hay)[from + i])
                    : (unit == 1 ? SPool[rng.Next(SPool.Length)] : UPool[rng.Next(UPool.Length)]);
                if (unit == 1)
                    needle[i] = (byte)c;
                else
                    BitConverter.TryWriteBytes(needle.AsSpan(i * 4), c);
            }
        }

        /// <summary>An output buffer with 32 sentinel bytes on each side.</summary>
        private sealed class Guarded
        {
            /// <summary>The sentinel byte.</summary>
            private const byte Sentinel = 0xCD;

            /// <summary>The margin on each side.</summary>
            private const int Margin = 32;

            /// <summary>The backing bytes.</summary>
            private readonly byte[] _bytes;

            /// <summary>The element width.</summary>
            internal readonly int Width;

            /// <summary>Creates a guarded element of <paramref name="width"/> bytes, pre-filled with a junk pattern.</summary>
            /// <param name="width">The element width.</param>
            internal Guarded(int width)
            {
                Width = width;
                _bytes = new byte[width + 2 * Margin];
                _bytes.AsSpan().Fill(Sentinel);
                // Junk inside the element: a body must write every byte of its output element.
                for (int i = 0; i < width; i++)
                    _bytes[Margin + i] = (byte)(0x5A ^ i);
            }

            /// <summary>Runs <paramref name="body"/> with the element's address.</summary>
            /// <param name="body">The writer.</param>
            internal void Use(Action<IntPtr> body)
            {
                fixed (byte* b = _bytes)
                    body((IntPtr)(b + Margin));
            }

            /// <summary>The element's bytes, after checking that both margins are untouched.</summary>
            /// <returns>The element.</returns>
            internal byte[] Element()
            {
                for (int i = 0; i < Margin; i++)
                {
                    if (_bytes[i] != Sentinel || _bytes[Margin + Width + i] != Sentinel)
                        throw new AssertFailedException($"a store left the {Width}-byte output element (margin byte {i})");
                }
                return _bytes.AsSpan(Margin, Width).ToArray();
            }
        }

        /// <summary>Formats an element for a failure message.</summary>
        /// <param name="bytes">The element bytes.</param>
        /// <returns>Hex.</returns>
        private static string Hex(byte[] bytes) => Convert.ToHexString(bytes);

        // =============================================================================================================
        // S / U: the lane bodies against the scalar bodies
        // =============================================================================================================

        /// <summary>
        ///     find / rfind / count / startswith / endswith: the lane bodies equal the span bodies for every width with lane
        ///     masks (1 to 64 units), every needle kind and every start / end form.
        /// </summary>
        [TestMethod]
        public void SearchFamily_LanesEqualScalar_EveryWidth()
        {
            RequireLanes();
            SearchFamily<byte>(1);
            SearchFamily<uint>(4);
        }

        /// <summary>The body of <see cref="SearchFamily_LanesEqualScalar_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void SearchFamily<TUnit>(int unit) where TUnit : unmanaged, IEquatable<TUnit>
        {
            var rng = new System.Random(2026 + unit);
            foreach (int w in Widths(unit).Concat(unit == 4 ? new[] { 160, 256 } : Array.Empty<int>()))
            {
                if (!NDTextElements.MasksFit<TUnit>(w))
                    continue;
                var hay = new byte[w];
                for (int trial = 0; trial < 120; trial++)
                {
                    fixed (byte* h = hay)
                    {
                        Fill(rng, h, w, unit);
                        Needle(rng, h, w, unit, out var sub);
                        fixed (byte* n = sub)
                        {
                            long s = Offsets[rng.Next(Offsets.Length)], e = Offsets[rng.Next(Offsets.Length)];
                            if (rng.Next(3) == 0)
                            {
                                s = 0;
                                e = long.MaxValue;
                            }
                            string ctx = $"w={w} unit={unit} hay={Hex(hay)} sub={Hex(sub)} start={s} end={e}";
                            Assert.AreEqual(NDTextElements.FindScalar<TUnit>(h, w, n, sub.Length, s, e),
                                            NDTextElements.FindLanes<TUnit>(h, w, n, sub.Length, s, e), "find " + ctx);
                            Assert.AreEqual(NDTextElements.RFindScalar<TUnit>(h, w, n, sub.Length, s, e),
                                            NDTextElements.RFindLanes<TUnit>(h, w, n, sub.Length, s, e), "rfind " + ctx);
                            Assert.AreEqual(NDTextElements.CountScalar<TUnit>(h, w, n, sub.Length, s, e),
                                            NDTextElements.CountLanes<TUnit>(h, w, n, sub.Length, s, e), "count " + ctx);
                            foreach (bool back in new[] { false, true })
                                Assert.AreEqual(NDTextElements.TailMatchScalar<TUnit>(h, w, n, sub.Length, s, e, back),
                                                NDTextElements.TailMatchLanes<TUnit>(h, w, n, sub.Length, s, e, back),
                                                (back ? "endswith " : "startswith ") + ctx);
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     startswith / endswith over a range from a non-negative start to the element's end (NumPy's default
        ///     <c>end=None</c>): the one-compare body (<see cref="NDTextElements.TailMatchWholeRange{TUnit}"/>, which reads no
        ///     length for startswith) equals the span body for every width (past the 64-unit lane cap too), every needle kind
        ///     (empty, longer than the element, NULs inside), every start from 0 to past the width, and an end of exactly
        ///     the width as well as <c>long.MaxValue</c>.
        /// </summary>
        [TestMethod]
        public void TailMatchWholeRange_EqualsTheSpanBody_EveryWidth()
        {
            WholeRange<byte>(1);
            WholeRange<uint>(4);
        }

        /// <summary>The body of <see cref="TailMatchWholeRange_EqualsTheSpanBody_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void WholeRange<TUnit>(int unit) where TUnit : unmanaged, IEquatable<TUnit>
        {
            var rng = new System.Random(4026 + unit);
            foreach (int w in Widths(unit).Concat(unit == 4 ? new[] { 160, 256, 400 } : new[] { 100, 200 }))
            {
                var hay = new byte[w];
                int units = w / unit;
                for (int trial = 0; trial < 120; trial++)
                {
                    fixed (byte* h = hay)
                    {
                        Fill(rng, h, w, unit);
                        Needle(rng, h, w, unit, out var sub);
                        fixed (byte* n = sub)
                        {
                            // 0 a quarter of the time (the default), else any start up to two units past the width.
                            long s = rng.Next(4) == 0 ? 0 : rng.Next(units + 3);
                            foreach (long e in new[] { (long)units, long.MaxValue })
                            {
                                foreach (bool back in new[] { false, true })
                                    Assert.AreEqual(NDTextElements.TailMatchScalar<TUnit>(h, w, n, sub.Length, s, e, back),
                                                    NDTextElements.TailMatchWholeRange<TUnit>(h, w, n, sub.Length, s, back),
                                                    $"{(back ? "endswith" : "startswith")} w={w} unit={unit} hay={Hex(hay)} " +
                                                    $"sub={Hex(sub)} start={s} end={e}");
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     strip / lstrip / rstrip (whitespace and chars) and slice: the lane bodies write the scalar bodies' output
        ///     element byte for byte (zero fill included) and nothing past it, for every width up to 64 bytes.
        /// </summary>
        [TestMethod]
        public void StripAndSlice_LanesEqualScalar_EveryWidth()
        {
            RequireLanes();
            StripAndSlice<byte>(1);
            StripAndSlice<uint>(4);
        }

        /// <summary>The body of <see cref="StripAndSlice_LanesEqualScalar_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void StripAndSlice<TUnit>(int unit) where TUnit : unmanaged, IEquatable<TUnit>
        {
            var rng = new System.Random(3026 + unit);
            foreach (int w in Widths(unit))
            {
                if (!NDTextElements.SelectsFit(w))
                    continue;
                var src = new byte[w];
                for (int trial = 0; trial < 60; trial++)
                {
                    fixed (byte* pf = src)
                    {
                        // A fixed local cannot be captured by a lambda; a copy of the pointer can.
                        byte* p = pf;
                        Fill(rng, p, w, unit);
                        Needle(rng, p, w, unit, out var chars);
                        string ctx = $"w={w} unit={unit} src={Hex(src)}";
                        for (int type = 0; type < 3; type++)
                        {
                            int t = type;
                            Compare(w, d => NDTextElements.StripWhitespaceScalar<TUnit>(p, w, d, t),
                                    d => NDTextElements.StripWhitespaceLanes<TUnit>(p, w, d, t), $"strip type {t} {ctx}");
                            fixed (byte* cf = chars)
                            {
                                byte* cc = cf;
                                int cw = chars.Length;
                                Compare(w, d => NDTextElements.StripCharsScalar<TUnit>(p, w, cc, cw, d, t),
                                        d => NDTextElements.StripCharsLanes<TUnit>(p, w, cc, cw, d, t),
                                        $"strip chars {Hex(chars)} type {t} {ctx}");
                            }
                        }
                        for (int k = 0; k < 6; k++)
                        {
                            long start = Offsets[rng.Next(Offsets.Length)], stop = Offsets[rng.Next(Offsets.Length)];
                            long step = Steps[rng.Next(Steps.Length)];
                            Compare(w, d => NDTextElements.SliceScalar<TUnit>(p, w, start, stop, step, d),
                                    d => NDTextElements.SliceLanes<TUnit>(p, w, start, stop, step, d),
                                    $"slice [{start}:{stop}:{step}] {ctx}");
                        }
                    }
                }
            }
        }

        /// <summary>The set units the wide chars strip tests are built from: letters, a space, a bar, a non-ASCII space, NUL.</summary>
        private static readonly uint[] WideSetPool = { 'a', ' ', '|', 0x3000, 0 };

        /// <summary>
        ///     A <c>U</c> element of <paramref name="units"/> units for the wide chars strip: a text of a random length whose
        ///     first and last runs (each of a random length, the whole text included) are units of <paramref name="set"/> and
        ///     whose middle comes from the pool, then trailing NULs. A set holding NUL makes NULs inside the stripped runs.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="p">The element.</param>
        /// <param name="units">Its width in units.</param>
        /// <param name="set">The set units.</param>
        private static void FillWideStrip(System.Random rng, uint* p, int units, uint[] set)
        {
            int len = rng.Next(units + 1);
            int lead = rng.Next(len + 1);
            int trail = rng.Next(len - lead + 1);
            for (int i = 0; i < units; i++)
                p[i] = i >= len ? 0u
                       : i < lead || i >= len - trail ? set[rng.Next(set.Length)]
                       : UPool[rng.Next(UPool.Length)];
        }

        /// <summary>A strip set of 1 to 5 units from <see cref="WideSetPool"/>, in an element with 0 to 2 trailing NUL units.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="set">Receives the set units.</param>
        /// <returns>The set element's bytes.</returns>
        private static byte[] WideSet(System.Random rng, out uint[] set)
        {
            set = new uint[1 + rng.Next(5)];
            var bytes = new byte[(set.Length + rng.Next(3)) * 4];
            for (int i = 0; i < set.Length; i++)
            {
                set[i] = WideSetPool[rng.Next(WideSetPool.Length)];
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), set[i]);
            }
            return bytes;
        }

        /// <summary>
        ///     The wide chars strip (<see cref="NDTextElements.StripCharsWide{TUnit}"/>: a <c>U</c> element of 17 to 64 units,
        ///     past the select helpers' 64 bytes) writes the scalar body's output element byte for byte, zero fill included,
        ///     for every width and strip type: stripped runs of every length from each end (inside one chunk, across chunk
        ///     boundaries, the whole text), a set holding NUL (NULs inside the stripped runs and among the trailing NULs the
        ///     text's length excludes), a set element with trailing NULs.
        /// </summary>
        [TestMethod]
        public void StripCharsWide_EqualsScalar_EveryWideWidth()
        {
            RequireLanes();
            var rng = new System.Random(3031);
            for (int units = 17; units <= NDTextElements.LaneUnits; units++)
            {
                int w = units * 4;
                var src = new byte[w];
                for (int trial = 0; trial < 60; trial++)
                {
                    var chars = WideSet(rng, out var set);
                    fixed (byte* pf = src)
                    fixed (byte* cf = chars)
                    {
                        FillWideStrip(rng, (uint*)pf, units, set);
                        // Fixed locals cannot be captured by a lambda; copies of the pointers can.
                        byte* p = pf, cc = cf;
                        int cw = chars.Length;
                        for (int t = 0; t < 3; t++)
                        {
                            int type = t;
                            Compare(w, d => NDTextElements.StripCharsScalar<uint>(p, w, cc, cw, d, type),
                                    d => NDTextElements.StripCharsWide<uint>(p, w, cc, cw, d, type),
                                    $"wide strip chars {Hex(chars)} type {type} w={w} src={Hex(src)}");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     The wide chars strip over elements that end at (or start right after) an inaccessible page, for every wide
        ///     width: its chunk reads stay inside <c>[src, src + width)</c> (an out-of-element read ends the test host), and
        ///     its output equals the scalar body's.
        /// </summary>
        [TestMethod]
        public void StripCharsWide_StaysInsideItsElements()
        {
            RequireLanes();
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(7031);
            using var inFence = new FencedMemory(pages: 1);
            using var setFence = new FencedMemory(pages: 1);
            using var outFence = new FencedMemory(pages: 1);
            var scratch = new byte[NDTextElements.LaneUnits * 4];
            for (int units = 17; units <= NDTextElements.LaneUnits; units++)
            {
                int w = units * 4;
                foreach (bool atEnd in new[] { true, false })
                {
                    for (int trial = 0; trial < 8; trial++)
                    {
                        var chars = WideSet(rng, out var set);
                        byte* p = inFence.Place(w, atEnd);
                        FillWideStrip(rng, (uint*)p, units, set);
                        byte* s = setFence.Place(chars.Length, atEnd);
                        chars.AsSpan().CopyTo(new Span<byte>(s, chars.Length));
                        int sw = chars.Length;
                        byte* d = outFence.Place(w, atEnd);
                        for (int t = 0; t < 3; t++)
                        {
                            NDTextElements.StripCharsWide<uint>(p, w, s, sw, d, t);
                            AssertSame(w, d, scratch, x => NDTextElements.StripCharsScalar<uint>(p, w, s, sw, x, t), "wide strip chars");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     center / ljust / rjust and zfill: the lane bodies write the scalar bodies' output element, for every input width
        ///     up to 64 bytes and output widths from the input's up to 64 bytes (an output wider than the result included).
        /// </summary>
        [TestMethod]
        public void PadAndZFill_LanesEqualScalar_EveryWidth()
        {
            RequireLanes();
            PadAndZFill<byte>(1);
            PadAndZFill<uint>(4);
        }

        /// <summary>The body of <see cref="PadAndZFill_LanesEqualScalar_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void PadAndZFill<TUnit>(int unit) where TUnit : unmanaged
        {
            var rng = new System.Random(4026 + unit);
            int maxUnits = 64 / unit;
            foreach (int w in Widths(unit))
            {
                if (!NDTextElements.SelectsFit(w))
                    continue;
                var src = new byte[w];
                var fill = new byte[unit];
                for (int trial = 0; trial < 40; trial++)
                {
                    fixed (byte* pf = src)
                    fixed (byte* ff = fill)
                    {
                        byte* p = pf, f = ff;
                        Fill(rng, p, w, unit);
                        // The fill character, then a requested width: below, at and above the length, negative too.
                        Fill(rng, f, unit, unit);
                        if (unit == 1)
                            f[0] = f[0] == 0 ? (byte)'*' : f[0];
                        else if (*(uint*)f == 0)
                            *(uint*)f = '*';
                        int units = w / unit;
                        long padWidth = rng.Next(-3, maxUnits + 1);
                        int outUnits = (int)System.Math.Min(maxUnits, System.Math.Max(units, padWidth) + rng.Next(3));
                        int ow = System.Math.Max(outUnits, units) * unit;
                        if (ow > 64)
                            continue;
                        string ctx = $"w={w} unit={unit} out={ow} width={padWidth} src={Hex(src)}";
                        for (int pos = 0; pos < 3; pos++)
                        {
                            int ps = pos;
                            byte* fp = f;
                            Compare(ow, d => NDTextElements.PadScalar<TUnit>(p, w, padWidth, fp, d, ow, ps),
                                    d => NDTextElements.PadLanes<TUnit>(p, w, padWidth, NDTextElementsUnit<TUnit>(fp), d, ow, ps),
                                    $"pad pos {ps} {ctx}");
                        }
                        Compare(ow, d => NDTextElements.ZFillCore<TUnit>(p, w, padWidth, d, ow, false),
                                d => NDTextElements.ZFillCore<TUnit>(p, w, padWidth, d, ow, true), $"zfill {ctx}");
                    }
                }
            }
        }

        /// <summary>The first unit of an element as the pad bodies' fill character.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="p">The element.</param>
        /// <returns>The unit.</returns>
        private static uint NDTextElementsUnit<TUnit>(byte* p) where TUnit : unmanaged
            => typeof(TUnit) == typeof(byte) ? p[0] : *(uint*)p;

        /// <summary>
        ///     partition / rpartition: the lane body writes the scalar body's three parts, for every width up to 64 bytes and
        ///     every separator kind.
        /// </summary>
        [TestMethod]
        public void Partition_LanesEqualScalar_EveryWidth()
        {
            RequireLanes();
            Partition<byte>(1);
            Partition<uint>(4);
        }

        /// <summary>The body of <see cref="Partition_LanesEqualScalar_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void Partition<TUnit>(int unit) where TUnit : unmanaged, IEquatable<TUnit>
        {
            var rng = new System.Random(5026 + unit);
            foreach (int w in Widths(unit))
            {
                if (!NDTextElements.SelectsFit(w))
                    continue;
                var src = new byte[w];
                for (int trial = 0; trial < 60; trial++)
                {
                    fixed (byte* pf = src)
                    {
                        byte* p = pf;
                        Fill(rng, p, w, unit);
                        Needle(rng, p, w, unit, out var sep);
                        // The loop never sees an empty separator (the wrapper raises first).
                        if (sep.All(b => b == 0))
                            sep[0] = (byte)'a';
                        fixed (byte* sf = sep)
                        {
                            byte* sp = sf;
                            int sw = sep.Length;
                            foreach (bool back in new[] { false, true })
                            {
                                var a1 = new Guarded(w);
                                var a2 = new Guarded(sw);
                                var a3 = new Guarded(w);
                                var b1 = new Guarded(w);
                                var b2 = new Guarded(sw);
                                var b3 = new Guarded(w);
                                a1.Use(o1 => a2.Use(o2 => a3.Use(o3 =>
                                    NDTextElements.PartitionScalar<TUnit>(p, w, sp, sw, back, (byte*)o1, w, (byte*)o2, sw, (byte*)o3, w))));
                                b1.Use(o1 => b2.Use(o2 => b3.Use(o3 =>
                                    NDTextElements.PartitionLanes<TUnit>(p, w, sp, sw, back, (byte*)o1, w, (byte*)o2, sw, (byte*)o3, w))));
                                string ctx = $"w={w} unit={unit} back={back} src={Hex(src)} sep={Hex(sep)}";
                                CollectionAssert.AreEqual(a1.Element(), b1.Element(), "before " + ctx);
                                CollectionAssert.AreEqual(a2.Element(), b2.Element(), "separator " + ctx);
                                CollectionAssert.AreEqual(a3.Element(), b3.Element(), "after " + ctx);
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     expandtabs: the lane size body equals the scalar one (the overflow answer included) and the lane fill writes the
        ///     scalar fill's output, over elements with and without line breaks and every tab size kind.
        /// </summary>
        [TestMethod]
        public void ExpandTabs_LanesEqualScalar_EveryWidth()
        {
            RequireLanes();
            ExpandTabs<byte>(1);
            ExpandTabs<uint>(4);
        }

        /// <summary>The body of <see cref="ExpandTabs_LanesEqualScalar_EveryWidth"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void ExpandTabs<TUnit>(int unit) where TUnit : unmanaged
        {
            var rng = new System.Random(6026 + unit);
            long[] tabsizes = { 8, 4, 3, 1, 0, -2, 5, 16, 1L << 40, long.MaxValue };
            foreach (int w in Widths(unit))
            {
                if (!NDTextElements.MasksFit<TUnit>(w))
                    continue;
                var src = new byte[w];
                for (int trial = 0; trial < 60; trial++)
                {
                    fixed (byte* pf = src)
                    {
                        byte* p = pf;
                        Fill(rng, p, w, unit);
                        long tabsize = tabsizes[rng.Next(tabsizes.Length)];
                        long scalar = NDTextElements.ExpandTabsLengthScalar<TUnit>(p, w, tabsize);
                        string ctx = $"w={w} unit={unit} tabsize={tabsize} src={Hex(src)}";
                        Assert.AreEqual(scalar, NDTextElements.ExpandTabsLengthLanes<TUnit>(p, w, tabsize), "length " + ctx);
                        // The fill writes an output of the size pass's width (NumPy's units: U sized four times wide).
                        if (scalar < 0 || scalar * unit > 4096)
                            continue;
                        int ow = (int)System.Math.Max(1, scalar) * unit;
                        Compare(ow, d => NDTextElements.ExpandTabsScalar<TUnit>(p, w, tabsize, d, ow),
                                d => NDTextElements.ExpandTabs<TUnit>(p, w, tabsize, d, ow), "fill " + ctx);
                    }
                }
            }
        }

        /// <summary>Runs a scalar and a lane body into two guarded outputs and compares the elements.</summary>
        /// <param name="width">The output width.</param>
        /// <param name="scalar">The scalar body.</param>
        /// <param name="lanes">The lane body.</param>
        /// <param name="ctx">The failure context.</param>
        private static void Compare(int width, BytePointerAction scalar, BytePointerAction lanes, string ctx)
            => CompareCore(width, d => scalar((byte*)d), d => lanes((byte*)d), ctx);

        /// <summary>A body writing one output element.</summary>
        /// <param name="dst">The output element.</param>
        private delegate void BytePointerAction(byte* dst);

        /// <summary>The comparison behind both <c>Compare</c> overloads.</summary>
        /// <param name="width">The output width.</param>
        /// <param name="scalar">The scalar body.</param>
        /// <param name="lanes">The lane body.</param>
        /// <param name="ctx">The failure context.</param>
        private static void CompareCore(int width, Action<IntPtr> scalar, Action<IntPtr> lanes, string ctx)
        {
            var a = new Guarded(width);
            var b = new Guarded(width);
            a.Use(scalar);
            b.Use(lanes);
            var ea = a.Element();
            var eb = b.Element();
            if (!ea.AsSpan().SequenceEqual(eb))
                Assert.Fail($"{ctx}: scalar {Hex(ea)} lanes {Hex(eb)}");
        }

        // =============================================================================================================
        // Fenced memory: no lane body reads or writes outside its elements
        // =============================================================================================================

        /// <summary>
        ///     Every lane body over elements that end at (or start right after) an inaccessible page, for every width: an
        ///     out-of-element read or write ends the test host; the values are checked against the scalar bodies.
        /// </summary>
        [TestMethod]
        public void LaneBodies_StayInsideTheirElements()
        {
            RequireLanes();
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            Fenced<byte>(1);
            Fenced<uint>(4);
        }

        /// <summary>The body of <see cref="LaneBodies_StayInsideTheirElements"/> for one unit type.</summary>
        /// <typeparam name="TUnit">The code unit.</typeparam>
        /// <param name="unit">Its size.</param>
        private static void Fenced<TUnit>(int unit) where TUnit : unmanaged, IEquatable<TUnit>
        {
            var rng = new System.Random(7026 + unit);
            using var inFence = new FencedMemory(pages: 1);
            using var subFence = new FencedMemory(pages: 1);
            using var outFence = new FencedMemory(pages: 1);
            var scratch = new byte[64];
            foreach (int w in Widths(unit))
            {
                if (!NDTextElements.SelectsFit(w))
                    continue;
                foreach (bool atEnd in new[] { true, false })
                {
                    byte* p = inFence.Place(w, atEnd);
                    Fill(rng, p, w, unit);
                    int sw = unit;
                    byte* s = subFence.Place(sw, atEnd);
                    Fill(rng, s, sw, unit);
                    if (unit == 1 ? s[0] == 0 : *(uint*)s == 0)
                    {
                        if (unit == 1)
                            s[0] = (byte)'a';
                        else
                            *(uint*)s = 'a';
                    }
                    byte* d = outFence.Place(w, atEnd);
                    _ = NDTextElements.FindLanes<TUnit>(p, w, s, sw, 0, long.MaxValue);
                    _ = NDTextElements.RFindLanes<TUnit>(p, w, s, sw, 0, long.MaxValue);
                    _ = NDTextElements.CountLanes<TUnit>(p, w, s, sw, 0, long.MaxValue);
                    _ = NDTextElements.TailMatchLanes<TUnit>(p, w, s, sw, 0, long.MaxValue, true);
                    _ = NDTextElements.TailMatchWholeRange<TUnit>(p, w, s, sw, 0, false);
                    _ = NDTextElements.TailMatchWholeRange<TUnit>(p, w, s, sw, 0, true);
                    _ = NDTextElements.Units<TUnit>(p, w);
                    _ = NDTextElements.ExpandTabsLengthLanes<TUnit>(p, w, 8);
                    for (int t = 0; t < 3; t++)
                    {
                        NDTextElements.StripWhitespaceLanes<TUnit>(p, w, d, t);
                        AssertSame(w, d, scratch, x => NDTextElements.StripWhitespaceScalar<TUnit>(p, w, x, t), "strip");
                        NDTextElements.StripCharsLanes<TUnit>(p, w, s, sw, d, t);
                        AssertSame(w, d, scratch, x => NDTextElements.StripCharsScalar<TUnit>(p, w, s, sw, x, t), "strip chars");
                    }
                    NDTextElements.SliceLanes<TUnit>(p, w, 0, long.MaxValue, -1, d);
                    AssertSame(w, d, scratch, x => NDTextElements.SliceScalar<TUnit>(p, w, 0, long.MaxValue, -1, x), "slice");
                    NDTextElements.PadLanes<TUnit>(p, w, w / unit, '*', d, w, NDTextElements.Center);
                    NDTextElements.ZFillCore<TUnit>(p, w, w / unit, d, w, true);
                    NDTextElements.PartitionLanes<TUnit>(p, w, s, sw, false, d, w, d, w, d, w);
                    long len = NDTextElements.ExpandTabsLengthLanes<TUnit>(p, w, 1);
                    if (len >= 0 && len * unit <= 4096 - 64)
                    {
                        byte* e = outFence.Place(System.Math.Max(1, len) * unit, atEnd);
                        NDTextElements.ExpandTabs<TUnit>(p, w, 1, e, (int)System.Math.Max(1, len) * unit);
                    }
                }
            }
        }

        /// <summary>Checks a fenced output against the scalar body's output written to ordinary memory.</summary>
        /// <param name="w">The width.</param>
        /// <param name="fenced">The fenced output.</param>
        /// <param name="scratch">Ordinary memory for the scalar output.</param>
        /// <param name="scalar">The scalar body.</param>
        /// <param name="what">The operation.</param>
        private static void AssertSame(int w, byte* fenced, byte[] scratch, BytePointerAction scalar, string what)
        {
            fixed (byte* x = scratch)
            {
                scalar(x);
                if (!new ReadOnlySpan<byte>(x, w).SequenceEqual(new ReadOnlySpan<byte>(fenced, w)))
                    Assert.Fail($"{what} w={w}: scalar {Convert.ToHexString(new ReadOnlySpan<byte>(x, w))} lanes " +
                                Convert.ToHexString(new ReadOnlySpan<byte>(fenced, w)));
            }
        }

        // =============================================================================================================
        // StringDType: the short-cell forms against the general UTF-8 cores
        // =============================================================================================================

        /// <summary>The code points the short cells are made of (1- to 4-byte UTF-8, NUL included).</summary>
        private static readonly string[] CellPool = { "a", "b", "c", "\0", " ", "é", "中", "😀", "ab" };

        /// <summary>
        ///     Builds a short cell: UTF-8 text of at most 15 bytes, the bytes past it junk (a short cell's padding is not
        ///     guaranteed zero), byte 15 the short flags and the size.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="cell">The 16-byte cell.</param>
        /// <returns>The text size.</returns>
        private static int ShortCell(System.Random rng, byte* cell)
        {
            var text = new StringBuilder();
            int target = rng.Next(16);
            while (true)
            {
                string piece = CellPool[rng.Next(CellPool.Length)];
                if (Encoding.UTF8.GetByteCount(text + piece) > target)
                    break;
                text.Append(piece);
            }
            var bytes = Encoding.UTF8.GetBytes(text.ToString());
            for (int i = 0; i < 15; i++)
                cell[i] = i < bytes.Length ? bytes[i] : (byte)rng.Next(1, 256);
            cell[15] = (byte)(VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA | bytes.Length);
            return bytes.Length;
        }

        /// <summary>
        ///     str_len / find / rfind / count / startswith / endswith over short cells: the vector forms equal the general
        ///     UTF-8 cores (code points without trailing NULs, full-byte needles, the one-byte cases, start / end in code
        ///     points), with junk past each cell's size.
        /// </summary>
        [TestMethod]
        public void ShortCells_EqualTheGeneralCores()
        {
            RequireLanes();
            var rng = new System.Random(8026);
            byte* h = stackalloc byte[16];
            byte* n = stackalloc byte[16];
            for (int trial = 0; trial < 20000; trial++)
            {
                int hs = ShortCell(rng, h);
                int ns = ShortCell(rng, n);
                if (rng.Next(3) == 0 && hs > 0)
                {
                    // A needle cut from the haystack (with its trailing junk replaced), so matches occur.
                    int from = rng.Next(hs), take = rng.Next(1, hs - from + 1);
                    for (int i = 0; i < 15; i++)
                        n[i] = i < take ? h[from + i] : (byte)0x77;
                    n[15] = (byte)(VStringAllocator.NPY_STRING_INITIALIZED | VStringAllocator.NPY_STRING_OUTSIDE_ARENA | take);
                    ns = take;
                }
                Assert.AreEqual(NDVStringMethods.CodePoints(h, hs), NDVStringMethods.ShortCodePoints(h, hs), "str_len");
                long s = Offsets[rng.Next(Offsets.Length)], e = Offsets[rng.Next(Offsets.Length)];
                if (rng.Next(2) == 0)
                {
                    s = 0;
                    e = long.MaxValue;
                }
                string ctx = $"h={Convert.ToHexString(new ReadOnlySpan<byte>(h, 16))} n={Convert.ToHexString(new ReadOnlySpan<byte>(n, 16))} start={s} end={e}";
                for (int mode = 0; mode < 3; mode++)
                    Assert.AreEqual(NDVStringMethods.FindCore(h, hs, n, ns, s, e, mode), NDVStringMethods.FindShort(h, n, s, e, mode),
                                    $"mode {mode} {ctx}");
                foreach (bool back in new[] { false, true })
                    Assert.AreEqual(NDVStringMethods.TailMatchCore(h, hs, n, ns, s, e, back), NDVStringMethods.TailMatchShort(h, n, s, e, back),
                                    $"tailmatch back={back} {ctx}");
            }
        }

        // =============================================================================================================
        // The IL kernels against the managed fallback kernels
        // =============================================================================================================

        /// <summary>
        ///     The IL shell's kernels (constant widths, inlined bodies) against the managed kernels the hosts without dynamic
        ///     code run (the same bodies with run-time widths), over a 300-element contiguous operand with broadcast
        ///     parameters: the outputs are equal byte for byte.
        /// </summary>
        [TestMethod]
        public void ILKernels_EqualManagedKernels()
        {
            if (!ILKernelGenerator.TextKernelsAvailable)
                Assert.Inconclusive("No dynamic code on this host: the managed kernels are the only ones.");
            foreach (int unit in new[] { 1, 4 })
            {
                foreach (int units in new[] { 1, 3, 8, 10, 16, 40 })
                {
                    int w = units * unit;
                    // find / count / endswith: (hay, text sub, i64 start = 1, i64 end = -1, out)
                    RunPair(new TextMethodKey(TextMethodOp.Find, unit, w, 2 * unit), unit, w,
                            new[] { Text(2 * unit, unit, "ab"), Int64(1), Int64(-1) }, 8);
                    RunPair(new TextMethodKey(TextMethodOp.Count, unit, w, unit), unit, w,
                            new[] { Text(unit, unit, "a"), Int64(0), Int64(long.MaxValue) }, 8);
                    RunPair(new TextMethodKey(TextMethodOp.EndsWith, unit, w, 2 * unit), unit, w,
                            new[] { Text(2 * unit, unit, "ab"), Int64(1), Int64(-1) }, 1);
                    // strip: (text, out of the input width)
                    RunPair(new TextMethodKey(TextMethodOp.StripWhitespace, unit, w, w, Extra: NDTextElements.BothStrip), unit, w,
                            Array.Empty<byte[]>(), w);
                    // slice: (text, i64 start, i64 stop, i64 step, out)
                    RunPair(new TextMethodKey(TextMethodOp.Slice, unit, w, w), unit, w, new[] { Int64(-2), Int64(0), Int64(-1) }, w);
                    // center: (text, i64 width, text fill, out of up to 64 bytes)
                    int ow = System.Math.Max(w, System.Math.Min(64, (units + 4) * unit));
                    RunPair(new TextMethodKey(TextMethodOp.Pad, unit, w, unit, ow, Extra: NDTextElements.Center), unit, w,
                            new[] { Int64(ow / unit), Text(unit, unit, "*") }, ow);
                }
            }
        }

        /// <summary>
        ///     Runs one key's IL and managed kernels over the same random operands (the text operand contiguous, the others
        ///     broadcast) and compares their output buffers.
        /// </summary>
        /// <param name="key">The kernel key.</param>
        /// <param name="unit">The code unit size.</param>
        /// <param name="w">The text element width.</param>
        /// <param name="parameters">The parameter operands' single elements (broadcast: stride 0).</param>
        /// <param name="outWidth">The output element width.</param>
        private static void RunPair(TextMethodKey key, int unit, int w, byte[][] parameters, int outWidth)
        {
            const int count = 300;
            var rng = new System.Random(9026 + (int)key.Op * 31 + w);
            var text = new byte[count * w];
            fixed (byte* t = text)
            {
                for (int i = 0; i < count; i++)
                    Fill(rng, t + (long)i * w, w, unit);
            }
            var il = ILKernelGenerator.GetTextMethodKernel(key);
            var managed = NDTextLoop.ManagedKernel(key, masked: false);
            Assert.IsNotNull(il, $"no IL kernel for {key}");
            byte[] Run(NDInnerLoopFunc kernel)
            {
                var output = new byte[count * outWidth];
                int nop = 2 + parameters.Length;
                var handles = new List<System.Runtime.InteropServices.GCHandle>();
                try
                {
                    void** ptrs = stackalloc void*[nop];
                    long* strides = stackalloc long[nop];
                    var all = new[] { text }.Concat(parameters).Append(output).ToArray();
                    for (int k = 0; k < nop; k++)
                    {
                        var hnd = System.Runtime.InteropServices.GCHandle.Alloc(all[k], System.Runtime.InteropServices.GCHandleType.Pinned);
                        handles.Add(hnd);
                        ptrs[k] = (void*)hnd.AddrOfPinnedObject();
                        strides[k] = k == 0 ? w : k == nop - 1 ? outWidth : 0;
                    }
                    kernel(ptrs, strides, count, null);
                }
                finally
                {
                    foreach (var hnd in handles)
                        hnd.Free();
                }
                return output;
            }
            CollectionAssert.AreEqual(Run(managed), Run(il), $"IL vs managed: {key}");
        }

        /// <summary>A text parameter element of <paramref name="width"/> bytes holding <paramref name="text"/>.</summary>
        /// <param name="width">The element width.</param>
        /// <param name="unit">The code unit size.</param>
        /// <param name="text">ASCII text (truncated to the width).</param>
        /// <returns>The element bytes.</returns>
        private static byte[] Text(int width, int unit, string text)
        {
            var e = new byte[width];
            for (int i = 0; i < width / unit && i < text.Length; i++)
            {
                if (unit == 1)
                    e[i] = (byte)text[i];
                else
                    BitConverter.TryWriteBytes(e.AsSpan(i * 4), (uint)text[i]);
            }
            return e;
        }

        /// <summary>An int64 parameter element.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The element bytes.</returns>
        private static byte[] Int64(long value) => BitConverter.GetBytes(value);
    }
}
