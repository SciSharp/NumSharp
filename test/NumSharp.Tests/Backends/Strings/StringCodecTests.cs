using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NumPy's <c>np.strings.encode</c> / <c>np.strings.decode</c> (strings phase 2, P2.6 milestone 4): the fast runs —
    ///     the vector inner runs over <c>U</c>, <c>S</c> and <c>StringDType</c> elements, the inline error handlers they write,
    ///     the namereplace run — against the general route (CPython's codecs and handlers in <see cref="NDCodec"/>, which the
    ///     <c>strcodec</c> / <c>tcodec</c> oracle sections gate), and the pieces the oracle cannot reach: memory safety at
    ///     inaccessible pages, the embedded Unicode name table's pin, the codecs NumSharp refuses.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The oracle holds the semantics over small operands; these tests reach what its operands do not: arrays long enough
    ///     for the inner runs (past the 64-element width estimate, across re-layouts, up to the result's last rows), element
    ///     widths around each vector boundary (8 / 16 / 24 / 32 units and past 32, where the runs step aside), strided,
    ///     reversed and transposed layouts, and every codec under every handler over data pools that make each route and each
    ///     error happen (lone surrogates and raw units past U+10FFFF in <c>U</c>, every kind of invalid UTF-8 in <c>S</c>,
    ///     BOMs and truncated units for UTF-16 / UTF-32).
    ///     </para>
    ///     <para>
    ///     The differential tests flip <see cref="NDCharObjectPath.CodecFastPathsDisabled"/>, a per-thread hook, so they run in
    ///     parallel with the rest of the suite — all but <see cref="NameReplace_ColdCaches_MatchTheGeneralRoute"/>, which
    ///     forgets the process-wide <c>namereplace</c> caches and runs alone.
    ///     </para>
    /// </remarks>
    [TestClass]
    public unsafe class StringCodecTests
    {
        // =============================================================================================================
        // Helpers
        // =============================================================================================================

        /// <summary>The codecs every differential test runs (one per implemented codec family and byte order).</summary>
        private static readonly string[] Codecs =
        {
            "utf-8", "ascii", "latin-1", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be", "cp1252",
            "utf-8-sig",
        };

        /// <summary>The errors arguments every differential test runs (null: none; the built-in handlers, inline or generic).</summary>
        private static readonly string[] Handlers =
        {
            null, "strict", "replace", "ignore", "xmlcharrefreplace", "backslashreplace", "namereplace", "surrogateescape",
            "surrogatepass",
        };

        /// <summary>The element widths every pool is built at: around each vector boundary of the runs, and past 32 units,
        ///     where the runs step aside for the exact-write paths.</summary>
        private static readonly int[] Widths = { 3, 8, 13, 16, 24, 32, 40 };

        /// <summary>The number of elements of a pool's base array: past the 64-element width estimate, an odd count.</summary>
        private const int BaseCount = 130;

        /// <summary>
        ///     Runs one call twice — through the fast runs and with them disabled — and requires the same outcome: the same
        ///     exception (type and text), or results of the same dtype, shape and bytes.
        /// </summary>
        /// <param name="call">The call (encode or decode of one operand).</param>
        /// <param name="because">The case's name (in the failure message).</param>
        private static void SameAsGeneral(Func<NDArray> call, string because)
        {
            NDArray fast = null, general = null;
            Exception fastError = null, generalError = null;
            try { fast = call(); }
            catch (Exception e) { fastError = e; }
            NDCharObjectPath.CodecFastPathsDisabled = true;
            try { general = call(); }
            catch (Exception e) { generalError = e; }
            finally { NDCharObjectPath.CodecFastPathsDisabled = false; }
            try
            {
                if (generalError is not null || fastError is not null)
                {
                    (fastError?.GetType()).Should().Be(generalError?.GetType(), because + " / " + (fastError ?? generalError)?.Message);
                    fastError!.Message.Should().Be(generalError!.Message, because);
                    return;
                }
                fast.dtype.str.Should().Be(general.dtype.str, because);
                fast.shape.Should().Equal(general.shape, because);
                fast.tobytes().Should().Equal(general.tobytes(), because);
            }
            finally
            {
                fast?.Dispose();
                general?.Dispose();
            }
        }

        /// <summary>
        ///     A 1-D <c>U</c> array of raw code units (which can hold what a C# string cannot: a lone surrogate, a unit past
        ///     U+10FFFF), each element's units then zeros.
        /// </summary>
        /// <param name="elements">The elements' units.</param>
        /// <param name="chars">The element width (each element at most this long).</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray RawU(uint[][] elements, int chars)
        {
            var a = np.zeros(new Shape(elements.Length), np.dtype("U" + chars));
            uint* p = (uint*)a.Storage.Address;
            for (int i = 0; i < elements.Length; i++)
                for (int j = 0; j < elements[i].Length; j++)
                    p[(long)i * chars + j] = elements[i][j];
            return a;
        }

        /// <summary>A 1-D <c>S</c> array of raw bytes, each element's bytes then zeros.</summary>
        /// <param name="elements">The elements' bytes.</param>
        /// <param name="chars">The element width (each element at most this long).</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray RawS(byte[][] elements, int chars)
        {
            var a = np.zeros(new Shape(elements.Length), np.dtype("S" + chars));
            byte* p = (byte*)a.Storage.Address;
            for (int i = 0; i < elements.Length; i++)
                for (int j = 0; j < elements[i].Length; j++)
                    p[(long)i * chars + j] = elements[i][j];
            return a;
        }

        /// <summary>
        ///     The views every differential test runs over a base array of <see cref="BaseCount"/> elements: the whole array
        ///     and three heads (contiguous: the runs' 32-byte loads, then the exact reads near the end), every second element
        ///     (a padded stride: exact reads), the array reversed (a negative stride) and a transposed 2-D view (a C-order walk
        ///     over non-contiguous chunks).
        /// </summary>
        /// <param name="a">The base array.</param>
        /// <returns>The views (each disposed by the caller).</returns>
        private static IEnumerable<(string Name, NDArray View)> Views(NDArray a)
        {
            yield return ("all", a["0:130"]);
            yield return ("head1", a["0:1"]);
            yield return ("head9", a["0:9"]);
            yield return ("head65", a["0:65"]);
            yield return ("step2", a["::2"]);
            yield return ("reversed", a["::-1"]);
            yield return ("transposed", a.reshape(13, 10).T);
        }

        /// <summary>One random code point of a <c>U</c> data pool (see <see cref="UPools"/>).</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="pool">The pool.</param>
        /// <returns>The code unit.</returns>
        private static uint UnitOf(System.Random rng, string pool)
        {
            uint ascii = (uint)rng.Next(0x20, 0x7F);
            switch (pool)
            {
                case "ascii": return ascii;
                case "latin1": return rng.Next(3) == 0 ? (uint)rng.Next(0xA0, 0x100) : ascii;
                case "latin": return rng.Next(3) == 0 ? (uint)rng.Next(0x100, 0x800) : ascii;
                case "cjk": return (uint)rng.Next(0x4E00, 0xA000);
                case "bmp":
                    return rng.Next(4) switch
                    {
                        0 => (uint)rng.Next(0x80, 0x800),
                        1 => (uint)rng.Next(0x800, 0xD800),
                        2 => (uint)rng.Next(0xE000, 0x10000),
                        _ => ascii,
                    };
                case "astral": return rng.Next(3) == 0 ? (uint)rng.Next(0x10000, 0x110000) : ascii;
                case "surrogate": return rng.Next(4) == 0 ? (uint)rng.Next(0xD800, 0xE000) : ascii;
                case "escapes": return rng.Next(3) == 0 ? (uint)rng.Next(0xDC80, 0xDD00) : ascii;
                case "names":
                    return rng.Next(4) switch
                    {
                        0 => (uint)rng.Next(0xF0000, 0xF0200),   // CPython's aliases and named sequences
                        1 => (uint)rng.Next(0xAC00, 0xD7A4),     // computed Hangul names
                        2 => (uint)rng.Next(0x370, 0x400),       // Greek (stored names, a few unnamed)
                        _ => ascii,
                    };
                case "raw":
                    // Raw units past U+10FFFF (and past 0x7FFFFFFF, which printf("%d") prints negative).
                    return rng.Next(5) == 0 ? (uint)rng.Next(0x110000, int.MaxValue) | (rng.Next(2) == 0 ? 0x80000000u : 0u) : ascii;
                default: return rng.Next(4) == 0 ? 0u : ascii;   // "nul": embedded NUL units
            }
        }

        /// <summary>The <c>U</c> data pools (see <see cref="UnitOf"/>).</summary>
        private static readonly string[] UPools =
            { "ascii", "latin1", "latin", "cjk", "bmp", "astral", "surrogate", "escapes", "names", "raw", "nul" };

        /// <summary>
        ///     A <c>U</c> base array of a pool: random lengths up to the width (empty ones included, the full width often);
        ///     the raw pool keeps its one-unit elements below U+110000, as a one-unit text past it ends every call at once
        ///     (<see cref="RawUnitsPastU10FFFF_KeepCPythonsOrder"/> tests that).
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="pool">The pool.</param>
        /// <param name="chars">The element width.</param>
        /// <returns>The array.</returns>
        private static NDArray UBase(System.Random rng, string pool, int chars)
        {
            var elements = new uint[BaseCount][];
            for (int i = 0; i < BaseCount; i++)
            {
                int len = rng.Next(4) == 0 ? chars : rng.Next(0, chars + 1);
                var e = new uint[len];
                for (int j = 0; j < len; j++)
                    e[j] = UnitOf(rng, pool);
                if (len == 1 && e[0] > 0x10FFFF)
                    e[0] = 'q';
                elements[i] = e;
            }
            return RawU(elements, chars);
        }

        /// <summary>The <c>S</c> data pools (see <see cref="BytesOf"/>).</summary>
        private static readonly string[] SPools =
            { "ascii", "latin1", "utf8", "utf8two", "utf8cjk", "bad", "invalid", "utf16", "utf32", "nul" };

        /// <summary>
        ///     One random element of an <c>S</c> data pool, at most <paramref name="chars"/> bytes: ASCII; any bytes (Latin-1);
        ///     valid UTF-8 of one- to four-byte code points, of one- / two-byte ones only, of three-byte ones only; ASCII with
        ///     isolated bytes past 0x7F; every kind of invalid UTF-8 (overlongs, surrogates, past U+10FFFF, truncated, stray and
        ///     adjacent bytes); UTF-16 / UTF-32 units with or without a BOM of either order (pairs, lone surrogates, a cut
        ///     unit); embedded NULs.
        /// </summary>
        /// <param name="rng">The generator.</param>
        /// <param name="pool">The pool.</param>
        /// <param name="chars">The element width.</param>
        /// <returns>The bytes.</returns>
        private static byte[] BytesOf(System.Random rng, string pool, int chars)
        {
            int budget = rng.Next(4) == 0 ? chars : rng.Next(0, chars + 1);
            var b = new List<byte>();
            void Add(params byte[] bytes)
            {
                if (b.Count + bytes.Length <= budget)
                    b.AddRange(bytes);
            }
            byte Letter() => (byte)rng.Next(0x61, 0x7B);
            for (int guard = 0; b.Count < budget && guard < 4 * chars + 8; guard++)
            {
                switch (pool)
                {
                    case "ascii":
                        Add(Letter());
                        break;
                    case "latin1":
                        Add((byte)rng.Next(1, 0x100));
                        break;
                    case "utf8":
                    case "utf8two":
                    case "utf8cjk":
                    {
                        int kind = pool == "utf8cjk" ? 2 : pool == "utf8two" ? rng.Next(2) : rng.Next(4);
                        int cp = kind switch
                        {
                            0 => rng.Next(0x20, 0x80),
                            1 => rng.Next(0x80, 0x800),
                            2 => rng.Next(0x800, 0xD800),
                            _ => rng.Next(0x10000, 0x110000),
                        };
                        Add(Encoding.UTF8.GetBytes(char.ConvertFromUtf32(cp)));
                        break;
                    }
                    case "bad":
                        // Isolated bytes past 0x7F: never two in a row.
                        if (b.Count > 0 && b[b.Count - 1] >= 0x80 || rng.Next(4) != 0)
                            Add(Letter());
                        else
                            Add((byte)rng.Next(0x80, 0x100));
                        break;
                    case "invalid":
                        switch (rng.Next(9))
                        {
                            case 0: Add(0xC0, 0x80); break;                   // overlong two-byte
                            case 1: Add(0xE0, 0x80, 0x80); break;             // overlong three-byte
                            case 2: Add(0xED, 0xA0, 0x80); break;             // a surrogate
                            case 3: Add(0xF4, 0x90, 0x80, 0x80); break;       // past U+10FFFF
                            case 4: Add((byte)rng.Next(0x80, 0xC0)); break;   // a stray continuation byte
                            case 5: Add((byte)rng.Next(0xC2, 0xF5)); break;   // a lead with whatever follows
                            case 6: Add(0xF0, 0x9F, 0x98); break;             // a truncated four-byte sequence
                            case 7: Add((byte)rng.Next(0xF5, 0x100)); break;  // an invalid lead
                            default: Add(Encoding.UTF8.GetBytes(char.ConvertFromUtf32(rng.Next(0x80, 0x800)))); break;
                        }
                        break;
                    case "utf16":
                    case "utf32":
                    {
                        bool wide = pool == "utf32";
                        bool big = rng.Next(2) == 0;
                        if (b.Count == 0 && rng.Next(2) == 0)
                            Add(wide ? (big ? new byte[] { 0, 0, 0xFE, 0xFF } : new byte[] { 0xFF, 0xFE, 0, 0 }) : big ? new byte[] { 0xFE, 0xFF } : new byte[] { 0xFF, 0xFE });
                        uint u = rng.Next(8) switch
                        {
                            0 => (uint)rng.Next(0xD800, 0xE000),                 // a lone surrogate (a pair half)
                            1 => (uint)rng.Next(0x10000, 0x110000),              // astral: a pair in UTF-16
                            2 => wide ? (uint)rng.Next(0x110000, 0x200000) : (uint)rng.Next(0x100, 0x800),
                            _ => (uint)rng.Next(0x21, 0x800),
                        };
                        if (wide)
                        {
                            Add(big ? new[] { (byte)(u >> 24), (byte)(u >> 16), (byte)(u >> 8), (byte)u }
                                    : new[] { (byte)u, (byte)(u >> 8), (byte)(u >> 16), (byte)(u >> 24) });
                        }
                        else if (u >= 0x10000)
                        {
                            uint hi = 0xD800 + ((u - 0x10000) >> 10), lo = 0xDC00 + ((u - 0x10000) & 0x3FF);
                            Add(big ? new[] { (byte)(hi >> 8), (byte)hi, (byte)(lo >> 8), (byte)lo }
                                    : new[] { (byte)hi, (byte)(hi >> 8), (byte)lo, (byte)(lo >> 8) });
                        }
                        else
                        {
                            Add(big ? new[] { (byte)(u >> 8), (byte)u } : new[] { (byte)u, (byte)(u >> 8) });
                        }
                        if (rng.Next(16) == 0)
                            Add((byte)rng.Next(0x21, 0x7F));   // a cut unit
                        break;
                    }
                    default:
                        Add(rng.Next(4) == 0 ? (byte)0 : Letter());
                        break;
                }
            }
            return b.ToArray();
        }

        /// <summary>An <c>S</c> base array of a pool (see <see cref="BytesOf"/>).</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="pool">The pool.</param>
        /// <param name="chars">The element width.</param>
        /// <returns>The array.</returns>
        private static NDArray SBase(System.Random rng, string pool, int chars)
        {
            var elements = new byte[BaseCount][];
            for (int i = 0; i < BaseCount; i++)
                elements[i] = BytesOf(rng, pool, chars);
            return RawS(elements, chars);
        }

        // =============================================================================================================
        // The fast runs against the general route
        // =============================================================================================================

        /// <summary>
        ///     <c>U</c> → <c>S</c>: every pool at every width through every view, under every codec and every handler — the fast
        ///     runs give the general route's bytes (or its error) exactly.
        /// </summary>
        [TestMethod]
        public void Encode_U_FastRunsMatchTheGeneralRoute()
        {
            var rng = new System.Random(2626);
            foreach (string pool in UPools)
            {
                foreach (int chars in Widths)
                {
                    using var a = UBase(rng, pool, chars);
                    foreach (var (name, view) in Views(a))
                    {
                        using (view)
                        {
                            foreach (string codec in Codecs)
                                foreach (string errors in Handlers)
                                    SameAsGeneral(() => np.strings.encode(view, codec, errors), $"U{chars} {pool} {name} {codec} {errors ?? "-"}");
                            SameAsGeneral(() => np.strings.encode(view), $"U{chars} {pool} {name} (no encoding)");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     <c>S</c> → <c>U</c>: every pool at every width through every view, under every codec and every handler (the
        ///     decode-only errors included: xmlcharrefreplace and namereplace raise CPython's TypeError at an error).
        /// </summary>
        [TestMethod]
        public void Decode_S_FastRunsMatchTheGeneralRoute()
        {
            var rng = new System.Random(2627);
            foreach (string pool in SPools)
            {
                foreach (int chars in Widths)
                {
                    using var a = SBase(rng, pool, chars);
                    foreach (var (name, view) in Views(a))
                    {
                        using (view)
                        {
                            foreach (string codec in Codecs)
                                foreach (string errors in Handlers)
                                    SameAsGeneral(() => np.strings.decode(view, codec, errors), $"S{chars} {pool} {name} {codec} {errors ?? "-"}");
                            SameAsGeneral(() => np.strings.decode(view), $"S{chars} {pool} {name} (no encoding)");
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     <c>StringDType</c> → <c>S</c>: short (inline), medium and long (arena) cells, empty cells and NA cells of every
        ///     NA kind (None, a str, NaN, none at all), under every codec and every handler.
        /// </summary>
        [TestMethod]
        public void Encode_StringDType_FastRunsMatchTheGeneralRoute()
        {
            var rng = new System.Random(2628);
            string[] pools = { "ascii", "latin1", "latin", "cjk", "astral", "names" };
            var naKinds = new (string Name, DType Dtype, object Na)[]
            {
                ("plain", np.dtypes.StringDType(), null),
                ("na-none", np.dtypes.StringDType(na_object: null), null),
                ("na-str", np.dtypes.StringDType(na_object: "NA"), "NA"),
                ("na-nan", np.dtypes.StringDType(na_object: double.NaN), double.NaN),
            };
            foreach (string pool in pools)
            {
                foreach (int maxLen in new[] { 4, 15, 16, 40 })
                {
                    foreach (var (naName, dtype, na) in naKinds)
                    {
                        var items = new object[BaseCount];
                        for (int i = 0; i < BaseCount; i++)
                        {
                            // NA cells only where the dtype has an NA object (and rarely: an NA cell ends a fast run).
                            if (naName != "plain" && rng.Next(40) == 0)
                            {
                                items[i] = na;
                                continue;
                            }
                            int len = rng.Next(4) == 0 ? maxLen : rng.Next(0, maxLen + 1);
                            var sb = new StringBuilder();
                            for (int j = 0; j < len; j++)
                            {
                                uint u = UnitOf(rng, pool);
                                sb.Append(char.ConvertFromUtf32((int)u));
                            }
                            items[i] = sb.ToString();
                        }
                        using var a = np.array(items, dtype);
                        foreach (var (name, view) in Views(a))
                        {
                            using (view)
                            {
                                foreach (string codec in Codecs)
                                    foreach (string errors in Handlers)
                                        SameAsGeneral(() => np.strings.encode(view, codec, errors), $"T{maxLen} {pool} {naName} {name} {codec} {errors ?? "-"}");
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        ///     An element far longer than the first 64 (the width estimate) late in a long array, and many elements each a
        ///     little longer than the last: the rows are re-laid out, at the exact length first and by half again later, and
        ///     narrowed once at the end — through the fast runs exactly as through the general route.
        /// </summary>
        [TestMethod]
        public void Relayouts_MatchTheGeneralRoute()
        {
            const int count = 300;
            foreach (bool growing in new[] { false, true })
            {
                var u = new uint[count][];
                var s = new byte[count][];
                for (int i = 0; i < count; i++)
                {
                    int len = growing ? 1 + i / 10 : i == 250 ? 30 : 3;
                    u[i] = Enumerable.Range(0, len).Select(j => (uint)(j % 3 == 0 ? 0xE9 : 'a' + j % 26)).ToArray();
                    s[i] = Enumerable.Range(0, len).Select(j => (byte)(j % 4 == 0 ? 0xE9 : 'a' + j % 26)).ToArray();
                }
                using var ua = RawU(u, 30);
                using var sa = RawS(s, 30);
                foreach (string codec in Codecs)
                {
                    foreach (string errors in Handlers)
                    {
                        SameAsGeneral(() => np.strings.encode(ua, codec, errors), $"relayout U growing={growing} {codec} {errors ?? "-"}");
                        SameAsGeneral(() => np.strings.decode(sa, codec, errors), $"relayout S growing={growing} {codec} {errors ?? "-"}");
                    }
                }
            }
        }

        /// <summary>
        ///     Encoding and errors arguments that vary per element (arrays broadcast against a): the fast runs need arguments
        ///     that stay put inside a chunk, so they step aside — the results still match the general route's.
        /// </summary>
        [TestMethod]
        public void VaryingArguments_MatchTheGeneralRoute()
        {
            var rng = new System.Random(2629);
            using var u = UBase(rng, "latin1", 8);
            using var s = SBase(rng, "bad", 8);
            var encodings = np.array(Enumerable.Range(0, BaseCount).Select(i => Codecs[i % Codecs.Length]).ToArray());
            var errors = np.array(Enumerable.Range(0, BaseCount).Select(i => i % 2 == 0 ? "replace" : "backslashreplace").ToArray());
            using (encodings)
            using (errors)
            {
                SameAsGeneral(() => np.strings.encode(u, encodings, "replace"), "U varying encoding");
                SameAsGeneral(() => np.strings.encode(u, "ascii", errors), "U varying errors");
                SameAsGeneral(() => np.strings.decode(s, "utf-8", errors), "S varying errors");
                SameAsGeneral(() => np.strings.decode(s, encodings, "replace"), "S varying encoding");
            }
        }

        /// <summary>
        ///     Raw <c>U</c> units past U+10FFFF: a one-unit element past it is CPython's SystemError at the scalar, raised in C
        ///     order (after the elements before it were coded), however the elements before it were coded; in longer elements
        ///     the codecs and handlers write what CPython writes for such units (UTF-32 as they are, a negative XML reference
        ///     past 0x7FFFFFFF, all eight hex digits under backslashreplace), wherever the unit sits in an element.
        /// </summary>
        /// <remarks>The mutation pass found a gap here: the run's check for XML references it does not write (a unit from
        ///     10,000,000) could skip the lanes past the eighth unnoticed, as the random elements were never the longest of
        ///     their arrays; the lane sweep pins it.</remarks>
        [TestMethod]
        public void RawUnitsPastU10FFFF_KeepCPythonsOrder()
        {
            foreach (int at in new[] { 0, 1, 63, 64, 65, 129 })
            {
                var elements = Enumerable.Range(0, BaseCount).Select(i => new uint[] { 'a', 'b', 0xE9 }).ToArray();
                elements[at] = new uint[] { 0x110000 };
                using var a = RawU(elements, 4);
                foreach (string codec in Codecs)
                    foreach (string errors in Handlers)
                        SameAsGeneral(() => np.strings.encode(a, codec, errors), $"one-unit past U+10FFFF at {at} {codec} {errors ?? "-"}");
                // The general route's error: CPython's SystemError at PyArray_ToScalar.
                Action call = () => np.strings.encode(a);
                call.Should().Throw<SystemError>();
            }
            var mixed = new[] { new uint[] { 'x', 0x110000, 0xE9 }, new uint[] { 0xFFFFFFFF, 'y' }, new uint[] { 0x7FFFFFFF, 0x80000000 } };
            using var m = RawU(Enumerable.Range(0, BaseCount).Select(i => mixed[i % 3]).ToArray(), 3);
            foreach (string codec in Codecs)
                foreach (string errors in Handlers)
                    SameAsGeneral(() => np.strings.encode(m, codec, errors), $"multi-unit past U+10FFFF {codec} {errors ?? "-"}");
            using var xml = np.strings.encode(m["0:3"], "ascii", "xmlcharrefreplace");
            Encoding.ASCII.GetString((byte[])xml.item(1)).Should().Be("&#-1;y");
            Encoding.ASCII.GetString((byte[])xml.item(2)).Should().Be("&#2147483647;&#-2147483648;");
            // backslashreplace writes all eight hex digits of the unit; Latin-1 writes U+00E9 as the byte 0xE9.
            using var bs = np.strings.encode(m["0:3"], "latin-1", "backslashreplace");
            ((byte[])bs.item(0)).Should().Equal(Encoding.ASCII.GetBytes("x\\U00110000").Concat(new byte[] { 0xE9 }));
            // An XML reference of 8+ digits or a negative one (a unit from 10,000,000, which the run leaves to the codec) and
            // the longest one the run writes (9,999,999), at every lane of every vector of the array's longest element. The
            // first 70 elements are one byte shorter than it, so the rows the width estimate sizes (and the first element,
            // which the general route codes before any run) are one byte too narrow for it: only a length the run counts
            // right re-lays them out; one it undercounts writes the element past its row, and the short elements at the end
            // (the outer loop's exact writes) cannot restore the width.
            foreach (int lane in new[] { 0, 7, 8, 15, 16, 23, 24, 31 })
            {
                foreach (uint big in new uint[] { 9_999_999, 10_000_000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF })
                {
                    // ASCII before the unit, and one after it below lane 31 (never a one-unit element: that is the
                    // SystemError above).
                    var e = new uint[lane < 31 ? lane + 2 : 32];
                    for (int j = 0; j < lane; j++)
                        e[j] = (uint)('a' + j % 26);
                    e[lane] = big;
                    if (lane < 31)
                        e[lane + 1] = 'z';
                    // e's length under xmlcharrefreplace: "&#" + the unit printed as a C int + ";", one byte per other unit.
                    int length = e.Length - 1 + 3 + ((int)big).ToString(System.Globalization.CultureInfo.InvariantCulture).Length;
                    // One byte shorter: references of 9,999,999 (ten bytes each, which the run writes) padded with ASCII, two
                    // references when one would leave more than 32 units, or plain ASCII when shorter than one reference.
                    int refs = length - 1 < 11 ? 0 : length - 1 > 42 ? 2 : 1;
                    var shorter = new uint[refs + (length - 1 - 10 * refs)];
                    for (int j = 0; j < shorter.Length; j++)
                        shorter[j] = j < refs ? 9_999_999u : (uint)('a' + j % 26);
                    var shortText = new uint[] { 'a', 'b' };
                    using var r = RawU(Enumerable.Range(0, BaseCount).Select(i => i < 70 ? shorter : i < BaseCount - 10 ? e : shortText).ToArray(), 32);
                    foreach (string codec in new[] { "ascii", "latin-1" })
                        foreach (string errors in new[] { "xmlcharrefreplace", "backslashreplace" })
                            SameAsGeneral(() => np.strings.encode(r, codec, errors), $"unit {big:X} at lane {lane} {codec} {errors}");
                }
            }
        }

        /// <summary>
        ///     Short <c>StringDType</c> cells whose bytes past their size hold an older, longer text (a cell rewritten with a
        ///     shorter one keeps its old bytes; here the size nibbles are shrunk directly): the cell run reads exactly the
        ///     cell's text — the 16-byte load is masked to the size.
        /// </summary>
        /// <remarks>The mutation pass found this gap: a freshly packed cell holds zeros past its text, so a mask to all 15
        ///     text bytes gave the same results.</remarks>
        [TestMethod]
        public void ShortCells_BytesPastTheSize_AreNeverRead()
        {
            // 14-byte texts (short cells), cut to 5 bytes (ASCII) and to 3 bytes ("hé": a whole sequence).
            var items = Enumerable.Range(0, BaseCount).Select(i => i % 2 == 0 ? "abcdefghijklmn" : "héllo wörld!").ToArray();
            using var a = np.array(items, np.dtypes.StringDType());
            byte* cells = (byte*)a.Storage.Address + a.Shape.offset * 16;
            for (int i = 0; i < BaseCount; i++)
            {
                byte* cell = cells + 16L * i;
                cell[15] = (byte)((cell[15] & 0xF0) | (i % 2 == 0 ? 5 : 3));
            }
            foreach (string codec in new[] { "utf-8", "latin-1", "ascii" })
                foreach (string errors in new[] { null, "replace", "xmlcharrefreplace" })
                    SameAsGeneral(() => np.strings.encode(a, codec, errors), $"shrunk cells {codec} {errors ?? "-"}");
            using var r8 = np.strings.encode(a, "utf-8");
            ((byte[])r8.item(0)).Should().Equal(Encoding.ASCII.GetBytes("abcde"));
            ((byte[])r8.item(1)).Should().Equal(Encoding.UTF8.GetBytes("hé"));
            using var r1 = np.strings.encode(a, "latin-1");
            ((byte[])r1.item(1)).Should().Equal(new byte[] { 0x68, 0xE9 });
        }

        /// <summary>
        ///     Every result row is written whole by the runs (the rows are <c>ForOverwrite</c> buffers, with no NUL fill): each
        ///     result is computed after buffers of its rows' possible sizes were filled with junk and returned to the pool, so
        ///     the run takes one of them, and any byte it fails to write — a row's padding, a handler's tail on rows wider than
        ///     32 bytes — shows as junk against the general route's result.
        /// </summary>
        /// <remarks>The mutation pass found this gap: the pool hands a call the buffer the previous identical call returned,
        ///     which already holds the right answer in every byte, so a row the run skips still read correctly.</remarks>
        [TestMethod]
        public void Rows_AreWrittenWhole_OverAJunkFilledPool()
        {
            var rng = new System.Random(2631);
            var cases = new List<(string Name, NDArray Input, Func<NDArray, NDArray> Call)>();
            // The inputs, each once (an NDArray is unhashable, so no Distinct over the cases).
            var inputs = new List<NDArray>();
            foreach (string pool in new[] { "latin1", "bmp", "escapes", "names", "ascii" })
            {
                foreach (int chars in new[] { 8, 16, 24 })
                {
                    var u = UBase(rng, pool, chars);
                    inputs.Add(u);
                    foreach (string codec in new[] { "utf-8", "ascii", "latin-1", "utf-16-be", "utf-32" })
                        foreach (string errors in new[] { null, "replace", "ignore", "xmlcharrefreplace", "backslashreplace", "namereplace", "surrogateescape" })
                            cases.Add(($"U{chars} {pool} {codec} {errors ?? "-"}", u, x => np.strings.encode(x, codec, errors)));
                }
            }
            foreach (string pool in new[] { "ascii", "latin1", "utf8two", "utf8cjk", "bad", "utf16" })
            {
                foreach (int chars in new[] { 8, 16, 24 })
                {
                    var s = SBase(rng, pool, chars);
                    inputs.Add(s);
                    foreach (string codec in new[] { "utf-8", "ascii", "latin-1", "utf-16" })
                        foreach (string errors in new[] { null, "replace", "ignore", "backslashreplace", "surrogateescape" })
                            cases.Add(($"S{chars} {pool} {codec} {errors ?? "-"}", s, x => np.strings.decode(x, codec, errors)));
                }
            }
            try
            {
                foreach (var (name, input, call) in cases)
                {
                    NDArray general = null;
                    NDCharObjectPath.CodecFastPathsDisabled = true;
                    try { general = call(input); }
                    catch (Exception) { }
                    finally { NDCharObjectPath.CodecFastPathsDisabled = false; }
                    if (general is null)
                        continue;   // an error: the differential tests compare those
                    using (general)
                    {
                        // The rows start at the estimate's width (any width up to the final one) and may be re-laid out through
                        // other widths or narrowed: poison every row width up to past the final one.
                        long count = general.size;
                        long rowUnit = general.dtype.kind == 'U' ? 4 : 1;
                        long width = general.dtype.itemsize / rowUnit;
                        for (long w = 1; w <= width + 8; w++)
                            PoisonPool(count * w * rowUnit);
                        using var fast = call(input);
                        fast.dtype.str.Should().Be(general.dtype.str, name);
                        fast.tobytes().Should().Equal(general.tobytes(), name);
                    }
                }
            }
            finally
            {
                foreach (var input in inputs)
                    input.Dispose();
            }
        }

        /// <summary>Returns three buffers of exactly <paramref name="bytes"/> bytes, filled with junk, to the buffer pool (the
        ///     next allocation of that size takes one).</summary>
        /// <param name="bytes">The byte size.</param>
        private static void PoisonPool(long bytes)
        {
            if (bytes < 1)
                return;
            var held = new NDArray[3];
            for (int i = 0; i < held.Length; i++)
            {
                held[i] = np.empty(new Shape(bytes), np.uint8);
                new Span<byte>((void*)held[i].Storage.Address, checked((int)bytes)).Fill(0xA5);
            }
            foreach (var h in held)
                h.Dispose();
        }

        /// <summary>
        ///     <c>namereplace</c> with both of its caches cold — the built replacements and the memoized name lengths forgotten
        ///     before each call — so the exact pre-pass sizes elements through its scalar rule while filling the memo, and the
        ///     inner run meets characters whose replacement is not built yet in the middle of the call (each such element takes
        ///     the outer loop, which builds the replacement, and the run resumes after it). The characters first appear late in
        ///     the array, one after another: names of one, two and three 32-byte chunks (U+FBF9's <c>\N{...}</c> is 87 bytes),
        ///     computed names (Hangul, CJK), an unnamed BMP character (its backslash form) and a character past U+FFFF (never
        ///     cached: always the outer loop), under ASCII and Latin-1, at element widths inside and at the run's limit.
        /// </summary>
        /// <remarks>Forgets process-wide caches, so it runs alone: a call running beside it would only rebuild entries, but the
        ///     cold start this test claims would no longer be certain.</remarks>
        [TestMethod]
        [DoNotParallelize]
        public void NameReplace_ColdCaches_MatchTheGeneralRoute()
        {
            uint[] rare = { 0xE9, 0x101, 0x3B1, 0xFBF9, 0x4E00, 0xAC00, 0xE000, 0x1F600, 0xFFFD, 0x2028, 0xFBFA };
            foreach (string codec in new[] { "ascii", "latin-1" })
            {
                foreach (int chars in new[] { 4, 8, 13, 32 })
                {
                    var rng = new System.Random(2632 + chars);
                    var elements = new uint[300][];
                    for (int i = 0; i < elements.Length; i++)
                    {
                        // The rare characters met so far: the k-th first appears in element 20 + 25k (forced into it).
                        int known = System.Math.Min(rare.Length, i < 20 ? 0 : (i - 20) / 25 + 1);
                        int len = rng.Next(1, chars + 1);
                        var e = new uint[len];
                        for (int j = 0; j < len; j++)
                            e[j] = known > 0 && rng.Next(3) == 0 ? rare[rng.Next(known)] : (uint)rng.Next(0x61, 0x7B);
                        if (i >= 20 && (i - 20) % 25 == 0 && known <= rare.Length)
                            e[len - 1] = rare[known - 1];
                        elements[i] = e;
                    }
                    using var a = RawU(elements, chars);
                    PyUnicodeNames.ClearNameLengthsForTests();
                    NDCharObjectPath.ResetNameReplacementsForTests();
                    SameAsGeneral(() => np.strings.encode(a, codec, "namereplace"), $"cold {codec} U{chars}");
                    // And warm: every replacement built, the whole array through the inner run.
                    SameAsGeneral(() => np.strings.encode(a, codec, "namereplace"), $"warm {codec} U{chars}");
                }
            }
        }

        // =============================================================================================================
        // Memory safety
        // =============================================================================================================

        /// <summary>
        ///     <c>U</c> and <c>S</c> arrays whose last element ends at an inaccessible page, or whose first starts right after
        ///     one, at every width from 1 to 36 units and every count from 1 to 40, under the codecs and handlers of every run:
        ///     the runs' loads (masked loads, 32-byte loads near the chunk's end, exact reads) stay inside the elements — an
        ///     out-of-bounds read ends the test host. The results are checked against the general route on the way.
        /// </summary>
        [TestMethod]
        public void Runs_StayInsideTheirElements()
        {
            if (!FencedMemory.IsSupported)
                Assert.Inconclusive("FencedMemory needs Windows (reserved pages).");
            var rng = new System.Random(2630);
            string[] encodeCodecs = { "utf-8", "ascii", "latin-1", "utf-16", "utf-32-be" };
            string[] encodeHandlers = { null, "replace", "xmlcharrefreplace", "namereplace" };
            string[] decodeCodecs = { "utf-8", "ascii", "latin-1", "utf-16", "utf-32" };
            string[] decodeHandlers = { null, "replace", "backslashreplace" };
            using var fence = new FencedMemory(pages: 2);
            foreach (bool atEnd in new[] { true, false })
            {
                for (int chars = 1; chars <= 36; chars++)
                {
                    foreach (int count in new[] { 1, 2, 7, 33, 40 })
                    {
                        // U: the units written straight into the fenced block.
                        long ubytes = (long)count * chars * 4;
                        byte* up = fence.Place(ubytes, atEnd);
                        for (long i = 0; i < (long)count * chars; i++)
                            ((uint*)up)[i] = rng.Next(3) == 0 ? 0u : UnitOf(rng, rng.Next(2) == 0 ? "latin1" : "bmp");
                        using (var u = np.frombuffer((IntPtr)up, ubytes, np.dtype("U" + chars)))
                        {
                            foreach (string codec in encodeCodecs)
                                foreach (string errors in encodeHandlers)
                                    SameAsGeneral(() => np.strings.encode(u, codec, errors), $"fenced U{chars} x{count} atEnd={atEnd} {codec} {errors ?? "-"}");
                        }
                        long sbytes = (long)count * chars;
                        byte* sp = fence.Place(sbytes, atEnd);
                        for (long i = 0; i < sbytes; i++)
                            sp[i] = rng.Next(4) == 0 ? (byte)0 : (byte)(rng.Next(3) == 0 ? rng.Next(0x80, 0x100) : rng.Next(0x61, 0x7B));
                        using (var s = np.frombuffer((IntPtr)sp, sbytes, np.dtype("S" + chars)))
                        {
                            foreach (string codec in decodeCodecs)
                                foreach (string errors in decodeHandlers)
                                    SameAsGeneral(() => np.strings.decode(s, codec, errors), $"fenced S{chars} x{count} atEnd={atEnd} {codec} {errors ?? "-"}");
                        }
                    }
                }
            }
        }

        // =============================================================================================================
        // The Unicode name table
        // =============================================================================================================

        /// <summary>
        ///     The embedded names decode to exactly what the generator recorded from CPython 3.12.12 (every code point's
        ///     namereplace name, hashed); and the BMP length memo agrees with the names themselves.
        /// </summary>
        [TestMethod]
        public void UnicodeNames_MatchThePinnedTable()
        {
            PyUnicodeNames.ComputeNamesHash().Should().Be(PyUnicodeNames.NamesHash);
            Span<byte> name = stackalloc byte[256];
            for (uint cp = 0; cp < 0x10000; cp++)
                PyUnicodeNames.GetNameLength(cp).Should().Be(PyUnicodeNames.GetName(cp, name), $"U+{cp:X4}");
            PyUnicodeNames.GetNameLength(0x1F600).Should().Be(PyUnicodeNames.GetName(0x1F600, name));
            PyUnicodeNames.GetNameLength(0x110000).Should().Be(0);
        }

        // =============================================================================================================
        // Edges
        // =============================================================================================================

        /// <summary>
        ///     The codecs CPython has and NumSharp does not implement are refused with a NotSupportedException that names the
        ///     codec and what NumSharp implements (never a wrong result); an unknown name is CPython's LookupError.
        /// </summary>
        [TestMethod]
        public void UnsupportedCodecs_AreRefused()
        {
            using var u = np.array(new[] { "abc" });
            using var s = np.array(new object[] { Encoding.ASCII.GetBytes("abc") }, np.dtype("S3"));
            foreach (string codec in new[] { "utf-7", "idna", "unicode_escape", "shift_jis", "gb2312", "punycode" })
            {
                Action enc = () => np.strings.encode(u, codec);
                enc.Should().Throw<NotSupportedException>().WithMessage("*np.strings.encode / np.strings.decode*");
                Action dec = () => np.strings.decode(s, codec);
                dec.Should().Throw<NotSupportedException>();
            }
            Action unknown = () => np.strings.encode(u, "no-such-codec");
            unknown.Should().Throw<LookupError>().WithMessage("unknown encoding: no-such-codec");
        }

        /// <summary>
        ///     An empty input is an empty <c>S1</c> / <c>U1</c> result of the broadcast shape (NumPy's astype of the empty
        ///     object result), and a 0-d input a 0-d result, through the fast runs as through the general route.
        /// </summary>
        [TestMethod]
        public void EmptyAndZeroD()
        {
            using var eu = np.zeros(new Shape(0, 3), np.dtype("U4"));
            using var r1 = np.strings.encode(eu, "utf-8");
            r1.dtype.str.Should().Be("|S1");
            r1.shape.Should().Equal(new long[] { 0, 3 });
            using var es = np.zeros(new Shape(2, 0), np.dtype("S4"));
            using var r2 = np.strings.decode(es);
            r2.dtype.str.Should().Be("<U1");
            r2.shape.Should().Equal(new long[] { 2, 0 });
            using var zu = np.array("héllo");
            SameAsGeneral(() => np.strings.encode(zu, "utf-8"), "0-d U");
            using var z8 = np.strings.encode(zu, "utf-8");
            z8.ndim.Should().Be(0);
            ((byte[])z8.item()).Should().Equal(Encoding.UTF8.GetBytes("héllo"));
        }
    }
}
