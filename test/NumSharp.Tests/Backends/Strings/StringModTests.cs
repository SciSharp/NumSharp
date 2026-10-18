using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using NumSharp.Backends.Strings;

namespace NumSharp.Tests.Backends.Strings
{
    /// <summary>
    ///     NumPy's <c>np.strings.mod</c> (strings phase 2, P2.6 milestone 3): its fast paths — the chunk loop that writes a
    ///     number's or a text's field straight into the result, the element path, the row storage of a <c>U</c> / <c>S</c>
    ///     result and the per-element packing of a <c>StringDType</c> one — against the general route (CPython's argument
    ///     machinery and PrintfFormatter's conversions, which the <c>strmod</c> / <c>tmod</c> oracle sections gate), and the
    ///     public function against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The oracle holds the semantics over small operands; these tests reach what its operands do not: arrays past the
    ///     row sample (64 elements) whose width changes later (re-layout, narrowing), row widths at the vector-copy
    ///     boundaries, the last rows near the result's end, every value dtype and special value through every conversion of
    ///     a one-argument format, C order of the first error in a large array, and the deferred code-point error of a
    ///     <c>StringDType</c> result.
    ///     </para>
    ///     <para>
    ///     The differential tests flip <see cref="NDCharObjectPath.ModFastPathsDisabled"/>, a per-thread hook, so they run
    ///     in parallel with the rest of the suite. Every expected literal was probed against NumPy 2.4.2.
    ///     </para>
    /// </remarks>
    [TestClass]
    public unsafe class StringModTests
    {
        // =============================================================================================================
        // Helpers
        // =============================================================================================================

        /// <summary>The elements of a <c>U</c> / <c>StringDType</c> result in C order, each as <c>item()</c> gives it (a
        ///     string).</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its elements.</returns>
        private static object[] Strings(NDArray r) => Enumerable.Range(0, (int)r.size).Select(i => r.item(i)).ToArray();

        /// <summary>The elements of an <c>S</c> result as Latin-1 strings (one char per byte; <c>item()</c> strips the
        ///     trailing NULs).</summary>
        /// <param name="r">The result.</param>
        /// <returns>Its elements.</returns>
        private static string[] Bytes(NDArray r) => Enumerable.Range(0, (int)r.size)
            .Select(i => Encoding.Latin1.GetString((byte[])r.item(i))).ToArray();

        /// <summary>A 1-element <c>S</c> array of an ASCII format (the bytes formatting operand).</summary>
        /// <param name="format">The format (ASCII; Latin-1 encoded).</param>
        /// <returns>The array, exactly as wide as the format (at least one byte).</returns>
        private static NDArray S(string format)
            => np.array(new object[] { Encoding.Latin1.GetBytes(format) }, np.dtype("S" + System.Math.Max(1, format.Length)));

        /// <summary>A 1-D <c>U2</c> array of raw code units (two per element), which can hold what a C# string cannot: a
        ///     unit past U+10FFFF.</summary>
        /// <param name="units">The units, two per element, in element order.</param>
        /// <returns>The array (the caller disposes it).</returns>
        private static NDArray RawU2(uint[] units)
        {
            var a = np.zeros(new Shape(units.Length / 2), np.dtype("U2"));
            using var v = a.view(np.uint32);
            for (int i = 0; i < units.Length; i++)
                v.SetUInt32(units[i], i);
            return a;
        }

        /// <summary>A 1-element <c>StringDType</c> array of a format.</summary>
        /// <param name="format">The format.</param>
        /// <returns>The array.</returns>
        private static NDArray T(string format) => np.array(new[] { format }, np.dtypes.StringDType());

        /// <summary>
        ///     Runs one call twice — through the fast paths and with them disabled — and requires the same outcome: the same
        ///     exception (type and text), or results of the same dtype, shape and elements.
        /// </summary>
        /// <param name="format">The format operand.</param>
        /// <param name="values">The values operand.</param>
        /// <param name="because">The case's name (in the failure message).</param>
        private static void SameAsGeneral(NDArray format, object values, string because)
        {
            NDArray fast = null, general = null;
            Exception fastError = null, generalError = null;
            try { fast = np.strings.mod(format, values); }
            catch (Exception e) { fastError = e; }
            NDCharObjectPath.ModFastPathsDisabled = true;
            try { general = np.strings.mod(format, values); }
            catch (Exception e) { generalError = e; }
            finally { NDCharObjectPath.ModFastPathsDisabled = false; }
            try
            {
                if (generalError is not null || fastError is not null)
                {
                    (fastError?.GetType()).Should().Be(generalError?.GetType(), because);
                    fastError!.Message.Should().Be(generalError!.Message, because);
                    return;
                }
                fast.dtype.Should().Be(general.dtype, because);
                fast.shape.Should().Equal(general.shape, because);
                if (fast.dtype.kind == 'S')
                    fast.tobytes().Should().Equal(general.tobytes(), because);
                else
                    Strings(fast).Should().Equal(Strings(general), because);
            }
            finally
            {
                fast?.Dispose();
                general?.Dispose();
            }
        }

        /// <summary>Special and ordinary values of a numeric dtype (the dtype's extremes, zeros, NaN / infinities for
        ///     floats, ties, tiny and huge magnitudes), as an array of that dtype.</summary>
        /// <param name="tc">The dtype.</param>
        /// <returns>The values.</returns>
        private static NDArray Specials(NPTypeCode tc)
        {
            double[] floats =
            {
                0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 0.125, 0.375, 9.9996, 123456.7, -98765.4321, 1e16, 1e-5, 1e-4, 1e100, 1e-100,
                1e300, -1e300, 5e-324, 2.2250738585072014e-308, double.NaN, double.PositiveInfinity, double.NegativeInfinity,
                System.Math.ScaleB(1, -25), 1.7976931348623157e308, 999999.5, 0.1, 1.0 / 3, 9223372036854775807.0,
                -9223372036854775808.0, 18446744073709551615.0,
            };
            switch (tc)
            {
                case NPTypeCode.Double:
                    return np.array(floats);
                case NPTypeCode.Single:
                    return np.array(floats.Select(d => (float)d).ToArray());
                case NPTypeCode.Half:
                    return np.array(floats.Select(d => (Half)d).ToArray());
                default:
                {
                    var info = np.iinfo(tc);
                    // uint64's maximum is not a long: its top values are appended below instead.
                    long min = info.min, max = tc == NPTypeCode.UInt64 ? long.MaxValue : info.max;
                    var v = new List<long> { 0, 1, -1, 7, -7, 42, 255, 1000, -1000, 123456, min, max, min + 1, max - 1 };
                    var arr = np.array(v.ToArray()).astype(tc);
                    if (tc == NPTypeCode.UInt64)
                    {
                        // The top of uint64 (past int64): its own values.
                        using var top = np.array(new[] { ulong.MaxValue, ulong.MaxValue - 1, 1UL << 63, 9007199254740993UL });
                        var joined = np.concatenate(new[] { arr, top });
                        arr.Dispose();
                        return joined;
                    }
                    return arr;
                }
            }
        }

        /// <summary>The numeric dtypes the fast paths specialize (and two they do not: bool and complex take the general
        ///     route in both runs).</summary>
        private static readonly NPTypeCode[] ValueTypes =
        {
            NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
        };

        /// <summary>One-argument formats over every conversion, flag, width and precision the fast paths write, with literal
        ///     text around them and <c>%%</c>; and a few formats that take the general route (two specs, a key).</summary>
        private static readonly string[] Formats =
        {
            "%d", "%i", "%u", "%5d", "%-5d|", "%+d", "% d", "%05d", "%.3d", "%08.3d", "%x", "%#X", "%#o", "%8.3x", "%#010x",
            "%s", "%10s", "%-10s|", "%.3s", "%r", "%a", "%15r", "%e", "%.2e", "%E", "%12.3e", "%#.0e", "%f", "%.2f", "%#.0f",
            "%+08.2f", "%-10.1f|", "%F", "%.0f", "%g", "%.3g", "%#g", "%G", "%10.4g", "%.0g", "%c", "x=%d;", "%%%d%%",
            "<%.2f>", "%.17e", "%.18e", "%.19f", "%.20f", "%d %d", "%(a)s",
        };

        // =============================================================================================================
        // The fast paths against the general route
        // =============================================================================================================

        /// <summary>
        ///     Every one-argument format over every specialized dtype's special values, through a <c>U</c>, an <c>S</c> and a
        ///     <c>StringDType</c> format: the fast paths give the general route's result (or its error) exactly — over the
        ///     whole array (the chunk path) and over each value alone (a one-element view at its offset).
        /// </summary>
        /// <remarks>
        ///     The whole array stops at its first element that raises (NaN under <c>%d</c>, a negative code point under
        ///     <c>%c</c>), which hides every later value from the comparison: the mutation pass found 2^63 under <c>%d</c>
        ///     (the truncation's range check) reached only one value at a time.
        /// </remarks>
        [TestMethod]
        public void FastPaths_MatchTheGeneralRoute_EveryFormatAndDtype()
        {
            foreach (var tc in ValueTypes)
            {
                using var values = Specials(tc);
                foreach (var f in Formats)
                {
                    using var sf = S(f);
                    using var tf = T(f);
                    SameAsGeneral(f, values, $"U '{f}' over {tc}");
                    SameAsGeneral(sf, values, $"S '{f}' over {tc}");
                    SameAsGeneral(tf, values, $"T '{f}' over {tc}");
                    for (int i = 0; i < values.size; i++)
                    {
                        using var one = values[$"{i}:{i + 1}"];
                        SameAsGeneral(f, one, $"U '{f}' over {tc}[{i}]");
                        SameAsGeneral(sf, one, $"S '{f}' over {tc}[{i}]");
                        SameAsGeneral(tf, one, $"T '{f}' over {tc}[{i}]");
                    }
                }
            }
        }

        /// <summary>
        ///     Large arrays in every layout the walk meets (contiguous, reversed, strided, 2-D, a broadcast column of values
        ///     against a row of formats): sizes around the row sample (63, 64, 65), a few hundred and a few thousand elements,
        ///     so the direct row writes, the last rows' arena path and the per-element format route all run.
        /// </summary>
        [TestMethod]
        public void FastPaths_MatchTheGeneralRoute_LargeArraysAndLayouts()
        {
            var rng = new System.Random(20261018);
            foreach (int n in new[] { 1, 2, 63, 64, 65, 300, 2500 })
            {
                var longs = Enumerable.Range(0, 2 * n).Select(_ => rng.NextInt64(-1_000_000_000, 1_000_000_000) >> rng.Next(0, 30)).ToArray();
                var doubles = Enumerable.Range(0, 2 * n).Select(_ => (rng.NextDouble() - 0.5) * System.Math.Pow(10, rng.Next(-8, 12))).ToArray();
                using var l = np.array(longs);
                using var d = np.array(doubles);
                foreach (var (name, arr) in new[] { ("i64", l), ("f64", d) })
                {
                    using var contiguous = arr[$"0:{n}"];
                    using var reversed = arr[$"{n - 1}::-1"];
                    using var strided = arr["::2"];
                    foreach (var f in new[] { "%d", "%s", "%.2f", "%e", "%g", "%r", "[%8.3f]", "%x" })
                    {
                        SameAsGeneral(f, contiguous, $"{f} {name} contiguous n={n}");
                        SameAsGeneral(f, reversed, $"{f} {name} reversed n={n}");
                        SameAsGeneral(f, strided, $"{f} {name} strided n={n}");
                        using (var t = T(f))
                            SameAsGeneral(t, contiguous, $"T {f} {name} n={n}");
                        using (var s = S(f))
                            SameAsGeneral(s, contiguous, $"S {f} {name} n={n}");
                    }
                }
                // A (k, 2) block of values against a row of two formats: the format moves along the inner axis.
                if (n >= 2)
                {
                    int rows = n / 2;
                    using var block = np.array(longs.Take(rows * 2).ToArray()).reshape(rows, 2);
                    using var fmts = np.array(new[] { "%d", "<%5x>" });
                    SameAsGeneral(fmts, block, $"row formats n={n}");
                    // A per-element format array cycling through four formats (the element path and the format cache).
                    var cycle = new[] { "%d", "%5d", "%-6d|", "%+d" };
                    using var perElement = np.array(Enumerable.Range(0, n).Select(i => cycle[i % 4]).ToArray());
                    using var head = l[$"0:{n}"];
                    SameAsGeneral(perElement, head, $"per-element formats n={n}");
                }
            }
        }

        /// <summary>
        ///     Text values: <c>%s</c> of <c>U</c> values under str formatting and <c>%s</c> / <c>%b</c> of <c>S</c> values
        ///     under bytes formatting (the chunk path), with precision cuts, widths, empty elements and elements filling the
        ///     whole width; other conversions of a text keep the general route (the same errors).
        /// </summary>
        [TestMethod]
        public void FastPaths_MatchTheGeneralRoute_TextValues()
        {
            var words = Enumerable.Range(0, 200).Select(i => new string((char)('a' + i % 26), i % 9)).ToArray();
            using var u = np.array(words, np.dtype("U8"));
            using var sv = np.array(words.Select(w => (object)Encoding.ASCII.GetBytes(w)).ToArray(), np.dtype("S8"));
            foreach (var f in new[] { "%s", "<%s>", "%10s|", "%-10s|", "%.3s", "%5.2s", "%r", "%d", "%c" })
            {
                SameAsGeneral(f, u, $"U '{f}' over U");
                using (var t = T(f))
                    SameAsGeneral(t, u, $"T '{f}' over U");
                using (var s = S(f))
                    SameAsGeneral(s, sv, $"S '{f}' over S");
            }
            using (var s = S("[%b]"))
                SameAsGeneral(s, sv, "S '[%b]' over S");
        }

        // =============================================================================================================
        // The row storage of a U / S result
        // =============================================================================================================

        /// <summary>
        ///     A longer element after the 64-element sample re-lays the rows out, lengths that keep growing re-lay them out
        ///     several times (exactly first, then by half again) and the result is narrowed once to NumPy's width; every row
        ///     width at the vector-copy boundaries (8, 16, 24, 32 code points; 32, 64 bytes) and past them.
        /// </summary>
        [TestMethod]
        public void Rows_ReLayoutNarrowAndVectorBoundaries()
        {
            // Probed: a long element at position 70 of 101.
            var late = Enumerable.Repeat(1L, 70).Concat(new[] { 123456789L }).Concat(Enumerable.Repeat(2L, 30)).ToArray();
            using (var r = np.strings.mod("%d", np.array(late)))
            {
                r.dtype.Should().Be(np.dtype("U9"));
                Strings(r)[70].Should().Be("123456789");
                Strings(r)[0].Should().Be("1");
                // Every row: the re-layout keeps the rows written before it (the row just before the long element too).
                AssertDecimalRows(r, late, "", "");
            }
            // Probed: 10**0 .. 10**18 five times over: '<1000000000000000000>' sets the width, U21.
            var growing = Enumerable.Range(0, 5).SelectMany(_ => Enumerable.Range(0, 19).Select(k => (long)System.Math.Pow(10, k))).ToArray();
            using (var r = np.strings.mod("<%d>", np.array(growing)))
            {
                r.dtype.Should().Be(np.dtype("U21"));
                Strings(r)[^1].Should().Be("<1000000000000000000>");
            }
            // Strictly increasing lengths past the sample: re-layouts every few elements, then the narrowing. Both routes
            // share the row storage, so the rows are also held to their own text (str(int) is '%d' % int) and width.
            var increasing = Enumerable.Range(0, 400).Select(i => i < 64 ? 1L : (long)System.Math.Pow(10, System.Math.Min(18, (i - 64) / 18))).ToArray();
            SameAsGeneral("%d", np.array(increasing), "increasing lengths");
            SameAsGeneral(S("%d"), np.array(increasing), "S increasing lengths");
            using (var r = np.strings.mod("[%d]", np.array(increasing)))
                AssertDecimalRows(r, increasing, "[", "]");
            using (var sd = S("%d"))
            using (var r = np.strings.mod(sd, np.array(increasing)))
                AssertDecimalRows(r, increasing, "", "");
            // A long element in the LAST rows (the rows whose room the direct writes must not assume: a write that ran past
            // the row would run past the result; NUMSHARP_DEBUG_GUARD_PAGES=1 turns that into a crash).
            foreach (int n in new[] { 65, 100, 257 })
            foreach (int back in new[] { 1, 2, 3 })
            {
                var tail = Enumerable.Repeat(7L, n).ToArray();
                tail[n - back] = 1_000_000_000_000_000_000L;
                using (var r = np.strings.mod("%d", np.array(tail)))
                    AssertDecimalRows(r, tail, "", "");
                using (var sd = S("%d"))
                using (var r = np.strings.mod(sd, np.array(tail)))
                    AssertDecimalRows(r, tail, "", "");
            }
            // Every row width from 1 to 70 code points / bytes, with 200 elements (the last rows take the exact copy).
            for (int width = 1; width <= 70; width++)
            {
                var vals = Enumerable.Range(0, 200).Select(i => (long)(i % 10)).ToArray();
                string f = "%" + width + "d";
                SameAsGeneral(f, np.array(vals), $"U width {width}");
                using (var s = S(f))
                    SameAsGeneral(s, np.array(vals), $"S width {width}");
                // Elements of varying length within the width (zeros to the row's end).
                string g = "%-" + width + "d|";
                SameAsGeneral(g, np.array(vals.Select(v => v * 1000003L).ToArray()), $"U left width {width}");
            }
        }

        /// <summary>
        ///     Holds a <c>U</c> / <c>S</c> result of a <c>%d</c> format over integers to its own text — each element
        ///     <paramref name="prefix"/> + the integer's decimal text + <paramref name="suffix"/> (CPython's <c>'%d' % i</c> is
        ///     <c>str(i)</c>) — and its width (the longest element), a reference that needs neither route.
        /// </summary>
        /// <param name="r">The result.</param>
        /// <param name="values">The integers it formatted.</param>
        /// <param name="prefix">The format's text before the spec.</param>
        /// <param name="suffix">The format's text after the spec.</param>
        private static void AssertDecimalRows(NDArray r, long[] values, string prefix, string suffix)
        {
            var want = values.Select(v => prefix + v.ToString(System.Globalization.CultureInfo.InvariantCulture) + suffix).ToArray();
            int width = want.Max(w => w.Length);
            bool bytes = r.dtype.kind == 'S';
            r.dtype.Should().Be(np.dtype((bytes ? "S" : "U") + width));
            var got = bytes ? Bytes(r) : Strings(r).Cast<string>().ToArray();
            got.Should().Equal(want);
        }

        // =============================================================================================================
        // Errors in C order, the StringDType code-point error
        // =============================================================================================================

        /// <summary>
        ///     The first failing element in C order raises, whichever route formats it: NaN under <c>%d</c> at position 100
        ///     of a large array, a <c>U</c> value of one unit past U+10FFFF (the scalar's SystemError) in the chunk path, and a
        ///     <c>StringDType</c> result whose invalid code point raises only after every element was formatted (a later
        ///     formatting error wins).
        /// </summary>
        [TestMethod]
        public void Errors_KeepCOrder_InLargeArrays()
        {
            var withNaN = Enumerable.Repeat(1.0, 100).Concat(new[] { double.NaN }).Concat(Enumerable.Repeat(2.0, 10)).ToArray();
            Action nan = () => np.strings.mod("%d", np.array(withNaN));
            nan.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");

            // A U value of one unit past U+10FFFF at position 150 (raw units: a C# string cannot hold it); the elements
            // before it ("a") take the chunk path's text field, so the error must still come from the right element.
            var raw = new uint[600];
            for (int i = 0; i < 300; i++)
                raw[2 * i] = 'a';
            raw[2 * 150] = 0x110000;
            using var u = RawU2(raw);
            Action big = () => np.strings.mod("%s", u);
            big.Should().Throw<SystemError>().WithMessage("invalid maximum character passed to PyUnicode_New");
            // Two units, one of them past U+10FFFF, are not an error (CPython caps the maximum character of a longer str)
            // and the result keeps the units (probed: NumPy 2.4.2 gives U2 [.., 0x110000, 0x62, ..] and U4 for '<%s>').
            raw[2 * 150 + 1] = 'b';
            using var u2 = RawU2(raw);
            using (var kept = np.strings.mod("%s", u2))
            using (var units = kept.view(np.uint32))
            {
                kept.dtype.Should().Be(np.dtype("U2"));
                units.GetUInt32(300).Should().Be(0x110000u);
                units.GetUInt32(301).Should().Be('b');
            }
            using (var kept = np.strings.mod("<%s>", u2))
            using (var units = kept.view(np.uint32))
            {
                kept.dtype.Should().Be(np.dtype("U4"));
                Enumerable.Range(600, 4).Select(i => units.GetUInt32(i)).Should().Equal('<', 0x110000u, 'b', '>');
            }

            // A StringDType result: a lone surrogate (a valid U unit) cannot be encoded — raised after the walk...
            var texts = Enumerable.Repeat("ok", 300).ToArray();
            texts[120] = "\ud800";
            using var withSurrogate = np.array(texts, np.dtype("U2"));
            using var tfmt = T("%s");
            Action surrogate = () => np.strings.mod(tfmt, withSurrogate);
            surrogate.Should().Throw<TypeError>().WithMessage("Invalid unicode code point found");
            // ...a U result keeps it (probed: NumPy returns '\ud800' at 120)...
            using (var keptU = np.strings.mod("%s", withSurrogate))
                keptU.item(120).Should().Be("\ud800");
            // ...and a formatting error of a LATER element wins over it: %c of the one-char surrogate formats, %c of a
            // two-char text raises.
            var chars = Enumerable.Repeat("k", 300).ToArray();
            chars[120] = "\ud800";
            using var onlySurrogate = np.array(chars, np.dtype("U2"));
            chars[250] = "ab";
            using var thenTwoChars = np.array(chars, np.dtype("U2"));
            using var cfmt = T("%c");
            Action deferred = () => np.strings.mod(cfmt, onlySurrogate);
            deferred.Should().Throw<TypeError>().WithMessage("Invalid unicode code point found");
            Action order = () => np.strings.mod(cfmt, thenTwoChars);
            order.Should().Throw<TypeError>().WithMessage("%c requires int or char");
        }

        // =============================================================================================================
        // The public function against NumPy 2.4.2
        // =============================================================================================================

        /// <summary>
        ///     Probed outputs (NumPy 2.4.2) of one-argument formats over numbers: integer flags, widths, precision, the
        ///     '#' prefixes, int64 / uint64 extremes, NumPy's scalar repr, float rounding (half to even on the exact value),
        ///     carries, specials with the zero flag, values past FloatDigits' range, and bytes / StringDType formats.
        /// </summary>
        [TestMethod]
        public void Numbers_MatchNumPy()
        {
            Strings(np.strings.mod("%.2f", np.array(new[] { -2.5e17, 0.125, 0.375, 2.5 })))
                .Should().Equal("-250000000000000000.00", "0.12", "0.38", "2.50");
            var huge = (string)Strings(np.strings.mod("%.2f", np.array(new[] { 1e300 })))[0];
            huge.Should().StartWith("1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375");
            huge.Should().EndWith("540160.00");
            Strings(np.strings.mod("%.0f", np.array(new[] { 0.5, 1.5, 2.5, -0.5 }))).Should().Equal("0", "2", "2", "-0");
            Strings(np.strings.mod("%.3e", np.array(new[] { 9.9996, 1e-300, 0.0, -0.0, 1e100 })))
                .Should().Equal("1.000e+01", "1.000e-300", "0.000e+00", "-0.000e+00", "1.000e+100");
            Strings(np.strings.mod("%g", np.array(new[] { 100000.0, 1e-5, 0.0001, 123456789.0, 0.0 })))
                .Should().Equal("100000", "1e-05", "0.0001", "1.23457e+08", "0");
            Strings(np.strings.mod("%#.0f", np.array(new[] { 3.0, -0.0 }))).Should().Equal("3.", "-0.");
            Strings(np.strings.mod("%#.0e", np.array(new[] { 3.0 }))).Should().Equal("3.e+00");
            Strings(np.strings.mod("%+08.2f", np.array(new[] { 3.14159, -2.5, double.PositiveInfinity, double.NegativeInfinity, double.NaN })))
                .Should().Equal("+0003.14", "-0002.50", "+0000inf", "-0000inf", "+0000nan");
            Strings(np.strings.mod("%-8d|", np.array(new[] { -5L, 12345678L, 0L }))).Should().Equal("-5      |", "12345678|", "0       |");
            Strings(np.strings.mod("%#010x", np.array(new[] { 255L, -255L }))).Should().Equal("0x000000ff", "-0x00000ff");
            Strings(np.strings.mod("%#o", np.array(new short[] { 8, 0 }))).Should().Equal("0o10", "0o0");
            Strings(np.strings.mod("%.5d", np.array(new[] { 42, -42 }))).Should().Equal("00042", "-00042");
            Strings(np.strings.mod("%r", np.array(new sbyte[] { 7, -7 }))).Should().Equal("np.int8(7)", "np.int8(-7)");
            Strings(np.strings.mod("%r", np.array(new[] { 0.1f }))).Should().Equal("np.float32(0.1)");
            Strings(np.strings.mod("%r", np.array(new[] { (Half)0.1 }))).Should().Equal("np.float16(0.1)");
            Strings(np.strings.mod("%d", np.array(new[] { long.MinValue, long.MaxValue })))
                .Should().Equal("-9223372036854775808", "9223372036854775807");
            Strings(np.strings.mod("%x", np.array(new[] { ulong.MaxValue }))).Should().Equal("ffffffffffffffff");
            Strings(np.strings.mod("%e", np.array(new[] { ulong.MaxValue }))).Should().Equal("1.844674e+19");
            Strings(np.strings.mod("%.3f", np.array(new[] { long.MinValue }))).Should().Equal("-9223372036854775808.000");
            Bytes(np.strings.mod(S("<%s>"), np.array(new[] { (object)Encoding.ASCII.GetBytes("ab"), Encoding.ASCII.GetBytes("c") }, np.dtype("S2"))))
                .Should().Equal("<ab>", "<c>");
            Bytes(np.strings.mod(S("%5.1b|"), np.array(new[] { (object)Encoding.ASCII.GetBytes("xyz") }, np.dtype("S3"))))
                .Should().Equal("    x|");
            Bytes(np.strings.mod(S("%i"), np.array(new[] { -3L }))).Should().Equal("-3");
            Bytes(np.strings.mod(S("%.2e"), np.array(new[] { 1234.5 }))).Should().Equal("1.23e+03");
            using var t = np.strings.mod(T("%s"), np.array(new[] { 1.5, -2.0 }));
            t.dtype.Should().Be(np.dtypes.StringDType());
            Strings(t).Should().Equal("1.5", "-2.0");
            using var cycle = np.strings.mod(np.array(Enumerable.Range(0, 12).Select(i => new[] { "%d", "%5d", "%-6d|", "%+d" }[i % 4]).ToArray()),
                np.arange(12));
            cycle.dtype.Should().Be(np.dtype("U7"));
            Strings(cycle).Should().Equal("0", "    1", "2     |", "+3", "4", "    5", "6     |", "+7", "8", "    9", "10    |", "+11");
        }

        /// <summary>
        ///     Probed outputs (NumPy 2.4.2) of text conversions: precision 0 keeps no character (<c>'%.0s|'</c> is <c>'|'</c>,
        ///     padded by the width), for <c>U</c>, <c>S</c> and <c>StringDType</c> formats and for a number's str / repr; a
        ///     <c>StringDType</c> format array whose elements differ but share a length; and a <c>StringDType</c> result
        ///     drops its trailing NULs (NumPy builds it through a <c>U</c> array: <c>'%s' % 'a\0'</c> is <c>'a'</c>, an inner
        ///     NUL stays), while the same text inside a format's literal run keeps them only when they are not trailing.
        /// </summary>
        [TestMethod]
        public void Texts_MatchNumPy()
        {
            Strings(np.strings.mod("%.0s|", np.array(new[] { "abc", "" }))).Should().Equal("|", "|");
            using (var r = np.strings.mod("%5.0s|", np.array(new[] { "abc" })))
            {
                r.dtype.Should().Be(np.dtype("U6"));
                Strings(r).Should().Equal("     |");
            }
            using (var f = S("%.0s|"))
                Bytes(np.strings.mod(f, np.array(new object[] { Encoding.ASCII.GetBytes("abc") }, np.dtype("S3")))).Should().Equal("|");
            using (var f = T("%.0s|"))
                Strings(np.strings.mod(f, np.array(new[] { "abc" }))).Should().Equal("|");
            // A number's str / repr cut to precision 0 too (the fast paths' ASCII text field).
            Strings(np.strings.mod("%.0s|", np.array(new[] { 42L, -7L }))).Should().Equal("|", "|");
            Strings(np.strings.mod("%.0r|", np.array(new[] { 1.5 }))).Should().Equal("|");
            using (var r = np.strings.mod("%5.0s|", np.array(new[] { 1.5 })))
            {
                r.dtype.Should().Be(np.dtype("U6"));
                Strings(r).Should().Equal("     |");
            }
            // StringDType formats of one length that differ per element: each element's format is decoded into the same
            // scratch, so the format cache must compare the text, not the address.
            using (var fmts = np.array(new[] { "%d", "%x", "%o", "%X" }, np.dtypes.StringDType()))
                Strings(np.strings.mod(fmts, np.array(new[] { 255L, 255L, 255L, 255L }))).Should().Equal("255", "ff", "377", "FF");
            using var nul = np.array(new[] { "a\0", "a\0b\0\0", "\0", "q" }, np.dtypes.StringDType());
            using (var f = T("%s"))
            using (var r = np.strings.mod(f, nul))
                Strings(r).Should().Equal("a", "a\0b", "", "q");
            using (var f = T("<%s>"))
            using (var r = np.strings.mod(f, nul))
                Strings(r).Should().Equal("<a\0>", "<a\0b\0\0>", "<\0>", "<q>");
        }

        /// <summary>
        ///     Values given as Python sequences (probed: NumPy 2.4.2): a C# list of numbers is NumPy's list (<c>[1, 2]</c> is an
        ///     int64 array, whatever the C# element type), a tuple is a sequence too (one value per format element), and a list
        ///     of bools is NumPy's bool array.
        /// </summary>
        [TestMethod]
        public void PythonSequences_MatchNumPy()
        {
            using (var r = np.strings.mod("%r", new System.Collections.Generic.List<int> { 1, 2 }))
            {
                r.dtype.Should().Be(np.dtype("U11"));
                Strings(r).Should().Equal("np.int64(1)", "np.int64(2)");
            }
            using (var fmts = np.array(new[] { "%s", "%s" }))
                Strings(np.strings.mod(fmts, (1, 2))).Should().Equal("1", "2");
            // A tuple of C# ints is a tuple of Python ints: int64, not the C# int32 (%r shows the dtype).
            Strings(np.strings.mod("%r", (1, 2))).Should().Equal("np.int64(1)", "np.int64(2)");
            using (var r = np.strings.mod("%r", ((1, 2), (3, 4))))
            {
                r.shape.Should().Equal(2L, 2L);
                Strings(r).Should().Equal("np.int64(1)", "np.int64(2)", "np.int64(3)", "np.int64(4)");
            }
            Strings(np.strings.mod("%r", new System.Collections.Generic.List<bool> { true, false })).Should().Equal("np.True_", "np.False_");
        }

        /// <summary>
        ///     NumPy prints a power of two the way CPython's repr does, 17 digits where 16 would read back as the neighbouring
        ///     double: <c>str(np.float64(2**-25))</c> is <c>2.9802322387695312e-08</c> (.NET's own shortest format writes
        ///     <c>2.980232238769531E-08</c>, which reads back as another double). The scalar str, the float → text cast and
        ///     <c>np.strings.mod</c>'s <c>%s</c> / <c>%r</c> all print NumPy's digits.
        /// </summary>
        [TestMethod]
        public void PowersOfTwo_PrintNumPysShortestDigits()
        {
            double a = System.Math.ScaleB(1, -25), b = System.Math.ScaleB(1, -958);
            Strings(np.strings.mod("%s", np.array(new[] { a })))[0].Should().Be("2.9802322387695312e-08");
            Strings(np.strings.mod("%r", np.array(new[] { a, 1.5 }))).Should().Equal("np.float64(2.9802322387695312e-08)", "np.float64(1.5)");
            Strings(np.array(new[] { a, b }).astype(np.dtype("U32"))).Should().Equal("2.9802322387695312e-08", "4.1045368012983762e-289");
            np.array(a).ToString().Should().Be("2.9802322387695312e-08");
            np.array(b).ToString().Should().Be("4.1045368012983762e-289");
        }
    }
}
