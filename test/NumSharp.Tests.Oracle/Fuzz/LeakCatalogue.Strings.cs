using System;
using System.Collections.Generic;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     Catalogue entries for the <c>numpy.strings</c> functions of strings phase 2, P2.5 (<c>str_len</c>, the find family,
    ///     <c>startswith</c> / <c>endswith</c>, the strips, the pads, <c>zfill</c>, <c>expandtabs</c>, <c>replace</c>,
    ///     <c>partition</c> / <c>rpartition</c>, <c>slice</c> and the six comparison names) and P2.6 (the predicates
    ///     <c>isalpha</c> … <c>isnumeric</c>). The string oracle's sections <c>strfn</c> / <c>tfn</c> / <c>strprop</c> /
    ///     <c>tprop</c> replay them through its own harness, outside the corpus sweep, so each is measured here.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every entry takes the U, S and StringDType (T) loops where NumPy has them, the NA forms of a StringDType (a string
    ///     <c>na_object</c>, a NaN-like one, None) and the refusals (index's ValueError, an NA the loop cannot use, the
    ///     TypeErrors of an integer operand), whose temporaries — the operands the promoters cast, the arena of a T result
    ///     the error abandons, the scratch buffer of a stepped slice — must be reclaimed as well as the success path's.
    ///     </para>
    ///     <para>
    ///     A Python str or int operand is HELD in a variable (<c>using NDArray b = "b";</c>), never written as a literal at
    ///     the call: the implicit conversion's 0-d array is the CALLER's temporary, which no library fix can dispose, so the
    ///     gate would count it against the function.
    ///     </para>
    /// </remarks>
    internal static partial class LeakCatalogue
    {
        /// <summary>The P2.5 <c>np.strings</c> functions, one entry or more per member.</summary>
        /// <param name="l">The entry list.</param>
        private static void AddStringsFunctions(List<LeakCase> l)
        {
            E(l, "np.strings.str_len", "U, S, T with a string NA; out= and where=; the null length's ValueError", f =>
            {
                using var u = np.array(new[] { "ab", "cde" });
                using var s = np.array(new object[] { new byte[] { 97 }, new byte[] { 98, 99 } });
                using var t = np.array(new object[] { "ab", null }, np.dtypes.StringDType("NA"));
                using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                using var o = np.zeros(new Shape(2), np.int64);
                using var mask = np.array(new[] { true, false });
                using var bytes = np.strings.str_len(s);
                using var vstring = np.strings.str_len(t);
                np.strings.str_len(u, o, mask);
                try
                {
                    np.strings.str_len(none);
                }
                catch (ValueError)
                {
                }
                return np.strings.str_len(u);
            });
            E(l, "np.strings.find", "U / S / T, start and end arrays, a Python str and a bytes needle, a string NA", f =>
            {
                using var u = np.array(new[] { "abcab", "b" });
                using var s = np.array(new object[] { new byte[] { 97, 98 }, new byte[] { 98 } });
                using var t = np.array(new object[] { "abcab", null }, np.dtypes.StringDType("ab"));
                using var start = np.array(new[] { 1L, 0L });
                using var end = np.array(new[] { 5L, -1L });
                using NDArray b = "b";
                using var bb = NDArray.Scalar(new byte[] { 98 });
                using var bytes = np.strings.find(s, bb);
                using var vstring = np.strings.find(t, b, start, end);
                return np.strings.find(u, b, start);
            });
            E(l, "np.strings.rfind", "U and T (a long needle, an end array), the None NA's ValueError", f =>
            {
                using var u = np.array(new[] { "abcab", "b" });
                using var t = np.array(new[] { "abcabcabcabcabcabcab", "b" }, np.dtypes.StringDType());
                using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                using var end = np.array(new[] { 4L, 1L });
                using NDArray ab = "ab";
                using var vstring = np.strings.rfind(t, ab, null, end);
                try
                {
                    np.strings.rfind(none, ab);
                }
                catch (ValueError)
                {
                }
                return np.strings.rfind(u, ab);
            });
            E(l, "np.strings.index", "U and T found; U and T not found (ValueError)", f =>
            {
                using var u = np.array(new[] { "abc", "b" });
                using var t = np.array(new[] { "abc", "b" }, np.dtypes.StringDType());
                using NDArray b = "b";
                using NDArray z = "z";
                using var vstring = np.strings.index(t, b);
                try
                {
                    np.strings.index(u, z);
                }
                catch (ValueError)
                {
                }
                try
                {
                    np.strings.index(t, z);
                }
                catch (ValueError)
                {
                }
                return np.strings.index(u, b);
            });
            E(l, "np.strings.rindex", "U and T found; S not found (ValueError)", f =>
            {
                using var u = np.array(new[] { "abcb", "b" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "abcb", "b" }, np.dtypes.StringDType());
                using NDArray b = "b";
                using var z = NDArray.Scalar(new byte[] { 122 });
                using var vstring = np.strings.rindex(t, b);
                try
                {
                    np.strings.rindex(s, z);
                }
                catch (ValueError)
                {
                }
                return np.strings.rindex(u, b);
            });
            E(l, "np.strings.count", "U, S, T, an empty needle, start / end", f =>
            {
                using var u = np.array(new[] { "aaa", "ab" });
                using var s = np.array(new object[] { new byte[] { 97, 97 } });
                using var t = np.array(new[] { "aaéaé", "" }, np.dtypes.StringDType());
                using NDArray a = "a";
                using NDArray empty = "";
                using NDArray one = 1;
                using var aa = NDArray.Scalar(new byte[] { 97 });
                using var bytes = np.strings.count(s, aa);
                using var vstring = np.strings.count(t, a, one);
                using var everywhere = np.strings.count(t, empty);
                return np.strings.count(u, a);
            });
            E(l, "np.strings.startswith", "U, S, T with start / end, a NaN NA (False), the None NA's ValueError", f =>
            {
                using var u = np.array(new[] { "abc", "b" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new object[] { "abc", null }, np.dtypes.StringDType(double.NaN));
                using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                using var start = np.array(new[] { 0L, 1L });
                using NDArray a = "a";
                using var aa = NDArray.Scalar(new byte[] { 97 });
                using var bytes = np.strings.startswith(s, aa);
                using var vstring = np.strings.startswith(t, a, start);
                try
                {
                    np.strings.startswith(none, a);
                }
                catch (ValueError)
                {
                }
                return np.strings.startswith(u, a);
            });
            E(l, "np.strings.endswith", "U and T with an end array, an S suffix", f =>
            {
                using var u = np.array(new[] { "abc", "bc" });
                using var s = np.array(new object[] { new byte[] { 97, 98 } });
                using var t = np.array(new[] { "abc", "c" }, np.dtypes.StringDType());
                using var end = np.array(new[] { 3L, -1L });
                using NDArray c = "c";
                using var bb = NDArray.Scalar(new byte[] { 98 });
                using var bytes = np.strings.endswith(s, bb);
                using var vstring = np.strings.endswith(t, c, null, end);
                return np.strings.endswith(u, c);
            });
            E(l, "np.strings.strip", "U whitespace, S and T chars, a T NA string, the None NA's ValueError", f =>
            {
                using var u = np.array(new[] { "  ab ", "\tc" });
                using var s = np.array(new object[] { new byte[] { 120, 97, 120 } });
                using var t = np.array(new object[] { "xxabx", null }, np.dtypes.StringDType("xx"));
                using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                using NDArray x = "x";
                using var xx = NDArray.Scalar(new byte[] { 120 });
                using var bytes = np.strings.strip(s, xx);
                using var vstring = np.strings.strip(t, x);
                using var white = np.strings.strip(t);
                try
                {
                    np.strings.strip(none);
                }
                catch (ValueError)
                {
                }
                return np.strings.strip(u);
            });
            E(l, "np.strings.lstrip", "U and T, whitespace and chars", f =>
            {
                using var u = np.array(new[] { "  ab ", "xc" });
                using var t = np.array(new[] { "  ab ", "xé" }, np.dtypes.StringDType());
                using NDArray x = "x";
                using var vstring = np.strings.lstrip(t, x);
                using var white = np.strings.lstrip(t);
                return np.strings.lstrip(u, x);
            });
            E(l, "np.strings.rstrip", "U and T, whitespace and chars, a T chars array", f =>
            {
                using var u = np.array(new[] { "  ab ", "cx" });
                using var t = np.array(new[] { " ab  ", "éx" }, np.dtypes.StringDType());
                using var chars = np.array(new[] { " ", "x" }, np.dtypes.StringDType());
                using var vstring = np.strings.rstrip(t, chars);
                using var white = np.strings.rstrip(t);
                return np.strings.rstrip(u);
            });
            E(l, "np.strings.center", "U / S / T, a width array, a fill character; a two-character fill and a float width (TypeError)", f =>
            {
                using var u = np.array(new[] { "ab", "c" });
                using var s = np.array(new object[] { new byte[] { 97 } });
                using var t = np.array(new object[] { "ab", null }, np.dtypes.StringDType(double.NaN));
                using var widths = np.array(new[] { 5L, 2L });
                using NDArray star = "*";
                using NDArray two = "**";
                using NDArray width = 6;
                using NDArray half = 2.5;
                using var bytes = np.strings.center(s, width);
                using var vstring = np.strings.center(t, widths, star);
                try
                {
                    np.strings.center(u, width, two);
                }
                catch (TypeError)
                {
                }
                try
                {
                    np.strings.center(t, half);
                }
                catch (TypeError)
                {
                }
                return np.strings.center(u, widths, star);
            });
            E(l, "np.strings.ljust", "U and T with a fill character, the None NA's ValueError", f =>
            {
                using var u = np.array(new[] { "ab", "c" });
                using var t = np.array(new[] { "ab", "é" }, np.dtypes.StringDType());
                using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                using NDArray dash = "-";
                using NDArray width = 4;
                using var vstring = np.strings.ljust(t, width, dash);
                try
                {
                    np.strings.ljust(none, width);
                }
                catch (ValueError)
                {
                }
                return np.strings.ljust(u, width);
            });
            E(l, "np.strings.rjust", "U and T, a width array", f =>
            {
                using var u = np.array(new[] { "ab", "c" });
                using var t = np.array(new[] { "ab", "é" }, np.dtypes.StringDType());
                using var widths = np.array(new[] { 4L, 3L });
                using var vstring = np.strings.rjust(t, widths);
                return np.strings.rjust(u, widths);
            });
            E(l, "np.strings.zfill", "U, S, T with signs; a float width (TypeError)", f =>
            {
                using var u = np.array(new[] { "-7", "+12" });
                using var s = np.array(new object[] { new byte[] { 45, 55 } });
                using var t = np.array(new[] { "-7", "é" }, np.dtypes.StringDType());
                using NDArray width = 5;
                using NDArray half = 2.5;
                using var bytes = np.strings.zfill(s, width);
                using var vstring = np.strings.zfill(t, width);
                try
                {
                    np.strings.zfill(u, half);
                }
                catch (TypeError)
                {
                }
                return np.strings.zfill(u, width);
            });
            E(l, "np.strings.expandtabs", "U, S, T, the default and a tabsize array; the None NA's ValueError", f =>
            {
                using var u = np.array(new[] { "a\tb", "\t" });
                using var s = np.array(new object[] { new byte[] { 97, 9, 98 } });
                using var t = np.array(new[] { "a\tb", "é\t" }, np.dtypes.StringDType());
                using var none = np.array(new object[] { "a\t", null }, np.dtypes.StringDType(null));
                using var sizes = np.array(new[] { 4L, 2L });
                using var bytes = np.strings.expandtabs(s);
                using var vstring = np.strings.expandtabs(t, sizes);
                try
                {
                    np.strings.expandtabs(none);
                }
                catch (ValueError)
                {
                }
                return np.strings.expandtabs(u, sizes);
            });
            E(l, "np.strings.replace", "U / S / T, a count array, a longer and an empty replacement, a T NA string", f =>
            {
                using var u = np.array(new[] { "aXbXc", "X" });
                using var s = np.array(new object[] { new byte[] { 97, 88, 98 } });
                using var t = np.array(new object[] { "aXbXc", null }, np.dtypes.StringDType("X"));
                using var counts = np.array(new[] { 1L, -1L });
                using NDArray x = "X";
                using NDArray longer = "-longer-";
                using NDArray empty = "";
                using var xb = NDArray.Scalar(new byte[] { 88 });
                using var eb = NDArray.Scalar(new byte[] { 45, 45 });
                using var bytes = np.strings.replace(s, xb, eb);
                using var vstring = np.strings.replace(t, x, longer, counts);
                using var removed = np.strings.replace(t, x, empty);
                return np.strings.replace(u, x, longer, counts);
            });
            // A StringDType text takes a StringDType separator: NumPy's _partition has no promoter, so a U separator is the
            // no-loop TypeError (a refusal whose temporaries the entry measures too).
            E(l, "np.strings.partition", "U / S / T separators; the U-separator TypeError and the empty separator's ValueError", f =>
            {
                using var u = np.array(new[] { "a-b-c", "c" });
                using var s = np.array(new object[] { new byte[] { 97, 45, 98 } });
                using var t = np.array(new[] { "a-b-c", "é" }, np.dtypes.StringDType());
                using var tdash = np.array(new[] { "-" }, np.dtypes.StringDType());
                using NDArray dash = "-";
                using NDArray empty = "";
                using var db = NDArray.Scalar(new byte[] { 45 });
                var (sb, ss, sa) = np.strings.partition(s, db);
                sb.Dispose();
                ss.Dispose();
                sa.Dispose();
                var (tb, ts, ta) = np.strings.partition(t, tdash);
                tb.Dispose();
                ts.Dispose();
                ta.Dispose();
                try
                {
                    np.strings.partition(t, dash);
                }
                catch (TypeError)
                {
                }
                try
                {
                    np.strings.partition(u, empty);
                }
                catch (ValueError)
                {
                }
                return np.strings.partition(u, dash);
            });
            E(l, "np.strings.rpartition", "U and T separators, a T NA string; the empty separator's ValueError", f =>
            {
                using var u = np.array(new[] { "a-b-c", "c" });
                using var t = np.array(new object[] { "a-b-c", null }, np.dtypes.StringDType("x-y"));
                using var tdash = np.array(new[] { "-" }, np.dtypes.StringDType("x-y"));
                using var tempty = np.array(new[] { "" }, np.dtypes.StringDType("x-y"));
                using NDArray dash = "-";
                var (tb, ts, ta) = np.strings.rpartition(t, tdash);
                tb.Dispose();
                ts.Dispose();
                ta.Dispose();
                try
                {
                    np.strings.rpartition(t, tempty);
                }
                catch (ValueError)
                {
                }
                return np.strings.rpartition(u, dash);
            });
            E(l, "np.strings.slice", "the four arities over U, S and T (a stepped non-ASCII slice); a zero step's ValueError", f =>
            {
                using var u = np.array(new[] { "abcdef", "xy" });
                using var s = np.array(new object[] { new byte[] { 97, 98, 99 } });
                using var t = np.array(new[] { "aébécé", "xy" }, np.dtypes.StringDType());
                using NDArray one = 1;
                using NDArray two = 2;
                using NDArray back = -1;
                using NDArray zero = 0;
                using var whole = np.strings.slice(u);
                using var bytes = np.strings.slice(s, two);
                using var middle = np.strings.slice(t, one, back);
                using var stepped = np.strings.slice(t, null, null, two);
                using var reversed = np.strings.slice(t, null, null, back);
                try
                {
                    np.strings.slice(u, null, null, zero);
                }
                catch (ValueError)
                {
                }
                return np.strings.slice(u, one, null, two);
            });
            // The comparison names are the comparison ufuncs: one shape of entry, U against U, S against S, T against U (the
            // promoter's cast of the U operand).
            foreach (var (name, compare) in new (string, Func<NDArray, NDArray, NDArray>)[]
                     {
                         ("equal", (a, b) => np.strings.equal(a, b)),
                         ("not_equal", (a, b) => np.strings.not_equal(a, b)),
                         ("less", (a, b) => np.strings.less(a, b)),
                         ("less_equal", (a, b) => np.strings.less_equal(a, b)),
                         ("greater", (a, b) => np.strings.greater(a, b)),
                         ("greater_equal", (a, b) => np.strings.greater_equal(a, b)),
                     })
            {
                E(l, "np.strings." + name, "U against U, S against S, T against U", f =>
                {
                    using var u = np.array(new[] { "ab", "c" });
                    using var v = np.array(new[] { "ab", "b" });
                    using var s = np.array(new object[] { new byte[] { 97 }, new byte[] { 98 } });
                    using var t = np.array(new[] { "ab", "d" }, np.dtypes.StringDType());
                    using var bytes = compare(s, s);
                    using var vstring = compare(t, u);
                    return compare(u, v);
                });
            }
            AddStringsPredicates(l);
        }

        /// <summary>
        ///     The P2.6 predicates <c>isalpha</c> … <c>isnumeric</c>: one entry per ufunc over every loop and route that holds a
        ///     temporary — the <c>U</c> loop on ASCII and non-ASCII text (the call's escalation), the <c>S</c> loop or its
        ///     no-loop TypeError (no <c>S</c> loop for <c>isdecimal</c> / <c>isnumeric</c>), the <c>StringDType</c> loop with a
        ///     string NA, a NaN-like NA and the ValueError of a None NA (the arena locks and the default string's pin), an out= of
        ///     another dtype under a mask (the bool temporary NumPy's output cast reads), and <c>dtype=</c>'s TypeError.
        /// </summary>
        /// <param name="l">The entry list.</param>
        /// <remarks>Each operand is held and disposed by the entry, so the gate counts only what the ufunc itself leaves
        ///     behind.</remarks>
        private static void AddStringsPredicates(List<LeakCase> l)
        {
            foreach (var (name, predicate) in new (string, Func<NDArray, NDArray, NDArray, DType, NDArray>)[]
                     {
                         ("isalpha", (x, o, w, d) => np.strings.isalpha(x, o, w, d)),
                         ("isdigit", (x, o, w, d) => np.strings.isdigit(x, o, w, d)),
                         ("isspace", (x, o, w, d) => np.strings.isspace(x, o, w, d)),
                         ("isalnum", (x, o, w, d) => np.strings.isalnum(x, o, w, d)),
                         ("islower", (x, o, w, d) => np.strings.islower(x, o, w, d)),
                         ("isupper", (x, o, w, d) => np.strings.isupper(x, o, w, d)),
                         ("istitle", (x, o, w, d) => np.strings.istitle(x, o, w, d)),
                         ("isdecimal", (x, o, w, d) => np.strings.isdecimal(x, o, w, d)),
                         ("isnumeric", (x, o, w, d) => np.strings.isnumeric(x, o, w, d)),
                     })
            {
                // NumPy registers the S (ENCODING::ASCII) loop for the first seven predicates only.
                bool bytesLoop = name is not ("isdecimal" or "isnumeric");
                E(l, "np.strings." + name,
                  "U ASCII and not, S (or its no-loop TypeError), T with string / NaN NAs and None's ValueError, out= int8 under where=, dtype='i8''s TypeError",
                  f =>
                  {
                      using var u = np.array(new[] { "Ab1", "é ²Ω" });
                      using var s = np.array(new object[] { new byte[] { 65, 98 }, new byte[] { 49 } });
                      using var t = np.array(new object[] { "Ab Cd", null, "héllo wörld and a text past the cell" }, np.dtypes.StringDType("NA"));
                      using var nan = np.array(new object[] { "ab", double.NaN }, np.dtypes.StringDType(double.NaN));
                      using var none = np.array(new object[] { "ab", null }, np.dtypes.StringDType(null));
                      using var o = np.zeros(new Shape(2), np.int8);
                      using var mask = np.array(new[] { true, false });
                      using var vstring = predicate(t, null, null, null);
                      using var nans = predicate(nan, null, null, null);
                      predicate(u, o, mask, null);
                      if (bytesLoop)
                      {
                          using var bytes = predicate(s, null, null, null);
                      }
                      else
                      {
                          try
                          {
                              predicate(s, null, null, null);
                          }
                          catch (TypeError)
                          {
                          }
                      }
                      try
                      {
                          predicate(none, null, null, null);
                      }
                      catch (ValueError)
                      {
                      }
                      try
                      {
                          predicate(u, null, null, np.int64);
                      }
                      catch (TypeError)
                      {
                      }
                      return predicate(u, null, null, null);
                  });
            }
        }
    }
}
