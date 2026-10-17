using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.strings.Case.cs — numpy.strings' case maps upper / lower / swapcase / capitalize / title (strings phase 2, P2.6)
// =============================================================================
//
// NumPy's upper(a) is `a_arr = np.asarray(a); return _vec_string(a_arr, a_arr.dtype, 'upper')` (the other four alike):
// not a ufunc (no out= / where= / dtype=), a fresh C-ordered result of a's own dtype, each element the bytes / str method's
// result (S / U cut to the element width). np.asarray drops a chararray's class, so a chararray operand gives a plain
// array. The engine is NDStringMethods.CaseMap (S / U) and NDVStringMethods.CaseMap (StringDType).
// =============================================================================

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            /// <summary>
            ///     NumPy's <c>np.strings.upper</c>: each element's text uppercased — CPython's <c>str.upper</c> (the full
            ///     mappings of Unicode 15.0.0: <c>'ß'</c> becomes <c>'SS'</c>, <c>'ﬃ'</c> <c>'FFI'</c>) for <c>U</c> and
            ///     <c>StringDType</c>, <c>bytes.upper</c> (ASCII letters only) for <c>S</c>.
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c> (a C# <c>string</c> converts to a 0-d
            ///     <c>U</c> array).</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype. An <c>S</c> / <c>U</c> result keeps the
            ///     element width, so a mapping that grows is cut: <c>upper</c> of <c>'ß'</c> in a <c>U1</c> array is
            ///     <c>'S'</c>. A <c>StringDType</c> result has <paramref name="a"/>'s NA object and coercion.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
            ///     <c>StringDType</c> NA whose NA object is not a str (<c>descriptor 'upper' for 'str' objects doesn't apply to
            ///     a 'float' object</c>, quoting the object's Python type).</exception>
            /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF (CPython's <c>invalid
            ///     maximum character passed to PyUnicode_New</c>).</exception>
            /// <remarks>
            ///     <para>The element is the text without its trailing NULs for <c>S</c> / <c>U</c> (NumPy's scalar), the whole
            ///     text for <c>StringDType</c> (a NUL is a character there). Under a string NA object
            ///     (<c>StringDType(na_object: "N/A")</c>) an NA is uppercased as that text, and any result equal to the NA's
            ///     text is stored as the NA, as NumPy's setitem stores it.</para>
            ///     <para><c>np.strings.upper(np.array(new[] { "aBc", "straße" }))</c> is <c>["ABC", "STRASS"]</c> (<c>U6</c>).
            ///     A <c>numpy.char.chararray</c> operand gives a plain array (NumPy's <c>np.asarray</c>).</para>
            /// </remarks>
            public static NDArray upper(NDArray a)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain(a, static s => upper(s));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDStringMethods.CaseMap(NDTextElements.UpperMap, a);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.lower</c>: each element's text lowercased — CPython's <c>str.lower</c> (the full
            ///     mappings, <c>'İ'</c> becomes <c>'i̇'</c>, and the final sigma: <c>'ΑΣ'</c> becomes <c>'ας'</c>) for <c>U</c>
            ///     and <c>StringDType</c>, <c>bytes.lower</c> (ASCII letters only) for <c>S</c>.
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c>.</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype (an <c>S</c> / <c>U</c> result cut to the
            ///     element width; a <c>StringDType</c> result with <paramref name="a"/>'s NA object and coercion).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text; a <c>StringDType</c> NA whose NA object is not a
            ///     str (<c>descriptor 'lower' for 'str' objects doesn't apply to a 'NoneType' object</c>).</exception>
            /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
            /// <remarks>U+03A3 lowers to the final sigma U+03C2 when a cased character precedes it (case-ignorable
            ///     characters skipped) and none follows, else to U+03C3, as CPython's <c>lower_ucs4</c> decides.</remarks>
            public static NDArray lower(NDArray a)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain(a, static s => lower(s));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDStringMethods.CaseMap(NDTextElements.LowerMap, a);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.swapcase</c>: each element's uppercase characters lowered and lowercase characters
            ///     uppercased (CPython's <c>str.swapcase</c> / <c>bytes.swapcase</c>); a titlecase character such as
            ///     <c>'ǅ'</c> is neither and stays.
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c>.</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype (an <c>S</c> / <c>U</c> result cut to the
            ///     element width).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text; a <c>StringDType</c> NA whose NA object is not a
            ///     str.</exception>
            /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
            /// <remarks><c>swapcase</c> applied twice is not always the identity: <c>'ß'</c> swaps to <c>'SS'</c>, which swaps
            ///     back to <c>'ss'</c>.</remarks>
            public static NDArray swapcase(NDArray a)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain(a, static s => swapcase(s));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDStringMethods.CaseMap(NDTextElements.SwapcaseMap, a);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.capitalize</c>: each element's first character in titlecase and the rest lowered
            ///     (CPython 3.8+'s <c>str.capitalize</c>: <c>'ǆemal'</c> becomes <c>'ǅemal'</c>); <c>bytes.capitalize</c>
            ///     uppercases the first byte and lowers the other ASCII letters.
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c>.</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype (an <c>S</c> / <c>U</c> result cut to the
            ///     element width).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text; a <c>StringDType</c> NA whose NA object is not a
            ///     str.</exception>
            /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
            /// <remarks>The first character is the element's first one even when it is not a letter: <c>'1abc'</c> stays
            ///     <c>'1abc'</c>, and a leading NUL stays first.</remarks>
            public static NDArray capitalize(NDArray a)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain(a, static s => capitalize(s));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDStringMethods.CaseMap(NDTextElements.CapitalizeMap, a);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.title</c>: each element in titlecase — a character after a cased character lowered,
            ///     any other titlecased (CPython's <c>str.title</c>; <c>bytes.title</c> over ASCII letters).
            /// </summary>
            /// <param name="a">The strings: <c>S</c>, <c>U</c> or <c>StringDType</c>.</param>
            /// <returns>A fresh C-ordered array of <paramref name="a"/>'s dtype (an <c>S</c> / <c>U</c> result cut to the
            ///     element width).</returns>
            /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="a"/> is not text; a <c>StringDType</c> NA whose NA object is not a
            ///     str.</exception>
            /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
            /// <remarks>Words are not detected: any uncased character starts a new "word", so <c>"they're bill's"</c>
            ///     becomes <c>"They'Re Bill'S"</c> and <c>"hello2world"</c> <c>"Hello2World"</c>, as in Python.</remarks>
            public static NDArray title(NDArray a)
            {
                if (NDCharArrayBoundary.Sees(a))
                    return NDCharArrayBoundary.Plain(a, static s => title(s));
                if (a is null)
                    throw new ArgumentNullException(nameof(a));
                return NDStringMethods.CaseMap(NDTextElements.TitleMap, a);
            }
        }
    }
}
