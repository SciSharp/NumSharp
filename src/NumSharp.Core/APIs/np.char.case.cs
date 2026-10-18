using System;

// =============================================================================
// np.char.case.cs — the case maps and translate numpy.char re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports capitalize, lower, swapcase, title, upper and translate from
// numpy.strings: the same function objects (np.char.upper is np.strings.upper). Each is `a_arr = np.asarray(a)` and then
// _vec_string over the str / bytes method: not a ufunc (no out= / where= / dtype=), a fresh C-ordered result of a's own
// dtype, and np.asarray drops a chararray's class, so a chararray operand gives a PLAIN array (a 0-d array for a 0-d
// operand or a Python str, never a NumPy scalar). The re-exports convert their operand with the module's text rule
// (CharModule.TextOperand: a byte[] is Python bytes) and call the np.strings function, so every answer (the result, its
// layout, every error and its order) is np.strings'.
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        /// <summary>
        ///     NumPy's <c>np.char.capitalize(a)</c>, which IS <see cref="np.strings.capitalize"/>: each element's first
        ///     character in titlecase and the rest lowered (CPython 3.8+'s <c>str.capitalize</c>: <c>'ǆemal'</c> becomes
        ///     <c>'ǅemal'</c>); <c>bytes.capitalize</c> uppercases the first byte and lowers the other ASCII letters. The text
        ///     is the stored one, a chararray's trailing whitespace included.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes).</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype (a chararray operand too: NumPy's
        ///     <c>np.asarray</c> drops the class), 0-d for a scalar. An <c>S</c> / <c>U</c> result keeps the element width, so
        ///     a mapping that grows is cut.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a str.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has no NumSharp array form.</exception>
        [NDScoped]
        public NDArray capitalize(object a) => np.strings.capitalize(TextOperand(a, nameof(a)));

        /// <summary>
        ///     NumPy's <c>np.char.lower(a)</c>, which IS <see cref="np.strings.lower"/>: each element's text lowercased —
        ///     CPython's <c>str.lower</c> (the full mappings and the final sigma) for <c>U</c> and <c>StringDType</c>,
        ///     <c>bytes.lower</c> (ASCII letters only) for <c>S</c>.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks).</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype (a chararray operand too), 0-d for a
        ///     scalar; an <c>S</c> / <c>U</c> result keeps the element width.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a str.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has no NumSharp array form.</exception>
        [NDScoped]
        public NDArray lower(object a) => np.strings.lower(TextOperand(a, nameof(a)));

        /// <summary>
        ///     NumPy's <c>np.char.swapcase(a)</c>, which IS <see cref="np.strings.swapcase"/>: each element's uppercase
        ///     characters lowered and lowercase characters uppercased (CPython's <c>str.swapcase</c> /
        ///     <c>bytes.swapcase</c>); a titlecase character such as <c>'ǅ'</c> is neither and stays.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks).</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype (a chararray operand too), 0-d for a
        ///     scalar; an <c>S</c> / <c>U</c> result keeps the element width (<c>'ß'</c> swaps to <c>'SS'</c>, cut to
        ///     fit).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a str.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has no NumSharp array form.</exception>
        [NDScoped]
        public NDArray swapcase(object a) => np.strings.swapcase(TextOperand(a, nameof(a)));

        /// <summary>
        ///     NumPy's <c>np.char.title(a)</c>, which IS <see cref="np.strings.title"/>: each element in titlecase — a
        ///     character after a cased character lowered, any other titlecased (CPython's <c>str.title</c>;
        ///     <c>bytes.title</c> over ASCII letters). Any uncased character starts a new "word": <c>'12a'</c> becomes
        ///     <c>'12A'</c>.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks).</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype (a chararray operand too), 0-d for a
        ///     scalar; an <c>S</c> / <c>U</c> result keeps the element width.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a str.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has no NumSharp array form.</exception>
        [NDScoped]
        public NDArray title(object a) => np.strings.title(TextOperand(a, nameof(a)));

        /// <summary>
        ///     NumPy's <c>np.char.upper(a)</c>, which IS <see cref="np.strings.upper"/>: each element's text uppercased —
        ///     CPython's <c>str.upper</c> (the full mappings: <c>'ß'</c> becomes <c>'SS'</c>) for <c>U</c> and
        ///     <c>StringDType</c>, <c>bytes.upper</c> (ASCII letters only) for <c>S</c>.
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks).</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype (a chararray operand too), 0-d for a
        ///     scalar. An <c>S</c> / <c>U</c> result keeps the element width, so a mapping that grows is cut: <c>'ß'</c> in a
        ///     <c>U1</c> array becomes <c>'S'</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     <c>StringDType</c> NA whose NA object is not a str.</exception>
        /// <exception cref="SystemError">A <c>U</c> element holding a code unit past U+10FFFF.</exception>
        /// <exception cref="NotSupportedException"><paramref name="a"/> has no NumSharp array form.</exception>
        [NDScoped]
        public NDArray upper(object a) => np.strings.upper(TextOperand(a, nameof(a)));

        /// <summary>
        ///     NumPy's <c>np.char.translate(a, table, deletechars=None)</c>, which IS <see cref="np.strings.translate"/>:
        ///     every character of each element mapped through <paramref name="table"/> — CPython's <c>str.translate</c> for
        ///     <c>U</c> and <c>StringDType</c>, <c>bytes.translate</c> for <c>S</c> (whose <paramref name="deletechars"/>
        ///     are then deleted).
        /// </summary>
        /// <param name="a">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks).</param>
        /// <param name="table">The table, read as the Python object it stands for and broadcast against
        ///     <paramref name="a"/> (see <see cref="np.strings.translate"/>): a dict keyed by code point for <c>U</c> /
        ///     <c>StringDType</c>, a 256-byte table or null for <c>S</c>.</param>
        /// <param name="deletechars">The bytes to delete (an <c>S</c> operand only); null passes nothing. NumPy ignores it
        ///     for <c>U</c> and passes it to <c>str.translate</c> for <c>StringDType</c>, which refuses it.</param>
        /// <returns>A fresh C-ordered PLAIN array of <paramref name="a"/>'s dtype over the broadcast shape (a chararray
        ///     operand too: NumPy's <c>np.asarray</c> drops the class), 0-d for a scalar; an <c>S</c> / <c>U</c> result
        ///     keeps the element width.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is not text (<c>string operation on non-string array</c>); a
        ///     table or delete set of the wrong kind; <paramref name="deletechars"/> with a <c>StringDType</c> operand
        ///     (<c>str.translate() takes exactly one argument (2 given)</c>); a <c>StringDType</c> NA whose NA object is not
        ///     a str.</exception>
        /// <exception cref="ValueError">A mapping outside <c>range(0x110000)</c>; an <c>S</c> table that is not 256 bytes
        ///     long (<c>translation table must be 256 characters long</c>).</exception>
        /// <exception cref="SystemError">A <c>U</c> unit past U+10FFFF that the call would keep.</exception>
        /// <exception cref="IncorrectShapeException">The arguments do not broadcast.</exception>
        /// <exception cref="NotSupportedException">An argument with no NumSharp array form.</exception>
        [NDScoped]
        public NDArray translate(object a, object table, object deletechars = null)
            => np.strings.translate(TextOperand(a, nameof(a)), table, deletechars);
    }
}
