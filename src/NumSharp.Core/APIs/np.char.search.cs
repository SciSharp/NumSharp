using System;

// =============================================================================
// np.char.search.cs — the length and search functions numpy.char re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports str_len, count, find, rfind, index, rindex, startswith and endswith
// from numpy.strings, so np.char.find IS np.strings.find (the same function object; str_len is the ufunc itself). Each
// returns a PLAIN int64 / bool array, for a chararray operand too: chararray.__array_wrap__ keeps the class only for a
// text result. The re-exports below convert their array_likes with the module's text rule (CharModule.TextOperand: a
// byte[] is Python bytes) and call the np.strings function, so every answer — the result, its layout, every error and its
// order — is np.strings', as it is in NumPy.
//
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        // =============================================================================================================
        // str_len (numpy.strings' ufunc, re-exported)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.str_len(x)</c>, which IS the <see cref="np.strings.str_len"/> ufunc: the length of every
        ///     element as int64 — characters for <c>U</c>, bytes for <c>S</c>, code points for <c>StringDType</c>. Trailing
        ///     NULs never count, embedded ones do; trailing whitespace counts (a chararray's stored text is measured, not its
        ///     stripped elements).
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the class
        ///     remarks). A number array has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype int64 casts to under same_kind), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only int64 has a loop; null leaves the loop to the operand.</param>
        /// <returns>A plain int64 array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back as a
        ///     new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not int64 (NumPy's
        ///     no-loop text).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive int64 (the same_kind output cast); a
        ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a <c>StringDType</c> NA whose descriptor's NA
        ///     object is not a string (<c>The length of a null string is undefined</c>).</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        [NDScoped]
        public NDArray str_len(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.str_len(TextOperand(x, nameof(x)), @out, where, dtype);

        // =============================================================================================================
        // count / find / rfind / index / rindex / startswith / endswith (numpy.strings' functions, re-exported)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.char.count(a, sub, start=0, end=None)</c>, which IS <see cref="np.strings.count"/>: the number of
        ///     non-overlapping occurrences of <paramref name="sub"/> in each element's <c>a[start:end]</c>, counted greedily
        ///     from the left; an empty substring occurs <c>len + 1</c> times in the range.
        /// </summary>
        /// <param name="a">The searched strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="sub">The substrings, broadcast: the same class as <paramref name="a"/> (a <c>U</c> one beside a
        ///     <c>StringDType</c> <paramref name="a"/>, or the reverse, is cast to <c>StringDType</c>; a <c>bytes</c> one
        ///     beside <c>U</c> or a number has no loop).</param>
        /// <param name="start">The range starts in slice notation (negative counts from the end), any integer dtype,
        ///     broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c> (the whole
        ///     string).</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for scalar operands); a chararray operand still gives a
        ///     plain array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes (an <c>S</c> array with a <c>str</c> substring, a
        ///     number substring: NumPy's no-loop text); two <c>StringDType</c> operands with different NA objects.</exception>
        /// <exception cref="ArgumentException"><paramref name="start"/> / <paramref name="end"/> cannot reach int64 by a
        ///     same_kind cast (a float or text: <c>Cannot cast ufunc 'count' input 2 from dtype('float64') to
        ///     dtype('int64') with casting rule 'same_kind'</c>).</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
        [NDScoped]
        public NDArray count(object a, object sub, object start = null, object end = null)
            => np.strings.count(TextOperand(a, nameof(a)), TextOperand(sub, nameof(sub)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.find(a, sub, start=0, end=None)</c>, which IS <see cref="np.strings.find"/>: for each
        ///     element, the lowest index where <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1. Indices are
        ///     characters (code points for <c>StringDType</c>); trailing NULs never count.
        /// </summary>
        /// <param name="a">The searched strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="sub">The substrings, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
        [NDScoped]
        public NDArray find(object a, object sub, object start = null, object end = null)
            => np.strings.find(TextOperand(a, nameof(a)), TextOperand(sub, nameof(sub)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.rfind(a, sub, start=0, end=None)</c>, which IS <see cref="np.strings.rfind"/>: for each
        ///     element, the highest index where <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1 (an empty
        ///     substring is found at the end of the range).
        /// </summary>
        /// <param name="a">The searched strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="sub">The substrings, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
        [NDScoped]
        public NDArray rfind(object a, object sub, object start = null, object end = null)
            => np.strings.rfind(TextOperand(a, nameof(a)), TextOperand(sub, nameof(sub)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.index(a, sub, start=0, end=None)</c>, which IS <see cref="np.strings.index"/>:
        ///     <see cref="find"/>, but an element with no match raises <c>ValueError: substring not found</c>.
        /// </summary>
        /// <param name="a">The searched strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="sub">The substrings, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An element without a match (<c>substring not found</c>); a <c>StringDType</c> NA
        ///     whose NA object is not a string.</exception>
        [NDScoped]
        public NDArray index(object a, object sub, object start = null, object end = null)
            => np.strings.index(TextOperand(a, nameof(a)), TextOperand(sub, nameof(sub)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.rindex(a, sub, start=0, end=None)</c>, which IS <see cref="np.strings.rindex"/>:
        ///     <see cref="rfind"/>, but an element with no match raises <c>ValueError: substring not found</c>.
        /// </summary>
        /// <param name="a">The searched strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="sub">The substrings, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">An element without a match; a <c>StringDType</c> NA whose NA object is not a
        ///     string.</exception>
        [NDScoped]
        public NDArray rindex(object a, object sub, object start = null, object end = null)
            => np.strings.rindex(TextOperand(a, nameof(a)), TextOperand(sub, nameof(sub)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.startswith(a, prefix, start=0, end=None)</c>, which IS <see cref="np.strings.startswith"/>:
        ///     whether each element's <c>a[start:end]</c> begins with <paramref name="prefix"/>.
        /// </summary>
        /// <param name="a">The tested strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="prefix">The prefixes, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain bool array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="prefix"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is neither a string nor NaN-like (a
        ///     NaN-like NA is False).</exception>
        [NDScoped]
        public NDArray startswith(object a, object prefix, object start = null, object end = null)
            => np.strings.startswith(TextOperand(a, nameof(a)), TextOperand(prefix, nameof(prefix)), OptionalOperand(start), OptionalOperand(end));

        /// <summary>
        ///     NumPy's <c>np.char.endswith(a, suffix, start=0, end=None)</c>, which IS <see cref="np.strings.endswith"/>:
        ///     whether each element's <c>a[start:end]</c> ends with <paramref name="suffix"/>.
        /// </summary>
        /// <param name="a">The tested strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like.</param>
        /// <param name="suffix">The suffixes, broadcast (the classes of <see cref="count"/>).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain bool array of the broadcast shape (0-d for scalar operands).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="suffix"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
        /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is neither a string nor NaN-like.</exception>
        [NDScoped]
        public NDArray endswith(object a, object suffix, object start = null, object end = null)
            => np.strings.endswith(TextOperand(a, nameof(a)), TextOperand(suffix, nameof(suffix)), OptionalOperand(start), OptionalOperand(end));
    }
}
