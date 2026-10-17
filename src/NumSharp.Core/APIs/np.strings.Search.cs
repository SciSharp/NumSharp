using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Strings;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            // =========================================================================================================
            // str_len and the comparison ufuncs (numpy.strings re-exports NumPy's ufunc objects themselves)
            // =========================================================================================================

            /// <summary>
            ///     NumPy's <c>np.strings.str_len</c> ufunc: the length of every element as int64 — characters for <c>U</c>,
            ///     bytes for <c>S</c>, code points for <c>StringDType</c>. Trailing NULs never count, embedded ones do
            ///     (<c>'a\0b'</c> has length 3, <c>'ab\0'</c> length 2).
            /// </summary>
            /// <param name="x">The strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="out">An output to write into and return (any dtype int64 casts to under same_kind), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only int64 has a loop.</param>
            /// <returns>The lengths, or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not int64 (NumPy's
            ///     no-loop text).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive int64 (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a <c>StringDType</c> NA whose descriptor's NA
            ///     object is not a string (<c>The length of a null string is undefined</c>).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>A <c>numpy.char.chararray</c> operand gives a plain int64 result (<c>chararray.__array_wrap__</c>
            ///     keeps only text results); a chararray <paramref name="out"/> comes back as a new view of it.</remarks>
            public static NDArray str_len(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => str_len(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.StrLen(x, @out, where, dtype);
            }

            /// <summary>NumPy's <c>np.strings.equal</c>, which IS the <c>np.equal</c> ufunc (<c>np.strings.equal is np.equal</c>).</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands (NumPy's no-loop text).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.equal(x1, x2, @out, where, dtype);

            /// <summary>NumPy's <c>np.strings.not_equal</c>, which IS the <c>np.not_equal</c> ufunc.</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray not_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.not_equal(x1, x2, @out, where, dtype);

            /// <summary>NumPy's <c>np.strings.less</c>, which IS the <c>np.less</c> ufunc.</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray less(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.less(x1, x2, @out, where, dtype);

            /// <summary>NumPy's <c>np.strings.less_equal</c>, which IS the <c>np.less_equal</c> ufunc.</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray less_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.less_equal(x1, x2, @out, where, dtype);

            /// <summary>NumPy's <c>np.strings.greater</c>, which IS the <c>np.greater</c> ufunc.</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray greater(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.greater(x1, x2, @out, where, dtype);

            /// <summary>NumPy's <c>np.strings.greater_equal</c>, which IS the <c>np.greater_equal</c> ufunc.</summary>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <param name="out">An output to write into and return, or null.</param>
            /// <param name="where">A bool mask, or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c> (bool loops only).</param>
            /// <returns>The bool result, or <paramref name="out"/>.</returns>
            /// <exception cref="TypeError">No comparison loop for the operands.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            public static NDArray greater_equal(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
                => np.greater_equal(x1, x2, @out, where, dtype);

            // =========================================================================================================
            // find / rfind / index / rindex / count / startswith / endswith
            // =========================================================================================================

            /// <summary>
            ///     NumPy's <c>np.strings.find(a, sub, start=0, end=None)</c>: for each element, the lowest index where
            ///     <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1. Indices are characters (code points for
            ///     <c>StringDType</c>); trailing NULs of either string never count.
            /// </summary>
            /// <param name="a">The searched strings (<c>S</c>, <c>U</c> or <c>StringDType</c>).</param>
            /// <param name="sub">The substrings: the same class as <paramref name="a"/> (a <c>U</c> one with a
            ///     <c>StringDType</c> <paramref name="a"/>, or the reverse, is cast to <c>StringDType</c>).</param>
            /// <param name="start">The range start, slice notation (negative counts from the end), any integer dtype; null is
            ///     NumPy's default 0.</param>
            /// <param name="end">The range end, slice notation; null is NumPy's <c>None</c> (the whole string).</param>
            /// <returns>An int64 array of the broadcast shape (0-d for scalar operands).</returns>
            /// <exception cref="TypeError">No loop for the operand classes (an <c>S</c> array with a <c>str</c> substring, a
            ///     number, …: NumPy's no-loop text); two <c>StringDType</c> operands with different NA objects.</exception>
            /// <exception cref="ArgumentException"><paramref name="start"/> / <paramref name="end"/> cannot reach int64 by a
            ///     same_kind cast (a float: <c>Cannot cast ufunc 'find' input 2 from dtype('float64') …</c>).</exception>
            /// <exception cref="OverflowException">A start / end literal past int64 (<c>Python int too large to convert to C
            ///     long</c>).</exception>
            /// <exception cref="IncorrectShapeException">The four operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
            public static NDArray find(NDArray a, NDArray sub, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, sub))
                    return NDCharArrayBoundary.Wrapped((a, sub, start, end), static s => find(s.a, s.sub, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(sub, nameof(sub));
                return NDStringMethods.FindLike("find", TextMethodOp.Find, a, sub, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.rfind(a, sub, start=0, end=None)</c>: for each element, the highest index where
            ///     <paramref name="sub"/> occurs inside <c>a[start:end]</c>, or -1 (an empty substring is found at the end).
            /// </summary>
            /// <param name="a">The searched strings.</param>
            /// <param name="sub">The substrings (see <see cref="find"/>).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>An int64 array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
            public static NDArray rfind(NDArray a, NDArray sub, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, sub))
                    return NDCharArrayBoundary.Wrapped((a, sub, start, end), static s => rfind(s.a, s.sub, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(sub, nameof(sub));
                return NDStringMethods.FindLike("rfind", TextMethodOp.RFind, a, sub, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.index(a, sub, start=0, end=None)</c>: <see cref="find"/>, but an element with no
            ///     match raises <c>ValueError: substring not found</c>.
            /// </summary>
            /// <param name="a">The searched strings.</param>
            /// <param name="sub">The substrings (see <see cref="find"/>).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>An int64 array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An element without a match (<c>substring not found</c>); a <c>StringDType</c> NA
            ///     whose NA object is not a string.</exception>
            public static NDArray index(NDArray a, NDArray sub, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, sub))
                    return NDCharArrayBoundary.Wrapped((a, sub, start, end), static s => index(s.a, s.sub, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(sub, nameof(sub));
                return NDStringMethods.FindLike("index", TextMethodOp.Find, a, sub, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.rindex(a, sub, start=0, end=None)</c>: <see cref="rfind"/>, but an element with no
            ///     match raises <c>ValueError: substring not found</c>.
            /// </summary>
            /// <param name="a">The searched strings.</param>
            /// <param name="sub">The substrings (see <see cref="find"/>).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>An int64 array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">An element without a match; a <c>StringDType</c> NA whose NA object is not a
            ///     string.</exception>
            public static NDArray rindex(NDArray a, NDArray sub, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, sub))
                    return NDCharArrayBoundary.Wrapped((a, sub, start, end), static s => rindex(s.a, s.sub, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(sub, nameof(sub));
                return NDStringMethods.FindLike("rindex", TextMethodOp.RFind, a, sub, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.count(a, sub, start=0, end=None)</c>: the number of NON-OVERLAPPING occurrences of
            ///     <paramref name="sub"/> in <c>a[start:end]</c>, counted greedily from the left (<c>count('aaaa', 'aa')</c>
            ///     is 2); an empty substring occurs <c>len + 1</c> times in the range.
            /// </summary>
            /// <param name="a">The searched strings.</param>
            /// <param name="sub">The substrings (see <see cref="find"/>).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>An int64 array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is not a string.</exception>
            public static NDArray count(NDArray a, NDArray sub, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, sub))
                    return NDCharArrayBoundary.Wrapped((a, sub, start, end), static s => count(s.a, s.sub, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(sub, nameof(sub));
                return NDStringMethods.FindLike("count", TextMethodOp.Count, a, sub, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.startswith(a, prefix, start=0, end=None)</c>: whether each element's
            ///     <c>a[start:end]</c> begins with <paramref name="prefix"/>.
            /// </summary>
            /// <param name="a">The tested strings.</param>
            /// <param name="prefix">The prefixes (see <see cref="find"/> for the classes).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>A bool array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is neither a string nor NaN-like (a
            ///     NaN-like NA is False).</exception>
            public static NDArray startswith(NDArray a, NDArray prefix, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, prefix))
                    return NDCharArrayBoundary.Wrapped((a, prefix, start, end), static s => startswith(s.a, s.prefix, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(prefix, nameof(prefix));
                return NDStringMethods.FindLike("startswith", TextMethodOp.StartsWith, a, prefix, start, end);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.endswith(a, suffix, start=0, end=None)</c>: whether each element's
            ///     <c>a[start:end]</c> ends with <paramref name="suffix"/>.
            /// </summary>
            /// <param name="a">The tested strings.</param>
            /// <param name="suffix">The suffixes (see <see cref="find"/> for the classes).</param>
            /// <param name="start">The range start (null = 0).</param>
            /// <param name="end">The range end (null = the whole string).</param>
            /// <returns>A bool array of the broadcast shape.</returns>
            /// <exception cref="TypeError">No loop for the operand classes; two different <c>StringDType</c> NA objects.</exception>
            /// <exception cref="ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
            /// <exception cref="OverflowException">A start / end literal past int64.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <exception cref="ValueError">A <c>StringDType</c> NA whose NA object is neither a string nor NaN-like.</exception>
            public static NDArray endswith(NDArray a, NDArray suffix, NDArray start = null, NDArray end = null)
            {
                if (NDCharArrayBoundary.Sees(a, suffix))
                    return NDCharArrayBoundary.Wrapped((a, suffix, start, end), static s => endswith(s.a, s.suffix, s.start, s.end));
                RequireOperand(a, nameof(a));
                RequireOperand(suffix, nameof(suffix));
                return NDStringMethods.FindLike("endswith", TextMethodOp.EndsWith, a, suffix, start, end);
            }

            /// <summary>Rejects a null operand (NumPy's <c>None</c> is an object array NumSharp does not model; a null here is a
            ///     caller bug).</summary>
            /// <param name="x">The operand.</param>
            /// <param name="name">Its parameter name.</param>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            private static void RequireOperand(NDArray x, string name)
            {
                if (x is null)
                    throw new ArgumentNullException(name);
            }
        }
    }
}
