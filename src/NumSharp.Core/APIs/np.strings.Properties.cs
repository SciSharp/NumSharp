using System;
using NumSharp.Backends.Strings;

// =============================================================================
// np.strings.Properties.cs — numpy.strings' predicate ufuncs (strings phase 2, P2.6)
// =============================================================================
//
// NumPy's isalpha / isdigit / isspace / isalnum / islower / isupper / istitle / isdecimal / isnumeric are ufuncs of
// numpy._core.umath (np.strings re-exports the ufunc objects): one input, a bool output, NumPy's out= / where= /
// dtype=. Each public method keeps the chararray boundary's rule in its own body, so the boundary audit reads the
// public name; the engine is NDStringMethods.Predicate (S / U) and NDVStringMethods.Predicate (StringDType).
// =============================================================================

namespace NumSharp
{
    public static partial class np
    {
        public static partial class strings
        {
            /// <summary>
            ///     NumPy's <c>np.strings.isalpha</c> ufunc: True for each element whose text has at least one character
            ///     and only alphabetic characters (CPython's <c>str.isalpha</c>: the Unicode letter categories Lu, Ll,
            ///     Lt, Lm, Lo). The text is the element's characters without its trailing NULs (an embedded NUL is a
            ///     character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'isalpha' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isalpha function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isalpha(np.array(new[] { "abc", "a1", "" }))</c> is <c>[True, False, False]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isalpha(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isalpha(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsAlphaKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isdigit</c> ufunc: True for each element whose text has at least one character
            ///     and only digits (CPython's <c>str.isdigit</c>: the decimal digits and the other digits, such as
            ///     <c>'²'</c>; not <c>'½'</c>). The text is the element's characters without its trailing NULs (an
            ///     embedded NUL is a character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'isdigit' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isdigit function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isdigit(np.array(new[] { "123", "²", "½" }))</c> is <c>[True, True, False]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isdigit(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isdigit(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsDigitKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isspace</c> ufunc: True for each element whose text has at least one character
            ///     and only whitespace (CPython's <c>str.isspace</c>: space, <c>\t</c>–<c>\r</c>, U+001C–U+001F, NEL,
            ///     the no-break, Ogham, ideographic and U+2000–U+200A spaces, the line and paragraph separators). The
            ///     text is the element's characters without its trailing NULs (an embedded NUL is a character, with no
            ///     property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'isspace' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isspace function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isspace(np.array(new[] { " \t", "\x1c", "a " }))</c> is <c>[True, True, False]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isspace(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isspace(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsSpaceKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isalnum</c> ufunc: True for each element whose text has at least one character
            ///     and only alphanumeric characters (CPython's <c>str.isalnum</c>: alphabetic, decimal, digit or
            ///     numeric). The text is the element's characters without its trailing NULs (an embedded NUL is a
            ///     character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'isalnum' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isalnum function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isalnum(np.array(new[] { "a1", "½", "a b" }))</c> is <c>[True, True, False]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isalnum(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isalnum(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsAlnumKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.islower</c> ufunc: True for each element whose text has at least one cased
            ///     character and every cased character lowercase (an uppercase or titlecase character makes it False).
            ///     The text is the element's characters without its trailing NULs (an embedded NUL is a character, with
            ///     no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'islower' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the islower function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.islower(np.array(new[] { "abc", "aBc", "1a" }))</c> is <c>[True, False, True]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray islower(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => islower(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsLowerKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isupper</c> ufunc: True for each element whose text has at least one cased
            ///     character and every cased character uppercase (a lowercase or titlecase character makes it False).
            ///     The text is the element's characters without its trailing NULs (an embedded NUL is a character, with
            ///     no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'isupper' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isupper function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isupper(np.array(new[] { "ABC", "aBc", "1A" }))</c> is <c>[True, False, True]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isupper(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isupper(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsUpperKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.istitle</c> ufunc: True for each element whose text is titlecased: an
            ///     uppercase or titlecase character (<c>'ǅ'</c>) never follows a cased character, a lowercase one
            ///     always does, and at least one character is cased. The text is the element's characters without its
            ///     trailing NULs (an embedded NUL is a character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>S</c> (NumPy's ASCII rules: a byte at or past 0x80 has no property;
            ///     whitespace is space and <c>\t</c>–<c>\r</c> only), <c>U</c> or <c>StringDType</c>.</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
            ///     (NumPy's no-loop text: <c>ufunc 'istitle' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.Int64DType'&gt; -&gt; None</c>).</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the istitle function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.istitle(np.array(new[] { "Hello World", "Hello world", "1A" }))</c> is <c>[True,
            ///     False, True]</c>. A <c>numpy.char.chararray</c> operand gives a plain bool result
            ///     (<c>chararray.__array_wrap__</c> keeps only text results); a chararray <paramref name="out"/> comes
            ///     back as a new view of it.
            /// </remarks>
            public static NDArray istitle(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => istitle(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsTitleKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isdecimal</c> ufunc: True for each element whose text has at least one
            ///     character and only decimal characters (CPython's <c>str.isdecimal</c>: <c>'٣'</c> is one, <c>'²'</c>
            ///     is not). The text is the element's characters without its trailing NULs (an embedded NUL is a
            ///     character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>U</c> or <c>StringDType</c> (NumPy has no <c>S</c> loop).</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not <c>U</c> / <c>StringDType</c> (an <c>S</c> operand
            ///     included: <c>ufunc 'isdecimal' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.BytesDType'&gt; -&gt; None</c>), or <paramref name="dtype"/> is not bool.</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isdecimal function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isdecimal(np.array(new[] { "123", "²", "٣" }))</c> is <c>[True, False, True]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isdecimal(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isdecimal(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsDecimalKind, x, @out, where, dtype);
            }

            /// <summary>
            ///     NumPy's <c>np.strings.isnumeric</c> ufunc: True for each element whose text has at least one
            ///     character and only numeric characters (CPython's <c>str.isnumeric</c>: digits, fractions such as
            ///     <c>'½'</c>, Roman numerals <c>'Ⅻ'</c>, CJK numerals <c>'万'</c>). The text is the element's
            ///     characters without its trailing NULs (an embedded NUL is a character, with no property).
            /// </summary>
            /// <param name="x">The strings: <c>U</c> or <c>StringDType</c> (NumPy has no <c>S</c> loop).</param>
            /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
            ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
            /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
            ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null.</param>
            /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop.</param>
            /// <returns>The bool results (0-d for a 0-d operand), or <paramref name="out"/>.</returns>
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            /// <exception cref="TypeError"><paramref name="x"/> is not <c>U</c> / <c>StringDType</c> (an <c>S</c> operand
            ///     included: <c>ufunc 'isnumeric' did not contain a loop with signature matching types &lt;class
            ///     'numpy.dtypes.BytesDType'&gt; -&gt; None</c>), or <paramref name="dtype"/> is not bool.</exception>
            /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast); a
            ///     non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
            ///     <paramref name="where"/>.</exception>
            /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
            ///     descriptor's NA object is neither NaN-like nor a string (<c>Cannot use the isnumeric function with a null that
            ///     is not a nan-like value</c>; a NaN-like NA is False, a string NA reads as its string).</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
            /// <remarks>
            ///     <c>np.strings.isnumeric(np.array(new[] { "123", "½", "a" }))</c> is <c>[True, True, False]</c>. A
            ///     <c>numpy.char.chararray</c> operand gives a plain bool result (<c>chararray.__array_wrap__</c> keeps
            ///     only text results); a chararray <paramref name="out"/> comes back as a new view of it.
            /// </remarks>
            public static NDArray isnumeric(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
            {
                if (NDCharArrayBoundary.Sees(x, @out))
                    return NDCharArrayBoundary.WrappedUfunc((x, @out, where, dtype), static s => isnumeric(s.x, s.@out, s.where, s.dtype), x, null, @out);
                if (x is null)
                    throw new ArgumentNullException(nameof(x));
                return NDStringMethods.Predicate(NDTextElements.IsNumericKind, x, @out, where, dtype);
            }
        }
    }
}
