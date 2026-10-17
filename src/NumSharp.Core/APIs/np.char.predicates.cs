using System;

// =============================================================================
// np.char.predicates.cs — the nine predicates numpy.char re-exports from numpy.strings
// =============================================================================
//
// numpy/_core/defchararray.py (NumPy 2.4.2) imports isalnum, isalpha, isdecimal, isdigit, islower, isnumeric, isspace,
// istitle and isupper from numpy.strings, and each IS a ufunc of numpy._core.umath (np.char.isalpha is
// np.strings.isalpha is the ufunc object): one input, a bool output, the ufunc's out= / where= / dtype=. A chararray
// operand gives a PLAIN bool array (chararray.__array_wrap__ keeps the class only for a text result), a 0-d array for a
// 0-d chararray (the wrap ignores return_scalar); a chararray out= comes back as a new chararray view of it. The
// re-exports convert their operand with the module's text rule (CharModule.TextOperand: a byte[] is Python bytes) and
// call the np.strings ufunc, so every answer (the result, its layout, every error and its order) is np.strings'.
//
// =============================================================================

namespace NumSharp
{
    public sealed partial class CharModule
    {
        /// <summary>
        ///     NumPy's <c>np.char.isalnum(x)</c>, which IS the <see cref="np.strings.isalnum"/> ufunc: True for each
        ///     element whose text has at least one character and only alphanumeric characters (CPython's
        ///     <c>str.isalnum</c>: alphabetic, decimal, digit or numeric). The text is the element's characters without
        ///     its trailing NULs, so a chararray's trailing whitespace counts: <c>'a1'</c> is True; a stored
        ///     <c>'ab  '</c> is False (its trailing spaces are not alphanumeric).
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isalnum(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isalnum(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isalpha(x)</c>, which IS the <see cref="np.strings.isalpha"/> ufunc: True for each
        ///     element whose text has at least one character and only alphabetic characters (CPython's
        ///     <c>str.isalpha</c>: the Unicode letter categories). The text is the element's characters without its
        ///     trailing NULs, so a chararray's trailing whitespace counts: <c>'abc'</c> is True; a stored <c>'ab  '</c>
        ///     is False, though its element <c>c[i]</c> reads <c>'ab'</c>.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isalpha(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isalpha(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isdecimal(x)</c>, which IS the <see cref="np.strings.isdecimal"/> ufunc: True for
        ///     each element whose text has at least one character and only decimal characters (CPython's
        ///     <c>str.isdecimal</c>: <c>'٣'</c> is one, <c>'²'</c> is not); <c>U</c> and <c>StringDType</c> only: an
        ///     <c>S</c> operand has no loop. The text is the element's characters without its trailing NULs, so a
        ///     chararray's trailing whitespace counts: <c>'123'</c> is True; a stored <c>'12 '</c> is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text (an <c>S</c> operand too: NumPy's ufunc has no
        ///     bytes loop), or <paramref name="dtype"/> is not bool (NumPy's no-loop text, which names a C# number by
        ///     its default DType: <c>Int64DType</c>, <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isdecimal(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isdecimal(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isdigit(x)</c>, which IS the <see cref="np.strings.isdigit"/> ufunc: True for each
        ///     element whose text has at least one character and only digits (CPython's <c>str.isdigit</c>: the decimal
        ///     digits and the other digits, such as <c>'²'</c>; not <c>'½'</c>). The text is the element's characters
        ///     without its trailing NULs, so a chararray's trailing whitespace counts: <c>'123'</c> is True; <c>'½'</c>
        ///     is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isdigit(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isdigit(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.islower(x)</c>, which IS the <see cref="np.strings.islower"/> ufunc: True for each
        ///     element whose text has at least one cased character and every cased character lowercase. The text is the
        ///     element's characters without its trailing NULs, so a chararray's trailing whitespace counts: a stored
        ///     <c>'ab  '</c> is True (spaces are uncased); <c>'aBc'</c> is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray islower(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.islower(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isnumeric(x)</c>, which IS the <see cref="np.strings.isnumeric"/> ufunc: True for
        ///     each element whose text has at least one character and only numeric characters (CPython's
        ///     <c>str.isnumeric</c>: digits, fractions such as <c>'½'</c>, Roman and CJK numerals); <c>U</c> and
        ///     <c>StringDType</c> only: an <c>S</c> operand has no loop. The text is the element's characters without
        ///     its trailing NULs, so a chararray's trailing whitespace counts: <c>'½'</c> is True; <c>'a1'</c> is
        ///     False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text (an <c>S</c> operand too: NumPy's ufunc has no
        ///     bytes loop), or <paramref name="dtype"/> is not bool (NumPy's no-loop text, which names a C# number by
        ///     its default DType: <c>Int64DType</c>, <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isnumeric(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isnumeric(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isspace(x)</c>, which IS the <see cref="np.strings.isspace"/> ufunc: True for each
        ///     element whose text has at least one character and only whitespace (CPython's <c>str.isspace</c> for
        ///     <c>U</c> / <c>StringDType</c>; space and <c>\t</c>–<c>\r</c> for <c>S</c>). The text is the element's
        ///     characters without its trailing NULs, so a chararray's trailing whitespace counts: <c>'  '</c> is True;
        ///     an empty element is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isspace(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isspace(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.istitle(x)</c>, which IS the <see cref="np.strings.istitle"/> ufunc: True for each
        ///     element whose text is titlecased: an uppercase or titlecase character never follows a cased character, a
        ///     lowercase one always does, and at least one character is cased. The text is the element's characters
        ///     without its trailing NULs, so a chararray's trailing whitespace counts: <c>'Abc Def'</c> is True;
        ///     <c>'ABC'</c> is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray istitle(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.istitle(TextOperand(x, nameof(x)), @out, where, dtype);

        /// <summary>
        ///     NumPy's <c>np.char.isupper(x)</c>, which IS the <see cref="np.strings.isupper"/> ufunc: True for each
        ///     element whose text has at least one cased character and every cased character uppercase. The text is the
        ///     element's characters without its trailing NULs, so a chararray's trailing whitespace counts: a stored
        ///     <c>'AB  '</c> is True (spaces are uncased); <c>'Ab'</c> is False.
        /// </summary>
        /// <param name="x">The strings: an <c>S</c> / <c>U</c> / <c>StringDType</c> array or a text array_like (see the
        ///     class remarks: a <c>byte[]</c> is Python bytes). A number has no loop.</param>
        /// <param name="out">An output to write into and return (any dtype bool casts to under same_kind: a number
        ///     receives 0 / 1, a text output <c>'True'</c> / <c>'False'</c> cut to its width), or null.</param>
        /// <param name="where">A bool mask: only elements where it is True are computed, the others keep
        ///     <paramref name="out"/>'s values (uninitialized in a fresh result), or null for NumPy's
        ///     <c>where=True</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: only bool has a loop; null leaves the loop to the
        ///     operand.</param>
        /// <returns>A plain bool array of <paramref name="x"/>'s shape (0-d for a scalar), or <paramref name="out"/>. A
        ///     chararray <paramref name="x"/> still gives a plain array; a chararray <paramref name="out"/> comes back
        ///     as a new view of it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null (see the class remarks).</exception>
        /// <exception cref="TypeError"><paramref name="x"/> is not text, or <paramref name="dtype"/> is not bool
        ///     (NumPy's no-loop text, which names a C# number by its default DType: <c>Int64DType</c>,
        ///     <c>Float64DType</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> cannot receive bool (the same_kind output cast);
        ///     a non-bool <paramref name="where"/>; operands that do not broadcast with <paramref name="out"/> /
        ///     <paramref name="where"/>.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>; a selected <c>StringDType</c> NA whose
        ///     descriptor's NA object is neither NaN-like nor a string.</exception>
        /// <exception cref="NotSupportedException"><paramref name="x"/> has no NumSharp array form (an object NumPy
        ///     would hold in an object array, which has no loop either).</exception>
        [NDScoped]
        public NDArray isupper(object x, NDArray @out = null, NDArray where = null, DType dtype = null)
            => np.strings.isupper(TextOperand(x, nameof(x)), @out, where, dtype);
    }
}
