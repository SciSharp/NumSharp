using System;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.4.2 reference: numpy/_core/src/multiarray/multiarraymodule.c::array_fromstring ->
        // ctors.c::PyArray_FromString -> array_from_text with fromstr_next_element / fromstr_skip_separator ->
        // arraytypes.c.src's <TYPE>_fromstr (PyOS_strtol / PyOS_strtoul with a 64-bit C long, NumPyOS_ascii_strtod).
        //
        //   fromstring(string, dtype=float, count=-1, *, sep, like=None)
        //
        // The scanners live in IO/TextReading/NumPyTextScan*.cs. The BINARY mode (no sep, or sep="") was removed in
        // NumPy 1.22: it raises rather than parse the string's bytes; np.frombuffer is the replacement.

        /// <summary>
        ///     Construct a 1-D array from the numbers in a text string.
        /// </summary>
        /// <param name="string">The text (NumPy's <c>str</c> argument: read as its UTF-8 bytes).</param>
        /// <param name="dtype">Element type of the result (default float64). Signed integers read with <c>PyOS_strtol</c>
        ///     (a sign, then whitespace and digits; past the 64-bit range it saturates), unsigned ones with
        ///     <c>PyOS_strtoul</c> (no sign at all), bools, floats and complex numbers with NumPy's float reader. A non-native
        ///     byte order reads as the native dtype.</param>
        /// <param name="count">Elements to read; negative (default -1) reads all of them. With a count, the result ALWAYS
        ///     has <paramref name="count"/> elements (NumPy's): the element after the last one read holds the value the failed
        ///     scan produced (0 for an integer, -1.0 for a float), and the rest are 0 (uninitialized in NumPy).</param>
        /// <param name="sep">Separator between numbers. A space of it matches any run of whitespace (none included), and
        ///     whitespace around it is always allowed. Null or empty is the removed binary mode.</param>
        /// <param name="like">Reference array for NumPy's array-function dispatch: accepted for signature parity, no
        ///     effect.</param>
        /// <remarks>
        ///     <para>
        ///     A read stops at the end of the text, after <paramref name="count"/> elements, or at a separator that does not
        ///     match. The scanners read what NumPy's do, which is more than a number in places: whitespace alone reads as 0
        ///     (or -1.0 for a float), <c>"- 1"</c> as -1, <c>"1+2"</c> as the complex <c>1+0j</c>, <c>"j"</c> as <c>-1j</c>.
        ///     </para>
        ///     <para>
        ///     The integer readers work in a C <c>long</c>: NumSharp pins its LP64 (Linux / macOS) width, so a number past the
        ///     32-bit range wraps into a narrow dtype from its 64-bit value; Windows NumPy saturates it at 2^31 first.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html
        /// </remarks>
        /// <returns>The 1-D array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="string"/> is null (NumPy's <c>None</c> is a
        ///     <c>TypeError</c>: <c>a bytes-like object is required, not 'NoneType'</c>).</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="string"/> or <paramref name="sep"/> holds a lone surrogate
        ///     (CPython's <c>'utf-8' codec can't encode character '\ud800' in position 5: surrogates not allowed</c>).</exception>
        /// <exception cref="ValueError">NumPy's refusals, in NumPy's order: a NUL in <paramref name="sep"/> (<c>embedded null
        ///     character</c>); no separator (<c>The binary mode of fromstring is removed, use frombuffer instead</c>); a
        ///     <c>StringDType</c> (<c>Cannot create an object array from a string</c>); an unsized string dtype (<c>zero-valued
        ///     itemsize</c>); a sized string dtype (<c>don't know how to read character strings with that array type</c>); a
        ///     count no array can hold (<c>array is too big</c>); text that does not read to its end (<c>string or file could
        ///     not be read to its end due to unmatched data</c>).</exception>
        public static NDArray fromstring(string @string, DType dtype = null, long count = -1, string sep = null,
            NDArray like = null)
        {
            if (@string is null)
                throw new ArgumentNullException(nameof(@string));
            // "s#": a str is its UTF-8 bytes, converted before every other argument.
            byte[] data = Utf8Strict(@string);
            return FromStringBytes(data, dtype, count, sep, like);
        }

        /// <summary>
        ///     Construct a 1-D array from the numbers in a byte string (NumPy's <c>bytes</c> argument, read as it is).
        /// </summary>
        /// <param name="string">The text's bytes.</param>
        /// <param name="dtype">Element type of the result (default float64).</param>
        /// <param name="count">Elements to read; negative reads all of them.</param>
        /// <param name="sep">Separator between numbers; null or empty is the removed binary mode.</param>
        /// <param name="like">Accepted for NumPy's signature; no effect.</param>
        /// <returns>The 1-D array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="string"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="ValueError">As <see cref="fromstring(string,DType,long,string,NDArray)"/>.</exception>
        public static NDArray fromstring(byte[] @string, DType dtype = null, long count = -1, string sep = null,
            NDArray like = null)
        {
            if (@string is null)
                throw new ArgumentNullException(nameof(@string));
            return FromStringBytes(@string, dtype, count, sep, like);
        }

        /// <summary>
        ///     <c>array_fromstring</c> after its first argument: <paramref name="sep"/>'s <c>"s"</c> conversion, the removed
        ///     binary mode, then <c>PyArray_FromString</c>'s dtype checks and the text read.
        /// </summary>
        /// <param name="data">The text's bytes.</param>
        /// <param name="dtype">The dtype (null: float64).</param>
        /// <param name="count">Elements to read; negative reads all.</param>
        /// <param name="sep">The separator (null: NumPy's omitted argument).</param>
        /// <param name="like">Ignored.</param>
        /// <returns>The array.</returns>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="ValueError">NumPy's refusals and read errors (see the public overloads).</exception>
        private static NDArray FromStringBytes(byte[] data, DType dtype, long count, string sep, NDArray like)
        {
            byte[] sepBytes = sep == null ? null : CStringArgument(sep);
            _ = like;
            // The binary mode (no separator) is gone in NumPy 2.x: it raises rather than reinterpret the bytes.
            if (sepBytes == null || sepBytes.Length == 0)
                throw new ValueError("The binary mode of fromstring is removed, use frombuffer instead");
            dtype ??= float64;
            // A dtype whose items hold references (StringDType: a cell points into its buffer's allocator) cannot be filled
            // from text, whatever the text or the count.
            if (dtype.Meta is VStringDTypeMeta)
                throw new ValueError("Cannot create an object array from a string");
            // A fixed-width string dtype: an unsized class first, then any dtype without a text element reader.
            if (Backends.Strings.NDStringStorage.IsString(dtype))
                throw new ValueError(dtype.itemsize == 0
                    ? "zero-valued itemsize"
                    : "don't know how to read character strings with that array type");
            return NumPyTextScan.FromString(data, dtype, count, sepBytes);
        }
    }
}
