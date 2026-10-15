using System;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.4.2 reference: numpy/_core/multiarray.py::fromstring ->
        // numpy/_core/src/multiarray/ctors.c::PyArray_FromString.
        //
        //   fromstring(string, dtype=float, count=-1, *, sep)
        //
        // Text mode (a non-empty `sep`) is the SAME item reader `fromfile` uses for a text file, so this
        // shares fromfile's SplitTokens / SelectTokens / TokensToArray helpers (numpy shares `fromstr`). The
        // BINARY mode (`sep=""`) was REMOVED in NumPy 1.22 — it now raises verbatim rather than parsing the
        // string's bytes; `np.frombuffer` is the replacement.

        /// <summary>
        ///     Construct a 1-D array from the numbers in a text <paramref name="string"/>.
        /// </summary>
        /// <param name="string">The text to parse. Numbers are separated by <paramref name="sep"/>.</param>
        /// <param name="dtype">Element type of the result (default <see cref="double"/>).</param>
        /// <param name="count">Number of items to read; <c>-1</c> (default) reads all of them.</param>
        /// <param name="sep">
        ///     Separator between numbers. A separator containing spaces matches runs of whitespace (a
        ///     whitespace-only separator splits on any whitespace run). An empty or <c>null</c> separator
        ///     selects the removed binary mode and raises <see cref="ValueError"/> — use
        ///     <see cref="frombuffer(byte[],Type,long,long)"/> instead.
        /// </param>
        /// <remarks>
        ///     Parity with NumPy 2.4.2's <c>np.fromstring</c> (text mode). Shares <see cref="fromfile(string,DType,int,string,long)"/>'s
        ///     text reader, so item parsing, the single-trailing-separator rule and the verbatim
        ///     "unmatched data" error all match it.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html
        /// </remarks>
        /// <returns>The 1-D array of the parsed items.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="string"/> is null.</exception>
        /// <exception cref="ValueError">NumPy's refusals, in NumPy's order: an empty separator (<c>The binary mode of
        ///     fromstring is removed, use frombuffer instead</c>); a <c>StringDType</c> dtype (<c>Cannot create an object
        ///     array from a string</c>); an unsized string dtype (<c>zero-valued itemsize</c>); a sized string dtype (<c>don't
        ///     know how to read character strings with that array type</c>); text the reader cannot read to its end.</exception>
        public static NDArray fromstring(string @string, DType dtype = null, int count = -1, string sep = null)
        {
            if (@string is null) throw new ArgumentNullException(nameof(@string));

            // The binary mode (empty separator) is gone in NumPy 2.x — it raises rather than reinterpreting
            // the string's bytes, so a caller reaching for it is redirected to frombuffer.
            if (string.IsNullOrEmpty(sep))
                throw new ValueError("The binary mode of fromstring is removed, use frombuffer instead");

            // PyArray_FromString's first check: a dtype whose items hold references (StringDType: a cell points into its
            // buffer's allocator) cannot be filled from text, whatever the text or the count (an empty string too).
            if (dtype?.Meta is VStringDTypeMeta)
                throw new ValueError("Cannot create an object array from a string");

            // A fixed-width string dtype: PyArray_FromString refuses an unsized class first, then any dtype without a
            // text element reader (the string classes have none).
            if (Backends.Strings.NDStringStorage.IsString(dtype))
                throw new ValueError(dtype.itemsize == 0
                    ? "zero-valued itemsize"
                    : "don't know how to read character strings with that array type");

            NPTypeCode tc = dtype?.GetTypeCode() ?? NPTypeCode.Double;
            string[] tokens = SplitTokens(@string, sep, out bool nonWhitespaceSep);
            string[] items = SelectTokens(tokens, count, nonWhitespaceSep);
            return TokensToArray(items, items.Length, tc);
        }
    }
}
