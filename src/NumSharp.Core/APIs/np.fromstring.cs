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
        ///     Parity with NumPy 2.4.2's <c>np.fromstring</c> (text mode). Shares <see cref="fromfile(string,NPTypeCode,int,string,long)"/>'s
        ///     text reader, so item parsing, the single-trailing-separator rule and the verbatim
        ///     "unmatched data" error all match it.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html
        /// </remarks>
        public static NDArray fromstring(string @string, NPTypeCode dtype, int count = -1, string sep = null)
            => fromstring(@string, dtype.AsType(), count, sep);

        /// <inheritdoc cref="fromstring(string,NPTypeCode,int,string)"/>
        public static NDArray fromstring(string @string, Type dtype = null, int count = -1, string sep = null)
        {
            if (@string is null) throw new ArgumentNullException(nameof(@string));

            // The binary mode (empty separator) is gone in NumPy 2.x — it raises rather than reinterpreting
            // the string's bytes, so a caller reaching for it is redirected to frombuffer.
            if (string.IsNullOrEmpty(sep))
                throw new ValueError("The binary mode of fromstring is removed, use frombuffer instead");

            NPTypeCode tc = (dtype ?? typeof(double)).GetTypeCode();
            string[] tokens = SplitTokens(@string, sep, out bool nonWhitespaceSep);
            string[] items = SelectTokens(tokens, count, nonWhitespaceSep);
            return TokensToArray(items, items.Length, tc);
        }
    }
}
