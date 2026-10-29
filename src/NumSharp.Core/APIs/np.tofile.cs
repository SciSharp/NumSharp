using System;
using System.IO;
using NumSharp.IO;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Write the array to a file as binary (default) or text.
        ///     <para>
        ///     Data is always written in C (row-major) order, independent of the array's own layout —
        ///     a sliced / strided / transposed / broadcast view writes its logical elements, NOT the raw
        ///     underlying buffer. The data produced can be recovered with <see cref="np.fromfile(string,System.Type)"/>.
        ///     </para>
        /// </summary>
        /// <param name="fid">A filename. The file is created (truncated if it exists).</param>
        /// <param name="sep">
        ///     Separator between items for text output. If "" (empty, the default) a binary file is
        ///     written, equivalent to <c>stream.Write(a.tobytes('C'))</c>. Null is Python's <c>None</c>, which NumPy
        ///     refuses.
        /// </param>
        /// <param name="format">
        ///     Python-style <c>%</c> format string for text output (ignored in binary mode). Each entry
        ///     is written as <c>format % (item,)</c>, where <c>item</c> is the Python object NumPy's <c>getitem</c>
        ///     returns (a <c>float</c> for every float dtype, an <c>int</c>, a <c>bool</c>, a <c>complex</c>, a
        ///     <c>str</c>), with CPython's rules and errors. <c>"%s"</c> (the default) and <c>""</c> write
        ///     <c>str(item)</c> (e.g. <c>1.5</c>, <c>(1+2j)</c>, <c>True</c>). Null is Python's <c>None</c>, which NumPy
        ///     refuses.
        /// </param>
        /// <remarks>
        ///     <para>
        ///     <paramref name="sep"/> and <paramref name="format"/> are checked before the file is opened, as NumPy's
        ///     argument parsing checks them: neither may be null (<c>None</c>) or hold a NUL or a lone surrogate, so a bad one
        ///     creates no file and leaves an existing one as it is.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</para>
        /// </remarks>
        /// <exception cref="ValueError"><paramref name="sep"/> or <paramref name="format"/> holds a NUL (<c>embedded null
        ///     character</c>), or a malformed format / a NaN under an integer conversion (see the stream overload).</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> or <paramref name="format"/> holds a lone surrogate,
        ///     or an element's text is not ASCII (see the stream overload).</exception>
        /// <exception cref="TypeError">A null <paramref name="sep"/> or <paramref name="format"/> (NumPy's <c>"s"</c>
        ///     converter: <c>tofile() argument 2 must be str, not None</c>, argument 3 for the format), or a conversion the
        ///     elements do not take (see the stream overload).</exception>
        /// <exception cref="OverflowException">See the stream overload.</exception>
        /// <exception cref="OSError">Binary mode over a <c>StringDType</c> array.</exception>
        public void tofile(string fid, string sep = "", string format = "%s")
        {
            // NumPy parses sep and format ("O|ss": tofile's arguments 2 and 3) before it opens the file: a bad one creates
            // no file.
            CheckToFileArgument(sep, 2);
            CheckToFileArgument(format, 3);
            // NumPy opens the filename with mode "wb" (create + truncate).
            using var fs = new FileStream(fid, FileMode.Create, FileAccess.Write);
            tofile(fs, sep, format);
        }

        /// <summary>
        ///     Write the array to an open <see cref="Stream"/> as binary (default) or text. The stream is
        ///     written from its current position and left open (the caller owns it), matching NumPy's
        ///     file-object <c>tofile</c>. Data is always in C (row-major) order.
        /// </summary>
        /// <param name="stream">An open, writeable stream.</param>
        /// <param name="sep">Separator between items for text output; "" (default) writes binary. It is written as its
        ///     UTF-8 bytes, as NumPy writes the C string. Null is Python's <c>None</c>, which NumPy refuses.</param>
        /// <param name="format">Python-style <c>%</c> format for text output (default <c>"%s"</c>): each element is written
        ///     as <c>format % (item,)</c> with the Python object NumPy's <c>getitem</c> returns. Null is Python's
        ///     <c>None</c>, which NumPy refuses.</param>
        /// <remarks>
        ///     <para>
        ///     Text mode is NumPy's <c>PyArray_ToFile</c>: for each element in C order, <c>str(item)</c> (format <c>""</c> or
        ///     <c>"%s"</c>) or <c>format % (item,)</c>, then the separator except after the last. <c>item</c> is a Python
        ///     <c>float</c> for every float dtype (a <c>float32</c> 0.1 writes <c>0.10000000149011612</c>), an <c>int</c>,
        ///     a <c>bool</c> (an <c>int</c> to Python's <c>%</c>: <c>'%x'</c> of <c>True</c> is <c>1</c>), a
        ///     <c>complex</c>, a <c>str</c> / <c>bytes</c>, or a <c>StringDType</c> element (a <c>str</c>, or the NA
        ///     object). NumSharp's <c>Char</c> elements are <c>int</c>s (a <c>uint16</c>) and its <c>Decimal</c> elements
        ///     <c>float</c>s.
        ///     </para>
        ///     <para>
        ///     Each element's text is written as ASCII (NumPy's <c>PyUnicode_AsASCIIString</c>); a non-ASCII text — a
        ///     string element, a <c>%c</c> past 127, non-ASCII literal text in the format — raises CPython's
        ///     <see cref="UnicodeEncodeError"/>, after the elements before it are written. (NumPy 2.4.2 does not check that
        ///     encode's result and crashes the process there.)
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null (after the checks of
        ///     <paramref name="sep"/> and <paramref name="format"/>, which NumPy's argument parsing makes first).</exception>
        /// <exception cref="ValueError"><paramref name="sep"/> or <paramref name="format"/> holds a NUL (<c>embedded null
        ///     character</c>); a malformed format (<c>incomplete format</c>, <c>unsupported format character 'y' (0x79) at
        ///     index 1</c>); a NaN under an integer conversion.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> or <paramref name="format"/> holds a lone surrogate
        ///     (CPython's UTF-8 error); an element's text is not ASCII.</exception>
        /// <exception cref="TypeError">A null <paramref name="sep"/> or <paramref name="format"/> (<c>tofile() argument 2 must
        ///     be str, not None</c>, argument 3 for the format), checked before anything is written; or CPython's
        ///     <c>TypeError</c> of <c>format % (item,)</c>, unwrapped as NumPy leaves it: <c>not enough arguments for format
        ///     string</c>, <c>%x format: an integer is required, not float</c>, <c>* wants int</c>.</exception>
        /// <exception cref="OverflowException">CPython's <c>OverflowError</c>: an infinity under an integer conversion, a
        ///     <c>%c</c> code point out of range, a <c>*</c> width past a C <c>ssize_t</c>.</exception>
        /// <exception cref="OSError">Binary mode over a <c>StringDType</c> array.</exception>
        public void tofile(Stream stream, string sep = "", string format = "%s")
        {
            // NumPy's argument parsing ("O|ss") converts sep and format before tofile looks at its file object.
            CheckToFileArgument(sep, 2);
            CheckToFileArgument(format, 3);
            if (stream is null) throw new System.ArgumentNullException(nameof(stream));

            ArrayFileIO.ToFile(this, stream, sep, format);
        }

        /// <summary>
        ///     <c>PyArg_ParseTupleAndKeywords</c>' <c>"s"</c> converter, which <c>tofile</c> applies to <c>sep</c> (its
        ///     argument 2) and <c>format</c> (argument 3): <c>None</c> is not a <c>str</c>, the text must encode to UTF-8
        ///     (CPython refuses a lone surrogate) and hold no NUL (the C string would end there).
        /// </summary>
        /// <param name="s">The argument; null is Python's <c>None</c>, empty has nothing to check.</param>
        /// <param name="position">The argument's position in NumPy's <c>tofile(fid, sep, format)</c>, which the
        ///     <c>None</c> text names.</param>
        /// <exception cref="TypeError">A null argument: <c>tofile() argument 2 must be str, not None</c> (CPython's text for
        ///     a <c>"s"</c> argument given <c>None</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A lone surrogate: <c>'utf-8' codec can't encode character '\ud800' in
        ///     position 0: surrogates not allowed</c>, checked first as CPython encodes before it looks for a NUL.</exception>
        /// <exception cref="ValueError"><c>embedded null character</c>.</exception>
        internal static void CheckToFileArgument(string s, int position)
        {
            if (s is null)
                throw new TypeError($"tofile() argument {position} must be str, not None");
            // The encode, then the NUL test (the bytes themselves are written by the text mode later, encoded again).
            ArrayFileIO.CStringArgument(s);
        }
    }
}
