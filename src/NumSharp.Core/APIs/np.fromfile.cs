using System;
using System.Globalization;
using System.IO;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        // NumPy 2.4.2 reference: numpy/_core/src/multiarray/multiarraymodule.c::array_fromfile ->
        // ctors.c::PyArray_FromFile (array_fromfile_binary / array_from_text with fromfile_next_element and
        // fromfile_skip_separator) -> arraytypes.c.src's <TYPE>_scan, the C library's fscanf (glibc's, ported) and
        // numpyos.c's NumPyOS_ascii_ftolf. The text scanners live in IO/TextReading/NumPyTextScan*.cs.

        /// <summary>
        ///     Construct an array from data in a text or binary file. Efficient for binary data of a known dtype, and parses
        ///     simply-formatted text files. Data written with <see cref="NDArray.tofile(string,string,string)"/> can be read
        ///     back with this function.
        /// </summary>
        /// <param name="file">A file name.</param>
        /// <param name="dtype">Element type (default float64). Binary: its item size; text: the scanner NumPy's dtype has
        ///     (signed / unsigned integers through glibc's <c>fscanf</c>, floats and bools through NumPy's own float reader,
        ///     complex through <c>CDOUBLE_scan</c>). A non-native byte order (<c>'&gt;i4'</c>) reads big-endian binary data
        ///     into a native array.</param>
        /// <param name="count">Elements to read; negative (default -1) reads the whole file.</param>
        /// <param name="sep">Separator between items of a text file. Empty (default) reads binary. A space of it matches any
        ///     run of whitespace (none included), and whitespace around the separator is always allowed.</param>
        /// <param name="offset">Bytes to skip from the file's current position first. Binary files only.</param>
        /// <param name="like">Reference array for NumPy's array-function dispatch: accepted for signature parity, no
        ///     effect.</param>
        /// <remarks>
        ///     <para>
        ///     NumPy's checks, in its order: <paramref name="sep"/>'s conversion (a lone surrogate is CPython's
        ///     <see cref="UnicodeEncodeError"/>, a NUL <c>embedded null character</c>), an <paramref name="offset"/> with a
        ///     separator, the file, the offset seek, then the dtype (a <c>StringDType</c> cannot be read, an unsized string dtype
        ///     builds <paramref name="count"/> zero-width elements, a string dtype has no text scanner).
        ///     </para>
        ///     <para>
        ///     A text read stops at the end of the file, after <paramref name="count"/> elements (the separator after the last
        ///     one is then optional), or at a separator that does not match; an element that does not scan is NumPy's
        ///     <see cref="ValueError"/>. The scanners consume what NumPy's do: <c>"1.5e,2"</c> reads 1.5 and 2 (the dangling
        ///     <c>e</c> is consumed), <c>"1, "</c> is an error for a float and not for an integer.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
        /// </remarks>
        /// <returns>The 1-D array of the elements read.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="file"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="TypeError"><paramref name="sep"/> is null (NumPy's <c>sep=None</c>: <c>fromfile() argument 4 must be
        ///     str, not None</c>), or an <paramref name="offset"/> with a text separator (<c>'offset' argument only permitted
        ///     for binary files</c>), checked before the file is opened.</exception>
        /// <exception cref="FileNotFoundException">No such file (Python's <c>[Errno 2] No such file or directory: 'name'</c>).</exception>
        /// <exception cref="OSError">An offset before the start of the file (<c>[Errno 22] Invalid argument</c>).</exception>
        /// <exception cref="ValueError"><paramref name="sep"/> holds a NUL; a <c>StringDType</c> (<c>Cannot read into object
        ///     array</c>); an unsized string dtype with a negative count, or a count past the end of a binary file
        ///     (<c>negative dimensions are not allowed</c>); a string dtype with a separator (<c>Unable to read character files
        ///     of that array type</c>); text that does not scan to its end (<c>string or file could not be read to its end due
        ///     to unmatched data</c>); a count no array can hold (<c>array is too big</c>).</exception>
        /// <exception cref="NotSupportedException">An unsized string dtype with a count of 0 or more: NumPy builds zero-width
        ///     elements, which NumSharp's string buffers cannot hold.</exception>
        public static NDArray fromfile(string file, DType dtype = null, long count = -1, string sep = "", long offset = 0,
            NDArray like = null)
        {
            if (file is null) throw new ArgumentNullException(nameof(file));
            // PyArg_ParseTupleAndKeywords converts sep ("s") before the body looks at the file.
            byte[] sepBytes = FromFileSep(sep);
            _ = like;
            // array_fromfile tests offset-with-a-separator before it opens the file, so a missing file reports this first.
            if (offset != 0 && sepBytes.Length != 0)
                throw new TypeError("'offset' argument only permitted for binary files");
            FileStream fs;
            try
            {
                fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            }
            catch (FileNotFoundException e)
            {
                throw new FileNotFoundException(
                    $"[Errno 2] No such file or directory: {Backends.Printing.PyRepr.Str(file)}", file, e);
            }
            using (fs)
                return ArrayFileIO.FromFile(fs, dtype, count, sepBytes, offset);
        }

        /// <summary>
        ///     Construct an array from data in an open <see cref="Stream"/> (NumPy's file-object form), from its current
        ///     position. The stream stays open and ends where NumPy leaves the Python file: after the bytes the read consumed
        ///     (a binary read's partial trailing item included, a text scanner's look-ahead byte excluded), also when the read
        ///     raises.
        /// </summary>
        /// <param name="stream">The stream. A non-seekable one is read as NumPy cannot (it needs a seekable file
        ///     descriptor): a binary count of -1 reads to the end, and a text read's look-ahead stays consumed.</param>
        /// <param name="dtype">Element type (default float64).</param>
        /// <param name="count">Elements to read; negative (default -1) reads to the end.</param>
        /// <param name="sep">Separator between items of a text file. Empty (default) reads binary.</param>
        /// <param name="offset">Bytes to skip from the stream's position first. Binary only.</param>
        /// <param name="like">Accepted for NumPy's signature; no effect.</param>
        /// <returns>The 1-D array of the elements read.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="TypeError"><paramref name="sep"/> is null, or an <paramref name="offset"/> with a text
        ///     separator.</exception>
        /// <exception cref="OSError">An offset before the start of the stream (<c>[Errno 22] Invalid argument</c>).</exception>
        /// <exception cref="ValueError">As <see cref="fromfile(string,DType,long,string,long,NDArray)"/>.</exception>
        /// <exception cref="NotSupportedException">An unsized string dtype with a count of 0 or more.</exception>
        public static NDArray fromfile(Stream stream, DType dtype = null, long count = -1, string sep = "", long offset = 0,
            NDArray like = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            byte[] sepBytes = FromFileSep(sep);
            _ = like;
            if (offset != 0 && sepBytes.Length != 0)
                throw new TypeError("'offset' argument only permitted for binary files");
            return ArrayFileIO.FromFile(stream, dtype, count, sepBytes, offset);
        }

        /// <summary>
        ///     <c>np.fromfile</c>'s <c>sep</c> through its <c>"s"</c> converter: <c>""</c> is NumPy's default (binary), and a
        ///     null is an explicit <c>None</c>, which the <c>"s"</c> format refuses.
        /// </summary>
        /// <param name="sep">The separator argument.</param>
        /// <returns>Its UTF-8 bytes.</returns>
        /// <exception cref="TypeError">A null separator (<c>fromfile() argument 4 must be str, not None</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A lone surrogate.</exception>
        /// <exception cref="ValueError">A NUL (<c>embedded null character</c>).</exception>
        private static byte[] FromFileSep(string sep)
            => sep is null
                ? throw new TypeError("fromfile() argument 4 must be str, not None")
                : ArrayFileIO.CStringArgument(sep);
    }
}
