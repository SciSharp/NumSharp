using System;
using System.Collections.Generic;
using System.IO;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class lib
        {
            /// <summary>
            ///     NumPy's <c>numpy.lib.format</c>: the public face of the <c>.npy</c> format - the magic string, the header
            ///     dictionary, the header readers and writers, and whole-array reads and writes - with NumPy's names over
            ///     NumSharp's <see cref="NpyFormat"/>, which np.save / np.load already run (so the bytes are the ones those
            ///     write and read).
            /// </summary>
            /// <remarks>
            ///     Typed as C# sees NumPy's values: a format version is <c>(int major, int minor)</c>, a header a dictionary
            ///     of <c>descr</c> (string), <c>fortran_order</c> (bool) and <c>shape</c> (<c>long[]</c>), a file a
            ///     <see cref="Stream"/>. Not offered: <c>open_memmap</c> (NumSharp has no <c>np.memmap</c>) and
            ///     <c>isfileobj</c> (a test for CPython's file objects, which a C# <see cref="Stream"/> never is). NumPy's
            ///     ValueErrors are raised as <see cref="ValueError"/> with NumPy's texts. API-parity batch 13.
            /// </remarks>
            [ModuleName("np.lib.format")]
            public static class format
            {
                /// <summary>The alignment of the data that follows a header, in bytes (NumPy's <c>ARRAY_ALIGN</c>): every
                ///     header is padded so the data starts on a multiple of it.</summary>
                public const int ARRAY_ALIGN = NpyFormat.ArrayAlign;

                /// <summary>The chunk size NumPy reads and writes data in, in bytes (NumPy's <c>BUFFER_SIZE</c>, 2**18).</summary>
                public const int BUFFER_SIZE = NpyFormat.BufferSize;

                /// <summary>The magic string's length, version bytes included (NumPy's <c>MAGIC_LEN</c>).</summary>
                public const int MAGIC_LEN = NpyFormat.MagicLen;

                /// <summary>The digits a growing shape's axis is padded to in a header (NumPy's
                ///     <c>GROWTH_AXIS_MAX_DIGITS</c>): room for <c>np.save</c>-and-append writers to rewrite the header in
                ///     place.</summary>
                public const int GROWTH_AXIS_MAX_DIGITS = NpyFormat.GrowthAxisMaxDigits;

                /// <summary>The magic prefix <c>b'\x93NUMPY'</c> (NumPy's <c>MAGIC_PREFIX</c>), as a new array on every read
                ///     (NumPy's bytes are immutable; a shared array would not be).</summary>
                public static byte[] MAGIC_PREFIX => NpyFormat.MagicPrefix.ToArray();

                /// <summary>The keys a header dictionary holds (NumPy's <c>EXPECTED_KEYS</c>): <c>descr</c>,
                ///     <c>fortran_order</c>, <c>shape</c>.</summary>
                public static IReadOnlyCollection<string> EXPECTED_KEYS { get; } = new[] { "descr", "fortran_order", "shape" };

                /// <summary>
                ///     NumPy's <c>magic(major, minor)</c>: the 8-byte magic string of a format version.
                /// </summary>
                /// <param name="major">The major version, <c>0 &lt;= major &lt; 256</c>.</param>
                /// <param name="minor">The minor version, <c>0 &lt;= minor &lt; 256</c>.</param>
                /// <returns>A new 8-byte array: <c>b'\x93NUMPY'</c> then the two version bytes.</returns>
                /// <exception cref="ValueError">A version byte out of range (<c>major version must be 0 &lt;= major &lt;
                ///     256</c>, then the minor's text), checked in NumPy's order.</exception>
                public static byte[] magic(int major, int minor)
                {
                    // NumPy checks the major first, each with its own text.
                    if (major < 0 || major > 255)
                        throw new ValueError("major version must be 0 <= major < 256");
                    if (minor < 0 || minor > 255)
                        throw new ValueError("minor version must be 0 <= minor < 256");
                    return NpyFormat.Magic(new NpyFormat.FormatVersion((byte)major, (byte)minor));
                }

                /// <summary>
                ///     NumPy's <c>read_magic(fp)</c>: reads the magic string and returns the file's format version, leaving the
                ///     stream on the header-length field.
                /// </summary>
                /// <param name="fp">The stream, positioned at the start of a <c>.npy</c> file.</param>
                /// <returns>The <c>(major, minor)</c> version the file declares (not checked against the supported ones, as
                ///     NumPy's does not).</returns>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> is null.</exception>
                /// <exception cref="ValueError">A wrong magic string (<c>the magic string is not correct; expected
                ///     b'\x93NUMPY', got b'abcdef'</c>) or a stream that ends inside it.</exception>
                public static (int major, int minor) read_magic(Stream fp)
                {
                    if (fp is null)
                        throw new ArgumentNullException(nameof(fp));
                    try
                    {
                        var version = NpyFormat.ReadMagic(fp);
                        return (version.Major, version.Minor);
                    }
                    catch (FormatException e)
                    {
                        // NpyFormat raises the house FormatException for a malformed file; NumPy's module raises ValueError
                        // with the same text.
                        throw new ValueError(e.Message, e);
                    }
                }

                /// <summary>
                ///     NumPy's <c>dtype_to_descr(dtype)</c>: the <c>descr</c> a header stores for <paramref name="dtype"/> -
                ///     its <c>str</c> (<c>'&lt;i8'</c>, <c>'&gt;f4'</c>, <c>'&lt;U3'</c>), and <c>'|O'</c> for a NEP 55
                ///     <c>StringDType</c> (NumPy "lies that the dtype is object" and pickles the array).
                /// </summary>
                /// <param name="dtype">The descriptor.</param>
                /// <returns>The descr string.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
                /// <exception cref="NotSupportedException">A dtype with no NumPy descr (decimal).</exception>
                public static string dtype_to_descr(DType dtype)
                    // The descriptor logic lives in the IO layer (NpyFormat.DescrOf: dtype.str for a non-native legacy
                    // descriptor, the writer's native mapping otherwise); the facade only forwards.
                    => NpyFormat.DescrOf(dtype);

                /// <summary>
                ///     NumPy's <c>descr_to_dtype(descr)</c>: the descriptor a header's <c>descr</c> string names - for a
                ///     string NumPy's statement is <c>numpy.dtype(descr)</c>, so this is <see cref="np.dtype(string)"/>.
                /// </summary>
                /// <param name="descr">The descr (<c>'&lt;f8'</c>, <c>'&gt;f4'</c>, <c>'|b1'</c>, <c>'&lt;U3'</c>, ...).</param>
                /// <returns>The descriptor, byte order included.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
                /// <exception cref="TypeError">Not a dtype NumPy understands (<c>data type 'garbage' not
                ///     understood</c>), as NumPy's <c>numpy.dtype</c> raises.</exception>
                /// <exception cref="NotSupportedException">A NumPy dtype NumSharp cannot represent (the object dtype
                ///     <c>'|O'</c>, datetime64, ...).</exception>
                /// <remarks>NumPy's list form (a structured descr) has no counterpart: NumSharp has no structured dtypes.</remarks>
                public static DType descr_to_dtype(string descr)
                {
                    if (descr is null)
                        throw new ArgumentNullException(nameof(descr));
                    // NumPy: `if isinstance(descr, str): return numpy.dtype(descr)` (no padding removal for a plain string).
                    return np.dtype(descr);
                }

                /// <summary>
                ///     NumPy's <c>drop_metadata(dtype)</c>: the descriptor without its metadata. NumSharp's descriptors carry
                ///     none, so this is <paramref name="dtype"/> itself.
                /// </summary>
                /// <param name="dtype">The descriptor.</param>
                /// <returns><paramref name="dtype"/>.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
                public static DType drop_metadata(DType dtype)
                    => dtype ?? throw new ArgumentNullException(nameof(dtype));

                /// <summary>
                ///     NumPy's <c>header_data_from_array_1_0(array)</c>: the header dictionary of <paramref name="array"/>:
                ///     <c>descr</c>, <c>fortran_order</c> (true only for an F-contiguous array that is not C-contiguous) and
                ///     <c>shape</c>.
                /// </summary>
                /// <param name="array">The array.</param>
                /// <returns>A new dictionary (its <c>shape</c> a new <c>long[]</c>, never the array's own), the input
                ///     <see cref="write_array_header_1_0"/> / <see cref="write_array_header_2_0"/> take.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="array"/> is null.</exception>
                /// <exception cref="NotSupportedException">A dtype with no NumPy descr (decimal).</exception>
                public static IDictionary<string, object> header_data_from_array_1_0(NDArray array)
                {
                    if (array is null)
                        throw new ArgumentNullException(nameof(array));
                    var d = NpyFormat.HeaderDataFromArray(array);
                    // HeaderDataFromArray hands back the array's own dimensions (shapes share them); a caller may write into a
                    // dictionary it was given, so it gets a copy.
                    d["shape"] = ((long[])d["shape"]).Clone();
                    return d;
                }

                /// <summary>NumPy's <c>write_array_header_1_0(fp, d)</c>: writes a version 1.0 header for the dictionary.</summary>
                /// <param name="fp">The stream to write to.</param>
                /// <param name="d">The header dictionary (<c>descr</c>, <c>fortran_order</c>, <c>shape</c>).</param>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> or <paramref name="d"/> is null.</exception>
                /// <exception cref="ArgumentException"><paramref name="d"/> has no <c>shape</c>, or one that is not a
                ///     <c>long[]</c> / <c>int[]</c>.</exception>
                /// <exception cref="ValueError">The header does not fit version 1.0's 2-byte length (<c>Header length 70017
                ///     too big for version=(1, 0)</c>).</exception>
                public static void write_array_header_1_0(Stream fp, IDictionary<string, object> d)
                    => WriteHeader(fp, d, NpyFormat.FormatVersion.V1_0);

                /// <summary>NumPy's <c>write_array_header_2_0(fp, d)</c>: writes a version 2.0 header for the dictionary
                ///     (a 4-byte header length).</summary>
                /// <param name="fp">The stream to write to.</param>
                /// <param name="d">The header dictionary.</param>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> or <paramref name="d"/> is null.</exception>
                /// <exception cref="ArgumentException"><paramref name="d"/> has no usable <c>shape</c>.</exception>
                public static void write_array_header_2_0(Stream fp, IDictionary<string, object> d)
                    => WriteHeader(fp, d, NpyFormat.FormatVersion.V2_0);

                /// <summary>The shared body of the two header writers: NpyFormat's writer at the given version.</summary>
                /// <param name="fp">The stream.</param>
                /// <param name="d">The header dictionary.</param>
                /// <param name="version">The version to write.</param>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> or <paramref name="d"/> is null.</exception>
                /// <exception cref="ArgumentException"><paramref name="d"/> has no usable <c>shape</c>.</exception>
                /// <exception cref="ValueError">The header does not fit the version's length field.</exception>
                private static void WriteHeader(Stream fp, IDictionary<string, object> d, NpyFormat.FormatVersion version)
                {
                    if (fp is null)
                        throw new ArgumentNullException(nameof(fp));
                    if (d is null)
                        throw new ArgumentNullException(nameof(d));
                    NpyFormat.WriteArrayHeader(fp, new Dictionary<string, object>(d, StringComparer.Ordinal), version);
                }

                /// <summary>
                ///     NumPy's <c>read_array_header_1_0(fp, max_header_size=10000)</c>: reads a version 1.0 header (the stream
                ///     positioned just past the magic string) and returns what it declares.
                /// </summary>
                /// <param name="fp">The stream.</param>
                /// <param name="max_header_size">The longest header accepted (a guard against crafted files).</param>
                /// <returns>The shape, the Fortran-order flag and the descriptor (<see cref="descr_to_dtype"/> of the
                ///     header's descr, byte order included).</returns>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> is null.</exception>
                /// <exception cref="ValueError">A malformed header (NumPy's texts).</exception>
                /// <exception cref="NotSupportedException">A descr NumSharp cannot represent (the <c>'|O'</c> a
                ///     <c>StringDType</c> file declares, datetime64, structured).</exception>
                public static (long[] shape, bool fortran_order, DType dtype) read_array_header_1_0(Stream fp,
                    long max_header_size = NpyFormat.MaxHeaderSize)
                    => ReadHeader(fp, NpyFormat.FormatVersion.V1_0, max_header_size);

                /// <summary>
                ///     NumPy's <c>read_array_header_2_0(fp, max_header_size=10000)</c>: reads a version 2.0 header (a 4-byte
                ///     header length) and returns what it declares.
                /// </summary>
                /// <param name="fp">The stream, positioned just past the magic string.</param>
                /// <param name="max_header_size">The longest header accepted.</param>
                /// <returns>The shape, the Fortran-order flag and the descriptor.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> is null.</exception>
                /// <exception cref="ValueError">A malformed header (NumPy's texts).</exception>
                /// <exception cref="NotSupportedException">A descr NumSharp cannot represent.</exception>
                public static (long[] shape, bool fortran_order, DType dtype) read_array_header_2_0(Stream fp,
                    long max_header_size = NpyFormat.MaxHeaderSize)
                    => ReadHeader(fp, NpyFormat.FormatVersion.V2_0, max_header_size);

                /// <summary>The shared body of the two header readers.</summary>
                /// <param name="fp">The stream.</param>
                /// <param name="version">The header's version.</param>
                /// <param name="max_header_size">The longest header accepted.</param>
                /// <returns>The declared shape, order flag and descriptor.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> is null.</exception>
                /// <exception cref="ValueError">A malformed header.</exception>
                /// <exception cref="NotSupportedException">A descr NumSharp cannot represent (the object dtype a NEP 55
                ///     <c>StringDType</c> file declares, datetime64, structured).</exception>
                private static (long[] shape, bool fortran_order, DType dtype) ReadHeader(Stream fp,
                    NpyFormat.FormatVersion version, long max_header_size)
                {
                    if (fp is null)
                        throw new ArgumentNullException(nameof(fp));
                    NpyFormat.HeaderData header;
                    try
                    {
                        header = NpyFormat.ReadArrayHeader(fp, version, max_header_size);
                    }
                    catch (FormatException e)
                    {
                        // NpyFormat raises the house FormatException with NumPy's texts; NumPy's module raises ValueError.
                        throw new ValueError(e.Message, e);
                    }
                    // NumPy's `_read_array_header` returns `descr_to_dtype(d['descr'])`: the header's own descriptor, byte
                    // order and complex64 included, not the storage lane NpyFormat materializes the data in.
                    return ((long[])header.Shape.Clone(), header.FortranOrder, descr_to_dtype(header.DescrText));
                }

                /// <summary>
                ///     NumPy's <c>write_array(fp, array, version=None, allow_pickle=True)</c>: writes <paramref name="array"/>
                ///     as a whole <c>.npy</c> file - the bytes <c>np.save</c> writes.
                /// </summary>
                /// <param name="fp">The stream to write to.</param>
                /// <param name="array">The array.</param>
                /// <param name="version">The format version, or null for the oldest one that fits the header.</param>
                /// <param name="allow_pickle">Whether a pickled payload (a <c>StringDType</c> array) may be written.</param>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> or <paramref name="array"/> is null.</exception>
                /// <exception cref="ValueError">An unsupported version (<c>we only support format version (1,0), (2,0), and
                ///     (3,0), not (4, 0)</c>), a version byte out of range, or a <c>StringDType</c> array with
                ///     <paramref name="allow_pickle"/> false.</exception>
                /// <exception cref="NotSupportedException">A dtype with no NumPy descr (decimal).</exception>
                public static void write_array(Stream fp, NDArray array, (int major, int minor)? version = null,
                                               bool allow_pickle = true)
                {
                    NpyFormat.FormatVersion? v = null;
                    if (version.HasValue)
                    {
                        // NumPy's _check_version refuses anything but (1,0), (2,0), (3,0); a byte out of range cannot even
                        // name a version.
                        var (major, minor) = version.Value;
                        if (major < 0 || major > 255 || minor < 0 || minor > 255)
                            throw new ValueError(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                                $"we only support format version (1,0), (2,0), and (3,0), not ({major}, {minor})"));
                        v = new NpyFormat.FormatVersion((byte)major, (byte)minor);
                    }
                    try
                    {
                        NpyFormat.WriteArray(fp, array, v, allow_pickle);
                    }
                    catch (FormatException e)
                    {
                        throw new ValueError(e.Message, e);
                    }
                }

                /// <summary>
                ///     NumPy's <c>read_array(fp, allow_pickle=False, *, max_header_size=10000)</c>: reads a whole
                ///     <c>.npy</c> file - what <c>np.load</c> reads.
                /// </summary>
                /// <param name="fp">The stream, positioned at the start of the file.</param>
                /// <param name="allow_pickle">Whether a pickled payload may be read (a trusted file).</param>
                /// <param name="max_header_size">The longest header accepted (ignored when <paramref name="allow_pickle"/>
                ///     declares the file trusted, as NumPy's).</param>
                /// <returns>A new array.</returns>
                /// <exception cref="ArgumentNullException"><paramref name="fp"/> is null.</exception>
                /// <exception cref="ValueError">A malformed file, or a pickled payload with <paramref name="allow_pickle"/>
                ///     false (NumPy's texts).</exception>
                /// <exception cref="NotSupportedException">A dtype NumSharp cannot represent.</exception>
                public static NDArray read_array(Stream fp, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
                {
                    try
                    {
                        return NpyFormat.ReadArray(fp, allow_pickle, max_header_size);
                    }
                    catch (FormatException e)
                    {
                        // The house FormatException (a malformed file, NumPy's "Object arrays cannot be loaded when
                        // allow_pickle=False") is NumPy's ValueError, text unchanged.
                        throw new ValueError(e.Message, e);
                    }
                }
            }
        }
    }
}
