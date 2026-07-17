using System;
using System.IO;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>The encodings NumPy accepts; anything else can silently corrupt numeric data.</summary>
        private static readonly string[] ValidPickleEncodings = { "ASCII", "latin1", "bytes" };

        /// <summary>The memory-map modes NumPy accepts.</summary>
        private static readonly string[] ValidMmapModes = { "r", "r+", "w+", "c" };

        #region load

        /// <summary>
        ///     Load an array or archive from a <c>.npy</c> or <c>.npz</c> file.
        /// </summary>
        /// <param name="file">
        ///     Path to the file. The type is detected from its magic bytes, not its extension.
        /// </param>
        /// <param name="mmap_mode">
        ///     Memory-map mode: <c>"r"</c>, <c>"r+"</c>, <c>"w+"</c> or <c>"c"</c>; null (default) reads
        ///     normally. Not yet implemented — see <see cref="CheckMmapMode"/>.
        /// </param>
        /// <param name="allow_pickle">
        ///     Whether to trust the file. Default false, as in NumPy since 1.16.3: an object array is a
        ///     Python pickle, and unpickling untrusted data can execute arbitrary code. NumSharp cannot
        ///     unpickle at all, so this only selects the error message — and, per NumPy, lifts
        ///     <paramref name="max_header_size"/>.
        /// </param>
        /// <param name="fix_imports">
        ///     Present for NumPy parity. Only affects unpickling Python 2 files, which NumSharp does not do.
        /// </param>
        /// <param name="encoding">
        ///     Present for NumPy parity; validated but otherwise unused. Must be <c>"ASCII"</c>,
        ///     <c>"latin1"</c> or <c>"bytes"</c>.
        /// </param>
        /// <param name="max_header_size">
        ///     Reject headers larger than this (default 10000). Guards against a header crafted to make
        ///     parsing pathologically expensive. Ignored when <paramref name="allow_pickle"/> is true.
        /// </param>
        /// <returns>
        ///     An <see cref="NDArray"/> for a <c>.npy</c> file, or an <see cref="NpzFile"/> for a
        ///     <c>.npz</c> archive — mirroring NumPy, whose return type also depends on the content.
        ///     Prefer <see cref="load_npy(string, bool, long)"/> or <see cref="load_npz(string, bool, long)"/>
        ///     when the kind is known: they are typed and need no cast.
        ///     <para>An <see cref="NpzFile"/> owns a file handle and must be disposed.</para>
        /// </returns>
        /// <exception cref="EndOfStreamException">The file is empty.</exception>
        /// <exception cref="FormatException">The content is not a .npy or .npz file, or is malformed.</exception>
        /// <example>
        /// <code>
        /// var arr = (NDArray)np.load("data.npy");
        ///
        /// if (np.load("data.npz") is NpzFile npz)
        ///     using (npz)
        ///         { NDArray w = npz["weights"]; }
        /// </code>
        /// </example>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.load.html</remarks>
        public static object load(string file, string mmap_mode = null, bool allow_pickle = false,
                                 bool fix_imports = true, string encoding = "ASCII",
                                 long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            CheckEncoding(encoding);
            CheckMmapMode(mmap_mode);

            // The stream is handed to NpzFile on the .npz branch, which then owns it; otherwise it is
            // closed here.
            var stream = new FileStream(file, FileMode.Open, FileAccess.Read);
            bool transferred = false;
            try
            {
                object result = LoadCore(stream, ownStream: true, allow_pickle, max_header_size);
                transferred = result is NpzFile;
                return result;
            }
            finally
            {
                if (!transferred)
                    stream.Dispose();
            }
        }

        /// <summary>
        ///     Load an array or archive from an open stream. The stream must be readable and seekable,
        ///     and is left open — the caller owns it.
        /// </summary>
        /// <inheritdoc cref="load(string, string, bool, bool, string, long)"/>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.load.html</remarks>
        public static object load(Stream file, string mmap_mode = null, bool allow_pickle = false,
                                  bool fix_imports = true, string encoding = "ASCII",
                                  long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            CheckEncoding(encoding);
            CheckMmapMode(mmap_mode);

            return LoadCore(file, ownStream: false, allow_pickle, max_header_size);
        }

        /// <summary>Load an array or archive from an in-memory <c>.npy</c>/<c>.npz</c> image.</summary>
        /// <inheritdoc cref="load(string, string, bool, bool, string, long)"/>
        public static object load(byte[] bytes, string mmap_mode = null, bool allow_pickle = false,
                                  bool fix_imports = true, string encoding = "ASCII",
                                  long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));
            CheckEncoding(encoding);
            CheckMmapMode(mmap_mode);

            var stream = new MemoryStream(bytes, writable: false);
            bool transferred = false;
            try
            {
                object result = LoadCore(stream, ownStream: true, allow_pickle, max_header_size);
                transferred = result is NpzFile;
                return result;
            }
            finally
            {
                if (!transferred)
                    stream.Dispose();
            }
        }

        #endregion

        #region load_npy

        /// <summary>
        ///     Load a single array from a <c>.npy</c> file — <see cref="load(string, string, bool, bool, string, long)"/>
        ///     without the cast.
        /// </summary>
        /// <param name="file">Path to a <c>.npy</c> file.</param>
        /// <param name="allow_pickle">Whether the file is trusted; see <see cref="load(string, string, bool, bool, string, long)"/>.</param>
        /// <param name="max_header_size">Reject headers larger than this.</param>
        /// <exception cref="FormatException">The file is not a .npy file, or is malformed.</exception>
        public static NDArray load_npy(string file, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            using (var stream = new FileStream(file, FileMode.Open, FileAccess.Read))
                return LoadNpyChecked(stream, file, allow_pickle, max_header_size);
        }

        /// <summary>
        ///     Read one array from a stream positioned at a <c>.npy</c> magic string. The stream is left
        ///     just past that array's data, so successive calls read successively saved arrays.
        /// </summary>
        /// <param name="file">An open, readable stream.</param>
        /// <param name="allow_pickle">Whether the file is trusted.</param>
        /// <param name="max_header_size">Reject headers larger than this.</param>
        /// <exception cref="EndOfStreamException">The stream is already at its end.</exception>
        public static NDArray load_npy(Stream file, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            return LoadNpyChecked(file, "stream", allow_pickle, max_header_size);
        }

        /// <summary>Load a single array from an in-memory <c>.npy</c> image.</summary>
        /// <inheritdoc cref="load_npy(string, bool, long)"/>
        public static NDArray load_npy(byte[] bytes, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));

            using (var stream = new MemoryStream(bytes, writable: false))
                return LoadNpyChecked(stream, "bytes", allow_pickle, max_header_size);
        }

        // Report a .npz handed to a .npy reader by name rather than letting the magic check produce
        // "the magic string is not correct", which would not tell the caller what to do instead.
        private static NDArray LoadNpyChecked(Stream stream, string what, bool allowPickle, long maxHeaderSize)
        {
            if (stream.CanSeek)
            {
                if (stream.Position >= stream.Length)
                    throw new EndOfStreamException("No data left in file");
                if (NpyFormat.IsNpzFile(stream))
                    throw new FormatException($"'{what}' is a .npz archive, not a .npy file. Use np.load_npz to open it.");
            }

            return NpyFormat.ReadArray(stream, allowPickle, maxHeaderSize);
        }

        #endregion

        #region load_npz

        /// <summary>
        ///     Open a <c>.npz</c> archive — <see cref="load(string, string, bool, bool, string, long)"/>
        ///     without the cast.
        /// </summary>
        /// <param name="file">Path to a <c>.npz</c> archive.</param>
        /// <param name="allow_pickle">Whether members are trusted.</param>
        /// <param name="max_header_size">Reject member headers larger than this.</param>
        /// <returns>
        ///     A lazily-loading, dictionary-like archive. It holds an open file handle — dispose it:
        ///     <c>using var npz = np.load_npz("m.npz");</c>
        /// </returns>
        /// <exception cref="FormatException">The file is not a ZIP archive.</exception>
        public static NpzFile load_npz(string file, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            var stream = new FileStream(file, FileMode.Open, FileAccess.Read);
            try
            {
                CheckIsNpz(stream, file);
                return new NpzFile(stream, ownStream: true, allow_pickle, max_header_size);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Open a <c>.npz</c> archive over a stream.
        /// </summary>
        /// <param name="file">A readable, seekable stream.</param>
        /// <param name="own_stream">When true, disposing the archive also disposes the stream.</param>
        /// <param name="allow_pickle">Whether members are trusted.</param>
        /// <param name="max_header_size">Reject member headers larger than this.</param>
        public static NpzFile load_npz(Stream file, bool own_stream = false, bool allow_pickle = false,
                                       long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            CheckIsNpz(file, "stream");
            return new NpzFile(file, own_stream, allow_pickle, max_header_size);
        }

        /// <summary>Open a <c>.npz</c> archive from an in-memory image.</summary>
        /// <inheritdoc cref="load_npz(string, bool, long)"/>
        public static NpzFile load_npz(byte[] bytes, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
        {
            if (bytes == null) throw new ArgumentNullException(nameof(bytes));

            var stream = new MemoryStream(bytes, writable: false);
            try
            {
                CheckIsNpz(stream, "bytes");
                return new NpzFile(stream, ownStream: true, allow_pickle, max_header_size);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }

        private static void CheckIsNpz(Stream stream, string what)
        {
            if (stream.CanSeek && stream.Position >= stream.Length)
                throw new EndOfStreamException("No data left in file");
            if (!NpyFormat.IsNpzFile(stream))
                throw new FormatException(
                    NpyFormat.IsNpyFile(stream)
                        ? $"'{what}' is a .npy file, not a .npz archive. Use np.load_npy to load it."
                        : $"'{what}' is not a .npz archive (no ZIP signature).");
        }

        #endregion

        #region internals

        // File-type detection by magic, in NumPy's order: ZIP, then NPY, then "must be a pickle".
        private static object LoadCore(Stream stream, bool ownStream, bool allowPickle, long maxHeaderSize)
        {
            if (!stream.CanSeek)
                throw new ArgumentException(
                    "Stream must be seekable so the file type can be detected from its magic bytes.", nameof(stream));

            if (stream.Position >= stream.Length)
                throw new EndOfStreamException("No data left in file");

            if (NpyFormat.IsNpzFile(stream))
                return new NpzFile(stream, ownStream, allowPickle, maxHeaderSize);

            if (NpyFormat.IsNpyFile(stream))
                return NpyFormat.ReadArray(stream, allowPickle, maxHeaderSize);

            // Neither magic matched. NumPy assumes a bare pickle here and tries to unpickle it;
            // NumSharp has no pickle reader, so both branches end in an error — but the allow_pickle=False
            // one is NumPy's verbatim message, which is what a user porting code will recognize.
            if (!allowPickle)
                throw new FormatException(
                    "This file contains pickled (object) data. If you trust the file you can load it unsafely " +
                    "using the `allow_pickle=` keyword argument or `pickle.load()`.");

            throw new NotSupportedException(
                "This file is not a .npy or .npz file. NumPy would try to unpickle it, which NumSharp cannot do: " +
                "the Python pickle protocol is not implemented. Re-save the data from NumPy with np.save.");
        }

        private static void CheckEncoding(string encoding)
        {
            if (Array.IndexOf(ValidPickleEncodings, encoding) < 0)
                throw new ArgumentException("encoding must be 'ASCII', 'latin1', or 'bytes'", nameof(encoding));
        }

        /// <summary>
        ///     Validate <c>mmap_mode</c> and reject it as unimplemented.
        /// </summary>
        /// <remarks>
        ///     Implementing this means backing an <c>NDArray</c> with a mapped view instead of owned
        ///     unmanaged memory. Sketch, if someone picks it up:
        ///     <list type="number">
        ///       <item>Read the header via <see cref="NpyFormat.ReadArrayHeader"/> and keep the data
        ///             offset — it is 64-byte aligned precisely so the view can start there.</item>
        ///       <item>Map with <c>MemoryMappedFile.CreateFromFile</c> and take a
        ///             <c>MemoryMappedViewAccessor</c> at that offset. Note the view offset must be a
        ///             multiple of the system allocation granularity (64 KB on Windows), so map from 0
        ///             and index past the header rather than mapping at the data offset.</item>
        ///       <item>Wrap the view's pointer in an <c>UnmanagedMemoryBlock</c> whose free-callback
        ///             releases the view and the map, so the NDArray keeps the mapping alive exactly as
        ///             long as it lives. This is the load-bearing part: today
        ///             <c>UnmanagedMemoryBlock</c> assumes it owns its allocation.</item>
        ///       <item>Honor the mode: <c>"r"</c> must produce a non-writeable Shape; <c>"c"</c> needs
        ///             <c>MemoryMappedFileAccess.CopyOnWrite</c>; <c>"r+"</c> and <c>"w+"</c> write
        ///             through, and <c>"w+"</c> must pre-size the file and write a header first.</item>
        ///       <item>Fortran-order files need the reshape-and-transpose from
        ///             <see cref="NpyFormat.ReadArray"/> applied to the mapped buffer.</item>
        ///       <item>Byte-swapped (big-endian) files cannot be mapped at all — NumSharp converts on
        ///             read, which a zero-copy view rules out. Reject them explicitly.</item>
        ///     </list>
        /// </remarks>
        /// <exception cref="ArgumentException">The mode is not one of NumPy's.</exception>
        /// <exception cref="NotImplementedException">A valid mode was requested.</exception>
        private static void CheckMmapMode(string mmap_mode)
        {
            if (mmap_mode == null)
                return;

            if (Array.IndexOf(ValidMmapModes, mmap_mode) < 0)
                throw new ArgumentException(
                    $"mode must be one of {{'r+', 'r', 'w+', 'c'}} (got '{mmap_mode}')", nameof(mmap_mode));

            throw new NotImplementedException(
                $"mmap_mode='{mmap_mode}' is not implemented yet — NumSharp arrays own their unmanaged memory and " +
                "cannot yet be backed by a mapped file view. Load without mmap_mode to read the file normally.");
        }

        #endregion
    }
}
