using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        #region save (.npy)

        /// <summary>
        ///     Save an array to a <c>.npy</c> binary file.
        /// </summary>
        /// <param name="file">
        ///     Target path. <c>.npy</c> is appended if the name does not already end with it, matching
        ///     NumPy.
        /// </param>
        /// <param name="arr">The array to save. Any layout; a Fortran-contiguous array is stored as such.</param>
        /// <param name="allow_pickle">
        ///     Present for NumPy parity. NumSharp has no object dtype, so nothing can reach the pickle
        ///     path and this never changes the outcome.
        /// </param>
        /// <exception cref="NotSupportedException">
        ///     The dtype has no NumPy equivalent — <see cref="NPTypeCode.Decimal"/>.
        /// </exception>
        /// <remarks>
        ///     The file is byte-for-byte what NumPy 2.4.2's own <c>np.save</c> writes for the same array.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.save.html
        /// </remarks>
        public static void save(string file, NDArray arr, bool allow_pickle = true)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            if (!file.EndsWith(".npy", StringComparison.Ordinal))
                file += ".npy";

            using (var stream = new FileStream(file, FileMode.Create, FileAccess.Write))
                NpyFormat.WriteArray(stream, arr, null, allow_pickle);
        }

        /// <summary>
        ///     Save an array to a <c>.npy</c> file, converting <paramref name="arr"/> with
        ///     <see cref="np.asanyarray(in object, Type)"/> first.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.save.html</remarks>
        public static void save(string file, Array arr, bool allow_pickle = true)
            => save(file, asanyarray(arr), allow_pickle);

        /// <summary>
        ///     Write an array in <c>.npy</c> format to an open stream.
        /// </summary>
        /// <param name="file">
        ///     An open, writable stream. Written from its current position and left open, so successive
        ///     calls append — several arrays can share one file and be read back in order by
        ///     <see cref="load_npy(Stream, bool, long)"/>.
        /// </param>
        /// <param name="arr">The array to save.</param>
        /// <param name="allow_pickle">Present for NumPy parity; see <see cref="save(string, NDArray, bool)"/>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.save.html</remarks>
        public static void save(Stream file, NDArray arr, bool allow_pickle = true)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            NpyFormat.WriteArray(file, arr, null, allow_pickle);
        }

        /// <summary>
        ///     Encode an array as <c>.npy</c> and return the bytes.
        /// </summary>
        /// <remarks>A NumSharp convenience; NumPy has no in-memory equivalent.</remarks>
        public static byte[] save(NDArray arr, bool allow_pickle = true)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            using (var stream = new MemoryStream())
            {
                NpyFormat.WriteArray(stream, arr, null, allow_pickle);
                return stream.ToArray();
            }
        }

        /// <summary>
        ///     Write an array in <c>.npy</c> format using an explicit format version — NumPy's
        ///     <c>numpy.lib.format.write_array</c>.
        /// </summary>
        /// <param name="file">An open, writable stream.</param>
        /// <param name="arr">The array to save.</param>
        /// <param name="version">
        ///     (1,0), (2,0), (3,0), or null to use the oldest that can hold the header. Version 2.0
        ///     widens the header-length field to 4 bytes; 3.0 also switches the header to UTF-8.
        /// </param>
        /// <param name="allow_pickle">Present for NumPy parity; see <see cref="save(string, NDArray, bool)"/>.</param>
        /// <exception cref="FormatException">The version is not one of (1,0), (2,0) or (3,0).</exception>
        public static void save_version(Stream file, NDArray arr, NpyFormat.FormatVersion? version, bool allow_pickle = true)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            NpyFormat.WriteArray(file, arr, version, allow_pickle);
        }

        #endregion

        #region savez (.npz)

        /// <summary>
        ///     Save several arrays into an uncompressed <c>.npz</c> archive, named <c>arr_0</c>,
        ///     <c>arr_1</c>, … in order.
        /// </summary>
        /// <param name="file">Target path. <c>.npz</c> is appended if not already present.</param>
        /// <param name="args">The arrays, in order.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(string file, params NDArray[] args)
            => savez(file, args, null);

        /// <summary>
        ///     Save named arrays into an uncompressed <c>.npz</c> archive — NumPy's keyword form,
        ///     <c>np.savez(file, weights=w, biases=b)</c>.
        /// </summary>
        /// <param name="file">Target path. <c>.npz</c> is appended if not already present.</param>
        /// <param name="kwds">Name/array pairs. Each becomes <c>&lt;name&gt;.npy</c> in the archive.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(string file, IDictionary<string, NDArray> kwds)
            => savez(file, null, kwds);

        /// <summary>
        ///     Save positional and named arrays into an uncompressed <c>.npz</c> archive.
        /// </summary>
        /// <param name="file">Target path. <c>.npz</c> is appended if not already present.</param>
        /// <param name="args">Positional arrays, stored as <c>arr_0</c>, <c>arr_1</c>, …</param>
        /// <param name="kwds">Named arrays.</param>
        /// <exception cref="ArgumentException">A name in <paramref name="kwds"/> collides with a generated <c>arr_N</c>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(string file, NDArray[] args, IDictionary<string, NDArray> kwds)
            => SaveZip(file, args, kwds, CompressionLevel.NoCompression);

        /// <summary>Write an uncompressed <c>.npz</c> archive to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(Stream file, params NDArray[] args)
            => SaveZip(file, args, null, CompressionLevel.NoCompression);

        /// <summary>Write an uncompressed <c>.npz</c> archive of named arrays to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(Stream file, IDictionary<string, NDArray> kwds)
            => SaveZip(file, null, kwds, CompressionLevel.NoCompression);

        /// <summary>Write an uncompressed <c>.npz</c> archive of positional and named arrays to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez.html</remarks>
        public static void savez(Stream file, NDArray[] args, IDictionary<string, NDArray> kwds)
            => SaveZip(file, args, kwds, CompressionLevel.NoCompression);

        /// <summary>Encode an uncompressed <c>.npz</c> archive of <c>arr_0</c>… arrays and return the bytes.</summary>
        /// <remarks>A NumSharp convenience; NumPy has no in-memory equivalent.</remarks>
        public static byte[] savez(params NDArray[] args)
            => SaveZipBytes(args, null, CompressionLevel.NoCompression);

        /// <summary>Encode an uncompressed <c>.npz</c> archive of named arrays and return the bytes.</summary>
        /// <remarks>A NumSharp convenience; NumPy has no in-memory equivalent.</remarks>
        public static byte[] savez(IDictionary<string, NDArray> kwds)
            => SaveZipBytes(null, kwds, CompressionLevel.NoCompression);

        #endregion

        #region savez_compressed (.npz, deflated)

        /// <summary>
        ///     Save several arrays into a compressed <c>.npz</c> archive, named <c>arr_0</c>,
        ///     <c>arr_1</c>, … in order.
        /// </summary>
        /// <param name="file">Target path. <c>.npz</c> is appended if not already present.</param>
        /// <param name="args">The arrays, in order.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(string file, params NDArray[] args)
            => savez_compressed(file, args, null);

        /// <summary>Save named arrays into a compressed <c>.npz</c> archive.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(string file, IDictionary<string, NDArray> kwds)
            => savez_compressed(file, null, kwds);

        /// <summary>Save positional and named arrays into a compressed <c>.npz</c> archive.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(string file, NDArray[] args, IDictionary<string, NDArray> kwds)
            => SaveZip(file, args, kwds, CompressionLevel.Optimal);

        /// <summary>Write a compressed <c>.npz</c> archive to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(Stream file, params NDArray[] args)
            => SaveZip(file, args, null, CompressionLevel.Optimal);

        /// <summary>Write a compressed <c>.npz</c> archive of named arrays to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(Stream file, IDictionary<string, NDArray> kwds)
            => SaveZip(file, null, kwds, CompressionLevel.Optimal);

        /// <summary>Write a compressed <c>.npz</c> archive of positional and named arrays to an open stream.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html</remarks>
        public static void savez_compressed(Stream file, NDArray[] args, IDictionary<string, NDArray> kwds)
            => SaveZip(file, args, kwds, CompressionLevel.Optimal);

        /// <summary>Encode a compressed <c>.npz</c> archive of <c>arr_0</c>… arrays and return the bytes.</summary>
        /// <remarks>A NumSharp convenience; NumPy has no in-memory equivalent.</remarks>
        public static byte[] savez_compressed(params NDArray[] args)
            => SaveZipBytes(args, null, CompressionLevel.Optimal);

        /// <summary>Encode a compressed <c>.npz</c> archive of named arrays and return the bytes.</summary>
        /// <remarks>A NumSharp convenience; NumPy has no in-memory equivalent.</remarks>
        public static byte[] savez_compressed(IDictionary<string, NDArray> kwds)
            => SaveZipBytes(null, kwds, CompressionLevel.Optimal);

        #endregion

        #region npz internals

        private static void SaveZip(string file, NDArray[] args, IDictionary<string, NDArray> kwds, CompressionLevel level)
        {
            if (file == null) throw new ArgumentNullException(nameof(file));

            if (!file.EndsWith(".npz", StringComparison.Ordinal))
                file += ".npz";

            using (var stream = new FileStream(file, FileMode.Create, FileAccess.Write))
                SaveZip(stream, args, kwds, level);
        }

        private static byte[] SaveZipBytes(NDArray[] args, IDictionary<string, NDArray> kwds, CompressionLevel level)
        {
            using (var stream = new MemoryStream())
            {
                SaveZip(stream, args, kwds, level);
                return stream.ToArray();
            }
        }

        private static void SaveZip(Stream stream, NDArray[] args, IDictionary<string, NDArray> kwds, CompressionLevel level)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            // NumPy's _savez: keyword arrays keep their names and come first, then positional arrays
            // take arr_0, arr_1, …; a positional name that collides with a keyword is an error rather
            // than a silent overwrite.
            var namedict = new List<KeyValuePair<string, NDArray>>();
            var seen = new HashSet<string>(StringComparer.Ordinal);

            if (kwds != null)
            {
                foreach (KeyValuePair<string, NDArray> kv in kwds)
                {
                    if (kv.Value is null) throw new ArgumentNullException(nameof(kwds), $"Array '{kv.Key}' is null.");
                    namedict.Add(kv);
                    seen.Add(kv.Key);
                }
            }

            if (args != null)
            {
                for (int i = 0; i < args.Length; i++)
                {
                    string key = $"arr_{i}";
                    if (seen.Contains(key))
                        throw new ArgumentException($"Cannot use un-named variables and keyword {key}", nameof(kwds));
                    if (args[i] is null) throw new ArgumentNullException(nameof(args), $"Array at index {i} is null.");
                    namedict.Add(new KeyValuePair<string, NDArray>(key, args[i]));
                    seen.Add(key);
                }
            }

            // ZipArchive writes Zip64 records as needed, so archives above 4 GB work — NumPy passes
            // force_zip64=True for the same reason (numpy gh-10776).
            using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: true))
            {
                foreach (KeyValuePair<string, NDArray> kv in namedict)
                {
                    ZipArchiveEntry entry = zip.CreateEntry(kv.Key + ".npy", level);
                    using (Stream s = entry.Open())
                        NpyFormat.WriteArray(s, kv.Value);
                }
            }
        }

        #endregion
    }
}
