using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        #region np.save

        /// <summary>
        /// Save an array to a binary file in NumPy .npy format.
        /// </summary>
        /// <param name="file">File path. If it doesn't end with .npy, the extension is added.</param>
        /// <param name="arr">Array data to be saved.</param>
        /// <remarks>
        /// The .npy format is the standard binary file format in NumPy for persisting
        /// a single arbitrary NumPy array on disk. The format stores all of the shape
        /// and dtype information necessary to reconstruct the array correctly.
        ///
        /// For saving multiple arrays, use <see cref="savez"/> or <see cref="savez_compressed"/>.
        /// </remarks>
        /// <example>
        /// <code>
        /// var arr = np.arange(10);
        /// np.save("data.npy", arr);
        ///
        /// // .npy extension is added automatically
        /// np.save("data", arr);  // saves as data.npy
        /// </code>
        /// </example>
        public static void save(string file, NDArray arr)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));

            // Add .npy extension if not present
            if (!file.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                file += ".npy";

            using var stream = new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.None);
            save(stream, arr);
        }

        /// <summary>
        /// Save an array to a stream in NumPy .npy format.
        /// </summary>
        /// <param name="stream">Stream to write to.</param>
        /// <param name="arr">Array data to be saved.</param>
        /// <remarks>
        /// Data is appended to the stream. Multiple arrays can be written to the same
        /// file by calling save multiple times on the same stream.
        /// </remarks>
        public static void save(Stream stream, NDArray arr)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            if (arr is null)
                throw new ArgumentNullException(nameof(arr));

            NpyFormat.WriteArray(stream, arr);
        }

        #endregion

        #region np.savez

        /// <summary>
        /// Save several arrays into a single file in uncompressed .npz format.
        /// </summary>
        /// <param name="file">File path. If it doesn't end with .npz, the extension is added.</param>
        /// <param name="arrays">Arrays to save. Will be named arr_0, arr_1, etc.</param>
        /// <remarks>
        /// The .npz file format is a zipped archive of .npy files. Each file in the archive
        /// contains one array in .npy format.
        ///
        /// For compression, use <see cref="savez_compressed"/>.
        /// </remarks>
        /// <example>
        /// <code>
        /// var x = np.arange(10);
        /// var y = np.sin(x);
        /// np.savez("data.npz", x, y);
        ///
        /// // Load back
        /// using var npz = np.load("data.npz") as NpzFile;
        /// var x_loaded = npz["arr_0"];
        /// var y_loaded = npz["arr_1"];
        /// </code>
        /// </example>
        public static void savez(string file, params NDArray[] arrays)
        {
            var dict = new Dictionary<string, NDArray>();
            for (int i = 0; i < arrays.Length; i++)
                dict[$"arr_{i}"] = arrays[i];

            SaveNpzInternal(file, dict, compress: false);
        }

        /// <summary>
        /// Save several arrays into a single file in uncompressed .npz format with named keys.
        /// </summary>
        /// <param name="file">File path. If it doesn't end with .npz, the extension is added.</param>
        /// <param name="arrays">Dictionary mapping names to arrays.</param>
        /// <remarks>
        /// Keys should be valid filenames (avoid / or . characters).
        /// </remarks>
        /// <example>
        /// <code>
        /// var weights = np.random.randn(100, 50);
        /// var biases = np.zeros(50);
        /// np.savez("model.npz", new Dictionary&lt;string, NDArray&gt; {
        ///     ["weights"] = weights,
        ///     ["biases"] = biases
        /// });
        ///
        /// // Load back
        /// using var npz = np.load("model.npz") as NpzFile;
        /// var w = npz["weights"];
        /// var b = npz["biases"];
        /// </code>
        /// </example>
        public static void savez(string file, Dictionary<string, NDArray> arrays)
        {
            SaveNpzInternal(file, arrays, compress: false);
        }

        /// <summary>
        /// Save several arrays into a single file in uncompressed .npz format.
        /// Combines positional and named arrays.
        /// </summary>
        /// <param name="file">File path.</param>
        /// <param name="positionalArrays">Arrays named arr_0, arr_1, etc.</param>
        /// <param name="namedArrays">Arrays with explicit names.</param>
        public static void savez(string file, NDArray[] positionalArrays, Dictionary<string, NDArray> namedArrays)
        {
            var combined = new Dictionary<string, NDArray>(namedArrays);
            for (int i = 0; i < positionalArrays.Length; i++)
            {
                string key = $"arr_{i}";
                if (combined.ContainsKey(key))
                    throw new ArgumentException($"Cannot use positional array and keyword '{key}'");
                combined[key] = positionalArrays[i];
            }

            SaveNpzInternal(file, combined, compress: false);
        }

        #endregion

        #region np.savez_compressed

        /// <summary>
        /// Save several arrays into a single file in compressed .npz format.
        /// </summary>
        /// <param name="file">File path. If it doesn't end with .npz, the extension is added.</param>
        /// <param name="arrays">Arrays to save. Will be named arr_0, arr_1, etc.</param>
        /// <remarks>
        /// Uses ZIP_DEFLATED compression. For uncompressed archives, use <see cref="savez"/>.
        /// </remarks>
        public static void savez_compressed(string file, params NDArray[] arrays)
        {
            var dict = new Dictionary<string, NDArray>();
            for (int i = 0; i < arrays.Length; i++)
                dict[$"arr_{i}"] = arrays[i];

            SaveNpzInternal(file, dict, compress: true);
        }

        /// <summary>
        /// Save several arrays into a single file in compressed .npz format with named keys.
        /// </summary>
        /// <param name="file">File path. If it doesn't end with .npz, the extension is added.</param>
        /// <param name="arrays">Dictionary mapping names to arrays.</param>
        public static void savez_compressed(string file, Dictionary<string, NDArray> arrays)
        {
            SaveNpzInternal(file, arrays, compress: true);
        }

        /// <summary>
        /// Save several arrays into a single file in compressed .npz format.
        /// Combines positional and named arrays.
        /// </summary>
        public static void savez_compressed(string file, NDArray[] positionalArrays, Dictionary<string, NDArray> namedArrays)
        {
            var combined = new Dictionary<string, NDArray>(namedArrays);
            for (int i = 0; i < positionalArrays.Length; i++)
            {
                string key = $"arr_{i}";
                if (combined.ContainsKey(key))
                    throw new ArgumentException($"Cannot use positional array and keyword '{key}'");
                combined[key] = positionalArrays[i];
            }

            SaveNpzInternal(file, combined, compress: true);
        }

        #endregion

        #region Internal

        /// <summary>
        /// Internal implementation for saving .npz files.
        /// </summary>
        private static void SaveNpzInternal(string file, Dictionary<string, NDArray> arrays, bool compress)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            // Add .npz extension if not present
            if (!file.EndsWith(".npz", StringComparison.OrdinalIgnoreCase))
                file += ".npz";

            var compression = compress ? CompressionLevel.Optimal : CompressionLevel.NoCompression;

            using var fileStream = new FileStream(file, FileMode.Create, FileAccess.Write, FileShare.None);
            using var archive = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);

            foreach (var kvp in arrays)
            {
                string entryName = kvp.Key;
                if (!entryName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                    entryName += ".npy";

                var entry = archive.CreateEntry(entryName, compression);
                using var entryStream = entry.Open();
                NpyFormat.WriteArray(entryStream, kvp.Value);
            }
        }

        /// <summary>
        /// Save .npz to stream.
        /// </summary>
        internal static void SaveNpzToStream(Stream stream, Dictionary<string, NDArray> arrays, bool compress, bool leaveOpen = false)
        {
            var compression = compress ? CompressionLevel.Optimal : CompressionLevel.NoCompression;

            using var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen);

            foreach (var kvp in arrays)
            {
                string entryName = kvp.Key;
                if (!entryName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                    entryName += ".npy";

                var entry = archive.CreateEntry(entryName, compression);
                using var entryStream = entry.Open();
                NpyFormat.WriteArray(entryStream, kvp.Value);
            }
        }

        #endregion

        #region Convenience Overloads

        /// <summary>
        /// Save NDArray to .npy file (NumPy-compatible binary format).
        /// </summary>
        public static void save(string file, Array arr)
        {
            save(file, np.array(arr));
        }

        /// <summary>
        /// Save to byte array in .npy format.
        /// </summary>
        public static byte[] save(NDArray arr)
        {
            using var stream = new MemoryStream();
            save(stream, arr);
            return stream.ToArray();
        }

        /// <summary>
        /// Save multiple arrays to byte array in .npz format.
        /// </summary>
        public static byte[] savez(params NDArray[] arrays)
        {
            using var stream = new MemoryStream();
            var dict = new Dictionary<string, NDArray>();
            for (int i = 0; i < arrays.Length; i++)
                dict[$"arr_{i}"] = arrays[i];
            SaveNpzToStream(stream, dict, compress: false, leaveOpen: true);
            return stream.ToArray();
        }

        /// <summary>
        /// Save multiple arrays to byte array in compressed .npz format.
        /// </summary>
        public static byte[] savez_compressed(params NDArray[] arrays)
        {
            using var stream = new MemoryStream();
            var dict = new Dictionary<string, NDArray>();
            for (int i = 0; i < arrays.Length; i++)
                dict[$"arr_{i}"] = arrays[i];
            SaveNpzToStream(stream, dict, compress: true, leaveOpen: true);
            return stream.ToArray();
        }

        #endregion
    }
}
