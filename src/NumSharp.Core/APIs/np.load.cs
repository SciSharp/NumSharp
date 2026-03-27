using System;
using System.IO;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        #region np.load

        /// <summary>
        /// Load arrays from .npy or .npz files.
        /// </summary>
        /// <param name="file">File path to load from.</param>
        /// <param name="maxHeaderSize">Maximum header size for security (default 10000).</param>
        /// <returns>
        /// For .npy files: the loaded NDArray.
        /// For .npz files: an NpzFile (dictionary-like, must be disposed).
        /// </returns>
        /// <remarks>
        /// File type is automatically detected by magic bytes:
        /// - \x93NUMPY → .npy file
        /// - PK\x03\x04 or PK\x05\x06 → .npz (ZIP) file
        ///
        /// For .npz files, arrays are loaded lazily. The returned NpzFile must be
        /// disposed (use 'using' statement) to release file handles.
        /// </remarks>
        /// <example>
        /// <code>
        /// // Load single array
        /// var arr = np.load("data.npy");
        ///
        /// // Load multiple arrays from .npz
        /// using var npz = np.load("data.npz") as NpzFile;
        /// var x = npz["arr_0"];
        /// var y = npz["arr_1"];
        ///
        /// // Or with pattern matching
        /// var result = np.load("data.npz");
        /// if (result is NpzFile npz)
        /// {
        ///     using (npz)
        ///     {
        ///         var x = npz["x"];
        ///     }
        /// }
        /// </code>
        /// </example>
        public static object load(string file, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));

            var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);

            try
            {
                return LoadFromStream(stream, ownStream: true, maxHeaderSize);
            }
            catch
            {
                stream.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Load array from stream.
        /// </summary>
        /// <param name="stream">Stream to read from. Must support seeking for file type detection.</param>
        /// <param name="maxHeaderSize">Maximum header size for security.</param>
        /// <returns>NDArray or NpzFile depending on content.</returns>
        public static object load(Stream stream, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            return LoadFromStream(stream, ownStream: false, maxHeaderSize);
        }

        /// <summary>
        /// Load array from byte array.
        /// </summary>
        /// <param name="bytes">Byte array containing .npy or .npz data.</param>
        /// <param name="maxHeaderSize">Maximum header size for security.</param>
        /// <returns>NDArray or NpzFile depending on content.</returns>
        public static object load(byte[] bytes, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            if (bytes == null)
                throw new ArgumentNullException(nameof(bytes));

            var stream = new MemoryStream(bytes, writable: false);
            return LoadFromStream(stream, ownStream: true, maxHeaderSize);
        }

        #endregion

        #region Typed Load Methods

        /// <summary>
        /// Load a single .npy file and return as NDArray.
        /// </summary>
        /// <param name="file">Path to .npy file.</param>
        /// <param name="maxHeaderSize">Maximum header size for security.</param>
        /// <returns>The loaded NDArray.</returns>
        /// <exception cref="FormatException">If the file is not a valid .npy file.</exception>
        public static NDArray load_npy(string file, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            return NpyFormat.ReadArray(stream, maxHeaderSize);
        }

        /// <summary>
        /// Load a single .npy file from stream.
        /// </summary>
        public static NDArray load_npy(Stream stream, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            return NpyFormat.ReadArray(stream, maxHeaderSize);
        }

        /// <summary>
        /// Load a single .npy file from byte array.
        /// </summary>
        public static NDArray load_npy(byte[] bytes, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            using var stream = new MemoryStream(bytes, writable: false);
            return NpyFormat.ReadArray(stream, maxHeaderSize);
        }

        /// <summary>
        /// Load a .npz file and return as NpzFile.
        /// </summary>
        /// <param name="file">Path to .npz file.</param>
        /// <param name="maxHeaderSize">Maximum header size for security.</param>
        /// <returns>NpzFile with lazy-loading arrays. Must be disposed.</returns>
        public static NpzFile load_npz(string file, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            return new NpzFile(stream, ownStream: true, maxHeaderSize);
        }

        /// <summary>
        /// Load a .npz file from stream.
        /// </summary>
        public static NpzFile load_npz(Stream stream, bool ownStream = false, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            return new NpzFile(stream, ownStream, maxHeaderSize);
        }

        /// <summary>
        /// Load a .npz file from byte array.
        /// </summary>
        public static NpzFile load_npz(byte[] bytes, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            var stream = new MemoryStream(bytes, writable: false);
            return new NpzFile(stream, ownStream: true, maxHeaderSize);
        }

        #endregion

        #region Internal

        /// <summary>
        /// Internal implementation for loading from stream.
        /// </summary>
        private static object LoadFromStream(Stream stream, bool ownStream, int maxHeaderSize)
        {
            if (!stream.CanSeek)
                throw new ArgumentException("Stream must support seeking for file type detection", nameof(stream));

            // Read magic bytes for type detection
            const int magicLen = 6;
            long startPosition = stream.Position;

            if (stream.Length - startPosition < magicLen)
                throw new EndOfStreamException("File too small to be a valid .npy or .npz file");

            byte[] magic = new byte[magicLen];
            int bytesRead = stream.Read(magic, 0, magicLen);

            if (bytesRead == 0)
                throw new EndOfStreamException("No data left in file");

            // Seek back to start position (not to 0, to support multiple arrays in one stream)
            stream.Position = startPosition;

            // Check for ZIP (.npz) magic: PK\x03\x04 or PK\x05\x06 (empty)
            if ((magic[0] == 0x50 && magic[1] == 0x4B) &&
                ((magic[2] == 0x03 && magic[3] == 0x04) ||
                 (magic[2] == 0x05 && magic[3] == 0x06)))
            {
                // NPZ file - return NpzFile for lazy loading
                return new NpzFile(stream, ownStream, maxHeaderSize);
            }

            // Check for NPY magic: \x93NUMPY
            if (magic[0] == 0x93 &&
                magic[1] == 'N' &&
                magic[2] == 'U' &&
                magic[3] == 'M' &&
                magic[4] == 'P' &&
                magic[5] == 'Y')
            {
                // NPY file
                try
                {
                    return NpyFormat.ReadArray(stream, maxHeaderSize);
                }
                finally
                {
                    if (ownStream)
                        stream.Dispose();
                }
            }

            // Unknown format
            if (ownStream)
                stream.Dispose();

            throw new FormatException(
                $"Unknown file format. Expected .npy (\\x93NUMPY) or .npz (PK), " +
                $"got: \\x{magic[0]:X2}\\x{magic[1]:X2}\\x{magic[2]:X2}\\x{magic[3]:X2}");
        }

        #endregion
    }
}
