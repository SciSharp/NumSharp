using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace NumSharp.IO
{
    /// <summary>
    /// A dictionary-like object with lazy-loading of arrays from a .npz archive.
    /// </summary>
    /// <remarks>
    /// NpzFile provides read-only access to arrays stored in a .npz file.
    /// Arrays are loaded lazily when first accessed.
    ///
    /// Usage:
    /// <code>
    /// using (var npz = np.load("data.npz") as NpzFile)
    /// {
    ///     var arr1 = npz["arr_0"];
    ///     var arr2 = npz["weights"];
    /// }
    /// </code>
    ///
    /// Both stripped and unstripped names work:
    /// - npz["arr_0"] and npz["arr_0.npy"] refer to the same array
    /// </remarks>
    public sealed class NpzFile : IReadOnlyDictionary<string, NDArray>, IDisposable
    {
        #region Fields

        private Stream? _stream;
        private ZipArchive? _archive;
        private readonly bool _ownStream;
        private readonly int _maxHeaderSize;

        /// <summary>Maps user-facing keys (without .npy) to entry names</summary>
        private readonly Dictionary<string, string> _keyToEntry;

        /// <summary>Maps entry names to user-facing keys</summary>
        private readonly Dictionary<string, string> _entryToKey;

        /// <summary>Cache of loaded arrays</summary>
        private readonly Dictionary<string, NDArray> _cache;

        /// <summary>List of user-facing keys (without .npy extension)</summary>
        private readonly List<string> _files;

        private bool _disposed;

        #endregion

        #region Properties

        /// <summary>
        /// List of all array names in the archive (without .npy extension).
        /// </summary>
        public IReadOnlyList<string> Files => _files;

        /// <summary>
        /// The underlying ZipArchive (for advanced access).
        /// </summary>
        public ZipArchive? Zip => _archive;

        /// <summary>
        /// Number of arrays in the archive.
        /// </summary>
        public int Count => _files.Count;

        /// <summary>
        /// All keys (array names) in the archive.
        /// </summary>
        public IEnumerable<string> Keys => _files;

        /// <summary>
        /// All arrays in the archive (triggers loading all arrays).
        /// </summary>
        public IEnumerable<NDArray> Values
        {
            get
            {
                foreach (var key in _files)
                    yield return this[key];
            }
        }

        #endregion

        #region Constructor

        /// <summary>
        /// Create NpzFile from a stream.
        /// </summary>
        /// <param name="stream">Stream containing the .npz archive</param>
        /// <param name="ownStream">If true, the stream will be closed when NpzFile is disposed</param>
        /// <param name="maxHeaderSize">Maximum allowed header size for security</param>
        public NpzFile(Stream stream, bool ownStream = false, int maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _ownStream = ownStream;
            _maxHeaderSize = maxHeaderSize;
            _archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);

            _keyToEntry = new Dictionary<string, string>(StringComparer.Ordinal);
            _entryToKey = new Dictionary<string, string>(StringComparer.Ordinal);
            _cache = new Dictionary<string, NDArray>(StringComparer.Ordinal);
            _files = new List<string>();

            // Build key mappings
            foreach (var entry in _archive.Entries)
            {
                string entryName = entry.FullName;
                string key = entryName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase)
                    ? entryName.Substring(0, entryName.Length - 4)
                    : entryName;

                _files.Add(key);
                _keyToEntry[key] = entryName;
                _keyToEntry[entryName] = entryName; // Also allow full name
                _entryToKey[entryName] = key;
            }
        }

        #endregion

        #region Indexer

        /// <summary>
        /// Get array by name. Both "arr_0" and "arr_0.npy" work.
        /// </summary>
        public NDArray this[string key]
        {
            get
            {
                ThrowIfDisposed();

                if (!_keyToEntry.TryGetValue(key, out var entryName))
                    throw new KeyNotFoundException($"'{key}' is not a file in the archive");

                // Check cache first
                if (_cache.TryGetValue(entryName, out var cached))
                    return cached;

                // Load from archive
                var entry = _archive!.GetEntry(entryName);
                if (entry == null)
                    throw new KeyNotFoundException($"Entry '{entryName}' not found in archive");

                using var entryStream = entry.Open();
                using var memStream = new MemoryStream();

                // Copy to memory stream (ZipArchiveEntry streams don't support seeking)
                entryStream.CopyTo(memStream);
                memStream.Position = 0;

                // Check if it's a .npy file
                if (NpyFormat.IsNpyFile(memStream))
                {
                    memStream.Position = 0;
                    var array = NpyFormat.ReadArray(memStream, _maxHeaderSize);
                    _cache[entryName] = array;
                    return array;
                }
                else
                {
                    // Non-.npy file - return as byte array
                    memStream.Position = 0;
                    var bytes = memStream.ToArray();
                    var array = np.array(bytes);
                    _cache[entryName] = array;
                    return array;
                }
            }
        }

        #endregion

        #region IReadOnlyDictionary Implementation

        /// <summary>
        /// Check if key exists in archive.
        /// </summary>
        public bool ContainsKey(string key)
        {
            ThrowIfDisposed();
            return _keyToEntry.ContainsKey(key);
        }

        /// <summary>
        /// Try to get array by name.
        /// </summary>
        public bool TryGetValue(string key, out NDArray value)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.ContainsKey(key))
            {
                value = default!;
                return false;
            }

            value = this[key];
            return true;
        }

        /// <summary>
        /// Enumerate all key-value pairs.
        /// </summary>
        public IEnumerator<KeyValuePair<string, NDArray>> GetEnumerator()
        {
            ThrowIfDisposed();

            foreach (var key in _files)
                yield return new KeyValuePair<string, NDArray>(key, this[key]);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        #endregion

        #region IDisposable

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(NpzFile));
        }

        /// <summary>
        /// Close the archive and release resources.
        /// </summary>
        public void Close()
        {
            Dispose();
        }

        /// <summary>
        /// Dispose the NpzFile.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;

            _archive?.Dispose();
            _archive = null;

            if (_ownStream)
            {
                _stream?.Dispose();
            }
            _stream = null;

            _cache.Clear();
        }

        #endregion

        #region Object Overrides

        /// <summary>
        /// String representation showing filename and keys.
        /// </summary>
        public override string ToString()
        {
            string name = _stream switch
            {
                FileStream fs => fs.Name,
                _ => "stream"
            };

            const int maxKeys = 5;
            string keys = string.Join(", ", _files.Take(maxKeys));
            if (_files.Count > maxKeys)
                keys += "...";

            return $"NpzFile '{name}' with keys: {keys}";
        }

        #endregion
    }
}
