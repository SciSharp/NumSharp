using System;
using System.Collections;
using System.Collections.Generic;
using System.Dynamic;
using System.IO;
using System.IO.Compression;
using System.Linq;

namespace NumSharp.IO
{
    /// <summary>
    ///     A lazily-loaded <c>.npz</c> archive — NumPy's <c>NpzFile</c>.
    /// </summary>
    /// <remarks>
    ///     A <c>.npz</c> is a ZIP archive of <c>.npy</c> members. Nothing is decoded until a key is
    ///     accessed, and each array is cached from then on, so a huge archive costs only what is read
    ///     out of it.
    ///
    ///     Keys work with or without the <c>.npy</c> suffix — <c>npz["weights"]</c> and
    ///     <c>npz["weights.npy"]</c> are the same member — while <see cref="Files"/> reports the stripped
    ///     names, matching NumPy.
    ///
    ///     The archive holds an open file handle, so dispose it:
    ///     <code>
    ///     using var npz = np.load_npz("model.npz");
    ///     NDArray w = npz["weights"];
    ///     NDArray b = npz.f.biases;   // dot access, NumPy's BagObj
    ///     </code>
    /// </remarks>
    public sealed class NpzFile : IReadOnlyDictionary<string, NDArray>, IDisposable
    {
        /// <summary>How many keys <see cref="ToString"/> lists before eliding — NumPy's <c>_MAX_REPR_ARRAY_COUNT</c>.</summary>
        private const int MaxReprArrayCount = 5;

        private readonly bool _ownStream;
        private readonly string _name;

        /// <summary>
        ///     Maps every accepted key — stripped AND suffixed — to its zip entry.
        /// </summary>
        /// <remarks>
        ///     Holds the <see cref="ZipArchiveEntry"/> rather than its name because a zip may contain
        ///     duplicate names, and the two runtimes disagree on which one wins:
        ///     <see cref="ZipArchive.GetEntry"/> returns the FIRST match, while Python's zipfile builds a
        ///     name→info dict as it scans, so the LAST wins. Building this map the same way — later
        ///     entries overwrite earlier ones — reproduces NumPy's choice.
        /// </remarks>
        private readonly Dictionary<string, ZipArchiveEntry> _keyToEntry;

        /// <summary>Cache keyed by entry identity, so duplicate names cannot alias each other.</summary>
        private readonly Dictionary<ZipArchiveEntry, NDArray> _cache;

        private readonly List<string> _files;

        private Stream _stream;
        private ZipArchive _archive;
        private bool _disposed;

        /// <summary>
        ///     Open an archive over a stream.
        /// </summary>
        /// <param name="stream">A readable, seekable stream holding the ZIP archive.</param>
        /// <param name="ownStream">When true, disposing this also disposes <paramref name="stream"/>.</param>
        /// <param name="allowPickle">Whether members are trusted; see <see cref="NpyFormat.ReadArray"/>.</param>
        /// <param name="maxHeaderSize">Per-member header size cap.</param>
        public NpzFile(Stream stream, bool ownStream = false, bool allowPickle = false,
                       long maxHeaderSize = NpyFormat.MaxHeaderSize)
        {
            _stream = stream ?? throw new ArgumentNullException(nameof(stream));
            _ownStream = ownStream;
            AllowPickle = allowPickle;
            MaxHeaderSize = maxHeaderSize;
            _name = (stream as FileStream)?.Name ?? "object";

            try
            {
                _archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true);
            }
            catch
            {
                if (ownStream) stream.Dispose();
                throw;
            }

            _keyToEntry = new Dictionary<string, ZipArchiveEntry>(StringComparer.Ordinal);
            _cache = new Dictionary<ZipArchiveEntry, NDArray>();
            _files = new List<string>(_archive.Entries.Count);

            // Mirrors NumPy's NpzFile.__init__ exactly:
            //     self.files  = [name.removesuffix(".npy") for name in _files]
            //     self._files = dict(zip(self.files, _files))   # pass 1: stripped -> entry
            //     self._files.update(zip(_files, _files))       # pass 2: full -> entry
            //
            // The TWO passes are load-bearing, not a stylistic detail. A zip may hold both 'a' and
            // 'a.npy', and both strip to the key 'a'; pass 2 then re-points 'a' at the entry literally
            // NAMED 'a'. A single interleaved loop resolves 'a' to whichever came last instead —
            // for entries ['a', 'a.npy'] NumPy answers with the first, one loop answers with the second.
            // Within each pass later entries overwrite earlier ones, because a zip may legally repeat a
            // name and Python's dict keeps the last.
            foreach (ZipArchiveEntry entry in _archive.Entries)
            {
                string key = StripNpy(entry.FullName);

                // Files lists every entry, duplicates included — it is a plain list built from
                // namelist(), so a repeated name appears twice here while the lookup keeps only one.
                _files.Add(key);
                _keyToEntry[key] = entry;
            }

            foreach (ZipArchiveEntry entry in _archive.Entries)
                _keyToEntry[entry.FullName] = entry;

            F = new BagObj(this);
        }

        /// <summary>Open an archive from a file path. The file handle is owned and closed on dispose.</summary>
        public NpzFile(string path, bool allowPickle = false, long maxHeaderSize = NpyFormat.MaxHeaderSize)
            : this(new FileStream(path, FileMode.Open, FileAccess.Read), ownStream: true, allowPickle, maxHeaderSize)
        {
            _name = path;
        }

        /// <summary>The array names in the archive, with <c>.npy</c> stripped — NumPy's <c>.files</c>.</summary>
        public IReadOnlyList<string> Files => _files;

        /// <summary>Whether members are loaded as trusted input.</summary>
        public bool AllowPickle { get; }

        /// <summary>The per-member header size cap.</summary>
        public long MaxHeaderSize { get; }

        /// <summary>
        ///     Dot-notation access to members — NumPy's <c>npz.f.weights</c>. Requires a <c>dynamic</c>
        ///     receiver: <c>NDArray w = npz.f.weights;</c>.
        /// </summary>
        public dynamic F { get; private set; }

        /// <summary>Lower-case alias of <see cref="F"/>, spelled as NumPy spells it.</summary>
        public dynamic f => F;

        /// <summary>The underlying archive, for callers that need entry metadata.</summary>
        public ZipArchive Zip
        {
            get { ThrowIfDisposed(); return _archive; }
        }

        /// <summary>Number of members.</summary>
        public int Count => _files.Count;

        /// <summary>The member names — same as <see cref="Files"/>.</summary>
        public IEnumerable<string> Keys => _files;

        /// <summary>Every member's array. Enumerating this loads and caches all of them.</summary>
        public IEnumerable<NDArray> Values
        {
            get
            {
                foreach (string key in _files)
                    yield return this[key];
            }
        }

        /// <summary>
        ///     The array stored under <paramref name="key"/>, with or without the <c>.npy</c> suffix.
        ///     Loaded on first access and cached.
        /// </summary>
        /// <exception cref="KeyNotFoundException">No such member.</exception>
        /// <exception cref="FormatException">The member is not a .npy file — use <see cref="GetRawBytes"/>.</exception>
        public NDArray this[string key]
        {
            get
            {
                ThrowIfDisposed();

                if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                    throw new KeyNotFoundException($"{key} is not a file in the archive");

                if (_cache.TryGetValue(entry, out NDArray cached))
                    return cached;

                using (MemoryStream member = OpenMember(entry))
                {
                    // NumPy checks the magic and hands back raw bytes for anything that is not a .npy.
                    // NumSharp's indexer is typed, so route those to GetRawBytes instead of widening
                    // every access to object.
                    if (!NpyFormat.IsNpyFile(member))
                        throw new FormatException(
                            $"'{entry.FullName}' is not a .npy member (its magic string is missing), so it has no " +
                            $"array to return. Use GetRawBytes(\"{key}\") to read it as bytes.");

                    NDArray array = NpyFormat.ReadArray(member, AllowPickle, MaxHeaderSize);
                    _cache[entry] = array;
                    return array;
                }
            }
        }

        /// <summary>
        ///     A member's raw bytes, whatever it holds. NumPy returns these from its indexer for
        ///     non-<c>.npy</c> members; for a <c>.npy</c> member this is the encoded file itself.
        /// </summary>
        /// <exception cref="KeyNotFoundException">No such member.</exception>
        public byte[] GetRawBytes(string key)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                throw new KeyNotFoundException($"{key} is not a file in the archive");

            using (MemoryStream member = OpenMember(entry))
                return member.ToArray();
        }

        /// <summary>Whether <paramref name="key"/> names a member that holds a .npy array.</summary>
        public bool IsArray(string key)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                return false;
            if (_cache.ContainsKey(entry))
                return true;

            using (MemoryStream member = OpenMember(entry))
                return NpyFormat.IsNpyFile(member);
        }

        /// <summary>NumPy's <c>name.removesuffix(".npy")</c>.</summary>
        private static string StripNpy(string entryName) =>
            entryName.EndsWith(".npy", StringComparison.Ordinal)
                ? entryName.Substring(0, entryName.Length - 4)
                : entryName;

        // A zip entry stream cannot seek, and both the magic sniff and the reader need to. Members are
        // one array each, so buffering the whole entry is bounded by the array we are about to build.
        private static MemoryStream OpenMember(ZipArchiveEntry entry)
        {
            var buffer = new MemoryStream(entry.Length > 0 && entry.Length <= int.MaxValue ? (int)entry.Length : 0);
            using (Stream member = entry.Open())
                member.CopyTo(buffer);
            buffer.Position = 0;
            return buffer;
        }

        /// <summary>Whether the archive has this member (with or without the <c>.npy</c> suffix).</summary>
        public bool ContainsKey(string key)
        {
            ThrowIfDisposed();
            return _keyToEntry.ContainsKey(key);
        }

        /// <summary>The array under <paramref name="key"/>, or false if there is no such member.</summary>
        public bool TryGetValue(string key, out NDArray value)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.ContainsKey(key))
            {
                value = null;
                return false;
            }

            value = this[key];
            return true;
        }

        /// <summary>Enumerate every member as a name/array pair, loading each in turn.</summary>
        public IEnumerator<KeyValuePair<string, NDArray>> GetEnumerator()
        {
            ThrowIfDisposed();

            foreach (string key in _files)
                yield return new KeyValuePair<string, NDArray>(key, this[key]);
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Close the archive and release the file handle — NumPy's <c>close()</c>.</summary>
        public void Close() => Dispose();

        /// <inheritdoc cref="Close"/>
        public void Dispose()
        {
            if (_disposed)
                return;

            _disposed = true;
            F = null;

            _archive?.Dispose();
            _archive = null;

            if (_ownStream)
                _stream?.Dispose();
            _stream = null;

            _cache.Clear();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(NpzFile));
        }

        /// <summary>Formatted as NumPy's repr: <c>NpzFile 'model.npz' with keys: a, b, c</c>.</summary>
        public override string ToString()
        {
            string keys = string.Join(", ", _files.Take(MaxReprArrayCount));
            if (_files.Count > MaxReprArrayCount)
                keys += "...";
            return $"NpzFile '{_name}' with keys: {keys}";
        }

        /// <summary>
        ///     Turns member lookups into property reads — NumPy's <c>BagObj</c>, reached via
        ///     <see cref="NpzFile.F"/>.
        /// </summary>
        private sealed class BagObj : DynamicObject
        {
            // NumPy uses a weakref here so the NpzFile stays collectable despite the cycle. .NET's GC
            // collects cycles, so a direct reference is fine.
            private readonly NpzFile _owner;

            public BagObj(NpzFile owner) => _owner = owner;

            public override bool TryGetMember(GetMemberBinder binder, out object result)
            {
                if (!_owner.ContainsKey(binder.Name))
                    throw new KeyNotFoundException($"{binder.Name} is not a file in the archive");

                result = _owner[binder.Name];
                return true;
            }

            public override IEnumerable<string> GetDynamicMemberNames() => _owner.Files;

            public override string ToString() => _owner.ToString();
        }
    }
}
