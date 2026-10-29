using System;
using System.Collections;
using System.Collections.Concurrent;
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
    ///
    ///     Any number of threads may read members of one archive at the same time (NumPy's <c>NpzFile</c> is not
    ///     thread-safe; this one is). A member already loaded is returned without a lock. Everything that reads the
    ///     archive itself (loading a member, <see cref="GetRawBytes"/>, the magic check of <see cref="IsArray"/>) runs
    ///     under one gate, because the archive reads a single underlying stream, so new members load one at a time.
    ///     Two threads asking for the same member first get the same instance. <see cref="Dispose"/> waits for a load in
    ///     progress, and every access after it raises <see cref="ObjectDisposedException"/>. <see cref="Zip"/> hands out
    ///     the archive itself and bypasses the gate.
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

        /// <summary>
        ///     Cache keyed by entry identity, so duplicate names cannot alias each other. A
        ///     <see cref="ConcurrentDictionary{TKey,TValue}"/> so a cached member is read without the gate while another
        ///     thread loads a different member; every write happens under <see cref="_archiveGate"/>.
        /// </summary>
        // Borrowed: an array is handed to whoever first reads it (the indexer / Values / enumeration
        // return the SAME instance on every access) and that reader owns it — an archive that disposed
        // its cache on Close would pull the buffer out from under `using var z = np.load_npz(p);
        // var a = z["a"];`. Dispose only forgets the memo.
        [NDBorrowed] private readonly ConcurrentDictionary<ZipArchiveEntry, NDArray> _cache;

        /// <summary>
        ///     Serializes every read of <see cref="_archive"/> and <see cref="Dispose"/>. A <see cref="ZipArchive"/> is not
        ///     thread-safe: each entry stream seeks and reads the one underlying stream, so two members read at once would
        ///     interleave their seeks and decode each other's bytes.
        /// </summary>
        private readonly object _archiveGate = new object();

        private readonly List<string> _files;

        private Stream _stream;
        private ZipArchive _archive;

        /// <summary>
        ///     Set once, under <see cref="_archiveGate"/>, by <see cref="Dispose"/>. <c>volatile</c> because the lock-free
        ///     checks at the top of each member read it outside the gate; the members that touch the archive read it again
        ///     inside the gate, where the answer is final.
        /// </summary>
        private volatile bool _disposed;

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
            // One write lock is enough (every write is under _archiveGate already), and the member count bounds the
            // cache, so it never resizes.
            _cache = new ConcurrentDictionary<ZipArchiveEntry, NDArray>(concurrencyLevel: 1, capacity: _archive.Entries.Count);
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

        /// <summary>
        ///     The underlying archive, for callers that need entry metadata. Reading entries through it bypasses the gate
        ///     that serializes this class's own reads, so do not read it while other threads load members.
        /// </summary>
        /// <value>The open archive.</value>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
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
        ///     Loaded on first access and cached; every later access, from any thread, returns the same instance.
        /// </summary>
        /// <param name="key">The member name, with or without the <c>.npy</c> suffix.</param>
        /// <returns>The member's array, owned by the caller that first read it (the archive never disposes it).</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        /// <exception cref="KeyNotFoundException">No such member.</exception>
        /// <exception cref="FormatException">The member is not a .npy file — use <see cref="GetRawBytes"/> — or its header is malformed.</exception>
        /// <exception cref="NotSupportedException">The member holds a dtype NumSharp cannot load (an object array, say).</exception>
        /// <exception cref="InvalidDataException">A compressed member's data is corrupt.</exception>
        /// <remarks>
        ///     A cached member is returned without a lock. A member that is not cached yet is loaded under the archive's
        ///     gate, so a thread loading one member waits for a thread loading another.
        /// </remarks>
        public NDArray this[string key]
        {
            get
            {
                ThrowIfDisposed();

                if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                    throw new KeyNotFoundException($"{key} is not a file in the archive");

                if (_cache.TryGetValue(entry, out NDArray cached))
                    return cached;

                lock (_archiveGate)
                {
                    // Dispose takes the gate too, so here the archive is either open or closed for good, and a closed one
                    // must raise ObjectDisposedException instead of failing inside the zip reader.
                    ThrowIfDisposed();

                    // Another thread may have loaded this member while this one waited. Returning its array keeps the
                    // promise that every access to a member gets the same instance.
                    if (_cache.TryGetValue(entry, out cached))
                        return cached;

                    // NumPy checks the magic and hands back raw bytes for anything that is not a .npy.
                    // NumSharp's indexer is typed, so route those to GetRawBytes instead of widening every
                    // access to object.
                    if (!StartsWithNpyMagic(entry))
                        throw new FormatException(
                            $"'{entry.FullName}' is not a .npy member (its magic string is missing), so it has no " +
                            $"array to return. Use GetRawBytes(\"{key}\") to read it as bytes.");

                    // Stream the member straight into the reader. Buffering it first would cap members at
                    // 2 GB (MemoryStream's limit) and double the peak cost of every load — see OpenMember.
                    using (Stream member = entry.Open())
                    {
                        NDArray array = NpyFormat.ReadArray(member, AllowPickle, MaxHeaderSize);
                        _cache[entry] = array;
                        return array;
                    }
                }
            }
        }

        /// <summary>
        ///     A member's raw bytes, whatever it holds. NumPy returns these from its indexer for
        ///     non-<c>.npy</c> members; for a <c>.npy</c> member this is the encoded file itself.
        /// </summary>
        /// <remarks>
        ///     Capped at ~2 GB by <see cref="Array"/> itself. A member larger than that can still be read
        ///     as an array through the indexer, which streams instead of buffering. Reads the archive under its gate,
        ///     like a load.
        /// </remarks>
        /// <param name="key">The member name, with or without the <c>.npy</c> suffix.</param>
        /// <returns>A new array holding the member's bytes.</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        /// <exception cref="KeyNotFoundException">No such member.</exception>
        /// <exception cref="InvalidDataException">A compressed member's data is corrupt.</exception>
        public byte[] GetRawBytes(string key)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                throw new KeyNotFoundException($"{key} is not a file in the archive");

            lock (_archiveGate)
            {
                // Dispose may have closed the archive while this thread waited for the gate.
                ThrowIfDisposed();

                var buffer = new MemoryStream(entry.Length > 0 && entry.Length <= int.MaxValue ? (int)entry.Length : 0);
                using (Stream member = entry.Open())
                    member.CopyTo(buffer);
                return buffer.ToArray();
            }
        }

        /// <summary>Whether <paramref name="key"/> names a member that holds a .npy array.</summary>
        /// <param name="key">The member name, with or without the <c>.npy</c> suffix.</param>
        /// <returns>True when the member is cached or starts with the .npy magic string; false when there is no such member or it is not a .npy file.</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        /// <remarks>A cached member answers without a lock; otherwise the magic check reads the archive under its gate.</remarks>
        public bool IsArray(string key)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                return false;
            if (_cache.ContainsKey(entry))
                return true;

            lock (_archiveGate)
            {
                // Dispose may have closed the archive while this thread waited for the gate.
                ThrowIfDisposed();
                return StartsWithNpyMagic(entry);
            }
        }

        /// <summary>NumPy's <c>name.removesuffix(".npy")</c>.</summary>
        private static string StripNpy(string entryName) =>
            entryName.EndsWith(".npy", StringComparison.Ordinal)
                ? entryName.Substring(0, entryName.Length - 4)
                : entryName;

        /// <summary>
        ///     Peek at a member's magic string without consuming the stream the reader will use.
        /// </summary>
        /// <remarks>
        ///     NumPy sniffs the magic and then rewinds (<c>bytes.seek(0)</c>), which Python's ZipExtFile
        ///     supports but .NET's entry stream does not — it is forward-only. So sniff on one stream and
        ///     read on a fresh one; in Read mode an entry may be opened any number of times, and the
        ///     sniff only ever pulls the 6 magic bytes.
        ///
        ///     The obvious alternative — buffer the member into a MemoryStream and seek within it — is
        ///     what this code used to do, and it was wrong twice over: MemoryStream tops out at 2 GB, so
        ///     a >2 GB member NumSharp had happily WRITTEN failed to load with "Stream was too long",
        ///     and every ordinary load paid double (the buffer plus the array built from it).
        /// </remarks>
        private static bool StartsWithNpyMagic(ZipArchiveEntry entry)
        {
            ReadOnlySpan<byte> magic = NpyFormat.MagicPrefix;
            Span<byte> head = stackalloc byte[6];

            using (Stream sniff = entry.Open())
            {
                int got = 0;
                while (got < head.Length)
                {
                    int read = sniff.Read(head.Slice(got));
                    if (read == 0) break;
                    got += read;
                }
                return got == head.Length && head.SequenceEqual(magic);
            }
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
        /// <remarks>
        ///     Takes the archive's gate, so a member another thread is loading finishes first (a large member can make
        ///     this wait), and that thread's next access raises <see cref="ObjectDisposedException"/>. The arrays already
        ///     handed out stay valid: they belong to the callers that read them. Calling it again does nothing.
        /// </remarks>
        public void Dispose()
        {
            lock (_archiveGate)
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
        [NDBorrowed] // dot-access sugar over the owning archive; it holds nothing of its own
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
