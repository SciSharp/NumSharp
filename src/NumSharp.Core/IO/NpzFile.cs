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
    ///     <c>npz["weights.npy"]</c> are the same member — while <see cref="files"/> reports the stripped
    ///     names, matching NumPy.
    ///
    ///     The members are NumPy's, names included (API-parity batch 13): <see cref="files"/>, <see cref="f"/>,
    ///     <see cref="zip"/>, <see cref="fid"/>, <see cref="allow_pickle"/>, <see cref="max_header_size"/>,
    ///     <see cref="keys"/>, <see cref="values"/>, <see cref="items"/>, <see cref="get"/> and <see cref="close"/>. The
    ///     dictionary interface's <c>Count</c> / <c>Keys</c> / <c>Values</c> are explicit (NumPy has <c>len(z)</c>,
    ///     <c>keys()</c> and <c>values()</c>); <see cref="ContainsKey"/> and <see cref="TryGetValue"/> stay the C#
    ///     spellings of NumPy's <c>key in z</c>, and <c>foreach</c> keeps the dictionary's name / array pairs.
    ///
    ///     A missing key raises NumPy's errors: the indexer a <see cref="KeyError"/> (<c>'x is not a file in the
    ///     archive'</c>) and dot access an <see cref="AttributeError"/> (<c>x</c>). Footgun: although the class is an
    ///     <see cref="IReadOnlyDictionary{TKey, TValue}"/>, its indexer does NOT throw .NET's
    ///     <see cref="KeyNotFoundException"/>; test with <see cref="ContainsKey"/> or <see cref="TryGetValue"/> first, or
    ///     catch <see cref="KeyError"/>.
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
    ///     progress, and every access after it raises <see cref="ObjectDisposedException"/>. <see cref="zip"/> hands out
    ///     the archive itself and bypasses the gate.
    /// </remarks>
    public sealed class NpzFile : IReadOnlyDictionary<string, NDArray>, IDisposable
    {
        /// <summary>How many keys <see cref="ToString"/> lists before eliding — NumPy's <c>_MAX_REPR_ARRAY_COUNT</c>.</summary>
        private const int MaxReprArrayCount = 5;

        private readonly bool _ownStream;
        private readonly string _name;

        /// <summary>The dot-access bag (NumPy's <c>BagObj</c>); null once <see cref="close"/> ran (NumPy's <c>self.f =
        ///     None</c>, which breaks the reference cycle).</summary>
        private dynamic _bag;

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
        ///     Open an archive over a stream - NumPy's <c>NpzFile(fid, own_fid=False, allow_pickle=False, *,
        ///     max_header_size=10000)</c>.
        /// </summary>
        /// <param name="fid">A readable, seekable stream holding the ZIP archive (NumPy's file object).</param>
        /// <param name="own_fid">When true, closing the archive also closes <paramref name="fid"/>, and <see cref="fid"/>
        ///     reports it (NumPy keeps <c>self.fid</c> only for an owned file).</param>
        /// <param name="allow_pickle">Whether members are trusted; see <see cref="NpyFormat.ReadArray"/>.</param>
        /// <param name="max_header_size">Per-member header size cap.</param>
        /// <exception cref="ArgumentNullException"><paramref name="fid"/> is null.</exception>
        /// <exception cref="InvalidDataException">The stream is not a ZIP archive (an owned stream is closed first).</exception>
        /// <remarks>NumPy's names since API-parity batch 13 (they were <c>stream</c>, <c>ownStream</c>,
        ///     <c>allowPickle</c>, <c>maxHeaderSize</c>); NumPy's <c>pickle_kwargs</c> has no NumSharp counterpart (its
        ///     unpickler takes no Python keyword dictionary).</remarks>
        public NpzFile(Stream fid, bool own_fid = false, bool allow_pickle = false,
                       long max_header_size = NpyFormat.MaxHeaderSize)
        {
            _stream = fid ?? throw new ArgumentNullException(nameof(fid));
            _ownStream = own_fid;
            allow_pickle_ = allow_pickle;
            max_header_size_ = max_header_size;
            _name = (fid as FileStream)?.Name ?? "object";

            try
            {
                _archive = new ZipArchive(fid, ZipArchiveMode.Read, leaveOpen: true);
            }
            catch
            {
                // An owned stream would otherwise stay open behind an archive that never existed.
                if (own_fid) fid.Dispose();
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

            _bag = new BagObj(this);
        }

        /// <summary>
        ///     Open an archive from a file path (NumPy's <c>NpzFile(fid)</c> with a path, which <c>zipfile</c> opens). The
        ///     file handle is owned: <see cref="fid"/> reports it and <see cref="close"/> closes it.
        /// </summary>
        /// <param name="fid">The archive's path.</param>
        /// <param name="allow_pickle">Whether members are trusted; see <see cref="NpyFormat.ReadArray"/>.</param>
        /// <param name="max_header_size">Per-member header size cap.</param>
        /// <exception cref="ArgumentNullException"><paramref name="fid"/> is null.</exception>
        /// <exception cref="FileNotFoundException">No such file.</exception>
        /// <exception cref="InvalidDataException">The file is not a ZIP archive (the file is closed first).</exception>
        public NpzFile(string fid, bool allow_pickle = false, long max_header_size = NpyFormat.MaxHeaderSize)
            : this(NDFileShare.OpenRead(fid), own_fid: true, allow_pickle, max_header_size)
        {
            _name = fid;
        }

        /// <summary>The array names in the archive, with <c>.npy</c> stripped — NumPy's <c>.files</c> (a repeated name
        ///     appears twice, as NumPy's list built from the zip's name list).</summary>
        public IReadOnlyList<string> files => _files;

        /// <summary>Whether members are loaded as trusted input (NumPy's <c>allow_pickle</c>).</summary>
        public bool allow_pickle => allow_pickle_;

        /// <summary>The value behind <see cref="allow_pickle"/> (a field: the property already takes the name).</summary>
        private readonly bool allow_pickle_;

        /// <summary>The per-member header size cap (NumPy's <c>max_header_size</c>).</summary>
        public long max_header_size => max_header_size_;

        /// <summary>The value behind <see cref="max_header_size"/>.</summary>
        private readonly long max_header_size_;

        /// <summary>
        ///     Dot-notation access to members — NumPy's <c>npz.f.weights</c> (its <c>BagObj</c>). Requires a
        ///     <c>dynamic</c> receiver: <c>NDArray w = npz.f.weights;</c>. Null once <see cref="close"/> ran, as NumPy's
        ///     <c>self.f = None</c>.
        /// </summary>
        public dynamic f => _bag;

        /// <summary>
        ///     The underlying archive (NumPy's <c>zip</c>), for callers that need entry metadata; null once
        ///     <see cref="close"/> ran, as NumPy's. Reading entries through it bypasses the gate that serializes this
        ///     class's own reads, so do not read it while other threads load members.
        /// </summary>
        /// <value>The open archive, or null after <see cref="close"/>.</value>
        public ZipArchive zip => _archive;

        /// <summary>
        ///     The file NumPy's <c>fid</c> names: the stream this archive OWNS (a path's file, or a stream given with
        ///     <c>own_fid: true</c>), null for a borrowed stream (NumPy keeps <c>self.fid</c> only when it owns it) and after
        ///     <see cref="close"/>.
        /// </summary>
        public Stream fid => _ownStream ? _stream : null;

        /// <summary>The number of members (NumPy's <c>len(z)</c>; <c>files.Count</c> is the same number).</summary>
        int IReadOnlyCollection<KeyValuePair<string, NDArray>>.Count => _files.Count;

        /// <summary>The member names (the dictionary interface's; NumPy's spelling is <see cref="keys"/>).</summary>
        IEnumerable<string> IReadOnlyDictionary<string, NDArray>.Keys => _files;

        /// <summary>Every member's array (the dictionary interface's; NumPy's spelling is <see cref="values"/>).</summary>
        IEnumerable<NDArray> IReadOnlyDictionary<string, NDArray>.Values => values();

        /// <summary>The member names — NumPy's <c>keys()</c>, the same names as <see cref="files"/>.</summary>
        /// <returns>The names, in the archive's order.</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        public IEnumerable<string> keys()
        {
            ThrowIfDisposed();
            return _files;
        }

        /// <summary>Every member's array — NumPy's <c>values()</c>. Enumerating it loads and caches every member.</summary>
        /// <returns>The arrays, in the archive's order (each the instance the indexer returns).</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed (checked as each member is read).</exception>
        /// <exception cref="FormatException">A member is not a .npy file.</exception>
        public IEnumerable<NDArray> values()
        {
            foreach (string key in _files)
                yield return this[key];
        }

        /// <summary>Every member as a name / array pair — NumPy's <c>items()</c>. Enumerating it loads and caches every
        ///     member.</summary>
        /// <returns>The pairs, in the archive's order (<c>foreach (var (name, array) in z.items())</c>).</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed (checked as each member is read).</exception>
        /// <exception cref="FormatException">A member is not a .npy file.</exception>
        public IEnumerable<(string key, NDArray value)> items()
        {
            foreach (string key in _files)
                yield return (key, this[key]);
        }

        /// <summary>
        ///     The array under <paramref name="key"/>, or <paramref name="default"/> when there is no such member —
        ///     NumPy's <c>get(key, default=None)</c> (the Mapping method: no KeyError).
        /// </summary>
        /// <param name="key">The member name, with or without the <c>.npy</c> suffix.</param>
        /// <param name="default">The answer for a missing member; null (NumPy's None) by default. A C# number converts.</param>
        /// <returns>The member's array (the indexer's instance), or <paramref name="default"/>.</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        /// <exception cref="FormatException">The member is not a .npy file.</exception>
        public NDArray get(string key, NDArray @default = null)
            => TryGetValue(key, out NDArray value) ? value : @default;

        /// <summary>
        ///     The array stored under <paramref name="key"/>, with or without the <c>.npy</c> suffix.
        ///     Loaded on first access and cached; every later access, from any thread, returns the same instance.
        /// </summary>
        /// <param name="key">The member name, with or without the <c>.npy</c> suffix.</param>
        /// <returns>The member's array, owned by the caller that first read it (the archive never disposes it).</returns>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
        /// <exception cref="KeyError">No such member: NumPy's <c>KeyError</c>, whose text is Python's <c>str()</c> of it
        ///     (<c>'x is not a file in the archive'</c>, quoted).</exception>
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
                    throw new KeyError(PyLiteral.Repr($"{key} is not a file in the archive"));

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
                        NDArray array = NpyFormat.ReadArray(member, allow_pickle_, max_header_size_);
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
        /// <exception cref="KeyError">No such member (the indexer's NumPy <c>KeyError</c> and text).</exception>
        /// <exception cref="InvalidDataException">A compressed member's data is corrupt.</exception>
        public byte[] GetRawBytes(string key)
        {
            ThrowIfDisposed();

            if (!_keyToEntry.TryGetValue(key, out ZipArchiveEntry entry))
                throw new KeyError(PyLiteral.Repr($"{key} is not a file in the archive"));

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

        /// <summary>Close the archive and release an owned file handle — NumPy's <c>close()</c> (<see cref="zip"/>,
        ///     <see cref="fid"/> and <see cref="f"/> become null).</summary>
        public void close() => Dispose();

        /// <summary>Close the archive and release an owned file handle (<see cref="close"/>, for <c>using</c>).</summary>
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
                _bag = null;

                _archive?.Dispose();
                _archive = null;

                if (_ownStream)
                    _stream?.Dispose();
                _stream = null;

                _cache.Clear();
            }
        }

        /// <summary>Throws once the archive is closed: the guard every member that reads the archive runs first.</summary>
        /// <exception cref="ObjectDisposedException">The archive was disposed.</exception>
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
        ///     <see cref="NpzFile.f"/>.
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
                // NumPy's BagObj.__getattribute__: `raise AttributeError(key) from None` around the archive's KeyError,
                // which CPython's attribute lookup then marks with the name and the BagObj (set_attribute_error_context).
                if (!_owner.ContainsKey(binder.Name))
                    throw new AttributeError(binder.Name, binder.Name, this);

                result = _owner[binder.Name];
                return true;
            }

            /// <summary>The member names the bag answers to (NumPy's <c>dir(npz.f)</c>): the archive's
            ///     <see cref="NpzFile.files"/>, the stripped names a debugger or a dynamic binder lists.</summary>
            /// <returns>The archive's member names, in its order.</returns>
            public override IEnumerable<string> GetDynamicMemberNames() => _owner.files;

            public override string ToString() => _owner.ToString();
        }
    }
}
