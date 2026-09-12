using System;
using System.Collections.Generic;
using System.Dynamic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using Parquet;
using Parquet.Schema;

namespace NumSharp.Interop.ParquetNet
{
    /// <summary>
    /// A lazily-read, column-addressable view over a Parquet file whose columns materialize as NumSharp
    /// <see cref="NDArray"/>s backed by raw unmanaged memory. Modeled on NumSharp's <c>NpzFile</c>: index by
    /// column name (<c>pf["close"]</c>), dot-access via <see cref="f"/> (<c>pf.f.close</c>), enumerate
    /// <see cref="Columns"/>, stream <see cref="RowGroups"/>, and dispose to release the underlying reader.
    ///
    /// <para>Columns are decoded on first access and (by default) cached, so only the columns you touch commit
    /// unmanaged memory (and, with a projection, only those columns are read from disk at all). This is the
    /// "NumSharp face" of the bridge; the "Parquet.Net face" is the set of extension methods in the
    /// <c>Parquet</c> namespace (see <c>NumSharpParquetExtensions</c>).</para>
    /// </summary>
    public sealed class ParquetFile : IDisposable
    {
        private readonly ParquetReader _reader;
        private readonly Stream _stream;
        private readonly bool _ownsStream;
        private readonly ParquetLoadOptions _options;
        private readonly Dictionary<string, DataField> _fields;
        private readonly Dictionary<string, NDArray> _cache = new Dictionary<string, NDArray>(StringComparer.Ordinal);
        private readonly object _gate = new object();
        private bool _disposed;

        private ParquetFile(ParquetReader reader, Stream stream, bool ownsStream, ParquetLoadOptions options)
        {
            _reader = reader;
            _stream = stream;
            _ownsStream = ownsStream;
            _options = options;
            _fields = BuildFieldMap(reader, options);
        }

        // Loadable columns (flat, supported dtype), honoring the optional projection, keyed by field name.
        private static Dictionary<string, DataField> BuildFieldMap(ParquetReader reader, ParquetLoadOptions options)
        {
            var map = new Dictionary<string, DataField>(StringComparer.Ordinal);
            HashSet<string> projection = options.Columns is null
                ? null
                : new HashSet<string>(options.Columns, StringComparer.Ordinal);

            foreach (DataField f in reader.Schema.GetDataFields())
            {
                if (!ParquetConvert.IsLoadable(f))
                    continue;
                if (projection != null && !projection.Contains(f.Name))
                    continue;
                map[f.Name] = f;   // last wins on the rare duplicate nested-leaf-name case
            }
            return map;
        }

        // ------------------------------------------------------------------ factories

        /// <summary>Open a Parquet file by path (synchronous).</summary>
        public static ParquetFile Load(string path, ParquetLoadOptions options = null)
            => RunSync(() => LoadAsync(path, options));

        /// <summary>Open a Parquet file over an existing stream (synchronous). The stream is left open by default.</summary>
        public static ParquetFile Load(Stream stream, ParquetLoadOptions options = null, bool leaveOpen = true)
            => RunSync(() => LoadAsync(stream, options, leaveOpen));

        /// <summary>Open a Parquet file by path.</summary>
        public static async Task<ParquetFile> LoadAsync(string path, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            FileStream fs = File.OpenRead(path);
            try
            {
                ParquetReader reader = await ParquetReader.CreateAsync(fs, cancellationToken: ct).ConfigureAwait(false);
                return new ParquetFile(reader, fs, ownsStream: true, ParquetLoadOptions.OrDefault(options));
            }
            catch
            {
                fs.Dispose();
                throw;
            }
        }

        /// <summary>Open a Parquet file over an existing stream. The stream is left open on dispose unless <paramref name="leaveOpen"/> is false.</summary>
        public static async Task<ParquetFile> LoadAsync(Stream stream, ParquetLoadOptions options = null, bool leaveOpen = true, CancellationToken ct = default)
        {
            ParquetReader reader = await ParquetReader.CreateAsync(stream, cancellationToken: ct).ConfigureAwait(false);
            return new ParquetFile(reader, stream, ownsStream: !leaveOpen, ParquetLoadOptions.OrDefault(options));
        }

        // ------------------------------------------------------------------ schema

        /// <summary>The raw Parquet.Net schema.</summary>
        public ParquetSchema Schema => _reader.Schema;

        /// <summary>Names of the loadable columns (flat, supported dtype, after any projection).</summary>
        public string[] Columns => _fields.Keys.ToArray();

        /// <summary>Total number of rows in the file.</summary>
        public long RowCount => _reader.Metadata?.NumRows ?? 0;

        /// <summary>Number of row groups.</summary>
        public int RowGroupCount => _reader.RowGroupCount;

        // ------------------------------------------------------------------ column access

        /// <summary>Materialize a column as an NDArray (lazy + cached). Shorthand for <see cref="Column(string, object)"/>.</summary>
        public NDArray this[string name] => Column(name);

        /// <summary>
        /// Materialize a column as an NDArray backed by raw unmanaged memory. Decoded on first access and,
        /// when <see cref="ParquetLoadOptions.CacheColumns"/> is set (default), cached so repeated access
        /// returns the same instance. Pass <paramref name="nullFill"/> to override the null-fill value for this
        /// call only (such calls are never cached).
        /// </summary>
        public NDArray Column(string name, object nullFill = null)
        {
            ThrowIfDisposed();
            if (!_fields.TryGetValue(name, out DataField field))
                throw new ArgumentException(
                    $"column '{name}' not found or not loadable. Available: {string.Join(", ", _fields.Keys)}", nameof(name));

            if (nullFill != null)
            {
                // A per-call fill is not cached — it would clash with the file-default entry under the same key.
                var one = new ParquetLoadOptions
                {
                    Columns = _options.Columns,
                    Nulls = _options.Nulls,
                    NullFill = nullFill,
                    CacheColumns = false
                };
                return RunSync(() => ParquetConvert.ReadColumnAsync(_reader, field, one, CancellationToken.None).AsTask());
            }

            lock (_gate)
            {
                if (_options.CacheColumns && _cache.TryGetValue(name, out NDArray cached))
                    return cached;

                NDArray nd = RunSync(() => ParquetConvert.ReadColumnAsync(_reader, field, _options, CancellationToken.None).AsTask());
                if (_options.CacheColumns)
                    _cache[name] = nd;
                return nd;
            }
        }

        /// <summary>Try to materialize a column; returns false when the name is unknown / not loadable.</summary>
        public bool TryGetColumn(string name, out NDArray value)
        {
            if (!_fields.ContainsKey(name))
            {
                value = null;
                return false;
            }
            value = Column(name);
            return true;
        }

        /// <summary>Materialize several columns into a name → NDArray dictionary (all loadable columns when none are named).</summary>
        public IReadOnlyDictionary<string, NDArray> ToDictionary(params string[] names)
        {
            IEnumerable<string> keys = names is { Length: > 0 } ? names : _fields.Keys;
            var result = new Dictionary<string, NDArray>(StringComparer.Ordinal);
            foreach (string k in keys)
                result[k] = Column(k);
            return result;
        }

        /// <summary>Dynamic dot-access to columns, e.g. <c>pf.f.close</c> (mirrors NpzFile's <c>f</c>).</summary>
        public dynamic f => new BagObj(this);

        // ------------------------------------------------------------------ streaming

        /// <summary>
        /// Stream the file one row group at a time (bounded memory). Each <see cref="ParquetRowGroup"/>
        /// materializes its columns as short-lived NDArrays of that group's row count — nothing scales with the
        /// whole file. Dispose each group (a <c>foreach</c> with <c>using</c>) to release its reader promptly.
        /// </summary>
        public IEnumerable<ParquetRowGroup> RowGroups
        {
            get
            {
                ThrowIfDisposed();
                for (int g = 0; g < _reader.RowGroupCount; g++)
                    yield return new ParquetRowGroup(_reader.OpenRowGroupReader(g), _fields, _options);
            }
        }

        // ------------------------------------------------------------------ lifetime

        /// <summary>
        /// Dispose the underlying reader (and the stream, if this instance opened it). Column NDArrays already
        /// handed out remain valid — each owns its pooled unmanaged buffer through NumSharp's reference
        /// counting, independent of the file.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _reader.DisposeAsync().AsTask().GetAwaiter().GetResult();
            if (_ownsStream)
                _stream?.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ParquetFile));
        }

        /// <summary>
        /// Run an async operation synchronously off the thread pool. Using <see cref="Task.Run{TResult}(Func{Task{TResult}})"/>
        /// (rather than a bare <c>.GetAwaiter().GetResult()</c> on the caller's context) avoids the classic
        /// sync-over-async deadlock in UI / ASP.NET synchronization contexts.
        /// </summary>
        internal static T RunSync<T>(Func<Task<T>> func) => Task.Run(func).GetAwaiter().GetResult();

        // pf.f.<name> dot access
        private sealed class BagObj : DynamicObject
        {
            private readonly ParquetFile _file;
            public BagObj(ParquetFile file) => _file = file;

            public override bool TryGetMember(GetMemberBinder binder, out object result)
            {
                result = _file[binder.Name];
                return true;
            }

            public override IEnumerable<string> GetDynamicMemberNames() => _file.Columns;
        }
    }

    /// <summary>
    /// One row group of a <see cref="ParquetFile"/>, for bounded-memory streaming. Materialize a column with
    /// <see cref="Column"/> to get an NDArray of <see cref="RowCount"/> elements.
    /// </summary>
    public sealed class ParquetRowGroup : IDisposable
    {
        private readonly ParquetRowGroupReader _rg;
        private readonly Dictionary<string, DataField> _fields;
        private readonly ParquetLoadOptions _options;

        internal ParquetRowGroup(ParquetRowGroupReader rg, Dictionary<string, DataField> fields, ParquetLoadOptions options)
        {
            _rg = rg;
            _fields = fields;
            _options = options;
        }

        /// <summary>Rows in this row group.</summary>
        public long RowCount => _rg.RowCount;

        /// <summary>Materialize one column of this row group as an NDArray of <see cref="RowCount"/> elements.</summary>
        public NDArray Column(string name)
        {
            if (!_fields.TryGetValue(name, out DataField field))
                throw new ArgumentException(
                    $"column '{name}' not found or not loadable. Available: {string.Join(", ", _fields.Keys)}", nameof(name));
            return ParquetFile.RunSync(() => ParquetConvert.ReadRowGroupColumnAsync(_rg, field, _options, CancellationToken.None).AsTask());
        }

        /// <summary>Release the underlying row-group reader.</summary>
        public void Dispose() => _rg.Dispose();
    }
}
