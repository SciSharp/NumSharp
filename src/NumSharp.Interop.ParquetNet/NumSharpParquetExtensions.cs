using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Interop.ParquetNet;
using Parquet.Schema;

// Deliberately declared in the `Parquet` namespace — like Parquet.Net's own `AnalysisExtensions`
// (the DataFrame bridge) — so these methods appear on Parquet.Net's own types (Stream, ParquetReader,
// ParquetRowGroupReader) via IntelliSense the moment this package is referenced, without a new `using`.
// The type lives in the NumSharp.Interop.ParquetNet assembly, so it can call the internal ParquetConvert.
namespace Parquet
{
    /// <summary>
    /// Extension methods bridging Parquet.Net to NumSharp <see cref="NDArray"/>s. These are the "Parquet.Net
    /// face" of the bridge: a Parquet.Net user calls them on the objects they already hold, and the result is
    /// an NDArray backed by raw unmanaged memory (see <see cref="ParquetFile"/> for the NumSharp-facing object).
    /// </summary>
    public static class NumSharpParquetExtensions
    {
        /// <summary>
        /// Read every loadable column of a Parquet stream into a name → <see cref="NDArray"/> dictionary — the
        /// NDArray analog of Parquet.Net's <c>Stream.ReadParquetAsDataFrameAsync()</c>.
        /// </summary>
        public static async Task<Dictionary<string, NDArray>> ReadParquetAsNDArraysAsync(
            this Stream stream, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            options = ParquetLoadOptions.OrDefault(options);

            await using ParquetReader reader = await ParquetReader.CreateAsync(stream, cancellationToken: ct).ConfigureAwait(false);

            HashSet<string> projection = options.Columns is null
                ? null
                : new HashSet<string>(options.Columns, StringComparer.Ordinal);

            var result = new Dictionary<string, NDArray>(StringComparer.Ordinal);
            foreach (DataField field in reader.Schema.GetDataFields())
            {
                if (!ParquetConvert.IsLoadable(field))
                    continue;
                if (projection != null && !projection.Contains(field.Name))
                    continue;
                result[field.Name] = await ParquetConvert.ReadColumnAsync(reader, field, options, ct).ConfigureAwait(false);
            }
            return result;
        }

        /// <summary>Read a single column (by name) of an open <see cref="ParquetReader"/> into an NDArray.</summary>
        public static ValueTask<NDArray> ReadColumnAsNDArrayAsync(
            this ParquetReader reader, string columnName, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            DataField field = Array.Find(reader.Schema.GetDataFields(), f => f.Name == columnName)
                ?? throw new ArgumentException($"column '{columnName}' not found", nameof(columnName));
            return ParquetConvert.ReadColumnAsync(reader, field, ParquetLoadOptions.OrDefault(options), ct);
        }

        /// <summary>Read a single column (by <see cref="DataField"/>) of an open <see cref="ParquetReader"/> into an NDArray.</summary>
        public static ValueTask<NDArray> ReadColumnAsNDArrayAsync(
            this ParquetReader reader, DataField field, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            if (field is null) throw new ArgumentNullException(nameof(field));
            return ParquetConvert.ReadColumnAsync(reader, field, ParquetLoadOptions.OrDefault(options), ct);
        }

        /// <summary>
        /// Read one row group's column into an NDArray of that group's row count — the streaming-friendly form,
        /// callable inside a user's own per-row-group loop.
        /// </summary>
        public static ValueTask<NDArray> ReadAsNDArrayAsync(
            this ParquetRowGroupReader rowGroup, DataField field, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (rowGroup is null) throw new ArgumentNullException(nameof(rowGroup));
            if (field is null) throw new ArgumentNullException(nameof(field));
            return ParquetConvert.ReadRowGroupColumnAsync(rowGroup, field, ParquetLoadOptions.OrDefault(options), ct);
        }

        /// <summary>
        /// A <see cref="Memory{T}"/> over the NDArray's own unmanaged buffer, so a Parquet.Net user can decode
        /// straight into an existing NDArray using Parquet.Net's native call, no new API to learn:
        /// <code>await rg.ReadAsync&lt;double&gt;(field, nd.AsMemory&lt;double&gt;());</code>
        /// Requires a C-contiguous array of the matching dtype and ≤ <see cref="int.MaxValue"/> elements
        /// (throws otherwise — the constraints of <c>NDArray.Unsafe.Memory&lt;T&gt;()</c>, which this forwards to).
        /// </summary>
        public static Memory<T> AsMemory<T>(this NDArray nd) where T : unmanaged
        {
            if (nd is null) throw new ArgumentNullException(nameof(nd));
            return nd.Unsafe.Memory<T>();
        }
    }
}
