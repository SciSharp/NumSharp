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
        /// <param name="stream">An open, readable stream positioned at the start of a Parquet file.</param>
        /// <param name="options">
        /// Load options (column projection, null policy, caching); <c>null</c> uses the defaults — all loadable
        /// columns, <see cref="NullHandling.Fill"/>.
        /// </param>
        /// <param name="ct">Token to cancel the read.</param>
        /// <returns>
        /// A dictionary mapping each loadable column's name to its <see cref="NDArray"/>. Columns that are not
        /// loadable (repeated/list fields, or CLR types with no NumSharp dtype) are silently skipped rather than
        /// thrown on; request a specific unloadable column by name to get the explanatory exception instead.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        public static async Task<Dictionary<string, NDArray>> ReadParquetAsNDArraysAsync(
            this Stream stream, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            options = ParquetLoadOptions.OrDefault(options);   // normalize null → default options once, up front

            // Open the reader over the stream; `await using` disposes it at scope exit so the stream is released.
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream, cancellationToken: ct).ConfigureAwait(false);

            // Materialize the projection into an ordinal set once, so the per-field membership test below is O(1)
            // instead of a linear scan of options.Columns for every field. null projection ⇒ take all columns.
            HashSet<string> projection = options.Columns is null
                ? null
                : new HashSet<string>(options.Columns, StringComparer.Ordinal);

            var result = new Dictionary<string, NDArray>(StringComparer.Ordinal);
            foreach (DataField field in reader.Schema.GetDataFields())
            {
                if (!ParquetConvert.IsLoadable(field))
                    continue;   // repeated/list field, or a CLR type with no NumSharp dtype — not representable
                if (projection != null && !projection.Contains(field.Name))
                    continue;   // caller asked for a subset and this column is not in it
                result[field.Name] = await ParquetConvert.ReadColumnAsync(reader, field, options, ct).ConfigureAwait(false);
            }
            return result;
        }

        /// <summary>
        /// Read a single column, named by <paramref name="columnName"/>, of an open <see cref="ParquetReader"/>
        /// into an NDArray spanning every row group in the file.
        /// </summary>
        /// <param name="reader">An open Parquet reader.</param>
        /// <param name="columnName">The flat column's name (a <see cref="DataField"/> name in the schema).</param>
        /// <param name="options">Load options; <c>null</c> uses the defaults.</param>
        /// <param name="ct">Token to cancel the read.</param>
        /// <returns>An <see cref="NDArray"/> holding the whole column, backed by raw unmanaged memory.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="reader"/> is null.</exception>
        /// <exception cref="ArgumentException">No column named <paramref name="columnName"/> exists in the schema.</exception>
        /// <exception cref="NotSupportedException">
        /// The named column is a repeated/list field, or its CLR type has no NumSharp dtype (surfaced when the
        /// returned task is awaited).
        /// </exception>
        public static ValueTask<NDArray> ReadColumnAsNDArrayAsync(
            this ParquetReader reader, string columnName, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            // Resolve the name to a DataField up front so a bad name fails fast here (as ArgumentException),
            // rather than deep inside the async read where the diagnostic would be less clear.
            DataField field = Array.Find(reader.Schema.GetDataFields(), f => f.Name == columnName)
                ?? throw new ArgumentException($"column '{columnName}' not found", nameof(columnName));
            return ParquetConvert.ReadColumnAsync(reader, field, ParquetLoadOptions.OrDefault(options), ct);
        }

        /// <summary>
        /// Read a single column, given its <see cref="DataField"/>, of an open <see cref="ParquetReader"/> into
        /// an NDArray spanning every row group in the file. The field overload skips the by-name schema lookup.
        /// </summary>
        /// <param name="reader">An open Parquet reader.</param>
        /// <param name="field">The schema field to read (obtain it from <c>reader.Schema.GetDataFields()</c>).</param>
        /// <param name="options">Load options; <c>null</c> uses the defaults.</param>
        /// <param name="ct">Token to cancel the read.</param>
        /// <returns>An <see cref="NDArray"/> holding the whole column, backed by raw unmanaged memory.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="reader"/> or <paramref name="field"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        /// The column is a repeated/list field, or its CLR type has no NumSharp dtype (surfaced when awaited).
        /// </exception>
        public static ValueTask<NDArray> ReadColumnAsNDArrayAsync(
            this ParquetReader reader, DataField field, ParquetLoadOptions options = null, CancellationToken ct = default)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            if (field is null) throw new ArgumentNullException(nameof(field));
            return ParquetConvert.ReadColumnAsync(reader, field, ParquetLoadOptions.OrDefault(options), ct);
        }

        /// <summary>
        /// Read one row group's column into an NDArray of that group's row count — the streaming-friendly form,
        /// callable inside a user's own per-row-group loop so peak memory stays bounded by one group, not the file.
        /// </summary>
        /// <param name="rowGroup">An open row-group reader (e.g. from <c>reader.OpenRowGroupReader(g)</c>).</param>
        /// <param name="field">The schema field to read from this row group.</param>
        /// <param name="options">Load options; <c>null</c> uses the defaults.</param>
        /// <param name="ct">Token to cancel the read.</param>
        /// <returns>An <see cref="NDArray"/> of this row group's row count, backed by raw unmanaged memory.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="rowGroup"/> or <paramref name="field"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        /// The column is a repeated/list field, or its CLR type has no NumSharp dtype (surfaced when awaited).
        /// </exception>
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
        /// <typeparam name="T">The element type; must match the array's dtype exactly (no reinterpretation).</typeparam>
        /// <param name="nd">The destination array whose buffer the returned <see cref="Memory{T}"/> aliases.</param>
        /// <returns>
        /// A writable <see cref="Memory{T}"/> aliasing <paramref name="nd"/>'s buffer — writes through it mutate
        /// the array. The window does not root <paramref name="nd"/>; keep the array alive while the memory is in use.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="nd"/> is null.</exception>
        /// <exception cref="InvalidOperationException">
        /// <paramref name="nd"/> is not C-contiguous or has more than <see cref="int.MaxValue"/> elements.
        /// </exception>
        /// <exception cref="ArgumentException"><typeparamref name="T"/> does not match the array's dtype.</exception>
        public static Memory<T> AsMemory<T>(this NDArray nd) where T : unmanaged
        {
            if (nd is null) throw new ArgumentNullException(nameof(nd));
            // Forward to the NDArray's own unmanaged-view accessor; it enforces C-contiguity, the dtype match,
            // and the int-length ceiling, and never takes a reference on the array (see the <returns> note).
            return nd.Unsafe.Memory<T>();
        }
    }
}
