using System;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using Parquet;
using Parquet.Schema;

namespace NumSharp.Interop.ParquetNet
{
    /// <summary>
    /// Core conversion between a Parquet column (Parquet.Net) and a NumSharp <see cref="NDArray"/> whose data
    /// lives in raw unmanaged memory.
    ///
    /// <para><b>How the bytes get into NDArray memory.</b> Parquet.Net's read API is buffer-based:
    /// <c>ParquetRowGroupReader.ReadAsync&lt;T&gt;(field, Memory&lt;T&gt;)</c> decodes a column chunk into a
    /// caller-supplied buffer (for every encoding — plain, RLE/bit-packed, delta, and dictionary, whose values
    /// it "explodes" into that same buffer). We hand it a <see cref="System.Memory{T}"/> that is a window over
    /// an NDArray's own <c>UnmanagedMemoryBlock&lt;T&gt;</c> (via <see cref="UnmanagedMemoryManager{T}"/>), so
    /// for a <b>required</b> (non-null) column the decoded values land directly in NDArray memory — no
    /// intermediate managed array, no extra copy.</para>
    ///
    /// <para><b>Nullable columns.</b> That fast overload cannot be used when the field is nullable (Parquet.Net
    /// requires a definition-levels buffer then), so nullable columns are read through the nullable
    /// <c>Memory&lt;T?&gt;</c> overload into a managed <c>T?[]</c> and then scattered into the NDArray window,
    /// applying the null-fill policy (see <see cref="NullHandling"/>).</para>
    ///
    /// <para><b>Supported dtypes.</b> The twelve Parquet.Net CLR column types that map 1:1 onto a NumSharp
    /// <see cref="NPTypeCode"/>: bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double,
    /// decimal. Everything else (BigDecimal, BigInteger, DateTime / DateOnly / TimeOnly, Interval, Guid,
    /// string / byte[], and any nested or repeated field) is rejected with a clear message. NumSharp's Half,
    /// Char and Complex dtypes have no Parquet.Net source type on the read path.</para>
    /// </summary>
    internal static class ParquetConvert
    {
        // -------------------------------------------------------------- dtype support

        /// <summary>Maps a Parquet.Net CLR column type to a NumSharp <see cref="NPTypeCode"/>, or null if unsupported.</summary>
        internal static NPTypeCode? MapOrNull(Type t)
        {
            if (t == typeof(bool)) return NPTypeCode.Boolean;
            if (t == typeof(byte)) return NPTypeCode.Byte;
            if (t == typeof(sbyte)) return NPTypeCode.SByte;
            if (t == typeof(short)) return NPTypeCode.Int16;
            if (t == typeof(ushort)) return NPTypeCode.UInt16;
            if (t == typeof(int)) return NPTypeCode.Int32;
            if (t == typeof(uint)) return NPTypeCode.UInt32;
            if (t == typeof(long)) return NPTypeCode.Int64;
            if (t == typeof(ulong)) return NPTypeCode.UInt64;
            if (t == typeof(float)) return NPTypeCode.Single;
            if (t == typeof(double)) return NPTypeCode.Double;
            if (t == typeof(decimal)) return NPTypeCode.Decimal;
            return null;
        }

        /// <summary>True when the field is a flat (non-repeated) column of a supported dtype.</summary>
        internal static bool IsLoadable(DataField field)
            => field.MaxRepetitionLevel == 0 && MapOrNull(field.ClrType).HasValue;

        private static void RejectIfUnloadable(DataField field)
        {
            if (field.MaxRepetitionLevel > 0)
                throw new NotSupportedException(
                    $"Parquet column '{field.Path}' is a repeated/list field; it cannot be loaded into a rectangular NDArray.");
            if (!MapOrNull(field.ClrType).HasValue)
                throw new NotSupportedException(
                    $"Parquet column '{field.Path}' has type {field.ClrType}, which has no NumSharp dtype. " +
                    "Supported: bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal.");
        }

        // -------------------------------------------------------------- whole-column read (all row groups)

        /// <summary>Reads an entire column (across every row group) into one NDArray backed by pooled unmanaged memory.</summary>
        /// <param name="reader">The open Parquet reader.</param>
        /// <param name="field">The flat, supported column to read.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>The whole column as an <see cref="NDArray"/>.</returns>
        /// <exception cref="NotSupportedException">The field is repeated/list or its CLR type has no NumSharp dtype.</exception>
        internal static ValueTask<NDArray> ReadColumnAsync(ParquetReader reader, DataField field, ParquetLoadOptions options, CancellationToken ct)
        {
            RejectIfUnloadable(field);
            // Prefer the footer's row count; fall back to summing per-group counts if metadata is absent.
            long total = reader.Metadata?.NumRows ?? SumRows(reader);
            Type t = field.ClrType;

            // Monomorphizing dispatch: turn the runtime CLR column type into the compile-time generic T so the
            // whole read path (buffer allocation, the decode window, null-scatter) is strongly typed and unmanaged.
            if (t == typeof(bool)) return ReadTypedAsync<bool>(reader, field, total, options, ct);
            if (t == typeof(byte)) return ReadTypedAsync<byte>(reader, field, total, options, ct);
            if (t == typeof(sbyte)) return ReadTypedAsync<sbyte>(reader, field, total, options, ct);
            if (t == typeof(short)) return ReadTypedAsync<short>(reader, field, total, options, ct);
            if (t == typeof(ushort)) return ReadTypedAsync<ushort>(reader, field, total, options, ct);
            if (t == typeof(int)) return ReadTypedAsync<int>(reader, field, total, options, ct);
            if (t == typeof(uint)) return ReadTypedAsync<uint>(reader, field, total, options, ct);
            if (t == typeof(long)) return ReadTypedAsync<long>(reader, field, total, options, ct);
            if (t == typeof(ulong)) return ReadTypedAsync<ulong>(reader, field, total, options, ct);
            if (t == typeof(float)) return ReadTypedAsync<float>(reader, field, total, options, ct);
            if (t == typeof(double)) return ReadTypedAsync<double>(reader, field, total, options, ct);
            if (t == typeof(decimal)) return ReadTypedAsync<decimal>(reader, field, total, options, ct);
            throw new NotSupportedException($"unsupported dtype {t}");
        }

        private static long SumRows(ParquetReader reader)
        {
            long total = 0;
            for (int g = 0; g < reader.RowGroupCount; g++)
            {
                using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                total += rg.RowCount;
            }
            return total;
        }

        private static async ValueTask<NDArray> ReadTypedAsync<T>(ParquetReader reader, DataField field, long total, ParquetLoadOptions options, CancellationToken ct)
            where T : unmanaged
        {
            // One pooled unmanaged block sized to the whole column; each row group fills its own sub-window.
            var block = new UnmanagedMemoryBlock<T>(total);   // NumSharp-owned raw memory
            try
            {
                T fill = ResolveFill<T>(options);
                bool nullable = field.MaxDefinitionLevel > 0;
                long off = 0;
                for (int g = 0; g < reader.RowGroupCount; g++)
                {
                    using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                    int n = checked((int)rg.RowCount);
                    if (n == 0)
                        continue;
                    await ReadGroupIntoAsync(rg, field, block, off, n, nullable, options.Nulls, fill, ct).ConfigureAwait(false);
                    off += n;
                }
                return new NDArray(new ArraySlice<T>(block));
            }
            catch
            {
                block.Free();   // release the pooled block on failure — nothing else references it yet
                throw;
            }
        }

        // -------------------------------------------------------------- single row-group read (streaming)

        /// <summary>Reads a single row group's column into an NDArray of that group's row count.</summary>
        /// <param name="rg">The open row-group reader.</param>
        /// <param name="field">The flat, supported column to read.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>This row group's slice of the column as an <see cref="NDArray"/>.</returns>
        /// <exception cref="NotSupportedException">The field is repeated/list or its CLR type has no NumSharp dtype.</exception>
        internal static ValueTask<NDArray> ReadRowGroupColumnAsync(ParquetRowGroupReader rg, DataField field, ParquetLoadOptions options, CancellationToken ct)
        {
            RejectIfUnloadable(field);
            Type t = field.ClrType;

            // Same monomorphizing dispatch as ReadColumnAsync, over a single row group's row count.
            if (t == typeof(bool)) return ReadOneGroupTypedAsync<bool>(rg, field, options, ct);
            if (t == typeof(byte)) return ReadOneGroupTypedAsync<byte>(rg, field, options, ct);
            if (t == typeof(sbyte)) return ReadOneGroupTypedAsync<sbyte>(rg, field, options, ct);
            if (t == typeof(short)) return ReadOneGroupTypedAsync<short>(rg, field, options, ct);
            if (t == typeof(ushort)) return ReadOneGroupTypedAsync<ushort>(rg, field, options, ct);
            if (t == typeof(int)) return ReadOneGroupTypedAsync<int>(rg, field, options, ct);
            if (t == typeof(uint)) return ReadOneGroupTypedAsync<uint>(rg, field, options, ct);
            if (t == typeof(long)) return ReadOneGroupTypedAsync<long>(rg, field, options, ct);
            if (t == typeof(ulong)) return ReadOneGroupTypedAsync<ulong>(rg, field, options, ct);
            if (t == typeof(float)) return ReadOneGroupTypedAsync<float>(rg, field, options, ct);
            if (t == typeof(double)) return ReadOneGroupTypedAsync<double>(rg, field, options, ct);
            if (t == typeof(decimal)) return ReadOneGroupTypedAsync<decimal>(rg, field, options, ct);
            throw new NotSupportedException($"unsupported dtype {t}");
        }

        private static async ValueTask<NDArray> ReadOneGroupTypedAsync<T>(ParquetRowGroupReader rg, DataField field, ParquetLoadOptions options, CancellationToken ct)
            where T : unmanaged
        {
            int n = checked((int)rg.RowCount);
            var block = new UnmanagedMemoryBlock<T>(n);
            try
            {
                if (n > 0)
                {
                    T fill = ResolveFill<T>(options);
                    bool nullable = field.MaxDefinitionLevel > 0;
                    await ReadGroupIntoAsync(rg, field, block, 0, n, nullable, options.Nulls, fill, ct).ConfigureAwait(false);
                }
                return new NDArray(new ArraySlice<T>(block));
            }
            catch
            {
                block.Free();
                throw;
            }
        }

        // -------------------------------------------------------------- the shared group -> window read

        private static async ValueTask ReadGroupIntoAsync<T>(ParquetRowGroupReader rg, DataField field, UnmanagedMemoryBlock<T> block, long off, int n, bool nullable, NullHandling nulls, T fill, CancellationToken ct)
            where T : unmanaged
        {
            if (!nullable)
            {
                // Required column: decode straight into the NDArray's memory window — zero extra copy.
                using UnmanagedMemoryManager<T> window = MakeWindow(block, off, n);
                await rg.ReadAsync<T>(field, window.Memory, null, ct).ConfigureAwait(false);
                return;
            }

            // Nullable column: read into a managed T?[] (the nullable overload — same one Parquet.Net's own
            // DataFrame bridge uses), then scatter into the window applying the null-fill policy.
            var tmp = new T?[n];
            await rg.ReadAsync<T>(field, tmp.AsMemory(), null, ct).ConfigureAwait(false);
            ScatterFill(tmp, block, off, fill, nulls, field);
        }

        // Build a Memory<T> window over block[off .. off+n). Isolated as an unsafe method so the pointer never
        // has to live in an async method across an await (the manager object holds it on the heap instead).
        private static unsafe UnmanagedMemoryManager<T> MakeWindow<T>(UnmanagedMemoryBlock<T> block, long off, int n)
            where T : unmanaged
            => new UnmanagedMemoryManager<T>(block.Address + off, n);

        // Expand a nullable T?[] into the unmanaged window: real values as-is, nulls -> fill (or throw on Raise).
        private static unsafe void ScatterFill<T>(T?[] src, UnmanagedMemoryBlock<T> block, long off, T fill, NullHandling nulls, DataField field)
            where T : unmanaged
        {
            T* dst = block.Address + off;
            for (int i = 0; i < src.Length; i++)
            {
                T? v = src[i];
                if (v.HasValue)
                    dst[i] = v.GetValueOrDefault();
                else if (nulls == NullHandling.Raise)
                    throw new InvalidOperationException(
                        $"Parquet column '{field.Path}' contains null values but NullHandling.Raise was requested.");
                else
                    dst[i] = fill;
            }
        }

        // Coerce the option's fill value to T, or default(T) when none was provided.
        private static T ResolveFill<T>(ParquetLoadOptions options) where T : unmanaged
        {
            if (options.NullFill is null)
                return default;
            return (T)Convert.ChangeType(options.NullFill, typeof(T), CultureInfo.InvariantCulture);
        }
    }
}
