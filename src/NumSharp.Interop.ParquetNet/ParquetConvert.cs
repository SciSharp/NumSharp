using System;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;
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
    /// decimal. Text and bytes read into NumSharp's text dtypes (<see cref="TextKindOf"/>): a UTF-8 column as a
    /// <c>StringDType(na_object=None)</c> array — numpy's view of it is an object array of <c>str</c> and <c>None</c> —
    /// and a binary column as an <c>S</c> array as wide as its longest value. Everything else (BigDecimal,
    /// BigInteger, DateTime / DateOnly / TimeOnly, Interval, Guid, and any nested or repeated field) is rejected
    /// with a clear message. NumSharp's Half, Char and Complex dtypes have no Parquet.Net source type on the read
    /// path (Parquet.Net 6.1.0 refuses a float16 column's schema outright).</para>
    ///
    /// <para><b>Text is a copy.</b> Parquet.Net decodes text to managed strings and bytes to managed arrays, so a
    /// text column is gathered row group by row group into one <c>object[]</c> and packed by <c>np.array</c> — the
    /// same element setter (NumPy's <c>PyArray_Pack</c>) that converts a fill value into its null slots.</para>
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

        /// <summary>True when the field is a flat (non-repeated) column of a supported dtype, text and bytes included.</summary>
        internal static bool IsLoadable(DataField field)
            => field.MaxRepetitionLevel == 0 && (MapOrNull(field.ClrType).HasValue || TextKindOf(field) != TextKind.None);

        /// <summary>What a column that is not a number holds, as the bridge reads it.</summary>
        internal enum TextKind
        {
            /// <summary>Neither text nor bytes: a numeric or boolean column, or one with no NumSharp dtype.</summary>
            None,

            /// <summary>UTF-8 text, read as <c>StringDType(na_object=None)</c> (a null is the NA).</summary>
            Text,

            /// <summary>Bytes, read as an <c>S</c> array as wide as the longest value (a null is <c>b""</c>).</summary>
            Binary,
        }

        /// <summary>
        ///     Classifies a column by both keys the bridge has: Parquet.Net's CLR type — <see cref="ReadOnlyMemory{T}"/> of
        ///     <see cref="char"/> / <see cref="byte"/> in Parquet.Net 6 (the package floor: every UTF-8, large-UTF-8,
        ///     dictionary-encoded and binary column, whoever wrote the file), and <see cref="string"/> / <c>byte[]</c>, which a
        ///     caller's own <c>DataField&lt;string&gt;</c> carries — and the column's Parquet schema, whose UTF-8 annotation
        ///     (<see cref="IsUtf8Annotated"/>) makes text of bytes Parquet.Net hands over undecoded (a JSON column is
        ///     <c>ReadOnlyMemory&lt;byte&gt;</c> in Parquet.Net 6.1.0).
        /// </summary>
        /// <param name="field">The column.</param>
        /// <returns><see cref="TextKind.Text"/>, <see cref="TextKind.Binary"/> or <see cref="TextKind.None"/>.</returns>
        internal static TextKind TextKindOf(DataField field)
        {
            Type t = field.ClrType;
            if (t == typeof(ReadOnlyMemory<char>) || t == typeof(string))
                return TextKind.Text;
            if (t == typeof(ReadOnlyMemory<byte>) || t == typeof(byte[]))
                return IsUtf8Annotated(field.SchemaElement) ? TextKind.Text : TextKind.Binary;
            return TextKind.None;
        }

        /// <summary>
        ///     True when the Parquet schema says the column's bytes are UTF-8 text: the STRING, ENUM or JSON logical type,
        ///     or the legacy UTF8 / ENUM / JSON converted type (BSON is binary).
        /// </summary>
        /// <param name="se">The column's schema element; <c>null</c> for a field built by hand.</param>
        /// <returns>True for a UTF-8 annotation.</returns>
        private static bool IsUtf8Annotated(Parquet.Meta.SchemaElement se)
            => se != null &&
               (se.LogicalType?.STRING != null || se.LogicalType?.ENUM != null || se.LogicalType?.JSON != null ||
                se.ConvertedType is Parquet.Meta.ConvertedType.UTF8 or Parquet.Meta.ConvertedType.ENUM or Parquet.Meta.ConvertedType.JSON);

        /// <summary>
        ///     Refuses a column the bridge cannot read, before any value is touched: a repeated (list) field, or one that is
        ///     neither numeric nor text / bytes — named as TEXT when the schema says UTF-8 (only the decode is missing, not a
        ///     NumSharp dtype), else as a type with no NumSharp dtype.
        /// </summary>
        /// <param name="field">The column about to be read.</param>
        /// <exception cref="NotSupportedException">The column is repeated, or its CLR type is not one the bridge reads.</exception>
        private static void RejectIfUnloadable(DataField field)
        {
            if (field.MaxRepetitionLevel > 0)
                throw new NotSupportedException(
                    $"Parquet column '{field.Path}' is a repeated/list field; it cannot be loaded into a rectangular NDArray.");
            if (!MapOrNull(field.ClrType).HasValue && TextKindOf(field) == TextKind.None)
                throw new NotSupportedException(
                    // The schema key: a column the schema calls UTF-8 text but that Parquet.Net decodes to a CLR type the
                    // bridge does not read (none in Parquet.Net 6.1.0; a guard for other versions) is named as text, not
                    // as a type with "no NumSharp dtype" — NumSharp has text dtypes, the bridge lacks that decode.
                    IsUtf8Annotated(field.SchemaElement)
                        ? $"Parquet column '{field.Path}' holds UTF-8 text, which Parquet.Net decodes as {field.ClrType}; this bridge " +
                          "reads text Parquet.Net decodes as ReadOnlyMemory<char> or string, or as UTF-8 bytes (ReadOnlyMemory<byte> or byte[])."
                        : $"Parquet column '{field.Path}' has type {field.ClrType}, which has no NumSharp dtype. " +
                          "Supported: bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal, text and binary.");
        }

        // -------------------------------------------------------------- whole-column read (all row groups)

        /// <summary>Reads an entire column (across every row group) into one NDArray backed by pooled unmanaged memory.</summary>
        /// <param name="reader">The open Parquet reader.</param>
        /// <param name="field">The flat, supported column to read.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>The whole column as an <see cref="NDArray"/>.</returns>
        /// <exception cref="NotSupportedException">The field is repeated/list or its CLR type is not one this bridge maps
        ///     (thrown by the call itself, before a task is returned).</exception>
        internal static ValueTask<NDArray> ReadColumnAsync(ParquetReader reader, DataField field, ParquetLoadOptions options, CancellationToken ct)
        {
            RejectIfUnloadable(field);
            // Prefer the footer's row count; fall back to summing per-group counts if metadata is absent.
            long total = reader.Metadata?.NumRows ?? SumRows(reader);
            TextKind kind = TextKindOf(field);
            if (kind != TextKind.None)
                return ReadTextColumnAsync(reader, field, kind, total, options, ct);
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

        /// <summary>
        ///     Reads a whole numeric column into one pooled unmanaged block, each row group decoding into its own window of it;
        ///     the block is freed if any group fails, since nothing else references it yet.
        /// </summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="reader">The open Parquet reader.</param>
        /// <param name="field">A flat numeric or boolean column.</param>
        /// <param name="total">The file's row count.</param>
        /// <param name="options">Resolved (non-null) load options: the null policy and fill.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>The column as an <see cref="NDArray"/> of <paramref name="total"/> elements.</returns>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        /// <exception cref="ValueError">The fill cannot be packed into the dtype at a null (NumPy's setitem texts).</exception>
        /// <exception cref="OverflowError">The fill is out of the dtype's range at a null (NumPy's OverflowError texts).</exception>
        private static async ValueTask<NDArray> ReadTypedAsync<T>(ParquetReader reader, DataField field, long total, ParquetLoadOptions options, CancellationToken ct)
            where T : unmanaged
        {
            // One pooled unmanaged block sized to the whole column; each row group fills its own sub-window.
            var block = new UnmanagedMemoryBlock<T>(total);   // NumSharp-owned raw memory
            try
            {
                bool nullable = field.MaxDefinitionLevel > 0;
                long off = 0;
                for (int g = 0; g < reader.RowGroupCount; g++)
                {
                    using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                    int n = checked((int)rg.RowCount);
                    if (n == 0)
                        continue;
                    await ReadGroupIntoAsync(rg, field, block, off, n, nullable, options, ct).ConfigureAwait(false);
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
        /// <exception cref="NotSupportedException">The field is repeated/list or its CLR type is not one this bridge maps.</exception>
        internal static ValueTask<NDArray> ReadRowGroupColumnAsync(ParquetRowGroupReader rg, DataField field, ParquetLoadOptions options, CancellationToken ct)
        {
            RejectIfUnloadable(field);
            TextKind kind = TextKindOf(field);
            if (kind != TextKind.None)
                return ReadTextGroupAsync(rg, field, kind, options, ct);
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

        /// <summary>Reads one row group's numeric column into its own pooled unmanaged block (freed if the read fails).</summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="rg">The open row-group reader.</param>
        /// <param name="field">A flat numeric or boolean column.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>This row group's slice of the column.</returns>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        /// <exception cref="ValueError">The fill cannot be packed into the dtype at a null.</exception>
        /// <exception cref="OverflowError">The fill is out of the dtype's range at a null.</exception>
        private static async ValueTask<NDArray> ReadOneGroupTypedAsync<T>(ParquetRowGroupReader rg, DataField field, ParquetLoadOptions options, CancellationToken ct)
            where T : unmanaged
        {
            int n = checked((int)rg.RowCount);
            var block = new UnmanagedMemoryBlock<T>(n);
            try
            {
                if (n > 0)
                {
                    bool nullable = field.MaxDefinitionLevel > 0;
                    await ReadGroupIntoAsync(rg, field, block, 0, n, nullable, options, ct).ConfigureAwait(false);
                }
                return new NDArray(new ArraySlice<T>(block));
            }
            catch
            {
                block.Free();
                throw;
            }
        }

        // -------------------------------------------------------------- text and bytes

        /// <summary>
        ///     Reads a whole text or byte column (every row group) into one <c>StringDType(na_object=None)</c> or <c>S</c>
        ///     array — numpy's view of the column is an object array of <c>str</c> / <c>bytes</c> and <c>None</c>.
        /// </summary>
        /// <param name="reader">The open Parquet reader.</param>
        /// <param name="field">A flat text or byte column.</param>
        /// <param name="kind">What it holds (<see cref="TextKindOf"/>).</param>
        /// <param name="total">The file's row count.</param>
        /// <param name="options">Resolved (non-null) load options: the null policy and fill.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>The column as a text array of <paramref name="total"/> elements.</returns>
        /// <exception cref="NotSupportedException">More rows than a managed array holds (<see cref="Array.MaxLength"/>).</exception>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        private static async ValueTask<NDArray> ReadTextColumnAsync(ParquetReader reader, DataField field, TextKind kind, long total, ParquetLoadOptions options, CancellationToken ct)
        {
            // The values gather in one managed object[] (Parquet.Net decodes text to strings and bytes to arrays), so the
            // column cannot exceed a managed array's length; refused before anything is read.
            if (total > Array.MaxLength)
                throw new NotSupportedException(
                    $"Parquet column '{field.Path}' has {total} rows; a text column is gathered into a managed array of at most {Array.MaxLength}.");
            var values = new object[total];
            int off = 0;
            for (int g = 0; g < reader.RowGroupCount; g++)
            {
                using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                int n = checked((int)rg.RowCount);
                if (n == 0)
                    continue;
                await ReadTextGroupIntoAsync(rg, field, kind, values, off, n, options, ct).ConfigureAwait(false);
                off += n;
            }

            return PackText(values, kind);
        }

        /// <summary>Reads one row group's text or byte column into a text array of that group's row count.</summary>
        /// <param name="rg">The open row-group reader.</param>
        /// <param name="field">A flat text or byte column.</param>
        /// <param name="kind">What it holds (<see cref="TextKindOf"/>).</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>This row group's slice of the column as a text array.</returns>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        private static async ValueTask<NDArray> ReadTextGroupAsync(ParquetRowGroupReader rg, DataField field, TextKind kind, ParquetLoadOptions options, CancellationToken ct)
        {
            int n = checked((int)rg.RowCount);
            var values = new object[n];
            if (n > 0)
                await ReadTextGroupIntoAsync(rg, field, kind, values, 0, n, options, ct).ConfigureAwait(false);
            return PackText(values, kind);
        }

        /// <summary>
        ///     Decodes one row group of a text or byte column into <paramref name="values"/> starting at
        ///     <paramref name="off"/>: a <see cref="string"/> per text value, a <c>byte[]</c> per byte value, and per null
        ///     the policy's value — the fill (converted later by <see cref="PackText"/>, at the null it fills), else the
        ///     NA (<c>null</c>) for text or <c>b""</c> for bytes, or a refusal under <see cref="NullHandling.Raise"/>.
        /// </summary>
        /// <param name="rg">The open row-group reader.</param>
        /// <param name="field">A flat text or byte column.</param>
        /// <param name="kind">What it holds: bytes the schema calls UTF-8 are decoded to text here.</param>
        /// <param name="values">The column's values, row by row.</param>
        /// <param name="off">The first row of this group in <paramref name="values"/>.</param>
        /// <param name="n">The group's row count.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>A task that completes when the group is decoded.</returns>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        private static async ValueTask ReadTextGroupIntoAsync(ParquetRowGroupReader rg, DataField field, TextKind kind, object[] values, int off, int n,
                                                              ParquetLoadOptions options, CancellationToken ct)
        {
            Type t = field.ClrType;
            if (t == typeof(ReadOnlyMemory<char>) || t == typeof(string))
            {
                // Parquet.Net's string overload decodes UTF-8 for any text field, required or optional; a null is null.
                var texts = new string[n];
                await rg.ReadAsync(field, texts.AsMemory(), null, ct).ConfigureAwait(false);
                for (int i = 0; i < n; i++)
                    values[off + i] = texts[i] ?? NullValue(field, kind, options);
                return;
            }

            var bytes = new byte[n][];
            await rg.ReadAsync(field, bytes.AsMemory(), null, ct).ConfigureAwait(false);
            for (int i = 0; i < n; i++)
            {
                byte[] b = bytes[i];
                // A UTF-8-annotated byte column (JSON in Parquet.Net 6.1.0) is text: decode it as Parquet.Net decodes its
                // own text fields (Encoding.UTF8, an invalid sequence as U+FFFD).
                values[off + i] = b is null ? NullValue(field, kind, options)
                    : kind == TextKind.Text ? System.Text.Encoding.UTF8.GetString(b)
                    : b;
            }
        }

        /// <summary>The value a null becomes under the load options (see <see cref="ReadTextGroupIntoAsync"/>).</summary>
        /// <param name="field">The column (named by a Raise refusal).</param>
        /// <param name="kind">What the column holds.</param>
        /// <param name="options">Resolved (non-null) load options.</param>
        /// <returns>The fill as given (packed with the rest), else <c>null</c> (text's NA) or an empty <c>byte[]</c>.</returns>
        /// <exception cref="InvalidOperationException">The policy is <see cref="NullHandling.Raise"/>.</exception>
        private static object NullValue(DataField field, TextKind kind, ParquetLoadOptions options)
        {
            if (options.Nulls == NullHandling.Raise)
                throw new InvalidOperationException(
                    $"Parquet column '{field.Path}' contains null values but NullHandling.Raise was requested.");
            // default(T) of the dtype: StringDType's NA (numpy's None) for text; S has no NA, so its empty value for bytes.
            return options.NullFill ?? (kind == TextKind.Text ? null : (object)Array.Empty<byte>());
        }

        /// <summary>
        ///     Packs the gathered values into the column's text array with <c>np.array</c>: NumPy's <c>PyArray_Pack</c> per
        ///     element, so a fill in a null's slot converts as NumPy's <c>a[i] = fill</c> does (text stays text; a number
        ///     becomes its text, NaN <c>'nan'</c>; a str into <c>S</c> is ASCII-encoded) — and only where a null was.
        /// </summary>
        /// <param name="values">The values (strings / byte arrays / fills / null).</param>
        /// <param name="kind">Text or bytes.</param>
        /// <returns>A <c>StringDType(na_object=None)</c> array, or an <c>S</c> array as wide as the longest value (at least 1).</returns>
        private static NDArray PackText(object[] values, TextKind kind)
            => kind == TextKind.Text
                ? np.array(values, np.dtypes.StringDType((object)null))
                : np.array(values, np.dtype("S"));

        // -------------------------------------------------------------- the shared group -> window read

        /// <summary>
        ///     Decodes one row group of a numeric column into <paramref name="block"/> at <paramref name="off"/>: a required
        ///     column straight into the window (no copy), a nullable one through a <c>T?[]</c> scattered with the null policy.
        /// </summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="rg">The open row-group reader.</param>
        /// <param name="field">A flat numeric or boolean column.</param>
        /// <param name="block">The column's unmanaged block.</param>
        /// <param name="off">The group's first row in <paramref name="block"/>.</param>
        /// <param name="n">The group's row count.</param>
        /// <param name="nullable">Whether the column has definition levels (a nullable column).</param>
        /// <param name="options">Resolved (non-null) load options: the null policy and fill.</param>
        /// <param name="ct">Cancellation token.</param>
        /// <returns>A task that completes when the group is decoded.</returns>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        /// <exception cref="ValueError">The fill cannot be packed into the dtype at a null.</exception>
        /// <exception cref="OverflowError">The fill is out of the dtype's range at a null.</exception>
        private static async ValueTask ReadGroupIntoAsync<T>(ParquetRowGroupReader rg, DataField field, UnmanagedMemoryBlock<T> block, long off, int n, bool nullable, ParquetLoadOptions options, CancellationToken ct)
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
            ScatterFill(tmp, block, off, options, field);
        }

        // Build a Memory<T> window over block[off .. off+n). Isolated as an unsafe method so the pointer never
        // has to live in an async method across an await (the manager object holds it on the heap instead).
        private static unsafe UnmanagedMemoryManager<T> MakeWindow<T>(UnmanagedMemoryBlock<T> block, long off, int n)
            where T : unmanaged
            => new UnmanagedMemoryManager<T>(block.Address + off, n);

        /// <summary>
        ///     Expands a nullable <c>T?[]</c> into the unmanaged window: values as they are, each null as the fill — converted
        ///     once, at the first null, as NumPy's <c>a[i] = fill</c> (<see cref="ResolveFill{T}"/>) — or a refusal under
        ///     <see cref="NullHandling.Raise"/>.
        /// </summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="src">The decoded values, null at a Parquet null.</param>
        /// <param name="block">The column's unmanaged block.</param>
        /// <param name="off">The first element of this group in <paramref name="block"/>.</param>
        /// <param name="options">Resolved (non-null) load options: the null policy and fill.</param>
        /// <param name="field">The column (named by a Raise refusal).</param>
        /// <exception cref="InvalidOperationException">A null under <see cref="NullHandling.Raise"/>.</exception>
        /// <exception cref="ValueError">The fill cannot be packed into the dtype (NaN into an integer, unparsable text).</exception>
        /// <exception cref="OverflowError">The fill is out of the dtype's range, or an infinity into an integer.</exception>
        private static unsafe void ScatterFill<T>(T?[] src, UnmanagedMemoryBlock<T> block, long off, ParquetLoadOptions options, DataField field)
            where T : unmanaged
        {
            T* dst = block.Address + off;
            T fill = default;
            bool resolved = false;
            for (int i = 0; i < src.Length; i++)
            {
                T? v = src[i];
                if (v.HasValue)
                    dst[i] = v.GetValueOrDefault();
                else if (options.Nulls == NullHandling.Raise)
                    throw new InvalidOperationException(
                        $"Parquet column '{field.Path}' contains null values but NullHandling.Raise was requested.");
                else
                {
                    // A fill replaces nulls only: it is converted at the first null this group holds, so a fill the
                    // column's dtype cannot hold breaks only a column that has a null to put it in.
                    if (!resolved)
                    {
                        fill = ResolveFill<T>(options);
                        resolved = true;
                    }

                    dst[i] = fill;
                }
            }
        }

        /// <summary>
        ///     Converts <see cref="ParquetLoadOptions.NullFill"/> into one <typeparamref name="T"/> element the way NumPy's
        ///     <c>a[i] = fill</c> does — Core's <c>PyArray_Pack</c> port: a float truncated toward zero, a Python int
        ///     range-checked with NumPy's texts, a str parsed (or its truth for bool), a bool as 0 / 1.
        /// </summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="options">The resolved load options; a null fill means <c>default(T)</c>.</param>
        /// <returns>The fill as a <typeparamref name="T"/>.</returns>
        /// <exception cref="OverflowError">An integer the dtype cannot hold, or an infinity into an integer (NumPy's
        ///     <c>OverflowError</c> texts).</exception>
        /// <exception cref="ValueError">NaN into an integer, or text that does not parse (NumPy's <c>ValueError</c> texts).</exception>
        private static unsafe T ResolveFill<T>(ParquetLoadOptions options) where T : unmanaged
        {
            if (options.NullFill is null)
                return default;
            // PackStore writes the element at the start of a cleared 16-byte slot, as setitem does before it copies the item.
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackStore(options.NullFill, InfoOf<T>.NPTypeCode, new Span<byte>(slot, ReductionInitial.SlotBytes));
            return *(T*)slot;
        }
    }
}
