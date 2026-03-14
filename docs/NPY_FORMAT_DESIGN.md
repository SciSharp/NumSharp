# NumPy .npy Format - C# Implementation Design

## Overview

This document describes a complete C# translation of NumPy's `numpy/lib/_format_impl.py`, designed for NumSharp with full long indexing support.

**Goals:**
- 1:1 behavioral compatibility with NumPy's .npy format
- Full support for `long` (int64) shapes, sizes, and counts
- Support for all format versions (1.0, 2.0, 3.0)
- **Full async/await with CancellationToken support**
- **Configurable buffer sizes for memory/performance tuning**
- Clean, maintainable code matching NumPy's structure

**Replaces:**
- `src/NumSharp.Core/APIs/np.save.cs`
- `src/NumSharp.Core/APIs/np.load.cs`

---

## File Structure

```
src/NumSharp.Core/
├── IO/
│   ├── NpyFormat.cs           # Core format implementation (this design)
│   ├── NpyFormat.Constants.cs # Magic bytes, version info, limits
│   ├── NpyFormat.Header.cs    # Header parsing/writing
│   ├── NpyFormat.Read.cs      # Async array reading
│   ├── NpyFormat.Write.cs     # Async array writing
│   ├── NpyOptions.cs          # Configuration options
│   └── NpzArchive.cs          # .npz support (uses NpyFormat internally)
├── APIs/
│   ├── np.save.cs             # Thin wrapper: np.save() → NpyFormat.WriteArrayAsync()
│   └── np.load.cs             # Thin wrapper: np.load() → NpyFormat.ReadArrayAsync()
```

---

## Options Configuration (NpyOptions.cs)

```csharp
namespace NumSharp.IO
{
    /// <summary>
    /// Configuration options for .npy/.npz file operations.
    /// </summary>
    public sealed class NpyOptions
    {
        /// <summary>
        /// Default options instance with standard settings.
        /// </summary>
        public static NpyOptions Default { get; } = new NpyOptions();

        /// <summary>
        /// Buffer size for reading operations.
        /// Default: 256 KiB (matches NumPy's BUFFER_SIZE).
        /// Larger values improve throughput for sequential reads but increase memory usage.
        /// </summary>
        public int ReadBufferSize { get; init; } = 256 * 1024; // 256 KiB

        /// <summary>
        /// Buffer size for writing operations.
        /// Default: 16 MiB (matches NumPy's write buffer for hiding loop overhead).
        /// Larger values improve throughput for large arrays but increase memory usage.
        /// </summary>
        public int WriteBufferSize { get; init; } = 16 * 1024 * 1024; // 16 MiB

        /// <summary>
        /// Maximum allowed header size for reading.
        /// Protects against malicious files with extremely large headers.
        /// Default: 10,000 bytes (matches NumPy's _MAX_HEADER_SIZE).
        /// </summary>
        public int MaxHeaderSize { get; init; } = 10_000;

        /// <summary>
        /// Whether to use async I/O for file operations.
        /// Default: true. Set to false for small files where async overhead exceeds benefit.
        /// </summary>
        public bool UseAsyncIO { get; init; } = true;

        /// <summary>
        /// Threshold in bytes below which sync I/O is used even if UseAsyncIO is true.
        /// Default: 64 KiB. Arrays smaller than this use sync I/O for lower latency.
        /// </summary>
        public long AsyncThreshold { get; init; } = 64 * 1024; // 64 KiB

        /// <summary>
        /// File options for FileStream creation.
        /// Default: Asynchronous | SequentialScan for optimal async performance.
        /// </summary>
        public FileOptions FileOptions { get; init; } = FileOptions.Asynchronous | FileOptions.SequentialScan;

        /// <summary>
        /// Compression level for .npz files.
        /// Default: Fastest for good balance of speed and size.
        /// </summary>
        public CompressionLevel CompressionLevel { get; init; } = CompressionLevel.Fastest;

        /// <summary>
        /// Optional progress callback. Called with (bytesProcessed, totalBytes).
        /// Useful for UI progress bars on large files.
        /// </summary>
        public IProgress<(long BytesProcessed, long TotalBytes)>? Progress { get; init; }

        /// <summary>
        /// Creates options optimized for small files (under 1 MB).
        /// Uses smaller buffers and sync I/O to minimize latency.
        /// </summary>
        public static NpyOptions ForSmallFiles() => new()
        {
            ReadBufferSize = 32 * 1024,      // 32 KiB
            WriteBufferSize = 64 * 1024,     // 64 KiB
            UseAsyncIO = false,
            FileOptions = FileOptions.SequentialScan
        };

        /// <summary>
        /// Creates options optimized for large files (over 100 MB).
        /// Uses larger buffers for maximum throughput.
        /// </summary>
        public static NpyOptions ForLargeFiles() => new()
        {
            ReadBufferSize = 1024 * 1024,    // 1 MiB
            WriteBufferSize = 64 * 1024 * 1024, // 64 MiB
            UseAsyncIO = true,
            FileOptions = FileOptions.Asynchronous | FileOptions.SequentialScan
        };

        /// <summary>
        /// Creates options with a progress reporter.
        /// </summary>
        public static NpyOptions WithProgress(IProgress<(long BytesProcessed, long TotalBytes)> progress) => new()
        {
            Progress = progress
        };
    }
}
```

---

## Constants (NpyFormat.Constants.cs)

```csharp
namespace NumSharp.IO
{
    /// <summary>
    /// Constants for the NumPy .npy file format.
    /// Matches numpy/lib/_format_impl.py
    /// </summary>
    public static partial class NpyFormat
    {
        #region Magic and Version

        /// <summary>Magic string prefix: \x93NUMPY</summary>
        public static readonly byte[] MagicPrefix = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

        /// <summary>Length of magic string including version bytes</summary>
        public const int MagicLength = 8; // 6 (prefix) + 2 (version)

        /// <summary>Alignment boundary for header (supports memory mapping)</summary>
        public const int ArrayAlign = 64;

        /// <summary>
        /// Max digits for growth axis (allows in-place header modification).
        /// = len(str(8 * 2**64 - 1)) for hypothetical int1 dtype
        /// </summary>
        public const int GrowthAxisMaxDigits = 21;

        #endregion

        #region Version Info

        /// <summary>Supported format versions</summary>
        public static readonly (byte Major, byte Minor)[] SupportedVersions =
        {
            (1, 0),
            (2, 0),
            (3, 0)
        };

        /// <summary>
        /// Header size field format per version.
        /// v1.0: 2-byte ushort, latin1 encoding
        /// v2.0: 4-byte uint, latin1 encoding
        /// v3.0: 4-byte uint, utf-8 encoding
        /// </summary>
        public static (int SizeBytes, Encoding Encoding) GetHeaderInfo((byte Major, byte Minor) version)
        {
            return version switch
            {
                (1, 0) => (2, Encoding.Latin1),
                (2, 0) => (4, Encoding.Latin1),
                (3, 0) => (4, Encoding.UTF8),
                _ => throw new NotSupportedException($"Unsupported format version: {version.Major}.{version.Minor}")
            };
        }

        #endregion

        #region Expected Header Keys

        /// <summary>Required keys in header dictionary</summary>
        public static readonly HashSet<string> ExpectedKeys = new() { "descr", "fortran_order", "shape" };

        #endregion
    }
}
```

---

## Header Data Structure

```csharp
namespace NumSharp.IO
{
    /// <summary>
    /// Represents the metadata in a .npy file header.
    /// Corresponds to the dictionary: {'descr': ..., 'fortran_order': ..., 'shape': ...}
    /// </summary>
    public readonly struct NpyHeader
    {
        /// <summary>
        /// Dtype descriptor string (e.g., "&lt;f8", "|b1", "&lt;i4").
        /// Can also be a complex descriptor for structured arrays.
        /// </summary>
        public readonly string Descr;

        /// <summary>
        /// True if data is Fortran-contiguous (column-major).
        /// False if C-contiguous (row-major).
        /// </summary>
        public readonly bool FortranOrder;

        /// <summary>
        /// Shape of the array. Uses long[] to support dimensions > int.MaxValue.
        /// Empty array for scalar.
        /// </summary>
        public readonly long[] Shape;

        /// <summary>
        /// Total number of elements. Computed from shape.
        /// </summary>
        public long Count => Shape.Length == 0 ? 1 : Shape.Aggregate(1L, (a, b) => a * b);

        public NpyHeader(string descr, bool fortranOrder, long[] shape)
        {
            Descr = descr ?? throw new ArgumentNullException(nameof(descr));
            FortranOrder = fortranOrder;
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
        }

        /// <summary>
        /// Creates header from an NDArray.
        /// </summary>
        public static NpyHeader FromArray(NDArray array)
        {
            string descr = GetDescr(array.dtype, array.typecode);

            // NumSharp only supports C-order; non-contiguous arrays are made contiguous
            bool fortranOrder = false;

            // Use long[] shape directly from NDArray
            long[] shape = array.Shape.IsScalar
                ? Array.Empty<long>()
                : array.Shape.Dimensions;

            return new NpyHeader(descr, fortranOrder, shape);
        }

        /// <summary>
        /// Gets dtype descriptor string for a type.
        /// </summary>
        private static string GetDescr(Type dtype, NPTypeCode typeCode)
        {
            // Byte order: '<' = little-endian, '>' = big-endian, '|' = not applicable
            char byteOrder = BitConverter.IsLittleEndian ? '<' : '>';

            return typeCode switch
            {
                NPTypeCode.Boolean => "|b1",
                NPTypeCode.Byte => "|u1",
                NPTypeCode.Int16 => $"{byteOrder}i2",
                NPTypeCode.UInt16 => $"{byteOrder}u2",
                NPTypeCode.Int32 => $"{byteOrder}i4",
                NPTypeCode.UInt32 => $"{byteOrder}u4",
                NPTypeCode.Int64 => $"{byteOrder}i8",
                NPTypeCode.UInt64 => $"{byteOrder}u8",
                NPTypeCode.Single => $"{byteOrder}f4",
                NPTypeCode.Double => $"{byteOrder}f8",
                NPTypeCode.Char => $"{byteOrder}U1",
                NPTypeCode.Decimal => throw new NotSupportedException("Decimal type not supported in .npy format"),
                _ => throw new NotSupportedException($"Unsupported type: {typeCode}")
            };
        }
    }
}
```

---

## Header Parsing (NpyFormat.Header.cs)

```csharp
namespace NumSharp.IO
{
    public static partial class NpyFormat
    {
        #region Magic String

        /// <summary>
        /// Creates the magic string for a given version.
        /// </summary>
        public static byte[] CreateMagic(byte major, byte minor)
        {
            if (major > 255 || minor > 255)
                throw new ArgumentException("Version numbers must be 0-255");

            var magic = new byte[MagicLength];
            Array.Copy(MagicPrefix, magic, MagicPrefix.Length);
            magic[6] = major;
            magic[7] = minor;
            return magic;
        }

        /// <summary>
        /// Reads and validates the magic string, returns version.
        /// </summary>
        public static async ValueTask<(byte Major, byte Minor)> ReadMagicAsync(
            Stream stream,
            CancellationToken cancellationToken = default)
        {
            var magic = new byte[MagicLength];
            await ReadExactlyAsync(stream, magic, cancellationToken).ConfigureAwait(false);

            for (int i = 0; i < MagicPrefix.Length; i++)
            {
                if (magic[i] != MagicPrefix[i])
                    throw new FormatException(
                        $"Invalid magic string. Expected \\x93NUMPY, got {BitConverter.ToString(magic, 0, 6)}");
            }

            return (magic[6], magic[7]);
        }

        #endregion

        #region Header Reading

        /// <summary>
        /// Reads array header from stream. Returns header and leaves stream positioned at data start.
        /// </summary>
        public static async ValueTask<NpyHeader> ReadHeaderAsync(
            Stream stream,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;
            var version = await ReadMagicAsync(stream, cancellationToken).ConfigureAwait(false);
            return await ReadHeaderAsync(stream, version, options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Reads array header with known version.
        /// </summary>
        public static async ValueTask<NpyHeader> ReadHeaderAsync(
            Stream stream,
            (byte Major, byte Minor) version,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;
            var (sizeBytes, encoding) = GetHeaderInfo(version);

            // Read header length
            var lengthBytes = new byte[sizeBytes];
            await ReadExactlyAsync(stream, lengthBytes, cancellationToken).ConfigureAwait(false);

            int headerLength = sizeBytes == 2
                ? BinaryPrimitives.ReadUInt16LittleEndian(lengthBytes)
                : (int)BinaryPrimitives.ReadUInt32LittleEndian(lengthBytes);

            if (headerLength > options.MaxHeaderSize)
                throw new FormatException(
                    $"Header size ({headerLength}) exceeds maximum ({options.MaxHeaderSize}). " +
                    "Increase MaxHeaderSize in options if you trust this file.");

            // Read and decode header string
            var headerBytes = new byte[headerLength];
            await ReadExactlyAsync(stream, headerBytes, cancellationToken).ConfigureAwait(false);
            string headerStr = encoding.GetString(headerBytes);

            // Parse the Python dictionary literal
            return ParseHeaderDict(headerStr);
        }

        /// <summary>
        /// Parses a Python dictionary literal into NpyHeader.
        /// Example: "{'descr': '<f8', 'fortran_order': False, 'shape': (100, 200), }"
        /// </summary>
        private static NpyHeader ParseHeaderDict(string header)
        {
            // Remove outer braces and whitespace
            header = header.Trim();
            if (!header.StartsWith("{") || !header.EndsWith("}"))
                throw new FormatException($"Header is not a dictionary: {header}");

            string? descr = null;
            bool? fortranOrder = null;
            long[]? shape = null;

            // Parse key-value pairs
            int pos = 1; // Skip opening brace
            while (pos < header.Length - 1)
            {
                // Skip whitespace and commas
                while (pos < header.Length && (char.IsWhiteSpace(header[pos]) || header[pos] == ','))
                    pos++;

                if (pos >= header.Length - 1 || header[pos] == '}')
                    break;

                // Parse key (single-quoted string)
                var (key, nextPos) = ParseQuotedString(header, pos);
                pos = nextPos;

                // Skip colon and whitespace
                while (pos < header.Length && (char.IsWhiteSpace(header[pos]) || header[pos] == ':'))
                    pos++;

                // Parse value based on key
                switch (key)
                {
                    case "descr":
                        (descr, pos) = ParseDescrValue(header, pos);
                        break;
                    case "fortran_order":
                        (fortranOrder, pos) = ParseBoolValue(header, pos);
                        break;
                    case "shape":
                        (shape, pos) = ParseShapeValue(header, pos);
                        break;
                    default:
                        throw new FormatException($"Unexpected key in header: '{key}'");
                }
            }

            // Validate all required keys present
            if (descr == null)
                throw new FormatException("Header missing 'descr' key");
            if (fortranOrder == null)
                throw new FormatException("Header missing 'fortran_order' key");
            if (shape == null)
                throw new FormatException("Header missing 'shape' key");

            return new NpyHeader(descr, fortranOrder.Value, shape);
        }

        private static (string Value, int NextPos) ParseQuotedString(string s, int pos)
        {
            if (s[pos] != '\'')
                throw new FormatException($"Expected single quote at position {pos}");

            int start = pos + 1;
            int end = s.IndexOf('\'', start);
            if (end < 0)
                throw new FormatException("Unterminated string");

            return (s.Substring(start, end - start), end + 1);
        }

        private static (string Value, int NextPos) ParseDescrValue(string s, int pos)
        {
            if (s[pos] == '\'')
            {
                return ParseQuotedString(s, pos);
            }
            else if (s[pos] == '[')
            {
                // Complex structured dtype - find matching bracket
                int depth = 1;
                int start = pos;
                pos++;
                while (pos < s.Length && depth > 0)
                {
                    if (s[pos] == '[') depth++;
                    else if (s[pos] == ']') depth--;
                    pos++;
                }
                return (s.Substring(start, pos - start), pos);
            }
            else
            {
                throw new FormatException($"Unexpected descr format at position {pos}");
            }
        }

        private static (bool Value, int NextPos) ParseBoolValue(string s, int pos)
        {
            if (s.AsSpan(pos).StartsWith("True"))
                return (true, pos + 4);
            if (s.AsSpan(pos).StartsWith("False"))
                return (false, pos + 5);
            throw new FormatException($"Expected True or False at position {pos}");
        }

        /// <summary>
        /// Parses shape tuple. Uses long to support dimensions > int.MaxValue.
        /// Examples: "()", "(10,)", "(100, 200)"
        /// </summary>
        private static (long[] Shape, int NextPos) ParseShapeValue(string s, int pos)
        {
            if (s[pos] != '(')
                throw new FormatException($"Expected '(' for shape at position {pos}");

            int end = s.IndexOf(')', pos);
            if (end < 0)
                throw new FormatException("Unterminated shape tuple");

            string tupleContent = s.Substring(pos + 1, end - pos - 1).Trim();

            if (string.IsNullOrEmpty(tupleContent))
            {
                // Scalar: ()
                return (Array.Empty<long>(), end + 1);
            }

            // Parse comma-separated integers
            var parts = tupleContent.Split(',', StringSplitOptions.RemoveEmptyEntries);
            var shape = new long[parts.Length];

            for (int i = 0; i < parts.Length; i++)
            {
                string part = parts[i].Trim();

                // Handle Python 2 long suffix 'L' (for backwards compatibility)
                if (part.EndsWith("L") || part.EndsWith("l"))
                    part = part.Substring(0, part.Length - 1);

                if (!long.TryParse(part, out shape[i]))
                    throw new FormatException($"Invalid shape dimension: '{parts[i]}'");

                if (shape[i] < 0)
                    throw new FormatException($"Negative dimension not allowed: {shape[i]}");
            }

            return (shape, end + 1);
        }

        #endregion

        #region Header Writing

        /// <summary>
        /// Writes array header to stream. Auto-selects minimum compatible version.
        /// </summary>
        public static async ValueTask WriteHeaderAsync(
            Stream stream,
            NpyHeader header,
            (byte, byte)? version = null,
            CancellationToken cancellationToken = default)
        {
            string headerStr = FormatHeaderDict(header);
            byte[] wrappedHeader = WrapHeader(headerStr, version);
            await stream.WriteAsync(wrappedHeader, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Formats header as Python dictionary literal.
        /// Keys are sorted alphabetically (NumPy convention for reproducibility).
        /// </summary>
        private static string FormatHeaderDict(NpyHeader header)
        {
            var sb = new StringBuilder(128);
            sb.Append('{');

            // descr
            sb.Append("'descr': ");
            if (header.Descr.StartsWith("["))
            {
                sb.Append(header.Descr);
            }
            else
            {
                sb.Append('\'');
                sb.Append(header.Descr);
                sb.Append('\'');
            }
            sb.Append(", ");

            // fortran_order
            sb.Append("'fortran_order': ");
            sb.Append(header.FortranOrder ? "True" : "False");
            sb.Append(", ");

            // shape
            sb.Append("'shape': (");
            for (int i = 0; i < header.Shape.Length; i++)
            {
                sb.Append(header.Shape[i]);
                sb.Append(", ");
            }
            sb.Append("), ");
            sb.Append('}');

            // Add padding space for potential in-place growth
            if (header.Shape.Length > 0)
            {
                long lastDim = header.FortranOrder ? header.Shape[0] : header.Shape[^1];
                int currentDigits = lastDim.ToString().Length;
                int paddingNeeded = GrowthAxisMaxDigits - currentDigits;
                if (paddingNeeded > 0)
                    sb.Append(' ', paddingNeeded);
            }

            return sb.ToString();
        }

        private static byte[] WrapHeader(string header, (byte Major, byte Minor)? version)
        {
            if (version == null)
            {
                // Try version 1.0 first (most compatible)
                try { return WrapHeaderWithVersion(header, (1, 0)); }
                catch (InvalidOperationException) { }

                // Try version 2.0
                try { return WrapHeaderWithVersion(header, (2, 0)); }
                catch (EncoderFallbackException) { }

                // Fall back to version 3.0
                return WrapHeaderWithVersion(header, (3, 0));
            }
            else
            {
                return WrapHeaderWithVersion(header, version.Value);
            }
        }

        private static byte[] WrapHeaderWithVersion(string header, (byte Major, byte Minor) version)
        {
            var (sizeBytes, encoding) = GetHeaderInfo(version);

            byte[] headerBytes = encoding.GetBytes(header);
            int headerLen = headerBytes.Length + 1; // +1 for newline

            int preambleLen = MagicLength + sizeBytes;
            int totalBeforePad = preambleLen + headerLen;
            int padLen = (ArrayAlign - (totalBeforePad % ArrayAlign)) % ArrayAlign;
            int totalHeaderLen = headerLen + padLen;

            int maxLen = sizeBytes == 2 ? ushort.MaxValue : int.MaxValue;
            if (totalHeaderLen > maxLen)
                throw new InvalidOperationException($"Header length {totalHeaderLen} exceeds max {maxLen} for version {version}");

            var result = new byte[preambleLen + totalHeaderLen];
            int pos = 0;

            // Magic + version
            var magic = CreateMagic(version.Major, version.Minor);
            Array.Copy(magic, 0, result, pos, magic.Length);
            pos += magic.Length;

            // Header length
            if (sizeBytes == 2)
                BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(pos), (ushort)totalHeaderLen);
            else
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(pos), (uint)totalHeaderLen);
            pos += sizeBytes;

            // Header content
            Array.Copy(headerBytes, 0, result, pos, headerBytes.Length);
            pos += headerBytes.Length;

            // Padding (spaces)
            result.AsSpan(pos, padLen).Fill((byte)' ');
            pos += padLen;

            // Newline
            result[pos] = (byte)'\n';

            return result;
        }

        #endregion

        #region Async Utilities

        /// <summary>
        /// Reads exactly the specified number of bytes, throwing on EOF.
        /// </summary>
        private static async ValueTask ReadExactlyAsync(
            Stream stream,
            Memory<byte> buffer,
            CancellationToken cancellationToken)
        {
            int totalRead = 0;
            while (totalRead < buffer.Length)
            {
                int read = await stream.ReadAsync(buffer.Slice(totalRead), cancellationToken).ConfigureAwait(false);
                if (read == 0)
                    throw new EndOfStreamException(
                        $"Unexpected EOF: expected {buffer.Length} bytes, got {totalRead}");
                totalRead += read;
            }
        }

        #endregion
    }
}
```

---

## Array Writing (NpyFormat.Write.cs)

```csharp
namespace NumSharp.IO
{
    public static partial class NpyFormat
    {
        #region Public Write API

        /// <summary>
        /// Writes an NDArray to a .npy file asynchronously.
        /// </summary>
        public static async ValueTask WriteArrayAsync(
            string path,
            NDArray array,
            NpyOptions? options = null,
            (byte, byte)? version = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            await using var stream = new FileStream(
                path,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: options.WriteBufferSize,
                options.FileOptions);

            await WriteArrayAsync(stream, array, options, version, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Writes an NDArray to a stream in .npy format asynchronously.
        /// </summary>
        public static async ValueTask WriteArrayAsync(
            Stream stream,
            NDArray array,
            NpyOptions? options = null,
            (byte, byte)? version = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            // Create and write header
            var header = NpyHeader.FromArray(array);
            await WriteHeaderAsync(stream, header, version, cancellationToken).ConfigureAwait(false);

            // Write data
            await WriteArrayDataAsync(stream, array, options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Writes an NDArray to a byte array.
        /// </summary>
        public static async ValueTask<byte[]> WriteArrayToBytesAsync(
            NDArray array,
            NpyOptions? options = null,
            (byte, byte)? version = null,
            CancellationToken cancellationToken = default)
        {
            using var stream = new MemoryStream();
            await WriteArrayAsync(stream, array, options, version, cancellationToken).ConfigureAwait(false);
            return stream.ToArray();
        }

        /// <summary>
        /// Synchronous wrapper for simple usage. Prefer async version for large arrays.
        /// </summary>
        public static void WriteArray(string path, NDArray array, NpyOptions? options = null)
        {
            WriteArrayAsync(path, array, options).AsTask().GetAwaiter().GetResult();
        }

        #endregion

        #region Data Writing

        private static async ValueTask WriteArrayDataAsync(
            Stream stream,
            NDArray array,
            NpyOptions options,
            CancellationToken cancellationToken)
        {
            if (array.size == 0)
                return;

            long totalBytes = array.size * array.dtypesize;
            long bytesWritten = 0;

            // Choose sync vs async based on size threshold
            bool useAsync = options.UseAsyncIO && totalBytes >= options.AsyncThreshold;

            if (array.Shape.IsContiguous)
            {
                await WriteContiguousDataAsync(stream, array, options, useAsync,
                    b => ReportProgress(options, bytesWritten += b, totalBytes),
                    cancellationToken).ConfigureAwait(false);
            }
            else
            {
                await WriteNonContiguousDataAsync(stream, array, options, useAsync,
                    b => ReportProgress(options, bytesWritten += b, totalBytes),
                    cancellationToken).ConfigureAwait(false);
            }
        }

        private static async ValueTask WriteContiguousDataAsync(
            Stream stream,
            NDArray array,
            NpyOptions options,
            bool useAsync,
            Action<long> onBytesWritten,
            CancellationToken cancellationToken)
        {
            long totalBytes = array.size * array.dtypesize;
            int bufferSize = options.WriteBufferSize;

            unsafe
            {
                byte* ptr = (byte*)array.Address;
                ptr += array.Shape.offset * array.dtypesize;

                long remaining = totalBytes;

                // Rent a buffer for chunked writing
                byte[] buffer = ArrayPool<byte>.Shared.Rent(bufferSize);
                try
                {
                    while (remaining > 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int chunkSize = (int)Math.Min(remaining, bufferSize);

                        // Copy from unmanaged to managed buffer
                        new ReadOnlySpan<byte>(ptr, chunkSize).CopyTo(buffer);

                        if (useAsync)
                        {
                            await stream.WriteAsync(buffer.AsMemory(0, chunkSize), cancellationToken)
                                .ConfigureAwait(false);
                        }
                        else
                        {
                            stream.Write(buffer, 0, chunkSize);
                        }

                        ptr += chunkSize;
                        remaining -= chunkSize;
                        onBytesWritten(chunkSize);
                    }
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(buffer);
                }
            }
        }

        private static async ValueTask WriteNonContiguousDataAsync(
            Stream stream,
            NDArray array,
            NpyOptions options,
            bool useAsync,
            Action<long> onBytesWritten,
            CancellationToken cancellationToken)
        {
            int itemSize = array.dtypesize;
            int bufferSize = options.WriteBufferSize;
            int elementsPerBuffer = Math.Max(1, bufferSize / itemSize);

            byte[] buffer = ArrayPool<byte>.Shared.Rent(elementsPerBuffer * itemSize);
            try
            {
                int bufferPos = 0;

                using var iter = new NDIterator(array);
                while (iter.HasNext())
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    // Copy element to buffer
                    var elementSpan = iter.Current;
                    elementSpan.CopyTo(buffer.AsSpan(bufferPos, itemSize));
                    bufferPos += itemSize;

                    // Flush buffer when full
                    if (bufferPos >= buffer.Length)
                    {
                        if (useAsync)
                        {
                            await stream.WriteAsync(buffer.AsMemory(0, bufferPos), cancellationToken)
                                .ConfigureAwait(false);
                        }
                        else
                        {
                            stream.Write(buffer, 0, bufferPos);
                        }
                        onBytesWritten(bufferPos);
                        bufferPos = 0;
                    }

                    iter.MoveNext();
                }

                // Flush remaining data
                if (bufferPos > 0)
                {
                    if (useAsync)
                    {
                        await stream.WriteAsync(buffer.AsMemory(0, bufferPos), cancellationToken)
                            .ConfigureAwait(false);
                    }
                    else
                    {
                        stream.Write(buffer, 0, bufferPos);
                    }
                    onBytesWritten(bufferPos);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        private static void ReportProgress(NpyOptions options, long bytesProcessed, long totalBytes)
        {
            options.Progress?.Report((bytesProcessed, totalBytes));
        }

        #endregion
    }
}
```

---

## Array Reading (NpyFormat.Read.cs)

```csharp
namespace NumSharp.IO
{
    public static partial class NpyFormat
    {
        #region Public Read API

        /// <summary>
        /// Reads an NDArray from a .npy file asynchronously.
        /// </summary>
        public static async ValueTask<NDArray> ReadArrayAsync(
            string path,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            await using var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: options.ReadBufferSize,
                options.FileOptions);

            return await ReadArrayAsync(stream, options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Reads an NDArray from a stream in .npy format asynchronously.
        /// </summary>
        public static async ValueTask<NDArray> ReadArrayAsync(
            Stream stream,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            // Read header
            var header = await ReadHeaderAsync(stream, options, cancellationToken).ConfigureAwait(false);

            // Get dtype from descriptor
            var (dtype, typeCode) = ParseDescr(header.Descr);

            // Create array
            var array = new NDArray(typeCode, new Shape(header.Shape));

            // Read data
            await ReadArrayDataAsync(stream, array, header.FortranOrder, options, cancellationToken)
                .ConfigureAwait(false);

            return array;
        }

        /// <summary>
        /// Reads just the header from a .npy file (for inspection without loading data).
        /// </summary>
        public static async ValueTask<NpyHeader> ReadHeaderOnlyAsync(
            string path,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            return await ReadHeaderAsync(stream, options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Synchronous wrapper for simple usage. Prefer async version for large arrays.
        /// </summary>
        public static NDArray ReadArray(string path, NpyOptions? options = null)
        {
            return ReadArrayAsync(path, options).AsTask().GetAwaiter().GetResult();
        }

        #endregion

        #region Dtype Parsing

        private static (Type dtype, NPTypeCode typeCode) ParseDescr(string descr)
        {
            // Handle byte order prefix
            bool needsSwap = false;
            if (descr.Length > 0)
            {
                char byteOrder = descr[0];
                switch (byteOrder)
                {
                    case '<':
                        needsSwap = !BitConverter.IsLittleEndian;
                        descr = descr.Substring(1);
                        break;
                    case '>':
                        needsSwap = BitConverter.IsLittleEndian;
                        descr = descr.Substring(1);
                        break;
                    case '|':
                    case '=':
                        descr = descr.Substring(1);
                        break;
                }
            }

            if (needsSwap)
                throw new NotSupportedException("Cross-endian .npy files not yet supported");

            return descr switch
            {
                "b1" => (typeof(bool), NPTypeCode.Boolean),
                "u1" => (typeof(byte), NPTypeCode.Byte),
                "i1" => (typeof(sbyte), NPTypeCode.Byte),
                "i2" => (typeof(short), NPTypeCode.Int16),
                "u2" => (typeof(ushort), NPTypeCode.UInt16),
                "i4" => (typeof(int), NPTypeCode.Int32),
                "u4" => (typeof(uint), NPTypeCode.UInt32),
                "i8" => (typeof(long), NPTypeCode.Int64),
                "u8" => (typeof(ulong), NPTypeCode.UInt64),
                "f4" => (typeof(float), NPTypeCode.Single),
                "f8" => (typeof(double), NPTypeCode.Double),
                "U1" => (typeof(char), NPTypeCode.Char),
                "f2" => throw new NotSupportedException("float16 not supported"),
                "c8" => throw new NotSupportedException("complex64 not supported"),
                "c16" => throw new NotSupportedException("complex128 not supported"),
                _ when descr.StartsWith("S") => throw new NotSupportedException("Fixed-length byte strings not supported"),
                _ when descr.StartsWith("U") => throw new NotSupportedException("Fixed-length unicode strings not supported"),
                _ => throw new NotSupportedException($"Unsupported dtype: {descr}")
            };
        }

        #endregion

        #region Data Reading

        private static async ValueTask ReadArrayDataAsync(
            Stream stream,
            NDArray array,
            bool fortranOrder,
            NpyOptions options,
            CancellationToken cancellationToken)
        {
            if (array.size == 0)
                return;

            long totalBytes = array.size * array.dtypesize;
            long bytesRead = 0;

            bool useAsync = options.UseAsyncIO && totalBytes >= options.AsyncThreshold;

            await ReadDataChunkedAsync(stream, array, options, useAsync,
                b => ReportProgress(options, bytesRead += b, totalBytes),
                cancellationToken).ConfigureAwait(false);

            // Handle Fortran order by transposing
            if (fortranOrder && array.ndim > 1)
            {
                // Note: Proper F-order support would read into reversed shape then transpose.
                // For now, we only support C-order arrays.
                throw new NotSupportedException("Fortran-order arrays not yet supported");
            }
        }

        private static async ValueTask ReadDataChunkedAsync(
            Stream stream,
            NDArray array,
            NpyOptions options,
            bool useAsync,
            Action<long> onBytesRead,
            CancellationToken cancellationToken)
        {
            long totalBytes = array.size * array.dtypesize;
            int bufferSize = options.ReadBufferSize;

            byte[] buffer = ArrayPool<byte>.Shared.Rent(bufferSize);
            try
            {
                unsafe
                {
                    byte* ptr = (byte*)array.Address;
                    long remaining = totalBytes;

                    while (remaining > 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        int toRead = (int)Math.Min(remaining, bufferSize);
                        int totalRead = 0;

                        // Read chunk (may require multiple reads for streams like GZipStream)
                        while (totalRead < toRead)
                        {
                            int read;
                            if (useAsync)
                            {
                                read = await stream.ReadAsync(
                                    buffer.AsMemory(totalRead, toRead - totalRead),
                                    cancellationToken).ConfigureAwait(false);
                            }
                            else
                            {
                                read = stream.Read(buffer, totalRead, toRead - totalRead);
                            }

                            if (read == 0)
                                throw new EndOfStreamException(
                                    $"Unexpected EOF: expected {totalBytes} bytes, got {totalBytes - remaining + totalRead}");

                            totalRead += read;
                        }

                        // Copy to unmanaged memory
                        new ReadOnlySpan<byte>(buffer, 0, totalRead).CopyTo(new Span<byte>(ptr, totalRead));

                        ptr += totalRead;
                        remaining -= totalRead;
                        onBytesRead(totalRead);
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        #endregion
    }
}
```

---

## NPZ Archive Support (NpzArchive.cs)

```csharp
namespace NumSharp.IO
{
    /// <summary>
    /// Handles .npz files (zip archives containing multiple .npy arrays).
    /// </summary>
    public sealed class NpzArchive : IAsyncDisposable, IDisposable
    {
        private readonly ZipArchive _archive;
        private readonly Stream _stream;
        private readonly bool _ownsStream;
        private readonly NpyOptions _options;

        #region Constructors

        private NpzArchive(ZipArchive archive, Stream stream, bool ownsStream, NpyOptions options)
        {
            _archive = archive;
            _stream = stream;
            _ownsStream = ownsStream;
            _options = options;
        }

        /// <summary>
        /// Opens an .npz file for reading.
        /// </summary>
        public static async ValueTask<NpzArchive> OpenAsync(
            string path,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            var stream = new FileStream(
                path,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: options.ReadBufferSize,
                options.FileOptions);

            var archive = new ZipArchive(stream, ZipArchiveMode.Read);
            return new NpzArchive(archive, stream, ownsStream: true, options);
        }

        /// <summary>
        /// Opens an .npz archive from a stream.
        /// </summary>
        public static NpzArchive Open(Stream stream, bool leaveOpen = false, NpyOptions? options = null)
        {
            options ??= NpyOptions.Default;
            var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen);
            return new NpzArchive(archive, stream, ownsStream: !leaveOpen, options);
        }

        #endregion

        #region Reading

        /// <summary>
        /// Gets names of all arrays in the archive.
        /// </summary>
        public IEnumerable<string> keys => _archive.Entries
            .Select(e => e.Name.EndsWith(".npy") ? e.Name[..^4] : e.Name);

        /// <summary>
        /// Gets an array by name asynchronously.
        /// </summary>
        public async ValueTask<NDArray> get_async(
            string name,
            CancellationToken cancellationToken = default)
        {
            var entry = _archive.GetEntry(name) ?? _archive.GetEntry(name + ".npy");
            if (entry == null)
                throw new KeyNotFoundException($"Array '{name}' not found in archive");

            await using var stream = entry.Open();
            return await NpyFormat.ReadArrayAsync(stream, _options, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Gets an array by name (synchronous).
        /// </summary>
        public NDArray this[string name] => get_async(name).AsTask().GetAwaiter().GetResult();

        /// <summary>
        /// Checks if an array exists.
        /// </summary>
        public bool contains_key(string name)
        {
            return _archive.GetEntry(name) != null || _archive.GetEntry(name + ".npy") != null;
        }

        /// <summary>
        /// Loads all arrays into a dictionary (async).
        /// </summary>
        public async ValueTask<Dictionary<string, NDArray>> to_dict_async(
            CancellationToken cancellationToken = default)
        {
            var result = new Dictionary<string, NDArray>();
            foreach (var key in keys)
            {
                result[key] = await get_async(key, cancellationToken).ConfigureAwait(false);
            }
            return result;
        }

        /// <summary>
        /// Loads all arrays into a dictionary (sync).
        /// </summary>
        public Dictionary<string, NDArray> to_dict()
        {
            return to_dict_async().AsTask().GetAwaiter().GetResult();
        }

        #endregion

        #region Static Creation Methods

        /// <summary>
        /// Creates an .npz file from multiple arrays asynchronously.
        /// </summary>
        public static async ValueTask SaveAsync(
            string path,
            Dictionary<string, NDArray> arrays,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            await using var stream = new FileStream(
                path,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: options.WriteBufferSize,
                options.FileOptions);

            await SaveAsync(stream, arrays, options, leaveOpen: false, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Creates an .npz archive in a stream asynchronously.
        /// </summary>
        public static async ValueTask SaveAsync(
            Stream stream,
            Dictionary<string, NDArray> arrays,
            NpyOptions? options = null,
            bool leaveOpen = false,
            CancellationToken cancellationToken = default)
        {
            options ??= NpyOptions.Default;

            using var archive = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen);

            foreach (var (name, array) in arrays)
            {
                cancellationToken.ThrowIfCancellationRequested();

                string entryName = name.EndsWith(".npy") ? name : name + ".npy";
                var entry = archive.CreateEntry(entryName, options.CompressionLevel);

                await using var entryStream = entry.Open();
                await NpyFormat.WriteArrayAsync(entryStream, array, options, cancellationToken: cancellationToken)
                    .ConfigureAwait(false);
            }
        }

        /// <summary>
        /// Creates an .npz file from a single array (stored as "arr_0").
        /// </summary>
        public static ValueTask SaveAsync(
            string path,
            NDArray array,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            return SaveAsync(path, new Dictionary<string, NDArray> { ["arr_0"] = array }, options, cancellationToken);
        }

        #endregion

        #region IDisposable / IAsyncDisposable

        public void Dispose()
        {
            _archive.Dispose();
            if (_ownsStream)
                _stream.Dispose();
        }

        public async ValueTask DisposeAsync()
        {
            _archive.Dispose();
            if (_ownsStream)
                await _stream.DisposeAsync().ConfigureAwait(false);
        }

        #endregion
    }
}
```

---

## Updated np.save / np.load APIs

**Naming Convention:** Public `np.*` async methods use snake_case (`save_async`, `load_async`) to match NumPy's naming style. Internal `NpyFormat.*` methods use C# convention (`WriteArrayAsync`, `ReadArrayAsync`).

```csharp
// np.save.cs
namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Save an array to a binary file in NumPy .npy format (async).
        /// </summary>
        public static ValueTask save_async(
            string filepath,
            NDArray arr,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            string path = filepath.EndsWith(".npy") ? filepath : filepath + ".npy";
            return IO.NpyFormat.WriteArrayAsync(path, arr, options, cancellationToken: cancellationToken);
        }

        /// <summary>
        /// Save an array to a binary file in NumPy .npy format (sync).
        /// </summary>
        public static void save(string filepath, NDArray arr, NpyOptions? options = null)
        {
            save_async(filepath, arr, options).AsTask().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Save multiple arrays to a .npz file (async).
        /// </summary>
        public static ValueTask savez_async(
            string filepath,
            Dictionary<string, NDArray> arrays,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            string path = filepath.EndsWith(".npz") ? filepath : filepath + ".npz";
            return IO.NpzArchive.SaveAsync(path, arrays, options, cancellationToken);
        }

        /// <summary>
        /// Save multiple arrays to a .npz file (sync).
        /// </summary>
        public static void savez(string filepath, Dictionary<string, NDArray> arrays, NpyOptions? options = null)
        {
            savez_async(filepath, arrays, options).AsTask().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Save multiple arrays to a .npz file with positional names (arr_0, arr_1, ...).
        /// </summary>
        public static ValueTask savez_async(
            string filepath,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default,
            params NDArray[] arrays)
        {
            var dict = new Dictionary<string, NDArray>();
            for (int i = 0; i < arrays.Length; i++)
                dict[$"arr_{i}"] = arrays[i];
            return savez_async(filepath, dict, options, cancellationToken);
        }

        /// <summary>
        /// Save multiple arrays to a compressed .npz file (async).
        /// </summary>
        public static ValueTask savez_compressed_async(
            string filepath,
            Dictionary<string, NDArray> arrays,
            CancellationToken cancellationToken = default)
        {
            var options = new NpyOptions { CompressionLevel = CompressionLevel.Optimal };
            return savez_async(filepath, arrays, options, cancellationToken);
        }

        /// <summary>
        /// Save multiple arrays to a compressed .npz file (sync).
        /// </summary>
        public static void savez_compressed(string filepath, Dictionary<string, NDArray> arrays)
        {
            savez_compressed_async(filepath, arrays).AsTask().GetAwaiter().GetResult();
        }
    }
}

// np.load.cs
namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Load an array from a .npy file (async).
        /// </summary>
        public static ValueTask<NDArray> load_async(
            string filepath,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            if (filepath.EndsWith(".npz"))
            {
                return load_first_from_npz_async(filepath, options, cancellationToken);
            }

            string path = filepath.EndsWith(".npy") ? filepath : filepath + ".npy";
            return IO.NpyFormat.ReadArrayAsync(path, options, cancellationToken);
        }

        private static async ValueTask<NDArray> load_first_from_npz_async(
            string path,
            NpyOptions? options,
            CancellationToken cancellationToken)
        {
            await using var archive = await IO.NpzArchive.OpenAsync(path, options, cancellationToken)
                .ConfigureAwait(false);
            return await archive.GetAsync(archive.Keys.First(), cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Load an array from a .npy file (sync).
        /// </summary>
        public static NDArray load(string filepath, NpyOptions? options = null)
        {
            return load_async(filepath, options).AsTask().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Load a .npz archive (async).
        /// </summary>
        public static ValueTask<IO.NpzArchive> load_npz_async(
            string filepath,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            return IO.NpzArchive.OpenAsync(filepath, options, cancellationToken);
        }

        /// <summary>
        /// Load a .npz archive (sync).
        /// </summary>
        public static IO.NpzArchive load_npz(string filepath, NpyOptions? options = null)
        {
            return load_npz_async(filepath, options).AsTask().GetAwaiter().GetResult();
        }

        /// <summary>
        /// Read just the header from a .npy file without loading data.
        /// Useful for inspecting array metadata (async).
        /// </summary>
        public static ValueTask<IO.NpyHeader> load_header_async(
            string filepath,
            NpyOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            return IO.NpyFormat.ReadHeaderOnlyAsync(filepath, options, cancellationToken);
        }

        /// <summary>
        /// Read just the header from a .npy file without loading data (sync).
        /// </summary>
        public static IO.NpyHeader load_header(string filepath, NpyOptions? options = null)
        {
            return load_header_async(filepath, options).AsTask().GetAwaiter().GetResult();
        }
    }
}
```

---

## Usage Examples

### Basic Usage

```csharp
// Simple save/load (sync - backward compatible)
np.save("data.npy", array);
var loaded = np.load("data.npy");

// Async with cancellation
using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5));
await np.save_async("data.npy", array, cancellationToken: cts.Token);
var loaded = await np.load_async("data.npy", cancellationToken: cts.Token);
```

### With Progress Reporting

```csharp
var progress = new Progress<(long BytesProcessed, long TotalBytes)>(p =>
{
    double percent = 100.0 * p.BytesProcessed / p.TotalBytes;
    Console.WriteLine($"Progress: {percent:F1}%");
});

var options = NpyOptions.WithProgress(progress);
await np.save_async("large_array.npy", array, options);
```

### Large File Optimization

```csharp
// For arrays > 100 MB
var options = NpyOptions.ForLargeFiles();
await np.save_async("huge_array.npy", array, options);

// Custom buffer sizes
var options = new NpyOptions
{
    ReadBufferSize = 4 * 1024 * 1024,   // 4 MiB read buffer
    WriteBufferSize = 32 * 1024 * 1024, // 32 MiB write buffer
};
```

### NPZ Archives

```csharp
// Save multiple arrays
var arrays = new Dictionary<string, NDArray>
{
    ["weights"] = weights,
    ["biases"] = biases,
    ["config"] = config
};
await np.savez_async("model.npz", arrays, cancellationToken: ct);

// Load from archive
await using var archive = await np.load_npz_async("model.npz");
var weights = await archive.get_async("weights", ct);
var biases = await archive.get_async("biases", ct);

// Or load all at once
var allArrays = await archive.to_dict_async(ct);
```

### Inspect Without Loading

```csharp
// Just read the header to check shape/dtype
var header = await np.load_header_async("large_file.npy");
Console.WriteLine($"Shape: [{string.Join(", ", header.Shape)}]");
Console.WriteLine($"Dtype: {header.Descr}");
Console.WriteLine($"Elements: {header.Count:N0}");
```

---

## Key Design Decisions

### 1. ValueTask for Hot Path

All async methods return `ValueTask` or `ValueTask<T>` for:
- Zero allocation when completing synchronously
- Optimal performance for small files that complete immediately
- Still fully async-capable for large files

### 2. ArrayPool<byte> for Buffers

```csharp
byte[] buffer = ArrayPool<byte>.Shared.Rent(bufferSize);
try { /* use buffer */ }
finally { ArrayPool<byte>.Shared.Return(buffer); }
```

Eliminates buffer allocations, reducing GC pressure for repeated operations.

### 3. Adaptive Sync/Async

```csharp
bool useAsync = options.UseAsyncIO && totalBytes >= options.AsyncThreshold;
```

Small arrays use sync I/O to avoid async overhead. Threshold configurable via `NpyOptions.AsyncThreshold`.

### 4. CancellationToken Throughout

Every async method accepts `CancellationToken`. Checked at each chunk boundary for responsive cancellation without checking on every byte.

### 5. Progress Reporting

Optional `IProgress<(long, long)>` for UI integration. Reports after each buffer chunk, not every byte.

### 6. FileOptions Optimization

```csharp
FileOptions = FileOptions.Asynchronous | FileOptions.SequentialScan
```

- `Asynchronous`: Enables true async I/O at OS level
- `SequentialScan`: Hints to OS for sequential access optimization

---

## Migration Checklist

1. [ ] Create `src/NumSharp.Core/IO/` directory
2. [ ] Implement `NpyOptions.cs`
3. [ ] Implement `NpyFormat.Constants.cs`
4. [ ] Implement `NpyFormat.Header.cs`
5. [ ] Implement `NpyFormat.Write.cs`
6. [ ] Implement `NpyFormat.Read.cs`
7. [ ] Implement `NpzArchive.cs`
8. [ ] Update `np.save.cs` with async API
9. [ ] Update `np.load.cs` with async API
10. [ ] Add unit tests:
    - [ ] Round-trip: write then read
    - [ ] Cancellation mid-operation
    - [ ] Progress reporting accuracy
    - [ ] Shape values > int.MaxValue
    - [ ] All dtypes
    - [ ] Version 1.0/2.0/3.0 headers
    - [ ] Buffer size variations
    - [ ] .npz archives
11. [ ] Performance benchmarks vs old implementation
12. [ ] Update CLAUDE.md documentation
