using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace NumSharp.IO
{
    /// <summary>
    /// NumPy .npy binary format implementation.
    /// Supports format versions 1.0, 2.0, and 3.0.
    /// </summary>
    /// <remarks>
    /// Based on numpy/lib/_format_impl.py (NumPy 2.x).
    ///
    /// Binary structure:
    /// - Magic string: \x93NUMPY (6 bytes)
    /// - Version: major, minor (2 bytes)
    /// - Header length: uint16 (v1.0) or uint32 (v2.0/v3.0), little-endian
    /// - Header: Python dict literal, padded to 64-byte alignment
    /// - Data: raw bytes in C-order or F-order
    /// </remarks>
    public static class NpyFormat
    {
        #region Constants

        /// <summary>Magic string prefix: \x93NUMPY</summary>
        public static readonly byte[] MagicPrefix = { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

        /// <summary>Length of magic string + version bytes</summary>
        public const int MagicLen = 8;

        /// <summary>Header alignment for memory-mapping compatibility</summary>
        public const int ArrayAlign = 64;

        /// <summary>Buffer size for chunked I/O (256 KB)</summary>
        public const int BufferSize = 262144;

        /// <summary>Maximum digits for growth axis padding</summary>
        public const int GrowthAxisMaxDigits = 21;

        /// <summary>Default maximum header size for security</summary>
        public const int MaxHeaderSize = 10000;

        /// <summary>Required header keys</summary>
        private static readonly HashSet<string> ExpectedKeys = new() { "descr", "fortran_order", "shape" };

        #endregion

        #region Format Version

        /// <summary>
        /// Format version information.
        /// </summary>
        public readonly struct FormatVersion : IEquatable<FormatVersion>
        {
            public readonly byte Major;
            public readonly byte Minor;

            public FormatVersion(byte major, byte minor)
            {
                Major = major;
                Minor = minor;
            }

            public static FormatVersion V1_0 => new(1, 0);
            public static FormatVersion V2_0 => new(2, 0);
            public static FormatVersion V3_0 => new(3, 0);

            /// <summary>Header length field size in bytes</summary>
            public int HeaderLengthSize => Major >= 2 ? 4 : 2;

            /// <summary>Header encoding</summary>
            public Encoding HeaderEncoding => Major >= 3 ? Encoding.UTF8 : Encoding.Latin1;

            /// <summary>Maximum header length</summary>
            public uint MaxHeaderLength => Major >= 2 ? uint.MaxValue : ushort.MaxValue;

            public bool Equals(FormatVersion other) => Major == other.Major && Minor == other.Minor;
            public override bool Equals(object? obj) => obj is FormatVersion v && Equals(v);
            public override int GetHashCode() => (Major << 8) | Minor;
            public override string ToString() => $"({Major}, {Minor})";

            public static bool operator ==(FormatVersion left, FormatVersion right) => left.Equals(right);
            public static bool operator !=(FormatVersion left, FormatVersion right) => !left.Equals(right);
            public static bool operator <=(FormatVersion left, FormatVersion right) =>
                left.Major < right.Major || (left.Major == right.Major && left.Minor <= right.Minor);
            public static bool operator >=(FormatVersion left, FormatVersion right) =>
                left.Major > right.Major || (left.Major == right.Major && left.Minor >= right.Minor);
        }

        #endregion

        #region Header Data

        /// <summary>
        /// Parsed header information from .npy file.
        /// </summary>
        public readonly struct HeaderData
        {
            public readonly long[] Shape;
            public readonly bool FortranOrder;
            public readonly NPTypeCode TypeCode;
            public readonly int ItemSize;
            public readonly bool LittleEndian;

            public HeaderData(long[] shape, bool fortranOrder, NPTypeCode typeCode, int itemSize, bool littleEndian)
            {
                Shape = shape;
                FortranOrder = fortranOrder;
                TypeCode = typeCode;
                ItemSize = itemSize;
                LittleEndian = littleEndian;
            }

            /// <summary>Total number of elements</summary>
            public long Count
            {
                get
                {
                    if (Shape.Length == 0) return 1; // Scalar
                    long count = 1;
                    for (int i = 0; i < Shape.Length; i++)
                        count *= Shape[i];
                    return count;
                }
            }

            /// <summary>Total data size in bytes</summary>
            public long DataSize => Count * ItemSize;
        }

        #endregion

        #region Magic & Version

        /// <summary>
        /// Create magic bytes for the given format version.
        /// </summary>
        public static byte[] Magic(FormatVersion version)
        {
            var magic = new byte[MagicLen];
            Array.Copy(MagicPrefix, magic, MagicPrefix.Length);
            magic[6] = version.Major;
            magic[7] = version.Minor;
            return magic;
        }

        /// <summary>
        /// Read and validate magic string, returning format version.
        /// </summary>
        public static FormatVersion ReadMagic(Stream stream)
        {
            var magic = ReadBytes(stream, MagicLen, "magic string");

            for (int i = 0; i < MagicPrefix.Length; i++)
            {
                if (magic[i] != MagicPrefix[i])
                {
                    throw new FormatException(
                        $"Invalid magic string. Expected \\x93NUMPY, got: " +
                        $"\\x{magic[0]:X2}{(char)magic[1]}{(char)magic[2]}{(char)magic[3]}{(char)magic[4]}{(char)magic[5]}");
                }
            }

            return new FormatVersion(magic[6], magic[7]);
        }

        /// <summary>
        /// Validate format version.
        /// </summary>
        public static void CheckVersion(FormatVersion version)
        {
            if (version != FormatVersion.V1_0 &&
                version != FormatVersion.V2_0 &&
                version != FormatVersion.V3_0)
            {
                throw new NotSupportedException(
                    $"Only format versions (1,0), (2,0), and (3,0) are supported, not {version}");
            }
        }

        #endregion

        #region dtype Conversion

        /// <summary>
        /// Convert NPTypeCode to NumPy dtype descriptor string.
        /// </summary>
        public static string TypeCodeToDescr(NPTypeCode typeCode)
        {
            // Use little-endian for multi-byte types, | for single-byte
            return typeCode switch
            {
                NPTypeCode.Boolean => "|b1",
                NPTypeCode.Byte => "|u1",
                NPTypeCode.Int16 => "<i2",
                NPTypeCode.UInt16 => "<u2",
                NPTypeCode.Int32 => "<i4",
                NPTypeCode.UInt32 => "<u4",
                NPTypeCode.Int64 => "<i8",
                NPTypeCode.UInt64 => "<u8",
                NPTypeCode.Single => "<f4",
                NPTypeCode.Double => "<f8",
                NPTypeCode.Char => "<U1", // Unicode char as 4-byte UTF-32
                NPTypeCode.Decimal => throw new NotSupportedException(
                    "Decimal type is not supported by NumPy format. Convert to Double first."),
                _ => throw new NotSupportedException($"Unsupported type: {typeCode}")
            };
        }

        /// <summary>
        /// Parse NumPy dtype descriptor string to type information.
        /// </summary>
        /// <returns>Tuple of (NPTypeCode, itemSize, isLittleEndian)</returns>
        public static (NPTypeCode typeCode, int itemSize, bool littleEndian) DescrToTypeCode(string descr)
        {
            if (string.IsNullOrEmpty(descr) || descr.Length < 2)
                throw new FormatException($"Invalid dtype descriptor: '{descr}'");

            // Parse endianness
            char endianChar = descr[0];
            bool littleEndian = endianChar switch
            {
                '<' => true,
                '>' => false,
                '|' => BitConverter.IsLittleEndian, // Not applicable, use native
                '=' => BitConverter.IsLittleEndian, // Native
                _ => throw new FormatException($"Invalid endian character: '{endianChar}'")
            };

            // Parse type code and size
            string typeStr = descr.Substring(1);
            char typeChar = typeStr[0];

            // Handle string types specially
            if (typeChar == 'S' || typeChar == 'U' || typeChar == 'V')
            {
                int size = typeStr.Length > 1 ? int.Parse(typeStr.Substring(1)) : 1;
                return typeChar switch
                {
                    'S' => (NPTypeCode.Byte, size, littleEndian), // Byte string
                    'U' => (NPTypeCode.Char, size * 4, littleEndian), // Unicode (4 bytes per char)
                    'V' => (NPTypeCode.Byte, size, littleEndian), // Void/opaque
                    _ => throw new FormatException($"Unexpected type: {typeChar}")
                };
            }

            // Parse numeric size
            if (!int.TryParse(typeStr.Substring(1), out int itemSize))
                throw new FormatException($"Invalid dtype size in: '{descr}'");

            NPTypeCode typeCode = (typeChar, itemSize) switch
            {
                ('b', 1) => NPTypeCode.Boolean,
                ('i', 1) => NPTypeCode.Byte, // NumPy i1 is signed, but we map to byte
                ('i', 2) => NPTypeCode.Int16,
                ('i', 4) => NPTypeCode.Int32,
                ('i', 8) => NPTypeCode.Int64,
                ('u', 1) => NPTypeCode.Byte,
                ('u', 2) => NPTypeCode.UInt16,
                ('u', 4) => NPTypeCode.UInt32,
                ('u', 8) => NPTypeCode.UInt64,
                ('f', 2) => throw new NotSupportedException("float16 is not supported"),
                ('f', 4) => NPTypeCode.Single,
                ('f', 8) => NPTypeCode.Double,
                ('f', 16) => throw new NotSupportedException("float128 is not supported"),
                ('c', 8) => throw new NotSupportedException("complex64 is not supported"),
                ('c', 16) => throw new NotSupportedException("complex128 is not supported"),
                _ => throw new FormatException($"Unknown dtype: '{descr}'")
            };

            return (typeCode, itemSize, littleEndian);
        }

        #endregion

        #region Header Writing

        /// <summary>
        /// Get header dictionary from NDArray.
        /// </summary>
        public static Dictionary<string, object> HeaderDataFromArray(NDArray array)
        {
            bool fortranOrder;
            if (array.Shape.IsContiguous)
            {
                // Both C and F contiguous for 1D/scalar - prefer C
                fortranOrder = false;
            }
            else
            {
                // Non-contiguous: will be copied as C-order
                fortranOrder = false;
            }

            return new Dictionary<string, object>
            {
                ["descr"] = TypeCodeToDescr(array.typecode),
                ["fortran_order"] = fortranOrder,
                ["shape"] = array.shape
            };
        }

        /// <summary>
        /// Format header dictionary as Python literal string.
        /// </summary>
        private static string FormatHeaderDict(Dictionary<string, object> d)
        {
            var sb = new StringBuilder("{");

            // Keys must be sorted alphabetically (NumPy convention)
            var sortedKeys = new List<string>(d.Keys);
            sortedKeys.Sort(StringComparer.Ordinal);

            foreach (var key in sortedKeys)
            {
                sb.Append($"'{key}': ");
                var value = d[key];

                if (value is string s)
                {
                    sb.Append($"'{s}'");
                }
                else if (value is bool b)
                {
                    sb.Append(b ? "True" : "False");
                }
                else if (value is int[] intShape)
                {
                    sb.Append('(');
                    for (int i = 0; i < intShape.Length; i++)
                    {
                        sb.Append(intShape[i]);
                        if (i < intShape.Length - 1)
                            sb.Append(", ");
                    }
                    // Trailing comma for 1-element tuple
                    if (intShape.Length == 1)
                        sb.Append(',');
                    sb.Append(')');
                }
                else if (value is long[] longShape)
                {
                    sb.Append('(');
                    for (int i = 0; i < longShape.Length; i++)
                    {
                        sb.Append(longShape[i]);
                        if (i < longShape.Length - 1)
                            sb.Append(", ");
                    }
                    // Trailing comma for 1-element tuple
                    if (longShape.Length == 1)
                        sb.Append(',');
                    sb.Append(')');
                }
                else
                {
                    sb.Append(value);
                }

                sb.Append(", ");
            }

            sb.Append('}');
            return sb.ToString();
        }

        /// <summary>
        /// Wrap header string with magic, version, and padding.
        /// </summary>
        private static byte[] WrapHeader(string header, FormatVersion version)
        {
            var encoding = version.HeaderEncoding;
            byte[] headerBytes = encoding.GetBytes(header);
            int hlen = headerBytes.Length + 1; // +1 for newline

            int headerLengthSize = version.HeaderLengthSize;
            int preambleSize = MagicLen + headerLengthSize;

            // Calculate padding to align to ArrayAlign
            int padlen = ArrayAlign - ((preambleSize + hlen) % ArrayAlign);
            if (padlen == ArrayAlign) padlen = 0;

            int totalHeaderLen = hlen + padlen;

            // Check if header fits in version
            if (totalHeaderLen > version.MaxHeaderLength)
            {
                throw new InvalidOperationException(
                    $"Header length {totalHeaderLen} too big for version {version}");
            }

            // Build complete header
            byte[] result = new byte[preambleSize + totalHeaderLen];
            int pos = 0;

            // Magic
            Array.Copy(MagicPrefix, 0, result, pos, MagicPrefix.Length);
            pos += MagicPrefix.Length;

            // Version
            result[pos++] = version.Major;
            result[pos++] = version.Minor;

            // Header length (little-endian)
            if (version.Major >= 2)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(pos), (uint)totalHeaderLen);
                pos += 4;
            }
            else
            {
                BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(pos), (ushort)totalHeaderLen);
                pos += 2;
            }

            // Header content
            Array.Copy(headerBytes, 0, result, pos, headerBytes.Length);
            pos += headerBytes.Length;

            // Padding (spaces)
            for (int i = 0; i < padlen; i++)
                result[pos++] = (byte)' ';

            // Trailing newline
            result[pos] = (byte)'\n';

            return result;
        }

        /// <summary>
        /// Automatically select version and wrap header.
        /// </summary>
        private static byte[] WrapHeaderGuessVersion(string header)
        {
            // Try v1.0 first (most compatible)
            try
            {
                return WrapHeader(header, FormatVersion.V1_0);
            }
            catch (InvalidOperationException)
            {
                // Header too large for v1.0
            }

            // Try v2.0
            try
            {
                var bytes = Encoding.Latin1.GetBytes(header);
                return WrapHeader(header, FormatVersion.V2_0);
            }
            catch (EncoderFallbackException)
            {
                // Contains non-Latin1 characters
            }
            catch (InvalidOperationException)
            {
                // Shouldn't happen for v2.0
            }

            // Fall back to v3.0 (UTF-8)
            return WrapHeader(header, FormatVersion.V3_0);
        }

        /// <summary>
        /// Write array header to stream.
        /// </summary>
        public static void WriteArrayHeader(Stream stream, Dictionary<string, object> d, FormatVersion? version = null)
        {
            string header = FormatHeaderDict(d);

            // Add growth axis padding (NumPy adds space for in-place header modification when appending)
            if (d.TryGetValue("shape", out var shapeObj))
            {
                int shapeLength = 0;
                long growthAxisValue = 0;

                if (shapeObj is long[] longShape && longShape.Length > 0)
                {
                    shapeLength = longShape.Length;
                    bool fortranOrder = d.TryGetValue("fortran_order", out var fo) && fo is bool b && b;
                    int growthAxis = fortranOrder ? shapeLength - 1 : 0;
                    growthAxisValue = longShape[growthAxis];
                }
                else if (shapeObj is int[] intShape && intShape.Length > 0)
                {
                    shapeLength = intShape.Length;
                    bool fortranOrder = d.TryGetValue("fortran_order", out var fo) && fo is bool b && b;
                    int growthAxis = fortranOrder ? shapeLength - 1 : 0;
                    growthAxisValue = intShape[growthAxis];
                }

                if (shapeLength > 0)
                {
                    int currentDigits = growthAxisValue.ToString().Length;
                    int padding = GrowthAxisMaxDigits - currentDigits;
                    if (padding > 0)
                        header += new string(' ', padding);
                }
            }

            byte[] wrappedHeader = version.HasValue
                ? WrapHeader(header, version.Value)
                : WrapHeaderGuessVersion(header);

            stream.Write(wrappedHeader, 0, wrappedHeader.Length);
        }

        #endregion

        #region Header Reading

        /// <summary>
        /// Read array header from stream.
        /// </summary>
        public static HeaderData ReadArrayHeader(Stream stream, FormatVersion version, int maxHeaderSize = MaxHeaderSize)
        {
            CheckVersion(version);

            // Read header length
            int headerLengthSize = version.HeaderLengthSize;
            byte[] lenBytes = ReadBytes(stream, headerLengthSize, "header length");

            uint headerLength = headerLengthSize == 2
                ? BinaryPrimitives.ReadUInt16LittleEndian(lenBytes)
                : BinaryPrimitives.ReadUInt32LittleEndian(lenBytes);

            if (headerLength > int.MaxValue)
                throw new FormatException($"Header length {headerLength} is too large");

            // Read header
            byte[] headerBytes = ReadBytes(stream, (int)headerLength, "header");
            string header = version.HeaderEncoding.GetString(headerBytes).TrimEnd();

            // Security check
            if (header.Length > maxHeaderSize)
            {
                throw new FormatException(
                    $"Header size ({header.Length}) exceeds maximum allowed ({maxHeaderSize}). " +
                    "Use a larger maxHeaderSize if you trust this file.");
            }

            // Parse header dictionary
            return ParseHeader(header, version);
        }

        /// <summary>
        /// Parse header string to HeaderData.
        /// </summary>
        private static HeaderData ParseHeader(string header, FormatVersion version)
        {
            // Simple regex-based parser for Python dict literal
            // Format: {'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }

            // Extract descr
            var descrMatch = Regex.Match(header, @"'descr'\s*:\s*'([^']+)'");
            if (!descrMatch.Success)
                throw new FormatException($"Cannot find 'descr' in header: {header}");
            string descr = descrMatch.Groups[1].Value;

            // Extract fortran_order
            var fortranMatch = Regex.Match(header, @"'fortran_order'\s*:\s*(True|False)");
            if (!fortranMatch.Success)
                throw new FormatException($"Cannot find 'fortran_order' in header: {header}");
            bool fortranOrder = fortranMatch.Groups[1].Value == "True";

            // Extract shape
            var shapeMatch = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)");
            if (!shapeMatch.Success)
                throw new FormatException($"Cannot find 'shape' in header: {header}");

            string shapeStr = shapeMatch.Groups[1].Value.Trim();
            long[] shape;
            if (string.IsNullOrEmpty(shapeStr))
            {
                shape = Array.Empty<long>(); // Scalar
            }
            else
            {
                // Handle trailing comma: (3,) or (3, 4,)
                var parts = shapeStr.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
                shape = new long[parts.Length];
                for (int i = 0; i < parts.Length; i++)
                {
                    string part = parts[i].Trim();
                    // Handle Python 2 'L' suffix
                    if (part.EndsWith("L", StringComparison.OrdinalIgnoreCase))
                        part = part.Substring(0, part.Length - 1);
                    if (!long.TryParse(part, out shape[i]))
                        throw new FormatException($"Invalid shape value: '{parts[i]}'");
                }
            }

            // Parse dtype
            var (typeCode, itemSize, littleEndian) = DescrToTypeCode(descr);

            return new HeaderData(shape, fortranOrder, typeCode, itemSize, littleEndian);
        }

        #endregion

        #region Array Writing

        /// <summary>
        /// Write NDArray to stream in .npy format.
        /// </summary>
        public static void WriteArray(Stream stream, NDArray array, FormatVersion? version = null)
        {
            if (array.typecode == NPTypeCode.Decimal)
                throw new NotSupportedException("Decimal type is not supported by NumPy format");

            // Write header
            var headerDict = HeaderDataFromArray(array);
            WriteArrayHeader(stream, headerDict, version);

            // Write data
            WriteArrayData(stream, array);
        }

        /// <summary>
        /// Write array data to stream.
        /// </summary>
        private static void WriteArrayData(Stream stream, NDArray array)
        {
            if (array.size == 0)
                return;

            int itemSize = array.dtypesize;

            // For contiguous arrays, write directly
            if (array.Shape.IsContiguous)
            {
                WriteContiguousData(stream, array);
            }
            else
            {
                // Non-contiguous: iterate and write
                WriteNonContiguousData(stream, array);
            }
        }

        /// <summary>
        /// Write contiguous array data.
        /// </summary>
        private static unsafe void WriteContiguousData(Stream stream, NDArray array)
        {
            long totalBytes = array.size * array.dtypesize;
            byte* ptr = (byte*)array.Address;

            // Write in chunks to avoid large buffer allocations
            const int chunkSize = 16 * 1024 * 1024; // 16 MB
            byte[] buffer = new byte[Math.Min(totalBytes, chunkSize)];

            long remaining = totalBytes;
            long offset = 0;

            while (remaining > 0)
            {
                int toWrite = (int)Math.Min(remaining, buffer.Length);
                Marshal.Copy((IntPtr)(ptr + offset), buffer, 0, toWrite);
                stream.Write(buffer, 0, toWrite);
                offset += toWrite;
                remaining -= toWrite;
            }
        }

        /// <summary>
        /// Write non-contiguous array data in C-order.
        /// </summary>
        private static unsafe void WriteNonContiguousData(Stream stream, NDArray array)
        {
            int itemSize = array.dtypesize;
            byte[] buffer = new byte[itemSize];
            long[] shape = array.shape;
            int ndim = shape.Length;

            if (ndim == 0)
            {
                // Scalar
                byte* ptr = (byte*)array.Address;
                Marshal.Copy((IntPtr)ptr, buffer, 0, itemSize);
                stream.Write(buffer, 0, itemSize);
                return;
            }

            // Iterate in C-order using coordinates
            long[] coords = new long[ndim];
            long[] strides = array.strides;
            long baseAddr = (long)array.Address;
            long size = array.size;
            long sliceOffset = array.Shape.offset; // Account for sliced views

            for (long i = 0; i < size; i++)
            {
                // Calculate offset from coordinates and strides
                // Start with slice offset for non-contiguous sliced views
                long offset = sliceOffset;
                for (int d = 0; d < ndim; d++)
                    offset += coords[d] * strides[d];

                byte* ptr = (byte*)(baseAddr + offset * itemSize);
                Marshal.Copy((IntPtr)ptr, buffer, 0, itemSize);
                stream.Write(buffer, 0, itemSize);

                // Increment coordinates (C-order: last dimension varies fastest)
                for (int d = ndim - 1; d >= 0; d--)
                {
                    coords[d]++;
                    if (coords[d] < shape[d])
                        break;
                    coords[d] = 0;
                }
            }
        }

        #endregion

        #region Array Reading

        /// <summary>
        /// Read NDArray from stream.
        /// </summary>
        public static NDArray ReadArray(Stream stream, int maxHeaderSize = MaxHeaderSize)
        {
            // Read magic and version
            var version = ReadMagic(stream);
            CheckVersion(version);

            // Read header
            var header = ReadArrayHeader(stream, version, maxHeaderSize);

            // Handle byte-order conversion if needed
            bool needsByteSwap = header.LittleEndian != BitConverter.IsLittleEndian;

            // Read data
            return ReadArrayData(stream, header, needsByteSwap);
        }

        /// <summary>
        /// Read array data from stream.
        /// </summary>
        private static NDArray ReadArrayData(Stream stream, HeaderData header, bool needsByteSwap)
        {
            long count = header.Count;
            int itemSize = header.ItemSize;
            long totalBytes = count * itemSize;

            // Create result array
            NDArray result = new NDArray(header.TypeCode, new Shape(header.Shape));

            if (count == 0)
                return result;

            // Read data
            ReadDataIntoArray(stream, result, totalBytes, needsByteSwap, itemSize);

            // Handle Fortran order
            if (header.FortranOrder && header.Shape.Length > 1)
            {
                // Data is in F-order, need to transpose
                // First reshape with reversed dimensions, then transpose
                long[] reversedShape = new long[header.Shape.Length];
                for (int i = 0; i < header.Shape.Length; i++)
                    reversedShape[i] = header.Shape[header.Shape.Length - 1 - i];

                result = result.reshape(reversedShape).T;
            }

            return result;
        }

        /// <summary>
        /// Read data directly into NDArray storage.
        /// </summary>
        private static unsafe void ReadDataIntoArray(Stream stream, NDArray array, long totalBytes, bool needsByteSwap, int itemSize)
        {
            byte* ptr = (byte*)array.Address;

            // Read in chunks
            byte[] buffer = new byte[Math.Min(totalBytes, BufferSize)];
            long remaining = totalBytes;
            long offset = 0;

            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                int bytesRead = ReadBytesIntoBuffer(stream, buffer, toRead);

                if (bytesRead != toRead)
                {
                    throw new EndOfStreamException(
                        $"Failed to read array data. Expected {totalBytes} bytes, got {offset + bytesRead}");
                }

                // Byte-swap if necessary
                if (needsByteSwap && itemSize > 1)
                {
                    SwapBytes(buffer, bytesRead, itemSize);
                }

                Marshal.Copy(buffer, 0, (IntPtr)(ptr + offset), bytesRead);
                offset += bytesRead;
                remaining -= bytesRead;
            }
        }

        /// <summary>
        /// Swap byte order for multi-byte elements.
        /// </summary>
        private static void SwapBytes(byte[] buffer, int length, int itemSize)
        {
            for (int i = 0; i < length; i += itemSize)
            {
                // Reverse bytes for each element
                int start = i;
                int end = i + itemSize - 1;
                while (start < end)
                {
                    (buffer[start], buffer[end]) = (buffer[end], buffer[start]);
                    start++;
                    end--;
                }
            }
        }

        #endregion

        #region Utility Methods

        /// <summary>
        /// Read exact number of bytes from stream.
        /// </summary>
        private static byte[] ReadBytes(Stream stream, int count, string description)
        {
            byte[] buffer = new byte[count];
            int totalRead = ReadBytesIntoBuffer(stream, buffer, count);

            if (totalRead != count)
            {
                throw new EndOfStreamException(
                    $"EOF reading {description}: expected {count} bytes, got {totalRead}");
            }

            return buffer;
        }

        /// <summary>
        /// Read bytes into existing buffer, handling partial reads.
        /// </summary>
        private static int ReadBytesIntoBuffer(Stream stream, byte[] buffer, int count)
        {
            int totalRead = 0;
            while (totalRead < count)
            {
                int read = stream.Read(buffer, totalRead, count - totalRead);
                if (read == 0)
                    break; // EOF
                totalRead += read;
            }
            return totalRead;
        }

        /// <summary>
        /// Check if stream appears to be a .npy file by checking magic bytes.
        /// Does not consume the stream.
        /// </summary>
        public static bool IsNpyFile(Stream stream)
        {
            if (!stream.CanSeek)
                throw new ArgumentException("Stream must be seekable");

            long pos = stream.Position;
            try
            {
                if (stream.Length - pos < MagicPrefix.Length)
                    return false;

                byte[] magic = new byte[MagicPrefix.Length];
                stream.Read(magic, 0, magic.Length);

                for (int i = 0; i < MagicPrefix.Length; i++)
                {
                    if (magic[i] != MagicPrefix[i])
                        return false;
                }

                return true;
            }
            finally
            {
                stream.Position = pos;
            }
        }

        /// <summary>
        /// Check if stream appears to be a ZIP (.npz) file.
        /// Does not consume the stream.
        /// </summary>
        public static bool IsNpzFile(Stream stream)
        {
            if (!stream.CanSeek)
                throw new ArgumentException("Stream must be seekable");

            long pos = stream.Position;
            try
            {
                if (stream.Length - pos < 4)
                    return false;

                byte[] magic = new byte[4];
                stream.Read(magic, 0, 4);

                // ZIP magic: PK\x03\x04 or PK\x05\x06 (empty zip)
                return (magic[0] == 0x50 && magic[1] == 0x4B &&
                        ((magic[2] == 0x03 && magic[3] == 0x04) ||
                         (magic[2] == 0x05 && magic[3] == 0x06)));
            }
            finally
            {
                stream.Position = pos;
            }
        }

        #endregion
    }
}
