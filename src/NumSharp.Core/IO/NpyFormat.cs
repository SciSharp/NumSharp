using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp.IO
{
    /// <summary>
    ///     The NumPy <c>.npy</c> binary format (NEP-01) — reader and writer for versions 1.0, 2.0 and 3.0.
    /// </summary>
    /// <remarks>
    ///     A port of <c>numpy/lib/_format_impl.py</c> (NumPy 2.4.2), structured to mirror it function for
    ///     function so the two can be diffed. Byte-for-byte identical output is a tested contract: see
    ///     <c>test/oracle/gen_npy_oracle.py</c>, which has real NumPy write every case this class claims
    ///     to support, and <c>NpyOracleTests</c>, which replays them.
    ///
    ///     File layout:
    ///     <code>
    ///     \x93NUMPY          6 bytes   magic (\x93 is byte 147, NOT the character '?')
    ///     major, minor       2 bytes   format version
    ///     HEADER_LEN         2 bytes   uint16 little-endian  (v1.0)
    ///                        4 bytes   uint32 little-endian  (v2.0 / v3.0)
    ///     header             HEADER_LEN bytes — a Python dict literal, space-padded, '\n'-terminated,
    ///                        sized so the data starts on a 64-byte (ARRAY_ALIGN) boundary
    ///     data               shape-product * itemsize raw bytes, C- or Fortran-order
    ///     </code>
    ///
    ///     Divergences from NumPy, all forced by NumSharp's type system and each covered by a test:
    ///     <list type="bullet">
    ///       <item>Big-endian files are byte-swapped to native on read. NumPy instead keeps the raw bytes
    ///             behind a byte-swapped dtype; NumSharp has no such dtype, so it converts. Values match;
    ///             the byte-order attribute is not preserved.</item>
    ///       <item><c>Char</c> (2-byte UTF-16) maps to NumPy's <c>'&lt;U1'</c> (4-byte UCS-4), converting in
    ///             both directions. <c>'&lt;U'</c> widths above 1 have no NumSharp analog.</item>
    ///       <item><c>'&lt;c8'</c> (complex64) widens to <see cref="System.Numerics.Complex"/> on read;
    ///             writing always emits <c>'&lt;c16'</c>.</item>
    ///       <item><c>Decimal</c> has no NumPy dtype and cannot be written.</item>
    ///       <item>Object arrays, structured dtypes, datetime64/timedelta64, byte strings and void are
    ///             parsed but rejected with a precise message (see the issue's "What We're Not Doing").</item>
    ///     </list>
    /// </remarks>
    public static class NpyFormat
    {
        #region Constants

        /// <summary>The magic string every .npy file starts with: <c>\x93NUMPY</c>.</summary>
        public static ReadOnlySpan<byte> MagicPrefix => new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

        /// <summary>Length of the magic string plus the two version bytes.</summary>
        public const int MagicLen = 8;

        /// <summary>
        ///     The header is padded so the data begins on a multiple of this. 64 bytes lets the data be
        ///     memory-mapped and read with aligned SIMD loads.
        /// </summary>
        public const int ArrayAlign = 64;

        /// <summary>Chunk size for streamed reads, matching NumPy's <c>BUFFER_SIZE</c> (256 KB).</summary>
        public const int BufferSize = 1 << 18;

        /// <summary>
        ///     Spare spaces reserved after the dict so the shape can be rewritten in place when a file is
        ///     grown along its first (C-order) or last (F-order) axis. <c>len(str(8 * 2**64 - 1))</c>.
        /// </summary>
        public const int GrowthAxisMaxDigits = 21;

        /// <summary>
        ///     Default cap on header size. Parsing an arbitrarily large header is a denial-of-service
        ///     vector, and no legitimate array needs more than this.
        /// </summary>
        public const int MaxHeaderSize = 10000;

        /// <summary>The only keys a valid header may contain.</summary>
        private static readonly string[] ExpectedKeys = { "descr", "fortran_order", "shape" };

        /// <summary>The first bytes of a ZIP archive — a .npz. <c>PK\x05\x06</c> marks an empty one.</summary>
        private static ReadOnlySpan<byte> ZipPrefix => new byte[] { 0x50, 0x4B, 0x03, 0x04 };
        private static ReadOnlySpan<byte> ZipSuffix => new byte[] { 0x50, 0x4B, 0x05, 0x06 };

        // latin1 that THROWS instead of silently substituting '?', so header encoding can detect
        // characters that need version 3.0 — mirroring Python's UnicodeEncodeError.
        private static readonly Encoding StrictLatin1 =
            Encoding.GetEncoding("ISO-8859-1", EncoderFallback.ExceptionFallback, DecoderFallback.ExceptionFallback);

        private static readonly Encoding Latin1 = Encoding.Latin1;
        private static readonly Encoding Utf8 = new UTF8Encoding(false);

        #endregion

        #region Format version

        /// <summary>A .npy format version. Only (1,0), (2,0) and (3,0) exist.</summary>
        public readonly struct FormatVersion : IEquatable<FormatVersion>
        {
            public readonly byte Major;
            public readonly byte Minor;

            public FormatVersion(byte major, byte minor)
            {
                Major = major;
                Minor = minor;
            }

            /// <summary>Version 1.0 — 2-byte header length, latin-1. The default and most portable.</summary>
            public static FormatVersion V1_0 => new FormatVersion(1, 0);

            /// <summary>Version 2.0 (NumPy 1.9) — 4-byte header length, for headers above 64 KB.</summary>
            public static FormatVersion V2_0 => new FormatVersion(2, 0);

            /// <summary>Version 3.0 (NumPy 1.17) — 4-byte header length, UTF-8, for Unicode field names.</summary>
            public static FormatVersion V3_0 => new FormatVersion(3, 0);

            /// <summary>Size of the header-length field: 2 bytes for v1.0, 4 for v2.0 and v3.0.</summary>
            public int HeaderLengthSize => Major >= 2 ? 4 : 2;

            /// <summary>The header's text encoding: UTF-8 from v3.0, latin-1 before it.</summary>
            public Encoding HeaderEncoding => Major >= 3 ? Utf8 : Latin1;

            /// <summary>The encoding used when writing, which throws rather than substituting characters.</summary>
            internal Encoding StrictHeaderEncoding => Major >= 3 ? Utf8 : StrictLatin1;

            /// <summary>Largest value the header-length field can hold.</summary>
            public uint MaxHeaderLength => Major >= 2 ? uint.MaxValue : ushort.MaxValue;

            public bool IsSupported =>
                (Major == 1 || Major == 2 || Major == 3) && Minor == 0;

            public bool Equals(FormatVersion other) => Major == other.Major && Minor == other.Minor;
            public override bool Equals(object obj) => obj is FormatVersion v && Equals(v);
            public override int GetHashCode() => (Major << 8) | Minor;

            /// <summary>Formatted as Python's tuple repr, e.g. <c>(1, 0)</c>, so it can go verbatim into messages.</summary>
            public override string ToString() => $"({Major}, {Minor})";

            public static bool operator ==(FormatVersion l, FormatVersion r) => l.Equals(r);
            public static bool operator !=(FormatVersion l, FormatVersion r) => !l.Equals(r);
            public static bool operator <=(FormatVersion l, FormatVersion r) =>
                l.Major < r.Major || (l.Major == r.Major && l.Minor <= r.Minor);
            public static bool operator >=(FormatVersion l, FormatVersion r) =>
                l.Major > r.Major || (l.Major == r.Major && l.Minor >= r.Minor);
        }

        #endregion

        #region dtype descriptors

        /// <summary>
        ///     How a file's elements must be transformed to become NumSharp elements. Everything except
        ///     <see cref="None"/> exists because NumSharp's type has a different width than NumPy's.
        /// </summary>
        internal enum ElementConversion
        {
            /// <summary>File bytes are NumSharp bytes (after any byte-swap).</summary>
            None,

            /// <summary>NumPy 4-byte UCS-4 code point ↔ NumSharp 2-byte UTF-16 <see cref="char"/>.</summary>
            Ucs4,

            /// <summary>NumPy <c>'c8'</c> (two float32) → <see cref="System.Numerics.Complex"/> (two float64).</summary>
            Complex64,
        }

        /// <summary>A parsed <c>descr</c>: what the file holds, and what NumSharp will hold.</summary>
        internal readonly struct DtypeInfo
        {
            /// <summary>The dtype NumSharp materializes.</summary>
            public readonly NPTypeCode TypeCode;

            /// <summary>Bytes per element as stored in the FILE (may differ from NumSharp's item size).</summary>
            public readonly int FileItemSize;

            /// <summary>
            ///     Size of one byte-swappable unit within an element. An element is not always one unit:
            ///     complex swaps per component, so <c>'&gt;c16'</c> is 2 units of 8 bytes, not 1 of 16.
            /// </summary>
            public readonly int SwapUnit;

            /// <summary>The file's byte order. Native for <c>'|'</c> and <c>'='</c>.</summary>
            public readonly bool LittleEndian;

            public readonly ElementConversion Conversion;

            /// <summary>True for <c>'|O'</c>, whose data is a Python pickle rather than raw elements.</summary>
            public readonly bool HasObject;

            public DtypeInfo(NPTypeCode typeCode, int fileItemSize, int swapUnit, bool littleEndian,
                             ElementConversion conversion = ElementConversion.None, bool hasObject = false)
            {
                TypeCode = typeCode;
                FileItemSize = fileItemSize;
                SwapUnit = swapUnit;
                LittleEndian = littleEndian;
                Conversion = conversion;
                HasObject = hasObject;
            }

            /// <summary>True when the file's byte order differs from this machine's.</summary>
            public bool NeedsSwap => LittleEndian != BitConverter.IsLittleEndian && SwapUnit > 1;
        }

        /// <summary>
        ///     The <c>descr</c> string for a NumSharp dtype — the inverse of <see cref="DescrToDtype"/>
        ///     and the equivalent of NumPy's <c>dtype_to_descr</c> for the types NumSharp has.
        /// </summary>
        /// <remarks>
        ///     Single-byte types take the <c>'|'</c> (not-applicable) prefix, everything else <c>'&lt;'</c>:
        ///     NumSharp is native-order internally and every platform it runs on is little-endian.
        /// </remarks>
        /// <exception cref="NotSupportedException">The dtype has no NumPy equivalent.</exception>
        public static string DtypeToDescr(NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Boolean: return "|b1";
                case NPTypeCode.SByte: return "|i1";
                case NPTypeCode.Byte: return "|u1";
                case NPTypeCode.Int16: return "<i2";
                case NPTypeCode.UInt16: return "<u2";
                case NPTypeCode.Int32: return "<i4";
                case NPTypeCode.UInt32: return "<u4";
                case NPTypeCode.Int64: return "<i8";
                case NPTypeCode.UInt64: return "<u8";
                case NPTypeCode.Half: return "<f2";
                case NPTypeCode.Single: return "<f4";
                case NPTypeCode.Double: return "<f8";
                case NPTypeCode.Complex: return "<c16";
                case NPTypeCode.Char: return "<U1"; // 2-byte UTF-16 widened to 4-byte UCS-4 on write
                case NPTypeCode.Decimal:
                    throw new NotSupportedException(
                        "Decimal has no NumPy dtype and cannot be saved to a .npy file. " +
                        "Convert first, e.g. arr.astype(np.float64) — note this loses precision.");
                default:
                    throw new NotSupportedException($"{typeCode} cannot be saved to a .npy file.");
            }
        }

        /// <summary>
        ///     Turn a header's <c>descr</c> into the dtype NumSharp will materialize — NumPy's
        ///     <c>descr_to_dtype</c>.
        /// </summary>
        /// <param name="descr">The parsed value of the header's <c>descr</c> key.</param>
        /// <exception cref="FormatException">The descriptor is not a valid NumPy dtype.</exception>
        /// <exception cref="NotSupportedException">A valid NumPy dtype NumSharp cannot represent.</exception>
        internal static DtypeInfo DescrToDtype(object descr)
        {
            // A list of field tuples is a structured (record) dtype; a tuple is a subarray dtype. Both
            // are valid NumPy and both are out of scope, so report them precisely rather than as
            // "invalid descriptor".
            if (descr is List<object>)
                throw new NotSupportedException(
                    $"Structured dtypes are not supported by NumSharp: {PyLiteral.Repr(descr)}");
            if (descr is PyTuple)
                throw new NotSupportedException(
                    $"Subarray dtypes are not supported by NumSharp: {PyLiteral.Repr(descr)}");

            if (!(descr is string s) || s.Length == 0)
                throw new FormatException($"descr is not a valid dtype descriptor: {PyLiteral.Repr(descr)}");

            // <endian><kind><size>[extra]
            int i = 0;
            bool littleEndian;
            switch (s[0])
            {
                case '<': littleEndian = true; i = 1; break;
                case '>': littleEndian = false; i = 1; break;
                case '|':
                case '=': littleEndian = BitConverter.IsLittleEndian; i = 1; break;
                default: littleEndian = BitConverter.IsLittleEndian; break; // no prefix: native
            }

            if (i >= s.Length)
                throw new FormatException($"descr is not a valid dtype descriptor: {PyLiteral.Repr(descr)}");

            char kind = s[i];
            string rest = s.Substring(i + 1);

            // Object: parses fine; read_array decides based on allow_pickle.
            if (kind == 'O')
                return new DtypeInfo(NPTypeCode.Empty, 0, 1, littleEndian, ElementConversion.None, hasObject: true);

            // datetime64/timedelta64 carry a unit suffix, e.g. '<M8[ns]'.
            if (kind == 'M' || kind == 'm')
            {
                string full = kind == 'M' ? "datetime64" : "timedelta64";
                throw new NotSupportedException(
                    $"{full} ('{s}') is not supported by NumSharp — it has no .NET analog yet. " +
                    "Convert to an integer count of units in NumPy before saving.");
            }

            if (kind == 'S' || kind == 'a')
                throw new NotSupportedException(
                    $"Byte-string dtypes ('{s}') are not supported by NumSharp, which has no string dtype.");

            if (kind == 'V')
                throw new NotSupportedException(
                    $"Void dtypes ('{s}') are not supported by NumSharp.");

            if (kind == 'U')
            {
                if (!int.TryParse(rest, NumberStyles.None, CultureInfo.InvariantCulture, out int chars) || chars < 1)
                    throw new FormatException($"descr is not a valid dtype descriptor: {PyLiteral.Repr(descr)}");
                if (chars != 1)
                    throw new NotSupportedException(
                        $"Unicode string dtype '{s}' is not supported — only '<U1' maps to NumSharp's Char " +
                        "(a single UTF-16 code unit). NumSharp has no string dtype.");
                return new DtypeInfo(NPTypeCode.Char, 4, 4, littleEndian, ElementConversion.Ucs4);
            }

            if (!int.TryParse(rest, NumberStyles.None, CultureInfo.InvariantCulture, out int size))
                throw new FormatException($"descr is not a valid dtype descriptor: {PyLiteral.Repr(descr)}");

            switch (kind)
            {
                case 'b' when size == 1: return new DtypeInfo(NPTypeCode.Boolean, 1, 1, littleEndian);
                case 'i' when size == 1: return new DtypeInfo(NPTypeCode.SByte, 1, 1, littleEndian);
                case 'i' when size == 2: return new DtypeInfo(NPTypeCode.Int16, 2, 2, littleEndian);
                case 'i' when size == 4: return new DtypeInfo(NPTypeCode.Int32, 4, 4, littleEndian);
                case 'i' when size == 8: return new DtypeInfo(NPTypeCode.Int64, 8, 8, littleEndian);
                case 'u' when size == 1: return new DtypeInfo(NPTypeCode.Byte, 1, 1, littleEndian);
                case 'u' when size == 2: return new DtypeInfo(NPTypeCode.UInt16, 2, 2, littleEndian);
                case 'u' when size == 4: return new DtypeInfo(NPTypeCode.UInt32, 4, 4, littleEndian);
                case 'u' when size == 8: return new DtypeInfo(NPTypeCode.UInt64, 8, 8, littleEndian);
                case 'f' when size == 2: return new DtypeInfo(NPTypeCode.Half, 2, 2, littleEndian);
                case 'f' when size == 4: return new DtypeInfo(NPTypeCode.Single, 4, 4, littleEndian);
                case 'f' when size == 8: return new DtypeInfo(NPTypeCode.Double, 8, 8, littleEndian);
                case 'f' when size == 16:
                    throw new NotSupportedException(
                        "float128 ('" + s + "') is not supported — .NET has no 128-bit float. " +
                        "Convert to float64 in NumPy before saving.");
                // Complex byte-swaps per component: '>c16' is two 8-byte units, not one 16-byte unit.
                case 'c' when size == 8: return new DtypeInfo(NPTypeCode.Complex, 8, 4, littleEndian, ElementConversion.Complex64);
                case 'c' when size == 16: return new DtypeInfo(NPTypeCode.Complex, 16, 8, littleEndian);
                case 'c' when size == 32:
                    throw new NotSupportedException(
                        "complex256 ('" + s + "') is not supported — .NET has no 256-bit complex. " +
                        "Convert to complex128 in NumPy before saving.");
                default:
                    throw new FormatException($"descr is not a valid dtype descriptor: {PyLiteral.Repr(descr)}");
            }
        }

        #endregion

        #region Magic and version

        /// <summary>The 8 magic bytes for a version — NumPy's <c>magic()</c>.</summary>
        public static byte[] Magic(FormatVersion version)
        {
            var magic = new byte[MagicLen];
            MagicPrefix.CopyTo(magic);
            magic[6] = version.Major;
            magic[7] = version.Minor;
            return magic;
        }

        /// <summary>
        ///     Read and validate the magic string, returning the file's format version — NumPy's
        ///     <c>read_magic()</c>. Leaves the stream on the header-length field.
        /// </summary>
        /// <exception cref="FormatException">The magic string is wrong or the stream ends inside it.</exception>
        public static FormatVersion ReadMagic(Stream stream)
        {
            byte[] magic = ReadBytes(stream, MagicLen, "magic string");
            if (!magic.AsSpan(0, MagicPrefix.Length).SequenceEqual(MagicPrefix))
                throw new FormatException(
                    "the magic string is not correct; expected " + PyBytesRepr(MagicPrefix) +
                    ", got " + PyBytesRepr(magic.AsSpan(0, MagicPrefix.Length)));

            return new FormatVersion(magic[6], magic[7]);
        }

        /// <summary>Reject versions this implementation does not know — NumPy's <c>_check_version()</c>.</summary>
        /// <exception cref="FormatException">The version is not (1,0), (2,0) or (3,0).</exception>
        public static void CheckVersion(FormatVersion version)
        {
            if (!version.IsSupported)
                throw new FormatException(
                    $"we only support format version (1,0), (2,0), and (3,0), not {version}");
        }

        /// <summary>Render bytes the way Python renders a bytes repr, for verbatim error messages.</summary>
        private static string PyBytesRepr(ReadOnlySpan<byte> bytes)
        {
            var sb = new StringBuilder("b'");
            foreach (byte b in bytes)
            {
                if (b == (byte)'\\') sb.Append("\\\\");
                else if (b == (byte)'\'') sb.Append("\\'");
                else if (b >= 0x20 && b < 0x7F) sb.Append((char)b);
                else sb.Append("\\x").Append(b.ToString("x2", CultureInfo.InvariantCulture));
            }
            return sb.Append('\'').ToString();
        }

        #endregion

        #region Header — writing

        /// <summary>
        ///     Build the header dict for an array — NumPy's <c>header_data_from_array_1_0()</c>.
        /// </summary>
        /// <remarks>
        ///     C-contiguity is tested first because a 1-D, 0-d or empty array is BOTH C- and
        ///     F-contiguous, and those must come out as <c>fortran_order: False</c>. An array that is
        ///     neither is copied to C-order at write time, so it too reports False.
        /// </remarks>
        public static Dictionary<string, object> HeaderDataFromArray(NDArray array)
        {
            bool fortranOrder = !array.Shape.IsContiguous && array.Shape.IsFContiguous;

            return new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["descr"] = DtypeToDescr(array.typecode),
                ["fortran_order"] = fortranOrder,
                ["shape"] = array.shape,
            };
        }

        // The dict as a Python literal, keys sorted — NumPy: "For repeatability and readability, the
        // dictionary keys are sorted in alphabetic order. A writer SHOULD implement this if possible.
        // A reader MUST NOT depend on this." (Alphabetical order happens to be descr, fortran_order,
        // shape.) Values go through repr(), which is why a 1-tuple keeps its trailing comma: (3,).
        private static string FormatHeaderDict(Dictionary<string, object> d)
        {
            var sb = new StringBuilder("{");
            foreach (string key in d.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                sb.Append('\'').Append(key).Append("': ");
                AppendRepr(sb, d[key]);
                sb.Append(", "); // NumPy emits a trailing comma after every entry, including the last
            }
            return sb.Append('}').ToString();
        }

        private static void AppendRepr(StringBuilder sb, object value)
        {
            switch (value)
            {
                case string s:
                    sb.Append('\'').Append(s).Append('\'');
                    return;
                case bool b:
                    sb.Append(b ? "True" : "False");
                    return;
                case long[] shape:
                    AppendShapeTuple(sb, shape);
                    return;
                case int[] ishape:
                    AppendShapeTuple(sb, Array.ConvertAll(ishape, x => (long)x));
                    return;
                default:
                    sb.Append(Convert.ToString(value, CultureInfo.InvariantCulture));
                    return;
            }
        }

        private static void AppendShapeTuple(StringBuilder sb, long[] shape)
        {
            sb.Append('(');
            for (int i = 0; i < shape.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append(shape[i].ToString(CultureInfo.InvariantCulture));
            }
            if (shape.Length == 1) sb.Append(','); // Python's 1-tuple repr: (3,) — never (3)
            sb.Append(')');
        }

        // Attach magic + length field + padding — NumPy's _wrap_header().
        private static byte[] WrapHeader(string header, FormatVersion version)
        {
            byte[] headerBytes;
            try
            {
                headerBytes = version.StrictHeaderEncoding.GetBytes(header);
            }
            catch (EncoderFallbackException)
            {
                // Not encodable in latin-1 -> the caller falls through to version 3.0 (UTF-8), the same
                // way NumPy catches UnicodeEncodeError.
                throw new UnicodeEncodeException();
            }

            int hlen = headerBytes.Length + 1; // + the terminating newline
            int prefixSize = MagicLen + version.HeaderLengthSize;

            // Pad so magic + length + header lands on an ARRAY_ALIGN boundary. Note this is never zero:
            // an already-aligned header gets a FULL 64 bytes. NumPy does exactly this, and matching it
            // is what makes the output byte-identical.
            int padlen = ArrayAlign - ((prefixSize + hlen) % ArrayAlign);
            long totalHeaderLen = (long)hlen + padlen;

            if (totalHeaderLen > version.MaxHeaderLength)
                throw new InvalidOperationException($"Header length {hlen} too big for version={version}");

            var result = new byte[prefixSize + totalHeaderLen];
            int pos = 0;

            MagicPrefix.CopyTo(result.AsSpan(pos));
            pos += MagicPrefix.Length;
            result[pos++] = version.Major;
            result[pos++] = version.Minor;

            if (version.HeaderLengthSize == 4)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(result.AsSpan(pos), (uint)totalHeaderLen);
                pos += 4;
            }
            else
            {
                BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan(pos), (ushort)totalHeaderLen);
                pos += 2;
            }

            headerBytes.CopyTo(result, pos);
            pos += headerBytes.Length;

            result.AsSpan(pos, padlen).Fill((byte)' ');
            pos += padlen;
            result[pos] = (byte)'\n';

            return result;
        }

        /// <summary>Signals a header that latin-1 cannot encode, mirroring Python's UnicodeEncodeError.</summary>
        private sealed class UnicodeEncodeException : Exception { }

        // Pick the oldest version that can hold this header — NumPy's _wrap_header_guess_version().
        private static byte[] WrapHeaderGuessVersion(string header)
        {
            try
            {
                return WrapHeader(header, FormatVersion.V1_0); // most compatible
            }
            catch (InvalidOperationException)
            {
                // Header exceeds the 2-byte length field -> needs v2.0.
            }
            catch (UnicodeEncodeException)
            {
                // Not latin-1 encodable -> v2.0 would fail the same way; fall through to v3.0.
                return WrapHeader(header, FormatVersion.V3_0);
            }

            try
            {
                return WrapHeader(header, FormatVersion.V2_0);
            }
            catch (UnicodeEncodeException)
            {
                return WrapHeader(header, FormatVersion.V3_0); // UTF-8
            }
        }

        /// <summary>
        ///     Write an array header — NumPy's <c>_write_array_header()</c>. Leaves the stream exactly at
        ///     the (64-byte aligned) start of the data.
        /// </summary>
        /// <param name="stream">Target stream.</param>
        /// <param name="d">Header dict: <c>descr</c>, <c>fortran_order</c> and <c>shape</c>.</param>
        /// <param name="version">The version to write, or null to pick the oldest that fits.</param>
        public static void WriteArrayHeader(Stream stream, Dictionary<string, object> d, FormatVersion? version = null)
        {
            string header = FormatHeaderDict(d);

            // Reserve room to rewrite the growth axis in place when the file is later extended: the
            // first axis in C-order, the last in F-order. A 0-d array has no axis to grow.
            long[] shape = ShapeOf(d);
            if (shape.Length > 0)
            {
                bool fortranOrder = d.TryGetValue("fortran_order", out object fo) && fo is bool b && b;
                long growthAxisValue = shape[fortranOrder ? shape.Length - 1 : 0];
                int padding = GrowthAxisMaxDigits - growthAxisValue.ToString(CultureInfo.InvariantCulture).Length;
                if (padding > 0)
                    header += new string(' ', padding);
            }

            byte[] wrapped;
            if (version.HasValue)
            {
                CheckVersion(version.Value);
                try
                {
                    wrapped = WrapHeader(header, version.Value);
                }
                catch (UnicodeEncodeException)
                {
                    throw new InvalidOperationException(
                        $"Header cannot be encoded in latin-1 for version={version.Value}; use version (3, 0).");
                }
            }
            else
            {
                wrapped = WrapHeaderGuessVersion(header);
            }

            stream.Write(wrapped, 0, wrapped.Length);
        }

        private static long[] ShapeOf(Dictionary<string, object> d)
        {
            if (!d.TryGetValue("shape", out object s))
                throw new ArgumentException("Header dict is missing the 'shape' key.", nameof(d));
            switch (s)
            {
                case long[] l: return l;
                case int[] i: return Array.ConvertAll(i, x => (long)x);
                default: throw new ArgumentException($"Header 'shape' must be long[] or int[], got {s?.GetType().Name ?? "null"}.", nameof(d));
            }
        }

        #endregion

        #region Header — reading

        /// <summary>A validated header: everything needed to materialize the array that follows.</summary>
        internal readonly struct HeaderData
        {
            public readonly long[] Shape;
            public readonly bool FortranOrder;
            public readonly DtypeInfo Dtype;

            public HeaderData(long[] shape, bool fortranOrder, DtypeInfo dtype)
            {
                Shape = shape;
                FortranOrder = fortranOrder;
                Dtype = dtype;
            }

            /// <summary>Element count. A 0-d array holds exactly one element, not zero.</summary>
            public long Count
            {
                get
                {
                    if (Shape.Length == 0) return 1;
                    long count = 1;
                    foreach (long d in Shape)
                        count = checked(count * d);
                    return count;
                }
            }
        }

        /// <summary>
        ///     Read and validate a header — NumPy's <c>_read_array_header()</c>. Leaves the stream at the
        ///     start of the data.
        /// </summary>
        /// <param name="stream">A stream positioned just past the magic and version bytes.</param>
        /// <param name="version">The version read from the magic.</param>
        /// <param name="maxHeaderSize">
        ///     Reject headers longer than this. The default guards against a header crafted to make
        ///     parsing pathologically expensive; raise it only for files you trust.
        /// </param>
        internal static HeaderData ReadArrayHeader(Stream stream, FormatVersion version, long maxHeaderSize = MaxHeaderSize)
        {
            CheckVersion(version);

            byte[] lenBytes = ReadBytes(stream, version.HeaderLengthSize, "array header length");
            uint headerLength = version.HeaderLengthSize == 2
                ? BinaryPrimitives.ReadUInt16LittleEndian(lenBytes)
                : BinaryPrimitives.ReadUInt32LittleEndian(lenBytes);

            if (headerLength > int.MaxValue)
                throw new FormatException($"Header length {headerLength} exceeds the maximum readable size.");

            byte[] headerBytes = ReadBytes(stream, (int)headerLength, "array header");
            string header = version.HeaderEncoding.GetString(headerBytes);

            if (header.Length > maxHeaderSize)
                throw new FormatException(
                    $"Header info length ({header.Length}) is large and may not be safe to load securely.\n" +
                    "To allow loading, adjust `maxHeaderSize` or fully trust the `.npy` file using `allow_pickle=True`.\n" +
                    "For safety against large resource use or crashes, sandboxing may be necessary.");

            object parsed;
            try
            {
                // The Python 2 'L' integer suffix only ever appeared in v1.0/v2.0 files; NumPy's
                // _filter_header retry is likewise gated on version <= (2, 0).
                parsed = PyLiteral.Parse(header, allowLongSuffix: version <= FormatVersion.V2_0);
            }
            catch (PyLiteral.SyntaxException e)
            {
                throw new FormatException($"Cannot parse header: {PyLiteral.Repr(header)}", e);
            }

            if (!(parsed is Dictionary<string, object> d))
                throw new FormatException($"Header is not a dictionary: {PyLiteral.Repr(parsed)}");

            if (d.Count != ExpectedKeys.Length || !ExpectedKeys.All(d.ContainsKey))
            {
                string keys = PyLiteral.Repr(d.Keys.OrderBy(k => k, StringComparer.Ordinal).Cast<object>().ToList());
                throw new FormatException($"Header does not contain the correct keys: {keys}");
            }

            if (!(d["shape"] is PyTuple shapeTuple) || !shapeTuple.All(x => x is long))
                throw new FormatException($"shape is not valid: {PyLiteral.Repr(d["shape"])}");

            if (!(d["fortran_order"] is bool fortranOrder))
                throw new FormatException($"fortran_order is not a valid bool: {PyLiteral.Repr(d["fortran_order"])}");

            var shape = new long[shapeTuple.Count];
            for (int i = 0; i < shapeTuple.Count; i++)
                shape[i] = (long)shapeTuple[i];

            return new HeaderData(shape, fortranOrder, DescrToDtype(d["descr"]));
        }

        #endregion

        #region Writing

        /// <summary>
        ///     Write an array with its header — NumPy's <c>write_array()</c>.
        /// </summary>
        /// <param name="stream">An open, writable stream. Written from its current position.</param>
        /// <param name="array">The array to write. Any layout; non-contiguous is materialized C-order.</param>
        /// <param name="version">The format version, or null (default) for the oldest that fits.</param>
        /// <param name="allowPickle">
        ///     Present for NumPy parity. NumSharp has no object dtype, so no array can reach the pickle
        ///     path and this parameter never changes the outcome.
        /// </param>
        /// <exception cref="NotSupportedException">The dtype has no NumPy equivalent (Decimal).</exception>
        public static void WriteArray(Stream stream, NDArray array, FormatVersion? version = null, bool allowPickle = true)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (version.HasValue) CheckVersion(version.Value);

            WriteArrayHeader(stream, HeaderDataFromArray(array), version);

            // NumPy writes an F-contiguous array as its transpose in C-order — the transpose of an
            // F-contiguous array is C-contiguous, so this streams the same bytes the file needs.
            bool fortran = !array.Shape.IsContiguous && array.Shape.IsFContiguous;
            WriteArrayData(stream, fortran ? array.T : array);
        }

        private static void WriteArrayData(Stream stream, NDArray array)
        {
            if (array.size == 0)
                return;

            if (array.typecode == NPTypeCode.Char)
            {
                WriteUcs4Data(stream, array);
                return;
            }

            // Only an offset-0 contiguous array can stream straight from its buffer; anything else
            // (strided / broadcast / sliced) is materialized C-order first, which is what NumPy's
            // nditer copy does. Matches NDArray.tofile's WriteBinary.
            NDArray src = array.Shape.IsContiguous && array.Shape.offset == 0 ? array : array.copy('C');

            unsafe
            {
                long len = checked(src.size * src.dtypesize);
                using (var ums = new UnmanagedMemoryStream((byte*)src.Storage.Address, len))
                    ums.CopyTo(stream, (int)Math.Min(len, BufferSize));
            }

            GC.KeepAlive(src);
        }

        // NumSharp's Char is one 2-byte UTF-16 code unit; NumPy's '<U1' is one 4-byte UCS-4 code point.
        // Widen element by element. A lone surrogate cannot be widened to a code point, so reject it
        // rather than write a file NumPy would read back as mojibake.
        private static unsafe void WriteUcs4Data(Stream stream, NDArray array)
        {
            NDArray src = array.Shape.IsContiguous && array.Shape.offset == 0 ? array : array.copy('C');
            long n = src.size;
            char* p = (char*)src.Storage.Address;

            var buffer = new byte[Math.Min(n, BufferSize / 4) * 4];
            int filled = 0;

            for (long i = 0; i < n; i++)
            {
                char c = p[i];
                if (char.IsSurrogate(c))
                    throw new NotSupportedException(
                        $"Char at flat index {i} is the surrogate U+{(int)c:X4}. A NumPy '<U1' element holds one " +
                        "whole code point, so a surrogate pair cannot be represented and would corrupt the file.");

                BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(filled), c);
                filled += 4;

                if (filled == buffer.Length)
                {
                    stream.Write(buffer, 0, filled);
                    filled = 0;
                }
            }

            if (filled > 0)
                stream.Write(buffer, 0, filled);

            GC.KeepAlive(src);
        }

        #endregion

        #region Reading

        /// <summary>
        ///     Read an array — NumPy's <c>read_array()</c>.
        /// </summary>
        /// <param name="stream">A stream positioned at the magic string.</param>
        /// <param name="allowPickle">
        ///     Whether the caller trusts this file. NumSharp cannot unpickle either way, but this
        ///     selects which error an object array produces and, per NumPy, lifts
        ///     <paramref name="maxHeaderSize"/>.
        /// </param>
        /// <param name="maxHeaderSize">Reject headers larger than this. Ignored when <paramref name="allowPickle"/>.</param>
        public static NDArray ReadArray(Stream stream, bool allowPickle = false, long maxHeaderSize = MaxHeaderSize)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));

            // allow_pickle means the caller has declared the file trusted, so the header guard — which
            // exists only to bound the cost of parsing hostile input — is moot.
            if (allowPickle)
                maxHeaderSize = long.MaxValue;

            FormatVersion version = ReadMagic(stream);
            CheckVersion(version);

            HeaderData header = ReadArrayHeader(stream, version, maxHeaderSize);

            if (header.Dtype.HasObject)
            {
                if (!allowPickle)
                    throw new FormatException("Object arrays cannot be loaded when allow_pickle=False");
                throw new NotSupportedException(
                    "Object arrays are stored as a Python pickle stream, which NumSharp cannot read. " +
                    "Re-save the array from NumPy with a concrete numeric dtype.");
            }

            foreach (long dim in header.Shape)
                if (dim < 0)
                    throw new FormatException("negative dimensions are not allowed");

            long count = header.Count;
            var result = new NDArray(header.Dtype.TypeCode, new Shape(header.Shape));

            if (count > 0)
                ReadArrayData(stream, result, header.Dtype, count);

            // The file holds F-order data: read it flat, give it the reversed shape, then transpose —
            // NumPy's `array.reshape(shape[::-1]).transpose()`.
            if (header.FortranOrder && header.Shape.Length > 1)
            {
                var reversed = new long[header.Shape.Length];
                for (int i = 0; i < header.Shape.Length; i++)
                    reversed[i] = header.Shape[header.Shape.Length - 1 - i];
                result = result.reshape(reversed).T;
            }

            return result;
        }

        private static unsafe void ReadArrayData(Stream stream, NDArray array, DtypeInfo dtype, long count)
        {
            long fileBytes = checked(count * dtype.FileItemSize);
            if (fileBytes == 0)
                return;

            // Read in BUFFER_SIZE chunks: a ZipExtFile or network stream can return short reads, and
            // NumPy notes crc32 breaks past 2**32 bytes on gzip streams. Chunking also caps peak memory.
            // The chunk is a whole number of elements so no element ever straddles two chunks.
            int chunkElements = Math.Max(1, BufferSize / dtype.FileItemSize);
            int chunkBytes = chunkElements * dtype.FileItemSize;
            var buffer = new byte[(int)Math.Min(fileBytes, chunkBytes)];

            byte* dst = (byte*)array.Storage.Address;
            long written = 0;
            long remaining = fileBytes;

            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                int got = ReadInto(stream, buffer, toRead);
                if (got != toRead)
                    throw new FormatException($"EOF: reading array data, expected {fileBytes} bytes got {fileBytes - remaining + got}");

                if (dtype.NeedsSwap)
                    SwapBytes(buffer, got, dtype.SwapUnit);

                written += ConvertChunk(buffer, got, dst + written, dtype);
                remaining -= got;
            }

            GC.KeepAlive(array);
        }

        // Copy one chunk of file bytes into the array, converting where NumPy's element width differs
        // from NumSharp's. Returns the number of destination bytes written.
        private static unsafe long ConvertChunk(byte[] buffer, int length, byte* dst, DtypeInfo dtype)
        {
            switch (dtype.Conversion)
            {
                case ElementConversion.None:
                    Marshal.Copy(buffer, 0, (IntPtr)dst, length);
                    return length;

                case ElementConversion.Ucs4:
                {
                    // 4-byte UCS-4 code point -> 2-byte UTF-16 code unit. Anything outside the BMP needs
                    // a surrogate pair and cannot fit a single Char.
                    int n = length / 4;
                    char* c = (char*)dst;
                    for (int i = 0; i < n; i++)
                    {
                        uint cp = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(i * 4));
                        if (cp > 0xFFFF)
                            throw new NotSupportedException(
                                $"'<U1' element U+{cp:X} is outside the Basic Multilingual Plane and does not fit " +
                                "NumSharp's 2-byte Char. Load this file in NumPy and convert it instead.");
                        if (cp >= 0xD800 && cp <= 0xDFFF)
                            throw new FormatException($"'<U1' element U+{cp:X4} is an unpaired surrogate and is not a valid code point.");
                        c[i] = (char)cp;
                    }
                    return (long)n * 2;
                }

                case ElementConversion.Complex64:
                {
                    // Two float32 -> two float64 (System.Numerics.Complex is real-then-imaginary, the
                    // same order NumPy stores).
                    int n = length / 4; // float32 components, not elements
                    double* d = (double*)dst;
                    for (int i = 0; i < n; i++)
                        d[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(buffer.AsSpan(i * 4)));
                    return (long)n * 8;
                }

                default:
                    throw new NotSupportedException($"Unknown element conversion {dtype.Conversion}.");
            }
        }

        // Reverse each `unit`-sized group in place. The unit is a component, not an element: '>c16' is
        // two 8-byte doubles, so it swaps in 8s.
        private static void SwapBytes(byte[] buffer, int length, int unit)
        {
            for (int i = 0; i + unit <= length; i += unit)
                buffer.AsSpan(i, unit).Reverse();
        }

        #endregion

        #region Stream helpers

        /// <summary>
        ///     Read exactly <paramref name="count"/> bytes — NumPy's <c>_read_bytes()</c>.
        /// </summary>
        /// <exception cref="FormatException">The stream ended early.</exception>
        private static byte[] ReadBytes(Stream stream, int count, string what)
        {
            var buffer = new byte[count];
            int got = ReadInto(stream, buffer, count);
            if (got != count)
                throw new FormatException($"EOF: reading {what}, expected {count} bytes got {got}");
            return buffer;
        }

        // Loop until the buffer is full or the stream ends: Stream.Read may legally return fewer bytes
        // than asked for, and ZipExtFile routinely does.
        private static int ReadInto(Stream stream, byte[] buffer, int count)
        {
            int total = 0;
            while (total < count)
            {
                int read = stream.Read(buffer, total, count - total);
                if (read == 0) break; // EOF
                total += read;
            }
            return total;
        }

        /// <summary>Whether the stream begins with the .npy magic. The position is restored.</summary>
        public static bool IsNpyFile(Stream stream) => StartsWith(stream, MagicPrefix);

        /// <summary>Whether the stream begins with a ZIP signature — a .npz. The position is restored.</summary>
        public static bool IsNpzFile(Stream stream) => StartsWith(stream, ZipPrefix) || StartsWith(stream, ZipSuffix);

        private static bool StartsWith(Stream stream, ReadOnlySpan<byte> prefix)
        {
            if (stream == null) throw new ArgumentNullException(nameof(stream));
            if (!stream.CanSeek) throw new ArgumentException("Stream must be seekable to detect its type.", nameof(stream));

            long pos = stream.Position;
            try
            {
                Span<byte> head = stackalloc byte[8];
                head = head.Slice(0, prefix.Length);
                int got = 0;
                while (got < prefix.Length)
                {
                    int read = stream.Read(head.Slice(got));
                    if (read == 0) break;
                    got += read;
                }
                return got == prefix.Length && head.SequenceEqual(prefix);
            }
            finally
            {
                stream.Position = pos;
            }
        }

        #endregion
    }
}
