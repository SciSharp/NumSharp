using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        /// <summary>
        ///     Construct an array from data in a text or binary file. Efficient for binary data of a known
        ///     dtype, and parses simply-formatted text files. Data written with <see cref="NDArray.tofile(string,string,string)"/>
        ///     can be read back with this function.
        /// </summary>
        /// <param name="file">A filename.</param>
        /// <param name="dtype">Element type. For binary files it sets the item size; defaults to <see cref="double"/>.</param>
        /// <param name="count">Number of items to read. <c>-1</c> (default) reads the whole file.</param>
        /// <param name="sep">
        ///     Separator between items for a text file. Empty (default) reads binary. A separator containing
        ///     spaces matches runs of whitespace; a whitespace-only separator splits on any whitespace run.
        /// </param>
        /// <param name="offset">Bytes to skip from the file's current position. Binary files only.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html</remarks>
        public static NDArray fromfile(string file, NPTypeCode dtype, int count = -1, string sep = "", long offset = 0)
            => fromfile(file, dtype.AsType(), count, sep, offset);

        /// <inheritdoc cref="fromfile(string,NPTypeCode,int,string,long)"/>
        public static NDArray fromfile(string file, Type dtype = null, int count = -1, string sep = "", long offset = 0)
        {
            using var fs = new FileStream(file, FileMode.Open, FileAccess.Read);
            return fromfile(fs, dtype, count, sep, offset);
        }

        /// <summary>
        ///     Construct an array from data in an open <see cref="Stream"/> (the file-object form). Reads from
        ///     the stream's current position and leaves it open.
        /// </summary>
        /// <inheritdoc cref="fromfile(string,NPTypeCode,int,string,long)"/>
        public static NDArray fromfile(Stream stream, NPTypeCode dtype, int count = -1, string sep = "", long offset = 0)
            => fromfile(stream, dtype.AsType(), count, sep, offset);

        /// <inheritdoc cref="fromfile(Stream,NPTypeCode,int,string,long)"/>
        public static NDArray fromfile(Stream stream, Type dtype = null, int count = -1, string sep = "", long offset = 0)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            NPTypeCode tc = (dtype ?? typeof(double)).GetTypeCode();

            if (string.IsNullOrEmpty(sep))
                return FromFileBinary(stream, tc, count, offset);

            // NumPy raises TypeError verbatim when offset is combined with a text separator.
            if (offset != 0)
                throw new TypeError("'offset' argument only permitted for binary files");

            return FromFileText(stream, tc, count, sep);
        }

        // ---- binary ------------------------------------------------------------------

        private static NDArray FromFileBinary(Stream stream, NPTypeCode tc, int count, long offset)
        {
            if (offset != 0)
                SkipBytes(stream, offset);

            int itemsize = tc.SizeOf();
            // count<0 => whole remainder; else count items, but never past EOF (NumPy reads what's there).
            long want = count < 0 ? long.MaxValue : (long)count * itemsize;

            byte[] bytes;
            if (stream.CanSeek)
            {
                // Fast path: the remaining length is known, so allocate ONE exact buffer and read straight
                // into it — a single disk->buffer copy that the array then views (pinned), matching NumPy's
                // read-into-result and avoiding the MemoryStream growth + ToArray double copy.
                long remaining = Math.Max(0, stream.Length - stream.Position);
                long take = Math.Min(want, remaining);
                take -= take % itemsize;
                bytes = new byte[take]; // >2GB is unrepresentable as a managed byte[] regardless (throws)
                ReadExact(stream, bytes, (int)take);
            }
            else
            {
                bytes = ReadUpTo(stream, want);
                int usable = bytes.Length - (bytes.Length % itemsize);
                if (usable != bytes.Length)
                    Array.Resize(ref bytes, usable);
            }
            return BytesToArray(bytes, tc);
        }

        private static void ReadExact(Stream s, byte[] buf, int count)
        {
            int off = 0;
            while (off < count)
            {
                int n = s.Read(buf, off, count - off);
                if (n <= 0) break; // EOF earlier than the reported length; use what we got
                off += n;
            }
        }

        private static void SkipBytes(Stream s, long offset)
        {
            if (s.CanSeek)
            {
                s.Seek(offset, SeekOrigin.Current);
                return;
            }
            // Non-seekable: read and discard.
            byte[] buf = new byte[Math.Min(offset, 81920)];
            long left = offset;
            while (left > 0)
            {
                int n = s.Read(buf, 0, (int)Math.Min(buf.Length, left));
                if (n <= 0) break;
                left -= n;
            }
        }

        private static byte[] ReadUpTo(Stream s, long max)
        {
            using var ms = new MemoryStream();
            byte[] buf = new byte[81920];
            long total = 0;
            while (total < max)
            {
                int toRead = (int)Math.Min(buf.Length, max - total);
                int n = s.Read(buf, 0, toRead);
                if (n <= 0) break;
                ms.Write(buf, 0, n);
                total += n;
            }
            return ms.ToArray();
        }

        // Reinterpret the byte buffer (already trimmed to whole items) as a 1-D array of <paramref name="tc"/>.
        private static NDArray BytesToArray(byte[] bytes, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return new NDArray(new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromBuffer(bytes, false)));
                case NPTypeCode.Byte: return new NDArray(new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromBuffer(bytes, false)));
                case NPTypeCode.SByte: return new NDArray(new ArraySlice<sbyte>(UnmanagedMemoryBlock<sbyte>.FromBuffer(bytes, false)));
                case NPTypeCode.Int16: return new NDArray(new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromBuffer(bytes, false)));
                case NPTypeCode.UInt16: return new NDArray(new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromBuffer(bytes, false)));
                case NPTypeCode.Int32: return new NDArray(new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromBuffer(bytes, false)));
                case NPTypeCode.UInt32: return new NDArray(new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromBuffer(bytes, false)));
                case NPTypeCode.Int64: return new NDArray(new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromBuffer(bytes, false)));
                case NPTypeCode.UInt64: return new NDArray(new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromBuffer(bytes, false)));
                case NPTypeCode.Char: return new NDArray(new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromBuffer(bytes, false)));
                case NPTypeCode.Half: return new NDArray(new ArraySlice<Half>(UnmanagedMemoryBlock<Half>.FromBuffer(bytes, false)));
                case NPTypeCode.Single: return new NDArray(new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromBuffer(bytes, false)));
                case NPTypeCode.Double: return new NDArray(new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromBuffer(bytes, false)));
                case NPTypeCode.Decimal: return new NDArray(new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromBuffer(bytes, false)));
                case NPTypeCode.Complex: return new NDArray(new ArraySlice<Complex>(UnmanagedMemoryBlock<Complex>.FromBuffer(bytes, false)));
                default: throw new NotSupportedException(tc.ToString());
            }
        }

        // ---- text --------------------------------------------------------------------

        private static NDArray FromFileText(Stream stream, NPTypeCode tc, int count, string sep)
        {
            string text;
            using (var reader = new StreamReader(stream, Encoding.ASCII, false, 4096, leaveOpen: true))
                text = reader.ReadToEnd();

            string[] tokens = SplitTokens(text, sep, out bool nonWhitespaceSep);
            string[] items = SelectTokens(tokens, count, nonWhitespaceSep);
            return TokensToArray(items, items.Length, tc);
        }

        // Split the text into raw item tokens. A whitespace-only separator splits on any whitespace run,
        // so no empty items are ever produced. Otherwise we split on the separator's non-whitespace core
        // and trim surrounding whitespace off each token (NumPy's swab_separator: whitespace around the
        // separator is a wildcard). Empty tokens are KEPT here — SelectTokens applies NumPy's rule that
        // only a single trailing separator is tolerated.
        private static string[] SplitTokens(string text, string sep, out bool nonWhitespaceSep)
        {
            var coreChars = new List<char>(sep.Length);
            foreach (char c in sep)
                if (!char.IsWhiteSpace(c)) coreChars.Add(c);

            if (coreChars.Count == 0)
            {
                nonWhitespaceSep = false;
                return text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
            }

            nonWhitespaceSep = true;
            string core = new string(coreChars.ToArray());
            string[] raw = text.Split(new[] { core }, StringSplitOptions.None);
            for (int i = 0; i < raw.Length; i++)
                raw[i] = raw[i].Trim();
            return raw;
        }

        // Select the items to parse, matching NumPy's text reader: read items until `count` is reached
        // (count < 0 = all). NumPy consumes value/sep/value/sep… and tolerates exactly one optional
        // trailing separator — a single empty item at the very end. Any OTHER empty item (leading,
        // internal, or a second trailing one) means it hit a separator where it expected a value, which
        // NumPy reports as unmatched data. A whitespace separator never yields empties, so this only
        // constrains the non-whitespace path. `count` is applied first: a bad/empty item beyond the
        // requested count is never examined, exactly as NumPy stops reading once it has `count` items.
        private static string[] SelectTokens(string[] tokens, int count, bool nonWhitespaceSep)
        {
            int want = count < 0 ? int.MaxValue : count;
            var items = new List<string>(count < 0 ? tokens.Length : Math.Min(count, tokens.Length));
            for (int i = 0; i < tokens.Length; i++)
            {
                if (items.Count >= want)
                    break; // count items already read; NumPy does not look at what follows

                string t = tokens[i];
                if (t.Length == 0)
                {
                    if (nonWhitespaceSep && i == tokens.Length - 1)
                        break; // a single trailing separator is allowed
                    throw Unmatched();
                }

                items.Add(t);
            }
            return items.ToArray();
        }

        private static NDArray TokensToArray(string[] tokens, int take, NPTypeCode tc)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: { var a = new bool[take]; for (int i = 0; i < take; i++) a[i] = ParseLong(tokens[i]) != 0; return array(a); }
                case NPTypeCode.Byte: { var a = new byte[take]; for (int i = 0; i < take; i++) a[i] = unchecked((byte)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.SByte: { var a = new sbyte[take]; for (int i = 0; i < take; i++) a[i] = unchecked((sbyte)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.Int16: { var a = new short[take]; for (int i = 0; i < take; i++) a[i] = unchecked((short)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.UInt16: { var a = new ushort[take]; for (int i = 0; i < take; i++) a[i] = unchecked((ushort)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.Int32: { var a = new int[take]; for (int i = 0; i < take; i++) a[i] = unchecked((int)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.UInt32: { var a = new uint[take]; for (int i = 0; i < take; i++) a[i] = unchecked((uint)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.Int64: { var a = new long[take]; for (int i = 0; i < take; i++) a[i] = ParseLong(tokens[i]); return array(a); }
                case NPTypeCode.UInt64: { var a = new ulong[take]; for (int i = 0; i < take; i++) a[i] = ParseULong(tokens[i]); return array(a); }
                case NPTypeCode.Char: { var a = new char[take]; for (int i = 0; i < take; i++) a[i] = (char)unchecked((ushort)ParseLong(tokens[i])); return array(a); }
                case NPTypeCode.Half: { var a = new Half[take]; for (int i = 0; i < take; i++) a[i] = (Half)ParseDouble(tokens[i]); return array(a); }
                case NPTypeCode.Single: { var a = new float[take]; for (int i = 0; i < take; i++) a[i] = (float)ParseDouble(tokens[i]); return array(a); }
                case NPTypeCode.Double: { var a = new double[take]; for (int i = 0; i < take; i++) a[i] = ParseDouble(tokens[i]); return array(a); }
                case NPTypeCode.Decimal: { var a = new decimal[take]; for (int i = 0; i < take; i++) a[i] = ParseDecimal(tokens[i]); return array(a); }
                case NPTypeCode.Complex: { var a = new Complex[take]; for (int i = 0; i < take; i++) a[i] = ParseComplex(tokens[i]); return array(a); }
                default: throw new NotSupportedException(tc.ToString());
            }
        }

        // NumPy raises this verbatim both for a token it cannot parse and for a missing (empty) item.
        private static ValueError Unmatched()
            => new ValueError("string or file could not be read to its end due to unmatched data");

        private static long ParseLong(string t)
        {
            if (long.TryParse(t, NumberStyles.Integer | NumberStyles.AllowLeadingSign, CI, out long v))
                return v;
            throw Unmatched();
        }

        private static ulong ParseULong(string t)
        {
            if (ulong.TryParse(t, NumberStyles.Integer, CI, out ulong u))
                return u;
            // negatives wrap two's-complement into the unsigned range (NumPy: uint64 "-1" -> 2^64-1).
            if (long.TryParse(t, NumberStyles.Integer | NumberStyles.AllowLeadingSign, CI, out long s))
                return unchecked((ulong)s);
            throw Unmatched();
        }

        private static decimal ParseDecimal(string t)
        {
            if (decimal.TryParse(t, NumberStyles.Float, CI, out decimal d))
                return d;
            throw Unmatched();
        }

        // NumPy/C parse "nan" to the POSITIVE quiet NaN (0x7FF8…). .NET's double.NaN is the NEGATIVE
        // quiet NaN (0xFFF8…), so use the positive bit pattern to stay byte-identical to NumPy (and so a
        // narrowing cast yields float 0x7FC00000 / half 0x7E00, matching NumPy's float16/float32 "nan").
        private static readonly double PositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);

        private static double ParseDouble(string t)
        {
            switch (t.ToLowerInvariant())
            {
                case "nan": case "+nan": return PositiveNaN;
                case "-nan": return double.NaN; // .NET double.NaN == 0xFFF8… (negative qNaN)
                case "inf": case "+inf": case "infinity": case "+infinity": return double.PositiveInfinity;
                case "-inf": case "-infinity": return double.NegativeInfinity;
            }
            if (double.TryParse(t, NumberStyles.Float, CI, out double v))
                return v;
            throw Unmatched();
        }

        // Parse a complex literal: "a+bj" / "a-bj" / "bj" / "a" (optionally wrapped in parentheses, so the
        // "(1+2j)" form NumPy's tofile writes round-trips here — NumPy's own text reader only accepts the
        // bare "1+2j" form, so this is a superset).
        private static Complex ParseComplex(string t)
        {
            string s = t.Trim();
            if (s.Length >= 2 && s[0] == '(' && s[s.Length - 1] == ')')
                s = s.Substring(1, s.Length - 2).Trim();

            if (s.Length == 0)
                throw Unmatched();

            char last = s[s.Length - 1];
            if (last != 'j' && last != 'J')
                return new Complex(ParseDouble(s), 0.0);

            string body = s.Substring(0, s.Length - 1);
            int split = ImaginarySplit(body);
            if (split < 0)
            {
                // pure imaginary: "j" / "+j" / "-j" / "2j"
                double im = body.Length == 0 ? 1.0
                          : body == "+" ? 1.0
                          : body == "-" ? -1.0
                          : ParseDouble(body);
                return new Complex(0.0, im);
            }

            double real = ParseDouble(body.Substring(0, split));
            string imagStr = body.Substring(split);
            double imag = imagStr == "+" ? 1.0 : imagStr == "-" ? -1.0 : ParseDouble(imagStr);
            return new Complex(real, imag);
        }

        // Index of the sign that separates real from imaginary — a '+'/'-' that is neither the leading
        // sign nor part of an exponent ("1.5e-3+2j").
        private static int ImaginarySplit(string body)
        {
            for (int i = body.Length - 1; i > 0; i--)
            {
                char c = body[i];
                if (c == '+' || c == '-')
                {
                    char prev = body[i - 1];
                    if (prev == 'e' || prev == 'E') continue; // exponent sign
                    return i;
                }
            }
            return -1;
        }
    }
}
