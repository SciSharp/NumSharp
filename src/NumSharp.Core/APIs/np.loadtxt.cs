using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.4.2 reference: numpy/lib/_npyio_impl.py::loadtxt -> _read, plus the C text reader
        // (numpy/_core/src/multiarray/textreading/{tokenize.cpp,conversions.c,str_to_int.c,rows.c}).
        //
        // The C parser always produces a 2-D (rows, cols) array; `ndmin` then squeezes extraneous size-1
        // axes or expands. bool parses through int64 (so "0"/"1", not "True"); ints range-check (int8 "200"
        // -> error) and unsigned reject negatives; float uses PyOS_string_to_double (rejects hex / "_" /
        // trailing junk, accepts case-insensitive inf/infinity/nan); complex is `to_complex_int`. Fields are
        // parsed straight from the line's char span (no per-field string allocation on the common path).

        #region loadtxt (public overloads)

        /// <summary>
        ///     Load data from a text file into a 1-D or 2-D array.
        /// </summary>
        /// <param name="fname">Path to the file; a <c>.gz</c> name is transparently decompressed.</param>
        /// <param name="dtype">Element type of the result (default <see cref="double"/>).</param>
        /// <param name="comments">
        ///     String marking the start of a comment (rest of the line ignored); a multi-character string is
        ///     stripped from each line. <c>null</c> disables comments.
        /// </param>
        /// <param name="delimiter">Column separator. <c>null</c> (default) splits on runs of whitespace; otherwise a single character.</param>
        /// <param name="converters">
        ///     Per-field parser(s): a <see cref="Func{String, Object}"/> applied to every column, or an
        ///     <see cref="IDictionary{TKey, TValue}"/> mapping a column index to a parser. <c>null</c> uses the dtype's parser.
        /// </param>
        /// <param name="skiprows">Skip this many leading lines (including comments/blanks).</param>
        /// <param name="usecols">Which columns to read (0-based, negatives count from the end). <c>null</c> reads all.</param>
        /// <param name="unpack">If true, transpose the result so columns can be unpacked as separate arrays.</param>
        /// <param name="ndmin">Minimum dimensions of the result (0, 1 or 2); otherwise size-1 axes are squeezed.</param>
        /// <param name="encoding">Text encoding used to decode the file (default UTF-8).</param>
        /// <param name="max_rows">Read at most this many data rows after <paramref name="skiprows"/> (blank/comment lines don't count).</param>
        /// <param name="quotechar">Quote character; delimiters and comments inside a quoted field are literal. <c>null</c> disables quoting.</param>
        /// <remarks>
        ///     Parity with NumPy 2.4.2's <c>np.loadtxt</c>. Reads back what <see cref="np.savetxt(string, NDArray, string, string, string, string, string, string, string)"/> writes.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
        /// </remarks>
        public static NDArray loadtxt(string fname, NPTypeCode dtype = NPTypeCode.Double, string comments = "#",
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (fname is null) throw new ArgumentNullException(nameof(fname));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows);
            using var lines = OpenTextFileLines(fname, encoding).GetEnumerator();
            return LoadTxtCore(lines, cfg, unpack);
        }

        /// <inheritdoc cref="loadtxt(string,NPTypeCode,string,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(string fname, Type dtype, string comments = "#",
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
            => loadtxt(fname, (dtype ?? typeof(double)).GetTypeCode(), comments, delimiter, converters, skiprows,
                usecols, unpack, ndmin, encoding, max_rows, quotechar);

        /// <summary>Load data from an open text <see cref="Stream"/> (read from the current position; left open).</summary>
        /// <inheritdoc cref="loadtxt(string,NPTypeCode,string,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(Stream stream, NPTypeCode dtype = NPTypeCode.Double, string comments = "#",
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows);
            using var reader = new StreamReader(stream, ResolveSaveTxtEncoding(encoding), true, 1 << 16, leaveOpen: true);
            using var lines = ReadLines(reader).GetEnumerator();
            return LoadTxtCore(lines, cfg, unpack);
        }

        /// <summary>Load data from an open <see cref="TextReader"/> (left open).</summary>
        /// <inheritdoc cref="loadtxt(string,NPTypeCode,string,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(TextReader reader, NPTypeCode dtype = NPTypeCode.Double, string comments = "#",
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, int? max_rows = null, string quotechar = null)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows);
            using var lines = ReadLines(reader).GetEnumerator();
            return LoadTxtCore(lines, cfg, unpack);
        }

        /// <summary>Load data from a sequence of lines (each string is one or more newline-separated lines).</summary>
        /// <inheritdoc cref="loadtxt(string,NPTypeCode,string,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(IEnumerable<string> lines, NPTypeCode dtype = NPTypeCode.Double, string comments = "#",
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, int? max_rows = null, string quotechar = null)
        {
            if (lines is null) throw new ArgumentNullException(nameof(lines));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows);
            using var it = FlattenLines(lines).GetEnumerator();
            return LoadTxtCore(it, cfg, unpack);
        }

        #endregion

        #region config + input normalization

        private struct LoadTxtConfig
        {
            public NPTypeCode Tc;
            public int ElemSize;
            public char? Delimiter;      // null = whitespace mode
            public bool Whitespace;
            public char? Comment;        // single-char comment (fast path)
            public string CommentMulti;  // multi-char comment stripped per line (else null)
            public char? Quote;
            public int SkipRows;
            public int MaxRows;          // -1 = all
            public int[] UseCols;        // null = all
            public int Ndmin;
            public Func<string, object> AllConverter;
            public IDictionary<int, Func<string, object>> DictConverter;
            public bool HasConverters => AllConverter != null || DictConverter != null;
        }

        private static LoadTxtConfig BuildLoadTxtConfig(NPTypeCode dtype, string comments, string delimiter,
            object converters, int[] usecols, int ndmin, int? max_rows, string quotechar, int skiprows)
        {
            if (ndmin != 0 && ndmin != 1 && ndmin != 2)
                throw new ValueError($"Illegal value of ndmin keyword: {ndmin}");
            if (skiprows < 0) throw new ValueError("argument must be nonnegative");
            if (max_rows.HasValue && max_rows.Value < 0) throw new ValueError("argument must be nonnegative");

            var cfg = new LoadTxtConfig
            {
                Tc = dtype,
                ElemSize = dtype.SizeOf(),
                SkipRows = skiprows,
                MaxRows = max_rows ?? -1,
                UseCols = usecols,
                Ndmin = ndmin,
            };

            if (delimiter == null)
            {
                cfg.Whitespace = true;
                cfg.Delimiter = null;
            }
            else
            {
                ValidateControlChar(delimiter, "delimiter");
                cfg.Delimiter = delimiter[0];
                cfg.Whitespace = false;
            }

            if (quotechar != null)
            {
                ValidateControlChar(quotechar, "quotechar");
                cfg.Quote = quotechar[0];
            }

            if (comments != null)
            {
                if (comments.Length == 0)
                    throw new ValueError("comments cannot be an empty string. Use comments=None to disable comments.");
                if (comments.Length == 1)
                    cfg.Comment = comments[0];
                else
                {
                    if (cfg.Quote.HasValue)
                        throw new ValueError("when multiple comments or a multi-character comment is given, quotes are not supported.  In this case quotechar must be set to None.");
                    cfg.CommentMulti = comments;
                }
            }

            if (converters is Func<string, object> f) cfg.AllConverter = f;
            else if (converters is IDictionary<int, Func<string, object>> d) cfg.DictConverter = d;
            else if (converters != null)
                throw new TypeError("converters must be a Func<string, object> or IDictionary<int, Func<string, object>>.");

            return cfg;
        }

        // NumPy's control-character validation, verbatim texts.
        private static void ValidateControlChar(string s, string name)
        {
            if (s.Length == 1)
            {
                if (s[0] == '\r' || s[0] == '\n')
                    throw new TypeError($"control character '{name}' cannot be a newline (`\\r` or `\\n`).");
                return;
            }
            throw new TypeError($"Text reading control character must be a single unicode character or None; but got: '{s}'");
        }

        private static IEnumerable<string> OpenTextFileLines(string fname, string encoding)
        {
            Stream fs = new FileStream(fname, FileMode.Open, FileAccess.Read);
            Stream inner = fname.EndsWith(".gz", StringComparison.OrdinalIgnoreCase)
                ? new GZipStream(fs, CompressionMode.Decompress)
                : fs;
            using (var reader = new StreamReader(inner, ResolveSaveTxtEncoding(encoding), true, 1 << 16))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                    yield return line;
            }
        }

        private static IEnumerable<string> ReadLines(TextReader reader)
        {
            string line;
            while ((line = reader.ReadLine()) != null)
                yield return line;
        }

        // A sequence element may itself hold several newline-separated lines (NumPy feeds list/generator
        // elements straight to the tokenizer, which splits on embedded newlines).
        private static IEnumerable<string> FlattenLines(IEnumerable<string> lines)
        {
            foreach (string s in lines)
            {
                if (s == null) continue;
                int start = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    if (s[i] == '\n')
                    {
                        int end = (i > start && s[i - 1] == '\r') ? i - 1 : i;
                        yield return s.Substring(start, end - start);
                        start = i + 1;
                    }
                }
                yield return start == 0 ? s : s.Substring(start);
            }
        }

        #endregion

        #region core

        private static NDArray LoadTxtCore(IEnumerator<string> lines, LoadTxtConfig cfg, bool unpack)
        {
            for (int i = 0; i < cfg.SkipRows && lines.MoveNext(); i++) { }

            int elsize = cfg.ElemSize;
            byte[] data = null;
            long capRows = 0, nrows = 0;
            int ncols = -1, nsel = -1, rowBytes = 0;
            int[] sel = null;                 // resolved usecols (original column indices), or null = all
            Func<string, object>[] conv = null;

            // Field boundaries into `source` (the line itself, or a per-line unescaped buffer for quotes).
            var fStart = new List<int>(16);
            var fLen = new List<int>(16);
            var qbuf = cfg.Quote.HasValue ? new StringBuilder(64) : null;

            while ((cfg.MaxRows < 0 || nrows < cfg.MaxRows) && lines.MoveNext())
            {
                string line = lines.Current;
                if (cfg.CommentMulti != null)
                {
                    int c = line.IndexOf(cfg.CommentMulti, StringComparison.Ordinal);
                    if (c >= 0) line = line.Substring(0, c);
                }

                string source = TokenizeLine(line, in cfg, fStart, fLen, qbuf);
                int current = fStart.Count;
                if (current == 0)
                    continue; // blank / comment-only line — not counted toward max_rows

                if (ncols == -1)
                {
                    ncols = current;
                    sel = cfg.UseCols;
                    nsel = sel == null ? ncols : sel.Length;
                    rowBytes = checked(nsel * elsize);
                    conv = ResolveConverters(in cfg, sel, ncols);
                    capRows = 256;
                    data = new byte[checked(capRows * rowBytes)];
                }
                else if (sel == null && current != ncols)
                {
                    throw new ValueError(
                        $"the number of columns changed from {ncols} to {current} at row {nrows + 1}; " +
                        "use `usecols` to select a subset and avoid this error");
                }

                if (nrows == capRows)
                {
                    capRows *= 2;
                    Array.Resize(ref data, checked((int)(capRows * rowBytes)));
                }

                long rowBase = nrows * rowBytes;
                for (int j = 0; j < nsel; j++)
                {
                    int origCol = sel == null ? j : sel[j];
                    if (origCol < 0) origCol += current;
                    if (origCol < 0 || origCol >= current)
                        throw new ValueError($"invalid column index {sel[j]} at row {nrows + 1} with {current} columns");

                    ReadOnlySpan<char> tok = source.AsSpan(fStart[origCol], fLen[origCol]);
                    var dst = data.AsSpan((int)(rowBase + j * elsize), elsize);

                    bool ok = conv != null && conv[j] != null
                        ? PackConverted(conv[j], tok.ToString(), cfg.Tc, dst)
                        : LoadtxtWriteField(cfg.Tc, tok, dst);

                    if (!ok)
                        throw new ValueError(
                            $"could not convert string '{tok.ToString()}' to {cfg.Tc.AsNumpyDtypeName()} at row {nrows}, column {origCol + 1}.");
                }

                nrows++;
            }

            NDArray arr;
            if (nrows == 0)
            {
                arr = BytesToArray(Array.Empty<byte>(), cfg.Tc);          // NumPy's shape (0,) before ndmin
            }
            else
            {
                Array.Resize(ref data, checked((int)(nrows * rowBytes)));
                arr = BytesToArray(data, cfg.Tc).reshape((int)nrows, nsel);
            }

            arr = EnsureNdmin(arr, cfg.Ndmin);
            if (unpack) arr = arr.T;
            return arr;
        }

        // NumPy's _ensure_ndmin_ndarray: squeeze extraneous size-1 axes, then expand toward `ndmin`.
        private static NDArray EnsureNdmin(NDArray a, int ndmin)
        {
            if (a.ndim > ndmin)
                a = np.squeeze(a);
            if (a.ndim < ndmin)
            {
                if (ndmin == 1) a = np.atleast_1d(a);
                else if (ndmin == 2) a = np.atleast_2d(a).T;
            }
            return a;
        }

        private static Func<string, object>[] ResolveConverters(in LoadTxtConfig cfg, int[] sel, int ncols)
        {
            if (!cfg.HasConverters)
                return null;

            int nsel = sel == null ? ncols : sel.Length;
            var conv = new Func<string, object>[nsel];
            for (int j = 0; j < nsel; j++)
            {
                int origCol = sel == null ? j : sel[j];
                if (origCol < 0) origCol += ncols;
                if (cfg.AllConverter != null)
                    conv[j] = cfg.AllConverter;
                else if (cfg.DictConverter != null && cfg.DictConverter.TryGetValue(origCol, out var f))
                    conv[j] = f;
            }
            return conv;
        }

        private static bool PackConverted(Func<string, object> conv, string tok, NPTypeCode tc, Span<byte> dst)
        {
            object o;
            try { o = conv(tok); }
            catch { return false; }
            if (o == null) return false;

            try
            {
                switch (tc)
                {
                    case NPTypeCode.Boolean: { dst[0] = (byte)(Convert.ToBoolean(o, CI) ? 1 : 0); return true; }
                    case NPTypeCode.Byte: { byte v = Convert.ToByte(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.SByte: { sbyte v = Convert.ToSByte(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Int16: { short v = Convert.ToInt16(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.UInt16: { ushort v = Convert.ToUInt16(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Int32: { int v = Convert.ToInt32(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.UInt32: { uint v = Convert.ToUInt32(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Int64: { long v = Convert.ToInt64(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.UInt64: { ulong v = Convert.ToUInt64(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Char: { char v = Convert.ToChar(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Half: { Half v = (Half)Convert.ToDouble(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Single: { float v = Convert.ToSingle(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Double: { double v = Convert.ToDouble(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Decimal: { decimal v = Convert.ToDecimal(o, CI); MemoryMarshal.Write(dst, in v); return true; }
                    case NPTypeCode.Complex: { Complex v = o is Complex cc ? cc : new Complex(Convert.ToDouble(o, CI), 0.0); MemoryMarshal.Write(dst, in v); return true; }
                    default: return false;
                }
            }
            catch { return false; }
        }

        #endregion

        #region strict field parsers (loadtxt semantics)

        // One dtype-dispatched parser (mirrors NumPy's per-dtype set_from_ucs4 function pointer). Parses
        // `tok` and writes the value into `dst`, returning false on any parse/range failure.
        private static bool LoadtxtWriteField(NPTypeCode tc, ReadOnlySpan<char> tok, Span<byte> dst)
        {
            switch (tc)
            {
                case NPTypeCode.Double: { if (!TryDoubleLT(tok, out double v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Single: { if (!TryDoubleLT(tok, out double v)) return false; float f = (float)v; MemoryMarshal.Write(dst, in f); return true; }
                case NPTypeCode.Half: { if (!TryDoubleLT(tok, out double v)) return false; Half h = (Half)v; MemoryMarshal.Write(dst, in h); return true; }
                case NPTypeCode.Decimal: { if (!TryDecimalLT(tok, out decimal v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Complex: { if (!TryComplexLT(tok, out Complex v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Int64: { if (!TryLongLT(tok, long.MinValue, long.MaxValue, out long v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.Int32: { if (!TryLongLT(tok, int.MinValue, int.MaxValue, out long v)) return false; int x = (int)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.Int16: { if (!TryLongLT(tok, short.MinValue, short.MaxValue, out long v)) return false; short x = (short)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.SByte: { if (!TryLongLT(tok, sbyte.MinValue, sbyte.MaxValue, out long v)) return false; sbyte x = (sbyte)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.UInt64: { if (!TryULongLT(tok, ulong.MaxValue, out ulong v)) return false; MemoryMarshal.Write(dst, in v); return true; }
                case NPTypeCode.UInt32: { if (!TryULongLT(tok, uint.MaxValue, out ulong v)) return false; uint x = (uint)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.UInt16: { if (!TryULongLT(tok, ushort.MaxValue, out ulong v)) return false; ushort x = (ushort)v; MemoryMarshal.Write(dst, in x); return true; }
                case NPTypeCode.Byte: { if (!TryULongLT(tok, byte.MaxValue, out ulong v)) return false; dst[0] = (byte)v; return true; }
                case NPTypeCode.Boolean: { if (!TryLongLT(tok, long.MinValue, long.MaxValue, out long v)) return false; dst[0] = (byte)(v != 0 ? 1 : 0); return true; }
                case NPTypeCode.Char: { if (!TryULongLT(tok, ushort.MaxValue, out ulong v)) return false; char x = (char)v; MemoryMarshal.Write(dst, in x); return true; }
                default: return false;
            }
        }

        // NumPy/C parse "nan" to the POSITIVE quiet NaN (0x7FF8…). See np.fromfile.cs.
        private static readonly double LoadtxtPositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);

        private static bool TryDoubleLT(ReadOnlySpan<char> tok, out double value)
        {
            ReadOnlySpan<char> s = tok.Trim();
            if (s.Length > 0)
            {
                // inf / infinity / nan (case-insensitive, optional sign) — double.TryParse rejects these.
                ReadOnlySpan<char> body = s;
                int sign = 1;
                if (body[0] == '+') body = body.Slice(1);
                else if (body[0] == '-') { sign = -1; body = body.Slice(1); }
                if (body.Equals("inf", StringComparison.OrdinalIgnoreCase) || body.Equals("infinity", StringComparison.OrdinalIgnoreCase))
                { value = sign < 0 ? double.NegativeInfinity : double.PositiveInfinity; return true; }
                if (body.Equals("nan", StringComparison.OrdinalIgnoreCase))
                { value = sign < 0 ? double.NaN : LoadtxtPositiveNaN; return true; }
            }
            // Requires the whole token to be a valid float (NumPy's p_end == end): rejects hex / "_" / junk.
            return double.TryParse(s, NumberStyles.Float, CI, out value);
        }

        private static bool TryLongLT(ReadOnlySpan<char> tok, long min, long max, out long value)
        {
            if (long.TryParse(tok.Trim(), NumberStyles.Integer, CI, out value) && value >= min && value <= max)
                return true;
            value = 0;
            return false;
        }

        private static bool TryULongLT(ReadOnlySpan<char> tok, ulong max, out ulong value)
        {
            ReadOnlySpan<char> s = tok.Trim();
            value = 0;
            if (s.Length > 0 && s[0] == '-') return false; // unsigned rejects a negative sign outright
            return ulong.TryParse(s, NumberStyles.Integer, CI, out value) && value <= max;
        }

        private static bool TryDecimalLT(ReadOnlySpan<char> tok, out decimal value)
            => decimal.TryParse(tok.Trim(), NumberStyles.Float, CI, out value);

        // Port of NumPy's to_complex_int (conversions.c): real, then optional '+'/'-' imaginary part ending
        // in 'j', optionally wrapped in parentheses.
        private static bool TryComplexLT(ReadOnlySpan<char> tok, out Complex value)
        {
            value = default;
            ReadOnlySpan<char> s = tok.Trim();
            int i = 0, end = s.Length;
            bool paren = false;
            if (i < end && s[i] == '(') { paren = true; i++; while (i < end && char.IsWhiteSpace(s[i])) i++; }

            if (!ScanDouble(s, i, end, out double real, out int p)) return false;
            double imag;
            if (p == end)
            {
                if (paren) return false; // "(1" — unmatched paren
                value = new Complex(real, 0.0);
                return true;
            }

            if (s[p] == 'j' || s[p] == 'J')
            {
                imag = real; real = 0.0; p++;
            }
            else if (s[p] == '+' || s[p] == '-')
            {
                if (s[p] == '+') p++; // advance so "1+-2j" reads the '-' as the imaginary sign
                if (!ScanDouble(s, p, end, out imag, out p)) return false;
                if (p >= end || (s[p] != 'j' && s[p] != 'J')) return false;
                p++;
            }
            else imag = 0.0;

            if (paren)
            {
                while (p < end && char.IsWhiteSpace(s[p])) p++;
                if (p < end && s[p] == ')') p++;
                else return false;
            }
            while (p < end && char.IsWhiteSpace(s[p])) p++;
            if (p != end) return false;

            value = new Complex(real, imag);
            return true;
        }

        // Scan the longest valid float starting at `i` (used only to split complex components), reporting
        // the value and the index just past what was consumed.
        private static bool ScanDouble(ReadOnlySpan<char> s, int i, int end, out double value, out int consumed)
        {
            value = 0; consumed = i;
            int begin = i;
            if (i < end && (s[i] == '+' || s[i] == '-')) i++;

            if (MatchWord(s, i, end, "infinity")) { value = s[begin] == '-' ? double.NegativeInfinity : double.PositiveInfinity; consumed = i + 8; return true; }
            if (MatchWord(s, i, end, "inf")) { value = s[begin] == '-' ? double.NegativeInfinity : double.PositiveInfinity; consumed = i + 3; return true; }
            if (MatchWord(s, i, end, "nan")) { value = s[begin] == '-' ? double.NaN : LoadtxtPositiveNaN; consumed = i + 3; return true; }

            bool anyDigit = false;
            while (i < end && s[i] >= '0' && s[i] <= '9') { i++; anyDigit = true; }
            if (i < end && s[i] == '.') { i++; while (i < end && s[i] >= '0' && s[i] <= '9') { i++; anyDigit = true; } }
            if (!anyDigit) return false;

            if (i < end && (s[i] == 'e' || s[i] == 'E'))
            {
                int j = i + 1;
                if (j < end && (s[j] == '+' || s[j] == '-')) j++;
                if (j < end && s[j] >= '0' && s[j] <= '9') { j++; while (j < end && s[j] >= '0' && s[j] <= '9') j++; i = j; }
            }

            consumed = i;
            value = double.Parse(s.Slice(begin, i - begin), NumberStyles.Float, CI);
            return true;
        }

        private static bool MatchWord(ReadOnlySpan<char> s, int i, int end, string word)
        {
            if (i + word.Length > end) return false;
            for (int k = 0; k < word.Length; k++)
                if (char.ToLowerInvariant(s[i + k]) != word[k]) return false;
            return true;
        }

        #endregion

        #region tokenizer (port of numpy textreading/tokenize.cpp)

        // Split one line into field spans (a port of the C tokenizer). Returns the string the spans index
        // into — the line itself on the common path, or `qbuf` when a quote char requires unescaping. Blank
        // / comment-only lines yield 0 fields; whitespace mode collapses runs and drops a trailing empty
        // field; a set delimiter keeps empty fields; a quote character protects delimiters/comments.
        private static string TokenizeLine(string line, in LoadTxtConfig cfg, List<int> fStart, List<int> fLen, StringBuilder qbuf)
        {
            fStart.Clear();
            fLen.Clear();
            int n = line.Length, i = 0;
            bool ws = cfg.Whitespace;
            char delim = cfg.Delimiter ?? '\0';
            bool hasComment = cfg.Comment.HasValue;
            char comment = cfg.Comment ?? '\0';
            bool hasQuote = cfg.Quote.HasValue;
            char quote = cfg.Quote ?? '\0';
            bool lastQuoted = false;

            if (!hasQuote)
            {
                // Fast path: field spans index directly into `line` (no copying / no per-field strings).
                while (true)
                {
                    int fs;
                    if (ws)
                    {
                        while (i < n && char.IsWhiteSpace(line[i])) i++;
                        if (i >= n) break;
                        if (hasComment && line[i] == comment) break;
                        fs = i;
                    }
                    else if (hasComment && i < n && line[i] == comment)
                    {
                        fStart.Add(i); fLen.Add(0); // empty field, then the comment ends the line
                        break;
                    }
                    else fs = i;

                    bool commentEnds = false;
                    while (i < n)
                    {
                        char c = line[i];
                        if (ws) { if (char.IsWhiteSpace(c)) break; }
                        else { if (c == delim) break; }
                        if (hasComment && c == comment) { commentEnds = true; break; }
                        i++;
                    }

                    fStart.Add(fs); fLen.Add(i - fs);
                    if (commentEnds) break;
                    if (i >= n) break;
                    if (!ws)
                    {
                        i++; // consume the delimiter
                        if (i == n) { fStart.Add(n); fLen.Add(0); break; } // trailing delimiter -> empty field
                    }
                }
            }
            else
            {
                qbuf.Clear();
                while (true)
                {
                    int fs = qbuf.Length;
                    bool quoted = false;
                    if (ws)
                    {
                        while (i < n && char.IsWhiteSpace(line[i])) i++;
                        if (i >= n) break;
                        if (hasComment && line[i] == comment) break;
                    }
                    else if (hasComment && i < n && line[i] == comment)
                    {
                        fStart.Add(fs); fLen.Add(0);
                        break;
                    }

                    if (i < n && line[i] == quote)
                    {
                        quoted = true;
                        i++;
                        while (i < n)
                        {
                            char c = line[i];
                            if (c == quote)
                            {
                                if (i + 1 < n && line[i + 1] == quote) { qbuf.Append(quote); i += 2; }
                                else { i++; break; }
                            }
                            else { qbuf.Append(c); i++; }
                        }
                    }

                    bool commentEnds = false;
                    while (i < n)
                    {
                        char c = line[i];
                        if (ws) { if (char.IsWhiteSpace(c)) break; }
                        else { if (c == delim) break; }
                        if (hasComment && c == comment) { commentEnds = true; break; }
                        qbuf.Append(c); i++;
                    }

                    fStart.Add(fs); fLen.Add(qbuf.Length - fs);
                    lastQuoted = quoted;
                    if (commentEnds) break;
                    if (i >= n) break;
                    if (!ws)
                    {
                        i++;
                        if (i == n) { fStart.Add(qbuf.Length); fLen.Add(0); break; }
                    }
                }
            }

            // Trailing empty-field rules (tokenize.cpp lines 383-401):
            //  - exactly one empty unquoted field => the whole line is empty (blank/comment-only).
            //  - whitespace mode: drop a trailing empty unquoted field (Python's " 1 ".split()).
            int count = fStart.Count;
            if (count == 1)
            {
                if (fLen[0] == 0 && !lastQuoted) { fStart.Clear(); fLen.Clear(); }
            }
            else if (ws && count > 0 && fLen[count - 1] == 0 && !lastQuoted)
            {
                fStart.RemoveAt(count - 1);
                fLen.RemoveAt(count - 1);
            }

            return hasQuote ? qbuf.ToString() : line;
        }

        #endregion
    }
}
