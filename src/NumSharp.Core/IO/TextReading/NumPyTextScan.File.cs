using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Numerics;

namespace NumSharp.IO
{
    /// <summary>
    ///     A C <c>FILE*</c> over a <see cref="Stream"/>, as much of it as NumPy's <c>np.fromfile</c> text scanners use:
    ///     <c>getc</c>, <c>ungetc</c> of one byte (which need not be the byte read: <c>CDOUBLE_scan</c> pushes back an
    ///     <c>'a'</c>), and <c>ftell</c> — the position NumPy's <c>npy_PyFile_DupClose2</c> seeks the Python file object to
    ///     when the read ends.
    /// </summary>
    /// <remarks>
    ///     The stream is read ahead in pooled chunks, so the stream itself stands past the logical position until
    ///     <see cref="RestorePosition"/> seeks it back (a non-seekable stream keeps the read-ahead consumed: NumPy cannot read
    ///     one at all, it needs a file descriptor it can seek). Dispose returns the chunk buffer.
    /// </remarks>
    internal sealed class CFileReader : IDisposable
    {
        /// <summary>The size of one read-ahead chunk.</summary>
        private const int ChunkSize = 1 << 16;

        private readonly Stream _stream;
        private readonly long _startPosition;
        private byte[] _buffer;
        private int _pos, _len;
        private bool _eof;
        private int _pushed = -1;

        /// <summary>The bytes handed out so far, net of the pushed-back one (C's <c>ftell</c>, relative to the start).</summary>
        internal long Consumed { get; private set; }

        /// <summary>Wraps a stream at its current position.</summary>
        /// <param name="stream">The stream; it stays open.</param>
        internal CFileReader(Stream stream)
        {
            _stream = stream;
            _startPosition = stream.CanSeek ? stream.Position : 0;
            _buffer = ArrayPool<byte>.Shared.Rent(ChunkSize);
        }

        /// <summary>C's <c>getc</c>: the next byte, or -1 at the end of the stream.</summary>
        /// <returns>The byte (0-255) or -1.</returns>
        internal int Getc()
        {
            if (_pushed >= 0)
            {
                int c = _pushed;
                _pushed = -1;
                Consumed++;
                return c;
            }
            if (_pos == _len && !Fill())
                return -1;
            Consumed++;
            return _buffer[_pos++];
        }

        /// <summary>
        ///     C's <c>ungetc(c)</c>: <paramref name="c"/> is the next byte <see cref="Getc"/> returns. <c>ungetc(EOF)</c> fails
        ///     and changes nothing — which is also what happens to a <c>0xFF</c> byte that NumPy passed through a <c>char</c>
        ///     (<see cref="UngetcSignedChar"/>).
        /// </summary>
        /// <param name="c">The byte (0-255), or -1.</param>
        internal void Ungetc(int c)
        {
            if (c < 0)
                return;
            // NumPy's scanners never stack two pushbacks (each ungetc follows a getc), which C guarantees only one of.
            Debug.Assert(_pushed < 0, "two bytes pushed back");
            _pushed = c & 0xFF;
            Consumed--;
        }

        /// <summary>
        ///     <c>ungetc(next)</c> where NumPy held <c>next</c> in a (signed) <c>char</c>: a <c>0xFF</c> byte became
        ///     <c>(char)-1</c> == EOF, which <c>ungetc</c> refuses — that byte is lost. Every other byte is pushed back.
        /// </summary>
        /// <param name="c">The value <see cref="Getc"/> returned.</param>
        internal void UngetcSignedChar(int c)
        {
            if (c == 0xFF)
                return;
            Ungetc(c);
        }

        /// <summary>
        ///     Seeks the stream to the logical position (NumPy's <c>npy_PyFile_DupClose2</c>: the Python file object ends
        ///     where the C <c>FILE*</c> stopped). A non-seekable stream cannot be moved back.
        /// </summary>
        internal void RestorePosition()
        {
            if (_stream.CanSeek)
                _stream.Seek(_startPosition + Consumed, SeekOrigin.Begin);
        }

        /// <summary>Reads the next chunk.</summary>
        /// <returns>False at the end of the stream (sticky, like a C stream's EOF flag).</returns>
        private bool Fill()
        {
            if (_eof)
                return false;
            _len = _stream.Read(_buffer, 0, _buffer.Length);
            _pos = 0;
            if (_len <= 0)
            {
                _len = 0;
                _eof = true;
                return false;
            }
            return true;
        }

        /// <summary>Returns the chunk buffer to the pool.</summary>
        public void Dispose()
        {
            if (_buffer != null)
            {
                ArrayPool<byte>.Shared.Return(_buffer);
                _buffer = null;
            }
        }
    }

    internal static partial class NumPyTextScan
    {
        /// <summary>NumPy's <c>FLOAT_FORMATBUFLEN</c>: <c>read_numberlike_string</c> keeps at most this many bytes of a number.</summary>
        private const int FloatFormatBufLen = 120;

        /// <summary>
        ///     glibc's <c>fscanf</c> integer conversion, which NumPy's <c>SHORT_scan</c> / <c>INT_scan</c> / <c>LONG_scan</c> /
        ///     <c>BYTE_scan</c> (<c>%hd %d %ld</c>, signed) and their unsigned twins (<c>%hu %u %lu</c>) call: whitespace,
        ///     an optional sign, then at least one digit, converted by <c>strtol</c> (saturating at
        ///     <see cref="long.MinValue"/> / <see cref="long.MaxValue"/> by sign) or <c>strtoul</c> (a minus sign negates the
        ///     magnitude modulo 2^64; a magnitude past 2^64 - 1 is <see cref="ulong.MaxValue"/>). The C type's assignment then
        ///     truncates the 64-bit value.
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="unsignedConversion">True for the <c>%u</c> family.</param>
        /// <param name="value">The 64-bit result's bits (a <see cref="long"/> or a <see cref="ulong"/>).</param>
        /// <returns>1 when a number was read, 0 on a matching failure (the sign, if any, stays consumed), -1 when the file
        ///     ended before anything but whitespace.</returns>
        internal static int ScanInteger(CFileReader fp, bool unsignedConversion, out ulong value)
        {
            value = 0;
            int c = fp.Getc();
            while (IsCSpace(c))
                c = fp.Getc();
            if (c < 0)
                return -1;
            bool negative = false;
            if (c == '+' || c == '-')
            {
                negative = c == '-';
                c = fp.Getc();
            }
            ulong magnitude = 0;
            bool overflow = false;
            int digits = 0;
            while (IsDigit(c))
            {
                ulong d = (uint)(c - '0');
                if (!overflow)
                {
                    if (magnitude > (ulong.MaxValue - d) / 10)
                        overflow = true;
                    else
                        magnitude = magnitude * 10 + d;
                }
                digits++;
                c = fp.Getc();
            }
            // the byte that ended the number is not part of it
            fp.Ungetc(c);
            if (digits == 0)
                return 0;

            if (unsignedConversion)
                value = overflow ? ulong.MaxValue : negative ? unchecked(0UL - magnitude) : magnitude;
            else
            {
                long signedValue;
                if (!overflow && magnitude <= long.MaxValue)
                    signedValue = negative ? -(long)magnitude : (long)magnitude;
                else if (!overflow && negative && magnitude == 0x8000000000000000UL)
                    signedValue = long.MinValue;
                else
                    signedValue = negative ? long.MinValue : long.MaxValue;
                value = unchecked((ulong)signedValue);
            }
            return 1;
        }

        /// <summary>
        ///     NumPy's <c>read_numberlike_string</c>: whitespace, then the longest leading part of
        ///     <c>[+-]?( [0-9]*\.[0-9]+([eE][+-]?[0-9]+) | nan(\([alnum_]*\))? | inf(inity)? )</c> (case-insensitive) as the
        ///     C macros match it, at most <see cref="FloatFormatBufLen"/> bytes. Whatever the match took stays consumed: a
        ///     dangling <c>e</c> or <c>.</c> is read into the buffer and lost when the number is parsed from it.
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="buffer">At least <see cref="FloatFormatBufLen"/> bytes.</param>
        /// <param name="length">The bytes stored.</param>
        /// <returns>1 when something was stored, 0 when nothing was, -1 when the file was already at its end.</returns>
        private static int ReadNumberlikeString(CFileReader fp, Span<byte> buffer, out int length)
        {
            int n = 0;
            int c = fp.Getc();
            if (c < 0)
            {
                length = 0;
                return -1;
            }
            while (IsCSpace(c))
                c = fp.Getc();

            // 4.1 sign
            if (c == '+' || c == '-')
            {
                if (!Next(fp, buffer, ref n, ref c))
                    goto filled;
            }
            // 4.2 nan, inf, infinity
            if (c == 'n' || c == 'N')
            {
                if (!Next(fp, buffer, ref n, ref c) || !MatchAlpha(fp, buffer, ref n, ref c, "an"))
                    goto filled;
                if (c == '(')
                {
                    if (!Next(fp, buffer, ref n, ref c))
                        goto filled;
                    while (IsAlnum(c) || c == '_')
                        if (!Next(fp, buffer, ref n, ref c))
                            goto filled;
                    if (c == ')')
                        Next(fp, buffer, ref n, ref c);
                }
                goto filled;
            }
            if (c == 'i' || c == 'I')
            {
                if (Next(fp, buffer, ref n, ref c))
                    MatchAlpha(fp, buffer, ref n, ref c, "nfinity");
                goto filled;
            }
            // 4.3 mantissa
            while (IsDigit(c))
                if (!Next(fp, buffer, ref n, ref c))
                    goto filled;
            if (c == '.')
            {
                if (!Next(fp, buffer, ref n, ref c) || !MatchOneOrMoreDigits(fp, buffer, ref n, ref c))
                    goto filled;
            }
            // 4.4 exponent
            if (c == 'e' || c == 'E')
            {
                if (!Next(fp, buffer, ref n, ref c))
                    goto filled;
                if ((c == '+' || c == '-') && !Next(fp, buffer, ref n, ref c))
                    goto filled;
                MatchOneOrMoreDigits(fp, buffer, ref n, ref c);
            }

        filled:
            fp.Ungetc(c);
            length = n;
            return n == 0 ? 0 : 1;
        }

        /// <summary>The <c>NEXT_CHAR</c> macro: stores the current byte and reads the next one, unless the byte is EOF or the
        ///     buffer is full (<c>END_MATCH</c>).</summary>
        /// <param name="fp">The file.</param>
        /// <param name="buffer">The match buffer.</param>
        /// <param name="n">The bytes stored.</param>
        /// <param name="c">The current byte; becomes the next one.</param>
        /// <returns>False when the match ends here.</returns>
        private static bool Next(CFileReader fp, Span<byte> buffer, ref int n, ref int c)
        {
            if (c < 0 || n >= FloatFormatBufLen)
                return false;
            buffer[n++] = (byte)c;
            c = fp.Getc();
            return true;
        }

        /// <summary>The <c>MATCH_ALPHA_STRING_NOCASE</c> macro: each letter of <paramref name="word"/> in either case.</summary>
        /// <param name="fp">The file.</param>
        /// <param name="buffer">The match buffer.</param>
        /// <param name="n">The bytes stored.</param>
        /// <param name="c">The current byte.</param>
        /// <param name="word">The lower-case rest of the word.</param>
        /// <returns>False when the match ends (a letter differs, or <c>NEXT_CHAR</c> ended it).</returns>
        private static bool MatchAlpha(CFileReader fp, Span<byte> buffer, ref int n, ref int c, string word)
        {
            foreach (char letter in word)
            {
                // c == *p || c + ('a' - 'A') == *p: the letter itself or its upper case
                if (c != letter && c + ('a' - 'A') != letter)
                    return false;
                if (!Next(fp, buffer, ref n, ref c))
                    return false;
            }
            return true;
        }

        /// <summary>The <c>MATCH_ONE_OR_MORE(isdigit)</c> macro.</summary>
        /// <param name="fp">The file.</param>
        /// <param name="buffer">The match buffer.</param>
        /// <param name="n">The bytes stored.</param>
        /// <param name="c">The current byte.</param>
        /// <returns>False when no digit followed, or <c>NEXT_CHAR</c> ended the match.</returns>
        private static bool MatchOneOrMoreDigits(CFileReader fp, Span<byte> buffer, ref int n, ref int c)
        {
            bool ok = false;
            while (IsDigit(c))
            {
                if (!Next(fp, buffer, ref n, ref c))
                    return false;
                ok = true;
            }
            return ok;
        }

        /// <summary>
        ///     NumPy's <c>NumPyOS_ascii_ftolf</c> (behind <c>FLOAT_scan</c>, <c>DOUBLE_scan</c>, <c>HALF_scan</c> and
        ///     <c>BOOL_scan</c>): <see cref="ReadNumberlikeString"/>, then <see cref="NumPyOSStrtod"/> over what it stored.
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="value">The number (only meaningful when 1 is returned).</param>
        /// <returns>1 when a number was read, 0 when none (the bytes the match stored stay consumed), -1 at the end of the
        ///     file.</returns>
        internal static int Ftolf(CFileReader fp, out double value)
        {
            Span<byte> buffer = stackalloc byte[FloatFormatBufLen];
            int r = ReadNumberlikeString(fp, buffer, out int n);
            value = 0;
            if (r == 1)
            {
                value = NumPyOSStrtod(buffer.Slice(0, n), 0, out int p);
                r = p == 0 ? 0 : 1;
            }
            return r;
        }

        /// <summary>
        ///     NumSharp's Decimal file reader (no NumPy analog): <see cref="ReadNumberlikeString"/>, then the decimal number of
        ///     what it stored, exactly (<see cref="DecimalFromstr"/>); a NaN, an infinity or a number past the decimal range is
        ///     a matching failure.
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="value">The value (only meaningful when 1 is returned).</param>
        /// <returns>1, 0 or -1 as <see cref="Ftolf"/>.</returns>
        internal static int ScanDecimal(CFileReader fp, out decimal value)
        {
            Span<byte> buffer = stackalloc byte[FloatFormatBufLen];
            int r = ReadNumberlikeString(fp, buffer, out int n);
            value = 0m;
            if (r == 1)
                r = DecimalFromstr(buffer.Slice(0, n), 0, out int p, out value) == 0 && p != 0 ? 1 : 0;
            return r;
        }

        /// <summary>
        ///     NumPy's <c>CDOUBLE_scan</c>: a real part (<see cref="Ftolf"/>), then a peek at the next byte. A sign starts an
        ///     imaginary part, which counts only when a <c>j</c> follows it — otherwise the part is 0 and an <c>'a'</c> is
        ///     pushed back in place of the byte read, so that the separator match fails. A <c>j</c> makes the real part the
        ///     imaginary one. The peeked byte travels through a <c>char</c>, so a <c>0xFF</c> byte after the number is lost
        ///     (<see cref="CFileReader.UngetcSignedChar"/>).
        /// </summary>
        /// <param name="fp">The file.</param>
        /// <param name="value">The value (only meaningful when 1 is returned).</param>
        /// <returns>The REAL part's result: 1, 0 or -1 as <see cref="Ftolf"/>.</returns>
        internal static int ScanComplex(CFileReader fp, out Complex value)
        {
            int retReal = Ftolf(fp, out double result);
            int next = fp.Getc();
            if (next == '+' || next == '-')
            {
                double real = result;
                fp.Ungetc(next);
                int retImag = Ftolf(fp, out result);
                int after = fp.Getc();
                if (retImag == 1 && after == 'j')
                    value = new Complex(real, result);
                else
                {
                    value = new Complex(real, 0.0);
                    // "Push an invalid char to trigger the not everything is read error"
                    fp.Ungetc('a');
                }
            }
            else if (next == 'j')
                value = new Complex(0.0, result);
            else
            {
                value = new Complex(result, 0.0);
                fp.UngetcSignedChar(next);
            }
            return retReal;
        }

        /// <summary>
        ///     NumPy's <c>fromfile_skip_separator</c>: the file's version of <see cref="FromstrSkipSeparator"/>, whose space
        ///     wildcard counts as matched only after it took a whitespace byte (the <c>sep_start</c> bookkeeping).
        /// </summary>
        /// <remarks>
        ///     A separator byte of 0x80 or more NEVER matches: NumPy compares the separator's <c>char</c> (signed, so such a
        ///     byte is negative) with <c>fgetc</c>'s <c>int</c> (0-255). A non-ASCII separator therefore stops a file read at
        ///     its first separator (an error, or the end when the count is reached), while <c>np.fromstring</c>, which compares
        ///     two <c>char</c>s, matches it.
        /// </remarks>
        /// <param name="fp">The file.</param>
        /// <param name="sep">The swabbed separator (<see cref="SwabSeparator"/>).</param>
        /// <returns>0 when matched, -1 when the file ended first, -2 when it does not match.</returns>
        internal static int FromfileSkipSeparator(CFileReader fp, ReadOnlySpan<byte> sep)
        {
            int si = 0, sepStart = 0;
            while (true)
            {
                int c = fp.Getc();
                if (c < 0)
                    return -1;
                if (si == sep.Length)
                {
                    fp.Ungetc(c);
                    return si != sepStart ? 0 : -2;
                }
                if (sep[si] == ' ')
                {
                    if (!IsCSpace(c))
                    {
                        si++;
                        sepStart++;
                        fp.Ungetc(c);
                    }
                    else if (si == sepStart)
                        sepStart--;
                }
                else if (sep[si] >= 0x80 || sep[si] != c)
                {
                    // (char)sep[si] is negative for a byte >= 0x80 and never equals fgetc's 0-255 value
                    fp.Ungetc(c);
                    return -2;
                }
                else
                    si++;
            }
        }
    }
}
