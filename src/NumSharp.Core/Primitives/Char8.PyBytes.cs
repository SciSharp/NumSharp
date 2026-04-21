// Python-bytes-style array operations for Char8[]. Each method mirrors the
// behavior of `bytes.xxx(...)` in Python 3 with full parity — these are the
// primary integration surface for NumPy's `numpy.char` / `numpy.bytes_` APIs.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp
{
    public readonly partial struct Char8
    {
        /// <summary>ASCII whitespace bytes used by Python's <c>bytes.strip()</c>: space, tab, LF, VT, FF, CR.</summary>
        private static ReadOnlySpan<byte> AsciiWhitespace => [(byte)' ', (byte)'\t', (byte)'\n', (byte)'\v', (byte)'\f', (byte)'\r'];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsAsciiWs(byte b) => b == 0x20 || (b >= 0x09 && b <= 0x0D);

        // ========================================================================
        // Trim / Strip (Python bytes.strip, .lstrip, .rstrip)
        // ========================================================================

        /// <summary>Python <c>b.strip()</c> — strip ASCII whitespace from both ends.</summary>
        public static Char8[] Strip(ReadOnlySpan<Char8> input)
        {
            int start = 0, end = input.Length;
            while (start < end && IsAsciiWs(input[start].m_value)) start++;
            while (end > start && IsAsciiWs(input[end - 1].m_value)) end--;
            return input.Slice(start, end - start).ToArray();
        }

        /// <summary>Python <c>b.lstrip()</c> — strip leading ASCII whitespace.</summary>
        public static Char8[] LStrip(ReadOnlySpan<Char8> input)
        {
            int start = 0;
            while (start < input.Length && IsAsciiWs(input[start].m_value)) start++;
            return input.Slice(start).ToArray();
        }

        /// <summary>Python <c>b.rstrip()</c> — strip trailing ASCII whitespace.</summary>
        public static Char8[] RStrip(ReadOnlySpan<Char8> input)
        {
            int end = input.Length;
            while (end > 0 && IsAsciiWs(input[end - 1].m_value)) end--;
            return input.Slice(0, end).ToArray();
        }

        /// <summary>Python <c>b.strip(chars)</c> — strip any byte in <paramref name="chars"/> from both ends.</summary>
        public static Char8[] Strip(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> chars)
        {
            int start = 0, end = input.Length;
            while (start < end && chars.Contains(input[start])) start++;
            while (end > start && chars.Contains(input[end - 1])) end--;
            return input.Slice(start, end - start).ToArray();
        }

        public static Char8[] LStrip(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> chars)
        {
            int start = 0;
            while (start < input.Length && chars.Contains(input[start])) start++;
            return input.Slice(start).ToArray();
        }

        public static Char8[] RStrip(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> chars)
        {
            int end = input.Length;
            while (end > 0 && chars.Contains(input[end - 1])) end--;
            return input.Slice(0, end).ToArray();
        }

        // ========================================================================
        // Split (Python bytes.split, .rsplit, .splitlines, .partition)
        // ========================================================================

        /// <summary>
        /// Python <c>b.split()</c> (no args) — splits on runs of ASCII whitespace, no empty elements, max <paramref name="maxsplit"/> splits
        /// (negative = unlimited). Matches Python exactly including the "leading whitespace is skipped" rule.
        /// </summary>
        public static Char8[][] Split(ReadOnlySpan<Char8> input, int maxsplit = -1)
        {
            var result = new List<Char8[]>();
            int i = 0;
            while (i < input.Length)
            {
                while (i < input.Length && IsAsciiWs(input[i].m_value)) i++;
                if (i >= input.Length) break;
                int start = i;
                while (i < input.Length && !IsAsciiWs(input[i].m_value)) i++;
                result.Add(input.Slice(start, i - start).ToArray());
                if (maxsplit >= 0 && result.Count > maxsplit)
                {
                    // Merge the last added element with the remainder
                    Char8[] last = result[^1];
                    result.RemoveAt(result.Count - 1);
                    // Include everything from start (not i) to end
                    result.Add(input.Slice(start).ToArray());
                    return result.ToArray();
                }
            }
            return result.ToArray();
        }

        /// <summary>Python <c>b.split(sep)</c> — splits on <paramref name="separator"/>, preserves empty elements.</summary>
        public static Char8[][] Split(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> separator, int maxsplit = -1)
        {
            if (separator.Length == 0) throw new ArgumentException("Empty separator.", nameof(separator));
            var result = new List<Char8[]>();
            int from = 0;
            int splits = 0;
            while (true)
            {
                if (maxsplit >= 0 && splits >= maxsplit)
                {
                    result.Add(input.Slice(from).ToArray());
                    return result.ToArray();
                }
                int idx = input.Slice(from).IndexOf(separator);
                if (idx < 0)
                {
                    result.Add(input.Slice(from).ToArray());
                    return result.ToArray();
                }
                result.Add(input.Slice(from, idx).ToArray());
                from += idx + separator.Length;
                splits++;
            }
        }

        /// <summary>Python <c>b.rsplit()</c> — like Split but consumes from the right end.</summary>
        public static Char8[][] RSplit(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> separator, int maxsplit = -1)
        {
            if (separator.Length == 0) throw new ArgumentException("Empty separator.", nameof(separator));
            var result = new List<Char8[]>();
            int end = input.Length;
            int splits = 0;
            while (true)
            {
                if (maxsplit >= 0 && splits >= maxsplit) break;
                int idx = input.Slice(0, end).LastIndexOf(separator);
                if (idx < 0) break;
                result.Insert(0, input.Slice(idx + separator.Length, end - idx - separator.Length).ToArray());
                end = idx;
                splits++;
            }
            result.Insert(0, input.Slice(0, end).ToArray());
            return result.ToArray();
        }

        /// <summary>
        /// Python <c>bytes.splitlines(keepends)</c> — splits on <c>\n</c>, <c>\r</c>, and <c>\r\n</c> only.
        /// Unlike Python's <c>str.splitlines()</c>, <c>bytes</c> does NOT treat <c>\v</c>, <c>\f</c>,
        /// <c>\x1c</c>..<c>\x1e</c>, or <c>\x85</c> as line boundaries.
        /// </summary>
        public static Char8[][] SplitLines(ReadOnlySpan<Char8> input, bool keepEnds = false)
        {
            var result = new List<Char8[]>();
            int i = 0;
            while (i < input.Length)
            {
                int start = i;
                while (i < input.Length)
                {
                    byte b = input[i].m_value;
                    if (b == 0x0A || b == 0x0D) break;
                    i++;
                }
                int eolStart = i;
                if (i < input.Length)
                {
                    byte b = input[i].m_value;
                    i++;
                    if (b == 0x0D && i < input.Length && input[i].m_value == 0x0A) i++; // \r\n
                }
                int contentEnd = keepEnds ? i : eolStart;
                if (contentEnd > start || i > start)  // Python skips trailing empty line
                    result.Add(input.Slice(start, contentEnd - start).ToArray());
            }
            return result.ToArray();
        }

        /// <summary>Python <c>b.partition(sep)</c> — splits on first occurrence, returns (before, sep, after).</summary>
        public static (Char8[] Before, Char8[] Sep, Char8[] After) Partition(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> separator)
        {
            if (separator.Length == 0) throw new ArgumentException("Empty separator.", nameof(separator));
            int idx = input.IndexOf(separator);
            if (idx < 0)
                return (input.ToArray(), Array.Empty<Char8>(), Array.Empty<Char8>());
            return (
                input.Slice(0, idx).ToArray(),
                separator.ToArray(),
                input.Slice(idx + separator.Length).ToArray());
        }

        /// <summary>Python <c>b.rpartition(sep)</c> — splits on last occurrence.</summary>
        public static (Char8[] Before, Char8[] Sep, Char8[] After) RPartition(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> separator)
        {
            if (separator.Length == 0) throw new ArgumentException("Empty separator.", nameof(separator));
            int idx = input.LastIndexOf(separator);
            if (idx < 0)
                return (Array.Empty<Char8>(), Array.Empty<Char8>(), input.ToArray());
            return (
                input.Slice(0, idx).ToArray(),
                separator.ToArray(),
                input.Slice(idx + separator.Length).ToArray());
        }

        // ========================================================================
        // Join
        // ========================================================================

        /// <summary>Python <c>separator.join(iterable)</c>.</summary>
        public static Char8[] Join(ReadOnlySpan<Char8> separator, Char8[][] parts)
        {
            if (parts.Length == 0) return Array.Empty<Char8>();
            int total = 0;
            for (int i = 0; i < parts.Length; i++) total += parts[i].Length;
            if (parts.Length > 1) total += separator.Length * (parts.Length - 1);
            var result = new Char8[total];
            int dst = 0;
            for (int i = 0; i < parts.Length; i++)
            {
                if (i > 0)
                {
                    separator.CopyTo(result.AsSpan(dst));
                    dst += separator.Length;
                }
                parts[i].CopyTo(result.AsSpan(dst));
                dst += parts[i].Length;
            }
            return result;
        }

        // ========================================================================
        // Replace / Count (Python bytes.replace, .count)
        // ========================================================================

        /// <summary>Python <c>b.replace(old, new, count)</c>.</summary>
        public static Char8[] Replace(ReadOnlySpan<Char8> input, ReadOnlySpan<Char8> oldValue, ReadOnlySpan<Char8> newValue, int count = -1)
        {
            if (oldValue.Length == 0)
            {
                // Python: inserting new between every byte (and at start/end)
                if (count < 0) count = int.MaxValue;
                int inserts = Math.Min(count, input.Length + 1);
                int total = input.Length + inserts * newValue.Length;
                var r = new Char8[total];
                int dst = 0;
                for (int i = 0; i <= input.Length; i++)
                {
                    if (i < inserts)
                    {
                        newValue.CopyTo(r.AsSpan(dst));
                        dst += newValue.Length;
                    }
                    if (i < input.Length) r[dst++] = input[i];
                }
                return r;
            }

            var occurrences = new List<int>();
            int from = 0;
            while (count != 0)
            {
                int idx = input.Slice(from).IndexOf(oldValue);
                if (idx < 0) break;
                occurrences.Add(from + idx);
                from += idx + oldValue.Length;
                if (count > 0) count--;
            }

            if (occurrences.Count == 0) return input.ToArray();

            int delta = newValue.Length - oldValue.Length;
            int newLength = input.Length + delta * occurrences.Count;
            var result = new Char8[newLength];
            int srcIdx = 0, dstIdx = 0;
            foreach (int occ in occurrences)
            {
                int copyLen = occ - srcIdx;
                input.Slice(srcIdx, copyLen).CopyTo(result.AsSpan(dstIdx));
                dstIdx += copyLen;
                newValue.CopyTo(result.AsSpan(dstIdx));
                dstIdx += newValue.Length;
                srcIdx = occ + oldValue.Length;
            }
            input.Slice(srcIdx).CopyTo(result.AsSpan(dstIdx));
            return result;
        }

        // ========================================================================
        // Case conversion (array-level)
        // ========================================================================

        /// <summary>Python <c>b.upper()</c> — ASCII bit-flip of each byte.</summary>
        public static Char8[] Upper(ReadOnlySpan<Char8> input)
        {
            var r = new Char8[input.Length];
            for (int i = 0; i < input.Length; i++) r[i] = ToUpper(input[i]);
            return r;
        }

        /// <summary>Python <c>b.lower()</c>.</summary>
        public static Char8[] Lower(ReadOnlySpan<Char8> input)
        {
            var r = new Char8[input.Length];
            for (int i = 0; i < input.Length; i++) r[i] = ToLower(input[i]);
            return r;
        }

        /// <summary>Python <c>b.swapcase()</c>.</summary>
        public static Char8[] SwapCase(ReadOnlySpan<Char8> input)
        {
            var r = new Char8[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                Char8 c = input[i];
                r[i] = IsAsciiLetterUpper(c) ? new Char8((byte)(c.m_value | 0x20))
                     : IsAsciiLetterLower(c) ? new Char8((byte)(c.m_value & 0xDF))
                     : c;
            }
            return r;
        }

        /// <summary>Python <c>b.capitalize()</c> — first byte uppercase, rest lowercase.</summary>
        public static Char8[] Capitalize(ReadOnlySpan<Char8> input)
        {
            if (input.Length == 0) return Array.Empty<Char8>();
            var r = new Char8[input.Length];
            r[0] = ToUpper(input[0]);
            for (int i = 1; i < input.Length; i++) r[i] = ToLower(input[i]);
            return r;
        }

        /// <summary>Python <c>b.title()</c> — titlecase ASCII: uppercase byte after any non-letter byte, lowercase elsewhere.</summary>
        public static Char8[] Title(ReadOnlySpan<Char8> input)
        {
            var r = new Char8[input.Length];
            bool prevIsLetter = false;
            for (int i = 0; i < input.Length; i++)
            {
                Char8 c = input[i];
                if (IsAsciiLetter(c))
                {
                    r[i] = prevIsLetter ? ToLower(c) : ToUpper(c);
                    prevIsLetter = true;
                }
                else
                {
                    r[i] = c;
                    prevIsLetter = false;
                }
            }
            return r;
        }

        // ========================================================================
        // Padding (Python bytes.ljust, .rjust, .center, .zfill)
        // ========================================================================

        /// <summary>Python <c>b.ljust(width, fillchar)</c>.</summary>
        public static Char8[] LJust(ReadOnlySpan<Char8> input, int width, Char8 fillChar)
        {
            if (input.Length >= width) return input.ToArray();
            var r = new Char8[width];
            input.CopyTo(r.AsSpan(0, input.Length));
            r.AsSpan(input.Length).Fill(fillChar);
            return r;
        }

        /// <summary>Python <c>b.rjust(width, fillchar)</c>.</summary>
        public static Char8[] RJust(ReadOnlySpan<Char8> input, int width, Char8 fillChar)
        {
            if (input.Length >= width) return input.ToArray();
            var r = new Char8[width];
            int pad = width - input.Length;
            r.AsSpan(0, pad).Fill(fillChar);
            input.CopyTo(r.AsSpan(pad));
            return r;
        }

        /// <summary>
        /// Python <c>b.center(width, fillchar)</c>. Uses CPython's formula
        /// <c>left = pad/2 + (pad &amp; width &amp; 1)</c> — extra padding goes on the LEFT when
        /// <c>pad</c> is odd and <c>width</c> is also odd.
        /// </summary>
        public static Char8[] Center(ReadOnlySpan<Char8> input, int width, Char8 fillChar)
        {
            if (input.Length >= width) return input.ToArray();
            int pad = width - input.Length;
            int left = pad / 2 + (pad & width & 1);
            var r = new Char8[width];
            r.AsSpan(0, left).Fill(fillChar);
            input.CopyTo(r.AsSpan(left));
            r.AsSpan(left + input.Length).Fill(fillChar);
            return r;
        }

        /// <summary>Python <c>b.zfill(width)</c> — pads with '0' on the left. Preserves leading '+'/'-' sign byte.</summary>
        public static Char8[] ZFill(ReadOnlySpan<Char8> input, int width)
        {
            if (input.Length >= width) return input.ToArray();
            Char8 zero = new Char8((byte)'0');
            int pad = width - input.Length;
            var r = new Char8[width];
            if (input.Length > 0 && (input[0].m_value == (byte)'+' || input[0].m_value == (byte)'-'))
            {
                r[0] = input[0];
                r.AsSpan(1, pad).Fill(zero);
                input.Slice(1).CopyTo(r.AsSpan(1 + pad));
            }
            else
            {
                r.AsSpan(0, pad).Fill(zero);
                input.CopyTo(r.AsSpan(pad));
            }
            return r;
        }

        // ========================================================================
        // Classification on arrays (Python bytes.isdigit, .isalpha, etc.)
        // ========================================================================

        /// <summary>Python <c>b.isdigit()</c> — non-empty and every byte is '0'..'9'.</summary>
        public static bool IsDigits(ReadOnlySpan<Char8> input)
        {
            if (input.Length == 0) return false;
            for (int i = 0; i < input.Length; i++)
                if (!IsAsciiDigit(input[i])) return false;
            return true;
        }

        /// <summary>Python <c>b.isalpha()</c> — non-empty and every byte is an ASCII letter.</summary>
        public static bool IsAlphas(ReadOnlySpan<Char8> input)
        {
            if (input.Length == 0) return false;
            for (int i = 0; i < input.Length; i++)
                if (!IsAsciiLetter(input[i])) return false;
            return true;
        }

        /// <summary>Python <c>b.isalnum()</c>.</summary>
        public static bool IsAlnums(ReadOnlySpan<Char8> input)
        {
            if (input.Length == 0) return false;
            for (int i = 0; i < input.Length; i++)
                if (!IsAsciiLetterOrDigit(input[i])) return false;
            return true;
        }

        /// <summary>Python <c>b.isspace()</c>.</summary>
        public static bool IsSpaces(ReadOnlySpan<Char8> input)
        {
            if (input.Length == 0) return false;
            for (int i = 0; i < input.Length; i++)
                if (!IsWhiteSpace(input[i])) return false;
            return true;
        }

        /// <summary>
        /// Python <c>b.isupper()</c> — true if at least one cased byte exists and all cased bytes are uppercase.
        /// Non-cased bytes are permitted.
        /// </summary>
        public static bool IsUppers(ReadOnlySpan<Char8> input)
        {
            bool hasCased = false;
            for (int i = 0; i < input.Length; i++)
            {
                Char8 c = input[i];
                if (IsAsciiLetterUpper(c)) hasCased = true;
                else if (IsAsciiLetterLower(c)) return false;
            }
            return hasCased;
        }

        /// <summary>Python <c>b.islower()</c> — mirror of IsUppers.</summary>
        public static bool IsLowers(ReadOnlySpan<Char8> input)
        {
            bool hasCased = false;
            for (int i = 0; i < input.Length; i++)
            {
                Char8 c = input[i];
                if (IsAsciiLetterLower(c)) hasCased = true;
                else if (IsAsciiLetterUpper(c)) return false;
            }
            return hasCased;
        }

        /// <summary>Python <c>b.istitle()</c> — title case alternation of ASCII letters.</summary>
        public static bool IsTitles(ReadOnlySpan<Char8> input)
        {
            bool hasCased = false;
            bool prevIsLetter = false;
            for (int i = 0; i < input.Length; i++)
            {
                Char8 c = input[i];
                if (IsAsciiLetterUpper(c))
                {
                    if (prevIsLetter) return false;
                    hasCased = true; prevIsLetter = true;
                }
                else if (IsAsciiLetterLower(c))
                {
                    if (!prevIsLetter) return false;
                    hasCased = true; prevIsLetter = true;
                }
                else
                {
                    prevIsLetter = false;
                }
            }
            return hasCased;
        }

        /// <summary>Python <c>b.isascii()</c> — every byte in [0x00, 0x7F]. Empty → true.</summary>
        public static bool IsAsciis(ReadOnlySpan<Char8> input)
        {
            for (int i = 0; i < input.Length; i++)
                if (input[i].m_value > 0x7F) return false;
            return true;
        }

        /// <summary>Python <c>b.isprintable()</c> — every byte in 0x20..0x7E. Empty → true.</summary>
        public static bool IsPrintables(ReadOnlySpan<Char8> input)
        {
            for (int i = 0; i < input.Length; i++)
                if (input[i].m_value < 0x20 || input[i].m_value > 0x7E) return false;
            return true;
        }
    }
}
