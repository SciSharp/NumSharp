// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
//
// NumSharp port: adapted from System.Char (dotnet/runtime, src/dotnet/src/libraries/
// System.Private.CoreLib/src/System/Char.cs) to a 1-byte character type modelled on
// NumPy's `dtype('S1')` / `numpy.bytes_` and Python's single-byte `bytes`.
//
// Representation : one byte, values 0x00..0xFF (unsigned).
// Layout         : [StructLayout(LayoutKind.Sequential)] — binary-compatible with byte.
// NumPy parity   : classification predicates (IsLetter, IsDigit, IsUpper, IsLower,
//                  IsWhiteSpace, IsLetterOrDigit) are ASCII-only. Latin-1 bytes
//                  (0x80..0xFF) return false — matches `bytes.isalpha()`, etc.
// C# interop     : implicit widening  Char8 -> byte / int / char  (Latin-1 mapping).
//                  explicit narrowing char / byte / int -> Char8  (throws on > 0xFF).
//                  string <-> Char8[] via ASCII / Latin-1 helpers.
// Case mapping   : ASCII bit-flip for 'A'..'Z' / 'a'..'z'. The full Latin-1
//                  ToUpper/ToLower fold is available via ToUpperLatin1 / ToLowerLatin1
//                  for callers that want Char.cs semantics.

using System;
using System.Buffers.Binary;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp
{
    /// <summary>
    /// Represents a single byte as a character. Equivalent to NumPy's <c>dtype('S1')</c>
    /// / <c>numpy.bytes_</c> of length 1, and to a Python <c>bytes</c> object of length 1.
    /// Interoperable with <see cref="byte"/>, <see cref="char"/> (via Latin-1), and
    /// <see cref="string"/> (via ASCII/Latin-1 encoding).
    /// </summary>
    [Serializable]
    [StructLayout(LayoutKind.Sequential, Size = 1)]
    public readonly partial struct Char8
        : IComparable,
          IComparable<Char8>,
          IEquatable<Char8>,
          IConvertible,
          IFormattable,
          ISpanFormattable
    {
        // ========================================================================
        // Fields
        // ========================================================================

        private readonly byte m_value;

        // ========================================================================
        // Constants
        // ========================================================================

        /// <summary>The maximum value (0xFF).</summary>
        public static readonly Char8 MaxValue = new Char8(byte.MaxValue);

        /// <summary>The minimum value (0x00).</summary>
        public static readonly Char8 MinValue = new Char8(byte.MinValue);

        // Flag layout of Latin1CharInfo (copied verbatim from Char.cs).
        private const byte IsWhiteSpaceFlag = 0x80;
        private const byte IsUpperCaseLetterFlag = 0x40;
        private const byte IsLowerCaseLetterFlag = 0x20;
        private const byte UnicodeCategoryMask = 0x1F;

        // Contains information about the C0, Basic Latin, C1, and Latin-1 Supplement ranges [ U+0000..U+00FF ], with:
        // - 0x80 bit if set means 'is whitespace'
        // - 0x40 bit if set means 'is uppercase letter'
        // - 0x20 bit if set means 'is lowercase letter'
        // - bottom 5 bits are the UnicodeCategory of the character
        private static ReadOnlySpan<byte> Latin1CharInfo =>
        [
        //  0     1     2     3     4     5     6     7     8     9     A     B     C     D     E     F
            0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x8E, 0x8E, 0x8E, 0x8E, 0x8E, 0x0E, 0x0E, // U+0000..U+000F
            0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, // U+0010..U+001F
            0x8B, 0x18, 0x18, 0x18, 0x1A, 0x18, 0x18, 0x18, 0x14, 0x15, 0x18, 0x19, 0x18, 0x13, 0x18, 0x18, // U+0020..U+002F
            0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x18, 0x18, 0x19, 0x19, 0x19, 0x18, // U+0030..U+003F
            0x18, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, // U+0040..U+004F
            0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x14, 0x18, 0x15, 0x1B, 0x12, // U+0050..U+005F
            0x1B, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, // U+0060..U+006F
            0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x14, 0x19, 0x15, 0x19, 0x0E, // U+0070..U+007F
            0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x8E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, // U+0080..U+008F
            0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, 0x0E, // U+0090..U+009F
            0x8B, 0x18, 0x1A, 0x1A, 0x1A, 0x1A, 0x1C, 0x18, 0x1B, 0x1C, 0x04, 0x16, 0x19, 0x0F, 0x1C, 0x1B, // U+00A0..U+00AF
            0x1C, 0x19, 0x0A, 0x0A, 0x1B, 0x21, 0x18, 0x18, 0x1B, 0x0A, 0x04, 0x17, 0x0A, 0x0A, 0x0A, 0x18, // U+00B0..U+00BF
            0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, // U+00C0..U+00CF
            0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x19, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x21, // U+00D0..U+00DF
            0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, // U+00E0..U+00EF
            0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x19, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21, // U+00F0..U+00FF
        ];

        // ========================================================================
        // Construction
        // ========================================================================

        /// <summary>Constructs a <see cref="Char8"/> directly from a byte.</summary>
        public Char8(byte value) { m_value = value; }

        /// <summary>Constructs a <see cref="Char8"/> from a <see cref="char"/>. Throws if the char cannot be represented in one byte (Latin-1).</summary>
        public Char8(char value)
        {
            if ((uint)value > 0xFF)
                throw new ArgumentOutOfRangeException(nameof(value), "Char must be in the Latin-1 range [0x00..0xFF] to convert to Char8.");
            m_value = (byte)value;
        }

        // ========================================================================
        // Conversions
        // ========================================================================

        /// <summary>Exposes the raw byte value.</summary>
        public byte Value => m_value;

        public static implicit operator byte(Char8 c) => c.m_value;
        public static implicit operator int(Char8 c) => c.m_value;
        public static implicit operator uint(Char8 c) => c.m_value;

        /// <summary>Widens to <see cref="char"/> via Latin-1 (byte 0xE9 -> char 'é' at U+00E9).</summary>
        public static implicit operator char(Char8 c) => (char)c.m_value;

        public static implicit operator Char8(byte b) => new Char8(b);

        /// <summary>Narrows from <see cref="char"/>. Throws if the char is outside Latin-1 (> 0xFF).</summary>
        public static explicit operator Char8(char c)
        {
            if ((uint)c > 0xFF)
                throw new OverflowException("Char value " + (int)c + " exceeds Char8 max (0xFF).");
            return new Char8((byte)c);
        }

        /// <summary>Narrows from <see cref="int"/>. Throws if the int is outside [0, 255].</summary>
        public static explicit operator Char8(int v)
        {
            if ((uint)v > 0xFF)
                throw new OverflowException("Int value " + v + " outside Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        /// <summary>Truncates a char to its low byte without bounds checking.</summary>
        public static Char8 FromCharTruncating(char c) => new Char8((byte)c);

        /// <summary>Truncates an int to its low byte without bounds checking.</summary>
        public static Char8 FromInt32Truncating(int v) => new Char8((byte)v);

        // ========================================================================
        // Equality, comparison, hashing
        // ========================================================================

        public override int GetHashCode() => m_value;

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is Char8 c && c.m_value == m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(Char8 other) => m_value == other.m_value;

        public int CompareTo(object? value)
        {
            if (value is null) return 1;
            if (value is not Char8 c) throw new ArgumentException("Argument must be Char8.");
            return m_value - c.m_value;
        }

        public int CompareTo(Char8 value) => m_value - value.m_value;

        // ========================================================================
        // ToString / Parse / TryFormat
        // ========================================================================

        /// <summary>Returns a one-character <see cref="string"/>, mapping the byte to a <see cref="char"/> via Latin-1.</summary>
        public override string ToString() => ToString(this);

        public string ToString(IFormatProvider? provider) => ToString(this);

        /// <summary>Returns a one-character string for the given <see cref="Char8"/>.</summary>
        public static string ToString(Char8 c) => new string((char)c.m_value, 1);

        bool ISpanFormattable.TryFormat(Span<char> destination, out int charsWritten, ReadOnlySpan<char> format, IFormatProvider? provider)
        {
            if (!destination.IsEmpty)
            {
                destination[0] = (char)m_value;
                charsWritten = 1;
                return true;
            }
            charsWritten = 0;
            return false;
        }

        string IFormattable.ToString(string? format, IFormatProvider? formatProvider) => ToString(this);

        /// <summary>Parses a one-character string as <see cref="Char8"/>. Throws if the string is not length 1 or contains a non-Latin-1 char.</summary>
        public static Char8 Parse(string s)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            return Parse(s.AsSpan());
        }

        internal static Char8 Parse(ReadOnlySpan<char> s)
        {
            if (s.Length != 1) throw new FormatException("String must be exactly one character.");
            char c = s[0];
            if ((uint)c > 0xFF) throw new FormatException("Char must be in Latin-1 range for Char8.");
            return new Char8((byte)c);
        }

        public static bool TryParse([NotNullWhen(true)] string? s, out Char8 result)
        {
            if (s is null) { result = default; return false; }
            return TryParse(s.AsSpan(), out result);
        }

        internal static bool TryParse(ReadOnlySpan<char> s, out Char8 result)
        {
            if (s.Length != 1 || (uint)s[0] > 0xFF) { result = default; return false; }
            result = new Char8((byte)s[0]);
            return true;
        }

        // ========================================================================
        // Classification — ASCII strict (NumPy / Python bytes parity)
        // ========================================================================

        /// <summary>Returns true if the value is ASCII (0x00..0x7F).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAscii(Char8 c) => c.m_value <= 0x7F;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsBetween(Char8 c, Char8 minInclusive, Char8 maxInclusive)
            => (uint)(c.m_value - minInclusive.m_value) <= (uint)(maxInclusive.m_value - minInclusive.m_value);

        /// <summary>ASCII letter 'A'..'Z' or 'a'..'z'.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiLetter(Char8 c) => (uint)((c.m_value | 0x20) - 'a') <= ('z' - 'a');

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiLetterUpper(Char8 c) => IsBetween(c, (Char8)(byte)'A', (Char8)(byte)'Z');

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiLetterLower(Char8 c) => IsBetween(c, (Char8)(byte)'a', (Char8)(byte)'z');

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiDigit(Char8 c) => IsBetween(c, (Char8)(byte)'0', (Char8)(byte)'9');

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiLetterOrDigit(Char8 c) => IsAsciiLetter(c) | IsAsciiDigit(c);

        /// <summary>ASCII hex digit: '0'..'9', 'A'..'F', 'a'..'f'.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiHexDigit(Char8 c) => IsAsciiDigit(c) || (uint)((c.m_value | 0x20) - 'a') <= 'f' - 'a';

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiHexDigitUpper(Char8 c) => IsAsciiDigit(c) || IsBetween(c, (Char8)(byte)'A', (Char8)(byte)'F');

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiHexDigitLower(Char8 c) => IsAsciiDigit(c) || IsBetween(c, (Char8)(byte)'a', (Char8)(byte)'f');

        /// <summary>
        /// Returns true for ASCII digits '0'..'9'. <b>Non-ASCII bytes return false</b> — matches NumPy / Python's
        /// <c>bytes.isdigit()</c>. For Latin-1 digit categories use <see cref="IsDigitLatin1"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsDigit(Char8 c) => IsAsciiDigit(c);

        /// <summary>
        /// Returns true for ASCII letters 'A'..'Z' / 'a'..'z'. <b>Non-ASCII bytes return false</b> — matches
        /// NumPy / Python's <c>bytes.isalpha()</c>. For Latin-1 letters use <see cref="IsLetterLatin1"/>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsLetter(Char8 c) => IsAsciiLetter(c);

        /// <summary>ASCII uppercase 'A'..'Z'. Matches Python's <c>bytes.isupper()</c> semantics for a single byte.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUpper(Char8 c) => IsAsciiLetterUpper(c);

        /// <summary>ASCII lowercase 'a'..'z'. Matches Python's <c>bytes.islower()</c> semantics for a single byte.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsLower(Char8 c) => IsAsciiLetterLower(c);

        /// <summary>ASCII whitespace: space, tab, LF, VT, FF, CR. Matches Python's <c>bytes.isspace()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsWhiteSpace(Char8 c)
            => c.m_value == 0x20 || (c.m_value >= 0x09 && c.m_value <= 0x0D);

        /// <summary>ASCII letter or digit. Matches Python's <c>bytes.isalnum()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsLetterOrDigit(Char8 c) => IsAsciiLetterOrDigit(c);

        /// <summary>Alias matching Python's <c>bytes.isalnum()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAlnum(Char8 c) => IsAsciiLetterOrDigit(c);

        /// <summary>Alias matching Python's <c>bytes.isalpha()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAlpha(Char8 c) => IsAsciiLetter(c);

        /// <summary>Alias matching Python's <c>bytes.isspace()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSpace(Char8 c) => IsWhiteSpace(c);

        /// <summary>Alias matching Python's <c>bytes.isascii()</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsAsciiChar(Char8 c) => IsAscii(c);

        /// <summary>
        /// Matches Python's <c>bytes.isprintable()</c>: ASCII 0x20..0x7E are printable; all other bytes are not.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsPrintable(Char8 c) => IsBetween(c, (Char8)(byte)0x20, (Char8)(byte)0x7E);

        /// <summary>Control character: ASCII 0x00..0x1F or 0x7F (DEL). Also covers C1 0x80..0x9F for parity with <see cref="char.IsControl(char)"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsControl(Char8 c) => (((uint)c.m_value + 1) & ~0x80u) <= 0x20u;

        /// <summary>Returns true if the value is 0 (null). Useful for C-string / null-terminated parsing.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsNull(Char8 c) => c.m_value == 0;

        // ------------------------------------------------------------------------
        // Classification — Latin-1 (Char.cs heritage)
        // Use these when you want the System.Char semantics — i.e. treat the byte
        // as a Latin-1 code point. Divergent from NumPy for 0x80..0xFF.
        // ------------------------------------------------------------------------

        /// <summary>Latin-1 Unicode category (always defined — every byte maps to Latin-1).</summary>
        public static UnicodeCategory GetUnicodeCategory(Char8 c)
            => (UnicodeCategory)(Latin1CharInfo[c.m_value] & UnicodeCategoryMask);

        /// <summary>Latin-1 letter check: includes accented letters like 'é' (0xE9).</summary>
        public static bool IsLetterLatin1(Char8 c)
            => (Latin1CharInfo[c.m_value] & (IsUpperCaseLetterFlag | IsLowerCaseLetterFlag)) != 0;

        /// <summary>Latin-1 uppercase letter check.</summary>
        public static bool IsUpperLatin1(Char8 c)
            => (Latin1CharInfo[c.m_value] & IsUpperCaseLetterFlag) != 0;

        /// <summary>Latin-1 lowercase letter check.</summary>
        public static bool IsLowerLatin1(Char8 c)
            => (Latin1CharInfo[c.m_value] & IsLowerCaseLetterFlag) != 0;

        /// <summary>Latin-1 whitespace check: includes NBSP (0xA0) in addition to ASCII whitespace.</summary>
        public static bool IsWhiteSpaceLatin1(Char8 c)
            => (Latin1CharInfo[c.m_value] & IsWhiteSpaceFlag) != 0;

        /// <summary>Latin-1 digit check: 0x30..0x39 only. There are no decimal digits in Latin-1 supplement.</summary>
        public static bool IsDigitLatin1(Char8 c) => IsAsciiDigit(c);

        /// <summary>Latin-1 punctuation (ConnectorPunctuation..OtherPunctuation).</summary>
        public static bool IsPunctuation(Char8 c)
        {
            UnicodeCategory uc = GetUnicodeCategory(c);
            return uc >= UnicodeCategory.ConnectorPunctuation && uc <= UnicodeCategory.OtherPunctuation;
        }

        /// <summary>Latin-1 separator (SpaceSeparator..ParagraphSeparator). Only space (0x20) and NBSP (0xA0) qualify in Latin-1.</summary>
        public static bool IsSeparator(Char8 c) => c.m_value == 0x20 || c.m_value == 0xA0;

        /// <summary>Latin-1 symbol (MathSymbol..OtherSymbol).</summary>
        public static bool IsSymbol(Char8 c)
        {
            UnicodeCategory uc = GetUnicodeCategory(c);
            return uc >= UnicodeCategory.MathSymbol && uc <= UnicodeCategory.OtherSymbol;
        }

        /// <summary>Latin-1 number (DecimalDigitNumber..OtherNumber). Includes superscript and fraction chars in Latin-1.</summary>
        public static bool IsNumber(Char8 c)
        {
            UnicodeCategory uc = GetUnicodeCategory(c);
            return uc >= UnicodeCategory.DecimalDigitNumber && uc <= UnicodeCategory.OtherNumber;
        }

        /// <summary>
        /// Returns -1.0 for non-digit bytes, 0..9 for '0'..'9'. Full Latin-1 fractions/superscripts
        /// (e.g. '¼', '½', '²') are not covered — use <see cref="char.GetNumericValue(char)"/> via
        /// <c>char.GetNumericValue((char)c)</c> if you need them.
        /// </summary>
        public static double GetNumericValue(Char8 c)
        {
            if (IsAsciiDigit(c)) return c.m_value - (byte)'0';
            return -1.0;
        }

        // Always false — a byte cannot be a surrogate (surrogates live in U+D800..U+DFFF).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsSurrogate(Char8 c) => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsHighSurrogate(Char8 c) => false;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsLowSurrogate(Char8 c) => false;

        // ========================================================================
        // Case conversion
        // ========================================================================

        /// <summary>ASCII uppercase (NumPy parity): bit-flips 'a'..'z' to 'A'..'Z'. Non-ASCII bytes unchanged.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 ToUpper(Char8 c)
            => IsAsciiLetterLower(c) ? new Char8((byte)(c.m_value & 0xDF)) : c;

        /// <summary>ASCII lowercase (NumPy parity): bit-flips 'A'..'Z' to 'a'..'z'. Non-ASCII bytes unchanged.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 ToLower(Char8 c)
            => IsAsciiLetterUpper(c) ? new Char8((byte)(c.m_value | 0x20)) : c;

        /// <inheritdoc cref="ToUpper(Char8)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 ToUpperInvariant(Char8 c) => ToUpper(c);

        /// <inheritdoc cref="ToLower(Char8)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 ToLowerInvariant(Char8 c) => ToLower(c);

        /// <summary>Latin-1 uppercase: folds 'á'..'þ' (0xE0..0xFE, excluding 0xF7) to 'Á'..'Þ' as well as ASCII letters. Matches <see cref="char.ToUpperInvariant(char)"/> over Latin-1.</summary>
        public static Char8 ToUpperLatin1(Char8 c)
        {
            byte b = c.m_value;
            if (IsAsciiLetterLower(c)) return new Char8((byte)(b & 0xDF));
            // Latin-1 supplement lowercase 0xE0..0xFE (excluding 0xF7 = '÷') folds to 0xC0..0xDE
            if (b >= 0xE0 && b <= 0xFE && b != 0xF7) return new Char8((byte)(b - 0x20));
            // 0xDF is sharp-s ('ß') which has no single-char uppercase in Latin-1; leave unchanged.
            // 0xFF is 'ÿ' which uppercases to U+0178 (non-Latin-1); leave unchanged.
            return c;
        }

        /// <summary>Latin-1 lowercase: folds 'Á'..'Þ' (0xC0..0xDE, excluding 0xD7) to 'á'..'þ' as well as ASCII letters.</summary>
        public static Char8 ToLowerLatin1(Char8 c)
        {
            byte b = c.m_value;
            if (IsAsciiLetterUpper(c)) return new Char8((byte)(b | 0x20));
            // Latin-1 supplement uppercase 0xC0..0xDE (excluding 0xD7 = '×') folds to 0xE0..0xFE
            if (b >= 0xC0 && b <= 0xDE && b != 0xD7) return new Char8((byte)(b + 0x20));
            return c;
        }

        // ========================================================================
        // IConvertible
        // ========================================================================

        public TypeCode GetTypeCode() => TypeCode.Byte;

        bool IConvertible.ToBoolean(IFormatProvider? provider) => m_value != 0;
        char IConvertible.ToChar(IFormatProvider? provider) => (char)m_value;
        sbyte IConvertible.ToSByte(IFormatProvider? provider) => checked((sbyte)m_value);
        byte IConvertible.ToByte(IFormatProvider? provider) => m_value;
        short IConvertible.ToInt16(IFormatProvider? provider) => m_value;
        ushort IConvertible.ToUInt16(IFormatProvider? provider) => m_value;
        int IConvertible.ToInt32(IFormatProvider? provider) => m_value;
        uint IConvertible.ToUInt32(IFormatProvider? provider) => m_value;
        long IConvertible.ToInt64(IFormatProvider? provider) => m_value;
        ulong IConvertible.ToUInt64(IFormatProvider? provider) => m_value;
        float IConvertible.ToSingle(IFormatProvider? provider) => m_value;
        double IConvertible.ToDouble(IFormatProvider? provider) => m_value;
        decimal IConvertible.ToDecimal(IFormatProvider? provider) => m_value;

        DateTime IConvertible.ToDateTime(IFormatProvider? provider)
            => throw new InvalidCastException("Cannot cast Char8 to DateTime.");

        object IConvertible.ToType(Type type, IFormatProvider? provider)
            => System.Convert.ChangeType((byte)m_value, type, provider)!;

        // ========================================================================
        // Operators — byte-width arithmetic (wraps at 0xFF in unchecked context)
        // ========================================================================

        public static bool operator ==(Char8 left, Char8 right) => left.m_value == right.m_value;
        public static bool operator !=(Char8 left, Char8 right) => left.m_value != right.m_value;
        public static bool operator <(Char8 left, Char8 right) => left.m_value < right.m_value;
        public static bool operator >(Char8 left, Char8 right) => left.m_value > right.m_value;
        public static bool operator <=(Char8 left, Char8 right) => left.m_value <= right.m_value;
        public static bool operator >=(Char8 left, Char8 right) => left.m_value >= right.m_value;

        public static Char8 operator +(Char8 left, Char8 right) => new Char8((byte)(left.m_value + right.m_value));
        public static Char8 operator -(Char8 left, Char8 right) => new Char8((byte)(left.m_value - right.m_value));
        public static Char8 operator *(Char8 left, Char8 right) => new Char8((byte)(left.m_value * right.m_value));
        public static Char8 operator /(Char8 left, Char8 right) => new Char8((byte)(left.m_value / right.m_value));
        public static Char8 operator %(Char8 left, Char8 right) => new Char8((byte)(left.m_value % right.m_value));

        public static Char8 operator &(Char8 left, Char8 right) => new Char8((byte)(left.m_value & right.m_value));
        public static Char8 operator |(Char8 left, Char8 right) => new Char8((byte)(left.m_value | right.m_value));
        public static Char8 operator ^(Char8 left, Char8 right) => new Char8((byte)(left.m_value ^ right.m_value));
        public static Char8 operator ~(Char8 value) => new Char8((byte)(~value.m_value));

        public static Char8 operator <<(Char8 value, int shift) => new Char8((byte)(value.m_value << (shift & 7)));
        public static Char8 operator >>(Char8 value, int shift) => new Char8((byte)(value.m_value >> (shift & 7)));

        public static Char8 operator ++(Char8 value) => new Char8((byte)(value.m_value + 1));
        public static Char8 operator --(Char8 value) => new Char8((byte)(value.m_value - 1));

        public static Char8 operator +(Char8 value) => value;
        public static Char8 operator -(Char8 value) => new Char8((byte)-value.m_value);

        // ========================================================================
        // String <-> Char8[] interop
        // ========================================================================

        /// <summary>
        /// Encodes a string to a <c>Char8[]</c> assuming Latin-1 (ISO-8859-1). Throws if any char is
        /// outside the 0x00..0xFF range. This matches Python's <c>s.encode('latin-1')</c>.
        /// </summary>
        public static Char8[] FromStringLatin1(string s)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            var result = new Char8[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if ((uint)c > 0xFF)
                    throw new ArgumentException($"Character '{c}' (U+{(int)c:X4}) at index {i} cannot be encoded in Latin-1.", nameof(s));
                result[i] = new Char8((byte)c);
            }
            return result;
        }

        /// <summary>
        /// Encodes a string to a <c>Char8[]</c> assuming ASCII. Throws if any char is outside 0x00..0x7F.
        /// This matches Python's <c>s.encode('ascii')</c>.
        /// </summary>
        public static Char8[] FromStringAscii(string s)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            var result = new Char8[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if ((uint)c > 0x7F)
                    throw new ArgumentException($"Character '{c}' (U+{(int)c:X4}) at index {i} is not ASCII.", nameof(s));
                result[i] = new Char8((byte)c);
            }
            return result;
        }

        /// <summary>
        /// Encodes a string as UTF-8 bytes, returning them as <c>Char8[]</c>. This matches Python's
        /// <c>s.encode('utf-8')</c>.
        /// </summary>
        public static Char8[] FromStringUtf8(string s)
        {
            if (s is null) throw new ArgumentNullException(nameof(s));
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            var result = new Char8[bytes.Length];
            for (int i = 0; i < bytes.Length; i++) result[i] = new Char8(bytes[i]);
            return result;
        }

        /// <summary>Copies a <c>byte[]</c> into a new <c>Char8[]</c>.</summary>
        public static Char8[] FromBytes(byte[] bytes)
        {
            if (bytes is null) throw new ArgumentNullException(nameof(bytes));
            var result = new Char8[bytes.Length];
            for (int i = 0; i < bytes.Length; i++) result[i] = new Char8(bytes[i]);
            return result;
        }

        /// <summary>Decodes a <c>Char8[]</c> as Latin-1 into a string. Lossless for all bytes 0x00..0xFF.</summary>
        public static unsafe string ToStringLatin1(ReadOnlySpan<Char8> chars)
        {
            if (chars.Length == 0) return string.Empty;
            string result = new string('\0', chars.Length);
            fixed (char* dst = result)
            {
                for (int i = 0; i < chars.Length; i++) dst[i] = (char)chars[i].m_value;
            }
            return result;
        }

        /// <summary>Decodes a <c>Char8[]</c> as ASCII into a string. Throws if any byte > 0x7F.</summary>
        public static string ToStringAscii(ReadOnlySpan<Char8> chars)
        {
            if (chars.Length == 0) return string.Empty;
            for (int i = 0; i < chars.Length; i++)
            {
                if (chars[i].m_value > 0x7F)
                    throw new ArgumentException($"Byte 0x{chars[i].m_value:X2} at index {i} is not ASCII.");
            }
            return ToStringLatin1(chars);
        }

        /// <summary>Decodes a <c>Char8[]</c> as UTF-8 into a string.</summary>
        public static string ToStringUtf8(ReadOnlySpan<Char8> chars)
        {
            if (chars.Length == 0) return string.Empty;
            var bytes = new byte[chars.Length];
            for (int i = 0; i < chars.Length; i++) bytes[i] = chars[i].m_value;
            return Encoding.UTF8.GetString(bytes);
        }

        /// <summary>Copies a <c>Char8[]</c> into a new <c>byte[]</c>.</summary>
        public static byte[] ToBytes(ReadOnlySpan<Char8> chars)
        {
            var bytes = new byte[chars.Length];
            for (int i = 0; i < chars.Length; i++) bytes[i] = chars[i].m_value;
            return bytes;
        }

        // ========================================================================
        // IUtfChar-like static API (mirrors System.IUtfChar<T> but public)
        // ========================================================================

        /// <inheritdoc cref="char.ConvertFromUtf32(int)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 CastFrom(byte value) => new Char8(value);

        /// <summary>Casts a <see cref="char"/> to <see cref="Char8"/> by truncating to 8 bits.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 CastFrom(char value) => new Char8((byte)value);

        /// <summary>Casts an <see cref="int"/> to <see cref="Char8"/> by truncating to 8 bits.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 CastFrom(int value) => new Char8((byte)value);

        /// <summary>Casts a <see cref="uint"/> to <see cref="Char8"/> by truncating to 8 bits.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 CastFrom(uint value) => new Char8((byte)value);

        /// <summary>Casts a <see cref="ulong"/> to <see cref="Char8"/> by truncating to 8 bits.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 CastFrom(ulong value) => new Char8((byte)value);

        /// <summary>Casts a <see cref="Char8"/> to a <see cref="uint"/> (zero-extends).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint CastToUInt32(Char8 value) => value.m_value;

        // ========================================================================
        // Binary read/write helpers (1-byte trivial cases of IBinaryInteger.Try*)
        // ========================================================================

        /// <summary>Writes the value as a single byte to <paramref name="destination"/>.</summary>
        public bool TryWriteLittleEndian(Span<byte> destination, out int bytesWritten)
        {
            if (destination.IsEmpty) { bytesWritten = 0; return false; }
            destination[0] = m_value;
            bytesWritten = 1;
            return true;
        }

        /// <inheritdoc cref="TryWriteLittleEndian(Span{byte}, out int)"/>
        public bool TryWriteBigEndian(Span<byte> destination, out int bytesWritten)
            => TryWriteLittleEndian(destination, out bytesWritten);

        /// <summary>Reads a <see cref="Char8"/> from the last byte of <paramref name="source"/>.</summary>
        public static bool TryReadLittleEndian(ReadOnlySpan<byte> source, bool isUnsigned, out Char8 value)
        {
            if (source.IsEmpty) { value = default; return true; }
            if (!isUnsigned && (sbyte)source[0] < 0) { value = default; return false; }
            if (source.Length > 1)
            {
                for (int i = 1; i < source.Length; i++)
                {
                    if (source[i] != 0) { value = default; return false; }
                }
            }
            value = new Char8(source[0]);
            return true;
        }

        /// <inheritdoc cref="TryReadLittleEndian(ReadOnlySpan{byte}, bool, out Char8)"/>
        public static bool TryReadBigEndian(ReadOnlySpan<byte> source, bool isUnsigned, out Char8 value)
        {
            if (source.IsEmpty) { value = default; return true; }
            byte last = source[^1];
            if (!isUnsigned && (sbyte)last < 0) { value = default; return false; }
            if (source.Length > 1)
            {
                for (int i = 0; i < source.Length - 1; i++)
                {
                    if (source[i] != 0) { value = default; return false; }
                }
            }
            value = new Char8(last);
            return true;
        }

        /// <summary>The shortest bit length needed to represent the value (1..8).</summary>
        public int GetShortestBitLength()
            => m_value == 0 ? 0 : 32 - BitOperations.LeadingZeroCount(m_value);

        /// <summary>Always returns 1 — a Char8 is a single byte.</summary>
        public int GetByteCount() => 1;

        // ========================================================================
        // Other helpers (Char.cs parity)
        // ========================================================================

        /// <summary>Returns Max/Min/etc. — INumberBase-style one-offs.</summary>
        public static Char8 Abs(Char8 value) => value;
        public static Char8 Max(Char8 x, Char8 y) => x.m_value >= y.m_value ? x : y;
        public static Char8 Min(Char8 x, Char8 y) => x.m_value <= y.m_value ? x : y;
        public static bool IsZero(Char8 value) => value.m_value == 0;
        public static bool IsEvenInteger(Char8 value) => (value.m_value & 1) == 0;
        public static bool IsOddInteger(Char8 value) => (value.m_value & 1) != 0;
        public static bool IsPow2(Char8 value) => value.m_value != 0 && (value.m_value & (value.m_value - 1)) == 0;
        public static Char8 Log2(Char8 value)
            => new Char8((byte)(value.m_value == 0 ? 0 : 31 - BitOperations.LeadingZeroCount(value.m_value)));

        /// <summary>Leading zero count in 8-bit width.</summary>
        public static Char8 LeadingZeroCount(Char8 value)
            => new Char8((byte)(BitOperations.LeadingZeroCount(value.m_value) - 24));

        /// <summary>Trailing zero count in 8-bit width (returns 8 for Char8.MinValue).</summary>
        public static Char8 TrailingZeroCount(Char8 value)
            => new Char8((byte)(value.m_value == 0 ? 8 : BitOperations.TrailingZeroCount(value.m_value)));

        /// <summary>Population count (number of set bits).</summary>
        public static Char8 PopCount(Char8 value) => new Char8((byte)BitOperations.PopCount(value.m_value));

        /// <summary>Rotate left within 8 bits.</summary>
        public static Char8 RotateLeft(Char8 value, int rotateAmount)
        {
            int r = rotateAmount & 7;
            return new Char8((byte)((value.m_value << r) | (value.m_value >> (8 - r) & 0xFF)));
        }

        /// <summary>Rotate right within 8 bits.</summary>
        public static Char8 RotateRight(Char8 value, int rotateAmount)
        {
            int r = rotateAmount & 7;
            return new Char8((byte)((value.m_value >> r) | (value.m_value << (8 - r) & 0xFF)));
        }
    }
}
