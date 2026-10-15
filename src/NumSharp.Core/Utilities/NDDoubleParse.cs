using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     The correctly rounded double of a decimal number's text, fast: Clinger's fast path, then the Eisel-Lemire
    ///     algorithm as fast_float implements it (<c>compute_float</c>, with the Mushtak-Lemire proof that 19 significant
    ///     digits never need a fallback), and .NET's own parser only for the rare text with more than 19 significant digits
    ///     whose 19-digit truncation does not decide the rounding.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why.</b> The text readers (<c>np.loadtxt</c>, <c>np.fromstring</c>, <c>np.fromfile</c>) parse one double per
    ///     field, and .NET's <see cref="double.Parse(ReadOnlySpan{char}, NumberStyles, IFormatProvider)"/> costs about
    ///     52 ns for a short decimal and 130-140 ns for the 17-19 digit numbers <c>repr</c> and <c>%.18e</c> write — more
    ///     than everything else a field costs. CPython's <c>_Py_dg_strtod</c>, which NumPy calls, is correctly rounded like
    ///     .NET's parser, so every path here returns the one correctly rounded double (round half to even), the signed
    ///     zero of an underflow and the infinity of an overflow: the bits NumPy produces.
    ///     </para>
    ///     <para>
    ///     <b>Input contract.</b> The text is a VALIDATED decimal number: an optional sign, digits with an optional fraction
    ///     (at least one digit in all), and an optional exponent that has a digit. The callers match their own grammar first
    ///     (NumPy's differs per reader), so this class never rejects text.
    ///     </para>
    /// </remarks>
    internal static class NDDoubleParse
    {
        /// <summary>The smallest decimal exponent with a table entry (below it every 19-digit number underflows to 0).</summary>
        private const int SmallestPowerOfTen = -342;

        /// <summary>The largest decimal exponent with a table entry (above it every non-zero number overflows).</summary>
        private const int LargestPowerOfTen = 308;

        /// <summary>
        ///     fast_float's <c>power_of_five_128</c>: for every decimal exponent q in [-342, 308], the 128-bit truncated,
        ///     normalized 5^q (high word first). Built once from exact integers, as fast_float's table_generation.py does.
        /// </summary>
        private static readonly ulong[] PowerOfFive128 = BuildPowerOfFiveTable();

        /// <summary>The exact powers of ten Clinger's fast path multiplies or divides by (10^0 to 10^22).</summary>
        private static readonly double[] ExactPowersOfTen =
        {
            1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20,
            1e21, 1e22,
        };

        /// <summary>
        ///     The extent of a decimal number at <paramref name="start"/> in CPython's <c>_Py_dg_strtod</c> grammar (which NumPy's
        ///     text readers reach through <c>PyOS_string_to_double</c>): an optional sign, digits with an optional fraction —
        ///     at least one digit in all, so <c>"5."</c> and <c>".5"</c> are numbers and <c>"."</c> is not — and an exponent
        ///     only when a digit follows its optional sign (<c>"1e+"</c> stops before the <c>e</c>).
        /// </summary>
        /// <typeparam name="TChar"><see cref="char"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="start">Where the number would start.</param>
        /// <returns>The index just past the number, or <paramref name="start"/> when no number starts there.</returns>
        internal static int MatchDecimal<TChar>(ReadOnlySpan<TChar> s, int start) where TChar : unmanaged, IBinaryInteger<TChar>
        {
            int p = start;
            int c = Ch(s, p);
            if (c == '+' || c == '-')
                p++;
            bool digits = false;
            while (IsDigit(Ch(s, p)))
            {
                p++;
                digits = true;
            }
            if (Ch(s, p) == '.')
            {
                p++;
                while (IsDigit(Ch(s, p)))
                {
                    p++;
                    digits = true;
                }
            }
            if (!digits)
                return start;
            if ((Ch(s, p) | 0x20) == 'e')
            {
                int q = p + 1;
                int d = Ch(s, q);
                if (d == '+' || d == '-')
                    q++;
                if (IsDigit(Ch(s, q)))
                {
                    while (IsDigit(Ch(s, q)))
                        q++;
                    p = q;
                }
            }
            return p;
        }

        /// <summary>
        ///     Parses <paramref name="s"/> when ALL of it is one decimal number (<see cref="MatchDecimal{TChar}"/>), with no
        ///     whitespace, word or other text around it.
        /// </summary>
        /// <typeparam name="TChar"><see cref="char"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The text (already stripped by the caller's own whitespace rule).</param>
        /// <param name="value">The correctly rounded value, when it matched.</param>
        /// <returns>True when the whole text is one decimal number.</returns>
        internal static bool TryParseWhole<TChar>(ReadOnlySpan<TChar> s, out double value) where TChar : unmanaged, IBinaryInteger<TChar>
        {
            if (s.Length > 0 && MatchDecimal(s, 0) == s.Length)
            {
                value = Parse(s);
                return true;
            }
            value = 0;
            return false;
        }

        /// <summary>
        ///     The correctly rounded double of a validated decimal number (see the class remarks for the contract).
        /// </summary>
        /// <typeparam name="TChar">The text's unit: <see cref="char"/> (a field of <c>np.loadtxt</c>) or <see cref="byte"/>
        ///     (the UTF-8 / ASCII text of <c>np.fromstring</c> and <c>np.fromfile</c>).</typeparam>
        /// <param name="s">The number's text, nothing before or after it.</param>
        /// <returns>The value.</returns>
        internal static double Parse<TChar>(ReadOnlySpan<TChar> s) where TChar : unmanaged, IBinaryInteger<TChar>
        {
            int p = 0;
            bool negative = false;
            int c = Ch(s, 0);
            if (c == '-' || c == '+')
            {
                negative = c == '-';
                p++;
            }

            // The digits, accumulated without a check: up to 19 significant digits fit a ulong, and a longer number is
            // re-accumulated below.
            int startDigits = p;
            ulong w = 0;
            while (p < s.Length && IsDigit(c = Ch(s, p)))
            {
                w = unchecked(10 * w + (uint)(c - '0'));
                p++;
            }
            int endInteger = p;
            long digitCount = endInteger - startDigits;
            long exponent = 0;
            int fractionStart = p, fractionEnd = p;
            if (p < s.Length && Ch(s, p) == '.')
            {
                p++;
                fractionStart = p;
                while (p < s.Length && IsDigit(c = Ch(s, p)))
                {
                    w = unchecked(10 * w + (uint)(c - '0'));
                    p++;
                }
                fractionEnd = p;
                exponent = -(long)(fractionEnd - fractionStart);
                digitCount += fractionEnd - fractionStart;
            }
            long exponentNumber = 0;
            if (p < s.Length && (Ch(s, p) | 0x20) == 'e')
            {
                p++;
                bool negativeExponent = false;
                c = Ch(s, p);
                if (c == '-' || c == '+')
                {
                    negativeExponent = c == '-';
                    p++;
                }
                while (p < s.Length && IsDigit(c = Ch(s, p)))
                {
                    // saturate far past any exponent that matters (|q| > 342 decides 0 / infinity)
                    if (exponentNumber < 0x10000000)
                        exponentNumber = 10 * exponentNumber + (c - '0');
                    p++;
                }
                if (negativeExponent)
                    exponentNumber = -exponentNumber;
                exponent += exponentNumber;
            }

            bool tooManyDigits = false;
            if (digitCount > 19)
            {
                // Leading zeros (and the point) are not significant: 0.000...0123 still fits.
                for (int k = startDigits; k < s.Length; k++)
                {
                    int z = Ch(s, k);
                    if (z == '0')
                        digitCount--;
                    else if (z != '.')
                        break;
                }
                if (digitCount > 19)
                {
                    // Truncate to the first 19 significant digits and move the exponent with them.
                    tooManyDigits = true;
                    const ulong minimalNineteenDigitInteger = 1000000000000000000UL;
                    w = 0;
                    int q = startDigits;
                    while (w < minimalNineteenDigitInteger && q < endInteger)
                        w = w * 10 + (uint)(Ch(s, q++) - '0');
                    if (w >= minimalNineteenDigitInteger)
                        exponent = endInteger - q + exponentNumber;
                    else
                    {
                        q = fractionStart;
                        while (w < minimalNineteenDigitInteger && q < fractionEnd)
                            w = w * 10 + (uint)(Ch(s, q++) - '0');
                        exponent = fractionStart - q + exponentNumber;
                    }
                }
            }

            // Clinger: the mantissa and the power of ten are exact doubles, so one IEEE operation rounds correctly.
            if (!tooManyDigits && exponent >= -22 && exponent <= 22 && w <= 1UL << 53)
            {
                double v = w;
                v = exponent < 0 ? v / ExactPowersOfTen[-exponent] : v * ExactPowersOfTen[exponent];
                return negative ? -v : v;
            }

            ComputeFloat(exponent, w, out ulong mantissa, out int power2);
            if (tooManyDigits)
            {
                // The truncated digits decide the rounding only when w and w + 1 round to the same double.
                ComputeFloat(exponent, w + 1, out ulong m2, out int p2);
                if (m2 != mantissa || p2 != power2)
                    return Fallback(s);
            }
            ulong bits = mantissa | ((ulong)power2 << 52);
            if (negative)
                bits |= 1UL << 63;
            return BitConverter.UInt64BitsToDouble(bits);
        }

        /// <summary>
        ///     fast_float's <c>compute_float&lt;binary64&gt;</c>: the binary exponent and the 52-bit stored mantissa of the
        ///     double nearest <c>w * 10^q</c> (ties to even), from a 128-bit approximation of 5^q that is always precise
        ///     enough for <paramref name="w"/> below 10^19.
        /// </summary>
        /// <param name="q">The decimal exponent.</param>
        /// <param name="w">The decimal significand (19 digits at most).</param>
        /// <param name="mantissa">The stored mantissa (the implicit bit set only for the subnormal that rounds up into the
        ///     smallest normal, which the OR with the exponent field absorbs).</param>
        /// <param name="power2">The biased exponent field: 0 for zero / a subnormal, 0x7FF for infinity.</param>
        private static void ComputeFloat(long q, ulong w, out ulong mantissa, out int power2)
        {
            if (w == 0 || q < SmallestPowerOfTen)
            {
                mantissa = 0;
                power2 = 0;
                return;
            }
            if (q > LargestPowerOfTen)
            {
                mantissa = 0;
                power2 = 0x7FF;
                return;
            }
            int lz = BitOperations.LeadingZeroCount(w);
            w <<= lz;

            // compute_product_approximation<55>: the high product suffices unless its low 9 bits are all ones.
            int index = 2 * (int)(q - SmallestPowerOfTen);
            ulong high = Math.BigMul(w, PowerOfFive128[index], out ulong low);
            if ((high & 0x1FF) == 0x1FF)
            {
                ulong secondHigh = Math.BigMul(w, PowerOfFive128[index + 1], out _);
                low += secondHigh;
                if (secondHigh > low)
                    high++;
            }

            int upperbit = (int)(high >> 63);
            int shift = upperbit + 64 - 52 - 3;
            mantissa = high >> shift;
            // detail::power(q) = ((152170 + 65536) * q >> 16) + 63; minus binary64's minimum exponent (-1023)
            power2 = (int)(((152170 + 65536) * q) >> 16) + 63 + upperbit - lz + 1023;
            if (power2 <= 0)
            {
                // A subnormal (or zero): shift into place and round; it may round up into the smallest normal.
                if (-power2 + 1 >= 64)
                {
                    mantissa = 0;
                    power2 = 0;
                    return;
                }
                mantissa >>= -power2 + 1;
                mantissa += mantissa & 1;
                mantissa >>= 1;
                power2 = mantissa < 1UL << 52 ? 0 : 1;
                return;
            }

            // Exactly halfway between two doubles (possible only where 5^q fits a 64-bit word): round to even.
            if (low <= 1 && q >= -4 && q <= 23 && (mantissa & 3) == 1 && (mantissa << shift) == high)
                mantissa &= ~1UL;

            mantissa += mantissa & 1;
            mantissa >>= 1;
            if (mantissa >= 2UL << 52)
            {
                mantissa = 1UL << 52;
                power2++;
            }
            mantissa &= ~(1UL << 52);
            if (power2 >= 0x7FF)
            {
                power2 = 0x7FF;
                mantissa = 0;
            }
        }

        /// <summary>.NET's correctly rounded parser, for the undecided text with more than 19 significant digits.</summary>
        /// <typeparam name="TChar"><see cref="char"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The number's text.</param>
        /// <returns>The value.</returns>
        private static double Fallback<TChar>(ReadOnlySpan<TChar> s) where TChar : unmanaged, IBinaryInteger<TChar>
        {
            if (typeof(TChar) == typeof(char))
                return double.Parse(MemoryMarshal.Cast<TChar, char>(s), NumberStyles.Float, CultureInfo.InvariantCulture);
            return double.Parse(MemoryMarshal.Cast<TChar, byte>(s), NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        /// <summary>The unit at <paramref name="i"/> as an int, or 0 past the end.</summary>
        /// <typeparam name="TChar"><see cref="char"/> or <see cref="byte"/>.</typeparam>
        /// <param name="s">The text.</param>
        /// <param name="i">The index.</param>
        /// <returns>The unit's value.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int Ch<TChar>(ReadOnlySpan<TChar> s, int i) where TChar : unmanaged, IBinaryInteger<TChar>
            => (uint)i < (uint)s.Length ? int.CreateTruncating(s[i]) : 0;

        /// <summary>An ASCII digit.</summary>
        /// <param name="c">The unit.</param>
        /// <returns>True for '0' to '9'.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsDigit(int c) => (uint)(c - '0') <= 9;

        /// <summary>
        ///     fast_float's table_generation.py: for q &lt; 0 the 128-bit ceiling of 2^b / 5^-q (b chosen to normalize it), for
        ///     q &gt;= 0 the truncated top 128 bits of 5^q.
        /// </summary>
        /// <returns>The table, high word first for each exponent.</returns>
        private static ulong[] BuildPowerOfFiveTable()
        {
            var table = new ulong[2 * (LargestPowerOfTen - SmallestPowerOfTen + 1)];
            BigInteger power5 = BigInteger.One;
            for (int q = 0; q <= LargestPowerOfTen; q++)
            {
                BigInteger v = power5;
                int bits = (int)v.GetBitLength();
                v = bits <= 128 ? v << (128 - bits) : v >> (bits - 128);
                Store(table, q, v);
                power5 *= 5;
            }
            BigInteger inverse = BigInteger.One;
            for (int k = 1; k <= -SmallestPowerOfTen; k++)
            {
                inverse *= 5; // 5^k for q = -k
                // z: the smallest z with 2^z >= 5^k (5^k is never a power of two, so its bit length)
                int z = (int)inverse.GetBitLength();
                BigInteger c;
                if (k <= 27)
                    c = (BigInteger.One << (z + 127)) / inverse + 1;
                else
                {
                    c = (BigInteger.One << (2 * z + 128)) / inverse + 1;
                    int cb = (int)c.GetBitLength();
                    if (cb > 128)
                        c >>= cb - 128;
                }
                Store(table, -k, c);
            }
            return table;
        }

        /// <summary>Writes one 128-bit entry (high word first) for decimal exponent <paramref name="q"/>.</summary>
        /// <param name="table">The table.</param>
        /// <param name="q">The exponent.</param>
        /// <param name="v">The 128-bit value.</param>
        private static void Store(ulong[] table, int q, BigInteger v)
        {
            int index = 2 * (q - SmallestPowerOfTen);
            table[index] = (ulong)(v >> 64);
            table[index + 1] = (ulong)(v & ulong.MaxValue);
        }
    }
}
