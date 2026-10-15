using System;
using System.Collections.Generic;
using System.Numerics;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================================
        //  np.binary_repr / np.base_repr — ports of numpy/_core/numeric.py (binary_repr / base_repr).
        //
        //  These are SCALAR integer → string formatters, not array/ufunc operations: NumPy takes a single
        //  Python int (binary_repr via operator.index; base_repr via int()), so there is no NDArray operand,
        //  no dtype loop and no kernel. BigInteger is used internally so uint64 max, 2**63, and arbitrary
        //  Python-int magnitudes round-trip EXACTLY — NumPy does this arithmetic in unbounded Python ints,
        //  and the two's-complement width path (2**(poswidth+1) + num) overflows int64 for values near the
        //  signed boundary. The long overloads are the common entry; a ulong / larger magnitude binds the
        //  BigInteger overload through its implicit conversion (ulong→long is not implicit).
        // =====================================================================================

        /// <summary>
        ///     Returns the binary (base-2) string of an integer (np.binary_repr). WITHOUT <paramref name="width"/>
        ///     a negative number is rendered with a leading minus sign; WITH <paramref name="width"/> a negative
        ///     number is rendered as its two's complement to that width. About 25× faster than
        ///     <see cref="base_repr(long,int,int)"/> with base 2.
        /// </summary>
        /// <param name="num">The integer to convert.</param>
        /// <param name="width">Optional field width: the result is left-zero-padded (positive input) or sign-extended with 1s (negative, two's-complement) to at least this many characters. Must be at least the number of significant bits or a <see cref="ValueError"/> is raised.</param>
        /// <returns>The binary representation, or the two's complement when <paramref name="num"/> is negative and <paramref name="width"/> is given.</returns>
        /// <exception cref="ValueError"><paramref name="width"/> is smaller than the number of bits required to represent <paramref name="num"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html</remarks>
        public static string binary_repr(long num, int? width = null) => binary_repr((BigInteger)num, width);

        /// <summary>
        ///     Arbitrary-precision overload of <see cref="binary_repr(long,int?)"/> — accepts <c>ulong</c>
        ///     (via its implicit conversion) and integers beyond the 64-bit range, matching NumPy's
        ///     unbounded-Python-int input.
        /// </summary>
        /// <param name="num">The integer to convert.</param>
        /// <param name="width">Optional field width (see <see cref="binary_repr(long,int?)"/>).</param>
        /// <returns>The binary representation, or the two's complement when <paramref name="num"/> is negative and <paramref name="width"/> is given.</returns>
        /// <exception cref="ValueError"><paramref name="width"/> is smaller than the number of bits required to represent <paramref name="num"/>.</exception>
        public static string binary_repr(BigInteger num, int? width = null)
        {
            if (num.IsZero)
            {
                // NumPy: '0' * (width or 1) — a 0/None width becomes 1; a negative width yields '' ('0'*neg).
                int rep = (width.HasValue && width.Value != 0) ? width.Value : 1;
                return rep <= 0 ? string.Empty : new string('0', rep);
            }

            if (num.Sign > 0)
            {
                string binary = ToBinary(num);
                int binwidth = binary.Length;
                ThrowIfInsufficientWidth(width, binwidth);
                int outwidth = width.HasValue ? Math.Max(binwidth, width.Value) : binwidth;
                return binary.PadLeft(outwidth, '0');
            }

            // Negative, no width: a plain minus sign in front of the magnitude's binary.
            if (!width.HasValue)
                return "-" + ToBinary(-num);

            // Negative, with width: the two's complement to that width (NumPy's negative-with-width branch).
            int poswidth = ToBinary(-num).Length;
            // gh-8679: a value exactly on a power-of-two boundary needs one fewer bit.
            if (BigInteger.Pow(2, poswidth - 1) == -num)
                poswidth--;
            BigInteger twocomp = BigInteger.Pow(2, poswidth + 1) + num;   // num < 0, so this is < 2**(poswidth+1)
            string binaryTc = ToBinary(twocomp);
            int binwidthTc = binaryTc.Length;
            ThrowIfInsufficientWidth(width, binwidthTc);
            int outwidthTc = Math.Max(binwidthTc, width.Value);
            // Sign-extend with 1s to the requested width.
            return new string('1', outwidthTc - binwidthTc) + binaryTc;
        }

        /// <summary>
        ///     Returns the string of an integer in a given base 2–36 (np.base_repr). A negative number gets a
        ///     leading minus sign (NOT two's complement — that is <see cref="binary_repr(long,int?)"/>'s job).
        /// </summary>
        /// <param name="number">The integer to convert (its sign is honoured; only the magnitude is expanded).</param>
        /// <param name="base">The radix, 2–36 inclusive. Digits above 9 use uppercase A–Z.</param>
        /// <param name="padding">Number of extra leading zeros to prepend (placed AFTER the sign, so <c>base_repr(-7,5,3)</c> is <c>"-00012"</c>). Non-positive means no padding.</param>
        /// <returns>The base-<paramref name="base"/> representation of <paramref name="number"/>.</returns>
        /// <exception cref="ValueError"><paramref name="base"/> is greater than 36 or less than 2.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.base_repr.html</remarks>
        public static string base_repr(long number, int @base = 2, int padding = 0) => base_repr((BigInteger)number, @base, padding);

        /// <summary>
        ///     Arbitrary-precision overload of <see cref="base_repr(long,int,int)"/> — accepts <c>ulong</c>
        ///     (via its implicit conversion) and integers beyond the 64-bit range.
        /// </summary>
        /// <param name="number">The integer to convert.</param>
        /// <param name="base">The radix, 2–36 inclusive.</param>
        /// <param name="padding">Number of extra leading zeros to prepend (see <see cref="base_repr(long,int,int)"/>).</param>
        /// <returns>The base-<paramref name="base"/> representation of <paramref name="number"/>.</returns>
        /// <exception cref="ValueError"><paramref name="base"/> is greater than 36 or less than 2.</exception>
        public static string base_repr(BigInteger number, int @base = 2, int padding = 0)
        {
            const string digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            if (@base > 36)
                throw new ValueError("Bases greater than 36 not handled in base_repr.");
            if (@base < 2)
                throw new ValueError("Bases less than 2 not handled in base_repr.");

            BigInteger num = BigInteger.Abs(number);
            // NumPy builds a list [least-significant digit, …], then appends the whole padding block and the
            // sign as single entries, reverses the LIST and joins — so the padding lands between the digits
            // and the sign (base_repr(-7,5,3) == "-00012").
            var parts = new List<string>();
            while (num > 0)
            {
                parts.Add(digits[(int)(num % @base)].ToString());
                num /= @base;
            }
            if (padding > 0)
                parts.Add(new string('0', padding));
            if (number.Sign < 0)
                parts.Add("-");
            if (parts.Count == 0)
                parts.Add("0");   // NumPy's `reversed(res or '0')`: the zero, no-padding case
            parts.Reverse();
            return string.Concat(parts);
        }

        /// <summary>Renders a POSITIVE <see cref="BigInteger"/> as its minimal binary string (MSB first, no leading zeros).</summary>
        /// <param name="v">A strictly positive value.</param>
        /// <returns>The binary digits of <paramref name="v"/>.</returns>
        private static string ToBinary(BigInteger v)
        {
            int bits = (int)v.GetBitLength();   // v > 0 ⇒ position of the top set bit + 1 = digit count
            var chars = new char[bits];
            for (int i = bits - 1; i >= 0; i--)
            {
                chars[i] = (char)('0' + (int)(v & BigInteger.One));
                v >>= 1;
            }
            return new string(chars);
        }

        /// <summary>Raises NumPy's binary_repr insufficient-width <see cref="ValueError"/> when <paramref name="width"/> is smaller than the required bit count.</summary>
        /// <param name="width">The requested field width (a <c>null</c> width never raises).</param>
        /// <param name="binwidth">The number of significant bits the value needs.</param>
        /// <exception cref="ValueError"><paramref name="width"/> is present and less than <paramref name="binwidth"/>.</exception>
        private static void ThrowIfInsufficientWidth(int? width, int binwidth)
        {
            if (width.HasValue && width.Value < binwidth)
                throw new ValueError($"Insufficient bit width={width.Value} provided for binwidth={binwidth}");
        }
    }
}
