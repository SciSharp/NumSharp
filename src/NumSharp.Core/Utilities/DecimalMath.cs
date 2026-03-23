// Decimal math functions for NumSharp kernel operations.
// Based on algorithms from DecimalMath by Nathan P. Jones (MIT License).
// https://github.com/nathanpjones/DecimalMath

using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    /// Provides transcendental math functions for decimal type.
    /// .NET does not provide these natively since decimal is designed for financial calculations.
    /// </summary>
    internal static class DecimalMath
    {
        private const MethodImplOptions Inline = MethodImplOptions.AggressiveInlining;

        #region Constants

        /// <summary>The pi constant. Pi radians = 180 degrees.</summary>
        public const decimal Pi = 3.1415926535897932384626433833m;

        /// <summary>Pi / 2 = 90 degrees.</summary>
        public const decimal PiHalf = 1.5707963267948966192313216916m;

        /// <summary>Pi / 4 = 45 degrees.</summary>
        public const decimal PiQuarter = 0.7853981633974483096156608458m;

        /// <summary>2 * Pi = 360 degrees.</summary>
        public const decimal TwoPi = 6.2831853071795864769252867666m;

        /// <summary>Euler's number e.</summary>
        public const decimal E = 2.7182818284590452353602874714m;

        /// <summary>Natural logarithm of 10.</summary>
        public const decimal Ln10 = 2.3025850929940456840179914547m;

        /// <summary>Natural logarithm of 2.</summary>
        public const decimal Ln2 = 0.6931471805599453094172321215m;

        /// <summary>Smallest non-zero decimal value.</summary>
        private const decimal SmallestNonZero = 0.0000000000000000000000000001m;

        #endregion

        #region Sqrt

        /// <summary>
        /// Returns the square root of a decimal using the Babylonian method.
        /// </summary>
        /// <param name="s">A non-negative number.</param>
        [MethodImpl(Inline)]
        public static decimal Sqrt(decimal s)
        {
            if (s < 0)
                throw new ArgumentException("Square root not defined for negative values.", nameof(s));

            if (s == 0 || s == SmallestNonZero)
                return 0;

            decimal x;
            var halfS = s / 2m;
            var lastX = -1m;
            decimal nextX;

            // Start with hardware sqrt estimate
            x = (decimal)Math.Sqrt(decimal.ToDouble(s));

            while (true)
            {
                nextX = x / 2m + halfS / x;
                if (nextX == x || nextX == lastX)
                    break;
                lastX = x;
                x = nextX;
            }

            return nextX;
        }

        #endregion

        #region Pow

        /// <summary>
        /// Returns x raised to the power y.
        /// </summary>
        [MethodImpl(Inline)]
        public static decimal Pow(decimal x, decimal y)
        {
            decimal result;
            var isNegativeExponent = false;

            if (y < 0)
            {
                isNegativeExponent = true;
                y = Math.Abs(y);
            }

            if (y == 0)
            {
                result = 1;
            }
            else if (y == 1)
            {
                result = x;
            }
            else
            {
                var t = decimal.Truncate(y);

                if (y == t)
                {
                    // Integer power - use exponentiation by squaring
                    result = ExpBySquaring(x, y);
                }
                else
                {
                    // Fractional power: x^y = x^t * x^(y-t) = x^t * e^((y-t)*ln(x))
                    result = ExpBySquaring(x, t) * Exp((y - t) * Log(x));
                }
            }

            if (isNegativeExponent)
            {
                if (result == 0)
                    throw new OverflowException("Negative power of 0 is undefined.");
                result = 1 / result;
            }

            return result;
        }

        /// <summary>
        /// Exponentiation by squaring for integer powers.
        /// </summary>
        [MethodImpl(Inline)]
        private static decimal ExpBySquaring(decimal x, decimal y)
        {
            var result = 1m;
            var multiplier = x;

            while (y > 0)
            {
                if ((y % 2) == 1)
                {
                    result *= multiplier;
                    y -= 1;
                    if (y == 0)
                        break;
                }
                multiplier *= multiplier;
                y /= 2;
            }

            return result;
        }

        #endregion

        #region Exp

        /// <summary>
        /// Returns e raised to the specified power.
        /// </summary>
        [MethodImpl(Inline)]
        public static decimal Exp(decimal d)
        {
            decimal result;
            decimal nextAdd;
            int iteration;
            bool reciprocal;
            decimal t;

            reciprocal = d < 0;
            d = Math.Abs(d);

            t = decimal.Truncate(d);

            if (d == 0)
            {
                result = 1;
            }
            else if (d == 1)
            {
                result = E;
            }
            else if (Math.Abs(d) > 1 && t != d)
            {
                // Split into integer and fractional parts
                result = Exp(t) * Exp(d - t);
            }
            else if (d == t)
            {
                // Integer power
                result = ExpBySquaring(E, d);
            }
            else
            {
                // Fractional power < 1: Taylor series
                iteration = 0;
                nextAdd = 0;
                result = 0;

                while (true)
                {
                    if (iteration == 0)
                    {
                        nextAdd = 1;
                    }
                    else
                    {
                        nextAdd *= d / iteration;
                    }

                    if (nextAdd == 0)
                        break;

                    result += nextAdd;
                    iteration += 1;
                }
            }

            if (reciprocal)
                result = 1 / result;

            return result;
        }

        #endregion

        #region Log

        /// <summary>
        /// Returns the natural (base e) logarithm of a number.
        /// </summary>
        [MethodImpl(Inline)]
        public static decimal Log(decimal d)
        {
            if (d < 0)
                throw new ArgumentException("Natural logarithm is complex for negative values.", nameof(d));
            if (d == 0)
                throw new OverflowException("Natural logarithm is negative infinity at zero.");

            if (d == 1)
                return 0;

            if (d >= 1)
            {
                var power = 0m;
                var x = d;
                while (x > 1)
                {
                    x /= 10;
                    power += 1;
                }
                return Log(x) + power * Ln10;
            }

            // For 0 < d < 1, use faster-converging series
            decimal y;
            decimal ySquared;

            var iteration = 0;
            var exponent = 0m;
            var nextAdd = 0m;
            var result = 0m;

            y = (d - 1) / (d + 1);
            ySquared = y * y;

            while (true)
            {
                if (iteration == 0)
                {
                    exponent = 2 * y;
                }
                else
                {
                    exponent = exponent * ySquared;
                }

                nextAdd = exponent / (2 * iteration + 1);

                if (nextAdd == 0)
                    break;

                result += nextAdd;
                iteration += 1;
            }

            return result;
        }

        /// <summary>
        /// Returns the base 10 logarithm of a number.
        /// </summary>
        [MethodImpl(Inline)]
        public static decimal Log10(decimal d)
        {
            if (d < 0)
                throw new ArgumentException("Logarithm is complex for negative values.", nameof(d));
            if (d == 0)
                throw new OverflowException("Logarithm is negative infinity at zero.");

            return Log(d) / Ln10;
        }

        #endregion

        #region Trigonometric

        /// <summary>
        /// Returns the angle whose tangent is the quotient of two numbers.
        /// </summary>
        /// <param name="y">The y coordinate.</param>
        /// <param name="x">The x coordinate.</param>
        /// <returns>Angle in radians, -pi to pi.</returns>
        [MethodImpl(Inline)]
        public static decimal ATan2(decimal y, decimal x)
        {
            if (x == 0 && y == 0)
                return 0;

            if (x == 0)
                return y > 0 ? PiHalf : -PiHalf;

            if (y == 0)
                return x > 0 ? 0 : Pi;

            var aTan = ATan(y / x);

            if (x > 0)
                return aTan;

            return y > 0 ? aTan + Pi : aTan - Pi;
        }

        /// <summary>
        /// Returns the angle whose tangent is the specified number.
        /// </summary>
        [MethodImpl(Inline)]
        public static decimal ATan(decimal x)
        {
            if (x == -1)
                return -PiQuarter;
            if (x == 0)
                return 0;
            if (x == 1)
                return PiQuarter;

            // Force to -1..1 range for faster convergence
            if (x < -1)
                return -PiHalf - ATan(1 / x);
            if (x > 1)
                return PiHalf - ATan(1 / x);

            var result = 0m;
            var doubleIteration = 0;
            var y = (x * x) / (1 + x * x);
            var nextAdd = 0m;

            while (true)
            {
                if (doubleIteration == 0)
                {
                    nextAdd = x / (1 + x * x);
                }
                else
                {
                    nextAdd *= y * doubleIteration / (doubleIteration + 1);
                }

                if (nextAdd == 0)
                    break;

                result += nextAdd;
                doubleIteration += 2;
            }

            return result;
        }

        #endregion

        #region Helpers

        /// <summary>
        /// Computes remainder with maximum precision retained.
        /// </summary>
        [MethodImpl(Inline)]
        internal static decimal Remainder(decimal d1, decimal d2)
        {
            if (Math.Abs(d1) < Math.Abs(d2))
                return d1;

            var timesInto = decimal.Truncate(d1 / d2);
            return d1 - timesInto * d2;
        }

        #endregion
    }
}
