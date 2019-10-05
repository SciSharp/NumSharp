//All rights reserved to nathanpjones, author of DecimalMath (https://github.com/nathanpjones/DecimalMath).

using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace DecimalMath
{
    /// <summary>
    ///     Contains mathematical operations performed in Decimal precision.
    /// </summary>
    public static partial class DecimalEx
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Abs(decimal a)
        {
            if (a >= 0)
                return a;

            return -a;
        }

        /// <summary>
        /// Returns the square root of a given number. 
        /// </summary>
        /// <param name="s">A non-negative number.</param>
        /// <remarks> 
        /// Uses an implementation of the "Babylonian Method".
        /// See http://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Babylonian_method 
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Sqrt(decimal s)
        {
            if (s < 0)
                throw new ArgumentException("Square root not defined for Decimal data type when less than zero!", "s");

            // Prevent divide-by-zero errors below. Dividing either
            // of the numbers below will yield a recurring 0 value
            // for halfS eventually converging on zero.
            if (s == 0 || s == SmallestNonZeroDec) return 0;

            decimal x;
            var halfS = s / 2m;
            var lastX = -1m;
            decimal nextX;

            // Begin with an estimate for the square root.
            // Use hardware to get us there quickly.
            x = (decimal)Math.Sqrt(decimal.ToDouble(s));

            while (true)
            {
                nextX = x / 2m + halfS / x;

                // The next check effectively sees if we've ran out of
                // precision for our data type.
                if (nextX == x || nextX == lastX) break;

                lastX = x;
                x = nextX;
            }

            return nextX;
        }

        /// <summary>
        /// Returns a specified number raised to the specified power.
        /// </summary>
        /// <param name="x">A number to be raised to a power.</param>
        /// <param name="y">A number that specifies a power.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Pow(decimal x, decimal y)
        {
            decimal result;
            var isNegativeExponent = false;

            // Handle negative exponents
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

                if (y == t) // Integer powers
                {
                    result = ExpBySquaring(x, y);
                }
                else // Fractional power < 1
                {
                    // See http://en.wikipedia.org/wiki/Exponent#Real_powers
                    // The next line is an optimization of Exp(y * Log(x)) for better precision
                    result = ExpBySquaring(x, t) * Exp((y - t) * Log(x));
                }
            }

            if (isNegativeExponent)
            {
                // Note, for IEEE floats this would be Infinity and not an exception...
                if (result == 0) throw new OverflowException("Negative power of 0 is undefined!");

                result = 1 / result;
            }

            return result;
        }

        /// <summary>
        /// Raises one number to an integral power.
        /// </summary>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Exponentiation_by_squaring
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        private static decimal ExpBySquaring(decimal x, decimal y)
        {
            Debug.Assert(y >= 0 && decimal.Truncate(y) == y, "Only non-negative, integer powers supported.");
            if (y < 0) throw new ArgumentOutOfRangeException("y", "Negative exponents not supported!");
            if (decimal.Truncate(y) != y) throw new ArgumentException("Exponent must be an integer!", "y");

            var result = 1m;
            var multiplier = x;

            while (y > 0)
            {
                if ((y % 2) == 1)
                {
                    result *= multiplier;
                    y -= 1;
                    if (y == 0) break;
                }

                multiplier *= multiplier;
                y /= 2;
            }

            return result;
        }

        /// <summary>
        /// Returns e raised to the specified power.
        /// </summary>
        /// <param name="d">A number specifying a power.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
                // Split up into integer and fractional
                result = Exp(t) * Exp(d - t);
            }
            else if (d == t) // Integer power
            {
                result = ExpBySquaring(E, d);
            }
            else // Fractional power < 1
            {
                // See http://mathworld.wolfram.com/ExponentialFunction.html
                iteration = 0;
                nextAdd = 0;
                result = 0;

                while (true)
                {
                    if (iteration == 0)
                    {
                        nextAdd = 1; // == Pow(d, 0) / Factorial(0) == 1 / 1 == 1
                    }
                    else
                    {
                        nextAdd *= d / iteration; // == Pow(d, iteration) / Factorial(iteration)
                    }

                    if (nextAdd == 0) break;

                    result += nextAdd;

                    iteration += 1;
                }
            }

            // Take reciprocal if this was a negative power
            // Note that result will never be zero at this point.
            if (reciprocal) result = 1 / result;

            return result;
        }

        /// <summary>
        /// Returns the natural (base e) logarithm of a specified number.
        /// </summary>
        /// <param name="d">A number whose logarithm is to be found.</param>
        /// <remarks>
        /// I'm still not satisfied with the speed. I tried several different
        /// algorithms that you can find in a historical version of this 
        /// source file. The one I settled on was the best of mediocrity.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Log(decimal d)
        {
            if (d < 0) throw new ArgumentException("Natural logarithm is a complex number for values less than zero!", "d");
            if (d == 0) throw new OverflowException("Natural logarithm is defined as negative infinity at zero which the Decimal data type can't represent!");

            if (d == 1) return 0;

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

            // See http://en.wikipedia.org/wiki/Natural_logarithm#Numerical_value
            // for more information on this faster-converging series.

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

                if (nextAdd == 0) break;

                result += nextAdd;

                iteration += 1;
            }

            return result;
        }

        /// <summary>
        /// Returns the logarithm of a specified number in a specified base.
        /// </summary>
        /// <param name="d">A number whose logarithm is to be found.</param>
        /// <param name="newBase">The base of the logarithm.</param>
        /// <remarks>
        /// This is a relatively naive implementation that simply divides the
        /// natural log of <paramref name="d"/> by the natural log of the base.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Log(decimal d, decimal newBase)
        {
            // Short circuit the checks below if d is 1 because
            // that will yield 0 in the numerator below and give us
            // 0 for any base, even ones that would yield infinity.
            if (d == 1) return 0m;

            if (newBase == 1) throw new InvalidOperationException("Logarithm for base 1 is undefined.");
            if (d < 0) throw new ArgumentException("Logarithm is a complex number for values less than zero!", nameof(d));
            if (d == 0) throw new OverflowException("Logarithm is defined as negative infinity at zero which the Decimal data type can't represent!");
            if (newBase < 0) throw new ArgumentException("Logarithm base would be a complex number for values less than zero!", nameof(newBase));
            if (newBase == 0) throw new OverflowException("Logarithm base would be negative infinity at zero which the Decimal data type can't represent!");

            return Log(d) / Log(newBase);
        }

        /// <summary>
        /// Returns the base 10 logarithm of a specified number.
        /// </summary>
        /// <param name="d">A number whose logarithm is to be found.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Log10(decimal d)
        {
            if (d < 0) throw new ArgumentException("Logarithm is a complex number for values less than zero!", nameof(d));
            if (d == 0) throw new OverflowException("Logarithm is defined as negative infinity at zero which the Decimal data type can't represent!");

            return Log(d) / Ln10;
        }

        /// <summary>
        /// Returns the base 2 logarithm of a specified number.
        /// </summary>
        /// <param name="d">A number whose logarithm is to be found.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Log2(decimal d)
        {
            if (d < 0) throw new ArgumentException("Logarithm is a complex number for values less than zero!", nameof(d));
            if (d == 0) throw new OverflowException("Logarithm is defined as negative infinity at zero which the Decimal data type can't represent!");

            return Log(d) / Ln2;
        }

        /// <summary>
        /// Returns the factorial of a number n expressed as n!. Factorial is
        /// calculated as follows: n * (n - 1) * (n - 2) * ... * 1
        /// </summary>
        /// <param name="n">An integer.</param>
        /// <remarks>
        /// Only supports non-negative integers.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Factorial(decimal n)
        {
            if (n < 0) throw new ArgumentException("Values less than zero are not supoprted!", "n");
            if (Decimal.Truncate(n) != n) throw new ArgumentException("Fractional values are not supoprted!", "n");

            var ret = 1m;

            for (var i = n; i >= 2; i += -1)
            {
                ret *= i;
            }

            return ret;
        }

        /// <summary>
        /// Uses the quadratic formula to factor and solve the equation ax^2 + bx + c = 0
        /// </summary>
        /// <param name="a">The coefficient of x^2.</param>
        /// <param name="b">The coefficient of x.</param>
        /// <param name="c">The constant.</param>
        /// <remarks>
        /// Will return empty results where there is no solution and for complex solutions.
        /// See http://www.wikihow.com/Factor-Second-Degree-Polynomials-%28Quadratic-Equations%29
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal[] SolveQuadratic(decimal a, decimal b, decimal c)
        {
            // Horizontal line is either 0 nowhere or everywhere so no solution.
            if ((a == 0) && (b == 0)) return new decimal[] { };

            if ((a == 0))
            {
                // This is actually a linear equation. Using quadratic would result in a
                // divide by zero so use the following equation.
                // 0 = b * x + c
                // -c = b * x
                // -c / b = x
                return new[] {-c / b};
            }

            // If all our coefficients have an absolute value less than 1,
            // then we'll lose precision in calculating the discriminant and
            // its root. Since we're solving for  ax^2 + bx + c = 0  we can
            // multiply the coefficients by whatever we want until they are 
            // in a more favorable range. In this case, we'll make sure here 
            // that at least one number is greater than 1 or less than -1.
            while ((-1 < a && a < 1) && (-1 < b && b < 1) && (-1 < c && c < 1))
            {
                a *= 10;
                b *= 10;
                c *= 10;
            }

            var discriminant = b * b - 4 * a * c;

            // Allow for a little rounding error and treat this as 0
            if (discriminant == -SmallestNonZeroDec) discriminant = 0;

            // Solution is complex -- shape does not intersect 0.
            if (discriminant < 0) return new decimal[] { };

            var sqrtOfDiscriminant = Sqrt(discriminant);

            // Select quadratic or "citardauq" depending on which one has a matching
            // sign between -b and the square root. This improves precision, sometimes
            // dramatically. See: http://math.stackexchange.com/a/56982
            var h = Math.Sign(b) == -1 ? (-b + sqrtOfDiscriminant) / (2 * a) : (2 * c) / (-b - sqrtOfDiscriminant);
            var k = Math.Sign(b) == +1 ? (-b - sqrtOfDiscriminant) / (2 * a) : (2 * c) / (-b + sqrtOfDiscriminant);

            // ax^2 + bx + c = (x - h)(x - k) 
            // (x - h)(x - k) = 0 means h and k are the values for x 
            //   that will make the equation = 0
            return h == k
                ? new[] {h}
                : new[] {h, k};
        }

        /// <summary>
        /// Returns the floor of a Decimal value at the given number of digits.
        /// </summary>
        /// <param name="value">A decimal value.</param>
        /// <param name="places">An integer representing the maximum number of digits 
        /// after the decimal point to end up with.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Floor(decimal value, int places = 0)
        {
            if (places < 0) throw new ArgumentOutOfRangeException("places", "Places must be greater than or equal to 0.");

            if (places == 0) return decimal.Floor(value);

            // At or beyond precision of decimal data type
            if (places >= 28) return value;
            
            return decimal.Floor(value * PowersOf10[places]) / PowersOf10[places];
        }

        /// <summary>
        /// Returns the ceiling of a Decimal value at the given number of digits.
        /// </summary>
        /// <param name="value">A decimal value.</param>
        /// <param name="places">An integer representing the maximum number of digits 
        /// after the decimal point to end up with.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Ceiling(decimal value, int places = 0)
        {
            if (places < 0) throw new ArgumentOutOfRangeException("places", "Places must be greater than or equal to 0.");

            if (places == 0) return decimal.Ceiling(value);

            // At or beyond precision of decimal data type
            if (places >= 28) return value;

            return decimal.Ceiling(value * PowersOf10[places]) / PowersOf10[places];
        }

        /// <summary>
        /// Calculates the greatest common factor of a and b to the highest level of
        /// precision represented by either number.
        /// </summary>
        /// <remarks>
        /// If either number is not an integer, the factor sought will be at the
        /// same precision as the most precise value.
        /// For example, 1.2 and 0.42 will yield 0.06.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal GCF(decimal a, decimal b)
        {
            // Run Euclid's algorithm
            while (true)
            {
                if (b == 0) break;
                var r = a % b;
                a = b;
                b = r;
            }

            return a;
        }

        /// <summary>
        /// Gets the greatest common factor of three or more numbers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal GCF(decimal a, decimal b, params decimal[] values)
        {
            return values.Aggregate(GCF(a, b), (current, value) => GCF(current, value));
        }

        /// <summary>
        /// Computes arithmetic-geometric mean which is the convergence of the
        /// series of the arithmetic and geometric means and their mean values.
        /// </summary>
        /// <param name="x">A number.</param>
        /// <param name="y">A number.</param>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Arithmetic-geometric_mean
        /// Originally implemented to try to get a fast approximation of the
        /// natural logarithm: http://en.wikipedia.org/wiki/Natural_logarithm#High_precision
        /// But it didn't yield a precise enough answer.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal AGMean(decimal x, decimal y)
        {
            decimal a;
            decimal g;

            // Handle special case
            if (x == 0 || y == 0) return 0;

            // Make sure signs match or we'll end up with a complex number
            var sign = Math.Sign(x);
            if (sign != Math.Sign(y))
                throw new Exception("Arithmetic geometric mean of these values is complex and cannot be expressed in Decimal data type!");

            // At this point, both signs match. If they're both negative, evaluate ag mean using them
            // as positive numbers and multiply result by -1.
            if (sign == -1)
            {
                x = decimal.Negate(x);
                y = decimal.Negate(y);
            }

            while (true)
            {
                a = x / 2 + y / 2;
                g = Sqrt(x * y);

                if (a == g) break;
                if (g == y && a == x) break;

                x = a;
                y = g;
            }

            return sign == -1 ? -a : a;
        }

        /// <summary>
        /// Calculates the average of the supplied numbers.
        /// </summary>
        /// <param name="values">The numbers to average.</param>
        /// <remarks>
        /// Simply uses LINQ's Average function, but switches to a potentially less
        /// accurate method of summing each value divided by the number of values.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Average(params decimal[] values)
        {
            decimal avg;

            try
            {
                avg = values.Average();
            }
            catch (OverflowException)
            {
                // Use less accurate method that won't overflow
                avg = values.Sum(v => v / values.Length);
            }

            return avg;
        }

        /// <summary>
        /// Gets the number of decimal places in a decimal value.
        /// </summary>
        /// <remarks>
        /// Started with something found here: http://stackoverflow.com/a/6092298/856595
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int GetDecimalPlaces(decimal dec, bool countTrailingZeros)
        {
            const int signMask = unchecked((int)0x80000000);
            const int scaleMask = 0x00FF0000;
            const int scaleShift = 16;

            int[] bits = Decimal.GetBits(dec);
            var result = (bits[3] & scaleMask) >> scaleShift; // extract exponent

            // Return immediately for values without a fractional portion or if we're counting trailing zeros
            if (countTrailingZeros || (result == 0)) return result;

            // Get a raw version of the decimal's integer
            bits[3] = bits[3] & ~unchecked(signMask | scaleMask); // clear out exponent and negative bit
            var rawValue = new decimal(bits);

            // Account for trailing zeros
            while ((result > 0) && ((rawValue % 10) == 0))
            {
                result--;
                rawValue /= 10;
            }

            return result;
        }

        /// <summary>
        /// Gets the remainder of one number divided by another number in such a way as to retain maximum precision.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Remainder(decimal d1, decimal d2)
        {
            if (Math.Abs(d1) < Math.Abs(d2)) return d1;

            var timesInto = decimal.Truncate(d1 / d2);
            var shiftingNumber = d2;
            var sign = Math.Sign(d1);

            for (var i = 0; i <= GetDecimalPlaces(d2, true); i++)
            {
                // Note that first "digit" will be the integer portion of d2
                var digit = decimal.Truncate(shiftingNumber);

                d1 -= timesInto * (digit / PowersOf10[i]);

                shiftingNumber = (shiftingNumber - digit) * 10m; // remove used digit and shift for next iteration
                if (shiftingNumber == 0m) break;
            }

            // If we've crossed zero because of the precision mismatch, 
            // we need to add a whole d2 to get a correct result.
            if (d1 != 0 && Math.Sign(d1) != sign)
            {
                d1 = Math.Sign(d2) == sign
                    ? d1 + d2
                    : d1 - d2;
            }

            return d1;
        }

        /// <summary> The pi (π) constant. Pi radians is equivalent to 180 degrees. </summary>
        /// <remarks> See http://en.wikipedia.org/wiki/Pi </remarks>
        public const decimal Pi = 3.1415926535897932384626433833m; // 180 degrees - see http://en.wikipedia.org/wiki/Pi

        /// <summary> π/2 - in radians is equivalent to 90 degrees. </summary> 
        public const decimal PiHalf = 1.5707963267948966192313216916m; //  90 degrees

        /// <summary> π/4 - in radians is equivalent to 45 degrees. </summary>
        public const decimal PiQuarter = 0.7853981633974483096156608458m; //  45 degrees

        /// <summary> π/12 - in radians is equivalent to 15 degrees. </summary>
        public const decimal PiTwelfth = 0.2617993877991494365385536153m; //  15 degrees

        /// <summary> 2π - in radians is equivalent to 360 degrees. </summary>
        public const decimal TwoPi = 6.2831853071795864769252867666m; // 360 degrees

        /// <summary>
        /// Smallest non-zero decimal value.
        /// </summary>
        public const decimal SmallestNonZeroDec = 0.0000000000000000000000000001m; // aka new decimal(1, 0, 0, false, 28); //1e-28m

        /// <summary>
        /// The e constant, also known as "Euler's number" or "Napier's constant"
        /// </summary>
        /// <remarks>
        /// Full value is 2.718281828459045235360287471352662497757, 
        /// see http://mathworld.wolfram.com/e.html
        /// </remarks>
        public const decimal E = 2.7182818284590452353602874714m;

        /// <summary>
        /// The value of the natural logarithm of 10.
        /// </summary>
        /// <remarks>
        /// Full value is: 2.30258509299404568401799145468436420760110148862877297603332790096757
        /// From: http://oeis.org/A002392/constant
        /// </remarks>
        public const decimal Ln10 = 2.3025850929940456840179914547m;

        /// <summary>
        /// The value of the natural logarithm of 2.
        /// </summary>
        /// <remarks>
        /// Full value is: .693147180559945309417232121458176568075500134360255254120680009493393621969694715605863326996418687
        /// From: http://oeis.org/A002162/constant
        /// </remarks>
        public const decimal Ln2 = 0.6931471805599453094172321215m;

        // Fast access for 10^n
        internal static readonly decimal[] PowersOf10 = {1m, 10m, 100m, 1000m, 10000m, 100000m, 1000000m, 10000000m, 100000000m, 1000000000m, 10000000000m, 100000000000m, 1000000000000m, 10000000000000m, 100000000000000m, 1000000000000000m, 10000000000000000m, 100000000000000000m, 1000000000000000000m, 10000000000000000000m, 100000000000000000000m, 1000000000000000000000m, 10000000000000000000000m, 100000000000000000000000m, 1000000000000000000000000m, 10000000000000000000000000m, 100000000000000000000000000m, 1000000000000000000000000000m, 10000000000000000000000000000m,};

        /// <summary>
        /// Converts degrees to radians. (π radians = 180 degrees)
        /// </summary>
        /// <param name="degrees">The degrees to convert.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToRad(decimal degrees)
        {
            if (degrees % 360m == 0)
            {
                return (degrees / 360m) * TwoPi;
            }

            if (degrees % 270m == 0)
            {
                return (degrees / 270m) * (Pi + PiHalf);
            }

            if (degrees % 180m == 0)
            {
                return (degrees / 180m) * Pi;
            }

            if (degrees % 90m == 0)
            {
                return (degrees / 90m) * PiHalf;
            }

            if (degrees % 45m == 0)
            {
                return (degrees / 45m) * PiQuarter;
            }

            if (degrees % 15m == 0)
            {
                return (degrees / 15m) * PiTwelfth;
            }

            return degrees * Pi / 180m;
        }

        /// <summary>
        /// Converts radians to degrees. (π radians = 180 degrees)
        /// </summary>
        /// <param name="radians">The radians to convert.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDeg(decimal radians)
        {
            const decimal ratio = 180m / Pi;

            return radians * ratio;
        }

        /// <summary>
        /// Normalizes an angle in radians to the 0 to 2Pi interval.
        /// </summary>
        /// <param name="radians">Angle in radians.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal NormalizeAngle(decimal radians)
        {
            radians = Remainder(radians, TwoPi);
            if (radians < 0) radians += TwoPi;
            return radians;
        }

        /// <summary>
        /// Normalizes an angle in degrees to the 0 to 360 degree interval.
        /// </summary>
        /// <param name="degrees">Angle in degrees.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal NormalizeAngleDeg(decimal degrees)
        {
            degrees = degrees % 360m;
            if (degrees < 0) degrees += 360m;
            return degrees;
        }

        /// <summary>
        /// Returns the sine of the specified angle.
        /// </summary>
        /// <param name="x">An angle, measured in radians.</param>
        /// <remarks>
        /// Uses a Taylor series to calculate sine. See 
        /// http://en.wikipedia.org/wiki/Trigonometric_functions for details.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Sin(decimal x)
        {
            // Normalize to between -2Pi <= x <= 2Pi
            x = Remainder(x, TwoPi);

            if (x == 0 || x == Pi || x == TwoPi)
            {
                return 0;
            }

            if (x == PiHalf)
            {
                return 1;
            }

            if (x == Pi + PiHalf)
            {
                return -1;
            }

            var result = 0m;
            var doubleIteration = 0; // current iteration * 2
            var xSquared = x * x;
            var nextAdd = 0m;

            while (true)
            {
                if (doubleIteration == 0)
                {
                    nextAdd = x;
                }
                else
                {
                    // We multiply by -1 each time so that the sign of the component
                    // changes each time. The first item is positive and it
                    // alternates back and forth after that.
                    // Following is equivalent to: nextAdd *= -1 * x * x / ((2 * iteration) * (2 * iteration + 1));
                    nextAdd *= -1 * xSquared / (doubleIteration * doubleIteration + doubleIteration);
                }

                Debug.WriteLine("{0:000}:{1,33:+0.0000000000000000000000000000;-0.0000000000000000000000000000} ->{2,33:+0.0000000000000000000000000000;-0.0000000000000000000000000000}",
                    doubleIteration / 2, nextAdd, result + nextAdd);
                if (nextAdd == 0) break;

                result += nextAdd;

                doubleIteration += 2;
            }

            return result;
        }

        /// <summary>
        /// Returns the cosine of the specified angle.
        /// </summary>
        /// <param name="x">An angle, measured in radians.</param>
        /// <remarks>
        /// Uses a Taylor series to calculate sine. See 
        /// http://en.wikipedia.org/wiki/Trigonometric_functions for details.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Cos(decimal x)
        {
            // Normalize to between -2Pi <= x <= 2Pi
            x = Remainder(x, TwoPi);

            if (x == 0 || x == TwoPi)
            {
                return 1;
            }

            if (x == Pi)
            {
                return -1;
            }

            if (x == PiHalf || x == Pi + PiHalf)
            {
                return 0;
            }

            var result = 0m;
            var doubleIteration = 0; // current iteration * 2
            var xSquared = x * x;
            var nextAdd = 0m;

            while (true)
            {
                if (doubleIteration == 0)
                {
                    nextAdd = 1;
                }
                else
                {
                    // We multiply by -1 each time so that the sign of the component
                    // changes each time. The first item is positive and it
                    // alternates back and forth after that.
                    // Following is equivalent to: nextAdd *= -1 * x * x / ((2 * iteration - 1) * (2 * iteration));
                    nextAdd *= -1 * xSquared / (doubleIteration * doubleIteration - doubleIteration);
                }

                if (nextAdd == 0) break;

                result += nextAdd;

                doubleIteration += 2;
            }

            return result;
        }

        /// <summary>
        /// Returns the tangent of the specified angle.
        /// </summary>
        /// <param name="radians">An angle, measured in radians.</param>
        /// <remarks>
        /// Uses a Taylor series to calculate sine. See 
        /// http://en.wikipedia.org/wiki/Trigonometric_functions for details.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal Tan(decimal radians)
        {
            try
            {
                return Sin(radians) / Cos(radians);
            }
            catch (DivideByZeroException)
            {
                throw new Exception("Tangent is undefined at this angle!");
            }
        }

        /// <summary>
        /// Returns the angle whose sine is the specified number.
        /// </summary>
        /// <param name="z">A number representing a sine, where -1 ≤d≤ 1.</param>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Inverse_trigonometric_function
        /// and http://mathworld.wolfram.com/InverseSine.html
        /// I originally used the Taylor series for ASin, but it was extremely slow
        /// around -1 and 1 (millions of iterations) and still ends up being less
        /// accurate than deriving from the ATan function.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ASin(decimal z)
        {
            if (z < -1 || z > 1)
            {
                throw new ArgumentOutOfRangeException("z", "Argument must be in the range -1 to 1 inclusive.");
            }

            // Special cases
            if (z == -1) return -PiHalf;
            if (z == 0) return 0;
            if (z == 1) return PiHalf;

            return 2m * ATan(z / (1 + Sqrt(1 - z * z)));
        }

        /// <summary>
        /// Returns the angle whose cosine is the specified number.
        /// </summary>
        /// <param name="z">A number representing a cosine, where -1 ≤d≤ 1.</param>
        /// <remarks>
        /// See http://en.wikipedia.org/wiki/Inverse_trigonometric_function
        /// and http://mathworld.wolfram.com/InverseCosine.html
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ACos(decimal z)
        {
            if (z < -1 || z > 1)
            {
                throw new ArgumentOutOfRangeException("z", "Argument must be in the range -1 to 1 inclusive.");
            }

            // Special cases
            if (z == -1) return Pi;
            if (z == 0) return PiHalf;
            if (z == 1) return 0;

            return 2m * ATan(Sqrt(1 - z * z) / (1 + z));
        }

        /// <summary>
        /// Returns the angle whose tangent is the quotient of two specified numbers.
        /// </summary>
        /// <param name="x">A number representing a tangent.</param>
        /// <remarks>
        /// See http://mathworld.wolfram.com/InverseTangent.html for faster converging 
        /// series from Euler that was used here.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ATan(decimal x)
        {
            // Special cases
            if (x == -1) return -PiQuarter;
            if (x == 0) return 0;
            if (x == 1) return PiQuarter;
            if (x < -1)
            {
                // Force down to -1 to 1 interval for faster convergence
                return -PiHalf - ATan(1 / x);
            }

            if (x > 1)
            {
                // Force down to -1 to 1 interval for faster convergence
                return PiHalf - ATan(1 / x);
            }

            var result = 0m;
            var doubleIteration = 0; // current iteration * 2
            var y = (x * x) / (1 + x * x);
            var nextAdd = 0m;

            while (true)
            {
                if (doubleIteration == 0)
                {
                    nextAdd = x / (1 + x * x); // is = y / x  but this is better for very small numbers where y = 9
                }
                else
                {
                    // We multiply by -1 each time so that the sign of the component
                    // changes each time. The first item is positive and it
                    // alternates back and forth after that.
                    // Following is equivalent to: nextAdd *= y * (iteration * 2) / (iteration * 2 + 1);
                    nextAdd *= y * doubleIteration / (doubleIteration + 1);
                }

                if (nextAdd == 0) break;

                result += nextAdd;

                doubleIteration += 2;
            }

            return result;
        }

        /// <summary>
        /// Returns the angle whose tangent is the quotient of two specified numbers.
        /// </summary>
        /// <param name="y">The y coordinate of a point.</param>
        /// <param name="x">The x coordinate of a point.</param>
        /// <returns>
        /// An angle, θ, measured in radians, such that -π≤θ≤π, and tan(θ) = y / x,
        /// where (x, y) is a point in the Cartesian plane. Observe the following: 
        /// For (x, y) in quadrant 1, 0 &lt; θ &lt; π/2.
        /// For (x, y) in quadrant 2, π/2 &lt; θ ≤ π.
        /// For (x, y) in quadrant 3, -π &lt; θ &lt; -π/2.
        /// For (x, y) in quadrant 4, -π/2 &lt; θ &lt; 0.
        /// </returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ATan2(decimal y, decimal x)
        {
            if (x == 0 && y == 0)
            {
                return 0; // X0, Y0
            }

            if (x == 0)
            {
                return y > 0
                    ? PiHalf // X0, Y+
                    : -PiHalf; // X0, Y-
            }

            if (y == 0)
            {
                return x > 0
                    ? 0 // X+, Y0
                    : Pi; // X-, Y0
            }

            var aTan = ATan(y / x);

            if (x > 0) return aTan; // Q1&4: X+, Y+-

            return y > 0
                ? aTan + Pi //   Q2: X-, Y+
                : aTan - Pi; //   Q3: X-, Y-
        }
    }
}
