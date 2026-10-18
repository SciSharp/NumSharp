using System;
using System.Globalization;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Printing;

namespace NumSharp.Tests.Backends.Printing
{
    /// <summary>
    ///     <see cref="FloatDigits"/> — the shortest round-trip digits (Ryu) behind every float → text conversion, and the
    ///     exact <c>%f</c> / <c>%e</c> roundings of <c>np.strings.mod</c>'s fast path — against references computed here in
    ///     exact <see cref="BigInteger"/> arithmetic, never against .NET's own float formatting (whose shortest form is wrong
    ///     for some powers of two, which is what <see cref="FloatDigits"/> replaced).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The shortest digits are checked by their definition: they read back as the same float, no decimal with fewer
    ///     significant digits does (both neighbours on the shorter grid are tried, because the rounding interval of a
    ///     power of two is lopsided), and among the candidates of the same length they are the one closest to the exact
    ///     value. float16 is exhaustive; float32 and float64 take every power of two plus seeded random bit patterns.
    ///     </para>
    ///     <para>Deterministic (fixed seeds), no process-wide state: runs in parallel with the suite.</para>
    /// </remarks>
    [TestClass]
    public unsafe class FloatDigitsTests
    {
        // =============================================================================================================
        // Exact references
        // =============================================================================================================

        /// <summary>Powers of two up to 2^1100 (a double's exponent range plus margin), built once.</summary>
        private static readonly BigInteger[] Pow2Big = Enumerable.Range(0, 1101).Select(k => BigInteger.Pow(2, k)).ToArray();

        /// <summary>Powers of ten up to 10^400 (a double's decimal range plus a scale's margin), built once.</summary>
        private static readonly BigInteger[] Pow10Big = Enumerable.Range(0, 401).Select(k => BigInteger.Pow(10, k)).ToArray();

        /// <summary>2^k from the cache (the references run thousands of exact divisions; raising the powers again each
        ///     time would be the test's whole cost).</summary>
        /// <param name="k">The power (0–1100).</param>
        /// <returns>2^k.</returns>
        private static BigInteger P2(int k) => Pow2Big[k];

        /// <summary>10^k from the cache.</summary>
        /// <param name="k">The power (0–400).</param>
        /// <returns>10^k.</returns>
        private static BigInteger P10(int k) => Pow10Big[k];

        /// <summary>A finite double's exact value as <c>m × 2^e</c> (m = 0 for zero).</summary>
        /// <param name="value">The value (the sign is ignored).</param>
        /// <param name="m">Receives the integer significand.</param>
        /// <param name="e">Receives the binary exponent.</param>
        private static void Exact(double value, out BigInteger m, out int e)
        {
            long bits = BitConverter.DoubleToInt64Bits(value) & long.MaxValue;
            int biased = (int)(bits >> 52);
            long frac = bits & ((1L << 52) - 1);
            m = biased == 0 ? frac : frac | (1L << 52);
            e = (biased == 0 ? 1 : biased) - 1075;
        }

        /// <summary><c>round_half_even(|value| × 10^s)</c> of the exact value, for any decimal scale.</summary>
        /// <param name="value">The value.</param>
        /// <param name="s">The scale (negative divides).</param>
        /// <returns>The rounded integer.</returns>
        private static BigInteger RoundScaled(double value, int s)
        {
            Exact(value, out BigInteger m, out int e);
            BigInteger n = m * P2(System.Math.Max(e, 0)) * P10(System.Math.Max(s, 0));
            BigInteger d = P2(System.Math.Max(-e, 0)) * P10(System.Math.Max(-s, 0));
            BigInteger q = BigInteger.DivRem(n, d, out BigInteger r);
            BigInteger twice = 2 * r;
            if (twice > d || (twice == d && !q.IsEven))
                q += 1;
            return q;
        }

        /// <summary><c>floor(log10 |value|)</c> of a nonzero finite value, exactly.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The decimal exponent of its first significant digit.</returns>
        private static int FloorLog10(double value)
        {
            Exact(value, out BigInteger m, out int e);
            int guess = (int)System.Math.Floor(System.Math.Log10(System.Math.Abs(value)));
            // Compare m·2^e with 10^k exactly around the estimate (Log10 can be off by one near powers of ten).
            for (int k = guess + 1; ; k--)
            {
                BigInteger lhs = m * P2(System.Math.Max(e, 0)) * P10(System.Math.Max(-k, 0));
                BigInteger rhs = P10(System.Math.Max(k, 0)) * P2(System.Math.Max(-e, 0));
                if (lhs >= rhs)
                    return k;
            }
        }

        /// <summary>
        ///     Checks the shortest digits of a finite nonzero value by their definition (see the class remarks).
        /// </summary>
        /// <param name="value">The value, as a double (exact for float32 / float16 values).</param>
        /// <param name="digits">The digits under test.</param>
        /// <param name="exponent">The decimal exponent of their last digit.</param>
        /// <param name="roundTrips">Reads a decimal back in the value's own format and says whether it is the value.</param>
        /// <param name="label">The case's name.</param>
        private static void CheckShortest(double value, ulong digits, int exponent, Func<string, bool> roundTrips, string label)
        {
            digits.Should().NotBe(0UL, label);
            (digits % 10).Should().NotBe(0UL, $"{label}: trailing zeros are stripped");
            roundTrips($"{digits}E{exponent}").Should().BeTrue($"{label}: {digits}E{exponent} reads back");
            int length = FloatDigits.DecimalLength(digits);
            int top = FloorLog10(value);
            // No shorter decimal reads back: both neighbours on each shorter grid.
            for (int k = length - 1; k >= 1; k--)
            {
                int s = k - 1 - top;
                BigInteger down = RoundTowardZeroScaled(value, s);
                foreach (var candidate in new[] { down, down + 1 })
                    if (candidate > 0)
                        roundTrips($"{candidate}E{-s}").Should().BeFalse($"{label}: {candidate}E{-s} is shorter and reads back");
            }
            // Among the same-length candidates that read back, ours is the closest to the exact value (ties: even).
            int sl = length - 1 - top;
            BigInteger lo = RoundTowardZeroScaled(value, sl);
            // (A shortest form that carried to the next power of ten, 1e23 for 9.999999999999999e22, has no same-length
            // neighbours to compare with.)
            if (exponent == -sl)
            {
                BigInteger nearest = RoundScaled(value, sl);
                if (nearest != (BigInteger)digits && roundTrips($"{nearest}E{-sl}"))
                    throw new AssertFailedException($"{label}: {digits}E{exponent} is not the nearest shortest ({nearest}E{-sl})");
                ((BigInteger)digits == lo || (BigInteger)digits == lo + 1).Should().BeTrue($"{label}: digits are a neighbour");
            }
        }

        /// <summary><c>floor(|value| × 10^s)</c> of the exact value.</summary>
        /// <param name="value">The value.</param>
        /// <param name="s">The scale.</param>
        /// <returns>The truncated integer.</returns>
        private static BigInteger RoundTowardZeroScaled(double value, int s)
        {
            Exact(value, out BigInteger m, out int e);
            BigInteger n = m * P2(System.Math.Max(e, 0)) * P10(System.Math.Max(s, 0));
            BigInteger d = P2(System.Math.Max(-e, 0)) * P10(System.Math.Max(-s, 0));
            return BigInteger.Divide(n, d);
        }

        /// <summary>Every power of two a double holds (subnormal ones included), the extremes, and seeded random finite bit
        ///     patterns.</summary>
        /// <param name="count">How many random patterns.</param>
        /// <returns>The values (positive).</returns>
        private static double[] DoubleSamples(int count)
        {
            var rng = new System.Random(4242);
            var values = Enumerable.Range(-1074, 1074 + 1024).Select(k => System.Math.ScaleB(1.0, k)).ToList();
            values.AddRange(new[] { double.Epsilon, double.MaxValue, 2.2250738585072014e-308, 2.225073858507201e-308, 0.1, 0.3,
                1.0 / 3, 5e-324, 9007199254740993.0, 1e23, 8.41e21, 5.0e-324 * 3 });
            while (values.Count < 2098 + 12 + count)
            {
                double d = BitConverter.Int64BitsToDouble(rng.NextInt64() & long.MaxValue);
                if (double.IsFinite(d) && d != 0)
                    values.Add(d);
            }
            return values.ToArray();
        }

        // =============================================================================================================
        // Shortest digits
        // =============================================================================================================

        /// <summary>
        ///     The two doubles whose shortest form the .NET 10 runtime gets wrong (10.0.1 writes 16 digits that read back as
        ///     the neighbouring double; .NET 8 and CPython write 17): <see cref="FloatDigits"/> gives CPython's 17 on every
        ///     runtime.
        /// </summary>
        /// <remarks>
        ///     Measured over every power of two of float64 / float32, every float16 and 2,000,000 random doubles: only 2^-25
        ///     and 2^-958 differ, and only on .NET 10. The test does not pin the runtime's behaviour (a servicing update
        ///     may fix it); it pins ours.
        /// </remarks>
        [TestMethod]
        public void Shortest_PowersOfTwoDotNet10GetsWrong_GiveSeventeenDigits()
        {
            FloatDigits.Shortest(System.Math.ScaleB(1.0, -25), out int e1).Should().Be(29802322387695312UL);
            e1.Should().Be(-24);
            FloatDigits.Shortest(System.Math.ScaleB(1.0, -958), out int e2).Should().Be(41045368012983762UL);
            e2.Should().Be(-305);
        }

        /// <summary>float64: every power of two, the extremes and 10,000 random bit patterns satisfy the definition.</summary>
        [TestMethod]
        public void Shortest_Double_IsTheDefinition()
        {
            foreach (double v in DoubleSamples(10000))
            {
                ulong digits = FloatDigits.Shortest(v, out int exponent);
                CheckShortest(v, digits, exponent,
                    s => double.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture) == v, $"double {v:R}");
                // The sign is ignored.
                FloatDigits.Shortest(-v, out int ne).Should().Be(digits);
                ne.Should().Be(exponent);
            }
        }

        /// <summary>float32: every power of two and 50,000 random bit patterns satisfy the definition.</summary>
        [TestMethod]
        public void Shortest_Single_IsTheDefinition()
        {
            var rng = new System.Random(77);
            var values = Enumerable.Range(-149, 149 + 128).Select(k => (float)System.Math.ScaleB(1.0, k)).ToList();
            values.AddRange(new[] { float.MaxValue, float.Epsilon, 1.17549435e-38f, 0.1f, 16777217f, 3.4e38f });
            while (values.Count < 277 + 6 + 50000)
            {
                float f = BitConverter.Int32BitsToSingle(rng.Next() & int.MaxValue);
                if (float.IsFinite(f) && f != 0)
                    values.Add(f);
            }
            foreach (float f in values)
            {
                uint digits = FloatDigits.Shortest(f, out int exponent);
                CheckShortest(f, digits, exponent,
                    s => float.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture) == f, $"float {f:R}");
            }
        }

        /// <summary>float16: every finite nonzero value satisfies the definition.</summary>
        [TestMethod]
        public void Shortest_Half_IsTheDefinition_Exhaustive()
        {
            for (int bits = 1; bits < 0x7C00; bits++)
            {
                Half h = BitConverter.UInt16BitsToHalf((ushort)bits);
                uint digits = FloatDigits.Shortest(h, out int exponent);
                CheckShortest((double)h, digits, exponent,
                    s => Half.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture) == h, $"half 0x{bits:X4}");
                FloatDigits.Shortest(-h, out int ne).Should().Be(digits);
                ne.Should().Be(exponent);
            }
        }

        // =============================================================================================================
        // %f / %e roundings
        // =============================================================================================================

        /// <summary>
        ///     <see cref="FloatDigits.TryFixed"/> is <c>round_half_even(|x| × 10^p)</c> of the exact value whenever it
        ///     answers, and answers for every value whose scaled digits fit a ulong; ties round to even (0.125 → 12,
        ///     0.375 → 38, 2.5 → 2, 0.5 → 0), and a precision past 19 is refused.
        /// </summary>
        [TestMethod]
        public void TryFixed_IsTheExactHalfEvenRounding()
        {
            (double V, int P, ulong Digits)[] pins =
            {
                (0.125, 2, 12), (0.375, 2, 38), (2.5, 0, 2), (0.5, 0, 0), (1.5, 0, 2), (0.0, 6, 0), (1e300, 2, 0),
                (9.9996, 3, 10000), (0.1, 19, 1000000000000000056), (18446744073709551615.0, 0, 0),
            };
            foreach (var (v, p, want) in pins)
            {
                bool ok = FloatDigits.TryFixed(v, p, out ulong got);
                if (v >= 1e19)
                    ok.Should().BeFalse($"{v} at {p} does not fit");
                else
                {
                    ok.Should().BeTrue($"{v:R} at {p}");
                    got.Should().Be(want, $"{v:R} at {p}");
                }
            }
            FloatDigits.TryFixed(1.0, 20, out _).Should().BeFalse("precision past 19");
            FloatDigits.TryFixed(1.0, -1, out _).Should().BeFalse("negative precision");

            var rng = new System.Random(911);
            int answered = 0;
            for (int i = 0; i < 40000; i++)
            {
                double v = rng.NextDouble() * System.Math.Pow(10, rng.Next(-22, 20));
                int p = rng.Next(0, 20);
                BigInteger want = RoundScaled(v, p);
                bool fits = want <= ulong.MaxValue;
                bool ok = FloatDigits.TryFixed(v, p, out ulong got);
                ok.Should().Be(fits, $"{v:R} at {p}: answers exactly when the digits fit");
                if (ok)
                {
                    ((BigInteger)got).Should().Be(want, $"{v:R} at {p}");
                    answered++;
                }
            }
            answered.Should().BeGreaterThan(20000);
        }

        /// <summary>
        ///     <see cref="FloatDigits.TryScientific"/> gives <c>p + 1</c> significant digits of the exact value rounded half
        ///     to even and the exponent of the first, moving the exponent on a carry (9.9996 at 3 → 1000, exponent 1);
        ///     zero gives 0, 0; it answers for every value from 0.1 to 1e15 at every precision, and where it refuses the
        ///     scale really is out of its range.
        /// </summary>
        [TestMethod]
        public void TryScientific_IsTheExactHalfEvenRounding()
        {
            FloatDigits.TryScientific(9.9996, 3, out ulong c, out int ce).Should().BeTrue();
            c.Should().Be(1000UL);
            ce.Should().Be(1);
            FloatDigits.TryScientific(0.0, 5, out ulong z, out int ze).Should().BeTrue();
            z.Should().Be(0UL);
            ze.Should().Be(0);
            FloatDigits.TryScientific(1e15, 3, out ulong big, out int be).Should().BeTrue();
            big.Should().Be(1000UL);
            be.Should().Be(15);
            // At or past 2^64 the integer part does not fit a ulong: refused, the caller's general route formats it.
            FloatDigits.TryScientific(1e100, 3, out _, out _).Should().BeFalse("1e100's integer part does not fit a ulong");
            FloatDigits.TryScientific(1.0, 18, out _, out _).Should().BeFalse("precision past 17");

            var rng = new System.Random(1213);
            int answered = 0, total = 0;
            for (int i = 0; i < 40000; i++)
            {
                double v = i % 2 == 0
                    ? rng.NextDouble() * System.Math.Pow(10, rng.Next(-30, 30))
                    : BitConverter.Int64BitsToDouble(rng.NextInt64() & long.MaxValue);
                if (!double.IsFinite(v) || v == 0)
                    continue;
                int p = rng.Next(0, 18);
                total++;
                int top = FloorLog10(v);
                BigInteger q = RoundScaled(v, p - top);
                int exponent = top;
                if (q == BigInteger.Pow(10, p + 1))
                {
                    q = BigInteger.Pow(10, p);
                    exponent++;
                }
                bool inRange = v >= 0.1 && v <= 1e15;
                bool ok = FloatDigits.TryScientific(v, p, out ulong got, out int gotExponent);
                if (inRange)
                    ok.Should().BeTrue($"{v:R} at {p} is inside the guaranteed range");
                if (ok)
                {
                    ((BigInteger)got).Should().Be(q, $"{v:R} at {p}");
                    gotExponent.Should().Be(exponent, $"{v:R} at {p}");
                    answered++;
                }
                else
                {
                    // A refusal only where the scale leaves 19 digits or the value reaches 2^64.
                    // The scale is p - e0 with e0 = top or top - 1 (one less again on the retry): a refusal is legitimate
                    // only where that can leave the 19 digits a ulong holds, or the integer part reaches 2^64.
                    (System.Math.Abs(p - top) >= 19 || v >= 18446744073709551616.0)
                        .Should().BeTrue($"{v:R} at {p} was refused inside the scale's range");
                }
            }
            // Half the samples are random bit patterns, mostly far outside the range a ulong scale reaches.
            answered.Should().BeGreaterThan(total / 4);
        }

        // =============================================================================================================
        // Digit writers
        // =============================================================================================================

        /// <summary><see cref="FloatDigits.DecimalLength"/> and both <see cref="FloatDigits.WriteDecimalBackwards(ulong, byte*)"/>
        ///     forms against <see cref="ulong.ToString()"/> at every power of ten and its neighbours, the extremes and random
        ///     values; the counted form zero-fills on the left and writes nothing before its first slot.</summary>
        [TestMethod]
        public void DigitWriters_MatchToString()
        {
            var values = new System.Collections.Generic.List<ulong> { 0, 1, 9, 10, 99, 100, ulong.MaxValue, ulong.MaxValue - 1 };
            for (int k = 1; k < 20; k++)
            {
                ulong p = FloatDigits.Pow10[k];
                values.AddRange(new[] { p - 1, p, p + 1 });
            }
            var rng = new System.Random(5);
            for (int i = 0; i < 20000; i++)
                values.Add((ulong)rng.NextInt64() >> rng.Next(0, 64));
            byte* buffer = stackalloc byte[48];
            foreach (ulong v in values)
            {
                string want = v.ToString(CultureInfo.InvariantCulture);
                FloatDigits.DecimalLength(v).Should().Be(want.Length, $"{v}");
                new Span<byte>(buffer, 48).Fill(0xAB);
                int n = FloatDigits.WriteDecimalBackwards(v, buffer + 40);
                n.Should().Be(want.Length);
                System.Text.Encoding.ASCII.GetString(buffer + 40 - n, n).Should().Be(want);
                buffer[40 - n - 1].Should().Be(0xAB, "nothing before the first digit");
                buffer[40].Should().Be(0xAB, "nothing at the end slot");
                new Span<byte>(buffer, 48).Fill(0xAB);
                FloatDigits.WriteDecimalBackwards(v, buffer + 40, 22);
                System.Text.Encoding.ASCII.GetString(buffer + 18, 22).Should().Be(want.PadLeft(22, '0'));
                buffer[17].Should().Be(0xAB, "the counted form writes exactly its count");
            }
        }
    }
}
