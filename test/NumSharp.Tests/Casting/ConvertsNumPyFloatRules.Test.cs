using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Pins the two scalar float conversions where NumPy and the .NET BCL disagree — a NaN crossing float16, and
    ///     uint64 → float32 / float64 — at every level they are implemented: the <see cref="Converts"/> table, the
    ///     scalar converter (<see cref="NDIterCasting.ConvertValue"/>), astype's cast kernels (contiguous and strided),
    ///     the ufuncs that widen a uint64 operand to float64, and the kernels that widen a float16 operand for a
    ///     NaN-propagating op (maximum / minimum over mixed float16 and float32 / float64, plain and in np.evaluate).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         float16: NumPy converts bit by bit (<c>npy_halfbits_to_floatbits</c>, <c>npy_halfbits_to_doublebits</c>,
    ///         <c>npy_floatbits_to_halfbits</c>, <c>npy_doublebits_to_halfbits</c>), so a SIGNALLING NaN stays signalling
    ///         and keeps its payload's top bits, and a NaN whose kept payload bits are all zero becomes 0x7C01 (still a
    ///         NaN, still signalling). The BCL casts set the quiet bit (and give 0x7E00 for the latter). NumSharp used the
    ///         BCL casts in the Converts table, in the scalar converter and in the IL cast emitter — so a signalling NaN
    ///         was quieted on some routes and not on others. The finite values were already right; float64 → float16
    ///         must round ONCE (never through float32), which is pinned here too.
    ///     </para>
    ///     <para>
    ///         uint64: NumPy's C casts round once to nearest-even. .NET 8's <c>(float)ulong</c> goes through float64 and
    ///         rounds twice, and its <c>(double)ulong</c> of a value ≥ 2⁶³ converts as signed and adds 2⁶⁴, rounding twice
    ///         — so on .NET 8 NumSharp's astype, scalar conversions and ufunc operand widenings of a uint64 were an ulp off
    ///         for such values. (.NET 10's own conversions are correct, so these tests fail on .NET 8 only without the
    ///         fix.)
    ///     </para>
    ///     <para>
    ///         The references are independent of the code under test: finite float16 values are rebuilt arithmetically
    ///         from their fields, NaNs from NumPy's published bit formulas, and uint64 values are rounded by exact integer
    ///         arithmetic (<see cref="RoundUInt64"/>). NumPy 2.4.2 anchors (probed) are spelled out where a formula alone
    ///         would not convince. A failure message is formatted only when a check fails, so the tens of thousands of
    ///         passing checks cost no string work.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class ConvertsNumPyFloatRulesTests
    {
        /// <summary>
        ///     Every one of the 65 536 float16 bit patterns converts to float32, float64 and complex128 exactly as NumPy
        ///     converts it — finite values and infinities exactly, NaNs with their sign and payload and their quiet bit
        ///     unchanged — through the Converts table, the scalar converter, and astype (contiguous and strided).
        /// </summary>
        /// <exception cref="AssertFailedException">A route produced other bits than NumPy for some float16.</exception>
        /// <remarks>
        ///     Probed (numpy 2.4.2): float16 [0x7d01, 0x7fff, 0xfc01] → float32 [0x7fa02000, 0x7fffe000, 0xff802000],
        ///     float64 [0x7ff4040000000000, 0x7ffffc0000000000, 0xfff0040000000000] — the formula's outputs.
        /// </remarks>
        [TestMethod]
        public unsafe void Float16ToFloat32AndFloat64_EveryBitPattern_MatchesNumPy()
        {
            var failures = new FailureLog();
            var halves = new Half[65536];
            var wantSingles = new uint[65536];
            var wantDoubles = new ulong[65536];
            for (int b = 0; b < 65536; b++)
            {
                ushort bits = (ushort)b;
                Half h = BitConverter.UInt16BitsToHalf(bits);
                halves[b] = h;
                uint wantSingle = wantSingles[b] = ExpectedHalfToSingleBits(bits);
                ulong wantDouble = wantDoubles[b] = ExpectedHalfToDoubleBits(bits);

                Expect(failures, "Converts.ToSingle", bits, wantSingle, BitConverter.SingleToUInt32Bits(Converts.ToSingle(h)));
                Expect(failures, "Converts.ToDouble", bits, wantDouble, BitConverter.DoubleToUInt64Bits(Converts.ToDouble(h)));
                Complex z = Converts.ToComplex(h);
                Expect(failures, "Converts.ToComplex(...).Real", bits, wantDouble, BitConverter.DoubleToUInt64Bits(z.Real));
                Expect(failures, "Converts.ToComplex(...).Imaginary", bits, 0UL, BitConverter.DoubleToUInt64Bits(z.Imaginary));

                // The scalar converter astype's copy core (and ToArray's one-element and scalar-odometer routes) applies.
                float f;
                double d;
                Complex c;
                NDIterCasting.ConvertValue(&h, &f, NPTypeCode.Half, NPTypeCode.Single);
                NDIterCasting.ConvertValue(&h, &d, NPTypeCode.Half, NPTypeCode.Double);
                NDIterCasting.ConvertValue(&h, &c, NPTypeCode.Half, NPTypeCode.Complex);
                Expect(failures, "ConvertValue f16→f32", bits, wantSingle, BitConverter.SingleToUInt32Bits(f));
                Expect(failures, "ConvertValue f16→f64", bits, wantDouble, BitConverter.DoubleToUInt64Bits(d));
                Expect(failures, "ConvertValue f16→c128", bits, wantDouble, BitConverter.DoubleToUInt64Bits(c.Real));
            }

            // astype over all 65 536 values at once — the cast kernels' vector bodies and tails — contiguous and strided.
            using var all = np.array(halves);
            using var doubled = np.repeat(all, 2);
            using var strided = doubled["::2"];
            foreach (var (name, source) in new[] { ("contiguous", all), ("strided", strided) })
            {
                using var singles = source.astype(NPTypeCode.Single);
                using var doubles = source.astype(NPTypeCode.Double);
                using var complexes = source.astype(NPTypeCode.Complex);
                float[] fs = singles.ToArray<float>();
                double[] ds = doubles.ToArray<double>();
                Complex[] cs = complexes.ToArray<Complex>();
                string toSingle = $"astype {name} f16→f32", toDouble = $"astype {name} f16→f64", toComplex = $"astype {name} f16→c128";
                for (int b = 0; b < 65536; b++)
                {
                    Expect(failures, toSingle, (ulong)b, wantSingles[b], BitConverter.SingleToUInt32Bits(fs[b]));
                    Expect(failures, toDouble, (ulong)b, wantDoubles[b], BitConverter.DoubleToUInt64Bits(ds[b]));
                    Expect(failures, toComplex, (ulong)b, wantDoubles[b], BitConverter.DoubleToUInt64Bits(cs[b].Real));
                }
            }

            // The NumPy anchors, spelled out.
            ExpectedHalfToSingleBits(0x7D01).Should().Be(0x7FA02000u);
            ExpectedHalfToSingleBits(0x7FFF).Should().Be(0x7FFFE000u);
            ExpectedHalfToSingleBits(0xFC01).Should().Be(0xFF802000u);
            ExpectedHalfToDoubleBits(0x7D01).Should().Be(0x7FF4040000000000UL);
            ExpectedHalfToDoubleBits(0x7FFF).Should().Be(0x7FFFFC0000000000UL);
            ExpectedHalfToDoubleBits(0xFC01).Should().Be(0xFFF0040000000000UL);
            AssertNone(failures, "float16 → float32 / float64 / complex128");
        }

        /// <summary>
        ///     float32 and float64 NaNs convert to float16 as NumPy converts them: sign kept, the payload's top 10 bits kept
        ///     (so a signalling NaN stays signalling), and 0x7C01 — not the canonical 0x7E00 — when those 10 bits are all
        ///     zero; through the Converts table (float, double and complex), the scalar converter and astype (contiguous and
        ///     strided).
        /// </summary>
        /// <exception cref="AssertFailedException">A route produced other bits than NumPy for some NaN.</exception>
        /// <remarks>
        ///     <para>
        ///         The rule reads only the sign and the payload's top 10 bits, and whether those are zero: so the inputs are
        ///         EVERY value of those 10 bits, both signs, each under several low-bit patterns (none — the bits that
        ///         survive are the whole payload —, the lowest bit, the highest dropped bit, all dropped bits, a random
        ///         pattern), plus random payloads — rather than all 2²⁴ float32 NaNs, which would only repeat each case.
        ///     </para>
        ///     <para>
        ///         Probed (numpy 2.4.2): float32 [0x7fa00001, 0x7fc0beef, 0x7f800001, 0x7f802000, 0xff800001, 0x7fbfffff]
        ///         → float16 [0x7d00, 0x7e05, 0x7c01, 0x7c01, 0xfc01, 0x7dff]; float64 [0x7ff4000000000001,
        ///         0x7ff8deadbeef0001, 0x7ff0000000000001, 0x7ff0040000000000, 0xfff0000000000001, 0x7ff7ffffffffffff] →
        ///         float16 [0x7d00, 0x7e37, 0x7c01, 0x7c01, 0xfc01, 0x7dff]; complex128 takes its real part's.
        ///     </para>
        /// </remarks>
        [TestMethod]
        public unsafe void Float32AndFloat64NaNsToFloat16_KeepTheSignalAndTheTopPayloadBits()
        {
            var failures = new FailureLog();
            var random = new System.Random(20260925);

            // float32: every top-10 value × low-13-bit patterns × both signs, through the Converts table and the scalar
            // converter; kept for astype below.
            var singles = new List<float>();
            var singleWant = new List<ushort>();
            for (uint top = 0; top < 1024; top++)
                foreach (uint low in new[] { 0u, 1u, 1u << 12, (1u << 13) - 1, (uint)random.Next(1 << 13) })
                    for (uint sign = 0; sign <= 1; sign++)
                    {
                        uint payload = (top << 13) | low;
                        if (payload == 0)
                            continue; // an infinity, not a NaN
                        uint bits = (sign << 31) | 0x7F80_0000u | payload;
                        float value = BitConverter.UInt32BitsToSingle(bits);
                        ushort want = ExpectedNaNToHalfBits(sign, top);
                        singles.Add(value);
                        singleWant.Add(want);
                        Expect(failures, "Converts.ToHalf(float)", bits, want, BitConverter.HalfToUInt16Bits(Converts.ToHalf(value)));
                        Half h;
                        NDIterCasting.ConvertValue(&value, &h, NPTypeCode.Single, NPTypeCode.Half);
                        Expect(failures, "ConvertValue f32→f16", bits, want, BitConverter.HalfToUInt16Bits(h));
                    }

            // float64: every top-10 value × low-42-bit patterns, plus random payloads, both signs, through Converts (double
            // and complex) and the scalar converter (float64 and complex128 sources); kept for astype below.
            var payloads = new List<ulong>();
            for (ulong top = 0; top < 1024; top++)
                foreach (ulong low in new[] { 0UL, 1UL, 1UL << 41, (1UL << 42) - 1, (ulong)random.NextInt64(1L << 42) })
                    payloads.Add((top << 42) | low);
            for (int i = 0; i < 4096; i++)
                payloads.Add((ulong)random.NextInt64() & 0x000F_FFFF_FFFF_FFFFUL);
            var doubles = new List<double>();
            var doubleWant = new List<ushort>();
            foreach (ulong p in payloads)
            {
                if (p == 0)
                    continue; // an infinity, not a NaN
                for (ulong sign = 0; sign <= 1; sign++)
                {
                    ulong bits = (sign << 63) | 0x7FF0_0000_0000_0000UL | p;
                    double value = BitConverter.UInt64BitsToDouble(bits);
                    ushort want = ExpectedNaNToHalfBits((uint)sign, (uint)(p >> 42));
                    doubles.Add(value);
                    doubleWant.Add(want);
                    Expect(failures, "Converts.ToHalf(double)", bits, want, BitConverter.HalfToUInt16Bits(Converts.ToHalf(value)));
                    Expect(failures, "Converts.ToHalf(complex)", bits, want, BitConverter.HalfToUInt16Bits(Converts.ToHalf(new Complex(value, 3.5))));
                    Half fromDouble, fromComplex;
                    var complex = new Complex(value, -1);
                    NDIterCasting.ConvertValue(&value, &fromDouble, NPTypeCode.Double, NPTypeCode.Half);
                    NDIterCasting.ConvertValue(&complex, &fromComplex, NPTypeCode.Complex, NPTypeCode.Half);
                    Expect(failures, "ConvertValue f64→f16", bits, want, BitConverter.HalfToUInt16Bits(fromDouble));
                    Expect(failures, "ConvertValue c128→f16", bits, want, BitConverter.HalfToUInt16Bits(fromComplex));
                }
            }

            // astype over each set: contiguous and strided (the cast kernels' vector bodies and tails).
            using var singleSource = np.array(singles.ToArray());
            using var doubleSource = np.array(doubles.ToArray());
            foreach (var (label, source, want) in new[] { ("f32", singleSource, singleWant), ("f64", doubleSource, doubleWant) })
            {
                using var twice = np.repeat(source, 2);
                using var strided = twice["::2"];
                foreach (var (name, array) in new[] { ("contiguous", source), ("strided", strided) })
                {
                    using var halves = array.astype(NPTypeCode.Half);
                    Half[] got = halves.ToArray<Half>();
                    string route = $"astype {name} {label}→f16 (element index)";
                    for (int i = 0; i < got.Length; i++)
                        Expect(failures, route, (ulong)i, want[i], BitConverter.HalfToUInt16Bits(got[i]));
                }
            }
            AssertNone(failures, "NaN → float16");
        }

        /// <summary>
        ///     float64 → float16 rounds ONCE, straight from the double, as NumPy's <c>npy_doublebits_to_halfbits</c> does
        ///     — never through float32, whose own rounding can land on a float16 tie (or on 65 520, which overflows) and
        ///     then round the wrong way.
        /// </summary>
        /// <exception cref="AssertFailedException">A route rounded through float32.</exception>
        /// <remarks>
        ///     Probed (numpy 2.4.2), direct vs. through float32: 1 + 2⁻¹¹ + 2⁻⁴⁰ → 0x3c01 (via float32 0x3c00); its
        ///     negation → 0xbc01 (0xbc00); 5·2⁻²⁵ + 2⁻⁶⁰ → 0x0003 (0x0002); 2049 + 2⁻³⁰ → 0x6801 (0x6800);
        ///     65 519.99999 → 0x7bff (0x7c00, infinity).
        /// </remarks>
        [TestMethod]
        public unsafe void Float64ToFloat16_RoundsOnce_NeverThroughFloat32()
        {
            double[] values =
            {
                1 + System.Math.Pow(2, -11) + System.Math.Pow(2, -40), -(1 + System.Math.Pow(2, -11) + System.Math.Pow(2, -40)),
                5 * System.Math.Pow(2, -25) + System.Math.Pow(2, -60), 2049 + System.Math.Pow(2, -30), 65519.99999,
            };
            ushort[] numpy = { 0x3C01, 0xBC01, 0x0003, 0x6801, 0x7BFF };
            var failures = new FailureLog();
            for (int i = 0; i < values.Length; i++)
            {
                double v = values[i];
                ulong bits = BitConverter.DoubleToUInt64Bits(v);
                Expect(failures, "Converts.ToHalf(double)", bits, numpy[i], BitConverter.HalfToUInt16Bits(Converts.ToHalf(v)));
                Half h;
                NDIterCasting.ConvertValue(&v, &h, NPTypeCode.Double, NPTypeCode.Half);
                Expect(failures, "ConvertValue f64→f16", bits, numpy[i], BitConverter.HalfToUInt16Bits(h));
            }
            using var source = np.array(values);
            using var halves = source.astype(NPTypeCode.Half);
            Half[] got = halves.ToArray<Half>();
            for (int i = 0; i < values.Length; i++)
                Expect(failures, "astype f64→f16", BitConverter.DoubleToUInt64Bits(values[i]), numpy[i], BitConverter.HalfToUInt16Bits(got[i]));
            AssertNone(failures, "float64 → float16 single rounding");
        }

        /// <summary>
        ///     uint64 → float32, float64 and complex128 round ONCE to nearest-even, as NumPy's C casts do, on every runtime:
        ///     through the Converts table, the scalar converter, astype (contiguous and strided), and the ufuncs that widen
        ///     a uint64 operand to float64.
        /// </summary>
        /// <exception cref="AssertFailedException">A route rounded some value twice (on .NET 8 without the fix: many).</exception>
        /// <remarks>
        ///     <para>
        ///         Probed (numpy 2.4.2): <c>np.array([2**63+1025, 2**60+2**36+1], np.uint64)</c> → float64
        ///         [0x43e0000000000001, 0x43b0000010000000], float32 [0x5f000000, 0x5d800001]; and
        ///         <c>np.true_divide(u, 1.0)</c> / <c>np.add(u, 0.0)</c> / <c>u * 1.0</c> (result float64) give the same
        ///         float64 bits. .NET 8 gave 0x43e0000000000000 and 0x5d800000.
        ///     </para>
        ///     <para>
        ///         The reference rounds by exact integer arithmetic (<see cref="RoundUInt64"/>); the values mix every
        ///         magnitude with a bias toward the ones that separate one rounding from two (above 2⁵³, above 2⁶³, exact
        ///         ties of both widths).
        ///     </para>
        /// </remarks>
        [TestMethod]
        public unsafe void UInt64ToFloat32AndFloat64_RoundsOnceToNearestEven()
        {
            var values = new List<ulong>
            {
                0, 1, (1UL << 24) + 1, (1UL << 53) + 1, (1UL << 53) + (1UL << 29) + 1, (1UL << 54) + 3, (1UL << 60) + (1UL << 36) + 1,
                (1UL << 63) - 1, 1UL << 63, (1UL << 63) + 1, (1UL << 63) + 1024, (1UL << 63) + 1025, (1UL << 63) + 3072,
                (1UL << 63) + (1UL << 39) + 1, ulong.MaxValue - (1UL << 39), ulong.MaxValue - 1024, ulong.MaxValue,
            };
            var random = new System.Random(20260926);
            for (int i = 0; i < 20_000; i++)
            {
                ulong v = (ulong)random.NextInt64() | ((ulong)random.Next(2) << 63);
                // Shift some down to every magnitude, and plant exact float64 (bit 9) and float32 (bit 38) ties.
                v >>= random.Next(12);
                if (i % 7 == 0)
                    v = (v & ~((1UL << 10) - 1)) | (1UL << 9);
                if (i % 11 == 0)
                    v = (v & ~((1UL << 39) - 1)) | (1UL << 38);
                values.Add(v);
            }

            var failures = new FailureLog();
            ulong[] all = values.ToArray();
            var wantDoubles = new ulong[all.Length];
            var wantSingles = new uint[all.Length];
            for (int i = 0; i < all.Length; i++)
            {
                ulong v = all[i];
                ulong wantDouble = wantDoubles[i] = BitConverter.DoubleToUInt64Bits(ExactToDouble(v));
                uint wantSingle = wantSingles[i] = BitConverter.SingleToUInt32Bits(ExactToSingle(v));
                Expect(failures, "Converts.ToDouble(ulong)", v, wantDouble, BitConverter.DoubleToUInt64Bits(Converts.ToDouble(v)));
                Expect(failures, "Converts.ToSingle(ulong)", v, wantSingle, BitConverter.SingleToUInt32Bits(Converts.ToSingle(v)));
                Expect(failures, "Converts.ToComplex(ulong).Real", v, wantDouble, BitConverter.DoubleToUInt64Bits(Converts.ToComplex(v).Real));
                double d;
                float f;
                Complex c;
                NDIterCasting.ConvertValue(&v, &d, NPTypeCode.UInt64, NPTypeCode.Double);
                NDIterCasting.ConvertValue(&v, &f, NPTypeCode.UInt64, NPTypeCode.Single);
                NDIterCasting.ConvertValue(&v, &c, NPTypeCode.UInt64, NPTypeCode.Complex);
                Expect(failures, "ConvertValue u64→f64", v, wantDouble, BitConverter.DoubleToUInt64Bits(d));
                Expect(failures, "ConvertValue u64→f32", v, wantSingle, BitConverter.SingleToUInt32Bits(f));
                Expect(failures, "ConvertValue u64→c128", v, wantDouble, BitConverter.DoubleToUInt64Bits(c.Real));
            }

            // astype and the widening ufuncs over the whole set, contiguous and strided.
            using var source = np.array(all);
            using var twice = np.repeat(source, 2);
            using var strided = twice["::2"];
            foreach (var (name, array) in new[] { ("contiguous", source), ("strided", strided) })
            {
                using var asDouble = array.astype(NPTypeCode.Double);
                using var asSingle = array.astype(NPTypeCode.Single);
                using var asComplex = array.astype(NPTypeCode.Complex);
                using var divided = np.true_divide(array, 1.0);
                using var added = np.add(array, 0.0);
                using var multiplied = array * 1.0;
                double[] d = asDouble.ToArray<double>(), q = divided.ToArray<double>(), s = added.ToArray<double>(), m = multiplied.ToArray<double>();
                float[] f = asSingle.ToArray<float>();
                Complex[] c = asComplex.ToArray<Complex>();
                string[] routes = { $"astype {name} u64→f64", $"astype {name} u64→f32", $"astype {name} u64→c128", $"true_divide(u64 {name}, 1.0)", $"add(u64 {name}, 0.0)", $"u64 {name} * 1.0" };
                for (int i = 0; i < all.Length; i++)
                {
                    Expect(failures, routes[0], all[i], wantDoubles[i], BitConverter.DoubleToUInt64Bits(d[i]));
                    Expect(failures, routes[1], all[i], wantSingles[i], BitConverter.SingleToUInt32Bits(f[i]));
                    Expect(failures, routes[2], all[i], wantDoubles[i], BitConverter.DoubleToUInt64Bits(c[i].Real));
                    Expect(failures, routes[3], all[i], wantDoubles[i], BitConverter.DoubleToUInt64Bits(q[i]));
                    Expect(failures, routes[4], all[i], wantDoubles[i], BitConverter.DoubleToUInt64Bits(s[i]));
                    Expect(failures, routes[5], all[i], wantDoubles[i], BitConverter.DoubleToUInt64Bits(m[i]));
                }
            }

            // The NumPy anchors, through the reference.
            BitConverter.DoubleToUInt64Bits(ExactToDouble((1UL << 63) + 1025)).Should().Be(0x43E0000000000001UL);
            BitConverter.DoubleToUInt64Bits(ExactToDouble((1UL << 60) + (1UL << 36) + 1)).Should().Be(0x43B0000010000000UL);
            BitConverter.SingleToUInt32Bits(ExactToSingle((1UL << 63) + 1025)).Should().Be(0x5F000000u);
            BitConverter.SingleToUInt32Bits(ExactToSingle((1UL << 60) + (1UL << 36) + 1)).Should().Be(0x5D800001u);
            AssertNone(failures, "uint64 → float");
        }

        /// <summary>
        ///     np.maximum / np.minimum over a float16 operand mixed with a float32 or float64 one return the float16's NaN
        ///     — signalling or quiet — with NumPy's bits, plain, strided, either operand order, and through np.evaluate:
        ///     the kernel widens the float16 operand by NumPy's bit rule, and neither op quiets a NaN. add and multiply
        ///     quiet it (the arithmetic does, in NumPy too).
        /// </summary>
        /// <exception cref="AssertFailedException">A route returned other bits than NumPy.</exception>
        /// <remarks>
        ///     <para>
        ///         Probed (numpy 2.4.2), <c>h</c> = float16 [0x7d01, 0x7c01, 0xfd01, 0x3c00], <c>f</c> = float32 [1, 2, −3,
        ///         0.5]: <c>maximum(h, f)</c> and <c>maximum(f, h)</c> → [0x7fa02000, 0x7f802000, 0xffa02000, 0x3f800000];
        ///         <c>minimum(h, f)</c> → [0x7fa02000, 0x7f802000, 0xffa02000, 0x3f000000]; with <c>f</c> as float64,
        ///         <c>maximum</c> → [0x7ff4040000000000, 0x7ff0040000000000, 0xfff4040000000000, 0x3ff0000000000000];
        ///         <c>add(h, f)</c> → [0x7fe02000, 0x7fc02000, 0xffe02000, 0x3fc00000] and <c>multiply(h, f)</c> →
        ///         [0x7fe02000, 0x7fc02000, 0xffe02000, 0x3f000000]. NumPy keeps the signal over all 65 536 float16 values
        ///         too (its vectorized casts and maximum included), which the second half checks against the reference.
        ///     </para>
        ///     <para>
        ///         NumSharp quieted the NaN on these routes (maximum → 0x7fe02000, 0x7ffc040000000000): the kernels' IL
        ///         operand widening went through the BCL's Half → double cast. It now calls the NumPy-exact
        ///         <see cref="Converts"/> overloads, float16 → float32 directly.
        ///     </para>
        /// </remarks>
        [TestMethod]
        public void MixedFloat16Operands_MaximumAndMinimum_KeepTheNaNBitsLikeNumPy()
        {
            var failures = new FailureLog();

            // The probed anchors.
            using (var h = np.array(new ushort[] { 0x7D01, 0x7C01, 0xFD01, 0x3C00 }).view(np.float16))
            using (var f = np.array(new float[] { 1f, 2f, -3f, 0.5f }))
            using (var d = np.array(new double[] { 1.0, 2.0, -3.0, 0.5 }))
            {
                uint[] max32 = { 0x7FA02000, 0x7F802000, 0xFFA02000, 0x3F800000 };
                uint[] min32 = { 0x7FA02000, 0x7F802000, 0xFFA02000, 0x3F000000 };
                ulong[] max64 = { 0x7FF4040000000000, 0x7FF0040000000000, 0xFFF4040000000000, 0x3FF0000000000000 };
                uint[] add32 = { 0x7FE02000, 0x7FC02000, 0xFFE02000, 0x3FC00000 };
                uint[] mul32 = { 0x7FE02000, 0x7FC02000, 0xFFE02000, 0x3F000000 };
                ExpectSingles(failures, "maximum(h, f) anchor", np.maximum(h, f), i => max32[i]);
                ExpectSingles(failures, "maximum(f, h) anchor", np.maximum(f, h), i => max32[i]);
                ExpectSingles(failures, "minimum(h, f) anchor", np.minimum(h, f), i => min32[i]);
                ExpectDoubles(failures, "maximum(h, f64) anchor", np.maximum(h, d), i => max64[i]);
                ExpectSingles(failures, "add(h, f) anchor", np.add(h, f), i => add32[i]);
                ExpectSingles(failures, "multiply(h, f) anchor", np.multiply(h, f), i => mul32[i]);
            }

            // Every float16 value against finite partners that never tie with it (k + 0.25): a NaN comes back widened
            // by NumPy's rule, anything else is the plain maximum / minimum of the exactly widened values.
            var halves = new Half[65536];
            var partners = new float[65536];
            var partners64 = new double[65536];
            for (int b = 0; b < 65536; b++)
            {
                halves[b] = BitConverter.UInt16BitsToHalf((ushort)b);
                partners[b] = b % 97 + 0.25f;
                partners64[b] = b % 97 + 0.25;
            }

            ulong WantSingle(int b, bool max)
            {
                ushort bits = (ushort)b;
                if (Converts.IsHalfNaNBits(bits))
                    return ExpectedHalfToSingleBits(bits);
                float wide = BitConverter.UInt32BitsToSingle(ExpectedHalfToSingleBits(bits));
                return BitConverter.SingleToUInt32Bits(max ? MathF.Max(wide, partners[b]) : MathF.Min(wide, partners[b]));
            }

            ulong WantDouble(int b, bool max)
            {
                ushort bits = (ushort)b;
                if (Converts.IsHalfNaNBits(bits))
                    return ExpectedHalfToDoubleBits(bits);
                double wide = BitConverter.UInt64BitsToDouble(ExpectedHalfToDoubleBits(bits));
                return BitConverter.DoubleToUInt64Bits(max ? System.Math.Max(wide, partners64[b]) : System.Math.Min(wide, partners64[b]));
            }

            using var all = np.array(halves);
            using var f32 = np.array(partners);
            using var f64 = np.array(partners64);
            using var twice = np.repeat(all, 2);
            using var strided = twice["::2"];
            ExpectSingles(failures, "maximum(f16, f32)", np.maximum(all, f32), b => WantSingle(b, true));
            ExpectSingles(failures, "maximum(f32, f16)", np.maximum(f32, all), b => WantSingle(b, true));
            ExpectSingles(failures, "minimum(f16, f32)", np.minimum(all, f32), b => WantSingle(b, false));
            ExpectSingles(failures, "maximum(f16 strided, f32)", np.maximum(strided, f32), b => WantSingle(b, true));
            ExpectDoubles(failures, "maximum(f16, f64)", np.maximum(all, f64), b => WantDouble(b, true));
            ExpectDoubles(failures, "minimum(f64, f16)", np.minimum(f64, all), b => WantDouble(b, false));
            ExpectSingles(failures, "evaluate maximum(f16, f32)", np.evaluate(NDExpr.Maximum((NDExpr)all, (NDExpr)f32)), b => WantSingle(b, true));
            ExpectDoubles(failures, "evaluate minimum(f16, f64)", np.evaluate(NDExpr.Minimum((NDExpr)all, (NDExpr)f64)), b => WantDouble(b, false));
            AssertNone(failures, "maximum / minimum over mixed float16 operands");
        }

        // ============================================================================================== references

        /// <summary>
        ///     NumPy's float16 → float32 (<c>npy_halfbits_to_floatbits</c>), computed independently: a finite value or an
        ///     infinity is rebuilt arithmetically from its fields (exact in a double, then exactly representable as a
        ///     float32); a NaN keeps its sign and its 10 payload bits at the top of the 23-bit significand.
        /// </summary>
        /// <param name="h">A float16 encoding.</param>
        /// <returns>The float32 encoding NumPy produces.</returns>
        private static uint ExpectedHalfToSingleBits(ushort h)
        {
            uint sign = (uint)h >> 15, exponent = ((uint)h >> 10) & 0x1F, mantissa = h & 0x3FFu;
            if (exponent == 0x1F)
                return mantissa == 0 ? (sign << 31) | 0x7F80_0000u : (sign << 31) | 0x7F80_0000u | (mantissa << 13);
            return BitConverter.SingleToUInt32Bits((float)HalfValue(sign, exponent, mantissa));
        }

        /// <summary>
        ///     NumPy's float16 → float64 (<c>npy_halfbits_to_doublebits</c>), computed independently: finite values and
        ///     infinities arithmetically, a NaN with its sign and its 10 payload bits at the top of the 52-bit significand.
        /// </summary>
        /// <param name="h">A float16 encoding.</param>
        /// <returns>The float64 encoding NumPy produces.</returns>
        private static ulong ExpectedHalfToDoubleBits(ushort h)
        {
            ulong sign = (ulong)h >> 15, exponent = ((ulong)h >> 10) & 0x1F, mantissa = h & 0x3FFUL;
            if (exponent == 0x1F)
                return mantissa == 0 ? (sign << 63) | 0x7FF0_0000_0000_0000UL : (sign << 63) | 0x7FF0_0000_0000_0000UL | (mantissa << 42);
            return BitConverter.DoubleToUInt64Bits(HalfValue((uint)sign, (uint)exponent, (uint)mantissa));
        }

        /// <summary>The exact value of a finite float16 from its fields: ±mantissa·2⁻²⁴ (subnormal) or ±(1024 + mantissa)·2^(exponent − 25).</summary>
        /// <param name="sign">Sign bit.</param>
        /// <param name="exponent">Biased exponent, 0–30.</param>
        /// <param name="mantissa">10-bit significand field.</param>
        /// <returns>The value (exact in a double; −0.0 for a negative zero).</returns>
        private static double HalfValue(uint sign, uint exponent, uint mantissa)
        {
            double magnitude = exponent == 0 ? mantissa * System.Math.Pow(2, -24) : (1024 + mantissa) * System.Math.Pow(2, (int)exponent - 25);
            return sign == 1 ? -magnitude : magnitude;
        }

        /// <summary>
        ///     NumPy's float32 / float64 NaN → float16 (<c>npy_floatbits_to_halfbits</c> / <c>npy_doublebits_to_halfbits</c>):
        ///     the sign, an all-ones exponent, the payload's top 10 bits, and 1 when those are all zero (a NaN, never an
        ///     infinity).
        /// </summary>
        /// <param name="sign">The NaN's sign bit.</param>
        /// <param name="top10">The payload's top 10 bits.</param>
        /// <returns>The float16 encoding NumPy produces.</returns>
        private static ushort ExpectedNaNToHalfBits(uint sign, uint top10)
            => (ushort)((sign << 15) | 0x7C00u | (top10 == 0 ? 1u : top10));

        /// <summary>
        ///     Rounds a uint64 to a <paramref name="bits"/>-bit significand, to nearest with ties to even, by exact integer
        ///     arithmetic: the kept high bits, and the power of two they scale by.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="bits">Significand width (24 for float32, 53 for float64).</param>
        /// <returns>(kept, shift): the rounded value is kept · 2^shift (kept may reach 2^bits after a carry, still exact).</returns>
        private static (ulong Kept, int Shift) RoundUInt64(ulong value, int bits)
        {
            if (value == 0)
                return (0, 0);
            int width = 64 - BitOperations.LeadingZeroCount(value);
            int drop = width - bits;
            if (drop <= 0)
                return (value, 0);
            ulong kept = value >> drop;
            ulong remainder = value & ((1UL << drop) - 1);
            ulong half = 1UL << (drop - 1);
            if (remainder > half || (remainder == half && (kept & 1) == 1))
                kept++;
            return (kept, drop);
        }

        /// <summary>The float64 nearest <paramref name="value"/> (ties to even), from <see cref="RoundUInt64"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The correctly rounded double (kept ≤ 2⁵³ is exact, and scaling by a power of two is exact).</returns>
        private static double ExactToDouble(ulong value)
        {
            var (kept, shift) = RoundUInt64(value, 53);
            return System.Math.ScaleB((double)(long)kept, shift);
        }

        /// <summary>The float32 nearest <paramref name="value"/> (ties to even), from <see cref="RoundUInt64"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The correctly rounded float (kept ≤ 2²⁴ and its scale are exact in a double, and exactly a float).</returns>
        private static float ExactToSingle(ulong value)
        {
            var (kept, shift) = RoundUInt64(value, 24);
            return (float)System.Math.ScaleB((double)(long)kept, shift);
        }

        // ============================================================================================== bookkeeping

        /// <summary>
        ///     Records a failure when <paramref name="got"/> differs from <paramref name="want"/> (both as raw bits); the
        ///     message is formatted only then, so a passing check costs one compare.
        /// </summary>
        /// <param name="failures">Collected failures.</param>
        /// <param name="route">The conversion route (a constant string, or one hoisted out of the loop).</param>
        /// <param name="input">The input's bits (or index), for the message.</param>
        /// <param name="want">NumPy's bits.</param>
        /// <param name="got">NumSharp's bits.</param>
        private static void Expect(FailureLog failures, string route, ulong input, ulong want, ulong got)
        {
            if (want != got)
                failures.Record($"{route} of 0x{input:X} = 0x{got:X}, NumPy 0x{want:X}");
        }

        /// <summary>Checks every element of a float32 result against the expected bits, then disposes the result.</summary>
        /// <param name="failures">Collected failures.</param>
        /// <param name="route">The operation, for the message.</param>
        /// <param name="result">A float32 result (owned: disposed here).</param>
        /// <param name="want">The expected bits of element i.</param>
        private static void ExpectSingles(FailureLog failures, string route, NDArray result, Func<int, ulong> want)
        {
            using (result)
            {
                if (result.typecode != NPTypeCode.Single)
                {
                    failures.Record($"{route}: dtype {result.dtype.name}, NumPy float32");
                    return;
                }
                float[] got = result.ToArray<float>();
                string label = route + " (element index)";
                for (int i = 0; i < got.Length; i++)
                    Expect(failures, label, (ulong)i, want(i), BitConverter.SingleToUInt32Bits(got[i]));
            }
        }

        /// <summary>Checks every element of a float64 result against the expected bits, then disposes the result.</summary>
        /// <param name="failures">Collected failures.</param>
        /// <param name="route">The operation, for the message.</param>
        /// <param name="result">A float64 result (owned: disposed here).</param>
        /// <param name="want">The expected bits of element i.</param>
        private static void ExpectDoubles(FailureLog failures, string route, NDArray result, Func<int, ulong> want)
        {
            using (result)
            {
                if (result.typecode != NPTypeCode.Double)
                {
                    failures.Record($"{route}: dtype {result.dtype.name}, NumPy float64");
                    return;
                }
                double[] got = result.ToArray<double>();
                string label = route + " (element index)";
                for (int i = 0; i < got.Length; i++)
                    Expect(failures, label, (ulong)i, want(i), BitConverter.DoubleToUInt64Bits(got[i]));
            }
        }

        /// <summary>Fails with the first recorded failures (and their total count), if any.</summary>
        /// <param name="failures">Collected failures.</param>
        /// <param name="what">The test's subject.</param>
        /// <exception cref="AssertFailedException">Any failure was recorded.</exception>
        private static void AssertNone(FailureLog failures, string what)
        {
            if (failures.Count > 0)
                Assert.Fail($"{what}: {failures.Count} failures; first {failures.First.Count}:\n  " + string.Join("\n  ", failures.First));
        }

        /// <summary>
        ///     Counts failures but keeps only the first <see cref="Kept"/> messages: a regression here fails tens of
        ///     thousands of checks at once, and keeping every message would cost memory for nothing.
        /// </summary>
        private sealed class FailureLog
        {
            /// <summary>How many messages are kept for the report.</summary>
            private const int Kept = 30;

            /// <summary>The first <see cref="Kept"/> failure messages, in the order they were recorded.</summary>
            public List<string> First { get; } = new List<string>();

            /// <summary>Every failure recorded, kept or not.</summary>
            public long Count { get; private set; }

            /// <summary>Counts one failure and keeps its message while fewer than <see cref="Kept"/> are kept.</summary>
            /// <param name="message">What failed, with the input and both results.</param>
            public void Record(string message)
            {
                Count++;
                if (First.Count < Kept)
                    First.Add(message);
            }
        }
    }
}
