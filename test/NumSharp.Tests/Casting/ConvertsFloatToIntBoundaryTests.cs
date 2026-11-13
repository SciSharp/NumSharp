using System;
using NumSharp.Utilities;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     The float → {int32, int16, uint16, int8, uint8} converters at int32's edges: the scalar per-element body of every
    ///     float → small-int cast kernel on a host without the x86 vector cast (arm64, and Mono on Android, iOS and
    ///     WebAssembly). <see cref="Converts.ToInt32(double)"/> decides the edges with one range test on the untruncated
    ///     value instead of <see cref="System.Math.Truncate(double)"/>, which Mono compiles to a call (5x a plain cast loop on an
    ///     arm64 phone); these tests hold it to NumPy 2.4.2's results and to the truncate-then-test definition it replaced.
    /// </summary>
    [TestClass]
    public class ConvertsFloatToIntBoundaryTests
    {
        /// <summary>
        ///     NumPy 2.4.2 (win-amd64), <c>np.array(values, np.float64).astype(np.int32)</c>: a fraction just inside either
        ///     end truncates into range, everything outside, NaN and both infinities give int32.MinValue.
        /// </summary>
        [TestMethod]
        public void ToInt32_Double_AtInt32sEdges_MatchesNumPy()
        {
            double[] values = { 2147483647.4, 2147483647.9, -2147483648.9, -2147483648.0, 2147483648.0, -2147483649.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0.9, -0.9, -0.0, 1e300 };
            int[] numpy = { 2147483647, 2147483647, -2147483648, -2147483648, -2147483648, -2147483648, -2147483648, -2147483648, -2147483648, 0, 0, 0, -2147483648 };
            for (int i = 0; i < values.Length; i++)
                Converts.ToInt32(values[i]).Should().Be(numpy[i], $"ToInt32({values[i]:R})");
        }

        /// <summary>
        ///     NumPy 2.4.2, <c>np.array(values, np.float32).astype(dtype)</c> for int16 / uint16 / int8 / uint8: the int32
        ///     intermediate wraps to the narrow width (70000.7 → 4464), and values with no int32 value (2^31, NaN, inf) give
        ///     int32.MinValue's low bits, 0.
        /// </summary>
        [TestMethod]
        public void ToNarrowInts_Single_WrapThroughInt32_MatchNumPy()
        {
            float[] values = { 70000.7f, -70000.7f, 32767.9f, -32768.9f, 65535.5f, 2147483520f, 2147483648f, float.NaN, float.PositiveInfinity };
            short[] int16 = { 4464, -4464, 32767, -32768, -1, -128, 0, 0, 0 };
            ushort[] uint16 = { 4464, 61072, 32767, 32768, 65535, 65408, 0, 0, 0 };
            sbyte[] int8 = { 112, -112, -1, 0, -1, -128, 0, 0, 0 };
            byte[] uint8 = { 112, 144, 255, 0, 255, 128, 0, 0, 0 };
            for (int i = 0; i < values.Length; i++)
            {
                Converts.ToInt16(values[i]).Should().Be(int16[i], $"ToInt16({values[i]:R})");
                Converts.ToUInt16(values[i]).Should().Be(uint16[i], $"ToUInt16({values[i]:R})");
                Converts.ToSByte(values[i]).Should().Be(int8[i], $"ToSByte({values[i]:R})");
                Converts.ToByte(values[i]).Should().Be(uint8[i], $"ToByte({values[i]:R})");
            }
        }

        /// <summary>
        ///     The range test equals the definition it replaced — truncate toward zero, then NaN / ±inf / out of int32 →
        ///     int32.MinValue — over 2^24 float32 bit patterns spread evenly across all 2^32 (every exponent and sign, NaN
        ///     encodings included; ±inf are in the NumPy tests above), and over every double within four ulps of the four
        ///     edges -2^31 - 1, -2^31, 2^31 - 1 and 2^31. (The full 2^32 float32 sweep was run once by hand: no difference.)
        /// </summary>
        [TestMethod]
        public void ToInt32_Double_EqualsTruncateThenRangeCheck()
        {
            static int Definition(double value)
            {
                if (double.IsNaN(value) || double.IsInfinity(value)) return int.MinValue;
                double t = System.Math.Truncate(value);
                return t < int.MinValue || t > int.MaxValue ? int.MinValue : (int)t;
            }

            for (long k = 0; k < 1L << 24; k++)   // 2^24 patterns: every high 24 bits, the low byte fixed at an odd 0x7F
            {
                uint bits = (uint)(k << 8) | 0x7F;
                double v = BitConverter.UInt32BitsToSingle(bits);
                if (Converts.ToInt32(v) != Definition(v))
                    Assert.Fail($"float32 0x{bits:X8} ({v:R}): {Converts.ToInt32(v)}, the definition gives {Definition(v)}");
            }

            foreach (double edge in new[] { -2147483649.0, -2147483648.0, 2147483647.0, 2147483648.0 })
            {
                double down = edge, up = edge;
                for (int step = 0; step <= 4; step++)
                {
                    Converts.ToInt32(down).Should().Be(Definition(down), $"ToInt32({down:R})");
                    Converts.ToInt32(up).Should().Be(Definition(up), $"ToInt32({up:R})");
                    down = System.Math.BitDecrement(down);
                    up = System.Math.BitIncrement(up);
                }
            }
        }
    }
}
