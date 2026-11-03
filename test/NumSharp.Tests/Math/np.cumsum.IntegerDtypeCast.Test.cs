using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.cumsum / np.nancumsum with an integer <c>dtype=</c> over a float, complex or decimal input (parity QA-06).
    ///     NumPy's add.accumulate casts each element to the integer loop with C's conversion (truncate toward zero, then
    ///     wrap; NaN / inf / out-of-range values give the integer minimum truncated to the width), the cast <c>astype</c>
    ///     runs. The axis scan kernel converted with .NET's <c>CreateTruncating</c>, which saturates, so
    ///     <c>np.cumsum([-3.0, 1.0], axis=0, dtype=uint8)</c> was <c>[0, 1]</c> where NumPy (and NumSharp's flat route)
    ///     give <c>[253, 254]</c>.
    /// </summary>
    [TestClass]
    public class np_cumsum_integer_dtype_cast_tests
    {
        /// <summary>The integer loops NumPy has (NumSharp's char lane has no NumPy twin and is left out).</summary>
        private static readonly NPTypeCode[] IntegerLoops =
        {
            NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
            NPTypeCode.Int64, NPTypeCode.UInt64,
        };

        /// <summary>
        ///     The parity QA pin: the axis forms give NumPy's wrapped values (<c>[253, 254]</c>, <c>[[44, 45]]</c>), as the
        ///     flat form already did.
        /// </summary>
        [TestMethod]
        public void CumsumAxis_IntegerDtype_CastsLikeNumPy()
        {
            using var neg = np.array(new[] { -3.0, 1.0 });
            using var big = np.array(new[,] { { 300.0, 1.0 } });
            using var inf = np.array(new[] { double.PositiveInfinity, 1.0 });

            using var unsigned = np.cumsum(neg, axis: 0, typeCode: np.uint8);
            using var signed = np.cumsum(big, axis: 1, typeCode: np.int8);
            using var fromInf = np.cumsum(inf, axis: 0, typeCode: np.int16);

            unsigned.ToArray<byte>().Should().Equal((byte)253, (byte)254);
            signed.ToArray<sbyte>().Should().Equal((sbyte)44, (sbyte)45);
            fromInf.ToArray<short>().Should().Equal((short)0, (short)1);   // NumPy: int16 of inf is 0 (INT_MIN truncated)
        }

        /// <summary>
        ///     The parity QA pin for np.nancumsum: <c>np.nancumsum([[-3.0, nan, 1.0]], axis=1, dtype=uint8)</c> is
        ///     <c>[[253, 253, 254]]</c> (the NaN counts as 0, the floats wrap).
        /// </summary>
        [TestMethod]
        public void NancumsumAxis_IntegerDtype_CastsLikeNumPy()
        {
            var nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            using var a = np.array(new[,] { { -3.0, nan, 1.0 } });

            using var r = np.nancumsum(a, axis: 1, dtype: np.uint8);

            r.ToArray<byte>().Should().Equal((byte)253, (byte)253, (byte)254);
        }

        /// <summary>
        ///     For every floating input dtype and every integer loop, flat and along both axes, <c>np.cumsum(x, axis,
        ///     dtype=D)</c> equals <c>np.cumsum(x.astype(D), axis, dtype=D)</c>: the accumulation casts each element exactly
        ///     as astype does (NaN, ±inf, ±1e10, a fraction and a negative included).
        /// </summary>
        [TestMethod]
        public void Cumsum_FloatingInputs_EveryIntegerLoop_MatchesAstypeThenCumsum()
        {
            var nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            using var values = np.array(new[,]
            {
                { -3.0, 300.0, 2.7, -1e10 },
                { 1e10, double.PositiveInfinity, double.NegativeInfinity, nan },
            });
            using var finite = np.array(new[,] { { -3.0, 300.0, 2.7, -70000.5 }, { 1e10, 65535.9, -1.5, 4.0 } });

            foreach (var source in new[] { NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex })
            {
                using var x = values.astype(source);
                foreach (var loop in IntegerLoops)
                    CheckAgainstAstype(x, loop);
            }

            // decimal (NumSharp's own dtype, no NumPy twin) holds no NaN / inf, so its values are the finite set; only the
            // axis scans run here: the flat decimal kernel converts with .NET's checked decimal casts (an out-of-range value
            // raises OverflowException), a NumSharp-only behaviour outside this parity fix.
            using var dec = finite.astype(NPTypeCode.Decimal);
            foreach (var loop in IntegerLoops)
                CheckAgainstAstype(dec, loop, flat: false);
        }

        /// <summary>
        ///     np.nancumsum of a float input into every integer loop along an axis equals the cumsum of the NaN-zeroed input
        ///     cast with astype.
        /// </summary>
        [TestMethod]
        public void Nancumsum_FloatInput_EveryIntegerLoop_MatchesAstypeOfTheZeroedInput()
        {
            var nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            using var x = np.array(new[,] { { -3.0, nan, 300.0, 2.7 }, { nan, 1e10, -2.5, 70000.0 } });
            using var zeroed = np.where(np.isnan(x), np.zeros_like(x), x);

            foreach (var loop in IntegerLoops)
            {
                foreach (var axis in new[] { 0, 1 })
                {
                    using var actual = np.nancumsum(x, axis: axis, dtype: loop);
                    using var cast = zeroed.astype(loop);
                    using var expected = np.cumsum(cast, axis, loop);

                    actual.dtype.Should().Be(loop);
                    np.array_equal(actual, expected).Should().BeTrue("nancumsum(axis={0}, dtype={1})", axis, loop);
                }
            }
        }

        /// <summary>
        ///     Compares <c>np.cumsum(x, axis, loop)</c> with <c>np.cumsum(x.astype(loop), axis, loop)</c> flat and along
        ///     both axes of a 2-D input.
        /// </summary>
        /// <param name="x">A 2-D floating input.</param>
        /// <param name="loop">The integer loop dtype.</param>
        /// <param name="flat">Whether the flat form (no axis) is checked too.</param>
        /// <exception cref="AssertFailedException">A result differs from astype-then-cumsum.</exception>
        private static void CheckAgainstAstype(NDArray x, NPTypeCode loop, bool flat = true)
        {
            using var cast = x.astype(loop);
            foreach (var axis in flat ? new int?[] { null, 0, 1 } : new int?[] { 0, 1 })
            {
                using var actual = np.cumsum(x, axis, loop);
                using var expected = np.cumsum(cast, axis, loop);

                actual.dtype.Should().Be(loop);
                np.array_equal(actual, expected).Should().BeTrue("cumsum of {0} along axis {1} into {2}", x.typecode, axis, loop);
            }
        }
    }
}
