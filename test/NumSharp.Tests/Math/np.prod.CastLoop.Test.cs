using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     <c>np.prod</c> in a cast loop — a <c>dtype=</c> that is neither the input's own dtype nor its accumulator: NumPy's
    ///     buffered iterator casts every element into the loop before <c>multiply</c>'s loop reads it, so a negative integer
    ///     WRAPS into an unsigned loop, uint16 65535 is inf in a float16 loop, and a complex input keeps its real part in a
    ///     real loop. Every literal was probed against NumPy 2.4.2 (<c>np.prod(a, axis, dtype=...)</c>).
    /// </summary>
    /// <remarks>
    ///     Before the fix (gap MX08 of the matrix oracle tier), the flat product read a narrower input AS its float16 /
    ///     complex128 loop (the bytes past the end of an int8 buffer: a different answer on every run), and the axis kernels
    ///     converted per element with their own rules (a negative integer saturated to 0 in an unsigned loop). The engine now
    ///     casts the input first (<c>DefaultEngine.ReduceProductInCastLoop</c>), as its add reduction does.
    /// </remarks>
    [TestClass]
    public class np_prod_cast_loop_tests
    {
        /// <summary>The MX08 pin: the flat product of an int8 array in a bool, float16 and complex128 loop.</summary>
        [TestMethod]
        public void FlatProduct_InACastLoop_CastsEveryElementFirst()
        {
            using var a = np.array(new sbyte[,] { { -1, 0, 1, 2 }, { 3, -1, 0, 1 }, { 2, 3, -1, 0 } });

            // NumPy: False (the bool loop is logical and; a zero element decides it).
            using var asBool = np.prod(a, dtype: np.@bool);
            ((bool)asBool).Should().BeFalse();

            // NumPy: -0.0 (0x8000): -1 * 0 is -0.0, and the sign flips once more at the second -1 and back at the third.
            using var asHalf = np.prod(a, dtype: np.float16);
            BitConverter.HalfToUInt16Bits(asHalf.GetAtIndex<Half>(0)).Should().Be(0x8000);

            // NumPy: (-0+0j): the real part is -0.0, the imaginary part +0.0.
            using var asComplex = np.prod(a, dtype: np.complex128);
            Complex c = asComplex.GetAtIndex<Complex>(0);
            BitConverter.DoubleToInt64Bits(c.Real).Should().Be(unchecked((long)0x8000000000000000UL));
            BitConverter.DoubleToInt64Bits(c.Imaginary).Should().Be(0L);
        }

        /// <summary>
        ///     Along an axis, a negative integer wraps into an unsigned loop: <c>-1</c> as uint32 is 4294967295, and the
        ///     product wraps modulo 2^32 (the axis kernel saturated the -1 to 0).
        /// </summary>
        [TestMethod]
        public void AxisProduct_NegativeIntegers_WrapIntoAnUnsignedLoop()
        {
            using var a = np.array(new sbyte[,] { { -1, 0, 1, 2 }, { 3, -1, 0, 1 }, { 2, 3, -1, 0 } });
            using var r = np.prod(a, 0, np.uint32);
            r.typecode.Should().Be(NPTypeCode.UInt32);
            r.ToArray<uint>().Should().Equal(4294967290u, 0u, 0u, 0u);

            using var b = np.array(new sbyte[,] { { -1, 2, 1, 2 }, { 3, -1, 3, 1 }, { 2, 3, -1, 2 } });
            using var rb = np.prod(b, 0, np.uint32);
            rb.ToArray<uint>().Should().Equal(4294967290u, 4294967290u, 4294967293u, 4u);
        }

        /// <summary>
        ///     A uint16 65535 is inf in a float16 loop (it rounds past float16's 65504), and inf times 0 is NumPy's default
        ///     NaN (0xFE00 on x86): the column products are [inf, NaN, NaN, 0].
        /// </summary>
        [TestMethod]
        public void AxisProduct_IntoAFloat16Loop_RoundsEachElementFirst()
        {
            using var a = np.array(new ushort[,] { { 65535, 0, 1, 2 }, { 3, 65535, 0, 1 }, { 2, 3, 65535, 0 } });
            using var r = np.prod(a, 0, np.float16);
            r.typecode.Should().Be(NPTypeCode.Half);
            var bits = new ushort[4];
            for (int i = 0; i < 4; i++)
                bits[i] = BitConverter.HalfToUInt16Bits(r.GetAtIndex<Half>(i));
            bits.Should().Equal((ushort)0x7C00, (ushort)0xFE00, (ushort)0xFE00, (ushort)0x0000);
        }

        /// <summary>A complex input in a float64 loop keeps its real parts: each row's product is -0.0 here.</summary>
        [TestMethod]
        public void AxisProduct_ComplexIntoAFloat64Loop_KeepsTheRealParts()
        {
            var values = new Complex[,]
            {
                { new(-1, 1), new(0, 0), new(1, 1), new(2, 0) },
                { new(3, 1), new(-1, 1), new(0, 0), new(1, 1) },
                { new(2, 0), new(3, 1), new(-1, 1), new(0, 0) },
            };
            using var a = np.array(values);
            using var r = np.prod(a, 1, np.float64);
            r.typecode.Should().Be(NPTypeCode.Double);
            for (int i = 0; i < 3; i++)
                BitConverter.DoubleToInt64Bits(r.GetDouble(i)).Should().Be(unchecked((long)0x8000000000000000UL));
        }

        /// <summary>
        ///     A 0-d or one-element input in a cast loop: NumPy casts the element too (int8 -1 is 4294967295 as uint32 and
        ///     255 as uint8, uint16 65535 is inf as float16; a (2, 1) int8 [[-1], [3]] is 65533 as uint16).
        /// </summary>
        [TestMethod]
        public void OneElementInputs_InACastLoop_CastTheElement()
        {
            using (var r = np.prod(NDArray.Scalar((sbyte)-1), dtype: np.uint32))
                r.GetAtIndex<uint>(0).Should().Be(4294967295u);
            using (var a = np.array(new sbyte[] { -1 }))
            using (var r = np.prod(a, dtype: np.uint32))
                r.GetAtIndex<uint>(0).Should().Be(4294967295u);
            using (var r = np.prod(NDArray.Scalar((sbyte)-1), dtype: np.uint8))
                r.GetAtIndex<byte>(0).Should().Be((byte)255);
            using (var r = np.prod(NDArray.Scalar((ushort)65535), dtype: np.float16))
                Half.IsPositiveInfinity(r.GetAtIndex<Half>(0)).Should().BeTrue();
            using (var a = np.array(new sbyte[,] { { -1 }, { 3 } }))
            using (var r = np.prod(a, dtype: np.uint16))
                r.GetAtIndex<ushort>(0).Should().Be((ushort)65533);
        }
    }
}
