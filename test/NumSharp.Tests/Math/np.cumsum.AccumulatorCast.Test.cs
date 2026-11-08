using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     np.cumsum / np.cumprod with a float16 or complex128 <c>dtype=</c> over an input of another dtype. NumPy's
    ///     add.accumulate / multiply.accumulate cast the input to the loop dtype and accumulate there; NumSharp's scan
    ///     kernels emitted a plain add / mul for these struct accumulators, which the JIT rejected for float16
    ///     (InvalidProgramException), ran over the raw bits for complex128 (wrong values, and an access violation where
    ///     the 16-byte reads ran past the input buffer) and refused along an axis (NotSupportedException). Every expected
    ///     value below is NumPy 2.4.2's.
    /// </summary>
    [TestClass]
    public class np_cumsum_accumulator_cast_tests
    {
        /// <summary>
        ///     <c>np.cumsum(np.arange(1, 13, dtype=uint64), dtype=complex128)</c>: the flat case that ended the process with
        ///     an access violation (12 elements of 8 bytes read as 16-byte complex values).
        /// </summary>
        [TestMethod]
        public void Cumsum_UInt64ToComplex_Flat()
        {
            using var a = np.arange(1, 13).astype(np.uint64);
            using var r = np.cumsum(a, null, np.complex128);
            r.dtype.Should().Be(np.complex128);
            r.ToArray<Complex>().Should().Equal(new Complex[] { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78 });
        }

        /// <summary>
        ///     <c>np.cumsum([3, -1, 2, 0, 5, -4] (int8), dtype=float16)</c> is <c>[3, 2, 4, 4, 9, 5]</c>; the converting float16
        ///     kernel was invalid IL.
        /// </summary>
        [TestMethod]
        public void Cumsum_Int8ToHalf_Flat()
        {
            using var a = np.array(new sbyte[] { 3, -1, 2, 0, 5, -4 });
            using var r = np.cumsum(a, null, np.float16);
            r.dtype.Should().Be(np.float16);
            using var bits = r.view(np.uint16);
            bits.ToArray<ushort>().Should().Equal(new ushort[] { 0x4200, 0x4000, 0x4400, 0x4400, 0x4880, 0x4500 });
        }

        /// <summary>
        ///     <c>np.cumprod([3, -1, 2, 0, 5, -4] (int16), dtype=float16)</c> is <c>[3, -3, -6, -0.0, -0.0, 0.0]</c>: the
        ///     signed zeros show the product runs in float16 after the cast.
        /// </summary>
        [TestMethod]
        public void Cumprod_Int16ToHalf_Flat_KeepsSignedZeros()
        {
            using var a = np.array(new short[] { 3, -1, 2, 0, 5, -4 });
            using var r = np.cumprod(a, null, np.float16);
            using var bits = r.view(np.uint16);
            bits.ToArray<ushort>().Should().Equal(new ushort[] { 0x4200, 0xc200, 0xc600, 0x8000, 0x8000, 0x0000 });
        }

        /// <summary>
        ///     <c>np.cumsum(int16 (3, 4), axis=0, dtype=complex128)</c>: the axis path refused the conversion.
        /// </summary>
        [TestMethod]
        public void Cumsum_Int16ToComplex_Axis0()
        {
            using var a = np.array(new short[] { 3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3 }).reshape(3, 4);
            using var r = np.cumsum(a, 0, np.complex128);
            r.shape.Should().Equal(new long[] { 3, 4 });
            r.flags.c_contiguous.Should().BeTrue();
            r.ToArray<Complex>().Should().Equal(new Complex[] { 3, -1, 2, 0, 8, -5, 3, 7, 6, 1, 7, 4 });
        }

        /// <summary>
        ///     An F-ordered int32 input along axis 1: the result keeps the input's memory order (NumPy's KEEPORDER output),
        ///     as it does without the cast.
        /// </summary>
        [TestMethod]
        public void Cumsum_FOrderedInt32ToComplex_Axis1_KeepsFOrder()
        {
            using var c = np.array(new[] { 3, -1, 2, 0, 5, -4, 1, 7, -2, 6, 4, -3 }).reshape(3, 4);
            using var f = np.asfortranarray(c);
            using var r = np.cumsum(f, 1, np.complex128);
            r.flags.f_contiguous.Should().BeTrue();
            r.flags.c_contiguous.Should().BeFalse();
            using var cOrder = np.ascontiguousarray(r);
            cOrder.ToArray<Complex>().Should().Equal(new Complex[] { 3, 2, 4, 4, 5, 1, 2, 9, -2, 4, 8, 5 });
        }

        /// <summary>
        ///     float16 and bool inputs into complex128: <c>np.cumsum([1.5, -2.25, 4.0, 0.5] (float16), dtype=complex128)</c>
        ///     and <c>np.cumprod([True, True, False, True], dtype=complex128)</c>.
        /// </summary>
        [TestMethod]
        public void Cumsum_HalfAndBoolToComplex_Flat()
        {
            using var h = np.array(new[] { (System.Half)1.5, (System.Half)(-2.25), (System.Half)4.0, (System.Half)0.5 });
            using var s = np.cumsum(h, null, np.complex128);
            s.ToArray<Complex>().Should().Equal(new Complex[] { 1.5, -0.75, 3.25, 3.75 });

            using var b = np.array(new[] { true, true, false, true });
            using var p = np.cumprod(b, null, np.complex128);
            p.ToArray<Complex>().Should().Equal(new Complex[] { 1, 1, 0, 0 });
        }

        /// <summary>
        ///     <c>np.cumprod([[2, 3], [4, 5]] (uint8), dtype=float16)</c> flattened is <c>[2, 6, 24, 120]</c>.
        /// </summary>
        [TestMethod]
        public void Cumprod_UInt8ToHalf_2D_Flat()
        {
            using var a = np.array(new byte[] { 2, 3, 4, 5 }).reshape(2, 2);
            using var r = np.cumprod(a, null, np.float16);
            using var bits = r.view(np.uint16);
            bits.ToArray<ushort>().Should().Equal(new ushort[] { 0x4000, 0x4600, 0x4e00, 0x5780 });
        }
    }
}
