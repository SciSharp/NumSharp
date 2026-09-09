using System;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     Special / extreme values survive every path bit-for-bit — the interop is a raw byte reinterpret, so
    ///     uncanonicalized NaNs (payloads / sign), ±inf, signed zero, subnormals and integer extremes must be
    ///     preserved through export (AsTensorSpan flatten + ToTensor) and import (ToNDArray + AsNDArray).
    /// </summary>
    [TestClass]
    public class SpecialValueFidelityTests : TensorsTestBase
    {
        [TestMethod]
        public void Float32_SpecialValues_BitExact_EveryPath()
        {
            AssertBitExact(np.array(new float[]
            {
                float.NaN,
                BitConverter.UInt32BitsToSingle(0xffc00000u),   // negative-signed NaN
                BitConverter.UInt32BitsToSingle(0x7f800001u),   // signalling NaN payload
                float.PositiveInfinity, float.NegativeInfinity,
                -0.0f, 0.0f, float.Epsilon, float.MaxValue, float.MinValue,
            }), new float[10]);
        }

        [TestMethod]
        public void Float64_SpecialValues_BitExact_EveryPath()
        {
            AssertBitExact(np.array(new double[]
            {
                double.NaN,
                BitConverter.UInt64BitsToDouble(0xfff8000000000000ul),   // negative-signed NaN
                BitConverter.UInt64BitsToDouble(0x7ff0000000000001ul),   // signalling NaN payload
                double.PositiveInfinity, double.NegativeInfinity,
                -0.0, 0.0, double.Epsilon, double.MaxValue, double.MinValue,
            }), new double[10]);
        }

        [TestMethod]
        public void Half_SpecialValues_BitExact_EveryPath()
        {
            AssertBitExact(np.array(new[]
            {
                Half.NaN, Half.PositiveInfinity, Half.NegativeInfinity,
                Half.Epsilon, Half.MaxValue, Half.MinValue, (Half)(-0.0f), (Half)0.0f,
            }), new Half[8]);
        }

        [TestMethod]
        public void Int64_And_UInt64_Extremes_BitExact()
        {
            AssertBitExact(np.array(new[] { long.MinValue, long.MaxValue, -1L, 0L, 1L }), new long[5]);
            AssertBitExact(np.array(new[] { ulong.MaxValue, 0ul, (1ul << 63) + 1, (1ul << 53) + 7 }), new ulong[4]);
        }

        [TestMethod]
        public void Complex_SpecialComponents_BitExact()
        {
            AssertBitExact(np.array(new[]
            {
                new Complex(double.NaN, -0.0),
                new Complex(double.PositiveInfinity, double.NegativeInfinity),
                new Complex(1.5, -2.5),
            }), new Complex[3]);
        }

        [TestMethod]
        public void Decimal_Values_BitExact()
        {
            AssertBitExact(np.array(new[] { decimal.MaxValue, decimal.MinValue, 0m, -1.2345m, 79228162514264337593543950335m }), new decimal[5]);
        }

        /// <summary>
        ///     Assert byte-for-byte identity of <paramref name="nd"/> through AsTensorSpan (flatten), ToTensor →
        ///     ToNDArray, and Tensor → AsNDArray. <paramref name="scratch"/> only fixes the element type + length.
        /// </summary>
        private static void AssertBitExact<T>(NDArray nd, T[] scratch) where T : unmanaged
        {
            byte[] expected = BytesOf(nd);

            // 1) export zero-copy, flatten, compare bytes
            using (var h = nd.AsTensorSpan<T>())
            {
                h.ReadOnlySpan.FlattenTo(scratch);
                MemoryMarshal.AsBytes<T>(scratch).ToArray().Should().Equal(expected, "AsTensorSpan preserves bytes");
            }

            // 2) export copy -> import copy
            Tensor<T> t = nd.ToTensor<T>();
            using (NDArray back = t.ToNDArray())
                BytesOf(back).Should().Equal(expected, "ToTensor -> ToNDArray preserves bytes");

            // 3) import zero-copy from a fresh tensor built on the same bytes
            t.FlattenTo(scratch);
            var t2 = Tensor.Create((T[])scratch.Clone(), t.Lengths.ToArray());
            using (NDArray view = t2.AsNDArray())
                BytesOf(view).Should().Equal(expected, "Tensor -> AsNDArray preserves bytes");
        }
    }
}
