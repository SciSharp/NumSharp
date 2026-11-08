using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
///     Pins the scalar float16 → float32 widen every emitted kernel uses (<c>DirectILKernelGenerator.HalfToFloatScalarExact</c>,
///     <c>Converts.ToSingle(Half)</c>, and through them the scalar float16 arithmetic of <c>HalfArithBits</c> and every
///     <c>EmitConvertTo(Half, Single)</c>) to NumPy's <c>npy_halfbits_to_floatbits</c> for ALL 65,536 bit patterns.
/// </summary>
/// <remarks>
///     The finite values used to go through the BCL's <c>(float)Half</c>, a call the JIT never inlines: in the polynomial
///     kernels' scalar float16 chains it was most of every step (a root product at 13.9 ns per step, U3's polyval over an
///     F-ordered series 0.63× NumPy). The branch-free Giesen rescale replaced it, and must give the same bits: the BCL's for
///     every finite value and infinity, NumPy's NaN rule (sign, all-ones exponent, payload shifted up, a signalling NaN LEFT
///     signalling — which the BCL would quiet) for every NaN.
/// </remarks>
[TestClass]
public class HalfScalarWidenTests
{
    /// <summary>NumPy's float32 for one float16 pattern: the BCL's exact widen for a non-NaN, the NaN formula otherwise.</summary>
    /// <param name="h">The float16 bit pattern.</param>
    /// <returns>The float32 bit pattern NumPy's <c>npy_halfbits_to_floatbits</c> produces.</returns>
    private static uint Expected(ushort h)
    {
        if ((h & 0x7fff) > 0x7c00)
            return ((uint)(h & 0x8000) << 16) | 0x7f800000u | ((uint)(h & 0x3ff) << 13);
        return BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(h));
    }

    /// <summary>Every pattern: the kernel helper and the public converter both give NumPy's bits.</summary>
    [TestMethod]
    public void EveryPattern_WidensToNumPysBits()
    {
        int bad = 0;
        for (int i = 0; i <= ushort.MaxValue; i++)
        {
            ushort h = (ushort)i;
            uint want = Expected(h);
            if (BitConverter.SingleToUInt32Bits(DirectILKernelGenerator.HalfToFloatScalarExact(h)) != want) bad++;
            if (BitConverter.SingleToUInt32Bits(Converts.ToSingle(BitConverter.UInt16BitsToHalf(h))) != want) bad++;
        }
        bad.Should().Be(0);
    }

    /// <summary>The finite rescale on its own: exact for every finite pattern, -0.0 and the subnormals included.</summary>
    [TestMethod]
    public void FiniteRescale_IsExactForEveryFinitePattern()
    {
        for (int i = 0; i <= ushort.MaxValue; i++)
        {
            ushort h = (ushort)i;
            if ((h & 0x7c00) == 0x7c00)
                continue;   // infinities and NaNs are the caller's branch
            BitConverter.SingleToUInt32Bits(DirectILKernelGenerator.HalfFiniteBitsToSingle(h))
                .Should().Be(BitConverter.SingleToUInt32Bits((float)BitConverter.UInt16BitsToHalf(h)), $"pattern 0x{h:x4}");
        }
        BitConverter.SingleToUInt32Bits(DirectILKernelGenerator.HalfFiniteBitsToSingle(0x8000)).Should().Be(0x80000000u);
        DirectILKernelGenerator.HalfFiniteBitsToSingle(0x0001).Should().Be(MathF.Pow(2, -24));   // the smallest subnormal
    }

    /// <summary>A signalling NaN stays signalling (the BCL cast would set the quiet bit).</summary>
    [TestMethod]
    public void SignallingNaN_StaysSignalling()
    {
        BitConverter.SingleToUInt32Bits(DirectILKernelGenerator.HalfToFloatScalarExact(0x7c01)).Should().Be(0x7f802000u);
        BitConverter.SingleToUInt32Bits(Converts.ToSingle(BitConverter.UInt16BitsToHalf(0xfd55))).Should().Be(0xffaaa000u);
    }
}
