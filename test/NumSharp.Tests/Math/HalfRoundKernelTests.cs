using System;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins the float16 floor/ceil/trunc/rint bit-level kernels
/// (DirectILKernelGenerator.Unary.Round.Half.cs) against NumPy 2.4.2 — proven by an
/// exhaustive all-65,536-pattern sweep per op (0 diffs, dev-time); these tests pin the
/// seams that would break silently: sNaN quieting (|0x0200), ceil's negative-zero
/// production (ceil(-0.2) = -0), the sub-1 half-tie (rint(±0.5) = ±0), the mid-range
/// carry across an exponent boundary (floor(-1.001) = -2), RTNE ties at spacing-2
/// (rint via around), and the ≥1024 identity band.
/// </summary>
[TestClass]
public class HalfRoundKernelTests
{
    private static NDArray FromBits(params ushort[] bits)
    {
        var nd = np.zeros(new Shape(bits.Length), NPTypeCode.Half);
        unsafe
        {
            ushort* p = (ushort*)nd.Address;
            for (int i = 0; i < bits.Length; i++) p[i] = bits[i];
        }
        return nd;
    }

    private static unsafe ushort[] Bits(NDArray nd)
    {
        var r = new ushort[nd.size];
        ushort* p = (ushort*)nd.Address;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    // probed 2.4.2 over: sNaN+ / sNaN- / qNaN / -0 / -0.2 / +0.2 / -1.001 / +minsub /
    // -minsub / 10 / -10 / 0.5 / -0.5 / 1.5 / 3 / -1.5
    private static readonly ushort[] In =
    {
        0x7c01, 0xfc01, 0x7e05, 0x8000, 0xb266, 0x3266, 0xbc01, 0x0001,
        0x8001, 0x4900, 0xc900, 0x3800, 0xb800, 0x3e00, 0x4200, 0xbe00,
    };

    private static readonly ushort[] WantFloor =
    {
        0x7e01, 0xfe01, 0x7e05, 0x8000, 0xbc00, 0x0000, 0xc000, 0x0000,
        0xbc00, 0x4900, 0xc900, 0x0000, 0xbc00, 0x3c00, 0x4200, 0xc000,
    };

    private static readonly ushort[] WantCeil =
    {
        0x7e01, 0xfe01, 0x7e05, 0x8000, 0x8000, 0x3c00, 0xbc00, 0x3c00,
        0x8000, 0x4900, 0xc900, 0x3c00, 0x8000, 0x4000, 0x4200, 0xbc00,
    };

    private static readonly ushort[] WantTrunc =
    {
        0x7e01, 0xfe01, 0x7e05, 0x8000, 0x8000, 0x0000, 0xbc00, 0x0000,
        0x8000, 0x4900, 0xc900, 0x0000, 0x8000, 0x3c00, 0x4200, 0xbc00,
    };

    private static readonly ushort[] WantRint =
    {
        0x7e01, 0xfe01, 0x7e05, 0x8000, 0x8000, 0x0000, 0xbc00, 0x0000,
        0x8000, 0x4900, 0xc900, 0x0000, 0x8000, 0x4000, 0x4200, 0xc000,
    };

    [TestMethod]
    public void Round_ProbedMatrix_AllFourOps()
    {
        using var a = FromBits(In);
        Bits(np.floor(a)).Should().Equal(WantFloor);
        Bits(np.ceil(a)).Should().Equal(WantCeil);
        Bits(np.trunc(a)).Should().Equal(WantTrunc);
        Bits(np.rint(a)).Should().Equal(WantRint);
    }

    [TestMethod]
    public void Round_TailAndSimdSeams_Agree()
    {
        // n = 21: the 8-lane SIMD body handles 16, the scalar tail 5 — every element
        // must match the scalar bit formula (checked against HalfRoundBits directly).
        var bits = new ushort[21];
        var rng = new System.Random(77);
        for (int i = 0; i < 21; i++) bits[i] = (ushort)rng.Next(0x10000);
        using var a = FromBits(bits);
        var got = Bits(np.floor(a));
        for (int i = 0; i < 21; i++)
            got[i].Should().Be(NumSharp.Backends.Kernels.DirectILKernelGenerator.HalfRoundBits(bits[i], NumSharp.Backends.Kernels.UnaryOp.Floor), $"floor lane {i}");
    }

    [TestMethod]
    public void Round_Around_SharesRintKernel_RTNETies()
    {
        // np.around(decimals=0) rides UnaryOp.Round — the same kernel. RTNE ties:
        // 2.5 (0x4100) → 2 (0x4000); 3.5 (0x4300) → 4 (0x4400); 682.5 (0x6155) → 682 (0x6154)
        using var b = FromBits(0x4100, 0x4300, 0x6155);
        Bits(np.around(b)).Should().Equal(new ushort[] { 0x4000, 0x4400, 0x6154 });
        Bits(np.rint(b)).Should().Equal(new ushort[] { 0x4000, 0x4400, 0x6154 });
    }

    [TestMethod]
    public void Round_IdentityBand_1024AndAbove_IncludingInf()
    {
        using var a = FromBits(0x6400, 0x7bff, 0x7c00, 0xfc00, 0xe400);
        Bits(np.floor(a)).Should().Equal(new ushort[] { 0x6400, 0x7bff, 0x7c00, 0xfc00, 0xe400 });
        Bits(np.rint(a)).Should().Equal(new ushort[] { 0x6400, 0x7bff, 0x7c00, 0xfc00, 0xe400 });
    }
}
