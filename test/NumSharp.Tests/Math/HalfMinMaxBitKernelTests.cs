using System;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins the float16 flat min/max bit-level kernel
/// (DirectILKernelGenerator.HalfMaxBitsContiguous / HalfMinBitsContiguous) against NumPy 2.4.2's
/// HALF_maximum/HALF_minimum fold <c>(ge(acc,x) || isnan(acc)) ? acc : x</c>. Every expected bit
/// pattern below was probed live on NumPy 2.4.2 (np.max/np.min/np.ptp over uint16.view(float16)).
/// Two semantics are the reason these tests exist — both were WRONG on the pre-kernel scalar scan
/// (<c>x &gt; acc || IsNaN(x)</c> let a LATER NaN overwrite the accumulator):
///   • first-NaN wins, payload AND sign returned VERBATIM (max([nanB,nanA]) = nanB;
///     max([-nan(0xfe22), nanA]) = 0xfe22 — a NEGATIVE NaN result);
///   • ±0 ties keep the FIRST zero's sign (npy_half ge/le call -0 == +0), while the kernel's
///     order-key map ranks -0 &lt; +0 — the first-zero rescan restores NumPy's answer.
/// </summary>
[TestClass]
public class HalfMinMaxBitKernelTests
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

    private static ushort MaxBits(NDArray a) { using var r = np.max(a); return BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)); }
    private static ushort MinBits(NDArray a) { using var r = np.min(a); return BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)); }

    // ── first-NaN wins, payload + sign verbatim (probed 2.4.2) ───────────────

    [TestMethod]
    public void Max_FirstNaN_PayloadPreserved()
    {
        // np.max(view([0x7e01, 0x7d05])) -> 0x7e01; reversed order -> 0x7d05
        using var a = FromBits(0x7e01, 0x7d05);
        MaxBits(a).Should().Be((ushort)0x7e01);
        MinBits(a).Should().Be((ushort)0x7e01);
        using var b = FromBits(0x7d05, 0x7e01);
        MaxBits(b).Should().Be((ushort)0x7d05);
        MinBits(b).Should().Be((ushort)0x7d05);
    }

    [TestMethod]
    public void Max_NegativeNaN_ReturnedWithSign()
    {
        // np.max(view([0xfe22, 0x7e01])) -> 0xfe22 (NEGATIVE NaN survives verbatim)
        using var a = FromBits(0xfe22, 0x7e01);
        MaxBits(a).Should().Be((ushort)0xfe22);
        MinBits(a).Should().Be((ushort)0xfe22);
        // NaN in the middle of finite values: np.max(view([5, 0xfe22, 9])) -> 0xfe22
        using var b = FromBits(0x4500, 0xfe22, 0x4880);
        MaxBits(b).Should().Be((ushort)0xfe22);
        MinBits(b).Should().Be((ushort)0xfe22);
    }

    [TestMethod]
    public void Max_NaN_AtEveryLoopBoundary()
    {
        // NaN at scalar-tail / 16-lane / 64-unroll seams of the SIMD kernel must all be caught.
        foreach (var pos in new[] { 0, 15, 16, 63, 64, 65, 99 })
        {
            var bits = new ushort[100];
            for (int i = 0; i < bits.Length; i++) bits[i] = (ushort)(0x3c00 + i);
            bits[pos] = 0x7e11;
            using var a = FromBits(bits);
            MaxBits(a).Should().Be((ushort)0x7e11, $"NaN at position {pos} must propagate");
            MinBits(a).Should().Be((ushort)0x7e11, $"NaN at position {pos} must propagate");
        }
    }

    // ── ±0 ties keep the FIRST zero's sign (probed 2.4.2) ────────────────────

    [TestMethod]
    public void Max_SignedZeroTie_FirstZeroWins()
    {
        // np.max(view([0x8000, 0x0000])) -> 0x8000 (-0); np.max(view([0x0000, 0x8000])) -> 0x0000
        using var a = FromBits(0x8000, 0x0000);
        MaxBits(a).Should().Be((ushort)0x8000);
        MinBits(a).Should().Be((ushort)0x8000);
        using var b = FromBits(0x0000, 0x8000);
        MaxBits(b).Should().Be((ushort)0x0000);
        MinBits(b).Should().Be((ushort)0x0000);
    }

    [TestMethod]
    public void Max_ZeroAmongNegatives_FirstZeroSign()
    {
        // np.max(view([-1, -0, +0, -0])) -> 0x8000 (the FIRST zero, -0), min -> -1
        using var a = FromBits(0xbc00, 0x8000, 0x0000, 0x8000);
        MaxBits(a).Should().Be((ushort)0x8000);
        MinBits(a).Should().Be((ushort)0xbc00);
        // np.min(view([+1, +0, -0])) -> 0x0000 (the FIRST zero, +0)
        using var b = FromBits(0x3c00, 0x0000, 0x8000);
        MinBits(b).Should().Be((ushort)0x0000);
        MaxBits(b).Should().Be((ushort)0x3c00);
    }

    // ── ordinary ordering incl. inf / subnormals ─────────────────────────────

    [TestMethod]
    public void MinMax_Infinities_And_Subnormals()
    {
        using var a = FromBits(0x7c00, 0xfc00); // [+inf, -inf]
        MaxBits(a).Should().Be((ushort)0x7c00);
        MinBits(a).Should().Be((ushort)0xfc00);
        using var b = FromBits(0x0001, 0x8001); // [min subnormal, -min subnormal]
        MaxBits(b).Should().Be((ushort)0x0001);
        MinBits(b).Should().Be((ushort)0x8001);
    }

    [TestMethod]
    public void MinMax_LargeArray_MatchesScalarFold()
    {
        // Deterministic mixed-sign sweep long enough to exercise the 64-wide unroll,
        // the 16-wide remainder, and the scalar tail (n = 1000 = 15*64 + 2*16 + 8).
        var bits = new ushort[1000];
        var rng = new System.Random(42);
        for (int i = 0; i < bits.Length; i++)
        {
            ushort v = (ushort)rng.Next(0x10000);
            if ((v & 0x7fff) > 0x7c00) v &= 0x7c00; // NaN-free lane: ordering itself is under test
            bits[i] = v;
        }
        ushort expMax = bits[0], expMin = bits[0];
        for (int i = 1; i < bits.Length; i++)
        {
            if ((float)BitConverter.UInt16BitsToHalf(bits[i]) > (float)BitConverter.UInt16BitsToHalf(expMax)) expMax = bits[i];
            if ((float)BitConverter.UInt16BitsToHalf(bits[i]) < (float)BitConverter.UInt16BitsToHalf(expMin)) expMin = bits[i];
        }
        using var a = FromBits(bits);
        MaxBits(a).Should().Be(expMax);
        MinBits(a).Should().Be(expMin);
    }

    [TestMethod]
    public void Ptp_Half_InheritsKernel_NaNAndZeroCases()
    {
        // ptp = subtract(max, min): np.ptp(view([0x7d05, 0x7e01])) -> 0x7f05 (probed 2.4.2 —
        // nanB - nanB quiets the payload's MSB), and ptp over [-0,+0] -> +0 (max=-0, min=-0, -0-(-0)=+0).
        using var a = FromBits(0x7d05, 0x7e01);
        using var p = np.ptp(a);
        BitConverter.HalfToUInt16Bits((Half)p.GetAtIndex(0)).Should().Be((ushort)0x7f05);
        using var b = FromBits(0x8000, 0x0000);
        using var q = np.ptp(b);
        BitConverter.HalfToUInt16Bits((Half)q.GetAtIndex(0)).Should().Be((ushort)0x0000);
    }

    [TestMethod]
    public void MinMax_OffsetContiguousView_UsesKernelCorrectly()
    {
        // A sliced-contiguous view (offset != 0) rides the kernel path — the base pointer
        // must honor Shape.offset.
        var bits = new ushort[64];
        for (int i = 0; i < bits.Length; i++) bits[i] = (ushort)(0x4000 + i); // ascending positives
        bits[0] = 0x7e01;  // NaN OUTSIDE the view
        bits[63] = 0x7c00; // +inf OUTSIDE the view
        using var full = FromBits(bits);
        var view = full["1:63"];
        MaxBits(view).Should().Be((ushort)(0x4000 + 62));
        MinBits(view).Should().Be((ushort)(0x4000 + 1));
    }

    // ── nanmin/nanmax (fmax.reduce / fmin.reduce semantics; probed 2.4.2) ────

    [TestMethod]
    public void NanMinMax_SkipNaN_AllNaN_FirstElementVerbatim()
    {
        // nanmax([1, nan, 3]) skips the NaN -> 3
        using var a = FromBits(0x3c00, 0x7e01, 0x4200);
        using (var r = np.nanmax(a)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x4200);
        using (var r = np.nanmin(a)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x3c00);
        // all-NaN: the fold accumulator never leaves the FIRST element - returned VERBATIM
        // (payload + sign; the old scalar helper returned canonical Half.NaN)
        using var b = FromBits(0x7e01, 0xfe22);
        using (var r = np.nanmax(b)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x7e01);
        using var c = FromBits(0xfe22, 0x7d05);
        using (var r = np.nanmin(c)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0xfe22);
    }

    [TestMethod]
    public void NanMinMax_ZeroTies_FirstZeroWins_EvenAfterLeadingNaN()
    {
        // nanmax([-0,+0]) = -0 (first zero); leading NaN is skipped, zeros still tie to first
        using var a = FromBits(0x8000, 0x0000);
        using (var r = np.nanmax(a)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x8000);
        using var b = FromBits(0x7e01, 0x8000, 0x0000);
        using (var r = np.nanmax(b)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x8000);
        using (var r = np.nanmin(b)) BitConverter.HalfToUInt16Bits((Half)r.GetAtIndex(0)).Should().Be((ushort)0x8000);
    }
}
