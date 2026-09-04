using System;
using System.Runtime.InteropServices;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins the float16 add/subtract/multiply/divide widen-compute-narrow kernels
/// (DirectILKernelGenerator.Binary.Arith.Half.cs) against NumPy 2.4.2's HALF loops
/// (npy_float_to_half(npy_half_to_float(in1) OP npy_half_to_float(in2)) — float32
/// compute, RTNE narrow). Every expected pattern probed live on 2.4.2.
/// The NaN-payload PRIORITY is a host pin: the wheel's MSVC build compiled the
/// commutative ops with REVERSED operands, so add/multiply resolve in2's NaN first
/// while subtract/divide resolve in1's — and RyuJIT may re-swap commutative intrinsic
/// operands (observed), so the kernels encode the priority as an EXPLICIT blend,
/// which these tests pin at every path (streamed + both scalar-broadcast sides).
/// </summary>
[TestClass]
public class HalfArithKernelTests
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

    // 0/0 and inf + -inf GENERATE a fresh qNaN whose SIGN is the CPU's default-NaN sign — NOT
    // something the widen-compute-narrow kernel chooses (non-NaN operand lanes take the hardware
    // FP result untouched; see DirectILKernelGenerator.Binary.Arith.Half.cs, and note the SIMD path
    // is Avx2-gated so arm64 runs the scalar HalfArithBits fallback, which hits the same hardware
    // division). x86/x64 emit the NEGATIVE default NaN (float32 0xFFC00000 -> half 0xFE00), matching
    // the numpy 2.4.2 win-amd64 wheel this kernel is byte-pinned to; AArch64 emits the POSITIVE
    // default NaN (0x7FC00000 -> 0x7E00). Assert the arch-appropriate default so the x86 byte-parity
    // gate stays strict while the test still holds on the Apple-silicon (arm64) CI runner.
    private static readonly ushort DefaultQNaNHalf =
        RuntimeInformation.ProcessArchitecture is Architecture.X86 or Architecture.X64
            ? (ushort)0xfe00 : (ushort)0x7e00;

    [TestMethod]
    public void Arith_NaNPriority_ProbedMatrix()
    {
        // (in1, in2): qNaN+finite, finite+sNaN, qNaN+sNaN, both-qNaN — probed 2.4.2:
        //   add/mul resolve in2 first (quieted), sub/div resolve in1 first.
        using var a = FromBits(0x7e05, 0x3c00, 0x7e05, 0xfe22);
        using var b = FromBits(0x3c00, 0x7d11, 0x7d11, 0x7e05);
        Bits(np.add(a, b)).Should().Equal(new ushort[] { 0x7e05, 0x7f11, 0x7f11, 0x7e05 });
        Bits(np.multiply(a, b)).Should().Equal(new ushort[] { 0x7e05, 0x7f11, 0x7f11, 0x7e05 });
        Bits(np.subtract(a, b)).Should().Equal(new ushort[] { 0x7e05, 0x7f11, 0x7e05, 0xfe22 });
        Bits(np.divide(a, b)).Should().Equal(new ushort[] { 0x7e05, 0x7f11, 0x7e05, 0xfe22 });
    }

    [TestMethod]
    public void Arith_Specials_DefaultQNaN_Inf_SignedZero()
    {
        // probed 2.4.2: 0/0 and inf-inf produce the x86 default qNaN 0xfe00 (arm64: 0x7e00 — the
        // sign is the CPU's default-NaN sign, see DefaultQNaNHalf);
        // 1/0 = +inf; 1*0 with -0: sign rules; overflow saturates to inf.
        using var a = FromBits(0x0000, 0x7c00, 0x3c00, 0x7bff, 0xbc00);
        using var b = FromBits(0x0000, 0xfc00, 0x0000, 0x7bff, 0x8000);
        Bits(np.divide(a, b))[0].Should().Be(DefaultQNaNHalf, "0/0 -> default qNaN");
        Bits(np.add(a, b))[1].Should().Be(DefaultQNaNHalf, "inf + -inf -> default qNaN");
        Bits(np.divide(a, b))[2].Should().Be((ushort)0x7c00, "1/0 -> +inf");
        Bits(np.add(a, b))[3].Should().Be((ushort)0x7c00, "65504+65504 overflows to +inf");
        Bits(np.multiply(a, b))[4].Should().Be((ushort)0x0000, "-1 * -0 = +0");
    }

    [TestMethod]
    public void Arith_NarrowRTNE_TieSeam()
    {
        // The f32-compute + RTNE-narrow pipeline's rounding seam, probed 2.4.2:
        // 1024 + 1 = 1025 is exactly representable (spacing 1 there) -> 0x6401;
        // 2048 + 1 = 2049 is a TIE between 2048/2050 (spacing 2) -> even 2048;
        // 2048 + 3 = 2051 ties between 2050/2052 -> even 2052.
        using var a = FromBits(0x3c00, 0x6800, 0x6800); // 1, 2048, 2048
        using var b = FromBits(0x6400, 0x3c00, 0x4200); // 1024, 1, 3
        var g = Bits(np.add(a, b));
        g[0].Should().Be((ushort)0x6401, "1024 + 1 = 1025 exactly");
        g[1].Should().Be((ushort)0x6800, "2049 ties RTNE to 2048");
        g[2].Should().Be((ushort)0x6802, "2051 ties RTNE to 2052");
    }

    [TestMethod]
    public void Arith_ScalarBroadcast_BothSides_NaNPriorityHolds()
    {
        // scalar = sNaN 0xfd11; stream holds an sNaN 0x7c01 lane and finite lanes.
        var bits = new ushort[40];
        for (int i = 0; i < 40; i++) bits[i] = (ushort)(0x3c00 + i);
        bits[7] = 0x7c01;
        using var arr = FromBits(bits);
        using var sc = FromBits(0xfd11);
        sc.Storage.Reshape(Shape.Scalar);

        // add(arr, sc): in2 (scalar) NaN wins EVERY lane -> quiet(0xfd11) = 0xff11
        var g1 = Bits(np.add(arr, sc));
        for (int i = 0; i < 40; i++) g1[i].Should().Be((ushort)0xff11, $"add lane {i}");
        // add(sc, arr): in2 (stream) — finite lanes take quiet(scalar-in1)... in2First:
        // stream lane NaN -> quiet(stream); else scalar NaN -> quiet(scalar)
        var g2 = Bits(np.add(sc, arr));
        for (int i = 0; i < 40; i++)
            g2[i].Should().Be(i == 7 ? (ushort)0x7e01 : (ushort)0xff11, $"add rev lane {i}");
        // div(arr, sc): in1 first — the stream's sNaN lane keeps ITS quieted bits
        var g3 = Bits(np.divide(arr, sc));
        for (int i = 0; i < 40; i++)
            g3[i].Should().Be(i == 7 ? (ushort)0x7e01 : (ushort)0xff11, $"div lane {i}");
    }

    [TestMethod]
    public void Arith_SubnormalsExact_TailAndSimdAgree()
    {
        // subnormal + subnormal is exact in f32 and representable in f16; n = 21 runs
        // the 8-lane body + a 5-element scalar tail — both must agree with NumPy
        // (probed: 0x0001 + 0x0001 = 0x0002; 0x03ff + 0x0001 = 0x0400 crosses into normal).
        var a = new ushort[21]; var b = new ushort[21];
        for (int i = 0; i < 21; i++) { a[i] = 0x0001; b[i] = 0x0001; }
        a[20] = 0x03ff; b[20] = 0x0001;
        using var x = FromBits(a); using var y = FromBits(b);
        var g = Bits(np.add(x, y));
        for (int i = 0; i < 20; i++) g[i].Should().Be((ushort)0x0002);
        g[20].Should().Be((ushort)0x0400, "max subnormal + min subnormal crosses to normal");
    }
}
