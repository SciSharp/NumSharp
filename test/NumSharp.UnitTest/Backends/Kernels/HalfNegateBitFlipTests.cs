using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins float16 negation to the NumPy operation: a single IEEE sign-bit flip
/// (<c>bits ^ 0x8000</c>). NumSharp's f16 negate kernel used to emit
/// <c>Half.op_UnaryNegation</c>, whose BCL implementation is <c>(Half)(-(float)h)</c> — a
/// float roundtrip measured 7.3× slower than the bit flip, which made f16 negate the worst
/// cell in the whole elementwise matrix (~0.14× NumPy, while f16 abs — a sign-bit MASK —
/// was 1.6×). The kernel now flips the sign bit directly (≈1.5× NumPy, on par with abs).
///
/// The defining invariant of negate (and what makes it bit-exact with NumPy across normals,
/// ±0, ±inf and NaN-payload preservation) is: <c>bits(-x) == bits(x) ^ 0x8000</c>. These
/// tests assert exactly that, on contiguous and strided views, plus the double-negate
/// identity. (The float-roundtrip operator produced the SAME results here — only slower —
/// so this is a pure perf fix with the semantics nailed down so a future refactor can't
/// silently reintroduce a precision-changing path.)
/// </summary>
[TestClass]
public class HalfNegateBitFlipTests
{
    private static Half H(ushort bits) => BitConverter.UInt16BitsToHalf(bits);
    private static ushort Bits(Half h) => BitConverter.HalfToUInt16Bits(h);

    // normals, ±0, ±inf, a quiet NaN (payload 0x200), a signaling-ish NaN, max-finite, subnormals.
    private static readonly Half[] Samples =
    {
        H(0x0000), H(0x8000),                 // +0, -0
        H(0x3C00), H(0xBC00),                 // +1, -1
        H(0x7BFF), H(0xFBFF),                 // +max, -max finite
        H(0x7C00), H(0xFC00),                 // +inf, -inf
        H(0x7E00), H(0xFE00),                 // +qNaN, -qNaN
        H(0x0001), H(0x8001),                 // +min subnormal, -min subnormal
        H(0x3555), H(0xB555),                 // ~1/3, -1/3
    };

    [TestMethod]
    public void HalfNegate_FlipsSignBit_Contiguous()
    {
        var a = np.array(Samples);
        var n = -a;
        Assert.AreEqual(a.size, n.size);
        for (long i = 0; i < a.size; i++)
        {
            ushort inBits = Bits((Half)a.GetAtIndex(i));
            ushort outBits = Bits((Half)n.GetAtIndex(i));
            Assert.AreEqual((ushort)(inBits ^ 0x8000), outBits,
                $"negate({inBits:X4}) should be {(ushort)(inBits ^ 0x8000):X4}, got {outBits:X4}");
        }
    }

    [TestMethod]
    public void HalfNegate_FlipsSignBit_Strided()
    {
        // (4,4) reshaped, strided inner [:, ::2] and transposed — both non-contiguous.
        var grid = new Half[16];
        for (int i = 0; i < 16; i++) grid[i] = Samples[i % Samples.Length];
        var a = np.array(grid).reshape(4, 4);

        foreach (var v in new[] { a[":, ::2"], a.T, a["::-1, :"] })
        {
            var n = -v;
            for (long i = 0; i < v.size; i++)
            {
                ushort inBits = Bits((Half)v.GetAtIndex(i));
                ushort outBits = Bits((Half)n.GetAtIndex(i));
                Assert.AreEqual((ushort)(inBits ^ 0x8000), outBits,
                    $"strided negate({inBits:X4}) -> {outBits:X4}");
            }
        }
    }

    [TestMethod]
    public void HalfNegate_DoubleNegate_IsIdentity()
    {
        var a = np.array(Samples);
        var back = -(-a);
        for (long i = 0; i < a.size; i++)
            Assert.AreEqual(Bits((Half)a.GetAtIndex(i)), Bits((Half)back.GetAtIndex(i)),
                $"--x must equal x at [{i}]");
    }

    [TestMethod]
    public void HalfAbs_ClearsSignBit()
    {
        // Guard the sibling: abs is a sign-bit MASK (bits & 0x7FFF). Pins that negate and
        // abs stay distinct ops (a careless "fix" could alias them).
        var a = np.array(Samples);
        var r = np.abs(a);
        for (long i = 0; i < a.size; i++)
        {
            ushort inBits = Bits((Half)a.GetAtIndex(i));
            ushort outBits = Bits((Half)r.GetAtIndex(i));
            Assert.AreEqual((ushort)(inBits & 0x7FFF), outBits,
                $"abs({inBits:X4}) should clear the sign bit -> {(ushort)(inBits & 0x7FFF):X4}, got {outBits:X4}");
        }
    }
}
