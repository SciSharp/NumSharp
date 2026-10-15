using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
///     Pins the AVX2 widening cast kernels added for the numpy.polynomial audit: {int8, uint8, int16, uint16, char,
///     int32, uint32} → {float32, float64} and {int64, uint64} → float64 (<c>Cast.IntToFloat.cs</c>), and float16 →
///     float64 (<c>Cast.Half.cs</c>). Every one must be BIT-EXACT with the scalar conversion NumPy's cast loop
///     performs: exact for the 8/16-bit integers and for int32/uint32 → float64, ONE round-to-nearest-even rounding
///     for int32/uint32 → float32 and for int64/uint64 → float64 (the exponent-splice kernels), and float16 → float64
///     by NumPy's own <c>ToDoubleBits</c> formula — which leaves a signalling NaN signalling. Lengths are chosen so
///     the 8-lane vector bodies AND their scalar remainders both run.
/// </summary>
[TestClass]
public class IntToFloatCastParityTests
{
    /// <summary>Every 8/16-bit value (plus a remainder) converts exactly: the kernel must equal the int64 route.</summary>
    [DataTestMethod]
    [DataRow(NPTypeCode.SByte), DataRow(NPTypeCode.Byte), DataRow(NPTypeCode.Int16), DataRow(NPTypeCode.UInt16), DataRow(NPTypeCode.Char)]
    public void NarrowIntegers_ConvertExactly_ToBothFloatWidths(NPTypeCode src)
    {
        int n = src is NPTypeCode.SByte or NPTypeCode.Byte ? 256 + 13 : 65536 + 13;
        var values = np.arange(n).astype(src);   // wraps around: every value of the dtype at least once
        foreach (var dst in new[] { NPTypeCode.Single, NPTypeCode.Double })
        {
            var got = values.astype(dst);
            var want = values.astype(NPTypeCode.Int64).astype(dst);   // int64 -> float is exact for these ranges
            np.array_equal(got, want).Should().BeTrue($"{src} -> {dst}");
        }
    }

    /// <summary>
    ///     int32 / uint32 → float32 round ONCE (NumPy: <c>np.array([...], dtype).astype(np.float32)</c>): ties to even at
    ///     2^24 + 1, 2^31 - 1 rounding up to 2^31, and uint32 values past 2^31 — where a biased float32 sum would round
    ///     twice — land on the correctly rounded float.
    /// </summary>
    [TestMethod]
    public void Int32AndUInt32_ToFloat32_RoundOnce_LikeNumPy()
    {
        // NumPy 2.4.2: np.array([16777217, 16777219, 2147483647, -2147483648, -16777217, 33554433], np.int32).astype(np.float32)
        var i32 = np.array(new[] { 16777217, 16777219, int.MaxValue, int.MinValue, -16777217, 33554433 });
        float[] i32Want = { 16777216f, 16777220f, 2147483648f, -2147483648f, -16777216f, 33554432f };
        // np.array([4294967295, 2147483648, 2147483649, 4294967167, 4294967168, 16777217, 0], np.uint32).astype(np.float32)
        var u32 = np.array(new uint[] { uint.MaxValue, 0x80000000u, 0x80000001u, 4294967167u, 4294967168u, 16777217u, 0u });
        float[] u32Want = { 4294967296f, 2147483648f, 2147483648f, 4294967040f, 4294967296f, 16777216f, 0f };
        // Repeat each list past one 8-lane vector so the SIMD body (not only the scalar remainder) converts it.
        AssertFloat32(Tile(i32, 5).astype(NPTypeCode.Single), i32Want, "int32");
        AssertFloat32(Tile(u32, 5).astype(NPTypeCode.Single), u32Want, "uint32");
    }

    /// <summary>
    ///     Every uint32 bit pattern stepped through one chunk of the kernels matches the scalar conversions (the full
    ///     2^32 sweep ran clean when the kernels were written; this keeps a strided sample in CI).
    /// </summary>
    [TestMethod]
    public unsafe void Int32AndUInt32_SampledAcrossTheRange_EqualScalarConversions()
    {
        const int n = 1 << 16;
        var bits = new uint[n];
        var rnd = new System.Random(20260927);
        for (int k = 0; k < n; k++)
            bits[k] = k < 64 ? (uint)(k * 0x04000000u + (k & 1)) : (uint)rnd.NextInt64(0, 1L << 32);
        var u = np.array(bits);
        var i = u.view(np.int32);
        var u2f = u.astype(NPTypeCode.Single); var u2d = u.astype(NPTypeCode.Double);
        var i2f = i.astype(NPTypeCode.Single); var i2d = i.astype(NPTypeCode.Double);
        for (int k = 0; k < n; k++)
        {
            uint b = bits[k];
            BitConverter.SingleToInt32Bits(u2f.GetSingle(k)).Should().Be(BitConverter.SingleToInt32Bits((float)(double)b), $"u32->f32 {b:x8}");
            u2d.GetDouble(k).Should().Be((double)b, $"u32->f64 {b:x8}");
            BitConverter.SingleToInt32Bits(i2f.GetSingle(k)).Should().Be(BitConverter.SingleToInt32Bits((float)(int)b), $"i32->f32 {b:x8}");
            i2d.GetDouble(k).Should().Be((double)(int)b, $"i32->f64 {b:x8}");
        }
    }

    /// <summary>
    ///     int64 / uint64 → float64 round ONCE, to nearest-even (NumPy 2.4.2 <c>astype(np.float64)</c>, probed bit
    ///     patterns): the ties at 2^53 + 1 / 2^53 + 3 go to even, 2^63 + 1025 rounds UP to 2^63 + 2048 (a
    ///     convert-as-signed-then-add-2^64 route rounds twice and lands on 2^63), 2^64 - 1 reaches 2^64, and the int64
    ///     splice's 48-bit seam (±2^48, ±2^48 - 1) and extremes land exactly where the scalar cvtsi2sd puts them.
    /// </summary>
    [TestMethod]
    public void Int64AndUInt64_ToFloat64_RoundOnce_LikeNumPy()
    {
        // np.array([...], np.uint64).astype(np.float64).view(np.uint64)
        var u64 = np.array(new ulong[]
        {
            (1UL << 63) + 1025, ulong.MaxValue, (1UL << 53) + 1, (1UL << 53) + 3, (1UL << 63) + 1024, (1UL << 63) + 3072,
            0, 1, uint.MaxValue, 1UL << 32, (1UL << 52) + 1, 18446744073709549568UL, 18446744073709550591UL,
        });
        ulong[] u64Want =
        {
            0x43e0000000000001, 0x43f0000000000000, 0x4340000000000000, 0x4340000000000002, 0x43e0000000000000,
            0x43e0000000000002, 0x0, 0x3ff0000000000000, 0x41efffffffe00000, 0x41f0000000000000, 0x4330000000000001,
            0x43efffffffffffff, 0x43efffffffffffff,
        };
        // np.array([...], np.int64).astype(np.float64).view(np.uint64)
        var i64 = np.array(new long[]
        {
            -((1L << 53) + 1), -((1L << 53) + 3), long.MinValue, long.MaxValue, (1L << 53) + 1, -1, 0, -(1L << 48),
            (1L << 48) - 1, -(1L << 48) - 1, (1L << 62) + 513, -((1L << 62) + 513), -((1L << 62) + 512 + 1024),
        });
        ulong[] i64Want =
        {
            0xc340000000000000, 0xc340000000000002, 0xc3e0000000000000, 0x43e0000000000000, 0x4340000000000000,
            0xbff0000000000000, 0x0, 0xc2f0000000000000, 0x42efffffffffffe0, 0xc2f0000000000010, 0x43d0000000000001,
            0xc3d0000000000001, 0xc3d0000000000002,
        };
        AssertFloat64Bits(Tile(u64, 3).astype(NPTypeCode.Double), u64Want, "uint64");
        AssertFloat64Bits(Tile(i64, 3).astype(NPTypeCode.Double), i64Want, "int64");
    }

    /// <summary>
    ///     Random int64 / uint64 values at every magnitude (a random bit length 0–64, so the rounding region past 2^53
    ///     and the exact region below it are both dense) convert exactly as the scalar conversions do: the C# cast for
    ///     int64 (cvtsi2sd) and <c>Converts.ToDouble(ulong)</c> for uint64 — both round-once, like NumPy.
    /// </summary>
    [TestMethod]
    public void Int64AndUInt64_SampledAcrossMagnitudes_EqualScalarConversions()
    {
        const int n = 1 << 16;
        var bits = new ulong[n];
        var rnd = new System.Random(20260927);
        for (int k = 0; k < n; k++)
        {
            int width = rnd.Next(0, 65);
            ulong v = (ulong)rnd.NextInt64() ^ ((ulong)rnd.NextInt64() << 1);
            bits[k] = width == 64 ? v : v & ((1UL << width) - 1);
            if ((k & 7) == 3) bits[k] |= 1UL << 63;      // plenty of uint64 >= 2^63 / negative int64
        }
        var u = np.array(bits);
        var i = u.view(np.int64);
        var u2d = u.astype(NPTypeCode.Double);
        var i2d = i.astype(NPTypeCode.Double);
        for (int k = 0; k < n; k++)
        {
            ulong b = bits[k];
            BitConverter.DoubleToInt64Bits(u2d.GetDouble(k)).Should().Be(BitConverter.DoubleToInt64Bits(NumSharp.Utilities.Converts.ToDouble(b)), $"u64->f64 {b:x16}");
            BitConverter.DoubleToInt64Bits(i2d.GetDouble(k)).Should().Be(BitConverter.DoubleToInt64Bits((double)(long)b), $"i64->f64 {b:x16}");
        }
    }

    /// <summary>
    ///     float16 → float64 over all 65,536 bit patterns (contiguous, reversed and stride-3) equals NumPy's
    ///     <c>ToDoubleBits</c>: a finite value exactly, inf/NaN as <c>sign | 0x7ff0… | mantissa &lt;&lt; 42</c> — so
    ///     the signalling NaN 0x7c01 becomes the signalling 0x7ff0040000000000 (NumPy 2.4.2, probed), not the quieted
    ///     0x7ff8040000000000 a <c>cvtps2pd</c> or the BCL's <c>(double)Half</c> would give.
    /// </summary>
    [TestMethod]
    public void Half_ToFloat64_EveryPattern_MatchesNumPyBits()
    {
        var raw = new ushort[65536];
        for (int k = 0; k < raw.Length; k++) raw[k] = (ushort)k;
        var h = np.array(raw).view(np.float16);
        foreach (var (name, view) in new[] { ("c", h), ("r", h["::-1"]), ("s3", h["::3"]) })
        {
            var got = view.astype(NPTypeCode.Double);
            var src = view.copy().view(np.uint16);
            for (long k = 0; k < got.size; k++)
            {
                ushort b = src.GetUInt16((int)k);
                ulong want = DirectILKernelGenerator.HalfToDoubleBitsExact(b);
                ((ulong)BitConverter.DoubleToInt64Bits(got.GetDouble((int)k))).Should().Be(want, $"{name}: f16 {b:x4}");
            }
        }
        ((ulong)BitConverter.DoubleToInt64Bits(np.array(new ushort[] { 0x7c01 }).view(np.float16).astype(NPTypeCode.Double).GetDouble(0)))
            .Should().Be(0x7ff0040000000000UL, "NumPy keeps a float16 sNaN signalling in float64");
    }

    /// <summary>The kernels are what <see cref="DirectILKernelGenerator.TryGetCastKernel"/> hands out on an AVX2 host.</summary>
    [TestMethod]
    public void TheHouseCast_ResolvesToTheVectorKernels()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            Assert.Inconclusive("AVX2 kernels are not available on this host");
        foreach (var src in new[] { NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char, NPTypeCode.Int32, NPTypeCode.UInt32 })
            foreach (var dst in new[] { NPTypeCode.Single, NPTypeCode.Double })
            {
                DirectILKernelGenerator.TryGetIntToFloatKernel(src, dst).Should().NotBeNull($"{src} -> {dst}");
                DirectILKernelGenerator.IsVectorizedContiguousCast(src, dst).Should().BeTrue($"{src} -> {dst}");
            }
        foreach (var src in new[] { NPTypeCode.Int64, NPTypeCode.UInt64 })
        {
            DirectILKernelGenerator.TryGetIntToFloatKernel(src, NPTypeCode.Double).Should().NotBeNull($"{src} -> Double (exponent splice)");
            DirectILKernelGenerator.IsVectorizedContiguousCast(src, NPTypeCode.Double).Should().BeTrue($"{src} -> Double");
            DirectILKernelGenerator.TryGetIntToFloatKernel(src, NPTypeCode.Single).Should().BeNull($"{src} -> Single would round twice through float64");
            DirectILKernelGenerator.IsVectorizedContiguousCast(src, NPTypeCode.Single).Should().BeFalse($"{src} -> Single");
        }
    }

    /// <summary><paramref name="a"/> repeated <paramref name="times"/> times (a 1-D tile).</summary>
    /// <param name="a">The 1-D array.</param>
    /// <param name="times">Repetitions.</param>
    /// <returns>The tiled array.</returns>
    private static NDArray Tile(NDArray a, int times) => np.tile(a, times);

    /// <summary>Asserts a float64 array holds the bit patterns <paramref name="want"/> repeated.</summary>
    /// <param name="got">The converted array (a tiling of the want list).</param>
    /// <param name="want">One period of the expected bit patterns.</param>
    /// <param name="what">Label for the failure message.</param>
    private static void AssertFloat64Bits(NDArray got, ulong[] want, string what)
    {
        for (int k = 0; k < got.size; k++)
            ((ulong)BitConverter.DoubleToInt64Bits(got.GetDouble(k))).Should().Be(want[k % want.Length], $"{what}[{k}]");
    }

    /// <summary>Asserts a float32 array holds <paramref name="want"/> repeated, bit for bit.</summary>
    /// <param name="got">The converted array (a tiling of the want list).</param>
    /// <param name="want">One period of the expected values.</param>
    /// <param name="what">Label for the failure message.</param>
    private static void AssertFloat32(NDArray got, float[] want, string what)
    {
        for (int k = 0; k < got.size; k++)
            BitConverter.SingleToInt32Bits(got.GetSingle(k)).Should().Be(BitConverter.SingleToInt32Bits(want[k % want.Length]), $"{what}[{k}]");
    }
}
