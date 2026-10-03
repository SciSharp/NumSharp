using System;
using System.Linq;
using System.Text;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.MathTests;

/// <summary>
///     Pins WHICH NaN a float32 / float64 add or multiply returns when BOTH operands are NaN, against
///     numpy 2.4.2 win-amd64 (every expected pattern probed live).
/// </summary>
/// <remarks>
///     <para>
///     x86 addpd / mulpd return their FIRST source operand's NaN (quieted), and add / multiply are
///     commutative, so the answer is whatever operand order the compiled loop emits. NumPy's MSVC
///     build keeps the SECOND operand's NaN on every loop path except one: the AVX2 scalar-broadcast
///     SIMD loop keeps the BROADCAST operand's NaN in its blocks of two vectors (8 float64 / 16
///     float32 lanes) and the ARRAY operand's in its scalar tail, restarting per inner-loop call (per
///     where= run, per 8192-element buffer chunk of a cast operand). subtract / divide keep the first
///     operand's NaN, float16 the second everywhere.
///     </para>
///     <para>
///     NumSharp encodes the choice explicitly (<c>DirectILKernelGenerator.NaNOperandOrder.cs</c>)
///     because RyuJIT may swap a commutative operation's operands. Before the fix every path returned
///     the first operand's NaN. Linux NumPy (GCC) orders the operands differently on the same CPU, so
///     these patterns are a win-amd64 contract, like the float16 and complex NaN-sign pins.
///     </para>
///     <para>
///     Pattern strings: one character per output element in C order — <c>a</c> = the first operand's
///     NaN quieted, <c>b</c> = the second operand's, <c>0</c> = +0.0 (a masked-off out slot),
///     <c>?</c> = anything else. The oracle tier <c>nan_order.jsonl</c> gates the full layout matrix.
///     </para>
///     <para>
///     The paths NumSharp does not match yet — cumsum / cumprod, complex128 add / multiply / divide,
///     the N-D broadcasts NumPy runs per row or per buffer fill, a reversed cast operand, <c>where=</c>
///     with a cast, and deeper <c>np.evaluate</c> trees with a NaN broadcast input — are pinned with
///     NumPy's bits as <c>[OpenBugs]</c> in <c>OpenBugs.NaNOperandOrder.cs</c>.
///     </para>
/// </remarks>
[TestClass]
public class NaNOperandOrderTests
{
    /// <summary>The first operand's float64 NaN: negative, payload 1.</summary>
    private const ulong A64 = 0xfff8000000000001;

    /// <summary>The second operand's float64 NaN: positive, payload 2.</summary>
    private const ulong B64 = 0x7ff8000000000002;

    /// <summary>The first operand's float32 NaN: negative, payload 1.</summary>
    private const ulong A32 = 0xffc00001;

    /// <summary>The second operand's float32 NaN: positive, payload 2.</summary>
    private const ulong B32 = 0x7fc00002;

    /// <summary>The first operand's float16 NaN: negative, payload 1.</summary>
    private const ulong A16 = 0xfe01;

    /// <summary>The second operand's float16 NaN: positive, payload 2.</summary>
    private const ulong B16 = 0x7e02;

    /// <summary>
    ///     Builds an array whose every element holds the raw <paramref name="bits"/>.
    /// </summary>
    /// <param name="type">Element type: Half, Single or Double.</param>
    /// <param name="bits">The raw element bits, truncated to the element width.</param>
    /// <param name="shape">The dimensions; none gives a 0-d array.</param>
    /// <returns>A fresh C-contiguous array.</returns>
    /// <remarks>
    ///     Writes through a pointer, never through a float value, so a signalling NaN reaches the
    ///     array unquieted (a float load / store may quiet it).
    /// </remarks>
    private static unsafe NDArray Fill(NPTypeCode type, ulong bits, params long[] shape)
    {
        var nd = np.zeros(new Shape(shape), type);
        byte* p = (byte*)nd.Address;
        for (long i = 0; i < nd.size; i++)
        {
            switch (nd.itemsize)
            {
                case 8: ((ulong*)p)[i] = bits; break;
                case 4: ((uint*)p)[i] = (uint)bits; break;
                default: ((ushort*)p)[i] = (ushort)bits; break;
            }
        }

        return nd;
    }

    /// <summary>
    ///     Reads every element of <paramref name="r"/> in C order as raw bits, zero-extended.
    /// </summary>
    /// <param name="r">A Half, Single or Double array of any layout.</param>
    /// <returns>One value per element.</returns>
    private static unsafe ulong[] Bits(NDArray r)
    {
        // A C-contiguous copy so the pointer walk below reads logical C order for every layout.
        var c = np.ascontiguousarray(r);
        var bits = new ulong[c.size];
        byte* p = (byte*)c.Address;
        for (long i = 0; i < c.size; i++)
        {
            bits[i] = c.itemsize switch
            {
                8 => ((ulong*)p)[i],
                4 => ((uint*)p)[i],
                _ => ((ushort*)p)[i],
            };
        }

        return bits;
    }

    /// <summary>
    ///     Classifies each element of <paramref name="r"/> as <c>a</c> / <c>b</c> (the quieted NaN of
    ///     the operand that held <paramref name="a"/> / <paramref name="b"/>), <c>0</c> or <c>?</c>.
    /// </summary>
    /// <param name="r">The result array.</param>
    /// <param name="a">The raw NaN the first operand held.</param>
    /// <param name="b">The raw NaN the second operand held.</param>
    /// <returns>The pattern string, one character per element in C order.</returns>
    private static string Pattern(NDArray r, ulong a, ulong b)
    {
        // The quiet bit is the mantissa's top bit: bit 51 / 22 / 9 for float64 / float32 / float16.
        ulong quiet = r.itemsize switch { 8 => 1UL << 51, 4 => 1UL << 22, _ => 1UL << 9 };
        var sb = new StringBuilder();
        foreach (ulong v in Bits(r))
            sb.Append(v == (a | quiet) ? 'a' : v == (b | quiet) ? 'b' : v == 0 ? '0' : '?');
        return sb.ToString();
    }

    /// <summary>
    ///     Contiguous operands of every length class (scalar loop, 4x-unrolled SIMD body, vector
    ///     remainder, scalar tail) keep the second operand's NaN for add / multiply and the first
    ///     operand's for subtract / divide, at float64 and float32.
    /// </summary>
    [TestMethod]
    public void Contiguous_AddMultiplyKeepSecond_SubtractDivideKeepFirst()
    {
        foreach (var (type, a, b) in new[] { (NPTypeCode.Double, A64, B64), (NPTypeCode.Single, A32, B32) })
        {
            foreach (long n in new long[] { 1, 3, 8, 9, 17, 33, 257 })
            {
                using var x = Fill(type, a, n);
                using var y = Fill(type, b, n);
                string second = new string('b', (int)n), first = new string('a', (int)n);
                Pattern(np.add(x, y), a, b).Should().Be(second, $"{type} add n={n}");
                Pattern(np.multiply(x, y), a, b).Should().Be(second, $"{type} multiply n={n}");
                Pattern(np.subtract(x, y), a, b).Should().Be(first, $"{type} subtract n={n}");
                Pattern(np.divide(x, y), a, b).Should().Be(first, $"{type} divide n={n}");
                Pattern(x + y, a, b).Should().Be(second, $"{type} operator+ n={n}");
                Pattern(x * y, a, b).Should().Be(second, $"{type} operator* n={n}");
            }
        }
    }

    /// <summary>
    ///     A 0-d broadcast against a float64 array follows the region rule: blocks of 8 lanes keep the
    ///     broadcast operand's NaN, the tail keeps the array operand's, and an array of 8 elements or
    ///     fewer (no SIMD loop) keeps the second operand's throughout.
    /// </summary>
    [TestMethod]
    public void ScalarBroadcast_Float64_RegionRule()
    {
        // (n, left-broadcast pattern, right-broadcast pattern), probed numpy 2.4.2 win-amd64.
        var probed = new (long n, string left, string right)[]
        {
            (7, "bbbbbbb", "bbbbbbb"),
            (8, "bbbbbbbb", "bbbbbbbb"),
            (9, "aaaaaaaab", "bbbbbbbba"),
            (15, "aaaaaaaabbbbbbb", "bbbbbbbbaaaaaaa"),
            (16, "aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
            (17, "aaaaaaaaaaaaaaaab", "bbbbbbbbbbbbbbbba"),
            (25, "aaaaaaaaaaaaaaaaaaaaaaaab", "bbbbbbbbbbbbbbbbbbbbbbbba"),
            (33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"),
        };
        foreach (var (n, left, right) in probed)
        {
            using var a0 = Fill(NPTypeCode.Double, A64);
            using var b0 = Fill(NPTypeCode.Double, B64);
            using var x = Fill(NPTypeCode.Double, A64, n);
            using var y = Fill(NPTypeCode.Double, B64, n);
            Pattern(np.add(a0, y), A64, B64).Should().Be(left, $"add 0-d + ({n},)");
            Pattern(np.multiply(a0, y), A64, B64).Should().Be(left, $"multiply 0-d * ({n},)");
            Pattern(np.add(x, b0), A64, B64).Should().Be(right, $"add ({n},) + 0-d");
            Pattern(np.multiply(x, b0), A64, B64).Should().Be(right, $"multiply ({n},) * 0-d");
        }
    }

    /// <summary>
    ///     float32 runs the same region rule with blocks of 16 lanes, so lengths 9-16 still keep the
    ///     second operand's NaN throughout.
    /// </summary>
    [TestMethod]
    public void ScalarBroadcast_Float32_RegionRuleBlocksOf16()
    {
        var probed = new (long n, string left, string right)[]
        {
            (9, "bbbbbbbbb", "bbbbbbbbb"),
            (16, "bbbbbbbbbbbbbbbb", "bbbbbbbbbbbbbbbb"),
            (17, "aaaaaaaaaaaaaaaab", "bbbbbbbbbbbbbbbba"),
            (25, "aaaaaaaaaaaaaaaabbbbbbbbb", "bbbbbbbbbbbbbbbbaaaaaaaaa"),
            (33, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbba"),
        };
        foreach (var (n, left, right) in probed)
        {
            using var a0 = Fill(NPTypeCode.Single, A32);
            using var b0 = Fill(NPTypeCode.Single, B32);
            using var x = Fill(NPTypeCode.Single, A32, n);
            using var y = Fill(NPTypeCode.Single, B32, n);
            Pattern(np.add(a0, y), A32, B32).Should().Be(left, $"add 0-d + ({n},)");
            Pattern(np.multiply(x, b0), A32, B32).Should().Be(right, $"multiply ({n},) * 0-d");
        }
    }

    /// <summary>
    ///     A size-1 1-D operand broadcasts with stride 0 exactly like a 0-d one, so it takes the same
    ///     region rule.
    /// </summary>
    [TestMethod]
    public void SizeOneArrayBroadcast_SameRuleAsZeroD()
    {
        using var a1 = Fill(NPTypeCode.Double, A64, 1);
        using var b1 = Fill(NPTypeCode.Double, B64, 1);
        using var x = Fill(NPTypeCode.Double, A64, 17);
        using var y = Fill(NPTypeCode.Double, B64, 17);
        Pattern(np.add(a1, y), A64, B64).Should().Be("aaaaaaaaaaaaaaaab");
        Pattern(np.multiply(x, b1), A64, B64).Should().Be("bbbbbbbbbbbbbbbba");
    }

    /// <summary>
    ///     Strided, reversed and mixed-stride pairs leave the SIMD loops: the second operand's NaN
    ///     throughout, including a 0-d against a strided array (the broadcast SIMD loop needs a
    ///     contiguous array operand).
    /// </summary>
    [TestMethod]
    public void StridedAndReversed_KeepSecond()
    {
        foreach (long n in new long[] { 9, 33 })
        {
            string second = new string('b', (int)n);
            using var x2 = Fill(NPTypeCode.Double, A64, 2 * n);
            using var y2 = Fill(NPTypeCode.Double, B64, 2 * n);
            using var x = Fill(NPTypeCode.Double, A64, n);
            using var y = Fill(NPTypeCode.Double, B64, n);
            using var b0 = Fill(NPTypeCode.Double, B64);
            Pattern(np.add(x2["::2"], y2["::2"]), A64, B64).Should().Be(second, $"strided n={n}");
            Pattern(np.multiply(x["::-1"], y["::-1"]), A64, B64).Should().Be(second, $"reversed n={n}");
            Pattern(np.add(x2["::2"], b0), A64, B64).Should().Be(second, $"strided + 0-d n={n}");
        }
    }

    /// <summary>
    ///     Under where=, NumPy calls the inner loop once per run of True mask elements, so the region
    ///     rule restarts at every run: a run of 12 splits 8 + 4, runs of 1 and 4 never reach the SIMD
    ///     loop, a run of 20 splits 16 + 4; masked-off slots keep the out array's prior zeros.
    /// </summary>
    [TestMethod]
    public void WhereRuns_RegionRuleRestartsPerRun()
    {
        var runs = new bool[40];
        foreach (var (start, stop) in new[] { (0, 12), (13, 14), (15, 35), (36, 40) })
            for (int i = start; i < stop; i++)
                runs[i] = true;
        using var mask = np.array(runs);

        using var a0 = Fill(NPTypeCode.Double, A64);
        using var b0 = Fill(NPTypeCode.Double, B64);
        using var x = Fill(NPTypeCode.Double, A64, 40);
        using var y = Fill(NPTypeCode.Double, B64, 40);

        var outLeft = np.zeros(new Shape(40), NPTypeCode.Double);
        np.add(a0, y, outLeft, mask);
        Pattern(outLeft, A64, B64).Should().Be("aaaaaaaabbbb0b0aaaaaaaaaaaaaaaabbbb0bbbb");

        var outRight = np.zeros(new Shape(40), NPTypeCode.Double);
        np.multiply(x, b0, outRight, mask);
        Pattern(outRight, A64, B64).Should().Be("bbbbbbbbaaaa0b0bbbbbbbbbbbbbbbbaaaa0bbbb");
    }

    /// <summary>
    ///     A float32 array against a float64 size-1 operand is cast in buffer chunks of
    ///     <c>np.getbufsize()</c> = 8192 elements, one inner-loop call each, so the region rule
    ///     restarts per chunk: 8203 elements split 8192 (all blocks) + 11 (8 + 3).
    /// </summary>
    [TestMethod]
    public void CastOperand_RegionRuleRestartsPerBufferChunk()
    {
        // a's float32 NaN 0xffc00001 widened to float64 is 0xfff8000020000000.
        const ulong widenedA = 0xfff8000020000000;
        const ulong quietB = 0x7ff8000000000002;

        using var small = np.add(Fill(NPTypeCode.Single, A32, 17), Fill(NPTypeCode.Double, B64, 1));
        var bits = Bits(small);
        bits.Take(16).Should().AllBeEquivalentTo(quietB, "the 16-lane block keeps the broadcast operand's NaN");
        bits[16].Should().Be(widenedA, "the tail keeps the cast array operand's NaN");

        using var large = np.add(Fill(NPTypeCode.Single, A32, 8203), Fill(NPTypeCode.Double, B64, 1));
        bits = Bits(large);
        bits.Take(8200).Should().AllBeEquivalentTo(quietB, "chunk 1 (8192) and chunk 2's 8-lane block");
        bits.Skip(8200).Should().AllBeEquivalentTo(widenedA, "chunk 2's 3-element tail");
    }

    /// <summary>
    ///     A signalling NaN is quieted with its payload kept, and the winner follows the same rule as
    ///     for quiet NaNs.
    /// </summary>
    [TestMethod]
    public void SignallingNaN_QuietedWinnerFollowsTheRule()
    {
        const ulong sa = 0xfff0000000000011, sb = 0x7ff0000000000022;
        using var x = Fill(NPTypeCode.Double, sa, 17);
        using var y = Fill(NPTypeCode.Double, sb, 17);
        Bits(np.add(x, y)).Should().AllBeEquivalentTo(0x7ff8000000000022UL);

        using var a0 = Fill(NPTypeCode.Double, sa);
        var bits = Bits(np.multiply(a0, y));
        bits.Take(16).Should().AllBeEquivalentTo(0xfff8000000000011UL);
        bits[16].Should().Be(0x7ff8000000000022UL);
    }

    /// <summary>
    ///     out= aliasing an input does not change the rule: in-place contiguous keeps the second
    ///     operand's NaN, an in-place broadcast still splits at the block boundary.
    /// </summary>
    [TestMethod]
    public void OutAliasingAnInput_SameRule()
    {
        using var x = Fill(NPTypeCode.Double, A64, 17);
        using var y = Fill(NPTypeCode.Double, B64, 17);
        np.add(x, y, x);
        Pattern(x, A64, B64).Should().Be("bbbbbbbbbbbbbbbbb");

        using var z = Fill(NPTypeCode.Double, A64, 17);
        using var b0 = Fill(NPTypeCode.Double, B64);
        np.multiply(z, b0, z);
        Pattern(z, A64, B64).Should().Be("bbbbbbbbbbbbbbbba");
    }

    /// <summary>
    ///     0-d against 0-d runs one scalar call: the second operand's NaN.
    /// </summary>
    [TestMethod]
    public void ZeroDBoth_KeepSecond()
    {
        using var a0 = Fill(NPTypeCode.Double, A64);
        using var b0 = Fill(NPTypeCode.Double, B64);
        Pattern(np.add(a0, b0), A64, B64).Should().Be("b");
        Pattern(np.multiply(a0, b0), A64, B64).Should().Be("b");
    }

    /// <summary>
    ///     float16 has no SIMD broadcast loop in NumPy: the second operand's NaN everywhere, also past
    ///     the float32 / float64 block lengths.
    /// </summary>
    [TestMethod]
    public void Float16_KeepSecondEverywhere()
    {
        foreach (long n in new long[] { 9, 17, 33 })
        {
            using var a0 = Fill(NPTypeCode.Half, A16);
            using var b0 = Fill(NPTypeCode.Half, B16);
            using var x = Fill(NPTypeCode.Half, A16, n);
            using var y = Fill(NPTypeCode.Half, B16, n);
            string second = new string('b', (int)n);
            Pattern(np.add(a0, y), A16, B16).Should().Be(second, $"add 0-d + ({n},)");
            Pattern(np.multiply(x, b0), A16, B16).Should().Be(second, $"multiply ({n},) * 0-d");
        }
    }

    /// <summary>
    ///     np.evaluate's fused kernel keeps NumPy's per-node order: array-array nodes keep the second
    ///     operand's NaN, and a single node with a 0-d input runs the direct broadcast kernel and its
    ///     region rule.
    /// </summary>
    [TestMethod]
    public void Evaluate_FusedTreesFollowThePerNodeRule()
    {
        using var x = Fill(NPTypeCode.Double, A64, 257);
        using var y = Fill(NPTypeCode.Double, B64, 257);
        string second = new string('b', 257);
        Pattern(np.evaluate((NDExpr)x + y), A64, B64).Should().Be(second, "a + b");
        Pattern(np.evaluate((NDExpr)x * y), A64, B64).Should().Be(second, "a * b");
        Pattern(np.evaluate((NDExpr)x * 1.0 + y), A64, B64).Should().Be(second, "(a * 1) + b");
        Pattern(np.evaluate(((NDExpr)x + y) * y), A64, B64).Should().Be(second, "(a + b) * b");

        using var z = Fill(NPTypeCode.Double, A64, 17);
        using var k = Fill(NPTypeCode.Double, B64);
        Pattern(np.evaluate((NDExpr)z + k), A64, B64).Should().Be("bbbbbbbbbbbbbbbba", "a + 0-d k");
    }

    /// <summary>
    ///     The fuzz finding that started this fix: <c>max(lxor(isnan(b), a*0.0), (a+m)*(a+b))</c> over
    ///     257 float64 elements, with a = -NaN and b = +NaN at index 14. NumPy returns +NaN there (the
    ///     product keeps (a+b)'s NaN, which is b's); NumSharp returned -NaN. The whole result now
    ///     hashes like NumPy's.
    /// </summary>
    [TestMethod]
    public unsafe void Evaluate_FuzzFinding_IsByteIdentical()
    {
        using var a = np.linspace(-3.0, 3.0, 257);
        using var b = np.linspace(2.0, -2.0, 257);
        using var m = np.linspace(0.5, 1.5, 257);
        ((ulong*)a.Address)[14] = 0xfff8000000000000;
        ((ulong*)b.Address)[14] = 0x7ff8000000000000;

        NDExpr ea = a, eb = b, em = m;
        using var r = np.evaluate(NDExpr.Maximum(NDExpr.LogicalXor(NDExpr.IsNaN(eb), ea * 0.0), (ea + em) * (ea + eb)));

        Bits(r)[14].Should().Be(0x7ff8000000000000UL);
        var bytes = new byte[r.size * 8];
        System.Runtime.InteropServices.Marshal.Copy((IntPtr)r.Address, bytes, 0, bytes.Length);
        Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(bytes)).ToLowerInvariant()
            .Should().Be("1863489c534a07433247ecc828d2ccc46fa03a069cc65cd802266f68d3b78b58",
                "SHA-256 of numpy 2.4.2's result bytes for the same inputs");
    }
}
