using System;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins the float16 elementwise maximum/minimum/fmax/fmin bit-level kernels
/// (DirectILKernelGenerator.Binary.MinMax.Half.cs) against NumPy 2.4.2's HALF loops
/// `(ge/le(in1,in2) || isnan(in1|in2)) ? in1 : in2` where ge/le are the NaN-GUARDED
/// npy_half comparators (halffloat.cpp: npy_half_le → Half::operator&lt;=, NOT the
/// _nonan raw compare). Every expected bit pattern probed live on 2.4.2.
/// The reasons these exist:
///   • maximum/minimum return the NaN OPERAND VERBATIM (in1 preferred when both NaN) —
///     the old scalar helper returned canonical Half.NaN, losing payload and sign;
///   • fmax/fmin ignore NaN (non-NaN operand wins; both-NaN → in1);
///   • ±0 ties return in1 for all four ops;
///   • the scalar-broadcast paths (arr∘scalar and scalar∘arr) must agree with the
///     streamed path — operand PRIORITY is not symmetric, so left-scalar is its own core.
/// </summary>
[TestClass]
public class HalfPairMinMaxKernelTests
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

    // probed 2.4.2: columns are (in1, in2) -> maximum, minimum, fmax, fmin
    private static readonly (ushort a, ushort b, ushort max, ushort min, ushort fmax, ushort fmin)[] ProbedPairs =
    {
        (0x8000, 0x0000, 0x8000, 0x8000, 0x8000, 0x8000), // (-0,+0): all four return in1
        (0x0000, 0x8000, 0x0000, 0x0000, 0x0000, 0x0000), // (+0,-0): all four return in1
        (0xbc00, 0x8000, 0x8000, 0xbc00, 0x8000, 0xbc00), // (-1,-0)
        (0x7e01, 0x3c00, 0x7e01, 0x7e01, 0x3c00, 0x3c00), // (+NaN,1): max/min propagate VERBATIM, f* ignore
        (0x3c00, 0xfd65, 0xfd65, 0xfd65, 0x3c00, 0x3c00), // (1,-NaN): NaN operand verbatim incl. SIGN
        (0xfe22, 0x7e01, 0xfe22, 0xfe22, 0xfe22, 0xfe22), // (-NaN,+NaN): both NaN -> in1 for all four
        (0x7d05, 0x7e01, 0x7d05, 0x7d05, 0x7d05, 0x7d05), // (nanB,nanA): in1 preferred
        (0x7c00, 0xfc00, 0x7c00, 0xfc00, 0x7c00, 0xfc00), // (+inf,-inf)
        (0x0001, 0x8001, 0x0001, 0x8001, 0x0001, 0x8001), // subnormals order correctly
    };

    [TestMethod]
    public void PairOps_ProbedMatrix_ArrayArray()
    {
        var a = FromBits(Array.ConvertAll(ProbedPairs, p => p.a));
        var b = FromBits(Array.ConvertAll(ProbedPairs, p => p.b));
        using var mx = np.maximum(a, b);
        using var mn = np.minimum(a, b);
        using var fx = np.fmax(a, b);
        using var fn = np.fmin(a, b);
        var gmx = Bits(mx); var gmn = Bits(mn); var gfx = Bits(fx); var gfn = Bits(fn);
        for (int i = 0; i < ProbedPairs.Length; i++)
        {
            gmx[i].Should().Be(ProbedPairs[i].max, $"maximum pair {i} (a={ProbedPairs[i].a:x4}, b={ProbedPairs[i].b:x4})");
            gmn[i].Should().Be(ProbedPairs[i].min, $"minimum pair {i} (a={ProbedPairs[i].a:x4}, b={ProbedPairs[i].b:x4})");
            gfx[i].Should().Be(ProbedPairs[i].fmax, $"fmax pair {i} (a={ProbedPairs[i].a:x4}, b={ProbedPairs[i].b:x4})");
            gfn[i].Should().Be(ProbedPairs[i].fmin, $"fmin pair {i} (a={ProbedPairs[i].a:x4}, b={ProbedPairs[i].b:x4})");
        }
        a.Dispose(); b.Dispose();
    }

    [TestMethod]
    public void PairOps_ScalarBroadcast_BothSides_MatchStreamed()
    {
        // ~70 elems so the 16-lane loop + tail both run; scalar = -NaN (the quirkiest operand).
        var bits = new ushort[70];
        for (int i = 0; i < 70; i++) bits[i] = (ushort)(0x3c00 + i);
        bits[3] = 0x8000; bits[40] = 0x7e33; // a -0 and a +NaN in the stream
        using var arr = FromBits(bits);
        using var sc = FromBits(0xfd65);
        sc.Storage.Reshape(Shape.Scalar);

        // arr ∘ scalar (ScalarRight) vs scalar ∘ arr (ScalarLeft) — NaN preference flips sides.
        using var r1 = np.maximum(arr, sc);
        using var r2 = np.maximum(sc, arr);
        var g1 = Bits(r1); var g2 = Bits(r2);
        for (int i = 0; i < 70; i++)
        {
            bool streamNaN = (bits[i] & 0x7fff) > 0x7c00;
            // maximum(finite, -NaN) -> -NaN (in2 NaN propagates); maximum(+NaN, -NaN) -> in1 (+NaN)
            g1[i].Should().Be(streamNaN ? bits[i] : (ushort)0xfd65, $"maximum(arr[{i}], -nan)");
            // maximum(-NaN, anything) -> in1 (-NaN) always
            g2[i].Should().Be((ushort)0xfd65, $"maximum(-nan, arr[{i}])");
        }

        using var f1 = np.fmax(arr, sc);
        using var f2 = np.fmax(sc, arr);
        var h1 = Bits(f1); var h2 = Bits(f2);
        for (int i = 0; i < 70; i++)
        {
            bool streamNaN = (bits[i] & 0x7fff) > 0x7c00;
            // fmax ignores the NaN scalar: arr value wins unless it is itself NaN (both-NaN -> in1)
            h1[i].Should().Be(bits[i], $"fmax(arr[{i}], -nan)");
            h2[i].Should().Be(streamNaN ? (ushort)0xfd65 : bits[i], $"fmax(-nan, arr[{i}])");
        }
    }

    [TestMethod]
    public void Clip_Half_NaNPayloadPreserved()
    {
        // np.clip rides the same scalar clamp helpers; a NaN input must survive VERBATIM
        // (probed 2.4.2: clip(nan(0x7e05), 0, 1) = 0x7e05) — the old helper returned
        // canonical Half.NaN and destroyed the payload.
        using var x = FromBits(0x7e05, 0x4500, 0xfe22, 0x3800);
        using var c = np.clip(x, (Half)0f, (Half)1f);
        var g = Bits(c);
        g[0].Should().Be((ushort)0x7e05, "NaN payload must survive clip");
        g[1].Should().Be((ushort)0x3c00, "5.0 clips to hi=1.0");
        g[2].Should().Be((ushort)0xfe22, "negative NaN survives with sign");
        g[3].Should().Be((ushort)0x3800, "0.5 is inside the window");
    }
}
