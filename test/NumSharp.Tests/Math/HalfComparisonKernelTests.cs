using System;

namespace NumSharp.Tests.MathTests;

/// <summary>
/// Pins the float16 comparison bit-level kernels (DirectILKernelGenerator.Comparison.Half.cs)
/// against NumPy 2.4.2's HALF comparison loops (the NaN-GUARDED npy_half_eq/ne/lt/le/gt/ge).
/// The seams under test: ±0 compares EQUAL in every ordered op (the key map orders -0 &lt; +0,
/// so bothZero repair is load-bearing), NaN makes every comparison False except != (True),
/// and two NaNs with IDENTICAL bits are still != (probed live).
/// </summary>
[TestClass]
public class HalfComparisonKernelTests
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

    private static unsafe byte[] Bools(NDArray nd)
    {
        var r = new byte[nd.size];
        byte* p = (byte*)nd.Address;
        for (int i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    [TestMethod]
    public void Compare_SignedZeros_EqualInAllOrderedOps()
    {
        // probed 2.4.2: less(-0,+0)=F equal(-0,+0)=T le(-0,+0)=T ge(-0,+0)=T
        using var a = FromBits(0x8000, 0x0000);
        using var b = FromBits(0x0000, 0x8000);
        Bools(np.equal(a, b)).Should().Equal(new byte[] { 1, 1 });
        Bools(np.not_equal(a, b)).Should().Equal(new byte[] { 0, 0 });
        Bools(np.less(a, b)).Should().Equal(new byte[] { 0, 0 });
        Bools(np.less_equal(a, b)).Should().Equal(new byte[] { 1, 1 });
        Bools(np.greater(a, b)).Should().Equal(new byte[] { 0, 0 });
        Bools(np.greater_equal(a, b)).Should().Equal(new byte[] { 1, 1 });
    }

    [TestMethod]
    public void Compare_NaN_FalseExceptNotEqual_EvenSameBits()
    {
        // (nan, 1), (1, -nan), (nan, nan-same-bits)
        using var a = FromBits(0x7e01, 0x3c00, 0x7e01);
        using var b = FromBits(0x3c00, 0xfd65, 0x7e01);
        Bools(np.equal(a, b)).Should().Equal(new byte[] { 0, 0, 0 });
        Bools(np.not_equal(a, b)).Should().Equal(new byte[] { 1, 1, 1 });
        Bools(np.less(a, b)).Should().Equal(new byte[] { 0, 0, 0 });
        Bools(np.less_equal(a, b)).Should().Equal(new byte[] { 0, 0, 0 });
        Bools(np.greater(a, b)).Should().Equal(new byte[] { 0, 0, 0 });
        Bools(np.greater_equal(a, b)).Should().Equal(new byte[] { 0, 0, 0 });
    }

    [TestMethod]
    public void Compare_Ordering_InfSubnormalsNegatives()
    {
        // pairs: (-inf,+inf) (min-sub, -min-sub) (-1, -2) (+1, +2)
        using var a = FromBits(0xfc00, 0x0001, 0xbc00, 0x3c00);
        using var b = FromBits(0x7c00, 0x8001, 0xc000, 0x4000);
        Bools(np.less(a, b)).Should().Equal(new byte[] { 1, 0, 0, 1 });
        Bools(np.greater(a, b)).Should().Equal(new byte[] { 0, 1, 1, 0 });
        Bools(np.less_equal(a, b)).Should().Equal(new byte[] { 1, 0, 0, 1 });
        Bools(np.greater_equal(a, b)).Should().Equal(new byte[] { 0, 1, 1, 0 });
    }

    [TestMethod]
    public void Compare_ScalarBroadcast_BothSides_RolesStraight()
    {
        // Less/Greater are not symmetric — scalar-left must not reuse scalar-right's roles.
        var bits = new ushort[40];
        for (int i = 0; i < 40; i++) bits[i] = (ushort)(0x3c00 + i * 8); // ascending positives
        using var arr = FromBits(bits);
        using var sc = FromBits(0x3c40); // sits inside the range
        sc.Storage.Reshape(Shape.Scalar);
        var lt1 = Bools(np.less(arr, sc));   // arr[i] < s
        var lt2 = Bools(np.less(sc, arr));   // s < arr[i]
        for (int i = 0; i < 40; i++)
        {
            lt1[i].Should().Be((byte)(bits[i] < 0x3c40 ? 1 : 0), $"arr[{i}] < scalar");
            lt2[i].Should().Be((byte)(0x3c40 < bits[i] ? 1 : 0), $"scalar < arr[{i}]");
        }
    }

    [TestMethod]
    public void Compare_TailAndSimdSeams_Agree()
    {
        // n = 67 exercises the 32-wide SIMD body plus a 3-element scalar tail; results at the
        // seam positions must agree with the scalar formula.
        var da = new ushort[67];
        var db = new ushort[67];
        var rng = new System.Random(11);
        for (int i = 0; i < 67; i++) { da[i] = (ushort)rng.Next(0x10000); db[i] = (ushort)rng.Next(0x10000); }
        using var a = FromBits(da);
        using var b = FromBits(db);
        var got = Bools(np.less_equal(a, b));
        for (int i = 0; i < 67; i++)
        {
            bool anyNaN = (da[i] & 0x7fff) > 0x7c00 || (db[i] & 0x7fff) > 0x7c00;
            bool bothZero = ((da[i] | db[i]) & 0x7fff) == 0;
            ushort Key(ushort v) => (ushort)(v ^ (0x8000 | ((short)v >> 15)));
            byte want = (byte)(!anyNaN && (bothZero || Key(da[i]) <= Key(db[i])) ? 1 : 0);
            got[i].Should().Be(want, $"le at {i} (a={da[i]:x4}, b={db[i]:x4})");
        }
    }
}
