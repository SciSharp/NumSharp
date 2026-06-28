using System;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.MathTests;

/// <summary>
///     Tests for np.minimum element-wise minimum operation.
///     All tests verified against NumPy v2.4.2.
/// </summary>
[TestClass]
public class NpMinimumTests : TestClass
{
    [TestMethod]
    public void Minimum_IntBroadcast_CorrectValues()
    {
        // NumPy: minimum([1,5,3], [[2],[4]]) = [[1,2,2],[1,4,3]]
        var a = np.array(new int[] { 1, 5, 3 });
        var b = np.array(new int[,] { { 2 }, { 4 } });
        var r = np.minimum(a, b);

        r.shape.Should().BeEquivalentTo(new long[] { 2, 3 });

        var expected = np.array(new int[,] { { 1, 2, 2 }, { 1, 4, 3 } });
        np.array_equal(r, expected).Should().BeTrue(
            "np.minimum with broadcast should compute element-wise min correctly");
    }

    [TestMethod]
    public void Minimum_DoubleBroadcast_CorrectValues()
    {
        // NumPy: minimum([1.,5.,3.], [[2.],[4.]]) = [[1,2,2],[1,4,3]]
        var a = np.array(new double[] { 1, 5, 3 });
        var b = np.array(new double[,] { { 2 }, { 4 } });
        var r = np.minimum(a, b);

        r.shape.Should().BeEquivalentTo(new long[] { 2, 3 });

        var expected = np.array(new double[,] { { 1, 2, 2 }, { 1, 4, 3 } });
        np.array_equal(r, expected).Should().BeTrue(
            "np.minimum with double broadcast should compute element-wise min correctly");
    }

    [TestMethod]
    public void Minimum_FloatBroadcast_CorrectValues()
    {
        // NumPy: minimum([1f,5f,3f], [[2f],[4f]]) = [[1,2,2],[1,4,3]]
        var a = np.array(new float[] { 1f, 5f, 3f });
        var b = np.array(new float[,] { { 2f }, { 4f } });
        var r = np.minimum(a, b);

        r.shape.Should().BeEquivalentTo(new long[] { 2, 3 });

        var expected = np.array(new float[,] { { 1f, 2f, 2f }, { 1f, 4f, 3f } });
        np.array_equal(r, expected).Should().BeTrue(
            "np.minimum with float broadcast should compute element-wise min correctly");
    }

    [TestMethod]
    public void Minimum_SameShape_CorrectValues()
    {
        var a = np.array(new int[] { 1, 5, 3, 8 });
        var b = np.array(new int[] { 2, 4, 6, 1 });
        var r = np.minimum(a, b);

        var expected = np.array(new int[] { 1, 4, 3, 1 });
        np.array_equal(r, expected).Should().BeTrue();
    }

    [TestMethod]
    public void Minimum_ScalarBroadcast_CorrectValues()
    {
        var a = np.array(new int[] { 1, 5, 3, 8 });
        var b = np.array(4);
        var r = np.minimum(a, b);

        var expected = np.array(new int[] { 1, 4, 3, 4 });
        np.array_equal(r, expected).Should().BeTrue();
    }

    [TestMethod]
    public void Maximum_IntBroadcast_CorrectValues()
    {
        // Verify np.maximum also works correctly with broadcast
        var a = np.array(new int[] { 1, 5, 3 });
        var b = np.array(new int[,] { { 2 }, { 4 } });
        var r = np.maximum(a, b);

        r.shape.Should().BeEquivalentTo(new long[] { 2, 3 });

        var expected = np.array(new int[,] { { 2, 5, 3 }, { 4, 5, 4 } });
        np.array_equal(r, expected).Should().BeTrue(
            "np.maximum with broadcast should compute element-wise max correctly");
    }

    // ----------------------------------------------------------------------------------
    // NaN propagation (regression for W11-A).
    //
    // NumPy's maximum/minimum PROPAGATE NaN: the result is NaN whenever EITHER operand is
    // NaN. NumSharp routes maximum/minimum through the clip SIMD kernel, whose hardware
    // MAXPS/MINPD intrinsics (Avx.Max/Min) return the SECOND operand on an unordered (NaN)
    // compare — silently dropping the NaN on every full SIMD vector. The scalar tail path
    // was always correct, so the bug only showed once the array reached the vector width.
    // Arrays below are deliberately >= one vector wide (4 doubles / 8 floats) to exercise
    // the SIMD lanes. All expected values verified against NumPy v2.4.2.
    // ----------------------------------------------------------------------------------

    private static void AssertElems(NDArray actual, double[] expected, string because)
    {
        actual.size.Should().Be(expected.Length, because);
        var f = actual.astype(NPTypeCode.Double);
        for (int i = 0; i < expected.Length; i++)
        {
            double a = f.GetDouble(i);
            if (double.IsNaN(expected[i]))
                double.IsNaN(a).Should().BeTrue($"element {i} should be NaN ({because})");
            else
                a.Should().Be(expected[i], $"element {i} ({because})");
        }
    }

    // Logical (flat C-order) comparison — reads through `.flat` so it is layout-independent.
    // (Multi-dimensional reads via GetDouble(int) take the int as a first-axis coordinate, not a
    // flat index, so they can't verify element order on a 2-D/3-D result; .flat always iterates
    // logical row-major.) `expected` is the NumPy C-order ravel of the result.
    private static void AssertFlat(NDArray actual, double[] expected, string because)
    {
        actual.size.Should().Be(expected.Length, because);
        var flat = actual.astype(NPTypeCode.Double).flat;
        for (int i = 0; i < expected.Length; i++)
        {
            double a = Convert.ToDouble(flat.GetValue(i));
            if (double.IsNaN(expected[i]))
                double.IsNaN(a).Should().BeTrue($"element {i} should be NaN ({because})");
            else
                a.Should().Be(expected[i], $"element {i} ({because})");
        }
    }

    [TestMethod]
    public void Maximum_Float64_PropagatesNaN_SimdPath()
    {
        double nan = double.NaN;
        var a = np.array(new double[] { 1, nan, 3, nan, 5, nan, 7, 8, 9, 10, nan, 12 });
        var b = np.array(new double[] { 9, 2, nan, 4, nan, 6, nan, nan, 1, 11, 11, 0 });
        // NumPy: maximum(a,b) = [9, nan, nan, nan, nan, nan, nan, nan, 9, 11, nan, 12]
        AssertElems(np.maximum(a, b),
            new double[] { 9, nan, nan, nan, nan, nan, nan, nan, 9, 11, nan, 12 },
            "maximum must propagate NaN from either operand");
    }

    [TestMethod]
    public void Minimum_Float64_PropagatesNaN_SimdPath()
    {
        double nan = double.NaN;
        var a = np.array(new double[] { 1, nan, 3, nan, 5, nan, 7, 8, 9, 10, nan, 12 });
        var b = np.array(new double[] { 9, 2, nan, 4, nan, 6, nan, nan, 1, 11, 11, 0 });
        // NumPy: minimum(a,b) = [1, nan, nan, nan, nan, nan, nan, nan, 1, 10, nan, 0]
        AssertElems(np.minimum(a, b),
            new double[] { 1, nan, nan, nan, nan, nan, nan, nan, 1, 10, nan, 0 },
            "minimum must propagate NaN from either operand");
    }

    [TestMethod]
    public void Maximum_Float32_PropagatesNaN_SimdPath()
    {
        float nan = float.NaN;
        var a = np.array(new float[] { 1, nan, 3, nan, 5, nan, 7, 8, 9, 10, nan, 12 });
        var b = np.array(new float[] { 9, 2, nan, 4, nan, 6, nan, nan, 1, 11, 11, 0 });
        AssertElems(np.maximum(a, b),
            new double[] { 9, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN,
                           double.NaN, double.NaN, 9, 11, double.NaN, 12 },
            "float32 maximum must propagate NaN (8-lane V256<float> path + scalar tail)");
    }

    [TestMethod]
    public void Minimum_Float32_PropagatesNaN_SimdPath()
    {
        float nan = float.NaN;
        var a = np.array(new float[] { 1, nan, 3, nan, 5, nan, 7, 8, 9, 10, nan, 12 });
        var b = np.array(new float[] { 9, 2, nan, 4, nan, 6, nan, nan, 1, 11, 11, 0 });
        AssertElems(np.minimum(a, b),
            new double[] { 1, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN,
                           double.NaN, double.NaN, 1, 10, double.NaN, 0 },
            "float32 minimum must propagate NaN");
    }

    [TestMethod]
    public void MaximumMinimum_OutAlias_PropagatesNaN_W11A()
    {
        // The exact W11-A fuzz case: maximum(a, roll(a), out=a) where the rolled operand
        // places a finite value opposite the NaN. The in-place out= write must still yield
        // NaN wherever either operand is NaN.
        double nan = double.NaN;
        var c = np.array(new double[] { 1, nan, 3, nan, 5, nan, 7, 8 });
        var rolled = np.roll(c, 1);             // [8, 1, nan, 3, nan, 5, nan, 7]
        np.maximum(c, rolled, c);               // out = c (aliases the first input)
        // NumPy: [8, nan, nan, nan, nan, nan, nan, 8]
        AssertElems(c, new double[] { 8, nan, nan, nan, nan, nan, nan, 8 },
            "maximum(a, roll(a), out=a) must propagate NaN through the aliased out= write");

        var d = np.array(new double[] { 1, nan, 3, nan, 5, nan, 7, 8 });
        var rolledD = np.roll(d, 1);
        np.minimum(d, rolledD, d);
        // NumPy: minimum -> [1, nan, nan, nan, nan, nan, nan, 7]
        AssertElems(d, new double[] { 1, nan, nan, nan, nan, nan, nan, 7 },
            "minimum(a, roll(a), out=a) must propagate NaN through the aliased out= write");
    }

    [TestMethod]
    public void MaximumMinimum_StridedView_PropagatesNaN()
    {
        // Non-contiguous (strided) operands: NaN must still propagate. arange(24) with every
        // 3rd element set to NaN, then sliced into even/odd views.
        double nan = double.NaN;
        var big = np.arange(24).astype(NPTypeCode.Double);
        for (int i = 0; i < 24; i += 3) big.SetDouble(nan, i);
        var av = big["::2"]; // [nan,2,4,nan,8,10,nan,14,16,nan,20,22]
        var bv = big["1::2"]; // [1,nan,5,7,nan,11,13,nan,17,19,nan,23]
        AssertElems(np.maximum(av, bv),
            new double[] { nan, nan, 5, nan, nan, 11, nan, nan, 17, nan, nan, 23 },
            "strided maximum must propagate NaN");
        AssertElems(np.minimum(av, bv),
            new double[] { nan, nan, 4, nan, nan, 10, nan, nan, 16, nan, nan, 22 },
            "strided minimum must propagate NaN");
    }

    [TestMethod]
    public void MaximumMinimum_Int64_NoNaNRegression()
    {
        // Integers have no NaN; the NaN-aware float wrapper must NOT alter integer results.
        var ai = np.array(new long[] { 1, 5, 3, 9, 5, 2, 7, 8 });
        var bi = np.array(new long[] { 9, 2, 4, 4, 6, 6, 1, 8 });
        np.array_equal(np.maximum(ai, bi), np.array(new long[] { 9, 5, 4, 9, 6, 6, 7, 8 }))
            .Should().BeTrue("int64 maximum is unaffected by the NaN-propagation fix");
        np.array_equal(np.minimum(ai, bi), np.array(new long[] { 1, 2, 3, 4, 5, 2, 1, 8 }))
            .Should().BeTrue("int64 minimum is unaffected by the NaN-propagation fix");
    }

    // ----------------------------------------------------------------------------------
    // W7-A: F-contiguous / strided element-pairing. maximum/minimum/clip route through the
    // flat clip kernel, which walks src/dst/lo/hi linearly and pairs them by C-order index.
    // An F-contig / strided / transposed / broadcast operand used to be read in its raw buffer
    // order, mis-pairing every element. Default.ClipNDArray now normalizes all operands to
    // C-contiguous offset-0 first. AssertElems reads in logical (flat C-order), so a mis-paired
    // result fails. Expected values verified against NumPy 2.4.2.
    // A position-sensitive checkerboard ±100 bound makes every cell's pairing observable.
    // ----------------------------------------------------------------------------------

    // a = arange(12).reshape(3,4); checkerboard c flips sign by (row+col) parity.
    private static readonly double[] CheckerC =
        { 100, -100, 100, -100, -100, 100, -100, 100, 100, -100, 100, -100 };

    [TestMethod]
    public void Maximum_FContiguousBound_PairsByLogicalIndex_W7A()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);          // C-contig
        var cF = np.array(CheckerC).reshape(3, 4).copy('F');                    // F-contig bound
        // NumPy: maximum(a, cF) = [100,1,100,3,4,100,6,100,100,9,100,11]
        AssertFlat(np.maximum(a, cF),
            new double[] { 100, 1, 100, 3, 4, 100, 6, 100, 100, 9, 100, 11 },
            "maximum must pair an F-contiguous bound by logical index, not buffer order");
    }

    [TestMethod]
    public void Minimum_FContiguousBound_PairsByLogicalIndex_W7A()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
        var cF = np.array(CheckerC).reshape(3, 4).copy('F');
        // NumPy: minimum(a, cF) = [0,-100,2,-100,-100,5,-100,7,8,-100,10,-100]
        AssertFlat(np.minimum(a, cF),
            new double[] { 0, -100, 2, -100, -100, 5, -100, 7, 8, -100, 10, -100 },
            "minimum must pair an F-contiguous bound by logical index");
    }

    [TestMethod]
    public void Maximum_FContiguousSource_PairsByLogicalIndex_W7A()
    {
        var aF = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4).copy('F'); // F-contig source
        var c = np.array(CheckerC).reshape(3, 4);                                 // C-contig bound
        // NumPy: maximum(aF, c) = [100,1,100,3,4,100,6,100,100,9,100,11]
        AssertFlat(np.maximum(aF, c),
            new double[] { 100, 1, 100, 3, 4, 100, 6, 100, 100, 9, 100, 11 },
            "maximum must pair correctly when the SOURCE (not the bound) is F-contiguous");
    }

    [TestMethod]
    public void Clip_FContiguousBounds_PairByLogicalIndex_W7A()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
        var loF = np.array(new double[] { -100, -100, -100, -100, 5, 5, 5, 5, -100, -100, -100, -100 })
            .reshape(3, 4).copy('F');
        var hiF = np.array(new double[] { 2, 2, 2, 2, 100, 100, 100, 100, 2, 2, 2, 2 })
            .reshape(3, 4).copy('F');
        // NumPy: clip(a, loF, hiF) = [0,1,2,2,5,5,6,7,2,2,2,2]
        AssertFlat(np.clip(a, loF, hiF),
            new double[] { 0, 1, 2, 2, 5, 5, 6, 7, 2, 2, 2, 2 },
            "clip must pair two F-contiguous bounds by logical index");
    }

    [TestMethod]
    public void Maximum_StridedSliceBound_PairsByLogicalIndex_W7A()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
        var c = np.array(CheckerC).reshape(3, 4);
        // Strided views: every 2nd column of each. NumPy: maximum(a[:,::2], c[:,::2]) =
        // [[100,100],[4,6],[100,100]] -> flat [100,100,4,6,100,100]
        AssertFlat(np.maximum(a[":, ::2"], c[":, ::2"]),
            new double[] { 100, 100, 4, 6, 100, 100 },
            "maximum must pair strided (column-step) views by logical index");
    }

    [TestMethod]
    public void Maximum_OutNonCContiguous_ArrayBound_W7A()
    {
        var a = np.arange(12).astype(NPTypeCode.Double).reshape(3, 4);
        var cF = np.array(CheckerC).reshape(3, 4).copy('F');
        var outF = np.zeros(new Shape(3, 4)).copy('F');     // F-contiguous out=
        var ret = np.maximum(a, cF, outF);
        // The flat kernel cannot run in F-order out=; the engine clips a C-order temp and copies
        // back. NumPy: [100,1,100,3,4,100,6,100,100,9,100,11]
        AssertFlat(outF,
            new double[] { 100, 1, 100, 3, 4, 100, 6, 100, 100, 9, 100, 11 },
            "maximum with a non-C-contiguous out= must still write the correctly-paired result");
        ReferenceEquals(ret, outF).Should().BeTrue("maximum must return the supplied out= instance");
    }

    [TestMethod]
    public void Maximum3D_TransposedBound_PairsByLogicalIndex_W7A()
    {
        var a3 = np.arange(24).astype(NPTypeCode.Double).reshape(2, 3, 4);
        // 2x4x3 *10 transposed to 2x3x4 (strided, neither C nor F).
        var b3 = (np.arange(24).astype(NPTypeCode.Double).reshape(2, 4, 3) * 10.0)
            .transpose(new int[] { 0, 2, 1 });
        // NumPy maximum(a3, b3) (b3 dominates everywhere):
        AssertFlat(np.maximum(a3, b3),
            new double[] { 0, 30, 60, 90, 10, 40, 70, 100, 20, 50, 80, 110,
                           120, 150, 180, 210, 130, 160, 190, 220, 140, 170, 200, 230 },
            "maximum must pair a 3-D transposed/strided bound by logical index");
    }

    [TestMethod]
    public void MaximumMinimum_Half_SignedZeroTieToFirstOperand_W7A()
    {
        // NumPy's float16 maximum/minimum return the FIRST operand on a tie, so signed zero is
        // resolved as: max(+0,-0)=+0, max(-0,+0)=-0, min(+0,-0)=+0, min(-0,+0)=-0. NumSharp's
        // Half scalar path used strict >/< and returned the SECOND operand on the tie (wrong sign).
        var a = np.array(new Half[] { (Half)0.0, (Half)(-0.0) });   // [+0, -0]
        var b = np.array(new Half[] { (Half)(-0.0), (Half)0.0 });   // [-0, +0]

        var mx = np.maximum(a, b);
        double.IsNegative((double)mx.GetHalf(0)).Should().BeFalse("max(+0,-0)=+0 (first operand, NumPy float16)");
        double.IsNegative((double)mx.GetHalf(1)).Should().BeTrue("max(-0,+0)=-0 (first operand)");

        var mn = np.minimum(a, b);
        double.IsNegative((double)mn.GetHalf(0)).Should().BeFalse("min(+0,-0)=+0 (first operand)");
        double.IsNegative((double)mn.GetHalf(1)).Should().BeTrue("min(-0,+0)=-0 (first operand)");
    }

    [TestMethod]
    public void Maximum_Half_FContiguousBound_PairsByLogicalIndex_W7A()
    {
        // float16 takes the scalar clip path; pairing must still follow logical (C) order.
        var halfC = new Half[12];
        for (int i = 0; i < 12; i++) halfC[i] = (Half)CheckerC[i];
        var a = np.arange(12).astype(NPTypeCode.Half).reshape(3, 4);
        var cF = np.array(halfC).reshape(3, 4).copy('F');
        AssertFlat(np.maximum(a, cF),
            new double[] { 100, 1, 100, 3, 4, 100, 6, 100, 100, 9, 100, 11 },
            "float16 maximum must pair an F-contiguous bound by logical index");
    }
}
