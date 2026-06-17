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
}
