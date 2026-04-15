using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.MathTests;

/// <summary>
///     Tests for np.minimum element-wise minimum operation.
///     All tests verified against NumPy v2.4.2.
/// </summary>
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
}
