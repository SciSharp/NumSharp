using System.Threading.Tasks;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests to verify that axis reductions with shape[axis]==1 return independent copies,
/// not views that share memory with the original array (matching NumPy behavior).
/// Bug: Single-element axis reduction was returning views causing corruption when modified.
/// </summary>
[TestClass]
public class AxisReductionMemoryTests
{
    // ===== Sum =====

    [TestMethod]
    public async Task Sum_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0); // reduce axis with size 1

        // Modify the result
        result[0] = 999.0;

        // Original should be unchanged
        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Sum_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0, keepdims: true);

        // Modify the result
        result[0, 0] = 999.0;

        // Original should be unchanged
        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Prod =====

    [TestMethod]
    public async Task Prod_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.prod(original, axis: 0);

        result[0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Prod_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.prod(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Max =====

    [TestMethod]
    public async Task Max_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 0);

        result[0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Max_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Min =====

    [TestMethod]
    public async Task Min_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amin(original, axis: 0);

        result[0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Min_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amin(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Mean =====

    [TestMethod]
    public async Task Mean_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.mean(original, axis: 0);

        result[0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Mean_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.mean(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Cumsum (axis with size 1) =====

    [TestMethod]
    public async Task Cumsum_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.cumsum(original, axis: 0);

        result[0, 0] = 999.0;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Var (returns zeros for single element) =====

    [TestMethod]
    public async Task Var_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.var(original, axis: 0);

        // Variance of a single element is 0
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        ((double)result[0]).Should().Be(0.0);
        ((double)result[1]).Should().Be(0.0);
        ((double)result[2]).Should().Be(0.0);
    }

    [TestMethod]
    public async Task Var_AxisWithSize1_Keepdims_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.var(original, axis: 0, keepdims: true);

        result.shape.Should().BeEquivalentTo(new long[] { 1, 3 });
        ((double)result[0, 0]).Should().Be(0.0);
    }

    // ===== Std (returns zeros for single element) =====

    [TestMethod]
    public async Task Std_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.std(original, axis: 0);

        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        ((double)result[0]).Should().Be(0.0);
    }

    [TestMethod]
    public async Task Std_AxisWithSize1_Keepdims_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.std(original, axis: 0, keepdims: true);

        result.shape.Should().BeEquivalentTo(new long[] { 1, 3 });
        ((double)result[0, 0]).Should().Be(0.0);
    }

    // ===== ArgMax/ArgMin (already fixed, but verify they work) =====

    [TestMethod]
    public async Task ArgMax_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.argmax(original, axis: 0);

        // ArgMax on axis with size 1 always returns 0
        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        ((long)result[0]).Should().Be(0L);
        ((long)result[1]).Should().Be(0L);
        ((long)result[2]).Should().Be(0L);
    }

    [TestMethod]
    public async Task ArgMin_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.argmin(original, axis: 0);

        result.shape.Should().BeEquivalentTo(new long[] { 3 });
        ((long)result[0]).Should().Be(0L);
    }

    // ===== 3D array tests (more complex shapes) =====

    [TestMethod]
    public async Task Sum_3D_MiddleAxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.arange(6).reshape(2, 1, 3); // shape (2, 1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 1); // reduce middle axis

        result[0, 0] = 999;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    [TestMethod]
    public async Task Max_3D_LastAxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.arange(6).reshape(2, 3, 1); // shape (2, 3, 1)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 2); // reduce last axis

        result[0, 0] = 999;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }

    // ===== Integer dtype tests =====

    [TestMethod]
    public async Task Sum_IntArray_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1, 2, 3 } }); // int32, shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0);

        result[0] = 999;

        np.array_equal(original, originalCopy).Should().BeTrue();
    }
}
