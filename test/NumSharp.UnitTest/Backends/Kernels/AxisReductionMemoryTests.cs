using System.Threading.Tasks;
using NumSharp;
using TUnit.Assertions;
using TUnit.Assertions.Extensions;
using TUnit.Core;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Tests to verify that axis reductions with shape[axis]==1 return independent copies,
/// not views that share memory with the original array (matching NumPy behavior).
/// Bug: Single-element axis reduction was returning views causing corruption when modified.
/// </summary>
public class AxisReductionMemoryTests
{
    // ===== Sum =====

    [Test]
    public async Task Sum_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0); // reduce axis with size 1

        // Modify the result
        result[0] = 999.0;

        // Original should be unchanged
        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Sum_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0, keepdims: true);

        // Modify the result
        result[0, 0] = 999.0;

        // Original should be unchanged
        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Prod =====

    [Test]
    public async Task Prod_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.prod(original, axis: 0);

        result[0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Prod_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.prod(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Max =====

    [Test]
    public async Task Max_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 0);

        result[0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Max_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Min =====

    [Test]
    public async Task Min_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amin(original, axis: 0);

        result[0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Min_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.amin(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Mean =====

    [Test]
    public async Task Mean_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.mean(original, axis: 0);

        result[0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Mean_AxisWithSize1_Keepdims_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.mean(original, axis: 0, keepdims: true);

        result[0, 0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Cumsum (axis with size 1) =====

    [Test]
    public async Task Cumsum_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)
        var originalCopy = original.copy();

        var result = np.cumsum(original, axis: 0);

        result[0, 0] = 999.0;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Var (returns zeros for single element) =====

    [Test]
    public async Task Var_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.var(original, axis: 0);

        // Variance of a single element is 0
        await Assert.That(result.shape).IsEquivalentTo(new[] { 3 });
        await Assert.That((double)result[0]).IsEqualTo(0.0);
        await Assert.That((double)result[1]).IsEqualTo(0.0);
        await Assert.That((double)result[2]).IsEqualTo(0.0);
    }

    [Test]
    public async Task Var_AxisWithSize1_Keepdims_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.var(original, axis: 0, keepdims: true);

        await Assert.That(result.shape).IsEquivalentTo(new[] { 1, 3 });
        await Assert.That((double)result[0, 0]).IsEqualTo(0.0);
    }

    // ===== Std (returns zeros for single element) =====

    [Test]
    public async Task Std_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.std(original, axis: 0);

        await Assert.That(result.shape).IsEquivalentTo(new[] { 3 });
        await Assert.That((double)result[0]).IsEqualTo(0.0);
    }

    [Test]
    public async Task Std_AxisWithSize1_Keepdims_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.std(original, axis: 0, keepdims: true);

        await Assert.That(result.shape).IsEquivalentTo(new[] { 1, 3 });
        await Assert.That((double)result[0, 0]).IsEqualTo(0.0);
    }

    // ===== ArgMax/ArgMin (already fixed, but verify they work) =====

    [Test]
    public async Task ArgMax_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.argmax(original, axis: 0);

        // ArgMax on axis with size 1 always returns 0
        await Assert.That(result.shape).IsEquivalentTo(new[] { 3 });
        await Assert.That((long)result[0]).IsEqualTo(0L);
        await Assert.That((long)result[1]).IsEqualTo(0L);
        await Assert.That((long)result[2]).IsEqualTo(0L);
    }

    [Test]
    public async Task ArgMin_AxisWithSize1_ReturnsZeros()
    {
        var original = np.array(new[,] { { 1.0, 2.0, 3.0 } }); // shape (1, 3)

        var result = np.argmin(original, axis: 0);

        await Assert.That(result.shape).IsEquivalentTo(new[] { 3 });
        await Assert.That((long)result[0]).IsEqualTo(0L);
    }

    // ===== 3D array tests (more complex shapes) =====

    [Test]
    public async Task Sum_3D_MiddleAxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.arange(6).reshape(2, 1, 3); // shape (2, 1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 1); // reduce middle axis

        result[0, 0] = 999;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    [Test]
    public async Task Max_3D_LastAxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.arange(6).reshape(2, 3, 1); // shape (2, 3, 1)
        var originalCopy = original.copy();

        var result = np.amax(original, axis: 2); // reduce last axis

        result[0, 0] = 999;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }

    // ===== Integer dtype tests =====

    [Test]
    public async Task Sum_IntArray_AxisWithSize1_ReturnsIndependentCopy()
    {
        var original = np.array(new[,] { { 1, 2, 3 } }); // int32, shape (1, 3)
        var originalCopy = original.copy();

        var result = np.sum(original, axis: 0);

        result[0] = 999;

        await Assert.That(np.array_equal(original, originalCopy)).IsTrue();
    }
}
