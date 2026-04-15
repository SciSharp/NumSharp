using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.shuffle matching NumPy legacy API behavior.
/// NumPy's legacy shuffle only shuffles along axis 0 (no axis parameter).
/// For axis support, use the Generator API (not yet implemented in NumSharp).
/// </summary>

    public class ShuffleNumPyAlignedTests : TestClass
{
    [TestMethod]
    public void Shuffle_1D_ShufflesElements()
    {
        var arr = np.arange(10);
        var originalSum = (int)np.sum(arr);

        var rng = np.random.RandomState(42);
        rng.shuffle(arr);

        // Sum should be unchanged (same elements, different order)
        Assert.AreEqual(originalSum, (int)np.sum(arr));
    }

    [TestMethod]
    public void Shuffle_2D_ShufflesRowsOnly()
    {
        // NumPy: shuffle only shuffles along axis 0 (rows for 2D)
        var arr = np.arange(9).reshape(3, 3);
        // rows: [0,1,2], [3,4,5], [6,7,8]

        var rng = np.random.RandomState(42);
        rng.shuffle(arr);

        // Total sum unchanged
        Assert.AreEqual(36, (int)np.sum(arr));

        // Each row's sum should be one of 3, 12, 21 (just in different order)
        var row0Sum = (int)np.sum(arr[0]);
        var row1Sum = (int)np.sum(arr[1]);
        var row2Sum = (int)np.sum(arr[2]);
        var sums = new[] { row0Sum, row1Sum, row2Sum }.OrderBy(x => x).ToArray();
        CollectionAssert.AreEqual(new[] { 3, 12, 21 }, sums);
    }

    [TestMethod]
    public void Shuffle_3D_ShufflesFirstDimension()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        var sumBefore = (int)np.sum(arr);

        // Get sums of each block before shuffle
        var block0Sum = (int)np.sum(arr[0]);
        var block1Sum = (int)np.sum(arr[1]);

        var rng = np.random.RandomState(42);
        rng.shuffle(arr);

        // Total sum should be unchanged
        Assert.AreEqual(sumBefore, (int)np.sum(arr));

        // Block sums should still exist (just potentially swapped)
        var newBlock0Sum = (int)np.sum(arr[0]);
        var newBlock1Sum = (int)np.sum(arr[1]);
        var sums = new[] { newBlock0Sum, newBlock1Sum }.OrderBy(x => x).ToArray();
        var expectedSums = new[] { block0Sum, block1Sum }.OrderBy(x => x).ToArray();
        CollectionAssert.AreEqual(expectedSums, sums);
    }

    [TestMethod]
    public void Shuffle_SingleElement_NoOp()
    {
        var arr = np.array(new[] { 42 });
        np.random.shuffle(arr);
        Assert.AreEqual(42, (int)arr[0]);
    }

    [TestMethod]
    public void Shuffle_4D_ShufflesFirstDimension()
    {
        var arr = np.arange(120).reshape(2, 3, 4, 5);
        var sumBefore = (int)np.sum(arr);

        var block0SumBefore = (int)np.sum(arr[0]);
        var block1SumBefore = (int)np.sum(arr[1]);

        var rng = np.random.RandomState(42);
        rng.shuffle(arr);

        // Total sum unchanged
        Assert.AreEqual(sumBefore, (int)np.sum(arr));

        // Block sums preserved
        var block0Sum = (int)np.sum(arr[0]);
        var block1Sum = (int)np.sum(arr[1]);
        var blockSums = new[] { block0Sum, block1Sum }.OrderBy(x => x).ToArray();
        var expectedSums = new[] { block0SumBefore, block1SumBefore }.OrderBy(x => x).ToArray();
        CollectionAssert.AreEqual(expectedSums, blockSums);
    }

    [TestMethod]
    public void Shuffle_PreservesRowContents()
    {
        // Key NumPy behavior: row contents are unchanged, only order changes
        var arr = np.arange(12).reshape(4, 3);

        // Each row has unique values that sum uniquely
        // Row 0: [0,1,2] sum=3
        // Row 1: [3,4,5] sum=12
        // Row 2: [6,7,8] sum=21
        // Row 3: [9,10,11] sum=30

        var rng = np.random.RandomState(123);
        rng.shuffle(arr);

        // Verify each row still has valid sum (no mixing of elements between rows)
        var rowSums = new HashSet<int>();
        for (int i = 0; i < 4; i++)
        {
            rowSums.Add((int)np.sum(arr[i]));
        }

        Assert.IsTrue(rowSums.Contains(3));
        Assert.IsTrue(rowSums.Contains(12));
        Assert.IsTrue(rowSums.Contains(21));
        Assert.IsTrue(rowSums.Contains(30));
    }
}
