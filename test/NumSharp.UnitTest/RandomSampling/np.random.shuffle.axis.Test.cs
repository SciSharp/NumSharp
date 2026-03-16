using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.RandomSampling;

/// <summary>
/// Tests for np.random.shuffle with axis parameter (NumPy Generator API).
/// Unlike legacy np.random.shuffle (axis=0 only), this supports arbitrary axis.
/// </summary>
public class ShuffleAxisTests : TestClass
{
    [Test]
    public void Shuffle_1D_ShufflesElements()
    {
        var arr = np.arange(10);
        var originalSum = (int)np.sum(arr);
        
        np.random.seed(42);
        np.random.shuffle(arr);
        
        // Sum should be unchanged (same elements, different order)
        Assert.AreEqual(originalSum, (int)np.sum(arr));
    }
    
    [Test]
    public void Shuffle_2D_Axis0_ShufflesRows()
    {
        // NumPy: shuffles rows (subarrays along axis 0)
        var arr = np.arange(9).reshape(3, 3);
        // rows: [0,1,2], [3,4,5], [6,7,8]
        
        np.random.seed(42);
        np.random.shuffle(arr, axis: 0);
        
        // Total sum unchanged
        Assert.AreEqual(36, (int)np.sum(arr));
        
        // Each row's sum should be one of 3, 12, 21 (just in different order)
        var row0Sum = (int)np.sum(arr[0]);
        var row1Sum = (int)np.sum(arr[1]);
        var row2Sum = (int)np.sum(arr[2]);
        var sums = new[] { row0Sum, row1Sum, row2Sum }.OrderBy(x => x).ToArray();
        CollectionAssert.AreEqual(new[] { 3, 12, 21 }, sums);
    }
    
    [Test]
    public void Shuffle_2D_Axis1_ShufflesWithinRows()
    {
        var arr = np.arange(9).reshape(3, 3);
        
        np.random.seed(42);
        np.random.shuffle(arr, axis: 1);
        
        // Row sums should be unchanged (elements shuffled within each row)
        Assert.AreEqual(3, (int)np.sum(arr[0]));   // 0+1+2
        Assert.AreEqual(12, (int)np.sum(arr[1]));  // 3+4+5
        Assert.AreEqual(21, (int)np.sum(arr[2]));  // 6+7+8
    }
    
    [Test]
    public void Shuffle_NegativeAxis_Works()
    {
        var arr = np.arange(9).reshape(3, 3);
        
        np.random.seed(42);
        np.random.shuffle(arr, axis: -1);  // Same as axis=1 for 2D
        
        // Row sums unchanged
        Assert.AreEqual(3, (int)np.sum(arr[0]));
        Assert.AreEqual(12, (int)np.sum(arr[1]));
        Assert.AreEqual(21, (int)np.sum(arr[2]));
    }
    
    [Test]
    public void Shuffle_3D_Axis0_ShufflesFirstDimension()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        var sumBefore = (int)np.sum(arr);
        
        np.random.seed(42);
        np.random.shuffle(arr, axis: 0);
        
        // Total sum should be unchanged
        Assert.AreEqual(sumBefore, (int)np.sum(arr));
    }
    
    [Test]
    public void Shuffle_3D_Axis1_ShufflesMiddleDimension()
    {
        var arr = np.arange(24).reshape(2, 3, 4);
        var sumBefore = (int)np.sum(arr);
        
        np.random.seed(42);
        np.random.shuffle(arr, axis: 1);
        
        // Total sum should be unchanged
        Assert.AreEqual(sumBefore, (int)np.sum(arr));
    }
    
    [Test]
    public void Shuffle_SingleElement_NoOp()
    {
        // In NumSharp, np.array(new[]{42}) creates a 1D array with 1 element
        // Shuffling does nothing (n <= 1 early return)
        var arr = np.array(new[] { 42 });
        np.random.shuffle(arr);
        Assert.AreEqual(42, (int)arr[0]);
    }
    
    [Test]
    public void Shuffle_InvalidAxis_Throws()
    {
        var arr = np.arange(9).reshape(3, 3);
        Assert.ThrowsException<ArgumentOutOfRangeException>(() => np.random.shuffle(arr, axis: 5));
    }
    
    [Test]
    public void Shuffle_4D_DefaultAxis_ShufflesFirstDimension()
    {
        // This tests the actual NumPy behavior - shuffle along axis 0
        var arr = np.arange(120).reshape(2, 3, 4, 5);
        var sumBefore = (int)np.sum(arr);
        
        // Get sums of each "block" along axis 0
        var block0SumBefore = (int)np.sum(arr[0]);
        var block1SumBefore = (int)np.sum(arr[1]);
        
        np.random.seed(42);
        np.random.shuffle(arr);  // Default axis=0
        
        // Total sum unchanged
        Assert.AreEqual(sumBefore, (int)np.sum(arr));
        
        // The two block sums should still exist (just potentially swapped)
        var block0Sum = (int)np.sum(arr[0]);
        var block1Sum = (int)np.sum(arr[1]);
        var blockSums = new[] { block0Sum, block1Sum }.OrderBy(x => x).ToArray();
        var expectedSums = new[] { block0SumBefore, block1SumBefore }.OrderBy(x => x).ToArray();
        CollectionAssert.AreEqual(expectedSums, blockSums);
    }
}
