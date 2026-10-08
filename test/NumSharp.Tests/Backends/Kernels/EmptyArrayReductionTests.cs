using System;

namespace NumSharp.Tests.Backends.Kernels;

/// <summary>
/// Tests for empty array reduction behavior.
/// NumPy raises ValueError for max/min of empty arrays (no identity element).
/// NumPy returns identity values for sum/prod (0 and 1 respectively).
/// </summary>
[TestClass]
public class EmptyArrayReductionTests
{
    #region Max/Min Should Throw

    /// <summary>
    /// NumPy: np.max(np.array([])) raises ValueError
    /// "zero-size array to reduction operation maximum which has no identity"
    /// </summary>
    [TestMethod]
    public void Max_EmptyArray_ThrowsValueError()
    {
        var empty = np.array(new double[0]);

        Assert.ThrowsException<ValueError>(() => np.max(empty));
    }

    /// <summary>
    /// NumPy: np.min(np.array([])) raises ValueError
    /// "zero-size array to reduction operation minimum which has no identity"
    /// </summary>
    [TestMethod]
    public void Min_EmptyArray_ThrowsValueError()
    {
        var empty = np.array(new double[0]);

        Assert.ThrowsException<ValueError>(() => np.min(empty));
    }

    /// <summary>
    /// NumPy: np.amax(np.array([])) raises ValueError
    /// </summary>
    [TestMethod]
    public void AMax_EmptyArray_ThrowsValueError()
    {
        var empty = np.array(new double[0]);

        Assert.ThrowsException<ValueError>(() => np.amax(empty));
    }

    /// <summary>
    /// NumPy: np.amin(np.array([])) raises ValueError
    /// </summary>
    [TestMethod]
    public void AMin_EmptyArray_ThrowsValueError()
    {
        var empty = np.array(new double[0]);

        Assert.ThrowsException<ValueError>(() => np.amin(empty));
    }

    /// <summary>
    /// The error is NumPy's ValueError with NumPy's exact text.
    /// </summary>
    [TestMethod]
    public void Max_EmptyArray_ErrorMessageMatchesNumPy()
    {
        var empty = np.array(new double[0]);

        var ex = Assert.ThrowsException<ValueError>(() => np.max(empty));
        Assert.AreEqual("zero-size array to reduction operation maximum which has no identity", ex.Message);
    }

    /// <summary>
    /// The error is NumPy's ValueError with NumPy's exact text.
    /// </summary>
    [TestMethod]
    public void Min_EmptyArray_ErrorMessageMatchesNumPy()
    {
        var empty = np.array(new double[0]);

        var ex = Assert.ThrowsException<ValueError>(() => np.min(empty));
        Assert.AreEqual("zero-size array to reduction operation minimum which has no identity", ex.Message);
    }

    #endregion

    #region Sum/Prod Should Return Identity

    /// <summary>
    /// NumPy: np.sum(np.array([])) returns 0.0 (identity element for addition)
    /// </summary>
    [TestMethod]
    public void Sum_EmptyArray_ReturnsZero()
    {
        var empty = np.array(new double[0]);

        var result = np.sum(empty);

        Assert.AreEqual(0.0, (double)result);
    }

    /// <summary>
    /// NumPy: np.prod(np.array([])) returns 1.0 (identity element for multiplication)
    /// </summary>
    [TestMethod]
    public void Prod_EmptyArray_ReturnsOne()
    {
        var empty = np.array(new double[0]);

        var result = np.prod(empty);

        Assert.AreEqual(1.0, (double)result);
    }

    #endregion

    #region Non-Empty Arrays Should Still Work

    /// <summary>
    /// Verify non-empty arrays still work correctly for max.
    /// </summary>
    [TestMethod]
    public void Max_NonEmptyArray_ReturnsMax()
    {
        var arr = np.array(new double[] { 1.0, 5.0, 3.0 });

        var result = np.max(arr);

        Assert.AreEqual(5.0, (double)result);
    }

    /// <summary>
    /// Verify non-empty arrays still work correctly for min.
    /// </summary>
    [TestMethod]
    public void Min_NonEmptyArray_ReturnsMin()
    {
        var arr = np.array(new double[] { 1.0, 5.0, 3.0 });

        var result = np.min(arr);

        Assert.AreEqual(1.0, (double)result);
    }

    /// <summary>
    /// Single element array should work for max.
    /// </summary>
    [TestMethod]
    public void Max_SingleElementArray_ReturnsElement()
    {
        var arr = np.array(new double[] { 42.0 });

        var result = np.max(arr);

        Assert.AreEqual(42.0, (double)result);
    }

    /// <summary>
    /// Single element array should work for min.
    /// </summary>
    [TestMethod]
    public void Min_SingleElementArray_ReturnsElement()
    {
        var arr = np.array(new double[] { 42.0 });

        var result = np.min(arr);

        Assert.AreEqual(42.0, (double)result);
    }

    #endregion

    #region Different Dtypes

    /// <summary>
    /// Empty int32 array should throw for max.
    /// </summary>
    [TestMethod]
    public void Max_EmptyInt32Array_ThrowsValueError()
    {
        var empty = np.array(new int[0]);

        Assert.ThrowsException<ValueError>(() => np.max(empty));
    }

    /// <summary>
    /// Empty float32 array should throw for max.
    /// </summary>
    [TestMethod]
    public void Max_EmptyFloat32Array_ThrowsValueError()
    {
        var empty = np.array(new float[0]);

        Assert.ThrowsException<ValueError>(() => np.max(empty));
    }

    #endregion
}
