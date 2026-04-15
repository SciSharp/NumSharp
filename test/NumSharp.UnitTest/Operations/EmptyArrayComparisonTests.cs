using System;
using System.Linq;
using NumSharp;

namespace NumSharp.UnitTest.Operations;

/// <summary>
/// Tests for empty array comparison operations.
/// NumPy behavior: comparing empty arrays with scalars should return empty arrays, not scalars.
///
/// NumPy reference:
/// >>> (np.array([], dtype=int) == 0).shape
/// (0,)
/// >>> (np.array([], dtype=int) != 0).shape
/// (0,)
/// >>> (np.array([], dtype=int) > 0).shape
/// (0,)
/// </summary>
[TestClass]
public class EmptyArrayComparisonTests
{
    [TestMethod]
    public void EmptyArray_Equals_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) == 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt == 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_NotEquals_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) != 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt != 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_GreaterThan_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) > 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt > 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_GreaterThanOrEqual_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) >= 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt >= 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_LessThan_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) < 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt < 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_LessThanOrEqual_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=int) <= 0).shape returns (0,)
        var emptyInt = np.array(Array.Empty<int>());
        var result = emptyInt <= 0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void Empty2DArray_Equals_Scalar_PreservesShape()
    {
        // NumPy: (np.zeros((0,3), dtype=int) == 0).shape returns (0, 3)
        var empty2D = np.zeros(new Shape(0, 3), NPTypeCode.Int32);
        var result = empty2D == 0;

        result.shape.SequenceEqual(new long[] { 0, 3 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void Empty2DArray_Reverse_Equals_Scalar_PreservesShape()
    {
        // NumPy: (np.zeros((3,0), dtype=int) == 0).shape returns (3, 0)
        var empty2D = np.zeros(new Shape(3, 0), NPTypeCode.Int32);
        var result = empty2D == 0;

        result.shape.SequenceEqual(new long[] { 3, 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyFloatArray_Equals_Scalar_ReturnsEmptyArray()
    {
        // NumPy: (np.array([], dtype=float) == 0.0).shape returns (0,)
        var emptyFloat = np.array(Array.Empty<double>());
        var result = emptyFloat == 0.0;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }

    [TestMethod]
    public void EmptyArray_ArrayEquals_Preserves_Shape()
    {
        // Compare two empty arrays
        var empty1 = np.array(Array.Empty<int>());
        var empty2 = np.array(Array.Empty<int>());
        var result = empty1 == empty2;

        result.shape.SequenceEqual(new long[] { 0 }).Should().BeTrue();
        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(bool));
    }
}
