using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Sorting_Searching_Counting;

/// <summary>
///     np.bincount parity tests. Every expected value/shape/dtype/error was captured from
///     NumPy 2.4.2 (see np.bincount(...) probes). bincount is also NumPy's histogram accumulate
///     primitive, so these pin the counting kernel the histogram family will later build on.
/// </summary>
[TestClass]
public class NpBincountTests
{
    private static void AssertInt64(NDArray r, params long[] expected)
    {
        Assert.AreEqual(NPTypeCode.Int64, r.typecode, "dtype");
        Assert.AreEqual(1, r.ndim, "ndim");
        Assert.AreEqual(expected.Length, (int)r.size, "size");
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], r.GetInt64(i), $"bin {i}");
    }

    private static void AssertDouble(NDArray r, params double[] expected)
    {
        Assert.AreEqual(NPTypeCode.Double, r.typecode, "dtype");
        Assert.AreEqual(1, r.ndim, "ndim");
        Assert.AreEqual(expected.Length, (int)r.size, "size");
        for (int i = 0; i < expected.Length; i++)
            // Bit-exact: bincount's weighted sum must match NumPy's sequential accumulation exactly.
            Assert.AreEqual(BitConverter.DoubleToInt64Bits(expected[i]),
                            BitConverter.DoubleToInt64Bits(r.GetDouble(i)), $"bin {i}");
    }

    // ---- Basic counts ----

    [TestMethod]
    public void Basic_Counts()
    {
        // np.bincount([0,1,1,2,2,2]) = [1, 2, 3]
        AssertInt64(np.bincount(np.array(new long[] { 0, 1, 1, 2, 2, 2 })), 1, 2, 3);
        // np.bincount([1,3]) = [0, 1, 0, 1]  (gaps are zeros)
        AssertInt64(np.bincount(np.array(new long[] { 1, 3 })), 0, 1, 0, 1);
        // np.bincount([5]) = [0,0,0,0,0,1]
        AssertInt64(np.bincount(np.array(new long[] { 5 })), 0, 0, 0, 0, 0, 1);
        // np.bincount([3,3,3]) = [0,0,0,3]
        AssertInt64(np.bincount(np.array(new long[] { 3, 3, 3 })), 0, 0, 0, 3);
        // np.bincount([0,0,0]) = [3]
        AssertInt64(np.bincount(np.array(new long[] { 0, 0, 0 })), 3);
    }

    // ---- minlength ----

    [TestMethod]
    public void MinLength_SmallerEqualLarger()
    {
        var x = np.array(new long[] { 0, 1, 2 });
        AssertInt64(np.bincount(x, minlength: 2), 1, 1, 1);         // < natural size: ignored
        AssertInt64(np.bincount(x, minlength: 3), 1, 1, 1);         // == natural size
        AssertInt64(np.bincount(x, minlength: 8), 1, 1, 1, 0, 0, 0, 0, 0); // > natural: pad zeros
        AssertInt64(np.bincount(x, minlength: 0), 1, 1, 1);
    }

    // ---- Empty input ----

    [TestMethod]
    public void Empty_ReturnsInt64ZerosOfMinlength()
    {
        // np.bincount([]) -> array([], dtype=int64)
        AssertInt64(np.bincount(np.array(new long[] { })));
        // np.bincount([], minlength=5) -> zeros(5) int64
        AssertInt64(np.bincount(np.array(new long[] { }), minlength: 5), 0, 0, 0, 0, 0);
    }

    [TestMethod]
    public void Empty_IgnoresMismatchedWeights()
    {
        // NumPy returns zeros(minlength) int64 BEFORE ever validating weights on empty input.
        var r = np.bincount(np.array(new long[] { }), weights: np.array(new double[] { 1.0 }));
        AssertInt64(r);
        Assert.AreEqual(NPTypeCode.Int64, r.typecode); // int64, NOT float64
    }

    // ---- Weights ----

    [TestMethod]
    public void Weights_Basic()
    {
        // np.bincount([0,1,1,2], weights=[.1,.2,.3,.4]) = [0.1, 0.5, 0.4]
        AssertDouble(np.bincount(np.array(new long[] { 0, 1, 1, 2 }),
                                 np.array(new double[] { 0.1, 0.2, 0.3, 0.4 })), 0.1, 0.5, 0.4);
        // np.bincount([0,1], weights=[5,6], minlength=4) = [5,6,0,0]
        AssertDouble(np.bincount(np.array(new long[] { 0, 1 }),
                                 np.array(new double[] { 5.0, 6.0 }), minlength: 4), 5.0, 6.0, 0.0, 0.0);
    }

    [TestMethod]
    public void Weights_IntCastToFloat64()
    {
        // np.bincount([0,1,1], weights=np.array([10,20,30],int32)) = [10., 50.]
        AssertDouble(np.bincount(np.array(new long[] { 0, 1, 1 }),
                                 np.array(new int[] { 10, 20, 30 })), 10.0, 50.0);
    }

    [TestMethod]
    public void Weights_NaNAndInfPropagate()
    {
        AssertDouble(np.bincount(np.array(new long[] { 0, 1, 0 }),
                                 np.array(new double[] { 1.0, double.NaN, 2.0 })), 3.0, double.NaN);
        AssertDouble(np.bincount(np.array(new long[] { 0, 1, 0 }),
                                 np.array(new double[] { 1.0, double.PositiveInfinity, 2.0 })), 3.0, double.PositiveInfinity);
    }

    [TestMethod]
    public void Weights_SequentialSummation_BitExact()
    {
        // Catastrophic cancellation: sequential left-to-right sum ((0+1e16)+1)-1e16)+1 == 1.0.
        // A privatized (reordered) accumulate would NOT give exactly 1.0 — pins the sequential path.
        AssertDouble(np.bincount(np.array(new long[] { 0, 0, 0, 0 }),
                                 np.array(new double[] { 1e16, 1.0, -1e16, 1.0 })), 1.0);
    }

    // ---- Input dtypes (all cast to int64) ----

    [TestMethod]
    public void Dtype_Bool()
    {
        // np.bincount([True,False,True,True]) = [1, 3]
        AssertInt64(np.bincount(np.array(new bool[] { true, false, true, true })), 1, 3);
    }

    [TestMethod]
    public void Dtype_IntegerFamily()
    {
        AssertInt64(np.bincount(np.array(new sbyte[] { 0, 1, 2, 2 })), 1, 1, 2);
        AssertInt64(np.bincount(np.array(new byte[] { 0, 1, 2, 2 })), 1, 1, 2);
        AssertInt64(np.bincount(np.array(new short[] { 0, 1, 2, 2 })), 1, 1, 2);
        AssertInt64(np.bincount(np.array(new ushort[] { 0, 1, 2, 2 })), 1, 1, 2);
        AssertInt64(np.bincount(np.array(new uint[] { 0, 5 })), 1, 0, 0, 0, 0, 1);
        AssertInt64(np.bincount(np.array(new long[] { 0, 5 })), 1, 0, 0, 0, 0, 1);
    }

    [TestMethod]
    public void Dtype_Char()
    {
        // Char is a uint16 code unit; cast to int64 by value.
        AssertInt64(np.bincount(np.array(new char[] { (char)0, (char)1, (char)1, (char)3 })), 1, 2, 0, 1);
    }

    [TestMethod]
    public void Dtype_UInt64WrapsNegative_Raises()
    {
        // FORCECAST uint64 -> int64 wraps 2^63 to int64.MinValue (negative) => negative-element error.
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new ulong[] { 9223372036854775808UL })));
        Assert.AreEqual("'list' argument must have no negative elements", ex.Message);
    }

    // ---- Non-contiguous / view inputs ----

    [TestMethod]
    public void View_StridedAndReversed()
    {
        // a[::2] of 0..19 -> evens counted once each
        AssertInt64(np.bincount(np.arange(20).astype(NPTypeCode.Int64)["::2"]),
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
        // reversed view of 0..4 -> each value once
        AssertInt64(np.bincount(np.arange(5).astype(NPTypeCode.Int64)["::-1"]), 1, 1, 1, 1, 1);
    }

    // ---- Result dtypes ----

    [TestMethod]
    public void ResultDtype_Int64OrFloat64()
    {
        Assert.AreEqual(NPTypeCode.Int64, np.bincount(np.array(new long[] { 0, 1, 2 })).typecode);
        Assert.AreEqual(NPTypeCode.Double,
            np.bincount(np.array(new long[] { 0, 1, 2 }), np.array(new double[] { 1.0, 2.0, 3.0 })).typecode);
    }

    // ---- Errors (verbatim NumPy messages) ----

    [TestMethod]
    public void Error_NegativeElements()
    {
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new long[] { -1, 0, 1 })));
        Assert.AreEqual("'list' argument must have no negative elements", ex.Message);
    }

    [TestMethod]
    public void Error_MinlengthNegative()
    {
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new long[] { 0, 1 }), minlength: -1));
        Assert.AreEqual("'minlength' must not be negative", ex.Message);
    }

    [TestMethod]
    public void Error_WeightsLengthMismatch()
    {
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new long[] { 0, 1 }), np.array(new double[] { 1.0, 2.0, 3.0 })));
        Assert.AreEqual("The weights and list don't have the same length.", ex.Message);
    }

    [TestMethod]
    public void Error_Ordering_NegativeBeforeWeightsMismatch()
    {
        // NumPy checks negative elements (minmax) BEFORE it ever validates weights length.
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new long[] { -1, 2 }), np.array(new double[] { 1.0 })));
        Assert.AreEqual("'list' argument must have no negative elements", ex.Message);
    }

    [TestMethod]
    public void Error_Ordering_MinlengthBeforeNegative()
    {
        // minlength<0 is checked before the negative-element scan.
        var ex = Assert.ThrowsException<ArgumentException>(
            () => np.bincount(np.array(new long[] { -1 }), minlength: -1));
        Assert.AreEqual("'minlength' must not be negative", ex.Message);
    }

    [TestMethod]
    public void Error_InputMustBe1D()
    {
        var deep = Assert.ThrowsException<IncorrectShapeException>(
            () => np.bincount(np.array(new long[,] { { 0, 1 }, { 2, 3 } })));
        Assert.AreEqual("object too deep for desired array", deep.Message);

        var shallow = Assert.ThrowsException<IncorrectShapeException>(
            () => np.bincount(NDArray.Scalar(5L)));
        Assert.AreEqual("object of too small depth for desired array", shallow.Message);
    }

    [TestMethod]
    public void Error_WeightsMustBe1D()
    {
        var deep = Assert.ThrowsException<IncorrectShapeException>(
            () => np.bincount(np.array(new long[] { 0, 1 }), np.array(new double[,] { { 1.0, 2.0 } })));
        Assert.AreEqual("object too deep for desired array", deep.Message);

        var shallow = Assert.ThrowsException<IncorrectShapeException>(
            () => np.bincount(np.array(new long[] { 0, 1 }), NDArray.Scalar(5.0)));
        Assert.AreEqual("object of too small depth for desired array", shallow.Message);
    }

    [TestMethod]
    public void Error_FloatInputRaises()
    {
        // An actual ndarray of a non-integer dtype fails the 'safe' cast (NumPy TypeError).
        var ex = Assert.ThrowsException<InvalidCastException>(
            () => np.bincount(np.array(new double[] { 0.0, 1.0, 2.9 })));
        Assert.AreEqual(
            "Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'",
            ex.Message);
    }

    [TestMethod]
    public void Error_ComplexWeightsRaise()
    {
        var ex = Assert.ThrowsException<InvalidCastException>(
            () => np.bincount(np.array(new long[] { 0, 1 }),
                              np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4) })));
        Assert.AreEqual(
            "Cannot cast array data from dtype('complex128') to dtype('float64') according to the rule 'safe'",
            ex.Message);
    }

    [TestMethod]
    public void NullInput_Throws()
    {
        Assert.ThrowsException<ArgumentNullException>(() => np.bincount(null));
    }

    // ---- Privatization stress (many repeats into few bins) ----

    [TestMethod]
    public void Privatization_AllSameBin()
    {
        // len >> ansSize triggers the privatized count path; merge must total exactly.
        var x = np.zeros(Shape.Vector(100_000), NPTypeCode.Int64);
        var r = np.bincount(x);
        Assert.AreEqual(1, (int)r.size);
        Assert.AreEqual(100_000L, r.GetInt64(0));
    }

    [TestMethod]
    public void Privatization_MultiBin_MatchesReference()
    {
        // Random values in a small range (privatized path) vs a naive reference count.
        var rng = new Random(4242);
        const int len = 200_003, bins = 500;
        var data = new long[len];
        var reference = new long[bins];
        for (int i = 0; i < len; i++) { long v = rng.Next(0, bins); data[i] = v; reference[v]++; }
        var r = np.bincount(np.array(data));
        Assert.AreEqual(bins, (int)r.size); // some bin near bins-1 is hit -> full width
        for (int b = 0; b < bins; b++)
            Assert.AreEqual(reference[b], r.GetInt64(b), $"bin {b}");
    }
}
