using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins complex128 axis reductions (sum/prod/min/max/mean) to NumPy 2.4.2 output.
/// These run through the NDIter 2-operand REDUCE path
/// (DefaultEngine.ExecuteAxisReductionNDIter + ILKernelGenerator complex kernels).
///
/// Expected values produced by NumPy 2.4.2, e.g.:
///   a = (np.arange(12) + 1j*(12-np.arange(12))).reshape(3,4)
///   np.sum/prod/min/max/mean(a, axis=0|1)
/// </summary>
[TestClass]
public class ComplexAxisReductionTests
{
    // a[r,c] = (r*4+c) + (12-(r*4+c))i  for a 3x4 grid.
    private static NDArray Known3x4()
    {
        var data = new Complex[12];
        for (int i = 0; i < 12; i++) data[i] = new Complex(i, 12 - i);
        return np.array(data).reshape(3, 4);
    }

    private static void AssertClose(Complex expected, Complex actual, string ctx)
    {
        bool re = NearlyEqual(expected.Real, actual.Real);
        bool im = NearlyEqual(expected.Imaginary, actual.Imaginary);
        Assert.IsTrue(re && im, $"{ctx}: expected {expected} but got {actual}");
    }

    private static bool NearlyEqual(double a, double b)
    {
        if (double.IsNaN(a) && double.IsNaN(b)) return true;
        if (double.IsInfinity(a) || double.IsInfinity(b)) return a == b;
        return Math.Abs(a - b) <= 1e-9 * (1 + Math.Abs(b));
    }

    private static void AssertVec(NDArray r, Complex[] expected, string ctx)
    {
        Assert.AreEqual(expected.Length, (int)r.size, $"{ctx}: size");
        for (long i = 0; i < expected.Length; i++)
            AssertClose(expected[i], (Complex)r.GetAtIndex(i), $"{ctx}[{i}]");
    }

    [TestMethod]
    public void Sum_Axis0_Axis1()
    {
        var a = Known3x4();
        AssertVec(np.sum(a, axis: 0), new[] { C(12, 24), C(15, 21), C(18, 18), C(21, 15) }, "sum axis0");
        AssertVec(np.sum(a, axis: 1), new[] { C(6, 42), C(22, 26), C(38, 10) }, "sum axis1");
    }

    [TestMethod]
    public void Prod_Axis0_Axis1()
    {
        var a = Known3x4();
        AssertVec(np.prod(a, axis: 0), new[] { C(-960, 0), C(-834, 342), C(-624, 624), C(-342, 834) }, "prod axis0");
        AssertVec(np.prod(a, axis: 1), new[] { C(10512, -7344), C(-5328, -1776), C(4560, 8400) }, "prod axis1");
    }

    [TestMethod]
    public void Min_Axis0_Axis1()
    {
        var a = Known3x4();
        // NumPy complex min/max is lexicographic on (real, imag).
        AssertVec(np.amin(a, axis: 0), new[] { C(0, 12), C(1, 11), C(2, 10), C(3, 9) }, "min axis0");
        AssertVec(np.amin(a, axis: 1), new[] { C(0, 12), C(4, 8), C(8, 4) }, "min axis1");
    }

    [TestMethod]
    public void Max_Axis0_Axis1()
    {
        var a = Known3x4();
        AssertVec(np.amax(a, axis: 0), new[] { C(8, 4), C(9, 3), C(10, 2), C(11, 1) }, "max axis0");
        AssertVec(np.amax(a, axis: 1), new[] { C(3, 9), C(7, 5), C(11, 1) }, "max axis1");
    }

    [TestMethod]
    public void Mean_Axis0_Axis1()
    {
        var a = Known3x4();
        // mean = sum / count (NumPy: both components divided by the real count).
        AssertVec(np.mean(a, axis: 0), new[] { C(4, 8), C(5, 7), C(6, 6), C(7, 5) }, "mean axis0");
        AssertVec(np.mean(a, axis: 1), new[] { C(1.5, 10.5), C(5.5, 6.5), C(9.5, 2.5) }, "mean axis1");
    }

    [TestMethod]
    public void NaN_Propagation_Axis()
    {
        // [1+1j, nan+0j, 2+2j] reshaped (3,1), reduce axis 0.
        var a = np.array(new[] { C(1, 1), new Complex(double.NaN, 0), C(2, 2) }).reshape(3, 1);
        AssertVec(np.sum(a, axis: 0), new[] { new Complex(double.NaN, 3) }, "sum nan");
        // min/max propagate the NaN-containing element verbatim (nan+0j), per NumPy.
        AssertVec(np.amin(a, axis: 0), new[] { new Complex(double.NaN, 0) }, "min nan");
        AssertVec(np.amax(a, axis: 0), new[] { new Complex(double.NaN, 0) }, "max nan");
    }

    [TestMethod]
    public void Flat_MinMax_NaN_ReturnsElementVerbatim()
    {
        // Regression: flat (axis=None) complex min/max went through min_elementwise_il →
        // Min/MaxElementwiseComplexFallback, which synthesized (nan,nan) on the first NaN.
        // NumPy returns the NaN-bearing element VERBATIM (first NaN in iteration order wins).
        // Values verified against NumPy 2.4.2 (np.array(...).min()/.max()).
        AssertClose(new Complex(double.NaN, 0), (Complex)np.amin(np.array(new[] { C(1, 1), new Complex(double.NaN, 0), C(2, 2) })).GetAtIndex(0), "min [1+1j,nan+0j,2+2j]");
        AssertClose(new Complex(double.NaN, 0), (Complex)np.amax(np.array(new[] { C(1, 1), new Complex(double.NaN, 0), C(2, 2) })).GetAtIndex(0), "max [1+1j,nan+0j,2+2j]");
        // NaN in the imaginary component only → real part preserved.
        AssertClose(new Complex(0, double.NaN), (Complex)np.amin(np.array(new[] { C(1, 1), new Complex(0, double.NaN), C(2, 2) })).GetAtIndex(0), "min [1+1j,0+nanj,2+2j]");
        // Two NaN elements → the FIRST in iteration order wins (left-fold), verbatim.
        AssertClose(new Complex(double.NaN, 5), (Complex)np.amin(np.array(new[] { new Complex(double.NaN, 5), C(1, 1), new Complex(double.NaN, 0) })).GetAtIndex(0), "min [nan+5j,1+1j,nan+0j]");
        AssertClose(new Complex(double.NaN, 5), (Complex)np.amax(np.array(new[] { new Complex(double.NaN, 5), C(1, 1), new Complex(double.NaN, 0) })).GetAtIndex(0), "max [nan+5j,1+1j,nan+0j]");
        // Genuinely (nan,nan) element is returned as-is.
        AssertClose(new Complex(double.NaN, double.NaN), (Complex)np.amin(np.array(new[] { C(1, 1), C(2, 2), new Complex(double.NaN, double.NaN) })).GetAtIndex(0), "min [1+1j,2+2j,nan+nanj]");
        // No-NaN flat path stays lexicographic (real, then imag).
        AssertClose(C(3, 1), (Complex)np.amin(np.array(new[] { C(3, 1), C(3, 9) })).GetAtIndex(0), "min [3+1j,3+9j]");
        AssertClose(C(3, 9), (Complex)np.amax(np.array(new[] { C(3, 1), C(3, 9) })).GetAtIndex(0), "max [3+1j,3+9j]");
    }

    [TestMethod]
    public void Keepdims_Shape()
    {
        var a = Known3x4();
        var r = np.sum(a, axis: 0, keepdims: true);
        Assert.AreEqual(2, r.ndim, "keepdims ndim");
        Assert.AreEqual(1, (int)r.shape[0], "keepdims dim0");
        Assert.AreEqual(4, (int)r.shape[1], "keepdims dim1");
        AssertClose(C(12, 24), (Complex)r.GetAtIndex(0), "keepdims [0]");
    }

    [TestMethod]
    public void LayoutInvariance_FOrder_Transpose_Sliced()
    {
        var a = Known3x4();
        var expected0 = new[] { C(12, 24), C(15, 21), C(18, 18), C(21, 15) };

        // F-order copy must reduce identically to the C-order original.
        AssertVec(np.sum(a.copy(order: 'F'), axis: 0), expected0, "sum axis0 (F-order)");

        // Transposed view: (a.T) sum axis1 == a sum axis0 (same logical columns).
        AssertVec(np.sum(a.T, axis: 1), expected0, "sum axis1 of a.T");

        // Sliced view a[::2] (rows 0 and 2) summed along axis 0.
        var sliced = a["::2"];
        AssertVec(np.sum(sliced, axis: 0),
            new[] { C(0 + 8, 12 + 4), C(1 + 9, 11 + 3), C(2 + 10, 10 + 2), C(3 + 11, 9 + 1) },
            "sum axis0 (sliced ::2)");
    }

    [TestMethod]
    public void ThreeDimensional_AllAxes()
    {
        // 2x3x4 complex, deterministic; compare each axis reduction to a brute-force fold.
        int d0 = 2, d1 = 3, d2 = 4;
        long n = d0 * d1 * d2;
        var data = new Complex[n];
        for (long i = 0; i < n; i++) data[i] = new Complex(i * 0.5 - 3, 5 - i * 0.25);
        var a = np.array(data).reshape(d0, d1, d2);

        for (int axis = 0; axis < 3; axis++)
        {
            var got = np.sum(a, axis: axis);
            var (refv, _) = BruteForceSum(a, axis);
            for (long i = 0; i < refv.Length; i++)
                AssertClose(refv[i], (Complex)got.GetAtIndex(i), $"3D sum axis{axis}[{i}]");
        }
    }

    [TestMethod]
    public void Out_Parameter_ReturnsSameInstanceAndValues()
    {
        var a = Known3x4();
        var outArr = np.zeros(new Shape(4), NPTypeCode.Complex);
        var eng = (DefaultEngine)a.TensorEngine;
        var r = eng.ReduceAdd(a, 0, false, null, outArr);
        Assert.AreSame(outArr, r, "out= identity");
        AssertVec(outArr, new[] { C(12, 24), C(15, 21), C(18, 18), C(21, 15) }, "out= values");
    }

    private static (Complex[] vals, long[] shape) BruteForceSum(NDArray a, int axis)
    {
        int ndim = a.ndim;
        var dims = new long[ndim];
        for (int i = 0; i < ndim; i++) dims[i] = a.shape[i];
        long axisN = dims[axis];
        var outDims = new System.Collections.Generic.List<long>();
        for (int i = 0; i < ndim; i++) if (i != axis) outDims.Add(dims[i]);
        long outSize = 1; foreach (var d in outDims) outSize *= d;
        var result = new Complex[outSize];
        var outCoord = new long[outDims.Count];
        for (long oi = 0; oi < outSize; oi++)
        {
            long rem = oi;
            for (int d = outDims.Count - 1; d >= 0; d--) { outCoord[d] = rem % outDims[d]; rem /= outDims[d]; }
            var full = new long[ndim];
            for (int i = 0, od = 0; i < ndim; i++) if (i != axis) full[i] = outCoord[od++];
            Complex acc = Complex.Zero;
            for (long k = 0; k < axisN; k++)
            {
                full[axis] = k;
                long flat = 0, stride = 1;
                for (int i = ndim - 1; i >= 0; i--) { flat += full[i] * stride; stride *= dims[i]; }
                acc += (Complex)a.GetAtIndex(flat);
            }
            result[oi] = acc;
        }
        return (result, outDims.ToArray());
    }

    private static Complex C(double re, double im) => new Complex(re, im);
}
