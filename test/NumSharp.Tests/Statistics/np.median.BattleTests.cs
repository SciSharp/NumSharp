using System;
using System.Numerics;

namespace NumSharp.Tests.Statistics;

/// <summary>
/// Battle tests for np.median, comparing against NumPy 2.4.2 reference values.
/// Covers axis variants, dtype preservation, NaN propagation, multi-axis tuples,
/// keepdims, and the strided/transposed input layouts from the variation matrix.
/// </summary>
[TestClass]
public class np_median_BattleTests
{
    private static double At(NDArray nd, int i) => Convert.ToDouble(nd.GetAtIndex(i));
    private static bool Near(double got, double want, double tol = 1e-9)
        => System.Math.Abs(got - want) < tol || (double.IsNaN(got) && double.IsNaN(want));

    // ── basic 1D / 2D ─────────────────────────────────────────────────────

    [TestMethod]
    public void Median_1D_Odd()
    {
        // NumPy: np.median([3,1,4,1,5,9,2]) -> 3.0
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2 });
        var m = np.median(a);
        At(m, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void Median_1D_Even()
    {
        // NumPy: np.median([3,1,4,1,5,9,2,6]) -> 3.5
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2, 6 });
        var m = np.median(a);
        At(m, 0).Should().Be(3.5);
    }

    [TestMethod]
    public void Median_2D_AxisNone()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var m = np.median(a);
        At(m, 0).Should().Be(3.5);
    }

    [TestMethod]
    public void Median_2D_Axis0()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var m = np.median(a, axis: 0);
        m.shape.Should().Equal(new long[] { 3 });
        At(m, 0).Should().Be(6.5);
        At(m, 1).Should().Be(4.5);
        At(m, 2).Should().Be(2.5);
    }

    [TestMethod]
    public void Median_2D_Axis1()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var m = np.median(a, axis: 1);
        m.shape.Should().Equal(new long[] { 2 });
        At(m, 0).Should().Be(7.0);
        At(m, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void Median_NegativeAxis_MatchesPositive()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var m1 = np.median(a, axis: -1);
        var m2 = np.median(a, axis: 2);
        m1.shape.Should().Equal(m2.shape);
        for (int i = 0; i < (int)m1.size; i++)
            At(m1, i).Should().Be(At(m2, i));
    }

    // ── keepdims ─────────────────────────────────────────────────────────

    [TestMethod]
    public void Median_Axis1_Keepdims_PreservesShape()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var m = np.median(a, axis: 1, keepdims: true);
        m.shape.Should().Equal(new long[] { 2, 1 });
        At(m, 0).Should().Be(7.0);
        At(m, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void Median_AxisNone_Keepdims_AllOnes()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var m = np.median(a, axis: (int?)null, keepdims: true);
        m.shape.Should().Equal(new long[] { 1, 1, 1 });
    }

    // ── tuple axis ───────────────────────────────────────────────────────

    [TestMethod]
    public void Median_3D_TupleAxis_01()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var m = np.median(a, axis: new[] { 0, 1 });
        m.shape.Should().Equal(new long[] { 4 });
        // Expected: [10, 11, 12, 13]
        At(m, 0).Should().Be(10.0);
        At(m, 1).Should().Be(11.0);
        At(m, 2).Should().Be(12.0);
        At(m, 3).Should().Be(13.0);
    }

    [TestMethod]
    public void Median_3D_TupleAxis_12()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var m = np.median(a, axis: new[] { 1, 2 });
        m.shape.Should().Equal(new long[] { 2 });
        At(m, 0).Should().Be(5.5);
        At(m, 1).Should().Be(17.5);
    }

    [TestMethod]
    public void Median_3D_TupleAxis_AllAxes_IsScalar()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var m = np.median(a, axis: new[] { 0, 1, 2 });
        m.shape.Should().Equal(new long[] { });
        At(m, 0).Should().Be(11.5);
    }

    // ── dtype handling ───────────────────────────────────────────────────

    [TestMethod]
    public void Median_Int32_PromotesToFloat64()
    {
        var a = np.array(new int[] { 1, 2, 3, 4, 5 });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void Median_Bool_PromotesToFloat64()
    {
        var a = np.array(new bool[] { true, false, true, true });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(1.0);
    }

    [TestMethod]
    public void Median_Float32_PreservesDtype()
    {
        var a = np.array(new float[] { 1, 2, 3, 4, 5 });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Single);
        m.GetAtIndex<float>(0).Should().Be(3.0f);
    }

    [TestMethod]
    public void Median_Float64_PreservesDtype()
    {
        var a = np.array(new double[] { 1, 2, 3, 4, 5 });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Double);
        At(m, 0).Should().Be(3.0);
    }

    [TestMethod]
    public void Median_Half_PreservesDtype()
    {
        var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5 });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Half);
        ((float)m.GetAtIndex<Half>(0)).Should().BeApproximately(3.0f, 0.01f);
    }

    [TestMethod]
    public void Median_Decimal_PreservesDtype()
    {
        var a = np.array(new decimal[] { 1m, 2m, 3m, 4m, 5m });
        var m = np.median(a);
        m.typecode.Should().Be(NPTypeCode.Decimal);
        m.GetAtIndex<decimal>(0).Should().Be(3.0m);
    }

    // ── NaN propagation ──────────────────────────────────────────────────

    [TestMethod]
    public void Median_WithNaN_PropagatesNaN()
    {
        var a = np.array(new double[] { 1, 2, double.NaN, 4 });
        var m = np.median(a);
        double.IsNaN(At(m, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Median_WithNaN_Float32_PropagatesNaN()
    {
        var a = np.array(new float[] { 1, 2, float.NaN, 4 });
        var m = np.median(a);
        float.IsNaN(m.GetAtIndex<float>(0)).Should().BeTrue();
    }

    [TestMethod]
    public void Median_PerAxis_NaNOnlyInOneSlice()
    {
        // NumPy: np.median([[1,2,NaN],[3,4,5]], axis=1) -> [NaN, 4.0]
        var a = np.array(new double[,] { { 1, 2, double.NaN }, { 3, 4, 5 } });
        var m = np.median(a, axis: 1);
        m.shape.Should().Equal(new long[] { 2 });
        double.IsNaN(At(m, 0)).Should().BeTrue();
        At(m, 1).Should().Be(4.0);
    }

    // ── complex with a NaN: NumPy's _median_nancheck ──────────────────────

    /// <summary>Python's <c>float('nan')</c>: the POSITIVE quiet NaN (.NET's <see cref="double.NaN"/> is the negative one).</summary>
    private static readonly double PyNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);

    /// <summary>A complex result's two parts as raw bits, so a NaN's sign and payload are compared too.</summary>
    /// <param name="z">The value.</param>
    /// <returns>(real bits, imaginary bits).</returns>
    private static (long, long) Bits(Complex z) => (BitConverter.DoubleToInt64Bits(z.Real), BitConverter.DoubleToInt64Bits(z.Imaginary));

    /// <summary>
    ///     np.median of complex data holding a NaN in either part returns the ELEMENT NumPy's complex order puts last
    ///     (<c>_median</c> partitions with kth -1, <c>_median_nancheck</c> returns that element when it is NaN): the order is
    ///     <c>[R + Rj, R + nanj, nan + Rj, nan + nanj]</c>, <c>R + nanj</c> by its real part, <c>nan + Rj</c> by its imaginary
    ///     part. NumPy 2.4.2's answers (probed); NumSharp gave the mean of the middle values with .NET's NaN-first sort
    ///     (the np-layer plan's NL2).
    /// </summary>
    [TestMethod]
    public void Median_Complex_WithANaN_IsTheElementNumPysOrderPutsLast()
    {
        Complex C(double re, double im) => new(re, im);
        (string Name, Complex[] Data, Complex Expected)[] cases =
        {
            ("[1+2j, nan+1j, 3, 4+1j]", new[] { C(1, 2), C(PyNaN, 1), C(3, 0), C(4, 1) }, C(PyNaN, 1)),
            ("[1+nanj, 2, 3]", new[] { C(1, PyNaN), C(2, 0), C(3, 0) }, C(1, PyNaN)),
            ("[1+nanj, nan+1j, 2+3j]", new[] { C(1, PyNaN), C(PyNaN, 1), C(2, 3) }, C(PyNaN, 1)),
            ("[nan+5j, nan+1j, 2]", new[] { C(PyNaN, 5), C(PyNaN, 1), C(2, 0) }, C(PyNaN, 5)),
            ("[1+nanj, 3+nanj, 2]", new[] { C(1, PyNaN), C(3, PyNaN), C(2, 0) }, C(3, PyNaN)),
            ("[nan+nanj, nan+1j, 2]", new[] { C(PyNaN, PyNaN), C(PyNaN, 1), C(2, 0) }, C(PyNaN, PyNaN)),
            ("[3+nanj, 1+nanj, 2, 5]", new[] { C(3, PyNaN), C(1, PyNaN), C(2, 0), C(5, 0) }, C(3, PyNaN)),
            ("[nan+1j]", new[] { C(PyNaN, 1) }, C(PyNaN, 1)),
            ("[1+1j, 2, 3, 4] (no NaN: the mean of the middle)", new[] { C(1, 1), C(2, 0), C(3, 0), C(4, 0) }, C(2.5, 0)),
        };
        foreach (var (name, data, expected) in cases)
        {
            using var a = np.array(data);
            using var m = np.median(a);
            Bits((Complex)m.GetValue()).Should().Be(Bits(expected), name);
        }
    }

    /// <summary>
    ///     Along an axis the NaN check runs per slice: <c>np.median([[1+2j, nan+1j, 3], [4+1j, 5, 6]], axis=1)</c> is
    ///     <c>[nan+1j, 5+0j]</c> and <c>np.median([[1, nan+1j], [3, 4]], axis=0)</c> is <c>[2+0j, nan+1j]</c> (NumPy 2.4.2).
    /// </summary>
    [TestMethod]
    public void Median_Complex_WithANaN_AlongAnAxis_ChecksEverySlice()
    {
        using (var a = np.array(new[,] { { new Complex(1, 2), new Complex(PyNaN, 1), new Complex(3, 0) }, { new Complex(4, 1), new Complex(5, 0), new Complex(6, 0) } }))
        using (var m = np.median(a, axis: 1))
        {
            Bits((Complex)m.GetValue(0)).Should().Be(Bits(new Complex(PyNaN, 1)));
            Bits((Complex)m.GetValue(1)).Should().Be(Bits(new Complex(5, 0)));
        }
        using (var a = np.array(new[,] { { new Complex(1, 0), new Complex(PyNaN, 1) }, { new Complex(3, 0), new Complex(4, 0) } }))
        using (var m = np.median(a, axis: 0))
        {
            Bits((Complex)m.GetValue(0)).Should().Be(Bits(new Complex(2, 0)));
            Bits((Complex)m.GetValue(1)).Should().Be(Bits(new Complex(PyNaN, 1)));
        }
    }

    // ── error paths ──────────────────────────────────────────────────────

    [TestMethod]
    public void Median_Complex_LexicographicSort()
    {
        // NumPy parity: complex median sorts lexicographically (real first,
        // imag tie-break) then picks middle. np.median([1+0j, 2+0j, 3+0j]) = (2+0j).
        var a = np.array(new Complex[] { new(1, 0), new(2, 0), new(3, 0) });
        var r = np.median(a);
        r.GetComplex(0).Should().Be(new Complex(2, 0));
    }

    [TestMethod]
    public void Median_Complex_TieBreakOnImaginary()
    {
        // Tie on Real → break on Imaginary; lexicographic ascending.
        // Sorted: (1+0j), (1+5j), (2+3j). Middle = (1+5j).
        var a = np.array(new Complex[] { new(2, 3), new(1, 0), new(1, 5) });
        var r = np.median(a);
        r.GetComplex(0).Should().Be(new Complex(1, 5));
    }

    [TestMethod]
    public void Median_Null_Throws()
    {
        Action act = () => np.median((NDArray)null);
        act.Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void Median_DuplicateAxis_Throws()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        Action act = () => np.median(a, axis: new[] { 0, 0 });
        act.Should().Throw<ArgumentException>();
    }

    // ── strided layouts ──────────────────────────────────────────────────

    [TestMethod]
    public void Median_StridedView()
    {
        // np.arange(24).reshape(2,3,4)[:, ::2, ::-1] has median 11.5
        var a = np.arange(24).reshape(2, 3, 4);
        var v = a[":, ::2, ::-1"];
        var m = np.median(v);
        At(m, 0).Should().Be(11.5);
    }

    [TestMethod]
    public void Median_Transposed_MatchesNumPy()
    {
        var a = np.arange(20).reshape(4, 5);
        var t = a.T;
        // NumPy: np.median(a.T, axis=0) == np.median(a, axis=1) -> [2.,7.,12.,17.]
        var m = np.median(t, axis: 0);
        m.shape.Should().Equal(new long[] { 4 });
        At(m, 0).Should().Be(2.0);
        At(m, 1).Should().Be(7.0);
        At(m, 2).Should().Be(12.0);
        At(m, 3).Should().Be(17.0);
    }

    // ── 0-D scalar input ─────────────────────────────────────────────────

    [TestMethod]
    public void Median_ZeroD_Scalar()
    {
        var s = NDArray.Scalar(5.0);
        var m = np.median(s);
        At(m, 0).Should().Be(5.0);
    }

    // ── out= parameter ───────────────────────────────────────────────────

    [TestMethod]
    public void Median_OutParameter_WritesAndReturnsOut()
    {
        var a = np.array(new long[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
        var outNd = np.zeros(new Shape(3), dtype: typeof(double));
        var r = np.median(a, axis: 1, @out: outNd);
        ReferenceEquals(r.Storage, outNd.Storage).Should().BeTrue();
        At(outNd, 0).Should().Be(1.5);
        At(outNd, 1).Should().Be(3.5);
        At(outNd, 2).Should().Be(5.5);
    }
}
