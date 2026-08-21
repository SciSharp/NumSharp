using System;
using System.Numerics;

namespace NumSharp.Tests.Statistics;

/// <summary>
/// Battle tests for np.corrcoef, comparing against NumPy 2.4.2 reference values.
/// corrcoef is a thin wrapper over np.cov (R_ij = C_ij / sqrt(C_ii*C_jj), clipped to [-1,1]),
/// so these cover the wrapper's own surface: 2-D/rowvar/y/dtype/complex, the scalar-covariance
/// c/c branch, the nan-propagation of a constant variable, the [-1,1] clip, empty inputs, and the
/// error taxonomy inherited from cov (ValueError "m/y has more than 2 dimensions").
/// </summary>
[TestClass]
public class np_corrcoef_Test
{
    // Flat C-order element access as double. astype(Double) first so Half (which is not IConvertible)
    // and every other numeric dtype read uniformly.
    private static double At(NDArray nd, int i)
        => nd.astype(NPTypeCode.Double).reshape(nd.size == 0 ? 0 : nd.size).GetAtIndex<double>(i);
    private static double Re(NDArray nd, int i) => Convert.ToDouble(np.real(nd).reshape(nd.size).GetAtIndex(i));
    private static double Im(NDArray nd, int i) => Convert.ToDouble(np.imag(nd).reshape(nd.size).GetAtIndex(i));
    private const double Tol = 1e-8;

    // ── basic shapes ───────────────────────────────────────────────────

    [TestMethod]
    public void Corrcoef_2D_PerfectAntiCorrelation()
    {
        // NumPy: np.corrcoef([[0,1,2],[2,1,0]]) -> [[1,-1],[-1,1]]
        var r = np.corrcoef(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 1).Should().BeApproximately(-1.0, Tol);
        At(r, 2).Should().BeApproximately(-1.0, Tol);
        At(r, 3).Should().BeApproximately(1.0, Tol);
    }

    [TestMethod]
    public void Corrcoef_3Var()
    {
        // NumPy: np.corrcoef([[1,2,3,4,5],[2,4,6,8,10],[5,4,3,2,1]]) ->
        //   [[1,1,-1],[1,1,-1],[-1,-1,1]] (within rounding, clipped)
        var r = np.corrcoef(np.array(new double[,] { { 1, 2, 3, 4, 5 }, { 2, 4, 6, 8, 10 }, { 5, 4, 3, 2, 1 } }));
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 1).Should().BeApproximately(1.0, Tol);
        At(r, 2).Should().BeApproximately(-1.0, Tol);
        At(r, 5).Should().BeApproximately(-1.0, Tol);
        At(r, 8).Should().BeApproximately(1.0, Tol);
    }

    [TestMethod]
    public void Corrcoef_RowvarFalse_ConstantColumnProducesNaN()
    {
        // NumPy: np.corrcoef([[0,1,2],[2,1,0]], rowvar=False) ->
        //   [[1,nan,-1],[nan,nan,nan],[-1,nan,1]]  (column 1 is constant [1,1] -> stddev 0)
        var r = np.corrcoef(np.array(new double[,] { { 0, 1, 2 }, { 2, 1, 0 } }), rowvar: false);
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 2).Should().BeApproximately(-1.0, Tol);
        At(r, 6).Should().BeApproximately(-1.0, Tol);
        At(r, 8).Should().BeApproximately(1.0, Tol);
        // Entire middle row/column is nan.
        for (int i = 3; i <= 5; i++) double.IsNaN(At(r, i)).Should().BeTrue();
        double.IsNaN(At(r, 1)).Should().BeTrue();
        double.IsNaN(At(r, 7)).Should().BeTrue();
    }

    // ── y (second data set) ────────────────────────────────────────────

    [TestMethod]
    public void Corrcoef_WithY_1D()
    {
        // NumPy: np.corrcoef([1,2,3,4],[4,6,8,7]) -> [[1, 0.83152184],[0.83152184, 1]]
        var r = np.corrcoef(np.array(new double[] { 1, 2, 3, 4 }), np.array(new double[] { 4, 6, 8, 7 }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 1).Should().BeApproximately(0.8315218406202999, 1e-8);
        At(r, 2).Should().BeApproximately(0.8315218406202999, 1e-8);
        At(r, 3).Should().BeApproximately(1.0, Tol);
    }

    [TestMethod]
    public void Corrcoef_WithY_ConstantVariable_PartialNaN()
    {
        // NumPy: np.corrcoef([1,1,1],[1,2,3]) -> [[nan,nan],[nan,1]]
        var r = np.corrcoef(np.array(new double[] { 1, 1, 1 }), np.array(new double[] { 1, 2, 3 }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        double.IsNaN(At(r, 0)).Should().BeTrue();
        double.IsNaN(At(r, 1)).Should().BeTrue();
        double.IsNaN(At(r, 2)).Should().BeTrue();
        At(r, 3).Should().BeApproximately(1.0, Tol);
    }

    // ── scalar-covariance branch (c / c) ───────────────────────────────

    [TestMethod]
    public void Corrcoef_1D_SingleVariable_ReturnsScalarOne()
    {
        // NumPy: np.corrcoef([1,2,3,4,5]) -> array(1.) (0-d; cov is scalar -> c/c == 1)
        var r = np.corrcoef(np.array(new double[] { 1, 2, 3, 4, 5 }));
        r.shape.Should().Equal(Array.Empty<long>());
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().BeApproximately(1.0, Tol);
    }

    [TestMethod]
    public void Corrcoef_1D_ConstantVariable_ReturnsScalarNaN()
    {
        // NumPy: np.corrcoef([2,2,2]) -> array(nan) (cov == 0 -> 0/0 == nan)
        var r = np.corrcoef(np.array(new double[] { 2, 2, 2 }));
        r.shape.Should().Equal(Array.Empty<long>());
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    // ── dtype handling (inherited from cov) ────────────────────────────

    [TestMethod]
    public void Corrcoef_IntInput_PromotesToFloat64()
    {
        // NumPy: int input -> cov promotes via result_type(m, float64) -> float64.
        var r = np.corrcoef(np.array(new int[,] { { 1, 2, 3, 4 }, { 4, 5, 7, 6 }, { 9, 1, 2, 8 } }));
        r.typecode.Should().Be(NPTypeCode.Double);
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 1).Should().BeApproximately(0.7999999999999999, 1e-8);
        At(r, 2).Should().BeApproximately(-0.0632455532033676, 1e-8);
        At(r, 5).Should().BeApproximately(-0.44271887242357316, 1e-8);
    }

    [TestMethod]
    public void Corrcoef_Float32Input_PromotesToFloat64()
    {
        // NumPy: float32 input (no dtype) -> result_type(float32, float64) == float64.
        var r = np.corrcoef(np.array(new float[,] { { 1, 2, 3, 4 }, { 4, 5, 7, 6 }, { 9, 1, 2, 8 } }));
        r.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Corrcoef_DtypeSingle_ResultIsSingle()
    {
        // NumPy: explicit dtype=float32 keeps the result float32.
        var r = np.corrcoef(np.array(new int[,] { { 1, 2, 3, 4 }, { 4, 5, 7, 6 }, { 9, 1, 2, 8 } }), dtype: NPTypeCode.Single);
        r.typecode.Should().Be(NPTypeCode.Single);
        r.shape.Should().Equal(new long[] { 3, 3 });
        // Within float32 ULP tolerance of NumPy's float32 result.
        At(r, 0).Should().BeApproximately(1.0, 1e-5);
        At(r, 1).Should().BeApproximately(0.8000001311302185, 1e-5);
        At(r, 2).Should().BeApproximately(-0.06324556469917297, 1e-5);
    }

    [TestMethod]
    public void Corrcoef_DtypeHalf_ResultIsHalf()
    {
        var r = np.corrcoef(np.array(new int[,] { { 1, 2, 3, 4 }, { 4, 5, 7, 6 }, { 9, 1, 2, 8 } }), dtype: NPTypeCode.Half);
        r.typecode.Should().Be(NPTypeCode.Half);
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(1.0, 1e-2);
    }

    // ── complex ────────────────────────────────────────────────────────

    [TestMethod]
    public void Corrcoef_Complex2D()
    {
        // NumPy: np.corrcoef([[1+2j,2-1j,3+0j],[0+1j,1+1j,2-2j]]) ->
        //   [[1+0j, 0.41079192+0.13693064j],[0.41079192-0.13693064j, 1+0j]]
        var xc = np.array(new Complex[,] { { new(1, 2), new(2, -1), new(3, 0) }, { new(0, 1), new(1, 1), new(2, -2) } });
        var r = np.corrcoef(xc);
        r.typecode.Should().Be(NPTypeCode.Complex);
        r.shape.Should().Equal(new long[] { 2, 2 });
        Re(r, 0).Should().BeApproximately(1.0, Tol); Im(r, 0).Should().BeApproximately(0.0, Tol);
        Re(r, 1).Should().BeApproximately(0.4107919181288746, 1e-8); Im(r, 1).Should().BeApproximately(0.13693063937629155, 1e-8);
        Re(r, 2).Should().BeApproximately(0.4107919181288746, 1e-8); Im(r, 2).Should().BeApproximately(-0.13693063937629155, 1e-8);
        Re(r, 3).Should().BeApproximately(1.0, Tol); Im(r, 3).Should().BeApproximately(0.0, Tol);
    }

    [TestMethod]
    public void Corrcoef_Complex1D_SingleVariable_ReturnsScalarOnePlusZeroJ()
    {
        // NumPy: np.corrcoef([1+1j,2+0j,3-1j]) -> (1+0j) (0-d, c/c)
        var r = np.corrcoef(np.array(new Complex[] { new(1, 1), new(2, 0), new(3, -1) }));
        r.shape.Should().Equal(Array.Empty<long>());
        r.typecode.Should().Be(NPTypeCode.Complex);
        Re(r, 0).Should().BeApproximately(1.0, Tol);
        Im(r, 0).Should().BeApproximately(0.0, Tol);
    }

    // ── clipping to [-1, 1] ────────────────────────────────────────────

    [TestMethod]
    public void Corrcoef_ClipsToUnitInterval()
    {
        // Perfectly (anti)correlated rows can drift past ±1 by rounding; corrcoef clips.
        var r = np.corrcoef(np.array(new double[,] { { 1, 2, 3, 4, 5 }, { 2, 4, 6, 8, 10 }, { 5, 4, 3, 2, 1 } }));
        for (int i = 0; i < (int)r.size; i++)
        {
            double v = At(r, i);
            (v >= -1.0 && v <= 1.0).Should().BeTrue($"element {i} = {v} must be within [-1, 1]");
        }
    }

    // ── layout independence (cov normalises via atleast_2d + astype copy) ──

    [TestMethod]
    public void Corrcoef_TransposedInput_MatchesContiguous()
    {
        var baseArr = np.array(new double[,] { { 0.1, 0.5, 0.9, 0.3 }, { 0.7, 0.2, 0.6, 0.4 }, { 0.8, 0.1, 0.3, 0.95 } });
        var contig = np.corrcoef(baseArr, rowvar: false);          // (4,4) over columns
        var transposed = np.corrcoef(baseArr.T);                   // rows of the transpose == columns
        transposed.shape.Should().Equal(new long[] { 4, 4 });
        for (int i = 0; i < 16; i++)
            At(transposed, i).Should().BeApproximately(At(contig, i), 1e-10);
    }

    [TestMethod]
    public void Corrcoef_NegativeStrideView_Works()
    {
        var baseArr = np.array(new double[,] { { 1.0, 2, 3, 4 }, { 4, 3, 2, 1 } });
        var rev = baseArr["::-1, ::-1"];
        var r = np.corrcoef(rev);
        r.shape.Should().Equal(new long[] { 2, 2 });
        // Two rows that are exact reverses -> still perfectly anti-correlated within [-1,1].
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 3).Should().BeApproximately(1.0, Tol);
        (At(r, 1) >= -1.0 && At(r, 1) <= 1.0).Should().BeTrue();
    }

    // ── empty ──────────────────────────────────────────────────────────

    [TestMethod]
    public void Corrcoef_Empty_ReturnsZeroByZeroFloat64()
    {
        // NumPy: np.corrcoef(np.zeros((0,4))) -> shape (0,0) float64
        var r = np.corrcoef(np.zeros(new Shape(0, 4)));
        r.shape.Should().Equal(new long[] { 0, 0 });
        r.typecode.Should().Be(NPTypeCode.Double);
    }

    // ── error taxonomy (inherited from cov) ────────────────────────────

    [TestMethod]
    public void Corrcoef_XMoreThan2Dimensions_ThrowsValueError()
    {
        new Action(() => np.corrcoef(np.zeros(new Shape(2, 2, 2))))
            .Should().Throw<ValueError>().WithMessage("m has more than 2 dimensions");
    }

    [TestMethod]
    public void Corrcoef_YMoreThan2Dimensions_ThrowsValueError()
    {
        new Action(() => np.corrcoef(np.zeros(new Shape(2, 3)), np.zeros(new Shape(2, 2, 2))))
            .Should().Throw<ValueError>().WithMessage("y has more than 2 dimensions");
    }

    [TestMethod]
    public void Corrcoef_NullInput_ThrowsArgumentNullException()
    {
        new Action(() => np.corrcoef(null)).Should().Throw<ArgumentNullException>();
    }
}
