using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest.Statistics;

/// <summary>
/// Battle tests for np.cov, comparing against NumPy 2.4.2 reference values.
/// Covers 1-D/2-D, rowvar, y, bias, ddof, fweights/aweights, dtype, complex,
/// empty/degenerate cases, non-contiguous layouts, and the error taxonomy
/// (ValueError / TypeError / RuntimeError with verbatim messages).
/// </summary>
[TestClass]
public class np_cov_BattleTests
{
    // Flat C-order element access as double.
    private static double At(NDArray nd, int i) => Convert.ToDouble(nd.reshape(nd.size == 0 ? 0 : nd.size).GetAtIndex(i));
    private static double Re(NDArray nd, int i) => Convert.ToDouble(np.real(nd).reshape(nd.size).GetAtIndex(i));
    private static double Im(NDArray nd, int i) => Convert.ToDouble(np.imag(nd).reshape(nd.size).GetAtIndex(i));
    private const double Tol = 1e-8;

    // ── basic shapes ───────────────────────────────────────────────────

    [TestMethod]
    public void Cov_1D_ReturnsScalarVariance()
    {
        // NumPy: np.cov([1,2,3,4,5]) -> array(2.5)
        var r = np.cov(np.array(new double[] { 1, 2, 3, 4, 5 }));
        r.shape.Should().Equal(Array.Empty<long>());
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().BeApproximately(2.5, Tol);
    }

    [TestMethod]
    public void Cov_2D_RowvarTrue()
    {
        // NumPy: np.cov([[0,1,2],[2,1,0]]) -> [[1,-1],[-1,1]]
        var r = np.cov(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        r.typecode.Should().Be(NPTypeCode.Double);
        At(r, 0).Should().BeApproximately(1.0, Tol);
        At(r, 1).Should().BeApproximately(-1.0, Tol);
        At(r, 2).Should().BeApproximately(-1.0, Tol);
        At(r, 3).Should().BeApproximately(1.0, Tol);
    }

    [TestMethod]
    public void Cov_2D_RowvarFalse_TransposesRelationship()
    {
        // NumPy: np.cov([[0,1,2],[2,1,0]], rowvar=False) -> [[2,0,-2],[0,0,0],[-2,0,2]]
        var r = np.cov(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), rowvar: false);
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(2.0, Tol);
        At(r, 2).Should().BeApproximately(-2.0, Tol);
        At(r, 4).Should().BeApproximately(0.0, Tol);
        At(r, 8).Should().BeApproximately(2.0, Tol);
    }

    // ── y (second data set) ────────────────────────────────────────────

    [TestMethod]
    public void Cov_WithY_1D()
    {
        // NumPy: np.cov([-2.1,-1,4.3],[3,1.1,0.12]) -> [[11.71,-4.286],[-4.286,2.14413333]]
        var r = np.cov(np.array(new double[] { -2.1, -1, 4.3 }), np.array(new double[] { 3, 1.1, 0.12 }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        At(r, 0).Should().BeApproximately(11.71, 1e-6);
        At(r, 1).Should().BeApproximately(-4.286, 1e-6);
        At(r, 3).Should().BeApproximately(2.14413333, 1e-6);
    }

    [TestMethod]
    public void Cov_WithY_2D_And_1D()
    {
        // NumPy: np.cov([[1,2,3],[4,5,6]], [7,8,9]).shape -> (3,3)
        var r = np.cov(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }), np.array(new double[] { 7, 8, 9 }));
        r.shape.Should().Equal(new long[] { 3, 3 });
    }

    // ── bias / ddof ────────────────────────────────────────────────────

    [TestMethod]
    public void Cov_BiasTrue_NormalizesByN()
    {
        // NumPy: np.cov([1,2,3,4,5], bias=True) -> array(2.)
        At(np.cov(np.array(new double[] { 1, 2, 3, 4, 5 }), bias: true), 0).Should().BeApproximately(2.0, Tol);
    }

    [TestMethod]
    public void Cov_Ddof0_EqualsBias()
    {
        // NumPy: np.cov([1,2,3,4,5], ddof=0) -> array(2.)
        At(np.cov(np.array(new double[] { 1, 2, 3, 4, 5 }), ddof: 0), 0).Should().BeApproximately(2.0, Tol);
    }

    [TestMethod]
    public void Cov_Ddof2_OverridesDefault()
    {
        // NumPy: np.cov([1,2,3,4,5], ddof=2) -> array(3.33333333)
        At(np.cov(np.array(new double[] { 1, 2, 3, 4, 5 }), ddof: 2), 0).Should().BeApproximately(3.33333333, 1e-6);
    }

    // ── weights ────────────────────────────────────────────────────────

    [TestMethod]
    public void Cov_FWeights()
    {
        // NumPy: np.cov([[0,1,2],[2,1,0]], fweights=[1,2,3]) -> [[0.66666667,-0.66666667],[-0.66666667,0.66666667]]
        var r = np.cov(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), fweights: np.array(new int[] { 1, 2, 3 }));
        At(r, 0).Should().BeApproximately(0.66666667, 1e-6);
        At(r, 1).Should().BeApproximately(-0.66666667, 1e-6);
    }

    [TestMethod]
    public void Cov_AWeights()
    {
        // NumPy: np.cov([[0,1,2],[2,1,0]], aweights=[1,2,3]) -> [[0.90909091,-0.90909091],...]
        var r = np.cov(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }), aweights: np.array(new double[] { 1, 2, 3 }));
        At(r, 0).Should().BeApproximately(0.90909091, 1e-6);
        At(r, 3).Should().BeApproximately(0.90909091, 1e-6);
    }

    [TestMethod]
    public void Cov_FWeights_And_AWeights()
    {
        // NumPy: np.cov([[0,1,2],[2,1,0]], fweights=[1,2,3], aweights=[0.5,1,2]) -> [[0.43103448,...]]
        var r = np.cov(np.array(new int[,] { { 0, 1, 2 }, { 2, 1, 0 } }),
            fweights: np.array(new int[] { 1, 2, 3 }), aweights: np.array(new double[] { 0.5, 1.0, 2.0 }));
        At(r, 0).Should().BeApproximately(0.43103448, 1e-6);
    }

    // ── dtype ──────────────────────────────────────────────────────────

    [TestMethod]
    public void Cov_IntInput_PromotesToFloat64()
    {
        np.cov(np.array(new int[] { 1, 2, 3, 4, 5 })).typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Cov_DtypeFloat32()
    {
        // NumPy: np.cov([1,2,3,4,5], dtype=np.float32) -> array(2.5, dtype=float32)
        var r = np.cov(np.array(new int[] { 1, 2, 3, 4, 5 }), dtype: NPTypeCode.Single);
        r.typecode.Should().Be(NPTypeCode.Single);
        At(r, 0).Should().BeApproximately(2.5, 1e-5);
    }

    [TestMethod]
    public void Cov_Complex()
    {
        // NumPy: np.cov([[1+2j,3+4j,5+6j],[1-1j,0,2+3j]]) ->
        //   [[8+0j, 5-3j],[5+3j, 5.33333333+0j]]
        var m = np.array(new Complex[,]
        {
            { new(1, 2), new(3, 4), new(5, 6) },
            { new(1, -1), new(0, 0), new(2, 3) }
        });
        var r = np.cov(m);
        r.shape.Should().Equal(new long[] { 2, 2 });
        r.typecode.Should().Be(NPTypeCode.Complex);
        Re(r, 0).Should().BeApproximately(8.0, 1e-8); Im(r, 0).Should().BeApproximately(0.0, 1e-8);
        Re(r, 1).Should().BeApproximately(5.0, 1e-8); Im(r, 1).Should().BeApproximately(-3.0, 1e-8);
        Re(r, 2).Should().BeApproximately(5.0, 1e-8); Im(r, 2).Should().BeApproximately(3.0, 1e-8);
        Re(r, 3).Should().BeApproximately(5.33333333, 1e-6); Im(r, 3).Should().BeApproximately(0.0, 1e-8);
    }

    // ── empty / degenerate ─────────────────────────────────────────────

    [TestMethod]
    public void Cov_Empty1D_ReturnsNaN()
    {
        // NumPy: np.cov([]) -> array(nan)
        var r = np.cov(np.array(new double[] { }));
        r.shape.Should().Equal(Array.Empty<long>());
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Cov_ZeroVariables_ReturnsEmpty00()
    {
        // NumPy: np.cov(np.zeros((0,3))) -> array of shape (0,0), dtype float64
        var r = np.cov(np.zeros(new Shape(0, 3), NPTypeCode.Double));
        r.shape.Should().Equal(new long[] { 0, 0 });
        r.typecode.Should().Be(NPTypeCode.Double);
    }

    [TestMethod]
    public void Cov_ZeroVariables_RowvarFalse_ReturnsEmpty00()
    {
        // NumPy: np.cov(np.zeros((3,0)), rowvar=False) -> (0,0)
        np.cov(np.zeros(new Shape(3, 0), NPTypeCode.Double), rowvar: false).shape.Should().Equal(new long[] { 0, 0 });
    }

    [TestMethod]
    public void Cov_ZeroObservations_ReturnsNaNMatrix()
    {
        // NumPy: np.cov(np.zeros((3,0))) -> (3,3) all nan
        var r = np.cov(np.zeros(new Shape(3, 0), NPTypeCode.Double));
        r.shape.Should().Equal(new long[] { 3, 3 });
        for (int i = 0; i < 9; i++) double.IsNaN(At(r, i)).Should().BeTrue();
    }

    [TestMethod]
    public void Cov_SingleObservation_ReturnsNaN()
    {
        // NumPy: np.cov([[1],[2]]) -> (2,2) all nan  (ddof=1 => fact=0)
        var r = np.cov(np.array(new double[,] { { 1 }, { 2 } }));
        r.shape.Should().Equal(new long[] { 2, 2 });
        double.IsNaN(At(r, 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Cov_DdofTooLarge_ReturnsInf()
    {
        // NumPy: np.cov([1,2,3], ddof=5) -> array(inf)
        double.IsPositiveInfinity(At(np.cov(np.array(new double[] { 1, 2, 3 }), ddof: 5), 0)).Should().BeTrue();
    }

    [TestMethod]
    public void Cov_Scalar_ReturnsNaN()
    {
        // NumPy: np.cov(5.0) -> array(nan)
        double.IsNaN(At(np.cov(NDArray.Scalar(5.0)), 0)).Should().BeTrue();
    }

    // ── non-contiguous input layouts (must match the contiguous result) ─

    [TestMethod]
    public void Cov_Transposed_Input()
    {
        // NumPy: np.cov(base.T) with base = [[1..4],[5..8],[9..12]] -> (4,4) all 16.0
        var b = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
        var r = np.cov(b.T);
        r.shape.Should().Equal(new long[] { 4, 4 });
        At(r, 0).Should().BeApproximately(16.0, Tol);
    }

    [TestMethod]
    public void Cov_NegativeStride_Input()
    {
        // NumPy: np.cov(base[:, ::-1]) == np.cov(base) -> (3,3) all 1.66666667
        var b = np.array(new double[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
        var r = np.cov(b[":, ::-1"]);
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(1.66666667, 1e-6);
    }

    [TestMethod]
    public void Cov_BroadcastInput()
    {
        // NumPy: np.cov(np.broadcast_to([1,2,3,4],(3,4))) -> (3,3) all 1.66666667
        var r = np.cov(np.broadcast_to(np.array(new double[] { 1, 2, 3, 4 }), new Shape(3, 4)));
        r.shape.Should().Equal(new long[] { 3, 3 });
        At(r, 0).Should().BeApproximately(1.66666667, 1e-6);
    }

    // ── error taxonomy ─────────────────────────────────────────────────

    [TestMethod]
    public void Cov_M_MoreThan2D_Throws()
    {
        new Action(() => np.cov(np.zeros(new Shape(2, 2, 2), NPTypeCode.Double)))
            .Should().Throw<ValueError>().WithMessage("m has more than 2 dimensions");
    }

    [TestMethod]
    public void Cov_Y_MoreThan2D_Throws()
    {
        new Action(() => np.cov(np.array(new double[] { 1, 2, 3 }), y: np.zeros(new Shape(2, 2, 2), NPTypeCode.Double)))
            .Should().Throw<ValueError>().WithMessage("y has more than 2 dimensions");
    }

    [TestMethod]
    public void Cov_FWeights_NonInteger_ThrowsTypeError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), fweights: np.array(new double[] { 1.5, 2, 3 })))
            .Should().Throw<TypeError>().WithMessage("fweights must be integer");
    }

    [TestMethod]
    public void Cov_FWeights_2D_ThrowsRuntimeError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), fweights: np.array(new int[,] { { 1, 2, 3 } })))
            .Should().Throw<RuntimeError>().WithMessage("cannot handle multidimensional fweights");
    }

    [TestMethod]
    public void Cov_FWeights_WrongLength_ThrowsRuntimeError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), fweights: np.array(new int[] { 1, 2 })))
            .Should().Throw<RuntimeError>().WithMessage("incompatible numbers of samples and fweights");
    }

    [TestMethod]
    public void Cov_FWeights_Negative_ThrowsValueError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), fweights: np.array(new int[] { 1, -2, 3 })))
            .Should().Throw<ValueError>().WithMessage("fweights cannot be negative");
    }

    [TestMethod]
    public void Cov_AWeights_2D_ThrowsRuntimeError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), aweights: np.array(new double[,] { { 1, 2, 3 } })))
            .Should().Throw<RuntimeError>().WithMessage("cannot handle multidimensional aweights");
    }

    [TestMethod]
    public void Cov_AWeights_WrongLength_ThrowsRuntimeError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), aweights: np.array(new double[] { 1, 2 })))
            .Should().Throw<RuntimeError>().WithMessage("incompatible numbers of samples and aweights");
    }

    [TestMethod]
    public void Cov_AWeights_Negative_ThrowsValueError()
    {
        new Action(() => np.cov(np.array(new double[,] { { 1, 2, 3 } }), aweights: np.array(new double[] { 1, -2, 3 })))
            .Should().Throw<ValueError>().WithMessage("aweights cannot be negative");
    }

    [TestMethod]
    public void Cov_Null_ThrowsArgumentNull()
    {
        new Action(() => np.cov(null)).Should().Throw<ArgumentNullException>();
    }
}
