using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Polynomial
{
    /// <summary>
    ///     The NumPy polynomial family — <c>vander</c>, <c>polyval</c>, <c>polyder</c>, <c>polyint</c>,
    ///     <c>polyadd</c>/<c>polysub</c>/<c>polymul</c>/<c>polydiv</c>, <c>poly</c> (1-D branch) and the
    ///     <c>poly1d</c> class. Every expectation was probed against NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     These functions are pure-managed compositions and need NO backend. The BLAS-backed members
    ///     (<c>roots</c>, <c>poly</c> of a matrix, <c>polyfit</c>) route through <c>eigvals</c>/<c>lstsq</c>;
    ///     their no-backend contract is pinned in <see cref="Polynomial_NoBackend_Contract"/> and their
    ///     live byte-parity in <c>NumSharp.Interop.UnitTests/PolynomialLiveParityTests</c>.
    /// </remarks>
    [TestClass]
    public class PolynomialTests
    {
        private static void Close(NDArray actual, params double[] expected)
        {
            actual.size.Should().Be(expected.Length);
            np.allclose(actual, np.array(expected).reshape(actual.Shape), rtol: 1e-12, atol: 1e-14)
                .Should().BeTrue($"expected [{string.Join(", ", expected)}] but was {actual.ToString(false)}");
        }

        #region vander

        [TestMethod]
        public void Vander_Default_IsSquareDescendingPowers()
        {
            // np.vander([1,2,3,5]) -> 4x4, columns x^3 x^2 x^1 x^0
            np.vander(np.array(new long[] { 1, 2, 3, 5 }))
                .Should().BeOfValues(1, 1, 1, 1, 8, 4, 2, 1, 27, 9, 3, 1, 125, 25, 5, 1)
                .And.BeShaped(4, 4).And.BeOfType(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Vander_N_And_Increasing()
        {
            np.vander(np.array(new long[] { 1, 2, 3, 5 }), 3)
                .Should().BeOfValues(1, 1, 1, 4, 2, 1, 9, 3, 1, 25, 5, 1).And.BeShaped(4, 3);
            np.vander(np.array(new long[] { 1, 2, 3, 5 }), 3, increasing: true)
                .Should().BeOfValues(1, 1, 1, 1, 2, 4, 1, 3, 9, 1, 5, 25).And.BeShaped(4, 3);
        }

        [TestMethod]
        public void Vander_FloatInput_PromotesWithIntToFloat64()
        {
            // promote_types(float32, int64) == float64
            np.vander(np.array(new float[] { 1, 2, 3 })).Should().BeOfType(NPTypeCode.Double);
            np.vander(np.array(new long[] { 1, 2, 3 }), 0).Should().BeShaped(3, 0).And.BeOfType(NPTypeCode.Int64);
        }

        #endregion

        #region polyval

        [TestMethod]
        public void Polyval_ArrayX_HornerScheme()
        {
            // np.polyval([3,0,1], [1,2,3]) -> [4, 13, 28]
            np.polyval(np.array(new long[] { 3, 0, 1 }), np.array(new long[] { 1, 2, 3 }))
                .Should().BeOfValues(4, 13, 28).And.BeShaped(3).And.BeOfType(NPTypeCode.Int64);
        }

        [TestMethod]
        public void Polyval_ScalarX_ReturnsZeroD()
        {
            np.polyval(np.array(new long[] { 3, 0, 1 }), NDArray.Scalar(5L)).Should().BeScalar().And.BeOfValues(76);
        }

        [TestMethod]
        public void Polyval_DtypePromotion_ResultType()
        {
            np.polyval(np.array(new float[] { 3, 0, 1 }), np.array(new float[] { 1, 2, 3 })).Should().BeOfType(NPTypeCode.Single);
            np.polyval(np.array(new float[] { 3, 0, 1 }), np.array(new long[] { 1, 2, 3 })).Should().BeOfType(NPTypeCode.Double);
        }

        [TestMethod]
        public void Polyval_EmptyCoeffs_IsZerosLikeX()
        {
            np.polyval(np.array(Array.Empty<double>()), np.array(new double[] { 1, 2 })).Should().BeOfValues(0.0, 0.0);
        }

        [TestMethod]
        public void Polyval_Infinity_MatchesNumpyNaN()
        {
            // NumPy's `0*inf` first Horner step yields nan.
            double v = (double)np.polyval(np.array(new double[] { 3, 0, 1 }), NDArray.Scalar(double.PositiveInfinity)).GetAtIndex(0);
            double.IsNaN(v).Should().BeTrue();
        }

        [TestMethod]
        public void Polyval_ZeroDCoeffs_Throws()
        {
            // NumPy's `for pv in p` cannot iterate a 0-d coefficient array (a scalar is not `[c]`).
            new Action(() => np.polyval(NDArray.Scalar(5L), np.array(new long[] { 1, 2, 3 })))
                .Should().Throw<TypeError>().WithMessage("iteration over a 0-d array");
            // ...but a 1-D length-1 array is a valid constant polynomial.
            np.polyval(np.array(new long[] { 5 }), np.array(new long[] { 1, 2, 3 })).Should().BeOfValues(5, 5, 5);
        }

        #endregion

        #region polyder / polyint

        [TestMethod]
        public void Polyder_IntPromotesToInt64()
        {
            np.polyder(np.array(new int[] { 1, 2, 3, 4 })).Should().BeOfValues(3, 4, 3).And.BeOfType(NPTypeCode.Int64);
            np.polyder(np.array(new long[] { 1, 1, 1, 1 }), 2).Should().BeOfValues(6, 2);
        }

        [TestMethod]
        public void Polyder_Constant_And_OrderBeyondDegree_AreEmpty()
        {
            np.polyder(np.array(new int[] { 5 })).Should().BeShaped(0);
            np.polyder(np.array(new long[] { 1, 2, 3 }), 5).Should().BeShaped(0);
        }

        [TestMethod]
        public void Polyder_NegativeOrder_Throws()
        {
            new Action(() => np.polyder(np.array(new double[] { 1, 2, 3 }), -1))
                .Should().Throw<ValueError>().WithMessage("Order of derivative must be positive (see polyint)");
        }

        [TestMethod]
        public void Polyint_AlwaysFloat_WithConstants()
        {
            np.polyint(np.array(new long[] { 1, 1, 1 })).Should().BeOfType(NPTypeCode.Double);
            Close(np.polyint(np.array(new long[] { 1, 1, 1 })), 1.0 / 3, 0.5, 1, 0);
            Close(np.polyint(np.array(new long[] { 1, 1, 1 }), 3, np.array(new double[] { 6, 5, 3 })),
                0.016666666666666666, 0.041666666666666664, 0.16666666666666666, 3, 5, 3);
        }

        [TestMethod]
        public void Polyint_ScalarConstant_Broadcasts()
        {
            Close(np.polyint(np.array(new long[] { 1, 1, 1 }), 1, 5.0), 1.0 / 3, 0.5, 1, 5);
        }

        #endregion

        #region polyadd / polysub / polymul / polydiv

        [TestMethod]
        public void PolyAdd_ZeroExtendsShorter()
        {
            np.polyadd(np.array(new long[] { 1, 2 }), np.array(new long[] { 9, 5, 4 })).Should().BeOfValues(9, 6, 6);
        }

        [TestMethod]
        public void PolySub()
        {
            np.polysub(np.array(new long[] { 2, 10, -2 }), np.array(new long[] { 3, 10, -4 })).Should().BeOfValues(-1, 0, 2);
        }

        [TestMethod]
        public void PolyMul_ConvolvesAndTrimsLeadingZeros()
        {
            np.polymul(np.array(new long[] { 1, 2, 3 }), np.array(new long[] { 9, 5, 1 })).Should().BeOfValues(9, 23, 38, 17, 3);
            // poly1d normalisation drops the leading zeros of [0,0,1,2] before convolving.
            np.polymul(np.array(new long[] { 0, 0, 1, 2 }), np.array(new long[] { 1, 1 })).Should().BeOfValues(1, 3, 2);
        }

        [TestMethod]
        public void PolyDiv_QuotientAndRemainder()
        {
            var (q, r) = np.polydiv(np.array(new double[] { 3, 5, 2 }), np.array(new double[] { 2, 1 }));
            Close(q, 1.5, 1.75);
            Close(r, 0.25);
        }

        #endregion

        #region poly (1-D branch — pure)

        [TestMethod]
        public void Poly_FromRoots_BuildsMonicCoefficients()
        {
            // np.poly([0,0,0]) -> [1,0,0,0]; np.poly([-.5,0,.5]) -> [1,0,-0.25,0]
            np.poly(np.array(new double[] { 0, 0, 0 })).Should().BeOfValues(1.0, 0.0, 0.0, 0.0);
            Close(np.poly(np.array(new double[] { -0.5, 0, 0.5 })), 1, 0, -0.25, 0);
        }

        [TestMethod]
        public void Poly_IntInput_IsFloat64_And_Empty_IsScalarOne()
        {
            np.poly(np.array(new long[] { 1, 2, 3 })).Should().BeOfType(NPTypeCode.Double);
            np.poly(np.array(Array.Empty<double>())).Should().BeScalar().And.BeOfValues(1.0);
        }

        [TestMethod]
        public void Poly_ComplexConjugatePair_CollapsesToReal()
        {
            // roots {1+2j, 1-2j} -> real coefficients [1, -2, 5] (float64)
            np.poly(np.array(new Complex[] { new(1, 2), new(1, -2) }))
                .Should().BeOfValues(1.0, -2.0, 5.0).And.BeOfType(NPTypeCode.Double);
        }

        [TestMethod]
        public void Poly_InvalidShape_Throws()
        {
            new Action(() => np.poly(np.array(new double[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3)))
                .Should().Throw<ValueError>().WithMessage("input must be 1d or non-empty square 2d array.");
        }

        #endregion

        #region poly1d

        [TestMethod]
        public void Poly1d_ConstructionAndProperties()
        {
            var p = new poly1d(np.array(new long[] { 1, 2, 3 }));
            p.coeffs.Should().BeOfValues(1, 2, 3);
            p.order.Should().Be(2);
            p.c.Should().BeOfValues(1, 2, 3);
            p.ToString().Should().Be("poly1d([1, 2, 3])");
        }

        [TestMethod]
        public void Poly1d_Repr_ByteExactToNumpy()
        {
            // NumPy's poly1d.__repr__ is repr(coeffs)[6:-1] wrapped — including alignment, the trailing
            // dot on whole floats, and the dtype suffix for non-default dtypes.
            new poly1d(np.array(new double[] { 0.5, 1, 3 })).ToString().Should().Be("poly1d([0.5, 1. , 3. ])");
            new poly1d(np.array(new sbyte[] { 1, 2, 3 })).ToString().Should().Be("poly1d([1, 2, 3], dtype=int8)");
            new poly1d(np.array(Array.Empty<double>())).ToString().Should().Be("poly1d([0.])");
        }

        [TestMethod]
        public void Poly1d_Arithmetic()
        {
            var p = new poly1d(np.array(new long[] { 1, 2, 3 }));
            (p * p).coeffs.Should().BeOfValues(1, 4, 10, 12, 9);
            (p + new poly1d(np.array(new long[] { 9, 5, 4 }))).coeffs.Should().BeOfValues(10, 7, 7);
            (p - new poly1d(np.array(new long[] { 1, 1, 1 }))).coeffs.Should().BeOfValues(1, 2); // [0,1,2] trims to [1,2]
            (-p).coeffs.Should().BeOfValues(-1, -2, -3);
            (p * 2.0).coeffs.Should().BeOfValues(2, 4, 6);
        }

        [TestMethod]
        public void Poly1d_DerivIntegAndDivision()
        {
            var p = new poly1d(np.array(new long[] { 1, 2, 3 }));
            p.deriv().coeffs.Should().BeOfValues(2, 2);
            Close(p.integ().coeffs, 1.0 / 3, 1, 3, 0);
            var (q, r) = new poly1d(np.array(new double[] { 1, 0, 0, 0, 4 })) / p;
            Close(q.coeffs, 1, -2, 1);
            Close(r.coeffs, 4, 1);
        }

        [TestMethod]
        public void Poly1d_RootsMode_And_Indexer()
        {
            // poly1d([1,2], True) -> coefficients of (x-1)(x-2) = [1,-3,2]
            new poly1d(np.array(new long[] { 1, 2 }), r: true).coeffs.Should().BeOfValues(1.0, -3.0, 2.0);
            var p = new poly1d(np.array(new long[] { 1, 2, 3 }));
            ((long)p[1].GetAtIndex(0)).Should().Be(2);   // coefficient of x^1
            ((long)p[5].GetAtIndex(0)).Should().Be(0);   // out of range -> 0
            (p == new poly1d(np.array(new long[] { 1, 2, 3 }))).Should().BeTrue();
        }

        [TestMethod]
        public void Poly1d_FlowsIntoNpFunctions_AsCoefficients()
        {
            // implicit poly1d -> NDArray (NumPy's __array__): a poly1d IS its coefficient vector.
            var p = new poly1d(np.array(new long[] { 1, 2, 3 }));
            np.polyder(p).Should().BeOfValues(2, 2);
        }

        #endregion

        #region no-backend contract (BLAS-routed members)

        [TestMethod]
        public void Polynomial_NoBackend_Contract()
        {
            // Without an OpenBLAS backend, the eigvals/lstsq-routed members raise — exactly as the
            // linalg factorisations do (see LinAlgEngineSeamTests). The trivial roots paths that never
            // reach eigvals still work.
            np.roots(np.array(new double[] { 0, 0, 0 })).Should().BeShaped(0);          // all-zero -> empty, no eigvals
            np.roots(np.array(new double[] { 5 })).Should().BeShaped(0);                // single -> empty, no eigvals

            new Action(() => np.roots(np.array(new double[] { 1, 0, -1 }))).Should().Throw<NotSupportedException>();
            new Action(() => np.polyfit(np.array(new double[] { 0, 1, 2 }), np.array(new double[] { 0, 1, 4 }), 2))
                .Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Roots_Rank1Only()
        {
            new Action(() => np.roots(np.array(new double[] { 1, 2, 3, 4 }).reshape(2, 2)))
                .Should().Throw<ValueError>().WithMessage("Input must be a rank-1 array.");
        }

        #endregion
    }
}
