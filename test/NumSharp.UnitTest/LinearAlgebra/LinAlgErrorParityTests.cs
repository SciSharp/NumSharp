using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     Argument, rank and dtype rejection across <c>np.linalg</c>, verbatim from NumPy 2.4.2.
    /// </summary>
    /// <remarks>
    ///     These are the tests the API-shell pass exists to make possible: the numerics of most of
    ///     this module wait on a LAPACK backend, but the CONTRACT in front of them does not — every
    ///     assertion here is reached before any factorisation is attempted, so it is gated today and
    ///     stays gated when the implementations land.
    /// </remarks>
    [TestClass]
    public class LinAlgErrorParityTests
    {
        private static NDArray V3 => np.arange(3.0);
        private static NDArray M23 => np.arange(6.0).reshape(2, 3);
        private static NDArray Sq3 => np.eye(3);

        [TestMethod]
        public void LinAlgError_IsAValueError_JustAsUpstream()
        {
            // numpy.linalg.LinAlgError derives from ValueError, so `except ValueError` catches it.
            typeof(LinAlgError).Should().BeAssignableTo<ValueError>();
        }

        #region rank

        [TestMethod]
        public void StackedRoutines_RejectFewerThanTwoDimensions_ReportingTheRankGiven()
        {
            new Action(() => np.linalg.inv(V3)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");

            new Action(() => np.linalg.inv(NDArray.Scalar(2.0))).Should().Throw<LinAlgError>()
                .WithMessage("0-dimensional array given. Array must be at least two-dimensional");

            new Action(() => np.linalg.det(V3)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be at least two-dimensional");
        }

        [TestMethod]
        public void Lstsq_DemandsExactlyTwoDimensions_SoItsMessageOmitsAtLeast()
        {
            // lstsq does not stack, and the wording differs by those two words.
            new Action(() => np.linalg.lstsq(V3, V3)).Should().Throw<LinAlgError>()
                .WithMessage("1-dimensional array given. Array must be two-dimensional");
        }

        [TestMethod]
        public void SquareRoutines_RejectNonSquareTrailingAxes()
        {
            new Action(() => np.linalg.inv(M23)).Should().Throw<LinAlgError>()
                .WithMessage("Last 2 dimensions of the array must be square");

            new Action(() => np.linalg.cholesky(M23)).Should().Throw<LinAlgError>()
                .WithMessage("Last 2 dimensions of the array must be square");

            new Action(() => np.linalg.solve(M23, V3)).Should().Throw<LinAlgError>()
                .WithMessage("Last 2 dimensions of the array must be square");

            new Action(() => np.linalg.matrix_power(M23, 2)).Should().Throw<LinAlgError>()
                .WithMessage("Last 2 dimensions of the array must be square");
        }

        #endregion

        #region dtype

        [TestMethod]
        public void Float16_IsRejectedRatherThanWidened()
        {
            // Every other narrow dtype widens to float64; float16 alone has no LAPACK loop and
            // NumPy refuses it outright.
            new Action(() => np.linalg.inv(Sq3.astype(np.float16)))
                .Should().Throw<TypeError>().WithMessage("array type float16 is unsupported in linalg");
        }

        [TestMethod]
        public void NumSharpsTwoDtypesWithNoNumPyCounterpart_TakeTheSameExit()
        {
            // Decimal and Char have no NumPy dtype at all, so no LAPACK routine has a loop for
            // them either — they follow float16's precedent rather than inventing a new error.
            new Action(() => np.linalg.inv(Sq3.astype(NPTypeCode.Decimal)))
                .Should().Throw<TypeError>().WithMessage("array type decimal is unsupported in linalg");
        }

        #endregion

        #region arguments

        [TestMethod]
        public void Eigh_ValidatesUPLO_BeforeTouchingTheArray()
        {
            new Action(() => np.linalg.eigh(Sq3, 'X'))
                .Should().Throw<ValueError>().WithMessage("UPLO argument must be 'L' or 'U'");

            // Lower case is accepted, so this must NOT be the UPLO rejection.
            new Action(() => np.linalg.eigh(Sq3, 'u'))
                .Should().NotThrow<ValueError>();
        }

        [TestMethod]
        public void Qr_ValidatesMode()
        {
            new Action(() => np.linalg.qr(Sq3, "bogus"))
                .Should().Throw<ValueError>().WithMessage("Unrecognized mode 'bogus'");
        }

        [TestMethod]
        public void Tensorinv_RejectsANonPositiveInd()
        {
            new Action(() => np.linalg.tensorinv(np.eye(4).reshape(2, 2, 2, 2), 0))
                .Should().Throw<ValueError>().WithMessage("Invalid ind argument.");
        }

        [TestMethod]
        public void Tensorsolve_CarriesNumPysRunOfSpaces()
        {
            // NumPy's literal is a backslash-continued Python string, so the source indentation
            // lands inside the message. Reproduced character for character.
            new Action(() => np.linalg.tensorsolve(np.ones(new Shape(2, 3)), np.ones(new Shape(2))))
                .Should().Throw<LinAlgError>().WithMessage(
                    "Input arrays must satisfy the requirement             " +
                    "prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])");
        }

        [TestMethod]
        public void MultiDot_NeedsTwoArrays()
        {
            new Action(() => np.linalg.multi_dot(new[] {Sq3}))
                .Should().Throw<ValueError>().WithMessage("Expecting at least two arrays.");

            new Action(() => np.linalg.multi_dot(Array.Empty<NDArray>()))
                .Should().Throw<ValueError>().WithMessage("Expecting at least two arrays.");
        }

        [TestMethod]
        public void Solve_ReportsTheGufuncItSelectedByBsRank()
        {
            // A 1-D right-hand side picks solve1's (m,m),(m)->(m); anything else picks solve's
            // (m,m),(m,n)->(m,n). The two report different signatures AND different names.
            new Action(() => np.linalg.solve(Sq3, np.arange(2.0)))
                .Should().Throw<ValueError>().WithMessage(
                    "solve1: Input operand 1 has a mismatch in its core dimension 0, " +
                    "with gufunc signature (m,m),(m)->(m) (size 2 is different from 3)");

            new Action(() => np.linalg.solve(Sq3, NDArray.Scalar(2.0)))
                .Should().Throw<ValueError>().WithMessage(
                    "solve: Input operand 1 does not have enough dimensions " +
                    "(has 0, gufunc core with signature (m,m),(m,n)->(m,n) requires 2)");
        }

        #endregion

        #region the Array-API forms are stricter than their main-namespace twins

        [TestMethod]
        public void LinalgOuter_DemandsVectors_WhereNpOuterFlattensAnything()
        {
            new Action(() => np.linalg.outer(M23, V3))
                .Should().Throw<ValueError>().WithMessage(
                    "Input arrays must be one-dimensional, but they are x1.ndim=2 and x2.ndim=1.");

            // The main-namespace function takes the same pair happily.
            np.outer(M23, V3).Should().NotBeNull();
        }

        [TestMethod]
        public void LinalgCross_DemandsThreeVectors()
        {
            new Action(() => np.linalg.cross(np.ones(new Shape(2)), np.ones(new Shape(2))))
                .Should().Throw<ValueError>().WithMessage(
                    "Both input arrays must be (arrays of) 3-dimensional vectors, " +
                    "but they are 2 and 2 dimensional instead.");
        }

        [TestMethod]
        public void Norm_HasThreeDifferentRejections_ChosenByHowManyAxesAreReduced()
        {
            // A vector order and a matrix order are different vocabularies, and a rank that is
            // neither is a third case — so these three messages are not interchangeable.
            new Action(() => np.linalg.norm(V3, "fro"))
                .Should().Throw<ValueError>().WithMessage("Invalid norm order 'fro' for vectors");

            new Action(() => np.linalg.norm(M23, 3))
                .Should().Throw<ValueError>().WithMessage("Invalid norm order for matrices.");

            new Action(() => np.linalg.norm(np.ones(new Shape(2, 2, 2)), 2))
                .Should().Throw<ValueError>().WithMessage("Improper number of dimensions to norm.");
        }

        [TestMethod]
        public void Norm_RejectsARepeatedAxis()
        {
            new Action(() => np.linalg.norm(M23, 1, new[] {0, 0}))
                .Should().Throw<ValueError>().WithMessage("Duplicate axes given.");
        }

        #endregion
    }
}
