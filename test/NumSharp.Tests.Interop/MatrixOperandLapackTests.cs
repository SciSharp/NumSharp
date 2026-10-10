using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

namespace NumSharp.Tests.Interop
{
    /// <summary>
    ///     The class and shape <c>np.linalg</c> returns for a <c>numpy.matrix</c> operand, for the functions that
    ///     compute only through a LAPACK backend. NumPy takes a <c>wrap</c> from one operand
    ///     (<c>a, wrap = _makearray(a)</c>) and applies it to SOME of its results: the factors and the vectors come
    ///     back as matrices, the values (eigenvalues, singular values) and the scalars as plain arrays.
    /// </summary>
    /// <remarks>
    ///     Every expectation was printed by NumPy 2.4.2 for the same call with
    ///     <c>sq = np.matrix([[2., 1.], [1., 3.]])</c>, <c>sa = np.asarray(sq)</c> and
    ///     <c>m = np.matrix([[1., 2., 3.], [4., 5., 6.]])</c>. The values themselves are the subject of the
    ///     live-parity classes of this project and of the <c>linalg_parity</c> oracle tier; here only what the
    ///     matrix boundary adds is asserted. The functions with a managed fallback (solve, inv, det) are covered
    ///     without a backend by <c>NDMatrixBoundaryTests</c> in the unit test project.
    /// </remarks>
    [TestClass]
    public class MatrixOperandLapackTests
    {
        /// <summary>Skips the test on a host without the LAPACK backend.</summary>
        private static void RequireLapack()
        {
            if (!OpenBlasEngine.LapackAvailable)
                Assert.Inconclusive("OpenBLAS LAPACK backend not available on this host.");
        }

        /// <summary>The probe's symmetric positive definite matrix as the reference a function receives.</summary>
        /// <returns>A new matrix object <c>[[2, 1], [1, 3]]</c> (float64) typed <see cref="NDArray"/>.</returns>
        private static NDArray Sq() => np.matrix(np.array(new double[,] { { 2, 1 }, { 1, 3 } }));

        /// <summary>The same values as a plain array.</summary>
        /// <returns>A new plain array <c>[[2, 1], [1, 3]]</c> (float64).</returns>
        private static NDArray Sa() => np.array(new double[,] { { 2, 1 }, { 1, 3 } });

        /// <summary>The probe's 2 x 3 matrix.</summary>
        /// <returns>A new matrix object <c>[[1, 2, 3], [4, 5, 6]]</c> (float64) typed <see cref="NDArray"/>.</returns>
        private static NDArray M() => np.matrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));

        /// <summary>Asserts the class and the shape of a result.</summary>
        /// <param name="result">What the function returned.</param>
        /// <param name="matrix">Whether NumPy returns a <c>numpy.matrix</c>.</param>
        /// <param name="shape">NumPy's shape.</param>
        private static void Check(NDArray result, bool matrix, params long[] shape)
        {
            (result is NDMatrix).Should().Be(matrix, "NumPy returns {0}", matrix ? "a matrix" : "a plain array");
            result.shape.Should().Equal(shape);
        }

        /// <summary>np.linalg.qr returns wrap(q), wrap(r); mode 'r' returns wrap(r).</summary>
        [TestMethod]
        public void Qr_ReturnsMatrices()
        {
            RequireLapack();
            var (q, r) = np.linalg.qr(Sq());
            Check(q, true, 2, 2);
            Check(r, true, 2, 2);
            np.allclose(np.matmul(q, r), Sa()).Should().BeTrue();

            var (_, onlyR) = np.linalg.qr(Sq(), "r");
            Check(onlyR, true, 2, 2);

            var (plainQ, plainR) = np.linalg.qr(Sa());
            (plainQ is NDMatrix).Should().BeFalse();
            (plainR is NDMatrix).Should().BeFalse();
        }

        /// <summary>np.linalg.eig / eigh return plain eigenvalues and the eigenvectors as a matrix; eigvals and
        ///     eigvalsh return plain arrays.</summary>
        [TestMethod]
        public void Eig_ReturnsPlainValuesAndMatrixVectors()
        {
            RequireLapack();
            var (w, v) = np.linalg.eig(Sq());
            Check(w, false, 2);
            Check(v, true, 2, 2);

            var (wh, vh) = np.linalg.eigh(Sq());
            Check(wh, false, 2);
            Check(vh, true, 2, 2);

            Check(np.linalg.eigvals(Sq()), false, 2);
            Check(np.linalg.eigvalsh(Sq()), false, 2);
        }

        /// <summary>np.linalg.svd returns wrap(u), the plain singular values and wrap(vh); svdvals is plain.</summary>
        [TestMethod]
        public void Svd_ReturnsMatrixFactorsAndPlainValues()
        {
            RequireLapack();
            var (u, s, vh) = np.linalg.svd(Sq());
            Check(u, true, 2, 2);
            Check(s, false, 2);
            Check(vh, true, 2, 2);

            Check(np.linalg.svdvals(Sq()), false, 2);
        }

        /// <summary>np.linalg.lstsq takes its wrap from b: a matrix b gives the solution and the residuals as
        ///     matrices (a 1 x 0 matrix for the empty residuals), a matrix a with a plain b plain arrays.</summary>
        [TestMethod]
        public void Lstsq_FollowsB()
        {
            RequireLapack();
            var (x, residuals, _, s) = np.linalg.lstsq(Sa(), Sq());
            Check(x, true, 2, 2);
            Check(residuals, true, 1, 0);
            (s is NDMatrix).Should().BeFalse();

            var (plainX, plainResiduals, _, _) = np.linalg.lstsq(Sq(), Sa());
            Check(plainX, false, 2, 2);
            Check(plainResiduals, false, 0);
        }

        /// <summary>np.linalg.pinv and cholesky return a matrix; the scalar functions return scalars.</summary>
        [TestMethod]
        public void Pinv_Cholesky_ReturnMatrices_ScalarsStayScalars()
        {
            RequireLapack();
            Check(np.linalg.pinv(M()), true, 3, 2);
            Check(np.linalg.cholesky(Sq()), true, 2, 2);
            Check(np.linalg.inv(Sq()), true, 2, 2);
            Check(np.linalg.solve(Sa(), Sq()), true, 2, 2);
            Check(np.linalg.solve(Sq(), Sa()), false, 2, 2);

            np.linalg.det(Sq()).ndim.Should().Be(0);
            np.linalg.cond(Sq()).ndim.Should().Be(0);
            np.linalg.matrix_rank(Sq()).ndim.Should().Be(0);
            var (sign, logdet) = np.linalg.slogdet(Sq());
            sign.ndim.Should().Be(0);
            logdet.ndim.Should().Be(0);
        }
    }
}
