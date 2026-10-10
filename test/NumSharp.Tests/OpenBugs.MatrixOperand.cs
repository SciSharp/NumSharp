using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Open gaps of the matrix boundary (<see cref="NDMatrixBoundary"/>) that the oracle tier
    ///     <c>matrix_operand.jsonl</c> cannot hold: functions whose result is a struct of arrays, where NumPy
    ///     2.4.2 returns <c>numpy.matrix</c> objects for a matrix operand and NumSharp returns plain arrays. Each
    ///     test asserts NumPy's answer, so it FAILS while the gap is open. Remove <c>[OpenBugs]</c> when it is
    ///     closed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     These functions carry a plain-array region, so their VALUES and their plain shapes are right and no
    ///     matrix semantics reach their internals; only the class of the returned arrays differs. The gaps the
    ///     tier does hold (MO01 to MO05) are replayed by <c>FuzzCorpusTests.MatrixOperand_KnownGaps</c>.
    ///     </para>
    ///     <para>
    ///     Probed on NumPy 2.4.2 with <c>m = np.matrix([[1., 2., 3.], [4., 5., 6.]])</c>:
    ///     <c>np.gradient(m)</c> is a list of two 2 x 3 matrices, <c>np.gradient(m, axis=0)</c> one 2 x 3 matrix,
    ///     and <c>np.meshgrid(m[0], m[0])</c> two 3 x 3 matrices.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class OpenBugsMatrixOperandTests
    {
        /// <summary>The probe's 2 x 3 float matrix, as the <see cref="NDArray"/> reference a function receives.</summary>
        /// <returns>A new matrix object <c>[[1, 2, 3], [4, 5, 6]]</c> (float64) typed <see cref="NDArray"/>.</returns>
        private static NDArray M() => np.matrix(np.array(new double[,] { { 1, 2, 3 }, { 4, 5, 6 } }));

        /// <summary>Gap MO06: np.gradient of a matrix returns matrices in NumPy (its differences are computed on
        ///     the matrix and keep its class); NumSharp's GradientResult holds plain arrays.</summary>
        [TestMethod]
        [OpenBugs]
        public void MO06_Gradient_OfAMatrix_ReturnsMatrices()
        {
            NDArray[] gradients = np.gradient(M());
            gradients.Length.Should().Be(2);
            foreach (NDArray g in gradients)
            {
                g.shape.Should().Equal(2L, 3L);
                (g is NDMatrix).Should().BeTrue("NumPy: np.gradient(m) is a list of matrices");
            }
        }

        /// <summary>Gap MO07: np.meshgrid of matrix rows returns matrices in NumPy (reshape and broadcast_arrays
        ///     with subok=True keep the class); NumSharp's MeshgridResult holds plain arrays.</summary>
        [TestMethod]
        [OpenBugs]
        public void MO07_Meshgrid_OfMatrixRows_ReturnsMatrices()
        {
            NDArray row = np.matrix(np.array(new double[,] { { 1, 2, 3 } }));
            NDArray[] grids = np.meshgrid(row, row);
            grids.Length.Should().Be(2);
            foreach (NDArray g in grids)
            {
                g.shape.Should().Equal(3L, 3L);
                (g is NDMatrix).Should().BeTrue("NumPy: np.meshgrid(m[0], m[0]) is a list of matrices");
            }
        }
    }
}
