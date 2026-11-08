using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp.Backends.Iteration;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The np.linalg factorisations over EMPTY operands (an empty core matrix or a zero-size batch) without a matrix
    ///     backend — the open-bugs rundown's bug 5 (plan api-parameter-parity X34). NumPy answers them without arithmetic,
    ///     so NumSharp.Core gives every one (<c>Backends/Default/LinearAlgebra/LinAlgEmpty.cs</c>); they used to throw
    ///     <see cref="OpenBlasMissingBackendException"/>. Every expected value, dtype and error text is NumPy 2.4.2's; the
    ///     <c>linalg_empty.jsonl</c> oracle tier gates the whole matrix, these pin one case of each rule.
    /// </summary>
    /// <remarks>
    ///     These tests run with whatever backend the process has (none in a plain test host): the empty path is taken
    ///     BEFORE the engine asks a backend, so the answer is the same either way.
    /// </remarks>
    [TestClass]
    public class np_linalg_empty_numpy_port_tests
    {
        /// <summary>Asserts an array's dtype and dimensions (a type-exact check of each).</summary>
        /// <param name="a">The result.</param>
        /// <param name="dtype">NumPy's dtype.</param>
        /// <param name="dims">NumPy's shape.</param>
        private static void Is(NDArray a, DType dtype, params long[] dims)
        {
            a.Should().NotBeNull();
            a.dtype.Should().Be(dtype);
            a.shape.Should().Equal(dims);
        }

        /// <summary>
        ///     A full SVD of an empty MATRIX is not empty: LAPACK leaves U / VT uninitialised and NumPy's
        ///     <c>svd_wrapper</c> writes an identity into the non-empty one. Reduced factors and the singular values are
        ///     empty; S is the real type (float64 for complex128), U / Vh keep the operand's.
        /// </summary>
        [TestMethod]
        public void Svd_FullOfAnEmptyMatrix_CarriesNumPysIdentity()
        {
            var (u, s, vh) = np.linalg.svd(np.zeros(new Shape(0, 3)));
            Is(u, np.float64, 0, 0);
            Is(s, np.float64, 0);
            Is(vh, np.float64, 3, 3);
            vh.ToArray<double>().Should().Equal(1, 0, 0, 0, 1, 0, 0, 0, 1);

            var (u2, s2, vh2) = np.linalg.svd(np.zeros(new Shape(2, 3, 0), np.complex128));
            Is(u2, np.complex128, 2, 3, 3);
            Is(s2, np.float64, 2, 0);
            Is(vh2, np.complex128, 2, 0, 0);
            u2.ToArray<Complex>().Should().Equal(
                1, 0, 0, 0, 1, 0, 0, 0, 1,
                1, 0, 0, 0, 1, 0, 0, 0, 1);

            var (ur, sr, vhr) = np.linalg.svd(np.zeros(new Shape(0, 3), np.float32), full_matrices: false);
            Is(ur, np.float32, 0, 0);
            Is(sr, np.float32, 0);
            Is(vhr, np.float32, 0, 3);
            Is(np.linalg.svdvals(np.zeros(new Shape(0, 2, 3))), np.float64, 0, 2);
            Is(np.linalg.svd(np.zeros(new Shape(3, 0), np.int64), compute_uv: false).S, np.float64, 0);
        }

        /// <summary>
        ///     QR's per-mode shapes, and the complete Q of an M×0 matrix is the M×M identity (<c>orgqr</c> with no
        ///     reflectors). The raw mode's h is the transposed packed matrix, tau has K = min(M, N) entries.
        /// </summary>
        [TestMethod]
        public void Qr_EveryMode_AndTheCompleteIdentity()
        {
            var (q, r) = np.linalg.qr(np.zeros(new Shape(3, 0)), "complete");
            Is(q, np.float64, 3, 3);
            q.ToArray<double>().Should().Equal(1, 0, 0, 0, 1, 0, 0, 0, 1);
            Is(r, np.float64, 3, 0);

            var (q2, r2) = np.linalg.qr(np.zeros(new Shape(3, 0)));
            Is(q2, np.float64, 3, 0);
            Is(r2, np.float64, 0, 0);

            var (h, tau) = np.linalg.qr(np.zeros(new Shape(0, 3), np.complex128), "raw");
            Is(h, np.complex128, 3, 0);
            Is(tau, np.complex128, 0);

            Is(np.linalg.qr(np.zeros(new Shape(0, 2, 3), np.float32), "r").R, np.float32, 0, 2, 3);
            Is(np.linalg.qr(np.zeros(new Shape(2, 0, 3)), "economic").Q, np.float64, 2, 0, 3);
        }

        /// <summary>
        ///     eig's result of an empty REAL operand is real (NumPy's <c>all(w.imag == 0)</c> is vacuously true); eigh's
        ///     eigenvalues are the real type; cholesky keeps the operand's dtype; a zero-size batch keeps its core shape.
        /// </summary>
        [TestMethod]
        public void Eigensolvers_And_Cholesky_OfEmpties()
        {
            var (w, v) = np.linalg.eig(np.zeros(new Shape(0, 0)));
            Is(w, np.float64, 0);
            Is(v, np.float64, 0, 0);
            Is(np.linalg.eigvals(np.zeros(new Shape(0, 3, 3), np.float32)), np.float32, 0, 3);
            Is(np.linalg.eigvals(np.zeros(new Shape(2, 0, 0), np.complex128)), np.complex128, 2, 0);

            var (wh, vh) = np.linalg.eigh(np.zeros(new Shape(0, 0), np.complex128));
            Is(wh, np.float64, 0);
            Is(vh, np.complex128, 0, 0);
            Is(np.linalg.eigvalsh(np.zeros(new Shape(2, 1, 0, 0), np.int64), 'U'), np.float64, 2, 1, 0);

            Is(np.linalg.cholesky(np.zeros(new Shape(2, 0, 0), np.float32)), np.float32, 2, 0, 0);
            Is(np.linalg.cholesky(np.zeros(new Shape(0, 3, 3)), upper: true), np.float64, 0, 3, 3);
        }

        /// <summary>
        ///     lstsq of an empty coefficient matrix: x zeros, rank int32 0, no singular values; for an M×0 matrix the
        ///     residuals are NumPy's sequential sum of squares of each column of b (computed in double, cast to the real
        ///     result type), for 0×N they are discarded (empty).
        /// </summary>
        [TestMethod]
        public void Lstsq_OfAnEmptyMatrix()
        {
            var (x, res, rank, s) = np.linalg.lstsq(np.zeros(new Shape(3, 0)), np.array(new[] { 1.0, 2.0, 3.0 }));
            Is(x, np.float64, 0);
            Is(res, np.float64, 1);
            res.GetDouble(0).Should().Be(14.0);
            Is(rank, np.int32);
            rank.GetAtIndex(0).Should().Be(0);
            Is(s, np.float64, 0);

            var b = np.array(new float[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
            var (x2, res2, _, s2) = np.linalg.lstsq(np.zeros(new Shape(3, 0), np.float32), b);
            Is(x2, np.float32, 0, 2);
            Is(res2, np.float32, 2);
            res2.ToArray<float>().Should().Equal(35f, 56f);
            Is(s2, np.float32, 0);

            var (x3, res3, _, _) = np.linalg.lstsq(np.zeros(new Shape(0, 3), np.complex128), np.zeros(new Shape(0, 2), np.complex128));
            Is(x3, np.complex128, 3, 2);
            Is(res3, np.float64, 0);

            // The ORDER of the residual sum, on a single column of 20 rows (NumPy 2.4.2's bits): one large square and
            // nineteen ones. NumPy's loop adds them in sequence, so every 1 is lost to 1e16's rounding (ulp 2, ties to
            // even) and the residual is exactly 1e16 (complex: 2e16). np.sum over one column is a pairwise sum, which
            // keeps them (1e16 + 16), so this pin fails if the residuals are ever summed with np.sum's flat schedule.
            // A column of three rows, or two columns of any length, cannot show it: both orders agree there.
            var tall = new double[20];
            tall[0] = 1e8;
            for (int i = 1; i < tall.Length; i++)
                tall[i] = 1.0;
            var (_, res4, _, _) = np.linalg.lstsq(np.zeros(new Shape(20, 0)), np.array(tall));
            Is(res4, np.float64, 1);
            BitConverter.DoubleToInt64Bits(res4.GetDouble(0)).Should().Be(BitConverter.DoubleToInt64Bits(1e16));
            using var tallComplex = np.array(tall).astype(np.complex128) * new Complex(1, 1);
            var (_, res5, _, _) = np.linalg.lstsq(np.zeros(new Shape(20, 0), np.complex128), tallComplex.reshape(20, 1));
            Is(res5, np.float64, 1);
            BitConverter.DoubleToInt64Bits(res5.GetDouble(0)).Should().Be(BitConverter.DoubleToInt64Bits(2e16));
        }

        /// <summary>
        ///     The composites follow NumPy's own statements over the empty factorisations: matrix_rank of an empty matrix
        ///     raises <c>S.max()</c>'s zero-size ValueError (with tol= it counts 0), the 2-norm is 0 through
        ///     <c>amax(initial=0)</c> while -2 raises <c>amin</c>'s, cond refuses an empty matrix, pinv of an empty matrix
        ///     keeps the operand's dtype, and a zero-size batch gives empty results.
        /// </summary>
        [TestMethod]
        public void Composites_OfEmpties()
        {
            Action rank = () => np.linalg.matrix_rank(np.zeros(new Shape(0, 3)));
            rank.Should().Throw<ValueError>().Which.Message.Should().Be(
                "zero-size array to reduction operation maximum which has no identity");
            Convert.ToInt64(np.linalg.matrix_rank(np.zeros(new Shape(0, 3)), tol: 0.5).GetAtIndex(0)).Should().Be(0);
            Is(np.linalg.matrix_rank(np.zeros(new Shape(0, 3, 3))), np.int64, 0);

            Convert.ToDouble(np.linalg.norm(np.zeros(new Shape(0, 3)), 2, new[] { -2, -1 }).GetAtIndex(0)).Should().Be(0.0);
            Action minus2 = () => np.linalg.norm(np.zeros(new Shape(0, 3)), -2, new[] { -2, -1 });
            minus2.Should().Throw<ValueError>().Which.Message.Should().Be(
                "zero-size array to reduction operation minimum which has no identity");
            Is(np.linalg.norm(np.zeros(new Shape(2, 0, 0), np.complex128), "fro", new[] { -2, -1 }), np.float64, 2);

            Action cond = () => np.linalg.cond(np.zeros(new Shape(0, 3)));
            cond.Should().Throw<LinAlgError>().Which.Message.Should().Be("cond is not defined on empty arrays");
            Is(np.linalg.cond(np.zeros(new Shape(0, 3, 3), np.float32)), np.float32, 0);

            Is(np.linalg.pinv(np.zeros(new Shape(0, 3), np.int64)), np.int64, 3, 0);
            Is(np.linalg.pinv(np.zeros(new Shape(0, 2, 3))), np.float64, 0, 3, 2);
            Action square = () => np.linalg.eig(np.zeros(new Shape(0, 3)));
            square.Should().Throw<LinAlgError>().Which.Message.Should().Be("Last 2 dimensions of the array must be square");
        }

        /// <summary>
        ///     The iterator neighbour the empty Frobenius norm found: NumPy marks a written operand's axis as a reduction
        ///     only where the OPERAND is broadcast (its dimension is 1 or missing), not wherever a stride is 0 — a zero-size
        ///     array's strides are 0 without a broadcast. Its error is NumPy's ValueError numbering the axis from the inside.
        /// </summary>
        [TestMethod]
        public void NDIter_ZeroStrideIsNotAReduction_TheBroadcastIs()
        {
            // A (3, 0) array read-write: NumPy iterates it (nothing to visit). The typed iterator threw.
            int seen = 0;
            foreach (ref double _ in np.nditer<double>(np.zeros(new Shape(3, 0)), writeable: true))
                seen++;
            seen.Should().Be(0);
            using (var it = np.nditer(np.zeros(new Shape(3, 0)), new[] { "zerosize_ok" }, new[] { "readwrite" }))
                it.itersize.Should().Be(0);

            // A writeable array with an explicit 0 stride is not a broadcast either: NumPy writes the one element thrice.
            var buffer = new byte[8];
            var shared = np.ndarray(new Shape(3), np.float64, buffer, strides: new long[] { 0 });
            int n = 0;
            foreach (ref double x in np.nditer<double>(shared, writeable: true))
                x = n++;
            n.Should().Be(3);
            shared.ToArray<double>().Should().Equal(2.0, 2.0, 2.0);

            // A genuinely broadcast output still refuses, with NumPy's axis numbering (innermost first).
            Action empty = () => np.nditer(new[] { np.zeros(new Shape(3, 0)), np.zeros(new Shape(1, 0)) },
                new[] { "zerosize_ok" }, new[] { new[] { "readonly" }, new[] { "readwrite" } }).Dispose();
            empty.Should().Throw<ValueError>().Which.Message.Should().Be(
                "output operand requires a reduction along dimension 1, but the reduction is not enabled. " +
                "The dimension size of 1 does not match the expected output shape.");
            foreach (var (outShape, dim) in new[] { (new long[] { 1, 3, 4 }, 2), (new long[] { 2, 1, 4 }, 1),
                                                    (new long[] { 2, 3, 1 }, 0), (new long[] { 1, 1, 4 }, 1),
                                                    (new long[] { 3, 4 }, 2) })
            {
                Action act = () => NDIterRef.MultiNew(2, new[] { np.zeros(new Shape(2, 3, 4)), np.zeros(new Shape(outShape)) },
                    NDIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READWRITE }).Dispose();
                act.Should().Throw<ValueError>().Which.Message.Should().Be(
                    $"output operand requires a reduction along dimension {dim}, but the reduction is not enabled. " +
                    "The dimension size of 1 does not match the expected output shape.");
            }
        }
    }
}
