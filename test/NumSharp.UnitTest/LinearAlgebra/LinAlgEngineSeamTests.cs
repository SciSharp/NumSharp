using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.LinearAlgebra
{
    /// <summary>
    ///     Where <c>np.linalg</c> stops without a LAPACK backend — and, just as importantly, where
    ///     it does NOT stop.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Delete a case from the first region as each implementation lands.</b> Every
    ///     <see cref="NotSupportedException"/> asserted below is a placeholder for numerics that do
    ///     not exist yet; the assertions in the later regions are permanent.
    ///     </para>
    ///     <para>
    ///     The line these tests draw is the one <c>IBlasBackend</c> documents: the matrix PRODUCTS
    ///     always have a managed kernel to fall back to, so a backend changes which implementation
    ///     runs and nothing else, while the FACTORISATIONS have no managed fallback at all.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class LinAlgEngineSeamTests
    {
        private static NDArray Sq2 => np.arange(4.0).reshape(2, 2);
        private static NDArray M23 => np.arange(6.0).reshape(2, 3);

        #region pending numerics — remove a case when its implementation lands

        [TestMethod]
        public void EveryFactorisation_RaisesNotSupported_UntilABackendIsInstalled()
        {
            var sq = Sq2;
            var m = M23;

            new Action(() => np.linalg.inv(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.det(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.slogdet(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.cholesky(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.eig(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.eigvals(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.eigh(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.eigvalsh(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.qr(m)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.svd(m)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.svdvals(m)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.pinv(m)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.solve(sq, np.arange(2.0))).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.lstsq(m, np.arange(2.0))).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.cond(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.matrix_rank(sq)).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.tensorinv(np.eye(4).reshape(2, 2, 2, 2))).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.tensorsolve(np.eye(4).reshape(2, 2, 2, 2), np.ones(new Shape(2, 2))))
                .Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void TheMessage_NamesTheApi_TheLapackRoutine_AndTheSeamMemberToImplement()
        {
            // An exception that only says "not supported" leaves the reader nowhere; this one has
            // to say what to install.
            new Action(() => np.linalg.inv(Sq2))
                .Should().Throw<NotSupportedException>()
                .Which.Message.Should().Contain("np.linalg.inv")
                .And.Contain("gesv")
                .And.Contain(nameof(IBlasBackend.TryInv))
                .And.Contain("TensorEngine.Blas");
        }

        [TestMethod]
        public void OnlyTheThreeSingularValueOrders_StopInNorm()
        {
            new Action(() => np.linalg.norm(M23, 2, new[] {0, 1})).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.norm(M23, -2, new[] {0, 1})).Should().Throw<NotSupportedException>();
            new Action(() => np.linalg.norm(M23, "nuc", new[] {0, 1})).Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void Einsum_IsASignatureStub_AndSaysSoDistinctly()
        {
            // Not waiting on a backend — waiting on a subscript parser. The message must not read
            // like the LAPACK ones or a reader will go looking for a package to install.
            new Action(() => np.einsum("ij,jk->ik", M23, np.arange(6.0).reshape(3, 2)))
                .Should().Throw<NotSupportedException>()
                .Which.Message.Should().Contain("np.einsum").And.Contain("np.tensordot");
        }

        [TestMethod]
        public void MatrixPower_StopsOnlyOnANegativeExponent()
        {
            // a**-n is inv(a)**n, so this is the one power that needs a backend.
            new Action(() => np.linalg.matrix_power(Sq2, -1)).Should().Throw<NotSupportedException>();
        }

        #endregion

        #region permanent — these compose out of primitives NumSharp already has

        [TestMethod]
        public void MatrixPower_NonNegativeExponents_Work()
        {
            np.linalg.matrix_power(Sq2, 1).Should().BeOfValues(0, 1, 2, 3).And.BeShaped(2, 2);
            np.linalg.matrix_power(Sq2, 2).Should().BeOfValues(2, 3, 6, 11).And.BeShaped(2, 2);
            np.linalg.matrix_power(Sq2, 3).Should().BeOfValues(6, 11, 22, 39).And.BeShaped(2, 2);
            // Binary exponentiation, so this is three multiplies rather than four.
            np.linalg.matrix_power(Sq2, 5).Should().BeOfValues(78, 139, 278, 495).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void MatrixPower_ZeroGivesTheIdentityInTheOperandsOwnDtype()
        {
            var identity = np.linalg.matrix_power(np.eye(3).astype(np.float32) * 2.0f, 0);
            identity.Should().BeOfValues(1, 0, 0, 0, 1, 0, 0, 0, 1).And.BeShaped(3, 3);
            identity.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void NDArrayMatrixPower_NoLongerRejectsNegativePowersOutright()
        {
            // It used to raise a bare Exception("matrix_power just work with int >= 0"), which was
            // never NumPy's rule. It now takes the inv() route — and so reports a MISSING BACKEND.
            new Action(() => np.arange(4.0).reshape(2, 2).matrix_power(-1))
                .Should().Throw<NotSupportedException>();

            np.arange(4.0).reshape(2, 2).matrix_power(2).Should().BeOfValues(2, 3, 6, 11);
        }

        [TestMethod]
        public void MultiDot_ChainsAnyNumberOfMatrices_AndUnwrapsVectorEndpoints()
        {
            np.linalg.multi_dot(new[] {np.ones(new Shape(2, 3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4, 5))})
                .Should().BeShaped(2, 5);

            np.linalg.multi_dot(new[]
                    {np.ones(new Shape(2, 3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4, 2)), np.ones(new Shape(2, 6))})
                .Should().BeShaped(2, 6);

            // A 1-D first operand is a row, a 1-D last operand a column; both axes are dropped again.
            np.linalg.multi_dot(new[] {np.ones(new Shape(3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4, 2))})
                .Should().BeShaped(2);
            np.linalg.multi_dot(new[] {np.ones(new Shape(2, 3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4))})
                .Should().BeShaped(2);
            np.linalg.multi_dot(new[] {np.ones(new Shape(3)), np.ones(new Shape(3, 4)), np.ones(new Shape(4))})
                .Should().BeOfValues(12).And.BeShaped();
        }

        [TestMethod]
        public void MultiDot_MatchesNumPyValues_AcrossChainLengthsAndDtypes()
        {
            // Both parenthesisation paths — the three-matrix shortcut and the Cormen DP (n >= 4) —
            // replayed against NumPy 2.4.2.
            np.linalg.multi_dot(new[]
                    {np.arange(6.0).reshape(2, 3), np.arange(12.0).reshape(3, 4), np.arange(8.0).reshape(4, 2)})
                .Should().BeOfValues(324, 422, 1008, 1304).And.BeShaped(2, 2);

            np.linalg.multi_dot(new[]
            {
                np.arange(6.0).reshape(2, 3), np.arange(15.0).reshape(3, 5), np.arange(10.0).reshape(5, 2),
                np.arange(8.0).reshape(2, 4), np.arange(12.0).reshape(4, 3)
            }).Should().BeOfValues(123750, 146200, 168650, 384300, 454000, 523700).And.BeShaped(2, 3);

            // An all-int64 chain stays integer, exactly as NumPy's dot preserves it.
            np.linalg.multi_dot(new[]
            {
                np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3),
                np.arange(12).astype(NPTypeCode.Int64).reshape(3, 4),
                np.arange(8).astype(NPTypeCode.Int64).reshape(4, 2)
            }).Should().BeOfValues(324, 422, 1008, 1304).And.BeShaped(2, 2);
        }

        [TestMethod]
        public void MultiDot_OutReceivesTheTwoDimensionalProduct_AndReturnsAReshapedView()
        {
            var A = np.arange(6.0).reshape(2, 3);
            var B = np.arange(12.0).reshape(3, 4);
            var C = np.arange(8.0).reshape(4, 2);

            // Plain 2-D result: out has the result shape, is filled, and is what comes back.
            var o22 = np.empty(new Shape(2, 2));
            np.linalg.multi_dot(new[] {A, B, C}, o22).Should().BeShaped(2, 2).And.BeOfValues(324, 422, 1008, 1304);
            o22.Should().BeOfValues(324, 422, 1008, 1304);

            // NumPy threads out into the FINAL dot, so with a 1-D first operand `out` is the 2-D
            // (1, k) shape and the array returned is a raveled (k,) VIEW of it — not (k,) in and out.
            var o12 = np.empty(new Shape(1, 2));
            np.linalg.multi_dot(new[] {np.arange(3.0), B, C}, o12)
                .Should().BeShaped(2).And.BeOfValues(324, 422);
            o12.Should().BeOfValues(324, 422);
        }

        [TestMethod]
        public void ArrayApiForms_ReduceTheLASTTwoAxes_WhereTheirMainNamespaceTwinsTakeTheFirst()
        {
            var stack = np.arange(18.0).reshape(2, 3, 3);

            // One trace per matrix...
            np.linalg.trace(stack).Should().BeOfValues(12, 39).And.BeShaped(2);
            // ...versus np.trace's first-two-axes default, which is a different answer entirely.
            np.trace(stack).Should().BeOfValues(12, 14, 16).And.BeShaped(3);

            np.linalg.diagonal(stack).Should().BeOfValues(0, 4, 8, 9, 13, 17).And.BeShaped(2, 3);
            np.diagonal(stack).Should().BeShaped(3, 2);
        }

        [TestMethod]
        public void Cross_IsTheThreeVectorProduct_OverAnyAxis()
        {
            np.linalg.cross(np.array(new[] {1.0, 2.0, 3.0}), np.array(new[] {4.0, 5.0, 6.0}))
                .Should().BeOfValues(-3, 6, -3).And.BeShaped(3);

            np.linalg.cross(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3) + 1.0)
                .Should().BeOfValues(-1, 2, -1, -1, 2, -1).And.BeShaped(2, 3);

            np.linalg.cross(np.arange(6.0).reshape(3, 2), np.arange(6.0).reshape(3, 2) + 1.0, 0)
                .Should().BeOfValues(-2, -2, 4, 4, -2, -2).And.BeShaped(3, 2);
        }

        [TestMethod]
        public void Norm_EveryOrderThatIsNotDefinedBySingularValues_Computes()
        {
            var m = M23;

            np.linalg.norm(m).Should().BeOfValues(7.416198487095663);
            np.linalg.norm(m, null, 0).Should().BeShaped(3);
            np.linalg.norm(m, null, 0, true).Should().BeShaped(1, 3);
            np.linalg.norm(m, 1, new[] {0, 1}).Should().BeOfValues(7);          // max column sum
            np.linalg.norm(m, -1, new[] {0, 1}).Should().BeOfValues(3);         // min column sum
            np.linalg.norm(m, double.PositiveInfinity, new[] {0, 1}).Should().BeOfValues(12);
            np.linalg.norm(m, double.NegativeInfinity, new[] {0, 1}).Should().BeOfValues(3);
            np.linalg.norm(m, "fro", new[] {0, 1}).Should().BeOfValues(7.416198487095663);

            var v = np.array(new[] {1.0, -5.0, 3.0});
            np.linalg.norm(v, double.PositiveInfinity, 0).Should().BeOfValues(5);
            np.linalg.norm(v, double.NegativeInfinity, 0).Should().BeOfValues(1);
            np.linalg.norm(v, 1, 0).Should().BeOfValues(9);
            np.linalg.norm(np.array(new[] {0.0, 1.0, 2.0}), 0, 0).Should().BeOfValues(2);
        }

        [TestMethod]
        public void MatrixRank_BelowRankTwo_IsAPredicateNotACount()
        {
            // NumPy short-circuits with `int(not all(A == 0))`, so a vector is rank 1 unless every
            // element is zero — matrix_rank([1,2,3]) is 1, NOT 3. Getting this wrong returns a
            // plausible number for every non-degenerate input, which is why it needs its own test.
            np.linalg.matrix_rank(np.array(new[] {1.0, 2.0, 3.0})).Should().BeOfValues(1).And.BeShaped();
            np.linalg.matrix_rank(np.array(new[] {0.0, 5.0, 0.0})).Should().BeOfValues(1);
            np.linalg.matrix_rank(np.zeros(new Shape(3))).Should().BeOfValues(0);
            np.linalg.matrix_rank(NDArray.Scalar(7.0)).Should().BeOfValues(1);
            np.linalg.matrix_rank(NDArray.Scalar(0.0)).Should().BeOfValues(0);

            // all([]) is true, so an empty operand is rank 0.
            np.linalg.matrix_rank(np.zeros(new Shape(0))).Should().BeOfValues(0);

            // NaN is not zero, so it counts as content.
            np.linalg.matrix_rank(np.array(new[] {double.NaN, 0.0})).Should().BeOfValues(1);

            np.linalg.matrix_rank(np.array(new[] {1.0, 2.0})).dtype.Should().Be(typeof(long));
        }

        [TestMethod]
        public void Norm_ComputesIntegerOperandsInFloatingPoint()
        {
            var result = np.linalg.norm(np.arange(3));
            result.dtype.Should().Be(typeof(double));
            result.Should().BeOfValues(2.23606797749979);
        }

        [TestMethod]
        public void VectorNormAndMatrixNorm_CarryTheirOwnDefaults()
        {
            var m = M23;
            np.linalg.vector_norm(m).Should().BeOfValues(7.416198487095663).And.BeShaped();
            np.linalg.vector_norm(m, null, true).Should().BeShaped(1, 1);
            np.linalg.vector_norm(m, new[] {0}).Should().BeShaped(3);

            // NumPy's axis= takes an int as readily as a tuple, so both spellings exist.
            np.linalg.vector_norm(m, 0).Should().BeOfValues(3, 4.123105625617661, 5.385164807134504);
            np.linalg.vector_norm(m, ord: 1).Should().BeOfValues(15);

            // matrix_norm always takes the last two axes, so a stack gives one value per matrix.
            np.linalg.matrix_norm(m).Should().BeOfValues(7.416198487095663);
            np.linalg.matrix_norm(m, ord: 1).Should().BeOfValues(7);
            np.linalg.matrix_norm(np.arange(18.0).reshape(2, 3, 3)).Should().BeShaped(2);
        }

        #endregion

        #region the seam itself

        [TestMethod]
        public void TheProductFamily_NeedsNoBackend_AndTheFactorisationsDo()
        {
            // The invariant IBlasBackend states, exercised from both sides in one place: with
            // TensorEngine.Blas left null (the default for a plain NumSharp.Core), the products
            // still answer and only the factorisations stop.
            var engine = np.arange(4.0).reshape(2, 2).TensorEngine;
            engine.Blas.Should().BeNull("NumSharp.Core installs no backend of its own");

            np.inner(np.arange(3.0), np.arange(3.0)).Should().BeOfValues(5);
            np.vdot(np.arange(3.0), np.arange(3.0)).Should().BeOfValues(5);
            np.vecdot(np.arange(3.0), np.arange(3.0)).Should().BeOfValues(5);
            np.matvec(M23, np.arange(3.0)).Should().BeOfValues(5, 14);
            np.vecmat(np.arange(2.0), M23).Should().BeOfValues(3, 4, 5);

            new Action(() => np.linalg.inv(Sq2)).Should().Throw<NotSupportedException>();
        }

        [TestMethod]
        public void EveryNewSeamMemberDefaultsToDeclining_SoAnExistingBackendKeepsCompiling()
        {
            // The whole reason the linalg half of IBlasBackend is spelled with default
            // implementations: a backend that predates it — OpenBlasBackend serves dot/matmul at
            // float32/float64 and nothing else — must not have to mention any of them.
            //
            // The variable is typed as the INTERFACE deliberately: a default interface member is
            // interface-dispatched and is not visible on the implementing class, which is exactly
            // why adding these members could not break an existing backend's own surface.
            IBlasBackend declining = new DecliningBackend();

            declining.TryInv(Sq2, out _).Should().BeFalse();
            declining.TryDet(Sq2, out _).Should().BeFalse();
            declining.TrySvd(Sq2, false, true, out _, out _, out _).Should().BeFalse();
            declining.TryQr(Sq2, "reduced", out _, out _).Should().BeFalse();
            declining.TryEig(Sq2, true, out _, out _).Should().BeFalse();
            declining.TryEigh(Sq2, 'L', true, out _, out _).Should().BeFalse();
            declining.TryCholesky(Sq2, false, out _).Should().BeFalse();
            declining.TrySlogdet(Sq2, out _, out _).Should().BeFalse();
            declining.TrySolve(Sq2, Sq2, false, out _).Should().BeFalse();
            declining.TryLstsq(Sq2, Sq2, -1d, out _, out _, out _, out _).Should().BeFalse();
            declining.TryInner(Sq2, Sq2, out _).Should().BeFalse();
            declining.TryVdot(Sq2, Sq2, out _).Should().BeFalse();
            declining.TryVecdot(Sq2, Sq2, out _).Should().BeFalse();
            declining.TryMatvec(Sq2, Sq2, out _).Should().BeFalse();
            declining.TryVecmat(Sq2, Sq2, out _).Should().BeFalse();
        }

        /// <summary>A backend that implements only what the interface has always required.</summary>
        private sealed class DecliningBackend : IBlasBackend
        {
            public string Info => "declines everything";

            public bool TryDot(NDArray left, NDArray right, out NDArray result)
            {
                result = null;
                return false;
            }

            public bool TryMatMul2D(NDArray left, NDArray right, NDArray result) => false;
        }

        #endregion
    }
}
