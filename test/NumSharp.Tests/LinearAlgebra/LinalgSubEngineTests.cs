using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp.Backends;

namespace NumSharp.Tests.LinearAlgebra
{
    /// <summary>
    ///     The <see cref="TensorEngine.Linalg"/> sub-engine (the np-layer dtype plan's phase PE, E2): every
    ///     <c>np.linalg</c> function reaches the <see cref="LinalgEngine"/> assigned to the cached engine, arrays created
    ///     before the assignment included; a <see cref="DefaultLinalgEngine"/> subclass that overrides a kernel computes the
    ///     factorisation while NumPy's statements (checks, casts, tails) stay; null restores the default.
    /// </summary>
    /// <remarks>
    ///     Every test assigns the property of the process-wide cached engine, which every other test's arrays use, so the
    ///     class runs alone (<see cref="DoNotParallelizeAttribute"/>) and restores the default in a <c>finally</c>.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class LinalgSubEngineTests
    {
        /// <summary>
        ///     A probe over the default statements: each of the 31 members records its name (the second C# form of
        ///     tensordot and vector_norm under its own key) and then runs the default.
        /// </summary>
        private sealed class RecordingLinalg : DefaultLinalgEngine
        {
            public readonly HashSet<string> Calls = new();

            public override NDArray Cholesky(NDArray a, bool upper = false) { Calls.Add("cholesky"); return base.Cholesky(a, upper); }
            public override NDArray Cond(NDArray x, object p = null) { Calls.Add("cond"); return base.Cond(x, p); }
            public override NDArray Cross(NDArray x1, NDArray x2, int axis = -1) { Calls.Add("cross"); return base.Cross(x1, x2, axis); }
            public override NDArray Det(NDArray a) { Calls.Add("det"); return base.Det(a); }
            public override (NDArray eigenvalues, NDArray eigenvectors) Eig(NDArray a) { Calls.Add("eig"); return base.Eig(a); }
            public override (NDArray eigenvalues, NDArray eigenvectors) Eigh(NDArray a, char UPLO = 'L') { Calls.Add("eigh"); return base.Eigh(a, UPLO); }
            public override NDArray Eigvals(NDArray a) { Calls.Add("eigvals"); return base.Eigvals(a); }
            public override NDArray Eigvalsh(NDArray a, char UPLO = 'L') { Calls.Add("eigvalsh"); return base.Eigvalsh(a, UPLO); }
            public override NDArray Inv(NDArray a) { Calls.Add("inv"); return base.Inv(a); }
            public override (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(NDArray a, NDArray b, double? rcond = null) { Calls.Add("lstsq"); return base.Lstsq(a, b, rcond); }
            public override NDArray Matmul(NDArray x1, NDArray x2) { Calls.Add("matmul"); return base.Matmul(x1, x2); }
            public override NDArray MatrixNorm(NDArray x, bool keepdims = false, object ord = null) { Calls.Add("matrix_norm"); return base.MatrixNorm(x, keepdims, ord); }
            public override NDArray MatrixPower(NDArray a, int n) { Calls.Add("matrix_power"); return base.MatrixPower(a, n); }
            public override NDArray MatrixRank(NDArray A, double? tol = null, bool hermitian = false, double? rtol = null) { Calls.Add("matrix_rank"); return base.MatrixRank(A, tol, hermitian, rtol); }
            public override NDArray MultiDot(NDArray[] arrays, NDArray @out = null) { Calls.Add("multi_dot"); return base.MultiDot(arrays, @out); }
            public override NDArray Norm(NDArray x, object ord = null, int[] axis = null, bool keepdims = false) { Calls.Add("norm"); return base.Norm(x, ord, axis, keepdims); }
            public override NDArray Outer(NDArray x1, NDArray x2) { Calls.Add("outer"); return base.Outer(x1, x2); }
            public override NDArray Pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null) { Calls.Add("pinv"); return base.Pinv(a, rcond, hermitian, rtol); }
            public override (NDArray Q, NDArray R) Qr(NDArray a, string mode = "reduced") { Calls.Add("qr"); return base.Qr(a, mode); }
            public override (NDArray sign, NDArray logabsdet) Slogdet(NDArray a) { Calls.Add("slogdet"); return base.Slogdet(a); }
            public override NDArray Solve(NDArray a, NDArray b) { Calls.Add("solve"); return base.Solve(a, b); }
            public override (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool full_matrices = true, bool compute_uv = true, bool hermitian = false) { Calls.Add("svd"); return base.Svd(a, full_matrices, compute_uv, hermitian); }
            public override NDArray Svdvals(NDArray x) { Calls.Add("svdvals"); return base.Svdvals(x); }
            public override NDArray Tensordot(NDArray x1, NDArray x2, int axes = 2) { Calls.Add("tensordot"); return base.Tensordot(x1, x2, axes); }
            public override NDArray Tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB) { Calls.Add("tensordot(axes, axes)"); return base.Tensordot(x1, x2, axesA, axesB); }
            public override NDArray Tensorinv(NDArray a, int ind = 2) { Calls.Add("tensorinv"); return base.Tensorinv(a, ind); }
            public override NDArray Tensorsolve(NDArray a, NDArray b, int[] axes = null) { Calls.Add("tensorsolve"); return base.Tensorsolve(a, b, axes); }
            public override NDArray Trace(NDArray x, int offset = 0, DType dtype = null) { Calls.Add("trace"); return base.Trace(x, offset, dtype); }
            public override NDArray Vecdot(NDArray x1, NDArray x2, int axis = -1) { Calls.Add("vecdot"); return base.Vecdot(x1, x2, axis); }
            public override NDArray VectorNorm(NDArray x, int[] axis = null, bool keepdims = false, object ord = null) { Calls.Add("vector_norm"); return base.VectorNorm(x, axis, keepdims, ord); }
            public override NDArray VectorNorm(NDArray x, int axis, bool keepdims = false, object ord = null) { Calls.Add("vector_norm(int)"); return base.VectorNorm(x, axis, keepdims, ord); }
        }

        /// <summary>
        ///     A provider in miniature: it serves the Cholesky, eig and svd kernels itself (fixed answers that no LAPACK
        ///     gives), counts what reached them, and records the operands' dtypes, so a test can see that NumPy's statements
        ///     ran around the kernel.
        /// </summary>
        private sealed class KernelProvider : DefaultLinalgEngine
        {
            public int CholeskyCalls, EigCalls, SvdCalls;
            public NPTypeCode LastOperandType;

            protected override NDArray CholeskyKernel(NDArray a, bool upper)
            {
                CholeskyCalls++;
                LastOperandType = a.typecode;
                return np.full(Shape.Adopt((long[])a.Shape.dimensions.Clone()), 42.0, a.dtype);
            }

            protected override (NDArray eigenvalues, NDArray eigenvectors) EigKernel(NDArray a, bool computeVectors)
            {
                EigCalls++;
                LastOperandType = a.typecode;
                // geev's dtype (complex128) with zero imaginary parts: eig's tail must collapse them to a real result.
                long n = a.Shape.dimensions[a.ndim - 1];
                var w = np.arange(1.0, n + 1.0).astype(NPTypeCode.Complex);
                var v = computeVectors ? np.eye((int)n).astype(NPTypeCode.Complex) : null;
                return (w, v);
            }

            protected override (NDArray U, NDArray S, NDArray Vh) SvdKernel(NDArray a, bool fullMatrices, bool computeUv)
            {
                SvdCalls++;
                LastOperandType = a.typecode;
                // Singular values 3 and 0: matrix_rank counts one, the 2-norm is 3.
                return (null, np.array(new[] {3.0, 0.0}), null);
            }
        }

        [TestMethod]
        public void TheDefault_IsTheSharedDefaultLinalgEngine_AndAssigningNullRestoresIt()
        {
            var engine = BackendFactory.GetEngine();
            var before = engine.Linalg;
            before.Should().BeOfType<DefaultLinalgEngine>("NumSharp.Core installs its own numpy.linalg statements");
            var probe = new RecordingLinalg();
            try
            {
                engine.Linalg = probe;
                engine.Linalg.Should().BeSameAs(probe);
                engine.Linalg = null;
                engine.Linalg.Should().BeSameAs(before, "null stands for the shared default, as a null Blas does for the managed kernels");
            }
            finally
            {
                engine.Linalg = null;
            }
        }

        [TestMethod]
        public void EveryNpLinalgFunction_CallsTheAssignedEngine_ForArraysCreatedBeforeTheAssignment()
        {
            // Created BEFORE the assignment: the property sits on the cached engine, so it reaches existing arrays.
            var sq = np.array(new double[,] {{4.0, 1.0}, {1.0, 3.0}});
            var m = np.arange(6.0).reshape(2, 3);
            var v = np.array(new[] {1.0, 2.0, 3.0});
            var w = np.array(new[] {0.0, 1.0, 0.0});
            var b = np.array(new[] {1.0, 2.0});
            var engine = sq.TensorEngine;
            var probe = new RecordingLinalg();
            try
            {
                engine.Linalg = probe;

                // The factorisations without managed numerics stop in the kernel (no backend in this project): the probe
                // has recorded the call by then, which is what this test asserts.
                void Call(Action action)
                {
                    try { action(); }
                    catch (NotSupportedException) { }
                }

                Call(() => np.linalg.cholesky(sq));
                Call(() => np.linalg.cond(sq));
                np.linalg.cross(v, w);
                np.linalg.det(sq);
                Call(() => np.linalg.eig(sq));
                Call(() => np.linalg.eigh(sq));
                Call(() => np.linalg.eigvals(sq));
                Call(() => np.linalg.eigvalsh(sq));
                np.linalg.inv(sq);
                Call(() => np.linalg.lstsq(sq, b));
                np.linalg.matmul(sq, sq);
                np.linalg.matrix_norm(sq);
                np.linalg.matrix_power(sq, 3);
                Call(() => np.linalg.matrix_rank(sq));
                np.linalg.multi_dot(sq, sq, sq);
                np.linalg.norm(sq);
                np.linalg.outer(v, w);
                Call(() => np.linalg.pinv(sq));
                Call(() => np.linalg.qr(m));
                np.linalg.slogdet(sq);
                np.linalg.solve(sq, b);
                Call(() => np.linalg.svd(m));
                Call(() => np.linalg.svdvals(m));
                np.linalg.tensordot(sq, sq);
                np.linalg.tensordot(sq, sq, new[] {1}, new[] {0});
                np.linalg.tensorinv(np.eye(4).reshape(4, 2, 2), 1);
                np.linalg.tensorsolve(sq, b);
                np.linalg.trace(sq);
                np.linalg.vecdot(v, w);
                np.linalg.vector_norm(m);
                np.linalg.vector_norm(m, 1);
            }
            finally
            {
                engine.Linalg = null;
            }

            probe.Calls.Should().BeEquivalentTo(new[]
            {
                "cholesky", "cond", "cross", "det", "eig", "eigh", "eigvals", "eigvalsh", "inv", "lstsq", "matmul",
                "matrix_norm", "matrix_power", "matrix_rank", "multi_dot", "norm", "outer", "pinv", "qr", "slogdet",
                "solve", "svd", "svdvals", "tensordot", "tensordot(axes, axes)", "tensorinv", "tensorsolve", "trace",
                "vecdot", "vector_norm", "vector_norm(int)",
            }, "each of numpy.linalg's 29 computing functions (31 C# forms) is one Linalg member");
        }

        [TestMethod]
        public void ANestedCall_GoesThroughTheAssignedEngineToo()
        {
            // NumPy's tensorinv calls the module's inv, and matrix_norm calls norm: their defaults call the public
            // np.linalg functions, so the assigned engine computes the nested function as well.
            var sq = np.array(new double[,] {{4.0, 1.0}, {1.0, 3.0}});
            var engine = sq.TensorEngine;
            var probe = new RecordingLinalg();
            try
            {
                engine.Linalg = probe;
                np.linalg.tensorinv(sq, 1);
                np.linalg.matrix_norm(sq);
            }
            finally
            {
                engine.Linalg = null;
            }

            probe.Calls.Should().Contain(new[] {"tensorinv", "inv", "matrix_norm", "norm"});
        }

        [TestMethod]
        public void AKernelOverride_ComputesTheFactorisation_WhileNumPysStatementsStay()
        {
            var ints = np.array(new[,] {{4, 1}, {1, 3}});   // int32: _commonType's loop dtype is float64
            var engine = ints.TensorEngine;
            var provider = new KernelProvider();
            try
            {
                engine.Linalg = provider;

                // cholesky: no OpenBlasMissingBackendException without a backend - the override answered, with the
                // operand already cast to the loop dtype.
                var l = np.linalg.cholesky(ints);
                provider.CholeskyCalls.Should().Be(1);
                provider.LastOperandType.Should().Be(NPTypeCode.Double);
                l.typecode.Should().Be(NPTypeCode.Double);
                l.GetDouble(0, 0).Should().Be(42.0);

                // The checks before the kernel still refuse: float16 (_commonType) and a non-square operand.
                new Action(() => np.linalg.cholesky(np.ones(new Shape(2, 2), NPTypeCode.Half)))
                    .Should().Throw<TypeError>().WithMessage("array type float16 is unsupported in linalg");
                new Action(() => np.linalg.cholesky(np.ones(new Shape(2, 3), np.float64)))
                    .Should().Throw<LinAlgError>().WithMessage("Last 2 dimensions of the array must be square");
                provider.CholeskyCalls.Should().Be(1, "a refused operand never reaches the kernel");

                // eig's tail collapses the kernel's complex result with zero imaginary parts to a real one.
                var eigenvalues = np.linalg.eigvals(ints);
                provider.EigCalls.Should().Be(1);
                eigenvalues.typecode.Should().Be(NPTypeCode.Double);
                eigenvalues.ToArray<double>().Should().Equal(1.0, 2.0);

                // matrix_rank counts the kernel's singular values above the tolerance; the 2-norm is the largest one.
                np.linalg.matrix_rank(ints).GetInt64().Should().Be(1);
                np.linalg.norm(ints, 2).GetDouble().Should().Be(3.0);
                provider.SvdCalls.Should().Be(2);
            }
            finally
            {
                engine.Linalg = null;
            }

            // Back on the default: no backend in this project, so the kernel raises again.
            new Action(() => np.linalg.cholesky(ints)).Should().Throw<OpenBlasMissingBackendException>();
        }

        [TestMethod]
        public void TheMissingBackendMessage_NamesTheOperandsEngine_AsBeforeTheMove()
        {
            // The kernels left TensorEngine in 0.71.0; the message still names the engine that holds the Blas property a
            // backend is assigned to (DefaultEngine for every NumSharp.Core array), not the sub-engine's class.
            var sq = np.array(new double[,] {{4.0, 1.0}, {1.0, 3.0}});
            new Action(() => np.linalg.svd(sq)).Should().Throw<OpenBlasMissingBackendException>()
                .WithMessage("np.linalg.svd requires a matrix backend, and DefaultEngine has none installed or the installed one declined these operands.*IBlasBackend.TrySvd.");
        }
    }
}
