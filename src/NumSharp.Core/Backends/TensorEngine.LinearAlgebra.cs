using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     The linear-algebra entry points: the products NumPy routes through CBLAS beyond
    ///     <c>dot</c>/<c>matmul</c>, and the factorisations it routes through LAPACK.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every member reads <see cref="Blas"/> into a LOCAL before using it. That is not style —
    ///     the property is settable and a concurrent <c>Blas.Disable()</c> turns a test-then-call
    ///     into a null dereference (measured at ~2 % of calls; see <c>docs/GEMM_PARITY.md</c> §9).
    ///     </para>
    ///     <para>
    ///     <b>Two behaviours live here and the difference matters to callers.</b> The product family
    ///     falls back to NumSharp's own managed kernels, so a backend changes which implementation
    ///     runs and nothing else. The LAPACK family has no managed fallback — NumSharp.Core carries
    ///     no LU, QR, SVD or eigensolver — so it raises <see cref="NotSupportedException"/> when no
    ///     backend serves the operands. The members are <c>virtual</c> rather than <c>abstract</c>
    ///     so an alternative engine overrides only what it actually implements.
    ///     </para>
    /// </remarks>
    public abstract partial class TensorEngine
    {
        #region Products — managed fallback, backend optional

        /// <summary>
        ///     <c>np.inner</c> — a sum product over the LAST axis of both operands.
        /// </summary>
        /// <remarks>
        ///     NumPy's <c>PyArray_InnerProduct</c> swaps the last two axes of <paramref name="b"/>
        ///     (when <c>a.ndim >= 1</c> and <c>b.ndim >= 2</c>) and hands the pair to the very same
        ///     <c>PyArray_MatrixProduct2</c> that backs <c>np.dot</c> — which is why an unaligned
        ///     <c>inner</c> reports <paramref name="b"/>'s shape ALREADY TRANSPOSED.
        /// </remarks>
        public virtual NDArray Inner(NDArray a, NDArray b)
        {
            var blas = Blas;
            if (blas != null && blas.TryInner(a, b, out var accelerated))
                return accelerated;

            var right = (a.ndim >= 1 && b.ndim >= 2) ? SwapAxes(b, -1, -2) : b;
            return Dot(a, right);
        }

        /// <summary>
        ///     <c>np.vdot</c> — both operands flattened to 1-D, <paramref name="a"/> conjugated when
        ///     complex, then a vector dot product. Always 0-d.
        /// </summary>
        public virtual NDArray Vdot(NDArray a, NDArray b)
        {
            var blas = Blas;
            if (blas != null && blas.TryVdot(a, b, out var accelerated))
                return accelerated;

            // NumPy's array_vdot reshapes BOTH operands with one reused PyArray_Dims buffer whose
            // single entry starts at -1. _fix_unknown_dimension writes the resolved length back into
            // that buffer, so the second reshape no longer asks for (-1,) but for (a.size,) — which
            // is why a length mismatch surfaces as a reshape error and never as the
            // "vectors have different lengths" the C code below it still carries.
            var left = np.ravel(a);
            var right = np.reshape(b, new long[] {left.size});
            return Dot(LinAlgHelper.Conjugate(left), right);
        }

        /// <summary>
        ///     The <c>np.vecdot</c> gufunc <c>(n),(n)->()</c> — <paramref name="x1"/> conjugated when
        ///     complex. Operands arrive validated; leading axes broadcast.
        /// </summary>
        public virtual NDArray Vecdot(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryVecdot(x1, x2, out var accelerated))
                return accelerated;

            // The reduction dtype is the LOOP dtype, not NEP50's accumulator: np.vecdot's registered
            // loops are 'ii->i', so an int32 pair reduces to int32 where np.sum would give int64.
            var product = Multiply(LinAlgHelper.Conjugate(x1), x2);
            return ReduceAdd(product, -1, false, product.typecode);
        }

        /// <summary>
        ///     The <c>np.matvec</c> gufunc <c>(m,n),(n)->(m)</c> — NumPy's <c>gemv</c> route.
        /// </summary>
        public virtual NDArray Matvec(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryMatvec(x1, x2, out var accelerated))
                return accelerated;

            // (…,m,n) @ (…,n,1) -> (…,m,1), then drop the trailing length-1 axis. Promoting x2 to a
            // column first is what keeps matmul out of its own 1-D special case.
            var product = Matmul(x1, np.expand_dims(x2, -1));
            return LinAlgHelper.DropAxis(product, product.ndim - 1);
        }

        /// <summary>
        ///     The <c>np.vecmat</c> gufunc <c>(n),(n,m)->(m)</c> — <paramref name="x1"/> conjugated
        ///     when complex.
        /// </summary>
        public virtual NDArray Vecmat(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryVecmat(x1, x2, out var accelerated))
                return accelerated;

            // (…,1,n) @ (…,n,m) -> (…,1,m), then drop the length-1 axis at -2.
            var product = Matmul(np.expand_dims(LinAlgHelper.Conjugate(x1), -2), x2);
            return LinAlgHelper.DropAxis(product, product.ndim - 2);
        }

        #endregion

        #region Factorisations — LAPACK, no managed fallback

        /// <summary><c>np.linalg.cholesky</c> — LAPACK <c>potrf</c>.</summary>
        public virtual NDArray Cholesky(NDArray a, bool upper)
        {
            var blas = Blas;
            if (blas != null && blas.TryCholesky(a, upper, out var result))
                return result;
            throw NoLapack("np.linalg.cholesky", "potrf", nameof(Backends.IBlasBackend.TryCholesky));
        }

        /// <summary><c>np.linalg.det</c> — LAPACK <c>getrf</c>.</summary>
        public virtual NDArray Det(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TryDet(a, out var result))
                return result;
            throw NoLapack("np.linalg.det", "getrf", nameof(Backends.IBlasBackend.TryDet));
        }

        /// <summary><c>np.linalg.slogdet</c> — LAPACK <c>getrf</c>.</summary>
        public virtual (NDArray Sign, NDArray LogAbsDet) Slogdet(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TrySlogdet(a, out var sign, out var logabsdet))
                return (sign, logabsdet);
            throw NoLapack("np.linalg.slogdet", "getrf", nameof(Backends.IBlasBackend.TrySlogdet));
        }

        /// <summary><c>np.linalg.eig</c> / <c>np.linalg.eigvals</c> — LAPACK <c>geev</c>.</summary>
        public virtual (NDArray Eigenvalues, NDArray Eigenvectors) Eig(NDArray a, bool computeVectors)
        {
            var blas = Blas;
            if (blas != null && blas.TryEig(a, computeVectors, out var w, out var v))
                return (w, v);
            throw NoLapack(computeVectors ? "np.linalg.eig" : "np.linalg.eigvals", "geev",
                nameof(Backends.IBlasBackend.TryEig));
        }

        /// <summary>
        ///     <c>np.linalg.eigh</c> / <c>np.linalg.eigvalsh</c> — LAPACK <c>syevd</c> (real
        ///     symmetric) or <c>heevd</c> (complex Hermitian).
        /// </summary>
        public virtual (NDArray Eigenvalues, NDArray Eigenvectors) Eigh(NDArray a, char uplo, bool computeVectors)
        {
            var blas = Blas;
            if (blas != null && blas.TryEigh(a, uplo, computeVectors, out var w, out var v))
                return (w, v);
            throw NoLapack(computeVectors ? "np.linalg.eigh" : "np.linalg.eigvalsh", "syevd/heevd",
                nameof(Backends.IBlasBackend.TryEigh));
        }

        /// <summary><c>np.linalg.inv</c> — LAPACK <c>gesv</c> against the identity.</summary>
        public virtual NDArray Inv(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TryInv(a, out var result))
                return result;
            throw NoLapack("np.linalg.inv", "gesv", nameof(Backends.IBlasBackend.TryInv));
        }

        /// <summary><c>np.linalg.lstsq</c> — LAPACK <c>gelsd</c>.</summary>
        public virtual (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(
            NDArray a, NDArray b, double rcond)
        {
            var blas = Blas;
            if (blas != null && blas.TryLstsq(a, b, rcond, out var x, out var res, out var rank, out var s))
                return (x, res, rank, s);
            throw NoLapack("np.linalg.lstsq", "gelsd", nameof(Backends.IBlasBackend.TryLstsq));
        }

        /// <summary>
        ///     <c>np.linalg.qr</c> — LAPACK <c>geqrf</c> plus <c>orgqr</c>/<c>ungqr</c>.
        /// </summary>
        public virtual (NDArray Q, NDArray R) Qr(NDArray a, string mode)
        {
            var blas = Blas;
            if (blas != null && blas.TryQr(a, mode, out var q, out var r))
                return (q, r);
            throw NoLapack("np.linalg.qr", "geqrf/orgqr", nameof(Backends.IBlasBackend.TryQr));
        }

        /// <summary><c>np.linalg.solve</c> — LAPACK <c>gesv</c>.</summary>
        public virtual NDArray Solve(NDArray a, NDArray b, bool oneDimensionalRhs)
        {
            var blas = Blas;
            if (blas != null && blas.TrySolve(a, b, oneDimensionalRhs, out var result))
                return result;
            throw NoLapack("np.linalg.solve", "gesv", nameof(Backends.IBlasBackend.TrySolve));
        }

        /// <summary>
        ///     <c>np.linalg.svd</c> / <c>np.linalg.svdvals</c> — LAPACK <c>gesdd</c>. Also the engine
        ///     behind <c>pinv</c>, <c>matrix_rank</c>, <c>cond</c> and the spectral/nuclear norms.
        /// </summary>
        public virtual (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool fullMatrices, bool computeUv)
        {
            var blas = Blas;
            if (blas != null && blas.TrySvd(a, fullMatrices, computeUv, out var u, out var s, out var vh))
                return (u, s, vh);
            throw NoLapack("np.linalg.svd", "gesdd", nameof(Backends.IBlasBackend.TrySvd));
        }

        #endregion

        #region Einstein summation

        /// <summary>
        ///     <c>np.einsum</c> — the contraction itself, reached only once the subscripts have
        ///     parsed and every operand validated.
        /// </summary>
        /// <param name="outputShape">
        ///     The shape the contraction produces, already resolved by the parser — so an
        ///     implementation inherits the ellipsis, diagonal and broadcast bookkeeping instead of
        ///     redoing it.
        /// </param>
        /// <remarks>
        ///     Deliberately NOT routed through <see cref="Blas"/>: only the pairwise contractions
        ///     <c>optimize=</c> plans can reach a BLAS, and they reach it through
        ///     <see cref="Dot"/>/<c>np.tensordot</c> already. What is missing here is a summation
        ///     kernel over an arbitrary label set and a contraction-path planner — NumSharp's own
        ///     work — so there is no operand shape a backend could answer for, and no
        ///     <c>TryEinsum</c> on the seam.
        /// </remarks>
        public virtual NDArray Einsum(string subscripts, NDArray[] operands, NDArray @out,
            NPTypeCode? dtype, char order, string casting, object optimize, long[] outputShape)
            => throw new NotSupportedException(
                $"np.einsum cannot compute \"{subscripts}\" — {GetType().Name} implements no " +
                "contraction kernel. The subscripts parsed cleanly and the result would have shape (" +
                string.Join(",", outputShape) + "), so what is missing is the KERNEL, not the " +
                "expression. Unlike the np.linalg entry points this is not waiting on a backend: " +
                "einsum needs a summation kernel over an arbitrary label set plus a contraction-path " +
                "planner, both NumSharp's own work. Express the contraction with np.tensordot, " +
                "np.dot, np.matmul or np.vecdot in the meantime.");

        #endregion

        /// <summary>
        ///     The one message every unserved factorisation raises. It names the NumPy API, the
        ///     LAPACK routine NumPy uses, and the seam member a backend has to implement — so the
        ///     exception says what to install rather than merely that something is missing.
        /// </summary>
        private NotSupportedException NoLapack(string api, string routines, string seamMember)
            => new NotSupportedException(
                $"{api} requires a LAPACK backend, and {GetType().Name} has none installed or the " +
                $"installed one declined these operands. NumSharp.Core is 100 % managed C# and ships " +
                $"no matrix factorisation of its own, so — unlike the matrix products, which always " +
                $"have a managed kernel to fall back to — there is nothing to compute this with. " +
                $"NumPy computes it with LAPACK {routines}. Reference a package that assigns " +
                $"TensorEngine.Blas an IBlasBackend implementing {seamMember}.");
    }
}

namespace NumSharp.Backends
{
    /// <summary>
    ///     Shape and dtype plumbing shared by the linear-algebra entry points.
    /// </summary>
    internal static class LinAlgHelper
    {
        /// <summary>
        ///     The complex conjugate, and the identity at every other dtype — which is exactly
        ///     NumPy's rule for <c>vdot</c>/<c>vecdot</c>/<c>vecmat</c>, whose "conjugate the first
        ///     operand" step is a no-op for real loops.
        /// </summary>
        /// <remarks>
        ///     Kept internal: <c>np.conj</c>/<c>np.conjugate</c> is a NumPy function in its own right
        ///     and adding it belongs to its own pass, not to this one.
        /// </remarks>
        internal static NDArray Conjugate(NDArray a)
        {
            if (a.typecode != NPTypeCode.Complex)
                return a;

            var result = a.copy();
            foreach (ref Complex z in np.nditer<Complex>(result, writeable: true))
                z = Complex.Conjugate(z);

            return result;
        }

        /// <summary>
        ///     Drops a length-1 axis from a freshly computed C-contiguous product — the step that
        ///     turns matmul's <c>(…,m,1)</c> back into <c>matvec</c>'s <c>(…,m)</c> and its
        ///     <c>(…,1,m)</c> back into <c>vecmat</c>'s <c>(…,m)</c>.
        /// </summary>
        internal static NDArray DropAxis(NDArray a, int axis)
        {
            var source = a.Shape.dimensions;
            var dims = new long[source.Length - 1];
            for (int i = 0, j = 0; i < source.Length; i++)
            {
                if (i == axis)
                    continue;
                dims[j++] = source[i];
            }

            return np.reshape(a, dims);
        }
    }
}
