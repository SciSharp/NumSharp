using System;
using System.Numerics;
using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     The linear-algebra entry points beyond <c>dot</c>/<c>matmul</c>: the matrix products (managed
    ///     fallback, backend optional), the LU-based factorisations <c>det</c>/<c>slogdet</c>/
    ///     <c>solve</c>/<c>inv</c> (managed fallback via <see cref="Backends.ManagedLu"/>, backend
    ///     optional), and the remaining factorisations (backend required, no fallback).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every member reads <see cref="Blas"/> into a LOCAL before using it. That is not style —
    ///     the property is settable and a concurrent <c>OpenBlasEngine.Disable()</c> turns a test-then-call
    ///     into a null dereference (measured at ~2 % of calls; see <c>docs/stale-docs/GEMM_PARITY.md</c> §9).
    ///     </para>
    ///     <para>
    ///     <b>Two behaviours live here and the difference matters to callers.</b> The product family
    ///     falls back to NumSharp's own managed kernels, so a backend changes which implementation
    ///     runs and nothing else. The factorisation family splits: the LU-based four — <c>det</c>,
    ///     <c>slogdet</c>, <c>solve</c>, <c>inv</c> — fall back to <see cref="Backends.ManagedLu"/>, a
    ///     pure-managed unblocked LU that computes them (allclose to NumPy, since NumSharp.Core carries
    ///     no BLOCKED LU to reproduce LAPACK's exact accumulation) rather than raising. The remaining
    ///     factorisations — Cholesky, QR, SVD, the eigensolvers, <c>lstsq</c> — still have no managed
    ///     numerics and raise <see cref="OpenBlasMissingBackendException"/> when no backend serves the
    ///     operands. The members are <c>virtual</c> rather than <c>abstract</c> so an alternative
    ///     engine overrides only what it actually implements.
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
        [NDScoped] // engine boundary (also called directly, not only via np.vdot): the ravel/reshape/conjugate temps are reclaimed
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
        [NDScoped] // engine boundary: the conjugate + elementwise product feeding the reduction are reclaimed
        public virtual NDArray Vecdot(NDArray x1, NDArray x2)
        {
            var blas = Blas;
            if (blas != null && blas.TryVecdot(x1, x2, out var accelerated))
                return accelerated;

            // The reduction dtype is the LOOP dtype, not NEP50's accumulator: np.vecdot's registered
            // loops are 'ii->i', so an int32 pair reduces to int32 where np.sum would give int64.
            var product = Multiply(LinAlgHelper.Conjugate(x1), x2);
            return ReduceAdd(product, -1, false, product.typecode.AsType());
        }

        /// <summary>
        ///     The <c>np.matvec</c> gufunc <c>(m,n),(n)->(m)</c> — NumPy's <c>gemv</c> route.
        /// </summary>
        [NDScoped] // engine boundary: the column-promoted operand and the pre-DropAxis product are reclaimed
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
        [NDScoped] // engine boundary: the conjugate/expand_dims operand and the pre-DropAxis product are reclaimed
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

        #region Factorisations — LU family has a managed fallback; the rest need a backend

        /// <summary><c>np.linalg.cholesky</c> — OpenBLAS <c>potrf</c>.</summary>
        public virtual NDArray Cholesky(NDArray a, bool upper)
        {
            var blas = Blas;
            if (blas != null && blas.TryCholesky(a, upper, out var result))
                return result;
            throw MissingBackend("np.linalg.cholesky", nameof(Backends.IBlasBackend.TryCholesky));
        }

        /// <summary>
        ///     <c>np.linalg.det</c> — OpenBLAS <c>getrf</c>, or NumSharp's managed LU
        ///     (<see cref="Backends.ManagedLu"/>) when no backend serves the operand.
        /// </summary>
        public virtual NDArray Det(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TryDet(a, out var result))
                return result;
            return Backends.ManagedLu.Det(a);
        }

        /// <summary>
        ///     <c>np.linalg.slogdet</c> — OpenBLAS <c>getrf</c>, or NumSharp's managed LU
        ///     (<see cref="Backends.ManagedLu"/>) when no backend serves the operand.
        /// </summary>
        public virtual (NDArray sign, NDArray logabsdet) Slogdet(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TrySlogdet(a, out var sign, out var logabsdet))
                return (sign, logabsdet);
            return Backends.ManagedLu.Slogdet(a);
        }

        /// <summary><c>np.linalg.eig</c> / <c>np.linalg.eigvals</c> — OpenBLAS <c>geev</c>.</summary>
        public virtual (NDArray eigenvalues, NDArray eigenvectors) Eig(NDArray a, bool computeVectors)
        {
            var blas = Blas;
            if (blas != null && blas.TryEig(a, computeVectors, out var w, out var v))
                return (w, v);
            throw MissingBackend(computeVectors ? "np.linalg.eig" : "np.linalg.eigvals",
                nameof(Backends.IBlasBackend.TryEig));
        }

        /// <summary>
        ///     <c>np.linalg.eigh</c> / <c>np.linalg.eigvalsh</c> — OpenBLAS <c>syevd</c> (real
        ///     symmetric) or <c>heevd</c> (complex Hermitian).
        /// </summary>
        public virtual (NDArray eigenvalues, NDArray eigenvectors) Eigh(NDArray a, char uplo, bool computeVectors)
        {
            var blas = Blas;
            if (blas != null && blas.TryEigh(a, uplo, computeVectors, out var w, out var v))
                return (w, v);
            throw MissingBackend(computeVectors ? "np.linalg.eigh" : "np.linalg.eigvalsh",
                nameof(Backends.IBlasBackend.TryEigh));
        }

        /// <summary>
        ///     <c>np.linalg.inv</c> — OpenBLAS <c>gesv</c> against the identity, or NumSharp's managed
        ///     LU (<see cref="Backends.ManagedLu"/>) when no backend serves the operand. A singular
        ///     operand surfaces as <c>LinAlgError("Singular matrix")</c> from the factorisation either
        ///     way, exactly as NumPy's does.
        /// </summary>
        public virtual NDArray Inv(NDArray a)
        {
            var blas = Blas;
            if (blas != null && blas.TryInv(a, out var result))
                return result;
            return Backends.ManagedLu.Inv(a);
        }

        /// <summary><c>np.linalg.lstsq</c> — OpenBLAS <c>gelsd</c>.</summary>
        public virtual (NDArray Solution, NDArray Residuals, NDArray Rank, NDArray SingularValues) Lstsq(
            NDArray a, NDArray b, double rcond)
        {
            var blas = Blas;
            if (blas != null && blas.TryLstsq(a, b, rcond, out var x, out var res, out var rank, out var s))
                return (x, res, rank, s);
            throw MissingBackend("np.linalg.lstsq", nameof(Backends.IBlasBackend.TryLstsq));
        }

        /// <summary>
        ///     <c>np.linalg.qr</c> — OpenBLAS <c>geqrf</c> plus <c>orgqr</c>/<c>ungqr</c>.
        /// </summary>
        public virtual (NDArray Q, NDArray R) Qr(NDArray a, string mode)
        {
            var blas = Blas;
            if (blas != null && blas.TryQr(a, mode, out var q, out var r))
                return (q, r);
            throw MissingBackend("np.linalg.qr", nameof(Backends.IBlasBackend.TryQr));
        }

        /// <summary>
        ///     <c>np.linalg.solve</c> — OpenBLAS <c>gesv</c>, or NumSharp's managed LU
        ///     (<see cref="Backends.ManagedLu"/>) when no backend serves the operands.
        /// </summary>
        public virtual NDArray Solve(NDArray a, NDArray b, bool oneDimensionalRhs)
        {
            var blas = Blas;
            if (blas != null && blas.TrySolve(a, b, oneDimensionalRhs, out var result))
                return result;
            return Backends.ManagedLu.Solve(a, b, oneDimensionalRhs);
        }

        /// <summary>
        ///     <c>np.linalg.svd</c> / <c>np.linalg.svdvals</c> — OpenBLAS <c>gesdd</c>. Also the engine
        ///     behind <c>pinv</c>, <c>matrix_rank</c>, <c>cond</c> and the spectral/nuclear norms.
        /// </summary>
        public virtual (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool fullMatrices, bool computeUv)
        {
            var blas = Blas;
            if (blas != null && blas.TrySvd(a, fullMatrices, computeUv, out var u, out var s, out var vh))
                return (u, s, vh);
            throw MissingBackend("np.linalg.svd", nameof(Backends.IBlasBackend.TrySvd));
        }

        #endregion

        // np.einsum's contraction does NOT live here — it is a pure composition over the matrix
        // products (Matmul/Multiply) with no engine state and no seam, so it sits in the np layer next
        // to np.tensordot and np.linalg.multi_dot (LinearAlgebra/np.einsum.Contract.cs). It inherits
        // Blas routing the same indirect way they do, through np.matmul, exactly as NumPy's
        // einsumfunc.py composes over c_einsum + matmul rather than adding a low-level kernel.

        /// <summary>
        ///     The one message every unserved factorisation raises. It names the NumPy API, the seam
        ///     member a backend has to implement, and the package that supplies one — so the exception
        ///     says what to install rather than merely that something is missing.
        /// </summary>
        /// <remarks>
        ///     There is no separate seam for the factorisations in NumSharp: they ride the very same
        ///     <see cref="Blas"/> property as the matrix products, filled by a single
        ///     <c>NumSharp.Interop.OpenBLAS</c> reference — so the message points at that package rather
        ///     than at "a backend" in the abstract.
        /// </remarks>
        private OpenBlasMissingBackendException MissingBackend(string api, string seamMember)
            => new OpenBlasMissingBackendException(
                $"{api} requires a matrix backend, and {GetType().Name} has none installed or the " +
                $"installed one declined these operands. NumSharp.Core is 100 % managed C# and ships " +
                $"no matrix factorisation of its own, so — unlike the matrix products, which always " +
                $"have a managed kernel to fall back to — there is nothing to compute this with. " +
                $"{OpenBlasMissingBackendException.HowToFix} The backend serves this operation through " +
                $"its IBlasBackend.{seamMember}.");
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
