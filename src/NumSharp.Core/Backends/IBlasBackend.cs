namespace NumSharp.Backends
{
    /// <summary>
    ///     An external BLAS/LAPACK a <see cref="TensorEngine"/> may delegate its matrix products and
    ///     factorisations to.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     NumSharp is 100 % managed C#: it has no native dependency and computes every matrix
    ///     product with its own SIMD kernels. This interface is the ONE place an optional package
    ///     can offer an alternative — it is a seam, not a requirement. With
    ///     <see cref="TensorEngine.Blas"/> left null (the default) nothing here is ever consulted.
    ///     </para>
    ///     <para>
    ///     Every member is <c>Try</c>-shaped on purpose: a backend answers only for the operand
    ///     combinations it actually implements (an external CBLAS covers <c>float32</c>/
    ///     <c>float64</c> and nothing else) and returns false for the rest, which the engine then
    ///     computes with its own kernels. So installing a backend can change WHICH implementation
    ///     runs, never WHETHER NumSharp can compute the product.
    ///     </para>
    ///     <para>
    ///     The reason to want one is not only speed. Two correct matrix products that sum in
    ///     different orders give different bits, so a workload that must agree with another stack
    ///     to the last bit — e.g. training a network twice and byte-comparing the weights — has to
    ///     call that stack's own BLAS. See <c>NumSharp.Interop.OpenBLAS</c> and
    ///     <c>docs/stale-docs/GEMM_PARITY.md</c>.
    ///     </para>
    ///     <para>
    ///     <b>Reading the operands.</b> An implementation needs raw pointers and strides, and
    ///     NumSharp's public surface offers two spellings that are NOT interchangeable. Use
    ///     <c>(T*)a.GetData().Address + a.Shape.Offset</c> together with <c>a.Shape.Strides</c> —
    ///     that pair is consistent for every layout. Do NOT use <c>a.GetData&lt;T&gt;().Address</c>:
    ///     for a non-contiguous view it returns a DENSIFIED COPY, so combining it with
    ///     <c>Shape.Strides</c> reads outside the buffer and yields wrong numbers on exactly the
    ///     sliced and transposed operands a matrix product is most often handed. Element strides,
    ///     not bytes. Nothing here requires access to NumSharp's internals — a backend can live in
    ///     any assembly.
    ///     </para>
    ///     <para>
    ///     <b>The linear-algebra members below (the CBLAS product family and the LAPACK
    ///     factorisations) are DEFAULT implementations returning false.</b> That is what lets them be
    ///     added without breaking the one backend that ships today
    ///     (<c>NumSharp.Interop.OpenBLAS</c>'s <c>OpenBlasBackend</c>, which answers only for
    ///     <c>dot</c>/<c>matmul</c> at float32/float64) and any third-party implementation — a
    ///     backend mentions only the members it serves, exactly as with
    ///     <see cref="TryMatMulBatched"/>.
    ///     </para>
    ///     <para>
    ///     <b>Those two groups differ in one important way, and it is not their shape.</b> The
    ///     product family (<c>inner</c>/<c>vdot</c>/<c>vecdot</c>/<c>matvec</c>/<c>vecmat</c>) keeps
    ///     the invariant the rest of this interface states: NumSharp computes them with its own
    ///     managed kernels, so a backend changes WHICH implementation runs, never WHETHER the answer
    ///     can be produced. The factorisation family CANNOT keep it — NumSharp.Core ships no managed
    ///     LU, QR, SVD or eigensolver, so with no backend installed (or one that declines) the engine
    ///     has nothing to fall back to and raises <see cref="NumSharp.OpenBlasMissingBackendException"/>.
    ///     Callers of <c>np.linalg</c> must expect that; callers of the product family need not.
    ///     </para>
    ///     <para>
    ///     <b>LAPACK operands.</b> The operand-reading rule above applies unchanged, but LAPACK
    ///     additionally wants column-major input, so an implementation will usually transpose or ask
    ///     LAPACKE for row-major order; the engine hands over the operands untouched rather than
    ///     guessing which a backend prefers.
    ///     </para>
    ///     <para>
    ///     <b>Stacked operands.</b> NumPy's linalg entry points are gufuncs over the trailing one or
    ///     two axes, so every operand in the factorisation family may carry leading batch dimensions.
    ///     A backend that only serves a single matrix returns false for a stacked one.
    ///     </para>
    /// </remarks>
    public interface IBlasBackend
    {
        /// <summary>
        ///     A one-line description of the underlying library (path, symbols, thread count),
        ///     for diagnostics. Never null.
        /// </summary>
        string Info { get; }

        /// <summary>
        ///     Computes <c>np.dot(left, right)</c>, allocating the result.
        /// </summary>
        /// <param name="result">The product, when this returns true; otherwise null.</param>
        /// <returns>
        ///     False when this backend does not serve these operands — the caller must then fall
        ///     back to its own kernels, and <paramref name="result"/> is meaningless.
        /// </returns>
        /// <remarks>
        ///     <c>dot</c> is a separate entry point from <see cref="TryMatMul2D"/> because the two
        ///     are not the same operation for every shape: NumPy implements them with two different
        ///     dispatchers, which disagree on some non-contiguous operands, and a backend claiming
        ///     bit-parity has to reproduce that.
        /// </remarks>
        bool TryDot(NDArray left, NDArray right, out NDArray result);

        /// <summary>
        ///     Computes the 2-D matrix product <c>left @ right</c> INTO <paramref name="result"/> —
        ///     the core of <c>np.matmul</c>, of <c>@</c>, and of every element of a stacked product.
        /// </summary>
        /// <param name="result">
        ///     A pre-allocated <c>(M, N)</c> array of the promoted dtype. Written in full when this
        ///     returns true; left untouched when it returns false.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryMatMul2D(NDArray left, NDArray right, NDArray result);

        /// <summary>
        ///     Computes a STACKED matrix product — the whole of <c>np.matmul</c> for operands of
        ///     three dimensions or more — INTO <paramref name="result"/>.
        /// </summary>
        /// <param name="left">Already broadcast to <c>(batch…, M, K)</c>.</param>
        /// <param name="right">Already broadcast to <c>(batch…, K, N)</c>.</param>
        /// <param name="result">A pre-allocated <c>(batch…, M, N)</c> array of the promoted dtype.</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        /// <remarks>
        ///     <b>Optional.</b> The default returns false and the engine calls
        ///     <see cref="TryMatMul2D"/> once per batch element instead, so a backend that does not
        ///     care about stacked products need not mention this member at all — which is the point
        ///     of it being a default implementation rather than a new required one.
        ///     <para>
        ///     Implement it when per-product setup can be amortised. The trailing two strides of
        ///     each operand are the SAME for every element of a stack, so any route decision or
        ///     scratch allocation derived from them is computed once — which is exactly why NumPy
        ///     hoists that work out of its matmul gufunc's outer loop. Doing it per element instead
        ///     costs more than the product itself once the matrices are small: measured on 2000
        ///     stacked 8×8 float32 products, per-element setup made an external OpenBLAS 20 % SLOWER
        ///     than NumSharp's own managed GEMM.
        ///     </para>
        /// </remarks>
        bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;

        #region Products (CBLAS)

        /// <summary>
        ///     Computes <c>np.inner(a, b)</c> — a sum product over the LAST axis of both operands.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryInner(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.vdot(a, b)</c> — both operands flattened to 1-D, <paramref name="a"/>
        ///     conjugated when complex, then a vector dot product. Always a 0-d result.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVdot(NDArray a, NDArray b, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.vecdot</c> gufunc <c>(n),(n)->()</c>, allocating the result —
        ///     <paramref name="x1"/> conjugated when complex.
        /// </summary>
        /// <remarks>
        ///     Allocating rather than filling a caller-provided buffer (the shape
        ///     <see cref="TryMatMul2D"/> takes) is deliberate for the three gufuncs: the
        ///     loop shape is the broadcast of the operands' leading axes, and making the engine
        ///     derive it up front would compute it twice on the path where the backend declines.
        /// </remarks>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVecdot(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.matvec</c> gufunc <c>(m,n),(n)->(m)</c>, allocating the result —
        ///     NumPy's <c>gemv</c> route.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryMatvec(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes the <c>np.vecmat</c> gufunc <c>(n),(n,m)->(m)</c>, allocating the result —
        ///     <paramref name="x1"/> conjugated when complex.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryVecmat(NDArray x1, NDArray x2, out NDArray result)
        {
            result = null;
            return false;
        }

        #endregion

        #region Factorisations (LAPACK)

        /// <summary>Computes <c>np.linalg.cholesky</c> (LAPACK <c>potrf</c>).</summary>
        /// <param name="upper">False for the lower-triangular factor (NumPy's default).</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryCholesky(NDArray a, bool upper, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>Computes <c>np.linalg.det</c> (LAPACK <c>getrf</c>).</summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryDet(NDArray a, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.slogdet</c> (LAPACK <c>getrf</c>) — the sign of the determinant
        ///     and the natural log of its absolute value, which stays finite where <c>det</c>
        ///     overflows.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySlogdet(NDArray a, out NDArray sign, out NDArray logabsdet)
        {
            sign = null;
            logabsdet = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.eig</c> / <c>np.linalg.eigvals</c> (LAPACK <c>geev</c>) for a
        ///     general, not necessarily symmetric, matrix.
        /// </summary>
        /// <param name="computeVectors">
        ///     False for <c>eigvals</c> — <paramref name="eigenvectors"/> is then null.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryEig(NDArray a, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.eigh</c> / <c>np.linalg.eigvalsh</c> (LAPACK <c>syevd</c> for a
        ///     real symmetric operand, <c>heevd</c> for a complex Hermitian one).
        /// </summary>
        /// <param name="uplo">'L' or 'U' — which triangle holds the data.</param>
        /// <param name="computeVectors">False for <c>eigvalsh</c>.</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryEigh(NDArray a, char uplo, bool computeVectors, out NDArray eigenvalues, out NDArray eigenvectors)
        {
            eigenvalues = null;
            eigenvectors = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.inv</c> — NumPy solves <c>a x = I</c> with <c>gesv</c> rather
        ///     than calling an explicit inversion routine.
        /// </summary>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryInv(NDArray a, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.lstsq</c> (LAPACK <c>gelsd</c>) — the least-squares solution
        ///     plus residuals, the effective rank and the singular values.
        /// </summary>
        /// <param name="rcond">Singular-value cutoff, relative to the largest singular value.</param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryLstsq(NDArray a, NDArray b, double rcond,
            out NDArray solution, out NDArray residuals, out NDArray rank, out NDArray singularValues)
        {
            solution = null;
            residuals = null;
            rank = null;
            singularValues = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.qr</c> (LAPACK <c>geqrf</c> plus <c>orgqr</c>/<c>ungqr</c>).
        /// </summary>
        /// <param name="mode">
        ///     Already validated to one of "reduced", "complete", "r", "raw". For "r" only
        ///     <paramref name="r"/> is produced; for "raw" the pair is LAPACK's packed
        ///     (<c>h</c>, <c>tau</c>) rather than (Q, R).
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TryQr(NDArray a, string mode, out NDArray q, out NDArray r)
        {
            q = null;
            r = null;
            return false;
        }

        /// <summary>Computes <c>np.linalg.solve</c> (LAPACK <c>gesv</c>).</summary>
        /// <param name="oneDimensionalRhs">
        ///     True when <paramref name="b"/> entered as a single vector — NumPy dispatches that to
        ///     its <c>solve1</c> gufunc <c>(m,m),(m)->(m)</c> rather than <c>(m,m),(m,n)->(m,n)</c>.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySolve(NDArray a, NDArray b, bool oneDimensionalRhs, out NDArray result)
        {
            result = null;
            return false;
        }

        /// <summary>
        ///     Computes <c>np.linalg.svd</c> / <c>np.linalg.svdvals</c> (LAPACK <c>gesdd</c>).
        /// </summary>
        /// <param name="computeUv">
        ///     False for the singular values alone — <paramref name="u"/> and <paramref name="vh"/>
        ///     are then null.
        /// </param>
        /// <returns>False when this backend does not serve these operands.</returns>
        bool TrySvd(NDArray a, bool fullMatrices, bool computeUv, out NDArray u, out NDArray s, out NDArray vh)
        {
            u = null;
            s = null;
            vh = null;
            return false;
        }

        #endregion
    }
}
