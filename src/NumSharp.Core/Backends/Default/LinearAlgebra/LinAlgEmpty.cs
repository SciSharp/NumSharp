using System;
using NumSharp;

namespace NumSharp.Backends
{
    /// <summary>
    ///     The <c>np.linalg</c> factorisations of an EMPTY operand — an empty core matrix (<c>M == 0</c> or <c>N == 0</c>) or a
    ///     zero-size batch (<c>(0, 3, 3)</c>) — which NumPy answers without computing anything: its gufunc loops zero times
    ///     over an empty batch, and LAPACK returns at once for an empty matrix, after which NumPy's C wrappers fill the
    ///     outputs LAPACK left untouched. NumSharp.Core ships no Cholesky, eigensolver, QR, SVD or least-squares solver, so
    ///     without a backend these are the only results it can produce for that family; the engine takes them BEFORE it
    ///     asks a backend, so an empty operand gets the same answer with or without <c>NumSharp.Interop.OpenBLAS</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Every result is a fresh, writeable, C-contiguous array of the dtype the backend's result would have (the
    ///     operand arrives cast to NumPy's <c>result_t</c>: float32, float64 or complex128; the real outputs take its real
    ///     type). An empty result has no values. The only values are exact: the identity matrices NumPy writes where LAPACK
    ///     leaves a non-empty factor of an empty matrix uninitialised (a complete QR's Q of an M×0 matrix: <c>orgqr</c>
    ///     with no reflectors; a full SVD's U or Vh: <c>umath_linalg</c>'s <c>identity_matrix</c>), and <c>lstsq</c>'s
    ///     residuals for an M×0 matrix (NumPy's sequential sum of squares of <c>b</c>).
    ///     </para>
    ///     <para>
    ///     Probed against NumPy 2.4.2 over the empty and zero-batch shapes and the float32 / float64 / complex128 / int
    ///     dtypes; gated by the portable <c>linalg_empty.jsonl</c> oracle tier. The composites built on these (pinv,
    ///     matrix_rank, cond, the SVD norms, svdvals) inherit them through their own NumPy statements: for example
    ///     <c>matrix_rank</c> of an empty matrix raises NumPy's zero-size <c>maximum</c> error from <c>S.max()</c>, and
    ///     <c>norm(x, 2)</c> is 0 through <c>amax(..., initial=0)</c>.
    ///     </para>
    /// </remarks>
    internal static class LinAlgEmpty
    {
        /// <summary>
        ///     <c>np.linalg.cholesky</c> of an empty operand: the empty <c>(…, N, N)</c> factor of the operand's dtype (the
        ///     <c>upper</c> flag only picks which triangle a non-empty factor fills).
        /// </summary>
        /// <param name="a">The operand, cast to <c>result_t</c>, with <c>a.size == 0</c> (validated square by the caller).</param>
        /// <returns>A fresh zero-size array of <paramref name="a"/>'s shape and dtype.</returns>
        internal static NDArray Cholesky(NDArray a) => np.zeros(Shape.Adopt(a.shape), a.typecode);

        /// <summary>
        ///     <c>np.linalg.eig</c> / <c>eigvals</c> of an empty operand, as the <c>geev</c> seam returns it: complex128
        ///     eigenvalues <c>(…, N)</c> and eigenvectors <c>(…, N, N)</c>. The caller's realness collapse then makes them
        ///     real for a real operand, because NumPy's <c>all(w.imag == 0)</c> is vacuously true for an empty result.
        /// </summary>
        /// <param name="a">The operand, cast to <c>result_t</c>, with <c>a.size == 0</c> (validated square by the caller).</param>
        /// <param name="computeVectors">Whether the eigenvectors are wanted; when false their slot is null.</param>
        /// <returns>The zero-size complex128 eigenvalues and (optionally) eigenvectors.</returns>
        internal static (NDArray w, NDArray v) Eig(NDArray a, bool computeVectors)
        {
            long n = a.Shape.dimensions[a.ndim - 1];
            var w = np.zeros(Stacked(a, n), NPTypeCode.Complex);
            var v = computeVectors ? np.zeros(Stacked(a, n, n), NPTypeCode.Complex) : null;
            return (w, v);
        }

        /// <summary>
        ///     <c>np.linalg.eigh</c> / <c>eigvalsh</c> of an empty operand: REAL eigenvalues <c>(…, N)</c> (the operand's
        ///     real type: float64 for complex128) and eigenvectors <c>(…, N, N)</c> of the operand's dtype.
        /// </summary>
        /// <param name="a">The operand, cast to <c>result_t</c>, with <c>a.size == 0</c> (validated square by the caller).</param>
        /// <param name="computeVectors">Whether the eigenvectors are wanted; when false their slot is null.</param>
        /// <returns>The zero-size eigenvalues and (optionally) eigenvectors.</returns>
        internal static (NDArray w, NDArray v) Eigh(NDArray a, bool computeVectors)
        {
            long n = a.Shape.dimensions[a.ndim - 1];
            var w = np.zeros(Stacked(a, n), RealOf(a.typecode));
            var v = computeVectors ? np.zeros(Shape.Adopt(a.shape), a.typecode) : null;
            return (w, v);
        }

        /// <summary>
        ///     <c>np.linalg.svd</c> / <c>svdvals</c> of an empty operand: singular values <c>(…, K)</c> (<c>K = min(M, N)</c>,
        ///     the real type), and for <paramref name="computeUv"/> the factors: full <c>U (…, M, M)</c> and
        ///     <c>Vh (…, N, N)</c>, or reduced <c>U (…, M, K)</c> and <c>Vh (…, K, N)</c>.
        /// </summary>
        /// <param name="a">The operand, cast to <c>result_t</c>, with <c>a.size == 0</c>.</param>
        /// <param name="fullMatrices">NumPy's <c>full_matrices</c> (gufunc <c>svd_f</c> versus <c>svd_s</c>).</param>
        /// <param name="computeUv">NumPy's <c>compute_uv</c>; when false the U and Vh slots are null.</param>
        /// <returns>The empty decomposition.</returns>
        /// <remarks>
        ///     A full decomposition of an empty MATRIX (K == 0, a non-empty batch) is not empty: <c>gesdd</c> with
        ///     <c>JOBZ='A'</c> leaves U and VT uninitialised, so NumPy's <c>svd_wrapper</c> writes an identity into whichever
        ///     is non-empty (<c>np.linalg.svd(np.zeros((0, 3)))</c> has <c>Vh = eye(3)</c>). A zero-size batch has K &gt; 0
        ///     possible, but every output is then zero-size.
        /// </remarks>
        internal static (NDArray u, NDArray s, NDArray vh) Svd(NDArray a, bool fullMatrices, bool computeUv)
        {
            long m = a.Shape.dimensions[a.ndim - 2];
            long n = a.Shape.dimensions[a.ndim - 1];
            long k = Math.Min(m, n);
            var s = np.zeros(Stacked(a, k), RealOf(a.typecode));
            if (!computeUv)
                return (null, s, null);
            if (fullMatrices)
            {
                // `if ('A' == params.JOBZ && min_m_n == 0) { identity_matrix(U, M); identity_matrix(VT, N); }`.
                var u = k == 0 ? IdentityStack(a, m, a.typecode) : np.zeros(Stacked(a, m, m), a.typecode);
                var vh = k == 0 ? IdentityStack(a, n, a.typecode) : np.zeros(Stacked(a, n, n), a.typecode);
                return (u, s, vh);
            }
            return (np.zeros(Stacked(a, m, k), a.typecode), s, np.zeros(Stacked(a, k, n), a.typecode));
        }

        /// <summary>
        ///     <c>np.linalg.qr</c> of an empty operand, in the engine seam's per-mode tuple: <c>reduced</c> (Q <c>(…, M, K)</c>,
        ///     R <c>(…, K, N)</c>), <c>complete</c> (Q <c>(…, M, M)</c> and R <c>(…, M, N)</c> when M &gt; N, else reduced),
        ///     <c>r</c> (null, R <c>(…, K, N)</c>), <c>raw</c> (h <c>(…, N, M)</c>, tau <c>(…, K)</c>) and the deprecated
        ///     <c>economic</c> (the packed <c>(…, M, N)</c> matrix, null).
        /// </summary>
        /// <param name="a">The operand, cast to <c>result_t</c>, with <c>a.size == 0</c>.</param>
        /// <param name="mode">The resolved mode (<c>reduced</c>, <c>complete</c>, <c>r</c>, <c>raw</c> or <c>economic</c>).</param>
        /// <returns>The empty factorisation.</returns>
        /// <exception cref="ArgumentException">An unresolved mode (np.linalg.qr resolves it first).</exception>
        /// <remarks>
        ///     A complete QR of an M×0 matrix (a non-empty batch) is not empty: <c>orgqr</c> expands zero reflectors into
        ///     the M×M identity, which is NumPy's Q (<c>np.linalg.qr(np.zeros((3, 0)), 'complete')</c> has <c>Q = eye(3)</c>).
        /// </remarks>
        internal static (NDArray q, NDArray r) Qr(NDArray a, string mode)
        {
            long m = a.Shape.dimensions[a.ndim - 2];
            long n = a.Shape.dimensions[a.ndim - 1];
            long k = Math.Min(m, n);
            var tc = a.typecode;
            switch (mode)
            {
                case "reduced":
                    return (np.zeros(Stacked(a, m, k), tc), np.zeros(Stacked(a, k, n), tc));
                case "complete":
                    // NumPy's `mc = m if mode == 'complete' and m > n else mn`; with M > N the operand is empty only when
                    // N == 0 (an identity Q) or the batch is (an empty identity stack).
                    if (m > n)
                        return (IdentityStack(a, m, tc), np.zeros(Stacked(a, m, n), tc));
                    return (np.zeros(Stacked(a, m, k), tc), np.zeros(Stacked(a, k, n), tc));
                case "r":
                    return (null, np.zeros(Stacked(a, k, n), tc));
                case "raw":
                    return (np.zeros(Stacked(a, n, m), tc), np.zeros(Stacked(a, k), tc));
                case "economic":
                    return (np.zeros(Shape.Adopt(a.shape), tc), null);
                default:
                    throw new ArgumentException($"Unrecognized mode '{mode}'");
            }
        }

        /// <summary>
        ///     <c>np.linalg.lstsq</c>'s gufunc for an EMPTY coefficient matrix (<c>M == 0</c> or <c>N == 0</c>), as the
        ///     backend's degenerate branch returns it: the solution <c>(N, K)</c> zeros, rank 0 (an int32 0-d array), no
        ///     singular values, and the residuals of NumPy's <c>excess &gt;= 0 &amp;&amp; rank == n</c> formula.
        /// </summary>
        /// <param name="a">The coefficient matrix <c>(M, N)</c>, cast to <c>result_t</c>, with <c>a.size == 0</c>.</param>
        /// <param name="b">The right-hand sides <c>(M, K)</c>, cast to <c>result_t</c> (the wrapper pads K == 0 to one
        ///     column, as NumPy does).</param>
        /// <returns>The solution, the residuals (float64, <c>(K,)</c>), the rank and the singular values (float64,
        ///     <c>(0,)</c>); the wrapper casts them to NumPy's result types and discards the residuals where NumPy does.</returns>
        /// <remarks>
        ///     <c>gelsd</c> returns at once for an empty matrix and leaves <c>b</c> as it is, so for an M×0 matrix every row
        ///     of <c>b</c> is an "excess" row and NumPy's residual of each column is <c>res += el*el</c> over them
        ///     (<c>re*re + im*im</c> for complex), a SEQUENTIAL sum in double: <c>lstsq(zeros((3, 0)), [1, 2, 3])</c> has
        ///     residuals <c>[14.]</c>. The sum runs as the last row of a <c>cumsum</c> down the columns, which is that order;
        ///     each square is formed in double (a float32 element is widened exactly first, as NumPy computes in double). For
        ///     M == 0 the solution is NumPy's <c>x[...] = 0</c> and the residuals (an empty sum) are discarded by the wrapper
        ///     (<c>m &lt;= n</c>).
        /// </remarks>
        internal static (NDArray x, NDArray resids, NDArray rank, NDArray s) Lstsq(NDArray a, NDArray b)
        {
            long m = a.Shape.dimensions[0];
            long n = a.Shape.dimensions[1];
            long nrhs = b.Shape.dimensions[1];
            var x = np.zeros(Shape.Adopt(new[] { n, nrhs }), a.typecode);
            var s = np.zeros(Shape.Adopt(new long[] { 0 }), NPTypeCode.Double);
            var rank = NDArray.Scalar(0);
            NDArray resids;
            if (n == 0 && m > 0)
            {
                using var squares = Abs2(b);
                using var running = np.cumsum(squares, 0);
                // The last row of the running sums is each column's sequential sum (a copy, so `running` can go).
                using var last = running[Slice.Index(m - 1)];
                resids = last.copy();
            }
            else
            {
                resids = np.zeros(Shape.Adopt(new[] { nrhs }), NPTypeCode.Double);
            }
            return (x, resids, rank, s);
        }

        /// <summary>
        ///     NumPy's per-element <c>abs2</c> in double: <c>el*el</c> for a real element (widened to float64 first), and
        ///     <c>re*re + im*im</c> for a complex one — two products and one sum, each rounded, never fused (NumPy's
        ///     baseline-compiled <c>umath_linalg</c> has no FMA to contract into).
        /// </summary>
        /// <param name="b">The right-hand sides, float32, float64 or complex128.</param>
        /// <returns>A new float64 array of <paramref name="b"/>'s shape.</returns>
        private static NDArray Abs2(NDArray b)
        {
            if (b.typecode == NPTypeCode.Complex)
            {
                using var re = np.real(b);
                using var im = np.imag(b);
                using var re2 = np.multiply(re, re);
                using var im2 = np.multiply(im, im);
                return np.add(re2, im2);
            }
            using var d = b.astype(NPTypeCode.Double);
            return np.multiply(d, d);
        }

        /// <summary>
        ///     A stack of <paramref name="k"/>×<paramref name="k"/> identity matrices, one per batch element of
        ///     <paramref name="a"/> (NumPy's <c>identity_matrix</c> per gufunc call: zeros with ones on the diagonal).
        /// </summary>
        /// <param name="a">The operand whose batch dimensions the stack takes.</param>
        /// <param name="k">The identity's order.</param>
        /// <param name="tc">The dtype (a complex identity is <c>1+0j</c> on the diagonal).</param>
        /// <returns>A fresh C-contiguous <c>(…, k, k)</c> array; zero-size (no values) for an empty batch or k == 0.</returns>
        private static NDArray IdentityStack(NDArray a, long k, NPTypeCode tc)
        {
            var shape = Stacked(a, k, k);
            if (shape.size == 0)
                return np.zeros(shape, tc);
            var eye = np.eye(checked((int)k), dtype: tc);
            if (a.ndim == 2)
                return eye;
            // One identity broadcast over the batch and materialized in C order: the copy owns its buffer, the
            // read-only broadcast view and the single identity are released right after.
            using (eye)
            using (var view = np.broadcast_to(eye, shape))
                return view.copy();
        }

        /// <summary>The shape of a result per batch element of <paramref name="a"/>: its batch dimensions (all but the last
        ///     two) followed by <paramref name="core"/>.</summary>
        /// <param name="a">The stacked operand.</param>
        /// <param name="core">The result's core dimensions.</param>
        /// <returns>A shape over a fresh dimension array (never the operand's own).</returns>
        private static Shape Stacked(NDArray a, params long[] core)
        {
            int nb = a.ndim - 2;
            var dims = new long[nb + core.Length];
            for (int i = 0; i < nb; i++)
                dims[i] = a.Shape.dimensions[i];
            Array.Copy(core, 0, dims, nb, core.Length);
            return Shape.Adopt(dims);
        }

        /// <summary>The real counterpart of a <c>result_t</c>: float64 for complex128, the dtype itself otherwise.</summary>
        /// <param name="tc">float32, float64 or complex128.</param>
        /// <returns>The singular-value / eigenvalue dtype.</returns>
        private static NPTypeCode RealOf(NPTypeCode tc) => tc == NPTypeCode.Complex ? NPTypeCode.Double : tc;
    }
}
