using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Eigenvalues and right eigenvectors of a general (not necessarily symmetric) matrix.
            /// </summary>
            /// <returns><c>(eigenvalues, eigenvectors)</c>, the columns of the second being the vectors.</returns>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
            ///     <para>
            ///     The result dtype is DATA-dependent: a real matrix with a complex-conjugate pair of
            ///     eigenvalues yields complex output, so the dtype cannot be predicted from the input
            ///     dtype alone. Use <see cref="eigh"/> where the operand is known symmetric or
            ///     Hermitian — it is both faster and guaranteed to give real eigenvalues.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped] // reclaims the ToCommon cast temp and the pre-collapse complex w/v CollapseEig supersedes
            public static (NDArray eigenvalues, NDArray eigenvectors) eig(NDArray a)
            {
                AssertStackedSquare(a);
                AssertFinite(a);
                var common = CommonType(a);
                var (w, v) = a.TensorEngine.Eig(ToCommon(a, common), computeVectors: true);
                return CollapseEig(w, v, common);
            }

            /// <summary>
            ///     Eigenvalues of a general matrix, without the eigenvectors.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped] // reclaims the ToCommon cast temp and the pre-collapse complex eigenvalues
            public static NDArray eigvals(NDArray a)
            {
                AssertStackedSquare(a);
                AssertFinite(a);
                var common = CommonType(a);
                var w = a.TensorEngine.Eig(ToCommon(a, common), computeVectors: false).eigenvalues;
                return CollapseEig(w, null, common).eigenvalues;
            }

            /// <summary>
            ///     NumPy's <c>eig</c>/<c>eigvals</c> Python-layer tail. The LAPACK <c>geev</c> gufunc
            ///     ALWAYS returns complex128 (a real matrix can carry a complex-conjugate eigenpair), so
            ///     for a REAL operand the result is collapsed to a real dtype when every imaginary part
            ///     is zero (<c>if not isComplexType(t) and all(w.imag == 0)</c>); otherwise it stays
            ///     complex. The final width then follows the operand: <c>float32 → float32</c> when real
            ///     (<c>complex128</c> when complex — NumSharp has no complex64), <c>float64</c>/int/bool
            ///     <c>→ float64</c> when real, and a complex operand stays complex.
            /// </summary>
            private static (NDArray eigenvalues, NDArray eigenvectors) CollapseEig(NDArray w, NDArray v, NPTypeCode common)
            {
                NPTypeCode resultType;
                // NumPy's `all(w.imag == 0)`. Expressed as `!any(imag)` — for a float array `np.any`
                // is "any truthy", and a NaN is truthy (`np.any([nan])` is True) exactly as a nonzero
                // is, so `!any(imag)` is True iff every imaginary part is 0 AND none is NaN — bit-for-bit
                // the same predicate as `all(equal(imag, 0))` (verified on zero/nonzero/NaN), but a single
                // reduction over the strided imaginary view instead of `equal` (which allocates a bool
                // array) followed by `all`.
                if (common != NPTypeCode.Complex && !np.any(np.imag(w)))
                {
                    // Every eigenvalue is real — drop the (zero) imaginary part and take the real dtype.
                    w = np.real(w);
                    v = v is null ? null : np.real(v);
                    resultType = common == NPTypeCode.Single ? NPTypeCode.Single : NPTypeCode.Double;
                }
                else
                {
                    // csingle and cdouble both collapse onto NumSharp's single complex width.
                    resultType = NPTypeCode.Complex;
                }

                // NumPy's `w.astype(result_t, copy=False)`: for a real operand np.real already produced
                // the float64 result (so astype is a no-op), and the complex branch's w/v are already
                // the complex128 the geev seam returned — copy:false avoids re-copying the (up to n×n)
                // eigenvector buffer whenever the dtype already matches. Only float32's real collapse
                // and any genuine width change still cast.
                w = w.astype(resultType, copy: false);
                v = v is null ? null : v.astype(resultType, copy: false);
                return (w, v);
            }

            /// <summary>
            ///     Eigenvalues and eigenvectors of a real symmetric or complex Hermitian matrix.
            /// </summary>
            /// <param name="UPLO">
            ///     Which triangle holds the data — <c>'L'</c> (default) or <c>'U'</c>. Case
            ///     insensitive. The other triangle is NOT read, so a non-symmetric operand is
            ///     silently interpreted as its own reflection rather than rejected.
            /// </param>
            /// <returns>
            ///     <c>(eigenvalues, eigenvectors)</c> with the eigenvalues in ASCENDING order and
            ///     always REAL — even for a complex operand, where they come back as the real
            ///     counterpart dtype.
            /// </returns>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped] // reclaims the ToCommon cast temp; the tuple is yielded component-wise
            public static (NDArray eigenvalues, NDArray eigenvectors) eigh(NDArray a, char UPLO = 'L')
            {
                char uplo = RequireUplo(UPLO);
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eigh(ToCommon(a, common), uplo, computeVectors: true);
            }

            /// <summary>
            ///     Eigenvalues of a real symmetric or complex Hermitian matrix, ascending.
            /// </summary>
            /// <inheritdoc cref="eigh"/>
            [NDScoped] // reclaims the ToCommon cast temp (the discarded eigenvector slot is null here)
            public static NDArray eigvalsh(NDArray a, char UPLO = 'L')
            {
                char uplo = RequireUplo(UPLO);
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Eigh(ToCommon(a, common), uplo, computeVectors: false).eigenvalues;
            }

            /// <summary>
            ///     NumPy validates UPLO before touching the array, and accepts either case.
            /// </summary>
            private static char RequireUplo(char uplo)
            {
                char upper = char.ToUpperInvariant(uplo);
                if (upper != 'L' && upper != 'U')
                    throw new ValueError("UPLO argument must be 'L' or 'U'");
                return upper;
            }
        }
    }
}
