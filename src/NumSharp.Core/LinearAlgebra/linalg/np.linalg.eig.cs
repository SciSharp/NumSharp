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
                if (common != NPTypeCode.Complex && np.all(np.equal(np.imag(w), NDArray.Scalar(0.0))))
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

                w = w.astype(resultType);
                v = v is null ? null : v.astype(resultType);
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
