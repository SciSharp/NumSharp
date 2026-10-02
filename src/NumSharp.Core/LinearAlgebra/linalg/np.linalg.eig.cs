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
                    // A float32 operand's complex result is NumPy's csingle: geev ran in double and `astype(complex64)`
                    // rounded each component to float32. NumSharp keeps the complex128 dtype (#569) but carries exactly
                    // those values — so a float32 matrix's eigenvalues, eigenvectors and everything sorted from them
                    // (numpy.polynomial's {p}roots) are NumPy's complex64 values, widened exactly. The imaginary-part
                    // test above already ran on the unrounded result, as NumPy's does.
                    if (common == NPTypeCode.Single)
                    {
                        RoundComponentsToSingle(w);
                        if (v is not null)
                            RoundComponentsToSingle(v);
                    }
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
            ///     NumPy's <c>astype(complex64)</c> of a complex128 result, in place and keeping the complex128 dtype: each
            ///     component narrowed to float32 (round to nearest even, the house cast) and widened back, which is exact.
            /// </summary>
            /// <param name="z">A complex128 array the caller owns (the geev seam's fresh result), overwritten.</param>
            /// <remarks>
            ///     A C-contiguous result (the eigenvalues always) is rounded through ONE float64 view of its interleaved
            ///     components — a cast and a copy back, half the array operations of the per-lane form, which matters at
            ///     the small sizes numpy.polynomial's <c>{p}roots</c> runs (0.7 of 1.4 µs on a degree-10 float32 series).
            ///     Any other layout (an F-ordered eigenvector matrix: <c>view</c> needs a contiguous last axis) rounds the
            ///     real and the imaginary lane separately, strided float64 views of the same buffer.
            /// </remarks>
            private static void RoundComponentsToSingle(NDArray z)
            {
                if (z.Shape.IsContiguous)
                {
                    var parts = z.view(np.float64);
                    np.copyto(parts, parts.astype(NPTypeCode.Single));
                    return;
                }
                var re = np.real(z);
                np.copyto(re, re.astype(NPTypeCode.Single));
                var im = np.imag(z);
                np.copyto(im, im.astype(NPTypeCode.Single));
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
