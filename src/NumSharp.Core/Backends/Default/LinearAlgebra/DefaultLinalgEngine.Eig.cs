using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_assert_finite</c>, <c>_commonType</c>,
        ///     <see cref="EigKernel"/> at the loop dtype, then NumPy's collapse of a real operand's all-real result
        ///     (<see cref="CollapseEig"/>).
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp and the pre-collapse complex w/v CollapseEig supersedes
        public override (NDArray eigenvalues, NDArray eigenvectors) Eig(NDArray a)
        {
            AssertStackedSquare(a);
            AssertFinite(a);
            var common = CommonType(a);
            var (w, v) = EigKernel(ToCommon(a, common), computeVectors: true);
            return CollapseEig(w, v, common);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     <see cref="Eig(NDArray)"/>'s statements without the eigenvectors (the kernel computes none).
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp and the pre-collapse complex eigenvalues
        public override NDArray Eigvals(NDArray a)
        {
            AssertStackedSquare(a);
            AssertFinite(a);
            var common = CommonType(a);
            var w = EigKernel(ToCommon(a, common), computeVectors: false).eigenvalues;
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
        /// <param name="w">The gufunc's complex128 eigenvalues (owned by the caller's scope).</param>
        /// <param name="v">The gufunc's complex128 eigenvectors, or null for <c>eigvals</c>.</param>
        /// <param name="common">The loop dtype the operand ran at.</param>
        /// <returns>The eigenvalues and eigenvectors (null when <paramref name="v"/> is) in NumPy's result dtype.</returns>
        /// <remarks>
        ///     The accessors and the reduction are the members np.imag, np.real and np.any dispatch to
        ///     (<see cref="NDArray.imag_"/>, <see cref="NDArray.real_"/>, <see cref="TensorEngine.Any(NDArray)"/>):
        ///     <paramref name="w"/> and <paramref name="v"/> are the kernel's plain results, so the facades' class rules
        ///     have nothing to see.
        /// </remarks>
        private static (NDArray eigenvalues, NDArray eigenvectors) CollapseEig(NDArray w, NDArray v, NPTypeCode common)
        {
            NPTypeCode resultType;
            // NumPy's `all(w.imag == 0)`. Expressed as `!any(imag)` — for a float array `np.any`
            // is "any truthy", and a NaN is truthy (`np.any([nan])` is True) exactly as a nonzero
            // is, so `!any(imag)` is True iff every imaginary part is 0 AND none is NaN — bit-for-bit
            // the same predicate as `all(equal(imag, 0))` (verified on zero/nonzero/NaN), but a single
            // reduction over the strided imaginary view instead of `equal` (which allocates a bool
            // array) followed by `all`. The view is taken only for a real operand (the && short-circuits first), and
            // np.any's member runs on the view's own engine, as the facade's `a.TensorEngine.Any(a)` did.
            if (common != NPTypeCode.Complex && NDArray.imag_(w) is var imag && !imag.TensorEngine.Any(imag))
            {
                // Every eigenvalue is real — drop the (zero) imaginary part and take the real dtype.
                w = NDArray.real_(w);
                v = v is null ? null : NDArray.real_(v);
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
        ///     <para>Each copy back is np.copyto's member, <c>UnmanagedStorage.Copyto</c> with np.copyto's default
        ///     <c>same_kind</c> rule (a float32 into float64 is a safe cast), and the lanes are np.real / np.imag's
        ///     members (<see cref="NDArray.real_"/>, <see cref="NDArray.imag_"/>): no np.* facade runs.</para>
        /// </remarks>
        private static void RoundComponentsToSingle(NDArray z)
        {
            if (z.Shape.IsContiguous)
            {
                var parts = z.view(np.float64);
                UnmanagedStorage.Copyto(parts, parts.astype(NPTypeCode.Single), NPY_CASTING.NPY_SAME_KIND_CASTING, null);
                return;
            }
            var re = NDArray.real_(z);
            UnmanagedStorage.Copyto(re, re.astype(NPTypeCode.Single), NPY_CASTING.NPY_SAME_KIND_CASTING, null);
            var im = NDArray.imag_(z);
            UnmanagedStorage.Copyto(im, im.astype(NPTypeCode.Single), NPY_CASTING.NPY_SAME_KIND_CASTING, null);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the UPLO check (before the operand is read), <c>_assert_stacked_square</c>,
        ///     <c>_commonType</c>, the text refusal of the <c>eigh_lo</c> / <c>eigh_up</c> gufunc's input cast, then
        ///     <see cref="EighKernel"/> at the loop dtype.
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp; the tuple is yielded component-wise
        public override (NDArray eigenvalues, NDArray eigenvectors) Eigh(NDArray a, char UPLO = 'L')
        {
            char uplo = RequireUplo(UPLO);
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.eigh_lo / eigh_up: its input cast refuses text.
            ThrowIfTextGufunc(uplo == 'U' ? "eigh_up" : "eigh_lo", a);
            return EighKernel(ToCommon(a, common), uplo, computeVectors: true);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     <see cref="Eigh(NDArray, char)"/>'s statements without the eigenvectors (the <c>eigvalsh_lo</c> /
        ///     <c>eigvalsh_up</c> gufunc; the kernel computes no vectors).
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp (the discarded eigenvector slot is null here)
        public override NDArray Eigvalsh(NDArray a, char UPLO = 'L')
        {
            char uplo = RequireUplo(UPLO);
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.eigvalsh_lo / eigvalsh_up: its input cast refuses text.
            ThrowIfTextGufunc(uplo == 'U' ? "eigvalsh_up" : "eigvalsh_lo", a);
            return EighKernel(ToCommon(a, common), uplo, computeVectors: false).eigenvalues;
        }

        /// <summary>
        ///     NumPy validates UPLO before touching the array, and accepts either case.
        /// </summary>
        /// <param name="uplo">The caller's UPLO.</param>
        /// <returns>The upper-cased letter, <c>'L'</c> or <c>'U'</c>.</returns>
        /// <exception cref="ValueError">Any other letter: <c>UPLO argument must be 'L' or 'U'</c>.</exception>
        private static char RequireUplo(char uplo)
        {
            char upper = char.ToUpperInvariant(uplo);
            if (upper != 'L' && upper != 'U')
                throw new ValueError("UPLO argument must be 'L' or 'U'");
            return upper;
        }
    }
}
