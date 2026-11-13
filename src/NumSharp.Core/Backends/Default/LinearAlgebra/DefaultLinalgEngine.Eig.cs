using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_assert_finite</c>, <c>_commonType</c>,
        ///     <see cref="EigKernel"/> at the loop dtype, then the results in the operand's complex type
        ///     (<see cref="ComplexEigResult"/>): always complex since NumPy 2.5, a real matrix with real eigenvalues too.
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp and the kernel's w/v when ComplexEigResult replaces them
        public override (NDArray eigenvalues, NDArray eigenvectors) Eig(NDArray a)
        {
            AssertStackedSquare(a);
            AssertFinite(a);
            var common = CommonType(a);
            var (w, v) = EigKernel(ToCommon(a, common), computeVectors: true);
            return ComplexEigResult(w, v, common);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     <see cref="Eig(NDArray)"/>'s statements without the eigenvectors (the kernel computes none).
        /// </remarks>
        [NDScoped] // reclaims the ToCommon cast temp and the kernel's eigenvalues when ComplexEigResult replaces them
        public override NDArray Eigvals(NDArray a)
        {
            AssertStackedSquare(a);
            AssertFinite(a);
            var common = CommonType(a);
            var w = EigKernel(ToCommon(a, common), computeVectors: false).eigenvalues;
            return ComplexEigResult(w, null, common).eigenvalues;
        }

        /// <summary>
        ///     NumPy's <c>eig</c>/<c>eigvals</c> Python-layer tail since NumPy 2.5 (numpy#30411):
        ///     <c>w.astype(_complexType(result_t), copy=False)</c>, the same for the eigenvectors. The LAPACK <c>geev</c>
        ///     gufunc always returns complex128, and the result now always STAYS complex: a real matrix whose eigenvalues
        ///     are all real gives complex values with zero imaginary parts (NumPy 2.4 collapsed that case to a real dtype,
        ///     which made the result dtype depend on the values). Use <c>eigh</c> / <c>eigvalsh</c> for real results of a
        ///     symmetric matrix, or take <c>np.real</c> when every imaginary part is zero.
        /// </summary>
        /// <param name="w">The gufunc's complex128 eigenvalues (owned by the caller's scope).</param>
        /// <param name="v">The gufunc's complex128 eigenvectors, or null for <c>eigvals</c>.</param>
        /// <param name="common">The loop dtype the operand ran at.</param>
        /// <returns>The eigenvalues and eigenvectors (null when <paramref name="v"/> is), complex128.</returns>
        /// <remarks>
        ///     <c>_complexType(result_t)</c> is complex64 for a float32 operand and complex128 otherwise. NumSharp has one
        ///     complex width (#569), so a float32 operand's result keeps the complex128 dtype but carries NumPy's
        ///     complex64 VALUES: geev ran in double and NumPy's <c>astype(complex64)</c> rounded each component to float32
        ///     (<see cref="RoundComponentsToSingle"/>), so the eigenvalues, the eigenvectors and everything sorted from
        ///     them (numpy.polynomial's <c>{p}roots</c>) are NumPy's values, widened exactly. For every other operand the
        ///     geev seam's complex128 already is the result: <c>astype(copy=False)</c> returns it as it is.
        /// </remarks>
        private static (NDArray eigenvalues, NDArray eigenvectors) ComplexEigResult(NDArray w, NDArray v, NPTypeCode common)
        {
            if (common == NPTypeCode.Single)
            {
                RoundComponentsToSingle(w);
                if (v is not null)
                    RoundComponentsToSingle(v);
            }

            // `w.astype(result_t, copy=False)`: the seam's arrays are already complex128, so no buffer is copied.
            w = w.astype(NPTypeCode.Complex, copy: false);
            v = v is null ? null : v.astype(NPTypeCode.Complex, copy: false);
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

        /// <summary>
        ///     NumPy 2.5's <c>numpy.linalg._linalg._to_real_if_imag_zero(w, t)</c>, the "backwards compat" step that keeps
        ///     <c>np.roots</c>, <c>polyroots</c>, <c>lagroots</c> and <c>hermroots</c> real when <c>eigvals</c> became always
        ///     complex: the real parts when the matrix was real and every imaginary part is zero, else the complex values.
        /// </summary>
        /// <param name="w">The (always complex) eigenvalues, already sorted by the caller.</param>
        /// <param name="t">The dtype of the matrix they are the eigenvalues of (NumPy's <c>t.dtype.type</c>).</param>
        /// <returns>
        ///     For a real <paramref name="t"/> whose eigenvalues are all real: <c>w.real</c> as float32 for a float32 matrix
        ///     and float64 otherwise (<c>_realType</c>) — a real-part VIEW either way (OWNDATA false, every other element of
        ///     its buffer, as NumPy's <c>w.real</c> of a complex128 or complex64 array is): of <paramref name="w"/> itself
        ///     for float64, of a float32 buffer laid out as NumPy's complex64 (re, im) pairs for float32 (NumSharp carries
        ///     complex64 values in complex128). Otherwise <paramref name="w"/> as complex (<c>_complexType</c>: csingle and
        ///     cdouble are NumSharp's one complex width), not copied.
        /// </returns>
        /// <remarks>
        ///     <c>all(w.imag == 0.0)</c> is evaluated as <c>!any(w.imag)</c>: <c>np.any</c> treats a NaN as truthy exactly as a
        ///     nonzero, so the two predicates agree on zero, nonzero and NaN parts (a -0.0 part is zero in both), and an
        ///     empty <paramref name="w"/> is real in both. Only the four roots functions above call this: <c>chebroots</c>,
        ///     <c>legroots</c> and <c>hermeroots</c> return eigvals' complex result as it is. It lives in the engine layer
        ///     because <c>np.roots</c>' engine half (<c>DefaultEngine.Roots</c>) calls it, and the engine never calls up into
        ///     the np.* facade.
        /// </remarks>
        [NDScoped]
        internal static NDArray ToRealIfImagZero(NDArray w, NPTypeCode t)
        {
            // The scope reclaims the imaginary-part view the test reads; the returned real view (or w) escapes it.
            if (t != NPTypeCode.Complex)
            {
                NDArray imag = NDArray.imag_(w);
                if (!imag.TensorEngine.Any(imag))
                    return t == NPTypeCode.Single ? Complex64RealView(w) : NDArray.real_(w).astype(NPTypeCode.Double, copy: false);
            }
            return w.astype(NPTypeCode.Complex, copy: false);
        }

        /// <summary>
        ///     NumPy's <c>w.real</c> of a complex64 array — a float32 view of every other element of the complex64 buffer
        ///     (byte stride 8 along the last axis, OWNDATA false) — for complex64 values NumSharp carries in complex128:
        ///     the (re, im) pairs are written as complex64 memory holds them, and the real lane is returned as that view.
        /// </summary>
        /// <param name="w">The complex128 eigenvalues whose components are float32 values (rounded by eigvals for a
        ///     float32 matrix), any shape.</param>
        /// <returns>The float32 real-part view, of <paramref name="w"/>'s shape; it keeps its pair buffer alive.</returns>
        /// <remarks>
        ///     A plain float32 copy would carry the same values, but its layout would not be NumPy's: NumPy returns the
        ///     view itself (<c>astype(single, copy=False)</c> of a float32 view is that view), so a C-contiguous owning
        ///     result would differ from NumPy's in its flags and strides.
        /// </remarks>
        private static NDArray Complex64RealView(NDArray w)
        {
            long[] dims = w.Shape.dimensions;
            var pairDims = new long[dims.Length + 1];
            dims.CopyTo(pairDims, 0);
            pairDims[dims.Length] = 2;
            var pairs = new NDArray(NPTypeCode.Single, new Shape(pairDims, copy: false), false);
            // C order of w's logical elements: element k occupies pairs [2k, 2k + 1]. The components are float32-exact,
            // so the narrowing casts are exact.
            long n = w.size;
            for (long k = 0; k < n; k++)
            {
                var z = w.GetAtIndex<System.Numerics.Complex>(k);
                pairs.SetAtIndex((float)z.Real, 2 * k);
                pairs.SetAtIndex((float)z.Imaginary, 2 * k + 1);
            }
            return pairs[Slice.Ellipsis, Slice.Index(0)];
        }
    }
}
