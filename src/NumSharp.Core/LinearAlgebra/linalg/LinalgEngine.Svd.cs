using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.svd</c> past the class boundary: the rank check (the square check of the hermitian
        ///     route), <c>_commonType</c>, the gufunc's text refusal, then <see cref="TensorEngine.Svd"/> at the loop
        ///     dtype.
        /// </summary>
        /// <param name="a">The matrix, or a stack of them (plain: the facade applied the class boundary).</param>
        /// <param name="full_matrices">True for square U and Vh, false for <c>K = min(M, N)</c> columns / rows.</param>
        /// <param name="compute_uv">False for the singular values alone (U and Vh null).</param>
        /// <param name="hermitian">Assume the operand is Hermitian (its square check applies).</param>
        /// <returns><c>(U, S, Vh)</c>, the singular values descending.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand with <paramref name="hermitian"/>.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the ToCommon cast temp; the (U, S, Vh) tuple is yielded component-wise
        internal static (NDArray U, NDArray S, NDArray Vh) Svd(NDArray a, bool full_matrices, bool compute_uv, bool hermitian)
        {
            // NumPy's hermitian=True route is eigh / eigvalsh: their square check, then their gufunc; otherwise
            // _assert_stacked_2d and one of the three svd gufuncs.
            if (hermitian)
                AssertStackedSquare(a);
            else
                AssertStacked2d(a);
            var common = CommonType(a);
            ThrowIfTextGufunc(hermitian ? (compute_uv ? "eigh_lo" : "eigvalsh_lo")
                                        : compute_uv ? (full_matrices ? "svd_f" : "svd_s") : "svd", a);
            return a.TensorEngine.Svd(ToCommon(a, common), full_matrices, compute_uv);
        }

        /// <summary>
        ///     The body of <c>np.linalg.svdvals</c>: <c>svd(x, compute_uv=False)</c>'s statements.
        /// </summary>
        /// <param name="x">The matrix, or a stack of them.</param>
        /// <returns>The singular values, descending.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped] // reclaims the ToCommon cast temp (U/Vh are null under compute_uv: false)
        internal static NDArray Svdvals(NDArray x)
        {
            AssertStacked2d(x);
            var common = CommonType(x);
            // NumPy's svd(x, compute_uv=False) calls _umath_linalg.svd: its input cast refuses text.
            ThrowIfTextGufunc("svd", x);
            return x.TensorEngine.Svd(ToCommon(x, common), fullMatrices: false, computeUv: false).S;
        }

        /// <summary>
        ///     The body of <c>np.linalg.matrix_rank</c>: the tolerance arguments, NumPy's predicate below two axes, and
        ///     otherwise the count of singular values above the tolerance.
        /// </summary>
        /// <param name="A">The matrix, or a stack of them.</param>
        /// <param name="tol">The absolute threshold; null for <c>S.max() * max(M, N) * eps</c> (or <paramref name="rtol"/>).</param>
        /// <param name="hermitian">Assume the operand is Hermitian (its square check applies).</param>
        /// <param name="rtol">The relative threshold.</param>
        /// <returns>The rank (an int64 0-d array for one matrix, one per matrix of a stack).</returns>
        /// <exception cref="ValueError">Both <paramref name="tol"/> and <paramref name="rtol"/> are set.</exception>
        /// <exception cref="LinAlgError">A non-square operand with <paramref name="hermitian"/>.</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand of two or more axes (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped]
        internal static NDArray MatrixRank(NDArray A, double? tol, bool hermitian, double? rtol)
        {
            if (tol is not null && rtol is not null)
                throw new ValueError("`tol` and `rtol` can't be both set.");

            // NumPy short-circuits below rank 2 without any decomposition: `int(not all(A == 0))`.
            // That is a PREDICATE, not a count — a 0-d operand and a vector alike are rank 1
            // unless every element is zero, so matrix_rank([1,2,3]) is 1 and not 3. An empty
            // operand is rank 0, because all([]) is true.
            if (A.ndim < 2)
            {
                // Text: NumPy's `A == 0` has no string loop, so ndarray.__eq__ answers all-False (a 0-d operand:
                // Python's False) without reading the text — rank 1, or 0 for an empty operand (all([]) is True).
                if (Backends.Strings.NDStringUfuncs.IsText(A))
                    return NDArray.Scalar(A.size == 0 ? 0L : 1L);
                return NDArray.Scalar(A.TensorEngine.CountNonZero(A) == 0 ? 0L : 1L);
            }

            // NumPy's svd(A, compute_uv=False, hermitian=hermitian): the eigvalsh route checks squareness first.
            if (hermitian)
                AssertStackedSquare(A);
            var common = CommonType(A);
            ThrowIfTextGufunc(hermitian ? "eigvalsh_lo" : "svd", A);
            var s = A.TensorEngine.Svd(ToCommon(A, common), fullMatrices: false, computeUv: false).S;

            int nd = A.ndim;
            long m = A.Shape.dimensions[nd - 2];
            long n = A.Shape.dimensions[nd - 1];

            NDArray tolArr;
            if (tol is null)
            {
                // Default: max(M,N) * eps(S.dtype) * largest singular value (MATLAB's rule). The eps is
                // the singular values' own — float32's for a float32 operand, float64's otherwise.
                double eps = s.typecode == NPTypeCode.Single ? 1.1920928955078125e-07 : 2.220446049250313e-16;
                double rtolVal = rtol ?? (Math.Max(m, n) * eps);
                tolArr = np.multiply(np.amax(s, -1, keepdims: true), NDArray.Scalar(rtolVal));
            }
            else
            {
                // asarray(tol)[..., newaxis] — a scalar threshold broadcasts against the last axis.
                tolArr = NDArray.Scalar(tol.Value);
            }

            return np.count_nonzero(np.greater(s, tolArr), -1);
        }

        /// <summary>
        ///     The body of <c>np.linalg.cond</c>: the empty-operand refusal, the hashability of <paramref name="p"/>, the
        ///     singular-value orders (through the public <c>np.linalg.svdvals</c>), or <c>norm(x, p) * norm(inv(x), p)</c>
        ///     for the others (the inverse in float64 / complex128, NaN for a singular matrix), and NumPy's NaN-to-inf tail.
        /// </summary>
        /// <param name="x">The matrix, or a stack of them.</param>
        /// <param name="p">The order: null / ±2 (singular values), ±1, ±inf, <c>"fro"</c>, <c>"nuc"</c>.</param>
        /// <returns>The condition number (a 0-d array for one matrix, one per matrix of a stack), real; inf for a singular
        ///     matrix.</returns>
        /// <exception cref="LinAlgError">An empty operand (<c>cond is not defined on empty arrays</c>), a non-square one
        ///     for an inverse-based order, or the nuclear norm of a singular matrix (its NaN inverse: <c>SVD did not
        ///     converge</c>).</exception>
        /// <exception cref="TypeError">An array or list <paramref name="p"/> (unhashable), or a float16, decimal or char
        ///     operand (<c>_commonType</c>).</exception>
        /// <exception cref="ValueError">An order the norm refuses (<c>Invalid norm order for matrices.</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped]
        internal static NDArray Cond(NDArray x, object p)
        {
            if (IsEmpty2d(x))
                throw new LinAlgError("cond is not defined on empty arrays");
            // `p is None or p in {2, -2}`: the set membership hashes p, which an ndarray (a typed C# array too, the house
            // map) and a Python list cannot be — whatever they hold, text included.
            if (p is NDArray || (p is Array && p is not string) || PolySequenceLike(p))
                throw new TypeError($"unhashable type: '{(p is NDArray || p.GetType().IsArray && p.GetType().GetElementType() != typeof(object) ? "numpy.ndarray" : "list")}'");

            NDArray r;
            if (p is null || IsOrder(p, 2) || IsOrder(p, -2))
            {
                // The default and ±2 orders come straight from the singular values: the ratio of the
                // largest to the smallest (2/None), or its reciprocal (-2). Division by a zero
                // singular value gives inf/nan here rather than raising, matching NumPy's errstate.
                var s = np.linalg.svdvals(x);
                // NumPy's `s[..., 0]` / `s[..., -1]`. Built as Slice[] rather than the "..., 0"
                // STRING form, whose per-call Regex.Split+Regex.Match parse (Slice.ParseSlices)
                // costs ~1.7µs EACH — the dominant managed overhead of this tiny wrapper, on top
                // of an already parity-or-faster gesdd. The Slice[] path resolves identically
                // (`Slice.Index` is index-reduction on the last axis after Ellipsis fills the
                // leading ones), so the extracted first/last singular values are byte-identical.
                var sMax = s[new[] {Slice.Ellipsis, Slice.Index(0)}];
                var sMin = s[new[] {Slice.Ellipsis, Slice.Index(-1)}];
                r = IsOrder(p, -2) ? np.divide(sMin, sMax) : np.divide(sMax, sMin);
            }
            else
            {
                AssertStackedSquare(x);
                var common = CommonType(x);
                // NumPy calls _umath_linalg.inv(x, signature=...) next: its input cast refuses text.
                ThrowIfTextGufunc("inv", x);
                // result_t = _realType(result_t): the condition number is real.
                var realT = common == NPTypeCode.Complex ? NPTypeCode.Double : common;

                // NumPy's statements, in order:
                //     signature = 'D->D' if isComplexType(t) else 'd->d'
                //     with errstate(all='ignore'):
                //         invx = _umath_linalg.inv(x, signature=signature)
                //         r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
                //     r = r.astype(result_t, copy=False)
                // The inverse runs in float64 / complex128 whatever result_t is (a float32 matrix's inverse and its norm
                // are float64), and a singular matrix's inverse is NaN instead of an error. Both norms are evaluated, x's
                // first: a bad order is refused by norm ("Invalid norm order for matrices.") even for a singular matrix,
                // and the nuclear norm of a singular matrix's NaN inverse raises the SVD's "SVD did not converge".
                var invx = InvIgnoringErrors(ToCommon(x, common == NPTypeCode.Complex ? NPTypeCode.Complex : NPTypeCode.Double));
                var normX = np.linalg.norm(x, p, new[] {-2, -1});
                var normInv = np.linalg.norm(invx, p, new[] {-2, -1});
                // One matrix's two norms are NumPy scalars, so `*` is scalarmath (the float64 product: a float32 norm
                // widens exactly) and astype keeps a scalar; a stack's are arrays, multiplied by the ufunc.
                r = normX.ndim == 0
                    ? NDArray.Scalar(normX.astype(np.float64).GetDouble() * normInv.GetDouble()).astype(realT).MarkReductionScalar()
                    : np.multiply(normX, normInv).astype(realT, false);
            }

            return CondNanToInf(r, x);
        }

        /// <summary>
        ///     NumPy's <c>_umath_linalg.inv(x, signature=...)</c> under <c>errstate(all='ignore')</c>, as np.linalg.cond calls
        ///     it: every matrix is inverted on its own (the gufunc's loop), and one that the factorisation finds singular is
        ///     filled with NaN (umath_linalg's <c>nan_matrix</c>) where np.linalg.inv raises.
        /// </summary>
        /// <param name="x">The square matrices, float64 or complex128 (the <c>'d->d'</c> / <c>'D->D'</c> loop).</param>
        /// <returns>The inverses, of <paramref name="x"/>'s shape and dtype; NaN (NumPy's positive quiet NaN, both parts of a
        ///     complex element) throughout each singular matrix.</returns>
        /// <remarks>
        ///     Nothing is raised for a singular matrix, and no backend is needed: without one the managed LU inverts
        ///     float64 and complex128 matrices. The engine's inverse raises <c>LinAlgError("Singular matrix")</c> at the
        ///     first singular matrix, so a stack is inverted whole first (one call, the common case) and only after that
        ///     error matrix by matrix. Each matrix is factorised independently on both routes (the OpenBLAS <c>gesv</c> loop
        ///     and the managed LU), so every regular matrix's inverse is the same bits either way.
        /// </remarks>
        [NDScopedCovered] // only caller: [NDScoped] cond; the per-matrix views, inverses and the NaN scalar ride its scope
        private static NDArray InvIgnoringErrors(NDArray x)
        {
            try
            {
                return x.TensorEngine.Inv(x);
            }
            catch (LinAlgError e) when (e.Message == "Singular matrix")
            {
                // errstate(all='ignore') swallows only the gufunc's invalid flag, which a singular factorisation sets;
                // any other error still propagates.
            }

            int nd = x.ndim;
            long m = x.Shape.dimensions[nd - 1];
            // a singular matrix was found, so there is at least one element: m > 0 and the matrix count is size / m^2
            long count = x.size / (m * m);
            var matrices = np.reshape(x, Shape.Adopt(new[] {count, m, m}));
            var inverses = np.empty(Shape.Adopt(new[] {count, m, m}), x.dtype);
            double nan = BitConverter.Int64BitsToDouble(0x7FF8000000000000L); // NPY_NAN; .NET's double.NaN is negative
            var nanValue = x.typecode == NPTypeCode.Complex
                ? NDArray.Scalar(new System.Numerics.Complex(nan, nan))
                : NDArray.Scalar(nan);
            for (long k = 0; k < count; k++)
            {
                var slot = inverses[new[] {Slice.Index(k)}];
                try
                {
                    np.copyto(slot, x.TensorEngine.Inv(matrices[new[] {Slice.Index(k)}]));
                }
                catch (LinAlgError e) when (e.Message == "Singular matrix")
                {
                    np.copyto(slot, nanValue);
                }
            }

            return np.reshape(inverses, Shape.Adopt((long[])x.Shape.dimensions.Clone()));
        }

        /// <summary>Whether an order is the number <paramref name="value"/> (NumPy's <c>p in {2, -2}</c> and <c>p == -2</c>).</summary>
        /// <param name="p">The order argument.</param>
        /// <param name="value">The number to compare with.</param>
        /// <returns>True for a non-null, non-string order equal to <paramref name="value"/>.</returns>
        /// <exception cref="InvalidCastException">An order that does not convert to a double.</exception>
        /// <exception cref="FormatException">An order whose conversion to a double parses text and fails.</exception>
        private static bool IsOrder(object p, double value)
            => p is not null && p is not string && Convert.ToDouble(p) == value;

        /// <summary>Whether <paramref name="p"/> is a Python list for NumPy's purposes (an <c>object[]</c> or another
        ///     non-array <see cref="System.Collections.IList"/>): unhashable, unlike a tuple.</summary>
        /// <param name="p">The order argument.</param>
        /// <returns>True for a list.</returns>
        private static bool PolySequenceLike(object p)
            => p is System.Collections.IList && p is not System.Runtime.CompilerServices.ITuple;

        /// <summary>
        ///     NumPy's <c>_is_empty_2d</c> — <c>size == 0</c> AND the product of the last two
        ///     dimensions is 0 (so a zero-length BATCH like <c>(0,3,3)</c> is NOT empty here).
        /// </summary>
        /// <param name="x">The operand.</param>
        /// <returns>True for an empty operand whose last two dimensions hold no element.</returns>
        private static bool IsEmpty2d(NDArray x)
        {
            if (x.size != 0)
                return false;

            int nd = x.ndim;
            long prod = 1;
            for (int i = Math.Max(0, nd - 2); i < nd; i++)
                prod *= x.Shape.dimensions[i];
            return prod == 0;
        }

        /// <summary>
        ///     NumPy's tail of <c>cond</c>: turn a NaN condition number into <c>inf</c> — UNLESS the
        ///     matrix itself held a NaN (over its last two axes), in which case the NaN is kept. A
        ///     NaN arises from a <c>0/0</c> (an all-zero matrix) or from a singular matrix's NaN
        ///     inverse, both infinitely ill-conditioned.
        /// </summary>
        /// <param name="r">The condition numbers.</param>
        /// <param name="x">The operand they came from.</param>
        /// <returns><paramref name="r"/> itself when it holds no NaN to replace; else, for one matrix, NumPy's scalar
        ///     <c>r.dtype.type(inf)</c> (a read-only 0-d array), and for a stack a fresh array with the NaNs replaced.</returns>
        [NDScopedCovered] // only caller: [NDScoped] cond (the isnan/logical/inf-scalar temps are scope-reclaimed)
        private static NDArray CondNanToInf(NDArray r, NDArray x)
        {
            var nanMask = np.isnan(r);
            if (!np.any(nanMask))
                return r;

            NDArray apply = nanMask;
            if (x.typecode is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half or NPTypeCode.Complex)
            {
                var xNan = np.any(np.isnan(x), new[] {-2, -1}, null);
                apply = np.logical_and(nanMask, np.logical_not(xNan));
            }

            // NumPy's `elif nan_mask: r = r.dtype.type(inf)` ("Convention is to return scalars instead of 0d arrays"): a
            // new scalar of r's dtype, or r itself when the matrix held the NaN.
            if (r.ndim == 0)
                return apply.GetBoolean()
                    ? NDArray.Scalar(double.PositiveInfinity).astype(r.typecode).MarkReductionScalar()
                    : r;

            return np.where(apply, NDArray.Scalar(double.PositiveInfinity).astype(r.typecode), r);
        }
    }
}
