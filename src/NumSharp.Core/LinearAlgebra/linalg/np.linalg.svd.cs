using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Singular value decomposition — <c>a = U diag(S) Vh</c>.
            /// </summary>
            /// <param name="full_matrices">
            ///     True (the default) makes U and Vh full square factors; false truncates them to
            ///     <c>K = min(M, N)</c> columns/rows.
            /// </param>
            /// <param name="compute_uv">
            ///     False returns the singular values alone — <c>U</c> and <c>Vh</c> are then null.
            /// </param>
            /// <param name="hermitian">
            ///     Assume the operand is Hermitian, allowing the cheaper symmetric eigensolver.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
            ///     <para>
            ///     Singular values come back in DESCENDING order. This is the routine behind
            ///     <see cref="pinv"/>, <see cref="matrix_rank"/>, <see cref="cond"/> and the
            ///     spectral and nuclear matrix norms — which is why all of those stop here too.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            public static (NDArray U, NDArray S, NDArray Vh) svd(NDArray a, bool full_matrices = true,
                bool compute_uv = true, bool hermitian = false)
            {
                AssertStacked2d(a);
                var common = CommonType(a);
                return a.TensorEngine.Svd(ToCommon(a, common), full_matrices, compute_uv);
            }

            /// <summary>
            ///     Singular values of a matrix, descending — the Array-API spelling of
            ///     <c>svd(x, compute_uv: false)</c>.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.svdvals.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            public static NDArray svdvals(NDArray x)
            {
                AssertStacked2d(x);
                var common = CommonType(x);
                return x.TensorEngine.Svd(ToCommon(x, common), fullMatrices: false, computeUv: false).S;
            }

            /// <summary>
            ///     Rank of a matrix — the number of singular values greater than the tolerance.
            /// </summary>
            /// <param name="tol">
            ///     Absolute threshold. Defaults to <c>S.max() * max(M, N) * eps</c>.
            /// </param>
            /// <param name="hermitian">Assume the operand is Hermitian.</param>
            /// <param name="rtol">Relative threshold; supplying it together with <paramref name="tol"/> is an error.</param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            public static NDArray matrix_rank(NDArray A, double? tol = null, bool hermitian = false,
                double? rtol = null)
            {
                if (tol is not null && rtol is not null)
                    throw new ValueError("`tol` and `rtol` can't be both set.");

                // NumPy short-circuits below rank 2 without any decomposition: `int(not all(A == 0))`.
                // That is a PREDICATE, not a count — a 0-d operand and a vector alike are rank 1
                // unless every element is zero, so matrix_rank([1,2,3]) is 1 and not 3. An empty
                // operand is rank 0, because all([]) is true.
                if (A.ndim < 2)
                    return NDArray.Scalar(A.TensorEngine.CountNonZero(A) == 0 ? 0L : 1L);

                var common = CommonType(A);
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
            ///     Condition number of a matrix in the given norm.
            /// </summary>
            /// <param name="p">
            ///     The order: <c>null</c> (default) and <c>±2</c> use singular values;
            ///     <c>±1</c>, <c>±inf</c> and <c>"fro"</c> are defined as
            ///     <c>norm(x, p) * norm(inv(x), p)</c>.
            /// </param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.cond.html
            ///     <para>
            ///     Every order needs a factorisation — the singular-value ones directly, the rest
            ///     through <see cref="inv"/> — so unlike <see cref="norm"/> there is no order that
            ///     works without a backend.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            public static NDArray cond(NDArray x, object p = null)
            {
                if (IsEmpty2d(x))
                    throw new LinAlgError("cond is not defined on empty arrays");

                NDArray r;
                if (p is null || IsOrder(p, 2) || IsOrder(p, -2))
                {
                    // The default and ±2 orders come straight from the singular values: the ratio of the
                    // largest to the smallest (2/None), or its reciprocal (-2). Division by a zero
                    // singular value gives inf/nan here rather than raising, matching NumPy's errstate.
                    var s = svdvals(x);
                    var sMax = s["..., 0"];
                    var sMin = s["..., -1"];
                    r = IsOrder(p, -2) ? np.divide(sMin, sMax) : np.divide(sMax, sMin);
                }
                else
                {
                    AssertStackedSquare(x);
                    var common = CommonType(x);
                    var realT = common == NPTypeCode.Complex ? NPTypeCode.Double : common;

                    // NumPy's own composition, verbatim — and the reason a bad order is reported by
                    // `norm` in the caller's words ("Invalid norm order for matrices.") rather than by
                    // `cond` itself. NumPy inverts IGNORING errors: a singular operand yields nan from
                    // the raw inv gufunc, which the nan→inf tail below turns into inf. NumSharp's
                    // `np.linalg.inv` RAISES on a singular matrix, so for a single (2-D) operand the raise
                    // is caught and the same nan→inf path is taken. A singular element inside a STACK
                    // still raises (per-element nan-fill would need a non-raising inv) — documented
                    // [Misaligned] divergence.
                    NDArray invx;
                    try
                    {
                        invx = inv(ToCommon(x, common));
                    }
                    catch (LinAlgError) when (x.ndim == 2)
                    {
                        return CondNanToInf(NDArray.Scalar(double.NaN).astype(realT), x);
                    }

                    r = np.multiply(norm(x, p, new[] {-2, -1}), norm(invx, p, new[] {-2, -1}));
                    r = r.astype(realT);
                }

                return CondNanToInf(r, x);
            }

            private static bool IsOrder(object p, double value)
                => p is not null && p is not string && Convert.ToDouble(p) == value;

            /// <summary>
            ///     NumPy's <c>_is_empty_2d</c> — <c>size == 0</c> AND the product of the last two
            ///     dimensions is 0 (so a zero-length BATCH like <c>(0,3,3)</c> is NOT empty here).
            /// </summary>
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
            ///     NaN only arises here from a <c>0/0</c> (an all-zero matrix), which is infinitely
            ///     ill-conditioned.
            /// </summary>
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

                return np.where(apply, NDArray.Scalar(double.PositiveInfinity).astype(r.typecode), r);
            }
        }
    }
}
