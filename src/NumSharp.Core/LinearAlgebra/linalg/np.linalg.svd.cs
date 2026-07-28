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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
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
                A.TensorEngine.Svd(ToCommon(A, common), fullMatrices: false, computeUv: false);

                throw new NotSupportedException(
                    "np.linalg.matrix_rank is not implemented for operands of two dimensions or more. " +
                    "Validation and the rank-0/rank-1 short circuits are complete and the decomposition " +
                    "the general case is built on is reached above, which is where the call stops while " +
                    "no LAPACK backend is installed. This line is the REMAINING work rather than the " +
                    "missing backend: counting the singular values above the tolerance still has to be " +
                    "written.");
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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static NDArray cond(NDArray x, object p = null)
            {
                if (p is null || IsOrder(p, 2) || IsOrder(p, -2))
                {
                    AssertStacked2d(x);
                    var svdCommon = CommonType(x);
                    x.TensorEngine.Svd(ToCommon(x, svdCommon), fullMatrices: false, computeUv: false);

                    throw new NotSupportedException(
                        "np.linalg.cond is not implemented. Validation is complete and the " +
                        "decomposition this order is defined by is reached above, which is where the " +
                        "call stops while no LAPACK backend is installed. This line is the REMAINING " +
                        "work rather than the missing backend: forming the ratio of the largest and " +
                        "smallest singular values still has to be written.");
                }

                AssertStackedSquare(x);
                var common = CommonType(x);

                // NumPy's own composition, verbatim — and the reason a bad order is reported by
                // `norm` in the caller's words ("Invalid norm order for matrices.") rather than by
                // `cond` itself.
                return np.multiply(norm(x, p, new[] {-2, -1}), norm(inv(ToCommon(x, common)), p, new[] {-2, -1}));
            }

            private static bool IsOrder(object p, double value)
                => p is not null && p is not string && Convert.ToDouble(p) == value;
        }
    }
}
