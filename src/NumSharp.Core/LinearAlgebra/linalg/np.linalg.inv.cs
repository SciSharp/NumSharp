using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Multiplicative inverse of a matrix, or of each matrix in a stack.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html
            ///     <para>
            ///     NumPy does not call an explicit inversion routine: it solves <c>a x = I</c> with
            ///     <c>gesv</c>, which is why a singular operand surfaces as
            ///     <c>LinAlgError("Singular matrix")</c> from the factorisation rather than from a
            ///     determinant test.
            ///     </para>
            /// </remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            public static NDArray inv(NDArray a)
            {
                AssertStackedSquare(a);
                var common = CommonType(a);
                return a.TensorEngine.Inv(ToCommon(a, common));
            }

            /// <summary>
            ///     Moore-Penrose pseudo-inverse, computed from the singular value decomposition.
            /// </summary>
            /// <param name="rcond">
            ///     Singular values below <c>rcond * largest</c> are treated as zero. NumPy's default
            ///     is <c>1e-15</c>.
            /// </param>
            /// <param name="hermitian">
            ///     Assume the operand is Hermitian, allowing the cheaper symmetric eigensolver.
            /// </param>
            /// <param name="rtol">
            ///     The Array-API spelling of <paramref name="rcond"/>. Supplying both is an error, as
            ///     upstream.
            /// </param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped]
            public static NDArray pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null)
            {
                if (rcond is not null && rtol is not null)
                    throw new ValueError("`rtol` and `rcond` can't be both set.");

                AssertStacked2d(a);

                // C# cannot tell NumPy's rtol=_NoValue (default → 1e-15) apart from rtol=None
                // (Array-API default → max(M,N)*eps), so an explicit rcond or rtol is honoured and with
                // neither set NumPy's 1e-15 default applies.
                double rc = rcond ?? rtol ?? 1e-15;

                int nd = a.ndim;
                long m = a.Shape.dimensions[nd - 2];
                long n = a.Shape.dimensions[nd - 1];

                // Empty matrix: NumPy returns an uninitialised (…, N, M) WITHOUT decomposing, keeping the
                // ORIGINAL dtype (unlike the SVD path, which is always floating).
                if (m == 0 || n == 0)
                {
                    var emptyDims = (long[]) a.Shape.dimensions.Clone();
                    emptyDims[nd - 2] = n;
                    emptyDims[nd - 1] = m;
                    return np.zeros(new Shape(emptyDims), a.typecode);
                }

                var common = CommonType(a);
                var aConj = np.conjugate(ToCommon(a, common));
                var (u, s, vt) = aConj.TensorEngine.Svd(aConj, fullMatrices: false, computeUv: true);

                // Discard singular values at or below rcond * largest, reciprocate the rest, and recombine
                // with the (plain, since `a` was conjugated) transposed factors — NumPy's exact sequence.
                var cutoff = np.multiply(np.amax(s, -1, keepdims: true), NDArray.Scalar(rc));
                var large = np.greater(s, cutoff);
                var sInv = np.where(large, np.reciprocal(s), np.zeros_like(s));
                return np.matmul(np.swapaxes(vt, -1, -2),
                    np.multiply(np.expand_dims(sInv, -1), np.swapaxes(u, -1, -2)));
            }

            /// <summary>
            ///     Inverse of an N-dimensional array with respect to a <see cref="tensordot"/>
            ///     contraction over its last <c>a.ndim - ind</c> axes.
            /// </summary>
            /// <param name="ind">
            ///     How many LEADING axes form the "row" side. Must be positive.
            /// </param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html</remarks>
            /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
            [NDScoped]
            public static NDArray tensorinv(NDArray a, int ind = 2)
            {
                if (ind <= 0)
                    throw new ValueError("Invalid ind argument.");

                var dims = a.Shape.dimensions;
                long leading = 1;
                for (int i = 0; i < ind && i < a.ndim; i++)
                    leading *= dims[i];
                long trailing = 1;
                for (int i = ind; i < a.ndim; i++)
                    trailing *= dims[i];

                var invertedShape = new long[a.ndim - ind + ind];
                for (int i = ind, j = 0; i < a.ndim; i++, j++)
                    invertedShape[j] = dims[i];
                for (int i = 0, j = a.ndim - ind; i < ind; i++, j++)
                    invertedShape[j] = dims[i];

                var square = np.reshape(a, new[] {leading, trailing});
                return np.reshape(inv(square), invertedShape);
            }
        }
    }
}
