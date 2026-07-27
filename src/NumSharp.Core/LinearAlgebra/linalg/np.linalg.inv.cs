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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
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
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
            public static NDArray pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null)
            {
                if (rcond is not null && rtol is not null)
                    throw new ValueError("`rtol` and `rcond` can't be both set.");

                AssertStacked2d(a);
                var common = CommonType(a);

                // Every route through pinv is defined by singular values, so the operand reaches the
                // decomposition before any of pinv's own arithmetic — and that is where the call
                // stops today. Neither the cutoff nor the Hermitian shortcut changes it.
                a.TensorEngine.Svd(ToCommon(a, common), fullMatrices: false, computeUv: true);

                throw new NotSupportedException(
                    "np.linalg.pinv is not implemented. Its validation and dtype resolution are " +
                    "complete and the decomposition it is built on is reached above, which is where " +
                    "the call stops while no LAPACK backend is installed. This line is the REMAINING " +
                    "work rather than the missing backend: pinv still needs its own reconstruction — " +
                    "discard the singular values below the cutoff, reciprocate the rest, and recombine " +
                    "with the conjugate-transposed factors.");
            }

            /// <summary>
            ///     Inverse of an N-dimensional array with respect to a <see cref="tensordot"/>
            ///     contraction over its last <c>a.ndim - ind</c> axes.
            /// </summary>
            /// <param name="ind">
            ///     How many LEADING axes form the "row" side. Must be positive.
            /// </param>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html</remarks>
            /// <exception cref="NotSupportedException">No LAPACK backend serves these operands.</exception>
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
