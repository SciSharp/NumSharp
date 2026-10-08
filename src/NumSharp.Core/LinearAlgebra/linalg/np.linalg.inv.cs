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
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            // Scope: ToCommon mints a dtype-cast temp when `a` is not already the common type
            // (a.typecode == common ? a : Cast(...)) — a leftover otherwise dropped to the finalizer.
            // [NDScoped] reclaims it while leaving a passthrough input untouched (rule R2); the
            // engine's fresh inverse is yielded. Matches the scoped siblings pinv/tensorinv.
            [NDScoped]
            public static NDArray inv(NDArray a)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped(a, static s => inv(s));
                AssertStackedSquare(a);
                var common = CommonType(a);
                // NumPy calls _umath_linalg.inv: its input cast refuses text, an empty operand included.
                ThrowIfTextGufunc("inv", a);
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
            /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves (an empty operand or a zero-size batch needs none).</exception>
            /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
            [NDScoped]
            public static NDArray pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null)
            {
                // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
                if (NDMatrixBoundary.Sees(a))
                    return NDMatrixBoundary.Wrapped((a, rcond, hermitian, rtol), static s => pinv(s.a, s.rcond, s.hermitian, s.rtol));
                if (rcond is not null && rtol is not null)
                    throw new ValueError("`rtol` and `rcond` can't be both set.");

                // C# cannot tell NumPy's rtol=_NoValue (default → 1e-15) apart from rtol=None
                // (Array-API default → max(M,N)*eps), so an explicit rcond or rtol is honoured and with
                // neither set NumPy's 1e-15 default applies.
                double rc = rcond ?? rtol ?? 1e-15;

                int nd = a.ndim;

                // NumPy's _is_empty_2d(a) comes before anything reads the matrix: an empty operand returns an
                // uninitialised (…, N, M) WITHOUT decomposing, keeping the ORIGINAL dtype (a string operand too —
                // unlike the SVD path, which is always floating). A 1-D empty operand fails NumPy's
                // `m, n = a.shape[-2:]` unpack.
                if (IsEmpty2d(a))
                {
                    if (nd < 2)
                        throw new ValueError($"not enough values to unpack (expected 2, got {nd})");
                    var emptyDims = (long[]) a.Shape.dimensions.Clone();
                    long m0 = emptyDims[nd - 2];
                    emptyDims[nd - 2] = emptyDims[nd - 1];
                    emptyDims[nd - 1] = m0;
                    return np.zeros(Shape.Adopt(emptyDims), a.dtype);
                }

                // NumPy's `a = a.conjugate()` is next: ndarray.conjugate refuses a non-numeric dtype before svd
                // checks the rank, so text never reaches the rank check below.
                if (Backends.Strings.NDStringUfuncs.IsString(a))
                    throw new TypeError("cannot conjugate non-numeric dtype");

                // svd's _assert_stacked_2d, and with hermitian=True its eigh route's square check.
                AssertStacked2d(a);
                if (hermitian)
                    AssertStackedSquare(a);

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
                using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
                if (ind <= 0)
                    throw new ValueError("Invalid ind argument.");

                // NumPy's statements verbatim: invshape = oldshape[ind:] + oldshape[:ind] and prod = the product of
                // oldshape[ind:] — Python slices clamp, so an `ind` past the rank leaves prod 1 and invshape the old
                // shape (a 0-d or 1-D operand still reaches inv, which reports its own error).
                var dims = a.Shape.dimensions;
                int nd = a.ndim;
                int split = Math.Min(ind, nd);
                long prod = 1;
                var invertedShape = new long[nd];
                int j = 0;
                for (int i = split; i < nd; i++)
                {
                    invertedShape[j++] = dims[i];
                    prod *= dims[i];
                }
                for (int i = 0; i < split; i++)
                    invertedShape[j++] = dims[i];

                // a.reshape(prod, -1): an empty operand with prod 0 is NumPy's own reshape refusal
                // ("cannot reshape array of size 0 into shape (0,newaxis)").
                var square = np.reshape(a, new[] {prod, -1L});
                return np.reshape(inv(square), invertedShape);
            }
        }
    }
}
