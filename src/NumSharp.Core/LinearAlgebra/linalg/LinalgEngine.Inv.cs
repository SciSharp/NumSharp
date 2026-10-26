using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.inv</c>: <c>_assert_stacked_square</c>, <c>_commonType</c>, the gufunc's text
        ///     refusal, then <see cref="TensorEngine.Inv"/> at the loop dtype.
        /// </summary>
        /// <param name="a">The square matrix, or a stack of them (plain: the facade applied the class boundary).</param>
        /// <returns>The inverse, a fresh array of the loop dtype.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a singular matrix (<c>Singular
        ///     matrix</c>, from the factorisation).</exception>
        /// <exception cref="TypeError">A float16, decimal or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="ArgumentException">A text operand (the gufunc's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        // Scope: ToCommon mints a dtype-cast temp when `a` is not already the common type
        // (a.typecode == common ? a : Cast(...)) — a leftover otherwise dropped to the finalizer.
        // [NDScoped] reclaims it while leaving a passthrough input untouched (rule R2); the
        // engine's fresh inverse is yielded. Matches the scoped siblings pinv/tensorinv.
        [NDScoped]
        internal static NDArray Inv(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.inv: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("inv", a);
            return a.TensorEngine.Inv(ToCommon(a, common));
        }

        /// <summary>
        ///     The body of <c>np.linalg.pinv</c>: the tolerance arguments, NumPy's empty-operand answer, the conjugation,
        ///     the singular value decomposition, the cutoff and the recombination of the factors.
        /// </summary>
        /// <param name="a">The matrix, or a stack of them (plain: the facade applied the class boundary).</param>
        /// <param name="rcond">The relative cutoff; null with <paramref name="rtol"/> null means NumPy's 1e-15.</param>
        /// <param name="hermitian">Assume the operand is Hermitian (its square check applies).</param>
        /// <param name="rtol">The Array-API spelling of <paramref name="rcond"/>.</param>
        /// <returns>The pseudo-inverse, shape <c>(..., N, M)</c>: float64 / float32 / complex128 by the loop dtype, the
        ///     operand's own dtype for an empty operand.</returns>
        /// <exception cref="ValueError">Both <paramref name="rcond"/> and <paramref name="rtol"/> are set, or an empty
        ///     operand of fewer than two axes (Python's unpack of <c>a.shape[-2:]</c>).</exception>
        /// <exception cref="TypeError">A text operand (<c>cannot conjugate non-numeric dtype</c>), or a float16, decimal
        ///     or char operand (<c>_commonType</c>).</exception>
        /// <exception cref="LinAlgError">Fewer than two axes, or a non-square operand with <paramref name="hermitian"/>.</exception>
        /// <exception cref="OpenBlasMissingBackendException">A non-empty operand that no matrix backend serves.</exception>
        [NDScoped]
        internal static NDArray Pinv(NDArray a, double? rcond, bool hermitian, double? rtol)
        {
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
            if (Backends.Strings.NDStringUfuncs.IsText(a))
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
        ///     The body of <c>np.linalg.tensorinv</c>: NumPy's shape statements, the reshape to a square matrix, its
        ///     inverse (through the public <c>np.linalg.inv</c>) and the reshape to the inverted shape.
        /// </summary>
        /// <param name="a">The tensor to invert.</param>
        /// <param name="ind">How many LEADING axes form the "row" side; must be positive.</param>
        /// <returns>The inverse, shape <c>a.shape[ind:] + a.shape[:ind]</c>.</returns>
        /// <exception cref="ValueError"><paramref name="ind"/> is not positive (<c>Invalid ind argument.</c>), or the
        ///     reshape to a square matrix is impossible.</exception>
        /// <exception cref="LinAlgError">The reshaped matrix is not square, or is singular.</exception>
        /// <exception cref="OpenBlasMissingBackendException">No matrix backend serves these operands.</exception>
        [NDScoped]
        internal static NDArray Tensorinv(NDArray a, int ind)
        {
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
            return np.reshape(np.linalg.inv(square), invertedShape);
        }
    }
}
