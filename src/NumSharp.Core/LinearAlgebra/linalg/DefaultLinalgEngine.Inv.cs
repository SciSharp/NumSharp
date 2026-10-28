using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, <c>_commonType</c>, the text refusal of the <c>inv</c>
        ///     gufunc's input cast, then <see cref="InvKernel"/> at the loop dtype.
        /// </remarks>
        // Scope: ToCommon mints a dtype-cast temp when `a` is not already the common type
        // (a.typecode == common ? a : Cast(...)) — a leftover otherwise dropped to the finalizer.
        // [NDScoped] reclaims it while leaving a passthrough input untouched (rule R2); the
        // engine's fresh inverse is yielded. Matches the scoped siblings pinv/tensorinv.
        [NDScoped]
        public override NDArray Inv(NDArray a)
        {
            AssertStackedSquare(a);
            var common = CommonType(a);
            // NumPy calls _umath_linalg.inv: its input cast refuses text, an empty operand included.
            ThrowIfTextGufunc("inv", a);
            return InvKernel(ToCommon(a, common));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the tolerance arguments, NumPy's empty-operand answer (<c>_is_empty_2d</c>, the
        ///     operand's own dtype), the conjugation, the singular value decomposition through this engine's
        ///     <see cref="Svd(NDArray, bool, bool, bool)"/>, the cutoff and the recombination of the factors.
        /// </remarks>
        [NDScoped]
        public override NDArray Pinv(NDArray a, double? rcond = null, bool hermitian = false, double? rtol = null)
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
            // NumPy's `u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)`: with hermitian, the eigh-based factors
            // of the matrix the conjugate's lower triangle holds.
            var (u, s, vt) = Svd(aConj, full_matrices: false, compute_uv: true, hermitian);

            // Discard singular values at or below rcond * largest, reciprocate the rest, and recombine
            // with the (plain, since `a` was conjugated) transposed factors — NumPy's exact sequence.
            var cutoff = np.multiply(np.amax(s, -1, keepdims: true), NDArray.Scalar(rc));
            var large = np.greater(s, cutoff);
            var sInv = np.where(large, np.reciprocal(s), np.zeros_like(s));
            return np.matmul(np.swapaxes(vt, -1, -2),
                np.multiply(np.expand_dims(sInv, -1), np.swapaxes(u, -1, -2)));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the shape arithmetic, the reshape to a square matrix, its inverse through the public
        ///     <c>np.linalg.inv</c> (so the engine's <see cref="Inv"/> computes it) and the reshape to the inverted shape.
        /// </remarks>
        [NDScoped]
        public override NDArray Tensorinv(NDArray a, int ind = 2)
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
            var inverse = np.linalg.inv(square);
            // `return ia.reshape(*invshape)`: a 0-d operand's invshape is (), and Python calls reshape() with no argument,
            // which ndarray.reshape refuses - after the inverse, so a singular or unsupported 0-d operand reports inv's
            // error first.
            if (invertedShape.Length == 0)
                throw new TypeError("reshape() takes exactly 1 argument (0 given)");
            return np.reshape(inverse, invertedShape);
        }
    }
}
