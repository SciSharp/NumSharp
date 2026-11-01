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
        ///     <see cref="Svd(NDArray, bool, bool, bool)"/>, the cutoff and the recombination of the factors. Every
        ///     main-namespace call of those statements is the member its facade dispatches to (the operand engine's
        ///     conjugate, amax, multiply, greater, reciprocal, swapaxes and matmul; the array object's allocation,
        ///     where and expand_dims), evaluated in the statements' order.
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
                // np.zeros(shape, a.dtype)'s own allocation (a zero-filled owner of the operand's dtype, text included),
                // without the call up into np.*.
                return new NDArray(a.dtype, Shape.Adopt(emptyDims), true);
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
            // `a = a.conjugate()`: np.conjugate's member on the loop-dtype operand.
            var operand = ToCommon(a, common);
            var aConj = operand.TensorEngine.Conjugate(operand);
            // NumPy's `u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)`: with hermitian, the eigh-based factors
            // of the matrix the conjugate's lower triangle holds.
            var (u, s, vt) = Svd(aConj, full_matrices: false, compute_uv: true, hermitian);

            // Discard singular values at or below rcond * largest, reciprocate the rest, and recombine
            // with the (plain, since `a` was conjugated) transposed factors — NumPy's exact sequence.
            // NumPy's `cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)`: np.amax's member, then
            // np.multiply's, the maximum on the left as this statement always had it.
            var largest = s.TensorEngine.ReduceAMax(s, -1, true, null, null, null, null);
            var cutoff = largest.TensorEngine.Multiply(largest, NDArray.Scalar(rc));
            // `large = s > cutoff` (np.greater's member), then NumPy's `s = divide(1, s, where=large, out=s);
            // s[~large] = 0` as np.where(large, reciprocal(s), zeros_like(s)) - its arguments evaluated left to right,
            // as the call evaluated them.
            var large = s.TensorEngine.Greater(s, cutoff);
            var reciprocals = s.TensorEngine.Reciprocal(s);
            // np.zeros_like(s)'s own allocation: order 'K' keeps s's axis permutation, zero-filled, s's dtype.
            var zeros = new NDArray(s.dtype, Shape.NewLikeLayout(s.Shape, 'K', (long[])s.Shape.dimensions.Clone()), true);
            var sInv = NDArray.where(large, reciprocals, zeros);
            // `res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))`, the operands in that order.
            var vtT = vt.TensorEngine.SwapAxes(vt, -1, -2);
            var column = NDArray.expand_dims(sInv, -1);
            var uT = u.TensorEngine.SwapAxes(u, -1, -2);
            return CoreMatmul(vtT, column.TensorEngine.Multiply(column, uT));
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: the shape arithmetic, the reshape to a square matrix (the array object's
        ///     <c>reshape</c>, np.reshape's member), its inverse through this engine's <see cref="Inv"/> (NumPy's
        ///     <c>inv(a)</c>, so a subclass that overrides it computes tensorinv's inverse too) and the reshape to the
        ///     inverted shape.
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
            // ("cannot reshape array of size 0 into shape (0,newaxis)"). The array object's reshape, np.reshape's member:
            // np.linalg.tensorinv's plain region already kept np.reshape's class rules from firing on a matrix or a
            // numpy.char.chararray operand, so the view is the one np.reshape returned here.
            var square = a.reshape(new[] {prod, -1L});
            var inverse = Inv(square);
            // `return ia.reshape(*invshape)`: a 0-d operand's invshape is (), and Python calls reshape() with no argument,
            // which ndarray.reshape refuses - after the inverse, so a singular or unsupported 0-d operand reports inv's
            // error first.
            if (invertedShape.Length == 0)
                throw new TypeError("reshape() takes exactly 1 argument (0 given)");
            return inverse.reshape(invertedShape);
        }
    }
}
