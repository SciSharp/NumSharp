using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's statements: <c>_assert_stacked_square</c>, the inverse for a negative power (through this engine's
        ///     <see cref="Inv"/>, NumPy's <c>inv(a)</c>, so a subclass that overrides it computes this inverse too), the
        ///     identity for 0, the operand itself for 1, and binary exponentiation with the core <c>matmul</c> above
        ///     (<see cref="CoreMatmul"/>, the member <see cref="np.matmul"/> dispatches to).
        /// </remarks>
        [NDScoped]
        public override NDArray MatrixPower(NDArray a, int n)
        {
            AssertStackedSquare(a);

            var work = a;
            if (n < 0)
            {
                // Raises without a backend — deliberately BEFORE the identity/short-power cases,
                // exactly as NumPy inverts first and then exponentiates the inverse.
                work = Inv(a);
                n = -n;
            }

            if (n == 0)
                return Identity(work);
            // NumPy's short-cut `if n == 1: return a` hands back the operand itself (or, for n == -1, the fresh
            // inverse) — the same object, so `matrix_power(a, 1) is a` and the result shares a's memory and flags.
            if (n == 1)
                return work;
            if (n == 2)
                return CoreMatmul(work, work);
            if (n == 3)
                return CoreMatmul(CoreMatmul(work, work), work);

            // Binary exponentiation. `result` stays null until the first set bit so the identity
            // multiply is skipped, which is what NumPy's `if result is None` does.
            NDArray result = null;
            var z = work;
            while (n > 0)
            {
                if ((n & 1) == 1)
                    result = result is null ? z : CoreMatmul(result, z);

                n >>= 1;
                if (n > 0)
                    z = CoreMatmul(z, z);
            }

            return result;
        }

        /// <summary>
        ///     What <c>matrix_power(a, 0)</c> returns, NumPy's two statements <c>a = empty_like(a)</c> and
        ///     <c>a[...] = eye(a.shape[-2], dtype=a.dtype)</c>: a new array of <paramref name="a"/>'s dtype, shape and memory
        ///     order (empty_like's order 'K': an F-ordered operand gives an F-ordered identity) holding the identity in every
        ///     matrix of the stack.
        /// </summary>
        /// <param name="a">The (square-stacked) operand; a numpy.char.chararray one arrives as it is.</param>
        /// <returns>A new plain array of <paramref name="a"/>'s dtype, shape and K-order layout holding the identity.</returns>
        /// <exception cref="OverflowException">The matrix side exceeds <see cref="int.MaxValue"/>.</exception>
        /// <remarks>
        ///     <para>The identity is built IN the operand's dtype, as NumPy's <c>eye(..., dtype=a.dtype)</c> is: for a text
        ///     dtype the diagonal stores the Python int 1 as its <c>str()</c> (<c>'1'</c>) and the rest stays empty, where
        ///     a float64 identity cast to text would read <c>'1.0'</c> / <c>'0.0'</c>.</para>
        ///     <para>The three statements are the members their facades dispatch to: np.eye's <see cref="NDArray.Eye"/>,
        ///     np.empty_like's allocation (a new plain array over <see cref="Shape.NewLikeLayout"/>'s K layout) and
        ///     np.copyto's <c>UnmanagedStorage.Copyto</c> with its default <c>same_kind</c> rule. None of them has a class
        ///     rule, so a numpy.char.chararray operand (the np.linalg.matrix_power facade has no chararray rule) gets the
        ///     plain identity this function always returned, without the plain region np.empty_like's chararray rule
        ///     needed (NumPy's empty_like keeps the chararray class: the class boundary's work, not this function's).</para>
        /// </remarks>
        [NDScopedCovered /* only caller: [NDScoped] MatrixPower; the eye temp rides its scope */]
        private static NDArray Identity(NDArray a)
        {
            long side = a.Shape.dimensions[a.ndim - 1];
            if (side > int.MaxValue)
                throw new OverflowException($"Matrix dimension {side} exceeds int.MaxValue for np.eye");

            // np.eye(side, dtype=a.dtype): the facade's default order 'C' (its order and device checks pass for the
            // defaults), then the array object's identity.
            var eye = NDArray.Eye((int)side, null, 0, a.dtype, 'C');
            // empty_like keeps the operand's order (a C-ordered copy of eye did not), and the assignment broadcasts the one
            // identity over the stack's leading axes. np.empty_like(a) is an uninitialised array of a's dtype in a's K
            // layout; np.copyto(result, eye) the same_kind assignment.
            var result = new NDArray(a.dtype, Shape.NewLikeLayout(a.Shape, 'K', (long[])a.Shape.dimensions.Clone()), false);
            UnmanagedStorage.Copyto(result, eye, NPY_CASTING.NPY_SAME_KIND_CASTING, null);
            return result;
        }
    }
}
