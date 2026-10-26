using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.matrix_power</c>: <c>_assert_stacked_square</c>, the inverse for a negative
        ///     power (through the public <c>np.linalg.inv</c>), the identity for 0, the operand itself for 1, and binary
        ///     exponentiation with <c>np.matmul</c> above.
        /// </summary>
        /// <param name="a">The square matrix, or a stack of them: plain for a numpy.matrix operand (the facade applied its
        ///     rule), a numpy.char.chararray one as it is (the facade has no chararray rule).</param>
        /// <param name="n">The power: any integer.</param>
        /// <returns>The power. <b>Footgun (NumPy's own):</b> <c>n == 1</c> returns <paramref name="a"/> ITSELF.</returns>
        /// <exception cref="LinAlgError">Fewer than two axes, a non-square operand, or a negative power of a singular
        ///     matrix.</exception>
        /// <exception cref="TypeError">A text matrix with <c>n</c> of 2 or more (matmul has no string loop) or a negative
        ///     <c>n</c> (inv's input cast).</exception>
        /// <exception cref="OpenBlasMissingBackendException">A negative power of a matrix the managed LU cannot take
        ///     with no backend installed.</exception>
        [NDScoped]
        internal static NDArray MatrixPower(NDArray a, int n)
        {
            AssertStackedSquare(a);

            var work = a;
            if (n < 0)
            {
                // Raises without a backend — deliberately BEFORE the identity/short-power cases,
                // exactly as NumPy inverts first and then exponentiates the inverse.
                work = np.linalg.inv(a);
                n = -n;
            }

            if (n == 0)
                return Identity(work);
            // NumPy's short-cut `if n == 1: return a` hands back the operand itself (or, for n == -1, the fresh
            // inverse) — the same object, so `matrix_power(a, 1) is a` and the result shares a's memory and flags.
            if (n == 1)
                return work;
            if (n == 2)
                return np.matmul(work, work);
            if (n == 3)
                return np.matmul(np.matmul(work, work), work);

            // Binary exponentiation. `result` stays null until the first set bit so the identity
            // multiply is skipped, which is what NumPy's `if result is None` does.
            NDArray result = null;
            var z = work;
            while (n > 0)
            {
                if ((n & 1) == 1)
                    result = result is null ? z : np.matmul(result, z);

                n >>= 1;
                if (n > 0)
                    z = np.matmul(z, z);
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
        /// <remarks>The identity is built IN the operand's dtype, as NumPy's <c>eye(..., dtype=a.dtype)</c> is: for a text
        ///     dtype the diagonal stores the Python int 1 as its <c>str()</c> (<c>'1'</c>) and the rest stays empty, where
        ///     a float64 identity cast to text would read <c>'1.0'</c> / <c>'0.0'</c>.</remarks>
        [NDScopedCovered] // only caller: [NDScoped] MatrixPower; the eye temp rides its scope
        private static NDArray Identity(NDArray a)
        {
            long side = a.Shape.dimensions[a.ndim - 1];
            if (side > int.MaxValue)
                throw new OverflowException($"Matrix dimension {side} exceeds int.MaxValue for np.eye");

            var eye = np.eye((int)side, dtype: a.dtype);
            // A plain region: a numpy.char.chararray operand arrives as it is (the facade has no chararray rule), and
            // np.empty_like must not apply its chararray rule from inside the library. The identity stays the plain array
            // this function always returned (NumPy's empty_like keeps the chararray class: the class boundary's work,
            // not this function's). The region does nothing for a plain operand.
            using var plain = NDMatrixBoundary.PlainFor(a);
            // empty_like keeps the operand's order (a C-ordered copy of eye did not), and the assignment broadcasts the one
            // identity over the stack's leading axes.
            var result = np.empty_like(a);
            np.copyto(result, eye);
            return result;
        }
    }
}
