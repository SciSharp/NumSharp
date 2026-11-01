using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the roots of a polynomial with coefficients given in <paramref name="p"/>.
        ///     If <c>p</c> has length <c>n+1</c> the polynomial is
        ///     <c>p[0]*x**n + p[1]*x**(n-1) + ... + p[n]</c>.
        /// </summary>
        /// <param name="p">Rank-1 array of polynomial coefficients.</param>
        /// <returns>The roots — the eigenvalues of the companion matrix.</returns>
        /// <exception cref="ValueError"><paramref name="p"/> is not rank 1 after <c>atleast_1d</c> (<c>Input must be a rank-1
        ///     array.</c>).</exception>
        /// <exception cref="ValueError">A text coefficient between the first and last non-empty one that does not parse as a
        ///     float (NumPy's <c>p.astype(float)</c>: <c>could not convert string to float: np.str_('x')</c>).</exception>
        /// <exception cref="NotSupportedException">Two or more non-zero coefficients and no LAPACK backend installed.</exception>
        /// <exception cref="TypeError">A 0-d <paramref name="p"/>, or null (NumPy's None): NumPy's
        ///     <c>array_function_dispatch</c> iterates what roots' dispatcher returns, the argument itself, and neither is
        ///     iterable (<c>dispatcher for __array_function__ did not return an iterable</c>), before roots' body
        ///     runs.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.roots.html</remarks>
        [NDScoped] // reclaims the ravel/nonzero/strip/astype chain and the companion-matrix + eigvals intermediates
        public static NDArray roots(NDArray p)
        {
            // NumPy's dispatch comes first: roots' dispatcher returns the argument, which array_function_dispatch iterates
            // to find overriding classes; a 0-d array (or None) is not iterable.
            if (p is null || p.ndim == 0)
                throw new TypeError("dispatcher for __array_function__ did not return an iterable");
            using var plain = NDMatrixBoundary.PlainFor(p);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return p.TensorEngine.Roots(p);
        }
    }
}
