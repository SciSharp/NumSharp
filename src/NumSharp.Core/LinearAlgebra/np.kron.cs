using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Kronecker product of two arrays.
        ///     <para>
        ///     Computes the Kronecker product, a composite array made of blocks of the second array
        ///     scaled by the first. If <c>a.shape = (r0,r1,...,rN)</c> and <c>b.shape = (s0,s1,...,sN)</c>
        ///     the result has shape <c>(r0*s0, r1*s1, ..., rN*sN)</c> with
        ///     <c>kron(a,b)[k0,...,kN] = a[i0,...,iN] * b[j0,...,jN]</c> where <c>kt = it*st + jt</c>.
        ///     </para>
        ///     The number of dimensions of <paramref name="a"/> and <paramref name="b"/> need not match —
        ///     the smaller is treated as if prepended with size-1 axes (NumPy's <c>ndmin</c> behaviour).
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <returns>
        ///     A fresh, writeable, C-contiguous array. The result dtype follows NumPy's <c>multiply</c>
        ///     promotion (NEP50). If <paramref name="b"/> is 0-d the result is the element-wise
        ///     <c>a * b</c> (which is itself the degenerate Kronecker product).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.kron.html
        ///     <para>The body is <see cref="TensorEngine.Kron"/> (the np-layer dtype plan); this facade keeps the
        ///     <c>numpy.matrix</c> and <c>numpy.char.chararray</c> rules.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        [NDScoped]
        public static NDArray kron(NDArray a, NDArray b)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a, b))
                return NDMatrixBoundary.Wrapped((a, b), static s => kron(s.a, s.b));
            // a numpy.char.chararray operand: NumPy keeps it (asanyarray, array(subok=True)) and multiplies with
            // subok=True, so chararray.__array_wrap__ decides the class of the product — a text result would be a
            // chararray view (text has no multiply loop: the ufunc's TypeError), an int8 one too (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a, b))
                return NDCharArrayBoundary.Wrapped((a, b), static s => kron(s.a, s.b));
            return a.TensorEngine.Kron(a, b);
        }
    }
}
