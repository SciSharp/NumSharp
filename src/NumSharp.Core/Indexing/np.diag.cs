using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Extract a diagonal or construct a diagonal array.
        /// </summary>
        /// <param name="v">
        ///     If <paramref name="v"/> is 2-D, return a copy of its <paramref name="k"/>-th
        ///     diagonal. If <paramref name="v"/> is 1-D, return a 2-D array with
        ///     <paramref name="v"/> on the <paramref name="k"/>-th diagonal.
        /// </param>
        /// <param name="k">
        ///     Diagonal in question. Use <c>k &gt; 0</c> for diagonals above the main diagonal,
        ///     and <c>k &lt; 0</c> for diagonals below the main diagonal.
        /// </param>
        /// <returns>The extracted diagonal or constructed diagonal array.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentException">
        ///     <paramref name="v"/> is not 1- or 2-dimensional —
        ///     <c>Input must be 1- or 2-d.</c> (NumPy's <c>ValueError</c>).
        /// </exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diag.html
        ///     <para>
        ///     <b>The two branches differ in view-ness</b>, exactly as in NumPy:
        ///     the 1-D branch <em>constructs</em> a fresh, writeable, C-contiguous
        ///     <c>(n, n)</c> array (<c>n = v.size + |k|</c>); the 2-D branch delegates to
        ///     <see cref="diagonal"/> and therefore returns a <b>read-only view</b> that shares
        ///     storage with <paramref name="v"/> — despite the NumPy docstring's talk of "a
        ///     copy" (probed against 2.4.2: <c>np.shares_memory</c> is True and
        ///     <c>flags.writeable</c> is False).
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray diag(NDArray v, int k = 0)
        {
            // a numpy.char.chararray operand: a 2-d one is np.diagonal's chararray view; a 1-d one builds np.zeros (plain)
            if (v is not null && v.ndim == 2 && NDCharArrayBoundary.Sees(v))
                return NDCharArrayBoundary.Kept((v, k), static s => diag(s.v, s.k));
            using var plain = NDMatrixBoundary.PlainFor(v);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (v is null) throw new ArgumentNullException(nameof(v));
            return NDArray.diag(v, k);
        }

        /// <summary>
        ///     Create a two-dimensional array with the flattened input as a diagonal.
        /// </summary>
        /// <param name="v">Input data, which is flattened (in C order) and set as the <paramref name="k"/>-th diagonal of the output.</param>
        /// <param name="k">
        ///     Diagonal to set; 0, the default, corresponds to the "main" diagonal, a positive
        ///     (negative) <paramref name="k"/> giving the number of the diagonal above (below)
        ///     the main.
        /// </param>
        /// <returns>
        ///     The 2-D output array of shape <c>(n, n)</c> where <c>n = v.size + |k|</c>.
        ///     Always a freshly allocated, C-contiguous, writeable array.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.diagflat.html
        ///     <para>
        ///     Unlike <see cref="diag"/>, <c>diagflat</c> accepts <b>any</b> dimensionality —
        ///     the input is raveled first, so a 0-d scalar produces a <c>(1, 1)</c> array and a
        ///     3-D input is flattened in C order. Ravel order is <b>logical</b> C order, so an
        ///     F-contiguous or strided input is read in row-major index order, not memory order
        ///     (probed against NumPy 2.4.2).
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray diagflat(NDArray v, int k = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(v))
                return NDMatrixBoundary.Wrapped((v, k), static s => diagflat(s.v, s.k));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(v))
                return NDCharArrayBoundary.Wrapped((v, k), static s => diagflat(s.v, s.k));
            if (v is null) throw new ArgumentNullException(nameof(v));
            return NDArray.diagflat(v, k);
        }
    }
}
