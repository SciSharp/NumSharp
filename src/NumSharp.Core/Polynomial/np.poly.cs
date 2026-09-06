using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the coefficients of a polynomial with the given sequence of roots (leading coefficient 1).
        ///     A square 2-D array is treated as a matrix, returning the coefficients of its characteristic
        ///     polynomial (via <see cref="linalg.eigvals"/>).
        /// </summary>
        /// <param name="seq_of_zeros">A 1-D sequence of roots, or a square 2-D array.</param>
        /// <returns>1-D coefficients highest degree first, <c>c[0] == 1</c>.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.poly.html
        ///     <para>An empty input returns the 0-d scalar <c>1.0</c> (NumPy's Python float <c>1.0</c>).</para>
        /// </remarks>
        [NDScoped]
        public static NDArray poly(NDArray seq_of_zeros)
        {
            seq_of_zeros = np.atleast_1d(seq_of_zeros);
            int nd = seq_of_zeros.ndim;

            NDArray seq;
            if (nd == 2 && seq_of_zeros.shape[0] == seq_of_zeros.shape[1] && seq_of_zeros.shape[0] != 0)
            {
                // Characteristic polynomial: build from the matrix's eigenvalues.
                seq = np.linalg.eigvals(seq_of_zeros);
            }
            else if (nd == 1)
            {
                // A real (non-object) sequence is cast to its mintypecode float — float32 stays float32,
                // everything else (int/bool/float16/float64) becomes float64; complex stays complex.
                seq = seq_of_zeros.astype(PolyMinType(seq_of_zeros.typecode));
            }
            else
            {
                throw new ValueError("input must be 1d or non-empty square 2d array.");
            }

            if (seq.size == 0)
                return NDArray.Scalar(1.0);

            NPTypeCode dt = seq.typecode;
            NDArray onesScalar = np.ones(new Shape(1), dt);
            NDArray negSeq = np.negative(seq);

            // a = ones(1); for each root: a = convolve(a, [1, -root])  (accumulates (x - root) factors).
            NDArray a = onesScalar;
            for (long i = 0; i < seq.size; i++)
            {
                NDArray factor = np.concatenate(new[] { onesScalar, negSeq[$"{i}:{i + 1}"] }, 0);
                a = np.convolve(a, factor, "full");
            }

            // If the roots come as complex conjugate pairs, the coefficients are real — take the real part.
            if (dt == NPTypeCode.Complex)
            {
                NDArray rootsC = seq.astype(NPTypeCode.Complex);
                if (np.all(np.equal(np.sort(rootsC), np.sort(np.conjugate(rootsC)))))
                    a = np.real(a).copy();
            }

            return a;
        }

        /// <summary>
        ///     NumPy's <c>mintypecode(dtype.char)</c> as applied by <c>poly</c>: the smallest floating type
        ///     that holds the roots. float32 preserves; complex preserves; every other type widens to float64.
        /// </summary>
        internal static NPTypeCode PolyMinType(NPTypeCode t)
        {
            switch (t)
            {
                case NPTypeCode.Single: return NPTypeCode.Single;
                case NPTypeCode.Complex: return NPTypeCode.Complex;
                default: return NPTypeCode.Double;
            }
        }
    }
}
