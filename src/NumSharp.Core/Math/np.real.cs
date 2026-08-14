using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the real part of the complex argument, element-wise.
        ///     One of the four basic complex-number accessors (with <see cref="imag"/>, <see cref="angle"/>
        ///     and <see cref="conjugate"/>) — the standard post-FFT spectrum component extractors: for
        ///     <c>A = np.fft.fft(a)</c>, <c>A.real</c> / <c>np.real(A)</c> is the real component.
        /// </summary>
        /// <param name="val">Input array.</param>
        /// <returns>
        ///     For a COMPLEX input: a float64 VIEW onto the real lane — it SHARES memory with
        ///     <paramref name="val"/> and is writeable, so <c>np.real(z)[i] = x</c> writes through to
        ///     <c>z[i]</c>'s real part (reproducing NumPy's <c>z.real</c>; complex128 -&gt; float64).
        ///     For a REAL / integer / boolean input: NumPy returns the array itself (the real part of a
        ///     real number is the number), so the input is returned unchanged with its dtype preserved.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.real.html</remarks>
        public static NDArray real(NDArray val)
        {
            // Complex: a strided float64 view onto the real lane (writes through, shares memory) — the
            // same strided complex->real lane read the FFT driver relies on, exposed as a view.
            if (val.typecode == NPTypeCode.Complex)
                return new NDArray(val.Storage.AliasComplexLane(imaginary: false));

            // Real / int / bool: NumPy's `val.real` IS `val` for a real array (same object, shared memory,
            // dtype preserved). Return it unchanged.
            return val;
        }
    }
}
