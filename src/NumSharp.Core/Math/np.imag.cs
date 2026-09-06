using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the imaginary part of the complex argument, element-wise.
        ///     One of the four basic complex-number accessors (with <see cref="real"/>, <see cref="angle"/>
        ///     and <see cref="conjugate"/>) — the standard post-FFT spectrum component extractors: for
        ///     <c>A = np.fft.fft(a)</c>, <c>A.imag</c> / <c>np.imag(A)</c> is the imaginary component.
        /// </summary>
        /// <param name="val">Input array.</param>
        /// <returns>
        ///     For a COMPLEX input: a float64 VIEW onto the imaginary lane — it SHARES memory with
        ///     <paramref name="val"/> and is writeable, so <c>np.imag(z)[i] = x</c> writes through to
        ///     <c>z[i]</c>'s imaginary part (reproducing NumPy's <c>z.imag</c>; complex128 -&gt; float64).
        ///     For a REAL / integer / boolean input: a fresh, READ-ONLY all-zeros array of the same shape
        ///     and dtype (the imaginary part of a real number is zero), matching NumPy exactly.
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.imag.html</remarks>
        public static NDArray imag(NDArray val)
        {
            // Complex: a strided float64 view onto the imaginary lane (writes through, shares memory).
            if (val.typecode == NPTypeCode.Complex)
                return new NDArray(val.Storage.AliasComplexLane(imaginary: true));

            // Real / int / bool: NumPy returns a READ-ONLY all-zeros array of the SAME shape and dtype
            // (probed: `np.imag(real).flags.writeable` is False, dtype preserved — and it OWNS its zeros:
            // owndata=True, base=None, so clear WRITEABLE on the owned array itself rather than aliasing
            // a view, which would misreport owndata=False. setflags(write=True) on it then succeeds,
            // exactly as NumPy's _IsWriteable allows for an owner).
            var zeros = np.zeros(new Shape(val.shape), val.typecode);
            zeros.Storage.SetShapeUnsafe(zeros.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            return zeros;
        }
    }
}
