namespace NumSharp
{
    public partial class FourierModule
    {
        // Real transforms (rfft: real -> complex half-spectrum; irfft: complex half -> real).
        // Ports of numpy/fft/_pocketfft.py. rfft/irfft resolve fully then reach the 1-D seam;
        // rfftn/irfftn (and the 2-D wrappers) are pure compositions.

        /// <summary>
        ///     Compute the one-dimensional discrete Fourier Transform for real input. The output length
        ///     along the axis is <c>n//2 + 1</c> (the non-negative-frequency half).
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="n">Number of points along the transform axis to use (default <c>a.shape[axis]</c>).</param>
        /// <param name="axis">Axis over which to compute the FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output of shape <c>(..., n//2+1, ...)</c>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html</remarks>
        [NDScoped]
        public NDArray rfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? ShapeAt(a, axis);
            return RawFft(a, nn, axis, isReal: true, isForward: true, norm, @out, "rfft", IsSinglePrecision(a));
        }

        /// <summary>
        ///     Compute the inverse of <see cref="rfft"/> — a real-valued output. For <c>n</c> output
        ///     points, <c>n//2+1</c> input points are used; when <c>n</c> is omitted it defaults to
        ///     <c>2*(m-1)</c> where <c>m</c> is the input length along the axis.
        /// </summary>
        /// <param name="a">Input array (the non-negative-frequency half-spectrum).</param>
        /// <param name="n">Length of the transformed (real) axis of the output. Default <c>2*(m-1)</c>.</param>
        /// <param name="axis">Axis over which to compute the inverse FFT (default the last axis).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated real (float64) output of length <c>n</c> on the axis.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html</remarks>
        [NDScoped]
        public NDArray irfft(NDArray a, int? n = null, int axis = -1, string norm = null, NDArray @out = null)
        {
            int nn = n ?? (ShapeAt(a, axis) - 1) * 2;
            return RawFft(a, nn, axis, isReal: true, isForward: false, norm, @out, "irfft", IsSinglePrecision(a));
        }

        /// <summary>
        ///     Compute the 2-dimensional FFT of a real array. <c>rfft2</c> is <c>rfftn</c> with the
        ///     default <paramref name="axes"/> of <c>(-2, -1)</c>.
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html</remarks>
        [NDScoped]
        public NDArray rfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => rfftn(a, s, axes ?? new[] { -2, -1 }, norm, @out);

        /// <summary>
        ///     Compute the N-dimensional discrete Fourier Transform for real input: <see cref="rfft"/>
        ///     over the LAST transformed axis, then <see cref="fft"/> over the remaining axes.
        /// </summary>
        /// <param name="a">Input array (taken to be real).</param>
        /// <param name="s">Per-axis output lengths; the final element is <c>n</c> for <c>rfft</c>, the
        ///     rest are <c>n</c> for <c>fft</c>. <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated complex output.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html</remarks>
        [NDScoped]
        public NDArray rfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: thread it so every leaf reproduces its own numpy loop
            var (ss, aa) = CookNdArgs(a, s, axes, invreal: false);
            // NumPy indexes s[-1]/axes[-1] with no axes to transform (0-d input, or an explicit
            // empty `axes`), leaking a bare IndexError("list index out of range"). Reproduce it here
            // rather than let C# surface an IndexOutOfRangeException.
            if (aa.Length == 0)
                throw new IndexError("list index out of range");
            a = RawFft(a, ss[aa.Length - 1], aa[aa.Length - 1], isReal: true, isForward: true, norm, @out, "rfft", floatPrec);
            for (int ii = aa.Length - 2; ii >= 0; ii--)
                a = RawFft(a, ss[ii], aa[ii], isReal: false, isForward: true, norm, @out, "fft", floatPrec);
            return a;
        }

        /// <summary>
        ///     Compute the inverse of <see cref="rfft2"/>. <c>irfft2</c> is <c>irfftn</c> with the
        ///     default <paramref name="axes"/> of <c>(-2, -1)</c>. Like NumPy, the provided
        ///     <paramref name="out"/> is not threaded through the composition.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT. Default <c>(-2, -1)</c>.</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Accepted for signature parity; NumPy passes <c>out=None</c> internally.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html</remarks>
        [NDScoped]
        public NDArray irfft2(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
            => irfftn(a, s, axes ?? new[] { -2, -1 }, norm, @out: null);

        /// <summary>
        ///     Compute the inverse of <see cref="rfftn"/>: <see cref="ifft"/> over all axes but the last,
        ///     then <see cref="irfft"/> over the last axis (real output). The last-axis default length is
        ///     <c>2*(m-1)</c> (see <see cref="CookNdArgs"/> with <c>invreal</c>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="s">Per-axis output lengths; <c>-1</c> uses the full input length.</param>
        /// <param name="axes">Axes over which to compute the inverse FFT (default: all axes).</param>
        /// <param name="norm">Normalization mode: <c>null</c>/"backward"/"ortho"/"forward".</param>
        /// <param name="out">Optional pre-allocated real (float64) output for the final transform.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html</remarks>
        [NDScoped]
        public NDArray irfftn(NDArray a, int[] s = null, int[] axes = null, string norm = null, NDArray @out = null)
        {
            bool floatPrec = IsSinglePrecision(a); // float-precision source: thread it so every leaf reproduces its own numpy loop
            var (ss, aa) = CookNdArgs(a, s, axes, invreal: true);
            // As in rfftn: no axes to transform leaks NumPy's IndexError("list index out of range")
            // at the s[-1]/axes[-1] access. (For a shapeless `s`, the invreal defaulting inside
            // CookNdArgs reaches that access first — see the report's shared-core escalation.)
            if (aa.Length == 0)
                throw new IndexError("list index out of range");
            for (int ii = 0; ii < aa.Length - 1; ii++)
                a = RawFft(a, ss[ii], aa[ii], isReal: false, isForward: false, norm, @out: null, "ifft", floatPrec);
            a = RawFft(a, ss[aa.Length - 1], aa[aa.Length - 1], isReal: true, isForward: false, norm, @out, "irfft", floatPrec);
            return a;
        }
    }
}
