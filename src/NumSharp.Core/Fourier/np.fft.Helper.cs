using System;

namespace NumSharp
{
    public partial class FourierModule
    {
        // Helper routines — ports of numpy/fft/_helper.py. These are pure compositions of existing
        // np.* functions (roll / arange / concatenate / arithmetic) and compute FULLY (no engine seam).

        /// <summary>
        ///     Shift the zero-frequency component to the center of the spectrum. Rolls every axis by
        ///     <c>shape[axis] // 2</c>. (All-axes form; <paramref name="axes"/> defaults to all.)
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">Axes over which to shift; <c>null</c> shifts all axes.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html</remarks>
        public NDArray fftshift(NDArray x, int[] axes = null)
            => Shift(x, axes, negate: false);

        /// <summary>
        ///     Shift the zero-frequency component to the center of the spectrum, over a single axis.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">Axis over which to shift (NumPy's <c>axes</c> also accepts a single int).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html</remarks>
        public NDArray fftshift(NDArray x, int axes)
            => Shift(x, new[] { axes }, negate: false);

        /// <summary>
        ///     The inverse of <see cref="fftshift(NDArray, int[])"/>. Rolls every axis by
        ///     <c>-(shape[axis] // 2)</c> (identical to <c>fftshift</c> for even lengths; differs by one
        ///     sample for odd lengths). (All-axes form; <paramref name="axes"/> defaults to all.)
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">Axes over which to shift; <c>null</c> shifts all axes.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html</remarks>
        public NDArray ifftshift(NDArray x, int[] axes = null)
            => Shift(x, axes, negate: true);

        /// <summary>
        ///     The inverse of <see cref="fftshift(NDArray, int)"/>, over a single axis.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">Axis over which to shift (NumPy's <c>axes</c> also accepts a single int).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html</remarks>
        public NDArray ifftshift(NDArray x, int axes)
            => Shift(x, new[] { axes }, negate: true);

        /// <summary>
        ///     Roll <paramref name="x"/> along each requested axis by <c>±(shape[axis] // 2)</c>.
        ///     Composing per-axis rolls reproduces NumPy's single multi-axis <c>roll(x, shift, axes)</c>
        ///     (each axis is an independent cyclic permutation; repeated axes accumulate exactly as
        ///     NumPy's roll accumulates them).
        ///
        ///     <para>NumPy builds the shift list by indexing <c>x.shape[ax]</c> BEFORE it calls
        ///     <c>roll</c>, so an out-of-range axis raises <see cref="IndexError"/>
        ///     ("tuple index out of range") — the Python tuple-subscript error — NOT <c>AxisError</c>
        ///     (which <c>roll</c>'s own axis normalization would raise but never reaches). Every axis is
        ///     validated first, then the rolls run.</para>
        /// </summary>
        private static NDArray Shift(NDArray x, int[] axes, bool negate)
        {
            int nd = x.ndim;
            if (axes == null)
            {
                // NumPy: axes = tuple(range(ndim)). A 0-d input then hits NumPy's internal
                // roll-unpack ValueError ("not enough values to unpack (expected 2, got 0)") — an
                // artifact of rolling a scalar. NumSharp treats fftshift/ifftshift of a 0-d array as a
                // no-op (returns a copy). [Misaligned]
                if (nd == 0)
                    return x.copy();
                axes = new int[nd];
                for (int i = 0; i < nd; i++)
                    axes[i] = i;
            }

            // Validate every axis the way NumPy indexes x.shape[ax] — up front, before any roll.
            foreach (int ax in axes)
            {
                int a = ax < 0 ? ax + nd : ax;
                if (a < 0 || a >= nd)
                    throw new IndexError("tuple index out of range");
            }

            // NumPy's roll ALWAYS returns a fresh array; reproduce that for the no-shift cases
            // (empty axes list, or an all-length-1 array) so the result never aliases the input.
            if (axes.Length == 0)
                return x.copy();

            NDArray result = x;
            foreach (int ax in axes)
            {
                int a = ax < 0 ? ax + nd : ax;
                long shift = x.shape[a] / 2;   // dim // 2 (a >= 0 here, so int-div == floor-div)
                result = np.roll(result, negate ? -shift : shift, a);
            }
            return ReferenceEquals(result, x) ? x.copy() : result;
        }

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies.
        ///     <c>f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)</c> (n even);
        ///     <c>f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)</c> (n odd).
        /// </summary>
        /// <param name="n">Window length.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <param name="device">Array-API device; must be <c>null</c> or <c>"cpu"</c>.</param>
        /// <returns>A float64 array of length <paramref name="n"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html</remarks>
        public NDArray fftfreq(int n, double d = 1.0, string device = null)
            => fftfreq((long)n, d, device);

        /// <summary>
        ///     <see cref="fftfreq(int, double, string)"/> for a 64-bit window length. NumPy accepts ANY
        ///     integer type (Python <c>int</c> / <c>np.integer</c>), so a C# <c>long</c> must compute
        ///     rather than fall to the float-rejecting overload.
        /// </summary>
        public NDArray fftfreq(long n, double d = 1.0, string device = null)
        {
            // NumPy's error ORDER (probed 2.4.2): val = 1.0/(n*d) FIRST — ZeroDivisionError when
            // n*d == 0 (i.e. n == 0 OR d == 0, for ANY sign of n; .NET's float 1/0 is +inf, not a
            // throw, so the divisor is checked explicitly). THEN empty(n, int, device=device), which
            // validates the device before it rejects a negative length.
            double nd = (double)n * d;
            if (nd == 0.0)
                throw new DivideByZeroException("float division by zero");
            ValidateDevice(device);
            if (n < 0)
                throw new ValueError("negative dimensions are not allowed");
            double val = 1.0 / nd;
            long N = (n - 1) / 2 + 1;                     // n>=1 here, so int division == floor division.
            var p1 = np.arange(0L, N);                    // [0 .. N-1]
            var p2 = np.arange(-(n / 2), 0L);             // [-(n/2) .. -1]
            var results = np.concatenate((p1, p2));       // length n
            return results * val;                         // -> float64
        }

        /// <summary>
        ///     Non-integer <c>n</c> is rejected exactly like NumPy (which requires an <c>int</c>-typed
        ///     window length): <c>ValueError("n should be an integer")</c> — and this fires BEFORE the
        ///     device/spacing checks, matching NumPy's <c>isinstance</c> guard at the top.
        /// </summary>
        /// <param name="n">Window length (a floating value — always rejected).</param>
        /// <param name="d">Sample spacing.</param>
        /// <param name="device">Array-API device (unreached — the integer check throws first).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html</remarks>
        public NDArray fftfreq(double n, double d = 1.0, string device = null)
            => throw new ValueError("n should be an integer");

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies (for usage with <c>rfft</c>,
        ///     <c>irfft</c>). <c>f = [0, 1, ..., n/2] / (d*n)</c> — the Nyquist component is positive.
        /// </summary>
        /// <param name="n">Window length.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <param name="device">Array-API device; must be <c>null</c> or <c>"cpu"</c>.</param>
        /// <returns>A float64 array of length <c>n//2 + 1</c>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html</remarks>
        public NDArray rfftfreq(int n, double d = 1.0, string device = null)
            => rfftfreq((long)n, d, device);

        /// <summary>
        ///     <see cref="rfftfreq(int, double, string)"/> for a 64-bit window length (see the
        ///     <c>long</c> rationale on <see cref="fftfreq(long, double, string)"/>).
        /// </summary>
        public NDArray rfftfreq(long n, double d = 1.0, string device = null)
        {
            // NumPy: val = 1.0/(n*d) FIRST (ZeroDivisionError when n*d == 0), THEN arange(0, n//2+1,
            // device=device) (device validated here). There is NO negative-dimension check — a
            // negative n makes n//2+1 <= 0 so arange yields an EMPTY float64 array
            // (rfftfreq(-5) -> [] rather than an error; unlike fftfreq, which calls empty(n)).
            double nd = (double)n * d;
            if (nd == 0.0)
                throw new DivideByZeroException("float division by zero");
            ValidateDevice(device);
            double val = 1.0 / nd;
            // N = n//2 + 1. Arithmetic right-shift is floor-division-by-2 for two's-complement ints
            // (n>>1 == n//2 for BOTH signs), so a negative n gives N<=0 exactly as NumPy's floor does
            // (C#'s '/' truncates toward zero, which would wrongly make rfftfreq(-1) -> [0.0]).
            long N = (n >> 1) + 1;
            var results = np.arange(0L, Math.Max(0L, N)); // arange(0, N); empty float64 when N<=0
            return results * val;                         // -> float64
        }

        /// <summary>
        ///     Non-integer <c>n</c> is rejected exactly like NumPy: <c>ValueError("n should be an integer")</c>
        ///     (fires before the device/spacing checks).
        /// </summary>
        /// <param name="n">Window length (a floating value — always rejected).</param>
        /// <param name="d">Sample spacing.</param>
        /// <param name="device">Array-API device (unreached — the integer check throws first).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html</remarks>
        public NDArray rfftfreq(double n, double d = 1.0, string device = null)
            => throw new ValueError("n should be an integer");

        /// <summary>
        ///     Array-API <c>device</c> validation shared by <c>fftfreq</c>/<c>rfftfreq</c>. NumPy accepts
        ///     only <c>None</c> or <c>"cpu"</c>, else <c>ValueError</c> with the verbatim message.
        /// </summary>
        private static void ValidateDevice(string device)
        {
            if (device != null && device != "cpu")
                throw new ValueError(
                    $"Device not understood. Only \"cpu\" is allowed, but received: {device}");
        }
    }
}
