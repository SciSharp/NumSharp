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
        {
            if (axes == null)
            {
                for (int ax = 0; ax < x.ndim; ax++)
                    x = ShiftRoll(x, ax, negate: false);
                return x;
            }
            foreach (int ax in axes)
                x = ShiftRoll(x, ax, negate: false);
            return x;
        }

        /// <summary>
        ///     Shift the zero-frequency component to the center of the spectrum, over a single axis.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axis">Axis over which to shift.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html</remarks>
        public NDArray fftshift(NDArray x, int axis)
            => ShiftRoll(x, axis, negate: false);

        /// <summary>
        ///     The inverse of <see cref="fftshift(NDArray, int[])"/>. Rolls every axis by
        ///     <c>-(shape[axis] // 2)</c> (identical to <c>fftshift</c> for even lengths; differs by one
        ///     sample for odd lengths). (All-axes form; <paramref name="axes"/> defaults to all.)
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">Axes over which to shift; <c>null</c> shifts all axes.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html</remarks>
        public NDArray ifftshift(NDArray x, int[] axes = null)
        {
            if (axes == null)
            {
                for (int ax = 0; ax < x.ndim; ax++)
                    x = ShiftRoll(x, ax, negate: true);
                return x;
            }
            foreach (int ax in axes)
                x = ShiftRoll(x, ax, negate: true);
            return x;
        }

        /// <summary>
        ///     The inverse of <see cref="fftshift(NDArray, int)"/>, over a single axis.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axis">Axis over which to shift.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html</remarks>
        public NDArray ifftshift(NDArray x, int axis)
            => ShiftRoll(x, axis, negate: true);

        /// <summary>
        ///     Roll <paramref name="x"/> along <paramref name="axis"/> by <c>±(shape[axis] // 2)</c>.
        ///     Composing per-axis rolls reproduces NumPy's single multi-axis <c>roll</c> (each axis is an
        ///     independent cyclic permutation). Reproduces <c>np.roll</c>'s axis error message on an
        ///     out-of-range axis.
        /// </summary>
        private static NDArray ShiftRoll(NDArray x, int axis, bool negate)
        {
            int nd = x.ndim;
            int a = axis < 0 ? axis + nd : axis;
            if (a < 0 || a >= nd)
                throw new ArgumentException($"axis {axis} is out of bounds for array of dimension {nd}");
            long shift = x.shape[a] / 2;
            if (negate) shift = -shift;
            return np.roll(x, shift, a);
        }

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies.
        ///     <c>f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)</c> (n even);
        ///     <c>f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)</c> (n odd).
        /// </summary>
        /// <param name="n">Window length.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <returns>A float64 array of length <paramref name="n"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html</remarks>
        public NDArray fftfreq(int n, double d = 1.0)
        {
            // NumPy's order: val = 1.0/(n*d) (ZeroDivisionError at n==0) then empty(n) (negative dims).
            if (n < 0)
                throw new ValueError("negative dimensions are not allowed");
            if (n == 0)
                throw new DivideByZeroException("float division by zero");
            double val = 1.0 / (n * d);
            int N = (n - 1) / 2 + 1;                      // n>=1 here, so int division == floor division.
            var p1 = np.arange(0, N);                     // int64 [0 .. N-1]
            var p2 = np.arange(-(n / 2), 0);              // int64 [-(n/2) .. -1]
            var results = np.concatenate((p1, p2));       // int64, length n
            return results * val;                         // -> float64
        }

        /// <summary>
        ///     Non-integer <c>n</c> is rejected exactly like NumPy (which requires an <c>int</c>-typed
        ///     window length): <c>ValueError("n should be an integer")</c>.
        /// </summary>
        /// <param name="n">Window length (a floating value — always rejected).</param>
        /// <param name="d">Sample spacing.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html</remarks>
        public NDArray fftfreq(double n, double d = 1.0)
            => throw new ValueError("n should be an integer");

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies (for usage with <c>rfft</c>,
        ///     <c>irfft</c>). <c>f = [0, 1, ..., n/2] / (d*n)</c> — the Nyquist component is positive.
        /// </summary>
        /// <param name="n">Window length.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <returns>A float64 array of length <c>n//2 + 1</c>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html</remarks>
        public NDArray rfftfreq(int n, double d = 1.0)
        {
            if (n < 0)
                throw new ValueError("negative dimensions are not allowed");
            if (n == 0)
                throw new DivideByZeroException("float division by zero");
            double val = 1.0 / (n * d);
            int N = n / 2 + 1;
            var results = np.arange(0, N);                // int64 [0 .. N-1]
            return results * val;                         // -> float64
        }

        /// <summary>
        ///     Non-integer <c>n</c> is rejected exactly like NumPy: <c>ValueError("n should be an integer")</c>.
        /// </summary>
        /// <param name="n">Window length (a floating value — always rejected).</param>
        /// <param name="d">Sample spacing.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html</remarks>
        public NDArray rfftfreq(double n, double d = 1.0)
            => throw new ValueError("n should be an integer");
    }
}
