using System;

namespace NumSharp
{
    public partial class FourierModule
    {
        // Helper routines — ports of numpy/fft/_helper.py. fftshift / ifftshift roll the array object
        // (NumPy's statements over np.roll, kept here: the shift read from x.shape, then ONE roll over the axes);
        // fftfreq / rfftfreq compute values, so they are members of the TensorEngine.Fft sub-engine
        // (Backends/Default/Fourier/DefaultFftEngine.Helper.cs holds the default's statements).

        /// <summary>
        ///     Shift the zero-frequency component to the center of the spectrum: NumPy's <c>fftshift</c>, which rolls each
        ///     listed axis by <c>shape[axis] // 2</c> in ONE <see cref="np.roll(NDArray, NDArray, NDAxis?)"/> call.
        /// </summary>
        /// <param name="x">Input array (a subclass operand gives a plain array: NumPy's <c>asarray</c>).</param>
        /// <param name="axes">NumPy's <c>axes</c>: <c>null</c> (every axis), one axis, or a tuple of axes (repeats roll
        ///     that axis again; <c>()</c> rolls nothing and returns a copy).</param>
        /// <returns>A new array, the input's shape and dtype, NumPy's layout (roll's <c>empty_like</c>).</returns>
        /// <exception cref="IndexError">An axis outside <c>[-ndim, ndim)</c>: NumPy reads <c>x.shape[ax]</c> first, so it is
        ///     Python's <c>tuple index out of range</c>, not an AxisError.</exception>
        /// <exception cref="ValueError">A 0-d <paramref name="x"/> with <c>null</c> or <c>()</c> axes: NumPy's roll over no axis
        ///     of a 0-d array ends in Python's <c>not enough values to unpack (expected 2, got 0)</c>.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html. NumPy's statements:
        ///     <c>axes is None</c> → <c>axes = tuple(range(ndim))</c>, <c>shift = [dim // 2 for dim in x.shape]</c>; an int →
        ///     <c>shift = x.shape[axes] // 2</c>; a tuple → <c>shift = [x.shape[ax] // 2 for ax in axes]</c>; then
        ///     <c>roll(x, shift, axes)</c>. One signature for NumPy's int-or-tuple <c>axes</c> since API-parity batch 6; the
        ///     <c>int</c> and <c>int[]</c> overloads forward here.
        /// </remarks>
        [NDScoped]
        public NDArray fftshift(NDArray x, NDAxis? axes = null)
            => Shift(x, axes, negate: false);

        /// <summary>
        ///     <see cref="fftshift(NDArray, NDAxis?)"/> with one <c>int</c> axis: the overload a C# integer binds, forwarding
        ///     to the NDAxis? form, so every answer and error is that one's.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">The axis (a negative value counts from the end).</param>
        /// <returns>The shifted array.</returns>
        /// <inheritdoc cref="fftshift(NDArray, NDAxis?)"/>
        public NDArray fftshift(NDArray x, int axes)
            => fftshift(x, (NDAxis?)axes);

        /// <summary>
        ///     <see cref="fftshift(NDArray, NDAxis?)"/> with the axes as an <c>int[]</c>: the overload an <c>int[]</c> and a
        ///     bare <c>null</c> bind, forwarding to the NDAxis? form (a null array is None, any other array NumPy's tuple).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">The axes as NumPy's tuple, or <c>null</c> for every axis.</param>
        /// <returns>The shifted array.</returns>
        /// <inheritdoc cref="fftshift(NDArray, NDAxis?)"/>
        public NDArray fftshift(NDArray x, int[] axes)
            => fftshift(x, (NDAxis?)axes);

        /// <summary>
        ///     The inverse of <see cref="fftshift(NDArray, NDAxis?)"/>: NumPy's <c>ifftshift</c>, which rolls each listed
        ///     axis by <c>-(shape[axis] // 2)</c> (the same as fftshift for an even length, one sample apart for an odd one).
        /// </summary>
        /// <param name="x">Input array (a subclass operand gives a plain array: NumPy's <c>asarray</c>).</param>
        /// <param name="axes">NumPy's <c>axes</c>: <c>null</c> (every axis), one axis, or a tuple of axes.</param>
        /// <returns>A new array, the input's shape and dtype.</returns>
        /// <exception cref="IndexError">An axis outside <c>[-ndim, ndim)</c> (<c>tuple index out of range</c>).</exception>
        /// <exception cref="ValueError">A 0-d <paramref name="x"/> with <c>null</c> or <c>()</c> axes (<c>not enough values to
        ///     unpack (expected 2, got 0)</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html: fftshift's statements with
        ///     <c>-(dim // 2)</c>.</remarks>
        [NDScoped]
        public NDArray ifftshift(NDArray x, NDAxis? axes = null)
            => Shift(x, axes, negate: true);

        /// <summary>
        ///     <see cref="ifftshift(NDArray, NDAxis?)"/> with one <c>int</c> axis: the overload a C# integer binds, forwarding
        ///     to the NDAxis? form.
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">The axis (a negative value counts from the end).</param>
        /// <returns>The shifted array.</returns>
        /// <inheritdoc cref="ifftshift(NDArray, NDAxis?)"/>
        public NDArray ifftshift(NDArray x, int axes)
            => ifftshift(x, (NDAxis?)axes);

        /// <summary>
        ///     <see cref="ifftshift(NDArray, NDAxis?)"/> with the axes as an <c>int[]</c>: the overload an <c>int[]</c> and a
        ///     bare <c>null</c> bind, forwarding to the NDAxis? form (a null array is None).
        /// </summary>
        /// <param name="x">Input array.</param>
        /// <param name="axes">The axes as NumPy's tuple, or <c>null</c> for every axis.</param>
        /// <returns>The shifted array.</returns>
        /// <inheritdoc cref="ifftshift(NDArray, NDAxis?)"/>
        public NDArray ifftshift(NDArray x, int[] axes)
            => ifftshift(x, (NDAxis?)axes);

        /// <summary>
        ///     NumPy's fftshift / ifftshift statements: the shift of every listed axis read from <c>x.shape</c> (Python's
        ///     tuple subscript, so an axis past the rank is an <see cref="IndexError"/> before anything rolls), then ONE
        ///     <see cref="np.roll(NDArray, NDArray, NDAxis?)"/> over the axes as given (repeats accumulate in roll).
        /// </summary>
        /// <param name="x">Input array; [AsArray] hands the body its plain view (NumPy's <c>asarray</c>), so a subclass
        ///     operand gives a plain result.</param>
        /// <param name="axes">The axes argument as the caller gave it.</param>
        /// <param name="negate">True for ifftshift (<c>-(dim // 2)</c>).</param>
        /// <returns>roll's result.</returns>
        /// <exception cref="IndexError">An axis outside <c>[-ndim, ndim)</c>.</exception>
        /// <exception cref="ValueError">roll's 0-d refusal (no axes to roll over).</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the shift array is this call's own and released; roll's result is
        ///     yielded.</remarks>
        [AsArray]
        [NDScoped]
        private static NDArray Shift(NDArray x, NDAxis? axes, bool negate)
        {
            int nd = x.ndim;
            long[] dims = x.Shape.dimensions;

            // NumPy's Python subscript x.shape[ax]: a negative ax counts from the end, anything else past the tuple is
            // its IndexError (raised for the first bad axis, before roll normalizes any of them).
            long HalfOf(int ax)
            {
                int a = ax < 0 ? ax + nd : ax;
                if (a < 0 || a >= nd)
                    throw new IndexError("tuple index out of range");
                long half = dims[a] / 2;   // a length is never negative: C#'s division is Python's floor division here
                return negate ? -half : half;
            }

            NDArray shift;
            NDAxis rollAxes;
            if (axes is not { } given)
            {
                // axes = tuple(range(x.ndim)); shift = [dim // 2 for dim in x.shape] - both empty for a 0-d input, whose
                // roll then raises NumPy's unpack ValueError.
                var all = new int[nd];
                var halves = new long[nd];
                for (int i = 0; i < nd; i++)
                {
                    all[i] = i;
                    halves[i] = HalfOf(i);
                }
                shift = np.array(halves);
                rollAxes = new NDAxis(all);
            }
            else if (!given.IsTuple)
            {
                // An int: roll(x, x.shape[axes] // 2, axes) - a scalar shift (a C# long is NumPy's Python int).
                shift = HalfOf(given[0]);
                rollAxes = given;
            }
            else
            {
                // A tuple: [x.shape[ax] // 2 for ax in axes], in the given order (repeats included).
                int[] items = given.ToArray();
                var halves = new long[items.Length];
                for (int i = 0; i < items.Length; i++)
                    halves[i] = HalfOf(items[i]);
                shift = np.array(halves);
                rollAxes = given;
            }

            return np.roll(x, shift, rollAxes);
        }

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies:
        ///     <c>f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)</c> (n even);
        ///     <c>f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)</c> (n odd).
        /// </summary>
        /// <param name="n">Window length: any C# integer converts (NumPy's Python int / <c>np.integer</c>). A floating n does
        ///     not compile, C#'s form of NumPy's <c>ValueError: n should be an integer</c>; a <c>ulong</c> needs a cast.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <param name="device">Array-API device; must be <c>null</c> or <c>"cpu"</c>.</param>
        /// <returns>A float64 array of length <paramref name="n"/>.</returns>
        /// <exception cref="ZeroDivisionError"><c>n * d</c> is zero (<c>float division by zero</c>, checked first).</exception>
        /// <exception cref="ValueError">A device other than <c>"cpu"</c> (<c>Device not understood. ...</c>), then a negative
        ///     <paramref name="n"/> (<c>negative dimensions are not allowed</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
        ///     <para>One integer signature since API-parity batch 6 (the <c>int</c> form and the <c>double</c> form that only
        ///     threw NumPy's ValueError are gone). Computed by <see cref="TensorEngine.Fft"/> of the default engine
        ///     (<see cref="FftEngine.Fftfreq"/>): there is no operand to choose another.</para></remarks>
        public NDArray fftfreq(long n, double d = 1.0, string device = null)
            => TensorEngine.Default.Fft.Fftfreq(n, d, device);

        /// <summary>
        ///     Return the Discrete Fourier Transform sample frequencies for use with <c>rfft</c> / <c>irfft</c>:
        ///     <c>f = [0, 1, ..., n/2] / (d*n)</c> (the Nyquist component is positive).
        /// </summary>
        /// <param name="n">Window length: any C# integer converts; a floating n does not compile (NumPy's
        ///     <c>ValueError: n should be an integer</c>). A negative one gives an empty result, as NumPy's <c>arange</c>
        ///     does.</param>
        /// <param name="d">Sample spacing (inverse of the sampling rate). Default 1.</param>
        /// <param name="device">Array-API device; must be <c>null</c> or <c>"cpu"</c>.</param>
        /// <returns>A float64 array of length <c>n//2 + 1</c> (empty when that is not positive).</returns>
        /// <exception cref="ZeroDivisionError"><c>n * d</c> is zero (<c>float division by zero</c>, checked first).</exception>
        /// <exception cref="ValueError">A device other than <c>"cpu"</c>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html
        ///     <para>One integer signature since API-parity batch 6. Computed by <see cref="TensorEngine.Fft"/> of the default
        ///     engine (<see cref="FftEngine.Rfftfreq"/>).</para></remarks>
        public NDArray rfftfreq(long n, double d = 1.0, string device = null)
            => TensorEngine.Default.Fft.Rfftfreq(n, d, device);
    }
}
