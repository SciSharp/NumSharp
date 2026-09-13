using System;

namespace NumSharp
{
    public static partial class np
    {
        // =====================================================================
        //  np.apply_over_axes — apply a function repeatedly over multiple axes.
        //
        //  Port of NumPy 2.x numpy.apply_over_axes (numpy/lib/_shape_base_impl.py):
        //
        //      val = asarray(a); N = a.ndim
        //      if scalar(axes): axes = (axes,)
        //      for axis in axes:
        //          if axis < 0: axis = N + axis           # negatives use the ORIGINAL ndim
        //          res = func(val, axis)
        //          if res.ndim == val.ndim: val = res     # func kept the rank (e.g. keepdims=True)
        //          else:
        //              res = expand_dims(res, axis)        # func dropped one dim — re-insert it
        //              if res.ndim == val.ndim: val = res
        //              else: raise ValueError("function is not returning an array of the correct shape")
        //      return val
        //
        //  A pure composition over func + expand_dims: func is arbitrary user code (typically a
        //  reduction such as np.sum), so there is no kernel to emit. This is the reduce-over-a-tuple-
        //  of-axes idiom — equivalent to a reorderable ufunc with keepdims=True over `axes`.
        // =====================================================================

        /// <summary>
        ///     Applies <paramref name="func"/> once per entry of <paramref name="axes"/>, threading
        ///     each result into the next call as <c>func(result, axis)</c>. Intended for reductions:
        ///     <paramref name="func"/> must return an array with either the same rank as its input
        ///     (a keepdims-style reduction) or exactly one fewer dimension — in which case the reduced
        ///     axis is re-inserted (as size 1) before the next call, so the rank is preserved across
        ///     the whole sequence. The result therefore has the SAME number of dimensions as
        ///     <paramref name="a"/>, with the sizes along <paramref name="axes"/> collapsed to 1.
        /// </summary>
        /// <param name="func">
        ///     Called as <c>func(current, axis)</c>. Must return an <see cref="NDArray"/> of the input's
        ///     rank or one less; any other rank raises (see below). A drop of one dimension is the
        ///     common case (a plain reduction), which this method re-expands for you.
        /// </param>
        /// <param name="a">Input array.</param>
        /// <param name="axes">
        ///     The axes to apply <paramref name="func"/> over, in order. Negative axes count from the
        ///     end using <paramref name="a"/>'s ORIGINAL rank (matching NumPy — the normalization does
        ///     not track the running result's rank, which the re-expansion keeps equal anyway).
        /// </param>
        /// <returns>
        ///     The array after applying <paramref name="func"/> over every axis; same rank as
        ///     <paramref name="a"/>, sizes along <paramref name="axes"/> reduced (typically to 1).
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="func"/>, <paramref name="a"/>, or <paramref name="axes"/> is null.</exception>
        /// <exception cref="ValueError">
        ///     <paramref name="func"/> returns an array whose rank is neither equal to nor exactly one
        ///     less than its input's, so the reduced axis cannot be re-inserted —
        ///     <c>"function is not returning an array of the correct shape"</c>, verbatim with NumPy.
        /// </exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html</remarks>
        public static NDArray apply_over_axes(Func<NDArray, int, NDArray> func, NDArray a, int[] axes)
        {
            if (func is null)
                throw new ArgumentNullException(nameof(func));
            if (a is null)
                throw new ArgumentNullException(nameof(a));
            if (axes is null)
                throw new ArgumentNullException(nameof(axes));

            // asarray(a) is a no-op for an ndarray (views preserved), so `val` starts as `a`.
            NDArray val = a;
            int N = a.ndim;   // negatives normalize against the ORIGINAL rank, exactly as NumPy does

            foreach (int rawAxis in axes)
            {
                int axis = rawAxis < 0 ? N + rawAxis : rawAxis;

                NDArray res = func(val, axis);
                if (res is null)
                    throw new ValueError("apply_over_axes: func returned null");

                if (res.ndim == val.ndim)
                {
                    // func kept the rank (e.g. a keepdims=True reduction) — take it as-is.
                    val = res;
                }
                else
                {
                    // func dropped a dimension — re-insert it at `axis` so the running rank is stable.
                    // expand_dims validates `axis` against the output rank and raises AxisError if the
                    // drop was by more than one dimension leaves it out of range; a still-wrong rank is
                    // the "correct shape" ValueError below (both match NumPy).
                    res = np.expand_dims(res, axis);
                    if (res.ndim == val.ndim)
                        val = res;
                    else
                        throw new ValueError("function is not returning an array of the correct shape");
                }
            }

            return val;
        }

        /// <summary>
        ///     Scalar-axis overload — NumPy accepts a single integer for <c>axes</c> (its
        ///     <c>array(axes).ndim == 0</c> branch wraps it into a one-tuple). Forwards to the array
        ///     overload with a single-element axis list.
        /// </summary>
        /// <param name="func">Called as <c>func(current, axis)</c>; see the array overload.</param>
        /// <param name="a">Input array.</param>
        /// <param name="axes">The single axis to apply <paramref name="func"/> over; negative counts from the end.</param>
        /// <returns>See the array overload.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="func"/> or <paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="func"/> returns an array of the wrong rank.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html</remarks>
        public static NDArray apply_over_axes(Func<NDArray, int, NDArray> func, NDArray a, int axes)
            => apply_over_axes(func, a, new[] { axes });
    }
}
