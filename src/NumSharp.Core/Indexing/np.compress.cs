using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return selected slices of <paramref name="a"/> along the given
        ///     <paramref name="axis"/> at positions where the 1-D
        ///     <paramref name="condition"/> is truthy.
        /// </summary>
        /// <param name="condition">
        ///     <strong>1-D</strong> array of booleans (or any dtype interpreted as
        ///     truthy). Must be 1-D — a 2-D or 0-D condition raises
        ///     <see cref="ArgumentException"/>, mirroring NumPy's
        ///     <c>ValueError("condition must be a 1-d array")</c>. If
        ///     <c>len(condition) &lt; a.shape[axis]</c>, only the first
        ///     <c>len(condition)</c> positions along <paramref name="axis"/> are
        ///     considered; if longer, any True beyond <c>a.shape[axis]</c> raises
        ///     <see cref="IndexOutOfRangeException"/>.
        /// </param>
        /// <param name="a">Source array.</param>
        /// <param name="axis">
        ///     Axis along which to slice. <c>null</c> (default) flattens
        ///     <paramref name="a"/> first.
        /// </param>
        /// <param name="out">
        ///     Optional destination. When supplied, shape must match the natural
        ///     output; values are cast to <paramref name="out"/>'s dtype via
        ///     <see cref="np.copyto"/> with unsafe casting and the method returns
        ///     <paramref name="out"/> itself (matches NumPy's out= dispatch).
        /// </param>
        /// <returns>
        ///     A copy of <paramref name="a"/> without the slices along
        ///     <paramref name="axis"/> for which <paramref name="condition"/> is
        ///     false. Dtype matches <paramref name="a"/> (or
        ///     <paramref name="out"/>'s dtype when supplied).
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.compress.html
        ///     <para>
        ///     Mirrors NumPy's C-side <c>PyArray_Compress</c>: validate
        ///     <paramref name="condition"/> is 1-D, materialise nonzero indices,
        ///     then dispatch to <see cref="take(NDArray, NDArray, int?, NDArray, string)"/>
        ///     with <c>mode="raise"</c>.
        ///     </para>
        ///     <para>
        ///     When <paramref name="condition"/> is 1-D <see cref="np.extract"/> with
        ///     <paramref name="axis"/> = <c>null</c> is equivalent.
        ///     </para>
        /// </remarks>
        public static NDArray compress(NDArray condition, NDArray a, int? axis = null, NDArray @out = null)
        {
            if (condition is null) throw new ArgumentNullException(nameof(condition));
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy hard-requires 1-D condition; 0-D and 2-D+ both fail here.
            if (condition.ndim != 1)
                throw new ArgumentException(
                    "condition must be a 1-d array",
                    nameof(condition));

            var indices = np.flatnonzero(condition);
            try
            {
                // axis=None flattens internally inside take; otherwise gather along axis.
                // mode defaults to "raise" — matches NumPy's NPY_RAISE pass in
                // PyArray_Compress (item_selection.c).
                return np.take(a, indices, axis, @out);
            }
            finally
            {
                indices.Dispose();
            }
        }
    }
}
