using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array with the specified shape.
        ///     <para>
        ///     If the new array is larger than the original array, then the new array is filled
        ///     with repeated copies of <paramref name="a"/> (iterating over <paramref name="a"/>
        ///     in C-order, cycling back from the start). Note that this behavior is different from
        ///     <see cref="NDArray.resize(long[])"/> which fills with zeros instead.
        ///     </para>
        ///     <para>
        ///     NumPy's <c>np.resize</c> takes <c>new_shape</c> as a <b>single</b> argument (an int or a
        ///     sequence) — the multi-argument form <c>np.resize(a, 2, 3)</c> is a NumPy <c>TypeError</c>,
        ///     so it is not offered here. Pass the shape as an int, a value tuple, an array, or a
        ///     <see cref="Shape"/>; all resolve through <see cref="Shape"/>'s implicit conversions:
        ///     <c>np.resize(a, 6)</c>, <c>np.resize(a, (2, 3))</c>, <c>np.resize(a, new[]{2, 3})</c>,
        ///     <c>np.resize(a, new Shape(2, 3))</c>.
        ///     </para>
        /// </summary>
        /// <param name="a">Array to be resized.</param>
        /// <param name="new_shape">Shape of resized array (int / tuple / array / <see cref="Shape"/>).</param>
        /// <returns>
        ///     The new array is formed from the data in the old array, repeated if necessary to
        ///     fill out the required number of elements. The data are repeated iterating over the
        ///     array in C-order. Result is C-contiguous; dtype matches <paramref name="a"/>. Like NumPy's (a reshape of a
        ///     slice of the concatenated copies) it is a VIEW of a fresh buffer — <c>flags.owndata</c> is false, so an
        ///     in-place <c>ndarray.resize</c> of it is refused — except for an empty source or result, which is NumPy's
        ///     owning <c>np.zeros_like(a, shape=new_shape)</c>. For a <c>numpy.char.chararray</c> operand that zero-filled
        ///     result is a chararray (<c>zeros_like</c> keeps the subclass); every other result is a plain array (NumPy's
        ///     <c>np.concatenate</c> drops it).
        /// </returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.resize.html</remarks>
        /// <exception cref="ArgumentNullException">If <paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentException">If any element of <paramref name="new_shape"/> is negative.</exception>
        public static NDArray resize(NDArray a, Shape new_shape)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));

            return UnmanagedStorage.ResizeRepeating(a, new_shape);
        }
    }
}
