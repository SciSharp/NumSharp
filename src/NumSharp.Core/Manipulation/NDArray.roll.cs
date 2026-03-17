using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Roll array elements along a given axis.
        ///
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// </summary>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <param name="axis">Axis along which elements are shifted.</param>
        /// <returns>Output array, with the same shape as the input.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.roll.html</remarks>
        public NDArray roll(long shift, int axis)
            => np.roll(this, shift, axis);

        /// <inheritdoc cref="roll(long, int)"/>
        public NDArray roll(int shift, int axis)
            => np.roll(this, (long)shift, axis);

        /// <summary>
        /// Roll array elements along a given axis.
        ///
        /// Elements that roll beyond the last position are re-introduced at the first.
        /// The array is flattened before shifting, after which the original shape is restored.
        /// </summary>
        /// <param name="shift">The number of places by which elements are shifted.</param>
        /// <returns>Output array, with the same shape as the input.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.roll.html</remarks>
        public NDArray roll(long shift)
            => np.roll(this, shift);

        /// <inheritdoc cref="roll(long)"/>
        public NDArray roll(int shift)
            => np.roll(this, (long)shift);
    }
}
