namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Permute the axes (dimensions) of an array.<br></br>
        ///     Array API standard alias of <see cref="transpose(NDArray,int[])"/>.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="axes">
        ///     If specified, it must be a permutation of <c>[0, 1, ..., N-1]</c> where <c>N</c> is the number of
        ///     axes of <paramref name="a"/>. Negative indices can also be used. The i-th axis of the returned array
        ///     will correspond to the axis numbered <c>axes[i]</c> of the input. If not specified, defaults to
        ///     reversing the order of the axes.
        /// </param>
        /// <returns><paramref name="a"/> with its axes permuted. A view is returned whenever possible.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.permute_dims.html</remarks>
        public static NDArray permute_dims(NDArray a, int[] axes = null)
            => a.TensorEngine.Transpose(a, axes);
    }
}
