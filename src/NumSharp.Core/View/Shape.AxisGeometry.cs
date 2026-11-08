namespace NumSharp
{
    public readonly partial struct Shape
    {
        /// <summary>
        ///     The all-but-axis collapse of this shape's dimensions around <paramref name="axis"/>: the product of the
        ///     dimensions before it and the product of those after it. A C-contiguous array of this shape is then
        ///     <paramref name="outer"/> blocks of <c>dimensions[axis]</c> rows of <paramref name="inner"/> contiguous
        ///     elements: the block walk np.packbits / np.unpackbits run over (np.repeat has the same collapse as a private
        ///     helper of its own, <c>ComputeAxisGeometry</c>).
        /// </summary>
        /// <param name="axis">A non-negative axis of this shape; the caller has checked it (it is not checked here).</param>
        /// <param name="outer">The product of the dimensions before <paramref name="axis"/>: 1 for axis 0.</param>
        /// <param name="inner">The product of the dimensions after <paramref name="axis"/>: 1 for the last axis, which
        ///     is then innermost and contiguous.</param>
        /// <remarks>The products are unchecked, as the element count is: a shape's dimensions were checked when it was
        ///     allocated.</remarks>
        internal void SplitAtAxis(int axis, out long outer, out long inner)
        {
            outer = 1;
            for (int i = 0; i < axis; i++) outer *= dimensions[i];
            inner = 1;
            for (int i = axis + 1; i < dimensions.Length; i++) inner *= dimensions[i];
        }
    }
}
