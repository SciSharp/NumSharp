using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array.
        /// </summary>
        /// <param name="order">
        ///     Controls the memory layout of the copy.
        ///     'C' - row-major (C-style), 'F' - column-major (Fortran-style),
        ///     'A' - 'F' if this is F-contiguous (and not C-contiguous), else 'C',
        ///     'K' - match the layout of this array as closely as possible.
        /// </param>
        /// <returns>A copy of the array with the requested memory layout.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.copy.html</remarks>
        public NDArray copy(char order = 'C')
        {
            char physical = OrderResolver.Resolve(order, this.Shape);

            // Preserve current behavior for scalars / empty arrays — Clone() handles them.
            if (this.Shape.IsEmpty || this.Shape.IsScalar || this.Shape.size <= 1)
                return Clone();

            if (physical == 'C')
                return Clone();

            // Allocate destination with F-contiguous strides and copy values logically.
            var destShape = new Shape(this.Shape.dimensions, 'F');
            var dest = new NDArray(this.typecode, destShape, false);
            if (!NpyIter.TryCopySameType(dest.Storage, this.Storage))
                MultiIterator.Assign(dest.Storage, this.Storage);
            return dest;
        }
    }
}
