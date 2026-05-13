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

            if (physical == 'C' && this.Shape.IsContiguous)
                return Clone();

            // Allocate destination with the requested physical strides and copy values logically.
            // Clone dimensions to avoid aliasing — Shape(long[], char) does not clone,
            // and Shape exposes an indexer setter that could otherwise mutate both shapes.
            var destShape = new Shape((long[])this.Shape.dimensions.Clone(), physical);
            var dest = new NDArray(this.typecode, destShape, false) { TensorEngine = TensorEngine };
            NpyIter.Copy(dest, this);
            return dest;
        }
    }
}
