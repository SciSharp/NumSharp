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
            // The uninitialized-shape sentinel has no dimensions to order or fill — Clone() handles it.
            if (this.Shape.IsEmpty)
                return Clone();

            // Unified allocate-and-fill copy core: resolves order (C/F/A/K), allocates, and fills via
            // the NDIter copy primitive. Absorbs every layout — scalar, (1,), strided, broadcast,
            // transposed — so copy() and astype() share one path.
            return NDIter.CopyAs(this.typecode, this, order, TensorEngine);
        }
    }
}
