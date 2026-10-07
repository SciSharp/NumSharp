using System;
using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Return a copy of the array collapsed into one dimension.
        /// </summary>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major (C-style), 'F' - column-major (Fortran-style),
        ///     'A' - 'F' if this is F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - the order the elements occur in memory: NpyIter's KEEPORDER sequence (C / F for a C- / F-contiguous
        ///     array, the input's own axis order otherwise; each axis in increasing index, a broadcast axis abstaining).
        /// </param>
        /// <returns>A copy of the input array, flattened to one dimension.</returns>
        /// <exception cref="ArgumentException">An order code other than C, F, A or K.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
        ///     NumPy: flatten() ALWAYS returns a copy. Use ravel() for a view when possible.
        /// </remarks>
        public NDArray flatten(char order = 'C')
        {
            // 'K' reads the elements in NpyIter's KEEPORDER sequence (PyArray_Flatten -> PyArray_CopyAsFlat), which is C or F
            // for a C- or F-contiguous array and the input's own axis order otherwise.
            if (order == 'K' || order == 'k')
                return KeepOrderFlatten();

            char physical = OrderResolver.Resolve(order, this.Shape);

            if (physical == 'F' && this.Shape.NDim > 1 && this.size > 1)
            {
                // F-order flatten: the memory of a fresh F-contiguous copy contains
                // the values in column-major read-out order; reinterpret that buffer
                // as a 1-D array. copy('F') allocated a fresh MemoryBlock that nothing
                // else references; the returned NDArray's ctor bumps the shared
                // Disposer's refcount via InitializeArc, so disposing fcopy at scope
                // exit just drops fcopy's wrapper ref — storage stays alive through
                // the returned NDArray.
                using var fcopy = this.copy('F');
                return new NDArray(new UnmanagedStorage(fcopy.Array, Shape.Vector(size))) { TensorEngine = TensorEngine };
            }

            return new NDArray(new UnmanagedStorage(Storage.CloneData(), Shape.Vector(size))) { TensorEngine = TensorEngine };
        }

        /// <summary>
        ///     <c>flatten(order='K')</c> — NumPy's <c>PyArray_CopyAsFlat</c> with <c>NPY_KEEPORDER</c>: the elements in the
        ///     order NpyIter visits a single operand — its axes sorted by |stride| with stride-0 and extent-1 axes abstaining
        ///     (C order on a tie, <see cref="Shape.NpyIterOutputShape"/>), each axis in increasing index
        ///     (<c>NPY_ITER_DONT_NEGATE_STRIDES</c>: a reversed axis stays reversed).
        /// </summary>
        /// <returns>A fresh 1-D array owning its data.</returns>
        /// <remarks>
        ///     The copy is built as an array of this shape laid out densely in that axis order, filled element for element;
        ///     its buffer, read front to back, is the KEEPORDER sequence, and the result is a 1-D array over that buffer.
        /// </remarks>
        private NDArray KeepOrderFlatten()
        {
            if (size == 0)
                return new NDArray(dtype, Shape.Vector(0));
            var s = Shape;
            var layout = Shape.NpyIterOutputShape((long[])s.dimensions.Clone(), new ReadOnlySpan<Shape>(in s));
            using var ordered = new NDArray(dtype, layout, false);
            NumSharp.Backends.Iteration.NDIter.Copy(ordered.Storage, Storage);
            return new NDArray(new UnmanagedStorage(ordered.Array, Shape.Vector(size))) { TensorEngine = TensorEngine };
        }
    }
}
