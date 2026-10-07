using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned
        /// </summary>
        /// <param name="a">Input array. The elements in a are read in the order specified by order, and packed as a 1-D array.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        /// <remarks><br></br>If this array's <see cref="Shape"/> is a sliced or broadcasted, the a copy will be made.</remarks>
        public static NDArray ravel(NDArray a) => ravel(a, 'C');

        /// <summary>
        ///     Return a contiguous flattened array. A 1-D array, containing the elements of the input, is returned — a
        ///     VIEW whenever the requested reading order walks the input's memory front to back, otherwise a copy
        ///     (NumPy's <c>PyArray_Ravel</c>, <c>numpy/_core/src/multiarray/shape.c</c>).
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <param name="order">
        ///     The order in which to read the elements.
        ///     'C' - row-major, 'F' - column-major,
        ///     'A' - 'F' if a is F-contiguous (and not C-contiguous) else 'C',
        ///     'K' - the order the elements occur in memory: C / F for a C- / F-contiguous array, a view when the array is
        ///     dense in some other axis order (a transpose of a contiguous array), else a copy read in NpyIter's KEEPORDER
        ///     axis order (each axis in increasing index — a negative stride is NOT reversed, a broadcast axis abstains).
        /// </param>
        /// <returns>The 1-D result: a view that shares <paramref name="a"/>'s memory (and its writeability), or a fresh
        ///     copy.</returns>
        /// <exception cref="ArgumentException">An order code other than C, F, A or K.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        public static NDArray ravel(NDArray a, char order)
        {
            var s = a.Shape;
            // PyArray_Ravel's order resolution: 'K' is C / F for a C- / F-contiguous array (0-d and size<=1 included);
            // 'A' is F only for a strictly F-contiguous array.
            char o;
            switch (order)
            {
                case 'C':
                case 'c':
                    o = 'C';
                    break;
                case 'F':
                case 'f':
                    o = 'F';
                    break;
                case 'A':
                case 'a':
                    o = s.IsFContiguous && !s.IsContiguous ? 'F' : 'C';
                    break;
                case 'K':
                case 'k':
                    o = s.IsContiguous ? 'C' : s.IsFContiguous ? 'F' : 'K';
                    break;
                default:
                    // The house error for an unknown order code.
                    OrderResolver.Resolve(order, s);
                    throw new ArgumentException($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')", nameof(order));
            }

            if (o == 'C' && s.IsContiguous)
                return a.reshape(Shape.Vector(a.size));

            // An F-contiguous array read column-major, or an array dense in another axis order read in memory order, is
            // its buffer from the view's first element on: one 1-D view with stride one element.
            if ((o == 'F' && s.IsFContiguous) || (o == 'K' && IsDenseInSomeAxisOrder(s)))
            {
                var vec = new Shape(new long[] { a.size }, new long[] { 1 }, s.offset, s.bufferSize);
                // NumPy builds the view with the input's flags: a read-only input gives a read-only view.
                if (!s.IsWriteable)
                    vec = vec.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                return new NDArray(a.Storage.Alias(vec)) { TensorEngine = a.TensorEngine };
            }

            // PyArray_Flatten: a copy, read in the resolved order.
            return a.flatten(o);
        }

        /// <summary>
        ///     <c>PyArray_Ravel</c>'s KEEPORDER view test: sorted by |stride| (<see cref="Shape.SortedStridePerm"/>), the axes
        ///     of more than one element must have the strides of a dense C layout — 1, then the product of the faster axes'
        ///     extents — compared SIGNED, so a negative or a stride-0 (broadcast) axis never qualifies.
        /// </summary>
        /// <param name="s">The array's shape.</param>
        /// <returns>True when a 1-D view starting at the array's first element reads every element in memory order.</returns>
        private static bool IsDenseInSomeAxisOrder(in Shape s)
        {
            var perm = Shape.SortedStridePerm(s.strides);
            long expected = 1;
            for (int i = perm.Length - 1; i >= 0; i--)
            {
                int axis = perm[i];
                long extent = s.dimensions[axis];
                if (extent == 1)
                    continue; // a size-one axis is never stepped, so its stride does not matter
                if (s.strides[axis] != expected)
                    return false;
                expected *= extent;
            }
            return true;
        }
    }
}
