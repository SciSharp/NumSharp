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
        ///     copy.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray over the result (NumPy creates it as the
        ///     subclass), owning its data exactly when the plain result would.</returns>
        /// <exception cref="ArgumentException">An order code other than C, F, A or K.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ravel.html</remarks>
        public static NDArray ravel(NDArray a, char order)
        {
            // a numpy.char.chararray operand comes back as a chararray, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                return NDCharArrayBoundary.Kept((a, order), static s => ravel(s.a, s.order));
            return NDArray.Ravel(a, order);
        }
    }
}
