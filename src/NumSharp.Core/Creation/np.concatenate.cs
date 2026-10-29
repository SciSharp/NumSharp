using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Join a sequence of arrays along an existing axis.
        /// </summary>
        /// <param name="arrays">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to <paramref name="axis"/> (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined. If
        ///     <c>null</c>, arrays are flattened before use — a 0-d array included, which joins as one element (an axis
        ///     refuses it: "zero-dimensional arrays cannot be concatenated"). Default is 0.
        ///     Negative axes are normalized against the input ndim.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result. The shape must
        ///     be correct, matching what would have been returned with no
        ///     <c>out</c> argument. Cannot be used together with <paramref name="dtype"/>.
        /// </param>
        /// <param name="dtype">
        ///     If provided, the result array will have this dtype. Cannot be
        ///     used together with <paramref name="out"/>.
        /// </param>
        /// <param name="casting">
        ///     Controls what kind of data casting may occur. One of
        ///     <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c>
        ///     (default), or <c>"unsafe"</c>.
        /// </param>
        /// <returns>The concatenated array (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arrays"/> or one of its elements is null.</exception>
        /// <exception cref="ArgumentException">An unknown <paramref name="casting"/> name; both <paramref name="out"/> and
        ///     <paramref name="dtype"/>.</exception>
        /// <exception cref="ValueError">No arrays (<c>need at least one array to concatenate</c>); a 0-d array with an axis
        ///     (<c>zero-dimensional arrays cannot be concatenated</c>; NumPy's ValueError — <c>axis=None</c> ravels it into one
        ///     element instead).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds for the inputs' rank.</exception>
        /// <exception cref="IncorrectShapeException">The inputs' ranks or non-axis dimensions differ, or
        ///     <paramref name="out"/> has the wrong rank or shape (NumPy's ValueError texts).</exception>
        /// <exception cref="InvalidCastException">An input does not cast to the result dtype under
        ///     <paramref name="casting"/> (NumPy's TypeError text).</exception>
        /// <exception cref="ValueError">Numbers joined into a numeric <paramref name="dtype"/> with text that does not parse
        ///     as that number (the string route, in <c>NDArray.Multiarray.cs</c>, formats numbers into a text result and parses
        ///     text into a numeric one).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray concatenate(
            NDArray[] arrays,
            int? axis = 0,
            NDArray @out = null,
            DType dtype = null,
            string casting = "same_kind")
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arrays))
                return NDMatrixBoundary.Wrapped((arrays, axis, @out, dtype, casting), static s => concatenate(s.arrays, s.axis, s.@out, s.dtype, s.casting), @out);
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));
            return NDArray.ConcatenateInto(arrays, axis, @out, dtype, casting);
        }

        // ---------------- Tuple-arity convenience overloads ----------------
        // NumPy permits `np.concatenate((a, b, c))` as a tuple. These mirror
        // that ergonomic with the array-form's default keyword params.

        public static NDArray concatenate((NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8 }, axis);

        public static NDArray concatenate((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8, arrays.Item9 }, axis);
    }
}
