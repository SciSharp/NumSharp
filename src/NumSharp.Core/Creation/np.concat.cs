using System;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.0: `np.concat` is the array-API-compatible alias of
        // `np.concatenate` (numpy/_core/__init__.py: `concat = numeric.concatenate`).
        // Every overload below forwards verbatim — one alias per concatenate form.

        /// <summary>
        ///     Join a sequence of arrays along an existing axis.
        ///     Alias of <see cref="concatenate(NDArray[], int?, NDArray, NPTypeCode?, string)"/>
        ///     introduced in NumPy 2.0 for Array API compatibility.
        /// </summary>
        /// <param name="arrays">
        ///     The arrays must have the same shape, except in the dimension
        ///     corresponding to <paramref name="axis"/> (the first, by default).
        /// </param>
        /// <param name="axis">
        ///     The axis along which the arrays will be joined. If
        ///     <c>null</c>, arrays are flattened before use. Default is 0.
        /// </param>
        /// <param name="out">
        ///     If provided, the destination to place the result. Cannot be used
        ///     together with <paramref name="dtype"/>.
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
        /// <returns>The concatenated array.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.concat.html</remarks>
        public static NDArray concat(
            NDArray[] arrays,
            int? axis = 0,
            NDArray @out = null,
            NPTypeCode? dtype = null,
            string casting = "same_kind")
            => concatenate(arrays, axis, @out, dtype, casting);

        public static NDArray concat((NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8 }, axis);

        public static NDArray concat((NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray) arrays, int axis = 0)
            => concatenate(new[] { arrays.Item1, arrays.Item2, arrays.Item3, arrays.Item4, arrays.Item5, arrays.Item6, arrays.Item7, arrays.Item8, arrays.Item9 }, axis);
    }
}
