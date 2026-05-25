using System;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.append ==============================
        // Append values to the end of an array.
        //
        // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::append
        // Three-liner in NumPy — asanyarray then concatenate, ravel both inputs
        // when axis is None. We follow the same pattern exactly:
        //
        //     arr = asanyarray(arr)
        //     if axis is None:
        //         arr = arr.ravel() if arr.ndim != 1 else arr
        //         values = ravel(values)
        //         axis = arr.ndim - 1
        //     return concatenate((arr, values), axis=axis)
        //
        // Type promotion is fully delegated to np.concatenate (which already
        // honours NEP50 — empty values default to float64, mixed int/float
        // promotes to float, etc.).

        /// <summary>
        ///     Append <paramref name="values"/> to the end of <paramref name="arr"/>.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="values">Values to append. Shape must match
        ///     <paramref name="arr"/> on all dimensions except <paramref name="axis"/>
        ///     when <paramref name="axis"/> is given; otherwise it is flattened.</param>
        /// <param name="axis">Axis along which to append. <c>null</c> (default)
        ///     flattens both <paramref name="arr"/> and <paramref name="values"/>
        ///     to 1-D before concatenation.</param>
        /// <returns>A new array with <paramref name="values"/> appended to
        ///     <paramref name="arr"/> along <paramref name="axis"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.append.html</remarks>
        public static NDArray append(NDArray arr, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            if (axis is null)
            {
                // Both inputs ravel'd into 1-D, then concatenate along axis 0.
                // Mirrors NumPy's `arr.ravel() if arr.ndim != 1 else arr`.
                NDArray flatArr = arr.ndim == 1 ? arr : np.ravel(arr);
                NDArray flatVals = values.ndim == 1 ? values : np.ravel(values);
                try
                {
                    return np.concatenate(new[] { flatArr, flatVals }, axis: 0);
                }
                finally
                {
                    if (!ReferenceEquals(flatArr, arr)) flatArr.Dispose();
                    if (!ReferenceEquals(flatVals, values)) flatVals.Dispose();
                }
            }

            // Axis given — concatenate handles the shape validation and dtype
            // promotion (NEP50, mixed-dtype paths, etc.) directly.
            return np.concatenate(new[] { arr, values }, axis: axis.Value);
        }

        /// <summary>
        ///     Scalar / generic-value overload that wraps the value via
        ///     <see cref="np.asanyarray"/> before delegating. Matches NumPy's
        ///     <c>np.append([1,2,3], 4)</c> idiom — the scalar is auto-coerced
        ///     to a 0-D ndarray (whose ravel is shape (1,)).
        /// </summary>
        public static NDArray append(NDArray arr, object values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            var asArr = np.asanyarray(values);
            try { return append(arr, asArr, axis); }
            finally { asArr.Dispose(); }
        }
    }
}
