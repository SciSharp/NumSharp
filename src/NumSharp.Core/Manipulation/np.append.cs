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
        ///     <paramref name="arr"/> along <paramref name="axis"/>: a plain array for <c>numpy.char.chararray</c> operands too
        ///     (NumPy's <c>np.concatenate</c> drops the class).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="values"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.append.html
        ///     <para>NumPy's statements are the array object's (<c>NDArray.append</c>), which releases the raveled copies
        ///     (or views) of a multi-dimensional <paramref name="arr"/> or <paramref name="values"/>, exceptions included,
        ///     and yields the concatenated result. A 1-D input is passed to concatenate as-is.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the inputs' rank.</exception>
        /// <exception cref="IncorrectShapeException">With an axis, the inputs' ranks or other axes differ.</exception>
        /// <exception cref="InvalidCastException">An input that does not cast to the result dtype.</exception>
        public static NDArray append(NDArray arr, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr, values))
                return NDMatrixBoundary.Wrapped((arr, values, axis), static s => append(s.arr, s.values, s.axis));
            // a numpy.char.chararray operand: NumPy's append ends with np.concatenate, which drops the class; the rule's
            // region keeps its inner ravel calls plain (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr, values))
                return NDCharArrayBoundary.Plain((arr, values, axis), static s => append(s.arr, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // NumPy's statements (the ravels when axis is None, then concatenate) are the array object's.
            return NDArray.append(arr, values, axis);
        }

        /// <summary>
        ///     Scalar / generic-value overload that wraps the value via
        ///     <see cref="np.asanyarray(in object, DType, string)"/> before delegating. Matches NumPy's
        ///     <c>np.append([1,2,3], 4)</c> idiom — the scalar is auto-coerced
        ///     to a 0-D ndarray (whose ravel is shape (1,)).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="values">Values to append: a scalar, a C# array or collection, or an <see cref="NDArray"/>
        ///     whose static type is <see cref="object"/>. Converted with <see cref="np.asanyarray(in object, DType, string)"/>.</param>
        /// <param name="axis">Axis along which to append; <c>null</c> (default) flattens both operands first.</param>
        /// <returns>A new array with <paramref name="values"/> appended to <paramref name="arr"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
        ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="values"/> is null.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of a scalar or collection is released on return. An
        ///     <see cref="NDArray"/> passed as <see cref="object"/> comes back from <see cref="np.asanyarray(in object, DType, string)"/>
        ///     unchanged; it was built before the scope opened, so the scope never disposes it. (The hand-written
        ///     version disposed that conversion unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray append(NDArray arr, object values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, values, axis), static s => append(s.arr, s.values, s.axis));
            // a numpy.char.chararray operand (either side): a plain result, its conversions computed plain (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr, values as NDArray))
                return NDCharArrayBoundary.Plain((arr, values, axis), static s => append(s.arr, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // The C# value's conversion (NumPy's array_like), then the NDArray overload: its matrix rule still sees a
            // numpy.matrix passed as object, which this overload's own rule (arr only) does not.
            var asArr = np.asanyarray(values);
            return append(arr, asArr, axis);
        }
    }
}
