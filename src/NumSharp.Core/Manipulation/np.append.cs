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
        /// <param name="arr">Input array (a C# number is NumPy's Python number: see the remarks).</param>
        /// <param name="values">Values to append: an array, a C# array (its own dtype), or a C# number, bool or string
        ///     (NumPy's Python scalar). Shape must match <paramref name="arr"/> on all dimensions except
        ///     <paramref name="axis"/> when <paramref name="axis"/> is given; otherwise it is flattened.</param>
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
        ///     <para><b>A C# number is NumPy's Python number.</b> NumPy's <c>asanyarray(arr)</c> and <c>ravel(values)</c>
        ///     build an ARRAY of a Python scalar, in the Python number's default dtype - int64 (uint64 past the int64
        ///     range), float64, complex128 - whatever the C# width of the literal: <c>np.append(int32_array, 5)</c> and
        ///     <c>np.append(uint8_array, 300)</c> are int64, <c>np.append(float32_array, 1.5f)</c> float64. A bool stays
        ///     bool and a string is its <c>U</c> array, as in NumPy. A typed value keeps its dtype: pass
        ///     <c>np.array(new byte[] { 3 })</c> to append a uint8.</para>
        ///     <para>API-parity batch 12: one signature. The <c>object values</c> overload is removed (a C# array, number or
        ///     string binds this one through the implicit conversions; a <c>List</c> is passed as
        ///     <c>np.array(list)</c>), and the C# number's dtype is NumPy's (it was the C# width: int32 for <c>5</c>).</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>);
        ///     a 0-d input with an axis (<c>zero-dimensional arrays cannot be concatenated</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the inputs' rank.</exception>
        /// <exception cref="IncorrectShapeException">With an axis, the inputs' ranks or other axes differ.</exception>
        /// <exception cref="InvalidCastException">An input that does not cast to the result dtype.</exception>
        public static NDArray append(NDArray arr, NDArray values, int? axis = null)
        {
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (values is null) throw new ArgumentNullException(nameof(values));

            // NumPy's statements (the Python scalars' arrays, the ravels when axis is None, then concatenate) are the
            // array object's.
            return NDArray.append(arr, values, axis);
        }
    }
}
