using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.insert ==============================
        // Insert values along the given axis before the given indices.
        //
        // The body is NDArray.Insert (Manipulation/NDArray.FunctionBase.cs, beside this facade), which describes
        // the two branches; the facades keep their signatures, documentation, class boundaries and null checks.

        /// <summary>
        ///     Insert <paramref name="values"/> along <paramref name="axis"/>
        ///     before the position <paramref name="obj"/>. Scalar-obj path.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.insert.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        public static NDArray insert(NDArray arr, int obj, NDArray values, int? axis = null)
            => NDMatrixBoundary.Sees(arr)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis))
            : NDCharArrayBoundary.Sees(arr)   // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis))
            : insert(arr, (long)obj, values, axis);

        /// <summary>
        ///     Long-index overload of <see cref="insert(NDArray, int, NDArray, int?)"/>.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        [NDScoped] // reclaims PrepareAxisContext's ravel work array and the pre-order intermediate WithSourceOrder supersedes (np.delete's exact pattern)
        public static NDArray insert(NDArray arr, long obj, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.Insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     Scalar-obj overload accepting a scalar value (NumPy: <c>np.insert(a, 1, 99)</c>).
        ///     Broadcasts the value to the required shape using <c>arr.dtype</c>.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The index before which the value is inserted; negative values count from the end.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="value"/> is Python's <c>None</c>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c> or a
        ///     <c>StringDType</c> NA object, converted after the index checks; int()'s TypeError for an integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray insert(NDArray arr, long obj, object value, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = NDArray.ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     int-overload accepting a scalar value.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        public static NDArray insert(NDArray arr, int obj, object value, int? axis = null)
            => NDMatrixBoundary.Sees(arr)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis))
            : NDCharArrayBoundary.Sees(arr)   // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis))
            : insert(arr, (long)obj, value, axis);

        /// <summary>
        ///     Slice-obj overload. <paramref name="obj"/> is expanded via
        ///     Python <c>slice.indices(N)</c> into an indices array, then the
        ///     multi-index branch runs.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, Slice obj, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.Insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     Slice-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions, expanded with Python's <c>slice.indices(N)</c>.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="value"/> is Python's <c>None</c>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c> or a
        ///     <c>StringDType</c> NA object, converted after the index checks; int()'s TypeError for an integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray insert(NDArray arr, Slice obj, object value, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = NDArray.ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     int[]-obj overload — NumPy's sequence obj (<c>np.insert(arr, [1, 3], v)</c>).
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="values"/> is Python's <c>None</c>, converted after the index checks: int()'s TypeError for an
        ///     integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     A ONE-element sequence takes NumPy's <c>indices.size == 1</c> branch (every value inserted at that one
        ///     position: <c>np.insert(a, [0], [7, 8, 9])</c> inserts three), without the scalar-obj <c>moveaxis</c> — so
        ///     <c>np.insert(arr, [1], v) != np.insert(arr, 1, v)</c> when v has several axes. Longer sequences take the
        ///     multi-index branch. (A one-element sequence used to take the multi-index branch, which demands one value per
        ///     index and so raised a broadcast error for several values.)
        /// </remarks>
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, int[] obj, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.Insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     int[]-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="value"/> is Python's <c>None</c>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c> or a
        ///     <c>StringDType</c> NA object, converted after the index checks; int()'s TypeError for an integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray insert(NDArray arr, int[] obj, object value, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = NDArray.ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     long[]-obj overload — see <see cref="insert(NDArray, int[], NDArray, int?)"/> (a one-element sequence takes
        ///     NumPy's single-index branch, without the scalar-obj <c>moveaxis</c>).
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="values"/> is Python's <c>None</c>, converted after the index checks: int()'s TypeError for an
        ///     integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        [NDScoped] // same transient set as the long-obj overload
        public static NDArray insert(NDArray arr, long[] obj, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.Insert(arr, obj, values, axis);
        }

        /// <summary>
        ///     long[]-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">The insertion positions (each before the original element at that index).</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="value"/> is Python's <c>None</c>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c> or a
        ///     <c>StringDType</c> NA object, converted after the index checks; int()'s TypeError for an integer array).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray insert(NDArray arr, long[] obj, object value, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = NDArray.ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     NDArray-obj overload accepting a scalar value.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A 0-d or 1-D integer index array, or a 1-D boolean mask of insertion positions.</param>
        /// <param name="value">The value(s) to insert: a scalar, a C# array or collection, or an
        ///     <see cref="NDArray"/> whose static type is <see cref="object"/>. Cast to <paramref name="arr"/>'s
        ///     dtype.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the value(s) inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="value"/> is Python's <c>None</c>: NaN, <c>nan+nanj</c>, <c>False</c>, <c>'None'</c> or a
        ///     <c>StringDType</c> NA object, converted after the index checks; int()'s TypeError for an integer array).</exception>
        /// <exception cref="ValueError">See <see cref="insert(NDArray, NDArray, NDArray, int?)"/>.</exception>
        /// <exception cref="TypeError">See <see cref="insert(NDArray, NDArray, NDArray, int?)"/>.</exception>
        /// <exception cref="IndexError">See <see cref="insert(NDArray, NDArray, NDArray, int?)"/>.</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <remarks>
        ///     Scoped by <c>[NDScoped]</c>: the conversion of <paramref name="value"/> is built in THIS frame,
        ///     before the array overload's scope opens, so this overload releases it with a scope of its own. An
        ///     <see cref="NDArray"/> passed as <paramref name="value"/> comes back unchanged and was built before
        ///     the scope opened, so it is never disposed. (The hand-written version disposed the conversion
        ///     unconditionally, which released the caller's own array.)
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray insert(NDArray arr, NDArray obj, object value, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, value, axis), static s => insert(s.arr, s.obj, s.value, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            var asArr = NDArray.ScalarValueAsArray(value, arr.dtype);
            return insert(arr, obj, asArr, axis);
        }

        /// <summary>
        ///     NDArray-typed obj. Dispatches:
        ///     <list type="bullet">
        ///       <item>0-D / 1-element integer ⇒ scalar single-index path.</item>
        ///       <item>1-D bool ⇒ <see cref="np.flatnonzero"/> ⇒ multi-index path.</item>
        ///       <item>1-D integer ⇒ multi-index path.</item>
        ///       <item>&gt; 1-D ⇒ <see cref="ArgumentException"/> matching NumPy's
        ///             "index array argument obj to insert must be one dimensional or scalar".</item>
        ///     </list>
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A 0-d or 1-D integer index array, or a 1-D boolean mask of insertion positions.</param>
        /// <param name="values">The values to insert, cast to <paramref name="arr"/>'s dtype and broadcast to the
        ///     insertion slot.</param>
        /// <param name="axis">The axis to insert along; <c>null</c> inserts into the flattened array.</param>
        /// <returns>A fresh array with the values inserted (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null (a null <paramref name="values"/> is Python's <c>None</c>, converted after the index checks: int()'s TypeError for an
        ///     integer array).</exception>
        /// <exception cref="ValueError"><paramref name="obj"/> is a boolean mask that is not 1-D, or an index array of more
        ///     than one dimension (NumPy's two texts).</exception>
        /// <exception cref="IndexOutOfRangeException">An insertion index is outside <c>[-N, N]</c>.</exception>
        /// <exception cref="TypeError">A one-element text or complex <paramref name="obj"/> (<c>'&lt;' not supported between
        ///     instances of 'str' and 'int'</c>), a text <paramref name="obj"/> of any other size (<c>ufunc 'less' did not
        ///     contain a loop …</c>), or a one-element float within bounds (<c>slice indices must be integers …</c>).</exception>
        /// <exception cref="IndexError">A float / complex <paramref name="obj"/> of any size but one (<c>arrays used as indices
        ///     must be of integer (or boolean) type</c>), or a one-element float out of bounds.</exception>
        /// <exception cref="ValueError">Text values for a numeric array that do not parse as its number type (NumPy converts
        ///     the values with <c>array(values, dtype=arr.dtype)</c>: <c>could not convert string to float</c>).</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the same transient set as the long-obj overload, plus the
        ///     bool-mask flatnonzero temp, is released on return.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // same transient set as the long-obj overload, plus the bool-mask flatnonzero temp
        public static NDArray insert(NDArray arr, NDArray obj, NDArray values, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, values, axis), static s => insert(s.arr, s.obj, s.values, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            // A null `values` is Python's None, converted where NumPy converts the values: after the index checks
            // (NDArray.FunctionBase.cs's CoerceValuesForSingleIndex / CoerceValuesForMultiIndex).
            return NDArray.Insert(arr, obj, values, axis);
        }
    }
}
