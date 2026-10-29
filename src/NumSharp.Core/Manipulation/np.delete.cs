using System;
using NumSharp.Backends;
using NumSharp.Generic;

namespace NumSharp
{
    public static partial class np
    {
        // ============================== np.delete ==============================
        // Return a copy of an array with sub-arrays along an axis deleted.
        //
        // NumPy 2.4.2 reference: numpy/lib/_function_base_impl.py::delete
        // Three obj shapes drive three execution paths:
        //   1. scalar int       — drop one position along axis
        //   2. slice            — drop a stride-defined range along axis
        //   3. integer ndarray  — drop multiple (possibly duplicate / unordered) positions
        //   4. boolean ndarray  — keep where mask is False (length must match axis dim)
        //
        // axis=None ravels the input to 1-D before deletion.
        //
        // For 1D contiguous arrays, the scalar/slice paths reuse np.concatenate of
        // pre/post views (zero-copy slicing + Buffer.MemoryCopy on the dst). The
        // array/bool paths build a keep-mask and delegate to np.compress, which
        // owns the fused popcount + axis-gather kernel.

        /// <summary>
        ///     Return a new array with the element at <paramref name="obj"/>
        ///     along <paramref name="axis"/> removed.
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">Integer index of the position to remove. Accepts
        ///     negative indices (counted from the end). Raises
        ///     <see cref="IndexOutOfRangeException"/> when out of bounds for the
        ///     selected axis.</param>
        /// <param name="axis">Axis along which to delete. <c>null</c> (default)
        ///     flattens <paramref name="arr"/> first and returns a 1-D result.</param>
        /// <returns>A C-contiguous copy of <paramref name="arr"/> with one
        ///     sub-array removed along <paramref name="axis"/>.
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.delete.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray delete(NDArray arr, int obj, int? axis = null)
            => NDMatrixBoundary.Sees(arr)   // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            ? NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis))
            : NDCharArrayBoundary.Sees(arr)   // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            ? NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis))
            : delete(arr, (long)obj, axis);

        /// <summary>
        ///     Long-index overload of <see cref="delete(NDArray, int, int?)"/>.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        // Scope: delete is a composition that owns transients — PrepareAxisContext mints a ravel
        // `work` copy for axis=None on an N-D input, the DeleteXxx helpers build concatenated chunks,
        // and WithSourceOrder mints an asfortranarray copy for an F-contiguous source (orphaning the
        // C-contiguous intermediate). None were reclaimed (measured: one bucketed buffer escaped per
        // delete). [NDScoped] reclaims them all while yielding the fresh result; a passthrough `work`
        // (already 1-D input) is never tracked. The DeleteXxx helpers are [NDScopedCovered] and the
        // bool paths' keepArr is left to the scope too: no member of this family disposes by hand.
        [NDScoped]
        public static NDArray delete(NDArray arr, long obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }

        /// <summary>
        ///     Slice-index overload. <paramref name="obj"/> is interpreted via
        ///     Python <c>slice.indices(N)</c> against the axis length.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        [NDScoped]
        public static NDArray delete(NDArray arr, Slice obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }

        /// <summary>
        ///     Array-of-indices overload. Negative indices are normalized; duplicates
        ///     are silently collapsed (each axis position is removed at most once).
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        [NDScoped]
        public static NDArray delete(NDArray arr, int[] obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }

        /// <summary>
        ///     Long-array-of-indices overload.
        /// </summary>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <remarks>
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.
        /// </remarks>
        [NDScoped]
        public static NDArray delete(NDArray arr, long[] obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }

        /// <summary>
        ///     Bool-array overload — values are interpreted as a keep-mask
        ///     inversion. Length must match the targeted axis size (NumPy raises
        ///     <c>ValueError</c> otherwise).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">True at each position to remove along <paramref name="axis"/>.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the masked sub-arrays (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="obj"/>'s length differs from the axis length.</exception>
        /// <remarks>Scoped by <c>[NDScoped]</c>: the ravel work array, the keep mask and the C-order result that an
        ///     F-ordered relabel supersedes are released on return.</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray delete(NDArray arr, bool[] obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }

        /// <summary>
        ///     NDArray-typed obj dispatch — selects the integer-array path or the
        ///     boolean-mask path based on <c>obj.GetTypeCode</c>. 0-D and 1-element
        ///     integer arrays collapse to the scalar-index fast path (matching
        ///     NumPy's <c>obj.size == 1 and obj.dtype.kind in "ui": obj = obj.item()</c>).
        /// </summary>
        /// <param name="arr">Input array.</param>
        /// <param name="obj">A boolean mask (removed where true) or an integer index array.</param>
        /// <param name="axis">The axis to delete along; <c>null</c> deletes from the flattened array.</param>
        /// <returns>A fresh array without the selected sub-arrays (F-ordered when <paramref name="arr"/> is).
        ///     A <c>numpy.char.chararray</c> operand gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>: OWNDATA False); a result of another dtype stays plain.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arr"/> or <paramref name="obj"/> is null.</exception>
        /// <exception cref="ArgumentException">A boolean <paramref name="obj"/> is not 1-D or its length differs
        ///     from the axis length.</exception>
        /// <exception cref="IndexError"><paramref name="obj"/>'s dtype is neither boolean nor integer — text, a float,
        ///     complex, decimal — whatever its size (NumPy's <c>keep[obj,] = False</c>: <c>arrays used as indices must be of
        ///     integer (or boolean) type</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index is outside <c>[-N, N)</c>.</exception>
        /// <remarks>
        ///     An integer <paramref name="obj"/> of any rank is accepted, as NumPy's fancy <c>keep[obj,]</c> accepts it (its
        ///     elements are the indices; one element takes the single-index path).
        ///     <para>Scoped by <c>[NDScoped]</c>: the ravel work array, the keep mask, the helpers' chunk views and
        ///     the C-order result that an F-ordered relabel supersedes are released on return.</para>
        /// </remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped]
        public static NDArray delete(NDArray arr, NDArray obj, int? axis = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arr))
                return NDMatrixBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            // a numpy.char.chararray operand comes back as chararray.__array_wrap__'s view of the result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(arr))
                return NDCharArrayBoundary.Wrapped((arr, obj, axis), static s => delete(s.arr, s.obj, s.axis));
            if (arr is null) throw new ArgumentNullException(nameof(arr));
            if (obj is null) throw new ArgumentNullException(nameof(obj));

            // NumPy's statements after the class boundaries and the null checks: NDArray.Delete (NDArray.FunctionBase.cs).
            return NDArray.Delete(arr, obj, axis);
        }
    }
}
