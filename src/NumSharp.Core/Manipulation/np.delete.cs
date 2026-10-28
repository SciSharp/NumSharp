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
        ///     NumPy's <c>delete</c>/<c>insert</c> allocate their output in ONE of two layouts —
        ///     <c>arrorder = 'F' if arr.flags.fnc else 'C'</c> (<c>_function_base_impl.py</c>): F-contiguous for a
        ///     strictly F-contiguous source (probed: num 1286), C-contiguous for every other source, a transposed or
        ///     broadcast one included. The composition kernels here may build another layout (concatenate keeps its
        ///     inputs' axis order), so the result is copied into the NumPy layout when it is not already in it.
        ///     IDEMPOTENT (a result already in the layout passes through), so nested overload delegation is safe.
        /// </summary>
        /// <param name="arr">The source array (its flags decide the layout).</param>
        /// <param name="result">The composed result.</param>
        /// <returns><paramref name="result"/>, or a C / F copy of it.</returns>
        internal static NDArray WithSourceOrder(NDArray arr, NDArray result)
        {
            if (result.ndim <= 1)
                return result;
            bool wantF = arr.Shape.IsFContiguous && !arr.Shape.IsContiguous;
            if (wantF)
                return result.Shape.IsFContiguous ? result : asfortranarray(result);
            return result.Shape.IsContiguous ? result : ascontiguousarray(result);
        }

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

        // ---------------------------- helpers ----------------------------

        /// <summary>
        ///     Resolves <c>axis=None</c> to "ravel + axis=0" and normalises a
        ///     non-null axis against <c>arr.ndim</c>. The returned <c>work</c>
        ///     array is either <c>arr</c> itself or a freshly-allocated ravel
        ///     (which the caller can treat as owning since insert/delete always
        ///     produces a new array).
        /// </summary>
        /// <param name="arr">The input of insert / delete.</param>
        /// <param name="axis">The caller's axis, or null for the flattened array.</param>
        /// <returns>The array to work on and the non-negative axis.</returns>
        /// <exception cref="AxisError">The axis is out of bounds (NumPy's <c>normalize_axis_index</c> text).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static (NDArray work, int axis) PrepareAxisContext(NDArray arr, int? axis)
        {
            if (axis is null)
            {
                NDArray work = arr.ndim == 1 ? arr : np.ravel(arr);
                return (work, 0);
            }

            int ax = axis.Value;
            if (ax < 0) ax += arr.ndim;
            // NumPy's normalize_axis_index raises AxisError (a subclass of the ArgumentOutOfRangeException this used to be).
            if (ax < 0 || ax >= arr.ndim)
                throw new AxisError(axis.Value, arr.ndim);
            return (arr, ax);
        }

        /// <summary>
        ///     Casts <paramref name="obj"/> to <see cref="long"/> with NumPy-style
        ///     bounds error on out-of-range. Used by the scalar-index fast paths.
        /// </summary>
        /// <param name="obj">An integer index array; its first element is read (a uint64 above long's range wraps).</param>
        /// <param name="fn">The NumPy function's name for the error text (<c>"insert"</c> / <c>"delete"</c>).</param>
        /// <returns>The index as a <see cref="long"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="obj"/> is not an integer array (the callers test that
        ///     first, so this is not reached from them).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static long ToInt64Scalar(NDArray obj, string fn)
        {
            // Accept any integer dtype (NumPy: kind in "ui"). Bool is handled by
            // the caller before this point.
            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:   return obj.GetByte(0);
                case NPTypeCode.SByte:  return obj.GetSByte(0);
                case NPTypeCode.Char:   return obj.GetChar(0);
                case NPTypeCode.Int16:  return obj.GetInt16(0);
                case NPTypeCode.UInt16: return obj.GetUInt16(0);
                case NPTypeCode.Int32:  return obj.GetInt32(0);
                case NPTypeCode.UInt32: return obj.GetUInt32(0);
                case NPTypeCode.Int64:  return obj.GetInt64(0);
                case NPTypeCode.UInt64: return (long)obj.GetUInt64(0);
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.type.Name}",
                        "obj");
            }
        }

        /// <summary>
        ///     Materialises <paramref name="obj"/> as a managed <c>long[]</c>.
        ///     Any integer dtype is accepted; size-0 input returns an empty array
        ///     so the caller's "no-op" branch kicks in.
        /// </summary>
        /// <param name="obj">A 0-d or 1-D integer index array (a uint64 above long's range wraps).</param>
        /// <param name="fn">The NumPy function's name for the error texts (<c>"insert"</c> / <c>"delete"</c>).</param>
        /// <returns>A fresh array of the indices in order.</returns>
        /// <exception cref="ArgumentException"><paramref name="obj"/> has more than one dimension or is not an integer
        ///     array (the callers test both first, so neither is reached from them).</exception>
        /// <remarks>Internal for np.insert's and np.delete's statements (NDArray.FunctionBase.cs), which share it.</remarks>
        internal static long[] ToInt64Vector(NDArray obj, string fn)
        {
            if (obj.ndim > 1)
                throw new ArgumentException(
                    $"np.{fn}: index array argument obj must be one dimensional or scalar",
                    "obj");

            long n = obj.size;
            var result = new long[n];

            switch (obj.GetTypeCode)
            {
                case NPTypeCode.Byte:
                    for (long i = 0; i < n; i++) result[i] = obj.GetByte((int)i);
                    break;
                case NPTypeCode.SByte:
                    for (long i = 0; i < n; i++) result[i] = obj.GetSByte((int)i);
                    break;
                case NPTypeCode.Char:
                    for (long i = 0; i < n; i++) result[i] = obj.GetChar((int)i);
                    break;
                case NPTypeCode.Int16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt16((int)i);
                    break;
                case NPTypeCode.UInt16:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt16((int)i);
                    break;
                case NPTypeCode.Int32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt32((int)i);
                    break;
                case NPTypeCode.UInt32:
                    for (long i = 0; i < n; i++) result[i] = obj.GetUInt32((int)i);
                    break;
                case NPTypeCode.Int64:
                    for (long i = 0; i < n; i++) result[i] = obj.GetInt64((int)i);
                    break;
                case NPTypeCode.UInt64:
                    for (long i = 0; i < n; i++) result[i] = (long)obj.GetUInt64((int)i);
                    break;
                default:
                    throw new ArgumentException(
                        $"np.{fn}: obj must be int / slice / integer array / boolean mask, got {obj.dtype.type.Name}",
                        "obj");
            }

            return result;
        }

        /// <summary>
        ///     Allocates a fresh, owning copy of <paramref name="arr"/>. Used by
        ///     delete/insert no-op paths to honour NumPy's "delete does not occur
        ///     in-place" contract.
        /// </summary>
        internal static NDArray DuplicateArray(NDArray arr) => np.copy(arr);

        /// <summary>
        ///     Returns an axis-aligned slice view <c>arr[..., start:stop, ...]</c>
        ///     by building a <see cref="Slice"/> array. The result shares storage
        ///     with <paramref name="arr"/> when <c>arr.Shape.IsContiguous</c> /
        ///     stride-friendly; the caller is responsible for disposing the wrapper.
        /// </summary>
        internal static NDArray SliceAlongAxis(NDArray arr, int axis, long start, long stop)
        {
            int ndim = arr.ndim;
            var slices = new Slice[ndim];
            for (int i = 0; i < ndim; i++)
                slices[i] = i == axis ? new Slice(start, stop) : Slice.All;
            return arr[slices];
        }

        /// <summary>
        ///     Python-compatible <c>slice.indices(N)</c> implementation. Resolves
        ///     None/negative bounds to absolute positions and clamps to a legal
        ///     iteration range. Used by both <see cref="DeleteSlice"/> and
        ///     <see cref="NDArray.InsertExpandSliceObj"/>.
        /// </summary>
        internal static (long start, long stop, long step) PythonSliceIndices(Slice s, long N)
        {
            long step = s.Step == 0 ? 1 : s.Step;
            if (step == 0)
                throw new ArgumentException("slice step cannot be zero");

            long start, stop;
            long defaultStart = step > 0 ? 0 : N - 1;
            long defaultStop = step > 0 ? N : -1; // exclusive; -1 sentinel for backwards

            if (s.Start.HasValue)
            {
                start = s.Start.Value;
                if (start < 0) start += N;
                if (step > 0)
                {
                    if (start < 0) start = 0;
                    if (start > N) start = N;
                }
                else
                {
                    if (start < 0) start = -1;
                    if (start >= N) start = N - 1;
                }
            }
            else start = defaultStart;

            if (s.Stop.HasValue)
            {
                stop = s.Stop.Value;
                if (stop < 0) stop += N;
                if (step > 0)
                {
                    if (stop < 0) stop = 0;
                    if (stop > N) stop = N;
                }
                else
                {
                    if (stop < 0) stop = -1;
                    if (stop >= N) stop = N - 1;
                }
            }
            else stop = defaultStop;

            return (start, stop, step);
        }
    }
}
