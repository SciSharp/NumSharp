using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin in NumPy's signature:
    //     (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    // (plan docs/plans/api-parameter-parity.md, item A4). The engine is DefaultEngine.MaxMinNumPy
    // (Default.Reduction.MinMax.Keywords.cs); its header has NumPy's statement order and the two engines.
    //
    // OVERLOADS. Each function keeps NumSharp's earlier forms next to NumPy's, and overload-resolution priority sorts them:
    //   * NumPy's form, priority 1 — every ported call binds it (named or positional, out= in NumPy's third slot);
    //   * (a, int? axis, bool keepdims), priority 1 — the positional keepdims every earlier NumSharp caller wrote as
    //     np.max(a, 1, true). Without it that call would bind NumPy's form with `true` converted to a 0-d out array. C#
    //     prefers it there (bool → bool beats bool → NDArray), and when both bind by name it wins as the form whose every
    //     parameter is given. NumPy itself rejects a positional True as `out` ("return arrays must be of ArrayType"), so
    //     no ported call is affected;
    //   * (a, int? axis = null, bool keepdims = false, DType dtype = null), priority 0 — NumSharp's dtype extension (NumPy's
    //     np.max has no dtype=), reached only by a call that names or passes a dtype, since priority 1 hides it otherwise.
    //
    // A keyword-less call (no out, initial or where) takes the engine's long-standing route, ReduceAMax / ReduceAMin / NanMax
    // / NanMin, unchanged; any keyword goes to MaxMinNumPy.
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis — NumPy's <c>np.max</c>, i.e.
        ///     <c>np.maximum.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end); <c>null</c> reduces every axis.</param>
        /// <param name="out">The array the result is written into and returned. It must have the result's shape (the
        ///     keepdims shape with <paramref name="keepdims"/>); its dtype may differ — the loop runs in the promotion of
        ///     (<paramref name="out"/>, <paramref name="a"/>), and the result is cast into it unsafely (a float64 maximum lands
        ///     in a bool out as <c>True</c>). Footgun: a narrowing out is lossy DURING the reduction too, as in NumPy — each
        ///     output's first element (or the initial value) is stored in out's dtype before the rest folds in, and the running
        ///     value round-trips through it at every buffer fill (<c>np.max(int16 [300, 100], out=int8)</c> is 100).</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions, so the result broadcasts against <paramref name="a"/>.</param>
        /// <param name="initial">The value every output starts from, converted into the loop dtype the way NumPy's element
        ///     setter converts a Python value: a C# integer is a Python int (range-checked, never wrapped), a C#
        ///     <c>double</c> / <c>float</c> a Python float (truncated into an integer loop), a 0-d <see cref="NDArray"/> is cast.
        ///     It makes an empty reduction valid (the result is the value) and is REQUIRED with <paramref name="where"/>.
        ///     <c>null</c> is NumPy's <c>None</c>: no initial value.</param>
        /// <param name="where">A bool mask broadcast against <paramref name="a"/>: only the selected elements are reduced;
        ///     an output with none selected keeps <paramref name="initial"/>. The C# literals are NumPy's Python bools: <c>true</c>
        ///     (which converts to a 0-d array) is NumPy's default <c>where=True</c>, no mask at all, and <c>false</c> selects
        ///     nothing (it needs <paramref name="initial"/>). Footgun: a 0-d bool ARRAY holding true is read the same way, where
        ///     NumPy treats <c>np.array(True)</c> as a mask that needs <paramref name="initial"/> — C# cannot tell the two apart
        ///     (with <paramref name="initial"/> both give NumPy's result). <c>null</c> is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array of <paramref name="a"/>'s dtype — 0-d (NumPy's
        ///     read-only scalar) for a flat reduction without <paramref name="keepdims"/>.</returns>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>; a mask with more axes than
        ///     <paramref name="a"/>; <paramref name="where"/> without <paramref name="initial"/> ("reduction operation
        ///     'maximum' does not have an identity, so to use a where mask one has to specify 'initial'"); an empty reduced
        ///     axis without <paramref name="initial"/> ("zero-size array to reduction operation maximum which has no
        ///     identity"); an unconvertible <paramref name="initial"/> (a float NaN into an integer loop, a sequence).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast
        ///     against <paramref name="a"/> (NumPy's "remapped shapes" text).</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop ("Python integer
        ///     1000 out of bounds for int8").</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.max.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray max(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => max(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.ReduceAMax(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.Max);

        /// <summary>
        ///     <c>np.max(a, axis, keepdims)</c> with keepdims in the third position — NumSharp's earlier positional form (NumPy's
        ///     third parameter is <c>out</c>; see the file header for why this overload stays).
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The maximum (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis (NumPy's "zero-size array" text).</exception>
        [OverloadResolutionPriority(1)]
        public static NDArray max(NDArray a, int? axis, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => max(s.a, s.axis, s.keepdims), axis, keepdims, "max")
            : a.TensorEngine.ReduceAMax(a, axis, keepdims);

        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis — NumPy's <c>np.amax</c>, an alias of
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> with the same parameters.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (see <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>).</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>; requires <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amax.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray amax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => amax(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : max(a, axis, @out, keepdims, initial, @where);

        /// <summary><c>np.amax(a, axis, keepdims)</c> — the positional keepdims form (see <see cref="max(NDArray, int?, bool)"/>).</summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The maximum.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis.</exception>
        [OverloadResolutionPriority(1)]
        public static NDArray amax(NDArray a, int? axis, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => amax(s.a, s.axis, s.keepdims), axis, keepdims, "max")
            : a.TensorEngine.ReduceAMax(a, axis, keepdims);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis — NumPy's <c>np.min</c>, i.e.
        ///     <c>np.minimum.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>. The keywords
        ///     behave as in <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>, with the minimum.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (any dtype; a narrowing one is lossy, see
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>).</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('minimum' in the texts).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.min.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray min(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => min(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.ReduceAMin(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.Min);

        /// <summary><c>np.min(a, axis, keepdims)</c> — the positional keepdims form (see <see cref="max(NDArray, int?, bool)"/>).</summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The minimum.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis.</exception>
        [OverloadResolutionPriority(1)]
        public static NDArray min(NDArray a, int? axis, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => min(s.a, s.axis, s.keepdims), axis, keepdims, "min")
            : a.TensorEngine.ReduceAMin(a, axis, keepdims);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis — NumPy's <c>np.amin</c>, an alias of
        ///     <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/>.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>; requires <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">See <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amin.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray amin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => amin(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : min(a, axis, @out, keepdims, initial, @where);

        /// <summary><c>np.amin(a, axis, keepdims)</c> — the positional keepdims form (see <see cref="max(NDArray, int?, bool)"/>).</summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <returns>The minimum.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="System.ArgumentException">An empty reduced axis.</exception>
        [OverloadResolutionPriority(1)]
        public static NDArray amin(NDArray a, int? axis, bool keepdims)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, keepdims), static s => amin(s.a, s.axis, s.keepdims), axis, keepdims, "min")
            : a.TensorEngine.ReduceAMin(a, axis, keepdims);

        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis, ignoring NaNs — NumPy's <c>np.nanmax</c>, which for
        ///     an ndarray IS <c>np.fmax.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>. The
        ///     keywords behave as in <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>; the texts name 'fmax'
        ///     (for every dtype, an integer array included).
        /// </summary>
        /// <param name="a">Array containing numbers whose maximum is desired.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.
        ///     An output whose every selected element is NaN returns it (fmax skips NaN).</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).
        ///     An all-NaN slice without an initial value is NaN (NumPy's "All-NaN slice encountered" RuntimeWarning is not
        ///     modeled).</returns>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmax' in the texts).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray nanmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => nanmax(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.NanMax(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.FMax);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis, ignoring NaNs — NumPy's <c>np.nanmin</c>
        ///     (<c>np.fmin.reduce</c>); the keywords behave as in <see cref="nanmax(NDArray, int?, NDArray, bool, object, NDArray)"/>.
        /// </summary>
        /// <param name="a">Array containing numbers whose minimum is desired.</param>
        /// <param name="axis">The axis to reduce, or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmin' in the texts).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="TypeError"><paramref name="initial"/> is complex for a real loop, or of an unconvertible type.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html</remarks>
        [OverloadResolutionPriority(1)]
        public static NDArray nanmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => nanmin(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.NanMin(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.FMin);
    }
}
