using NumSharp.Backends;

namespace NumSharp
{
    // =============================================================================
    // np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin: ONE overload each, NumPy's whole signature
    //     (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    // (plan docs/plans/api-parameter-parity.md, items A4 and A4d). The engine is DefaultEngine.MaxMinNumPy
    // (Default.Reduction.MinMax.Keywords.cs); its header has NumPy's statement order and the two engines.
    //
    // ONE OVERLOAD PER FUNCTION, NOTHING ELSE (the owner's rule, 2026-10-17: "one whole api per function as numpy provides,
    // remove others"). NumSharp's earlier forms are gone:
    //   * the generic np.amax<T>(a) / np.amin<T>(a) — not NumPy, and they cast through the dtype extension below;
    //   * the dtype= extension (a, axis, keepdims, dtype) — NumPy's np.max has no dtype (`np.max(a, dtype=...)` is a
    //     TypeError there), and the extension returned silently wrong values for a cast it could not do (int64 [7, 1, 5]
    //     with dtype float64 gave NaN, with int32 -1). NumPy's own spelling of a typed reduction is
    //     np.maximum.reduce(a, axis, dtype=...), which the engine keeps (TensorEngine.ReduceAMax / ReduceAMin);
    //   * the positional (a, axis, keepdims) shim. It resolved only through [OverloadResolutionPriority], which C# 12
    //     ignores, so a default net8.0 consumer got CS0121 for plain calls like np.min(a, 0) and np.amin(a, axis: 0).
    //
    // A positional third argument is `out`, as in NumPy: np.max(a, 1, true) binds the bool literal to out, and the reduce
    // prologue refuses it with NumPy's own TypeError "return arrays must be of ArrayType" (ReductionArguments.
    // ValidateParsedArguments reads the mark the implicit scalar conversions put on their 0-d array). Write keepdims by
    // name: np.max(a, 1, keepdims: true) — as NumPy callers must too.
    //
    // A keyword-less call (no out, initial or where) takes the engine's long-standing route, ReduceAMax / ReduceAMin /
    // NanMax / NanMin, unchanged; any keyword goes to MaxMinNumPy.
    // =============================================================================

    public static partial class np
    {
        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis — NumPy's <c>np.max(a, axis=None, out=None,
        ///     keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e.
        ///     <c>np.maximum.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>. This is the only
        ///     overload: NumPy's whole signature, with NumPy's names, order and defaults.
        /// </summary>
        /// <param name="a">Input data. A NaN propagates (use <see cref="nanmax"/> to skip NaNs).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end); <c>null</c> reduces every axis.</param>
        /// <param name="out">The array the result is written into and returned. It must have the result's shape (the
        ///     keepdims shape with <paramref name="keepdims"/>); its dtype may differ — the loop runs in the promotion of
        ///     (<paramref name="out"/>, <paramref name="a"/>), and the result is cast into it unsafely (a float64 maximum lands
        ///     in a bool out as <c>True</c>). Footgun: a narrowing out is lossy DURING the reduction too, as in NumPy — each
        ///     output's first element (or the initial value) is stored in out's dtype before the rest folds in, and the running
        ///     value round-trips through it at every buffer fill (<c>np.max(int16 [300, 100], out=int8)</c> is 100). Footgun:
        ///     this is the THIRD positional parameter, as in NumPy, so <c>np.max(a, 1, true)</c> passes the bool as out and
        ///     raises; write <c>keepdims: true</c> by name.</param>
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
        ///     NumPy treats <c>np.array(True)</c> as a mask that needs <paramref name="initial"/> (with <paramref name="initial"/>
        ///     both give NumPy's result). <c>null</c> is no mask.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array of <paramref name="a"/>'s dtype — 0-d (NumPy's
        ///     read-only scalar) for a flat reduction without <paramref name="keepdims"/>.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar converted to a 0-d array (a bool, number, string,
        ///     <see cref="System.Half"/> or <c>char</c> literal, as in <c>np.max(a, 1, true)</c>): NumPy's "return arrays must be
        ///     of ArrayType", raised before any other argument is looked at. Also: <paramref name="initial"/> is complex for a
        ///     real loop or of an unconvertible type; a <c>numpy.matrix</c> operand with <c>keepdims: true</c> (NumPy calls the
        ///     matrix method, which has no such keyword; the text names that method, as in <c>matrix.sum() got an unexpected
        ///     keyword argument 'keepdims'</c> for <c>np.sum</c>).</exception>
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
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.max.html. There is no <c>dtype=</c>: NumPy's np.max has
        ///     none. A flat result is a 0-d array; read it as a C# value with a cast (<c>(double)np.max(a)</c>).
        /// </remarks>
        public static NDArray max(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => max(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.ReduceAMax(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.Max);

        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis — NumPy's <c>np.amax</c>, an alias of
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> with the same signature (NumPy's
        ///     <c>(a, axis=None, out=None, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>).
        ///     This is the only overload.
        /// </summary>
        /// <param name="a">Input data. A NaN propagates.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (any dtype; a narrowing one is lossy, see
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>). Footgun: the third positional parameter, so
        ///     <c>np.amax(a, 1, true)</c> raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>; requires <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amax.html</remarks>
        public static NDArray amax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => amax(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : max(a, axis, @out, keepdims, initial, @where);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis — NumPy's <c>np.min(a, axis=None, out=None,
        ///     keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e.
        ///     <c>np.minimum.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>. The keywords
        ///     behave as in <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>, with the minimum. This is the
        ///     only overload.
        /// </summary>
        /// <param name="a">Input data. A NaN propagates (use <see cref="nanmin"/> to skip NaNs).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned (any dtype; a narrowing one is lossy, see
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>). Footgun: the third positional parameter, so
        ///     <c>np.min(a, 1, true)</c> raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('minimum' in the texts).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.min.html</remarks>
        public static NDArray min(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => min(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.ReduceAMin(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.Min);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis — NumPy's <c>np.amin</c>, an alias of
        ///     <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/> with the same signature (NumPy's
        ///     <c>(a, axis=None, out=None, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>).
        ///     This is the only overload.
        /// </summary>
        /// <param name="a">Input data. A NaN propagates.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned. Footgun: the third positional parameter, so
        ///     <c>np.amin(a, 1, true)</c> raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>; requires <paramref name="initial"/>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>.</exception>
        /// <exception cref="ValueError">See <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amin.html</remarks>
        public static NDArray amin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => amin(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : min(a, axis, @out, keepdims, initial, @where);

        /// <summary>
        ///     Return the maximum of an array or the maximum along an axis, ignoring NaNs — NumPy's <c>np.nanmax(a, axis=None,
        ///     out=None, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, which for an ndarray IS
        ///     <c>np.fmax.reduce(a, axis, out=out, keepdims=keepdims, initial=initial, where=where)</c>, reproduced bit for bit:
        ///     which zero sign survives a <c>±0</c> tie and which NaN an all-NaN slice returns (payload intact, never a canonical
        ///     NaN) are the ones NumPy 2.4.2 returns for the same layout. The keywords behave as in
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>; the texts name 'fmax' (for every dtype, an
        ///     integer array included). This is the only overload.
        /// </summary>
        /// <param name="a">Array containing numbers whose maximum is desired.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned. Footgun: the third positional parameter, so
        ///     <c>np.nanmax(a, 1, true)</c> raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions (for a flat reduction: shape <c>(1,)*a.ndim</c>).</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.
        ///     An output whose every selected element is NaN returns it (fmax skips NaN).</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array of <paramref name="a"/>'s dtype (0-d for a flat
        ///     reduction without keepdims). An all-NaN slice without an initial value is NaN (NumPy's "All-NaN slice encountered"
        ///     RuntimeWarning is not modeled).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmax' in the
        ///     texts); an empty reduction without <paramref name="initial"/> raises (<c>fmax</c> has no identity) — it does not
        ///     return NaN.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>
        ///     <para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html. Per dtype:
        ///     <list type="bullet">
        ///     <item>float32 / float64 run NumPy's own <c>fmax</c> reduction schedule — the vector op <c>maxp</c> inside each
        ///     inner-loop call's vector section, the CRT <c>fmax</c> for its scalar tail — so a <c>±0</c> tie or an all-NaN slice
        ///     can come back differently for two layouts of the same values, exactly as in NumPy.</item>
        ///     <item>float16 and complex128 run NumPy's sequential loops: NaN elements are skipped (complex: a NaN in either
        ///     part), a tie keeps the EARLIER element, an all-NaN slice returns its first NaN verbatim; complex compares
        ///     lexicographically (real part, then imaginary).</item>
        ///     <item>Integer, bool and char arrays cannot hold a NaN, so this is <see cref="amax(NDArray, int?, NDArray, bool, object, NDArray)"/>;
        ///     decimal likewise (no NumPy analog).</item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     A non-empty reduced axis of an otherwise empty array returns the empty result.
        ///     </para>
        /// </remarks>
        public static NDArray nanmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => nanmax(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "max")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.NanMax(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.FMax);

        /// <summary>
        ///     Return the minimum of an array or the minimum along an axis, ignoring NaNs — NumPy's <c>np.nanmin(a, axis=None,
        ///     out=None, keepdims=&lt;no value&gt;, initial=&lt;no value&gt;, where=&lt;no value&gt;)</c>, i.e.
        ///     <c>np.fmin.reduce</c> for an ndarray, reproduced bit for bit; the keywords behave as in
        ///     <see cref="nanmax(NDArray, int?, NDArray, bool, object, NDArray)"/>. This is the only overload.
        /// </summary>
        /// <param name="a">Array containing numbers whose minimum is desired.</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or <c>null</c> for every axis.</param>
        /// <param name="out">The array the result is written into and returned. Footgun: the third positional parameter, so
        ///     <c>np.nanmin(a, 1, true)</c> raises; write <c>keepdims: true</c> by name.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions (for a flat reduction: shape <c>(1,)*a.ndim</c>).</param>
        /// <param name="initial">The value every output starts from, or <c>null</c>; required with <paramref name="where"/>.</param>
        /// <param name="where">A bool mask of the elements to reduce, or <c>null</c>.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array of <paramref name="a"/>'s dtype (0-d for a flat
        ///     reduction without keepdims). An all-NaN slice without an initial value is NaN.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmin' in the
        ///     texts); an empty reduction without <paramref name="initial"/> raises (<c>fmin</c> has no identity).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>
        ///     <para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html. Per dtype:
        ///     <list type="bullet">
        ///     <item>float32 / float64 run NumPy's own <c>fmin</c> reduction schedule — the vector op <c>minp</c> inside each
        ///     inner-loop call's vector section, the CRT <c>fmin</c> for its scalar tail (whose tie keeps <c>-0</c> over
        ///     <c>+0</c>) — so a <c>±0</c> tie or an all-NaN slice can come back differently for two layouts of the same
        ///     values, exactly as in NumPy.</item>
        ///     <item>float16 and complex128 run NumPy's sequential loops: NaN elements are skipped (complex: a NaN in either
        ///     part), a tie keeps the EARLIER element, an all-NaN slice returns its first NaN verbatim.</item>
        ///     <item>Integer, bool and char arrays cannot hold a NaN, so this is <see cref="amin(NDArray, int?, NDArray, bool, object, NDArray)"/>;
        ///     decimal likewise (no NumPy analog).</item>
        ///     </list>
        ///     </para>
        ///     <para>
        ///     A non-empty reduced axis of an otherwise empty array returns the empty result.
        ///     </para>
        /// </remarks>
        public static NDArray nanmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            ? NDMatrixBoundary.Reduced((a, axis, @out, keepdims, initial, @where), static s => nanmin(s.a, s.axis, s.@out, s.keepdims, s.initial, s.@where), axis, keepdims, "min")
            : @out is null && initial is null && @where is null
                ? a.TensorEngine.NanMin(a, axis, keepdims)
                : DefaultEngine.MaxMinNumPy(a, axis, @out, keepdims, initial, @where, MinMaxOp.FMin);
    }
}
