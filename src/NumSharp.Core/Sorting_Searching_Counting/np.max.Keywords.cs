using System;

namespace NumSharp
{
    // =============================================================================
    // np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin: ONE overload each, NumPy's whole signature
    //     (a, axis=None, out=None, keepdims=<no value>, initial=<no value>, where=<no value>)
    // (plan docs/plans/api-parameter-parity.md, items A4 and A4d). The engine members are TensorEngine.ReduceAMax /
    // ReduceAMin / NanMax / NanMin, which take out= / initial= / where= (np-layer dtype plan, phase PE, E5); DefaultEngine's
    // keyword path is MaxMinNumPy (Default.Reduction.MinMax.Keywords.cs), whose header has NumPy's statement order and the
    // two engines.
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
    // A keyword-less call (no out, initial or where) takes the engine members' long-standing route, unchanged; any keyword
    // takes their keyword path (MaxMinNumPy). The facades make one engine call either way.
    //
    // A numpy.matrix operand follows the matrix METHOD, as in NumPy (np-layer NL13): np.max / np.amax / np.min / np.amin
    // reach matrix.max(axis, out) / matrix.min(axis, out) through _wrapreduction with every keyword the caller passed, so
    // keepdims= / initial= / where= are the method's TypeError and an out= gets the method's answer (MatrixExtreme).
    // np.nanmax / np.nanmin take NumPy's branch for an operand that is not a base ndarray: _replace_nan, then np.amax /
    // np.amin of the copy (the same method rules), then NaN written where a whole slice was NaN (MatrixNanExtreme).
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
        ///     read-only scalar) for a flat reduction without <paramref name="keepdims"/>. A <c>numpy.matrix</c> operand gets the
        ///     matrix method's answer: the scalar without an axis, else a row <c>(1, N)</c> or a column <c>(N, 1)</c>; with a
        ///     matrix <paramref name="out"/>, that out along an axis and a copy of its element without one.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar converted to a 0-d array (a bool, number, string,
        ///     <see cref="System.Half"/> or <c>char</c> literal, as in <c>np.max(a, 1, true)</c>): NumPy's "return arrays must be
        ///     of ArrayType", raised before any other argument is looked at. Also: <paramref name="initial"/> is complex for a
        ///     real loop or of an unconvertible type; a <c>numpy.matrix</c> operand with <c>keepdims: true</c>,
        ///     <paramref name="initial"/> or <paramref name="where"/> (NumPy calls the matrix method, which has none of them; the
        ///     first one passed, in that order, is <c>matrix.max() got an unexpected keyword argument 'initial'</c>), raised
        ///     before anything is computed.</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain (non-matrix) <paramref name="out"/>:
        ///     NumPy's matrix method stores the maxima into it, then fails on <c>out._collapse</c> (<c>'numpy.ndarray' object has
        ///     no attribute '_collapse'</c>), so the out holds the values when this is raised.</exception>
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
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.max.html. There is no <c>dtype=</c>: NumPy's np.max has
        ///     none. A flat result is a 0-d array; read it as a C# value with a cast (<c>(double)np.max(a)</c>).
        /// </remarks>
        public static NDArray max(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: matrix.max(axis, out), the method NumPy hands it to
            ? MatrixExtreme(a, axis, @out, keepdims, initial, @where, isMax: true)
            // The whole call is the engine's: TensorEngine.ReduceAMax of a's engine takes the keywords (a null a is
            // refused above).
            : a.TensorEngine.ReduceAMax(a, axis, keepdims, null, @out, initial, @where);
        }

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
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims). A
        ///     <c>numpy.matrix</c> operand gets the matrix method's answer (see
        ///     <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>, <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.max() got an
        ///     unexpected keyword argument ...</c>).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain <paramref name="out"/>, after the values
        ///     were stored (<c>'numpy.ndarray' object has no attribute '_collapse'</c>).</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amax.html</remarks>
        public static NDArray amax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: matrix.max(axis, out), the method NumPy hands it to
            ? MatrixExtreme(a, axis, @out, keepdims, initial, @where, isMax: true)
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
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims). A
        ///     <c>numpy.matrix</c> operand gets the matrix method's answer: the scalar without an axis, else a row or a column;
        ///     with a matrix <paramref name="out"/>, that out along an axis and a copy of its element without one.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>, <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.min() got an
        ///     unexpected keyword argument ...</c>, the first one passed in that order).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain <paramref name="out"/>, after the values
        ///     were stored (<c>'numpy.ndarray' object has no attribute '_collapse'</c>).</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('minimum' in the texts).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.min.html</remarks>
        public static NDArray min(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: matrix.min(axis, out), the method NumPy hands it to
            ? MatrixExtreme(a, axis, @out, keepdims, initial, @where, isMax: false)
            // The whole call is the engine's: TensorEngine.ReduceAMin of a's engine takes the keywords (a null a is
            // refused above).
            : a.TensorEngine.ReduceAMin(a, axis, keepdims, null, @out, initial, @where);
        }

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
        /// <returns><paramref name="out"/> itself when given; else a fresh array (0-d for a flat reduction without keepdims). A
        ///     <c>numpy.matrix</c> operand gets the matrix method's answer (see
        ///     <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/>).</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>, <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.min() got an
        ///     unexpected keyword argument ...</c>).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain <paramref name="out"/>, after the values
        ///     were stored (<c>'numpy.ndarray' object has no attribute '_collapse'</c>).</exception>
        /// <exception cref="ValueError">See <see cref="min(NDArray, int?, NDArray, bool, object, NDArray)"/>.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.amin.html</remarks>
        public static NDArray amin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
            => NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: matrix.min(axis, out), the method NumPy hands it to
            ? MatrixExtreme(a, axis, @out, keepdims, initial, @where, isMax: false)
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
        ///     RuntimeWarning is not modeled). A <c>numpy.matrix</c> operand gets NumPy's subclass route (see the remarks): the
        ///     scalar without an axis, else a row or a column, an all-NaN slice as NumPy's own <c>np.nan</c>; with a matrix
        ///     <paramref name="out"/>, that out along an axis and the scalar without one.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>, <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.max() got an
        ///     unexpected keyword argument ...</c>: NumPy's route ends in the matrix method, which has none of them).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain <paramref name="out"/>: the method stores
        ///     the maxima of the NaN-replaced copy into it (an all-NaN slice as <c>-inf</c>), then fails on
        ///     <c>out._collapse</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmax' in the
        ///     texts); an empty reduction without <paramref name="initial"/> raises (<c>fmax</c> has no identity; 'maximum' for a
        ///     <c>numpy.matrix</c> operand, whose route reduces with the method) — it does not return NaN.</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
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
        ///     <para>
        ///     A <c>numpy.matrix</c> operand is not a base ndarray, so NumPy takes nanmax's "slow, but safe for subclasses"
        ///     branch: the NaNs of a copy become <c>-inf</c>, <c>np.amax</c> reduces it (the matrix method), and a slice that was
        ///     all NaN gets <c>np.nan</c> written back. Footgun: those bits are not fmax's — a <c>±0</c> tie follows
        ///     <c>maximum</c> (<c>np.nanmax(np.matrix([[0.0, -0.0]]))</c> is <c>-0.0</c>, where the plain array's is
        ///     <c>0.0</c>), an all-NaN slice is NumPy's positive quiet NaN whatever NaN it held, and a complex one is
        ///     <c>nan+0j</c>; a matrix out of a flat call keeps the <c>-inf</c> (only the returned scalar is NaN).
        ///     </para>
        /// </remarks>
        public static NDArray nanmax(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: NumPy's subclass branch, ending in matrix.max
            ? MatrixNanExtreme(a, axis, @out, keepdims, initial, @where, isMax: true)
            // The whole call is the engine's: TensorEngine.NanMax of a's engine takes the keywords (a null a is
            // refused above).
            : a.TensorEngine.NanMax(a, axis, keepdims, @out, initial, @where);
        }

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
        ///     reduction without keepdims). An all-NaN slice without an initial value is NaN. A <c>numpy.matrix</c> operand gets
        ///     NumPy's subclass route, as in <see cref="nanmax(NDArray, int?, NDArray, bool, object, NDArray)"/> with
        ///     <c>+inf</c> and <c>np.amin</c>.</returns>
        /// <exception cref="TypeError"><paramref name="out"/> is a C# scalar ("return arrays must be of ArrayType");
        ///     <paramref name="initial"/> is complex for a real loop or of an unconvertible type; a <c>numpy.matrix</c> operand
        ///     with <c>keepdims: true</c>, <paramref name="initial"/> or <paramref name="where"/> (<c>matrix.min() got an
        ///     unexpected keyword argument ...</c>).</exception>
        /// <exception cref="AttributeError">A <c>numpy.matrix</c> operand with a plain <paramref name="out"/>: the method stores
        ///     the minima of the NaN-replaced copy into it (an all-NaN slice as <c>+inf</c>), then fails on
        ///     <c>out._collapse</c>.</exception>
        /// <exception cref="ValueError">See <see cref="max(NDArray, int?, NDArray, bool, object, NDArray)"/> ('fmin' in the
        ///     texts); an empty reduction without <paramref name="initial"/> raises (<c>fmin</c> has no identity; 'minimum' for
        ///     a <c>numpy.matrix</c> operand).</exception>
        /// <exception cref="System.ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="where"/> or <paramref name="out"/> does not broadcast.</exception>
        /// <exception cref="System.OverflowException"><paramref name="initial"/> does not fit an integer loop.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
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
        ///     <para>
        ///     A <c>numpy.matrix</c> operand takes NumPy's subclass branch (NaNs of a copy become <c>+inf</c>, <c>np.amin</c>
        ///     reduces it, an all-NaN slice gets <c>np.nan</c>): a <c>±0</c> tie follows <c>minimum</c>, not fmin
        ///     (<c>np.nanmin(np.matrix([[-0.0, 0.0]]))</c> is <c>0.0</c>, where the plain array's is <c>-0.0</c>), and an all-NaN
        ///     slice is NumPy's positive quiet NaN (complex: <c>nan+0j</c>).
        ///     </para>
        /// </remarks>
        public static NDArray nanmin(NDArray a, int? axis = null, NDArray @out = null, bool keepdims = false, object initial = null,
            NDArray @where = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            return NDMatrixBoundary.Sees(a)   // a numpy.matrix operand: NumPy's subclass branch, ending in matrix.min
            ? MatrixNanExtreme(a, axis, @out, keepdims, initial, @where, isMax: false)
            // The whole call is the engine's: TensorEngine.NanMin of a's engine takes the keywords (a null a is
            // refused above).
            : a.TensorEngine.NanMin(a, axis, keepdims, @out, initial, @where);
        }

        /// <summary>
        ///     NumPy's np.max / np.amax / np.min / np.amin for a <c>numpy.matrix</c> operand: <c>_wrapreduction</c> finds the
        ///     operand's own <c>max</c> / <c>min</c> and calls <c>matrix.max(axis=axis, out=out, **passkwargs)</c>, which is
        ///     <c>ndarray.max(self, axis, out, keepdims=True)._collapse(axis)</c>. Use it only for an operand
        ///     <see cref="NDMatrixBoundary.Sees"/> as a matrix; a plain array never comes here.
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axis, or <c>null</c> for the scalar of every element.</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is the method's TypeError.</param>
        /// <param name="initial">The caller's initial; any value is the method's TypeError.</param>
        /// <param name="where">The caller's mask; any mask is the method's TypeError.</param>
        /// <param name="isMax"><c>true</c> for <c>matrix.max</c> (np.max / np.amax), <c>false</c> for <c>matrix.min</c>.</param>
        /// <returns>Without <paramref name="out"/>: the 0-d scalar for a <c>null</c> axis, else a row <c>(1, N)</c> or a
        ///     column <c>(N, 1)</c> matrix. With a matrix <paramref name="out"/>: that out along an axis, a 0-d copy of its
        ///     element without one.</returns>
        /// <exception cref="TypeError"><paramref name="keepdims"/> is true, or <paramref name="initial"/> or
        ///     <paramref name="where"/> is given (<c>matrix.max() got an unexpected keyword argument 'initial'</c>), the first
        ///     in that order, before anything is computed; or <paramref name="out"/> is a C# scalar (<c>return arrays must be of
        ///     ArrayType</c>).</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array: the values are stored into it first, as
        ///     NumPy's method stores them before <c>out._collapse</c> fails.</exception>
        /// <exception cref="ValueError">An empty reduction (<c>zero-size array to reduction operation maximum which has no
        ///     identity</c>), or an out that is read-only or of the wrong rank (the reduction's own texts).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <remarks>C# cannot tell an omitted keepdims from <c>keepdims: false</c>, which NumPy refuses too when it is
        ///     passed; only <c>keepdims: true</c> raises here (as for <c>np.sum</c>).</remarks>
        private static NDArray MatrixExtreme(NDArray a, int? axis, NDArray @out, bool keepdims, object initial, NDArray @where,
            bool isMax)
        {
            string method = isMax ? "max" : "min";
            // _wrapreduction passes the keywords the caller gave in amax's signature order (keepdims, initial, where), and
            // the method takes none of them: the first one passed is its TypeError, before anything else is looked at.
            if (keepdims)
                throw NDMatrixBoundary.KeywordError(method, "keepdims");
            if (initial is not null)
                throw NDMatrixBoundary.KeywordError(method, "initial");
            if (@where is not null)
                throw NDMatrixBoundary.KeywordError(method, "where");

            // matrix.max(axis, out): a matrix out gets the member's answer, a plain one the values and then the method's
            // AttributeError (ReducedInto); the plain-out reduction is the engine's keepdims=True call of the method body.
            if (@out is not null)
                return isMax
                    ? NDMatrixBoundary.ReducedInto(a, @out, axis,
                        static (m, o, ax) => m.max(ax, o),
                        static (m, o, ax) => m.TensorEngine.ReduceAMax(m, ax, keepdims: true, null, o))
                    : NDMatrixBoundary.ReducedInto(a, @out, axis,
                        static (m, o, ax) => m.min(ax, o),
                        static (m, o, ax) => m.TensorEngine.ReduceAMin(m, ax, keepdims: true, null, o));

            // a row, a column or the scalar, as NumPy's matrix method returns
            return isMax
                ? NDMatrixBoundary.Reduced((a, axis), static s => max(s.a, s.axis), axis, false, "max")
                : NDMatrixBoundary.Reduced((a, axis), static s => min(s.a, s.axis), axis, false, "min");
        }

        /// <summary>
        ///     NumPy's np.nanmax / np.nanmin for a <c>numpy.matrix</c> operand, its branch for anything that is not a base
        ///     ndarray (numpy/lib/_nanfunctions_impl.py), statement for statement:
        ///     <code>
        ///     a, mask = _replace_nan(a, -np.inf)          # +np.inf for nanmin
        ///     res = np.amax(a, axis=axis, out=out, **kwargs)
        ///     if mask is None:
        ///         return res
        ///     kwargs.pop("initial", None)
        ///     mask = np.all(mask, axis=axis, **kwargs)
        ///     if np.any(mask):
        ///         res = _copyto(res, np.nan, mask)
        ///     return res
        ///     </code>
        ///     Use it only for an operand <see cref="NDMatrixBoundary.Sees"/> as a matrix: it is NumPy's slow route, a copy
        ///     and two mask reductions per call, which a plain array never takes (its nanmax is <c>fmax.reduce</c>).
        /// </summary>
        /// <param name="a">The matrix operand.</param>
        /// <param name="axis">The axis, or <c>null</c> for the scalar of every element.</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims; <c>true</c> is the method's TypeError.</param>
        /// <param name="initial">The caller's initial; any value is the method's TypeError.</param>
        /// <param name="where">The caller's mask; any mask is the method's TypeError.</param>
        /// <param name="isMax"><c>true</c> for np.nanmax (<c>-inf</c>, <c>np.amax</c>), <c>false</c> for np.nanmin
        ///     (<c>+inf</c>, <c>np.amin</c>).</param>
        /// <returns>What <see cref="MatrixExtreme"/> returns for the NaN-replaced copy, with every all-NaN slice set to NumPy's
        ///     <c>np.nan</c>: a new NaN scalar of the operand's dtype for a <c>null</c> axis (a matrix out keeps the
        ///     <c>∓inf</c> then, as NumPy's does), else NaN written into the row, the column or the matrix out.</returns>
        /// <exception cref="TypeError">See <see cref="MatrixExtreme"/>.</exception>
        /// <exception cref="AttributeError"><paramref name="out"/> is a plain array, after it received the replaced copy's
        ///     extremes (<c>∓inf</c> for an all-NaN slice).</exception>
        /// <exception cref="ValueError">An empty reduction, named after <c>maximum</c> / <c>minimum</c> (the method's
        ///     reduction), or a read-only or misshapen out.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        [NDScoped]
        private static NDArray MatrixNanExtreme(NDArray a, int? axis, NDArray @out, bool keepdims, object initial,
            NDArray @where, bool isMax)
        {
            // The steps below compute plain and build the matrix answer here, where NumPy's statements pass matrices from
            // call to call: the boundary's contract is that no library method reaches a public function's matrix rule at
            // depth zero (NDMatrixBoundary's caller audit), so every np.* call runs inside this region.
            using var plain = NDMatrixBoundary.Inside();

            // a, mask = _replace_nan(a, -inf / +inf): the array object's statement (NDArray.ReplaceNaN) makes the mask, the
            // K-order copy and the replacement for an inexact dtype - the dtype decision lives there, below the facade
            // (ratchet R1). Without a mask nanmax IS np.amax of the operand itself.
            NDArray replaced = NDArray.ReplaceNaN(a, isMax ? double.NegativeInfinity : double.PositiveInfinity, out NDArray mask);
            if (mask is null)
                return MatrixExtreme(a, axis, @out, keepdims, initial, @where, isMax);

            // res = np.amax(a, axis=axis, out=out, **kwargs): the matrix method's rules, its TypeErrors and its
            // AttributeError for a plain out (raised before the all-NaN check below, so such an out keeps -inf / +inf). The
            // copy reaches them as a matrix view, as NumPy's subok copy is a matrix.
            NDArray res = MatrixExtreme(NDMatrix.FromWrapped(replaced), axis, @out, keepdims, initial, @where, isMax);

            // mask = np.all(mask, axis=axis): matrix.all is ndarray.all(keepdims=True)._collapse(axis), the scalar without
            // an axis and the kept-axis array along one, which lines up with the row / column result or the matrix out (no
            // keywords are left: any of them raised above). np.any(mask) asks whether any slice was all NaN.
            NDArray allNaN = axis is null ? all(mask) : all(mask, axis, keepdims: true);
            NDArray anyAllNaN = any(allNaN);
            if (!(bool)anyAllNaN)
                return res;

            // _copyto(res, np.nan, mask). np.nan is Python's positive quiet NaN (C#'s double.NaN carries the sign bit), cast
            // to the result's dtype: 0x7FF8000000000000, 0x7FC00000 for float32, 0x7E00 for float16, nan+0j for complex.
            double positiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000);
            if (axis is null)
                // res is NumPy's scalar (matrix._collapse(None)), not an ndarray: `a = a.dtype.type(val)` makes a NEW scalar,
                // and a matrix out keeps the -inf / +inf the method stored.
                return NDArray.Scalar(positiveNaN, res.dtype).MarkReductionScalar();

            // An ndarray result (the row / column, or the matrix out itself): np.copyto(res, np.nan, where=mask,
            // casting='unsafe') sets the all-NaN slices in place.
            copyto(res, positiveNaN, "unsafe", allNaN);
            return res;
        }
    }
}
