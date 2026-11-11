using System;

// =============================================================================
// NDSubclassRules - the NumPy answers no shared mechanism expresses
// =============================================================================
//
// Most public functions answer for a numpy.matrix / numpy.char.chararray operand by one of NumPy's shared mechanisms
// (NDSubclassAttributes.cs). The functions here end in an answer that depends on their ARGUMENTS (np.copy's subok,
// np.linalg.qr's mode, np.diag's rank) or on several operands at once (np.append's pair, np.frexp's two outputs,
// np.meshgrid's per-input grids), as NumPy's Python bodies do. Each is a [SubclassRule] declared on the computation the
// public function calls; NumSharp.Build binds every parameter after the first by NAME to the public function's argument
// (the CALLER's object: the matrix, the chararray), and the first receives the result computed on the plain views.
//
// A rule never computes: it decides which of NDSubclass's mechanisms applies, or rebuilds a result struct with them.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The <see cref="SubclassRuleAttribute"/> rules of the public functions whose NumPy answer for an array subclass
    ///     operand depends on their arguments: each picks the mechanism NumPy's body reaches for those arguments.
    /// </summary>
    /// <remarks>Called only by the subclass path NumSharp.Build weaves into the public functions, with the computation's
    ///     result on the plain views first and the caller's own arguments after it.</remarks>
    internal static class NDSubclassRules
    {
        #region Creation

        /// <summary>
        ///     <c>np.copy(a, order, subok)</c>: NumPy's <c>array(a, order=order, subok=subok, copy=True)</c> - created as the
        ///     operand's class only with <c>subok=True</c>; the default <c>subok=False</c> copy is a plain array.
        /// </summary>
        /// <param name="result">The copy computed on the plain view.</param>
        /// <param name="a">The caller's operand.</param>
        /// <param name="subok">The caller's <c>subok</c>.</param>
        /// <returns>The copy as <paramref name="a"/>'s class (subok), or the plain copy.</returns>
        internal static NDArray Copy(NDArray result, NDArray a, bool subok)
            => subok ? NDSubclass.NewSubtype(result, null, a, keepRank: false, NewSubtypeZeroDim.Array) : NDSubclass.AsArray(result);

        /// <summary>
        ///     <c>np.require(a, dtype, requirements)</c>: NumPy's <c>array(a, ..., subok=subok)</c> keeps the operand's class
        ///     unless the requirements hold <c>'E'</c> / <c>'ENSUREARRAY'</c> (any case), which sets <c>subok=False</c>; no
        ///     requirement at all is <c>asanyarray</c>, which keeps it too.
        /// </summary>
        /// <param name="result">The array computed on the plain view (the operand itself, a view or a copy).</param>
        /// <param name="a">The caller's operand.</param>
        /// <param name="requirements">The caller's requirements: a <c>string[]</c> of flag names, a <c>string</c> whose
        ///     characters are the flags, or <c>null</c>.</param>
        /// <returns>A plain array for ENSUREARRAY, otherwise the result as <paramref name="a"/>'s class.</returns>
        /// <exception cref="ValueError">A chararray result whose dtype is not string data.</exception>
        internal static NDArray Require(NDArray result, NDArray a, object requirements)
            => EnsuresArray(requirements)
                ? NDSubclass.AsArray(result)
                : NDSubclass.NewSubtype(result, null, a, keepRank: false, NewSubtypeZeroDim.Array);

        /// <summary>Whether <c>np.require</c>'s requirements ask for a base-class array (<c>'E'</c> / <c>'ENSUREARRAY'</c>).</summary>
        /// <param name="requirements">A <c>string[]</c> of names, a <c>string</c> of one-letter flags, or anything else (none).</param>
        /// <returns>True when ENSUREARRAY is requested.</returns>
        /// <remarks>An unknown flag never reaches here: the computation rejects it with NumPy's KeyError first.</remarks>
        private static bool EnsuresArray(object requirements)
        {
            switch (requirements)
            {
                case string[] names:
                    foreach (string name in names)
                        if (string.Equals(name, "E", StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(name, "ENSUREARRAY", StringComparison.OrdinalIgnoreCase))
                            return true;
                    return false;
                case string letters:
                    // The string overload iterates its characters, as NumPy iterates a str: each one is a flag.
                    return letters.IndexOf('E') >= 0 || letters.IndexOf('e') >= 0;
                default:
                    return false;
            }
        }

        /// <summary>
        ///     <c>np.meshgrid</c>: NumPy builds grid <c>i</c> from <c>np.asanyarray(xi[i])</c> (reshape,
        ///     <c>broadcast_arrays(subok=True)</c>, copy), so each grid keeps ITS input's class: a matrix input gives a matrix
        ///     grid, a chararray input a chararray grid, a plain one a plain grid (<c>np.meshgrid(m[0], x)</c> is a matrix and
        ///     an ndarray).
        /// </summary>
        /// <param name="result">The grids computed on the plain views. The computation already gave a matrix input's grid
        ///     the shape NumPy's reshape steps give it under the matrix finalize rule (<see cref="NDArray"/>'s meshgrid
        ///     statements), so the class's creation here changes no shape.</param>
        /// <param name="xi">The caller's inputs as a sequence, or <c>null</c> for the two- and three-input overloads.</param>
        /// <param name="x1">The first input of a fixed-arity overload, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <returns>The grids, each subclass input's as its class over its grid.</returns>
        /// <exception cref="ValueError">A matrix grid of more than two axes longer than 1 (a matrix input broadcast against
        ///     plain inputs of a higher rank: <c>shape too large to be a matrix.</c>).</exception>
        internal static np.MeshgridResult Meshgrid(np.MeshgridResult result, NDArray[] xi, NDArray x1, NDArray x2, NDArray x3)
        {
            NDArray[] inputs = xi ?? new[] { x1, x2, x3 };
            NDArray[] grids = (NDArray[])result.ToArray().Clone();
            for (int i = 0; i < grids.Length && i < inputs.Length; i++)
                if (NDSubclass.KindOf(inputs[i]) != NDSubclassKind.None)
                    grids[i] = NDSubclass.NewSubtype(grids[i], null, inputs[i], keepRank: false, NewSubtypeZeroDim.Array);
            return new np.MeshgridResult(grids);
        }

        /// <summary>
        ///     <c>np.gradient(f, *varargs, axis, edge_order)</c>: NumPy takes <c>f = np.asanyarray(f)</c> and creates every
        ///     component with <c>out = np.empty_like(f, dtype=otype)</c>, so each one is <paramref name="f"/>'s class (two
        ///     2 x 3 matrices for a 2 x 3 matrix; one for a single axis).
        /// </summary>
        /// <param name="result">The components computed on the plain view.</param>
        /// <param name="f">The caller's operand.</param>
        /// <returns>The components, each created as <paramref name="f"/>'s class; <paramref name="result"/> for a plain
        ///     operand.</returns>
        /// <remarks>The values are the plain computation's: where NumPy's own Python body would reach a matrix's
        ///     <c>*</c> (a non-uniform spacing's coefficient times the matrix slice is <c>np.dot</c> there), NumSharp keeps
        ///     the elementwise product, the rule of the documented gaps MO02 to MO05.</remarks>
        internal static np.GradientResult Gradient(np.GradientResult result, NDArray f)
        {
            if (NDSubclass.KindOf(f) == NDSubclassKind.None)
                return result;
            NDArray[] components = (NDArray[])result.ToArray().Clone();
            for (int i = 0; i < components.Length; i++)
                components[i] = NDSubclass.NewSubtype(components[i], null, f, keepRank: false, NewSubtypeZeroDim.Array);
            return new np.GradientResult(components, result.IsSingle);
        }

        /// <summary>
        ///     <c>np.unstack(x, axis)</c>: NumPy's <c>tuple(moveaxis(x, axis, 0))</c> ITERATES the moved operand, and iteration
        ///     is the subclass's <c>__getitem__</c> - a matrix yields 1 x N row matrices, an N-d chararray chararray views,
        ///     and a 1-d chararray its elements as Python <c>str</c> / <c>bytes</c> with the trailing whitespace stripped.
        /// </summary>
        /// <param name="result">The views computed on the plain view (their axis checks already ran).</param>
        /// <param name="x">The caller's operand.</param>
        /// <returns>The pieces as NumPy's iteration yields them.</returns>
        /// <exception cref="ValueError">A matrix piece with more than two axes longer than 1.</exception>
        internal static NDArray[] Unstack(NDArray[] result, NDArray x)
        {
            switch (NDSubclass.KindOf(x))
            {
                case NDSubclassKind.Matrix:
                    return NDSubclass.NewSubtype(result, null, x, keepRank: false, NewSubtypeZeroDim.Array);
                case NDSubclassKind.CharArray when x.ndim == 1:
                {
                    // The elements, read through the chararray's own indexer (the stripped stand-ins); the plain 0-d views
                    // are left to the subclass path's scope.
                    var chararray = (NDCharArray)x;
                    var values = new NDArray[checked((int)chararray.size)];
                    for (int i = 0; i < values.Length; i++)
                        values[i] = chararray[i];
                    return values;
                }
                case NDSubclassKind.CharArray:
                    return NDSubclass.NewSubtype(result, null, x, keepRank: false, NewSubtypeZeroDim.Array);
                default:
                    return result;
            }
        }

        #endregion

        #region Manipulation

        /// <summary>
        ///     <c>np.atleast_1d(*arys)</c> / <c>np.atleast_2d(*arys)</c>: NumPy's loop takes <c>asanyarray</c> of each array,
        ///     so result <c>i</c> keeps ITS operand's class (the operand itself when it already has the rank).
        /// </summary>
        /// <param name="result">The results computed on the plain views (the operands handed back are the caller's again).</param>
        /// <param name="arys">The caller's operands.</param>
        /// <returns><paramref name="result"/>, each element created as its operand's class.</returns>
        internal static NDArray[] AtLeastEach(NDArray[] result, NDArray[] arys)
            => EachAsItsOperand(result, arys, keepRank: false);

        /// <summary>
        ///     <c>np.atleast_3d(*arys)</c>: as <see cref="AtLeastEach"/>, where a matrix operand's result is
        ///     <c>m[:, :, newaxis]</c> - the 3-d matrix object, its rank kept.
        /// </summary>
        /// <param name="result">The results computed on the plain views.</param>
        /// <param name="arys">The caller's operands.</param>
        /// <returns><paramref name="result"/>, each element created as its operand's class.</returns>
        internal static NDArray[] AtLeast3dEach(NDArray[] result, NDArray[] arys)
            => EachAsItsOperand(result, arys, keepRank: true);

        /// <summary>The per-operand subtype creation of the <c>atleast_*d</c> loops.</summary>
        /// <param name="result">The results (a new array the computation built; its slots are replaced in place).</param>
        /// <param name="operands">The caller's operands, one per result.</param>
        /// <param name="keepRank">Skip the matrix finalize rule (<c>atleast_3d</c>'s new axis).</param>
        /// <returns><paramref name="result"/>.</returns>
        private static NDArray[] EachAsItsOperand(NDArray[] result, NDArray[] operands, bool keepRank)
        {
            if (result is null || operands is null)
                return result;
            for (int i = 0; i < result.Length && i < operands.Length; i++)
                result[i] = NDSubclass.NewSubtype(result[i], null, operands[i], keepRank, NewSubtypeZeroDim.Array);
            return result;
        }

        /// <summary>
        ///     <c>np.append(arr, values, axis)</c>: NumPy ends with <c>concatenate((arr, values), axis)</c>, whose result is
        ///     <c>PyArray_GetSubType</c>'s - a matrix in either place wins, a chararray never does.
        /// </summary>
        /// <param name="result">The concatenation computed on the plain views.</param>
        /// <param name="arr">The caller's array.</param>
        /// <param name="values">The caller's values (<c>null</c> when an overload takes them as a Python object).</param>
        /// <returns>The matrix view of the result, or the result.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray Append(NDArray result, NDArray arr, NDArray values)
            => NDSubclass.GetSubType(result, null, new[] { arr, values });

        /// <summary>
        ///     <c>np.resize(a, new_shape)</c>: NumPy's <c>a = ravel(a)</c> keeps a chararray's class, so an empty source or an
        ///     empty result is <c>np.zeros_like(a, shape=new_shape)</c> - a chararray owning its zeros; any other result is a
        ///     slice of <c>np.concatenate</c>'s plain array. A matrix is plain throughout (<c>ravel</c> converts it).
        /// </summary>
        /// <param name="result">The resized array computed on the plain view.</param>
        /// <param name="a">The caller's operand.</param>
        /// <returns>The chararray of an empty resize, otherwise the plain result.</returns>
        /// <exception cref="ValueError">Never: the zeros keep the chararray's string dtype.</exception>
        internal static NDArray Resize(NDArray result, NDArray a)
            => result is not null && NDSubclass.KindOf(a) == NDSubclassKind.CharArray && (a.size == 0 || result.size == 0)
                ? NDSubclass.NewSubtype(result, null, a, keepRank: false, NewSubtypeZeroDim.Array)
                : NDSubclass.AsArray(result);

        /// <summary>
        ///     <c>np.setdiff1d(ar1, ar2, assume_unique)</c>: NumPy's <c>unique(ar1)</c> keeps a chararray's class and its
        ///     boolean index of it is a chararray view; with <c>assume_unique</c> it converts with <c>np.asarray</c> (plain). A
        ///     matrix is plain (NumSharp's rule: NumPy's own Python body mis-reads the matrix's rows there).
        /// </summary>
        /// <param name="result">The difference computed on the plain views.</param>
        /// <param name="ar1">The caller's first operand.</param>
        /// <param name="assume_unique">The caller's flag.</param>
        /// <returns>The chararray view of the result, or the plain result.</returns>
        internal static NDArray Setdiff1d(NDArray result, NDArray ar1, bool assume_unique)
            => !assume_unique && NDSubclass.KindOf(ar1) == NDSubclassKind.CharArray
                ? NDSubclass.ArrayWrap(result, null, ar1, null, null)
                : NDSubclass.AsArray(result);

        /// <summary>
        ///     The unique family's values (<c>np.unique</c> without an axis or of a 1-d operand, <c>np.unique_counts</c>,
        ///     <c>np.unique_inverse</c>, <c>np.unique_all</c>): <c>_unique1d</c> hands the values back through
        ///     <c>_array_converter.wrap</c> or a boolean index of the chararray, so they are a chararray view; the index arrays
        ///     stay plain. A matrix gives plain outputs.
        /// </summary>
        /// <param name="values">The unique values computed on the plain view.</param>
        /// <param name="ar">The caller's operand.</param>
        /// <returns>The chararray view of the values, or the values.</returns>
        private static NDArray UniqueValues(NDArray values, NDArray ar)
            => NDSubclass.KindOf(ar) == NDSubclassKind.CharArray ? NDCharArray.WrapResult(values) : values;

        /// <summary>
        ///     <c>np.unique</c>: the values of the flat path (<c>axis=None</c>, or a 1-d operand) wrapped as
        ///     <see cref="UniqueValues"/>; the axis path of an N-d operand converts with <c>np.ascontiguousarray</c>, so every
        ///     output is plain.
        /// </summary>
        /// <param name="result">The outputs computed on the plain view.</param>
        /// <param name="ar">The caller's operand.</param>
        /// <param name="axis">The caller's axis.</param>
        /// <param name="return_index">The caller's flag (rebuilds the result's field layout).</param>
        /// <param name="return_inverse">The caller's flag.</param>
        /// <param name="return_counts">The caller's flag.</param>
        /// <returns>The outputs, the values wrapped for a chararray operand.</returns>
        internal static np.UniqueResult Unique(np.UniqueResult result, NDArray ar, int? axis, bool return_index, bool return_inverse,
                                              bool return_counts)
        {
            if (NDSubclass.KindOf(ar) != NDSubclassKind.CharArray || axis is not null && ar.ndim != 1)
                return result;
            var outputs = (NDArray[])result.ToArray().Clone();
            outputs[0] = UniqueValues(outputs[0], ar);
            return new np.UniqueResult(outputs, return_index, return_inverse, return_counts);
        }

        /// <summary><c>np.unique_counts</c>: the values wrapped as <see cref="UniqueValues"/>, the counts plain.</summary>
        /// <param name="result">The outputs computed on the plain view.</param>
        /// <param name="x">The caller's operand.</param>
        /// <returns>The outputs.</returns>
        internal static np.UniqueCountsResult UniqueCounts(np.UniqueCountsResult result, NDArray x)
            => NDSubclass.KindOf(x) == NDSubclassKind.CharArray
                ? new np.UniqueCountsResult(UniqueValues(result.values, x), result.counts)
                : result;

        /// <summary><c>np.unique_inverse</c>: the values wrapped as <see cref="UniqueValues"/>, the inverse plain.</summary>
        /// <param name="result">The outputs computed on the plain view.</param>
        /// <param name="x">The caller's operand.</param>
        /// <returns>The outputs.</returns>
        internal static np.UniqueInverseResult UniqueInverse(np.UniqueInverseResult result, NDArray x)
            => NDSubclass.KindOf(x) == NDSubclassKind.CharArray
                ? new np.UniqueInverseResult(UniqueValues(result.values, x), result.inverse_indices)
                : result;

        /// <summary><c>np.unique_all</c>: the values wrapped as <see cref="UniqueValues"/>, the three index outputs plain.</summary>
        /// <param name="result">The outputs computed on the plain view.</param>
        /// <param name="x">The caller's operand.</param>
        /// <returns>The outputs.</returns>
        internal static np.UniqueAllResult UniqueAll(np.UniqueAllResult result, NDArray x)
            => NDSubclass.KindOf(x) == NDSubclassKind.CharArray
                ? new np.UniqueAllResult(UniqueValues(result.values, x), result.indices, result.inverse_indices, result.counts)
                : result;

        #endregion

        #region Indexing

        /// <summary>
        ///     <c>np.diag(v, k)</c>: NumPy's 2-d branch is <c>np.diagonal(v, k)</c> - a chararray's view (<c>c.diagonal()</c>),
        ///     a matrix's plain array (<c>np.diagonal</c> converts a matrix with <c>asarray</c>); the 1-d branch builds
        ///     <c>np.zeros</c>, plain for every class.
        /// </summary>
        /// <param name="result">The diagonal or the built matrix, computed on the plain view.</param>
        /// <param name="v">The caller's operand.</param>
        /// <returns>The chararray view of a 2-d chararray's diagonal, otherwise the plain result.</returns>
        internal static NDArray Diag(NDArray result, NDArray v)
            => NDSubclass.KindOf(v) == NDSubclassKind.CharArray && v.ndim == 2
                ? NDSubclass.NewSubtype(result, null, v, keepRank: false, NewSubtypeZeroDim.Array)
                : NDSubclass.AsArray(result);

        /// <summary>
        ///     <c>np.take_along_axis(arr, indices, axis)</c>: NumPy indexes <c>arr</c> itself along an axis (the subclass's
        ///     <c>__getitem__</c>: a matrix, a chararray view), and <c>arr.flat</c> without one - a matrix's flat iterator gives a
        ///     plain 1-d array.
        /// </summary>
        /// <param name="result">The gathered values computed on the plain view.</param>
        /// <param name="arr">The caller's array.</param>
        /// <param name="axis">The caller's axis.</param>
        /// <returns>The wrapped result, or the plain one for a matrix without an axis.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray TakeAlongAxis(NDArray result, NDArray arr, int? axis)
            => axis is null && NDSubclass.KindOf(arr) == NDSubclassKind.Matrix
                ? NDSubclass.AsArray(result)
                : NDSubclass.ArrayWrap(result, null, arr, null, null);

        #endregion

        #region Sorting

        /// <summary>
        ///     <c>np.partition</c>'s entry check: NumPy calls <c>a.partition(kth, axis=axis, ...)</c> on its copy of the operand,
        ///     and a chararray's <c>partition</c> is the STRING partition, which takes no <c>axis</c>.
        /// </summary>
        /// <param name="a">The caller's operand.</param>
        /// <exception cref="TypeError"><paramref name="a"/> is a chararray: <c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>, before anything is computed.</exception>
        internal static void PartitionCheck(NDArray a)
        {
            if (NDSubclass.KindOf(a) == NDSubclassKind.CharArray)
                throw NDCharArray.PartitionKeywordError();
        }

        /// <summary>
        ///     <c>np.partition</c>: NumPy partitions <c>asanyarray(a).copy()</c> in place, so the result is the operand's class
        ///     (a chararray never gets here: <see cref="PartitionCheck"/>).
        /// </summary>
        /// <param name="result">The partitioned copy computed on the plain view.</param>
        /// <param name="a">The caller's operand.</param>
        /// <returns>The copy as <paramref name="a"/>'s class.</returns>
        internal static NDArray Partition(NDArray result, NDArray a)
            => NDSubclass.NewSubtype(result, null, a, keepRank: false, NewSubtypeZeroDim.Array);

        #endregion

        #region Ufuncs with several outputs

        /// <summary>
        ///     <c>np.frexp(x, out1, out2)</c>: a two-output ufunc - each output wrapped by <paramref name="x"/>'s class, except a
        ///     caller's out array, which comes back itself.
        /// </summary>
        /// <param name="result">The mantissa and exponent computed on the plain views.</param>
        /// <param name="x">The caller's input.</param>
        /// <param name="out1">The caller's mantissa out, or <c>null</c>.</param>
        /// <param name="out2">The caller's exponent out, or <c>null</c>.</param>
        /// <returns>NumPy's pair.</returns>
        /// <exception cref="ValueError">A matrix output with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) Frexp((NDArray, NDArray) result, NDArray x, NDArray out1, NDArray out2)
            => (NDSubclass.ArrayWrap(result.Item1, out1, x, null, null), NDSubclass.ArrayWrap(result.Item2, out2, x, null, null));

        /// <summary>
        ///     <c>np.modf(x, outFrac, outIntegral)</c>: a two-output ufunc - each output wrapped by <paramref name="x"/>'s class,
        ///     except a caller's out array, which comes back itself.
        /// </summary>
        /// <param name="result">The fractional and integral parts computed on the plain views.</param>
        /// <param name="x">The caller's input.</param>
        /// <param name="outFrac">The caller's fractional out, or <c>null</c>.</param>
        /// <param name="outIntegral">The caller's integral out, or <c>null</c>.</param>
        /// <returns>NumPy's pair.</returns>
        /// <exception cref="ValueError">A matrix output with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) Modf((NDArray, NDArray) result, NDArray x, NDArray outFrac, NDArray outIntegral)
            => (NDSubclass.ArrayWrap(result.Item1, outFrac, x, null, null),
                NDSubclass.ArrayWrap(result.Item2, outIntegral, x, null, null));

        /// <summary>
        ///     <c>np.divmod(x1, x2, out)</c>: a two-output ufunc - each output wrapped by the input chosen by priority, except a
        ///     caller's out array, which comes back itself.
        /// </summary>
        /// <param name="result">The quotient and remainder computed on the plain views.</param>
        /// <param name="x1">The caller's first input.</param>
        /// <param name="x2">The caller's second input (<c>null</c> when an overload takes it as a Python object).</param>
        /// <param name="out">The caller's pair of out arrays (<c>default</c> when the overload has none).</param>
        /// <returns>NumPy's pair.</returns>
        /// <exception cref="ValueError">A matrix output with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) DivMod((NDArray, NDArray) result, NDArray x1, NDArray x2,
                                                  (NDArray, NDArray) @out = default)
            => (NDSubclass.ArrayWrap(result.Item1, @out.Item1, x1, x2, null),
                NDSubclass.ArrayWrap(result.Item2, @out.Item2, x1, x2, null));

        #endregion

        #region Math and statistics

        /// <summary>
        ///     <c>np.cumulative_sum</c> / <c>np.cumulative_prod</c> (NumPy's <c>_cumulative_func</c>): the class depends on the
        ///     route. Without <c>include_initial</c> the result is <c>func.accumulate(x, out=out)</c>, a ufunc's (wrapped by
        ///     the operand's class, or out's own); with it and no out, NumPy prepends <c>np.full_like(res, identity, ...)</c>
        ///     (subok, so both pieces have <c>res</c>'s class) with <c>np.concat</c>, whose <c>PyArray_GetSubType</c> keeps a
        ///     matrix and drops a chararray; with it and an out, NumPy's own <c>return out</c> hands the caller's object back.
        /// </summary>
        /// <param name="result">The cumulative result computed on the plain view (the caller's <c>out</c> when given).</param>
        /// <param name="x">The caller's operand (bound by NAME to the public function's <c>x</c>).</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <param name="include_initial">The caller's include_initial.</param>
        /// <returns>NumPy's answer for the route the call took.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray Cumulative(NDArray result, NDArray x, NDArray @out, bool include_initial)
        {
            // out + include_initial: `return out` - the caller's object itself, not a ufunc's wrap of it.
            if (@out is not null && include_initial)
                return @out;
            return include_initial
                ? NDSubclass.GetSubType(result, null, new[] { x, x })      // concat([full_like(res), res]): both as x's class
                : NDSubclass.ArrayWrap(result, @out, x, null, null);       // func.accumulate(x, ..., out=out)
        }

        /// <summary>
        ///     <c>np.round(a, decimals, out)</c> / <c>np.around</c>: <c>_wrapfunc(a, 'round', ...)</c> is <c>ndarray.round</c>,
        ///     whose <c>decimals=0</c> route is the <c>rint</c> ufunc (wrapped by the operand's class); any other
        ///     <c>decimals</c> runs multiply / rint / true_divide into a result NumPy returns as a plain array (probed 2.4.2).
        /// </summary>
        /// <param name="result">The rounded array computed on the plain view (or the caller's <c>out</c>).</param>
        /// <param name="a">The caller's operand (bound by NAME to np.round's / np.around's <c>a</c>: a rename there
        ///     must rename this, or NumSharp.Build silently binds null and a matrix operand comes back plain).</param>
        /// <param name="decimals">The caller's decimals.</param>
        /// <param name="out">The caller's out, or <c>null</c>.</param>
        /// <returns>The wrapped result for <c>decimals=0</c>, otherwise a plain array (a matrix out's plain view).</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray Round(NDArray result, NDArray a, int decimals, NDArray @out)
            => decimals == 0 ? NDSubclass.ArrayWrap(result, @out, a, null, null) : NDSubclass.AsArray(result);

        /// <summary>
        ///     <c>np.trapezoid(y, x, dx, axis)</c>: NumPy ends in <c>(d * (y[1:] + y[:-1]) / 2.0).sum(axis)</c> - for a matrix
        ///     <c>y</c> that is the matrix METHOD's sum (<c>_collapse(axis)</c>: a row or a column); for a matrix <c>x</c> the
        ///     product of the two matrices raises inside NumPy, which then repeats the statement on plain arrays, so the result
        ///     is plain.
        /// </summary>
        /// <param name="result">The integral computed on the plain views.</param>
        /// <param name="y">The caller's ordinates.</param>
        /// <param name="x">The caller's sample points, or <c>null</c>.</param>
        /// <param name="axis">The caller's axis.</param>
        /// <returns>The collapsed matrix result for a matrix <paramref name="y"/>, otherwise the plain result.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray Trapezoid(NDArray result, NDArray y, NDArray x, int axis)
            => NDSubclass.KindOf(x) == NDSubclassKind.Matrix
                ? NDSubclass.AsArray(result)
                : NDSubclass.KindOf(y) == NDSubclassKind.Matrix
                    ? NDSubclass.Collapse(result, null, y, axis, subtypeOthers: false, NewSubtypeZeroDim.Array)
                    : result;

        /// <summary>
        ///     <c>np.average</c>'s entry check: NumPy hands <c>keepdims</c> to the matrix's own method - <c>a.mean(axis,
        ///     keepdims=...)</c> without weights, <c>np.multiply(a, wgt).sum(axis, keepdims=...)</c> with them - which has no
        ///     such keyword.
        /// </summary>
        /// <param name="a">The caller's operand.</param>
        /// <param name="weights">The caller's weights, or <c>null</c>.</param>
        /// <param name="keepdims">The caller's keepdims.</param>
        /// <exception cref="TypeError"><paramref name="a"/> is a matrix and <paramref name="keepdims"/> is true:
        ///     <c>matrix.mean() got an unexpected keyword argument 'keepdims'</c> (<c>matrix.sum()</c> with weights), before
        ///     anything is computed.</exception>
        internal static void AverageCheck(NDArray a, NDArray weights, bool keepdims)
        {
            if (keepdims && NDSubclass.KindOf(a) == NDSubclassKind.Matrix)
                throw NDSubclass.KeywordError("matrix", weights is null ? "mean" : "sum", "keepdims");
        }

        /// <summary>
        ///     <c>np.average(a, axis, weights)</c> of a matrix: the matrix METHOD's reduction - <c>keepdims=True</c>, then
        ///     <c>_collapse(axis)</c>, which is the scalar without an axis and the kept axes as they are otherwise. The
        ///     computation reduced a matrix operand's axes with keepdims (<c>NDArray.average</c>), so a given axis needs only the
        ///     class; any other operand keeps the plain answer.
        /// </summary>
        /// <param name="result">The average computed on the plain view.</param>
        /// <param name="a">The caller's operand.</param>
        /// <param name="axis">The caller's axis: <c>null</c>, an <c>int</c> (boxed) or an <c>int[]</c>.</param>
        /// <returns>The scalar or the matrix view of a matrix operand's average, otherwise <paramref name="result"/>.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray Average(NDArray result, NDArray a, object axis)
        {
            if (NDSubclass.KindOf(a) != NDSubclassKind.Matrix)
                return result;
            // _collapse(None): the computation reduced every axis to the scalar.
            if (axis is null)
                return NDSubclass.Collapse(result, null, a, null, subtypeOthers: false, NewSubtypeZeroDim.Array);
            return NDSubclass.NewSubtype(result, null, a, keepRank: false, NewSubtypeZeroDim.Array);
        }

        /// <summary>
        ///     <c>np.average(..., returned=True)</c>: the average as <see cref="Average"/>, and NumPy's sum of weights, which
        ///     ends in the average's shape (<c>np.broadcast_to(scl, avg_as_array.shape).copy()</c>, a plain array) - the
        ///     computation reduced it with the same keepdims as the average, so it already has that shape.
        /// </summary>
        /// <param name="result">The average and the sum of weights computed on the plain views.</param>
        /// <param name="a">The caller's operand.</param>
        /// <param name="axis">The caller's axis: <c>null</c>, an <c>int</c> (boxed) or an <c>int[]</c>.</param>
        /// <returns>The pair, the average shaped as <see cref="Average"/>.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) AverageReturned((NDArray, NDArray) result, NDArray a, object axis)
            => (Average(result.Item1, a, axis), result.Item2);

        #endregion

        #region Logic

        /// <summary>
        ///     <c>np.iscomplex(x)</c>: for a complex operand NumPy computes <c>ax.imag != 0</c>, a ufunc (wrapped by the
        ///     operand's class); any other dtype is a new <c>zeros(ax.shape, bool)</c>, plain.
        /// </summary>
        /// <param name="result">The mask computed on the plain view.</param>
        /// <param name="x">The caller's operand (bound by NAME to np.iscomplex's <c>x</c>: a rename there must rename
        ///     this, or NumSharp.Build silently binds null and a complex matrix's mask comes back plain).</param>
        /// <returns>The wrapped mask of a complex operand, otherwise the plain mask.</returns>
        internal static NDArray Iscomplex(NDArray result, NDArray x)
            => x is not null && x.typecode == NPTypeCode.Complex
                ? NDSubclass.ArrayWrap(result, null, x, null, null)
                : NDSubclass.AsArray(result);

        #endregion

        #region Linear algebra

        /// <summary>
        ///     <c>np.linalg.qr(a, mode)</c>: NumPy returns <c>wrap(q), wrap(r)</c> (<c>wrap(r)</c> alone for mode <c>'r'</c>,
        ///     where Q is absent), and for mode <c>'raw'</c> <c>wrap(h)</c> with the plain <c>tau</c>.
        /// </summary>
        /// <param name="result">Q (or h) and R (or tau), computed on the plain view.</param>
        /// <param name="a">The caller's operand.</param>
        /// <param name="mode">The caller's mode (already validated by the computation).</param>
        /// <returns>The pair, its wrapped slots as <paramref name="a"/>'s class.</returns>
        /// <exception cref="ValueError">A matrix output with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) Qr((NDArray, NDArray) result, NDArray a, string mode)
        {
            bool raw = Backends.DefaultLinalgEngine.ResolveQrMode(mode) == "raw";
            // Bit 0 alone wraps h; -1 wraps both slots (a null Q of mode 'r' stays null).
            return NDSubclass.ArrayWrap(result, null, a, null, null, raw ? 1 : -1);
        }

        #endregion
    }
}
