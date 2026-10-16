using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

// =============================================================================
// NDCharArrayBoundary - what a public np.* function returns for a numpy.char.chararray operand
// =============================================================================
//
// NumPy finds a chararray at run time, and each function hands one back (or refuses it) by its own mechanism:
//
//   * a function that calls the operand's METHOD or works on np.asanyarray(a) keeps the class: np.reshape, np.ravel,
//     np.transpose, np.squeeze, np.flip, np.take, np.sort, np.repeat, np.tile, the split family ...
//     (rule Kept: the plain result, its dtype checked by __array_finalize__, as a chararray over the same storage, so
//     it owns its data exactly when the result does; a function that returns its operand returns the chararray itself);
//   * a ufunc, and a function that ends with _array_converter.wrap (np.insert, np.delete, np.diagflat, np.unique's
//     values) or indexes the chararray with an array (np.take_along_axis, np.setdiff1d), returns
//     chararray.__array_wrap__'s VIEW of its result (rule Wrapped, OWNDATA False); a result that is not text stays an
//     ndarray, and a ufunc given a chararray out= returns a new view of that out;
//   * a function that converts with np.asarray / np.concatenate / np.where returns an ndarray (no rule; rule Plain where
//     the computation could hand the operand object back);
//   * a function that ALLOCATES a result of another dtype as the subclass raises __array_finalize__'s ValueError
//     ("Can only create a chararray from string data."): np.argmax / np.argmin / np.argpartition and their nan forms
//     (rule IndexResult, after NumPy's argument checks), np.zeros_like with a numeric dtype, and the ufuncs whose
//     trivial loop COPIES a short chararray input into the loop's dtype (np.logical_and casts to bool, a StringDType
//     operand casts the chararray to StringDType: TrivialLoopCopyError);
//   * a function that calls ndarray.partition with keywords meets chararray.partition (the STRING partition, which takes
//     only a separator): np.partition, np.median, np.quantile, np.percentile raise "chararray.partition() got an
//     unexpected keyword argument 'axis'".
//
// THE BOUNDARY IS THE OUTERMOST PUBLIC CALL, and its depth is NDMatrixBoundary's: a rule raises it around the plain
// computation, so nested public calls (np.unique calls np.sort) compute plain arrays for both subclasses, and
// NDMatrixBoundary.PlainFor opens a region for a chararray receiver (an NDArray member of a chararray object stays
// plain: the static rule, see NDCharArray.cs).
//
// THE SCALAR STAND-IN IS NOT A CHARARRAY OPERAND. c[0] is NumPy's Python str / bytes: a function that receives the 0-d
// stand-in computes as for a plain 0-d array (np.atleast_2d(c[0]) is an ndarray in NumPy).
//
// THE CONTRACT IS MEASURED by the string oracle's chararray sections (strcharops / strcharfn): every rule against
// NumPy's class, values, ownership and errors, and an audit that no library method reaches a chararray rule at depth
// zero (AuditInternalCallers).
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     The run-time rule of the public <c>np.*</c> functions for a <c>numpy.char.chararray</c> operand: whether the call
    ///     sees a chararray (<see cref="Sees(NDArray, string)"/>) and how NumPy answers with one (<see cref="Kept{TState}"/>,
    ///     <see cref="Wrapped{TState}"/>, <see cref="Plain{TState}"/>, the finalize and partition errors). Inside the
    ///     library a chararray is a plain string array.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The pattern of a public function matches <see cref="NDMatrixBoundary"/>'s: when <see cref="Sees(NDArray, string)"/>
    ///     is true, call a rule with the arguments packed in a tuple and a <c>static</c> lambda that calls the function
    ///     again. The rule raises the shared depth, so that second call computes with plain-array semantics, and then turns
    ///     the plain result into NumPy's. A <c>static</c> lambda with its state passed by value allocates nothing.
    ///     </para>
    ///     <para>
    ///     Thread safety: the depth and the ufunc operand marks are per thread, and a rule restores them in a
    ///     <c>finally</c>, so an exception from the computation leaves them at their entry values.
    ///     </para>
    /// </remarks>
    internal static class NDCharArrayBoundary
    {
        /// <summary>NumPy's error text when a function's <c>ndarray.partition</c> call meets <c>chararray.partition</c>.</summary>
        internal const string PartitionKeywordText = "chararray.partition() got an unexpected keyword argument 'axis'";

        /// <summary>
        ///     Whether <paramref name="a"/> is a chararray OPERAND: an <see cref="NDCharArray"/> object that is not the scalar
        ///     stand-in of an element access (NumPy's Python <c>str</c> / <c>bytes</c>). Does not look at the depth; use
        ///     <see cref="Sees(NDArray, string)"/> to decide a function's rule.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>True for a chararray of any rank.</returns>
        /// <remarks>An exact type test (one method-table compare): <see cref="NDCharArray"/> has no subclass in the library,
        ///     and <c>is</c> on a non-sealed class walks the type hierarchy of every plain array that passes through.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsCharArray(NDArray a)
            => a is not null && a.GetType() == typeof(NDCharArray) && !((NDCharArray)a).IsScalarStandIn;

        /// <summary>
        ///     Whether this call is the boundary for a chararray operand: <paramref name="a"/> is a chararray and no public
        ///     function above this one (of either array subclass) is already handling the call.
        /// </summary>
        /// <param name="a">The array operand, or <c>null</c>.</param>
        /// <param name="function">The calling public function's name. Leave it to the compiler; only the audit reads it.</param>
        /// <returns>True when the function must apply its chararray rule; false means compute as for a plain array.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, [CallerMemberName] string function = null)
            => IsCharArray(a) && NDMatrixBoundary.AtOutermostCall && (!AuditInternalCallers || Audit(function));

        /// <summary><see cref="Sees(NDArray, string)"/> for a function of two array operands: true when either is a chararray.</summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its chararray rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, NDArray b, [CallerMemberName] string function = null)
            => (IsCharArray(a) || IsCharArray(b)) && NDMatrixBoundary.AtOutermostCall && (!AuditInternalCallers || Audit(function));

        /// <summary><see cref="Sees(NDArray, string)"/> for a function of three array operands (a ufunc's two inputs and its
        ///     <c>out=</c>: NumPy looks up <c>__array_wrap__</c> on all of them).</summary>
        /// <param name="a">The first operand, or <c>null</c>.</param>
        /// <param name="b">The second operand, or <c>null</c>.</param>
        /// <param name="c">The third operand, or <c>null</c>.</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its chararray rule.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool Sees(NDArray a, NDArray b, NDArray c, [CallerMemberName] string function = null)
            => (IsCharArray(a) || IsCharArray(b) || IsCharArray(c)) && NDMatrixBoundary.AtOutermostCall
               && (!AuditInternalCallers || Audit(function));

        /// <summary><see cref="Sees(NDArray, string)"/> for operands passed as one sequence: true when any is a chararray.</summary>
        /// <param name="arrays">The operands, or <c>null</c>; a <c>null</c> element is skipped.</param>
        /// <param name="function">The calling public function's name (compiler-supplied; read by the audit).</param>
        /// <returns>True when the function must apply its chararray rule.</returns>
        internal static bool Sees(NDArray[] arrays, [CallerMemberName] string function = null)
        {
            if (arrays is null || !NDMatrixBoundary.AtOutermostCall)
                return false;
            for (int i = 0; i < arrays.Length; i++)
                if (IsCharArray(arrays[i]))
                    return !AuditInternalCallers || Audit(function);
            return false;
        }

        /// <summary>
        ///     Opens a region of plain-array semantics: the public functions the caller calls inside it compute for plain
        ///     arrays (no subclass rule applies). For a rule written out by hand in a function whose result needs a shape
        ///     the generic rules do not build (np.unique's result structs).
        /// </summary>
        /// <returns>The region; dispose it (<c>using</c>) when the plain computation is done.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDMatrixBoundary.PlainRegion Inside() => NDMatrixBoundary.Inside();

        #region The audit

        /// <summary>
        ///     Test hook: when set, <see cref="Sees(NDArray, string)"/> records every library method that reaches a chararray
        ///     rule at depth zero in <see cref="InternalCallers"/>. The oracle project replays its chararray cases with it on
        ///     and requires the set to stay empty: the rules are written for the USER's call, and a library method that
        ///     passes its operand on must compute plain.
        /// </summary>
        internal static bool AuditInternalCallers;

        /// <summary>What the audit found: <c>"{caller type}.{caller method} -&gt; {function}"</c> with the number of times.</summary>
        internal static readonly ConcurrentDictionary<string, int> InternalCallers = new();

        /// <summary>Records the caller of <paramref name="function"/> when that caller belongs to this library.</summary>
        /// <param name="function">The public function whose chararray rule is about to apply.</param>
        /// <returns>Always true, so the audit never changes what the function returns.</returns>
        /// <remarks>Reads the stack, so it runs only under the test hook. Frames of this class, of
        ///     <see cref="NDMatrixBoundary"/>, of <see cref="NDCharArray"/> (the chararray's operators and members call the
        ///     public functions on purpose, exactly as NumPy's methods do) and frames named like the function (the function
        ///     itself, a forwarding overload, <c>np.char.add</c> forwarding to <c>np.add</c>) are skipped; the first frame
        ///     left is the caller.</remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static bool Audit(string function)
        {
            var trace = new System.Diagnostics.StackTrace(1, false);
            for (int i = 0; i < trace.FrameCount; i++)
            {
                System.Reflection.MethodBase method = trace.GetFrame(i)?.GetMethod();
                Type owner = method?.DeclaringType;
                if (owner is null || owner == typeof(NDCharArrayBoundary) || owner == typeof(NDMatrixBoundary)
                    || owner == typeof(NDCharArray))
                    continue;
                if (method.Name == function)
                    continue;
                if (owner.Assembly == typeof(NDCharArrayBoundary).Assembly)
                    InternalCallers.AddOrUpdate($"{owner.FullName}.{method.Name} -> {function}", 1, static (_, n) => n + 1);
                break;
            }
            return true;
        }

        #endregion

        #region The rules

        /// <summary>
        ///     The rule of a function whose NumPy result is the operand's CLASS because NumPy creates it as the subclass (a
        ///     method call, an <c>np.asanyarray</c> view, <c>empty_like</c>'s <c>subok=True</c>): the plain result, its dtype
        ///     checked by <c>__array_finalize__</c>, as a chararray over the same storage — owning its data exactly when the
        ///     plain result does.
        /// </summary>
        /// <typeparam name="TState">The arguments of the function, packed by value (a tuple).</typeparam>
        /// <param name="state">The arguments, handed to <paramref name="call"/> unchanged.</param>
        /// <param name="call">Calls the public function again. It runs with the shared depth raised, so it computes with
        ///     plain-array semantics. Pass a <c>static</c> lambda.</param>
        /// <param name="out">The function's <c>out=</c> argument, or <c>null</c>: NumPy returns a given <c>out</c> itself.</param>
        /// <returns>The chararray; <paramref name="out"/> when the computation returned it; the computation's own result when
        ///     that already is a chararray (the operand handed back, as <c>np.asanyarray(c)</c> is <c>c</c>).</returns>
        /// <exception cref="ValueError">The result's dtype is not string data (NumPy's finalize error: the computation ran,
        ///     but nothing it computed can raise first — the functions that allocate before reading test the dtype
        ///     themselves, see <see cref="RequireStringResult(DType)"/>).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray Kept<TState>(TState state, Func<TState, NDArray> call, NDArray @out = null)
        {
            // The scope reclaims the hidden plain result: the chararray holds its own counted reference on the buffer.
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray result = call(state);
                if (result is null || ReferenceEquals(result, @out) || result is NDCharArray)
                    return scope.Returns(result);
                return scope.Returns(AsChararray(result));
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Kept{TState}"/> for a function that returns several arrays (<c>np.split</c>, <c>np.unstack</c>):
        ///     each one a chararray.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <returns>The arrays, each a chararray over its plain result's storage.</returns>
        /// <exception cref="ValueError">A result's dtype is not string data.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray[] KeptAll<TState>(TState state, Func<TState, NDArray[]> call)
        {
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray[] results = call(state);
                if (results is null)
                    return null;
                var kept = new NDArray[results.Length];
                for (int i = 0; i < results.Length; i++)
                    kept[i] = results[i] is null or NDCharArray ? results[i] : AsChararray(results[i]);
                for (int i = 0; i < kept.Length; i++)
                    scope.Returns(kept[i]);
                return kept;
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="KeptAll{TState}"/> for a function whose result <c>i</c> is computed from operand <c>i</c> alone and
        ///     keeps THAT operand's class (<c>np.meshgrid</c>: <c>np.asanyarray(x).reshape(...)</c>, then
        ///     <c>broadcast_arrays(subok=True)</c> and <c>copy()</c>, per input): result <c>i</c> becomes a chararray when
        ///     operand <c>i</c> is one, and stays plain otherwise.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="operands">The operands, one per result, in the results' order.</param>
        /// <returns>The results, the chararray operands' ones as chararrays over their plain result's storage.</returns>
        /// <exception cref="ValueError">The result of a chararray operand is not string data.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray[] KeptPerOperand<TState>(TState state, Func<TState, NDArray[]> call, NDArray[] operands)
        {
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray[] results = call(state);
                if (results is null)
                    return null;
                var kept = new NDArray[results.Length];
                for (int i = 0; i < results.Length; i++)
                    kept[i] = i < operands.Length && IsCharArray(operands[i]) && results[i] is not null and not NDCharArray
                        ? AsChararray(results[i])
                        : results[i];
                for (int i = 0; i < kept.Length; i++)
                    scope.Returns(kept[i]);
                return kept;
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Kept{TState}"/> for <c>np.take</c>: <c>ndarray.take</c> ends with <c>PyArray_Return</c>, so a 0-d
        ///     result WITHOUT <c>out=</c> is the element's NumPy scalar (<c>np.take(c, 0)</c> is <c>np.str_('b a ')</c>,
        ///     NOT stripped: <c>__getitem__</c> never runs) — NumSharp's read-only scalar stand-in holding the unstripped
        ///     text. Any other result is <see cref="Kept{TState}"/>'s.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls <c>np.take</c> again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="out">The <c>out=</c> argument, or <c>null</c>: a given <c>out</c> is returned itself, also 0-d.</param>
        /// <returns>The stand-in, the chararray, or <paramref name="out"/>.</returns>
        /// <exception cref="ValueError">The chararray holds data that is not string data (NumPy allocates the result as the
        ///     subclass before <c>PyArray_Return</c> looks at its rank).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray KeptTake<TState>(TState state, Func<TState, NDArray> call, NDArray @out)
            => KeptOrElement(state, call, @out, strip: false, finalizeFirst: true);

        /// <summary>
        ///     <see cref="Kept{TState}"/> for a function that ends by INDEXING the chararray (<c>np.flip</c>'s
        ///     <c>m[indexer]</c>): <c>chararray.__getitem__</c> turns a 0-d result into the element, its trailing whitespace
        ///     stripped (<c>np.flip(c0)</c> of a 0-d chararray is the Python <c>str</c>, as <c>c0[()]</c> is) — NumSharp's
        ///     read-only scalar stand-in. Any other result is <see cref="Kept{TState}"/>'s chararray view.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <returns>The stand-in or the chararray.</returns>
        /// <exception cref="ValueError">An array result whose dtype is not string data (an element is no array: it is never
        ///     refused).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray KeptIndexed<TState>(TState state, Func<TState, NDArray> call)
            => KeptOrElement(state, call, null, strip: true, finalizeFirst: false);

        /// <summary>The core of <see cref="KeptTake{TState}"/> and <see cref="KeptIndexed{TState}"/>: a 0-d result becomes the
        ///     element's stand-in, any other result the chararray over its storage.</summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics).</param>
        /// <param name="out">The <c>out=</c> argument, or <c>null</c>: a given <c>out</c> is returned itself, also 0-d.</param>
        /// <param name="strip">True for an element <c>chararray.__getitem__</c> returns (stripped); false for the NumPy scalar
        ///     <c>PyArray_Return</c> makes of a 0-d result (unstripped).</param>
        /// <param name="finalizeFirst">True when NumPy creates the 0-d result as the subclass before turning it into the
        ///     scalar (<c>ndarray.take</c>), so the finalize rule runs on its dtype first.</param>
        /// <returns>The stand-in, the chararray, or <paramref name="out"/>.</returns>
        /// <exception cref="ValueError">A result of a dtype that is not string data where NumPy creates it as the
        ///     subclass.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        private static NDArray KeptOrElement<TState>(TState state, Func<TState, NDArray> call, NDArray @out, bool strip,
                                                     bool finalizeFirst)
        {
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray result = call(state);
                if (result is null || ReferenceEquals(result, @out) || result is NDCharArray)
                    return scope.Returns(result);
                if (result.ndim == 0)
                {
                    if (finalizeFirst)
                        NDCharArray.Finalize(result.dtype);
                    // The stand-in copies the element; the plain 0-d result is reclaimed by the scope.
                    return scope.Returns(NDCharArray.ScalarOf(result.Storage, strip));
                }
                return scope.Returns(AsChararray(result));
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's <c>chararray.__array_wrap__</c> for a ufunc, or for a function that ends with
        ///     <c>_array_converter.wrap</c> / indexes the chararray with an array: a result of a string dtype
        ///     (<c>dtype.char in 'SUbc'</c>) becomes a chararray VIEW of the computed output (never an owner); any other
        ///     result stays a plain array. A chararray <c>out=</c> is wrapped too (a new view of it, not the object itself).
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="out">The ufunc's <c>out=</c>, or <c>null</c>. A plain <c>out</c> is returned itself (NumPy's wrap of an
        ///     exact ndarray is the identity); a chararray <c>out</c> through its own <c>__array_wrap__</c>: a new chararray
        ///     view of it, so <c>np.add(c, c, out=cout) is cout</c> is False in NumPy too.</param>
        /// <returns>The chararray view, the plain result, or <paramref name="out"/> / a view of it.</returns>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray Wrapped<TState>(TState state, Func<TState, NDArray> call, NDArray @out = null)
        {
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray result = call(state);
                if (result is null)
                    return null;
                if (ReferenceEquals(result, @out))
                    return IsCharArray(@out) ? scope.Returns(NDCharArray.ViewOf(@out)) : @out;
                return scope.Returns(WrapText(result));
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a ufunc call: also marks the chararray INPUTS of this call, so the string
        ///     ufunc guard can apply NumPy's trivial-loop copy to exactly these objects (<see cref="TrivialLoopCopyError"/>).
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the ufunc's public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="x1">The first input.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="out">The <c>out=</c> argument, or <c>null</c>.</param>
        /// <returns>What <see cref="Wrapped{TState}"/> returns.</returns>
        /// <exception cref="ValueError">The guard found a short chararray input its loop casts to a dtype the finalize rule
        ///     refuses (<c>np.logical_and(c, c)</c>, <c>np.add(c, stringdtype_array)</c>).</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth and the marks are restored first.</exception>
        /// <remarks>
        ///     The marks are the operand OBJECTS, not a depth: the ufunc that runs inside this rule is the user's call, and a
        ///     public function that a library method calls with the same chararray at a deeper level (a region of plain
        ///     semantics) must not raise NumPy's copy error where NumPy's own function converts with <c>np.asarray</c>.
        ///     The first guard that reads the marks consumes them.
        /// </remarks>
        internal static NDArray WrappedUfunc<TState>(TState state, Func<TState, NDArray> call, NDArray x1, NDArray x2, NDArray @out)
        {
            NDArray saved0 = t_ufuncInput0, saved1 = t_ufuncInput1;
            t_ufuncInput0 = IsCharArray(x1) ? x1 : null;
            t_ufuncInput1 = IsCharArray(x2) ? x2 : null;
            try
            {
                return Wrapped(state, call, @out);
            }
            finally
            {
                t_ufuncInput0 = saved0;
                t_ufuncInput1 = saved1;
            }
        }

        /// <summary>
        ///     <see cref="Wrapped{TState}"/> for a function that returns several arrays (<c>np.unique</c>'s tuple): only the
        ///     FIRST — the values, which NumPy hands back through <c>_array_converter.wrap</c> or a boolean index — becomes a
        ///     chararray view; the index arrays stay plain.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="stripRuns">True for the unique family's sort path (a return flag): NumPy's <c>_unique1d</c> sorts the
        ///     chararray and finds the runs with <c>aux[1:] != aux[:-1]</c>, which is the chararray's STRIPPING comparison, so
        ///     strings that differ only in trailing whitespace are one value (<see cref="StripUniqueRuns"/>).</param>
        /// <returns>The results, the first one wrapped.</returns>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth and the flag are restored first.</exception>
        internal static NDArray[] WrappedFirst<TState>(TState state, Func<TState, NDArray[]> call, bool stripRuns = false)
        {
            bool savedStrip = t_stripUniqueRuns;
            t_stripUniqueRuns = stripRuns;
            NDScope scope = NDScope.Open();
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray[] results = call(state);
                if (results is null || results.Length == 0)
                    return results;
                var wrapped = (NDArray[])results.Clone();
                wrapped[0] = WrapText(results[0]);
                for (int i = 0; i < wrapped.Length; i++)
                    scope.Returns(wrapped[i]);
                return wrapped;
            }
            finally
            {
                region.Dispose();
                scope.Dispose();
                t_stripUniqueRuns = savedStrip;
            }
        }

        /// <summary>Set while the unique family's chararray rule computes a call with a return flag (see
        ///     <see cref="WrappedFirst{TState}"/>).</summary>
        [ThreadStatic] private static bool t_stripUniqueRuns;

        /// <summary>
        ///     Whether the text unique being computed finds its runs with the chararray's stripping comparison: true inside
        ///     the unique family's chararray rule for a call with a return flag (NumPy's <c>_unique1d</c> compares the sorted
        ///     CHARARRAY with its own <c>!=</c>). The computation also requires its operand to be the chararray object, so a
        ///     unique of another array inside the same call compares plainly.
        /// </summary>
        internal static bool StripUniqueRuns => t_stripUniqueRuns;

        /// <summary>
        ///     The rule of a function that creates an INDEX result as the operand's subclass (<c>np.argmax</c>,
        ///     <c>np.argmin</c>, <c>np.nanargmax</c>, <c>np.nanargmin</c>, <c>np.argpartition</c>: <c>_wrapfunc</c> calls the
        ///     chararray's method, whose C code allocates the int64 result with <c>Py_TYPE(self)</c>): every argument check
        ///     of the plain computation first, then the finalize error — unless an <c>out=</c> receives the indices, in which
        ///     case nothing is created as the subclass and the call computes.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <param name="out">The <c>out=</c> argument, or <c>null</c>.</param>
        /// <returns><paramref name="out"/>, filled.</returns>
        /// <exception cref="ValueError">Without <paramref name="out"/>, always (<c>Can only create a chararray from string
        ///     data.</c>) once the computation's own checks passed.</exception>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws first (an axis error, an empty sequence, a
        ///     kth out of bounds, an <c>out=</c> of the wrong shape or dtype); the depth is restored first.</exception>
        internal static NDArray IndexResult<TState>(TState state, Func<TState, NDArray> call, NDArray @out)
        {
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray result = call(state);
                if (@out is not null)
                    return result;
                // The plain computation stands for NumPy's argument checks; its index is never handed out.
                result?.Dispose();
                throw FinalizeError();
            }
            finally
            {
                region.Dispose();
            }
        }

        /// <summary>
        ///     <see cref="IndexResult{TState}"/> for an overload that returns the flat index as a <see cref="long"/>
        ///     (<c>np.argmax(c)</c>): the plain computation's checks, then always the finalize error.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again (plain-array semantics). Pass a <c>static</c> lambda.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError">The finalize error, or the plain computation's own (an empty sequence).</exception>
        internal static long IndexScalar<TState>(TState state, Func<TState, long> call)
        {
            var region = NDMatrixBoundary.Inside();
            try
            {
                call(state);
                throw FinalizeError();
            }
            finally
            {
                region.Dispose();
            }
        }

        /// <summary>
        ///     The rule of a function NumPy converts its operand in with <c>np.asarray</c> (<c>np.asarray</c>,
        ///     <c>np.ascontiguousarray</c>, <c>np.array</c>): the plain result, computed with plain-array semantics; when that
        ///     is the chararray operand itself (the computation had nothing to do), a new plain view of it.
        /// </summary>
        /// <typeparam name="TState">The packed arguments.</typeparam>
        /// <param name="state">The arguments.</param>
        /// <param name="call">Calls the public function again. Pass a <c>static</c> lambda.</param>
        /// <returns>The plain result.</returns>
        /// <exception cref="Exception">Whatever <paramref name="call"/> throws; the depth is restored first.</exception>
        internal static NDArray Plain<TState>(TState state, Func<TState, NDArray> call)
        {
            var region = NDMatrixBoundary.Inside();
            try
            {
                NDArray result = call(state);
                if (result is not null && result.GetType() == typeof(NDCharArray))
                    return new NDArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };
                return result;
            }
            finally
            {
                region.Dispose();
            }
        }

        /// <summary>
        ///     <paramref name="a"/> as a plain array: a NEW plain view of a chararray operand's memory (the chararray's
        ///     <c>.A</c>-like view, writes reach it), any other array itself. For a module that computes with plain-array
        ///     semantics on data it keeps (numpy.ma: NumPy's masked chararray keeps the chararray as <c>_baseclass</c>, which
        ///     NumSharp does not model, so its masked array holds plain data, as it does for a matrix).
        /// </summary>
        /// <param name="a">The array, or <c>null</c>.</param>
        /// <returns>The plain view, or <paramref name="a"/>.</returns>
        /// <remarks>The view is a new object with its own counted reference on the buffer: the caller owns it (a scope
        ///     reclaims it, or the module keeps it as its data).</remarks>
        internal static NDArray PlainViewOf(NDArray a)
            => IsCharArray(a) ? new NDArray(a.Storage.Alias()) { TensorEngine = a.TensorEngine } : a;

        /// <summary>
        ///     <c>np.unstack</c> of a ONE-dimensional chararray: NumPy's <c>tuple(moveaxis(x, axis, 0))</c> iterates the
        ///     chararray, and iteration calls <c>chararray.__getitem__</c> per element — the Python <c>str</c> / <c>bytes</c>
        ///     with its trailing whitespace stripped. NumSharp's stand-ins for those values.
        /// </summary>
        /// <param name="x">The 1-d chararray (the axis was checked by the caller).</param>
        /// <returns>One read-only stand-in per element, stripped.</returns>
        internal static NDArray[] StrippedElements(NDCharArray x)
        {
            var values = new NDArray[checked((int)x.size)];
            for (int i = 0; i < values.Length; i++)
                values[i] = x[i];
            return values;
        }

        #endregion

        #region The finalize rule

        /// <summary>
        ///     Throws NumPy's finalize error when a result of <paramref name="dtype"/> cannot be a chararray: for a function
        ///     that ALLOCATES its result as the subclass before reading the data (<c>np.full_like(c, 'x', dtype=float)</c>),
        ///     so the error comes before anything is computed.
        /// </summary>
        /// <param name="dtype">The descriptor of the result about to be created.</param>
        /// <exception cref="ValueError">The dtype is not string data (<c>Can only create a chararray from string data.</c>).</exception>
        internal static void RequireStringResult(DType dtype) => NDCharArray.Finalize(dtype);

        /// <summary>NumPy's finalize error, for a function whose result is never string data (<c>np.argmax(c)</c>).</summary>
        /// <returns>The exception to throw: <c>ValueError: Can only create a chararray from string data.</c></returns>
        internal static ValueError FinalizeError() => new ValueError(NDCharArray.FinalizeText);

        /// <summary>
        ///     NumPy's error when a function calls <c>a.partition(kth, axis=...)</c> on a chararray (<c>np.partition</c>,
        ///     <c>np.median</c>, the unweighted <c>np.quantile</c> / <c>np.percentile</c>): the subclass's <c>partition</c> is
        ///     the STRING partition, whose only parameter is the separator.
        /// </summary>
        /// <returns>The exception to throw, with NumPy's text.</returns>
        internal static TypeError PartitionKeywordError() => new TypeError(PartitionKeywordText);

        /// <summary>
        ///     <c>chararray.__array_wrap__</c> of one result: a chararray VIEW over a result of a string dtype
        ///     (<c>dtype.char in 'SUbc'</c>), the result itself otherwise. The view never owns its data.
        /// </summary>
        /// <param name="result">The plain result (its wrapper is reclaimed by the caller's scope when it is replaced).</param>
        /// <returns>The view, or <paramref name="result"/>.</returns>
        internal static NDArray WrapText(NDArray result)
        {
            if (result is null)
                return null;
            char code = result.dtype.@char;
            if (code != 'S' && code != 'U' && code != 'b' && code != 'c')
                return result;
            // __array_wrap__: arr.view(type(self)) — a second object over the output's memory (OWNDATA False).
            return new NDCharArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };
        }

        /// <summary>
        ///     The chararray for a plain result NumPy creates as the subclass: the finalize rule on its dtype, then a chararray
        ///     over its storage — taken over when the result owns a buffer nothing else references (the chararray owns it
        ///     then, as NumPy's does), an alias otherwise (a view shares its buffer, and a storage object is never shared by
        ///     two live arrays).
        /// </summary>
        /// <param name="result">The plain result (its wrapper is reclaimed by the caller's scope).</param>
        /// <returns>The chararray.</returns>
        /// <exception cref="ValueError">The result's dtype is not string data.</exception>
        private static NDCharArray AsChararray(NDArray result)
        {
            NDCharArray.Finalize(result.dtype);
            var storage = result.Storage;
            // An owning storage object that another live array might also wrap (an engine shortcut can hand an operand's
            // own storage back in a second wrapper) holds more than this result's one reference: alias it then.
            if (!storage.OwnsData || !storage.InternalArray.IsUniquelyReferenced)
                storage = storage.Alias();
            return new NDCharArray(storage) { TensorEngine = result.TensorEngine };
        }

        #endregion

        #region NumPy's trivial-loop copy (check_for_trivial_loop)

        /// <summary>The first chararray input of the ufunc call <see cref="WrappedUfunc{TState}"/> is running, or null.</summary>
        [ThreadStatic] private static NDArray t_ufuncInput0;

        /// <summary>The second chararray input of that call, or null.</summary>
        [ThreadStatic] private static NDArray t_ufuncInput1;

        /// <summary>
        ///     NumPy's <c>check_for_trivial_loop</c> for a ufunc call over a chararray (ufunc_object.c): without a
        ///     <c>where=</c> mask, each input whose loop dtype differs from its own (or that is misaligned) is COPIED into the
        ///     loop's dtype when it is 0-d or 1-d with at most <c>np.getbufsize()</c> elements — and the copy
        ///     (<c>PyArray_CastToType</c>) keeps the subclass, so a chararray input copied into bool (the logical ufuncs) or
        ///     StringDType raises the finalize error. The first input that would need a copy but is larger refuses the trivial
        ///     loop: no later input is copied, and the call computes through the iterator (a plain result).
        /// </summary>
        /// <param name="x1">The first input.</param>
        /// <param name="loop1">The loop's descriptor for <paramref name="x1"/>.</param>
        /// <param name="x2">The second input, or null.</param>
        /// <param name="loop2">The loop's descriptor for <paramref name="x2"/> (ignored when it is null).</param>
        /// <param name="where">The <c>where=</c> mask, or null: a mask selects the masked loop, which never copies. The C#
        ///     literal <c>true</c> (its array marked <see cref="NDArray.WasPythonBool"/>) is NumPy's <c>where=True</c>, which
        ///     <c>_wheremask_converter</c> reads as no mask; every other value — <c>false</c>, a 0-d <c>np.array(True)</c>,
        ///     any bool array — is a mask.</param>
        /// <returns>The finalize error, or null when the call goes on.</returns>
        /// <remarks>
        ///     Run it where NumPy runs the copy: after the loop was resolved and its output cast into <c>out=</c> checked, before
        ///     the broadcast (so <c>np.logical_and(c, empty)</c> raises the finalize error, not the broadcast's). It only reads
        ///     the inputs <see cref="WrappedUfunc{TState}"/> marked, and consumes the marks.
        /// </remarks>
        internal static ValueError TrivialLoopCopyError(NDArray x1, DType loop1, NDArray x2, DType loop2, NDArray where)
        {
            NDArray mark0 = t_ufuncInput0, mark1 = t_ufuncInput1;
            if (mark0 is null && mark1 is null)
                return null;
            // The marks belong to the call's own guard: a nested ufunc of the computation must not read them.
            t_ufuncInput0 = t_ufuncInput1 = null;
            // _wheremask_converter: only the Python object True is "no mask" (obj == Py_True); np.array(True) is a mask.
            if (where is not null && !(where.WasPythonBool && where.GetAtIndex<bool>(0)))
                return null;
            long bufsize = np.getbufsize();
            if (CopyStep(x1, loop1, mark0, mark1, bufsize, out bool stop) is { } first)
                return first;
            if (stop || x2 is null)
                return null;
            return CopyStep(x2, loop2, mark0, mark1, bufsize, out _);
        }

        /// <summary>One input of <see cref="TrivialLoopCopyError"/>: copied, refused, or not concerned.</summary>
        /// <param name="op">The input.</param>
        /// <param name="loop">Its loop descriptor.</param>
        /// <param name="mark0">The first marked chararray input.</param>
        /// <param name="mark1">The second marked chararray input.</param>
        /// <param name="bufsize">NumPy's buffer size.</param>
        /// <param name="stop">Set when the input refuses the trivial loop (no later input is copied).</param>
        /// <returns>The finalize error when the copied input is a marked chararray and the loop dtype is not string data.</returns>
        private static ValueError CopyStep(NDArray op, DType loop, NDArray mark0, NDArray mark1, long bufsize, out bool stop)
        {
            stop = false;
            // A cast between two different descriptors is never a view here (text to bool, text to StringDType), so it
            // needs a copy; a misaligned input needs one too (PyArray_ISALIGNED).
            bool cast = loop is not null && !op.dtype.Equals(loop);
            bool aligned = (op.Shape.Flags & ArrayFlags.ALIGNED) != 0 && op.Storage.IsDataAligned;
            if (!cast && aligned)
                return null;
            if (op.ndim == 0 || op.ndim == 1 && op.size <= bufsize)
            {
                // The copy is made as the input's class: a chararray copy into a non-string loop dtype is refused.
                if (cast && (ReferenceEquals(op, mark0) || ReferenceEquals(op, mark1)) && !NDCharArray.AcceptsDType(loop))
                    return FinalizeError();
                return null;
            }
            stop = true;
            return null;
        }

        #endregion
    }
}
