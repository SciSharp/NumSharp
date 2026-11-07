using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

// =============================================================================
// NDSubclass - NumPy's array-subclass mechanisms, woven into the public np.* functions
// =============================================================================
//
// NumPy finds a subclass operand (numpy.matrix, numpy.char.chararray) at run time and answers by the mechanism its
// function's implementation uses (NDSubclassAttributes.cs lists them). NumSharp states that mechanism on the computation
// each public function calls, and NumSharp.Build weaves the handling into the function itself. The woven function has a
// one-branch prologue and a generated sibling, the SUBCLASS PATH:
//
//     public static NDArray add(NDArray x1, NDArray x2, NDArray @out = null, ...)
//     {
//         if (NDSubclass.IsOperand(x1) || NDSubclass.IsOperand(x2) || NDSubclass.IsOperand(@out))   // woven
//             return <add>subclass(x1, x2, @out, ...);                                               // woven
//         return x1.TensorEngine.Add(x1, x2, dtype, @out, where);                                    // the source
//     }
//
//     private static NDArray <add>subclass(NDArray x1, NDArray x2, NDArray @out, ...)             // generated
//     {
//         NDSubclass.AuditEntry("add");
//         NDScope scope = NDScope.Open();
//         try
//         {
//             NDArray __x1 = NDSubclass.PlainUfuncInput(x1), __x2 = ..., __out = NDSubclass.Plain(@out);
//             NDArray r = add(__x1, __x2, __out, ...);           // the function again, on plain views
//             r = NDSubclass.Unalias(r, __x1, x1); ...           // an operand handed back is the caller's object
//             return scope.Returns(NDSubclass.ArrayWrap(r, @out, x1, x2, null));   // the declared mechanism
//         }
//         finally { scope.Dispose(); }
//     }
//
// THE PLAIN VIEW REPLACES THE DEPTH. The second call receives plain aliases (new NDArray objects over the operands'
// storage), so it takes the function's plain path, and every computation and public function below it meets plain arrays
// only: no per-thread state says "a public function is already handling this call". An alias remembers the subclass it
// stands for (NDArray.SubclassOrigin) for the few NumPy answers that react to the class inside the computation (np.block's
// matrix leaves, the unique family's stripping comparison, the trivial-loop copy of a ufunc's chararray input).
//
// THE SCOPE reclaims the aliases and the hidden plain result the mechanism wrapped (the subclass object holds its own
// counted reference on the buffer).
//
// THE MECHANISMS below are NumPy's class selection (npy_find_array_wrap's priorities, PyArray_GetSubType's) followed by
// the chosen operand's own class members, as NumPy puts them on its classes: NDArray's virtual __array_priority__,
// __array_wrap__, __array_finalize__ and _collapse (NDArray.SubclassHooks.cs), which NDMatrix and NDCharArray override.
// No mechanism switches on the class. They return the plain result unchanged when no operand is a subclass.
//
// THE CLASS'S OWN CODE computes the same way. An NDMatrix / NDCharArray member or operator that computes its answer itself
// hands the public functions it calls its plain view (NDMatrix.A, PlainView, Plain, PlainUfuncInput), never `this`: the
// function would take its subclass path, whose generated method dispatch can call the very member again. An NDArray
// member reached through a subclass object is [PlainReceiver]: woven to run on plain views of its receiver and operands.
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's array-subclass mechanisms for the public <c>np.*</c> functions, called by the code NumSharp.Build weaves
    ///     into them: the operand test of the prologue, the plain views of the subclass path, and the mechanism that turns
    ///     the plain result into NumPy's answer.
    /// </summary>
    /// <remarks>
    ///     Plain arrays pay one or two exact type tests per operand (method-table compares) and nothing else. Library code
    ///     calls the mechanisms only through the woven code; a computation that must react to a subclass operand reads
    ///     <see cref="StandsForMatrix(NDArray)"/> / <see cref="StandsForCharArray(NDArray)"/>.
    /// </remarks>
    internal static class NDSubclass
    {
        #region Operand tests

        /// <summary>
        ///     Which subclass <paramref name="a"/> is as an OPERAND: a matrix with at least one axis, or a chararray that is not
        ///     the scalar stand-in of an element access. A 0-d <see cref="NDMatrix"/> (a C# number converted to the class) and
        ///     a chararray's element stand-in are NumPy scalars, not subclass operands.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>The subclass kind, or <see cref="NDSubclassKind.None"/> for a plain array, a scalar stand-in or
        ///     <c>null</c>.</returns>
        /// <remarks>Exact type tests: neither class has a subclass in the library, and an <c>is</c> test on a non-sealed
        ///     class would walk the type hierarchy of every plain array that passes through a public function.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDSubclassKind KindOf(NDArray a)
        {
            if (a is null)
                return NDSubclassKind.None;
            Type t = a.GetType();
            if (t == typeof(NDMatrix))
                return a.Shape.NDim != 0 ? NDSubclassKind.Matrix : NDSubclassKind.None;
            if (t == typeof(NDCharArray))
                return ((NDCharArray)a).IsScalarStandIn ? NDSubclassKind.None : NDSubclassKind.CharArray;
            return NDSubclassKind.None;
        }

        /// <summary>
        ///     The woven prologue's test: whether <paramref name="a"/> is a subclass operand the function must answer for
        ///     (see <see cref="KindOf"/>). False for every plain array.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>True when the call takes the subclass path.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsOperand(NDArray a)
            => KindOf(a) != NDSubclassKind.None;

        /// <summary><see cref="IsOperand(NDArray)"/> for operands passed as one sequence: true when any element is one.</summary>
        /// <param name="arrays">The operands, or <c>null</c>; a <c>null</c> element is skipped.</param>
        /// <returns>True when the call takes the subclass path.</returns>
        internal static bool IsOperand(NDArray[] arrays)
        {
            if (arrays is null)
                return false;
            for (int i = 0; i < arrays.Length; i++)
                if (KindOf(arrays[i]) != NDSubclassKind.None)
                    return true;
            return false;
        }

        /// <summary>
        ///     <see cref="IsOperand(NDArray)"/> for operands that arrive in a NESTED sequence (<c>np.block</c>,
        ///     <c>np.r_</c>): true when any array at any depth is a subclass operand.
        /// </summary>
        /// <param name="nested">An array, or a sequence (<see cref="System.Collections.IList"/>) of arrays and sequences;
        ///     anything else holds no operand.</param>
        /// <returns>True when the call takes the subclass path.</returns>
        internal static bool IsNestedOperand(object nested)
            => HoldsOperand(nested);

        /// <summary>
        ///     Whether a nested sequence holds a subclass operand of <paramref name="kind"/> at any depth: the class
        ///     <c>PyArray_GetSubType</c> would pick for a function that concatenates the leaves (<c>np.block</c>, <c>np.r_</c>:
        ///     a matrix leaf wins).
        /// </summary>
        /// <param name="nested">An array, a sequence of arrays and sequences, or anything else.</param>
        /// <param name="kind">The subclass looked for.</param>
        /// <returns>True when a leaf is an operand of that class.</returns>
        internal static bool HoldsKind(object nested, NDSubclassKind kind)
        {
            if (nested is NDArray array)
                return KindOf(array) == kind;
            if (!IsArraySequence(nested))
                return false;
            var list = (System.Collections.IList)nested;
            for (int i = 0; i < list.Count; i++)
                if (HoldsKind(list[i], kind))
                    return true;
            return false;
        }

        /// <summary>Whether a nested sequence holds a subclass operand at any depth.</summary>
        /// <param name="nested">An array, a sequence, or anything else.</param>
        /// <returns>True when one is found.</returns>
        private static bool HoldsOperand(object nested)
        {
            if (nested is NDArray array)
                return KindOf(array) != NDSubclassKind.None;
            if (!IsArraySequence(nested))
                return false;
            var list = (System.Collections.IList)nested;
            for (int i = 0; i < list.Count; i++)
                if (HoldsOperand(list[i]))
                    return true;
            return false;
        }

        /// <summary>
        ///     Whether <paramref name="nested"/> is a sequence that can hold array objects: a one-dimensional CLR array of
        ///     references or a non-generic / reference-element list. A string (a sequence of chars in .NET), an array of
        ///     numbers, a rectangular array and a list of numbers hold none (and are never boxed element by element).
        /// </summary>
        /// <param name="nested">Anything.</param>
        /// <returns>True for a sequence worth descending into.</returns>
        private static bool IsArraySequence(object nested)
        {
            if (nested is Array clr)
                return clr.Rank == 1 && !clr.GetType().GetElementType().IsValueType;
            if (nested is not System.Collections.IList || nested is string)
                return false;
            Type listType = nested.GetType();
            return !(listType.IsGenericType && listType.GetGenericArguments()[0].IsValueType);
        }

        /// <summary>
        ///     Whether <paramref name="a"/> stands for a matrix operand inside a computation: the matrix object itself, or the
        ///     plain alias the woven subclass path made of one (<see cref="NDArray.SubclassOrigin"/>).
        /// </summary>
        /// <param name="a">The array, or <c>null</c>.</param>
        /// <returns>True for a matrix operand or its alias.</returns>
        /// <remarks>For the computations whose NumPy answer reacts to the class on the way (<c>np.block</c> keeps a matrix leaf
        ///     at two axes); a result derived from the alias does not stand for the matrix.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool StandsForMatrix(NDArray a)
            => a is not null && (a.SubclassOrigin == NDSubclassKind.Matrix || KindOf(a) == NDSubclassKind.Matrix);

        /// <summary>
        ///     Whether <paramref name="a"/> stands for a chararray operand inside a computation: the chararray object itself, or
        ///     the plain alias the woven subclass path made of one.
        /// </summary>
        /// <param name="a">The array, or <c>null</c>.</param>
        /// <returns>True for a chararray operand or its alias.</returns>
        /// <remarks>For the computations whose NumPy answer reacts to the class on the way (the unique family's stripping
        ///     comparison of runs, the weighted quantile's refusal of chararray weights).</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool StandsForCharArray(NDArray a)
            => a is not null && (a.SubclassOrigin == NDSubclassKind.CharArray || KindOf(a) == NDSubclassKind.CharArray);

        #endregion

        #region Plain views

        /// <summary>
        ///     The plain view the subclass path hands the computation in place of a subclass operand: a NEW
        ///     <see cref="NDArray"/> over the operand's storage that remembers the subclass it stands for. Any other array
        ///     (plain, a scalar stand-in, <c>null</c>) is returned itself.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>The alias, or <paramref name="a"/>.</returns>
        /// <remarks>The alias holds its own counted reference on the buffer: the subclass path's scope reclaims it. Writes
        ///     through it reach the operand (an <c>out=</c> matrix receives the values).</remarks>
        internal static NDArray Plain(NDArray a)
        {
            NDSubclassKind kind = KindOf(a);
            if (kind == NDSubclassKind.None)
                return a;
            return new NDArray(a.Storage.Alias()) { TensorEngine = a.TensorEngine, SubclassOrigin = kind };
        }

        /// <summary>
        ///     <see cref="Plain(NDArray)"/> for an INPUT of a ufunc call: a chararray input's alias is also marked for NumPy's
        ///     trivial-loop copy (<see cref="NDArray.IsTextUfuncInput"/>), which the string ufunc's guard reads.
        /// </summary>
        /// <param name="a">The input, or <c>null</c>.</param>
        /// <returns>The alias, or <paramref name="a"/>.</returns>
        internal static NDArray PlainUfuncInput(NDArray a)
        {
            NDArray alias = Plain(a);
            if (alias is not null && alias.SubclassOrigin == NDSubclassKind.CharArray)
                alias.IsTextUfuncInput = true;
            return alias;
        }

        /// <summary>
        ///     <see cref="Plain(NDArray)"/> for operands passed as one sequence: a NEW array holding each element's plain view
        ///     when any element is a subclass operand; the sequence itself otherwise (the caller's array is never modified).
        /// </summary>
        /// <param name="arrays">The operands, or <c>null</c>.</param>
        /// <returns>The plain sequence.</returns>
        internal static NDArray[] Plain(NDArray[] arrays)
        {
            if (arrays is null)
                return null;
            NDArray[] plain = null;
            for (int i = 0; i < arrays.Length; i++)
            {
                if (KindOf(arrays[i]) == NDSubclassKind.None)
                    continue;
                plain ??= (NDArray[])arrays.Clone();
                plain[i] = Plain(arrays[i]);
            }
            return plain ?? arrays;
        }

        /// <summary>
        ///     <see cref="Plain(NDArray)"/> for operands in a NESTED sequence (<c>np.block</c>'s lists of lists): a copy of every
        ///     sequence that holds a subclass operand, with each operand replaced by its plain view; anything else is returned
        ///     itself.
        /// </summary>
        /// <param name="nested">An array, a sequence of arrays and sequences, or anything else.</param>
        /// <returns>The plain structure.</returns>
        /// <remarks>A CLR array is copied as an <c>object[]</c> of the same length and a list as a <c>List&lt;object&gt;</c>:
        ///     the nested-sequence readers of <c>np.block</c> and <c>np.r_</c> accept both, and the caller's sequences stay
        ///     untouched.</remarks>
        internal static object PlainNested(object nested)
        {
            if (nested is NDArray array)
                return Plain(array);
            if (!IsArraySequence(nested) || !HoldsOperand(nested))
                return nested;
            var list = (System.Collections.IList)nested;
            if (nested is Array)
            {
                var copy = new object[list.Count];
                for (int i = 0; i < copy.Length; i++)
                    copy[i] = PlainNested(list[i]);
                return copy;
            }
            var listCopy = new System.Collections.Generic.List<object>(list.Count);
            for (int i = 0; i < list.Count; i++)
                listCopy.Add(PlainNested(list[i]));
            return listCopy;
        }

        /// <summary>
        ///     The plain view a CONTAINER keeps of an operand (numpy.ma's data and mask, a sampler's converted parameter): the
        ///     operand itself when it is plain, otherwise a new plain view over its storage, which the container owns.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>The plain array to keep.</returns>
        /// <remarks>Unlike <see cref="Plain(NDArray)"/> the view carries no subclass origin: the container computes with it
        ///     long after the user's call, and nothing it computes later is that call's subclass answer.</remarks>
        internal static NDArray PlainView(NDArray a)
            => KindOf(a) == NDSubclassKind.None ? a : new NDArray(a.Storage.Alias()) { TensorEngine = a.TensorEngine };

        /// <summary>
        ///     <see cref="PlainView(NDArray)"/> for operands passed as one sequence (a <c>[PlainReceiver]</c> member's
        ///     <c>NDArray[]</c> parameter): a NEW array holding each subclass element's plain view when any element is one;
        ///     the sequence itself otherwise (the caller's array is never modified).
        /// </summary>
        /// <param name="arrays">The operands, or <c>null</c>.</param>
        /// <returns>The plain sequence.</returns>
        internal static NDArray[] PlainView(NDArray[] arrays)
        {
            if (arrays is null)
                return null;
            NDArray[] plain = null;
            for (int i = 0; i < arrays.Length; i++)
            {
                if (KindOf(arrays[i]) == NDSubclassKind.None)
                    continue;
                plain ??= (NDArray[])arrays.Clone();
                plain[i] = PlainView(arrays[i]);
            }
            return plain ?? arrays;
        }

        /// <summary>
        ///     <see cref="PlainView(NDArray)"/> for code that holds no <see cref="NDScope"/> and calls woven public functions
        ///     with a caller's operand (the numpy.polynomial package, which computes with plain arrays by design): the plain
        ///     view of a subclass operand, released when the lease is disposed; a plain operand is handed through and never
        ///     disposed.
        /// </summary>
        /// <param name="a">The operand, or <c>null</c>.</param>
        /// <returns>The lease: compute with <see cref="PlainLease.Array"/>, and dispose the lease (<c>using</c>) once the
        ///     computation is done.</returns>
        /// <remarks>Releasing the view never frees a result that views it (every array holds its own counted reference on the
        ///     buffer), but <see cref="PlainLease.Array"/> itself must not outlive the lease: never return it.</remarks>
        internal static PlainLease LeasePlainView(NDArray a) => new PlainLease(a);

        /// <summary>
        ///     A plain view leased by <see cref="LeasePlainView"/>: the array to compute with, and whether the lease made it
        ///     (and so releases it on <see cref="Dispose"/>).
        /// </summary>
        internal readonly struct PlainLease : IDisposable
        {
            /// <summary>The array to compute with: the operand's plain view, or the operand itself when it is
            ///     plain.</summary>
            internal readonly NDArray Array;

            /// <summary>Whether <see cref="Array"/> is a view this lease made, released by <see cref="Dispose"/>.</summary>
            private readonly bool _made;

            /// <summary>Leases the plain view of <paramref name="operand"/>.</summary>
            /// <param name="operand">The caller's operand, or <c>null</c>.</param>
            internal PlainLease(NDArray operand)
            {
                Array = PlainView(operand);
                _made = !ReferenceEquals(Array, operand);
            }

            /// <summary>Releases the view this lease made (its own counted reference on the buffer). A plain operand is the
            ///     caller's and is left alone; a second call is harmless (<see cref="NDArray.Dispose"/> is
            ///     idempotent).</summary>
            public void Dispose()
            {
                if (_made)
                    Array.Dispose();
            }
        }

        /// <summary>
        ///     For a subclass path that runs WITHOUT a scope (its result - an iterator over the operand, a broadcast object -
        ///     may keep the plain view): detaches the view from any ambient scope, so a scoped caller cannot dispose it under
        ///     the result; the finalizer reclaims it once nothing references it.
        /// </summary>
        /// <param name="alias">The plain view the path made (or the operand itself when it was plain).</param>
        /// <param name="original">The caller's operand.</param>
        internal static void KeepAlias(NDArray alias, NDArray original)
        {
            if (alias is not null && !ReferenceEquals(alias, original))
                NDScope.Detach(alias);
        }

        /// <summary><see cref="KeepAlias(NDArray, NDArray)"/> for a sequence operand's plain copy: every element that is a
        ///     view the path made (the caller's own elements are left to the caller's scope).</summary>
        /// <param name="aliases">The plain sequence the path made.</param>
        /// <param name="originals">The caller's sequence.</param>
        internal static void KeepAlias(NDArray[] aliases, NDArray[] originals)
        {
            if (aliases is null || ReferenceEquals(aliases, originals))
                return;
            for (int i = 0; i < aliases.Length; i++)
                if (originals is null || i >= originals.Length || !ReferenceEquals(aliases[i], originals[i]))
                    NDScope.Detach(aliases[i]);
        }

        #endregion

        #region Unaliasing (an operand handed back is the caller's object)

        /// <summary>
        ///     The result the subclass path continues with: the CALLER's operand when the computation handed back the plain
        ///     view of it (a function that returns its operand, <c>np.asanyarray(m)</c>, a ufunc's <c>out=</c>), the result
        ///     itself otherwise.
        /// </summary>
        /// <param name="result">What the function returned on the plain views.</param>
        /// <param name="alias">One plain view the subclass path made (or the operand itself when it was plain).</param>
        /// <param name="original">The caller's operand that view stands for.</param>
        /// <returns><paramref name="original"/> or <paramref name="result"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static NDArray Unalias(NDArray result, NDArray alias, NDArray original)
            => ReferenceEquals(result, alias) ? original : result;

        /// <summary><see cref="Unalias(NDArray, NDArray, NDArray)"/> for each element of a multi-array result, in place.</summary>
        /// <param name="result">The results (a new array the computation built; never the caller's).</param>
        /// <param name="alias">One plain view the subclass path made.</param>
        /// <param name="original">The caller's operand that view stands for.</param>
        /// <returns><paramref name="result"/>.</returns>
        internal static NDArray[] Unalias(NDArray[] result, NDArray alias, NDArray original)
        {
            if (result is null || ReferenceEquals(alias, original))
                return result;
            for (int i = 0; i < result.Length; i++)
                if (ReferenceEquals(result[i], alias))
                    result[i] = original;
            return result;
        }

        /// <summary>
        ///     <see cref="Unalias(NDArray, NDArray, NDArray)"/> for a SEQUENCE operand: when the result is the plain view of one
        ///     of its elements, the caller's element at that position.
        /// </summary>
        /// <param name="result">What the function returned on the plain sequence.</param>
        /// <param name="aliases">The plain sequence the subclass path made (the caller's own array when no element was a
        ///     subclass operand).</param>
        /// <param name="originals">The caller's sequence.</param>
        /// <returns>The caller's element, or <paramref name="result"/>.</returns>
        internal static NDArray Unalias(NDArray result, NDArray[] aliases, NDArray[] originals)
        {
            // Nothing was replaced when the sequence held no subclass operand (Plain returned it itself).
            if (result is null || aliases is null || ReferenceEquals(aliases, originals))
                return result;
            for (int i = 0; i < aliases.Length && i < originals.Length; i++)
                if (ReferenceEquals(result, aliases[i]))
                    return originals[i];
            return result;
        }

        /// <summary>
        ///     <see cref="Unalias(NDArray, NDArray[], NDArray[])"/> for each element of a multi-array result, in place
        ///     (<c>np.atleast_1d(m, c)</c> hands back each array that already has the rank).
        /// </summary>
        /// <param name="result">The results (a new array the computation built, or the plain sequence itself).</param>
        /// <param name="aliases">The plain sequence the subclass path made.</param>
        /// <param name="originals">The caller's sequence.</param>
        /// <returns><paramref name="result"/>, every element that is a plain view replaced by the caller's element.</returns>
        /// <remarks>Quadratic in the two lengths: it runs on the subclass path only, over a call's handful of operands.</remarks>
        internal static NDArray[] Unalias(NDArray[] result, NDArray[] aliases, NDArray[] originals)
        {
            if (result is null || aliases is null || ReferenceEquals(aliases, originals))
                return result;
            for (int r = 0; r < result.Length; r++)
                for (int i = 0; i < aliases.Length && i < originals.Length; i++)
                    if (ReferenceEquals(result[r], aliases[i]))
                    {
                        result[r] = originals[i];
                        break;
                    }
            return result;
        }

        /// <summary><see cref="Unalias(NDArray, NDArray, NDArray)"/> for each slot of a two-array result.</summary>
        /// <param name="result">The pair.</param>
        /// <param name="alias">One plain view the subclass path made.</param>
        /// <param name="original">The caller's operand that view stands for.</param>
        /// <returns>The pair with the view replaced.</returns>
        internal static (NDArray, NDArray) Unalias((NDArray, NDArray) result, NDArray alias, NDArray original)
            => (Unalias(result.Item1, alias, original), Unalias(result.Item2, alias, original));

        /// <summary><see cref="Unalias(NDArray, NDArray, NDArray)"/> for each slot of a three-array result.</summary>
        /// <param name="result">The triple.</param>
        /// <param name="alias">One plain view the subclass path made.</param>
        /// <param name="original">The caller's operand that view stands for.</param>
        /// <returns>The triple with the view replaced.</returns>
        internal static (NDArray, NDArray, NDArray) Unalias((NDArray, NDArray, NDArray) result, NDArray alias, NDArray original)
            => (Unalias(result.Item1, alias, original), Unalias(result.Item2, alias, original),
                Unalias(result.Item3, alias, original));

        /// <summary><see cref="Unalias(NDArray, NDArray, NDArray)"/> for each slot of a four-array result.</summary>
        /// <param name="result">The four arrays.</param>
        /// <param name="alias">One plain view the subclass path made.</param>
        /// <param name="original">The caller's operand that view stands for.</param>
        /// <returns>The four arrays with the view replaced.</returns>
        internal static (NDArray, NDArray, NDArray, NDArray) Unalias((NDArray, NDArray, NDArray, NDArray) result, NDArray alias,
                                                                    NDArray original)
            => (Unalias(result.Item1, alias, original), Unalias(result.Item2, alias, original),
                Unalias(result.Item3, alias, original), Unalias(result.Item4, alias, original));

        #endregion

        #region __array_wrap__ (ufuncs, _array_converter.wrap, numpy.linalg's wrap)

        /// <summary>
        ///     NumPy's <c>npy_find_array_wrap</c> + <c>npy_apply_wrap</c> (force_wrap, as a ufunc passes it). A result that is
        ///     the caller's <paramref name="out"/> is wrapped by out's own class: a matrix out (and a plain one) comes back
        ///     itself, a chararray out as a NEW chararray view (<c>np.add(c, c, out=co) is co</c> is False in NumPy). Any other
        ///     result is wrapped by the input chosen by priority: a matrix view (the finalize rule: a 1-d result becomes a
        ///     row), or a chararray view of a TEXT result (a numeric result stays plain).
        /// </summary>
        /// <param name="result">The function's result on the plain views (unaliased).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="x1">The first input whose class may wrap, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <returns>NumPy's answer; <paramref name="result"/> itself when no input is a subclass operand.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1 (<c>shape too large to be a
        ///     matrix.</c>); a chararray out whose dtype is not string data.</exception>
        internal static NDArray ArrayWrap(NDArray result, NDArray @out, NDArray x1, NDArray x2, NDArray x3)
        {
            if (result is null)
                return null;
            // A result that is out= is wrapped by out's own class: a matrix out comes back itself, a chararray out as a
            // NEW chararray view (chararray.__array_wrap__).
            if (@out is not null && ReferenceEquals(result, @out))
                return KindOf(@out) == NDSubclassKind.None ? @out : @out.__array_wrap__(@out);
            return WrapBy(result, FindWrap(x1, x2, x3));
        }

        /// <summary>
        ///     <see cref="ArrayWrap(NDArray, NDArray, NDArray, NDArray, NDArray)"/> for a multi-array result (<c>np.split</c>'s
        ///     pieces, <c>np.unique</c>'s tuple): the wrap of the chosen input applies to the selected slots.
        /// </summary>
        /// <param name="result">The results on the plain views (a new array; its slots are replaced in place).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c> (a slot that is it is wrapped by its own class).</param>
        /// <param name="x1">The first input whose class may wrap, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <param name="outputs">Bit <c>i</c> set: NumPy wraps slot <c>i</c> (<c>-1</c>: every slot).</param>
        /// <returns><paramref name="result"/>, the selected slots wrapped.</returns>
        /// <exception cref="ValueError">A wrapped matrix slot with more than two axes longer than 1.</exception>
        internal static NDArray[] ArrayWrap(NDArray[] result, NDArray @out, NDArray x1, NDArray x2, NDArray x3, int outputs)
        {
            if (result is null)
                return null;
            NDArray wrapper = FindWrap(x1, x2, x3);
            for (int i = 0; i < result.Length; i++)
                if (Selected(outputs, i))
                    result[i] = ReferenceEquals(result[i], @out) && @out is not null
                        ? ArrayWrap(result[i], @out, null, null, null)
                        : WrapBy(result[i], wrapper);
            return result;
        }

        /// <summary><see cref="ArrayWrap(NDArray[], NDArray, NDArray, NDArray, NDArray, int)"/> for a two-array result.</summary>
        /// <param name="result">The pair on the plain views.</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="x1">The first input whose class may wrap, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <param name="outputs">Bit <c>i</c> set: NumPy wraps slot <c>i</c> (<c>-1</c>: both).</param>
        /// <returns>The pair, the selected slots wrapped.</returns>
        /// <exception cref="ValueError">A wrapped matrix slot with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray) ArrayWrap((NDArray, NDArray) result, NDArray @out, NDArray x1, NDArray x2, NDArray x3,
                                                     int outputs)
        {
            NDArray wrapper = FindWrap(x1, x2, x3);
            return (Selected(outputs, 0) ? WrapBy(result.Item1, wrapper) : result.Item1,
                    Selected(outputs, 1) ? WrapBy(result.Item2, wrapper) : result.Item2);
        }

        /// <summary><see cref="ArrayWrap(NDArray[], NDArray, NDArray, NDArray, NDArray, int)"/> for a three-array result.</summary>
        /// <param name="result">The triple on the plain views.</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="x1">The first input whose class may wrap, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <param name="outputs">Bit <c>i</c> set: NumPy wraps slot <c>i</c> (<c>-1</c>: every slot).</param>
        /// <returns>The triple, the selected slots wrapped.</returns>
        /// <exception cref="ValueError">A wrapped matrix slot with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray, NDArray) ArrayWrap((NDArray, NDArray, NDArray) result, NDArray @out, NDArray x1,
                                                              NDArray x2, NDArray x3, int outputs)
        {
            NDArray wrapper = FindWrap(x1, x2, x3);
            return (Selected(outputs, 0) ? WrapBy(result.Item1, wrapper) : result.Item1,
                    Selected(outputs, 1) ? WrapBy(result.Item2, wrapper) : result.Item2,
                    Selected(outputs, 2) ? WrapBy(result.Item3, wrapper) : result.Item3);
        }

        /// <summary><see cref="ArrayWrap(NDArray[], NDArray, NDArray, NDArray, NDArray, int)"/> for a four-array result.</summary>
        /// <param name="result">The four arrays on the plain views.</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="x1">The first input whose class may wrap, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <param name="outputs">Bit <c>i</c> set: NumPy wraps slot <c>i</c> (<c>-1</c>: every slot).</param>
        /// <returns>The four arrays, the selected slots wrapped.</returns>
        /// <exception cref="ValueError">A wrapped matrix slot with more than two axes longer than 1.</exception>
        internal static (NDArray, NDArray, NDArray, NDArray) ArrayWrap((NDArray, NDArray, NDArray, NDArray) result, NDArray @out,
                                                                       NDArray x1, NDArray x2, NDArray x3, int outputs)
        {
            NDArray wrapper = FindWrap(x1, x2, x3);
            return (Selected(outputs, 0) ? WrapBy(result.Item1, wrapper) : result.Item1,
                    Selected(outputs, 1) ? WrapBy(result.Item2, wrapper) : result.Item2,
                    Selected(outputs, 2) ? WrapBy(result.Item3, wrapper) : result.Item3,
                    Selected(outputs, 3) ? WrapBy(result.Item4, wrapper) : result.Item4);
        }

        /// <summary>Whether bit <paramref name="slot"/> of an output selection is set (<c>-1</c> selects every slot).</summary>
        /// <param name="outputs">The selection.</param>
        /// <param name="slot">The zero-based slot.</param>
        /// <returns>True when the slot is wrapped.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool Selected(int outputs, int slot) => outputs == -1 || slot < 31 && (outputs & (1 << slot)) != 0;

        /// <summary>
        ///     NumPy's <c>npy_find_array_wrap</c> over the inputs: the first subclass operand wraps; a later one only with a
        ///     STRICTLY higher priority; a subclass at <c>NPY_PRIORITY</c> (a chararray, 0.0) still beats an exact ndarray seen
        ///     before it. So a matrix (10.0) wins over a chararray in either order.
        /// </summary>
        /// <param name="x1">The first input, or <c>null</c>.</param>
        /// <param name="x2">The second input, or <c>null</c>.</param>
        /// <param name="x3">The third input, or <c>null</c>.</param>
        /// <returns>The wrapping operand, or <c>null</c> when every input is plain.</returns>
        private static NDArray FindWrap(NDArray x1, NDArray x2, NDArray x3)
        {
            NDArray wrapper = null;
            double priority = double.NegativeInfinity;
            Consider(x1, ref wrapper, ref priority);
            Consider(x2, ref wrapper, ref priority);
            Consider(x3, ref wrapper, ref priority);
            return wrapper;
        }

        /// <summary>One input of <see cref="FindWrap"/>'s loop.</summary>
        /// <param name="x">The input, or <c>null</c> (skipped).</param>
        /// <param name="wrapper">The subclass operand chosen so far, or <c>null</c>.</param>
        /// <param name="priority">The priority of the current choice (-inf before any).</param>
        private static void Consider(NDArray x, ref NDArray wrapper, ref double priority)
        {
            if (KindOf(x) == NDSubclassKind.None)
                return;   // an exact ndarray has no wrap; it never displaces a subclass
            double current = x.__array_priority__;
            if (wrapper is null || priority < current)
            {
                wrapper = x;
                priority = current;
            }
        }

        /// <summary>The chosen operand's <c>__array_wrap__</c> of one result.</summary>
        /// <param name="result">The plain result (or the caller's operand handed back), or <c>null</c>.</param>
        /// <param name="wrapper">The operand whose class wraps, or <c>null</c> (nothing wraps).</param>
        /// <returns>The wrapping operand's <c>__array_wrap__</c> of the result (a matrix view, the chararray view of a text
        ///     result), or <paramref name="result"/> when nothing wraps.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        private static NDArray WrapBy(NDArray result, NDArray wrapper)
            => result is null || wrapper is null ? result : wrapper.__array_wrap__(result);

        #endregion

        #region Subtype creation (Py_TYPE(self))

        /// <summary>
        ///     NumPy's subtype creation: the result is created as <paramref name="self"/>'s class (a matrix view with the
        ///     finalize rule; a chararray after the finalize rule on the result's dtype, owning its data exactly when the
        ///     result does). A result that is the caller's <paramref name="out"/>, or already of the class (the operand handed
        ///     back), comes back itself.
        /// </summary>
        /// <param name="result">The function's result on the plain views (unaliased).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="self">The operand NumPy calls the method on.</param>
        /// <param name="keepRank">Skip the matrix finalize rule (NumPy indexed the matrix with a new axis).</param>
        /// <param name="zeroDim">How a 0-d result reaches the caller for a chararray (the element's scalar stand-in for
        ///     <see cref="NewSubtypeZeroDim.Scalar"/> / <see cref="NewSubtypeZeroDim.Element"/>).</param>
        /// <returns>NumPy's answer; <paramref name="result"/> itself when <paramref name="self"/> is plain.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1; a chararray result whose dtype
        ///     is not string data (<c>Can only create a chararray from string data.</c>).</exception>
        internal static NDArray NewSubtype(NDArray result, NDArray @out, NDArray self, bool keepRank, NewSubtypeZeroDim zeroDim)
        {
            if (result is null || @out is not null && ReferenceEquals(result, @out))
                return result;
            // Py_TYPE(self) creation and its __array_finalize__: the operand's class member (a plain operand creates
            // nothing).
            return KindOf(self) == NDSubclassKind.None ? result : self.__array_finalize__(result, keepRank, zeroDim);
        }

        /// <summary><see cref="NewSubtype(NDArray, NDArray, NDArray, bool, NewSubtypeZeroDim)"/> for each element of a
        ///     multi-array result (<c>np.split</c>'s pieces: each created as the operand's class).</summary>
        /// <param name="result">The results on the plain views (a new array; its slots are replaced in place).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="self">The operand NumPy calls the method on.</param>
        /// <param name="keepRank">Skip the matrix finalize rule.</param>
        /// <param name="zeroDim">How a 0-d piece reaches the caller for a chararray.</param>
        /// <returns><paramref name="result"/>, each piece created as the class.</returns>
        /// <exception cref="ValueError">A matrix piece with more than two axes longer than 1; a chararray piece whose dtype is
        ///     not string data.</exception>
        internal static NDArray[] NewSubtype(NDArray[] result, NDArray @out, NDArray self, bool keepRank, NewSubtypeZeroDim zeroDim)
        {
            if (result is null || KindOf(self) == NDSubclassKind.None)
                return result;
            for (int i = 0; i < result.Length; i++)
                result[i] = NewSubtype(result[i], @out, self, keepRank, zeroDim);
            return result;
        }

        /// <summary>
        ///     <see cref="NewSubtype(NDArray, NDArray, NDArray, bool, NewSubtypeZeroDim)"/> for a function that returns a C#
        ///     number (<c>np.argmax(c)</c> returns the flat index as a <see cref="long"/>): NumPy creates the 0-d index result
        ///     as the operand's class before <c>PyArray_Return</c> turns it into a scalar, so a chararray's finalize rule
        ///     refuses it; any other operand keeps the number.
        /// </summary>
        /// <param name="result">The computed index.</param>
        /// <param name="self">The operand NumPy calls the method on.</param>
        /// <returns><paramref name="result"/>.</returns>
        /// <exception cref="ValueError"><paramref name="self"/> is a chararray (<c>Can only create a chararray from string
        ///     data.</c>), raised after the computation's own checks, as in NumPy.</exception>
        internal static long NewSubtype(long result, NDArray self)
        {
            // The 0-d int64 result created as the operand's class: its finalize check (a chararray's refuses it).
            if (KindOf(self) != NDSubclassKind.None)
                self.__array_finalize__(DType.Int64);
            return result;
        }

        /// <summary>
        ///     The entry check of a function that creates its result as the operand's class BEFORE computing
        ///     (<c>np.zeros_like(c, dtype)</c>: <c>PyArray_NewLikeArray</c> runs the finalize rule on the new dtype first).
        /// </summary>
        /// <param name="self">The operand the result is created like.</param>
        /// <param name="dtype">The result dtype the caller asked for, or <c>null</c> for the operand's own.</param>
        /// <exception cref="ValueError"><paramref name="self"/> is a chararray and the dtype is not string data.</exception>
        internal static void FinalizeFirst(NDArray self, DType dtype)
        {
            if (KindOf(self) != NDSubclassKind.None)
                self.__array_finalize__(dtype ?? self.dtype);
        }

        #endregion

        #region PyArray_GetSubType (C functions over a sequence)

        /// <summary>
        ///     NumPy's <c>PyArray_GetSubType</c>: the result is created as the subtype of STRICTLY the highest priority above
        ///     ndarray's 0.0 among <paramref name="arrays"/> - a matrix (10.0) wins, a chararray (0.0) never does. A result
        ///     that is the caller's <paramref name="out"/> comes back itself.
        /// </summary>
        /// <param name="result">The function's result on the plain views (unaliased).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c>.</param>
        /// <param name="arrays">The caller's operands (<c>null</c> elements skipped).</param>
        /// <returns>The matrix view of the result, or the result itself.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        internal static NDArray GetSubType(NDArray result, NDArray @out, NDArray[] arrays)
        {
            if (result is null || arrays is null || @out is not null && ReferenceEquals(result, @out))
                return result;
            // The subtype of STRICTLY the highest __array_priority__ above ndarray's NPY_PRIORITY (0.0): a chararray never
            // wins, a matrix does; the first of equal priorities is kept.
            NDArray subtype = null;
            double priority = 0.0;
            for (int i = 0; i < arrays.Length; i++)
            {
                NDArray candidate = arrays[i];
                if (KindOf(candidate) == NDSubclassKind.None || candidate.__array_priority__ <= priority)
                    continue;
                subtype = candidate;
                priority = candidate.__array_priority__;
            }
            return subtype is null ? result : subtype.__array_finalize__(result, keepRank: false, NewSubtypeZeroDim.Array);
        }

        #endregion

        #region asarray

        /// <summary>
        ///     NumPy's <c>asarray</c> answer: a plain result. When the computation handed the caller's subclass operand back,
        ///     a NEW plain view of it, as <c>np.asarray(m)</c> is never <c>m</c>.
        /// </summary>
        /// <param name="result">The function's result on the plain views (unaliased).</param>
        /// <returns>The plain answer.</returns>
        internal static NDArray AsArray(NDArray result)
            // array(a, copy=False, subok=False) of a subclass object: a base-class view of its storage (a matrix's A is the
            // same view), never the object itself.
            => KindOf(result) == NDSubclassKind.None
                ? result
                : new NDArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };

        /// <summary><see cref="AsArray(NDArray)"/> for each element of a multi-array result, in place.</summary>
        /// <param name="result">The results on the plain views (a new array).</param>
        /// <returns><paramref name="result"/>.</returns>
        internal static NDArray[] AsArray(NDArray[] result)
        {
            if (result is not null)
                for (int i = 0; i < result.Length; i++)
                    result[i] = AsArray(result[i]);
            return result;
        }

        /// <summary><see cref="AsArray(NDArray)"/> for each slot of a two-array result (<c>np.broadcast_arrays(lhs, rhs)</c>).</summary>
        /// <param name="result">The pair on the plain views (unaliased).</param>
        /// <returns>The pair, each slot plain.</returns>
        internal static (NDArray, NDArray) AsArray((NDArray, NDArray) result)
            => (AsArray(result.Item1), AsArray(result.Item2));

        #endregion

        #region Reductions through the operand's method (_collapse)

        /// <summary>
        ///     The entry check of a function whose answer passes through <c>np.{method}</c> of a matrix intermediate
        ///     (<see cref="ReducesThroughAttribute"/>): NumPy forwards <c>keepdims</c> to the matrix method, which has no such
        ///     keyword.
        /// </summary>
        /// <param name="self">The operand.</param>
        /// <param name="keepdims">The caller's <c>keepdims</c>.</param>
        /// <param name="method">The method the keyword reaches, or <c>null</c> when none does.</param>
        /// <param name="inexactMethod">The method reached first for an inexact operand, or <c>null</c>.</param>
        /// <exception cref="TypeError"><paramref name="self"/> is a matrix and <paramref name="keepdims"/> is true:
        ///     <c>matrix.{method}() got an unexpected keyword argument 'keepdims'</c>, before anything is computed.</exception>
        /// <remarks>C# cannot tell an omitted <c>keepdims</c> from <c>keepdims: false</c>, and NumPy raises for both spellings
        ///     of a PASSED keyword; only <c>keepdims: true</c> raises here.</remarks>
        internal static void ReducesThroughCheck(NDArray self, bool keepdims, string method, string inexactMethod)
        {
            if (!keepdims || KindOf(self) != NDSubclassKind.Matrix)
                return;
            // _replace_nan returns a mask for an inexact dtype, and the masked path calls np.sum(~mask, keepdims=...) first.
            string reached = inexactMethod is not null && self.typecode is NPTypeCode.Half or NPTypeCode.Single
                                 or NPTypeCode.Double or NPTypeCode.Complex
                ? inexactMethod
                : method;
            if (reached is not null)
                throw KeywordError(NDMatrix.__name__, reached, "keepdims");
        }

        /// <summary>
        ///     <c>matrix._collapse(axis)</c> after NumPy's <c>keepdims=True</c> reduction, applied to the plain (keepdims=False)
        ///     result: without an axis the scalar, along the last axis a column <c>(N, 1)</c>, otherwise a row or the matrix
        ///     itself. Any other operand keeps the result.
        /// </summary>
        /// <param name="result">The plain reduction (unaliased).</param>
        /// <param name="out">The caller's <c>out=</c>, or <c>null</c> (handed to the other classes' subtype creation).</param>
        /// <param name="self">The operand.</param>
        /// <param name="axis">The caller's axis: it decides row or column for a 1-d plain result.</param>
        /// <param name="subtypeOthers">Whether a non-matrix subclass operand gets
        ///     <see cref="NewSubtype(NDArray, NDArray, NDArray, bool, NewSubtypeZeroDim)"/> (NumPy's body creates its result as
        ///     that class: <c>np.percentile</c> of a chararray partitions a subtype copy) instead of the plain result.</param>
        /// <param name="zeroDim">The 0-d form of that subtype creation.</param>
        /// <returns>The 0-d result, a matrix view of it, the other class's result, or <paramref name="result"/>.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1; a chararray result whose dtype
        ///     is not string data (<c>Can only create a chararray from string data.</c>).</exception>
        internal static NDArray Collapse(NDArray result, NDArray @out, NDArray self, int? axis, bool subtypeOthers,
                                         NewSubtypeZeroDim zeroDim)
        {
            if (KindOf(self) == NDSubclassKind.None)
                return result;
            // The class's _collapse (numpy.matrix's), else the class NumPy's body creates its result as (a chararray's
            // subtype copy) or the plain result.
            return self._collapse(result, axis)
                   ?? (subtypeOthers ? NewSubtype(result, @out, self, false, zeroDim) : result);
        }

        /// <summary>
        ///     <see cref="Collapse(NDArray, NDArray, NDArray, int?, bool, NewSubtypeZeroDim)"/> for a function that returns a C#
        ///     number (<c>np.nanargmax(a)</c>'s flat index): a matrix's <c>_collapse(None)</c> is that scalar itself; a
        ///     chararray's subtype creation refuses the int64 index (<see cref="NewSubtype(long, NDArray)"/>).
        /// </summary>
        /// <param name="result">The computed number.</param>
        /// <param name="out">Unused (a number has no <c>out=</c>); kept for the woven call's uniform shape.</param>
        /// <param name="self">The operand.</param>
        /// <param name="axis">Unused: a number is the scalar of every axis.</param>
        /// <param name="subtypeOthers">Whether a non-matrix subclass operand gets the subtype creation.</param>
        /// <param name="zeroDim">Unused for a number.</param>
        /// <returns><paramref name="result"/>.</returns>
        /// <exception cref="ValueError"><paramref name="self"/> is a chararray and <paramref name="subtypeOthers"/> is set
        ///     (<c>Can only create a chararray from string data.</c>), after the computation's own checks.</exception>
        internal static long Collapse(long result, NDArray @out, NDArray self, int? axis, bool subtypeOthers, NewSubtypeZeroDim zeroDim)
            => subtypeOthers ? NewSubtype(result, self) : result;

        /// <summary>
        ///     NumPy's error for a keyword a subclass method does not take (<c>np.sum(m, keepdims=True)</c>).
        /// </summary>
        /// <param name="pyClass">The subclass's Python name (<c>matrix</c>, <c>chararray</c>).</param>
        /// <param name="method">The method NumPy calls.</param>
        /// <param name="keyword">The keyword it was given.</param>
        /// <returns>The exception to throw, with NumPy's text.</returns>
        internal static TypeError KeywordError(string pyClass, string method, string keyword)
            => new TypeError($"{pyClass}.{method}() got an unexpected keyword argument '{keyword}'");

        #endregion

        #region Woven-code conversions

        /// <summary>
        ///     The <c>[AsArrayFor]</c> test of the woven code: whether NumPy's function converts <paramref name="self"/>'s class
        ///     with <c>asarray</c> first (an <c>isinstance</c> test in its body).
        /// </summary>
        /// <param name="self">The operand.</param>
        /// <param name="kinds">The converted classes, as bits <c>1 &lt;&lt; (int)NDSubclassKind</c>.</param>
        /// <returns>True when the operand is one of them.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool ConvertsWithAsArray(NDArray self, int kinds)
        {
            NDSubclassKind kind = KindOf(self);
            return kind != NDSubclassKind.None && (kinds & (1 << (int)kind)) != 0;
        }

        /// <summary>An <c>int</c> axis as the <c>int?</c> a mechanism or a subclass method takes (the woven code's
        ///     conversion: generating <c>Nullable&lt;int&gt;</c>'s constructor in IL is avoided).</summary>
        /// <param name="axis">The axis.</param>
        /// <returns>The same axis, as a nullable.</returns>
        internal static int? Axis(int axis) => axis;

        /// <summary>The <c>int?</c> axis of a function that has none (the woven code's <c>null</c> for an <c>int?</c>
        ///     argument).</summary>
        /// <returns><c>null</c>.</returns>
        internal static int? NoAxis() => null;

        #endregion

        #region The caller audit (a test hook)

        /// <summary>
        ///     Test hook: while true, every call that takes the subclass path records its CALLER when that caller belongs to
        ///     this library (<see cref="InternalCallers"/>). Off in production (one static read on the subclass path only).
        /// </summary>
        /// <remarks>
        ///     The plain views rest on one assumption: library code never hands a subclass OBJECT to a public function - it
        ///     computes on the plain view the subclass path gave it, or on its own receiver's plain view. A library method that
        ///     passes a matrix or chararray on gets that function's subclass answer where it was written for a plain one. The
        ///     oracle project replays its corpus with subclass operands and requires the set to stay empty.
        /// </remarks>
        internal static bool AuditInternalCallers;

        /// <summary>What the audit found: <c>"{caller type}.{caller method} -&gt; {function}"</c> with the number of times.</summary>
        internal static readonly ConcurrentDictionary<string, int> InternalCallers = new();

        /// <summary>
        ///     The first statement of every woven subclass path: records the caller under the audit hook.
        /// </summary>
        /// <param name="function">The public function's name.</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void AuditEntry(string function)
        {
            if (AuditInternalCallers)
                Audit(function);
        }

        /// <summary>
        ///     Records the caller of the public function <paramref name="function"/> when that caller belongs to this
        ///     library.
        /// </summary>
        /// <param name="function">The public function taking the subclass path.</param>
        /// <remarks>
        ///     Reads the stack, so it runs only under the test hook. Frames of this class, the woven function and its
        ///     subclass path (named <c>function</c> and <c>&lt;function&gt;subclass</c>), other overloads of the same name
        ///     (a forwarding overload), and the subclasses' own members (<see cref="NDMatrix"/> and <see cref="NDCharArray"/>
        ///     call the public functions with their operands on purpose, exactly as NumPy's methods do) are skipped; the first
        ///     frame left is the caller. A <see cref="SubclassAwareAttribute"/> member on the way - or an override of one, such
        ///     as <see cref="NDCharArray"/>'s <c>NanMax</c>, which hands the chararray to np.amax as NumPy's nanmax does - is
        ///     NumPy's own class-reading code passing the object on by design, so the walk ends there and records nothing; it
        ///     is the one reader of that attribute at run time.
        ///     <para>The walk sees only the frames optimized code keeps: an inlined method has none, and a call in tail position
        ///     can become a jump that replaces the caller's frame. So every override of a <see cref="SubclassAwareAttribute"/>
        ///     member carries <see cref="MethodImplOptions.NoInlining"/> | <see cref="MethodImplOptions.NoOptimization"/>
        ///     (held by the guard test <c>SubclassAuditFrameTests</c>): <see cref="NDCharArray"/>'s <c>NanMax</c>, once at tier
        ///     1, inlined np.amax and tail-jumped to its subclass path, and np.nanmax was recorded as the caller (PR #633's CI,
        ///     2026-11-07, ubuntu net10.0 only, as tiering timing decides when a method is optimized).</para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void Audit(string function)
        {
            var trace = new System.Diagnostics.StackTrace(1, false);
            string woven = "<" + function + ">subclass";
            for (int i = 0; i < trace.FrameCount; i++)
            {
                System.Reflection.MethodBase method = trace.GetFrame(i)?.GetMethod();
                // Tested before the owner skip below: the overrides that hand the object on live in the skipped classes, and
                // skipping them would make the public facade that called them (np.nanmax) look like the caller.
                if (IsSubclassAware(method))
                    break;
                Type owner = method?.DeclaringType;
                if (owner is null || owner == typeof(NDSubclass) || owner == typeof(NDMatrix) || owner == typeof(NDCharArray))
                    continue;
                if (method.Name == function || method.Name == woven)
                    continue;
                if (owner.Assembly == typeof(NDSubclass).Assembly)
                    InternalCallers.AddOrUpdate($"{owner.FullName}.{method.Name} -> {function}", 1, static (_, n) => n + 1);
                break;
            }
        }

        /// <summary>
        ///     Whether a stack frame's method is NumPy's own class-reading code: it carries
        ///     <see cref="SubclassAwareAttribute"/>, or it overrides a member that does (the attribute sits on the base
        ///     declaration the weave resolves, <c>NDArray.NanMax</c>, not on each class's override).
        /// </summary>
        /// <param name="method">The frame's method, or <c>null</c>.</param>
        /// <returns>True for a <c>[SubclassAware]</c> member or an override of one.</returns>
        /// <remarks>Reflection, under the audit's test hook only.</remarks>
        private static bool IsSubclassAware(System.Reflection.MethodBase method)
        {
            if (method is null)
                return false;
            System.Reflection.MethodBase declaration = method is System.Reflection.MethodInfo info ? info.GetBaseDefinition() : method;
            return declaration.IsDefined(typeof(SubclassAwareAttribute), false);
        }

        #endregion
    }
}
