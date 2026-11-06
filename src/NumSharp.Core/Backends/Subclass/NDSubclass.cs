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
// counted reference on the buffer), exactly as the hand-written rules this replaces did.
//
// THE MECHANISMS below are NumPy's class selection (npy_find_array_wrap's priorities, PyArray_GetSubType's) followed by
// the chosen class's own construction (NDMatrix.FromWrapped for a matrix, the chararray's finalize rule and view). They
// return the plain result unchanged when no operand of their kind is a subclass.
//
// MIGRATION SHIM: while a legacy NDMatrixBoundary / NDCharArrayBoundary region is open (an unconverted member computes
// plain below it), IsOperand is false, so a woven function called inside the region computes plain as it did before.
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
        /// <summary><c>numpy.matrix.__array_priority__</c> (defmatrix.py).</summary>
        internal const double MatrixPriority = 10.0;

        /// <summary><c>numpy.char.chararray</c> declares none: ndarray's <c>NPY_PRIORITY</c>, 0.0.</summary>
        internal const double CharArrayPriority = 0.0;

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
        /// <remarks>MIGRATION SHIM: false while a legacy boundary region is open, so a woven function called by unconverted
        ///     code inside such a region computes plain, as it did before the weave.</remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static bool IsOperand(NDArray a)
            => KindOf(a) != NDSubclassKind.None && NDMatrixBoundary.AtOutermostCall;

        /// <summary><see cref="IsOperand(NDArray)"/> for operands passed as one sequence: true when any element is one.</summary>
        /// <param name="arrays">The operands, or <c>null</c>; a <c>null</c> element is skipped.</param>
        /// <returns>True when the call takes the subclass path.</returns>
        internal static bool IsOperand(NDArray[] arrays)
        {
            if (arrays is null || !NDMatrixBoundary.AtOutermostCall)
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
            => NDMatrixBoundary.AtOutermostCall && HoldsOperand(nested);

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
            if (@out is not null && ReferenceEquals(result, @out))
                return KindOf(@out) == NDSubclassKind.CharArray ? NDCharArray.ViewOf(@out) : @out;
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
            double current;
            switch (KindOf(x))
            {
                case NDSubclassKind.Matrix:
                    current = MatrixPriority;
                    break;
                case NDSubclassKind.CharArray:
                    current = CharArrayPriority;
                    break;
                default:
                    return;   // an exact ndarray has no wrap; it never displaces a subclass
            }
            if (wrapper is null || priority < current)
            {
                wrapper = x;
                priority = current;
            }
        }

        /// <summary>The chosen operand's <c>__array_wrap__</c> of one result.</summary>
        /// <param name="result">The plain result (or the caller's operand handed back), or <c>null</c>.</param>
        /// <param name="wrapper">The operand whose class wraps, or <c>null</c> (nothing wraps).</param>
        /// <returns>The matrix view, the chararray view of a text result, or <paramref name="result"/>.</returns>
        /// <exception cref="ValueError">A matrix result with more than two axes longer than 1.</exception>
        private static NDArray WrapBy(NDArray result, NDArray wrapper)
        {
            if (result is null || wrapper is null)
                return result;
            if (KindOf(wrapper) == NDSubclassKind.Matrix)
                // ndarray.__array_wrap__ as the matrix class, then the finalize rule; an operand handed back stays itself.
                return result is NDMatrix ? result : NDMatrix.FromWrapped(result);
            // chararray.__array_wrap__ is arr.view(type(self)): a NEW view of a text result (OWNDATA False) - also of the
            // operand itself when the computation handed it back - and any other result unchanged.
            return NDCharArray.WrapResult(result);
        }

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
            switch (KindOf(self))
            {
                case NDSubclassKind.Matrix:
                    if (result is NDMatrix)
                        return result;
                    // np.atleast_3d is matrix.__getitem__ with a new axis: the finalize rule is off, the rank stays.
                    return keepRank ? new NDMatrix(result.Storage.Alias()) : NDMatrix.FromWrapped(result);
                case NDSubclassKind.CharArray:
                    if (result is NDCharArray)
                        return result;
                    if (result.ndim == 0 && zeroDim != NewSubtypeZeroDim.Array)
                    {
                        // PyArray_Return creates the 0-d result as the subclass first (the finalize rule reads its dtype),
                        // then hands back the unstripped NumPy scalar; __getitem__ hands back the stripped element.
                        if (zeroDim == NewSubtypeZeroDim.Scalar)
                            NDCharArray.Finalize(result.dtype);
                        return NDCharArray.ScalarOf(result.Storage, strip: zeroDim == NewSubtypeZeroDim.Element);
                    }
                    return NDCharArray.FromCreated(result);
                default:
                    return result;
            }
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
            if (KindOf(self) == NDSubclassKind.CharArray)
                NDCharArray.Finalize(DType.Int64);
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
            if (KindOf(self) == NDSubclassKind.CharArray)
                NDCharArray.Finalize(dtype ?? self.dtype);
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
            for (int i = 0; i < arrays.Length; i++)
                if (KindOf(arrays[i]) == NDSubclassKind.Matrix)
                    return result is NDMatrix ? result : NDMatrix.FromWrapped(result);
            return result;
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
        {
            switch (KindOf(result))
            {
                case NDSubclassKind.Matrix:
                    return ((NDMatrix)result).A;
                case NDSubclassKind.CharArray:
                    return new NDArray(result.Storage.Alias()) { TensorEngine = result.TensorEngine };
                default:
                    return result;
            }
        }

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
                throw KeywordError("matrix", reached, "keepdims");
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
            NDSubclassKind kind = KindOf(self);
            if (kind != NDSubclassKind.Matrix)
                return subtypeOthers && kind != NDSubclassKind.None ? NewSubtype(result, @out, self, false, zeroDim) : result;
            if (result is null || result is NDMatrix)
                return result;
            Shape shape = result.Storage.Shape;
            if (shape.NDim == 0)
                return result;   // matrix._collapse(None): the scalar
            // The reduced axis stays with extent 1 under keepdims=True; a plain reduction dropped it, so a 1-d result is put
            // back as a column for the last axis and as a row for the first.
            if (shape.NDim == 1 && axis is int along && (along == 1 || along == -1))
                return new NDMatrix(result.Storage.Alias(shape.AsMatrixColumn()));
            return NDMatrix.FromWrapped(result);
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
        ///     frame left is the caller.
        /// </remarks>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void Audit(string function)
        {
            var trace = new System.Diagnostics.StackTrace(1, false);
            string woven = "<" + function + ">subclass";
            for (int i = 0; i < trace.FrameCount; i++)
            {
                System.Reflection.MethodBase method = trace.GetFrame(i)?.GetMethod();
                Type owner = method?.DeclaringType;
                if (owner is null || owner == typeof(NDSubclass) || owner == typeof(NDMatrix) || owner == typeof(NDCharArray)
                    || owner == typeof(NDMatrixBoundary) || owner == typeof(NDCharArrayBoundary))
                    continue;
                if (method.Name == function || method.Name == woven)
                    continue;
                if (owner.Assembly == typeof(NDSubclass).Assembly)
                    InternalCallers.AddOrUpdate($"{owner.FullName}.{method.Name} -> {function}", 1, static (_, n) => n + 1);
                break;
            }
        }

        #endregion
    }
}
