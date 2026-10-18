using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListArray.Reduce.cs — the reductions and accumulations of NumPy's object array of lists: max / min / sum / prod,
// cumsum / cumprod, trace, clip, and the statistics that end in a TypeError (mean / std / var)
// =============================================================================
//
// An OBJECT reduction is the ufunc's object loop folded element by element in the reduction iterator's order
// (PyUFunc_Reduce; probed against NumPy 2.4.2):
//
//   * Order. axis=None visits the elements in NumPy's flat reduction order — the axes sorted by |stride|
//     (NumPySequentialReduce.FlatVisitOrder), each walked in increasing LOGICAL index (DONT_NEGATE_STRIDES): a transposed
//     array is visited in memory order, a reversed one in logical order. Along an axis each output folds its elements in
//     increasing index. Sum's concatenation shows the order: a.T.sum() is a's C order.
//   * Start. With `initial` the fold starts from it; otherwise from the first visited element (an object reduction uses
//     no identity for a non-empty slice). An EMPTY slice without `initial` is NumPy's error for maximum / minimum ("zero-
//     size array to reduction operation maximum which has no identity") and the identity for add / multiply (the int 0 /
//     1, which an NDListArray cannot hold: NotSupportedException). A where= mask without `initial` is NumPy's
//     "reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'".
//   * The steps. maximum is npy_ObjectMax (`acc >= x ? acc : x`: a tie keeps the earlier list), minimum npy_ObjectMin
//     (`acc <= x`), add the list concatenation (a new list), multiply `list * list`: CPython's TypeError "can't multiply
//     sequence by non-int of type 'list'" for a slice of two or more elements (one element is itself).
//   * The result. axis=None gives NumPy's list as a 0-d NDListArray (the house rule: a 0-d array stands for NumPy's scalar
//     — item() is the list); keepdims=True a (1, …, 1) one; an axis the reduced array in NumPy's reduction layout. A sum,
//     a product and an accumulation make new lists, in a heap of their own (a concatenation is a new list in NumPy too),
//     sized once before anything is copied; maximum / minimum / clip SELECT lists that exist, so their result shares this
//     array's heap (NumPy's result holds the same list objects) and copies no list; `out=` stores them into the given
//     list array.
//
// The argument checks are NumPy's reduce prologue (ReductionArguments: out=, where=, the axis, out's shape), run over the
// id arrays, which have the object arrays' layouts.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        /// <summary>The object loops a list reduction folds with.</summary>
        private enum ListFold
        {
            /// <summary>np.maximum: <c>acc &gt;= x ? acc : x</c>.</summary>
            Max,

            /// <summary>np.minimum: <c>acc &lt;= x ? acc : x</c>.</summary>
            Min,

            /// <summary>np.add: the list concatenation.</summary>
            Sum,

            /// <summary>np.multiply: <c>list * list</c> (CPython's TypeError).</summary>
            Prod,
        }

        // =============================================================================================================
        // max / min / sum / prod
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>max(axis=None, out=None, keepdims=False, initial=&lt;no value&gt;, where=True)</c>: the greatest
        ///     list in Python's list order (a tie keeps the first visited), see the file header for the order and the rules.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">The start of the fold: a Python list (an <see cref="NDListArray"/> of one element, or a C#
        ///     sequence of <see cref="string"/> / <c>byte[]</c> pieces), or another Python value (it raises CPython's
        ///     TypeError when compared with a list); null for none.</param>
        /// <param name="where">NumPy's <c>where=</c> mask (null, or the C# literal <c>true</c>, for none).</param>
        /// <returns>The result list array (<paramref name="out"/> when given).</returns>
        /// <exception cref="ValueError">An empty slice without <paramref name="initial"/>; a mask without
        ///     <paramref name="initial"/>; NumPy's out= errors.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">A comparison of a str item with a bytes item, or of a list with a non-list
        ///     <paramref name="initial"/>.</exception>
        /// <exception cref="NotSupportedException">A result element that is not a list (a non-list <paramref name="initial"/>
        ///     over an empty slice).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray max(int? axis = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => Reduce(ListFold.Max, axis, @out, keepdims, initial, where);

        /// <summary>NumPy's <c>min(...)</c>: the smallest list (see <see cref="max"/>).</summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">The start of the fold (see <see cref="max"/>), or null.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">An empty slice or a mask without <paramref name="initial"/>; out= errors.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">An unorderable pair.</exception>
        /// <exception cref="NotSupportedException">A result element that is not a list.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray min(int? axis = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => Reduce(ListFold.Min, axis, @out, keepdims, initial, where);

        /// <summary>
        ///     NumPy's <c>sum(axis=None, dtype=None, out=None, keepdims=False, initial=&lt;no value&gt;, where=True)</c> of
        ///     lists: their CONCATENATION in NumPy's visiting order (a transposed array's sum is its memory order), starting
        ///     from <paramref name="initial"/> when given.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (NumPy's <c>None</c>, the object loop: the concatenation), or
        ///     a dtype whose loop NumPy runs over the lists converted to it — see the remarks.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">A Python list to start from (see <see cref="max"/>), or another Python value (CPython's
        ///     TypeError, <c>unsupported operand type(s) for +: 'int' and 'list'</c>); null for none.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">A mask without <paramref name="initial"/> (the object loop); out= errors; a float
        ///     <paramref name="dtype"/> over lists (<c>setting an array element with a sequence.</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">A non-list <paramref name="initial"/> added to a list; a <paramref name="dtype"/>
        ///     NumPy refuses as a ufunc argument (<c>The `dtype` and `signature` arguments to ufuncs only select the general
        ///     DType ...</c> for a sized, byte-swapped or unit-carrying descriptor, <c>Cannot pass a new user DType instance
        ///     ...</c> for a StringDType), a dtype without a loop (<c>No loop matching the specified signature and casting
        ///     was found for ufunc add</c>), an integer or complex <paramref name="dtype"/> over lists.</exception>
        /// <exception cref="NotSupportedException">An empty slice without <paramref name="initial"/> (NumPy's int 0), a list
        ///     of str and bytes items (the classes of <paramref name="initial"/> and the array differ), or a non-list result
        ///     — every result of a <paramref name="dtype"/> NumPy computes (a bool, a number of the dtype).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     <para>One overload, NumPy's signature. With a non-null <paramref name="dtype"/> NumPy casts every list to the
        ///     loop's dtype and reduces those values: an integer, float or complex loop fails at the first list with the
        ///     conversion error <see cref="astype"/> gives, and the bool loop (each list's truth) or an empty array returns a
        ///     value of the dtype, which a list array cannot hold — NumPy's value is
        ///     <c>np.sum(lists.astype(dtype), ...)</c> in that dtype. NumPy's checks run first, in its order: the
        ///     dtype's class, <paramref name="out"/>, <paramref name="where"/>, the axis, the loop, the shape of
        ///     <paramref name="out"/>, <paramref name="initial"/>.</para>
        /// </remarks>
        public NDListArray sum(int? axis = null, DType dtype = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => dtype is null
                ? Reduce(ListFold.Sum, axis, @out, keepdims, initial, where)
                : ReduceInLoop(ListFold.Sum, axis, dtype, @out, keepdims, initial, where, typeNumberOnly: false, "sum");

        /// <summary>
        ///     NumPy's <c>prod(axis=None, dtype=None, out=None, keepdims=False, initial=&lt;no value&gt;, where=True)</c> of
        ///     lists: a slice of one list is that list, two or more are <c>list * list</c> (CPython's TypeError).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (the object loop), or a dtype NumPy converts the lists to (see
        ///     <see cref="sum"/>).</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">The start of the fold: an int repeats the first list (Python's <c>n * list</c>), a list
        ///     multiplies (TypeError), null for none.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">A mask without <paramref name="initial"/> (the object loop); out= errors; a float
        ///     <paramref name="dtype"/> over lists.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError"><c>can't multiply sequence by non-int of type 'list'</c>; a refused
        ///     <paramref name="dtype"/> (see <see cref="sum"/>), or an integer or complex one over lists.</exception>
        /// <exception cref="NotSupportedException">An empty slice without <paramref name="initial"/> (NumPy's int 1), or a
        ///     non-list result (every result of a <paramref name="dtype"/> NumPy computes).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature (see <see cref="sum"/> for <paramref name="dtype"/>).</remarks>
        public NDListArray prod(int? axis = null, DType dtype = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => dtype is null
                ? Reduce(ListFold.Prod, axis, @out, keepdims, initial, where)
                : ReduceInLoop(ListFold.Prod, axis, dtype, @out, keepdims, initial, where, typeNumberOnly: false, "prod");

        // =============================================================================================================
        // cumsum / cumprod / trace
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>cumsum(axis=None, dtype=None, out=None)</c> of lists: the running concatenation along
        ///     <paramref name="axis"/> (the flattened C-order array for null); the result has the array's shape in its memory
        ///     order (a 1-D array for null).
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (the object loop), or a dtype NumPy converts the lists to (see
        ///     <see cref="sum"/>; this C method passes the dtype's TYPE NUMBER, so a sized or byte-swapped descriptor names
        ///     its class here, where <see cref="sum"/> refuses it).</param>
        /// <param name="out">A list array to store into, or null: the running lists of each line are written over its first
        ///     elements along the axis, NumPy's accumulate rules deciding what fits (see the remarks).</param>
        /// <returns>The running concatenations, or <paramref name="out"/>.</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/> (<c>output array is read-only</c>); an
        ///     <paramref name="out"/> that does not fit (<c>provided out is the wrong size for the accumulation.</c>, or the
        ///     iterator's <c>operands could not be broadcast together with remapped shapes [original-&gt;remapped]: ...</c> /
        ///     <c>non-broadcastable output operand with shape ... [remapped to ...] ...</c> /
        ///     <c>Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which has 1 dimensions</c>); a float
        ///     <paramref name="dtype"/> over lists.</exception>
        /// <exception cref="TypeError">A StringDType <paramref name="dtype"/>, a dtype without a loop, or an integer / complex
        ///     one over lists (see <see cref="sum"/>).</exception>
        /// <exception cref="NotSupportedException">A non-list result (every result of a <paramref name="dtype"/> NumPy
        ///     computes over at least one list, or its empty result without <paramref name="out"/>).</exception>
        /// <exception cref="ObjectDisposedException">The array or <paramref name="out"/> was disposed.</exception>
        /// <remarks>
        ///     <para>One overload, NumPy's signature.</para>
        ///     <para><c>PyUFunc_Accumulate</c> takes one of two routes, with different <paramref name="out"/> checks: a 1-D
        ///     object accumulation into an <paramref name="out"/> that does not overlap the array runs without an iterator —
        ///     an empty array or an empty <paramref name="out"/> returns <paramref name="out"/> untouched, any other shape
        ///     than the array's is "the wrong size" —; everything else (two or more dimensions, a <paramref name="dtype"/>
        ///     cast, an overlapping <paramref name="out"/>) builds NumPy's outer iterator, whose texts are reproduced, and
        ///     whose <paramref name="out"/> may have MORE axes (the extra ones are read at index 0) or broadcast the array.
        ///     NumPy 2.4.2 crashes the process for a 0-d <paramref name="out"/> on the first route; NumSharp reports it as
        ///     the wrong size ([Misaligned]).</para>
        /// </remarks>
        public NDListArray cumsum(int? axis = null, DType dtype = null, NDListArray @out = null) => Accumulate(ListFold.Sum, axis, dtype, @out);

        /// <summary>
        ///     NumPy's <c>cumprod(axis=None, dtype=None, out=None)</c> of lists: the first element of each line is itself, the
        ///     second multiplies (CPython's TypeError).
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c> (see <see cref="cumsum"/>).</param>
        /// <param name="out">A list array to store into, or null (see <see cref="cumsum"/>).</param>
        /// <returns>The result (for lines of at most one element), or <paramref name="out"/>.</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError"><c>can't multiply sequence by non-int of type 'list'</c> for a line of two or more; a
        ///     refused <paramref name="dtype"/>, or an integer / complex one over lists.</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>, or one that does not fit (see
        ///     <see cref="cumsum"/>); a float <paramref name="dtype"/> over lists.</exception>
        /// <exception cref="NotSupportedException">A non-list result (see <see cref="cumsum"/>).</exception>
        /// <exception cref="ObjectDisposedException">The array or <paramref name="out"/> was disposed.</exception>
        /// <remarks>One overload, NumPy's signature (see <see cref="cumsum"/>).</remarks>
        public NDListArray cumprod(int? axis = null, DType dtype = null, NDListArray @out = null) => Accumulate(ListFold.Prod, axis, dtype, @out);

        /// <summary>
        ///     NumPy's <c>trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)</c> of lists: the concatenation of the
        ///     diagonal's lists (NumPy's add.reduce over the diagonal's last axis).
        /// </summary>
        /// <param name="offset">The diagonal's offset.</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays.</param>
        /// <param name="axis2">The second axis.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (the object loop), or a dtype NumPy converts the lists to (see
        ///     <see cref="cumsum"/>: the C method passes its type number).</param>
        /// <param name="out">A list array to store into, or null.</param>
        /// <returns>The concatenations (a 0-d list array for a 2-D array).</returns>
        /// <exception cref="ValueError">Fewer than two axes (<c>diag requires an array of at least two dimensions</c>); out=
        ///     errors; a float <paramref name="dtype"/> over lists.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">A refused <paramref name="dtype"/>, or an integer / complex one over lists.</exception>
        /// <exception cref="NotSupportedException">An empty diagonal (NumPy's int 0), or a non-list result.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature: <c>PyArray_Trace</c> is the diagonal, then
        ///     <c>add.reduce(diagonal, -1, dtype=PyArray_DescrFromType(rtype), out=out)</c>.</remarks>
        public NDListArray trace(int offset = 0, int axis1 = 0, int axis2 = 1, DType dtype = null, NDListArray @out = null)
        {
            using var diagonalView = diagonal(offset, axis1, axis2);
            return dtype is null
                ? diagonalView.Reduce(ListFold.Sum, -1, @out, keepdims: false, initial: null, where: null)
                : diagonalView.ReduceInLoop(ListFold.Sum, -1, dtype, @out, keepdims: false, initial: null, where: null, typeNumberOnly: true, "trace");
        }

        // =============================================================================================================
        // clip
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>clip(min=None, max=None, out=None, **kwargs)</c> of lists — NumPy's <c>_clip</c>: np.maximum with
        ///     <paramref name="min"/> (a list below it becomes it), np.minimum with <paramref name="max"/>, np.clip with both,
        ///     np.positive with neither, element by element over the broadcast operands, by Python's list order.
        /// </summary>
        /// <param name="min">The lower bounds (a list array, broadcast), a Python value, or null.</param>
        /// <param name="max">The upper bounds, or null.</param>
        /// <param name="out">A list array to store into, or null: the broadcast of every operand (it may be larger than the
        ///     inputs, which then broadcast to it; it is never stretched itself).</param>
        /// <param name="dtype">The ufunc's <c>dtype=</c> (null: the object loop). Every other loop needs the object array cast
        ///     to it under <c>same_kind</c>, which NumPy refuses, so a non-null dtype always raises (see the
        ///     exceptions).</param>
        /// <param name="where">The ufunc's <c>where=</c> mask (null, or the C# literal <c>true</c>, for none; a bool array
        ///     that broadcasts with the operands): the loop runs only where it is true, and <paramref name="out"/> keeps its
        ///     lists elsewhere.</param>
        /// <returns>The clipped lists (a 0-d list array stands for NumPy's list), or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's text: every operand's shape,
        ///     <paramref name="out"/> and <paramref name="where"/> included).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/> (<c>output array is read-only</c>), or one of
        ///     another shape than the broadcast (<c>non-broadcastable output operand with shape (1,) doesn't match the
        ///     broadcast shape (4,)</c>).</exception>
        /// <exception cref="ArgumentException">A <paramref name="where"/> that is not bool (the house type of NumPy's
        ///     <c>Cannot cast array data from dtype('int64') to dtype('bool') according to the rule 'safe'</c>).</exception>
        /// <exception cref="TypeError">A bound that is not a list (<c>'&gt;=' not supported between instances of 'list' and
        ///     'str'</c>), an unorderable item pair, or no bound at all (np.positive of a list: <c>bad operand type for unary
        ///     +: 'list'</c>) — each only where the loop runs; a <paramref name="dtype"/>: NumPy's refusal of the descriptor
        ///     as a ufunc argument, <c>No loop matching the specified signature and casting was found for ufunc clip</c>
        ///     (text dtypes), or <c>Cannot cast ufunc 'clip' input 0 from dtype('O') to dtype('float64') with casting rule
        ///     'same_kind'</c>.</exception>
        /// <exception cref="NotSupportedException">A new result with masked-off elements (NumPy leaves them None, which a
        ///     list array cannot hold); a bound with no NumSharp array form.</exception>
        /// <exception cref="ObjectDisposedException">The array or <paramref name="out"/> was disposed.</exception>
        /// <remarks>One overload, NumPy's signature with the two keywords of its <c>**kwargs</c> that NumSharp's
        ///     <see cref="NDArray.clip"/> takes. The checks run in the ufunc's order: the dtype's class, the bounds' conversion,
        ///     <paramref name="out"/>, <paramref name="where"/>, the loop, the broadcast.</remarks>
        public NDListArray clip(object min = null, object max = null, NDListArray @out = null, DType dtype = null, NDArray where = null)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            // _clip picks the ufunc by the bounds it is given; its name is in every loop error.
            string ufunc = min is null ? (max is null ? "positive" : "minimum") : (max is null ? "maximum" : "clip");
            // _get_fixed_signature: the dtype names a DType class.
            if (dtype is not null)
                CheckUfuncDtype(dtype, typeNumberOnly: false);
            // convert_ufunc_arguments: the inputs (np.asarray of each bound), out= (_set_out_array), then where=.
            var lo = BoundOperand(min);
            var hi = BoundOperand(max);
            if (@out is not null && !@out.Ids.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            NDArray mask = IsNoMask(where) ? null : where;
            DefaultEngine.ValidateWhereMask(mask);
            // promote_and_get_ufuncimpl / resolve_descriptors: an object input reaches no loop of another dtype.
            if (dtype is not null)
                throw ObjectUfuncLoopError(dtype, ufunc, lo.Present && hi.Present ? 3 : lo.Present || hi.Present ? 2 : 1);
            return ClipInto(lo, hi, @out, mask);
        }

        /// <summary>
        ///     The ufunc call of <see cref="clip"/> after its argument checks: the iterator's broadcast of every operand
        ///     (inputs, <paramref name="out"/>, the mask), the object loop at the positions the mask selects, then the result —
        ///     a new list array (a 0-d one stands for NumPy's list), or the lists stored into <paramref name="out"/>.
        /// </summary>
        /// <param name="lo">The lower bound operand (absent for np.minimum / np.positive).</param>
        /// <param name="hi">The upper bound operand (absent for np.maximum / np.positive).</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="mask">The caller's bool mask, or null.</param>
        /// <returns>The result, or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        /// <exception cref="ValueError"><paramref name="out"/> of another shape than the broadcast.</exception>
        /// <exception cref="TypeError">The loop's first failing operation (see <see cref="clip"/>).</exception>
        /// <exception cref="NotSupportedException">A new result with masked-off elements.</exception>
        private NDListArray ClipInto(Bound lo, Bound hi, NDListArray @out, NDArray mask)
        {
            // npyiter_fill_axisdata: every operand's shape in the ufunc's order — the inputs, the output, the mask.
            var shapes = new List<Shape> { Ids.Shape };
            if (lo.Present)
                shapes.Add(lo.Shape);
            if (hi.Present)
                shapes.Add(hi.Shape);
            var inputShapes = shapes.ToArray();
            if (@out is not null)
                shapes.Add(@out.Ids.Shape);
            if (mask is not null)
                shapes.Add(mask.Shape);
            var dims = BroadcastAll(shapes);
            // An output is NPY_ITER_NO_BROADCAST: it must have the broadcast shape itself.
            if (@out is not null && !@out.Ids.Shape.dimensions.AsSpan().SequenceEqual(dims))
                throw new ValueError($"non-broadcastable output operand with shape {@out.Ids.Shape.ToPythonTuple()} doesn't match " +
                                     $"the broadcast shape {Shape.Adopt((long[])dims.Clone()).ToPythonTuple()}");
            long count = 1;
            foreach (var d in dims)
                count *= d;
            NDArray maskView = null;
            try
            {
                // The positions the loop runs at: every one, or the mask's true ones (masked-off ones are never compared).
                bool anyRun = count > 0, anyMasked = false;
                if (mask is not null && count > 0)
                {
                    maskView = np.broadcast_to(mask, Shape.Adopt((long[])dims.Clone()));
                    long selected = np.count_nonzero(maskView);
                    anyRun = selected > 0;
                    anyMasked = selected < count;
                }
                if (!lo.Present && !hi.Present)
                {
                    // np.positive of a list: CPython's unary +, at the first position the loop runs.
                    if (anyRun)
                        throw new TypeError("bad operand type for unary +: 'list'");
                }
                else if (anyRun)
                {
                    // A bound that is not a list fails at the first comparison the loop makes (np.maximum's '>=' first).
                    if (lo.Present && !lo.IsLists)
                        throw new TypeError($"'>=' not supported between instances of 'list' and '{lo.TypeName}'");
                    if (hi.Present && !hi.IsLists)
                        throw new TypeError($"'<=' not supported between instances of 'list' and '{hi.TypeName}'");
                }
                if (@out is null)
                {
                    // A new result's masked-off elements are left as NumPy allocated them: None.
                    if (anyMasked)
                        throw new NotSupportedException(
                            "NumPy leaves the masked-off elements of a new clip result as None, which an NDListArray (a list " +
                            "array) cannot hold; pass out= to keep lists there");
                    if (!lo.Present && !hi.Present)
                    {
                        // Nothing ran (an empty broadcast): np.positive's empty object array in the ufunc's output layout.
                        return new NDListArray(new NDArray(NPTypeCode.Int64, Shape.NpyIterOutputShape(dims, inputShapes), false),
                            NDListHeap.Empty(Kind)).MarkStandIn();
                    }
                    // A 0-d ufunc result is unwrapped to the object: the clipped list.
                    return ClipCore(lo, hi, dims, Shape.NpyIterOutputShape(dims, mask is null ? inputShapes : shapes.ToArray()), null).MarkStandIn();
                }
                // out=: the loop writes the selected positions; every other element of out keeps its list.
                if (!anyRun)
                    return @out;
                using (var computed = ClipCore(lo, hi, dims, Shape.Adopt((long[])dims.Clone()), maskView))
                    @out.AssignWhere(@out.Ids, computed, maskView);
                return @out;
            }
            finally
            {
                maskView?.Dispose();
            }
        }

        // =============================================================================================================
        // mean / std / var
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)</c> of lists (NumPy's
        ///     <c>_mean</c>): the sum divided by the count. Whole-array: the concatenated list divided by NumPy's intp count
        ///     — an empty list is an empty float64 array, a list of str or bytes NumPy's divide TypeError, an empty array NaN;
        ///     along an axis (or with keepdims): each list divided in the object loop, CPython's TypeError; into
        ///     <paramref name="out"/>: NumPy's sum stored into it first (see the remarks).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (the object loop), or a dtype NumPy converts the lists to before
        ///     it sums them (the bool loop sums their truths; the others fail at the first list).</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape that receives the mean (returned), or
        ///     null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask: the object loop has no identity for it (ValueError over any list),
        ///     a <paramref name="dtype"/> loop reduces the selected elements and divides by the per-output counts.</param>
        /// <returns>The NumPy values a list array's mean has (a 0-d NaN, an empty float64 array, a value of
        ///     <paramref name="dtype"/>), or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError"><c>ufunc 'divide' not supported for the input types, and the inputs could not be safely
        ///     coerced to any supported types according to the casting rule ''safe''</c> (whole array);
        ///     <c>unsupported operand type(s) for /: 'list' and 'int'</c> (along an axis); a list stored into an integer or
        ///     complex <paramref name="out"/> or <paramref name="dtype"/> (<c>int() argument must be ... not 'list'</c>,
        ///     <c>must be real number, not list</c>); two lists added in a bool <paramref name="out"/>; a refused or loop-less
        ///     <paramref name="dtype"/> (see <see cref="sum"/>); a StringDType <paramref name="out"/>.</exception>
        /// <exception cref="ValueError">A mask in the object loop (add has no identity for it); a list stored into a float or
        ///     text <paramref name="out"/> or <paramref name="dtype"/> (<c>setting an array element with a sequence.</c>);
        ///     NumPy's out= checks (read-only, rank, shape).</exception>
        /// <exception cref="DivideByZeroException">An empty slice along an axis (NumPy's ZeroDivisionError, <c>division by
        ///     zero</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array (a 0-d array has none).</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array (an empty result along an
        ///     axis).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     One overload, NumPy's signature. <c>_mean</c> is <c>_count_reduce_items</c> (the axis and the mask's
        ///     broadcast), <c>ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)</c>, then
        ///     <c>true_divide(ret, rcount, out=ret, casting='unsafe')</c>. With <paramref name="out"/> the object sum casts each
        ///     output's first list into it — NumPy's conversion error for a numeric or text out — except a bool out, which
        ///     takes the list's truth and adds the next list to it (CPython's TypeError); an output that sums no list holds
        ///     the identity 0, divided by 0 in the out's dtype (NaN, NumPy's unsafe NaN cast for an integer, True for a
        ///     bool).
        /// </remarks>
        public NDArray mean(int? axis = null, DType dtype = null, NDArray @out = null, bool keepdims = false, NDArray where = null)
        {
            ThrowIfDisposed();
            NDArray mask = IsNoMask(where) ? null : where;
            // _count_reduce_items: the axis, then the mask broadcast to the array.
            long rcount = StatisticsCount(axis, mask);
            // ret = umr_sum(arr, axis, dtype, out, keepdims, where=where): the reduce prologue, the loop between its parsing
            // and the shape of out.
            if (dtype is not null)
                CheckUfuncDtype(dtype, typeNumberOnly: false);
            ReductionArguments.ValidateParsedArguments(@out, mask);
            bool[] flags = ReductionArguments.AxisFlags(ndim, axis);
            if (dtype is not null)
                CheckObjectReduceLoop(dtype, "add");
            else if (@out is not null)
                CheckObjectOutLoop(@out, "add");
            ReductionArguments.ValidateOperands(Ids, flags, keepdims, @out, mask, "add");
            if (dtype is not null)
                return MeanInLoop(axis, dtype, @out, keepdims, mask, rcount);
            // The object loop has no identity for a where= mask (an empty array uses it: every count is 0).
            if (mask is not null && size > 0)
                throw new ValueError("reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
            if (@out is not null)
                return MeanInto(@out, rcount);
            if (!keepdims && (axis is null || ndim <= 1))
            {
                // _mean's scalar branch — every axis reduced to a 0-d result (axis=None, or the one axis of a 1-D array):
                // the sum is a Python object (a list, or the int 0 of an empty array), divided by an np.intp count through
                // NumPy's scalar operator.
                if (size == 0)
                    return NDArray.Scalar(double.NaN);
                using var total = sum();
                long pieces = total.Heap.Length(total.FlatId(0));
                if (pieces == 0)
                    return new NDArray(NPTypeCode.Double, Shape.Vector(0), true);
                throw new TypeError("ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced " +
                                    "to any supported types according to the casting rule ''safe''");
            }
            throw DivideListsError(axis, "mean");
        }

        /// <summary>
        ///     NumPy's <c>var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=&lt;no
        ///     value&gt;)</c> of lists (NumPy's <c>_var</c>): its first step that touches a list — the division of the
        ///     kept-dims sum, or the subtraction of <paramref name="mean"/> — is CPython's TypeError; an array whose
        ///     deviations are EMPTY gets NumPy's value (see the remarks).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c>: null (the object loop), or a dtype NumPy converts the lists to for
        ///     the mean (the bool loop's mean is then subtracted from the lists: CPython's TypeError).</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape that receives the variance (returned), or
        ///     null.</param>
        /// <param name="ddof">NumPy's delta degrees of freedom (the divisor is <c>max(count - ddof, 0)</c>).</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <param name="mean">NumPy's keyword-only <c>mean=</c>: the mean to subtract instead of computing it (any value NumPy
        ///     takes: a list array, an array, a C# number or string as the Python value it stands for), or null (NumPy's
        ///     default, computed).</param>
        /// <returns>NumPy's value for a list array whose deviations are empty (a 0-d float64 0.0 or NaN, an array of
        ///     <paramref name="dtype"/>), or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError"><c>unsupported operand type(s) for /: 'list' and 'int'</c> (the mean's division);
        ///     <c>unsupported operand type(s) for -: 'list' and 'float'</c> (the subtraction, naming the mean's element type);
        ///     <c>ufunc 'divide' not supported ...</c> (a 0-d array of a non-empty list); a list converted to an integer or
        ///     complex <paramref name="dtype"/>; a refused or loop-less <paramref name="dtype"/>; std's sqrt of an object
        ///     result.</exception>
        /// <exception cref="DivideByZeroException">An empty slice (NumPy's ZeroDivisionError, <c>division by zero</c>).</exception>
        /// <exception cref="ValueError">A mask (the object loop has no identity for it); a list converted to a float
        ///     <paramref name="dtype"/>; NumPy's out= checks.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="mean"/> does not broadcast with the array (<c>operands
        ///     could not be broadcast together with shapes (4,) (3,) </c>).</exception>
        /// <exception cref="ArgumentException">std into an integer or bool <paramref name="out"/> (NumPy's sqrt output cast,
        ///     the house type).</exception>
        /// <exception cref="AxisError">An axis outside the array (a 0-d array has none).</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array (along an axis of an empty array); a
        ///     <paramref name="mean"/> with no NumSharp array form.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>
        ///     <para>One overload, NumPy's signature (ndarray.var takes no <c>correction=</c>: NumPy 2.4.2's <c>_var</c> refuses
        ///     it, where the function <c>np.var</c> takes it).</para>
        ///     <para><c>_var</c>'s statements over lists: <c>_count_reduce_items</c>; the mean (<c>umr_sum(arr, axis, dtype,
        ///     keepdims=True, where=where)</c> divided by the count) or <paramref name="mean"/>; <c>x = arr - arrmean</c>,
        ///     the object subtraction, CPython's TypeError for any element; then, for EMPTY deviations, <c>ret = umr_sum(x,
        ///     axis, dtype, out, keepdims=keepdims, where=where)</c> — every output its identity 0 — divided by
        ///     <c>max(count - ddof, 0)</c>: NaN or 0.0 for a scalar, ZeroDivisionError for an object array divided by 0, the
        ///     dtype's unsafe cast for <paramref name="dtype"/> or <paramref name="out"/>. <paramref name="out"/> is first
        ///     checked by that last sum, so an earlier TypeError wins over its errors.</para>
        /// </remarks>
        public NDArray var(int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false,
                           NDArray where = null, object mean = null)
            => VarStd(axis, dtype, @out, ddof, keepdims, where, mean, sqrt: false);

        /// <summary>
        ///     NumPy's <c>std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True, mean=&lt;no
        ///     value&gt;)</c> of lists: <c>sqrt(var(...))</c> (NumPy's <c>_std</c>), whose variance step raises for any list
        ///     (see <see cref="var"/>).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">NumPy's loop <c>dtype</c> (see <see cref="var"/>).</param>
        /// <param name="out">NumPy's <c>out=</c>: an array of the result's shape (returned), or null.</param>
        /// <param name="ddof">NumPy's delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <param name="mean">NumPy's keyword-only <c>mean=</c> (see <see cref="var"/>), or null.</param>
        /// <returns>NumPy's value for a list array whose deviations are empty, or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">See <see cref="var"/>; and <c>loop of ufunc does not support argument 0 of type float
        ///     which has no callable sqrt method</c> (the square root of an object array of floats).</exception>
        /// <exception cref="DivideByZeroException">An empty slice (NumPy's ZeroDivisionError).</exception>
        /// <exception cref="ValueError">See <see cref="var"/>.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="mean"/> does not broadcast with the array.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="out"/> or array result (NumPy's
        ///     <c>Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'</c>, the
        ///     house type).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array; a <paramref name="mean"/> with no
        ///     NumSharp array form.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        /// <remarks>One overload, NumPy's signature (see <see cref="var"/>).</remarks>
        public NDArray std(int? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false,
                           NDArray where = null, object mean = null)
            => VarStd(axis, dtype, @out, ddof, keepdims, where, mean, sqrt: true);

        /// <summary>
        ///     NumPy's <c>_var</c> / <c>_std</c> over lists (see <see cref="var"/>): the count, the mean or the given
        ///     <paramref name="mean"/>, the subtraction's TypeError for any element, and for empty deviations NumPy's value.
        /// </summary>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="ddof">The delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="where">The caller's mask (null or Python's True: none).</param>
        /// <param name="mean">The given mean, or null.</param>
        /// <param name="sqrt">True for std.</param>
        /// <returns>NumPy's value, or <paramref name="out"/>.</returns>
        private NDArray VarStd(int? axis, DType dtype, NDArray @out, int ddof, bool keepdims, NDArray where, object mean, bool sqrt)
        {
            ThrowIfDisposed();
            NDArray mask = IsNoMask(where) ? null : where;
            // _count_reduce_items: the axis (a 0-d array has none), then the mask broadcast to the array.
            long rcount = StatisticsCount(axis, mask);
            long[] deviations;
            if (mean is null)
            {
                // arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where), then its division by the count.
                if (dtype is not null)
                    CheckUfuncDtype(dtype, typeNumberOnly: false);
                ReductionArguments.ValidateParsedArguments(null, mask);
                bool[] flags = ReductionArguments.AxisFlags(ndim, axis);
                if (dtype is not null)
                    CheckObjectReduceLoop(dtype, "add");
                ReductionArguments.ValidateOperands(Ids, flags, keepdims: true, null, mask, "add");
                if (dtype is null)
                {
                    if (mask is not null && size > 0)
                        throw new ValueError("reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
                    if (ndim == 0)
                    {
                        // A 0-d array's kept-dims sum is unwrapped to the list itself, so the mean is NumPy's scalar division
                        // `list / rcount`: a non-empty list is the ufunc divide TypeError; an empty one is the empty float64
                        // mean, and the subtraction `arr - arrmean` broadcasts the 0-d array against its (0,) shape.
                        if (Heap.Length(FlatId(0)) > 0)
                            throw new TypeError("ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced " +
                                                "to any supported types according to the casting rule ''safe''");
                        deviations = new long[] { 0 };
                    }
                    else
                    {
                        // The kept-dims sums divided in the object loop: a list by an int is CPython's TypeError, the int 0 of
                        // an empty slice by 0 its ZeroDivisionError; no output divides nothing (then the deviations are empty).
                        long outputs = KeptOutputs(flags);
                        if (outputs > 0)
                        {
                            if (rcount == 0)
                                throw new DivideByZeroException("division by zero");
                            throw new TypeError("unsupported operand type(s) for /: 'list' and 'int'");
                        }
                        deviations = (long[])Ids.Shape.dimensions.Clone();
                    }
                }
                else
                {
                    // Another loop: every list is converted to it (its error); the bool loop's mean is then subtracted from
                    // the lists (CPython's TypeError, naming the mean's element type).
                    if (size > 0)
                    {
                        if (ListCastError(dtype) is { } conversion)
                            throw conversion;
                        throw new TypeError($"unsupported operand type(s) for -: 'list' and '{PyElementTypeName(dtype)}'");
                    }
                    deviations = (long[])Ids.Shape.dimensions.Clone();
                }
                return EmptyDeviations(deviations, axis, dtype, @out, keepdims, mask, Math.Max(rcount - ddof, 0), sqrt, dtypeChecked: true);
            }
            // x = arr - mean: np.subtract's broadcast, then the object loop at every element.
            var m = BoundOperand(mean);
            var meanShape = m.IsLists ? m.Lists.Ids.Shape : m.Shape;
            deviations = BroadcastAll(new List<Shape> { Ids.Shape, meanShape });
            long count = 1;
            foreach (var d in deviations)
                count *= d;
            if (count > 0)
                throw new TypeError($"unsupported operand type(s) for -: 'list' and '{(m.IsLists ? "list" : m.TypeName)}'");
            return EmptyDeviations(deviations, axis, dtype, @out, keepdims, mask, Math.Max(rcount - ddof, 0), sqrt, dtypeChecked: false);
        }

        /// <summary>
        ///     The last statements of <c>_var</c> / <c>_std</c> for EMPTY deviations <c>x</c> of shape
        ///     <paramref name="deviations"/>: <c>ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)</c> —
        ///     NumPy's reduce prologue over <c>x</c>, every output its identity 0 —, the division by
        ///     <paramref name="divisor"/> (NumPy's <c>max(rcount - ddof, 0)</c>) in the result's own arithmetic, and std's
        ///     square root.
        /// </summary>
        /// <param name="deviations">The shape of <c>x</c> (it may have more axes than the array, from a broadcast mean).</param>
        /// <param name="axis">The axis, read on <c>x</c>'s rank.</param>
        /// <param name="dtype">The loop dtype, or null (the object loop: the Python int 0).</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="mask">The mask, or null.</param>
        /// <param name="divisor">The divisor.</param>
        /// <param name="sqrt">True for std.</param>
        /// <param name="dtypeChecked">Whether the mean's sum already checked <paramref name="dtype"/>'s class (it reads the
        ///     dtype before this sum when no mean was given).</param>
        /// <returns>NumPy's value, or <paramref name="out"/>.</returns>
        /// <exception cref="ValueError">NumPy's out= checks.</exception>
        /// <exception cref="TypeError">A refused or loop-less <paramref name="dtype"/>; a text or StringDType
        ///     <paramref name="out"/>; std's sqrt of an object array of floats.</exception>
        /// <exception cref="DivideByZeroException">An object array divided by 0.</exception>
        /// <exception cref="ArgumentException">std's sqrt into an integer or bool array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array.</exception>
        private NDArray EmptyDeviations(long[] deviations, int? axis, DType dtype, NDArray @out, bool keepdims, NDArray mask,
                                        long divisor, bool sqrt, bool dtypeChecked)
        {
            // The zero-size stand-in of x carries only its shape (the prologue reads nothing else).
            using var x = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])deviations.Clone()), false);
            if (dtype is not null && !dtypeChecked)
                CheckUfuncDtype(dtype, typeNumberOnly: false);
            ReductionArguments.ValidateParsedArguments(@out, mask);
            bool[] flags = ReductionArguments.AxisFlags(x.ndim, axis);
            if (dtype is not null)
                CheckObjectReduceLoop(dtype, "add");
            else if (@out is not null)
                CheckObjectOutLoop(@out, "add");
            ReductionArguments.ValidateOperands(x, flags, keepdims, @out, mask, "add");
            if (@out is not null)
            {
                // The identity cast into out, then true_divide(out, rcount, out=out, casting='unsafe') (no text loop), then
                // std's sqrt(out, out=out).
                if (@out.dtype.kind is 'U' or 'S')
                    throw new TypeError("ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced " +
                                        "to any supported types according to the casting rule ''safe''");
                using (var zero = NDArray.Scalar(0L))
                    np.copyto(@out, zero, casting: "unsafe");
                DefaultEngine.DivideByCountLikeNumPy(@out, divisor);
                if (sqrt)
                    DefaultEngine.SqrtLikeNumPy(@out, scalarResult: false);
                return @out;
            }
            var outDims = new List<long>();
            for (int d = 0; d < flags.Length; d++)
            {
                if (!flags[d])
                    outDims.Add(deviations[d]);
                else if (keepdims)
                    outDims.Add(1);
            }
            bool scalar = outDims.Count == 0;
            if (dtype is null)
            {
                if (scalar)
                {
                    // The Python int 0 divided by an np.intp: NumPy's scalar division, an np.float64 (and its sqrt).
                    double v = divisor == 0 ? double.NaN : 0.0;
                    return NDArray.Scalar(sqrt ? Math.Sqrt(v) : v);
                }
                // An object array of int 0 divided in the object loop (the count an int there): 0 / 0 is ZeroDivisionError;
                // otherwise an object array of floats, whose square root has no object loop.
                long outputs = 1;
                foreach (var d in outDims)
                    outputs *= d;
                if (outputs > 0 && divisor == 0)
                    throw new DivideByZeroException("division by zero");
                if (sqrt && outputs > 0)
                    throw new TypeError("loop of ufunc does not support argument 0 of type float which has no callable sqrt method");
                throw new NotSupportedException(
                    $"NumPy's {(sqrt ? "std" : "var")} of this list array is an object array, which NumSharp has no type for");
            }
            // A numeric loop: the dtype's 0 per output, divided in NumPy's arithmetic (an array in place, a scalar through
            // dtype.type(ret / rcount): the same unsafe cast), then the square root (an array's same_kind refusal).
            var ret = new NDArray(dtype, Shape.Adopt(outDims.ToArray()), true);
            try
            {
                DefaultEngine.DivideByCountLikeNumPy(ret, divisor);
                if (sqrt)
                    DefaultEngine.SqrtLikeNumPy(ret, scalarResult: scalar);
                return ret;
            }
            catch
            {
                ret.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <see cref="mean"/> with a <paramref name="dtype"/> loop, after NumPy's checks: the lists converted to it (the
        ///     first one's error), the sum in that loop (the bool loop sums the truths), then the division — into
        ///     <paramref name="out"/> when given.
        /// </summary>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="dtype">The loop dtype (a numeric class with a loop).</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="mask">The mask, or null.</param>
        /// <param name="rcount">The count per output (without a mask; a mask's counts are per output, see
        ///     <see cref="MaskedMeanInLoop"/>).</param>
        /// <returns>The mean (a value of the dtype), or <paramref name="out"/>.</returns>
        /// <exception cref="ValueError">A float dtype over lists (NumPy casts every element, a masked-off one too).</exception>
        /// <exception cref="TypeError">An integer or complex dtype over lists.</exception>
        private NDArray MeanInLoop(int? axis, DType dtype, NDArray @out, bool keepdims, NDArray mask, long rcount)
        {
            // The buffered cast converts EVERY element, masked off or not: the first list's conversion error.
            if (size > 0 && ListCastError(dtype) is { } conversion)
                throw conversion;
            // The converted elements: the lists' truths for bool, an empty array of the dtype otherwise.
            using var converted = astype(dtype);
            if (mask is not null)
                return MaskedMeanInLoop(converted, axis, dtype, @out, keepdims, mask);
            if (@out is null)
                return converted.TensorEngine.Mean(converted, axis, dtype, keepdims);
            // umr_sum(arr, axis, dtype, out, keepdims), then true_divide(out, rcount, out=out, casting='unsafe').
#pragma warning disable NDW012 // not a leak: ReduceAdd(out=) returns the destination itself, which the caller owns
            converted.TensorEngine.ReduceAdd(converted, axis, keepdims, dtype, @out);
#pragma warning restore NDW012
            DefaultEngine.DivideByCountLikeNumPy(@out, rcount);
            return @out;
        }

        /// <summary>
        ///     <see cref="mean"/> with a <c>where=</c> mask in a <c>dtype=</c> loop — <c>_mean</c>'s statements:
        ///     <c>ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)</c>, the loop's masked add (the identity 0 at
        ///     every masked-off element: the bool loop's logical OR ignores a False), divided by <c>_count_reduce_items</c>'
        ///     per-output counts (the mask broadcast to the array, summed along the axis in intp) —
        ///     <c>true_divide(ret, rcount, out=ret, casting='unsafe')</c> for an array, <c>ret.dtype.type(ret / rcount)</c>
        ///     for a scalar: the same arithmetic.
        /// </summary>
        /// <param name="converted">The elements converted to the loop: the lists' truths for bool (only an EMPTY array
        ///     reaches another loop; every other conversion failed). A fresh writeable array, overwritten here.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="mask">The bool mask (its dtype and its broadcast to the array were checked).</param>
        /// <returns>The mean — a read-only 0-d scalar for a full reduction without keepdims, else a fresh array — or
        ///     <paramref name="out"/>.</returns>
        private NDArray MaskedMeanInLoop(NDArray converted, int? axis, DType dtype, NDArray @out, bool keepdims, NDArray mask)
        {
            using var maskView = np.broadcast_to(mask, Shape.Adopt((long[])Ids.Shape.dimensions.Clone()));
            // The masked-off elements hold the add identity (0 cast to the loop: False for bool), so the plain reduction
            // over them is the masked one.
            using (var off = np.logical_not(maskView))
                np.copyto(converted, 0, casting: "unsafe", where: off);
            // _count_reduce_items with where=: the selected elements per output, in intp.
            using var counts = maskView.TensorEngine.ReduceAdd(maskView, axis, keepdims, np.int64);
            if (@out is not null)
            {
#pragma warning disable NDW012 // not a leak: ReduceAdd(out=) returns the destination itself, which the caller owns
                converted.TensorEngine.ReduceAdd(converted, axis, keepdims, dtype, @out);
#pragma warning restore NDW012
                DefaultEngine.DivideByCountLikeNumPy(@out, counts);
                return @out;
            }
            var ret = converted.TensorEngine.ReduceAdd(converted, axis, keepdims, dtype);
            if (!ret.Shape.IsWriteable)
            {
                // A full reduction's read-only scalar: its value in a writeable 0-d array, divided there, then NumPy's
                // scalar again.
                var scalar = ret.copy();
                ret.Dispose();
                ret = scalar;
            }
            try
            {
                DefaultEngine.DivideByCountLikeNumPy(ret, counts);
                return ret.ndim == 0 ? ret.MarkReductionScalar() : ret;
            }
            catch
            {
                // The quotient never reached the caller: its buffer goes back now.
                ret.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <see cref="mean"/> of the object loop into <paramref name="out"/>: NumPy's object sum stored into it (each
        ///     output's first list cast to the out's dtype, the next ones added to the cast value), then the division.
        /// </summary>
        /// <param name="out">The destination (checked by the prologue: writeable, the result's shape).</param>
        /// <param name="rcount">The lists each output sums.</param>
        /// <returns><paramref name="out"/>.</returns>
        /// <exception cref="ValueError">A float or text out over at least one list.</exception>
        /// <exception cref="TypeError">An integer or complex out over at least one list; a bool out over two or more (the
        ///     truth plus a list); a text out over none (the divide has no loop).</exception>
        private NDArray MeanInto(NDArray @out, long rcount)
        {
            if (rcount > 0 && @out.size > 0)
            {
                if (@out.typecode == NPTypeCode.Boolean)
                {
                    // The first list's truth, cast back to a Python bool, plus the next list: CPython's TypeError.
                    if (rcount >= 2)
                        throw new TypeError("unsupported operand type(s) for +: 'bool' and 'list'");
                    // One list per output: its truth, divided by 1 (unchanged).
                    using var truth = Truth('C');
                    using var shaped = truth.reshape(Shape.Adopt((long[])@out.Shape.dimensions.Clone()));
                    np.copyto(@out, shaped);
                    return @out;
                }
                throw ListCastError(@out.dtype) ?? new TypeError($"cannot cast a list to {@out.dtype.ToString(true)}");
            }
            // No list is summed: every output holds the identity 0, then 0 / rcount (0: NaN in the out's arithmetic).
            if (@out.dtype.kind is 'U' or 'S')
                throw new TypeError("ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced " +
                                    "to any supported types according to the casting rule ''safe''");
            using (var zero = NDArray.Scalar(0L))
                np.copyto(@out, zero, casting: "unsafe");
            DefaultEngine.DivideByCountLikeNumPy(@out, rcount);
            return @out;
        }

        // =============================================================================================================
        // The engine
        // =============================================================================================================

        /// <summary>Whether <paramref name="where"/> is "no mask": null, or Python's True (the C# literal <c>true</c>).</summary>
        /// <param name="where">The caller's mask.</param>
        /// <returns>True for no mask.</returns>
        private static bool IsNoMask(NDArray where)
            => where is null || (where.ndim == 0 && where.WasPythonBool && where.GetBoolean());

        /// <summary>
        ///     One reduction (see the file header): NumPy's prologue, the start, the fold of every output in NumPy's order,
        ///     the result in a heap of its own, then <paramref name="out"/>.
        /// </summary>
        /// <param name="fold">The object loop.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="out">The list array to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="initial">The fold's start, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>The result (or <paramref name="out"/>).</returns>
        private NDListArray Reduce(ListFold fold, int? axis, NDListArray @out, bool keepdims, object initial, NDArray where)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            string ufunc = fold switch { ListFold.Max => "maximum", ListFold.Min => "minimum", ListFold.Sum => "add", _ => "multiply" };
            NDArray mask = IsNoMask(where) ? null : where;
            bool[] reduced = ReductionArguments.Validate(Ids, axis, keepdims, @out?.Ids, mask, ufunc);
            // get_initial_from_ufunc: an object add / multiply uses its identity (0 / 1) only when the iteration is empty;
            // maximum / minimum have none. Without an initial value a where= mask is refused.
            bool identityUsed = (fold == ListFold.Sum || fold == ListFold.Prod) && size == 0;
            if (mask is not null && initial is null && !identityUsed)
                throw new ValueError($"reduction operation '{ufunc}' does not have an identity, so to use a where mask one has to specify 'initial'");

            var start = Start.Of(initial, Kind);
            NDArray maskView = null;
            try
            {
                // The mask broadcast to the input (the prologue proved it broadcasts that way).
                if (mask is not null)
                    maskView = np.broadcast_to(mask, Shape.Adopt((long[])Ids.Shape.dimensions.Clone()));
                NDListArray computed;
                if (maskView is null && TrySelectAlongAxis(fold, reduced, start, out var selected))
                {
                    // The selected lists are this heap's own: the result shares it (a reference of its own).
                    var ids = ReductionIds(reduced, keepdims, selected);
                    computed = new NDListArray(ids, Heap.Share());
                }
                else
                {
                    computed = FoldSlices(fold, ufunc, Slices(reduced, maskView), start, reduced, keepdims);
                }
                // A reduction to one element without out= returns the object itself: the list.
                var result = computed.MarkStandIn();
                if (@out is null)
                    return result;
                @out.Receive(result);
                return @out;
            }
            finally
            {
                maskView?.Dispose();
                start.Dispose();
            }
        }

        /// <summary>
        ///     A reduction's slices, flat: output <c>o</c> (in C order over the kept axes) folds
        ///     <c>Ids[Bounds[o] .. Bounds[o + 1])</c> in NumPy's visiting order. One array for every output, where a list per
        ///     output cost an allocation per output (100,000 of them for a (100000, 10) sum along axis 1).
        /// </summary>
        private readonly struct SliceSet
        {
            /// <summary>Every output's ids, back to back.</summary>
            internal readonly long[] Ids;

            /// <summary>Output o's ids start at <c>Bounds[o]</c>; <c>Bounds[Count]</c> is the end of the last.</summary>
            internal readonly long[] Bounds;

            /// <summary>Creates the set.</summary>
            /// <param name="ids">The ids.</param>
            /// <param name="bounds">The bounds (one more than the outputs).</param>
            internal SliceSet(long[] ids, long[] bounds)
            {
                Ids = ids;
                Bounds = bounds;
            }

            /// <summary>The number of outputs.</summary>
            internal int Count => Bounds.Length - 1;

            /// <summary>Output <paramref name="o"/>'s ids.</summary>
            /// <param name="o">The output.</param>
            internal ReadOnlySpan<long> this[int o] => Ids.AsSpan(checked((int)Bounds[o]), checked((int)(Bounds[o + 1] - Bounds[o])));
        }

        /// <summary>
        ///     The reduction's slices: for each output (in C order over the kept axes), the ids it folds, in NumPy's visiting
        ///     order (masked-off elements skipped).
        /// </summary>
        /// <param name="reduced">The reduced-axis flags (every axis, or one).</param>
        /// <param name="mask">The mask broadcast to the input, or null.</param>
        /// <returns>The slices.</returns>
        private unsafe SliceSet Slices(bool[] reduced, NDArray mask)
        {
            var dims = Ids.Shape.dimensions;
            var strides = Ids.Shape.strides;
            int nd = dims.Length;
            long* ids = (long*)Ids.Address;
            bool* m = mask is null ? null : (bool*)mask.Address;
            var mstrides = mask?.Shape.strides;
            long moff = mask?.Shape.offset ?? 0;
            int axis = Array.IndexOf(reduced, false) < 0 ? -1 : Array.IndexOf(reduced, true);
            if (axis < 0 || nd == 0)
            {
                // Every axis: one output, NumPy's flat reduction order (the axes by |stride|, each in increasing index).
                var all = new long[size];
                long used = 0;
                if (size > 0)
                {
                    var perm = new int[Math.Max(nd, 1)];
                    int count = NumPySequentialReduce.FlatVisitOrder(dims, strides, perm);
                    var index = new long[count];
                    long offset = Ids.Shape.offset, mo = moff;
                    for (long k = 0; k < size; k++)
                    {
                        if (m is null || m[mo])
                            all[used++] = ids[offset];
                        for (int j = 0; j < count; j++)
                        {
                            int ax = perm[j];
                            index[j]++;
                            offset += strides[ax];
                            if (m is not null)
                                mo += mstrides[ax];
                            if (index[j] < dims[ax])
                                break;
                            offset -= index[j] * strides[ax];
                            if (m is not null)
                                mo -= index[j] * mstrides[ax];
                            index[j] = 0;
                        }
                    }
                }
                return new SliceSet(all, new long[] { 0, used });
            }
            // One axis: every output (C order over the other axes) folds its elements in increasing index.
            long n = dims[axis];
            long outer = 1;
            for (int d = 0; d < nd; d++)
                if (d != axis)
                    outer *= dims[d];
            var flat = new long[checked(outer * n)];
            var bounds = new long[outer + 1];
            long written = 0;
            var coord = new long[nd];
            long baseOffset = Ids.Shape.offset, baseMask = moff;
            for (long o = 0; o < outer; o++)
            {
                for (long k = 0; k < n; k++)
                {
                    if (m is null || m[baseMask + k * mstrides[axis]])
                        flat[written++] = ids[baseOffset + k * strides[axis]];
                }
                bounds[o + 1] = written;
                for (int d = nd - 1; d >= 0; d--)
                {
                    if (d == axis)
                        continue;
                    coord[d]++;
                    baseOffset += strides[d];
                    if (m is not null)
                        baseMask += mstrides[d];
                    if (coord[d] < dims[d])
                        break;
                    baseOffset -= coord[d] * strides[d];
                    if (m is not null)
                        baseMask -= coord[d] * mstrides[d];
                    coord[d] = 0;
                }
            }
            return new SliceSet(flat, bounds);
        }

        /// <summary>
        ///     maximum / minimum along ONE axis in NumPy's slab order, when nothing can raise: no mask, a non-empty axis, and
        ///     a start that is absent or a list of this heap (one piece class: every pair of lists is ordered). Each output's
        ///     fold still visits its elements in increasing index, so the selected lists are the per-output folds' —
        ///     only the interleaving of the outputs changes, which only an error could observe.
        /// </summary>
        /// <param name="fold">The object loop (only maximum / minimum are selected here).</param>
        /// <param name="reduced">The reduced-axis flags.</param>
        /// <param name="start">The fold's start.</param>
        /// <param name="selected">Receives each output's selected list id, in C order over the kept axes.</param>
        /// <returns>False when the general fold must run (another loop, every axis reduced, an empty axis, a start of another
        ///     heap or not a list).</returns>
        /// <remarks>
        ///     When the reduced axis is the innermost in memory each output's elements are adjacent and the outputs fold one
        ///     after another (a ROW); otherwise every step along the axis compares a whole row of accumulators with the next
        ///     row of elements (a SLAB) — NumPy's reduction iterator order, one pass over the heap in memory order. Folding
        ///     the outputs one after another there read the heap once per output: <c>max(axis=0)</c> of a (100000, 10) split
        ///     result walked its 50 MB heap ten times (1.26x NumPy, against 4.4x for the flat maximum).
        /// </remarks>
        private unsafe bool TrySelectAlongAxis(ListFold fold, bool[] reduced, Start start, out long[] selected)
        {
            selected = null;
            if (fold != ListFold.Max && fold != ListFold.Min)
                return false;
            int axis = Array.IndexOf(reduced, true);
            if (axis < 0 || Array.IndexOf(reduced, false) < 0)
                return false;
            if (start.Present && !(start.IsList && ReferenceEquals(start.Heap, Heap)))
                return false;
            var dims = Ids.Shape.dimensions;
            var strides = Ids.Shape.strides;
            int nd = dims.Length;
            long n = dims[axis];
            if (n == 0)
                return false;
            // Every output's first element, C order over the kept axes (an odometer over the kept axes, as Slices walks).
            long outer = 1;
            for (int d = 0; d < nd; d++)
                if (d != axis)
                    outer *= dims[d];
            var bases = new long[outer];
            var coord = new long[nd];
            long offset = Ids.Shape.offset;
            for (long o = 0; o < outer; o++)
            {
                bases[o] = offset;
                for (int d = nd - 1; d >= 0; d--)
                {
                    if (d == axis)
                        continue;
                    coord[d]++;
                    offset += strides[d];
                    if (coord[d] < dims[d])
                        break;
                    offset -= coord[d] * strides[d];
                    coord[d] = 0;
                }
            }
            long* ids = (long*)Ids.Address;
            long step = strides[axis];
            var lists = new ListPointers(Heap);
            bool greatest = fold == ListFold.Max;
            var acc = new long[outer];
            long first;
            if (start.Present)
            {
                Array.Fill(acc, start.Id);
                first = 0;
            }
            else
            {
                for (long o = 0; o < outer; o++)
                    acc[o] = ids[bases[o]];
                first = 1;
            }
            // ROW mode when no kept axis longer than 1 has a smaller |stride| than the reduced axis.
            bool row = true;
            long reducedStride = Math.Abs(step);
            for (int d = 0; d < nd; d++)
            {
                if (d != axis && dims[d] > 1 && Math.Abs(strides[d]) < reducedStride)
                    row = false;
            }
            if (row)
            {
                for (long o = 0; o < outer; o++)
                {
                    long a = acc[o];
                    long* line = ids + bases[o];
                    for (long k = first; k < n; k++)
                    {
                        // npy_ObjectMax keeps the accumulator while acc >= x, npy_ObjectMin while acc <= x (a tie keeps it).
                        long x = line[k * step];
                        int c = lists.Compare(a, x);
                        if (greatest ? c < 0 : c > 0)
                            a = x;
                    }
                    acc[o] = a;
                }
            }
            else
            {
                for (long k = first; k < n; k++)
                {
                    long* row0 = ids + k * step;
                    for (long o = 0; o < outer; o++)
                    {
                        long x = row0[bases[o]];
                        int c = lists.Compare(acc[o], x);
                        if (greatest ? c < 0 : c > 0)
                            acc[o] = x;
                    }
                }
            }
            selected = acc;
            return true;
        }

        /// <summary>
        ///     Folds every slice into one list and lays the results out as NumPy's reduction does. maximum / minimum SELECT
        ///     lists that exist, so their result points into this array's own heap (NumPy's result holds the same list
        ///     objects; a winning <c>initial</c> of another heap is appended to it) — unless the start is a non-empty list of
        ///     the other piece class (<see cref="IsForeignStart"/>), whose class the result then takes; a sum or a product
        ///     makes new lists, in a heap of their own — a sum's sized exactly before anything is copied.
        /// </summary>
        /// <param name="fold">The object loop.</param>
        /// <param name="ufunc">The ufunc's name (for NumPy's messages).</param>
        /// <param name="slices">The slices, one per output in C order over the kept axes.</param>
        /// <param name="start">The fold's start.</param>
        /// <param name="reduced">The reduced-axis flags.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">An empty slice of maximum / minimum without a start.</exception>
        /// <exception cref="TypeError">A fold whose operands cannot combine (the first output that raises).</exception>
        /// <exception cref="NotSupportedException">A result NumSharp cannot hold (an int identity, a list of str and bytes
        ///     items).</exception>
        private NDListArray FoldSlices(ListFold fold, string ufunc, SliceSet slices, Start start, bool[] reduced, bool keepdims)
        {
            int outputs = slices.Count;
            // NumPy's errors before any result: an empty slice of maximum / minimum, then the TypeErrors of a fold whose
            // operands cannot combine (the first output in order that raises), then the identities NumSharp cannot hold.
            for (int o = 0; o < outputs; o++)
            {
                if (slices[o].Length == 0 && !start.Present && (fold == ListFold.Max || fold == ListFold.Min))
                    throw new ValueError($"zero-size array to reduction operation {ufunc} which has no identity");
            }
            for (int o = 0; o < outputs; o++)
                CheckFoldTypes(fold, slices[o], start);
            bool selects = fold == ListFold.Max || fold == ListFold.Min;
            // A sum's heap is created at its exact size (the whole result counted first); a selection shares this heap.
            var heap = fold == ListFold.Sum ? ReservedSumHeap(slices, start)
                : selects && !IsForeignStart(start) ? Heap.Share()
                : NewHeapLike(start);
            var memo = new Dictionary<(NDListHeap, long), long>();
            var resultIds = new long[outputs];
            try
            {
                for (int o = 0; o < outputs; o++)
                    resultIds[o] = FoldOne(fold, slices[o], start, heap, memo);
                var ids = ReductionIds(reduced, keepdims, resultIds);
                return new NDListArray(ids, heap);
            }
            catch
            {
                heap.Release();
                throw;
            }
        }

        /// <summary>A sum's result heap, created at the whole result's exact size: per output, the start's pieces and
        ///     every element's (the class of <see cref="NewHeapLike"/>).</summary>
        /// <param name="slices">The slices.</param>
        /// <param name="start">The fold's start.</param>
        /// <returns>The heap (one reference, the caller's).</returns>
        /// <exception cref="OutOfMemoryException">The result's size passes what an array can address (MemoryError).</exception>
        private NDListHeap ReservedSumHeap(SliceSet slices, Start start)
        {
            var lists = new ListPointers(Heap);
            bool hasStart = start.Present && start.IsList;
            long startUnits = 0, startPieces = 0;
            if (hasStart)
            {
                var first = new ListPointers(start.Heap);
                startUnits = first.UnitCount(start.Id);
                startPieces = first.Length(start.Id);
            }
            long units = 0, pieces = 0;
            for (int o = 0; o < slices.Count; o++)
            {
                units = AddSize(units, startUnits);
                pieces = AddSize(pieces, startPieces);
                foreach (long id in slices[o])
                {
                    units = AddSize(units, lists.UnitCount(id));
                    pieces = AddSize(pieces, lists.Length(id));
                }
            }
            return NDListHeap.Reserved(IsForeignStart(start) ? start.Heap.Kind : Kind, units, pieces, slices.Count);
        }

        /// <summary>
        ///     The TypeErrors a slice's fold meets before it combines anything (NumPy raises at the first object op that
        ///     fails): a non-list start combined with a list, a list multiplied by a list.
        /// </summary>
        /// <param name="fold">The object loop.</param>
        /// <param name="slice">The slice.</param>
        /// <param name="start">The fold's start.</param>
        /// <exception cref="TypeError">CPython's text for the first failing operation.</exception>
        /// <exception cref="NotSupportedException">A non-list result (a non-list start over an empty slice, an empty slice of
        ///     add / multiply without a start).</exception>
        private void CheckFoldTypes(ListFold fold, ReadOnlySpan<long> slice, Start start)
        {
            if (slice.Length == 0)
            {
                if (start.Present && start.IsList)
                    return;
                if (start.Present)
                    throw new NotSupportedException(
                        $"the reduction's result is the Python {start.TypeName} 'initial' itself, which an NDListArray (a list array) cannot hold");
                throw new NotSupportedException(
                    $"an empty {(fold == ListFold.Sum ? "sum" : "product")} of lists is NumPy's int {(fold == ListFold.Sum ? 0 : 1)} " +
                    "(the ufunc's identity), which an NDListArray cannot hold; give a list as 'initial'");
            }
            if (start.Present && !start.IsList)
            {
                string t = start.TypeName;
                switch (fold)
                {
                    case ListFold.Max:
                        throw new TypeError($"'>=' not supported between instances of '{t}' and 'list'");
                    case ListFold.Min:
                        throw new TypeError($"'<=' not supported between instances of '{t}' and 'list'");
                    case ListFold.Sum:
                        throw new TypeError(t switch
                        {
                            "str" => "can only concatenate str (not \"list\") to str",
                            "bytes" => "can't concat list to bytes",
                            _ => $"unsupported operand type(s) for +: '{t}' and 'list'",
                        });
                    case ListFold.Prod:
                        // n * list repeats the list (an int or bool start); the second element multiplies two lists.
                        if (start.Repeat is not null)
                        {
                            if (slice.Length >= 2)
                                throw new TypeError("can't multiply sequence by non-int of type 'list'");
                            return;
                        }
                        throw new TypeError(t is "float" or "complex" or "str" or "bytes"
                            ? $"can't multiply sequence by non-int of type '{(t is "str" or "bytes" ? "list" : t)}'"
                            : $"unsupported operand type(s) for *: '{t}' and 'list'");
                }
            }
            if (fold == ListFold.Prod && (slice.Length >= 2 || start.Present))
                throw new TypeError("can't multiply sequence by non-int of type 'list'");
        }

        /// <summary>One slice's fold into <paramref name="heap"/>: a sum's new list (copied into the room
        ///     <see cref="ReserveSums"/> made), a product's repetition or copy, the selected list of maximum / minimum (this
        ///     array's own id when <paramref name="heap"/> is this array's heap).</summary>
        /// <param name="fold">The object loop.</param>
        /// <param name="slice">The ids folded, in order.</param>
        /// <param name="start">The fold's start.</param>
        /// <param name="heap">The result heap.</param>
        /// <param name="memo">Lists already copied into <paramref name="heap"/> (source heap and id → result id).</param>
        /// <returns>The result list's id in <paramref name="heap"/>.</returns>
        /// <exception cref="NotSupportedException">A concatenation of str and bytes items.</exception>
        /// <exception cref="TypeError">A str item ordered against a bytes item (a start of the other class).</exception>
        private long FoldOne(ListFold fold, ReadOnlySpan<long> slice, Start start, NDListHeap heap, Dictionary<(NDListHeap, long), long> memo)
        {
            switch (fold)
            {
                case ListFold.Sum:
                {
                    // A concatenation is a new list: the start's pieces, then every element's, in order.
                    CheckSameClass(start, slice);
                    var lists = new ListPointers(Heap);
                    heap.BeginList();
                    if (start.Present)
                        heap.AppendReserved(new ListPointers(start.Heap), start.Id);
                    foreach (long id in slice)
                        heap.AppendReserved(lists, id);
                    return heap.EndListReserved();
                }
                case ListFold.Prod:
                {
                    if (start.Repeat is { } times)
                    {
                        // n * x0: the list repeated (a non-positive n gives an empty list).
                        heap.BeginList();
                        heap.AppendListPieces(Heap, slice[0], times);
                        return heap.EndList();
                    }
                    return start.Present ? CopyList(start.Heap, start.Id, heap, memo) : CopyList(Heap, slice[0], heap, memo);
                }
                default:
                {
                    bool greatest = fold == ListFold.Max;
                    if (!start.Present || ReferenceEquals(start.Heap, Heap))
                    {
                        // One heap: Python's list order through the raw pointers (no str / bytes pair can meet, and nothing
                        // is appended while they are held).
                        var lists = new ListPointers(Heap);
                        long acc;
                        int first;
                        if (start.Present)
                        {
                            acc = start.Id;
                            first = 0;
                        }
                        else
                        {
                            acc = slice[0];
                            first = 1;
                        }
                        for (int k = first; k < slice.Length; k++)
                        {
                            // npy_ObjectMax keeps the accumulator while acc >= x, npy_ObjectMin while acc <= x (a tie keeps it).
                            int c = lists.Compare(acc, slice[k]);
                            if (greatest ? c < 0 : c > 0)
                                acc = slice[k];
                        }
                        return CopyList(Heap, acc, heap, memo);
                    }
                    // A start of another heap: Python's rich comparison across the heaps (its str / bytes TypeError).
                    var op = greatest ? PyCompareOp.Ge : PyCompareOp.Le;
                    NDListHeap accHeap = start.Heap;
                    long accId = start.Id;
                    for (int k = 0; k < slice.Length; k++)
                    {
                        if (!RichCompare(op, accHeap, accId, Heap, slice[k]))
                        {
                            accHeap = Heap;
                            accId = slice[k];
                        }
                    }
                    return CopyList(accHeap, accId, heap, memo);
                }
            }
        }

        /// <summary>
        ///     The result ids laid out as NumPy's reduction result: a 0-d array (or (1, …, 1) with keepdims) for every axis,
        ///     else the kept axes in the input's memory order (<see cref="Shape.ReductionOutputShape"/>).
        /// </summary>
        /// <param name="reduced">The reduced-axis flags.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="resultIds">The outputs' list ids, in C order over the kept axes.</param>
        /// <returns>The id array.</returns>
        private NDArray ReductionIds(bool[] reduced, bool keepdims, long[] resultIds)
        {
            var dims = Ids.Shape.dimensions;
            int nd = dims.Length;
            var outDims = new List<long>();
            int axis = -1;
            for (int d = 0; d < nd; d++)
            {
                if (!reduced[d])
                    outDims.Add(dims[d]);
                else
                {
                    axis = axis == -1 ? d : -2;
                    if (keepdims)
                        outDims.Add(1);
                }
            }
            var od = outDims.ToArray();
            Shape layout = axis >= 0 && od.Length > 0 && Array.IndexOf(reduced, false) >= 0
                ? Shape.ReductionOutputShape(od, Ids.Shape, axis)
                : Shape.Adopt(od);
            var ids = new NDArray(NPTypeCode.Int64, layout, false);
            // The outputs come in C order over the kept axes: write them through a C-ordered view of the result.
            using (var flat = ids.reshape(Shape.Adopt((long[])ids.Shape.dimensions.Clone())))
            {
                for (long i = 0; i < resultIds.Length; i++)
                    flat.SetAtIndex(resultIds[i], i);
            }
            return ids;
        }

        /// <summary>
        ///     NumPy's accumulate of lists (cumsum / cumprod): each line's running concatenation (or its first element and
        ///     the TypeError of the second product), in the array's memory order (1-D C order for <c>axis=None</c>) — after
        ///     <c>PyArray_CumSum</c>'s checks in its order: the axis, the dtype's class, a read-only <paramref name="out"/>,
        ///     the loop, then <paramref name="out"/>'s route checks (<see cref="AccumulateOutRegion"/>, whose zero-size returns
        ///     come before any element is read).
        /// </summary>
        /// <param name="fold">Sum or Prod.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="dtype">The loop dtype, or null (the object loop).</param>
        /// <param name="out">The list array to store into, or null.</param>
        /// <returns>The result (or <paramref name="out"/>).</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError">A read-only or ill-fitting <paramref name="out"/>; a float <paramref name="dtype"/>
        ///     over lists.</exception>
        /// <exception cref="TypeError">A refused or loop-less <paramref name="dtype"/>, or an integer / complex one over lists;
        ///     <c>can't multiply sequence by non-int of type 'list'</c>.</exception>
        /// <exception cref="NotSupportedException">The result of a <paramref name="dtype"/> loop (not lists).</exception>
        private unsafe NDListArray Accumulate(ListFold fold, int? axis, DType dtype, NDListArray @out)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            string ufunc = fold == ListFold.Sum ? "add" : "multiply";
            NDArray source;
            int ax;
            if (axis is null)
            {
                source = Ids.ndim == 1 ? Ids : Ids.ravel();
                ax = 0;
            }
            else
            {
                // PyArray_CheckAxis: a 0-d array is one element along axis 0 / -1.
                ax = ndim == 0 ? DefaultEngine.check_and_adjust_axis(1, axis.Value) : DefaultEngine.check_and_adjust_axis(ndim, axis.Value);
                source = ndim == 0 ? Ids.reshape(1) : Ids;
            }
            try
            {
                // ufunc.accumulate(arr, axis, dtype=PyArray_DescrFromType(rtype), out=out): _get_dtype (the type number's
                // descriptor), then _set_out_array.
                if (dtype is not null)
                    CheckUfuncDtype(dtype, typeNumberOnly: true);
                if (@out is not null && !@out.Ids.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
                // PyUFunc_Accumulate: the loop first, then the checks of its out route.
                if (dtype is not null)
                    CheckObjectReduceLoop(dtype, ufunc);
                NDArray region = null;
                if (@out is not null)
                {
                    region = AccumulateOutRegion(@out.Ids, source, ax, castLoop: dtype is not null);
                    // A zero-size input or out: NumPy returns out without touching it.
                    if (region is null)
                        return @out;
                }
                using (region)
                {
                    long n = source.ndim == 0 ? 1 : source.Shape.dimensions[ax];
                    if (dtype is not null)
                    {
                        // The first element's cast to the loop (a conversion error), else a result that is not lists: the
                        // bool loop's truths, or an empty array of the dtype.
                        if (source.size > 0 && ListCastError(dtype) is { } conversion)
                            throw conversion;
                        throw LoopResult(fold == ListFold.Sum ? "cumsum" : "cumprod", dtype);
                    }
                    if (fold == ListFold.Prod && n >= 2 && source.size > 0)
                        throw new TypeError("can't multiply sequence by non-int of type 'list'");
                    var result = AccumulateLists(source, ax, axis is null);
                    if (@out is null)
                        return result;
                    // The running lists over out's region: each line's first n positions (NumPy's count is the input's
                    // extent), the input broadcast over out's other axes, the lists put into out's heap.
                    using (result)
                        @out.AssignFrom(region, result);
                    return @out;
                }
            }
            finally
            {
                if (!ReferenceEquals(source, Ids))
                    source.Dispose();
            }
        }

        /// <summary>
        ///     The accumulation of <see cref="Accumulate"/>: each line's running concatenation along <paramref name="ax"/>
        ///     (a product's lines hold at most one element here), in a heap of its own sized once.
        /// </summary>
        /// <param name="source">The ids accumulated (this array's, its ravel, or a 0-d array's one-element form).</param>
        /// <param name="ax">The accumulation axis.</param>
        /// <param name="flat">Whether the result is the 1-D accumulation of the flattened array (NumPy's axis=None).</param>
        /// <returns>The running lists, in the source's memory order (1-D for <paramref name="flat"/>).</returns>
        private unsafe NDListArray AccumulateLists(NDArray source, int ax, bool flat)
        {
            var layout = flat ? Shape.Vector(source.size) : source.Shape.KeepOrder();
            NDListHeap heap = null;
            var ids = new NDArray(NPTypeCode.Int64, layout, false);
            try
            {
                long* src = (long*)source.Address;
                long* dst = (long*)ids.Address;
                var lists = new ListPointers(Heap);
                // Pass 1: the exact size of every running list (a line's k-th output holds its first k + 1 lists).
                long units = 0, pieces = 0;
                ForEachLine(source.Shape, ax, (start, stride, length) =>
                {
                    long lineUnits = 0, linePieces = 0;
                    for (long k = 0; k < length; k++)
                    {
                        long id = src[start + k * stride];
                        lineUnits += lists.UnitCount(id);
                        linePieces += lists.Length(id);
                        units = AddSize(units, lineUnits);
                        pieces = AddSize(pieces, linePieces);
                    }
                });
                // The heap at its exact size, its flags too when the source has any: pass 2 reads the result heap
                // through pointers captured once, and a flags array allocated by the first np.str_ piece AFTER that
                // capture would be invisible to it (the running lists would drop their pieces' np.str_ flags).
                heap = NDListHeap.Reserved(Kind, units, pieces, source.size, withFlags: Heap.PieceFlags != null);
                // Pass 2: each running list is the previous one (read from the result heap itself, whose arrays a
                // reserved heap never moves) followed by the next element's.
                var own = new ListPointers(heap);
                var dstLines = new List<long>();
                ForEachLine(ids.Shape, ax, (start, stride, length) => dstLines.Add(start));
                long dstStride = ids.Shape.strides[ax];
                int lineNo = 0;
                ForEachLine(source.Shape, ax, (start, stride, length) =>
                {
                    long dstStart = dstLines[lineNo++];
                    long previous = -1;
                    for (long k = 0; k < length; k++)
                    {
                        long id = src[start + k * stride];
                        heap.BeginList();
                        if (previous >= 0)
                            heap.AppendReserved(own, previous);
                        heap.AppendReserved(lists, id);
                        previous = heap.EndListReserved();
                        dst[dstStart + k * dstStride] = previous;
                    }
                });
                return new NDListArray(ids, heap);
            }
            catch
            {
                ids.Dispose();
                heap?.Release();
                throw;
            }
        }

        /// <summary>
        ///     <c>PyUFunc_Accumulate</c>'s checks of an <c>out=</c> array, route by route, and the region of it the
        ///     accumulation writes. The plain loop (a 1-D object accumulation into an out that does not overlap the input)
        ///     returns early for an empty input or out and refuses any other shape than the input's; the outer iterator (two
        ///     or more dimensions, a cast loop, an overlapping out) maps axis k of each operand to iterator axis k, out
        ///     READWRITE and NO_BROADCAST, the input READONLY, with <c>npyiter_fill_axisdata</c>'s two passes and texts, and
        ///     returns early for an empty input or out only after them.
        /// </summary>
        /// <param name="outIds">The out's ids.</param>
        /// <param name="source">The ids accumulated (1-D for a flattened or 0-d array).</param>
        /// <param name="ax">The accumulation axis of <paramref name="source"/>.</param>
        /// <param name="castLoop">Whether the loop casts the input (a dtype= of another class: always the iterator).</param>
        /// <returns>A view of <paramref name="outIds"/> to write (out's extra trailing axes read at index 0, the
        ///     accumulation axis cut to the input's extent), or null when NumPy returns out untouched.</returns>
        /// <exception cref="ValueError">NumPy's texts: <c>provided out is the wrong size for the accumulation.</c> (also for a
        ///     0-d out, where NumPy 2.4.2 crashes), <c>Iterator input op_axes[0][0] (==1) is not a valid axis of op[0], which
        ///     has 1 dimensions </c>, <c>operands could not be broadcast together with remapped shapes
        ///     [original-&gt;remapped]: (3,2)-&gt;(3,2) (2,3)-&gt;(2,3) </c>, <c>non-broadcastable output operand with shape
        ///     (1,3) [remapped to (3,1)] doesn't match the broadcast shape (2,3)</c>.</exception>
        private static NDArray AccumulateOutRegion(NDArray outIds, NDArray source, int ax, bool castLoop)
        {
            int nd = source.ndim;
            long[] o = outIds.Shape.dimensions, s = source.Shape.dimensions;
            // need_outer_iterator: more than one axis, a cast, or an out that may share memory with the input.
            bool iterator = nd > 1 || castLoop || NDMemOverlap.SolveMayShareMemory(outIds, source, 0) != MemOverlap.No;
            if (!iterator)
            {
                // The plain loop: its zero-size returns come before its size check.
                if (s[ax] == 0 || outIds.size == 0)
                    return null;
                if (outIds.ndim != 1 || o[0] != s[0])
                    throw new ValueError("provided out is the wrong size for the accumulation.");
                return outIds.view();
            }
            // Pass 1 (op_axes 0 .. nd-1): an out with fewer axes has no axis k; the first nd extents broadcast, out first.
            if (outIds.ndim < nd)
                throw new ValueError($"Iterator input op_axes[0][{nd - outIds.ndim - 1}] (=={outIds.ndim}) is not a valid axis of op[0], " +
                                     $"which has {outIds.ndim} dimensions ");
            var mapped = new long[nd];
            Array.Copy(o, mapped, nd);
            var broadcast = new long[nd];
            for (int d = 0; d < nd; d++)
            {
                long b = mapped[d];
                if (b == 1)
                    b = s[d];
                else if (s[d] != b && s[d] != 1)
                    throw new ValueError("operands could not be broadcast together with remapped shapes [original->remapped]: " +
                                         $"{outIds.Shape.ToPythonTuple()}->{Shape.Adopt(mapped).ToPythonTuple()} " +
                                         $"{source.Shape.ToPythonTuple()}->{source.Shape.ToPythonTuple()} ");
                broadcast[d] = b;
            }
            // Pass 2: out is NO_BROADCAST — an extent 1 where the broadcast is not 1 (printed in iterator order, reversed).
            for (int d = 0; d < nd; d++)
            {
                if (broadcast[d] != 1 && mapped[d] == 1)
                {
                    var reversed = (long[])mapped.Clone();
                    Array.Reverse(reversed);
                    throw new ValueError($"non-broadcastable output operand with shape {outIds.Shape.ToPythonTuple()} " +
                                         $"[remapped to {Shape.Adopt(reversed).ToPythonTuple()}] doesn't match the broadcast shape " +
                                         $"{Shape.Adopt(broadcast).ToPythonTuple()}");
                }
            }
            // The zero-size returns, after the iterator was built.
            if (s[ax] == 0 || outIds.size == 0)
                return null;
            // The written region: the iterator walks out's first nd axes from its data pointer (extra axes at index 0), and
            // each line runs the input's extent along the axis (an input of extent 1 writes one position).
            var slices = new Slice[outIds.ndim];
            for (int d = 0; d < outIds.ndim; d++)
                slices[d] = d >= nd ? Slice.Index(0) : d == ax && o[d] != s[d] ? new Slice(0, s[d]) : Slice.All;
            return outIds[slices];
        }

        /// <summary>
        ///     The clip loop over the operands broadcast to <paramref name="dims"/> (each bound a list array, or absent):
        ///     npy_ObjectMax with the lower bound, then npy_ObjectMin with the upper — at every position, or only where
        ///     <paramref name="maskView"/> is true (a masked-off position holds a valid id the caller never reads: this
        ///     array's own list there, or an empty list of the result's heap).
        /// </summary>
        /// <param name="lo">The lower bound operand (absent, or list arrays).</param>
        /// <param name="hi">The upper bound operand (absent, or list arrays).</param>
        /// <param name="dims">The broadcast dimensions.</param>
        /// <param name="layout">The result's layout (NumPy's ufunc output layout over the operands, for a new result).</param>
        /// <param name="maskView">The mask broadcast to <paramref name="dims"/>, or null for every position.</param>
        /// <returns>The clipped lists, in this array's heap (every operand in it) or a heap of their own.</returns>
        /// <exception cref="TypeError">A str list item ordered against a bytes one (Python's comparison across the
        ///     classes).</exception>
        /// <exception cref="NotSupportedException">A selected list of the other piece class in a result of this class.</exception>
        private unsafe NDListArray ClipCore(Bound lo, Bound hi, long[] dims, Shape layout, NDArray maskView)
        {
            long count = 1;
            foreach (var d in dims)
                count *= d;
            // An empty broadcast compares nothing: the empty result in the ufunc's output layout.
            if (count == 0)
                return new NDListArray(new NDArray(NPTypeCode.Int64, layout, false), NDListHeap.Empty(Kind));
            if ((!lo.Present || ReferenceEquals(lo.Lists.Heap, Heap)) && (!hi.Present || ReferenceEquals(hi.Lists.Heap, Heap)))
                return ClipOneHeap(lo, hi, layout, maskView);
            var heap = NDListHeap.Empty(Kind);
            var ids = new NDArray(NPTypeCode.Int64, layout, false);
            var memo = new Dictionary<(NDListHeap, long), long>();
            try
            {
                using var a = np.broadcast_to(Ids, Shape.Adopt((long[])dims.Clone()));
                using var bl = lo.Present ? np.broadcast_to(lo.Lists.Ids, Shape.Adopt((long[])dims.Clone())) : null;
                using var bh = hi.Present ? np.broadcast_to(hi.Lists.Ids, Shape.Adopt((long[])dims.Clone())) : null;
                using var flatA = a.flatten();
                using var flatL = bl?.flatten();
                using var flatH = bh?.flatten();
                using var flatMask = maskView?.flatten();
                using var flatOut = ids.reshape(Shape.Adopt((long[])dims.Clone()));
                // One empty list stands at the masked-off positions (created the first time one is met).
                long placeholder = -1;
                for (long i = 0; i < count; i++)
                {
                    if (flatMask is not null && !flatMask.GetBoolean(i))
                    {
                        if (placeholder < 0)
                        {
                            heap.BeginList();
                            placeholder = heap.EndList();
                        }
                        flatOut.SetAtIndex(placeholder, i);
                        continue;
                    }
                    NDListHeap h = Heap;
                    long id = flatA.GetInt64(i);
                    if (flatL is not null && !RichCompare(PyCompareOp.Ge, h, id, lo.Lists.Heap, flatL.GetInt64(i)))
                    {
                        h = lo.Lists.Heap;
                        id = flatL.GetInt64(i);
                    }
                    if (flatH is not null && !RichCompare(PyCompareOp.Le, h, id, hi.Lists.Heap, flatH.GetInt64(i)))
                    {
                        h = hi.Lists.Heap;
                        id = flatH.GetInt64(i);
                    }
                    flatOut.SetAtIndex(CopyList(h, id, heap, memo), i);
                }
                return new NDListArray(ids, heap);
            }
            catch
            {
                ids.Dispose();
                heap.Release();
                throw;
            }
        }

        /// <summary>
        ///     The clip loop when every operand's lists live in this array's heap: the selected lists are the operands' own
        ///     (NumPy's result holds the same list objects), so the result shares the heap and no list is copied, and the
        ///     broadcast operands are walked once in NumPy's ufunc order.
        /// </summary>
        /// <param name="lo">The lower bound operand (lists of this heap, or absent).</param>
        /// <param name="hi">The upper bound operand (lists of this heap, or absent).</param>
        /// <param name="layout">The result's layout over the broadcast dimensions (NumPy's ufunc output layout over the
        ///     operands, or plain C order when the result is delivered into an out=): the walk broadcasts every operand to
        ///     it.</param>
        /// <param name="maskView">The mask broadcast to the result's dimensions, or null: a masked-off position keeps this
        ///     array's own list (never compared, never read by the caller).</param>
        /// <returns>The clipped lists.</returns>
        private unsafe NDListArray ClipOneHeap(Bound lo, Bound hi, Shape layout, NDArray maskView)
        {
            var heap = Heap.Share();
            var ids = new NDArray(NPTypeCode.Int64, layout, false);
            try
            {
                var lists = new ListPointers(Heap);
                // The walk broadcasts the operands to the result: this array's ids, each bound's ids, the mask, the result.
                var ops = new List<NDArray> { Ids };
                if (lo.Present)
                    ops.Add(lo.Lists.Ids);
                if (hi.Present)
                    ops.Add(hi.Lists.Ids);
                if (maskView is not null)
                    ops.Add(maskView);
                ops.Add(ids);
                var flags = new NDIterPerOpFlags[ops.Count];
                for (int i = 0; i < flags.Length - 1; i++)
                    flags[i] = NDIterPerOpFlags.READONLY;
                flags[^1] = NDIterPerOpFlags.WRITEONLY;
                bool hasLo = lo.Present, hasHi = hi.Present, hasMask = maskView is not null;
                int maskIndex = 1 + (hasLo ? 1 : 0) + (hasHi ? 1 : 0);
                int outIndex = ops.Count - 1;
                Walk(ops.ToArray(), flags, (dataptrs, strides, n, aux) =>
                {
                    byte* px = (byte*)dataptrs[0];
                    byte* pl = hasLo ? (byte*)dataptrs[1] : null;
                    byte* ph = hasHi ? (byte*)dataptrs[hasLo ? 2 : 1] : null;
                    byte* pm = hasMask ? (byte*)dataptrs[maskIndex] : null;
                    byte* po = (byte*)dataptrs[outIndex];
                    long sx = strides[0], sl = hasLo ? strides[1] : 0, sh = hasHi ? strides[hasLo ? 2 : 1] : 0;
                    long sm = hasMask ? strides[maskIndex] : 0, so = strides[outIndex];
                    for (long k = 0; k < n; k++, px += sx, pl += sl, ph += sh, pm += sm, po += so)
                    {
                        long x = *(long*)px;
                        // A masked-off position is not compared: NumPy's masked loop never visits it.
                        if (hasMask && *pm == 0)
                        {
                            *(long*)po = x;
                            continue;
                        }
                        // npy_ObjectMax(x, lo): x unless x >= lo is false; then npy_ObjectMin(., hi): it unless it <= hi is false.
                        if (hasLo && lists.Compare(x, *(long*)pl) < 0)
                            x = *(long*)pl;
                        if (hasHi && lists.Compare(x, *(long*)ph) > 0)
                            x = *(long*)ph;
                        *(long*)po = x;
                    }
                });
                return new NDListArray(ids, heap);
            }
            catch
            {
                ids.Dispose();
                heap.Release();
                throw;
            }
        }

        /// <summary>The broadcast dimensions of every operand, with NumPy's ufunc text (every operand's shape, a trailing
        ///     space) when they do not broadcast.</summary>
        /// <param name="shapes">The operands' shapes.</param>
        /// <returns>The broadcast dimensions.</returns>
        /// <exception cref="IncorrectShapeException"><c>operands could not be broadcast together with shapes (4,) (3,) </c>.</exception>
        private static long[] BroadcastAll(List<Shape> shapes)
        {
            try
            {
                var b = Shape.Broadcast(shapes.ToArray());
                return (long[])b[0].dimensions.Clone();
            }
            catch (IncorrectShapeException e)
            {
                var sb = new StringBuilder("operands could not be broadcast together with shapes ");
                foreach (var s in shapes)
                    sb.Append(s.ToPythonTuple()).Append(' ');
                throw new IncorrectShapeException(sb.ToString(), e);
            }
        }

        /// <summary>
        ///     NumPy's <c>_count_reduce_items(arr, axis, keepdims, where)</c>, the first statement of <c>_mean</c> and
        ///     <c>_var</c>: the axis through <c>normalize_axis_index</c> (a 0-d array has no axis at all — ufunc.reduce's
        ///     0 / -1 exception does not apply), then the mask broadcast to the array, then the count each output reduces.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="mask">The caller's mask, or null.</param>
        /// <returns>The elements each output reduces (with a mask only an EMPTY array goes on past the object loop's missing
        ///     identity — and its counts are 0 —, so the mask's per-output counts are not needed).</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="IncorrectShapeException">The mask does not broadcast to the array (np.broadcast_to's error).</exception>
        private long StatisticsCount(int? axis, NDArray mask)
        {
            long items = axis is null ? size : Ids.Shape.dimensions[DefaultEngine.check_and_adjust_axis(ndim, axis.Value)];
            if (mask is not null)
            {
                // broadcast_to(where, arr.shape) runs before any reduction, whatever the mask's dtype.
                using var spread = np.broadcast_to(mask, Shape.Adopt((long[])Ids.Shape.dimensions.Clone()));
                if (size == 0)
                    items = 0;
            }
            return items;
        }

        /// <summary>The outputs of a reduction that keeps its reduced axes (the product of the kept extents).</summary>
        /// <param name="reduced">The reduced-axis flags.</param>
        /// <returns>The number of outputs (1 when every axis is reduced).</returns>
        private long KeptOutputs(bool[] reduced)
        {
            long n = 1;
            var dims = Ids.Shape.dimensions;
            for (int d = 0; d < dims.Length; d++)
            {
                if (!reduced[d])
                    n *= dims[d];
            }
            return n;
        }

        /// <summary>
        ///     <c>ufunc_object.c</c>'s <c>_get_dtype</c>: the <c>dtype=</c> of a ufunc call, reduce or accumulate names a
        ///     DType CLASS, so a descriptor that is not its class's canonical one is refused, and NEP 55's StringDType (a
        ///     new-style class) is refused as an instance. A C method (cumsum, cumprod, trace) passes
        ///     <c>PyArray_DescrFromType(type_num)</c> instead of the caller's descriptor, so only StringDType still fails
        ///     there.
        /// </summary>
        /// <param name="dtype">The caller's dtype.</param>
        /// <param name="typeNumberOnly">True for a C method's type-number descriptor.</param>
        /// <exception cref="TypeError">NumPy's two texts.</exception>
        private static void CheckUfuncDtype(DType dtype, bool typeNumberOnly)
        {
            if (dtype.Meta is VStringDTypeMeta)
                throw new TypeError("Cannot pass a new user DType instance to the `dtype` or `signature` arguments of ufuncs. Pass the DType class instead.");
            if (typeNumberOnly)
                return;
            // A sized text, a byte-swapped or a unit-carrying descriptor is not equivalent to its class's singleton.
            var canonical = dtype.Meta.singleton;
            if (canonical is null || !DTypeCasting.EquivTypes(canonical, dtype))
                throw new TypeError("The `dtype` and `signature` arguments to ufuncs only select the general DType and not details " +
                                    "such as the byte order or time unit. You can avoid this error by using the scalar types " +
                                    "`np.float64` or the dtype string notation.");
        }

        /// <summary>
        ///     <c>reducelike_promote_and_resolve</c> for an object array reduced or accumulated in the loop of
        ///     <paramref name="dtype"/>'s class (an input cast that a reduction makes unsafely): the text, void and datetime
        ///     classes have no add / multiply loop, timedelta has an add loop but no cast from object.
        /// </summary>
        /// <param name="dtype">The loop's class.</param>
        /// <param name="ufunc"><c>add</c> or <c>multiply</c>.</param>
        /// <exception cref="TypeError"><c>No loop matching the specified signature and casting was found for ufunc add</c>;
        ///     <c>cannot cast dtype object to &lt;class 'numpy.dtypes.TimeDelta64DType'&gt;.</c></exception>
        private static void CheckObjectReduceLoop(DType dtype, string ufunc)
        {
            switch (dtype.kind)
            {
                case 'U':
                case 'S':
                case 'V':
                case 'M':
                    throw new TypeError($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
                case 'm':
                    if (ufunc == "add")
                        throw new TypeError("cannot cast dtype object to <class 'numpy.dtypes.TimeDelta64DType'>.");
                    throw new TypeError($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
            }
        }

        /// <summary>
        ///     The loop of an object reduction into an <paramref name="out"/> of another class: every class reaches the object
        ///     loop's output by a reduction's unsafe cast, except NEP 55's StringDType, which has no loop with an object
        ///     operand.
        /// </summary>
        /// <param name="out">The out array.</param>
        /// <param name="ufunc">The ufunc's name.</param>
        /// <exception cref="TypeError">NumPy's UFuncTypeError text for a StringDType out.</exception>
        private static void CheckObjectOutLoop(NDArray @out, string ufunc)
        {
            if (@out.dtype.Meta is VStringDTypeMeta)
                throw new TypeError($"ufunc '{ufunc}' did not contain a loop with signature matching types " +
                                    "(<class 'numpy.dtypes.StringDType'>, <class 'numpy.dtypes.ObjectDType'>) -> None");
        }

        /// <summary>
        ///     The error a ufunc CALL (not a reduction) over an object input raises for a <c>dtype=</c> of another class:
        ///     a text or void class has no loop, any other class's loop needs the object input cast to it under
        ///     <c>same_kind</c>, which NumPy refuses (np.positive has no datetime loop either).
        /// </summary>
        /// <param name="dtype">The requested class (its descriptor already checked by <see cref="CheckUfuncDtype"/>).</param>
        /// <param name="ufunc">The ufunc's name (<c>clip</c>, <c>maximum</c>, <c>minimum</c>, <c>positive</c>).</param>
        /// <param name="nin">The ufunc's input count (a unary ufunc names no input index).</param>
        /// <returns>The exception to throw.</returns>
        private static Exception ObjectUfuncLoopError(DType dtype, string ufunc, int nin)
        {
            if (dtype.kind is 'U' or 'S' or 'V' || (dtype.kind == 'M' && ufunc == "positive"))
                return new TypeError($"No loop matching the specified signature and casting was found for ufunc {ufunc}");
            return new TypeError($"Cannot cast ufunc '{ufunc}' {(nin == 1 ? "input" : "input 0")} from dtype('O') to " +
                                 $"{dtype.ToString(true)} with casting rule 'same_kind'");
        }

        /// <summary>
        ///     The error NumPy's cast of a Python list to <paramref name="dtype"/> raises (the dtype's setitem, the first list
        ///     it meets) — the texts <see cref="astype"/> gives: float (and NumSharp's decimal) <c>setting an array element
        ///     with a sequence.</c>, integer <c>int() argument must be ... not 'list'</c>, complex <c>must be real number, not
        ///     list</c>, text <c>setting an array element with a sequence</c>; a bool cast is each list's truth (no error).
        /// </summary>
        /// <param name="dtype">The target dtype.</param>
        /// <returns>The exception, or null for bool.</returns>
        private static Exception ListCastError(DType dtype) => dtype.kind switch
        {
            'b' => null,
            'f' => new ValueError("setting an array element with a sequence."),
            'c' => new TypeError("must be real number, not list"),
            'U' or 'S' => new ValueError("setting an array element with a sequence"),
            _ => new TypeError("int() argument must be a string, a bytes-like object or a real number, not 'list'"),
        };

        /// <summary>
        ///     The refusal of a <c>dtype=</c> loop's result: NumPy returns the dtype's values (a bool, a number, an array of
        ///     them), which a list array cannot hold. The message names the call that gives NumPy's value.
        /// </summary>
        /// <param name="member">The member (<c>sum</c>, <c>prod</c>, <c>cumsum</c>, <c>cumprod</c>, <c>trace</c>).</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <returns>The exception to throw.</returns>
        private static NotSupportedException LoopResult(string member, DType dtype)
            => new($"NumPy's {member}(dtype={dtype}) of a list array runs the {dtype} loop over the lists converted to it and " +
                   $"returns {dtype} values, which an NDListArray (a list array) cannot hold; np.{member}(lists.astype(dtype), ...) " +
                   "in that dtype gives NumPy's value");

        /// <summary>
        ///     <c>PyUFunc_ReduceWrapper</c>'s pack of <c>initial</c> into a <c>dtype=</c> loop, before the array is read: a
        ///     list fails with the dtype's conversion error (none for bool), a C# number or bool converts; the conversion of any
        ///     other object (a string, a complex into a real loop, an array) is not modelled.
        /// </summary>
        /// <param name="initial">The caller's initial, or null.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <exception cref="ValueError">A list into a float loop.</exception>
        /// <exception cref="TypeError">A list into an integer or complex loop.</exception>
        /// <exception cref="NotSupportedException">An initial whose conversion NumSharp does not model.</exception>
        private void CheckInitialPacks(object initial, DType dtype)
        {
            if (initial is null || dtype.kind == 'b')
                return;
            if (initial is NDListArray { IsStandIn: true })
                throw ListCastError(dtype);
            if (TryPythonList(initial, Kind, out var heap, out _))
            {
                // A C# sequence standing for one Python list: converted like the array's lists.
                heap.Release();
                throw ListCastError(dtype);
            }
            bool number = initial is bool or sbyte or byte or short or ushort or int or uint or long or ulong or float or double
                                  or decimal or Half || (initial is System.Numerics.Complex && dtype.kind == 'c');
            if (!number)
                throw new NotSupportedException(
                    $"NumPy packs initial into the {dtype} loop with Python's conversion of a {NDPyObject.TypeName(initial)}, which " +
                    "this list-array reduction does not model");
        }

        /// <summary>
        ///     A reduction of lists in a <c>dtype=</c> loop (sum, prod, trace): NumPy's checks in its order, then the first
        ///     list's conversion error; every result NumPy returns (the bool loop's truths, an empty array's identity, the
        ///     initial) is the dtype's values, refused with <see cref="LoopResult"/>.
        /// </summary>
        /// <param name="fold">Sum or Prod.</param>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="out">The destination, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="initial">The initial, or null.</param>
        /// <param name="where">The caller's mask (null or Python's True: none).</param>
        /// <param name="typeNumberOnly">True for a C method (trace), which passes the dtype's type number.</param>
        /// <param name="member">The member's name for the refusal.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="TypeError">A refused or loop-less dtype; an integer or complex conversion.</exception>
        /// <exception cref="ValueError">NumPy's out= checks; a float conversion.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">The loop's result.</exception>
        private NDListArray ReduceInLoop(ListFold fold, int? axis, DType dtype, NDListArray @out, bool keepdims, object initial,
                                         NDArray where, bool typeNumberOnly, string member)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            string ufunc = fold == ListFold.Sum ? "add" : "multiply";
            // PyUFunc_GenericReduction: the dtype's class, out=, where=, the axis; PyUFunc_Reduce: the loop, then out's shape.
            CheckUfuncDtype(dtype, typeNumberOnly);
            NDArray mask = IsNoMask(where) ? null : where;
            ReductionArguments.ValidateParsedArguments(@out?.Ids, mask);
            bool[] flags = ReductionArguments.AxisFlags(ndim, axis);
            CheckObjectReduceLoop(dtype, ufunc);
            ReductionArguments.ValidateOperands(Ids, flags, keepdims, @out?.Ids, mask, ufunc);
            // A numeric loop has an identity, so a mask needs no initial; initial is packed before the elements are read.
            CheckInitialPacks(initial, dtype);
            if (size > 0 && ListCastError(dtype) is { } conversion)
                throw conversion;
            throw LoopResult(member, dtype);
        }

        /// <summary>
        ///     The object loop's division of the kept-dims sums by the count (NumPy's <c>true_divide(arrmean, rcount,
        ///     out=arrmean)</c>): a list divided by an int is CPython's TypeError, the int 0 of an empty slice divided by 0
        ///     its ZeroDivisionError (DivideByZeroException); a result with no element divides nothing (an object array NumSharp cannot return).
        /// </summary>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="name">The member (for the NotSupportedException).</param>
        /// <param name="keptDims">True for _var's kept-dims mean (axis=None divides one element).</param>
        /// <returns>The exception to throw.</returns>
        private Exception DivideListsError(int? axis, string name, bool keptDims = false)
        {
            long outputs, count;
            if (axis is null)
            {
                outputs = 1;
                count = size;
            }
            else
            {
                int ax = ndim == 0 ? 0 : DefaultEngine.check_and_adjust_axis(ndim, axis.Value);
                count = ndim == 0 ? 1 : Ids.Shape.dimensions[ax];
                outputs = count == 0 ? SizeWithout(ax) : size / count;
            }
            if (outputs == 0)
                return new NotSupportedException($"NumPy's {name} of this list array is an empty object array, which NumSharp has no type for");
            if (count == 0)
                return new DivideByZeroException("division by zero");
            _ = keptDims;
            return new TypeError("unsupported operand type(s) for /: 'list' and 'int'");
        }

        /// <summary>The number of elements of the array without axis <paramref name="axis"/>.</summary>
        /// <param name="axis">The axis.</param>
        /// <returns>The product of the other extents.</returns>
        private long SizeWithout(int axis)
        {
            long n = 1;
            var dims = Ids.Shape.dimensions;
            for (int d = 0; d < dims.Length; d++)
                if (d != axis)
                    n *= dims[d];
            return n;
        }

        // =============================================================================================================
        // Starts, bounds, heaps
        // =============================================================================================================

        /// <summary>
        ///     A reduction's <c>initial</c>: absent, a Python list (in a one-list heap of its own), or another Python value
        ///     (its type name for CPython's TypeError, and a repeat count for <c>n * list</c>).
        /// </summary>
        private sealed class Start : IDisposable
        {
            /// <summary>Whether an <c>initial</c> was given.</summary>
            internal bool Present;

            /// <summary>Whether it is a Python list.</summary>
            internal bool IsList;

            /// <summary>The list's heap (a list start).</summary>
            internal NDListHeap Heap;

            /// <summary>The list's id.</summary>
            internal long Id;

            /// <summary>The Python type name of a non-list start.</summary>
            internal string TypeName;

            /// <summary>The repeat count of an int / bool start (Python's <c>n * list</c>), or null.</summary>
            internal long? Repeat;

            /// <summary>Releases the list's heap reference.</summary>
            public void Dispose() => Heap?.Dispose();

            /// <summary>Reads a C# value as NumPy's <c>initial</c>.</summary>
            /// <param name="initial">The value (null: none).</param>
            /// <param name="kind">The list array's piece class (an empty list start takes it).</param>
            /// <returns>The start.</returns>
            /// <exception cref="NotSupportedException">A sequence of str and bytes pieces.</exception>
            internal static Start Of(object initial, char kind)
            {
                var s = new Start();
                if (initial is null)
                    return s;
                // NumPy packs `initial` into the object loop as it is: an ndarray (a list array that is not a list's
                // stand-in) would become the start of the fold and meet the lists' comparisons elementwise.
                if (initial is NDListArray { IsStandIn: false })
                    throw new NotSupportedException(
                        "an array as 'initial' is the start of NumPy's object fold as it is (an ndarray object), which an " +
                        "NDListArray reduction cannot hold; pass a list (lists[\"1\"], one element of a list array) or a C# sequence of pieces");
                s.Present = true;
                if (TryPythonList(initial, kind, out var heap, out long id))
                {
                    s.IsList = true;
                    s.Heap = heap;
                    s.Id = id;
                    return s;
                }
                s.TypeName = PythonTypeName(initial);
                s.Repeat = initial switch
                {
                    bool b => b ? 1 : 0,
                    sbyte or byte or short or ushort or int or uint or long => Convert.ToInt64(initial),
                    ulong u when u <= long.MaxValue => (long)u,
                    _ => null,
                };
                return s;
            }
        }

        /// <summary>
        ///     A clip bound: absent, a list array (its lists broadcast, borrowed from the caller), or the shape and Python
        ///     type of a value that is not a list.
        /// </summary>
        [NDBorrowed] // the bound's list array is the caller's argument; the bound never owns one
        private sealed class Bound
        {
            /// <summary>Whether a bound was given.</summary>
            internal bool Present;

            /// <summary>Whether it is a list array (or a Python list made one).</summary>
            internal bool IsLists => Lists is not null;

            /// <summary>The bound's lists (the caller's).</summary>
            internal NDListArray Lists;

            /// <summary>The shape the bound broadcasts with.</summary>
            internal Shape Shape;

            /// <summary>The Python type of a non-list bound's elements.</summary>
            internal string TypeName;
        }

        /// <summary>Reads a clip bound (null: none).</summary>
        /// <param name="value">The bound.</param>
        /// <returns>The bound.</returns>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        private Bound BoundOperand(object value)
        {
            var b = new Bound();
            if (value is null)
                return b;
            b.Present = true;
            if (value is NDListArray { IsStandIn: true } list)
            {
                // A Python list bound is np.asarray'd by the ufunc: the array of its items.
                list.ThrowIfDisposed();
                using var items = list.ItemsArray(list.FlatId(0));
                b.Shape = items.Shape;
                b.TypeName = PyElementTypeName(items.dtype);
                return b;
            }
            if (value is NDListArray lists)
            {
                lists.ThrowIfDisposed();
                b.Lists = lists;
                b.Shape = lists.Ids.Shape;
                return b;
            }
            if (TryPythonList(value, Kind, out var heap, out long id))
            {
                // A Python list as the bound: NumPy's np.asarray of a list of str is a str ARRAY, compared item by item.
                heap.Release();
            }
            if (value is NDArray array)
            {
                b.Shape = array.Shape;
                b.TypeName = PyElementTypeName(array.dtype);
                return b;
            }
            using var converted = PythonOperand(value);
            b.Shape = Shape.Adopt((long[])converted.Shape.dimensions.Clone());
            b.TypeName = PyElementTypeName(converted.dtype);
            return b;
        }

        /// <summary>
        ///     Reads a C# value standing for ONE Python list: a list stand-in (<see cref="IsStandIn"/>), or a C# sequence (not a
        ///     string, not a <c>byte[]</c>) of <see cref="string"/> pieces (str) or <c>byte[]</c> pieces (bytes).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="kind">The piece class an empty list takes.</param>
        /// <param name="heap">Receives a heap holding the list (a reference the caller releases).</param>
        /// <param name="id">Receives the list's id.</param>
        /// <returns>False when the value is not a list.</returns>
        /// <exception cref="NotSupportedException">A sequence of str and bytes pieces, or of other items.</exception>
        internal static bool TryPythonList(object value, char kind, out NDListHeap heap, out long id)
        {
            heap = null;
            id = -1;
            switch (value)
            {
                case NDListArray { IsStandIn: true } lists:
                    // The stand-in of a Python list (r["1"], r.max()): the list. Any other list array is an ndarray object,
                    // which NumPy would store as it is.
                    lists.ThrowIfDisposed();
                    heap = lists.Heap.Share();
                    id = lists.FlatId(0);
                    return true;
                case NDListArray:
                case string:
                case byte[]:
                case NDArray:
                    return false;
                case IEnumerable sequence:
                {
                    var pieces = new List<object>();
                    bool anyStr = false, anyBytes = false;
                    foreach (var p in sequence)
                    {
                        pieces.Add(p);
                        anyStr |= p is string;
                        anyBytes |= p is byte[];
                    }
                    if (anyStr && anyBytes)
                        throw new NotSupportedException("a list of str and bytes items has no NDListArray form (one piece class per list array)");
                    char k = anyBytes ? 'S' : anyStr ? (kind == 'S' ? 'U' : kind) : kind;
                    heap = NDListHeap.Empty(k);
                    try
                    {
                        id = heap.AppendList(pieces);
                    }
                    catch
                    {
                        heap.Release();
                        heap = null;
                        throw;
                    }
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <summary>The Python type name of a C# value as NumSharp's boundary map reads it.</summary>
        /// <param name="value">The value.</param>
        /// <returns><c>int</c>, <c>float</c>, <c>complex</c>, <c>bool</c>, <c>str</c>, <c>bytes</c>, or NumPy's name of an
        ///     array's elements.</returns>
        private static string PythonTypeName(object value) => value switch
        {
            bool => "bool",
            sbyte or byte or short or ushort or int or uint or long or ulong or System.Numerics.BigInteger => "int",
            float or double or decimal => "float",
            System.Numerics.Complex => "complex",
            string => "str",
            char => "str",
            byte[] => "bytes",
            NDArray a => PyElementTypeName(a.dtype),
            _ => NDPyObject.TypeName(value),
        };

        /// <summary>
        ///     Whether the fold's start is a NON-EMPTY list of the other piece class (bytes against str lists). Every result
        ///     NumSharp can hold is then the start's pieces or an empty list: a sum adds this array's lists to it, which
        ///     <see cref="CheckSameClass"/> refuses unless they are empty, and a maximum / minimum compares it with them,
        ///     which is CPython's TypeError unless one side is empty — so the result belongs in the start's class
        ///     (<c>e.max(initial=[b'x'])</c> over empty str lists is <c>[b'x']</c>).
        /// </summary>
        /// <param name="start">The fold's start.</param>
        /// <returns>True for a non-empty list of the other class.</returns>
        private bool IsForeignStart(Start start)
            => start.Present && start.IsList && (start.Heap.Kind == 'S') != (Kind == 'S') && start.Heap.Length(start.Id) > 0;

        /// <summary>A new empty heap for a result: the start's class when it is a non-empty list of the other class
        ///     (<see cref="IsForeignStart"/>), else this array's.</summary>
        /// <param name="start">The fold's start.</param>
        /// <returns>The heap (one reference, the caller's).</returns>
        private NDListHeap NewHeapLike(Start start) => NDListHeap.Empty(IsForeignStart(start) ? start.Heap.Kind : Kind);

        /// <summary>A concatenation's class check: a list of str and bytes items has no NDListArray form.</summary>
        /// <param name="start">The fold's start.</param>
        /// <param name="slice">The slice concatenated after it.</param>
        /// <exception cref="NotSupportedException">The start's pieces and the slice's are of different classes and both
        ///     contribute pieces.</exception>
        private void CheckSameClass(Start start, ReadOnlySpan<long> slice)
        {
            if (!start.Present || !start.IsList || (start.Heap.Kind == 'S') == (Kind == 'S'))
                return;
            if (start.Heap.Length(start.Id) == 0)
                return;
            foreach (long id in slice)
            {
                if (Heap.Length(id) != 0)
                    throw new NotSupportedException("the concatenation holds str and bytes items, which an NDListArray (one piece class) cannot hold");
            }
        }

        /// <summary>Copies list <paramref name="id"/> of <paramref name="source"/> into <paramref name="target"/> once
        ///     (memoized), returning its id there.</summary>
        /// <param name="source">The source heap.</param>
        /// <param name="id">The list.</param>
        /// <param name="target">The target heap.</param>
        /// <param name="memo">The copies already made.</param>
        /// <returns>The list's id in <paramref name="target"/>.</returns>
        /// <exception cref="NotSupportedException">A non-empty list of the other piece class.</exception>
        private static long CopyList(NDListHeap source, long id, NDListHeap target, Dictionary<(NDListHeap, long), long> memo)
        {
            if (ReferenceEquals(source, target))
                return id;
            if (memo.TryGetValue((source, id), out long copied))
                return copied;
            if ((source.Kind == 'S') != (target.Kind == 'S') && source.Length(id) != 0)
                throw new NotSupportedException("a list of the other piece class (str and bytes) cannot join an NDListArray of one piece class");
            target.BeginList();
            target.AppendListPieces(source, id);
            long result = target.EndList();
            memo[(source, id)] = result;
            return result;
        }

        /// <summary>
        ///     This array as an <c>out=</c> destination: the computed <paramref name="result"/>'s lists are stored into it
        ///     (NumPy's out= of an object ufunc: the pointers copied), and <paramref name="result"/> is disposed — the caller
        ///     hands it over.
        /// </summary>
        /// <param name="result">The computed result (always disposed here).</param>
        /// <exception cref="ValueError">This array is read-only (<c>output array is read-only</c>), or its shape differs
        ///     (<c>non-broadcastable output operand with shape (2,) doesn't match the broadcast shape (3,)</c>).</exception>
        /// <exception cref="NotSupportedException">A non-empty list of the other piece class.</exception>
        private void Receive(NDListArray result)
        {
            using (result)
            {
                if (!Ids.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
                if (!Ids.Shape.dimensions.AsSpan().SequenceEqual(result.Ids.Shape.dimensions))
                    throw new ValueError($"non-broadcastable output operand with shape {Ids.Shape.ToPythonTuple()} doesn't match " +
                                         $"the broadcast shape {result.Ids.Shape.ToPythonTuple()}");
                AssignFrom(Ids, result);
            }
        }
    }
}
