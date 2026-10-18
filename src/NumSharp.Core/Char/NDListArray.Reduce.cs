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
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">A Python list to start from (see <see cref="max"/>), or another Python value (CPython's
        ///     TypeError, <c>unsupported operand type(s) for +: 'int' and 'list'</c>); null for none.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">A mask without <paramref name="initial"/>; out= errors.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError">A non-list <paramref name="initial"/> added to a list.</exception>
        /// <exception cref="NotSupportedException">An empty slice without <paramref name="initial"/> (NumPy's int 0), a list
        ///     of str and bytes items (the classes of <paramref name="initial"/> and the array differ), or a non-list result.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray sum(int? axis = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => Reduce(ListFold.Sum, axis, @out, keepdims, initial, where);

        /// <summary>
        ///     NumPy's <c>sum</c> with a <c>dtype=</c>: the elements are converted to it first (NumPy's cast of the input to
        ///     the loop), then NDArray's sum — so <c>bool</c> is the OR of the lists' truths and every numeric dtype NumPy's
        ///     conversion error (see <see cref="astype"/>).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">The loop dtype.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <returns>The reduced array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is null.</exception>
        /// <exception cref="ValueError">A float / text dtype for a non-empty array.</exception>
        /// <exception cref="TypeError">An integer / complex dtype for a non-empty array.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray sum(int? axis, DType dtype, bool keepdims = false)
        {
            using var converted = astype(dtype);
            return np.sum(converted, axis, keepdims, dtype);
        }

        /// <summary>
        ///     NumPy's <c>prod(axis=None, dtype=None, out=None, keepdims=False, initial=&lt;no value&gt;, where=True)</c> of
        ///     lists: a slice of one list is that list, two or more are <c>list * list</c> (CPython's TypeError).
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="initial">The start of the fold: an int repeats the first list (Python's <c>n * list</c>), a list
        ///     multiplies (TypeError), null for none.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>The result list array.</returns>
        /// <exception cref="ValueError">A mask without <paramref name="initial"/>; out= errors.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError"><c>can't multiply sequence by non-int of type 'list'</c>.</exception>
        /// <exception cref="NotSupportedException">An empty slice without <paramref name="initial"/> (NumPy's int 1), or a
        ///     non-list result.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray prod(int? axis = null, NDListArray @out = null, bool keepdims = false, object initial = null, NDArray where = null)
            => Reduce(ListFold.Prod, axis, @out, keepdims, initial, where);

        // =============================================================================================================
        // cumsum / cumprod / trace
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>cumsum(axis=None, dtype=None, out=None)</c> of lists: the running concatenation along
        ///     <paramref name="axis"/> (the flattened C-order array for null); the result has the array's shape in its memory
        ///     order (a 1-D array for null).
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <returns>The running concatenations.</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError"><paramref name="out"/>'s shape differs.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray cumsum(int? axis = null, NDListArray @out = null) => Accumulate(ListFold.Sum, axis, @out);

        /// <summary>
        ///     NumPy's <c>cumprod(axis=None, dtype=None, out=None)</c> of lists: the first element of each line is itself, the
        ///     second multiplies (CPython's TypeError).
        /// </summary>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <returns>The result (for lines of at most one element).</returns>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="TypeError"><c>can't multiply sequence by non-int of type 'list'</c> for a line of two or more.</exception>
        /// <exception cref="ValueError"><paramref name="out"/>'s shape differs.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray cumprod(int? axis = null, NDListArray @out = null) => Accumulate(ListFold.Prod, axis, @out);

        /// <summary>
        ///     NumPy's <c>trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)</c> of lists: the concatenation of the
        ///     diagonal's lists (NumPy's add.reduce over the diagonal's last axis).
        /// </summary>
        /// <param name="offset">The diagonal's offset.</param>
        /// <param name="axis1">The first axis of the 2-D sub-arrays.</param>
        /// <param name="axis2">The second axis.</param>
        /// <param name="out">A list array to store into, or null.</param>
        /// <returns>The concatenations (a 0-d list array for a 2-D array).</returns>
        /// <exception cref="ValueError">Fewer than two axes (<c>diag requires an array of at least two dimensions</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">An empty diagonal (NumPy's int 0).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray trace(int offset = 0, int axis1 = 0, int axis2 = 1, NDListArray @out = null)
        {
            using var diagonalView = diagonal(offset, axis1, axis2);
            return diagonalView.sum(-1, @out);
        }

        // =============================================================================================================
        // clip
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>clip(min=None, max=None, out=None)</c> of lists: np.maximum with <paramref name="min"/> (a list
        ///     below it becomes it), then np.minimum with <paramref name="max"/>, element by element over the broadcast
        ///     operands, by Python's list order.
        /// </summary>
        /// <param name="min">The lower bounds (a list array, broadcast), a Python value, or null.</param>
        /// <param name="max">The upper bounds, or null.</param>
        /// <param name="out">A list array of the result's shape to store into, or null.</param>
        /// <returns>The clipped lists.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast (NumPy's text, every operand's shape).</exception>
        /// <exception cref="TypeError">A bound that is not a list (<c>'&gt;=' not supported between instances of 'list' and
        ///     'str'</c>), an unorderable item pair, or no bound at all (np.positive of a list: <c>bad operand type for unary
        ///     +: 'list'</c>).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray clip(object min = null, object max = null, NDListArray @out = null)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            var result = ClipResult(min, max);
            if (@out is null)
                return result;
            @out.Receive(result);
            return @out;
        }

        /// <summary>The clipped lists of <see cref="clip"/>, in a new list array (the caller delivers them).</summary>
        /// <param name="min">The lower bounds, or null.</param>
        /// <param name="max">The upper bounds, or null.</param>
        /// <returns>The clipped lists.</returns>
        /// <exception cref="TypeError">See <see cref="clip"/>.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast.</exception>
        private NDListArray ClipResult(object min, object max)
        {
            if (min is null && max is null)
            {
                // NumPy 2's _clip with no bound is np.positive: CPython's unary + of a list.
                if (size > 0)
                    throw new TypeError("bad operand type for unary +: 'list'");
                return copy();
            }
            // A 0-d ufunc result is unwrapped to the object: the clipped list.
            return ClipCore(BoundOperand(min), BoundOperand(max)).MarkStandIn();
        }

        // =============================================================================================================
        // mean / std / var
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)</c> of lists (NumPy's
        ///     <c>_mean</c>): the sum divided by the count. Whole-array: the concatenated list divided by NumPy's intp count
        ///     — an empty list is an empty float64 array, a list of str or bytes NumPy's divide TypeError, an empty array NaN;
        ///     along an axis (or with keepdims): each list divided in the object loop, CPython's TypeError.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">A loop dtype (the elements converted first), or null.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask (add's missing identity: ValueError).</param>
        /// <returns>The NumPy values a list array's mean has (a 0-d NaN, an empty float64 array).</returns>
        /// <exception cref="TypeError"><c>ufunc 'divide' not supported for the input types, and the inputs could not be safely
        ///     coerced to any supported types according to the casting rule ''safe''</c> (whole array), or
        ///     <c>unsupported operand type(s) for /: 'list' and 'int'</c> (along an axis).</exception>
        /// <exception cref="ValueError">A mask (add has no identity for it).</exception>
        /// <exception cref="DivideByZeroException">An empty slice along an axis (NumPy's ZeroDivisionError, <c>division by
        ///     zero</c>).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array (an empty result along an axis).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray mean(int? axis = null, DType dtype = null, bool keepdims = false, NDArray where = null)
        {
            ThrowIfDisposed();
            if (dtype is not null)
            {
                using var converted = astype(dtype);
                // NumPy's _mean over the converted elements in the requested loop (the engine's Mean is that port).
                RefuseMaskWithLoop(where);
                return converted.TensorEngine.Mean(converted, axis, dtype, keepdims);
            }
            CheckStatisticsArguments(axis, keepdims, where);
            if (!keepdims && (axis is null || ndim == 1))
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
        ///     NumPy's <c>var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)</c> of lists (NumPy's
        ///     <c>_var</c>): its first step divides the kept-dims sum in the object loop, CPython's TypeError for a list.
        /// </summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">A loop dtype (the elements converted first), or null.</param>
        /// <param name="ddof">NumPy's delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>Never returns for a list array of at least one list.</returns>
        /// <exception cref="TypeError"><c>unsupported operand type(s) for /: 'list' and 'int'</c>.</exception>
        /// <exception cref="DivideByZeroException">An empty slice (NumPy's ZeroDivisionError, <c>division by zero</c>).</exception>
        /// <exception cref="ValueError">A mask (add has no identity for it).</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array (an empty result along an axis); a mask
        ///     together with <paramref name="dtype"/>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray var(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, NDArray where = null)
        {
            ThrowIfDisposed();
            if (dtype is not null)
            {
                using var converted = astype(dtype);
                RefuseMaskWithLoop(where);
                return axis is { } ax
                    ? np.var(converted, ax, keepdims, ddof, dtype)
                    : np.var(converted, keepdims, ddof, dtype);
            }
            CheckStatisticsArguments(axis, keepdims, where);
            if (ndim == 0)
                return ScalarVariance(ddof, sqrt: false);
            throw DivideListsError(axis, "var", keptDims: true);
        }

        /// <summary>NumPy's <c>std(...)</c> of lists: <c>sqrt(var(...))</c>, whose first step raises (see <see cref="var"/>).</summary>
        /// <param name="axis">The axis, or null for every element.</param>
        /// <param name="dtype">A loop dtype, or null.</param>
        /// <param name="ddof">NumPy's delta degrees of freedom.</param>
        /// <param name="keepdims">Keep the reduced axes as length 1.</param>
        /// <param name="where">NumPy's <c>where=</c> mask.</param>
        /// <returns>Never returns for a list array of at least one list.</returns>
        /// <exception cref="TypeError"><c>unsupported operand type(s) for /: 'list' and 'int'</c>.</exception>
        /// <exception cref="DivideByZeroException">An empty slice (NumPy's ZeroDivisionError).</exception>
        /// <exception cref="ValueError">A mask.</exception>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="NotSupportedException">NumPy's result is an object array; a mask together with
        ///     <paramref name="dtype"/>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDArray std(int? axis = null, DType dtype = null, int ddof = 0, bool keepdims = false, NDArray where = null)
        {
            ThrowIfDisposed();
            if (dtype is not null)
            {
                using var converted = astype(dtype);
                RefuseMaskWithLoop(where);
                return axis is { } ax
                    ? np.std(converted, ax, keepdims, ddof, dtype)
                    : np.std(converted, keepdims, ddof, dtype);
            }
            CheckStatisticsArguments(axis, keepdims, where);
            if (ndim == 0)
                return ScalarVariance(ddof, sqrt: true);
            throw DivideListsError(axis, "std", keptDims: true);
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
        ///     the TypeError of the second product), in the array's memory order (1-D C order for <c>axis=None</c>).
        /// </summary>
        /// <param name="fold">Sum or Prod.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <param name="out">The list array to store into, or null.</param>
        /// <returns>The result (or <paramref name="out"/>).</returns>
        private unsafe NDListArray Accumulate(ListFold fold, int? axis, NDListArray @out)
        {
            ThrowIfDisposed();
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
                long n = source.ndim == 0 ? 1 : source.Shape.dimensions[ax];
                if (fold == ListFold.Prod && n >= 2 && source.size > 0)
                    throw new TypeError("can't multiply sequence by non-int of type 'list'");
                var layout = axis is null ? Shape.Vector(source.size) : source.Shape.KeepOrder();
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
                    var result = new NDListArray(ids, heap);
                    if (@out is null)
                        return result;
                    @out.Receive(result);
                    return @out;
                }
                catch
                {
                    ids.Dispose();
                    heap?.Release();
                    throw;
                }
            }
            finally
            {
                if (!ReferenceEquals(source, Ids))
                    source.Dispose();
            }
        }

        /// <summary>
        ///     The clip loop over the broadcast operands (each bound a list array or a Python value, or null for none):
        ///     npy_ObjectMax with the lower bound, then npy_ObjectMin with the upper.
        /// </summary>
        /// <param name="lo">The lower bound operand.</param>
        /// <param name="hi">The upper bound operand.</param>
        /// <returns>The clipped lists, in a heap of their own.</returns>
        private unsafe NDListArray ClipCore(Bound lo, Bound hi)
        {
            var shapes = new List<Shape> { Ids.Shape };
            if (lo.Present)
                shapes.Add(lo.Shape);
            if (hi.Present)
                shapes.Add(hi.Shape);
            var dims = BroadcastAll(shapes);
            long count = 1;
            foreach (var d in dims)
                count *= d;
            // An empty broadcast compares nothing: the empty result in the ufunc's output layout.
            if (count == 0)
                return new NDListArray(new NDArray(NPTypeCode.Int64, Shape.NpyIterOutputShape(dims, shapes.ToArray()), false),
                    NDListHeap.Empty(Kind));
            // A bound that is not a list fails at the first comparison (np.maximum's object loop runs first).
            if (lo.Present && !lo.IsLists)
                throw new TypeError($"'>=' not supported between instances of 'list' and '{lo.TypeName}'");
            if (hi.Present && !hi.IsLists)
                throw new TypeError($"'<=' not supported between instances of 'list' and '{hi.TypeName}'");
            var layout = Shape.NpyIterOutputShape(dims, shapes.ToArray());
            if ((!lo.Present || ReferenceEquals(lo.Lists.Heap, Heap)) && (!hi.Present || ReferenceEquals(hi.Lists.Heap, Heap)))
                return ClipOneHeap(lo, hi, layout);
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
                using var flatOut = ids.reshape(Shape.Adopt((long[])dims.Clone()));
                for (long i = 0; i < count; i++)
                {
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
        /// <param name="layout">The result's layout (NumPy's ufunc output layout over the operands).</param>
        /// <returns>The clipped lists.</returns>
        private unsafe NDListArray ClipOneHeap(Bound lo, Bound hi, Shape layout)
        {
            var heap = Heap.Share();
            var ids = new NDArray(NPTypeCode.Int64, layout, false);
            try
            {
                var lists = new ListPointers(Heap);
                var ops = new List<NDArray> { Ids };
                if (lo.Present)
                    ops.Add(lo.Lists.Ids);
                if (hi.Present)
                    ops.Add(hi.Lists.Ids);
                ops.Add(ids);
                var flags = new NDIterPerOpFlags[ops.Count];
                for (int i = 0; i < flags.Length - 1; i++)
                    flags[i] = NDIterPerOpFlags.READONLY;
                flags[^1] = NDIterPerOpFlags.WRITEONLY;
                bool hasLo = lo.Present, hasHi = hi.Present;
                int outIndex = ops.Count - 1;
                Walk(ops.ToArray(), flags, (dataptrs, strides, n, aux) =>
                {
                    byte* px = (byte*)dataptrs[0];
                    byte* pl = hasLo ? (byte*)dataptrs[1] : null;
                    byte* ph = hasHi ? (byte*)dataptrs[hasLo ? 2 : 1] : null;
                    byte* po = (byte*)dataptrs[outIndex];
                    long sx = strides[0], sl = hasLo ? strides[1] : 0, sh = hasHi ? strides[hasLo ? 2 : 1] : 0, so = strides[outIndex];
                    for (long k = 0; k < n; k++, px += sx, pl += sl, ph += sh, po += so)
                    {
                        long x = *(long*)px;
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
        ///     The checks NumPy's <c>_mean</c> / <c>_var</c> make before their division: the axis (and the reduce prologue
        ///     of their sum), then a mask, which the sum of an object array refuses without an <c>initial</c>.
        /// </summary>
        /// <param name="axis">The axis, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="where">The mask, or null.</param>
        /// <exception cref="AxisError">An axis outside the array.</exception>
        /// <exception cref="ValueError">A mask.</exception>
        private void CheckStatisticsArguments(int? axis, bool keepdims, NDArray where)
        {
            NDArray mask = IsNoMask(where) ? null : where;
            ReductionArguments.Validate(Ids, axis, keepdims, null, mask, "add");
            // The sum's where= rule (see Reduce): an empty array's sum takes add's identity, a non-empty one refuses the mask.
            if (mask is not null && size > 0)
                throw new ValueError("reduction operation 'add' does not have an identity, so to use a where mask one has to specify 'initial'");
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

        /// <summary>
        ///     Refuses a <c>where=</c> mask in a statistic with a <c>dtype=</c> loop: NumSharp's mean / var / std take no mask
        ///     (NumPy would reduce the converted elements under it).
        /// </summary>
        /// <param name="where">The caller's mask (null or Python's True: none).</param>
        /// <exception cref="NotSupportedException">A real mask.</exception>
        private static void RefuseMaskWithLoop(NDArray where)
        {
            if (!IsNoMask(where))
                throw new NotSupportedException("a where= mask together with dtype= in a list array's mean / var / std is not supported");
        }

        /// <summary>
        ///     <c>_var</c> of a 0-d array: its kept-dims sum is unwrapped to the list itself, so the mean takes the scalar
        ///     branch — a non-empty list divided by the count is NumPy's ufunc divide TypeError; an empty list is the empty
        ///     float64 mean, whose deviations sum to the int 0, so the variance is <c>0 / max(1 - ddof, 0)</c> (0.0, or NaN
        ///     when ddof reaches the count).
        /// </summary>
        /// <param name="ddof">NumPy's delta degrees of freedom.</param>
        /// <param name="sqrt">True for std (the square root of the variance).</param>
        /// <returns>The 0-d float64 result.</returns>
        /// <exception cref="TypeError">A non-empty list.</exception>
        private NDArray ScalarVariance(int ddof, bool sqrt)
        {
            if (Heap.Length(FlatId(0)) > 0)
                throw new TypeError("ufunc 'divide' not supported for the input types, and the inputs could not be safely coerced " +
                                    "to any supported types according to the casting rule ''safe''");
            long count = Math.Max(1L - ddof, 0L);
            double variance = count == 0 ? double.NaN : 0.0 / count;
            return NDArray.Scalar(sqrt ? Math.Sqrt(variance) : variance);
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
