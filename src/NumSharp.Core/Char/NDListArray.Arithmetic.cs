using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Strings;

// =============================================================================
// NDListArray.Arithmetic.cs — the arithmetic of NumPy's object array of lists: + (concatenation), * (repetition), dot,
// round, conj / conjugate, real / imag
// =============================================================================
//
// An object ufunc calls the Python operator on every element pair, over the broadcast operands, in the ufunc's
// iteration order, into a result laid out as NumPy lays out a ufunc output (the operands' memory order, C winning ties).
// For lists (probed against NumPy 2.4.2):
//
//   * a + b of two list arrays is each pair's concatenation — a NEW list (the result's lists live in a heap of their
//     own). Against anything else it is CPython's TypeError for the first pair: a list on the left "can only
//     concatenate list (not "int") to list"; a value on the left its own operator's text ("unsupported operand type(s)
//     for +: 'int' and 'list'", "can only concatenate str (not "list") to str", "can't concat list to bytes"). An empty
//     broadcast computes nothing and gives an empty result.
//   * a * n repeats each list n times (n <= 0: an empty list), n broadcast from an integer or bool array or value; n * a
//     is the same (list.__rmul__). A count of another type is "can't multiply sequence by non-int of type 'float'" (the
//     count's type), and a str / bytes value on the LEFT names the list ("can't multiply sequence by non-int of type
//     'list'": str's own repetition). A count past a C ssize_t is CPython's OverflowError "cannot fit 'int' into an
//     index-sized integer" (checked before the list's length: an empty list raises too); a count whose list would not
//     fit in memory is its MemoryError (OutOfMemoryException).
//   * A chararray beside a list array is the chararray's operator, which refuses the object result: + is "Can only
//     create a chararray from string data.", * "Can only multiply by integers".
//   * dot is OBJECT_dot: per output, the products a[..., k] * b[..., k, ...] summed by + in k order (the first product
//     starts the sum). A 0-d operand is a plain multiply. An empty contraction leaves NumPy's int 0, which a list array
//     cannot hold.
//   * round / conj / conjugate run ufuncs whose object loops look for a method a list does not have: TypeErrors for a
//     non-empty array, an empty result for an empty one. real is the array itself; imag is an array of int zeros.
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        // =============================================================================================================
        // + (concatenation)
        // =============================================================================================================

        /// <summary>NumPy's <c>a + b</c> of two list arrays: each broadcast pair's concatenation, a new list.</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The concatenations, in NumPy's ufunc output layout.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast (<c>operands could not be broadcast together
        ///     with shapes (4,) (3,) </c>).</exception>
        /// <exception cref="NotSupportedException">The results would hold str and bytes items together (NumPy's object lists
        ///     can; an NDListArray holds one piece class).</exception>
        /// <exception cref="ObjectDisposedException">An operand was disposed.</exception>
        public static NDListArray operator +(NDListArray left, NDListArray right) => Concatenate(left, right);

        /// <summary>NumPy's <c>lists + x</c> with an ndarray (or a C# value its implicit conversion made one): CPython's
        ///     TypeError for the first pair, <c>can only concatenate list (not "int") to list</c>.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The array.</param>
        /// <returns>An empty list array (nothing was added).</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's concatenation.</exception>
        public static NDListArray operator +(NDListArray left, NDArray right) => AddValue(left, right, reflected: false);

        /// <summary>NumPy's <c>x + lists</c> with an ndarray on the left: the element's own + (<c>unsupported operand
        ///     type(s) for +: 'int' and 'list'</c>, <c>can only concatenate str (not "list") to str</c>).</summary>
        /// <param name="left">The array.</param>
        /// <param name="right">The list array.</param>
        /// <returns>An empty list array (nothing was added).</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's addition.</exception>
        public static NDListArray operator +(NDArray left, NDListArray right) => AddValue(right, left, reflected: true);

        /// <summary>NumPy's <c>lists + x</c> with a C# value (null is Python's None; a list array concatenates).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The value.</param>
        /// <returns>The concatenations (a list array value), or an empty result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's concatenation.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDListArray operator +(NDListArray left, object right) => AddObject(left, right, reflected: false);

        /// <summary>NumPy's <c>x + lists</c> with a C# value on the left (see the array form).</summary>
        /// <param name="left">The value.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The concatenations (a list array value), or an empty result.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">The first pair's addition.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDListArray operator +(object left, NDListArray right) => AddObject(right, left, reflected: true);

        /// <summary>NumPy's <c>lists + c</c> with a chararray: the chararray's reflected add, whose object result cannot be a
        ///     chararray.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError"><c>Can only create a chararray from string data.</c></exception>
        public static NDListArray operator +(NDListArray left, NDCharArray right) => throw ChararrayObjectResult();

        /// <summary>NumPy's <c>c + lists</c> (see the reflected form).</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError"><c>Can only create a chararray from string data.</c></exception>
        public static NDListArray operator +(NDCharArray left, NDListArray right) => throw ChararrayObjectResult();

        // =============================================================================================================
        // * (repetition)
        // =============================================================================================================

        /// <summary>NumPy's <c>a * b</c> of two list arrays: <c>list * list</c>, CPython's TypeError for the first pair.</summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>An empty list array (nothing was multiplied).</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError"><c>can't multiply sequence by non-int of type 'list'</c>.</exception>
        public static NDListArray operator *(NDListArray left, NDListArray right)
        {
            if (left is null)
                throw new ArgumentNullException(nameof(left));
            if (right is null)
                throw new ArgumentNullException(nameof(right));
            left.ThrowIfDisposed();
            right.ThrowIfDisposed();
            if (left.IsStandIn != right.IsStandIn)
            {
                // A Python list facing an array is its items (np.asarray): str * list on the left names the list.
                var (list, array) = left.IsStandIn ? (left, right) : (right, left);
                using var items = list.ItemsArray(list.FlatId(0));
                return Repeat(array, items, reflected: left.IsStandIn);
            }
            var dims = BroadcastDims(left.Ids.Shape, right.Ids.Shape);
            if (Count(dims) > 0)
                throw new TypeError("can't multiply sequence by non-int of type 'list'");
            return EmptyResult(left, dims, new[] { left.Ids.Shape, right.Ids.Shape });
        }

        /// <summary>NumPy's <c>lists * n</c>: each list repeated by the broadcast integer (or bool) count.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The counts (any integer or bool array; a C# number converts to a 0-d one).</param>
        /// <returns>The repetitions, in NumPy's ufunc output layout.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">Counts that are not integers (<c>can't multiply sequence by non-int of type
        ///     'float'</c>).</exception>
        /// <exception cref="OverflowException">A count past a C ssize_t (<c>cannot fit 'int' into an index-sized
        ///     integer</c>).</exception>
        /// <exception cref="OutOfMemoryException">A repetition that cannot be allocated (NumPy's MemoryError).</exception>
        public static NDListArray operator *(NDListArray left, NDArray right) => Repeat(left, right, reflected: false);

        /// <summary>NumPy's <c>n * lists</c> (list.__rmul__): see the reflected form; a str / bytes array on the left is its
        ///     own repetition by a list (<c>can't multiply sequence by non-int of type 'list'</c>).</summary>
        /// <param name="left">The counts.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The repetitions.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">Counts that are not integers.</exception>
        /// <exception cref="OverflowException">A count past a C ssize_t.</exception>
        /// <exception cref="OutOfMemoryException">A repetition that cannot be allocated.</exception>
        public static NDListArray operator *(NDArray left, NDListArray right) => Repeat(right, left, reflected: true);

        /// <summary>NumPy's <c>lists * x</c> with a C# value (a <see cref="BigInteger"/> is a Python int of any size; null is
        ///     None; a list array is <c>list * list</c>).</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The count.</param>
        /// <returns>The repetitions.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A count that is not an integer.</exception>
        /// <exception cref="OverflowException">A count past a C ssize_t.</exception>
        /// <exception cref="OutOfMemoryException">A repetition that cannot be allocated.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDListArray operator *(NDListArray left, object right) => MultiplyObject(left, right, reflected: false);

        /// <summary>NumPy's <c>x * lists</c> with a C# value on the left.</summary>
        /// <param name="left">The count.</param>
        /// <param name="right">The list array.</param>
        /// <returns>The repetitions.</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        /// <exception cref="TypeError">A count that is not an integer.</exception>
        /// <exception cref="OverflowException">A count past a C ssize_t.</exception>
        /// <exception cref="OutOfMemoryException">A repetition that cannot be allocated.</exception>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        public static NDListArray operator *(object left, NDListArray right) => MultiplyObject(right, left, reflected: true);

        /// <summary>NumPy's <c>lists * c</c> with a chararray: the chararray's reflected multiply refuses a non-integer count.</summary>
        /// <param name="left">The list array.</param>
        /// <param name="right">The chararray.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError"><c>Can only multiply by integers</c>.</exception>
        public static NDListArray operator *(NDListArray left, NDCharArray right) => throw new ValueError("Can only multiply by integers");

        /// <summary>NumPy's <c>c * lists</c>: the chararray's multiply refuses the object counts.</summary>
        /// <param name="left">The chararray.</param>
        /// <param name="right">The list array.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="ValueError"><c>Can only multiply by integers</c>.</exception>
        public static NDListArray operator *(NDCharArray left, NDListArray right) => throw new ValueError("Can only multiply by integers");

        // =============================================================================================================
        // dot
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>dot(b, out=None)</c> of an object array of lists (OBJECT_dot): a 0-d operand multiplies (each list
        ///     repeated); otherwise each output is the sum (concatenation, in k order) of <c>a[..., k] * b[..., k, ...]</c>
        ///     over this array's last axis and <paramref name="b"/>'s second-to-last (its only one for 1-D).
        /// </summary>
        /// <param name="b">The other operand: an integer / bool array or C# number (repeat counts), or a list array /
        ///     another value (CPython's TypeError for the first product).</param>
        /// <param name="out">A C-contiguous, writeable list array of the result's shape to store into, or null.</param>
        /// <returns>The result (a 0-d list array for two 1-D operands: NumPy's list), or <paramref name="out"/>.</returns>
        /// <exception cref="IncorrectShapeException">The shapes are not aligned (<c>shapes (2,3) and (2,) not aligned: 3 (dim 1)
        ///     != 2 (dim 0)</c>), or a multiply's operands do not broadcast.</exception>
        /// <exception cref="ValueError">An unusable <paramref name="out"/> (<c>output array is not acceptable (must have the
        ///     right datatype, number of dimensions, and be a C-Array)</c>, <c>output array has wrong dimensions</c>).</exception>
        /// <exception cref="TypeError">A product a list does not take (<c>can't multiply sequence by non-int of type
        ///     'list'</c>).</exception>
        /// <exception cref="OverflowException">A count past a C ssize_t.</exception>
        /// <exception cref="OutOfMemoryException">A product that cannot be allocated.</exception>
        /// <exception cref="NotSupportedException">An empty contraction (NumPy's int 0 per output).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray dot(object b, NDListArray @out = null)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            var result = DotCore(b, @out);
            if (@out is null)
                return result;
            @out.Receive(result);
            return @out;
        }

        /// <summary>
        ///     The computation of <see cref="dot"/>: the products and sums into a new list array (the caller delivers it),
        ///     after NumPy's alignment and <c>out=</c> checks.
        /// </summary>
        /// <param name="b">The other operand.</param>
        /// <param name="out">The destination (checked only), or null.</param>
        /// <returns>The result.</returns>
        /// <exception cref="IncorrectShapeException">See <see cref="dot"/>.</exception>
        /// <exception cref="ValueError">See <see cref="dot"/>.</exception>
        /// <exception cref="TypeError">See <see cref="dot"/>.</exception>
        /// <exception cref="OverflowException">See <see cref="dot"/>.</exception>
        /// <exception cref="OutOfMemoryException">See <see cref="dot"/>.</exception>
        /// <exception cref="NotSupportedException">See <see cref="dot"/>.</exception>
        private NDListArray DotCore(object b, NDListArray @out)
        {
            if (b is NDListArray { IsStandIn: true } list)
            {
                // np.dot converts a Python list operand with np.asarray: its items.
                list.ThrowIfDisposed();
                using var items = list.ItemsArray(list.FlatId(0));
                return DotCore(items, @out);
            }
            if (b is NDListArray lists)
            {
                lists.ThrowIfDisposed();
                if (ndim == 0 || lists.ndim == 0)
                    return this * lists;
                var (outDims, length) = DotShapes(Ids.Shape, lists.Ids.Shape);
                CheckDotOut(@out, outDims);
                long outputs = Count(outDims);
                if (outputs > 0 && length > 0)
                    throw new TypeError("can't multiply sequence by non-int of type 'list'");
                return DotWithoutProducts(outDims, outputs);
            }
            // A 0-d operand (a Python int of any size, None) is a plain multiply.
            if (b is null || b is BigInteger)
                return MultiplyObject(this, b, reflected: false);
            NDArray counts = b as NDArray;
            bool ownCounts = counts is null;
            counts ??= PythonOperand(b);
            try
            {
                if (ndim == 0 || counts.ndim == 0)
                    return Repeat(this, counts, reflected: false);
                var (outDims, length) = DotShapes(Ids.Shape, counts.Shape);
                CheckDotOut(@out, outDims);
                long outputs = Count(outDims);
                if (outputs == 0 || length == 0)
                    return DotWithoutProducts(outDims, outputs);
                return DotRepeat(counts, outDims, length);
            }
            finally
            {
                if (ownCounts)
                    counts.Dispose();
            }
        }

        // =============================================================================================================
        // round / conj / real / imag
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>round(decimals=0, out=None)</c> of lists: <c>rint</c> (decimals 0), a multiply by <c>10.0 **
        ///     decimals</c> (positive) or a divide by <c>10.0 ** -decimals</c> (negative) — each CPython's TypeError for a
        ///     list; an empty array gives an empty result.
        /// </summary>
        /// <param name="decimals">The decimals.</param>
        /// <param name="out">A list array of the same SIZE, or null.</param>
        /// <returns>An empty result (or <paramref name="out"/>) for an empty array.</returns>
        /// <exception cref="ValueError"><paramref name="out"/> of another size (<c>invalid output shape</c>).</exception>
        /// <exception cref="TypeError"><c>loop of ufunc does not support argument 0 of type list which has no callable rint
        ///     method</c>, <c>can't multiply sequence by non-int of type 'float'</c>, <c>unsupported operand type(s) for /:
        ///     'list' and 'float'</c>.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray round(int decimals = 0, NDListArray @out = null)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            if (@out is not null && @out.size != size)
                throw new ValueError("invalid output shape");
            if (size > 0)
                throw new TypeError(decimals switch
                {
                    0 => "loop of ufunc does not support argument 0 of type list which has no callable rint method",
                    > 0 => "can't multiply sequence by non-int of type 'float'",
                    _ => "unsupported operand type(s) for /: 'list' and 'float'",
                });
            return @out ?? copy();
        }

        /// <summary>NumPy's <c>conj(out=None)</c> of lists: the conjugate ufunc's object loop looks for a <c>conjugate</c>
        ///     method a list does not have (an empty array gives an empty result).</summary>
        /// <param name="out">A list array of the result's shape, or null.</param>
        /// <returns>An empty result (or <paramref name="out"/>) for an empty array.</returns>
        /// <exception cref="TypeError"><c>loop of ufunc does not support argument 0 of type list which has no callable
        ///     conjugate method</c>.</exception>
        /// <exception cref="ValueError"><paramref name="out"/>'s shape differs.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray conj(NDListArray @out = null) => conjugate(@out);

        /// <summary>NumPy's <c>conjugate(out=None)</c> of lists (see <see cref="conj"/>).</summary>
        /// <param name="out">A list array of the result's shape, or null.</param>
        /// <returns>An empty result (or <paramref name="out"/>) for an empty array.</returns>
        /// <exception cref="TypeError">The object loop's TypeError for a non-empty array.</exception>
        /// <exception cref="ValueError"><paramref name="out"/>'s shape differs.</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray conjugate(NDListArray @out = null)
        {
            ThrowIfDisposed();
            @out?.ThrowIfDisposed();
            if (size > 0)
                throw new TypeError("loop of ufunc does not support argument 0 of type list which has no callable conjugate method");
            var result = copy();
            if (@out is null)
                return result;
            @out.Receive(result);
            return @out;
        }

        /// <summary>NumPy's <c>real</c> of an object array: the array ITSELF (<c>a.real is a</c>), so the caller must not
        ///     dispose it separately.</summary>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray real
        {
            get
            {
                ThrowIfDisposed();
                return this;
            }
        }

        /// <summary>NumPy's <c>imag</c> of an object array: a new READ-ONLY array of the int 0 — which a list array cannot
        ///     hold, except an empty one.</summary>
        /// <returns>An empty read-only list array for an empty array.</returns>
        /// <exception cref="NotSupportedException">A non-empty array (NumPy's elements are ints).</exception>
        /// <exception cref="ObjectDisposedException">The array was disposed.</exception>
        public NDListArray imag
        {
            get
            {
                ThrowIfDisposed();
                if (size > 0)
                    throw new NotSupportedException(
                        "NumPy's imag of an object array is an array of the int 0, which an NDListArray (a list array) cannot hold");
                var result = copy();
                result.Ids.setflags(write: false);
                return result;
            }
        }

        // =============================================================================================================
        // The engine
        // =============================================================================================================

        /// <summary>The element count of broadcast dimensions.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>Their product.</returns>
        private static long Count(long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return n;
        }

        /// <summary>An empty result (nothing computed) in NumPy's ufunc output layout over the operands.</summary>
        /// <param name="lists">The list operand (its piece class).</param>
        /// <param name="dims">The broadcast dimensions (an empty broadcast).</param>
        /// <param name="shapes">The operands' shapes.</param>
        /// <returns>The empty list array.</returns>
        private static NDListArray EmptyResult(NDListArray lists, long[] dims, Shape[] shapes)
            => new NDListArray(new NDArray(NPTypeCode.Int64, Shape.NpyIterOutputShape(dims, shapes), false), NDListHeap.Empty(lists.Kind));

        /// <summary>
        ///     Each broadcast pair's concatenation into a new heap (the class of the operand that holds pieces), walked in
        ///     NumPy's ufunc order.
        /// </summary>
        /// <param name="left">The left list array.</param>
        /// <param name="right">The right list array.</param>
        /// <returns>The concatenations.</returns>
        private static unsafe NDListArray Concatenate(NDListArray left, NDListArray right)
        {
            if (left is null)
                throw new ArgumentNullException(nameof(left));
            if (right is null)
                throw new ArgumentNullException(nameof(right));
            left.ThrowIfDisposed();
            right.ThrowIfDisposed();
            if (left.IsStandIn != right.IsStandIn)
            {
                // A Python list facing an array: ndarray's add runs np.asarray(list) — its items — against the lists.
                var (list, array) = left.IsStandIn ? (left, right) : (right, left);
                using var items = list.ItemsArray(list.FlatId(0));
                return AddValue(array, items, reflected: left.IsStandIn);
            }
            var dims = BroadcastDims(left.Ids.Shape, right.Ids.Shape);
            var shapes = new[] { left.Ids.Shape, right.Ids.Shape };
            if (Count(dims) == 0)
                return EmptyResult(left, dims, shapes);
            NDListHeap ha = left.Heap, hb = right.Heap;
            char kind = ha.Kind;
            if (ha.Kind != hb.Kind)
            {
                // One heap holds one piece class: a result made of str pieces from one side and bytes pieces from the other
                // is a list NumSharp has no form for (every left list meets some right list once the broadcast is not
                // empty). The truth passes run only here, where the classes (or the str kinds) differ.
                bool aHas = HoldsPieces(left), bHas = HoldsPieces(right);
                if ((ha.Kind == 'S') != (hb.Kind == 'S') && aHas && bHas)
                    throw new NotSupportedException(
                        "the concatenations would hold str and bytes items together, which an NDListArray (one piece class) cannot hold");
                kind = aHas || !bHas ? ha.Kind : hb.Kind;
            }
            long count = Count(dims);
            NDListHeap heap = null;
            var ids = new NDArray(NPTypeCode.Int64, Shape.NpyIterOutputShape(dims, shapes), false);
            try
            {
                // Pass 1, in NumPy's ufunc order: each output's pair of lists and the result's exact size. The result heap is
                // fresh, so output k (the k-th visited) becomes list k.
                var la = new ListPointers(ha);
                var lb = new ListPointers(hb);
                var pairs = new long[checked(2 * count)];
                long units = 0, pieces = 0, next = 0;
                Walk(new[] { left.Ids, right.Ids, ids },
                    new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY },
                    (dataptrs, strides, n, aux) =>
                    {
                        byte* pa = (byte*)dataptrs[0];
                        byte* pb = (byte*)dataptrs[1];
                        byte* po = (byte*)dataptrs[2];
                        long sa = strides[0], sb = strides[1], so = strides[2];
                        // The chunk's running totals in locals (the closure's fields would be stored on every element).
                        var a = la;
                        var b = lb;
                        long u = units, p = pieces, nx = next;
                        for (long k = 0; k < n; k++, pa += sa, pb += sb, po += so)
                        {
                            long ia = *(long*)pa, ib = *(long*)pb;
                            pairs[2 * nx] = ia;
                            pairs[2 * nx + 1] = ib;
                            // Checked: a broadcast can repeat a long list until the sizes pass what an array addresses
                            // (NumPy's MemoryError), and the one exact reservation must never be under-sized.
                            u = AddSize(u, a.UnitCount(ia) + b.UnitCount(ib));
                            p = AddSize(p, a.Length(ia) + b.Length(ib));
                            *(long*)po = nx++;
                        }
                        units = u;
                        pieces = p;
                        next = nx;
                    });
                // Pass 2: the heap at its exact size, then each concatenation copied in order.
                heap = NDListHeap.Reserved(kind, units, pieces, count);
                for (long k = 0; k < count; k++)
                {
                    heap.BeginList();
                    heap.AppendReserved(la, pairs[2 * k]);
                    heap.AppendReserved(lb, pairs[2 * k + 1]);
                    heap.EndListReserved();
                }
            }
            catch
            {
                ids.Dispose();
                heap?.Release();
                throw;
            }
            // A 0-d ufunc result is unwrapped to the object: the concatenated list.
            return new NDListArray(ids, heap).MarkStandIn();
        }

        /// <summary>Whether any element of a list array is a non-empty list.</summary>
        /// <param name="lists">The list array.</param>
        /// <returns>True when some list has a piece.</returns>
        private static bool HoldsPieces(NDListArray lists)
        {
            if (lists.size == 0)
                return false;
            using var truth = lists.Truth('K');
            using var any = truth.any();
            return (bool)any;
        }

        /// <summary>A list array plus an ndarray (either side): CPython's TypeError for the first pair; an empty broadcast is
        ///     an empty result.</summary>
        /// <param name="lists">The list array.</param>
        /// <param name="array">The ndarray operand.</param>
        /// <param name="reflected">True when the array was the LEFT operand.</param>
        /// <returns>An empty list array.</returns>
        private static NDListArray AddValue(NDListArray lists, NDArray array, bool reflected)
        {
            if (lists is null)
                throw new ArgumentNullException(nameof(lists));
            if (array is null)
                throw new ArgumentNullException(nameof(array));
            if (array is NDCharArray)
                throw ChararrayObjectResult();
            lists.ThrowIfDisposed();
            if (lists.IsStandIn)
                throw ItemsArithmetic(lists, array, reflected, "+");
            // np.add(x, lists) for x + lists: the broadcast (and its error text) in the expression's operand order.
            var dims = reflected ? BroadcastDims(array.Shape, lists.Ids.Shape) : BroadcastDims(lists.Ids.Shape, array.Shape);
            var shapes = reflected ? new[] { array.Shape, lists.Ids.Shape } : new[] { lists.Ids.Shape, array.Shape };
            if (Count(dims) == 0)
                return EmptyResult(lists, dims, shapes);
            throw AddTypeError(PyElementTypeName(array.dtype), reflected);
        }

        /// <summary>A list array plus a C# value: a list array concatenates, null is None, anything else its array form.</summary>
        /// <param name="lists">The list array.</param>
        /// <param name="value">The value.</param>
        /// <param name="reflected">True when the value was the LEFT operand.</param>
        /// <returns>The result.</returns>
        private static NDListArray AddObject(NDListArray lists, object value, bool reflected)
        {
            switch (value)
            {
                case NDListArray other:
                    return reflected ? Concatenate(other, lists) : Concatenate(lists, other);
                case NDCharArray:
                    throw ChararrayObjectResult();
                case NDArray array:
                    return AddValue(lists, array, reflected);
                case null:
                    if (lists is null)
                        throw new ArgumentNullException(nameof(lists));
                    lists.ThrowIfDisposed();
                    if (lists.size == 0)
                        return EmptyResult(lists, (long[])lists.Ids.Shape.dimensions.Clone(), new[] { lists.Ids.Shape });
                    throw AddTypeError("NoneType", reflected);
            }
            using var converted = PythonOperand(value);
            return AddValue(lists, converted, reflected);
        }

        /// <summary>CPython's TypeError of <c>list + x</c> (or <c>x + list</c>, reflected) for an element of Python type
        ///     <paramref name="type"/>.</summary>
        /// <param name="type">The other element's Python type name.</param>
        /// <param name="reflected">True when the other element is on the LEFT.</param>
        /// <returns>The exception.</returns>
        private static TypeError AddTypeError(string type, bool reflected)
        {
            if (!reflected)
                return new TypeError($"can only concatenate list (not \"{type}\") to list");
            return new TypeError(type switch
            {
                "str" => "can only concatenate str (not \"list\") to str",
                "bytes" => "can't concat list to bytes",
                _ => $"unsupported operand type(s) for +: '{type}' and 'list'",
            });
        }

        /// <summary>
        ///     A Python list (a stand-in) with an ndarray in + or *: ndarray's operator runs <c>np.asarray(list)</c> — its
        ///     items — against the array, a text or numeric computation whose result is not a list array. NumPy's own error
        ///     for it is raised (the ufunc's, through NDArray's operator); a result it computes is refused.
        /// </summary>
        /// <param name="list">The stand-in.</param>
        /// <param name="array">The ndarray.</param>
        /// <param name="reflected">True when the ARRAY was the left operand.</param>
        /// <param name="op"><c>"+"</c> or <c>"*"</c>.</param>
        /// <returns>The refusal (NumPy's own error is thrown instead when NumPy raises).</returns>
        private static Exception ItemsArithmetic(NDListArray list, NDArray array, bool reflected, string op)
        {
            using var items = list.ItemsArray(list.FlatId(0));
            var (l, r) = reflected ? (array, items) : (items, array);
            using var result = op == "+" ? l + r : l * r;
            return new NotSupportedException(
                $"a Python list {op} an ndarray is np.asarray(list) {op} the array, a {result.dtype} array, which an NDListArray " +
                "operator does not return; compute it on NDListArray.ItemsArray's NDArray form instead");
        }

        /// <summary>The chararray operators' refusal of an object result.</summary>
        /// <returns>The exception.</returns>
        private static ValueError ChararrayObjectResult() => new ValueError("Can only create a chararray from string data.");

        /// <summary>
        ///     Each list repeated by the broadcast count: the counts' dtype is checked first (a non-integer count is
        ///     CPython's TypeError for the first pair), then each pair in NumPy's ufunc order — the count's index-sized
        ///     conversion (OverflowError), the size (MemoryError), the repetition.
        /// </summary>
        /// <param name="lists">The list array.</param>
        /// <param name="counts">The counts.</param>
        /// <param name="reflected">True when the counts were the LEFT operand.</param>
        /// <returns>The repetitions, in a heap of their own.</returns>
        private static unsafe NDListArray Repeat(NDListArray lists, NDArray counts, bool reflected)
        {
            if (lists is null)
                throw new ArgumentNullException(nameof(lists));
            if (counts is null)
                throw new ArgumentNullException(nameof(counts));
            if (counts is NDCharArray)
                throw new ValueError("Can only multiply by integers");
            lists.ThrowIfDisposed();
            // A Python list times an ndarray (not a 0-d one: list * np.int64(2) repeats the list, as the 0-d path below
            // does) is the ndarray's multiply of the list's items.
            if (lists.IsStandIn && counts.ndim > 0)
                throw ItemsArithmetic(lists, counts, reflected, "*");
            // np.multiply(n, lists) for n * lists: the broadcast in the expression's operand order.
            var dims = reflected ? BroadcastDims(counts.Shape, lists.Ids.Shape) : BroadcastDims(lists.Ids.Shape, counts.Shape);
            var shapes = reflected ? new[] { counts.Shape, lists.Ids.Shape } : new[] { lists.Ids.Shape, counts.Shape };
            if (Count(dims) == 0)
                return EmptyResult(lists, dims, shapes);
            string type = PyElementTypeName(counts.dtype);
            if (type != "int" && type != "bool")
            {
                // A str / bytes element on the LEFT runs its own repetition with the list as the count.
                bool textLeft = reflected && (type == "str" || type == "bytes");
                throw new TypeError($"can't multiply sequence by non-int of type '{(textLeft ? "list" : type)}'");
            }
            // The counts as int64 (uint64 values past int64 wrap here; the overflow mask marks them for CPython's
            // OverflowError in visiting order).
            NDArray wide = counts.typecode == NPTypeCode.Int64 ? counts : counts.astype(NPTypeCode.Int64);
            NDArray overflow = null;
            try
            {
                if (counts.typecode == NPTypeCode.UInt64)
                {
                    using var limit = NDArray.Scalar((ulong)long.MaxValue);
                    overflow = np.greater(counts, limit);
                }
                NDListHeap source = lists.Heap;
                long count = Count(dims);
                NDListHeap heap = null;
                var ids = new NDArray(NPTypeCode.Int64, Shape.NpyIterOutputShape(dims, shapes), false);
                try
                {
                    var ops = overflow is null
                        ? new[] { lists.Ids, wide, ids }
                        : new[] { lists.Ids, wide, ids, overflow };
                    var flags = overflow is null
                        ? new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY }
                        : new[] { NDIterPerOpFlags.READONLY, NDIterPerOpFlags.READONLY, NDIterPerOpFlags.WRITEONLY, NDIterPerOpFlags.READONLY };
                    // Pass 1, in NumPy's ufunc order: each pair's checks (the index-sized count, then the size) raise at the
                    // first pair that fails, as the object loop does; the passing pairs record (list, count) and the
                    // result's exact size. The result heap is fresh, so output k (the k-th visited) becomes list k.
                    var src = new ListPointers(source);
                    var plan = new long[checked(2 * count)];
                    long units = 0, pieces = 0, next = 0;
                    Walk(ops, flags, (dataptrs, strides, n, aux) =>
                    {
                        byte* pl = (byte*)dataptrs[0];
                        byte* pc = (byte*)dataptrs[1];
                        byte* po = (byte*)dataptrs[2];
                        byte* pv = flags.Length > 3 ? (byte*)dataptrs[3] : null;
                        long sl = strides[0], sc = strides[1], so = strides[2], sv = flags.Length > 3 ? strides[3] : 0;
                        // The chunk's running totals in locals (the closure's fields would be stored on every element); a
                        // check that throws abandons the whole call, so they need no write-back then.
                        var lists0 = src;
                        long u = units, p = pieces, nx = next;
                        for (long k = 0; k < n; k++, pl += sl, pc += sc, po += so, pv += sv)
                        {
                            if (pv is not null && *(bool*)pv)
                                throw new OverflowException("cannot fit 'int' into an index-sized integer");
                            long id = *(long*)pl, times = *(long*)pc;
                            CheckRepeatSize(lists0, id, times);
                            plan[2 * nx] = id;
                            plan[2 * nx + 1] = times;
                            if (times > 0)
                            {
                                u = AddSize(u, lists0.UnitCount(id) * times);
                                p = AddSize(p, lists0.Length(id) * times);
                            }
                            *(long*)po = nx++;
                        }
                        units = u;
                        pieces = p;
                        next = nx;
                    });
                    // Pass 2: the heap at its exact size, then each repetition copied in order.
                    heap = NDListHeap.Reserved(source.Kind, units, pieces, count);
                    for (long k = 0; k < count; k++)
                    {
                        heap.BeginList();
                        heap.AppendReserved(src, plan[2 * k], plan[2 * k + 1]);
                        heap.EndListReserved();
                    }
                }
                catch
                {
                    ids.Dispose();
                    heap?.Release();
                    throw;
                }
                // A 0-d ufunc result is unwrapped to the object: the repeated list.
                return new NDListArray(ids, heap).MarkStandIn();
            }
            finally
            {
                overflow?.Dispose();
                if (!ReferenceEquals(wide, counts))
                    wide.Dispose();
            }
        }

        /// <summary>The size check of CPython's <c>list * n</c> for one list (n &lt;= 0 or an empty list is an empty list):
        ///     a result too large to allocate is its MemoryError.</summary>
        /// <param name="source">The list's heap.</param>
        /// <param name="id">The list.</param>
        /// <param name="times">The count (already index-sized).</param>
        /// <exception cref="OutOfMemoryException">The repetition's pieces or code units do not fit in memory.</exception>
        private static void CheckRepeatSize(in ListPointers source, long id, long times)
        {
            long pieces = source.Length(id);
            if (times <= 0 || pieces == 0)
                return;
            long units = source.UnitCount(id);
            // list_repeat's `len > PY_SSIZE_T_MAX / n` and its failed allocation, both MemoryError: the pieces' offsets
            // (8 bytes each) and code units must stay addressable.
            if (times > (long.MaxValue / 8) / pieces || (units > 0 && times > (long.MaxValue / 8) / units))
                throw new OutOfMemoryException(string.Empty); // NumPy's MemoryError (PyErr_NoMemory) has no text
        }

        /// <summary>A result size plus one more list's: past what an array can address, the allocation's MemoryError (each
        ///     list fits, but NumPy's object loop would run out of memory before the last one).</summary>
        /// <param name="total">The size so far.</param>
        /// <param name="more">The next list's size (at most <c>long.MaxValue / 8</c>).</param>
        /// <returns>The sum.</returns>
        /// <exception cref="OutOfMemoryException">The sum passes <c>long.MaxValue / 8</c>.</exception>
        private static long AddSize(long total, long more)
        {
            long sum = total + more;
            if (sum > long.MaxValue / 8 || sum < total)
                throw new OutOfMemoryException(string.Empty); // NumPy's MemoryError (PyErr_NoMemory) has no text
            return sum;
        }

        /// <summary>A list array multiplied by a C# value: a list array is <c>list * list</c>, a <see cref="BigInteger"/> a
        ///     Python int of any size, null None, anything else its array form.</summary>
        /// <param name="lists">The list array.</param>
        /// <param name="value">The value.</param>
        /// <param name="reflected">True when the value was the LEFT operand.</param>
        /// <returns>The repetitions.</returns>
        private static NDListArray MultiplyObject(NDListArray lists, object value, bool reflected)
        {
            switch (value)
            {
                case NDListArray other:
                    return reflected ? other * lists : lists * other;
                case NDCharArray:
                    throw new ValueError("Can only multiply by integers");
                case NDArray array:
                    return Repeat(lists, array, reflected);
                case null:
                    if (lists is null)
                        throw new ArgumentNullException(nameof(lists));
                    lists.ThrowIfDisposed();
                    if (lists.size == 0)
                        return EmptyResult(lists, (long[])lists.Ids.Shape.dimensions.Clone(), new[] { lists.Ids.Shape });
                    throw new TypeError("can't multiply sequence by non-int of type 'NoneType'");
                case BigInteger big:
                    if (lists is null)
                        throw new ArgumentNullException(nameof(lists));
                    lists.ThrowIfDisposed();
                    if (lists.size > 0 && (big > long.MaxValue || big < long.MinValue))
                        throw new OverflowException("cannot fit 'int' into an index-sized integer");
                    using (var count = NDArray.Scalar((long)BigInteger.Max(BigInteger.Min(big, long.MaxValue), long.MinValue)))
                        return Repeat(lists, count, reflected);
            }
            using var converted = PythonOperand(value);
            return Repeat(lists, converted, reflected);
        }

        /// <summary>
        ///     NumPy's dot alignment (PyArray_MatrixProduct2): this array's last axis against the other's second-to-last (its
        ///     only one for 1-D); the result's dimensions are this array's leading ones then the other's remaining ones.
        /// </summary>
        /// <param name="a">This array's shape (ndim at least 1).</param>
        /// <param name="b">The other's shape (ndim at least 1).</param>
        /// <returns>The result's dimensions and the contraction's length.</returns>
        /// <exception cref="IncorrectShapeException">NumPy's alignment text.</exception>
        private static (long[] OutDims, long Length) DotShapes(Shape a, Shape b)
        {
            var ad = a.dimensions;
            var bd = b.dimensions;
            int matchDim = bd.Length == 1 ? 0 : bd.Length - 2;
            long length = ad[ad.Length - 1];
            if (bd[matchDim] != length)
                throw new IncorrectShapeException(
                    $"shapes {a.ToPythonTuple()} and {b.ToPythonTuple()} not aligned: {length} (dim {ad.Length - 1}) != {bd[matchDim]} (dim {matchDim})");
            var outDims = new List<long>();
            for (int d = 0; d < ad.Length - 1; d++)
                outDims.Add(ad[d]);
            for (int d = 0; d < bd.Length; d++)
                if (d != matchDim)
                    outDims.Add(bd[d]);
            return (outDims.ToArray(), length);
        }

        /// <summary>NumPy's checks of dot's <c>out=</c> (new_array_for_sum): the rank, dtype and C layout (writeable) first,
        ///     then the dimensions.</summary>
        /// <param name="out">The destination, or null.</param>
        /// <param name="outDims">The result's dimensions.</param>
        /// <exception cref="ValueError">NumPy's two texts.</exception>
        private static void CheckDotOut(NDListArray @out, long[] outDims)
        {
            if (@out is null)
                return;
            var s = @out.Ids.Shape;
            if (s.NDim != outDims.Length || !s.IsContiguous || !s.IsWriteable)
                throw new ValueError("output array is not acceptable (must have the right datatype, number of dimensions, and be a C-Array)");
            if (!s.dimensions.AsSpan().SequenceEqual(outDims))
                throw new ValueError("output array has wrong dimensions");
        }

        /// <summary>A dot that computes no product: an empty result for no output; otherwise NumPy's outputs are the int 0
        ///     (an empty contraction), which a list array cannot hold.</summary>
        /// <param name="outDims">The result's dimensions.</param>
        /// <param name="outputs">The number of outputs.</param>
        /// <returns>The empty result.</returns>
        /// <exception cref="NotSupportedException">At least one output.</exception>
        private NDListArray DotWithoutProducts(long[] outDims, long outputs)
        {
            if (outputs > 0)
                throw new NotSupportedException(
                    "an empty contraction leaves NumPy's int 0 in every output of an object dot, which an NDListArray cannot hold");
            return new NDListArray(new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])outDims.Clone()), false), NDListHeap.Empty(Kind));
        }

        /// <summary>
        ///     OBJECT_dot with integer counts: each output (C order) is the concatenation, in k order, of this array's list
        ///     at k repeated by the count at k — the first product's errors (a non-integer count, an overflow, a memory
        ///     error) raised in that order.
        /// </summary>
        /// <param name="counts">The counts (ndim at least 1, aligned).</param>
        /// <param name="outDims">The result's dimensions.</param>
        /// <param name="length">The contraction's length (at least 1).</param>
        /// <returns>The result in C order.</returns>
        private unsafe NDListArray DotRepeat(NDArray counts, long[] outDims, long length)
        {
            string type = PyElementTypeName(counts.dtype);
            if (type != "int" && type != "bool")
                throw new TypeError($"can't multiply sequence by non-int of type '{type}'");
            int bm = counts.ndim == 1 ? 0 : counts.ndim - 2;
            // This array as (rows, length) and the counts as (columns, length), both C-ordered copies of their elements.
            using var a2 = Ids.reshape(Shape.Adopt(new[] { size / length, length }));
            using var aRows = a2.copy('C');
            using var moved = np.moveaxis(counts, bm, -1);
            using var bWide = moved.astype(NPTypeCode.Int64);
            using var bRows = bWide.reshape(Shape.Adopt(new[] { counts.size / length, length }));
            using var bCopy = bRows.copy('C');
            NDArray overflow = null;
            try
            {
                if (counts.typecode == NPTypeCode.UInt64)
                {
                    using var limit = NDArray.Scalar((ulong)long.MaxValue);
                    using var over = np.greater(moved, limit);
                    using var overRows = over.reshape(Shape.Adopt(new[] { counts.size / length, length }));
                    overflow = overRows.copy('C');
                }
                long rows = aRows.Shape.dimensions[0], cols = bCopy.Shape.dimensions[0];
                long* pa = (long*)aRows.Address;
                long* pb = (long*)bCopy.Address;
                bool* pv = overflow is null ? null : (bool*)overflow.Address;
                NDListHeap heap = null;
                var ids = new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])outDims.Clone()), false);
                try
                {
                    long* po = (long*)ids.Address;
                    var src = new ListPointers(Heap);
                    // Pass 1: every product of every output in OBJECT_dot's order (outputs in C order, each over k), checked
                    // as it computes them (the index-sized count, then the size), and the result's exact size.
                    long units = 0, pieces = 0;
                    for (long i = 0; i < rows; i++)
                    {
                        for (long j = 0; j < cols; j++)
                        {
                            for (long k = 0; k < length; k++)
                            {
                                if (pv is not null && pv[j * length + k])
                                    throw new OverflowException("cannot fit 'int' into an index-sized integer");
                                long id = pa[i * length + k];
                                long times = pb[j * length + k];
                                CheckRepeatSize(src, id, times);
                                if (times > 0)
                                {
                                    units = AddSize(units, src.UnitCount(id) * times);
                                    pieces = AddSize(pieces, src.Length(id) * times);
                                }
                            }
                        }
                    }
                    // Pass 2: the heap at its exact size; each output is the sum (concatenation) of its products in k order,
                    // and the fresh heap numbers the outputs in C order.
                    heap = NDListHeap.Reserved(Kind, units, pieces, rows * cols);
                    for (long i = 0; i < rows; i++)
                    {
                        for (long j = 0; j < cols; j++)
                        {
                            heap.BeginList();
                            for (long k = 0; k < length; k++)
                                heap.AppendReserved(src, pa[i * length + k], pb[j * length + k]);
                            po[i * cols + j] = heap.EndListReserved();
                        }
                    }
                }
                catch
                {
                    ids.Dispose();
                    heap?.Release();
                    throw;
                }
                // OBJECT_dot of two 1-D operands writes one object, which np.dot returns as it is: the list.
                return new NDListArray(ids, heap).MarkStandIn();
            }
            finally
            {
                overflow?.Dispose();
            }
        }
    }
}
