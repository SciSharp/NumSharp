using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

// =============================================================================
// NDListArray.Engine.cs — what NumPy's OBJECT loops do with Python lists, for the members that read the lists' VALUES:
// Python's list comparison, the type names Python prints in a TypeError, the element walks of an object ufunc and the
// ranks the sorting members order by
// =============================================================================
//
// An object array's comparisons, sorts and reductions call the ELEMENTS' Python methods: `a < b` is np.less's OBJECT loop
// (PyObject_RichCompare per pair), a sort is npy_quicksort with OBJECT_compare (Py_LT, then Py_GT), np.maximum's object
// loop is npy_ObjectMax (`i1 >= i2 ? i1 : i2`). For lists those are CPython's list_richcompare:
//
//   * == / != with different lengths answer at once (False / True); otherwise the items are compared with == from the
//     front, and the first unequal pair decides: == / != by it, an ordering by the operator applied to that pair (so
//     ['a', 'b'] < ['a', 'c'] compares 'b' < 'c'); with no unequal pair the lengths decide;
//   * a str item and a bytes item are never equal (no error), and ORDERING them is CPython's TypeError
//     "'<' not supported between instances of 'str' and 'bytes'" — the item type names are Python's: 'str', 'bytes', and
//     'numpy.str_' for an element's own np.str_ (a [self] piece).
//
// One NDListHeap holds one piece class, so the members that compare an array with itself (sort, max, min, argmax) never
// meet that error; it comes only from a second operand (another list array of the other class, a Python value).
// =============================================================================

namespace NumSharp
{
    public sealed partial class NDListArray
    {
        /// <summary>Python's six rich-comparison operators (<c>Py_LT</c> … <c>Py_GE</c>).</summary>
        internal enum PyCompareOp
        {
            /// <summary><c>&lt;</c>.</summary>
            Lt,

            /// <summary><c>&lt;=</c>.</summary>
            Le,

            /// <summary><c>==</c>.</summary>
            Eq,

            /// <summary><c>!=</c>.</summary>
            Ne,

            /// <summary><c>&gt;</c>.</summary>
            Gt,

            /// <summary><c>&gt;=</c>.</summary>
            Ge,
        }

        /// <summary>The operator's Python spelling, as CPython's TypeError prints it.</summary>
        /// <param name="op">The operator.</param>
        /// <returns><c>"&lt;"</c>, <c>"&lt;="</c>, <c>"=="</c>, <c>"!="</c>, <c>"&gt;"</c> or <c>"&gt;="</c>.</returns>
        internal static string Symbol(PyCompareOp op) => op switch
        {
            PyCompareOp.Lt => "<",
            PyCompareOp.Le => "<=",
            PyCompareOp.Eq => "==",
            PyCompareOp.Ne => "!=",
            PyCompareOp.Gt => ">",
            _ => ">=",
        };

        /// <summary>
        ///     The operator Python tries when the LEFT operand declines (<c>x &lt; y</c> → <c>y.__gt__(x)</c>): the reflected
        ///     comparison, so <c>1 &lt; lists</c> is NumPy's <c>np.greater(lists, 1)</c>.
        /// </summary>
        /// <param name="op">The operator as written.</param>
        /// <returns>The operator with its operands swapped (== and != are their own reflection).</returns>
        internal static PyCompareOp Reflected(PyCompareOp op) => op switch
        {
            PyCompareOp.Lt => PyCompareOp.Gt,
            PyCompareOp.Le => PyCompareOp.Ge,
            PyCompareOp.Gt => PyCompareOp.Lt,
            PyCompareOp.Ge => PyCompareOp.Le,
            _ => op,
        };

        /// <summary>The Python type name of piece <paramref name="piece"/>: <c>'bytes'</c>, <c>'str'</c>, or
        ///     <c>'numpy.str_'</c> for an element's own scalar.</summary>
        /// <param name="heap">The heap holding the piece.</param>
        /// <param name="piece">The piece index.</param>
        /// <returns>The name.</returns>
        private static string PieceTypeName(NDListHeap heap, long piece)
            => heap.Kind == 'S' ? "bytes" : heap.IsNpStr(piece) ? "numpy.str_" : "str";

        /// <summary>Whether two pieces are the same Python value: two str (or two bytes) with the same units; a str and a
        ///     bytes are never equal.</summary>
        /// <param name="ha">The first piece's heap.</param>
        /// <param name="pa">The first piece.</param>
        /// <param name="hb">The second piece's heap.</param>
        /// <param name="pb">The second piece.</param>
        /// <returns>The equality.</returns>
        private static unsafe bool PiecesEqual(NDListHeap ha, long pa, NDListHeap hb, long pb)
        {
            if ((ha.Kind == 'S') != (hb.Kind == 'S'))
                return false;
            byte* a = ha.Piece(pa, out long na);
            byte* b = hb.Piece(pb, out long nb);
            if (na != nb)
                return false;
            long bytes = na * ha.UnitSize;
            return new ReadOnlySpan<byte>(a, checked((int)bytes)).SequenceEqual(new ReadOnlySpan<byte>(b, checked((int)bytes)));
        }

        /// <summary>
        ///     The order of two pieces of ONE class: str by code points (Python's str comparison), bytes by unsigned bytes,
        ///     then by length.
        /// </summary>
        /// <param name="ha">The first piece's heap.</param>
        /// <param name="pa">The first piece.</param>
        /// <param name="hb">The second piece's heap (the same class).</param>
        /// <param name="pb">The second piece.</param>
        /// <returns>Negative, zero or positive.</returns>
        private static unsafe int ComparePieces(NDListHeap ha, long pa, NDListHeap hb, long pb)
        {
            byte* a = ha.Piece(pa, out long na);
            byte* b = hb.Piece(pb, out long nb);
            if (ha.UnitSize == 1)
                return new ReadOnlySpan<byte>(a, checked((int)na)).SequenceCompareTo(new ReadOnlySpan<byte>(b, checked((int)nb)));
            return new ReadOnlySpan<uint>((uint*)a, checked((int)na)).SequenceCompareTo(new ReadOnlySpan<uint>((uint*)b, checked((int)nb)));
        }

        /// <summary>
        ///     CPython's <c>list_richcompare(a, b, op)</c> of two lists (see the file header): the length shortcut of
        ///     == / !=, the first unequal item pair, then the operator on that pair or on the lengths.
        /// </summary>
        /// <param name="op">The operator.</param>
        /// <param name="ha">The left list's heap.</param>
        /// <param name="ida">The left list's id.</param>
        /// <param name="hb">The right list's heap.</param>
        /// <param name="idb">The right list's id.</param>
        /// <returns>The comparison's truth.</returns>
        /// <exception cref="TypeError">An ordering reaches a str item against a bytes item (<c>'&lt;' not supported between
        ///     instances of 'str' and 'bytes'</c>).</exception>
        internal static bool RichCompare(PyCompareOp op, NDListHeap ha, long ida, NDListHeap hb, long idb)
        {
            var (sa, ea) = ha.Range(ida);
            var (sb, eb) = hb.Range(idb);
            long la = ea - sa, lb = eb - sb;
            if (la != lb && (op == PyCompareOp.Eq || op == PyCompareOp.Ne))
                return op == PyCompareOp.Ne;
            long i = 0;
            while (i < la && i < lb && PiecesEqual(ha, sa + i, hb, sb + i))
                i++;
            if (i >= la || i >= lb)
                return CompareLengths(op, la, lb);
            if (op == PyCompareOp.Eq)
                return false;
            if (op == PyCompareOp.Ne)
                return true;
            // The differing pair under the operator itself: two items of one class order by their units; a str and a bytes
            // item are CPython's unorderable pair.
            if ((ha.Kind == 'S') != (hb.Kind == 'S'))
                throw new TypeError($"'{Symbol(op)}' not supported between instances of '{PieceTypeName(ha, sa + i)}' and " +
                                    $"'{PieceTypeName(hb, sb + i)}'");
            int c = ComparePieces(ha, sa + i, hb, sb + i);
            return op switch
            {
                PyCompareOp.Lt => c < 0,
                PyCompareOp.Le => c <= 0,
                PyCompareOp.Gt => c > 0,
                _ => c >= 0,
            };
        }

        /// <summary><c>Py_RETURN_RICHCOMPARE</c> over two list lengths.</summary>
        /// <param name="op">The operator.</param>
        /// <param name="la">The left length.</param>
        /// <param name="lb">The right length.</param>
        /// <returns>The comparison's truth.</returns>
        private static bool CompareLengths(PyCompareOp op, long la, long lb) => op switch
        {
            PyCompareOp.Lt => la < lb,
            PyCompareOp.Le => la <= lb,
            PyCompareOp.Eq => la == lb,
            PyCompareOp.Ne => la != lb,
            PyCompareOp.Gt => la > lb,
            _ => la >= lb,
        };

        /// <summary>
        ///     The total order of two lists whose items share one class (the lists of one heap, or of two heaps of one
        ///     class): <c>OBJECT_compare</c>'s answer, -1 when <c>a &lt; b</c>, 1 when <c>a &gt; b</c>, 0 otherwise.
        /// </summary>
        /// <param name="ha">The first list's heap.</param>
        /// <param name="ida">The first list.</param>
        /// <param name="hb">The second list's heap.</param>
        /// <param name="idb">The second list.</param>
        /// <returns>Negative, zero or positive.</returns>
        /// <exception cref="TypeError">The heaps hold different piece classes and a str / bytes pair decides.</exception>
        internal static int CompareLists(NDListHeap ha, long ida, NDListHeap hb, long idb)
        {
            if ((ha.Kind == 'S') != (hb.Kind == 'S'))
                return RichCompare(PyCompareOp.Lt, ha, ida, hb, idb) ? -1 : RichCompare(PyCompareOp.Gt, ha, ida, hb, idb) ? 1 : 0;
            var (sa, ea) = ha.Range(ida);
            var (sb, eb) = hb.Range(idb);
            long la = ea - sa, lb = eb - sb;
            for (long i = 0; i < la && i < lb; i++)
            {
                int c = ComparePieces(ha, sa + i, hb, sb + i);
                if (c != 0)
                    return c < 0 ? -1 : 1;
            }
            return la < lb ? -1 : la > lb ? 1 : 0;
        }

        // =============================================================================================================
        // Python values beside a list
        // =============================================================================================================

        /// <summary>
        ///     The Python type of the elements an array of <paramref name="dtype"/> becomes in NumPy's object loop (the
        ///     operand cast to object: <c>np.array([1]).astype(object)[0]</c> is a Python <c>int</c>), as a TypeError names
        ///     it: <c>bool</c>, <c>int</c>, <c>float</c>, <c>complex</c>, <c>str</c> or <c>bytes</c>.
        /// </summary>
        /// <param name="dtype">The operand's dtype.</param>
        /// <returns>The type name.</returns>
        /// <remarks>NumSharp's <c>char</c> converts as uint16 (a Python <c>int</c>) and its <c>decimal</c> as float64 (a Python
        ///     <c>float</c>), the house rule for the two dtypes NumPy lacks.</remarks>
        internal static string PyElementTypeName(DType dtype)
        {
            // The two dtypes NumPy lacks first: their kind letters describe NumSharp's storage, not a NumPy class.
            if (dtype.typecode == NPTypeCode.Char)
                return "int";
            if (dtype.typecode == NPTypeCode.Decimal)
                return "float";
            return dtype.kind switch
            {
                'b' => "bool",
                'i' or 'u' => "int",
                'f' => "float",
                'c' => "complex",
                'U' or 'T' => "str",
                'S' => "bytes",
                _ => dtype.name,
            };
        }

        /// <summary>
        ///     The array NumPy's ufunc makes of a non-list operand (<c>np.asarray(other)</c>): a C# <c>byte[]</c> is Python
        ///     <c>bytes</c>, a <c>string</c> a Python <c>str</c>, a number a Python number, an <see cref="NDArray"/> itself,
        ///     anything else <c>np.asanyarray</c>. A <c>null</c> is Python's <c>None</c> and is handled by the callers.
        /// </summary>
        /// <param name="value">The operand (not null).</param>
        /// <returns>The array (a new one, except an <see cref="NDArray"/> operand: the caller disposes it when it differs).</returns>
        /// <exception cref="NotSupportedException">The value has no NumSharp array form.</exception>
        internal static NDArray PythonOperand(object value) => CharModule.TextOperand(value, nameof(value));

        // =============================================================================================================
        // Element walks
        // =============================================================================================================

        /// <summary>
        ///     One NDIter walk over <paramref name="ops"/> in NumPy's ufunc order (K order: the operands' memory order, C
        ///     winning ties), the chunks handed to <paramref name="chunk"/> — the order an object ufunc visits its elements,
        ///     which decides which element's error is raised first.
        /// </summary>
        /// <param name="ops">The operands (broadcast together; an output among them is allocated in the broadcast shape).</param>
        /// <param name="flags">Each operand's access (READONLY inputs, a WRITEONLY output).</param>
        /// <param name="chunk">Called per chunk: <c>dataptrs[k]</c>, the byte <c>strides[k]</c>, the element count.</param>
        /// <exception cref="Exception">Whatever <paramref name="chunk"/> throws.</exception>
        internal static unsafe void Walk(NDArray[] ops, NDIterPerOpFlags[] flags, NDInnerLoopFunc chunk)
        {
            // An empty operand empties the whole (already validated) broadcast: nothing to visit. Checked here rather than
            // left to ZEROSIZE_OK, which handed a reversed view of a (0, 3) array to the chunk with a count of 3.
            foreach (var op in ops)
            {
                if (op.size == 0)
                    return;
            }
            using var iter = NDIterRef.AdvancedNew(ops.Length, ops,
                NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.ZEROSIZE_OK,
                NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING, flags, opDescrs: new DType[ops.Length]);
            iter.ForEach(chunk);
        }

        /// <summary>
        ///     The broadcast shape of two operands with NumPy's ufunc error text (<c>operands could not be broadcast together
        ///     with shapes (4,) (3,) </c>, an <see cref="IncorrectShapeException"/>, the house type for that ValueError).
        /// </summary>
        /// <param name="a">The first operand's shape.</param>
        /// <param name="b">The second operand's shape.</param>
        /// <returns>The broadcast dimensions (a new array).</returns>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast.</exception>
        internal static long[] BroadcastDims(Shape a, Shape b)
        {
            var (ra, _) = DefaultEngine.Broadcast(a, b);
            return (long[])ra.dimensions.Clone();
        }

        // =============================================================================================================
        // Ranks: the order the sorting members use
        // =============================================================================================================

        /// <summary>
        ///     Each list's DENSE rank among <paramref name="ids"/> (equal lists one rank, the smallest 0): the integers whose
        ///     order is the lists' order, so NumPy's sort and selection algorithms can run on them unchanged.
        /// </summary>
        /// <param name="heap">The heap of the lists.</param>
        /// <param name="ids">The list ids.</param>
        /// <param name="ranks">Receives each id's rank (same length as <paramref name="ids"/>).</param>
        /// <param name="order">Scratch of the same length (receives the positions in rank order).</param>
        /// <returns>The number of distinct ranks.</returns>
        internal static long DenseRanks(NDListHeap heap, ReadOnlySpan<long> ids, Span<long> ranks, long[] order)
        {
            int n = ids.Length;
            for (int i = 0; i < n; i++)
                order[i] = i;
            var local = ids.ToArray();
            // Any comparison sort gives the ranks (equal lists get one rank whatever their order).
            Array.Sort(order, 0, n, Comparer<long>.Create((x, y) => CompareLists(heap, local[x], heap, local[y])));
            long rank = -1;
            for (int k = 0; k < n; k++)
            {
                if (k == 0 || CompareLists(heap, local[order[k - 1]], heap, local[order[k]]) != 0)
                    rank++;
                ranks[(int)order[k]] = rank;
            }
            return rank + 1;
        }
    }
}
