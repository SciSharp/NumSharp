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
        ///     Each element's DENSE rank in Python's list order (equal lists one rank, the smallest 0): the integers whose order
        ///     is the lists' order, so NumPy's sort and selection algorithms can run on them unchanged.
        /// </summary>
        /// <param name="heap">The heap of the lists (one piece class: the lists are totally ordered).</param>
        /// <param name="ids">The elements' list ids.</param>
        /// <param name="ranks">Receives each element's rank.</param>
        /// <param name="count">The number of elements.</param>
        /// <returns>The number of distinct ranks.</returns>
        /// <remarks>
        ///     Equal lists are found first by CONTENT (a hash of every piece's units, then a full comparison on a hash match),
        ///     so only the distinct lists are sorted: a split result over a small vocabulary has a few hundred distinct lists
        ///     among millions of elements, and sorting all of them with a comparison was the whole cost of the sorting
        ///     members (the elements' ids are distinct even when their lists are equal). With every list distinct the hashing
        ///     adds one pass to the same sort.
        /// </remarks>
        internal static unsafe long DenseRanks(NDListHeap heap, long* ids, long* ranks, long count)
        {
            var lists = new ListPointers(heap);
            int n = checked((int)count);
            // The representatives of the distinct contents and each element's representative.
            var repIds = new List<long>();
            var repHashes = new List<ulong>();
            var repOf = new int[n];
            // Open addressing over the representatives (slot = representative + 1, 0 = empty), grown at half load.
            var slots = new int[64];
            int mask = slots.Length - 1;
            for (int i = 0; i < n; i++)
            {
                long id = ids[i];
                ulong h = lists.Hash(id);
                int s = (int)(h ^ (h >> 32)) & mask;
                while (true)
                {
                    int rep = slots[s] - 1;
                    if (rep < 0)
                    {
                        rep = repIds.Count;
                        repIds.Add(id);
                        repHashes.Add(h);
                        slots[s] = rep + 1;
                        if (repIds.Count * 2 > slots.Length)
                        {
                            // Rehash into a table twice as large (the representatives keep their indices).
                            slots = new int[slots.Length * 2];
                            mask = slots.Length - 1;
                            for (int r = 0; r < repIds.Count; r++)
                            {
                                ulong hr = repHashes[r];
                                int t = (int)(hr ^ (hr >> 32)) & mask;
                                while (slots[t] != 0)
                                    t = (t + 1) & mask;
                                slots[t] = r + 1;
                            }
                        }
                        repOf[i] = rep;
                        break;
                    }
                    if (repHashes[rep] == h && lists.Equal(repIds[rep], id))
                    {
                        repOf[i] = rep;
                        break;
                    }
                    s = (s + 1) & mask;
                }
            }
            // The distinct lists in order: any comparison sort (the contents are distinct, so no two compare equal).
            int k = repIds.Count;
            var order = new int[k];
            for (int r = 0; r < k; r++)
                order[r] = r;
            var reps = repIds.ToArray();
            Array.Sort(order, Comparer<int>.Create((x, y) => lists.Compare(reps[x], reps[y])));
            var rankOfRep = new long[k];
            for (int j = 0; j < k; j++)
                rankOfRep[order[j]] = j;
            for (int i = 0; i < n; i++)
                ranks[i] = rankOfRep[repOf[i]];
            return k;
        }
    }

    /// <summary>
    ///     One heap's lists read through raw pointers, for the scans that compare or hash many lists of that heap (the ranks,
    ///     argmax / argmin): valid while the heap does not grow, so only reading members may hold one.
    /// </summary>
    /// <remarks>
    ///     <see cref="NDListArray.CompareLists"/> re-reads the heap's arrays through their NDArray handles for every
    ///     comparison; a scan over millions of lists pays for those reads more than for the comparisons themselves.
    /// </remarks>
    internal readonly unsafe struct ListPointers
    {
        /// <summary>List i's first piece.</summary>
        private readonly long* _start;
        /// <summary>One past list i's last piece.</summary>
        private readonly long* _end;
        /// <summary>Piece k spans units <c>[_offsets[k], _offsets[k + 1])</c>.</summary>
        private readonly long* _offsets;
        /// <summary>The code units.</summary>
        private readonly byte* _units;
        /// <summary>The per-piece <c>np.str_</c> flags, or null when no piece has one.</summary>
        private readonly bool* _flags;
        /// <summary>The bytes of one code unit: 1 (bytes pieces) or 4 (str pieces, UCS4).</summary>
        private readonly int _unitSize;

        /// <summary>Captures <paramref name="heap"/>'s arrays.</summary>
        /// <param name="heap">The heap.</param>
        internal ListPointers(NDListHeap heap)
        {
            _start = heap.ListStart;
            _end = heap.ListEnd;
            _offsets = heap.Offsets;
            _units = heap.Units;
            _flags = heap.PieceFlags;
            _unitSize = heap.UnitSize;
        }

        /// <summary>List <paramref name="id"/>'s first piece.</summary>
        /// <param name="id">The list.</param>
        /// <returns>The piece index.</returns>
        internal long FirstPiece(long id) => _start[id];

        /// <summary>One past list <paramref name="id"/>'s last piece.</summary>
        /// <param name="id">The list.</param>
        /// <returns>The piece index.</returns>
        internal long EndPiece(long id) => _end[id];

        /// <summary>The number of pieces of list <paramref name="id"/> (Python's <c>len(list)</c>).</summary>
        /// <param name="id">The list.</param>
        /// <returns>The length.</returns>
        internal long Length(long id) => _end[id] - _start[id];

        /// <summary>The code units all pieces of list <paramref name="id"/> span (they are contiguous: a list's pieces were
        ///     appended one after another).</summary>
        /// <param name="id">The list.</param>
        /// <returns>The unit count.</returns>
        internal long UnitCount(long id) => _offsets[_end[id]] - _offsets[_start[id]];

        /// <summary>The piece offsets (piece k spans <c>[Offsets[k], Offsets[k + 1])</c>).</summary>
        internal long* Offsets => _offsets;

        /// <summary>The code units.</summary>
        internal byte* Units => _units;

        /// <summary>The per-piece <c>np.str_</c> flags, or null.</summary>
        internal bool* Flags => _flags;

        /// <summary>The bytes of one code unit: 1 (bytes pieces) or 4 (str pieces).</summary>
        internal int UnitSize => _unitSize;

        /// <summary>
        ///     Python's order of two lists of this heap (<c>OBJECT_compare</c>): the first unequal pieces decide (str by code
        ///     points, bytes by unsigned bytes, a prefix first), then the lengths.
        /// </summary>
        /// <param name="a">The first list.</param>
        /// <param name="b">The second list.</param>
        /// <returns>-1, 0 or 1.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal int Compare(long a, long b) => Compare(this, a, this, b);

        /// <summary>Whether two lists of this heap hold the same pieces (Python's <c>==</c>; an element's own
        ///     <c>np.str_</c> equals the same text).</summary>
        /// <param name="a">The first list.</param>
        /// <param name="b">The second list.</param>
        /// <returns>The equality.</returns>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        internal bool Equal(long a, long b) => Equal(this, a, this, b);

        /// <summary>
        ///     Python's order of list <paramref name="a"/> of <paramref name="ha"/> against list <paramref name="b"/> of
        ///     <paramref name="hb"/> (<c>list_richcompare</c>): the first unequal pieces decide (str by code points, bytes by
        ///     unsigned bytes, a prefix first), then the lengths. The two heaps must have one unit size — the same piece class
        ///     (<c>'U'</c> and <c>'T'</c> are both str): a str against a bytes piece is CPython's TypeError, which only
        ///     <see cref="NDListArray.RichCompare"/> raises.
        /// </summary>
        /// <param name="ha">The first list's heap.</param>
        /// <param name="a">The first list.</param>
        /// <param name="hb">The second list's heap (the same unit size).</param>
        /// <param name="b">The second list.</param>
        /// <returns>-1, 0 or 1.</returns>
        internal static int Compare(in ListPointers ha, long a, in ListPointers hb, long b)
        {
            long sa = ha._start[a], la = ha._end[a] - sa;
            long sb = hb._start[b], lb = hb._end[b] - sb;
            long common = la < lb ? la : lb;
            for (long i = 0; i < common; i++)
            {
                long pa = sa + i, pb = sb + i;
                long oa = ha._offsets[pa], na = ha._offsets[pa + 1] - oa;
                long ob = hb._offsets[pb], nb = hb._offsets[pb + 1] - ob;
                int c = ha._unitSize == 1
                    ? new ReadOnlySpan<byte>(ha._units + oa, checked((int)na)).SequenceCompareTo(new ReadOnlySpan<byte>(hb._units + ob, checked((int)nb)))
                    : new ReadOnlySpan<uint>((uint*)ha._units + oa, checked((int)na)).SequenceCompareTo(new ReadOnlySpan<uint>((uint*)hb._units + ob, checked((int)nb)));
                if (c != 0)
                    return c < 0 ? -1 : 1;
            }
            return la < lb ? -1 : la > lb ? 1 : 0;
        }

        /// <summary>Whether list <paramref name="a"/> of <paramref name="ha"/> and list <paramref name="b"/> of
        ///     <paramref name="hb"/> hold the same pieces (Python's <c>==</c>); the heaps must have one unit size.</summary>
        /// <param name="ha">The first list's heap.</param>
        /// <param name="a">The first list.</param>
        /// <param name="hb">The second list's heap (the same unit size).</param>
        /// <param name="b">The second list.</param>
        /// <returns>The equality.</returns>
        internal static bool Equal(in ListPointers ha, long a, in ListPointers hb, long b)
        {
            long sa = ha._start[a], la = ha._end[a] - sa;
            long sb = hb._start[b], lb = hb._end[b] - sb;
            if (la != lb)
                return false;
            int unit = ha._unitSize;
            for (long i = 0; i < la; i++)
            {
                long pa = sa + i, pb = sb + i;
                long oa = ha._offsets[pa], na = ha._offsets[pa + 1] - oa;
                long ob = hb._offsets[pb], nb = hb._offsets[pb + 1] - ob;
                if (na != nb)
                    return false;
                long bytes = na * unit;
                if (!new ReadOnlySpan<byte>(ha._units + oa * unit, checked((int)bytes))
                        .SequenceEqual(new ReadOnlySpan<byte>(hb._units + ob * unit, checked((int)bytes))))
                    return false;
            }
            return true;
        }

        /// <summary>A 64-bit hash of a list's content (its piece count, each piece's length and units): equal lists hash
        ///     alike.</summary>
        /// <param name="id">The list.</param>
        /// <returns>The hash.</returns>
        internal ulong Hash(long id)
        {
            const ulong K = 0x9E3779B97F4A7C15UL;
            long s = _start[id], l = _end[id] - s;
            ulong h = (ulong)l * K;
            for (long i = 0; i < l; i++)
            {
                long o = _offsets[s + i], n = _offsets[s + i + 1] - o;
                long bytes = n * _unitSize;
                byte* p = _units + o * _unitSize;
                // The piece's length separates ["ab"] from ["a", "b"]; its bytes go in eight at a time.
                h = (BitOperations.RotateLeft(h, 5) ^ (ulong)bytes) * K;
                long q = 0;
                for (; q + 8 <= bytes; q += 8)
                    h = (BitOperations.RotateLeft(h, 5) ^ System.Runtime.CompilerServices.Unsafe.ReadUnaligned<ulong>(p + q)) * K;
                ulong tail = 0;
                for (int t = 0; q < bytes; q++, t += 8)
                    tail |= (ulong)p[q] << t;
                h = (BitOperations.RotateLeft(h, 5) ^ tail) * K;
            }
            return h ^ (h >> 29);
        }
    }
}
