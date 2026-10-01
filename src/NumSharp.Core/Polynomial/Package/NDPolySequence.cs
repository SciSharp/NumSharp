using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

// =============================================================================
// NDPolySequence.cs — np.array of a Python sequence, for numpy.polynomial's arguments
// =============================================================================
//
// numpy.polynomial converts its array_like arguments with np.array / np.asarray, and a Python list or tuple is
// COERCED by array_coercion.c (PyArray_DiscoverDTypeAndShape): one depth-first walk that discovers the shape and
// the dtype together, then a fill. The house C# boundary (NDPolyNumber's header) makes these values Python
// SEQUENCES:
//   ITuple (ValueTuple / Tuple)                                       a tuple
//   object[] and every other C# array whose elements are not a NumSharp dtype — jagged double[][], NDArray[],
//     BigInteger[], ValueTuple[] …; a multi-dimensional one (object[,]) is a list of rows
//                                                                     a list
//   any other IList / IEnumerable (List<T>, LINQ)                     a list — one of a NumSharp dtype T (List<double>,
//                                                                     List<int> …) is coerced from its values in one pass
//                                                                     (TryTypedCollection), the same array the walk builds
// while an NDArray, a typed C# array (double[], int[,], Half[] …) and a Memory<T> of a dtype are NDARRAYS, and a
// string is a str scalar. Items classify the same way at every depth; any other item is a scalar: a Python scalar
// (bool, the integer primitives, BigInteger, float, double, Complex) or a NumPy scalar (Half, char, decimal).
//
// SHAPE. NumPy's walk is ported statement for statement (update_shape and the recursion), because its answers for
// irregular input are observable:
//   * the dims are fixed by the FIRST leaf reached (depth-first); a later leaf at another depth, or a sequence of
//     another length, is ragged: "setting an array element with a sequence. The requested array has an
//     inhomogeneous shape after N dimensions. The detected shape was (...) + inhomogeneous part.", where N and
//     the shape are what the walk still agreed on — [[[1],[1,2]], [[1],[1,2]]] reports (2, 2), not the inner
//     list's own (2,);
//   * an EMPTY sequence contributes no dtype and ends the dims at its depth: [np.zeros(0, np.int8), []] is an int8
//     (2, 0) array, and [np.zeros((0, 3)), []] a (2, 0) array into which the (0, 3) leaf cannot be assigned
//     ("could not broadcast input array from shape (0,3) into shape (0,)").
// DTYPE. Every leaf's DISCOVERED dtype — a Python bool/int/float/complex is bool/int64 (uint64 past int64)/
// float64/complex128, a NumPy scalar or array keeps its own — promoted with np.promote_types. Array coercion does
// NOT apply NEP 50's weak rules ([np.float32(1), 1.0] is float64). float64 when no leaf was reached ([] and
// [[], []]). A str or None leaf (NumPy's str / object arrays), any other unsupported object and a Python int past
// uint64 are refused with NotSupportedException — AFTER the walk, so a ragged input reports NumPy's ValueError.
//
// STR AND OBJECT ARRAYS AS SHAPES. polyutils.as_series checks its arrays' sizes and dims, and the other arguments',
// BEFORE the common type decides anything, and a str array never computes there at all ("Coefficient arrays have no
// common type"). ToArrayOrNonNumeric therefore reports such a sequence as NumPy's array SHAPE — its discovered dims,
// whether NumPy's dtype is object (a None / non-numeric / oversized-int leaf: str promotes with numbers to a str dtype,
// with object to object) or str — and the refusal, instead of raising it; NDPolySeries carries that as a non-numeric
// view and raises the refusal only where NumPy would start computing with Python objects.
//
// NDim is np.ndim of a sequence: the same walk without dtypes (a str / None leaf is simply a scalar there), which
// is what numpy.polynomial's "lbnd must be a scalar." / "scl must be a scalar." checks evaluate — and why a ragged
// lbnd raises the inhomogeneous ValueError instead.
//
// NumSharp has no NPY_MAXDIMS, so a sequence may nest to any depth (NumPy caps it at 64 dims).
//
// =============================================================================

namespace NumSharp
{
    /// <summary>
    ///     NumPy's array coercion of a Python sequence (<c>np.array(seq)</c>) for the <c>numpy.polynomial</c> package's
    ///     arguments, and the house test that decides which C# values ARE Python sequences (see the file header).
    /// </summary>
    internal static unsafe class PolySequence
    {
        // ---------------------------------------------------------------------------------------------
        //  Classification
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Whether a C# array's elements are a NumSharp dtype — which makes it an NDARRAY of that dtype (the house
        ///     mapping: <c>double[]</c>, <c>int[,]</c>, <c>Half[]</c>, <c>Complex[]</c> …), as opposed to a Python list of
        ///     its items (<c>object[]</c>, jagged <c>double[][]</c>, <c>NDArray[]</c>, <c>BigInteger[]</c> …).
        /// </summary>
        /// <param name="a">The array.</param>
        /// <returns>True for the 15 NumSharp element types.</returns>
        public static bool IsDtypeArray(Array a) => IsDtypeElement(a.GetType().GetElementType());

        /// <summary>
        ///     Whether <paramref name="o"/> is an ndarray-like value that <c>np.asanyarray</c> converts as a whole: a typed
        ///     C# array of a NumSharp dtype (<see cref="IsDtypeArray"/>) or a <see cref="Memory{T}"/> /
        ///     <see cref="ReadOnlyMemory{T}"/> of one. An <see cref="NDArray"/> is NOT included (callers take it as is).
        /// </summary>
        /// <param name="o">The value (may be null).</param>
        /// <returns>True for a typed array or memory of a dtype.</returns>
        public static bool IsArrayLike(object o) => o is Array a ? IsDtypeArray(a) : IsDtypeMemory(o);

        /// <summary>
        ///     Whether <paramref name="o"/> is a Python SEQUENCE under the house mapping: an <see cref="ITuple"/> (a tuple),
        ///     a C# array whose elements are not a NumSharp dtype, or any other <see cref="IEnumerable"/> (a list). A
        ///     string (a str scalar), an <see cref="NDArray"/> and a typed dtype array (ndarrays) are not.
        /// </summary>
        /// <param name="o">The value (may be null).</param>
        /// <returns>True when np.array would walk it as a sequence.</returns>
        public static bool IsSequence(object o) => o switch
        {
            null or string or NDArray => false,
            ITuple or ArrayRows => true,
            Array a => !IsDtypeArray(a),
            IEnumerable => true,
            _ => false,
        };

        /// <summary>
        ///     A Python sequence's items, in order (Python's <c>list(seq)</c>): a tuple's elements, an array's elements (a
        ///     multi-dimensional array yields its rows as nested sequences), an enumerable's items.
        /// </summary>
        /// <param name="seq">A value <see cref="IsSequence"/> accepts.</param>
        /// <returns>The items (the caller must not modify the array: an <c>object[]</c> argument is returned itself).</returns>
        /// <exception cref="ArgumentException"><paramref name="seq"/> is not a sequence.</exception>
        public static object[] Items(object seq)
        {
            switch (seq)
            {
                case object[] a:
                    // object[] and, by array covariance, every array of reference elements (double[][], NDArray[] …).
                    return a;
                case ITuple t:
                {
                    var r = new object[t.Length];
                    for (int i = 0; i < r.Length; i++)
                        r[i] = t[i];
                    return r;
                }
                case ArrayRows rows:
                    return rows.Items();
                case Array a when a.Rank > 1:
                    return new ArrayRows(a, System.Array.Empty<int>()).Items();
                case IEnumerable e:
                {
                    var list = e is ICollection c ? new List<object>(c.Count) : new List<object>();
                    foreach (var o in e)
                        list.Add(o);
                    return list.ToArray();
                }
                default:
                    throw new ArgumentException($"a {seq?.GetType().Name ?? "null"} is not a Python sequence", nameof(seq));
            }
        }

        /// <summary>
        ///     A Python tuple of <c>items[0..count)</c> under the house mapping — a <see cref="ValueTuple"/> of that arity
        ///     with <c>object</c> elements, nested the way C# nests eight or more (seven items and a <c>TRest</c> tuple of the
        ///     rest), so <see cref="ITuple"/> reads it back item for item. Used where NumPy returns a tuple built from a
        ///     tuple argument (a slice of it).
        /// </summary>
        /// <param name="items">The items.</param>
        /// <param name="count">How many leading items the tuple holds (0 … <c>items.Length</c>).</param>
        /// <returns>The boxed tuple.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="count"/> is negative or exceeds the items.</exception>
        public static object MakeTuple(object[] items, int count)
        {
            if ((uint)count > (uint)items.Length)
                throw new ArgumentOutOfRangeException(nameof(count));
            switch (count)
            {
                case 0: return ValueTuple.Create();
                case 1: return ValueTuple.Create(items[0]);
                case 2: return ValueTuple.Create(items[0], items[1]);
                case 3: return ValueTuple.Create(items[0], items[1], items[2]);
                case 4: return ValueTuple.Create(items[0], items[1], items[2], items[3]);
                case 5: return ValueTuple.Create(items[0], items[1], items[2], items[3], items[4]);
                case 6: return ValueTuple.Create(items[0], items[1], items[2], items[3], items[4], items[5]);
                case 7: return ValueTuple.Create(items[0], items[1], items[2], items[3], items[4], items[5], items[6]);
            }
            // Eight or more: ValueTuple<T1..T7, TRest>, whose TRest is the tuple of the remaining items — its type is only
            // known at run time, hence the constructed generic type.
            object rest = MakeTuple(items[7..count], count - 7);
            var o = typeof(object);
            var type = typeof(ValueTuple<,,,,,,,>).MakeGenericType(o, o, o, o, o, o, o, rest.GetType());
            return Activator.CreateInstance(type, items[0], items[1], items[2], items[3], items[4], items[5], items[6], rest);
        }

        // ---------------------------------------------------------------------------------------------
        //  np.array(seq) / np.ndim(seq)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     <c>np.array(seq)</c> of a Python sequence: NumPy's shape and dtype discovery over the nested items (see the
        ///     file header), then a fill of a new C-contiguous array.
        /// </summary>
        /// <param name="seq">A value <see cref="IsSequence"/> accepts.</param>
        /// <returns>A new, owning, C-contiguous array (float64 <c>(0,)</c> for an empty sequence).</returns>
        /// <exception cref="ValueError">A ragged sequence (NumPy's inhomogeneous-shape text), or an array item that cannot
        ///     be assigned into its slot after an empty sibling ended the dims (NumPy's broadcast text).</exception>
        /// <exception cref="NotSupportedException">A str / None / other non-numeric item, or a Python int past uint64 —
        ///     NumPy would build a str or object array, dtypes NumSharp does not have.</exception>
        public static NDArray ToArray(object seq)
            => ToArrayOrNonNumeric(seq, out var nonNumeric) ?? throw nonNumeric.Refusal;

        /// <summary>
        ///     <see cref="ToArray"/> for a caller that can carry NumPy's str / object array as a SHAPE (see the file header):
        ///     the same coercion, but a sequence NumPy turns into a str or object array returns null and describes that
        ///     array in <paramref name="nonNumeric"/> instead of raising.
        /// </summary>
        /// <param name="seq">A value <see cref="IsSequence"/> accepts.</param>
        /// <param name="nonNumeric">When the result is null: the dims NumPy's array would have, whether its dtype is
        ///     object (else str), and the refusal to raise where NumPy would compute with it. Default otherwise.</param>
        /// <returns>A new, owning, C-contiguous array (float64 <c>(0,)</c> for an empty sequence), or null for NumPy's str /
        ///     object array.</returns>
        /// <exception cref="ValueError">A ragged sequence (NumPy's inhomogeneous-shape text — raggedness is reported before
        ///     any refusal, as NumPy's walk does), or an array item that cannot be assigned into its slot after an empty
        ///     sibling ended the dims (NumPy's broadcast text).</exception>
        public static NDArray ToArrayOrNonNumeric(object seq, out PolyNonNumericArray nonNumeric)
        {
            nonNumeric = default;
            // The overwhelmingly common argument — a flat Python list of Python floats (`[1.5, -2.0, 0.25]`, a float
            // array's tolist()) or of Python ints — needs no discovery: its shape is (n,) and its dtype float64 / int64,
            // exactly what the walk below concludes for it, so it is filled directly (the walk costs ~50 ns an item in
            // node, dispatch and promotion bookkeeping; a 100-item list took longer to coerce than NumPy's whole call).
            if (seq is object[] flat && TryFlatScalars(flat, out var fast))
                return fast;
            // A typed C# COLLECTION (List<double>, List<int> …) is one flat Python list of scalars of a single C# kind: its
            // dtype needs no walk, and its values convert in one pass (item by item through the walk it cost ~40 ns an item —
            // a 1000-point List<double> took longer than NumPy's whole {p}vander). An empty one is the walk's (no dtype vote).
            if (TryTypedCollection(seq, out var values) && values.Length > 0)
                return CollectionArray(values);
            // Arrays converted from typed C# arrays during the walk are intermediates of this call (the fill copies
            // them): the Discovery releases them however the call ends.
            using var d = new Discovery(shapeOnly: false);
            var root = Walk(d, seq, 0);
            if (d.Ragged)
                throw Inhomogeneous(d);
            if (d.Refused is not null)
            {
                // NumPy's array exists — a str or object one — and only its shape can be observed before it computes: the
                // walk's dims are its dims (a str / None leaf is a scalar leaf, exactly as in NumPy's walk).
                var shape = new long[d.MaxDims];
                System.Array.Copy(d.Shape, shape, shape.Length);
                nonNumeric = new PolyNonNumericArray(shape, d.ObjectLeaf, d.Refused);
                return null;
            }
            NPTypeCode t = d.Dtype == NPTypeCode.Empty ? NPTypeCode.Double : d.Dtype;
            var dims = new long[d.MaxDims];
            System.Array.Copy(d.Shape, dims, dims.Length);
            var r = new NDArray(t, new Shape(dims), false);
            try
            {
                int size = r.dtypesize;
                // C-order byte strides (a zero dim counts as 1, like NumPy's — nothing is written through them then).
                var strides = new long[dims.Length];
                long acc = size;
                for (int k = dims.Length - 1; k >= 0; k--)
                {
                    strides[k] = acc;
                    acc *= Math.Max(dims[k], 1);
                }
                Fill(root, 0, (byte*)r.Storage.Address + r.Shape.offset * size, dims, strides, t, r);
                return r;
            }
            catch
            {
                // The fill can still refuse an array item (the empty-sibling broadcast case): nothing escapes.
                r.Dispose();
                throw;
            }
        }

        /// <summary>
        ///     <see cref="ToArray"/>'s fast path: a non-empty flat list whose items are ALL Python floats (boxed
        ///     <see cref="double"/>) — float64 — or ALL Python ints that fit int64 (boxed <see cref="long"/> or
        ///     <see cref="int"/>) — int64 —, filled straight into a new (n,) array. Anything else (a mix, a nested item, an
        ///     array, a bool, a NumPy scalar, an int past int64) declines to the full walk, which alone knows their rules.
        /// </summary>
        /// <param name="items">The list's items.</param>
        /// <param name="result">The array, when the fast path applies.</param>
        /// <returns>True when <paramref name="result"/> was built.</returns>
        private static bool TryFlatScalars(object[] items, out NDArray result)
        {
            result = null;
            int n = items.Length;
            if (n == 0)
                return false;
            // One classifying pass before any allocation: a mixed or nested list is the walk's business.
            bool doubles = items[0] is double, ints = items[0] is long || items[0] is int;
            if (!doubles && !ints)
                return false;
            for (int i = 1; i < n; i++)
            {
                object o = items[i];
                if (doubles ? o is not double : o is not long && o is not int)
                    return false;
            }
            if (doubles)
            {
                var r = new NDArray(NPTypeCode.Double, new Shape(n), false);
                double* p = (double*)r.Storage.Address;
                for (int i = 0; i < n; i++)
                    p[i] = (double)items[i];
                result = r;
            }
            else
            {
                var r = new NDArray(NPTypeCode.Int64, new Shape(n), false);
                long* p = (long*)r.Storage.Address;
                for (int i = 0; i < n; i++)
                    p[i] = items[i] is long l ? l : (int)items[i];
                result = r;
            }
            return true;
        }

        /// <summary>
        ///     <c>np.ndim(seq)</c> of a Python sequence (<c>asarray(seq).ndim</c>): the shape walk alone — a str, None or any
        ///     other object item is a scalar there — so a ragged sequence raises NumPy's inhomogeneous ValueError.
        /// </summary>
        /// <param name="seq">A value <see cref="IsSequence"/> accepts.</param>
        /// <returns>The rank (≥ 1: a sequence always has at least its own axis).</returns>
        /// <exception cref="ValueError">A ragged sequence.</exception>
        public static int NDim(object seq)
        {
            using var d = new Discovery(shapeOnly: true);
            Walk(d, seq, 0);
            if (d.Ragged)
                throw Inhomogeneous(d);
            return d.MaxDims;
        }

        // ---------------------------------------------------------------------------------------------
        //  The walk (PyArray_DiscoverDTypeAndShape_Recursive)
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     The walk's state: NumPy's <c>out_shape</c>, <c>max_dims</c>, discovery flags and <c>out_descr</c> — plus the
        ///     arrays the walk converted from typed C# arrays, which it owns and releases on <see cref="Dispose"/>.
        /// </summary>
        private sealed class Discovery : IDisposable
        {
            /// <summary>The discovered dims (grown on demand; entries past <see cref="MaxDims"/> are stale).</summary>
            public long[] Shape = new long[8];
            /// <summary>NumPy's <c>max_dims</c>: unbounded until the first leaf fixes it (NumSharp has no 64-dim cap).</summary>
            public int MaxDims = int.MaxValue;
            /// <summary><c>MAX_DIMS_WAS_REACHED</c>: a leaf fixed the dims, later leaves are compared against them.</summary>
            public bool Reached;
            /// <summary><c>FOUND_RAGGED_ARRAY</c>.</summary>
            public bool Ragged;
            /// <summary>The promoted dtype (<see cref="NPTypeCode.Empty"/> until a leaf contributes one).</summary>
            public NPTypeCode Dtype = NPTypeCode.Empty;
            /// <summary>np.ndim's walk: shapes only, every non-sequence non-array item a scalar.</summary>
            public readonly bool ShapeOnly;
            /// <summary>The first refused item, raised only if the input is not ragged (NumPy reports raggedness first).</summary>
            public NotSupportedException Refused;
            /// <summary>
            ///     A refused item makes NumPy's dtype OBJECT: None, a non-numeric object or a Python int past uint64 (str
            ///     items alone make a str dtype — NumPy promotes str with every number to str, and with object to object).
            /// </summary>
            public bool ObjectLeaf;
            /// <summary>Arrays converted from typed C# arrays (disposed after the fill).</summary>
            public List<NDArray> Owned;

            /// <summary>A fresh walk.</summary>
            /// <param name="shapeOnly">Discover shapes only (np.ndim).</param>
            public Discovery(bool shapeOnly) { ShapeOnly = shapeOnly; }

            /// <summary>
            ///     Releases the arrays converted from typed C# arrays: intermediates of the walk (the fill has copied them by
            ///     the time the walk ends, and a failed walk never needs them). Idempotent.
            /// </summary>
            public void Dispose()
            {
                if (Owned is null)
                    return;
                foreach (var a in Owned)
                    a.Dispose();
                Owned = null;
            }
        }

        /// <summary>A walked sequence: its items, and per item the child sequence or array leaf (null: a scalar item).</summary>
        private sealed class Node
        {
            /// <summary>The items.</summary>
            public readonly object[] Items;
            /// <summary>Per item: a <see cref="Node"/>, an <see cref="NDArray"/> leaf, or null (a scalar, converted at fill).</summary>
            public object[] Kids;

            /// <summary>A sequence node.</summary>
            /// <param name="items">The items.</param>
            public Node(object[] items) { Items = items; }
        }

        /// <summary>
        ///     One sequence at depth <paramref name="curr"/>: its length updates the shape as a SEQUENCE (it never fixes the
        ///     dim count), an empty one ends the dims one level down, and every item is walked in order — past a ragged one
        ///     too, as NumPy does, since later items can still shrink the reported shape.
        /// </summary>
        /// <param name="d">The walk.</param><param name="seq">The sequence.</param><param name="curr">Its depth.</param>
        /// <returns>The node, or null when the sequence itself was ragged (its items are not visited).</returns>
        private static Node Walk(Discovery d, object seq, int curr)
        {
            object[] items = Items(seq);
            long size = items.Length;
            if (!UpdateShape(d, curr, 1, &size, sequence: true))
            {
                d.Ragged = true;
                return null;
            }
            var node = new Node(items);
            if (items.Length == 0)
            {
                // "If the sequence is empty, this must be the last dimension": no dtype, dims end here.
                d.Reached = true;
                d.MaxDims = curr + 1;
                return node;
            }
            for (int i = 0; i < items.Length; i++)
            {
                object o = items[i];
                // Most items are C# numbers: one cheap type test before the sequence / ndarray classification.
                if (PolyNumber.IsScalarValue(o))
                {
                    ScalarLeaf(d, o, curr + 1);
                    continue;
                }
                if (IsSequence(o))
                {
                    // A flat list of one scalar kind — a typed collection, or an object[] of Python floats / of Python ints
                    // (the top level's fast path, nested: the 2-D / 3-D point stacks, the rows of an N-D series) — takes the
                    // same shape updates in two calls and fills as one array.
                    if (!d.ShapeOnly && TryFlatLeaf(o, out var flatLeaf))
                    {
                        FlatLeaf(d, node, i, flatLeaf, curr + 1);
                        continue;
                    }
                    var child = Walk(d, o, curr + 1);
                    if (child is not null && !d.ShapeOnly)
                        (node.Kids ??= new object[items.Length])[i] = child;
                }
                else if (o is NDArray nd)
                    ArrayLeaf(d, node, i, nd, curr + 1);
                else if (IsArrayLike(o))
                {
                    if (d.ShapeOnly)
                    {
                        // np.ndim needs the array's dims only: read them without converting.
                        if (!ArrayLikeDimsFit(d, o, curr + 1))
                            d.Ragged = true;
                        continue;
                    }
                    var a = np.asanyarray(o);
                    (d.Owned ??= new List<NDArray>()).Add(a);
                    ArrayLeaf(d, node, i, a, curr + 1);
                }
                else
                    ScalarLeaf(d, o, curr + 1);
            }
            return node;
        }

        /// <summary>An ndarray item: its dims fix (or are checked against) the shape, then its dtype joins the promotion.</summary>
        /// <param name="d">The walk.</param><param name="node">The parent.</param><param name="i">The item's index.</param>
        /// <param name="a">The array.</param><param name="curr">Its depth.</param>
        private static void ArrayLeaf(Discovery d, Node node, int i, NDArray a, int curr)
        {
            var dims = a.Shape.dimensions ?? System.Array.Empty<long>();
            fixed (long* p = dims)
            {
                if (!UpdateShape(d, curr, dims.Length, p, sequence: false))
                {
                    d.Ragged = true;
                    return;
                }
            }
            if (d.ShapeOnly)
                return;
            Promote(d, a.typecode);
            (node.Kids ??= new object[node.Items.Length])[i] = a;
        }

        /// <summary>
        ///     Whether a sequence item is a FLAT list of one scalar kind, and its values as one array: a non-empty typed
        ///     collection (<see cref="TryTypedCollection"/>), or a non-empty <c>object[]</c> whose items are all Python floats or
        ///     all Python ints within int64 (<see cref="TryFlatScalars"/>). Anything else — empty, mixed, nested — is walked item
        ///     by item.
        /// </summary>
        /// <param name="o">The sequence item.</param>
        /// <param name="a">Its values as a new (n,) array of the list's discovered dtype, when the result is true.</param>
        /// <returns>True for a flat list of one scalar kind.</returns>
        private static bool TryFlatLeaf(object o, out NDArray a)
        {
            a = null;
            if (o is object[] items)
                return TryFlatScalars(items, out a);
            if (TryTypedCollection(o, out var values) && values.Length > 0)
            {
                a = CollectionArray(values);
                return true;
            }
            return false;
        }

        /// <summary>
        ///     A flat-list item (<see cref="TryFlatLeaf"/>) at depth <paramref name="curr"/>: the shape updates the walk makes for
        ///     a sequence of scalars — the sequence's own at <paramref name="curr"/>, then ONE scalar leaf at
        ///     <paramref name="curr"/> + 1 (every item is a scalar at that depth, so the other n - 1 updates are the same
        ///     comparison and change nothing) — and its values as ONE array leaf of the list's dtype, which the fill casts into
        ///     the result like any array leaf (the same values the per-item conversion writes: every leaf dtype here converts
        ///     into the promoted dtype exactly, or with the one correctly rounded int64 / uint64 -> float64 step both take).
        /// </summary>
        /// <param name="d">The walk.</param><param name="node">The parent.</param><param name="i">The item's index.</param>
        /// <param name="a">The list's values (non-empty, owned by the walk from here on).</param><param name="curr">The list's depth.</param>
        private static void FlatLeaf(Discovery d, Node node, int i, NDArray a, int curr)
        {
            (d.Owned ??= new List<NDArray>()).Add(a);   // an intermediate: the fill copies it (or the walk is abandoned)
            long len = a.size;
            if (!UpdateShape(d, curr, 1, &len, sequence: true) || !UpdateShape(d, curr + 1, 0, null, sequence: false))
            {
                d.Ragged = true;
                return;
            }
            Promote(d, a.typecode);
            (node.Kids ??= new object[node.Items.Length])[i] = a;
        }

        /// <summary>np.ndim's view of a typed C# array / Memory item: its dims, read without a conversion.</summary>
        /// <param name="d">The walk.</param><param name="o">The array-like.</param><param name="curr">Its depth.</param>
        /// <returns>False when the item is ragged against the shape.</returns>
        private static bool ArrayLikeDimsFit(Discovery d, object o, int curr)
        {
            if (o is Array arr)
            {
                long* dims = stackalloc long[arr.Rank];
                for (int k = 0; k < arr.Rank; k++)
                    dims[k] = arr.GetLength(k);
                return UpdateShape(d, curr, arr.Rank, dims, sequence: false);
            }
            long len = (int)o.GetType().GetProperty("Length")!.GetValue(o)!;
            return UpdateShape(d, curr, 1, &len, sequence: false);
        }

        /// <summary>
        ///     A scalar item: it fixes (or is checked against) the dims at its depth, then its DISCOVERED dtype joins the
        ///     promotion. A str / None / unsupported object, or a Python int past uint64, is recorded as refused.
        /// </summary>
        /// <param name="d">The walk.</param><param name="o">The item.</param><param name="curr">Its depth.</param>
        private static void ScalarLeaf(Discovery d, object o, int curr)
        {
            if (!UpdateShape(d, curr, 0, null, sequence: false))
            {
                d.Ragged = true;
                return;
            }
            if (d.ShapeOnly)
                return;
            if (!PolyNumber.IsScalarValue(o))
            {
                d.Refused ??= new NotSupportedException(o switch
                {
                    null => "None has no NumSharp equivalent: NumPy would build an object array, a dtype NumSharp does not have",
                    string => "a Python str operand makes NumPy build a str/object array, a dtype NumSharp does not have",
                    _ => $"a {o.GetType().Name} item makes NumPy build an object array, a dtype NumSharp does not have",
                });
                // A str leaf alone leaves NumPy's dtype a str one; anything else here makes it object.
                if (o is not string)
                    d.ObjectLeaf = true;
                return;
            }
            // A scalar value always converts (FromObject throws only for None / str / unknown objects, handled above). A
            // Python int past uint64 makes NumPy's array an OBJECT one: classified without an exception — a throw and catch
            // per leaf cost ~1 µs, half of a two-term object series' whole {p}roots — and its refusal is built, not thrown,
            // raised only where NumPy would start computing with Python objects.
            var num = PolyNumber.FromObject(o);
            if (num.TryDiscoveredDtype(out NPTypeCode t))
                Promote(d, t);
            else
            {
                d.Refused ??= num.ObjectArrayRefusal();
                d.ObjectLeaf = true;
            }
        }

        /// <summary>NumPy's <c>handle_promotion</c>: the first dtype, then <c>np.promote_types</c> with each later one.</summary>
        /// <param name="d">The walk.</param><param name="t">A leaf's dtype.</param>
        private static void Promote(Discovery d, NPTypeCode t)
            => d.Dtype = d.Dtype == NPTypeCode.Empty ? t : PolyTyping.Promote(d.Dtype, t);

        /// <summary>
        ///     NumPy's <c>update_shape</c>, statement for statement: a leaf (not a sequence) FIXES the dim count at
        ///     <c>curr + newNdim</c> (a second, different count is ragged); until the first leaf the dims are written,
        ///     afterwards compared, and a mismatch shrinks the dim count to what still agrees — by the unusable dims for a
        ///     leaf, to <paramref name="curr"/> for a sequence.
        /// </summary>
        /// <param name="d">The walk.</param><param name="curr">The depth the new dims start at.</param>
        /// <param name="newNdim">How many dims the item contributes (a sequence 1, a scalar 0, an array its rank).</param>
        /// <param name="newShape">Those dims.</param><param name="sequence">The item is a sequence (not a leaf).</param>
        /// <returns>False when the item is ragged against the shape.</returns>
        private static bool UpdateShape(Discovery d, int curr, int newNdim, long* newShape, bool sequence)
        {
            bool success = true;
            bool reached = d.Reached;
            if ((long)curr + newNdim > d.MaxDims)
            {
                // Deeper than the fixed dim count: only the dims that exist are checked; max_dims is unchanged.
                success = false;
                newNdim = d.MaxDims - curr;
            }
            else if (!sequence && d.MaxDims != curr + newNdim)
            {
                // A leaf fixes the count (a sequence never does); a leaf at another depth than an earlier one is ragged.
                d.MaxDims = curr + newNdim;
                if (reached)
                    success = false;
            }
            for (int i = 0; i < newNdim; i++)
            {
                EnsureShape(d, curr + i + 1);
                if (!reached)
                    d.Shape[curr + i] = newShape[i];
                else if (newShape[i] != d.Shape[curr + i])
                {
                    success = false;
                    if (!sequence)
                        d.MaxDims -= newNdim - i;   // drop the dims this leaf made unusable
                    else
                        d.MaxDims = curr;           // a sequence never set max_dims, so set it now
                    break;
                }
            }
            if (!sequence)
                d.Reached = true;
            return success;
        }

        /// <summary>Grows the shape buffer to hold <paramref name="count"/> dims.</summary>
        /// <param name="d">The walk.</param><param name="count">Dims needed.</param>
        private static void EnsureShape(Discovery d, int count)
        {
            if (count > d.Shape.Length)
                System.Array.Resize(ref d.Shape, Math.Max(count, d.Shape.Length * 2));
        }

        // ---------------------------------------------------------------------------------------------
        //  The fill
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Writes a walked sequence into the result block at <paramref name="p"/>: a child sequence recursively, an array
        ///     leaf by a casting copy, a scalar by np.array's element conversion (<see cref="PolyNumber.WriteElement"/>).
        /// </summary>
        /// <param name="node">The sequence (not ragged: its length is <c>dims[depth]</c>).</param>
        /// <param name="depth">Its depth.</param><param name="p">Its first element in the result.</param>
        /// <param name="dims">The result dims.</param><param name="strides">The result's C-order byte strides.</param>
        /// <param name="t">The result dtype.</param><param name="r">The result (the base for array-leaf views).</param>
        /// <exception cref="ValueError">An array leaf whose dims differ from its slot (only possible after an empty
        ///     sibling ended the dims above the leaf's own).</exception>
        private static void Fill(Node node, int depth, byte* p, long[] dims, long[] strides, NPTypeCode t, NDArray r)
        {
            for (int i = 0; i < node.Items.Length; i++)
            {
                byte* q = p + i * strides[depth];
                switch (node.Kids?[i])
                {
                    case Node child:
                        Fill(child, depth + 1, q, dims, strides, t, r);
                        break;
                    case NDArray leaf:
                        CopyLeaf(leaf, depth + 1, q, dims, t, r);
                        break;
                    default:
                        PolyNumber.FromObject(node.Items[i]).WriteElement(t, q);
                        break;
                }
            }
        }

        /// <summary>
        ///     Assigns an array leaf into its slot, <c>dims[depth..]</c> at <paramref name="q"/> — NumPy's
        ///     <c>PyArray_AssignArray</c> of the coercion cache, an unsafe-cast copy through any layout.
        /// </summary>
        /// <param name="leaf">The array.</param><param name="depth">The slot's first dim.</param>
        /// <param name="q">The slot's first element.</param><param name="dims">The result dims.</param>
        /// <param name="t">The result dtype.</param><param name="r">The result.</param>
        /// <exception cref="ValueError"><c>could not broadcast input array from shape (a) into shape (b)</c>.</exception>
        private static void CopyLeaf(NDArray leaf, int depth, byte* q, long[] dims, NPTypeCode t, NDArray r)
        {
            var ld = leaf.Shape.dimensions ?? System.Array.Empty<long>();
            int slot = dims.Length - depth;
            bool same = ld.Length == slot;
            for (int k = 0; same && k < slot; k++)
                same = ld[k] == dims[depth + k];
            if (!same)
            {
                var sd = new long[Math.Max(slot, 0)];
                System.Array.Copy(dims, depth, sd, 0, sd.Length);
                throw new ValueError($"could not broadcast input array from shape {new Shape(ld).ToPythonTuple()} " +
                                     $"into shape {new Shape(sd).ToPythonTuple()}");
            }
            if (slot == 0)
            {
                PolyNumber.FromArray(leaf).WriteAs(t, q);   // a 0-d leaf: its element, cast
                return;
            }
            if (leaf.size == 0)
                return;
            // The slot is a C-contiguous block of the result: view it with the leaf's dims and copy (NDIter casts and
            // walks the leaf in any layout).
            var vd = (long[])ld.Clone();
            var vs = new long[slot];
            long acc = 1;
            for (int k = slot - 1; k >= 0; k--)
            {
                vs[k] = acc;
                acc *= vd[k];
            }
            long offset = (q - (byte*)r.Storage.Address) / r.dtypesize;
            var shape = new Shape(vd, vs, offset, r.Shape.bufferSize);
            using var view = new NDArray(r.Storage.Alias(ref shape), r.TensorEngine, skipEngineResolve: true);
            NDIter.Copy(view, leaf);
        }

        /// <summary>
        ///     NumPy's ragged-input error: the dim count and the dims the walk still agreed on, the shape as a Python tuple
        ///     repr (<c>(2,)</c>, <c>(2, 2)</c>).
        /// </summary>
        /// <param name="d">The walk.</param>
        /// <returns>The exception to throw.</returns>
        private static ValueError Inhomogeneous(Discovery d)
        {
            int n = Math.Max(d.MaxDims, 0);
            var parts = new string[n];
            for (int k = 0; k < n; k++)
                parts[k] = d.Shape[k].ToString(CultureInfo.InvariantCulture);
            string shape = n == 1 ? $"({parts[0]},)" : "(" + string.Join(", ", parts) + ")";
            return new ValueError("setting an array element with a sequence. The requested array has an inhomogeneous shape " +
                                  $"after {n} dimensions. The detected shape was {shape} + inhomogeneous part.");
        }

        // ---------------------------------------------------------------------------------------------
        //  Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>Whether <paramref name="e"/> is one of NumSharp's 15 element types (never an array type).</summary>
        /// <param name="e">The element type.</param>
        /// <returns>True for bool, char, the integer types, Half, float, double, decimal and Complex.</returns>
        private static bool IsDtypeElement(Type e)
        {
            if (e is null || e.IsArray || e.IsEnum)
                return false;
            switch (e.GetTypeCode())
            {
                case NPTypeCode.Boolean:
                case NPTypeCode.Char:
                case NPTypeCode.SByte:
                case NPTypeCode.Byte:
                case NPTypeCode.Int16:
                case NPTypeCode.UInt16:
                case NPTypeCode.Int32:
                case NPTypeCode.UInt32:
                case NPTypeCode.Int64:
                case NPTypeCode.UInt64:
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                case NPTypeCode.Double:
                case NPTypeCode.Decimal:
                case NPTypeCode.Complex:
                    return true;
                default:
                    return false;
            }
        }

        // ---------------------------------------------------------------------------------------------
        //  Typed collections: List<double>, List<int> … — Python lists of one scalar kind
        // ---------------------------------------------------------------------------------------------

        /// <summary>Per runtime type: the materializer of a typed collection (its values as a <c>T[]</c>), or null when the
        ///     type is not one. The interface scan runs once per type.</summary>
        private static readonly ConcurrentDictionary<Type, Func<object, Array>> s_collectionMaterializers = new();

        /// <summary>The open generic <see cref="MaterializeCollection{T}"/>, closed per element type.</summary>
        private static readonly MethodInfo s_materializeCollection =
            typeof(PolySequence).GetMethod(nameof(MaterializeCollection), BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new MissingMethodException(nameof(PolySequence), nameof(MaterializeCollection));

        /// <summary>
        ///     Whether <paramref name="o"/> is a typed C# COLLECTION — an <see cref="IEnumerable{T}"/> of one of NumSharp's element
        ///     types that is not a C# array (those are ndarrays), a string (a str), an NDArray or a tuple: a <c>List&lt;T&gt;</c>,
        ///     any other collection, a LINQ sequence — which the house map reads as a Python LIST of T's scalars. Its values are
        ///     copied out (one <see cref="ICollection{T}.CopyTo"/>, or one enumeration of a lazy sequence).
        /// </summary>
        /// <param name="o">The value (may be null).</param>
        /// <param name="values">The collection's values as a <c>T[]</c>, when the result is true.</param>
        /// <returns>True for a typed collection.</returns>
        private static bool TryTypedCollection(object o, out Array values)
        {
            values = null;
            if (o is null or Array or string or NDArray or ITuple or ArrayRows)
                return false;
            var materialize = s_collectionMaterializers.GetOrAdd(o.GetType(), static t =>
            {
                foreach (var itf in t.GetInterfaces())
                {
                    if (!itf.IsGenericType || itf.GetGenericTypeDefinition() != typeof(IEnumerable<>))
                        continue;
                    var e = itf.GetGenericArguments()[0];
                    if (IsDtypeElement(e))
                        return s_materializeCollection.MakeGenericMethod(e).CreateDelegate<Func<object, Array>>();
                }
                return null;
            });
            if (materialize is null)
                return false;
            values = materialize(o);
            return true;
        }

        /// <summary>A typed collection's values: one <see cref="ICollection{T}.CopyTo"/> into a new array, or one enumeration
        ///     of a lazy sequence.</summary>
        /// <typeparam name="T">The element type (a NumSharp dtype's).</typeparam>
        /// <param name="o">The collection (an <see cref="IEnumerable{T}"/>).</param>
        /// <returns>The values, in enumeration order.</returns>
        private static Array MaterializeCollection<T>(object o)
        {
            if (o is ICollection<T> c)
            {
                var a = new T[c.Count];
                c.CopyTo(a, 0);
                return a;
            }
            return System.Linq.Enumerable.ToArray((IEnumerable<T>)o);
        }

        /// <summary>
        ///     <c>np.array</c> of the Python list a typed collection stands for: its values in the dtype array coercion DISCOVERS
        ///     for the scalars its C# elements are — bool a Python bool (bool); every integer primitive a Python int (int64 —
        ///     a <c>ulong</c> past int64 is uint64, and a mix of both magnitudes float64, NumPy's promotion of the two); float
        ///     and double Python floats (float64: a C# float widens exactly); Complex a Python complex (complex128); Half,
        ///     char and decimal NumPy scalars (their own dtype).
        /// </summary>
        /// <param name="values">The values (a non-empty <c>T[]</c>).</param>
        /// <returns>A new (n,) array; the caller owns it.</returns>
        private static NDArray CollectionArray(Array values)
        {
            var a = np.asanyarray(values);   // an ndarray of T (a copy into NumSharp memory)
            NPTypeCode t = a.typecode switch
            {
                NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                    or NPTypeCode.UInt32 or NPTypeCode.Int64 => NPTypeCode.Int64,
                NPTypeCode.UInt64 => ULongListDtype(a),
                NPTypeCode.Single => NPTypeCode.Double,
                var same => same,
            };
            if (t == a.typecode)
                return a;
            try
            {
                return a.astype(t);
            }
            finally
            {
                a.Dispose();   // the T-typed copy was an intermediate of the conversion
            }
        }

        /// <summary>
        ///     The dtype of a Python list of non-negative ints (a <c>ulong</c> collection): int64 when every value fits int64,
        ///     uint64 when none does, float64 for a mix (np.promote_types(int64, uint64), as NumPy's discovery promotes them).
        /// </summary>
        /// <param name="a">The values as a uint64 array.</param>
        /// <returns>The dtype.</returns>
        private static NPTypeCode ULongListDtype(NDArray a)
        {
            using var max = np.amax(a);   // 0-d reductions (house kernels): intermediates of the classification
            using var min = np.amin(a);
            bool big = max.GetAtIndex<ulong>(0) > long.MaxValue;
            bool small = min.GetAtIndex<ulong>(0) <= long.MaxValue;
            return big ? (small ? NPTypeCode.Double : NPTypeCode.UInt64) : NPTypeCode.Int64;
        }

        /// <summary>Whether <paramref name="o"/> is a <see cref="Memory{T}"/> / <see cref="ReadOnlyMemory{T}"/> of a dtype.</summary>
        /// <param name="o">The value (may be null).</param>
        /// <returns>True for a dtype memory.</returns>
        private static bool IsDtypeMemory(object o)
        {
            var type = o?.GetType();
            if (type is null || !type.IsGenericType)
                return false;
            var g = type.GetGenericTypeDefinition();
            return (g == typeof(Memory<>) || g == typeof(ReadOnlyMemory<>)) && IsDtypeElement(type.GetGenericArguments()[0]);
        }

        /// <summary>
        ///     A multi-dimensional C# array of non-dtype elements (<c>object[,]</c>) seen as Python's list of rows: the
        ///     sub-array at a fixed index prefix, itself a sequence until the last dimension yields the elements.
        /// </summary>
        private sealed class ArrayRows
        {
            /// <summary>The array.</summary>
            private readonly Array _a;
            /// <summary>The fixed leading indices (absolute: lower bounds included).</summary>
            private readonly int[] _prefix;

            /// <summary>The rows under <paramref name="prefix"/>.</summary>
            /// <param name="a">The array.</param><param name="prefix">Fixed leading indices.</param>
            public ArrayRows(Array a, int[] prefix)
            {
                _a = a;
                _prefix = prefix;
            }

            /// <summary>The items along the next dimension: elements at the last one, nested rows before it.</summary>
            /// <returns>The items.</returns>
            public object[] Items()
            {
                int dim = _prefix.Length;
                int lo = _a.GetLowerBound(dim), n = _a.GetLength(dim);
                bool last = dim + 1 == _a.Rank;
                var idx = new int[dim + 1];
                System.Array.Copy(_prefix, idx, dim);
                var r = new object[n];
                for (int j = 0; j < n; j++)
                {
                    idx[dim] = lo + j;
                    r[j] = last ? _a.GetValue(idx) : new ArrayRows(_a, (int[])idx.Clone());
                }
                return r;
            }
        }
    }

    /// <summary>
    ///     The array NumPy's <c>np.array</c> builds from a sequence NumSharp cannot hold — a str or object one — described by
    ///     what can be observed of it before NumPy computes with it (see <see cref="PolySequence.ToArrayOrNonNumeric"/>).
    /// </summary>
    internal readonly struct PolyNonNumericArray
    {
        /// <summary>NumPy's array dims (the coercion walk's; at least one dim for a sequence).</summary>
        public readonly long[] Dims;
        /// <summary>NumPy's dtype is object (else a str dtype): None, a non-numeric object or an oversized Python int.</summary>
        public readonly bool IsObject;
        /// <summary>What NumSharp raises where NumPy would compute with the array (it has no str / object dtype).</summary>
        public readonly NotSupportedException Refusal;

        /// <summary>Describes one such array.</summary>
        /// <param name="dims">Its dims.</param>
        /// <param name="isObject">Its dtype is object.</param>
        /// <param name="refusal">The deferred refusal.</param>
        public PolyNonNumericArray(long[] dims, bool isObject, NotSupportedException refusal)
        {
            Dims = dims;
            IsObject = isObject;
            Refusal = refusal;
        }
    }
}
