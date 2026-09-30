using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
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
//   any other IList / IEnumerable (List<T>, LINQ)                     a list
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
        {
            // Arrays converted from typed C# arrays during the walk are intermediates of this call (the fill copies
            // them): the Discovery releases them however the call ends.
            using var d = new Discovery(shapeOnly: false);
            var root = Walk(d, seq, 0);
            if (d.Ragged)
                throw Inhomogeneous(d);
            if (d.Refused is not null)
                throw d.Refused;
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
                return;
            }
            try
            {
                Promote(d, PolyNumber.FromObject(o).DiscoveredDtype());
            }
            catch (NotSupportedException e)
            {
                d.Refused ??= e;   // a Python int past uint64: NumPy's object array
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
}
