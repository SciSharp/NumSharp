using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Produce an object that mimics broadcasting.
        /// </summary>
        /// <param name="arrays">
        ///     The arrays to broadcast against one another. NumPy caps this at 64 operands
        ///     (NPY_MAXARGS); NumSharp imposes no cap, matching its unlimited-operand <c>NDIter</c>.
        /// </param>
        /// <returns>Broadcast the input parameters against one another, and return an object that encapsulates the result. Amongst others, it has shape and nd properties, and may be used as an iterator.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html</remarks>
        public static Broadcast broadcast(params NDArray[] arrays)
        {
            return new Broadcast(arrays);
        }

        /// <summary>
        ///     Two-operand overload — the most common case. Retained as an explicit overload so
        ///     <c>np.broadcast(a, b)</c> stays binary-compatible and skips the params-array allocation.
        /// </summary>
        /// <param name="nd1">The first operand.</param>
        /// <param name="nd2">The second operand.</param>
        /// <returns>The broadcast object of the two operands.</returns>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast together (NumPy's text).</exception>
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            return new Broadcast(nd1, nd2);
        }

        /// <summary>
        ///     NumPy's <c>np.broadcast(*arrays)</c> over object arrays of lists - <c>np.char.split</c>'s
        ///     <see cref="NDListArray"/> results: their shapes broadcast together, and iterating yields one tuple of the
        ///     operands' lists (each a <see cref="List{T}"/> of its pieces, NumPy's Python list) per broadcast position.
        /// </summary>
        /// <param name="arrays">The list arrays (NumPy's sequence of operands); the object borrows them.</param>
        /// <returns>The broadcast object over the list arrays.</returns>
        /// <exception cref="ArgumentNullException">An operand is null.</exception>
        /// <exception cref="ObjectDisposedException">An operand was disposed.</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast together (NumPy's text).</exception>
        /// <remarks>
        ///     A sequence parameter, not <c>params</c>: a second <c>params</c> form would make <c>np.broadcast()</c>
        ///     ambiguous. NumSharp's lists are their own array type, so a list array and a numeric array do not broadcast in
        ///     one object here (NumPy's object array would). API-parity batch 13.
        /// </remarks>
        public static Broadcast broadcast(NDListArray[] arrays)
        {
            return new Broadcast(arrays);
        }

        /// <summary>
        ///     NumPy's <c>numpy.broadcast</c> — the broadcast result of N operands, usable as an
        ///     iterator. Like NumPy, the object is its OWN iterator (<c>iter(b) is b</c>): it keeps a
        ///     single live cursor exposed as <see cref="index"/>, iterating it yields one tuple of
        ///     per-operand values per step (advancing the cursor), and <see cref="reset"/> rewinds it.
        /// </summary>
        public class Broadcast : IEnumerable<object[]>, IEnumerator<object[]>
        {
            [NDBorrowed] private readonly NDArray[] _ops;   // the caller's operands (a list array's ids for list operands)
            [NDBorrowed] private readonly NDListArray[] _lists; // the caller's list arrays, or null for array operands
            private NDFlatIterator[] _iters;
            private NDArray[] _views;     // operands broadcast to the result shape (lazy) — OWNED, released by Dispose
            private int _index;           // live cursor — NumPy's broadcast.index
            private object[] _current;    // values at the previous cursor position

            /// <summary>The two-operand broadcast <see cref="np.broadcast(NDArray, NDArray)"/> builds.</summary>
            /// <param name="op1">The first operand (borrowed).</param>
            /// <param name="op2">The second operand (borrowed).</param>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast together.</exception>
            internal Broadcast(NDArray op1, NDArray op2) : this(new[] { op1, op2 }) { }

            /// <summary>
            ///     NumPy's <c>np.broadcast(*arrays)</c> class constructor: broadcast N operands together. The result
            ///     <see cref="shape"/> is the broadcast of every operand's shape (resolved eagerly;
            ///     iterators are built lazily on first <see cref="iters"/> access). Incompatible
            ///     shapes raise <see cref="IncorrectShapeException"/> (NumPy parity). Unlike NumPy's
            ///     fixed NPY_MAXARGS=64 cap, any number of operands is accepted — matching NumSharp's
            ///     unlimited-operand <c>NDIter</c>. No operand at all (<c>new np.Broadcast()</c>) is NumPy's
            ///     <c>np.broadcast()</c>: the 0-d broadcast, shape <c>()</c> and size 1.
            /// </summary>
            /// <param name="arrays">The operands (null is treated as none); the object borrows them, it does not own or
            ///     dispose them.</param>
            /// <remarks>Public since API-parity batch 13 (NumPy's class IS the function); the operand-less public
            ///     constructor it replaced built an object whose <c>shape</c> threw and whose <c>size</c> was 0.</remarks>
            /// <exception cref="IncorrectShapeException">Two operands disagree on an axis: NumPy's <c>shape mismatch: objects
            ///     cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (3,) and arg 1 with shape
            ///     (4,).</c> (<see cref="Shape.MultiIterShape"/>).</exception>
            /// <exception cref="ValueError">The broadcast shape's element count does not fit a 64-bit integer
            ///     (<c>broadcast dimensions too large.</c>).</exception>
            public Broadcast(params NDArray[] arrays)
            {
                _ops = arrays ?? Array.Empty<NDArray>();

                // NUMSHARP DIVERGENCE: NumPy caps the multi-iterator at NPY_MAXARGS=64, but
                // NumSharp's NDIter allocates all per-operand state dynamically and imposes no
                // operand cap (see NDIter.State.cs — "UNLIMITED ... matches NumSharp's core
                // philosophy"). np.broadcast follows that engine, so any number of operands is
                // accepted. 0 operands is legal and yields a 0-d (scalar) broadcast.
                // The shape is multiiter_new_impl's PyArray_Broadcast, so a mismatch names the first pair of operands that
                // disagree, as NumPy's ValueError does (parity QA-30; the generic resolver stopped at "...to a single shape").
                shape = _ops.Length == 0
                    ? Shape.Scalar
                    : Shape.MultiIterShape(_ops.Select(o => o.Shape).ToArray());
            }

            /// <summary>
            ///     NumPy's <c>np.broadcast(*arrays)</c> over object arrays of lists (<see cref="NDListArray"/>, the
            ///     <c>np.char.split</c> results): the list arrays' shapes broadcast together, and each step's tuple holds
            ///     the operands' LISTS (a <see cref="List{T}"/> of its pieces each, NumPy's Python list).
            /// </summary>
            /// <param name="arrays">The list arrays (null is treated as none); borrowed, never disposed here.</param>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="ObjectDisposedException">An operand was disposed.</exception>
            /// <exception cref="IncorrectShapeException">The operands do not broadcast together (NumPy's text).</exception>
            /// <remarks>
            ///     The broadcast runs over each list array's int64 ids (NumPy's object pointers), so the shape rule and its
            ///     texts are the array constructor's; a step reads each id and hands back the list it names. <see cref="iters"/>
            ///     is refused for list operands (its flat iterators read numbers). A sequence parameter, not <c>params</c>: a
            ///     second <c>params</c> form would make <c>new np.Broadcast()</c> ambiguous. API-parity batch 13 (the owner:
            ///     "Also NDListArray").
            /// </remarks>
            public Broadcast(NDListArray[] arrays)
                : this(ListIds(arrays))
            {
                _lists = arrays ?? Array.Empty<NDListArray>();
            }

            /// <summary>The ids of each list array, checked in order (null and disposed operands refused first).</summary>
            /// <param name="arrays">The list arrays, or null for none.</param>
            /// <returns>Each operand's id array (the operand's own: borrowed, never disposed here).</returns>
            /// <exception cref="ArgumentNullException">An operand is null.</exception>
            /// <exception cref="ObjectDisposedException">An operand was disposed.</exception>
            private static NDArray[] ListIds(NDListArray[] arrays)
            {
                if (arrays is null)
                    return Array.Empty<NDArray>();
                var ids = new NDArray[arrays.Length];
                for (int i = 0; i < arrays.Length; i++)
                {
                    var part = arrays[i] ?? throw new ArgumentNullException(nameof(arrays), "arrays holds a null list array");
                    part.ThrowIfDisposed();
                    ids[i] = part.Ids;
                }
                return ids;
            }

            public Shape shape { get; internal set; }

            /// <summary>
            ///     The live flat position of the iterator (NumPy's <c>broadcast.index</c>): the number
            ///     of elements already consumed — 0 before iterating, incrementing by one per step, and
            ///     equal to <see cref="size"/> once exhausted. Use <see cref="reset"/> to rewind.
            /// </summary>
            public int index => _index;

            /// <summary>Operands broadcast to the result <see cref="shape"/>, built once on demand.</summary>
            private NDArray[] Views => _views ??= _ops?.Select(o => broadcast_to(o, shape)).ToArray();

            /// <summary>
            ///     Per-operand flat iterators (NumPy's <c>broadcast.iters</c>): one entry per input,
            ///     each iterating that operand broadcast to the result <see cref="shape"/> in C-order
            ///     (e.g. np.broadcast([1,2,3], [[10],[20]]).iters[0] yields 1,2,3,1,2,3). Built lazily
            ///     on first access — np.broadcast() only resolves the shape eagerly — and backed by
            ///     <see cref="np.broadcast_to(NDArray, Shape)"/> + <see cref="NDFlatIterator"/>, the
            ///     NDIter-aligned replacement for the removed NDIterator. There are <see cref="numiter"/>
            ///     entries (none for a broadcast of no operand). Unlike NumPy's flatiters these are
            ///     independent and re-enumerable; they do not share the <see cref="index"/> cursor.
            /// </summary>
            /// <exception cref="NotSupportedException">The operands are list arrays (<see cref="NDListArray"/>): their
            ///     views hold ids, which a flat iterator would read as numbers; iterate the broadcast itself.</exception>
            public NDFlatIterator[] iters
            {
                get
                {
                    // A list operand's view holds ids, and a flat iterator would read them as numbers: refused rather than
                    // handing back pointers where NumPy hands back the lists.
                    if (_lists is not null)
                        throw new NotSupportedException(
                            "iters of a broadcast over NDListArray operands: iterate the broadcast itself (each step's tuple holds the lists)");
                    return _iters ??= _ops is null
                        ? null
                        : Views.Select(v => new NDFlatIterator(v)).ToArray();
                }
                internal set => _iters = value;
            }

            public int nd
            {
                get => ndim;
            }

            public int ndim => shape.NDim;
            public long size => shape.size;

            /// <summary>
            ///     Number of operands being broadcast together (NumPy's <c>broadcast.numiter</c>,
            ///     equal to <c>len(iters)</c>). Equals the operand count passed to
            ///     <see cref="np.broadcast(NDArray[])"/> (0 for a broadcast of no operand, <c>new np.Broadcast()</c>).
            /// </summary>
            public int numiter => _ops?.Length ?? 0;

            // ---- NumPy-style single-cursor iteration (iter(b) is b) ----

            /// <summary>
            ///     Returns the broadcast itself as its iterator — matching NumPy's <c>iter(b) is b</c>,
            ///     so iteration shares the single <see cref="index"/> cursor (call <see cref="reset"/>
            ///     to iterate again).
            /// </summary>
            public IEnumerator<object[]> GetEnumerator() => this;

            IEnumerator IEnumerable.GetEnumerator() => this;

            /// <summary>The tuple of per-operand values produced by the most recent step (one per <see cref="numiter"/>).</summary>
            public object[] Current => _current;

            object IEnumerator.Current => _current;

            /// <summary>
            ///     Advances the cursor one element and reads each operand's value at that flat
            ///     (C-order) position into <see cref="Current"/>. Returns false once <see cref="index"/>
            ///     reaches <see cref="size"/>. A broadcast of no operand has one step, an empty tuple (NumPy's
            ///     <c>list(np.broadcast())</c> is <c>[()]</c>); a list operand's value is its list.
            /// </summary>
            public bool MoveNext()
            {
                if (_ops is null || _index >= size)
                    return false;

                var views = Views;
                var vals = new object[views.Length];
                for (int i = 0; i < views.Length; i++)
                {
                    // A list operand's view holds its ids: the step's value is the list the id names (NumPy's object
                    // pointer read), a new .NET list of its pieces.
                    vals[i] = _lists is null
                        ? views[i].GetAtIndex(_index)
                        : _lists[i].Heap.ToList((long)views[i].GetAtIndex(_index));
                }

                _current = vals;
                _index++;
                return true;
            }

            /// <summary>
            ///     NumPy's <c>broadcast.reset()</c>: rewind the live <see cref="index"/> cursor to 0
            ///     so the object can be iterated again.
            /// </summary>
            public void reset()
            {
                _index = 0;
                _current = null;
            }

            void IEnumerator.Reset() => reset();

            /// <summary>
            ///     Releases the broadcast views built for iteration (each is a <c>broadcast_to</c> view
            ///     over an operand's buffer — a counted alias whose ref would otherwise wait for the
            ///     finalizer). Called by <c>foreach</c> at the end of a loop; the views and the
            ///     <see cref="iters"/> over them are rebuilt lazily on the next access, so the object
            ///     stays re-enumerable (NumPy's <c>iter(b) is b</c>; call <see cref="reset"/> to rewind).
            ///     The operands themselves are the caller's and are never touched.
            /// </summary>
            public void Dispose()
            {
                var views = _views;
                _views = null;
                _iters = null;
                if (views is null)
                    return;
                foreach (var v in views)
                    v?.Dispose();
            }
        }
    }
}
