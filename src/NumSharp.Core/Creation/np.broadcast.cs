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
        ///     (NPY_MAXARGS); NumSharp imposes no cap, matching its unlimited-operand <c>NpyIter</c>.
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
        public static Broadcast broadcast(NDArray nd1, NDArray nd2)
        {
            return new Broadcast(nd1, nd2);
        }

        /// <summary>
        ///     NumPy's <c>numpy.broadcast</c> — the broadcast result of N operands, usable as an
        ///     iterator. Like NumPy, the object is its OWN iterator (<c>iter(b) is b</c>): it keeps a
        ///     single live cursor exposed as <see cref="index"/>, iterating it yields one tuple of
        ///     per-operand values per step (advancing the cursor), and <see cref="reset"/> rewinds it.
        /// </summary>
        public class Broadcast : IEnumerable<object[]>, IEnumerator<object[]>
        {
            private readonly NDArray[] _ops;
            private NpyFlatIterator[] _iters;
            private NDArray[] _views;     // operands broadcast to the result shape (lazy)
            private int _index;           // live cursor — NumPy's broadcast.index
            private object[] _current;    // values at the previous cursor position

            /// <summary>
            ///     Parameterless constructor retained for source/binary compatibility
            ///     (the public surface previously exposed an implicit one). An instance
            ///     built this way has no operands, so <see cref="iters"/> stays null
            ///     unless explicitly assigned.
            /// </summary>
            public Broadcast() { }

            internal Broadcast(NDArray op1, NDArray op2) : this(new[] { op1, op2 }) { }

            /// <summary>
            ///     Broadcast N operands together (NumPy's <c>np.broadcast(*args)</c>). The result
            ///     <see cref="shape"/> is the broadcast of every operand's shape (resolved eagerly;
            ///     iterators are built lazily on first <see cref="iters"/> access). Incompatible
            ///     shapes raise <see cref="IncorrectShapeException"/> (NumPy parity). Unlike NumPy's
            ///     fixed NPY_MAXARGS=64 cap, any number of operands is accepted — matching NumSharp's
            ///     unlimited-operand <c>NpyIter</c>.
            /// </summary>
            internal Broadcast(params NDArray[] ops)
            {
                _ops = ops ?? Array.Empty<NDArray>();

                // NUMSHARP DIVERGENCE: NumPy caps the multi-iterator at NPY_MAXARGS=64, but
                // NumSharp's NpyIter allocates all per-operand state dynamically and imposes no
                // operand cap (see NpyIter.State.cs — "UNLIMITED ... matches NumSharp's core
                // philosophy"). np.broadcast follows that engine, so any number of operands is
                // accepted. 0 operands is legal and yields a 0-d (scalar) broadcast.
                shape = _ops.Length == 0
                    ? Shape.Scalar
                    : Shape.ResolveReturnShape(_ops.Select(o => o.Shape).ToArray());
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
            ///     <see cref="np.broadcast_to(NDArray, Shape)"/> + <see cref="NpyFlatIterator"/>, the
            ///     NpyIter-aligned replacement for the removed NDIterator. There are <see cref="numiter"/>
            ///     entries (0 when constructed without operands). Unlike NumPy's flatiters these are
            ///     independent and re-enumerable; they do not share the <see cref="index"/> cursor.
            /// </summary>
            public NpyFlatIterator[] iters
            {
                get => _iters ??= _ops is null
                    ? null
                    : Views.Select(v => new NpyFlatIterator(v)).ToArray();
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
            ///     <see cref="np.broadcast(NDArray[])"/> (0 for the parameterless constructor).
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
            ///     reaches <see cref="size"/> (or when constructed without operands).
            /// </summary>
            public bool MoveNext()
            {
                if (_ops is null || _index >= size)
                    return false;

                var views = Views;
                var vals = new object[views.Length];
                for (int i = 0; i < views.Length; i++)
                    vals[i] = views[i].GetAtIndex(_index);

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

            public void Dispose() { }
        }
    }
}
