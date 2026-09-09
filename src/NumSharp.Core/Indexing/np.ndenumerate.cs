using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Multidimensional index iterator — returns an iterator yielding pairs of array
        ///     coordinates and values.
        /// </summary>
        /// <param name="arr">
        ///     Input array. Anything implicitly convertible to <see cref="NDArray"/> works
        ///     (<c>new[,] {{1, 2}, {3, 4}}</c>, a scalar, …), matching NumPy's <c>np.asarray(arr)</c>
        ///     coercion of array_like input.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndenumerate.html</remarks>
        public static NDEnumerate ndenumerate(NDArray arr) => new NDEnumerate(arr);

        /// <summary>
        ///     Typed form of <see cref="ndenumerate(NDArray)"/> — yields <typeparamref name="T"/>
        ///     instead of a boxed <c>object</c>. NumSharp extension (NumPy has no typed variant,
        ///     because Python has no unboxed generics); prefer it in hot loops, where the boxing
        ///     of the untyped form dominates the walk.
        /// </summary>
        /// <typeparam name="T">Must be the array's element type — no conversion is performed.</typeparam>
        public static NDEnumerate<T> ndenumerate<T>(NDArray arr) where T : unmanaged => new NDEnumerate<T>(arr);

        /// <summary>
        ///     NumPy's <c>numpy.ndenumerate</c> — walks an array in C-order, yielding
        ///     <c>(index, value)</c> for every element.
        ///
        ///     <code>
        ///     // numpy: list(np.ndenumerate(np.array([[1, 2], [3, 4]])))
        ///     //     -> [((0,0), 1), ((0,1), 2), ((1,0), 3), ((1,1), 4)]
        ///     foreach (var (index, value) in np.ndenumerate(a)) { … }
        ///     </code>
        ///
        ///     Like NumPy the object is its OWN iterator (<c>iter(e) is e</c>): it keeps a single
        ///     live cursor, so a second enumeration resumes where the first stopped.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.4.2's <c>numpy/lib/_index_tricks_impl.py</c>, whose body is
        ///     <c>self.iter = np.asarray(arr).flat</c> and <c>return self.iter.coords, next(self.iter)</c>
        ///     — a <c>flatiter</c> walk, i.e. a LOGICAL C-order traversal that honours the array's
        ///     own strides. So the enumeration order is always C-order regardless of layout
        ///     (probed: an F-contiguous array enumerates (0,0), (0,1), (1,0), (1,1) — not its
        ///     memory order), reversed and broadcast views read through their strides, and a 0-d
        ///     array yields exactly one pair with an EMPTY index (NumPy's <c>()</c>).
        ///
        ///     NumSharp's <c>NDArray.flat</c> is a raveled <see cref="NDArray"/> rather than a
        ///     <c>flatiter</c> object (it has no <c>coords</c> cursor), so the coordinates come
        ///     from an odometer advanced in lockstep with the flat position — which is precisely
        ///     what <c>flatiter.coords</c> is. Indices are <c>long</c> (NumPy's <c>intp</c>), and
        ///     each step yields a FRESH index array, never a recycled buffer.
        /// </remarks>
        [NDBorrowed] // a cursor over the caller's array; owns no NDArray and no unmanaged state
        public class NDEnumerate : IEnumerable<(long[] index, object value)>, IEnumerator<(long[] index, object value)>
        {
            private readonly NDArray _arr;
            private readonly NDIndexWalker _walker;

            internal NDEnumerate(NDArray arr)
            {
                _arr = arr ?? throw new ArgumentNullException(nameof(arr));
                _walker = new NDIndexWalker(arr.Shape);
            }

            public (long[] index, object value) Current { get; private set; }

            object IEnumerator.Current => Current;

            /// <summary>
            ///     Returns the enumerator itself — matching NumPy's <c>iter(e) is e</c>, so
            ///     enumeration shares the single live cursor.
            /// </summary>
            public IEnumerator<(long[] index, object value)> GetEnumerator() => this;

            IEnumerator IEnumerable.GetEnumerator() => this;

            public bool MoveNext()
            {
                if (!_walker.MoveNext())
                {
                    Current = default;
                    return false;
                }

                // GetAtIndex maps a LOGICAL C-order position through the shape's strides
                // (Shape.TransformOffset), so sliced / transposed / reversed / broadcast views
                // all read correctly — the flatiter contract.
                Current = (_walker.Coords, _arr.GetAtIndex(_walker.Position));
                return true;
            }

            void IEnumerator.Reset() => _walker.Reset();

            public void Dispose() { }
        }

        /// <summary>
        ///     Typed <see cref="NDEnumerate"/> — identical traversal, but reads elements as
        ///     <typeparamref name="T"/> without boxing. See <see cref="ndenumerate{T}(NDArray)"/>.
        /// </summary>
        [NDBorrowed] // a cursor over the caller's array; owns no NDArray and no unmanaged state
        public class NDEnumerate<T> : IEnumerable<(long[] index, T value)>, IEnumerator<(long[] index, T value)>
            where T : unmanaged
        {
            private readonly NDArray _arr;
            private readonly NDIndexWalker _walker;

            internal NDEnumerate(NDArray arr)
            {
                _arr = arr ?? throw new ArgumentNullException(nameof(arr));

                if (_arr.dtype != typeof(T))
                    throw new ArgumentException(
                        $"ndenumerate<{typeof(T).Name}> called on a {_arr.dtype.type.Name} array; " +
                        $"cast it first (e.g. arr.astype(typeof({typeof(T).Name}))).", nameof(arr));

                _walker = new NDIndexWalker(arr.Shape);
            }

            public (long[] index, T value) Current { get; private set; }

            object IEnumerator.Current => Current;

            public IEnumerator<(long[] index, T value)> GetEnumerator() => this;

            IEnumerator IEnumerable.GetEnumerator() => this;

            public bool MoveNext()
            {
                if (!_walker.MoveNext())
                {
                    Current = default;
                    return false;
                }

                Current = (_walker.Coords, _arr.GetAtIndex<T>(_walker.Position));
                return true;
            }

            void IEnumerator.Reset() => _walker.Reset();

            public void Dispose() { }

            /// <summary>
            ///     The allocation-free, by-<c>ref T</c> form of this walk — the high-performance
            ///     counterpart. Where iterating this object hands out a <c>(long[] index, T value)</c>
            ///     tuple per step (T is unboxed, but the value is a COPY and each step allocates a
            ///     fresh <c>long[]</c> for the index), <see cref="NDEnumerateRef{T}"/> hands out
            ///     <c>ref T</c> straight into the array's memory plus a <see cref="ReadOnlySpan{T}"/>
            ///     over a REUSED coordinate buffer — no per-element allocation, and write-through:
            ///
            ///     <code>
            ///     foreach (var e in np.ndenumerate&lt;double&gt;(a).AsRef(writeable: true))
            ///     {
            ///         e.Value += e.Index[0];    // ref double — writes through to the array
            ///     }
            ///     </code>
            ///
            ///     Same logical C-order as the boxed/tuple walks (the value comes from the same
            ///     <see cref="NDIterRef"/> C-order cursor <see cref="np.flat{T}(NDArray, bool)"/>
            ///     uses, so the k-th step is the element at C-order position k, whatever the layout).
            ///     <c>Index</c> is a span over a buffer reused each step — copy it (<c>e.Index.ToArray()</c>)
            ///     if you need to keep it past the current iteration, exactly as with a
            ///     <c>Span&lt;T&gt;</c> chunk.
            /// </summary>
            /// <param name="writeable">
            ///     Open the array <c>readwrite</c> so assignments through <c>e.Value</c> reach it; a
            ///     read-only broadcast view is refused with NumPy's message. Defaults to read.
            /// </param>
            public NDEnumerateRef<T> AsRef(bool writeable = false) => new NDEnumerateRef<T>(_arr, writeable);
        }

        /// <summary>
        ///     The <c>foreach</c>-able returned by <see cref="NDEnumerate{T}.AsRef(bool)"/> — the
        ///     typed, allocation-free, by-<c>ref T</c> ndenumerate. Yields, per element, a
        ///     <see cref="Entry"/> carrying the coordinate (a <see cref="ReadOnlySpan{T}"/> over a
        ///     reused buffer) and the value BY REFERENCE, in logical C-order (matching
        ///     <see cref="ndenumerate(NDArray)"/>), write-through for every memory layout.
        ///
        ///     <para>
        ///     It composes two things that are already correct: the value <c>ref</c> comes from
        ///     <see cref="NDRefIter{T}.Enumerator"/> pinned to C-order (the same engine
        ///     <see cref="np.flat{T}(NDArray, bool)"/> drives), and the coordinate comes from a
        ///     C-order odometer advanced in lockstep — since a C-order walk visits logical positions
        ///     <c>0, 1, …, size-1</c>, the k-th value and the k-th unravelled coordinate always line
        ///     up. Everything documented on <see cref="NDRefIter{T}"/> about the <c>ref struct</c>
        ///     enumerator, disposal under <c>foreach</c>, and restart-on-re-enumeration applies.
        ///     </para>
        /// </summary>
        /// <typeparam name="T">The array's exact element type (a <c>ref</c> cannot convert).</typeparam>
        [NDBorrowed] // a cursor over the caller's array; each GetEnumerator builds (and foreach disposes) its own NDIterRef
        public readonly struct NDEnumerateRef<T> where T : unmanaged
        {
            private readonly NDArray _arr;
            private readonly bool _writeable;

            internal NDEnumerateRef(NDArray arr, bool writeable)
            {
                TypedIterHelpers.Validate<T>(arr, writeable, nameof(arr));
                _arr = arr;
                _writeable = writeable;
            }

            /// <summary>Builds a fresh C-order iterator; <c>foreach</c> disposes it for you.</summary>
            public Enumerator GetEnumerator() => new Enumerator(_arr, _writeable);

            /// <summary>One enumerated element: its coordinate and its value by reference.</summary>
            public ref struct Entry
            {
                private readonly long[] _coords;
                private readonly ref T _value;

                internal Entry(long[] coords, ref T value)
                {
                    _coords = coords;
                    _value = ref value;
                }

                /// <summary>
                ///     The current element's multi-index, in C-order. A view over a buffer REUSED each
                ///     step — copy it (<c>.ToArray()</c>) to keep it past the current iteration.
                /// </summary>
                public ReadOnlySpan<long> Index => _coords;

                /// <summary>The current element BY REFERENCE — assign to it (e.g. <c>e.Value = x</c>) to write through.</summary>
                public ref T Value => ref _value;
            }

            /// <summary>The cursor — pairs the C-order value <c>ref</c> with a lockstep coordinate odometer.</summary>
            public ref struct Enumerator
            {
                private NDRefIter<T>.Enumerator _inner;   // the C-order ref walk (never buffered)
                private readonly long[] _coords;          // reused odometer buffer (Index spans it)
                private readonly long[] _dims;
                private bool _started;

                internal Enumerator(NDArray arr, bool writeable)
                {
                    _inner = new NDRefIter<T>.Enumerator(arr, writeable, NPY_ORDER.NPY_CORDER);
                    int ndim = arr.ndim;
                    _coords = new long[ndim];
                    _dims = new long[ndim];
                    var shp = arr.shape;
                    for (int i = 0; i < ndim; i++)
                        _dims[i] = shp[i];
                    _started = false;
                }

                /// <summary>The current (coordinate, ref value) pair.</summary>
                public Entry Current => new Entry(_coords, ref _inner.Current);

                /// <summary>Advance one element, moving both the value ref and the coordinate odometer.</summary>
                public bool MoveNext()
                {
                    if (!_inner.MoveNext())
                        return false;

                    // The first element is C-order position 0 -> coords already all-zero. Every later
                    // step increments the last axis and carries left (C-order), REUSING _coords so no
                    // per-step allocation occurs. _dims holds no zero here (an empty array never enters
                    // this branch — _inner.MoveNext() returned false), and a 0-d array's empty loop
                    // simply leaves the empty coords in place.
                    if (_started)
                    {
                        for (int axis = _dims.Length - 1; axis >= 0; axis--)
                        {
                            if (++_coords[axis] < _dims[axis])
                                break;
                            _coords[axis] = 0;
                        }
                    }
                    else
                    {
                        _started = true;
                    }

                    return true;
                }

                /// <summary>Frees the unmanaged iterator state (called by <c>foreach</c>).</summary>
                public void Dispose() => _inner.Dispose();
            }
        }

        /// <summary>
        ///     The shared C-order cursor behind <see cref="NDEnumerate"/> — NumPy's
        ///     <c>flatiter</c> pair of <c>index</c> (flat position) and <c>coords</c> (multi-index),
        ///     advanced together so the coordinates never have to be recomputed by division.
        /// </summary>
        private sealed class NDIndexWalker
        {
            private readonly long[] _dims;
            private readonly long _size;
            private long[] _coords;
            private long _position = -1;

            internal NDIndexWalker(Shape shape)
            {
                _dims = shape.Dimensions ?? Array.Empty<long>();
                // A 0-d array has size 1 and an EMPTY coordinate array — one step, index ().
                _size = shape.size;
            }

            /// <summary>Flat C-order position of the current element.</summary>
            internal long Position => _position;

            /// <summary>Multi-index of the current element (a fresh array per step).</summary>
            internal long[] Coords { get; private set; }

            internal bool MoveNext()
            {
                if (_position + 1 >= _size)
                {
                    _position = _size;
                    Coords = null;
                    return false;
                }

                if (++_position == 0)
                {
                    _coords = new long[_dims.Length];
                }
                else
                {
                    // Odometer carry, last axis fastest. _dims can't contain a 0 here — that
                    // would have made _size 0 and stopped us above.
                    for (int axis = _dims.Length - 1; axis >= 0; axis--)
                    {
                        if (++_coords[axis] < _dims[axis])
                            break;

                        _coords[axis] = 0;
                    }
                }

                // Hand out a copy: NumPy's flatiter.coords is a fresh tuple per step, so
                // materializing the enumeration must not alias one mutating buffer.
                Coords = (long[])_coords.Clone();
                return true;
            }

            internal void Reset()
            {
                _position = -1;
                _coords = null;
                Coords = null;
            }
        }
    }
}
