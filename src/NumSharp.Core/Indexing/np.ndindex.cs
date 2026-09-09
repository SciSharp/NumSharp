using System;
using System.Collections;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     An N-dimensional iterator object to index arrays.
        ///
        ///     Given the shape of an array, an <see cref="NDIndex"/> instance iterates over the
        ///     N-dimensional index of the array. At each iteration an index array is returned;
        ///     the last dimension is iterated over first.
        /// </summary>
        /// <param name="shape">
        ///     The size of each dimension, passed as individual parameters (<c>np.ndindex(3, 2, 1)</c>)
        ///     or as a single array (<c>np.ndindex(arr.shape)</c>). Both spellings bind to this one
        ///     <c>params</c> overload, so NumPy's "ints, or a single tuple of ints" rule holds for free.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndindex.html</remarks>
        public static NDIndex ndindex(params long[] shape) => new NDIndex(shape);

        /// <summary>
        ///     <c>int[]</c> overload — <c>long[]</c> is the house shape type, but an existing
        ///     <c>int[]</c> does not convert to it by array covariance, so it gets its own entry.
        ///     Deliberately NOT <c>params</c>: individual <c>int</c> arguments already widen into
        ///     the <c>params long[]</c> form, and a second params overload would make the
        ///     zero-argument call <c>np.ndindex()</c> ambiguous.
        /// </summary>
        public static NDIndex ndindex(int[] shape) => new NDIndex(NDIndex.Widen(shape));

        /// <summary>
        ///     NumPy's <c>numpy.ndindex</c> — a C-order odometer over the index space of a shape,
        ///     yielding one index array per step with the LAST dimension varying fastest.
        ///
        ///     <code>
        ///     // numpy: list(np.ndindex(3, 2)) -> [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
        ///     foreach (var idx in np.ndindex(3, 2)) { /* idx = {0,0}, {0,1}, {1,0}, ... */ }
        ///     </code>
        ///
        ///     Like NumPy the object is its OWN iterator (<c>iter(i) is i</c>): it keeps a single
        ///     live cursor, so a second enumeration resumes where the first stopped rather than
        ///     restarting — the same contract <see cref="np.Broadcast"/> follows.
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.4.2's <c>numpy/lib/_index_tricks_impl.py</c>, whose body is
        ///     <c>product(*map(range, shape))</c> — a plain odometer. NumPy itself moved
        ///     <c>ndindex</c> OFF <c>nditer</c> onto <c>itertools.product</c>, so this class
        ///     deliberately allocates no iterator state: there are no operands to walk, only a
        ///     counter. (<see cref="np.nditer"/> is the API that drives NumSharp's <c>NDIterRef</c>.)
        ///
        ///     Indices are <c>long</c> — NumPy's <c>intp</c>, the house type for index-valued
        ///     output. Each step yields a FRESH array, never a recycled buffer, so materializing
        ///     the enumeration (NumPy's <c>list(np.ndindex(...))</c>) gives distinct indices.
        /// </remarks>
        public class NDIndex : IEnumerable<long[]>, IEnumerator<long[]>
        {
            private readonly long[] _shape;
            private long[] _current;
            private bool _started;
            private bool _finished;

            internal static long[] Widen(int[] shape)
            {
                if (shape == null)
                    return Array.Empty<long>();

                var ret = new long[shape.Length];
                for (int i = 0; i < shape.Length; i++)
                    ret[i] = shape[i];

                return ret;
            }

            /// <param name="shape">Dimension sizes; may be null/empty (a 0-d index space).</param>
            /// <exception cref="ArgumentException">
            ///     A dimension is negative — NumPy's
            ///     <c>ValueError("negative dimensions are not allowed")</c>.
            /// </exception>
            internal NDIndex(long[] shape)
            {
                shape ??= Array.Empty<long>();

                // NumPy: `if min(shape, default=0) < 0: raise ValueError(...)`. Validated at
                // CONSTRUCTION, before a single index is produced (probed: np.ndindex(2, -3)
                // raises without ever being iterated).
                for (int i = 0; i < shape.Length; i++)
                {
                    if (shape[i] < 0)
                        throw new ArgumentException("negative dimensions are not allowed");
                }

                _shape = (long[])shape.Clone();
            }

            /// <summary>
            ///     The index array produced by the most recent step. Its length is the number of
            ///     dimensions — 0 for the 0-d index space, whose single step yields an empty array
            ///     (NumPy's <c>()</c>).
            /// </summary>
            public long[] Current => _current;

            object IEnumerator.Current => _current;

            /// <summary>
            ///     Returns the odometer itself as its iterator — matching NumPy's <c>iter(i) is i</c>,
            ///     so enumeration shares the single live cursor.
            /// </summary>
            public IEnumerator<long[]> GetEnumerator() => this;

            IEnumerator IEnumerable.GetEnumerator() => this;

            /// <summary>
            ///     Advances the odometer one step, incrementing the LAST axis first and carrying
            ///     leftward (C-order). Returns false once every index has been produced.
            /// </summary>
            public bool MoveNext()
            {
                if (_finished)
                    return false;

                int ndim = _shape.Length;

                if (!_started)
                {
                    _started = true;

                    // A zero-length dimension makes the index space empty — product() over an
                    // empty range yields nothing (np.ndindex(0, 3) -> []).
                    for (int i = 0; i < ndim; i++)
                    {
                        if (_shape[i] == 0)
                        {
                            _finished = true;
                            _current = null;
                            return false;
                        }
                    }

                    // ndim == 0 falls through here: product() with no iterables yields exactly
                    // one empty tuple, so np.ndindex() and np.ndindex(()) both give [()].
                    _current = new long[ndim];
                    return true;
                }

                // Odometer carry, last axis fastest. Work on a copy so already-yielded index
                // arrays stay valid — NumPy hands out a fresh tuple per step.
                var next = (long[])_current.Clone();
                for (int axis = ndim - 1; axis >= 0; axis--)
                {
                    if (++next[axis] < _shape[axis])
                    {
                        _current = next;
                        return true;
                    }

                    next[axis] = 0;
                }

                // Carried past axis 0 — or ndim == 0, whose single step is already spent.
                _finished = true;
                _current = null;
                return false;
            }

            // NumPy's ndindex exposes no reset (unlike broadcast.reset()), so this stays an
            // explicit interface implementation rather than public surface.
            void IEnumerator.Reset()
            {
                _started = false;
                _finished = false;
                _current = null;
            }

            public void Dispose() { }

            /// <summary>
            ///     The allocation-free form of this odometer — the high-performance counterpart.
            ///     Where iterating this object yields a FRESH <c>long[]</c> per step,
            ///     <see cref="NDIndexSpans"/> yields a <see cref="ReadOnlySpan{T}"/> over a coordinate
            ///     buffer REUSED each step, so a walk of the whole index space allocates nothing:
            ///
            ///     <code>
            ///     foreach (var idx in np.ndindex(3, 2).AsSpans())   // idx is ReadOnlySpan&lt;long&gt;
            ///         Sum += a.GetValue(idx);                       // idx valid until the next step
            ///     </code>
            ///
            ///     <para>
            ///     <c>ndindex</c> walks an index SPACE, not an array — there are no element values, so
            ///     there is no <c>ref T</c> form (unlike <see cref="np.flat{T}(NDArray, bool)"/> or
            ///     <see cref="NDEnumerate{T}.AsRef(bool)"/>, which reference an array's elements). This
            ///     span form is its non-boxed, non-allocating equivalent. Copy the span
            ///     (<c>.ToArray()</c>) to keep an index past the current iteration.
            ///     </para>
            /// </summary>
            public NDIndexSpans AsSpans() => new NDIndexSpans(_shape);
        }

        /// <summary>
        ///     The <c>foreach</c>-able returned by <see cref="NDIndex.AsSpans"/> — the allocation-free
        ///     odometer. Yields the current multi-index as a <see cref="ReadOnlySpan{T}"/> over a
        ///     buffer REUSED each step (C-order, last axis fastest — identical order to
        ///     <see cref="NDIndex"/>), so an entire index-space walk allocates nothing per step.
        ///
        ///     <para>
        ///     Re-enumeration RESTARTS (each <c>foreach</c> builds a fresh enumerator with its own
        ///     buffer) — deliberately unlike <see cref="NDIndex"/>, which is its own iterator (NumPy's
        ///     <c>iter(i) is i</c>) and resumes. Copy the span (<c>.ToArray()</c>) to keep an index
        ///     past the current iteration; storing the span itself, or the whole enumeration, is a
        ///     use-after-overwrite.
        ///     </para>
        /// </summary>
        public readonly struct NDIndexSpans
        {
            private readonly long[] _shape;

            internal NDIndexSpans(long[] shape) => _shape = shape ?? Array.Empty<long>();

            /// <summary>Builds a fresh odometer with its own reused coordinate buffer.</summary>
            public Enumerator GetEnumerator() => new Enumerator(_shape);

            /// <summary>The cursor — one <see cref="ReadOnlySpan{T}"/> (over a reused buffer) per step.</summary>
            public ref struct Enumerator
            {
                private readonly long[] _shape;
                private readonly long[] _coords;   // reused; Current spans it
                private bool _started;
                private bool _finished;

                internal Enumerator(long[] shape)
                {
                    _shape = shape;
                    _coords = new long[shape.Length];
                    _started = false;
                    _finished = false;
                }

                /// <summary>The current index — a view over the reused buffer, valid until the next step.</summary>
                public ReadOnlySpan<long> Current => _coords;

                /// <summary>Advance the odometer (last axis fastest, C-order), reusing the buffer.</summary>
                public bool MoveNext()
                {
                    if (_finished)
                        return false;

                    int ndim = _shape.Length;

                    if (!_started)
                    {
                        _started = true;
                        // A zero-length dimension makes the space empty (product over an empty range).
                        for (int i = 0; i < ndim; i++)
                        {
                            if (_shape[i] == 0)
                            {
                                _finished = true;
                                return false;
                            }
                        }

                        // ndim == 0 falls through: one step, an empty index (NumPy's ()).
                        // _coords is already all-zero from allocation.
                        return true;
                    }

                    for (int axis = ndim - 1; axis >= 0; axis--)
                    {
                        if (++_coords[axis] < _shape[axis])
                            return true;
                        _coords[axis] = 0;
                    }

                    // Carried past axis 0 — or ndim == 0, whose single step is already spent.
                    _finished = true;
                    return false;
                }

                /// <summary>No unmanaged state; present so <c>foreach</c>'s dispose pattern is satisfied.</summary>
                public void Dispose() { }
            }
        }
    }
}
