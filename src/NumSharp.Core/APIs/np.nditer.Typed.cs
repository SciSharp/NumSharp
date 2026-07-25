using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Typed, allocation-free element iteration — the unboxed counterpart of
        ///     <see cref="nditer(NDArray, string[], string[], NPTypeCode[], char, string, int[][], long[], long)"/>.
        ///     Yields <c>ref T</c> straight into the operand's memory, so reading costs a
        ///     dereference and writing goes through to the array.
        ///
        ///     <code>
        ///     // read
        ///     foreach (ref double x in np.nditer&lt;double&gt;(a))
        ///         total += x;
        ///
        ///     // write in place
        ///     foreach (ref double x in np.nditer&lt;double&gt;(a, writeable: true))
        ///         x *= 2;
        ///     </code>
        /// </summary>
        /// <typeparam name="T">
        ///     Must be EXACTLY the array's element type — no conversion or casting is performed,
        ///     because a <c>ref</c> cannot convert. A mismatch throws rather than reinterpreting
        ///     the bytes.
        /// </typeparam>
        /// <param name="op">The array to iterate over.</param>
        /// <param name="writeable">
        ///     Open the operand <c>readwrite</c> so assignments through the <c>ref</c> reach the
        ///     array. Broadcast views are read-only and are rejected, with NumPy's message.
        /// </param>
        /// <param name="order">
        ///     Iteration order: <c>'K'</c> (default, memory order — matches NumPy's
        ///     <c>np.nditer</c>), <c>'C'</c>, <c>'F'</c> or <c>'A'</c>. See
        ///     <see cref="NDRefIter{T}"/> for why the default is NOT logical C-order.
        /// </param>
        /// <remarks>
        ///     NumSharp extension: NumPy has no typed iteration, because Python has no unboxed
        ///     generics. The traversal is NumSharp's <see cref="NDIterRef"/> — the same engine
        ///     <c>np.nditer</c> drives — so every memory layout behaves identically; all that is
        ///     gone is the per-element <see cref="NDArray"/> view, which is what made the boxed
        ///     form slow.
        ///
        ///     <para>
        ///     <b>Empty arrays iterate zero times</b>, where the boxed <c>np.nditer</c> raises
        ///     <c>"Iteration of zero-sized operands is not enabled"</c> unless given NumPy's
        ///     <c>zerosize_ok</c> flag. Deliberate: throwing would force every caller to guard a
        ///     <c>foreach</c> with <c>if (a.size &gt; 0)</c>, which is not how C# collections
        ///     behave, and this is an extension rather than a parity surface.
        ///     </para>
        /// </remarks>
        public static NDRefIter<T> nditer<T>(NDArray op, bool writeable = false, char order = 'K')
            where T : unmanaged
            => new NDRefIter<T>(op, writeable, order);

        /// <summary>
        ///     Typed CHUNK iteration — hands out a <see cref="Span{T}"/> per inner loop rather than
        ///     one element at a time, so the body can be vectorized or passed to
        ///     <c>TensorPrimitives</c>. The typed analogue of NumPy's <c>external_loop</c>, except
        ///     that a <c>Span&lt;T&gt;</c> is directly consumable by .NET's vector APIs where
        ///     NumPy's chunk is another ndarray.
        ///
        ///     <code>
        ///     foreach (Span&lt;double&gt; chunk in np.nditer_chunks&lt;double&gt;(a, writeable: true))
        ///         TensorPrimitives.Multiply(chunk, 2.0, chunk);
        ///     </code>
        ///
        ///     A C-contiguous array arrives as a SINGLE chunk covering the whole array — and so do
        ///     F-contiguous, transposed and reversed views, which the iterator coalesces.
        /// </summary>
        /// <inheritdoc cref="nditer{T}(NDArray, bool, char)"/>
        public static NDChunkIter<T> nditer_chunks<T>(NDArray op, bool writeable = false, char order = 'K')
            where T : unmanaged
            => new NDChunkIter<T>(op, writeable, order);

        /// <summary>
        ///     The <c>foreach</c>-able returned by <see cref="nditer{T}(NDArray, bool, char)"/>.
        ///
        ///     <para>
        ///     <b>Why a <c>ref struct</c> enumerator.</b> C#'s <c>foreach</c> is pattern-based — it
        ///     needs only <c>GetEnumerator</c>/<c>MoveNext</c>/<c>Current</c>, no interface — so an
        ///     enumerator exposing <c>ref T Current</c> gives <c>foreach (ref T x in …)</c> with no
        ///     allocation, no boxing and no interface dispatch. Being a <c>ref struct</c> also makes
        ///     the compiler enforce what this API needs anyway: neither the enumerator nor the
        ///     <c>ref</c> it hands out can escape to a field, a lambda or an <c>async</c> frame.
        ///     </para>
        ///
        ///     <para>
        ///     <b>This type holds no unmanaged state; the ENUMERATOR does.</b> Each
        ///     <see cref="GetEnumerator"/> builds a fresh <see cref="NDIterRef"/>, which
        ///     <c>foreach</c> then disposes through the same pattern (no <c>IDisposable</c>
        ///     required). That is why this value is safe to keep and re-enumerate, and why every
        ///     pass starts from the beginning — deliberately UNLIKE the class-based
        ///     <see cref="NDIterator"/>, which is its own iterator (NumPy's <c>iter(x) is x</c>) and
        ///     therefore resumes. Returning <c>this</c> from <c>GetEnumerator</c> — the
        ///     <see cref="np.Broadcast"/> pattern — would be a use-after-free here: the first
        ///     <c>foreach</c> frees the state that the second would then walk.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Order is <c>'K'</c>, i.e. MEMORY order — not logical C-order.</b> This matches
        ///     <c>np.nditer</c> exactly (probed against NumPy 2.4.2: a reversed view
        ///     <c>a[:, ::-1]</c> of <c>arange(6).reshape(2,3)</c> yields <c>0 1 2 3 4 5</c> under
        ///     the default order in BOTH libraries, and <c>2 1 0 5 4 3</c> under <c>order='C'</c>).
        ///     It is also what lets reversed / F-contiguous / transposed views coalesce to a single
        ///     chunk. For logical order — the order <see cref="ndenumerate(NDArray)"/> uses — pass
        ///     <c>order: 'C'</c>.
        ///     </para>
        /// </summary>
        /// <typeparam name="T">The array's exact element type.</typeparam>
        public readonly struct NDRefIter<T> where T : unmanaged
        {
            private readonly NDArray _op;
            private readonly bool _writeable;
            private readonly NPY_ORDER _order;

            internal NDRefIter(NDArray op, bool writeable, char order)
            {
                TypedIterHelpers.Validate<T>(op, writeable, nameof(op));
                _op = op;
                _writeable = writeable;
                _order = TypedIterHelpers.ParseOrder(order);
            }

            /// <summary>Builds a fresh iterator; <c>foreach</c> disposes it for you.</summary>
            public Enumerator GetEnumerator() => new Enumerator(_op, _writeable, _order);

            /// <summary>
            ///     The cursor. Walks the inner loop with plain pointer arithmetic and only touches
            ///     the iterator at a chunk boundary, so a contiguous array costs one iterator call
            ///     for the whole walk.
            /// </summary>
            public unsafe ref struct Enumerator
            {
                private NDIterRef _it;
                private byte* _current;
                private byte* _next;
                private long _remaining;   // elements left in this chunk, excluding _current
                private long _stride;      // BYTE stride of the inner loop
                private bool _live;

                internal Enumerator(NDArray op, bool writeable, NPY_ORDER order)
                {
                    _it = TypedIterHelpers.Build(op, writeable, order);
                    _current = null;
                    _next = null;
                    _remaining = 0;
                    _stride = 0;
                    _live = true;
                }

                /// <summary>The current element, BY REFERENCE — assign to it to write through.</summary>
                public ref T Current => ref *(T*)_current;

                /// <summary>Advance one element.</summary>
                public bool MoveNext()
                {
                    if (_remaining > 0)
                    {
                        _remaining--;
                        _current = _next;
                        _next += _stride;
                        return true;
                    }

                    return NextChunk();
                }

                private bool NextChunk()
                {
                    while (_live)
                    {
                        if (_it.Finished || _it.IterSize == 0)
                        {
                            _live = false;
                            return false;
                        }

                        TypedIterHelpers.ReadInnerLoop(ref _it, out long count, out long elementStride);
                        _stride = elementStride * sizeof(T);
                        byte* start = (byte*)_it.GetDataPtrArray()[0];

                        // Safe to advance before consuming: this iterator is never buffered, so the
                        // pointers are absolute into the operand and the captured chunk stays valid.
                        if (!_it.Iternext())
                            _live = false;

                        if (count > 0)
                        {
                            _current = start;
                            _next = start + _stride;
                            _remaining = count - 1;
                            return true;
                        }
                    }

                    return false;
                }

                /// <summary>
                ///     Frees the unmanaged iterator state. <c>foreach</c> calls this; hand-driven
                ///     <see cref="MoveNext"/> loops must call it themselves.
                /// </summary>
                public void Dispose() => _it.Dispose();
            }
        }

        /// <summary>
        ///     The <c>foreach</c>-able returned by
        ///     <see cref="nditer_chunks{T}(NDArray, bool, char)"/>. See <see cref="NDRefIter{T}"/>
        ///     for the <c>ref struct</c>, disposal, re-enumeration and iteration-order rationale,
        ///     which apply identically.
        ///
        ///     <para>
        ///     The yielded <see cref="Span{T}"/> points straight at the operand, so writes go
        ///     through and the span is invalidated by the next step.
        ///     </para>
        /// </summary>
        /// <typeparam name="T">The array's exact element type.</typeparam>
        public readonly struct NDChunkIter<T> where T : unmanaged
        {
            private readonly NDArray _op;
            private readonly bool _writeable;
            private readonly NPY_ORDER _order;

            internal NDChunkIter(NDArray op, bool writeable, char order)
            {
                TypedIterHelpers.Validate<T>(op, writeable, nameof(op));
                _op = op;
                _writeable = writeable;
                _order = TypedIterHelpers.ParseOrder(order);
            }

            /// <summary>Builds a fresh iterator; <c>foreach</c> disposes it for you.</summary>
            /// <exception cref="NotSupportedException">
            ///     The layout's inner loop is not unit-stride — a stepped view such as
            ///     <c>a[":, ::2"]</c>. A <see cref="Span{T}"/> is contiguous by definition and
            ///     cannot describe it. The single-operand inner stride is fixed for the whole
            ///     iteration, so this is detected up front and never surfaces mid-loop. Use
            ///     <see cref="nditer{T}(NDArray, bool, char)"/>, which handles any stride, or
            ///     iterate a <c>.copy()</c>.
            /// </exception>
            public Enumerator GetEnumerator() => new Enumerator(_op, _writeable, _order);

            /// <summary>The cursor — one <see cref="Span{T}"/> per inner loop.</summary>
            public unsafe ref struct Enumerator
            {
                private NDIterRef _it;
                private Span<T> _current;
                private bool _live;

                internal Enumerator(NDArray op, bool writeable, NPY_ORDER order)
                {
                    _it = TypedIterHelpers.Build(op, writeable, order);
                    _current = default;
                    _live = true;

                    if (!_it.Finished && _it.IterSize > 1)
                    {
                        TypedIterHelpers.ReadInnerLoop(ref _it, out _, out long elementStride);
                        if (elementStride != 1)
                        {
                            _it.Dispose();
                            _live = false;
                            throw new NotSupportedException(
                                $"np.nditer_chunks<{typeof(T).Name}> requires a unit-stride inner loop, but this " +
                                $"layout iterates with element stride {elementStride} (a stepped view such as " +
                                $"a[\":, ::2\"]). A Span<T> is contiguous by definition and cannot describe it. " +
                                $"Use np.nditer<{typeof(T).Name}>(...), which handles any stride, or iterate a .copy().");
                        }
                    }
                }

                /// <summary>The current inner loop, as a span over the operand's own memory.</summary>
                public Span<T> Current => _current;

                /// <summary>Advance one chunk.</summary>
                public bool MoveNext()
                {
                    if (!_live || _it.Finished || _it.IterSize == 0)
                    {
                        _live = false;
                        return false;
                    }

                    TypedIterHelpers.ReadInnerLoop(ref _it, out long count, out _);
                    _current = new Span<T>(_it.GetDataPtrArray()[0], checked((int)count));

                    if (!_it.Iternext())
                        _live = false;

                    return true;
                }

                /// <summary>
                ///     Frees the unmanaged iterator state. <c>foreach</c> calls this; hand-driven
                ///     <see cref="MoveNext"/> loops must call it themselves.
                /// </summary>
                public void Dispose() => _it.Dispose();
            }
        }
    }
}

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     Shared plumbing for the typed iterators (<c>np.nditer&lt;T&gt;</c> /
    ///     <c>np.nditer_chunks&lt;T&gt;</c>).
    /// </summary>
    internal static unsafe class TypedIterHelpers
    {
        /// <summary>
        ///     Guards applied before an iterator is built: exact dtype match (a <c>ref</c> cannot
        ///     convert, so a mismatch would silently REINTERPRET the bytes) and writeability.
        /// </summary>
        internal static void Validate<T>(NDArray op, bool writeable, string paramName) where T : unmanaged
        {
            if (op is null)
                throw new ArgumentNullException(paramName);

            if (op.dtype != typeof(T))
                throw new ArgumentException(
                    $"np.nditer<{typeof(T).Name}> called on a {op.dtype.Name} array. Typed iteration hands out " +
                    $"`ref {typeof(T).Name}` straight into the array's memory, so no conversion is possible — " +
                    $"reinterpreting the bytes would return garbage. Cast first: arr.astype(typeof({typeof(T).Name})).",
                    paramName);

            // NumPy's verbatim text for np.nditer(broadcast_view, op_flags=['readwrite']).
            if (writeable && !op.Shape.IsWriteable)
                throw new ArgumentException(
                    "operand array with iterator write flag set is read-only", paramName);
        }

        internal static NPY_ORDER ParseOrder(char order)
        {
            switch (order)
            {
                case 'C': return NPY_ORDER.NPY_CORDER;
                case 'F': return NPY_ORDER.NPY_FORTRANORDER;
                case 'A': return NPY_ORDER.NPY_ANYORDER;
                case 'K': return NPY_ORDER.NPY_KEEPORDER;
                default:
                    throw new ArgumentException($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }

        /// <summary>
        ///     The single-operand EXTERNAL_LOOP iterator both typed walks run on. Never buffered:
        ///     a <c>ref</c>/<c>Span</c> into a buffer would dangle the moment the next fill lands,
        ///     and unbuffered pointers are absolute into the operand.
        /// </summary>
        internal static NDIterRef Build(NDArray op, bool writeable, NPY_ORDER order)
            => NDIterRef.MultiNew(
                1, new[] {op},
                NDIterGlobalFlags.EXTERNAL_LOOP,
                order,
                NPY_CASTING.NPY_SAFE_CASTING,
                new[] {writeable ? NDIterPerOpFlags.READWRITE : NDIterPerOpFlags.READONLY});

        /// <summary>
        ///     The current inner loop's element count and ELEMENT stride.
        ///
        ///     <para>
        ///     Two traps in the public <see cref="NDIterRef"/> surface are absorbed here.
        ///     <see cref="NDIterRef.GetInnerLoopSizePtr"/> dereferences <c>Shape[NDim - 1]</c>,
        ///     which on a 0-d operand is <c>Shape[-1]</c> — an access violation, not an exception —
        ///     so 0-d is answered directly. And <see cref="NDIterRef.GetInnerStrideArray"/> reports
        ///     ELEMENT strides while the kernel contract (<c>ForEach</c>/<c>ExecuteGeneric</c>) uses
        ///     BYTE strides; this returns elements and the callers scale by <c>sizeof(T)</c>.
        ///     </para>
        /// </summary>
        internal static void ReadInnerLoop(ref NDIterRef it, out long count, out long elementStride)
        {
            if (it.NDim == 0)
            {
                count = 1;
                elementStride = 0;
                return;
            }

            count = *it.GetInnerLoopSizePtr();
            elementStride = it.GetInnerLoopElementStride(0);
        }
    }
}
