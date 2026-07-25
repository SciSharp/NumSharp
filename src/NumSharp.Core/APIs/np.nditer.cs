using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Efficient multi-dimensional iterator object to iterate over arrays.
        /// </summary>
        /// <param name="op">The array to iterate over.</param>
        /// <param name="flags">
        ///     Flags controlling iterator behaviour: <c>buffered</c>, <c>c_index</c>, <c>f_index</c>,
        ///     <c>multi_index</c>, <c>common_dtype</c>, <c>copy_if_overlap</c>, <c>delay_bufalloc</c>,
        ///     <c>external_loop</c>, <c>grow_inner</c> (a.k.a. <c>growinner</c>), <c>ranged</c>,
        ///     <c>refs_ok</c>, <c>reduce_ok</c>, <c>zerosize_ok</c>.
        /// </param>
        /// <param name="op_flags">
        ///     Per-operand flags: <c>readonly</c> (default), <c>readwrite</c>, <c>writeonly</c>,
        ///     <c>allocate</c>, <c>no_broadcast</c>, <c>contig</c>, <c>aligned</c>, <c>nbo</c>,
        ///     <c>copy</c>, <c>updateifcopy</c>, <c>no_subtype</c>, <c>arraymask</c>,
        ///     <c>writemasked</c>, <c>overlap_assume_elementwise</c>, <c>virtual</c>.
        /// </param>
        /// <param name="op_dtypes">The required data type(s) of the operands.</param>
        /// <param name="order">Iteration order: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (default).</param>
        /// <param name="casting">
        ///     Casting rule when making a copy or buffering: <c>"no"</c>, <c>"equiv"</c>,
        ///     <c>"safe"</c> (default), <c>"same_kind"</c>, <c>"unsafe"</c>.
        /// </param>
        /// <param name="op_axes">Per-operand list of axes, mapping iterator dimensions to operand dimensions (-1 = newaxis).</param>
        /// <param name="itershape">The desired shape of the iterator.</param>
        /// <param name="buffersize">Buffer size to use when buffering is enabled; 0 selects the default.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nditer.html</remarks>
        public static NDIterator nditer(
            NDArray op,
            string[] flags = null,
            string[] op_flags = null,
            NPTypeCode[] op_dtypes = null,
            char order = 'K',
            string casting = "safe",
            int[][] op_axes = null,
            long[] itershape = null,
            long buffersize = 0)
            => new NDIterator(
                new[] {op},
                flags,
                op_flags == null ? null : new[] {op_flags},
                op_dtypes, order, casting, op_axes, itershape, buffersize);

        /// <summary>
        ///     Multi-operand form — NumPy's <c>np.nditer([a, b, …])</c>. A null entry in
        ///     <paramref name="op"/> is an output slot to be ALLOCATED by the iterator (NumPy's
        ///     <c>None</c>), which then defaults to <c>writeonly, allocate</c>.
        /// </summary>
        /// <param name="op_flags">
        ///     One flag list per operand. A SINGLE inner list is broadcast to every operand —
        ///     NumPy's "flat list of strings applies to all operands" convenience.
        /// </param>
        /// <inheritdoc cref="nditer(NDArray, string[], string[], NPTypeCode[], char, string, int[][], long[], long)"/>
        public static NDIterator nditer(
            NDArray[] op,
            string[] flags = null,
            string[][] op_flags = null,
            NPTypeCode[] op_dtypes = null,
            char order = 'K',
            string casting = "safe",
            int[][] op_axes = null,
            long[] itershape = null,
            long buffersize = 0)
            => new NDIterator(op, flags, op_flags, op_dtypes, order, casting, op_axes, itershape, buffersize);

        /// <summary>
        ///     NumPy's <c>numpy.nditer</c> — the public, managed face of NumSharp's
        ///     <see cref="NDIterRef"/>.
        ///
        ///     <code>
        ///     // numpy: for x in np.nditer(a): total += x
        ///     foreach (var vals in np.nditer(a))
        ///         total += (int)vals[0];
        ///
        ///     // numpy: it = np.nditer(a, flags=['multi_index'])
        ///     //        while not it.finished: print(it.multi_index, it[0]); it.iternext()
        ///     var it = np.nditer(a, flags: new[] {"multi_index"});
        ///     while (!it.finished) { Use(it.multi_index, it[0]); it.iternext(); }
        ///     </code>
        /// </summary>
        /// <remarks>
        ///     Port of NumPy 2.4.2's <c>numpy/_core/src/multiarray/nditer_pywrap.c</c> — the Python
        ///     WRAPPER around the C iterator, which is what this class is: argument conversion,
        ///     flag-string parsing, the property surface and the iteration protocol. The iterator
        ///     itself is <see cref="NDIterRef"/> (NumPy's <c>NpyIter</c>), which already implements
        ///     the buffering, casting, coalescing, broadcasting and index tracking.
        ///
        ///     <para>
        ///     <b>Lifetime.</b> <see cref="NDIterRef"/> is a <c>ref struct</c> and cannot live in a
        ///     class field, so this class owns the heap <c>NDIterState</c> directly (handed over by
        ///     <c>NDIterRef.Detach</c>) and re-borrows a non-owning <see cref="NDIterRef"/> for the
        ///     duration of each call. It therefore MUST be disposed — <see cref="close"/>,
        ///     <see cref="Dispose"/> or a <c>using</c> — which frees the unmanaged state and
        ///     resolves any <c>copy_if_overlap</c> / <c>updateifcopy</c> write-backs, exactly like
        ///     NumPy's <c>with np.nditer(...) as it:</c>. A finalizer is the safety net.
        ///     </para>
        ///
        ///     <para>
        ///     <b>The yielded arrays alias the iterator.</b> <see cref="this[int]"/> and
        ///     <see cref="value"/> return views onto the iterator's LIVE data pointer (0-d
        ///     normally, 1-d under <c>external_loop</c>), so they change under you on the next
        ///     step and are invalid after disposal — the same contract as NumPy, where the loop
        ///     variable must be copied to be kept. Writing through them writes to the operand
        ///     (or its buffer), which is how <c>readwrite</c> iteration mutates an array.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Iteration yields <c>NDArray[]</c>, always.</b> NumPy yields a bare 0-d array for
        ///     one operand and a tuple for several; C# has no such union, so enumeration always
        ///     produces the operand array — <c>vals[0]</c> for the single-operand case. This is
        ///     the same choice <see cref="np.Broadcast"/> made (<c>object[]</c> always).
        ///     </para>
        /// </remarks>
        public unsafe class NDIterator : IEnumerable<NDArray[]>, IDisposable
        {
            private NDIterState* _state;
            private NDArray[] _operands;
            private NDArray[] _writebackOriginals;
            private NDIterNextFunc _cachedNext;
            private bool _exhausted;

            // ---------------------------------------------------------------
            // Construction — the port of NumPy's nditer_init argument handling
            // ---------------------------------------------------------------

            internal NDIterator(
                NDArray[] op,
                string[] flags,
                string[][] op_flags,
                NPTypeCode[] op_dtypes,
                char order,
                string casting,
                int[][] op_axes,
                long[] itershape,
                long buffersize)
            {
                if (op == null || op.Length == 0)
                    throw new ArgumentException("Must provide at least one operand");

                int nop = op.Length;

                var globalFlags = ParseGlobalFlags(flags);
                var npyOrder = ParseOrder(order);
                var npyCasting = ParseCasting(casting);
                var perOpFlags = ParseOpFlags(op_flags, op, nop);

                // NumPy rejects EXTERNAL_LOOP combined with index tracking up front
                // (nditer_constr.c) — reproduce the message verbatim.
                if ((globalFlags & NDIterGlobalFlags.EXTERNAL_LOOP) != 0 &&
                    (globalFlags & (NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.F_INDEX | NDIterGlobalFlags.MULTI_INDEX)) != 0)
                    throw new ArgumentException(
                        "Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");

                // Zero-sized operands need an explicit opt-in, like NumPy.
                if ((globalFlags & NDIterGlobalFlags.ZEROSIZE_OK) == 0)
                {
                    for (int i = 0; i < nop; i++)
                    {
                        if (op[i] is not null && op[i].size == 0)
                            throw new ArgumentException("Iteration of zero-sized operands is not enabled");
                    }
                }

                op_dtypes = InferAllocateDtypes(op, perOpFlags, op_dtypes, nop);

                var iter = NDIterRef.AdvancedNew(
                    nop, op, globalFlags, npyOrder, npyCasting, perOpFlags, op_dtypes,
                    op_axes == null ? -1 : (op_axes.Length > 0 && op_axes[0] != null ? op_axes[0].Length : -1),
                    op_axes, itershape, buffersize);

                _state = iter.Detach(out var detachedOperands, out var detachedWritebacks);
                _operands = detachedOperands;
                _writebackOriginals = detachedWritebacks;
            }

            ~NDIterator() => ReleaseUnmanaged();

            // ---------------------------------------------------------------
            // Flag / order / casting conversion (nditer_pywrap.c converters)
            // ---------------------------------------------------------------

            private static NDIterGlobalFlags ParseGlobalFlags(string[] flags)
            {
                var result = NDIterGlobalFlags.None;
                if (flags == null)
                    return result;

                foreach (var f in flags)
                {
                    switch (f)
                    {
                        case "buffered": result |= NDIterGlobalFlags.BUFFERED; break;
                        case "c_index": result |= NDIterGlobalFlags.C_INDEX; break;
                        case "f_index": result |= NDIterGlobalFlags.F_INDEX; break;
                        case "multi_index": result |= NDIterGlobalFlags.MULTI_INDEX; break;
                        case "common_dtype": result |= NDIterGlobalFlags.COMMON_DTYPE; break;
                        case "copy_if_overlap": result |= NDIterGlobalFlags.COPY_IF_OVERLAP; break;
                        case "delay_bufalloc": result |= NDIterGlobalFlags.DELAY_BUFALLOC; break;
                        case "external_loop": result |= NDIterGlobalFlags.EXTERNAL_LOOP; break;
                        // Documented as grow_inner; the original spelling growinner still works.
                        case "grow_inner":
                        case "growinner": result |= NDIterGlobalFlags.GROWINNER; break;
                        case "ranged": result |= NDIterGlobalFlags.RANGED; break;
                        case "refs_ok": result |= NDIterGlobalFlags.REFS_OK; break;
                        case "reduce_ok": result |= NDIterGlobalFlags.REDUCE_OK; break;
                        case "zerosize_ok": result |= NDIterGlobalFlags.ZEROSIZE_OK; break;
                        default:
                            throw new ArgumentException($"Unexpected iterator global flag \"{f}\"");
                    }
                }

                return result;
            }

            private static NDIterPerOpFlags ParseOneOpFlags(string[] flags)
            {
                var result = NDIterPerOpFlags.None;
                if (flags == null)
                    return result;

                foreach (var f in flags)
                {
                    switch (f)
                    {
                        case "readonly": result |= NDIterPerOpFlags.READONLY; break;
                        case "readwrite": result |= NDIterPerOpFlags.READWRITE; break;
                        case "writeonly": result |= NDIterPerOpFlags.WRITEONLY; break;
                        case "nbo": result |= NDIterPerOpFlags.NBO; break;
                        case "aligned": result |= NDIterPerOpFlags.ALIGNED; break;
                        case "contig": result |= NDIterPerOpFlags.CONTIG; break;
                        case "copy": result |= NDIterPerOpFlags.COPY; break;
                        case "updateifcopy": result |= NDIterPerOpFlags.UPDATEIFCOPY; break;
                        case "allocate": result |= NDIterPerOpFlags.ALLOCATE; break;
                        case "no_subtype": result |= NDIterPerOpFlags.NO_SUBTYPE; break;
                        case "arraymask": result |= NDIterPerOpFlags.ARRAYMASK; break;
                        case "writemasked": result |= NDIterPerOpFlags.WRITEMASKED; break;
                        case "no_broadcast": result |= NDIterPerOpFlags.NO_BROADCAST; break;
                        case "virtual": result |= NDIterPerOpFlags.VIRTUAL; break;
                        case "overlap_assume_elementwise":
                            result |= NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP; break;
                        default:
                            throw new ArgumentException($"Unexpected per-op iterator flag \"{f}\"");
                    }
                }

                return result;
            }

            private static NDIterPerOpFlags[] ParseOpFlags(string[][] op_flags, NDArray[] op, int nop)
            {
                var result = new NDIterPerOpFlags[nop];

                if (op_flags == null)
                {
                    // NumPy nditer_pywrap.c:640 — with no op_flags, a None (null) operand becomes
                    // WRITEONLY|ALLOCATE and everything else READONLY.
                    for (int i = 0; i < nop; i++)
                    {
                        result[i] = op[i] is null
                            ? NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.ALLOCATE
                            : NDIterPerOpFlags.READONLY;
                    }

                    return result;
                }

                if (op_flags.Length != nop && op_flags.Length != 1)
                    throw new ArgumentException(
                        $"op_flags must be a tuple or array of per-op flag-tuples ({op_flags.Length} given for {nop} operands)");

                for (int i = 0; i < nop; i++)
                {
                    // A single inner list broadcasts to every operand (NumPy's flat-list form).
                    var flags = op_flags.Length == 1 ? op_flags[0] : op_flags[i];
                    var parsed = ParseOneOpFlags(flags);

                    // No read/write mode named: same defaulting as the no-op_flags path.
                    if ((parsed & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEONLY)) == 0)
                    {
                        parsed |= op[i] is null
                            ? NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.ALLOCATE
                            : NDIterPerOpFlags.READONLY;
                    }
                    else if (op[i] is null)
                    {
                        parsed |= NDIterPerOpFlags.ALLOCATE;
                    }

                    result[i] = parsed;
                }

                return result;
            }

            /// <summary>
            ///     Supply a dtype for every ALLOCATE operand the caller left unspecified.
            ///
            ///     NumPy infers it — <c>np.nditer([a, None])</c> allocates an <c>int64</c> output
            ///     for an <c>int64</c> input, and <c>np.nditer([int_a, float_b, None])</c> a
            ///     <c>float64</c> one — by promoting the operands that DO have a dtype
            ///     (<c>npyiter_get_common_dtype</c>). NumSharp's <c>NDIterRef</c> instead demands
            ///     an explicit <c>opDtypes</c> entry and throws without one, so the inference
            ///     belongs here in the wrapper, exactly where NumPy puts it.
            /// </summary>
            private static NPTypeCode[] InferAllocateDtypes(
                NDArray[] op, NDIterPerOpFlags[] perOpFlags, NPTypeCode[] op_dtypes, int nop)
            {
                bool needsInference = false;
                for (int i = 0; i < nop; i++)
                {
                    if (op[i] is null &&
                        (perOpFlags[i] & NDIterPerOpFlags.ALLOCATE) != 0 &&
                        (op_dtypes == null || i >= op_dtypes.Length || op_dtypes[i] == NPTypeCode.Empty))
                    {
                        needsInference = true;
                        break;
                    }
                }

                if (!needsInference)
                    return op_dtypes;

                var provided = new List<NDArray>(nop);
                for (int i = 0; i < nop; i++)
                {
                    if (op[i] is not null)
                        provided.Add(op[i]);
                }

                if (provided.Count == 0)
                    throw new ArgumentException(
                        "Iterator operand required copying or buffering, but neither copying nor buffering was enabled");

                var common = result_type(provided.ToArray());

                var resolved = new NPTypeCode[nop];
                for (int i = 0; i < nop; i++)
                {
                    if (op_dtypes != null && i < op_dtypes.Length && op_dtypes[i] != NPTypeCode.Empty)
                        resolved[i] = op_dtypes[i];
                    else if (op[i] is null)
                        resolved[i] = common;
                    else
                        resolved[i] = op[i].typecode;
                }

                return resolved;
            }

            private static NPY_ORDER ParseOrder(char order)
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

            private static NPY_CASTING ParseCasting(string casting)
            {
                switch (casting)
                {
                    case "no": return NPY_CASTING.NPY_NO_CASTING;
                    case "equiv": return NPY_CASTING.NPY_EQUIV_CASTING;
                    case "safe": return NPY_CASTING.NPY_SAFE_CASTING;
                    case "same_kind": return NPY_CASTING.NPY_SAME_KIND_CASTING;
                    case "unsafe": return NPY_CASTING.NPY_UNSAFE_CASTING;
                    default:
                        throw new ArgumentException(
                            $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
                }
            }

            // ---------------------------------------------------------------
            // Borrowing
            // ---------------------------------------------------------------

            private NDIterRef Borrow()
            {
                if (_state == null)
                    throw new InvalidOperationException("Iterator is closed");

                return NDIterRef.Borrow(_state, _operands, _cachedNext);
            }

            // ---------------------------------------------------------------
            // Properties (NumPy's nditer getters)
            // ---------------------------------------------------------------

            /// <summary>The number of operands (NumPy's <c>nop</c>).</summary>
            public int nop => Borrow().NOp;

            /// <summary>The number of dimensions iterated (NumPy's <c>ndim</c>).</summary>
            public int ndim => Borrow().NDim;

            /// <summary>
            ///     The shape being iterated (NumPy's <c>shape</c>). Without <c>multi_index</c> the
            ///     iterator is free to coalesce and reorder axes, so this is the COALESCED shape
            ///     (probed: a C-contiguous (2,3) reports <c>(6,)</c>); with <c>multi_index</c> the
            ///     original axis order is preserved and it reports <c>(2, 3)</c>.
            /// </summary>
            public long[] shape => Borrow().Shape;

            /// <summary>Total number of elements the iterator will visit (NumPy's <c>itersize</c>).</summary>
            public long itersize => Borrow().IterSize;

            /// <summary>The operands, including any the iterator ALLOCATED (NumPy's <c>operands</c>).</summary>
            public NDArray[] operands => _operands;

            /// <summary>The per-operand iteration dtypes (NumPy's <c>dtypes</c>).</summary>
            public NPTypeCode[] dtypes => Borrow().GetDescrArray();

            /// <summary>True once iteration has run past the end (NumPy's <c>finished</c>).</summary>
            public bool finished => _state == null || Borrow().Finished;

            /// <summary>Whether a C- or F-order flat index is being tracked (NumPy's <c>has_index</c>).</summary>
            public bool has_index => (_state->ItFlags & (uint)(NDIterFlags.HASINDEX)) != 0;

            /// <summary>Whether a multi-index is being tracked (NumPy's <c>has_multi_index</c>).</summary>
            public bool has_multi_index => (_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0;

            /// <summary>Whether buffer allocation is still delayed pending a <see cref="reset"/> (NumPy's <c>has_delayed_bufalloc</c>).</summary>
            public bool has_delayed_bufalloc => (_state->ItFlags & (uint)NDIterFlags.DELAYBUF) != 0;

            /// <summary>
            ///     Whether iteration needs the Python C-API (NumPy's <c>iterationneedsapi</c>).
            ///     Always FALSE in NumSharp — there is no Python runtime, and the transfer flags
            ///     never carry <c>REQUIRES_PYAPI</c>.
            /// </summary>
            public bool iterationneedsapi
                => (Borrow().GetTransferFlags() & NDArrayMethodFlags.REQUIRES_PYAPI) != 0;

            /// <summary>
            ///     The tracked flat index (NumPy's <c>index</c>). Requires the <c>c_index</c> or
            ///     <c>f_index</c> flag.
            /// </summary>
            public long index => Borrow().GetIndex();

            /// <summary>
            ///     The tracked multi-index (NumPy's <c>multi_index</c>). Requires the
            ///     <c>multi_index</c> flag.
            /// </summary>
            public long[] multi_index
            {
                get
                {
                    var it = Borrow();
                    var coords = new long[it.NDim];
                    it.GetMultiIndex(coords);
                    return coords;
                }
            }

            /// <summary>The current flat iteration position (NumPy's <c>iterindex</c>).</summary>
            public long iterindex
            {
                get => Borrow().IterIndex;
                set
                {
                    var it = Borrow();
                    it.GotoIterIndex(value);
                    _cachedNext = it.PeekCachedIterNext();
                    _exhausted = false;
                }
            }

            /// <summary>
            ///     The <c>[start, end)</c> sub-range being iterated (NumPy's <c>iterrange</c>).
            ///     Setting it requires the <c>ranged</c> flag.
            /// </summary>
            public (long Start, long End) iterrange
            {
                get => Borrow().IterRange;
                set
                {
                    var it = Borrow();
                    it.ResetToIterIndexRange(value.Start, value.End);
                    _cachedNext = it.PeekCachedIterNext();
                    _exhausted = false;
                }
            }

            /// <summary>
            ///     Per-operand views with the iterator's internal axis ordering (NumPy's
            ///     <c>itviews</c>). Not available while buffering.
            /// </summary>
            public NDArray[] itviews
            {
                get
                {
                    var it = Borrow();
                    var result = new NDArray[it.NOp];
                    for (int i = 0; i < result.Length; i++)
                        result[i] = it.GetIterView(i);

                    return result;
                }
            }

            /// <summary>
            ///     The current values — one live view per operand (NumPy's <c>value</c>). NumPy
            ///     returns a bare array for a single operand and a tuple otherwise; this always
            ///     returns the array, so use <c>value[0]</c> for the single-operand case.
            /// </summary>
            public NDArray[] value
            {
                get
                {
                    var result = new NDArray[Borrow().NOp];
                    for (int i = 0; i < result.Length; i++)
                        result[i] = this[i];

                    return result;
                }
            }

            /// <summary>
            ///     A LIVE view of operand <paramref name="i"/> at the current position (NumPy's
            ///     <c>it[i]</c>) — 0-d normally, 1-d spanning the inner loop under
            ///     <c>external_loop</c>. Writing through it writes to the operand or its buffer;
            ///     the view is invalidated by the next step and by disposal.
            /// </summary>
            public NDArray this[int i]
            {
                get
                {
                    var it = Borrow();
                    if ((uint)i >= (uint)it.NOp)
                        throw new ArgumentOutOfRangeException(nameof(i), $"Operand index {i} out of range [0, {it.NOp})");

                    void* ptr = _state->GetDataPtr(i);
                    var dtype = _state->GetOpDType(i);

                    if (!it.HasExternalLoop)
                        return AliasView(ptr, dtype, 1, 1, scalar: true);

                    long count = *it.GetInnerLoopSizePtr();
                    long stride = it.GetInnerLoopElementStride(i);
                    return AliasView(ptr, dtype, count, stride, scalar: false);
                }
            }

            // ---------------------------------------------------------------
            // Methods (NumPy's nditer methods)
            // ---------------------------------------------------------------

            /// <summary>Advance to the next element/chunk (NumPy's <c>iternext()</c>). Returns false at the end.</summary>
            public bool iternext()
            {
                var it = Borrow();
                bool more = it.Iternext();
                _cachedNext = it.PeekCachedIterNext();
                if (!more)
                    _exhausted = true;

                return more;
            }

            /// <summary>Rewind to the start (NumPy's <c>reset()</c>); also allocates delayed buffers.</summary>
            public void reset()
            {
                var it = Borrow();
                it.Reset();
                _cachedNext = it.PeekCachedIterNext();
                _exhausted = false;
            }

            /// <summary>
            ///     Duplicate the iterator at its current position (NumPy's <c>copy()</c>). The copy
            ///     owns its own state and must be disposed independently.
            /// </summary>
            public NDIterator copy()
            {
                var copied = Borrow().Copy();
                return new NDIterator(copied);
            }

            /// <summary>Private ctor adopting an already-built NDIterRef (used by <see cref="copy"/>).</summary>
            private NDIterator(NDIterRef iter)
            {
                _state = iter.Detach(out var operands, out var writebacks);
                _operands = operands;
                _writebackOriginals = writebacks;
            }

            /// <summary>
            ///     Remove an axis from iteration (NumPy's <c>remove_axis(i)</c>). Requires the
            ///     <c>multi_index</c> flag.
            /// </summary>
            public void remove_axis(int axis)
            {
                var it = Borrow();
                if (!it.RemoveAxis(axis))
                    throw new ArgumentException($"Iterator axis {axis} cannot be removed");

                _cachedNext = it.PeekCachedIterNext();
            }

            /// <summary>
            ///     Stop tracking the multi-index, letting the iterator coalesce and reorder axes
            ///     (NumPy's <c>remove_multi_index()</c>). Resets the position to the start.
            /// </summary>
            public void remove_multi_index()
            {
                var it = Borrow();
                it.RemoveMultiIndex();
                _cachedNext = it.PeekCachedIterNext();
                _exhausted = false;
            }

            /// <summary>Switch to external-loop iteration after construction (NumPy's <c>enable_external_loop()</c>).</summary>
            public void enable_external_loop()
            {
                var it = Borrow();
                it.EnableExternalLoop();
                _cachedNext = it.PeekCachedIterNext();
                _exhausted = false;
            }

            /// <summary>Dump the iterator's internal state (NumPy's <c>debug_print()</c>).</summary>
            public void debug_print() => Borrow().DebugPrint();

            /// <summary>
            ///     Resolve write-backs and release the iterator (NumPy's <c>close()</c>, i.e. the
            ///     end of a <c>with np.nditer(...) as it:</c> block). Idempotent; the iterator is
            ///     unusable afterwards.
            /// </summary>
            public void close() => Dispose();

            public void Dispose()
            {
                ReleaseUnmanaged();
                GC.SuppressFinalize(this);
            }

            private void ReleaseUnmanaged()
            {
                var state = _state;
                if (state == null)
                    return;

                _state = null;
                _cachedNext = null;

                // Flush a pending buffered window before the write-backs, so buffer contents
                // reach a forced-copy temp before that temp is copied out (NDIterRef.Dispose
                // orders it the same way).
                if ((state->ItFlags & (uint)NDIterFlags.BUFFER) != 0 &&
                    (state->ItFlags & (uint)NDIterFlags.REDUCE) == 0 &&
                    (state->ItFlags & (uint)NDIterFlags.DELAYBUF) == 0)
                {
                    NDIterBufferManager.FlushBufferWindow(ref *state);
                }

                NDIterRef.ResolveDetachedWritebacks(_operands, _writebackOriginals);
                _writebackOriginals = null;

                NDIterRef.FreeState(state);
            }

            // ---------------------------------------------------------------
            // Enumeration
            // ---------------------------------------------------------------

            /// <summary>
            ///     Enumerates the iterator's single live cursor, so — as in NumPy, where
            ///     <c>iter(it) is it</c> — a second enumeration RESUMES where the first stopped
            ///     rather than restarting (call <see cref="reset"/> to go again).
            /// </summary>
            /// <remarks>
            ///     Deliberately a thin wrapper rather than <c>this</c>, even though returning
            ///     <c>this</c> would model <c>iter(it) is it</c> more literally: <c>foreach</c>
            ///     disposes the enumerator it obtains, and this class's <see cref="Dispose"/> frees
            ///     the unmanaged iterator state. Handing out <c>this</c> would therefore CLOSE the
            ///     iterator at the end of any <c>foreach</c>/LINQ pass, making every property read
            ///     afterwards throw. The wrapper's <c>Dispose</c> is a no-op and the cursor is
            ///     still shared, so the observable semantics are unchanged.
            ///     (<see cref="np.Broadcast"/> can safely return <c>this</c> only because it owns
            ///     no unmanaged resources.)
            /// </remarks>
            public IEnumerator<NDArray[]> GetEnumerator() => new Enumerator(this);

            IEnumerator IEnumerable.GetEnumerator() => new Enumerator(this);

            private sealed class Enumerator : IEnumerator<NDArray[]>
            {
                private readonly NDIterator _owner;

                internal Enumerator(NDIterator owner) => _owner = owner;

                public NDArray[] Current => _owner.Current;

                object IEnumerator.Current => _owner.Current;

                public bool MoveNext() => _owner.MoveNext();

                public void Reset() => _owner.reset();

                // NOT the iterator's Dispose — see GetEnumerator's remarks.
                public void Dispose() { }
            }

            /// <summary>The values published by the most recent <see cref="MoveNext"/>.</summary>
            public NDArray[] Current { get; private set; }

            /// <summary>
            ///     Publishes the values at the current position and THEN advances — the exact
            ///     shape of NumPy's <c>__next__</c> (<c>return self.value</c> followed by
            ///     <c>iternext()</c>), which is why after consuming one element the cursor already
            ///     reads 1 and a <see cref="copy"/> taken there continues from the SECOND element.
            /// </summary>
            /// <remarks>
            ///     Publishing before advancing is safe because <see cref="value"/> captures the
            ///     operand's ABSOLUTE data pointer, so the handed-out view keeps pointing at the
            ///     element it was made for. The exception is buffered <c>external_loop</c>
            ///     iteration, where the view aliases a buffer the next step refills — NumPy has
            ///     the identical hazard, hence its documented <c>[x.copy() for x in it]</c> idiom.
            /// </remarks>
            public bool MoveNext()
            {
                if (_state == null || _exhausted || Borrow().Finished)
                {
                    _exhausted = true;
                    Current = null;
                    return false;
                }

                Current = value;
                iternext();
                return true;
            }

            // ---------------------------------------------------------------
            // Live views over the iterator's data pointers
            // ---------------------------------------------------------------

            /// <summary>
            ///     Wrap a raw iterator data pointer in an <see cref="NDArray"/> WITHOUT copying or
            ///     owning the memory, so reads and writes go straight through to the operand (or
            ///     its buffer) — NumPy's <c>it[i]</c> semantics.
            /// </summary>
            private static NDArray AliasView(void* ptr, NPTypeCode dtype, long count, long stride, bool scalar)
            {
                switch (dtype)
                {
                    case NPTypeCode.Boolean: return AliasView<bool>(ptr, count, stride, scalar);
                    case NPTypeCode.Byte: return AliasView<byte>(ptr, count, stride, scalar);
                    case NPTypeCode.SByte: return AliasView<sbyte>(ptr, count, stride, scalar);
                    case NPTypeCode.Int16: return AliasView<short>(ptr, count, stride, scalar);
                    case NPTypeCode.UInt16: return AliasView<ushort>(ptr, count, stride, scalar);
                    case NPTypeCode.Int32: return AliasView<int>(ptr, count, stride, scalar);
                    case NPTypeCode.UInt32: return AliasView<uint>(ptr, count, stride, scalar);
                    case NPTypeCode.Int64: return AliasView<long>(ptr, count, stride, scalar);
                    case NPTypeCode.UInt64: return AliasView<ulong>(ptr, count, stride, scalar);
                    case NPTypeCode.Char: return AliasView<char>(ptr, count, stride, scalar);
                    case NPTypeCode.Half: return AliasView<Half>(ptr, count, stride, scalar);
                    case NPTypeCode.Single: return AliasView<float>(ptr, count, stride, scalar);
                    case NPTypeCode.Double: return AliasView<double>(ptr, count, stride, scalar);
                    case NPTypeCode.Decimal: return AliasView<decimal>(ptr, count, stride, scalar);
                    case NPTypeCode.Complex: return AliasView<System.Numerics.Complex>(ptr, count, stride, scalar);
                    default: throw new NotSupportedException($"Unsupported iterator dtype {dtype}");
                }
            }

            private static NDArray AliasView<T>(void* ptr, long count, long stride, bool scalar) where T : unmanaged
            {
                // The block must span every element the view can touch. With a negative stride the
                // first logical element sits at the HIGH end, so the block starts below the
                // iterator's pointer and the shape carries the compensating offset.
                long far = count <= 0 ? 0 : stride * (count - 1);
                long low = Math.Min(0, far);
                long high = Math.Max(0, far);
                long blockCount = high - low + 1;

                var block = new UnmanagedMemoryBlock<T>((T*)ptr + low, blockCount);
                var slice = new ArraySlice<T>(block);

                // HOT PATH: the 0-d per-element view, built once per operand per step. Its shape
                // size matches the slice exactly, so it can take NDArray's direct slice ctor —
                // measured 2x cheaper than the storage ctor below, which re-aliases the storage
                // into a second UnmanagedStorage (NDArray.cs: `storage.Alias(ref shape)`).
                if (scalar)
                    return new NDArray(slice, Shape.NewScalar());

                // external_loop: a strided inner loop spans more slots than it has elements
                // (count=2, stride=2 covers 3), which the slice ctor rejects — it demands
                // shape.size == slice.Count. So wrap the block as FLAT storage first and apply the
                // strided/offset shape through the view ctor, the same two-step
                // NDIterRef.GetIterView uses. This path runs once per CHUNK, not per element.
                var storage = new UnmanagedStorage(slice, Shape.Vector(blockCount));
                var shape = new Shape(new[] {count}, new[] {stride}, -low, blockCount);

                return new NDArray(storage, shape);
            }
        }
    }
}
