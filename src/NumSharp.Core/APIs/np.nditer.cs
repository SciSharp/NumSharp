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
        /// <param name="op_dtypes">
        ///     The required data type(s) of the operands — one per operand, null for "the operand's own". An unsized string
        ///     request (<c>"U"</c>, <c>"S"</c>) is adapted to its operand (<c>'U'</c> over an int64 array is <c>&lt;U21</c>). A
        ///     request other than the operand's dtype is a CAST: it must satisfy <paramref name="casting"/> in the read
        ///     direction (and the write-back direction for a written operand), and it needs <c>buffered</c>, a <c>copy</c> /
        ///     <c>updateifcopy</c> operand flag, or a read-only 0-d operand (copied automatically) — else NumPy's
        ///     <c>TypeError</c>. A list whose length is not the operand count is NumPy's <c>TypeError</c> too.
        /// </param>
        /// <param name="order">Iteration order: <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c> (default).</param>
        /// <param name="casting">
        ///     Casting rule when making a copy or buffering: <c>"no"</c>, <c>"equiv"</c>,
        ///     <c>"safe"</c> (default), <c>"same_kind"</c>, <c>"unsafe"</c>.
        /// </param>
        /// <param name="op_axes">Per-operand list of axes, mapping iterator dimensions to operand dimensions (-1 = newaxis).</param>
        /// <param name="itershape">The desired shape of the iterator.</param>
        /// <param name="buffersize">Buffer size to use when buffering is enabled; 0 selects the default.</param>
        /// <returns>The iterator (dispose it: <see cref="NDIterator.close"/> resolves its write-backs).</returns>
        /// <exception cref="ValueError">NumPy's argument and flag errors (verbatim texts).</exception>
        /// <exception cref="TypeError">A refused cast, or a cast operand neither copied nor buffered.</exception>
        /// <exception cref="StringComputationNotSupportedException">A conversion between text and numbers that NumPy computes
        ///     (the string dtypes' computational phase).</exception>
        /// <remarks>
        ///     Fixed-width strings (<c>S</c>, <c>U</c>) iterate like any dtype: each step yields 0-d <c>&lt;U3</c> views (1-D
        ///     chunks under <c>external_loop</c>), a buffered <c>op_dtypes</c> request widens, truncates or converts between
        ///     <c>S</c> and <c>U</c> (ASCII only, NumPy's Unicode errors otherwise), and a <c>readwrite</c> view writes through.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nditer.html
        /// </remarks>
        public static NDIterator nditer(
            NDArray op,
            string[] flags = null,
            string[] op_flags = null,
            DType[] op_dtypes = null,
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
        /// <inheritdoc cref="nditer(NDArray, string[], string[], DType[], char, string, int[][], long[], long)"/>
        public static NDIterator nditer(
            NDArray[] op,
            string[] flags = null,
            string[][] op_flags = null,
            DType[] op_dtypes = null,
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
        ///
        ///     <para>
        ///     <b>Descriptors.</b> The iterator is built over NumPy's resolved per-operand DESCRIPTORS
        ///     (<see cref="dtypes"/>), not storage lanes, so fixed-width strings iterate, and the cast rules are NumPy's:
        ///     <c>copy</c> / <c>updateifcopy</c> operands get a temporary in the iteration dtype laid out in the iterator's
        ///     order (written back with unsafe casting at <see cref="close"/>), a read-only 0-d operand is copied without
        ///     being asked, and every other cast needs <c>buffered</c>.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Known differences.</b> An allocated output requested as unsized <c>'U'</c> / <c>'S'</c> is allocated with
        ///     one character, as in NumPy, but NumPy's views of it are unsized (writing through them stores nothing) while
        ///     NumSharp's are one character wide. A non-native request (<c>'&gt;U3'</c>) is reported and cast-checked as
        ///     NumPy does, but its buffer holds native data. The <c>contig</c> flag does not raise NumPy's "required
        ///     buffering" error when an unbuffered operand is not contiguous, and <c>nbo</c> / <c>aligned</c> force no copy
        ///     (NumSharp arrays are native; see the string dtype plan's open items).
        ///     </para>
        /// </remarks>
        public unsafe partial class NDIterator : IEnumerable<NDArray[]>, IDisposable
        {
            private NDIterState* _state;
            private NDArray[] _operands;
            private NDArray[] _writebackOriginals;

            /// <summary>
            ///     The read-only copy-route temporaries' originals (a 0-d operand read through a cast, a read-only
            ///     <c>copy</c> operand): released at teardown with nothing written back. See
            ///     <see cref="NDIterRef.ResolveDetachedWritebacks"/>.
            /// </summary>
            private NDArray[] _releaseOnlyOriginals;

            /// <summary>
            ///     NumPy's per-operand iteration descriptors (<c>it.dtypes</c>), as resolved at construction: the requested
            ///     dtype adapted to its operand, the common dtype, or the operand's own. An allocated output keeps the request
            ///     as given — an unsized <c>'U'</c> reads <c>dtype('&lt;U')</c> although the array is allocated <c>&lt;U1</c>.
            /// </summary>
            private DType[] _iterDescrs;

            private NDIterNextFunc _cachedNext;

            /// <summary>
            ///     NumPy's wrapper flag <c>finished</c>: set when a pull or <see cref="iternext"/> runs past the end, and at
            ///     construction / every reset when there is nothing to visit; cleared by every seek. While it is set the
            ///     position getters and the element views raise NumPy's "Iterator is past the end", and a pull ends.
            /// </summary>
            private bool _finished;

            /// <summary>
            ///     NumPy's <c>get_multi_index != NULL</c>: the multi-index getter exists only when a multi-index is tracked and
            ///     no buffer allocation is pending at the last <c>npyiter_cache_values</c> — or since the last reset / range
            ///     reset. Without it <see cref="multi_index"/> reports why (not tracked, delayed allocation, invalid state).
            /// </summary>
            private bool _hasGetMultiIndex;

            /// <summary>
            ///     A buffered REDUCTION built with <c>delay_bufalloc</c>: NumSharp's reduce setup is construction-bound and
            ///     allocates at once, but NumPy's contract is still the delayed one — <see cref="has_delayed_bufalloc"/>
            ///     reads true and the element views refuse until <see cref="reset"/> — so the wrapper keeps that state
            ///     itself (the engine's <c>DELAYBUF</c> flag carries it for every other buffered iterator).
            /// </summary>
            private bool _reduceDelayPending;

            // --- np.nested_iters support (NumPy's nested_child chain) ---
            // A nested group's levels share NumPy's advance-on-entry protocol (npyiter_next's `started` flag) with the plain
            // iterator; what nesting adds is the child re-base cascade (npyiter_resetbasepointers), run after every advance
            // and seek of a level so the next level walks the sub-array at the parent's CURRENT position.
            private NDIterator _nestedChild;   // next inner level, or null (the innermost / non-nested)
            private bool _nestedMode;          // member of a np.nested_iters group (MoveNextNested + the re-base cascade)
            private bool _started;             // NumPy's `started` flag (advance only after the first pull)

            // ---------------------------------------------------------------
            // Construction — the port of NumPy's nditer_init argument handling
            // ---------------------------------------------------------------

            /// <summary>
            ///     The port of NumPy's <c>npyiter_init</c> (<c>nditer_pywrap.c</c>) and of the descriptor half of
            ///     <c>NpyIter_AdvancedNew</c>: the arguments are converted in NumPy's order (flags, order, casting, op_flags,
            ///     op_dtypes, op_axes), the global flags are checked, every operand is prepared in turn
            ///     (<c>npyiter_prepare_operands</c>: its flags, its writeability, its size, its requested dtype adapted to it),
            ///     the dtypes an allocation or <c>common_dtype</c> needs are resolved (<c>npyiter_get_common_dtype</c>), and the
            ///     iterator is built over the resulting DESCRIPTORS — which is what lets it iterate fixed-width strings and run
            ///     NumPy's cast checks and copy routes.
            /// </summary>
            /// <param name="op">The operands; a null entry is an output for the iterator to allocate.</param>
            /// <param name="flags">The global flags.</param>
            /// <param name="op_flags">The per-operand flags (one inner list per operand, or one list for all).</param>
            /// <param name="op_dtypes">The requested iteration dtypes (one per operand; null entries mean "the operand's").</param>
            /// <param name="order">The iteration order.</param>
            /// <param name="casting">The casting rule.</param>
            /// <param name="op_axes">The per-operand axis maps (null entries for natural broadcasting).</param>
            /// <param name="itershape">The iteration shape (with <paramref name="op_axes"/>).</param>
            /// <param name="buffersize">The buffer size (0 = <c>np.getbufsize()</c>).</param>
            /// <exception cref="ValueError">NumPy's argument and flag errors (verbatim texts).</exception>
            /// <exception cref="TypeError">A malformed <paramref name="op_dtypes"/> sequence, a refused cast, or a cast
            ///     operand neither copied nor buffered.</exception>
            /// <exception cref="StringComputationNotSupportedException">A conversion between text and numbers that NumPy
            ///     computes (the computational phase of the string dtypes).</exception>
            internal NDIterator(
                NDArray[] op,
                string[] flags,
                string[][] op_flags,
                DType[] op_dtypes,
                char order,
                string casting,
                int[][] op_axes,
                long[] itershape,
                long buffersize)
            {
                // npyiter_prepare_ops.
                if (op == null || op.Length == 0)
                    throw new ValueError("Must provide at least one operand");

                // NumPy never touches the caller's operand sequence, while the engine fills the ALLOCATE slots and swaps in
                // copy-route temporaries: it works on an array of its own.
                op = (NDArray[])op.Clone();
                int nop = op.Length;

                // PyArg_ParseTupleAndKeywords' converters, in NumPy's argument order, then npyiter_convert_ops (op_flags),
                // npyiter_convert_dtypes and npyiter_convert_op_axes.
                var globalFlags = ParseGlobalFlags(flags);
                var npyOrder = ParseOrder(order);
                var npyCasting = ParseCasting(casting);
                var perOpFlags = ParseOpFlags(op_flags, op, nop);
                var requests = ConvertDtypes(op_dtypes, nop);
                int oaNDim = ConvertOpAxes(op_axes, nop);

                // NpyIter_AdvancedNew: npyiter_check_global_flags, then each operand in turn, then the common dtype.
                CheckGlobalFlags(globalFlags);
                var descrs = PrepareOperands(op, perOpFlags, requests, globalFlags, nop);
                if (op_axes == null && itershape == null)
                    CheckBroadcast(op, nop);
                ResolveCommonDtype(op, perOpFlags, descrs, requests, globalFlags, nop);

                // The engine still takes the type codes (its allocation fallback); the descriptors carry the widths.
                var op_typecodes = new NPTypeCode[nop];
                for (int i = 0; i < nop; i++)
                    op_typecodes[i] = descrs[i] is null ? NPTypeCode.Empty : descrs[i].GetTypeCode();

                var iter = NDIterRef.AdvancedNew(
                    nop, op, globalFlags, npyOrder, npyCasting, perOpFlags, op_typecodes,
                    oaNDim, op_axes, itershape, buffersize, descrs);

                _state = iter.Detach(out var detachedOperands, out var detachedWritebacks, out var detachedReleaseOnly);
                _operands = detachedOperands;
                _writebackOriginals = detachedWritebacks;
                _releaseOnlyOriginals = detachedReleaseOnly;
                _iterDescrs = descrs;

                // A buffered reduction's delayed allocation is kept by the wrapper (see _reduceDelayPending): the engine
                // allocated it at construction, NumPy's contract says it is still pending.
                const NDIterGlobalFlags delayedBuffering = NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.DELAY_BUFALLOC;
                _reduceDelayPending = (globalFlags & delayedBuffering) == delayedBuffering &&
                                      (_state->ItFlags & (uint)NDIterFlags.REDUCE) != 0 &&
                                      (_state->ItFlags & (uint)NDIterFlags.DELAYBUF) == 0;

                // The end of npyiter_init: npyiter_cache_values, then an empty iteration starts both started and finished
                // (its first pull ends at once), any other starts fresh.
                CacheValues();
                _started = _finished = _state->IterSize == 0;
            }

            ~NDIterator() => ReleaseUnmanaged();

            // ---------------------------------------------------------------
            // Flag / order / casting conversion (nditer_pywrap.c converters)
            // ---------------------------------------------------------------

            /// <summary>NumPy's <c>NpyIter_GlobalFlagsConverter</c>: the global flag strings.</summary>
            /// <param name="flags">The flag strings, or null for none.</param>
            /// <returns>The flags.</returns>
            /// <exception cref="ValueError"><c>Unexpected iterator global flag "x"</c> (per-operand flags included).</exception>
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
                            throw new ValueError($"Unexpected iterator global flag \"{f}\"");
                    }
                }

                return result;
            }

            /// <summary>NumPy's <c>NpyIter_OpFlagsConverter</c>: one operand's flag strings.</summary>
            /// <param name="flags">The flag strings (not null — the caller rejects a missing entry).</param>
            /// <returns>The flags.</returns>
            /// <exception cref="ValueError"><c>Unexpected per-op iterator flag "x"</c> (global flags included).</exception>
            private static NDIterPerOpFlags ParseOneOpFlags(string[] flags)
            {
                var result = NDIterPerOpFlags.None;
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
                            throw new ValueError($"Unexpected per-op iterator flag \"{f}\"");
                    }
                }

                return result;
            }

            /// <summary>
            ///     NumPy's <c>npyiter_convert_ops</c> flag half: without <paramref name="op_flags"/> a null operand is
            ///     <c>writeonly, allocate</c> and every other one <c>readonly</c>; with them, each operand takes its own list
            ///     (or the one list given, NumPy's flat-list form) EXACTLY as written — nothing is defaulted, so a list without a
            ///     read/write mode, or a null operand without <c>allocate</c>, is refused later with NumPy's text.
            /// </summary>
            /// <param name="op_flags">The per-operand flag lists, or null.</param>
            /// <param name="op">The operands (null entries are outputs to allocate).</param>
            /// <param name="nop">The operand count.</param>
            /// <returns>The per-operand flags.</returns>
            /// <exception cref="ValueError"><c>op_flags must be a tuple or array of per-op flag-tuples</c> (a list count that
            ///     is neither one nor the operand count, or a null list); an unknown flag string.</exception>
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

                // npyiter_convert_op_flags_array: a sequence of the wrong length is tried as ONE set of flags, which a list of
                // lists is not — NumPy's error. A single inner list for several operands is NumSharp's spelling of NumPy's
                // flat list of strings (one set for all).
                if (op_flags.Length != nop && op_flags.Length != 1)
                    throw new ValueError("op_flags must be a tuple or array of per-op flag-tuples");

                for (int i = 0; i < nop; i++)
                {
                    var flags = op_flags.Length == 1 ? op_flags[0] : op_flags[i];
                    // NumPy's NpyIter_OpFlagsConverter refuses anything that is not a list of strings.
                    if (flags == null)
                        throw new ValueError("op_flags must be a tuple or array of per-op flag-tuples");
                    result[i] = ParseOneOpFlags(flags);
                }

                return result;
            }

            /// <summary>
            ///     NumPy's <c>npyiter_convert_dtypes</c>: one requested dtype per operand (null = the operand's own). A sequence
            ///     of any other length is tried by NumPy as ONE dtype — a list of dtypes reads as a structured-dtype field list,
            ///     which fails — so the same lengths fail here.
            /// </summary>
            /// <param name="op_dtypes">The requested dtypes, or null for none.</param>
            /// <param name="nop">The operand count.</param>
            /// <returns>The requests (an array of <paramref name="nop"/> entries), or null.</returns>
            /// <exception cref="TypeError"><c>Field elements must be 2- or 3-tuples, got '…'</c>: a non-empty list whose length
            ///     is not the operand count (NumPy's message, naming the first entry).</exception>
            /// <exception cref="NotSupportedException">An EMPTY list: NumPy reads it as the structured dtype <c>dtype([])</c>,
            ///     which NumSharp does not have.</exception>
            private static DType[] ConvertDtypes(DType[] op_dtypes, int nop)
            {
                if (op_dtypes == null)
                    return null;
                if (op_dtypes.Length == nop)
                    return (DType[])op_dtypes.Clone();
                if (op_dtypes.Length == 0)
                    throw new NotSupportedException(
                        "op_dtypes=[] is NumPy's empty structured dtype dtype([]) applied to every operand; NumSharp has no " +
                        "structured dtypes. Pass one dtype (or null) per operand.");
                var first = op_dtypes[0];
                throw new TypeError($"Field elements must be 2- or 3-tuples, got '{(first is null ? "None" : first.ToString(true))}'");
            }

            /// <summary>
            ///     NumPy's <c>npyiter_convert_op_axes</c>: the op_axes rank — the length of the first non-null entry, which every
            ///     other non-null entry must share.
            /// </summary>
            /// <param name="op_axes">The per-operand axis maps, or null.</param>
            /// <param name="nop">The operand count.</param>
            /// <returns>The op_axes rank, or -1 when <paramref name="op_axes"/> is null.</returns>
            /// <exception cref="ValueError">NumPy's texts: a list count other than the operand count; entries of different
            ///     lengths; every entry null.</exception>
            private static int ConvertOpAxes(int[][] op_axes, int nop)
            {
                if (op_axes == null)
                    return -1;
                if (op_axes.Length != nop)
                    throw new ValueError("op_axes must be a tuple/list matching the number of ops");

                int oaNDim = -1;
                foreach (var axes in op_axes)
                {
                    if (axes == null)
                        continue;
                    if (oaNDim == -1)
                        oaNDim = axes.Length;
                    if (axes.Length != oaNDim)
                        throw new ValueError("Each entry of op_axes must have the same size");
                }

                if (oaNDim == -1)
                    throw new ValueError("If op_axes is provided, at least one list of axes must be contained within it");
                return oaNDim;
            }

            /// <summary>NumPy's <c>npyiter_check_global_flags</c>: the global flag combinations it refuses.</summary>
            /// <param name="flags">The global flags.</param>
            /// <exception cref="ValueError">Both indices; an external loop with a tracked index; <c>ranged</c> with an
            ///     external loop but no buffering (NumPy's texts).</exception>
            private static void CheckGlobalFlags(NDIterGlobalFlags flags)
            {
                bool hasIndex = (flags & (NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.F_INDEX)) != 0;
                if ((flags & (NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.F_INDEX)) ==
                    (NDIterGlobalFlags.C_INDEX | NDIterGlobalFlags.F_INDEX))
                    throw new ValueError("Iterator flags C_INDEX and F_INDEX cannot both be specified");
                if ((flags & NDIterGlobalFlags.EXTERNAL_LOOP) != 0 &&
                    (hasIndex || (flags & NDIterGlobalFlags.MULTI_INDEX) != 0))
                    throw new ValueError("Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked");
                if ((flags & NDIterGlobalFlags.RANGED) != 0 &&
                    (flags & NDIterGlobalFlags.EXTERNAL_LOOP) != 0 &&
                    (flags & NDIterGlobalFlags.BUFFERED) == 0)
                    throw new ValueError(
                        "Iterator flag RANGED cannot be used with the flag EXTERNAL_LOOP unless BUFFERED is also enabled");
            }

            /// <summary>
            ///     NumPy's <c>npyiter_prepare_operands</c>: each operand in turn — its flags
            ///     (<c>npyiter_check_per_op_flags</c>), the single ARRAYMASK, then <c>npyiter_prepare_one_operand</c> (the null /
            ///     VIRTUAL rules, writeability, size, and the requested dtype adapted to the array) — then the
            ///     WRITEMASKED / ARRAYMASK pairing.
            /// </summary>
            /// <param name="op">The operands.</param>
            /// <param name="perOpFlags">The per-operand flags.</param>
            /// <param name="requests">The requested dtypes, or null.</param>
            /// <param name="globalFlags">The global flags.</param>
            /// <param name="nop">The operand count.</param>
            /// <returns>Per operand, its iteration descriptor so far: the adapted request or the operand's own; for a null
            ///     operand the request (or bool for an ARRAYMASK), else null until <see cref="ResolveCommonDtype"/>.</returns>
            /// <exception cref="ValueError">NumPy's per-operand texts, in NumPy's order.</exception>
            /// <remarks>
            ///     An unsized string request is adapted to its array (<c>PyArray_AdaptDescriptorToArray</c>): <c>U3</c> under
            ///     <c>'U'</c> stays <c>&lt;U3</c>, <c>|S3</c> under <c>'U'</c> becomes <c>&lt;U3</c>, an int64 array under
            ///     <c>'U'</c> becomes <c>&lt;U21</c> — the width a cast from the array would pick. NumPy's <c>refs_ok</c>, <c>nbo</c>
            ///     and <c>aligned</c> checks have nothing to act on (NumSharp has no object dtype, and its arrays are native);
            ///     they are not reproduced.
            /// </remarks>
            private static DType[] PrepareOperands(NDArray[] op, NDIterPerOpFlags[] perOpFlags, DType[] requests,
                                                   NDIterGlobalFlags globalFlags, int nop)
            {
                var descrs = new DType[nop];
                int maskOp = -1;
                bool anyWriteMasked = false;

                for (int i = 0; i < nop; i++)
                {
                    var f = perOpFlags[i];
                    bool read, write;
                    CheckPerOpFlags(f, out read, out write);

                    if ((f & NDIterPerOpFlags.ARRAYMASK) != 0)
                    {
                        if (maskOp != -1)
                            throw new ValueError("Only one iterator operand may receive an ARRAYMASK flag");
                        maskOp = i;
                    }
                    if ((f & NDIterPerOpFlags.WRITEMASKED) != 0)
                        anyWriteMasked = true;

                    var request = requests?[i];
                    if (op[i] is null)
                    {
                        if ((f & (NDIterPerOpFlags.ALLOCATE | NDIterPerOpFlags.VIRTUAL)) == 0)
                            throw new ValueError(
                                "Iterator operand was NULL, but neither the ALLOCATE nor the VIRTUAL flag was specified");
                        if ((f & NDIterPerOpFlags.ALLOCATE) != 0)
                        {
                            if (!write)
                                throw new ValueError(
                                    "Automatic allocation was requested for an iterator operand, but it wasn't flagged for writing");
                            // NumPy's text, double space included.
                            if ((globalFlags & (NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.DELAY_BUFALLOC)) == NDIterGlobalFlags.BUFFERED &&
                                read)
                                throw new ValueError(
                                    "Automatic allocation was requested for an iterator operand, and it was flagged as readable, " +
                                    "but buffering  without delayed allocation was enabled");
                            descrs[i] = request;
                        }
                        // A mask with no requested dtype is bool (a VIRTUAL non-mask operand waits for the common dtype).
                        if ((f & NDIterPerOpFlags.ARRAYMASK) != 0 && descrs[i] is null)
                            descrs[i] = DType.Boolean;
                        continue;
                    }

                    if ((f & NDIterPerOpFlags.VIRTUAL) != 0)
                        throw new ValueError("Iterator operand flag VIRTUAL was specified, but the operand was not NULL");
                    // PyArray_FailUnlessWriteable: even a copy / updateifcopy operand writes BACK to the original.
                    if (write && !op[i].Shape.IsWriteable)
                        throw new ValueError("operand array with iterator write flag set is read-only");
                    if ((globalFlags & NDIterGlobalFlags.ZEROSIZE_OK) == 0 && op[i].size == 0)
                        throw new ValueError("Iteration of zero-sized operands is not enabled");

                    var own = op[i].dtype;
                    descrs[i] = request is null || ReferenceEquals(request, own) ? own : AdaptDescriptorToArray(own, request);
                }

                if (anyWriteMasked && maskOp < 0)
                    throw new ValueError(
                        "An iterator operand was flagged as WRITEMASKED, but no ARRAYMASK operand was given to supply the mask");
                if (!anyWriteMasked && maskOp >= 0)
                    throw new ValueError(
                        "An iterator operand was flagged as the ARRAYMASK, but no WRITEMASKED operands were given to use the mask");
                return descrs;
            }

            /// <summary>
            ///     The broadcast check of NumPy's <c>npyiter_fill_axisdata</c> for the plain form (no <c>op_axes</c>, no
            ///     <c>itershape</c>): the operands' shapes, right-aligned, must agree wherever neither extent is 1 — raised with
            ///     NumPy's text, which lists every operand's shape followed by a space.
            /// </summary>
            /// <param name="op">The operands (null entries are outputs to allocate and take no part).</param>
            /// <param name="nop">The operand count.</param>
            /// <exception cref="ValueError"><c>operands could not be broadcast together with shapes (3,) (4,) </c>.</exception>
            /// <remarks>
            ///     The engine checks the same broadcast again with its own (house) wording; this check runs first, at NumPy's
            ///     point in the construction — after every per-operand check, before the common dtype.
            /// </remarks>
            private static void CheckBroadcast(NDArray[] op, int nop)
            {
                int nd = 0;
                for (int i = 0; i < nop; i++)
                    if (op[i] is not null && op[i].ndim > nd)
                        nd = op[i].ndim;

                // The running broadcast extents; on the stack for any common rank (no NPY_MAXDIMS cap, so the heap past it).
                Span<long> b = nd <= 32 ? stackalloc long[nd] : new long[nd];
                for (int d = 0; d < nd; d++)
                    b[d] = 1;
                bool ok = true;
                for (int i = 0; i < nop && ok; i++)
                {
                    if (op[i] is null)
                        continue;
                    var dims = op[i].Shape.dimensions;
                    int shift = nd - dims.Length;
                    for (int k = 0; k < dims.Length; k++)
                    {
                        long e = dims[k];
                        if (e == 1)
                            continue;
                        if (b[shift + k] == 1)
                            b[shift + k] = e;
                        else if (b[shift + k] != e)
                        {
                            ok = false;
                            break;
                        }
                    }
                }
                if (ok)
                    return;

                var sb = new System.Text.StringBuilder("operands could not be broadcast together with shapes ");
                for (int i = 0; i < nop; i++)
                    if (op[i] is not null)
                        sb.Append(op[i].Shape.ToPythonTuple()).Append(' ');
                throw new ValueError(sb.ToString());
            }

            /// <summary>NumPy's <c>npyiter_check_per_op_flags</c>, in its order and with its texts.</summary>
            /// <param name="f">The operand's flags.</param>
            /// <param name="read">The operand is read.</param>
            /// <param name="write">The operand is written.</param>
            /// <exception cref="ValueError">Conflicting or missing read/write modes; <c>copy</c> on a written operand;
            ///     WRITEMASKED without a write mode or with ARRAYMASK; VIRTUAL without READWRITE.</exception>
            private static void CheckPerOpFlags(NDIterPerOpFlags f, out bool read, out bool write)
            {
                const string OnlyOne =
                    "Only one of the iterator flags READWRITE, READONLY, and WRITEONLY may be specified for an operand";
                if ((f & NDIterPerOpFlags.READONLY) != 0)
                {
                    if ((f & (NDIterPerOpFlags.READWRITE | NDIterPerOpFlags.WRITEONLY)) != 0)
                        throw new ValueError(OnlyOne);
                    read = true;
                    write = false;
                }
                else if ((f & NDIterPerOpFlags.READWRITE) != 0)
                {
                    if ((f & NDIterPerOpFlags.WRITEONLY) != 0)
                        throw new ValueError(OnlyOne);
                    read = true;
                    write = true;
                }
                else if ((f & NDIterPerOpFlags.WRITEONLY) != 0)
                {
                    read = false;
                    write = true;
                }
                else
                {
                    throw new ValueError(
                        "None of the iterator flags READWRITE, READONLY, or WRITEONLY were specified for an operand");
                }

                if (write && (f & (NDIterPerOpFlags.COPY | NDIterPerOpFlags.UPDATEIFCOPY)) == NDIterPerOpFlags.COPY)
                    throw new ValueError("If an iterator operand is writeable, must use the flag UPDATEIFCOPY instead of COPY");
                if ((f & NDIterPerOpFlags.WRITEMASKED) != 0)
                {
                    if (!write)
                        throw new ValueError("The iterator flag WRITEMASKED may only be used with READWRITE or WRITEONLY");
                    if ((f & NDIterPerOpFlags.ARRAYMASK) != 0)
                        throw new ValueError("The iterator flag WRITEMASKED may not be used together with ARRAYMASK");
                }
                // NumPy's message carries a doubled "be" — matched verbatim.
                if ((f & NDIterPerOpFlags.VIRTUAL) != 0 && (f & NDIterPerOpFlags.READWRITE) == 0)
                    throw new ValueError("The iterator flag VIRTUAL should be be used together with READWRITE");
            }

            /// <summary>
            ///     NumPy's <c>PyArray_AdaptDescriptorToArray(arr, NULL, request)</c>: a sized request is used as given; an unsized
            ///     string request (<c>'S'</c>, <c>'U'</c>) stands for its class and becomes the instance a cast from the array's
            ///     descriptor resolves to (<see cref="DTypePromotion.CastDescrToDType"/>).
            /// </summary>
            /// <param name="own">The operand's descriptor.</param>
            /// <param name="request">The requested descriptor.</param>
            /// <returns>The iteration descriptor.</returns>
            /// <exception cref="TypeError">No cast exists from the operand's class to the requested one.</exception>
            private static DType AdaptDescriptorToArray(DType own, DType request)
            {
                if (request.itemsize == 0 && request.Meta is StringDTypeMeta)
                    return DTypePromotion.CastDescrToDType(own, request.Meta);
                return request;
            }

            /// <summary>
            ///     NumPy's <c>npyiter_get_common_dtype</c> and its use in <c>NpyIter_AdvancedNew</c>: when an allocated operand
            ///     has no dtype yet, or <c>common_dtype</c> is set, the result type of the operands' descriptors — only the read
            ///     operands unless <c>common_dtype</c> — fills the missing descriptors, or replaces every descriptor under
            ///     <c>common_dtype</c>.
            /// </summary>
            /// <param name="op">The operands.</param>
            /// <param name="perOpFlags">The per-operand flags.</param>
            /// <param name="descrs">The descriptors from <see cref="PrepareOperands"/> (updated in place).</param>
            /// <param name="requests">The requested dtypes, or null.</param>
            /// <param name="globalFlags">The global flags.</param>
            /// <param name="nop">The operand count.</param>
            /// <exception cref="TypeError"><c>no arrays or types available to calculate result type</c>: nothing contributes a
            ///     dtype (an iterator of outputs only, none of them with a requested dtype).</exception>
            /// <remarks>
            ///     The promotion is NEP 50's: a 0-d operand without a request is passed as an ARRAY, which under NEP 50 counts
            ///     only by its dtype, so the answer is the plain result type of the descriptors (<c>U3</c> with int64 is
            ///     <c>&lt;U21</c>, <c>|S2</c> with <c>&lt;U1</c> is <c>&lt;U2</c>).
            /// </remarks>
            private static void ResolveCommonDtype(NDArray[] op, NDIterPerOpFlags[] perOpFlags, DType[] descrs, DType[] requests,
                                                   NDIterGlobalFlags globalFlags, int nop)
            {
                bool common = (globalFlags & NDIterGlobalFlags.COMMON_DTYPE) != 0;
                bool anyMissing = false;
                for (int i = 0; i < nop; i++)
                    anyMissing |= op[i] is null && descrs[i] is null;
                if (!anyMissing && !common)
                    return;

                bool onlyInputs = !common;

                // Fast path: every contribution is the same builtin descriptor (the common case — np.nditer([a, b, None])
                // over arrays of one dtype). result_type of identical builtin descriptors is that descriptor, so the
                // promotion machinery (a list, an array and the promotion fold: ~80 ns) is skipped. A 0-d operand
                // contributes its dtype as a strong value (NEP 50), the same descriptor.
                DType same = null;
                bool allSame = true;
                bool anyContribution = false;
                for (int i = 0; i < nop && allSame; i++)
                {
                    if (descrs[i] is null)
                        continue;
                    bool read = (perOpFlags[i] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0;
                    if (onlyInputs && !read)
                        continue;
                    var d = descrs[i];
                    anyContribution = true;
                    if (same is null)
                        same = d;
                    else if (!ReferenceEquals(same, d))
                        allSame = false;
                }

                if (anyContribution && allSame && same.isbuiltin == 1)
                {
                    for (int i = 0; i < nop; i++)
                    {
                        if (common || descrs[i] is null)
                            descrs[i] = same;
                    }

                    return;
                }

                var contributions = new List<object>(nop);
                for (int i = 0; i < nop; i++)
                {
                    if (descrs[i] is null)
                        continue;
                    bool read = (perOpFlags[i] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0;
                    if (onlyInputs && !read)
                        continue;
                    // NumPy passes an unrequested 0-d operand as the array itself; under NEP 50 that is its dtype.
                    if (requests?[i] is null && op[i] is not null && op[i].ndim == 0)
                        contributions.Add(op[i]);
                    else
                        contributions.Add(descrs[i]);
                }

                if (contributions.Count == 0)
                    throw new TypeError("no arrays or types available to calculate result type");

                var result = DTypePromotion.ResultType(contributions.ToArray());
                for (int i = 0; i < nop; i++)
                {
                    if (common || descrs[i] is null)
                        descrs[i] = result;
                }
            }

            /// <summary>NumPy's <c>PyArray_OrderConverter</c> for the iteration order: the four letters in EITHER case.</summary>
            /// <param name="order">The order character.</param>
            /// <returns>The order.</returns>
            /// <exception cref="ValueError"><c>order must be one of 'C', 'F', 'A', or 'K' (got '…')</c>.</exception>
            /// <remarks>It refused the lowercase letters, which NumPy's parser accepts (probed 2.4.2:
            ///     <c>np.nditer(a, order='f')</c> iterates in F order).</remarks>
            private static NPY_ORDER ParseOrder(char order) => OrderResolver.Convert(order);

            /// <summary>NumPy's <c>PyArray_CastingConverter</c> for the casting rule.</summary>
            /// <param name="casting">The casting string.</param>
            /// <returns>The rule.</returns>
            /// <exception cref="ValueError"><c>casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')</c>.</exception>
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
                        throw new ValueError(
                            $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
                }
            }

            // ---------------------------------------------------------------
            // Borrowing
            // ---------------------------------------------------------------

            /// <summary>
            ///     A non-owning <see cref="NDIterRef"/> over this iterator's state — every member that NumPy guards with
            ///     <c>self->iter == NULL</c> starts here.
            /// </summary>
            /// <returns>The borrowed iterator (valid until the next member call that changes the state).</returns>
            /// <exception cref="ValueError">The iterator is closed: NumPy's "Iterator is invalid".</exception>
            private NDIterRef Borrow()
            {
                if (_state == null)
                    throw new ValueError("Iterator is invalid");

                return NDIterRef.Borrow(_state, _operands, _cachedNext);
            }

            /// <summary>The text NumPy raises while a <c>delay_bufalloc</c> iterator has not been reset.</summary>
            private const string DelayedBufAllocText =
                "Iterator construction used delayed buffer allocation, and no reset has been done yet";

            /// <summary>
            ///     Whether buffer allocation is still pending a reset — the engine's <c>DELAYBUF</c> flag, or the wrapper's
            ///     own flag for a buffered reduction (<see cref="_reduceDelayPending"/>). Requires an open iterator.
            /// </summary>
            private bool DelayedBufAlloc =>
                (_state->ItFlags & (uint)NDIterFlags.DELAYBUF) != 0 || _reduceDelayPending;

            /// <summary>
            ///     NumPy's <c>self->iter == NULL || self->finished</c> guard of the position getters and the element views.
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed or has run past the end: "Iterator is past the end".</exception>
            private void ThrowIfPastTheEnd()
            {
                if (_state == null || _finished)
                    throw new ValueError("Iterator is past the end");
            }

            /// <summary>
            ///     NumPy's <c>npyiter_cache_values</c> for the one value the wrapper caches: whether the multi-index getter
            ///     exists (a multi-index is tracked and no buffer allocation is pending). Run at construction, after a copy,
            ///     and after every member that changes the axes (<see cref="remove_axis"/>, <see cref="remove_multi_index"/>,
            ///     <see cref="enable_external_loop"/>).
            /// </summary>
            private void CacheValues()
                => _hasGetMultiIndex = (_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0 && !DelayedBufAlloc;

            // ---------------------------------------------------------------
            // Properties (NumPy's nditer getters)
            // ---------------------------------------------------------------

            /// <summary>The number of operands (NumPy's <c>nop</c>).</summary>
            public int nop => Borrow().NOp;

            /// <summary>The number of dimensions iterated (NumPy's <c>ndim</c>).</summary>
            public int ndim => Borrow().NDim;

            /// <summary>
            ///     The shape being iterated (NumPy's <c>shape</c>). Without <c>multi_index</c> the
            ///     iterator is free to coalesce and reorder axes, so this is the COALESCED shape listed
            ///     INNERMOST FIRST, NumPy's axisdata order (probed: a C-contiguous (2,3) reports <c>(6,)</c>,
            ///     an <c>f_index</c> walk of it — which cannot coalesce — <c>(3, 2)</c>); with <c>multi_index</c>
            ///     it is the operands' own shape in their axis order, <c>(2, 3)</c>.
            /// </summary>
            /// <remarks>
            ///     Code that reads the shape of an iterator without a multi-index gets the iteration's axes, not the array's:
            ///     a forced F-order walk of a C <c>(3, 4)</c> array reports <c>(3, 4)</c> (axis 0 is its inner axis), a C-order
            ///     walk of an F one reports <c>(4, 3)</c>.
            /// </remarks>
            /// <exception cref="ValueError">The iterator is closed or past the end — NumPy's <c>npyiter_shape_get</c>
            ///     guards it like a position ("Iterator is past the end"), so an empty iteration has no shape either.</exception>
            public long[] shape
            {
                get
                {
                    ThrowIfPastTheEnd();
                    var it = Borrow();
                    var internalShape = it.Shape;
                    if (it.HasMultiIndex)
                        return internalShape;

                    // NpyIter_GetShape without a multi-index lists the iterator's OWN axes in its axisdata order, innermost
                    // first ((4, 3) for a C (3, 4) iteration that did not coalesce, (3, 2) for a[::2, ::2] of a (4, 6)
                    // array). The engine keeps its axes outermost first, so the order is reversed here. With a multi-index
                    // the shape is the operands' own, which the engine already reports.
                    Array.Reverse(internalShape);
                    return internalShape;
                }
            }

            /// <summary>Total number of elements the iterator will visit (NumPy's <c>itersize</c>).</summary>
            public long itersize => Borrow().IterSize;

            /// <summary>
            ///     The operands, including any the iterator ALLOCATED and the temporaries of its copy routes (NumPy's
            ///     <c>operands</c>). Each read returns a new array of the same operand objects, as NumPy returns a new tuple,
            ///     so replacing an entry cannot change what the iterator walks.
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid"): the copy-route temporaries were
            ///     written back and released by <see cref="close"/>.</exception>
            public NDArray[] operands
            {
                get
                {
                    Borrow();
                    return (NDArray[])_operands.Clone();
                }
            }

            /// <summary>
            ///     The per-operand iteration dtypes (NumPy's <c>dtypes</c>): the descriptor each operand is iterated as —
            ///     a requested dtype adapted to its operand (<c>'U'</c> over a <c>&lt;U3</c> array is <c>&lt;U3</c>), the common
            ///     dtype, or the operand's own. A string reports its width and class (<c>dtype('&lt;U5')</c>), which a type code
            ///     cannot.
            /// </summary>
            /// <remarks>
            ///     An allocated output keeps its request as given, as NumPy does: <c>np.nditer([u, null], op_dtypes: [null,
            ///     "U"])</c> reports <c>dtype('&lt;U')</c> for the output although the array is allocated <c>&lt;U1</c>. A
            ///     non-native request (<c>'&gt;U3'</c>) is reported as requested although the buffer holds native data.
            /// </remarks>
            /// <exception cref="InvalidOperationException">The iterator is closed.</exception>
            public DType[] dtypes
            {
                get
                {
                    Borrow();
                    return (DType[])_iterDescrs.Clone();
                }
            }

            /// <summary>
            ///     True once iteration has run past the end (NumPy's <c>finished</c>): after the pull or
            ///     <see cref="iternext"/> that found no next element, and from the start for an empty iteration or range.
            ///     A seek (<see cref="iterindex"/>, <see cref="multi_index"/>, <see cref="index"/>) clears it.
            /// </summary>
            /// <remarks>A CLOSED iterator reads false, as NumPy's does (its getter tests the iterator before the flag).</remarks>
            public bool finished => _state != null && _finished;

            /// <summary>Whether a C- or F-order flat index is being tracked (NumPy's <c>has_index</c>).</summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            public bool has_index
            {
                get
                {
                    Borrow();
                    return (_state->ItFlags & (uint)NDIterFlags.HASINDEX) != 0;
                }
            }

            /// <summary>Whether a multi-index is being tracked (NumPy's <c>has_multi_index</c>).</summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            public bool has_multi_index
            {
                get
                {
                    Borrow();
                    return (_state->ItFlags & (uint)NDIterFlags.HASMULTIINDEX) != 0;
                }
            }

            /// <summary>
            ///     Whether buffer allocation is still delayed pending a <see cref="reset"/> (NumPy's
            ///     <c>has_delayed_bufalloc</c>). While it is, the element views and <see cref="multi_index"/> refuse — NumPy's
            ///     <c>delay_bufalloc</c> exists so an allocated output can be initialized before the buffers first read it.
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            public bool has_delayed_bufalloc
            {
                get
                {
                    Borrow();
                    return DelayedBufAlloc;
                }
            }

            /// <summary>
            ///     Whether iteration needs the Python C-API (NumPy's <c>iterationneedsapi</c>).
            ///     Always FALSE in NumSharp — there is no Python runtime, and the transfer flags
            ///     never carry <c>REQUIRES_PYAPI</c>.
            /// </summary>
            public bool iterationneedsapi
                => (Borrow().GetTransferFlags() & NDArrayMethodFlags.REQUIRES_PYAPI) != 0;

            /// <summary>
            ///     The tracked flat index (NumPy's <c>index</c>) — the position in C or F order of the operands' own shape,
            ///     whatever order the iterator walks. Requires the <c>c_index</c> or <c>f_index</c> flag. Setting it moves the
            ///     iterator there (NumPy's <c>NpyIter_GotoIndex</c>): the next pull yields that element.
            /// </summary>
            /// <exception cref="ValueError">Reading: the iterator is closed or past the end ("Iterator is past the end"), or no
            ///     index is tracked ("Iterator does not have an index"). Setting: the iterator is closed ("Iterator is
            ///     invalid"), no index is tracked, or it is buffered / external-loop (NumPy's "Cannot call GotoIndex on an
            ///     iterator which …" texts).</exception>
            /// <exception cref="IndexError">Setting: an index outside <c>[0, itersize)</c> ("Iterator GotoIndex called with an
            ///     out-of-bounds index"), or one outside the current <see cref="iterrange"/>.</exception>
            public long index
            {
                get
                {
                    ThrowIfPastTheEnd();
                    var it = Borrow();
                    if (!it.HasIndex)
                        throw new ValueError("Iterator does not have an index");
                    return it.GetIndex();
                }
                set
                {
                    var it = Borrow();
                    if (!it.HasIndex)
                        throw new ValueError("Iterator does not have an index");

                    // NpyIter_GotoIndex's checks, in its order; the range check needs the target position BEFORE moving.
                    uint itflags = _state->ItFlags;
                    if ((itflags & (uint)NDIterFlags.BUFFER) != 0)
                        throw new ValueError("Cannot call GotoIndex on an iterator which is buffered");
                    if ((itflags & (uint)NDIterFlags.EXLOOP) != 0)
                        throw new ValueError("Cannot call GotoIndex on an iterator which has the flag EXTERNAL_LOOP");
                    if (value < 0 || value >= it.IterSize)
                        throw new IndexError("Iterator GotoIndex called with an out-of-bounds index");
                    long target = it.IterIndexOfIndex(value);
                    if (target < it.IterStart || target >= it.IterEnd)
                        throw new IndexError("Iterator GotoIndex called with an index outside the restricted iteration range.");

                    it.GotoIndex(value);
                    _cachedNext = it.PeekCachedIterNext();
                    // npyiter_index_set: the next pull yields the element at the new position.
                    _started = _finished = false;
                    if (_nestedChild != null)
                        RebaseChildren();
                }
            }

            /// <summary>
            ///     The tracked multi-index (NumPy's <c>multi_index</c>) in the operands' own axis order. Requires the
            ///     <c>multi_index</c> flag. Setting it moves the iterator there (NumPy's <c>NpyIter_GotoMultiIndex</c>): the
            ///     next pull yields that element.
            /// </summary>
            /// <exception cref="ValueError">Reading: the iterator is closed or past the end ("Iterator is past the end"), no
            ///     multi-index is tracked, or a <c>delay_bufalloc</c> iterator has not been reset (NumPy's texts). Setting:
            ///     the iterator is closed ("Iterator is invalid"), no multi-index is tracked, a null value ("multi_index must
            ///     be set with a sequence"), a value whose length is not <see cref="ndim"/> ("Wrong number of indices"), or a
            ///     buffered iterator ("Cannot call GotoMultiIndex on an iterator which is buffered").</exception>
            /// <exception cref="IndexError">Setting: a coordinate outside its axis ("Iterator GotoMultiIndex called with an
            ///     out-of-bounds multi-index"), or a position outside the current <see cref="iterrange"/>.</exception>
            public long[] multi_index
            {
                get
                {
                    ThrowIfPastTheEnd();
                    var it = Borrow();
                    if (_hasGetMultiIndex)
                    {
                        var coords = new long[it.NDim];
                        it.GetMultiIndex(coords);
                        return coords;
                    }

                    // npyiter_multi_index_get without a getter: say why there is none.
                    if (!it.HasMultiIndex)
                        throw new ValueError("Iterator is not tracking a multi-index");
                    if (DelayedBufAlloc)
                        throw new ValueError(DelayedBufAllocText);
                    throw new ValueError("Iterator is in an invalid state");
                }
                set
                {
                    var it = Borrow();
                    if (!it.HasMultiIndex)
                        throw new ValueError("Iterator is not tracking a multi-index");
                    if (value == null)
                        throw new ValueError("multi_index must be set with a sequence");
                    if (value.Length != it.NDim)
                        throw new ValueError("Wrong number of indices");

                    // NpyIter_GotoMultiIndex's checks, in its order (the external-loop one is unreachable: construction and
                    // enable_external_loop both refuse a tracked multi-index).
                    uint itflags = _state->ItFlags;
                    if ((itflags & (uint)NDIterFlags.BUFFER) != 0)
                        throw new ValueError("Cannot call GotoMultiIndex on an iterator which is buffered");
                    if ((itflags & (uint)NDIterFlags.EXLOOP) != 0)
                        throw new ValueError("Cannot call GotoMultiIndex on an iterator which has the flag EXTERNAL_LOOP");
                    long target = it.IterIndexOfMultiIndex(value);
                    if (target < 0)
                        throw new IndexError("Iterator GotoMultiIndex called with an out-of-bounds multi-index");
                    if (target < it.IterStart || target >= it.IterEnd)
                        throw new IndexError(
                            "Iterator GotoMultiIndex called with a multi-index outside the restricted iteration range");

                    it.GotoMultiIndex(value);
                    _cachedNext = it.PeekCachedIterNext();
                    // npyiter_multi_index_set: the next pull yields the element at the new position.
                    _started = _finished = false;
                    if (_nestedChild != null)
                        RebaseChildren();
                }
            }

            /// <summary>
            ///     The current flat iteration position (NumPy's <c>iterindex</c>), counted in the iterator's own order.
            ///     Setting it moves the iterator there (NumPy's <c>NpyIter_GotoIterIndex</c>; a buffered iterator writes its
            ///     window back and fills the one at the new position): the next pull yields that element.
            /// </summary>
            /// <exception cref="ValueError">Reading: the iterator is closed or past the end ("Iterator is past the end").
            ///     Setting: the iterator is closed ("Iterator is invalid"), or it is external-loop ("Cannot call GotoIterIndex
            ///     on an iterator which has the flag EXTERNAL_LOOP").</exception>
            /// <exception cref="IndexError">Setting: a position outside the current <see cref="iterrange"/> ("Iterator
            ///     GotoIterIndex called with an iterindex outside the iteration range.").</exception>
            public long iterindex
            {
                get
                {
                    ThrowIfPastTheEnd();
                    return Borrow().IterIndex;
                }
                set
                {
                    var it = Borrow();
                    if ((_state->ItFlags & (uint)NDIterFlags.EXLOOP) != 0)
                        throw new ValueError("Cannot call GotoIterIndex on an iterator which has the flag EXTERNAL_LOOP");
                    if (value < it.IterStart || value >= it.IterEnd)
                        throw new IndexError("Iterator GotoIterIndex called with an iterindex outside the iteration range.");

                    it.GotoIterIndex(value);
                    _cachedNext = it.PeekCachedIterNext();
                    // NumPy's npyiter_iterindex_set: the next pull yields the element at the new index.
                    _started = _finished = false;
                    if (_nestedChild != null)
                        RebaseChildren();
                }
            }

            /// <summary>
            ///     The <c>[start, end)</c> sub-range being iterated (NumPy's <c>iterrange</c>). Setting it requires the
            ///     <c>ranged</c> flag, and resets the iterator to the start of the new range (NumPy's
            ///     <c>NpyIter_ResetToIterIndexRange</c>, which also allocates delayed buffers); an empty range is finished at
            ///     once.
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid"); setting without the <c>ranged</c>
            ///     flag, a range outside <c>[0, itersize]</c> ("Out-of-bounds range [s, e) passed to ResetToIterIndexRange"),
            ///     or an end before the start ("Invalid range [s, e) passed to ResetToIterIndexRange").</exception>
            public (long Start, long End) iterrange
            {
                get => Borrow().IterRange;
                set
                {
                    var it = Borrow();
                    if ((_state->ItFlags & (uint)NDIterFlags.RANGE) == 0)
                        throw new ValueError(
                            "Cannot call ResetToIterIndexRange on an iterator without requesting ranged iteration " +
                            "support in the constructor");
                    if (value.Start < 0 || value.End > it.IterSize)
                        throw new ValueError(
                            $"Out-of-bounds range [{value.Start}, {value.End}) passed to ResetToIterIndexRange");
                    if (value.End < value.Start)
                        throw new ValueError($"Invalid range [{value.Start}, {value.End}) passed to ResetToIterIndexRange");

                    // The engine sets the range and resets to its start (NpyIter_Reset: a windowed buffer is written back
                    // and refilled for the new range, delayed buffers are allocated).
                    if (!it.ResetToIterIndexRange(value.Start, value.End))
                        throw new OutOfMemoryException("Could not allocate the iterator's delayed buffers");
                    _reduceDelayPending = false;

                    _cachedNext = it.PeekCachedIterNext();
                    // NumPy's npyiter_iterrange_set: an empty range is finished at once, any other starts fresh, and the
                    // multi-index getter exists from now on.
                    _started = _finished = value.Start >= value.End;
                    if (!_hasGetMultiIndex && it.HasMultiIndex)
                        _hasGetMultiIndex = true;
                    if (_nestedChild != null)
                        RebaseChildren();
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
            /// <exception cref="ValueError">The iterator is closed or past the end ("Iterator is past the end"), or a
            ///     <c>delay_bufalloc</c> iterator has not been reset.</exception>
            public NDArray[] value
            {
                get
                {
                    // npyiter_value_get guards the position first, then builds each view (npyiter_seq_item).
                    ThrowIfPastTheEnd();
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
            ///     the view is invalidated by the next step and by disposal. A negative index counts from the last operand.
            ///     Setting it copies a value into the current element(s) — see the setter's remarks.
            /// </summary>
            /// <param name="i">The operand index (negative counts from the end).</param>
            /// <returns>The view; read-only unless the operand is written (<c>readwrite</c> / <c>writeonly</c>), as NumPy's
            ///     views are (its write flags).</returns>
            /// <exception cref="ValueError">The iterator is closed or past the end ("Iterator is past the end"), or a
            ///     <c>delay_bufalloc</c> iterator has not been reset (NumPy's text).</exception>
            /// <exception cref="IndexError">No operand <paramref name="i"/>: "Iterator operand index i is out of
            ///     bounds".</exception>
            /// <exception cref="RuntimeError">Setting a read-only operand: "Iterator operand i is not writeable".</exception>
            /// <exception cref="ArgumentNullException">Setting null (NumPy's <c>None</c> fails converting to the dtype; a
            ///     C# null is refused before that).</exception>
            /// <exception cref="OverflowException">Setting a 0-d integer value that does not fit the operand's dtype (NumPy's
            ///     OverflowError texts).</exception>
            /// <remarks>
            ///     The setter is NumPy's <c>npyiter_seq_ass_item</c>: the value is copied into a 1-D view of the current
            ///     inner loop (one element without <c>external_loop</c>). A 0-d value is a SCALAR there (<c>PyArray_Pack</c>:
            ///     a C# number is NumPy's Python scalar — range-checked, a float truncated into an integer operand, a string
            ///     parsed), so <c>it[1] = 300</c> on an int8 operand raises like NumPy's; a value with dimensions is assigned
            ///     with unsafe casting and broadcast to the view ("could not broadcast input array from shape (2,) into shape
            ///     (1,)"). NumSharp cannot tell an explicit 0-d array from a literal, so <c>np.array(300)</c> also raises
            ///     where NumPy's 0-d ndarray would wrap.
            /// </remarks>
            public NDArray this[int i]
            {
                get
                {
                    int op = CheckedOperandIndex(i);
                    var it = Borrow();

                    void* ptr = _state->GetDataPtr(op);
                    var dtype = _state->GetOpDType(op);
                    // A fixed-width string view needs its width and class: the descriptor of the side the pointer is on (the
                    // buffer under a buffered cast, the operand otherwise — the same descriptor without one).
                    var stringDescr = dtype == NPTypeCode.String ? _state->GetOpDescr(op) : null;
                    NDArray view;
                    if (!it.HasExternalLoop)
                    {
                        view = AliasView(ptr, dtype, stringDescr, 1, 1, scalar: true);
                    }

                    // A 0-d operand has NO inner dimension, and GetInnerLoopSizePtr reads
                    // Shape[NDim - 1] — i.e. Shape[-1] — which is an ACCESS VIOLATION, not an
                    // exception: it takes the process down. This is the same trap
                    // NDIterTyped.ReadInnerLoop absorbs (see np.nditer.Typed.cs); the public
                    // indexer has to absorb it too. Probed: NumPy yields ONE 1-element 1-D chunk
                    // here (np.nditer(np.array(3), flags=['external_loop']) -> array([3])).
                    else if (it.NDim == 0)
                    {
                        view = AliasView(ptr, dtype, stringDescr, 1, 1, scalar: false);
                    }
                    else
                    {
                        long count = *it.GetInnerLoopSizePtr();
                        long stride = it.GetInnerLoopElementStride(op);
                        view = AliasView(ptr, dtype, stringDescr, count, stride, scalar: false);
                    }

                    // NumPy's views carry the operand's write flag (PyArray_NewFromDescrAndBase with NPY_ARRAY_WRITEABLE or
                    // nothing): a read-only operand's view refuses assignment, and a written operand's view is writeable
                    // even where its stride is 0 — the external-loop chunk of a reduction operand along a reduced axis,
                    // which NumSharp's shape flags as a read-only broadcast. Writing it element by element accumulates
                    // into the one slot, which is how NumPy's reduction loops use it.
                    if ((_state->GetOpFlags(op) & NDIterOpFlags.WRITE) == 0)
                        MarkReadOnly(view);
                    else if (!view.Shape.IsWriteable)
                        view.Storage.SetShapeUnsafe(view.Shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE));
                    return view;
                }
                set
                {
                    int op = CheckedOperandIndex(i);
                    if ((_state->GetOpFlags(op) & NDIterOpFlags.WRITE) == 0)
                        throw new RuntimeError($"Iterator operand {i} is not writeable");
                    if (value is null)
                        throw new ArgumentNullException(nameof(value));

                    var it = Borrow();
                    void* ptr = _state->GetDataPtr(op);
                    var dtype = _state->GetOpDType(op);
                    var stringDescr = dtype == NPTypeCode.String ? _state->GetOpDescr(op) : null;

                    // npyiter_seq_ass_item's temporary: a 1-D view of the current inner loop — one element without
                    // external_loop, and a 0-d iteration has no inner axis to read (GetInnerLoopSizePtr would read Shape[-1]).
                    long count = 1, stride = 1;
                    if (it.HasExternalLoop && it.NDim != 0)
                    {
                        count = *it.GetInnerLoopSizePtr();
                        stride = it.GetInnerLoopElementStride(op);
                    }

                    using var target = AliasView(ptr, dtype, stringDescr, count, stride, scalar: false);
                    AssignToIteratorView(target, value);
                }
            }

            /// <summary>
            ///     NumPy's operand-index checks of <c>it[i]</c> (<c>npyiter_seq_item</c> / <c>npyiter_seq_ass_item</c>), in
            ///     its order: the position, the delayed allocation, then the index.
            /// </summary>
            /// <param name="i">The caller's index (negative counts from the end).</param>
            /// <returns>The operand index in <c>[0, nop)</c>.</returns>
            /// <exception cref="ValueError">"Iterator is past the end" (closed or finished), or the delayed-allocation
            ///     text.</exception>
            /// <exception cref="IndexError">"Iterator operand index i is out of bounds" — with the index AS GIVEN.</exception>
            private int CheckedOperandIndex(int i)
            {
                ThrowIfPastTheEnd();
                if (DelayedBufAlloc)
                    throw new ValueError(DelayedBufAllocText);

                int nop = _state->NOp;
                int op = i < 0 ? i + nop : i;
                if (op < 0 || op >= nop)
                    throw new IndexError($"Iterator operand index {i} is out of bounds");
                return op;
            }

            /// <summary>
            ///     Makes a view of an operand the iterator does not write read-only, as NumPy's is (its write flags), for
            ///     good: the storage is marked foreign read-only, so <c>flags.writeable = true</c> refuses it — NumPy's view
            ///     has the iterator, not an array, for its base, which cannot grant write access.
            /// </summary>
            /// <param name="view">A view <see cref="AliasView"/> just built (its storage is its own, not the operand's).</param>
            private static void MarkReadOnly(NDArray view)
            {
                view.Storage.WriteProtected = true;
                view.Storage.SetShapeUnsafe(view.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            }

            /// <summary>
            ///     <c>PyArray_CopyObject(view, value)</c> for the iterator's setter: a 0-d numeric or text value is packed as a
            ///     SCALAR (NumPy's <c>PyArray_Pack</c>, range-checked) and stored into every element of the view; anything
            ///     else — a value with dimensions, or a text view — is assigned with unsafe casting and NumPy's broadcasting.
            /// </summary>
            /// <param name="view">The writeable 1-D view of the current inner loop.</param>
            /// <param name="value">The value (not null).</param>
            /// <exception cref="OverflowException">A 0-d integer that does not fit the view's dtype.</exception>
            /// <exception cref="ValueError">A text that does not parse into a numeric view, a NaN into an integer view, or a
            ///     value that does not broadcast to the view.</exception>
            /// <exception cref="TypeError">A complex into a real view.</exception>
            private static void AssignToIteratorView(NDArray view, NDArray value)
            {
                if (value.ndim == 0 && view.typecode != NPTypeCode.String)
                {
                    byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
                    ReductionInitial.TryPack(value.GetAtIndex(0), view.typecode, new Span<byte>(slot, ReductionInitial.SlotBytes));
                    CopytoAssignRawScalar(view, slot, view.typecode, null, NPY_CASTING.NPY_UNSAFE_CASTING);
                    return;
                }

                // PyArray_AssignArray(dest, src, NULL, NPY_UNSAFE_CASTING): np.copyto's broadcasting and texts.
                copyto(view, value, casting: "unsafe");
            }

            // ---------------------------------------------------------------
            // Methods (NumPy's nditer methods)
            // ---------------------------------------------------------------

            /// <summary>
            ///     Advance to the next element/chunk (NumPy's <c>iternext()</c>). Returns false at the end — and for a closed
            ///     or finished iterator, without advancing (NumPy's <c>npyiter_iternext</c> raises nothing there).
            /// </summary>
            /// <returns>True when the iterator moved to another element; false when it is (now) finished.</returns>
            public bool iternext()
            {
                if (_state != null && !_finished)
                {
                    var it = Borrow();
                    bool more = it.Iternext();
                    _cachedNext = it.PeekCachedIterNext();
                    if (more)
                    {
                        if (_nestedChild != null)
                            RebaseChildren();   // NumPy: npyiter_iternext re-bases the nested chain after each advance
                        return true;
                    }
                }

                _finished = true;
                return false;
            }

            // Nested-iteration members (SetupNested / RebaseChildren / MoveNextNested) live in the
            // np.nditer.Nested.cs partial, next to np.nested_iters — this file keeps only the fields
            // and the three guarded hooks (in MoveNext / iternext / reset), all no-ops unless nested.

            /// <summary>
            ///     Rewind to the start of the range (NumPy's <c>reset()</c>); also allocates delayed buffers — the reset a
            ///     <c>delay_bufalloc</c> iterator needs before its elements can be read. A buffered window already at the
            ///     start is left as it is (NumPy's <c>NpyIter_Reset</c>).
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            /// <exception cref="OutOfMemoryException">Delayed buffers could not be allocated.</exception>
            public void reset()
            {
                var it = Borrow();
                if (!it.Reset())
                    throw new OutOfMemoryException("Could not allocate the iterator's delayed buffers");
                _reduceDelayPending = false;
                _cachedNext = it.PeekCachedIterNext();
                // npyiter_reset: an empty iteration is finished again at once, and the multi-index getter exists now.
                _started = _finished = it.IterSize == 0;
                if (!_hasGetMultiIndex && it.HasMultiIndex)
                    _hasGetMultiIndex = true;
                if (_nestedChild != null)
                    RebaseChildren();       // NumPy: npyiter_reset re-bases the nested chain
            }

            /// <summary>
            ///     Duplicate the iterator at its current position (NumPy's <c>copy()</c>). The copy
            ///     owns its own state and must be disposed independently.
            /// </summary>
            /// <returns>The copy, carrying NumPy's started / finished flags (so it resumes where the original would).</returns>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            public NDIterator copy()
            {
                var copied = Borrow().Copy();
                // The copy iterates the same operands as the same descriptors (it owns no write-backs: those resolve when
                // the original closes, as the copy shares the original's temporaries).
                // NumPy's npyiter_copy carries the started / finished flags too, so a copy taken after one pull resumes
                // with the SECOND element, as the original would.
                var duplicate = new NDIterator(copied)
                {
                    _iterDescrs = (DType[])_iterDescrs.Clone(),
                    _started = _started,
                    _finished = _finished,
                    _reduceDelayPending = _reduceDelayPending,
                };
                // npyiter_copy runs npyiter_cache_values on the copy: no multi-index getter while allocation is delayed.
                duplicate.CacheValues();
                return duplicate;
            }

            /// <summary>Private ctor adopting an already-built NDIterRef (used by <see cref="copy"/>).</summary>
            /// <param name="iter">The built iterator (its state is detached into this instance).</param>
            private NDIterator(NDIterRef iter)
            {
                _state = iter.Detach(out var operands, out var writebacks, out var releaseOnly);
                _operands = operands;
                _writebackOriginals = writebacks;
                _releaseOnlyOriginals = releaseOnly;
            }

            /// <summary>
            ///     Remove an axis from iteration (NumPy's <c>remove_axis(i)</c>): the iterator then walks every other axis with
            ///     that one fixed at index 0, and restarts. Requires the <c>multi_index</c> flag; the axis is in the operands'
            ///     own numbering, whatever order the iterator walks.
            /// </summary>
            /// <param name="axis">The axis to remove, in <c>[0, ndim)</c> (no negative indexing, as in NumPy).</param>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid"), or <paramref name="axis"/> is out
            ///     of bounds ("axis out of bounds in iterator RemoveAxis").</exception>
            /// <exception cref="RuntimeError">No multi-index is tracked, a flat index is tracked, or the iterator is buffered
            ///     (NumPy's texts).</exception>
            public void remove_axis(int axis)
            {
                var it = Borrow();
                it.RemoveAxis(axis);

                // npyiter_remove_axis: the cached values are stale, and RemoveAxis also reset the iterator.
                _cachedNext = it.PeekCachedIterNext();
                CacheValues();
                _started = _finished = it.IterSize == 0;
            }

            /// <summary>
            ///     Stop tracking the multi-index, letting the iterator coalesce and reorder axes
            ///     (NumPy's <c>remove_multi_index()</c>). Resets the position to the start, whether or not a multi-index was
            ///     tracked (a buffered window is written back first, delayed buffers are allocated).
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid").</exception>
            public void remove_multi_index()
            {
                var it = Borrow();
                it.RemoveMultiIndex();
                // NpyIter_RemoveMultiIndex starts with NpyIter_Reset, which allocates delayed buffers.
                _reduceDelayPending = false;
                _cachedNext = it.PeekCachedIterNext();
                // RemoveMultiIndex resets the iterator: the next pull yields the first element (NumPy's started = 0).
                CacheValues();
                _started = _finished = it.IterSize == 0;
            }

            /// <summary>
            ///     Switch to external-loop iteration after construction (NumPy's <c>enable_external_loop()</c>), restarting
            ///     from the first chunk (delayed buffers are allocated).
            /// </summary>
            /// <exception cref="ValueError">The iterator is closed ("Iterator is invalid"), or a flat index or multi-index is
            ///     tracked ("Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being
            ///     tracked").</exception>
            public void enable_external_loop()
            {
                var it = Borrow();
                it.EnableExternalLoop();
                // NpyIter_EnableExternalLoop ends with NpyIter_Reset, which allocates delayed buffers.
                _reduceDelayPending = false;
                _cachedNext = it.PeekCachedIterNext();
                // EnableExternalLoop resets the iterator: the next pull yields the first chunk (NumPy's started = 0).
                CacheValues();
                _started = _finished = it.IterSize == 0;
            }

            /// <summary>
            ///     Dump the iterator's internal state to standard output (NumPy's <c>debug_print()</c>); a closed iterator
            ///     prints NumPy's <c>Iterator: (nil)</c>.
            /// </summary>
            public void debug_print()
            {
                if (_state == null)
                {
                    Console.Out.Write("Iterator: (nil)\n");
                    return;
                }

                Borrow().DebugPrint();
            }

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

                // Write-backs (updateifcopy / copy_if_overlap temporaries) copy back into their originals; read-only copy
                // routes are only released. Either way each operand slot reverts to the original.
                NDIterRef.ResolveDetachedWritebacks(_operands, _writebackOriginals, _releaseOnlyOriginals);
                _writebackOriginals = null;
                _releaseOnlyOriginals = null;

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

            [NDBorrowed] // the foreach wrapper shares the owning iterator's live cursor; it owns nothing
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

            /// <summary>
            ///     The values published by the most recent <see cref="MoveNext"/>. Borrowed: the
            ///     per-step views are handed to the consumer, who owns them (NumPy's <c>[x.copy() for
            ///     x in it]</c> idiom is the one that keeps them).
            /// </summary>
            [NDBorrowed] public NDArray[] Current { get; private set; }

            /// <summary>
            ///     NumPy's <c>npyiter_next</c>: ADVANCES ON ENTRY (its <c>started</c> flag) and then publishes the values at the
            ///     new position — the first pull publishes the first element without moving, every later pull moves one step
            ///     first. So after one pull <see cref="iterindex"/> still reads 0, and a <see cref="copy"/> taken there (which
            ///     carries the flag) resumes with the SECOND element.
            /// </summary>
            /// <returns>True when a value was published; false at the end.</returns>
            /// <remarks>
            ///     The order is what makes a buffered <c>readwrite</c> loop correct: a write through the published view lands
            ///     in the buffer BEFORE the iterator moves past it, and moving past the end of a buffer is what flushes it. The
            ///     publish-then-advance form this replaced (journey4's surface gap SU42) moved first, flushed the buffer, and
            ///     lost the last write of every buffer. A published view stays valid until the next pull; under buffered
            ///     <c>external_loop</c> iteration it aliases a buffer the next pull refills — NumPy has the identical hazard,
            ///     hence its documented <c>[x.copy() for x in it]</c> idiom.
            /// </remarks>
            public bool MoveNext()
            {
                if (_nestedMode)
                    return MoveNextNested();

                // npyiter_next: a closed or finished iterator ends (an empty iteration or range starts finished).
                if (_state == null || _finished)
                {
                    Current = null;
                    return false;
                }

                if (_started)
                {
                    // iternext sets _finished at the end.
                    if (!iternext())
                    {
                        Current = null;
                        return false;
                    }
                }

                // NumPy sets `started` before building the values, so a pull that raises (a delay_bufalloc iterator not
                // reset yet) still counts as the first one.
                _started = true;
                Current = value;
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
            /// <param name="ptr">The iterator's data pointer for the operand.</param>
            /// <param name="dtype">The operand's iteration storage lane.</param>
            /// <param name="stringDescr">For the String lane, the sized descriptor of the memory the pointer is on; else null.</param>
            /// <param name="count">The element count (1 for the per-element view).</param>
            /// <param name="stride">The element stride (in elements of the view's dtype).</param>
            /// <param name="scalar">True for the 0-d per-element view.</param>
            /// <returns>The non-owning view.</returns>
            /// <exception cref="NotSupportedException">A lane without a view (none of the 15 storage lanes).</exception>
            private static NDArray AliasView(void* ptr, NPTypeCode dtype, DType stringDescr, long count, long stride, bool scalar)
            {
                switch (dtype)
                {
                    case NPTypeCode.String: return AliasStringView(ptr, stringDescr, count, stride, scalar);
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

            /// <summary>
            ///     The fixed-width string counterpart of <see cref="AliasView{T}"/>: a non-owning <see cref="StringArraySlice"/>
            ///     over the iterator's memory (bytes, sized by the descriptor), so reads and writes go straight through — NumPy's
            ///     0-d <c>&lt;U3</c> view per element, or the 1-D chunk under <c>external_loop</c>.
            /// </summary>
            /// <param name="ptr">The first element the view starts at.</param>
            /// <param name="descr">The sized, native string descriptor of the memory.</param>
            /// <param name="count">The element count.</param>
            /// <param name="stride">The element stride (in string elements; negative walks backward).</param>
            /// <param name="scalar">True for the 0-d per-element view.</param>
            /// <returns>The view (it does not own the memory and is invalid once the iterator moves past it or closes).</returns>
            private static NDArray AliasStringView(void* ptr, DType descr, long count, long stride, bool scalar)
            {
                int itemsize = descr.itemsize;
                // The block spans every element the view touches; a negative stride starts it below the pointer and the
                // shape carries the compensating offset (as AliasView<T>).
                long far = count <= 0 ? 0 : stride * (count - 1);
                long low = Math.Min(0, far);
                long high = Math.Max(0, far);
                long blockCount = high - low + 1;

                var block = new UnmanagedMemoryBlock<byte>((byte*)ptr + low * itemsize, blockCount * itemsize);
                var slice = new StringArraySlice(new ArraySlice<byte>(block), descr);
                if (scalar)
                    return new NDArray(new UnmanagedStorage(slice, Shape.NewScalar()));

                var storage = new UnmanagedStorage(slice, Shape.Vector(blockCount));
                var shape = new Shape(new[] {count}, new[] {stride}, -low, blockCount);
                return new NDArray(storage, shape);
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
