using System;
using System.Runtime.InteropServices;
using NumSharp.Backends.Strings;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     The managed side of an iterator's NEP 55 <c>StringDType</c> operands: for every operand with a <c>StringDType</c> side
    ///     (its array, its iteration descriptor, or both), the two descriptors and the allocators of the cells on each side. The
    ///     unmanaged <see cref="NDIterState"/> holds it as a <see cref="GCHandle"/> (<see cref="NDIterState.VStringContext"/>),
    ///     because the buffer transfers see only the state, and a <c>StringDType</c> cell is meaningless without the allocator of
    ///     the buffer it lives in (an arena offset or a heap pointer of that allocator).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>NumPy's model and NumSharp's.</b> In NumPy the allocator belongs to the DESCRIPTOR instance, so an iterator buffer of
    ///     the operand's own descriptor shares the array's allocator. In NumSharp it belongs to the cells' memory block (views of
    ///     one buffer share it, a copy gets its own), so every buffer with a <c>StringDType</c> side gets an allocator of its own
    ///     (<see cref="AllocateBuffer"/>), freed with the buffer (<see cref="FreeBuffer"/>). The values every route reads are
    ///     NumPy's either way; only the internal arena offsets of a buffer's cells differ, which no API shows.
    ///     </para>
    ///     <para>
    ///     <b>Transfers.</b> A fill (array → buffer) and a flush (buffer → array) of such an operand run
    ///     <see cref="NDVStringCasting.TransferRun"/> over each run of the iterator's walk: NumPy's <c>string_to_string</c> between
    ///     two <c>StringDType</c> descriptors (every string duplicated into the destination's allocator, the destination's old
    ///     string freed first), or the <c>StringDType</c> cast loop of the pair (<c>U</c> / <c>S</c> / bool / numbers). The run
    ///     bodies are built once per operand and direction (<see cref="Transfer"/>).
    ///     </para>
    ///     <para>
    ///     <b>Lifetime.</b> Attached once the operands are final (after the copy routes and COPY_IF_OVERLAP replaced any of them,
    ///     before the first buffer allocation); released by <see cref="NDIterState.FreeDimArrays"/>, the step every teardown of a
    ///     state runs (<c>Dispose</c>, <c>FreeState</c>, a failed construction or copy). <see cref="NDIterRef.Copy"/> gives the
    ///     copy a context of its own (<see cref="CloneForCopy"/>): the same array descriptors and allocators, new buffer allocators
    ///     over zeroed buffers, which the copy's first fill (NumPy's re-buffering at the copy's position) fills from the arrays.
    ///     </para>
    /// </remarks>
    internal sealed unsafe class NDIterVStringContext
    {
        /// <summary>Each operand's ARRAY-side descriptor (the operand array's dtype); null for an operand without a <c>StringDType</c> side.</summary>
        internal readonly DType[] ArrayDescrs;

        /// <summary>Each operand's ITERATION (buffer) descriptor; null for an operand without a <c>StringDType</c> side.</summary>
        internal readonly DType[] BufferDescrs;

        /// <summary>The allocator of each operand's array cells when its array side is <c>StringDType</c>, else null.</summary>
        internal readonly VStringAllocator[] ArrayAllocators;

        /// <summary>
        ///     The allocator of each operand's buffer cells while a buffer with a <c>StringDType</c> iteration side is allocated,
        ///     else null (<see cref="AllocateBuffer"/> / <see cref="FreeBuffer"/>).
        /// </summary>
        internal readonly VStringAllocator[] BufferAllocators;

        /// <summary>The cached fill bodies (array → buffer), per operand; built on first use.</summary>
        private readonly NDVStringCasting.RunAction[] _fills;

        /// <summary>The cached flush bodies (buffer → array), per operand; built on first use.</summary>
        private readonly NDVStringCasting.RunAction[] _flushes;

        /// <summary>Creates an empty context for <paramref name="nop"/> operands.</summary>
        /// <param name="nop">The number of operands.</param>
        private NDIterVStringContext(int nop)
        {
            ArrayDescrs = new DType[nop];
            BufferDescrs = new DType[nop];
            ArrayAllocators = new VStringAllocator[nop];
            BufferAllocators = new VStringAllocator[nop];
            _fills = new NDVStringCasting.RunAction[nop];
            _flushes = new NDVStringCasting.RunAction[nop];
        }

        /// <summary>
        ///     Attaches a context to <paramref name="state"/> when one of its operands has a <c>StringDType</c> side, and marks
        ///     those operands in the state (<see cref="NDIterState.SetOpDType(int, DType)"/> already set the string kind bits).
        /// </summary>
        /// <param name="state">The iterator state (its operand dtypes set, its operands final).</param>
        /// <param name="operands">The final operand arrays (an allocated output or a copy-route temporary included).</param>
        /// <param name="iterDescrs">The iteration descriptors of the descriptor plan, per operand.</param>
        /// <exception cref="InvalidOperationException">The state already has a context (a construction bug).</exception>
        internal static void AttachIfNeeded(ref NDIterState state, NDArray[] operands, DType[] iterDescrs)
        {
            int nop = state.NOp;
            NDIterVStringContext ctx = null;
            for (int op = 0; op < nop; op++)
            {
                var arr = op < operands.Length ? operands[op] : null;
                var arrDescr = arr?.dtype;
                var iterDescr = iterDescrs != null && op < iterDescrs.Length && iterDescrs[op] != null ? iterDescrs[op] : arrDescr;
                bool arrV = arrDescr?.Meta is VStringDTypeMeta;
                bool iterV = iterDescr?.Meta is VStringDTypeMeta;
                if (!arrV && !iterV)
                    continue;
                ctx ??= new NDIterVStringContext(nop);
                ctx.ArrayDescrs[op] = arrDescr;
                ctx.BufferDescrs[op] = iterDescr;
                // The operand's own cells: the caller's array, an allocated output, or the temporary that replaced it.
                ctx.ArrayAllocators[op] = arrV ? arr.Storage.VStringSlice.Allocator : null;
            }

            if (ctx == null)
                return;
            if (state.VStringContext != IntPtr.Zero)
                throw new InvalidOperationException("The iterator state already has a StringDType context.");
            state.VStringContext = GCHandle.ToIntPtr(GCHandle.Alloc(ctx));
        }

        /// <summary>The context of <paramref name="state"/>.</summary>
        /// <param name="state">An iterator state.</param>
        /// <returns>The context, or null when no operand has a <c>StringDType</c> side.</returns>
        internal static NDIterVStringContext Of(ref NDIterState state)
            => state.VStringContext == IntPtr.Zero ? null : (NDIterVStringContext)GCHandle.FromIntPtr(state.VStringContext).Target;

        /// <summary>
        ///     Releases <paramref name="state"/>'s context handle (the buffers and their allocators were freed before, by
        ///     <see cref="NDIterBufferManager.FreeBuffers"/>); a state without one is left alone.
        /// </summary>
        /// <param name="state">The state being torn down.</param>
        internal static void Detach(ref NDIterState state)
        {
            if (state.VStringContext == IntPtr.Zero)
                return;
            GCHandle.FromIntPtr(state.VStringContext).Free();
            state.VStringContext = IntPtr.Zero;
        }

        /// <summary>Whether operand <paramref name="op"/>'s ITERATION side is <c>StringDType</c> (its buffer holds cells).</summary>
        /// <param name="op">The operand index.</param>
        /// <returns>True for a <c>StringDType</c> iteration descriptor.</returns>
        internal bool IsBufferVString(int op) => BufferDescrs[op]?.Meta is VStringDTypeMeta;

        /// <summary>
        ///     Prepares a freshly allocated buffer of operand <paramref name="op"/>: when its iteration side is <c>StringDType</c>,
        ///     the cells are zeroed (an all-zero cell is the empty string and owns nothing, so the first fill frees nothing) and a
        ///     new allocator takes them.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="buffer">The buffer's first byte.</param>
        /// <param name="elements">The buffer's element count.</param>
        internal void AllocateBuffer(int op, void* buffer, long elements)
        {
            if (!IsBufferVString(op))
                return;
            NativeMemory.Clear(buffer, (nuint)(elements * VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING));
            BufferAllocators[op] = new VStringAllocator();
            // The new allocator replaces whatever a cached transfer of this operand captured.
            ForgetTransfers(op);
        }

        /// <summary>
        ///     Releases what a buffer of operand <paramref name="op"/> holds before its memory is freed: NumPy's clear loop over
        ///     the cells (the heap strings) and the allocator's arena.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="buffer">The buffer's first byte.</param>
        /// <param name="elements">The buffer's element count.</param>
        internal void FreeBuffer(int op, void* buffer, long elements)
        {
            var allocator = BufferAllocators[op];
            if (allocator == null)
                return;
            BufferAllocators[op] = null;
            // The operand's cached transfers write and read through this allocator: none may run once it is freed.
            ForgetTransfers(op);
            allocator.ClearAndFree((byte*)buffer, elements);
        }

        /// <summary>
        ///     Drops operand <paramref name="op"/>'s cached transfers (<see cref="Transfer"/>), which hold the buffer allocator
        ///     they were built over; the next fill or flush builds them again over the current one.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <remarks>
        ///     A state's buffers are freed and allocated again without a new context when the iterator runs its buffering setup
        ///     a second time (<c>RebuildBufferingSetup</c>, after <c>remove_multi_index</c> merged axes the old setup no longer
        ///     describes). A transfer kept from the first buffer would then write the refill's strings through the freed
        ///     allocator, and the new allocator could not read them back (NumSharp's "Failed to load string" error).
        /// </remarks>
        private void ForgetTransfers(int op)
        {
            _fills[op] = null;
            _flushes[op] = null;
        }

        /// <summary>
        ///     The run body of operand <paramref name="op"/>'s transfer in one direction, built once: a fill reads the array side
        ///     and writes the buffer side, a flush the reverse.
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="toBuffer">True for the fill, false for the flush.</param>
        /// <returns>The run body (the caller locks the allocators, see <see cref="Lock"/>).</returns>
        /// <exception cref="InvalidOperationException">A <c>StringDType</c> buffer side without its allocator (the buffer was
        ///     not allocated through <see cref="AllocateBuffer"/>: a routing bug).</exception>
        internal NDVStringCasting.RunAction Transfer(int op, bool toBuffer)
        {
            var cache = toBuffer ? _fills : _flushes;
            var run = cache[op];
            if (run != null)
                return run;
            if (IsBufferVString(op) && BufferAllocators[op] == null)
                throw new InvalidOperationException("A StringDType iterator buffer needs the allocator of its cells.");
            run = toBuffer
                ? NDVStringCasting.TransferRun(ArrayDescrs[op], ArrayAllocators[op], BufferDescrs[op], BufferAllocators[op])
                : NDVStringCasting.TransferRun(BufferDescrs[op], BufferAllocators[op], ArrayDescrs[op], ArrayAllocators[op]);
            cache[op] = run;
            return run;
        }

        /// <summary>Locks the allocators operand <paramref name="op"/>'s transfer touches (both sides' that exist).</summary>
        /// <param name="op">The operand index.</param>
        /// <remarks>Pair it with exactly one <see cref="Unlock"/> for the same operand.</remarks>
        internal void Lock(int op)
        {
            var a = ArrayAllocators[op];
            var b = BufferAllocators[op];
            if (a != null && b != null)
                VStringAllocator.acquire_allocators(a, b);
            else
                (a ?? b)?.acquire_allocator();
        }

        /// <summary>Releases what <see cref="Lock"/> took for operand <paramref name="op"/>.</summary>
        /// <param name="op">The operand index.</param>
        internal void Unlock(int op)
        {
            var a = ArrayAllocators[op];
            var b = BufferAllocators[op];
            if (a != null && b != null)
                VStringAllocator.release_allocators(a, b);
            else
                (a ?? b)?.release_allocator();
        }

        /// <summary>
        ///     The descriptor of the memory operand <paramref name="op"/>'s data pointer <paramref name="ptr"/> is on: the
        ///     iteration descriptor inside the operand's buffer, the array's descriptor anywhere else.
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The operand index.</param>
        /// <param name="ptr">The operand's current data pointer.</param>
        /// <returns>The descriptor, or null when operand <paramref name="op"/> has no <c>StringDType</c> side.</returns>
        internal DType DescrAt(ref NDIterState state, int op, void* ptr)
            => InBuffer(ref state, op, ptr) ? BufferDescrs[op] : ArrayDescrs[op];

        /// <summary>
        ///     The allocator of the cells <paramref name="ptr"/> points at: the buffer's inside the operand's buffer, the array's
        ///     anywhere else; null when that memory is not <c>StringDType</c>.
        /// </summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The operand index.</param>
        /// <param name="ptr">The operand's current data pointer.</param>
        /// <returns>The allocator, or null.</returns>
        internal VStringAllocator AllocatorAt(ref NDIterState state, int op, void* ptr)
            => InBuffer(ref state, op, ptr) ? BufferAllocators[op] : ArrayAllocators[op];

        /// <summary>Whether <paramref name="ptr"/> lies inside operand <paramref name="op"/>'s allocated buffer.</summary>
        /// <param name="state">The iterator state.</param>
        /// <param name="op">The operand index.</param>
        /// <param name="ptr">A pointer.</param>
        /// <returns>True inside the buffer.</returns>
        private static bool InBuffer(ref NDIterState state, int op, void* ptr)
        {
            byte* buffer = (byte*)state.GetBuffer(op);
            if (buffer == null)
                return false;
            long bytes = state.BufferSize * state.GetElementSize(op);
            return (byte*)ptr >= buffer && (byte*)ptr < buffer + bytes;
        }

        /// <summary>
        ///     A context for <see cref="NDIterRef.Copy"/>'s new state: the same descriptors and array allocators, and for every
        ///     <c>StringDType</c> buffer a new allocator over the copy's (zeroed) buffer — with each string of the source buffer
        ///     duplicated into it when <paramref name="duplicateBufferStrings"/> is set (the copy's cells never point into the
        ///     source's allocator either way).
        /// </summary>
        /// <param name="source">The source state.</param>
        /// <param name="copy">The new state, its buffers allocated (raw bytes) but not filled.</param>
        /// <param name="duplicateBufferStrings">True to duplicate the source buffers' strings into the copy's; false to leave
        ///     the copy's buffers empty (every cell the empty string) for a refill from the arrays — what NumPy's
        ///     <c>NpyIter_Copy</c> does, and what <see cref="NDIterRef.Copy"/> asks for.</param>
        /// <returns>The new context, attached to <paramref name="copy"/>.</returns>
        /// <exception cref="OutOfMemoryException">NumPy's <c>MemoryError</c> texts.</exception>
        internal NDIterVStringContext CloneForCopy(ref NDIterState source, ref NDIterState copy, bool duplicateBufferStrings = true)
        {
            int nop = ArrayDescrs.Length;
            var ctx = new NDIterVStringContext(nop);
            Array.Copy(ArrayDescrs, ctx.ArrayDescrs, nop);
            Array.Copy(BufferDescrs, ctx.BufferDescrs, nop);
            Array.Copy(ArrayAllocators, ctx.ArrayAllocators, nop);
            copy.VStringContext = GCHandle.ToIntPtr(GCHandle.Alloc(ctx));
            for (int op = 0; op < nop; op++)
            {
                byte* from = (byte*)source.GetBuffer(op);
                byte* to = (byte*)copy.GetBuffer(op);
                if (from == null || to == null || !ctx.IsBufferVString(op) || BufferAllocators[op] == null)
                    continue;
                ctx.AllocateBuffer(op, to, copy.BufferSize);
                if (!duplicateBufferStrings)
                    continue;   // the zeroed cells wait for the refill
                var fromAllocator = BufferAllocators[op];
                var toAllocator = ctx.BufferAllocators[op];
                VStringAllocator.acquire_allocators(fromAllocator, toAllocator);
                try
                {
                    NDVStringCopy.CopyRun(from, VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING, to, VStringAllocator.SIZEOF_NPY_PACKED_STATIC_STRING, source.BufferSize,
                        fromAllocator, toAllocator, naName: null);
                }
                finally
                {
                    VStringAllocator.release_allocators(fromAllocator, toAllocator);
                }
            }
            return ctx;
        }
    }
}
