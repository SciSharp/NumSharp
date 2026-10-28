using System;
using NumSharp.Backends.Strings;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     The views <c>np.nditer</c> hands out over the iterator's LIVE data pointers — NumPy's <c>npyiter_seq_item</c>
    ///     (<c>it[i]</c>, <c>it.value</c>, each pulled step) and the temporary of <c>npyiter_seq_ass_item</c>
    ///     (<c>it[i] = v</c>), both <c>PyArray_NewFromDescrAndBase</c> over the operand's current pointer with the operand's
    ///     iteration descriptor.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The dtype decisions of those views live here, in the iterator, rather than in np.nditer's wrapper: which storage
    ///     lane the pointer is read as, the width and class of a fixed-width string view (the descriptor of the memory the
    ///     pointer is on: the buffer under a buffered cast, the operand otherwise), and the string allocator a NEP 55
    ///     <c>StringDType</c> view reads and writes its cells through (the buffer's own inside a buffer, the operand's
    ///     elsewhere). The wrapper keeps NumPy's wrapper logic: the operand-index checks, the write flags, the store.
    ///     </para>
    ///     <para>
    ///     A view wraps the iterator's memory without owning it: it is invalid once the iterator moves past it or closes,
    ///     exactly as NumPy's are (hence NumPy's <c>[x.copy() for x in it]</c> idiom). Each call builds a new view.
    ///     </para>
    /// </remarks>
    public unsafe ref partial struct NDIterRef
    {
        /// <summary>
        ///     A view of operand <paramref name="op"/> at the iterator's current position: the 0-d element view
        ///     (<paramref name="scalar"/>), or a 1-D view of <paramref name="count"/> elements <paramref name="stride"/> apart
        ///     (the inner loop under <c>external_loop</c>, or the setter's one-element temporary).
        /// </summary>
        /// <param name="op">The operand index, already checked by the caller to be in <c>[0, nop)</c>.</param>
        /// <param name="count">The element count (1 for the element views).</param>
        /// <param name="stride">The element stride, in elements of the view's dtype (negative walks backward).</param>
        /// <param name="scalar">True for the 0-d per-element view.</param>
        /// <returns>
        ///     The view, which does not own its data (<c>flags.owndata</c> is false, as NumPy's views of an iterator report)
        ///     and keeps the shape's default write flag: the caller applies the operand's write flags.
        /// </returns>
        /// <exception cref="NotSupportedException">The operand's iteration lane has no view (none of the 15 storage
        ///     lanes nor a string lane).</exception>
        /// <exception cref="InvalidOperationException">A <c>StringDType</c> view without the allocator of its cells (a
        ///     routing bug: a <c>StringDType</c> cell is meaningless without its buffer's allocator).</exception>
        /// <remarks>
        ///     The caller must not ask for the inner loop of a 0-d iteration through <c>GetInnerLoopSizePtr</c>: a 0-d
        ///     iteration has no inner dimension (the pointer would read <c>Shape[-1]</c>, an access violation); np.nditer's
        ///     getter passes <c>count = 1</c> there instead, NumPy's one 1-element chunk.
        /// </remarks>
        internal NDArray CurrentView(int op, long count, long stride, bool scalar)
        {
            void* ptr = _state->GetDataPtr(op);
            var dtype = _state->GetOpDType(op);
            // A fixed-width string view needs its width and class: the descriptor of the side the pointer is on (the
            // buffer under a buffered cast, the operand otherwise — the same descriptor without one).
            var stringDescr = dtype == NPTypeCode.String ? OperandStringDescr(op, ptr) : null;
            var vstrings = VStringAllocatorOf(op, stringDescr, ptr);
            return AliasView(ptr, dtype, stringDescr, count, stride, scalar, vstrings);
        }

        /// <summary>
        ///     The descriptor of the string memory operand <paramref name="op"/>'s data pointer <paramref name="ptr"/> is on,
        ///     for its views: with a NEP 55 <c>StringDType</c> side, the iteration descriptor inside the operand's buffer and
        ///     the array's descriptor elsewhere (the iterator's <see cref="NDIterVStringContext"/> keeps both instances — the
        ///     state's type code and element size would rebuild a 16-byte <c>'|S16'</c> and expose the raw cells); a
        ///     fixed-width operand's buffer-side descriptor (the buffer under a buffered cast, the operand otherwise).
        /// </summary>
        /// <param name="op">The operand index (a string operand).</param>
        /// <param name="ptr">The operand's current data pointer.</param>
        /// <returns>The descriptor.</returns>
        private DType OperandStringDescr(int op, void* ptr)
        {
            if (NDIterVStringContext.Of(ref *_state) is { } ctx && ctx.DescrAt(ref *_state, op, ptr) is { } descr)
                return descr;
            return _operands[op] is { } a && a.Storage.IsVString ? a.dtype : _state->GetOpDescr(op);
        }

        /// <summary>
        ///     The allocator of the NEP 55 <c>StringDType</c> cells operand <paramref name="op"/>'s data pointer
        ///     <paramref name="ptr"/> is on, for its iterator views: the buffer's own allocator inside a buffer with a
        ///     <c>StringDType</c> iteration side, else the operand's (the caller's array, an allocated output, or the copy that
        ///     replaced it in the operand list).
        /// </summary>
        /// <param name="op">The operand index.</param>
        /// <param name="stringDescr">The descriptor of that memory (<see cref="OperandStringDescr"/>), or null for a numeric
        ///     operand.</param>
        /// <param name="ptr">The operand's current data pointer.</param>
        /// <returns>The allocator, or null when that memory is not <c>StringDType</c>.</returns>
        private VStringAllocator VStringAllocatorOf(int op, DType stringDescr, void* ptr)
        {
            if (stringDescr?.Meta is not VStringDTypeMeta)
                return null;
            if (NDIterVStringContext.Of(ref *_state) is { } ctx && ctx.AllocatorAt(ref *_state, op, ptr) is { } allocator)
                return allocator;
            return _operands[op].Storage.VStringSlice.Allocator;
        }

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
        /// <param name="vstrings">For a NEP 55 <c>StringDType</c> operand, the allocator of the cells the pointer is on (see
        ///     <see cref="VStringAllocatorOf"/>); else null.</param>
        /// <returns>The non-owning view.</returns>
        /// <exception cref="NotSupportedException">A lane without a view (none of the 15 storage lanes).</exception>
        private static NDArray AliasView(void* ptr, NPTypeCode dtype, DType stringDescr, long count, long stride, bool scalar,
                                         VStringAllocator vstrings = null)
        {
            switch (dtype)
            {
                case NPTypeCode.String: return AliasStringView(ptr, stringDescr, count, stride, scalar, vstrings);
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
        /// <param name="vstrings">For a NEP 55 <c>StringDType</c> descriptor, the allocator of the operand's cells: the
        ///     view is a <c>StringDType</c> view of those cells (texts read and written through that allocator, never the
        ///     raw cell bytes); else null.</param>
        /// <returns>The view (it does not own the memory and is invalid once the iterator moves past it or closes).</returns>
        /// <exception cref="InvalidOperationException">A <c>StringDType</c> descriptor without its allocator (a routing
        ///     bug: a <c>StringDType</c> cell is meaningless without the allocator of its buffer).</exception>
        private static NDArray AliasStringView(void* ptr, DType descr, long count, long stride, bool scalar,
                                               VStringAllocator vstrings = null)
        {
            int itemsize = descr.itemsize;
            // The block spans every element the view touches; a negative stride starts it below the pointer and the
            // shape carries the compensating offset (as AliasView<T>).
            long far = count <= 0 ? 0 : stride * (count - 1);
            long low = Math.Min(0, far);
            long high = Math.Max(0, far);
            long blockCount = high - low + 1;

            var block = new UnmanagedMemoryBlock<byte>((byte*)ptr + low * itemsize, blockCount * itemsize);
            IArraySlice slice;
            if (descr.Meta is VStringDTypeMeta)
            {
                // A StringDType view: the cells keep meaning only with their own buffer's allocator. The non-owning block
                // runs no clear loop on release (the operand's block owns the allocator and outlives the view).
                if (vstrings is null)
                    throw new InvalidOperationException("A StringDType iterator view needs the allocator of its operand's cells.");
                slice = new VStringArraySlice(new ArraySlice<byte>(block), vstrings, descr);
            }
            else
            {
                slice = new StringArraySlice(new ArraySlice<byte>(block), descr);
            }
            if (scalar)
            {
                var scalarStorage = new UnmanagedStorage(slice, NumSharp.Shape.NewScalar());
                // NumPy's view has the iterator for its base, so it does not own its data (see AliasView<T>).
                scalarStorage.ExternalBase = true;
                scalarStorage.OnReshaped();
                return new NDArray(scalarStorage);
            }

            var storage = new UnmanagedStorage(slice, NumSharp.Shape.Vector(blockCount));
            var shape = new Shape(new[] {count}, new[] {stride}, -low, blockCount);
            return new NDArray(storage, shape);
        }

        /// <summary>
        ///     Wraps the iterator's memory at <paramref name="ptr"/> as a non-owning view of element type
        ///     <typeparamref name="T"/>: the 0-d per-element view, or the 1-D inner-loop chunk under
        ///     <c>external_loop</c>.
        /// </summary>
        /// <typeparam name="T">The element type of the operand's iteration lane.</typeparam>
        /// <param name="ptr">The first element the view starts at.</param>
        /// <param name="count">The element count.</param>
        /// <param name="stride">The element stride (in elements; negative walks backward).</param>
        /// <param name="scalar">True for the 0-d per-element view.</param>
        /// <returns>The view. Neither form owns its data (<c>flags.owndata</c> is false, as NumPy's views of an iterator
        ///     report), and both are invalid once the iterator moves past them or closes.</returns>
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
            {
                var view = new NDArray(slice, NumSharp.Shape.NewScalar());
                // NumPy builds the view with PyArray_NewFromDescrAndBase over the iterator's pointer, the iterator its
                // base: the view does not own its data. The slice ctor wraps the block as an OWNER (no base storage),
                // so the storage is marked as having a foreign owner and the OWNDATA mirror re-synced. The 1-D form
                // below needs neither: its alias of the wrap storage has a base. A struct copy, no allocation.
                view.Storage.ExternalBase = true;
                view.Storage.OnReshaped();
                return view;
            }

            // external_loop: a strided inner loop spans more slots than it has elements
            // (count=2, stride=2 covers 3), which the slice ctor rejects — it demands
            // shape.size == slice.Count. So wrap the block as FLAT storage first and apply the
            // strided/offset shape through the view ctor, the same two-step
            // NDIterRef.GetIterView uses. This path runs once per CHUNK, not per element.
            var storage = new UnmanagedStorage(slice, NumSharp.Shape.Vector(blockCount));
            var shape = new Shape(new[] {count}, new[] {stride}, -low, blockCount);

            return new NDArray(storage, shape);
        }
    }
}
