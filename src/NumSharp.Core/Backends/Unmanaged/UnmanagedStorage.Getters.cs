using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Getters

        /// <summary>
        ///     <see cref="IArraySlice.Slice(long,long)"/> for a sub-view, tolerating the EMPTY case.
        /// </summary>
        /// <remarks>
        ///     A zero-length sub-view has no element to address, so its start offset is meaningless
        ///     — and it is routinely out of range: an empty window over a live buffer
        ///     (<c>x[:, 1:1]</c>) narrows <see cref="InternalArray"/> to Count 0 while the shape's
        ///     strides still step across the original rows, so <c>[1]</c> asks to slice at offset 2
        ///     of a 0-element array. NumPy hands back the empty view; <see cref="IArraySlice"/>'s
        ///     range guard is right to reject the call, so the empty case starts at 0 instead.
        ///     Nothing can be read or written through a length-0 slice, so where it points is
        ///     unobservable.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private IArraySlice SliceForSubview(long offset, long count)
            => InternalArray.Slice(count == 0 ? 0 : offset, count);

        /// <summary>
        ///     The length of a broadcast sub-array's slice, ended at the buffer's end: the parent's buffer size counts from
        ///     the parent's FIRST element and the slice from the sub-array's, so past the first row the requested length runs
        ///     off the end of the buffer (row i of <c>np.broadcast_to(b, (3, n)).T</c> starts at element i and asked for all
        ///     n).
        /// </summary>
        /// <param name="offset">The sub-array's first element in the buffer.</param>
        /// <param name="length">The requested length.</param>
        /// <returns>The length, at most the elements from <paramref name="offset"/> to the buffer's end (never negative).</returns>
        /// <remarks>
        ///     A broadcast view's strides are stride 0 or non-negative, so it reaches only elements its parent reaches: the
        ///     clamped slice still covers every element the view reads. A numeric slice never checked its length (the view
        ///     read inside the buffer anyway); a <c>StringDType</c> cell slice does, and refused the row
        ///     (<c>np.apply_along_axis</c> along an axis of a broadcast <c>StringDType</c> array).
        /// </remarks>
        private long ClampToBuffer(long offset, long length)
        {
            long available = InternalArray.Count - offset;
            return length > available ? Math.Max(available, 0) : length;
        }

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> boxed as the dtype's own CLR type (a text element as its
        ///     decoded C# value), after NumPy's coordinate check — the read behind <c>item(i, j, …)</c>.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its axis, and
        ///     axes left out read their first element.</param>
        /// <returns>The element, boxed.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing outside the array is read).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        public unsafe object GetValue(int[] indices)
        {
            // One checked offset for the whole switch (parity QA-33: an out-of-range coordinate used to read another row
            // or outside the buffer).
            long off = _shape.GetOffsetChecked(indices);
            return ElementAt(off);
        }

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> boxed as the dtype's own CLR type (a text element as its
        ///     decoded C# value), after NumPy's coordinate check — the read behind <c>item(i, j, …)</c>.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end
        ///     of its axis, and axes left out read their first element.</param>
        /// <returns>The element, boxed.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing outside the array is read).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        public unsafe object GetValue(params long[] indices)
        {
            long off = _shape.GetOffsetChecked(indices);
            return ElementAt(off);
        }

        /// <summary>
        ///     Reads the element at logical flat index <paramref name="index"/> (C order) boxed as the dtype's own CLR
        ///     type, after NumPy's flat-index check (<c>a.item(i)</c>, <c>a.flat[i]</c>).
        /// </summary>
        /// <param name="index">The flat index; a negative one counts from the end of the array.</param>
        /// <returns>The element, boxed.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is outside the array (NumPy's item texts).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        public unsafe object GetAtIndex(long index)
        {
            long off = _shape.TransformOffsetChecked(index);
            return ElementAt(off);
        }

        /// <summary>
        ///     Boxes the element at a resolved storage offset as the dtype's own CLR type; the shared tail of the checked
        ///     boxing getters.
        /// </summary>
        /// <param name="off">The element's offset in the buffer, already checked by the caller.</param>
        /// <returns>The element, boxed (a text element as its decoded C# value).</returns>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        private unsafe object ElementAt(long off)
        {
            switch (TypeCode)
            {
                case NPTypeCode.Boolean: return *((bool*)Address + off);
                case NPTypeCode.SByte: return *((sbyte*)Address + off);
                case NPTypeCode.Byte: return *((byte*)Address + off);
                case NPTypeCode.Int16: return *((short*)Address + off);
                case NPTypeCode.UInt16: return *((ushort*)Address + off);
                case NPTypeCode.Int32: return *((int*)Address + off);
                case NPTypeCode.UInt32: return *((uint*)Address + off);
                case NPTypeCode.Int64: return *((long*)Address + off);
                case NPTypeCode.UInt64: return *((ulong*)Address + off);
                case NPTypeCode.Char: return *((char*)Address + off);
                case NPTypeCode.Half: return HalfAt((ushort*)Address + off);
                case NPTypeCode.Double: return *((double*)Address + off);
                case NPTypeCode.Single: return *((float*)Address + off);
                case NPTypeCode.Decimal: return *((decimal*)Address + off);
                case NPTypeCode.Complex: return *((System.Numerics.Complex*)Address + off);
                // A string element decodes to its C# value through its slice: string for U, byte[] for S, string or the NA
                // object for StringDType (the slice of either layout implements IArraySlice.GetIndex(long)).
                case NPTypeCode.String: return InternalArray.GetIndex(off);
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     The <see cref="Half"/> at <paramref name="p"/>, built from its 16-bit pattern — the form the boxing getters
        ///     above return.
        /// </summary>
        /// <param name="p">The element's address.</param>
        /// <returns>The value.</returns>
        /// <remarks>
        ///     Boxing a <see cref="Half"/> straight from a pointer (<c>return *(Half*)p;</c> into an <see cref="object"/>)
        ///     makes the .NET 8 runtime's tier-0 code (the first calls of a method, before tiered compilation optimizes
        ///     it) read past the two bytes: an element that ends where mapped memory ends — a <c>frombuffer</c> over a
        ///     caller's page, a buffer under <c>NUMSHARP_DEBUG_GUARD_PAGES=1</c> — ended the process ("Internal CLR error",
        ///     reproduced with <c>VirtualAlloc</c>: .NET 8.0.29 crashes, .NET 10 and optimized code read two bytes). A
        ///     <see cref="Half"/> made from a <see cref="ushort"/> read is boxed from a register, never from the memory.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Half HalfAt(ushort* p) => BitConverter.UInt16BitsToHalf(*p);

        /// <summary>
        ///     Reads the element at logical flat index <paramref name="index"/> (C order) as <typeparamref name="T"/>: the
        ///     stored value for the dtype's own type, NumPy's cast of it for any other <typeparamref name="T"/>.
        /// </summary>
        /// <typeparam name="T">The type to read the element as.</typeparam>
        /// <param name="index">The flat index; a negative one counts from the end of the array.</param>
        /// <returns>The element as <typeparamref name="T"/>.</returns>
        /// <exception cref="IndexError"><paramref name="index"/> is outside the array (NumPy's item texts).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        /// <remarks>
        ///     Before parity QA-12 / QA-33 this loaded <c>sizeof(T)</c> bytes at an unchecked offset (a <c>Debug.Assert</c>
        ///     was the only type test), so a wrong <typeparamref name="T"/> or index read neighbouring elements or memory
        ///     past the buffer in Release.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        public unsafe T GetAtIndex<T>(long index) where T : unmanaged
            => ReadAs<T>(_shape.TransformOffsetChecked(index));

        /// <summary>
        /// Gets a sub-array based on the given indices, returning a view that shares memory.
        /// </summary>
        /// <param name="indices">
        /// The indices specifying which dimensions to select. Negative indices are supported
        /// and are converted to positive indices relative to the dimension size.
        /// </param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> representing the sub-array. This is a view
        /// that shares memory with the original storage. The returned storage's
        /// <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The returned storage shares memory with this storage.
        /// Modifications through either storage affect the same underlying data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner,
        /// ensuring all views in a chain point to the original storage, not intermediate views.
        /// </para>
        /// <para>
        /// <b>Code Paths:</b>
        /// <list type="bullet">
        ///   <item><b>Broadcasted shapes:</b> Creates a broadcasted view with base tracking</item>
        ///   <item><b>Non-contiguous shapes:</b> Delegates to <see cref="GetView(Slice[])"/> which uses <see cref="Alias(Shape)"/></item>
        ///   <item><b>Contiguous shapes:</b> Creates a direct memory slice with base tracking</item>
        /// </list>
        /// </para>
        /// </remarks>
        /// <seealso cref="GetView(Slice[])"/>
        /// <seealso cref="Alias()"/>
        [MethodImpl(OptimizeAndInline)]
        public UnmanagedStorage GetData(int[] indices)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            indices = Shape.InferNegativeCoordinates(Shape.dimensions, indices);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                // For non-broadcasted contiguous subshapes, use size (the actual data extent).
                // Only use BufferSize when the subshape itself is broadcasted.
                long sliceSize = ClampToBuffer(offset, shape.IsBroadcasted
                    ? (shape.BufferSize > 0 ? shape.BufferSize : shape.size)
                    : shape.size);
                // The slice below starts at the element GetSubshape names, so the view's shape starts at 0 — except for an
                // EMPTY index list, where GetSubshape returns this shape itself with offset 0 (the slice then starts at the
                // buffer's first element) and the shape's own offset must stay: a row of a broadcast at a non-zero offset
                // (np.broadcast_arrays(x[:, None], x)[0][1] has stride 0 and offset 1) took its stores (`row[...] = v`,
                // SetData) at element 0 when this offset was dropped.
                var adjustedShape = new Shape(shape.dimensions, shape.strides, indices.Length == 0 ? shape.offset : 0, sliceSize);
                // A sub-array of a read-only array is read-only (NumPy views inherit WRITEABLE): row 2 of a broadcast row
                // keeps no stride-0 axis, so its fresh flags would otherwise call it writeable.
                if (!this_shape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                // The mirror half, as Alias(Shape) applies it: a broadcast sub-array of a WRITEABLE broadcast parent
                // (np.broadcast_arrays' results, ndarray.setflags(write: true), an np.nditer view of a reduce operand's
                // stride-0 chunk) keeps the parent's write access — its computed read-only flag came from the broadcastness
                // it inherited, not from a protection decision. Without it a store into such a parent (`y[...] = v`, whose
                // copy takes this sub-array) raised "assignment destination is read-only" where NumPy writes.
                else if (adjustedShape.IsBroadcasted && !adjustedShape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(SliceForSubview(offset, sliceSize), adjustedShape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                return GetView(indices.Select(Slice.Index).ToArray());
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(indices);
                // A sub-array is a VIEW, so it inherits writeability: indexing a row/element out of a
                // read-only array (broadcast, or a contiguous 'r' memmap) must stay read-only —
                // GetSubshape rebuilds the shape with fresh (writeable) flags, so carry it through.
                if (!this_shape.IsWriteable)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                var view = new UnmanagedStorage(SliceForSubview(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
        }

        /// <summary>
        /// Gets a sub-array based on the given indices (long array version), returning a view that shares memory.
        /// </summary>
        /// <param name="indices">The dimension indices specifying the sub-array.</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> representing the sub-array. This is a view
        /// that shares memory with the original storage. The returned storage's
        /// <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <seealso cref="GetData(int[])"/>
        [MethodImpl(OptimizeAndInline)]
        public UnmanagedStorage GetData(params long[] indices)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            indices = Shape.InferNegativeCoordinates(Shape.dimensions, indices);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(indices);
                // For non-broadcasted contiguous subshapes, use size (the actual data extent).
                // Only use BufferSize when the subshape itself is broadcasted.
                long sliceSize = ClampToBuffer(offset, shape.IsBroadcasted
                    ? (shape.BufferSize > 0 ? shape.BufferSize : shape.size)
                    : shape.size);
                // The slice below starts at the element GetSubshape names, so the view's shape starts at 0 — except for an
                // EMPTY index list, where GetSubshape returns this shape itself with offset 0 (the slice then starts at the
                // buffer's first element) and the shape's own offset must stay: a row of a broadcast at a non-zero offset
                // (np.broadcast_arrays(x[:, None], x)[0][1] has stride 0 and offset 1) took its stores (`row[...] = v`,
                // SetData) at element 0 when this offset was dropped.
                var adjustedShape = new Shape(shape.dimensions, shape.strides, indices.Length == 0 ? shape.offset : 0, sliceSize);
                // A sub-array of a read-only array is read-only (NumPy views inherit WRITEABLE): row 2 of a broadcast row
                // keeps no stride-0 axis, so its fresh flags would otherwise call it writeable.
                if (!this_shape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                // The mirror half, as Alias(Shape) applies it: a broadcast sub-array of a WRITEABLE broadcast parent
                // (np.broadcast_arrays' results, ndarray.setflags(write: true), an np.nditer view of a reduce operand's
                // stride-0 chunk) keeps the parent's write access — its computed read-only flag came from the broadcastness
                // it inherited, not from a protection decision. Without it a store into such a parent (`y[...] = v`, whose
                // copy takes this sub-array) raised "assignment destination is read-only" where NumPy writes.
                else if (adjustedShape.IsBroadcasted && !adjustedShape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(SliceForSubview(offset, sliceSize), adjustedShape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                return GetView(indices.Select(i => Slice.Index(i)).ToArray());
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(indices);
                // A sub-array is a VIEW, so it inherits writeability: indexing a row/element out of a
                // read-only array (broadcast, or a contiguous 'r' memmap) must stay read-only —
                // GetSubshape rebuilds the shape with fresh (writeable) flags, so carry it through.
                if (!this_shape.IsWriteable)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                var view = new UnmanagedStorage(SliceForSubview(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
        }

        /// <summary>
        /// Gets a sub-array based on the given indices (pointer version), returning a view that shares memory.
        /// </summary>
        /// <param name="dims">Pointer to an array of dimension indices.</param>
        /// <param name="ndims">The number of indices in the array.</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> representing the sub-array. This is a view
        /// that shares memory with the original storage. The returned storage's
        /// <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The returned storage shares memory with this storage.
        /// Modifications through either storage affect the same underlying data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner,
        /// ensuring all views in a chain point to the original storage, not intermediate views.
        /// </para>
        /// <para>
        /// <b>Code Paths:</b>
        /// <list type="bullet">
        ///   <item><b>Broadcasted shapes:</b> Creates a broadcasted view with base tracking</item>
        ///   <item><b>Non-contiguous shapes:</b> Delegates to <see cref="GetView(Slice[])"/> which uses <see cref="Alias(Shape)"/></item>
        ///   <item><b>Contiguous shapes:</b> Creates a direct memory slice with base tracking</item>
        /// </list>
        /// </para>
        /// </remarks>
        /// <seealso cref="GetData(int[])"/>
        [MethodImpl(OptimizeAndInline)]
        public unsafe UnmanagedStorage GetData(int* dims, int ndims)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            Shape.InferNegativeCoordinates(Shape.dimensions, dims, ndims);

            // Convert int* to long* for GetSubshape
            long* longDims = stackalloc long[ndims];
            for (int i = 0; i < ndims; i++)
                longDims[i] = dims[i];

            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(longDims, ndims);
                // For non-broadcasted contiguous subshapes, use size (the actual data extent).
                // Only use BufferSize when the subshape itself is broadcasted.
                long sliceSize = ClampToBuffer(offset, shape.IsBroadcasted
                    ? (shape.BufferSize > 0 ? shape.BufferSize : shape.size)
                    : shape.size);
                // The slice below starts at the element GetSubshape names, so the view's shape starts at 0 — except for an
                // EMPTY index list, where GetSubshape returns this shape itself with offset 0 (the slice then starts at the
                // buffer's first element) and the shape's own offset must stay: a row of a broadcast at a non-zero offset
                // (np.broadcast_arrays(x[:, None], x)[0][1] has stride 0 and offset 1) took its stores (`row[...] = v`,
                // SetData) at element 0 when this offset was dropped.
                var adjustedShape = new Shape(shape.dimensions, shape.strides, ndims == 0 ? shape.offset : 0, sliceSize);
                // A sub-array of a read-only array is read-only (NumPy views inherit WRITEABLE): row 2 of a broadcast row
                // keeps no stride-0 axis, so its fresh flags would otherwise call it writeable.
                if (!this_shape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                // The mirror half, as Alias(Shape) applies it: a broadcast sub-array of a WRITEABLE broadcast parent
                // (np.broadcast_arrays' results, ndarray.setflags(write: true), an np.nditer view of a reduce operand's
                // stride-0 chunk) keeps the parent's write access — its computed read-only flag came from the broadcastness
                // it inherited, not from a protection decision. Without it a store into such a parent (`y[...] = v`, whose
                // copy takes this sub-array) raised "assignment destination is read-only" where NumPy writes.
                else if (adjustedShape.IsBroadcasted && !adjustedShape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(SliceForSubview(offset, sliceSize), adjustedShape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                var slices = new Slice[ndims];
                for (int i = 0; i < ndims; i++)
                {
                    slices[i] = Slice.Index(*(dims + i));
                }

                return GetView(slices);
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(longDims, ndims);
                // A sub-array view inherits writeability (see the int[] overload).
                if (!this_shape.IsWriteable)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                var view = new UnmanagedStorage(SliceForSubview(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
        }

        /// <summary>
        /// Gets a sub-array based on the given indices (long pointer version), returning a view that shares memory.
        /// </summary>
        /// <param name="dims">Pointer to an array of long dimension indices.</param>
        /// <param name="ndims">The number of indices in the array.</param>
        /// <returns>A new <see cref="UnmanagedStorage"/> representing the sub-array.</returns>
        /// <seealso cref="GetData(int*, int)"/>
        [MethodImpl(OptimizeAndInline)]
        public unsafe UnmanagedStorage GetData(long* dims, int ndims)
        {
            var this_shape = Shape;

            // ReSharper disable once ConvertIfStatementToReturnStatement
            Shape.InferNegativeCoordinates(Shape.dimensions, dims, ndims);
            if (this_shape.IsBroadcasted)
            {
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                // For non-broadcasted contiguous subshapes, use size (the actual data extent).
                // Only use BufferSize when the subshape itself is broadcasted.
                long sliceSize = ClampToBuffer(offset, shape.IsBroadcasted
                    ? (shape.BufferSize > 0 ? shape.BufferSize : shape.size)
                    : shape.size);
                // The slice below starts at the element GetSubshape names, so the view's shape starts at 0 — except for an
                // EMPTY index list, where GetSubshape returns this shape itself with offset 0 (the slice then starts at the
                // buffer's first element) and the shape's own offset must stay: a row of a broadcast at a non-zero offset
                // (np.broadcast_arrays(x[:, None], x)[0][1] has stride 0 and offset 1) took its stores (`row[...] = v`,
                // SetData) at element 0 when this offset was dropped.
                var adjustedShape = new Shape(shape.dimensions, shape.strides, ndims == 0 ? shape.offset : 0, sliceSize);
                // A sub-array of a read-only array is read-only (NumPy views inherit WRITEABLE): row 2 of a broadcast row
                // keeps no stride-0 axis, so its fresh flags would otherwise call it writeable.
                if (!this_shape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                // The mirror half, as Alias(Shape) applies it: a broadcast sub-array of a WRITEABLE broadcast parent
                // (np.broadcast_arrays' results, ndarray.setflags(write: true), an np.nditer view of a reduce operand's
                // stride-0 chunk) keeps the parent's write access — its computed read-only flag came from the broadcastness
                // it inherited, not from a protection decision. Without it a store into such a parent (`y[...] = v`, whose
                // copy takes this sub-array) raised "assignment destination is read-only" where NumPy writes.
                else if (adjustedShape.IsBroadcasted && !adjustedShape.IsWriteable)
                    adjustedShape = adjustedShape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);
                var view = UnmanagedStorage.CreateBroadcastedUnsafe(SliceForSubview(offset, sliceSize), adjustedShape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
            else if (!this_shape.IsContiguous)
            {
                // Non-contiguous shapes (stepped slices, transposed, etc.) cannot use
                // memory slicing. Create a view with indexed slices instead.
                var slices = new Slice[ndims];
                for (int i = 0; i < ndims; i++)
                {
                    slices[i] = Slice.Index(*(dims + i));
                }

                return GetView(slices);
            }
            else
            {
                // Contiguous shape: can take a direct memory slice.
                // GetSubshape computes the correct offset accounting for shape.offset.
                var (shape, offset) = this_shape.GetSubshape(dims, ndims);
                // A sub-array view inherits writeability (see the int[] overload).
                if (!this_shape.IsWriteable)
                    shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                var view = new UnmanagedStorage(SliceForSubview(offset, shape.Size), shape);
                view._baseStorage = _baseStorage ?? this;
                // a sub-array keeps its source's engine (Alias does too): the operations on it compute where the source's do
                view.Engine = Engine;
                // A lane-sharing descriptor (longdouble, Windows' intc) is not on the buffer: the new view carries it.
                view.AdoptLaneSharingDescr(_descr);
                view.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
                return view;
            }
        }

        /// <summary>
        ///     Get reference to internal data storage and cast (also copies) elements to new dtype if necessary
        /// </summary>
        /// <typeparam name="T">new storage data type</typeparam>
        /// <returns>reference to internal (casted) storage as T[]</returns>
        /// <remarks>Copies if <typeparamref name="T"/> does not equal to <see cref="DType"/> or if Shape is sliced.</remarks>
        public ArraySlice<T> GetData<T>() where T : unmanaged
        {
            if (!typeof(T).IsValidNPType())
                throw new NotSupportedException($"Type {typeof(T).Name} is not a valid np.dtype");

            if (!Shape.IsContiguous)
                return CloneData<T>();

            var internalArray = InternalArray;
            // Contiguity is offset-independent (NumPy-aligned): a contiguous
            // VIEW may occupy [offset, offset+size) inside a larger buffer —
            // np.split / np.unstack children are built that way. Narrow to the
            // logical window (zero-copy, still writes through) so both the
            // same-type return and the cast below see exactly the view's
            // elements rather than the whole backing buffer.
            if (internalArray != null && (Shape.offset != 0 || internalArray.Count != Shape.size))
                internalArray = internalArray.Slice(Shape.offset, Shape.size);

            if (internalArray is ArraySlice<T> ret)
                return ret;

            return _ChangeTypeOfArray<T>(internalArray);
        }

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as <typeparamref name="T"/>: the stored value for the
        ///     dtype's own type, NumPy's cast of it (the cast <c>astype</c> runs) for any other <typeparamref name="T"/>.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its axis, and
        ///     axes left out read their first element.</param>
        /// <typeparam name="T">The type to read the element as.</typeparam>
        /// <returns>The element as <typeparamref name="T"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing outside the array is read).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        /// <remarks>
        ///     Before parity QA-12 / QA-33 this loaded <c>sizeof(T)</c> bytes at an unchecked offset: <c>GetValue&lt;double&gt;(1)</c>
        ///     of a float32 array read elements 1 and 2 as one double, and a wrong coordinate read another row or memory
        ///     outside the array in Release.
        /// </remarks>
        public T GetValue<T>(int[] indices) where T : unmanaged
            => ReadAs<T>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as <typeparamref name="T"/>: the stored value for the
        ///     dtype's own type, NumPy's cast of it (the cast <c>astype</c> runs) for any other <typeparamref name="T"/>.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end
        ///     of its axis, and axes left out read their first element.</param>
        /// <typeparam name="T">The type to read the element as.</typeparam>
        /// <returns>The element as <typeparamref name="T"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's texts;
        ///     nothing outside the array is read).</exception>
        /// <exception cref="NotSupportedException">The storage has no element type.</exception>
        /// <remarks>See <see cref="GetValue{T}(int[])"/>.</remarks>
        public T GetValue<T>(params long[] indices) where T : unmanaged
            => ReadAs<T>(_shape.GetOffsetChecked(indices));

        /// <summary>
        /// Get reference to internal data storage
        /// </summary>
        /// <returns>reference to internal storage as System.Array</returns>
        [MethodImpl(Inline)]
        public IArraySlice GetData() => InternalArray;
        // #region Direct Getters
     
        // %foreach supported_dtypes,supported_dtypes_lowercase%
        // /// <summary>
        // ///     Retrieves value of type <see cref="#2"/> from internal storage.
        // /// </summary>
        // /// <param name="indices">The shape's indices to get.</param>
        // /// <returns></returns>
        // /// <exception cref="NullReferenceException">When <see cref="DType"/> is not <see cref="#2"/></exception>
        // public #2 Get#1(int[] indices)
            // => _slices.#1[_shape.GetOffset(indices)];

        // %
        // #endregion

        #region Direct Getters
     
        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="bool"/>: the stored value itself in a
        ///     bool array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="bool"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public bool GetBoolean(int[] indices)
            => ReadAs<bool>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="sbyte"/>: the stored value itself in a
        ///     int8 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="sbyte"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public sbyte GetSByte(int[] indices)
            => ReadAs<sbyte>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="byte"/>: the stored value itself in a
        ///     uint8 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="byte"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public byte GetByte(int[] indices)
            => ReadAs<byte>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="short"/>: the stored value itself in a
        ///     int16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="short"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public short GetInt16(int[] indices)
            => ReadAs<short>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="ushort"/>: the stored value itself in a
        ///     uint16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="ushort"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public ushort GetUInt16(int[] indices)
            => ReadAs<ushort>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="int"/>: the stored value itself in a
        ///     int32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="int"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public int GetInt32(int[] indices)
            => ReadAs<int>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="uint"/>: the stored value itself in a
        ///     uint32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="uint"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public uint GetUInt32(int[] indices)
            => ReadAs<uint>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="long"/>: the stored value itself in a
        ///     int64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="long"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public long GetInt64(int[] indices)
            => ReadAs<long>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="ulong"/>: the stored value itself in a
        ///     uint64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="ulong"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public ulong GetUInt64(int[] indices)
            => ReadAs<ulong>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="char"/>: the stored value itself in a
        ///     char array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="char"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public char GetChar(int[] indices)
            => ReadAs<char>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="Half"/>: the stored value itself in a
        ///     float16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="Half"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public Half GetHalf(int[] indices)
            => ReadAs<Half>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="double"/>: the stored value itself in a
        ///     float64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="double"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public double GetDouble(int[] indices)
            => ReadAs<double>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="float"/>: the stored value itself in a
        ///     float32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="float"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public float GetSingle(int[] indices)
            => ReadAs<float>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="decimal"/>: the stored value itself in a
        ///     decimal array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="decimal"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public decimal GetDecimal(int[] indices)
            => ReadAs<decimal>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="System.Numerics.Complex"/>: the stored value itself in a
        ///     complex128 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis; a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="System.Numerics.Complex"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public System.Numerics.Complex GetComplex(int[] indices)
            => ReadAs<System.Numerics.Complex>(_shape.GetOffsetChecked(indices));

        #endregion

        #region Direct Getters (long[] overloads)

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="bool"/>: the stored value itself in a
        ///     bool array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="bool"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public bool GetBoolean(params long[] indices)
            => ReadAs<bool>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="sbyte"/>: the stored value itself in a
        ///     int8 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="sbyte"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public sbyte GetSByte(params long[] indices)
            => ReadAs<sbyte>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="byte"/>: the stored value itself in a
        ///     uint8 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="byte"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public byte GetByte(params long[] indices)
            => ReadAs<byte>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="short"/>: the stored value itself in a
        ///     int16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="short"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public short GetInt16(params long[] indices)
            => ReadAs<short>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="ushort"/>: the stored value itself in a
        ///     uint16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="ushort"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public ushort GetUInt16(params long[] indices)
            => ReadAs<ushort>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="int"/>: the stored value itself in a
        ///     int32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="int"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public int GetInt32(params long[] indices)
            => ReadAs<int>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="uint"/>: the stored value itself in a
        ///     uint32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="uint"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public uint GetUInt32(params long[] indices)
            => ReadAs<uint>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="long"/>: the stored value itself in a
        ///     int64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="long"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public long GetInt64(params long[] indices)
            => ReadAs<long>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="ulong"/>: the stored value itself in a
        ///     uint64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="ulong"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public ulong GetUInt64(params long[] indices)
            => ReadAs<ulong>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="char"/>: the stored value itself in a
        ///     char array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="char"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public char GetChar(params long[] indices)
            => ReadAs<char>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="Half"/>: the stored value itself in a
        ///     float16 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="Half"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public Half GetHalf(params long[] indices)
            => ReadAs<Half>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="double"/>: the stored value itself in a
        ///     float64 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="double"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public double GetDouble(params long[] indices)
            => ReadAs<double>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="float"/>: the stored value itself in a
        ///     float32 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="float"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public float GetSingle(params long[] indices)
            => ReadAs<float>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="decimal"/>: the stored value itself in a
        ///     decimal array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="decimal"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public decimal GetDecimal(params long[] indices)
            => ReadAs<decimal>(_shape.GetOffsetChecked(indices));

        /// <summary>
        ///     Reads the element at <paramref name="indices"/> as a <see cref="System.Numerics.Complex"/>: the stored value itself in a
        ///     complex128 array, NumPy's cast of it (the cast <c>astype</c> runs) in an array of any other dtype.
        /// </summary>
        /// <param name="indices">One coordinate per leading axis (long version); a negative coordinate counts from the end of its
        ///     axis, and axes left out read their first element.</param>
        /// <returns>The element as a <see cref="System.Numerics.Complex"/>.</returns>
        /// <exception cref="IndexError">A coordinate outside its axis, or more coordinates than dimensions (NumPy's
        ///     texts; nothing outside the array is read).</exception>
        public System.Numerics.Complex GetComplex(params long[] indices)
            => ReadAs<System.Numerics.Complex>(_shape.GetOffsetChecked(indices));

        #endregion

        #endregion
    }
}
