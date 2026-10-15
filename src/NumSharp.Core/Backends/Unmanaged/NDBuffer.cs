using System;
using System.Buffers;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     One buffer handed to <c>np.ndarray(shape, buffer=...)</c>, resolved to a byte range: the C# port of NumPy's
    ///     buffer protocol as <c>PyArray_BufferConverter</c> (<c>numpy/_core/src/multiarray/conversion_utils.c</c>) uses it.
    ///     It records where the bytes start, how many there are, whether they may be written, and who keeps them alive.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Accepted buffers</b> (NumPy: any object that exports a contiguous buffer):
    ///     <list type="bullet">
    ///         <item><see cref="NDArray"/> that is C- or F-contiguous (NumPy's <c>ndarray</c>); its bytes are read in
    ///         memory order, and its writeability carries over.</item>
    ///         <item><see cref="np.MemoryView"/> (<c>ndarray.data</c>, NumPy's <c>memoryview</c>) over such an array.</item>
    ///         <item><see cref="IArraySlice"/>, NumSharp's own raw buffer.</item>
    ///         <item>A <see cref="Array"/> of any rank whose element type holds no references (<c>byte[]</c>,
    ///         <c>double[,]</c>, a user struct), the analog of <c>bytearray</c> or <c>array.array</c>: pinned for as long
    ///         as an array uses it, writeable.</item>
    ///         <item><see cref="ArraySegment{T}"/> and <see cref="Memory{T}"/> of <see cref="byte"/> (writeable) and
    ///         <see cref="ReadOnlyMemory{T}"/> of <see cref="byte"/> (read-only, NumPy's <c>bytes</c>): pinned through
    ///         <see cref="Memory{T}.Pin"/>.</item>
    ///         <item><see cref="Half"/>, <see cref="char"/> and <see cref="decimal"/>, the C# values NumSharp treats as
    ///         NumPy scalars (<c>np.float16(1.5)</c> exports its bytes): a read-only copy of the value's bytes.</item>
    ///     </list>
    ///     Anything else is NumPy's <c>TypeError: a bytes-like object is required, not 'list'</c>, with the Python type
    ///     name of the C# value (<see cref="PythonTypeName"/>).
    ///     </para>
    ///     <para>
    ///     <b>Lifetime.</b> A buffer NumSharp already manages (an array, a memoryview, a slice) is BORROWED: the result
    ///     counts its references on that memory's own block, so disposing the buffer array while the result lives does
    ///     not free the memory. A managed buffer is PINNED by <see cref="Acquire"/>; the pin belongs to this object until
    ///     <see cref="TakeRoot"/> hands it to the result's memory block, which frees it when the last array over it is
    ///     released or collected. Every failure between the two MUST call <see cref="Release"/>: this object has no
    ///     finalizer and neither has a <see cref="GCHandle"/>, so a missed release keeps the managed array pinned and
    ///     alive for the life of the process.
    ///     </para>
    /// </remarks>
    internal sealed unsafe class NDBuffer
    {
        /// <summary>The first byte of the buffer (NumPy's <c>PyArray_Chunk.ptr</c>).</summary>
        internal readonly byte* Start;

        /// <summary>The buffer's length in bytes (NumPy's <c>PyArray_Chunk.len</c>).</summary>
        internal readonly long Length;

        /// <summary>
        ///     Whether the exporter granted a WRITABLE buffer. NumPy asks for one first and retries read-only, so a
        ///     read-only exporter gives a read-only array whose <c>WRITEABLE</c> flag cannot be turned on again.
        /// </summary>
        internal readonly bool Writable;

        /// <summary>The storage that becomes the result's <c>base</c> for a borrowed NumSharp array; null otherwise.</summary>
        private readonly UnmanagedStorage _borrowedRoot;

        /// <summary>
        ///     The slice whose memory block counts the result's references for borrowed NumSharp memory (an array's or a
        ///     slice's own <see cref="IArraySlice"/>); null for a buffer pinned here.
        /// </summary>
        private readonly IArraySlice _borrowedArcOwner;

        /// <summary>The pin of a managed <see cref="Array"/> buffer, owned until <see cref="TakeRoot"/>.</summary>
        private GCHandle _pin;

        /// <summary>The pin of a <see cref="Memory{T}"/> or <see cref="ReadOnlyMemory{T}"/> buffer, owned until
        ///     <see cref="TakeRoot"/>.</summary>
        private MemoryHandle _memoryPin;

        /// <summary>True when <see cref="_memoryPin"/> holds the pin (instead of <see cref="_pin"/>).</summary>
        private readonly bool _hasMemoryPin;

        /// <summary>Set once the pin is handed over (<see cref="TakeRoot"/>) or freed (<see cref="Release"/>).</summary>
        private bool _done;

        /// <summary>Records a borrowed NumSharp buffer: nothing to free, the owner's block counts references.</summary>
        /// <param name="start">The first byte.</param>
        /// <param name="length">The length in bytes.</param>
        /// <param name="writable">Whether the bytes may be written.</param>
        /// <param name="borrowedRoot">The storage that becomes the result's base, or null to build one over the bytes.</param>
        /// <param name="borrowedArcOwner">The slice whose block counts the result's references.</param>
        private NDBuffer(byte* start, long length, bool writable, UnmanagedStorage borrowedRoot, IArraySlice borrowedArcOwner)
        {
            Start = start;
            Length = length;
            Writable = writable;
            _borrowedRoot = borrowedRoot;
            _borrowedArcOwner = borrowedArcOwner;
        }

        /// <summary>Records a managed array pinned here; the pin is this object's until handed over or released.</summary>
        /// <param name="pin">The pinned handle of the array.</param>
        /// <param name="length">The array's length in bytes.</param>
        /// <param name="writable">Whether the bytes may be written.</param>
        private NDBuffer(GCHandle pin, long length, bool writable)
        {
            _pin = pin;
            Start = (byte*)pin.AddrOfPinnedObject();
            Length = length;
            Writable = writable;
        }

        /// <summary>Records a <see cref="Memory{T}"/> pinned here; the pin is this object's until handed over or released.</summary>
        /// <param name="pin">The memory's pin.</param>
        /// <param name="length">The memory's length in bytes.</param>
        /// <param name="writable">Whether the bytes may be written.</param>
        private NDBuffer(MemoryHandle pin, long length, bool writable)
        {
            _memoryPin = pin;
            _hasMemoryPin = true;
            Start = (byte*)pin.Pointer;
            Length = length;
            Writable = writable;
        }

        /// <summary>
        ///     Resolves <paramref name="buffer"/> to its bytes — NumPy's <c>PyArray_BufferConverter</c>, which asks for a
        ///     contiguous writable buffer and falls back to a read-only one.
        /// </summary>
        /// <param name="buffer">The buffer argument of <c>np.ndarray</c>; null means "no buffer" (NumPy's <c>None</c>).</param>
        /// <returns>The resolved buffer, or null for a null <paramref name="buffer"/>.</returns>
        /// <exception cref="ValueError">An <see cref="NDArray"/> that is neither C- nor F-contiguous (NumPy:
        ///     <c>ndarray is not contiguous</c>).</exception>
        /// <exception cref="BufferError">A <see cref="np.MemoryView"/> over such an array (NumPy:
        ///     <c>memoryview: underlying buffer is not contiguous</c>).</exception>
        /// <exception cref="TypeError">An object that is not a buffer (NumPy: <c>a bytes-like object is required, not
        ///     'list'</c>), including a C# array whose elements hold references.</exception>
        /// <remarks>A pinned result must reach <see cref="TakeRoot"/> or <see cref="Release"/>.</remarks>
        internal static NDBuffer Acquire(object buffer)
        {
            switch (buffer)
            {
                case null:
                    return null;

                case NDArray nd:
                    // NumPy's array exporter refuses a non-contiguous array with a ValueError of its own.
                    if (!IsOneSegment(nd))
                        throw new ValueError("ndarray is not contiguous");
                    return FromNDArray(nd, nd.Shape.IsWriteable);

                case np.MemoryView view:
                    // A memoryview re-exports its array; CPython's memoryview raises BufferError for the contiguous request.
                    if (!IsOneSegment(view.obj))
                        throw new BufferError("memoryview: underlying buffer is not contiguous");
                    return FromNDArray(view.obj, !view.@readonly);

                case IArraySlice slice:
                    // NumSharp's raw buffer: borrowed, writable, and the base is a byte view built over it in TakeRoot.
                    return new NDBuffer((byte*)slice.Address, slice.BytesLength, writable: true, borrowedRoot: null, borrowedArcOwner: slice);

                case ArraySegment<byte> segment:
                    // A default segment has no array: it is an empty buffer, as NumPy's b'' is.
                    return FromMemory(segment.Array is null ? Memory<byte>.Empty : new Memory<byte>(segment.Array, segment.Offset, segment.Count));

                case Memory<byte> memory:
                    return FromMemory(memory);

                case ReadOnlyMemory<byte> readOnly:
                    // NumPy's bytes: readable, never writable, so the result is read-only for good.
                    return new NDBuffer(readOnly.Pin(), readOnly.Length, writable: false);

                case Half h:
                    return FromScalarBytes(h);

                case char c:
                    return FromScalarBytes(c);

                case decimal d:
                    return FromScalarBytes(d);

                case Array array when array.GetType().GetElementType()!.IsValueType:
                    return FromManagedArray(array);

                default:
                    throw new TypeError($"a bytes-like object is required, not '{PythonTypeName(buffer)}'");
            }
        }

        /// <summary>
        ///     Hands the buffer to the result: the storage that becomes the result's <c>base</c>, and the slice whose block
        ///     counts the result's references. A pin taken by <see cref="Acquire"/> moves into a new memory block here and
        ///     is freed when the last array over it is released.
        /// </summary>
        /// <param name="typeCode">The result's storage type. A buffer pinned here gets a root slice of this type, which
        ///     the result shares when its first element lies on the element grid (one slice instead of two).</param>
        /// <param name="itemSize">The size of one <paramref name="typeCode"/> element in bytes.</param>
        /// <param name="arcOwner">Receives the slice the result shares or forwards its reference counting to.</param>
        /// <returns>The base storage: the buffer array's own owner for a borrowed array, else a storage over the whole
        ///     buffer that does not own its data (NumPy's non-array base object): <c>uint8</c> for a NumSharp slice,
        ///     <paramref name="typeCode"/> elements for a buffer pinned here.</returns>
        /// <exception cref="InvalidOperationException">Called twice, or after <see cref="Release"/>.</exception>
        /// <remarks>Call it only after every check has passed: from here on the pin belongs to a memory block, and a later
        ///     failure leaves it to that block's finalizer instead of freeing it at once.</remarks>
        internal UnmanagedStorage TakeRoot(NPTypeCode typeCode, int itemSize, out IArraySlice arcOwner)
        {
            if (_done)
                throw new InvalidOperationException("The buffer was already handed over or released.");
            _done = true;

            if (_borrowedRoot != null)
            {
                arcOwner = _borrowedArcOwner;
                return _borrowedRoot;
            }

            IArraySlice rootSlice;
            if (_borrowedArcOwner != null)
            {
                // A NumSharp slice: a byte alias whose reference counting forwards to the slice's own block. The result
                // shares the slice itself when its type and grid match (arcOwner is the slice, not this alias).
                rootSlice = ArraySlice.WrapShared<byte>(Start, Length, _borrowedArcOwner);
            }
            else
            {
                // A pin taken here: the root slice owns it and frees it when the last array over the buffer is released
                // (or the block is finalized). It covers the buffer's whole elements; a trailing partial element is never
                // addressed by an array the bounds check accepted.
                Action release;
                if (_hasMemoryPin)
                {
                    var memoryPin = _memoryPin;
                    release = () => memoryPin.Dispose();
                }
                else
                {
                    var pin = _pin;
                    release = () => pin.Free();
                }

                rootSlice = ArraySlice.WrapOwned(typeCode, Start, Length / itemSize, release);
                arcOwner = rootSlice;
                return NewForeignRoot(rootSlice, Writable);
            }

            arcOwner = _borrowedArcOwner;
            return NewForeignRoot(rootSlice, Writable);
        }

        /// <summary>
        ///     The storage that stands for a foreign buffer as the result's base: over the whole buffer, not owning its
        ///     data, and read-only for good when the exporter refused a writable buffer.
        /// </summary>
        /// <param name="rootSlice">The slice over the buffer.</param>
        /// <param name="writable">Whether the buffer may be written.</param>
        /// <returns>The base storage.</returns>
        private static UnmanagedStorage NewForeignRoot(IArraySlice rootSlice, bool writable)
        {
            var root = new UnmanagedStorage(rootSlice, Shape.Vector(rootSlice.Count));
            // NumPy's base is the buffer object itself, which an ndarray over it does not own.
            root.ExternalBase = true;
            root.OnReshaped();
            if (!writable)
            {
                // A read-only exporter refuses a writable buffer later too: setflags(write=True) must fail on the result.
                root.WriteProtected = true;
                root.SetShapeUnsafe(root.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            }

            return root;
        }

        /// <summary>
        ///     Frees a pin taken by <see cref="Acquire"/> when the call fails before <see cref="TakeRoot"/>. A borrowed
        ///     buffer has nothing to free. Safe to call more than once and after <see cref="TakeRoot"/> (then it does
        ///     nothing).
        /// </summary>
        internal void Release()
        {
            if (_done)
                return;
            _done = true;

            if (_hasMemoryPin)
                _memoryPin.Dispose();
            else if (_pin.IsAllocated)
                _pin.Free();
        }

        /// <summary>NumPy's <c>PyArray_ISONESEGMENT</c>: is the array C- or F-contiguous (a buffer the protocol can
        ///     export)?</summary>
        /// <param name="nd">The array.</param>
        /// <returns>True for a C- or F-contiguous array, including every empty and 0-d one.</returns>
        private static bool IsOneSegment(NDArray nd) => nd.Shape.IsContiguous || nd.Shape.IsFContiguous;

        /// <summary>Borrows the bytes of a contiguous <see cref="NDArray"/>.</summary>
        /// <param name="nd">The array (C- or F-contiguous).</param>
        /// <param name="writable">Whether the exporter grants a writable buffer.</param>
        /// <returns>The borrowed buffer: base = the array's owner (NumPy collapses the base chain to the first owner),
        ///     references counted on the array's own slice.</returns>
        private static NDBuffer FromNDArray(NDArray nd, bool writable)
        {
            var storage = nd.Storage;
            int itemSize = nd.dtypesize;
            // The first element in memory order: a contiguous array's data starts at its offset, whatever the order.
            byte* start = (byte*)storage.Address + nd.Shape.offset * itemSize;
            return new NDBuffer(start, nd.size * itemSize, writable, storage._baseStorage ?? storage, storage.InternalArray);
        }

        /// <summary>Pins a <see cref="Memory{T}"/> of bytes as a writable buffer.</summary>
        /// <param name="memory">The memory (empty is allowed).</param>
        /// <returns>The pinned buffer.</returns>
        private static NDBuffer FromMemory(Memory<byte> memory) => new NDBuffer(memory.Pin(), memory.Length, writable: true);

        /// <summary>
        ///     Pins a managed array whose element type is a value type. The array's bytes are its elements in their
        ///     memory order (row-major for a multi-dimensional array).
        /// </summary>
        /// <param name="array">The array.</param>
        /// <returns>The pinned, writable buffer.</returns>
        /// <exception cref="TypeError">The element type holds references (a struct with a reference field): such an array
        ///     cannot be pinned, and it is not a buffer of bytes.</exception>
        private static NDBuffer FromManagedArray(Array array)
        {
            GCHandle pin;
            try
            {
                pin = GCHandle.Alloc(array, GCHandleType.Pinned);
            }
            catch (ArgumentException)
            {
                // "Object contains references": the C# analog of a Python list of objects.
                throw new TypeError("a bytes-like object is required, not 'list'");
            }

            // The element size is the distance between two element addresses (the CLR's component size; no public API
            // names it for an element type known only at run time). Both addresses are taken while pinned, so the array
            // cannot move between the two reads. The computation reads no element, so it is valid for an empty array too.
            long itemSize = (long)Marshal.UnsafeAddrOfPinnedArrayElement(array, 1) - (long)Marshal.UnsafeAddrOfPinnedArrayElement(array, 0);
            return new NDBuffer(pin, array.LongLength * itemSize, writable: true);
        }

        /// <summary>
        ///     The buffer of a C# value NumSharp treats as a NumPy scalar (<see cref="Half"/>, <see cref="char"/>,
        ///     <see cref="decimal"/>): a read-only copy of its bytes, as NumPy's scalar types export a read-only buffer.
        /// </summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="value">The value.</param>
        /// <returns>The pinned, read-only buffer over a copy of the value's bytes.</returns>
        private static NDBuffer FromScalarBytes<T>(T value) where T : unmanaged
        {
            byte[] bytes = MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref value, 1)).ToArray();
            return new NDBuffer(GCHandle.Alloc(bytes, GCHandleType.Pinned), bytes.Length, writable: false);
        }

        /// <summary>
        ///     The Python type name NumPy's buffer error quotes for a C# value, by NumSharp's mapping of C# values onto
        ///     Python objects: <c>str</c>, <c>bool</c>, <c>int</c>, <c>float</c>, <c>complex</c>, <c>tuple</c>,
        ///     <c>dict</c>, <c>set</c>, <c>list</c>; any other object reports its CLR type name.
        /// </summary>
        /// <param name="value">The rejected buffer (never null).</param>
        /// <returns>The name to quote.</returns>
        internal static string PythonTypeName(object value)
        {
            switch (value)
            {
                case string:
                    return "str";
                case bool:
                    return "bool";
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return "int";
                case float or double:
                    return "float";
                case Complex:
                    return "complex";
                case ITuple:
                    return "tuple";
                case IDictionary:
                    return "dict";
                case IList:
                    // object[], string[], jagged arrays, List<T>: Python's list.
                    return "list";
            }

            foreach (var i in value.GetType().GetInterfaces())
                if (i.IsGenericType && i.GetGenericTypeDefinition() == typeof(ISet<>))
                    return "set";

            return value.GetType().Name;
        }
    }
}
