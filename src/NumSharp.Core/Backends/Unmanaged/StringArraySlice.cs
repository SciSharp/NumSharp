using System;
using System.Collections;
using System.Collections.Generic;
using NumSharp.Backends.Strings;
using NumSharp.Utilities;

namespace NumSharp.Backends.Unmanaged
{
    /// <summary>
    ///     The ELEMENT view of a fixed-width string buffer (<c>S</c> or <c>U</c>) — the <see cref="IArraySlice"/> a string
    ///     <see cref="UnmanagedStorage"/> holds as its <see cref="UnmanagedStorage.InternalArray"/>. The bytes live in an
    ///     ordinary <see cref="ArraySlice{T}"/> of <see cref="byte"/> (<see cref="Bytes"/>, so pooling, reference counting and
    ///     lifetime are exactly the numeric lanes'); this object only says how to count them: one element is
    ///     <see cref="ItemSize"/> bytes, its type is <see cref="NPTypeCode.String"/>, and its descriptor is
    ///     <see cref="Descr"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why a dedicated slice instead of a raw byte slice.</b> Much of the storage layer rebuilds a storage from a
    ///     slice of its parent — contiguous slicing re-seats the data pointer (<c>InternalArray.Slice(offset, size)</c>), an
    ///     empty view allocates a zero-length slice, <c>Clone()</c> clones the slice, <c>GetData(int[])</c> slices a
    ///     sub-array. Those paths count in ELEMENTS and take the new storage's type from the slice's
    ///     <see cref="TypeCode"/>. A plain byte slice would silently turn a string array into a <c>uint8</c> array of the
    ///     wrong length on every one of them; this slice keeps element semantics (<see cref="Count"/>,
    ///     <see cref="Slice(long, long)"/> and <see cref="ItemLength"/> are per string element) and carries the descriptor,
    ///     so each of those paths produces a correct string storage without knowing about strings. Any path that instead
    ///     dispatches a numeric kernel on <see cref="TypeCode"/> meets <see cref="NPTypeCode.String"/> and refuses.
    ///     </para>
    ///     <para>
    ///     <b>Layout.</b> Element <c>i</c> occupies bytes <c>[i * ItemSize, (i + 1) * ItemSize)</c> of <see cref="Bytes"/>:
    ///     for <c>S</c> one byte per character, for <c>U</c> one native-endian <c>uint32</c> code point per character; both
    ///     NUL-padded. Typed access exists only for the raw units (<c>byte</c>, and <c>uint</c> for <c>U</c>), which is
    ///     what the unit-view movement kernels and the byte I/O paths use; everything else goes through the codec
    ///     (<see cref="GetIndex(long)"/>, <see cref="SetIndex(long, object)"/>).
    ///     </para>
    /// </remarks>
    public sealed unsafe class StringArraySlice : IArraySlice
    {
        /// <summary>
        ///     Wraps <paramref name="bytes"/> as <c>bytes.Count / descr.itemsize</c> string elements of <paramref name="descr"/>.
        /// </summary>
        /// <param name="bytes">The raw buffer; its length must be a whole number of elements.</param>
        /// <param name="descr">A sized, native-byte-order <c>S</c> or <c>U</c> descriptor.</param>
        /// <exception cref="ArgumentNullException"><paramref name="descr"/> is null.</exception>
        /// <exception cref="ArgumentException">The descriptor is not a sized native string descriptor, or the buffer is not a
        /// whole number of its elements.</exception>
        public StringArraySlice(ArraySlice<byte> bytes, DType descr)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            if (descr.Meta is not StringDTypeMeta meta)
                throw new ArgumentException($"{descr.ToString(true)} is not a fixed-width string dtype.", nameof(descr));
            // An array element has at least one character: NumPy grows an unsized S0/U0 to S1/U1 at allocation, and
            // NumSharp buffers are always native-endian (a '>U5' request is canonicalised before it gets here).
            if (descr.itemsize <= 0)
                throw new ArgumentException($"{descr.ToString(true)} is unsized; a string buffer needs a positive item size.", nameof(descr));
            if (!descr.isnative)
                throw new ArgumentException($"{descr.ToString(true)} is not native-endian; string buffers are stored in host order.", nameof(descr));
            if (bytes.Count % descr.itemsize != 0)
                throw new ArgumentException($"A buffer of {bytes.Count} bytes is not a whole number of {descr.itemsize}-byte elements.", nameof(bytes));

            Bytes = bytes;
            Descr = descr;
            ItemSize = descr.itemsize;
            IsUnicode = meta.IsUnicode;
            Chars = ItemSize / meta.UnitSize;
            Count = bytes.Count / ItemSize;
        }

        /// <summary>
        ///     Allocates a buffer for <paramref name="count"/> elements of <paramref name="descr"/> from the shared pool.
        /// </summary>
        /// <param name="descr">A sized, native string descriptor.</param>
        /// <param name="count">The number of string elements.</param>
        /// <param name="fillZeros">True to NUL-fill (every element reads <c>""</c> / <c>b""</c>); false leaves pool noise,
        /// for callers that write every byte.</param>
        /// <returns>The new slice.</returns>
        /// <exception cref="ArgumentException">See the constructor.</exception>
        /// <exception cref="OutOfMemoryException">The allocation failed.</exception>
        public static StringArraySlice Allocate(DType descr, long count, bool fillZeros)
        {
            if (descr is null)
                throw new ArgumentNullException(nameof(descr));
            // The caller has already run AllocationGuard on (dims, itemsize), so the product cannot overflow.
            return new StringArraySlice(ArraySlice<byte>.Allocate(count * descr.itemsize, fillZeros), descr);
        }

        /// <summary>The raw buffer: <see cref="Count"/> × <see cref="ItemSize"/> bytes.</summary>
        public ArraySlice<byte> Bytes { get; }

        /// <summary>The element descriptor (sized, native): <c>|S5</c>, <c>&lt;U3</c>.</summary>
        public DType Descr { get; }

        /// <summary>Bytes per element (<c>Descr.itemsize</c>).</summary>
        public int ItemSize { get; }

        /// <summary>Characters per element: <see cref="ItemSize"/> for <c>S</c>, <see cref="ItemSize"/> / 4 for <c>U</c>.</summary>
        public int Chars { get; }

        /// <summary>True for <c>U</c> (UCS-4 units), false for <c>S</c> (bytes).</summary>
        public bool IsUnicode { get; }

        // ---------------------------------------------------------------------------------------------
        // IMemoryBlock — element semantics
        // ---------------------------------------------------------------------------------------------

        /// <summary>The number of string ELEMENTS (not bytes).</summary>
        public long Count { get; }

        /// <summary>Bytes per element — the <see cref="IMemoryBlock"/> "size of a single item".</summary>
        public int ItemLength => ItemSize;

        /// <summary>Total bytes (<see cref="Count"/> × <see cref="ItemSize"/>).</summary>
        public long BytesLength => Count * ItemSize;

        /// <summary>Always <see cref="NPTypeCode.String"/>: the shared lane of both string classes.</summary>
        public NPTypeCode TypeCode => NPTypeCode.String;

        /// <summary>The first byte of the buffer.</summary>
        public byte* Address => Bytes.Address;

        /// <inheritdoc/>
        void* IMemoryBlock.Address => Bytes.VoidAddress;

        /// <summary>The underlying byte block (pool, reference count).</summary>
        public IMemoryBlock MemoryBlock => Bytes.MemoryBlock;

        // ---------------------------------------------------------------------------------------------
        // Element access through the codec
        // ---------------------------------------------------------------------------------------------

        /// <summary>
        ///     Element <paramref name="index"/> as its C# value: a <see cref="string"/> for <c>U</c>, a <c>byte[]</c> for
        ///     <c>S</c>, trailing NULs removed (NumPy's <c>getitem</c>).
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <returns>The value.</returns>
        /// <exception cref="SystemError">A <c>U</c> unit is above <c>U+10FFFF</c>.</exception>
        public object GetIndex(long index)
        {
            byte* p = Bytes.Address + index * ItemSize;
            return IsUnicode ? NDStringCodec.ReadUcs4((uint*)p, Chars) : NDStringCodec.ReadBytes(p, Chars);
        }

        /// <summary>
        ///     Stores <paramref name="value"/> into element <paramref name="index"/> with NumPy's <c>setitem</c> rules
        ///     (truncate to the width, NUL-pad; a <see cref="string"/> into <c>S</c> is ASCII-encoded, a <c>byte[]</c> into
        ///     <c>U</c> ASCII-decoded).
        /// </summary>
        /// <param name="index">The element index.</param>
        /// <param name="value">A <see cref="string"/>, <c>byte[]</c>, <see cref="char"/>, or a 0-d string <see cref="NDArray"/>.</param>
        /// <exception cref="UnicodeEncodeError">A non-ASCII string into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="NotSupportedException">A value that is not text (numbers become text in the computational phase).</exception>
        public void SetIndex(long index, object value)
            => NDStringStorage.WriteElement(Bytes.Address + index * ItemSize, IsUnicode, Chars, value);

        /// <summary>Element access by index (see <see cref="GetIndex(long)"/> / <see cref="SetIndex(long, object)"/>).</summary>
        /// <param name="index">The element index.</param>
        public object this[long index]
        {
            get => GetIndex(index);
            set => SetIndex(index, value);
        }

        /// <summary>Typed element access is meaningless for text; refused.</summary>
        /// <typeparam name="T">Any unmanaged type.</typeparam>
        /// <param name="index">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public T GetIndex<T>(long index) where T : unmanaged
            => throw new NotSupportedException($"A {Descr.ToString(true)} element is text; read it with GetIndex(long) (a string / byte[]), not as {typeof(T).Name}.");

        /// <summary>Typed element writes are meaningless for text; refused.</summary>
        /// <typeparam name="T">Any unmanaged type.</typeparam>
        /// <param name="index">Ignored.</param>
        /// <param name="value">Ignored.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void SetIndex<T>(long index, T value) where T : unmanaged
            => throw new NotSupportedException($"A {Descr.ToString(true)} element is text; write it with SetIndex(long, object) (a string / byte[]), not as {typeof(T).Name}.");

        /// <summary>
        ///     Writes <paramref name="value"/> into every element: the value is encoded ONCE into the first element and the
        ///     bytes are replicated by doubling block copies (<c>np.full</c>, <c>fill</c>).
        /// </summary>
        /// <param name="value">The text value (see <see cref="SetIndex(long, object)"/>).</param>
        /// <exception cref="UnicodeEncodeError">See <see cref="SetIndex(long, object)"/>.</exception>
        /// <exception cref="UnicodeDecodeError">See <see cref="SetIndex(long, object)"/>.</exception>
        /// <exception cref="NotSupportedException">See <see cref="SetIndex(long, object)"/>.</exception>
        public void Fill(object value)
        {
            if (Count == 0)
                return;
            SetIndex(0, value);
            NDStringStorage.ReplicateFirstElement(Bytes.Address, ItemSize, Count);
        }

        // ---------------------------------------------------------------------------------------------
        // Slicing / cloning — element units
        // ---------------------------------------------------------------------------------------------

        /// <summary>The elements from <paramref name="start"/> to the end, sharing memory.</summary>
        /// <param name="start">The first element.</param>
        /// <returns>A string slice over the same bytes.</returns>
        public IArraySlice Slice(long start) => Slice(start, Count - start);

        /// <summary><paramref name="count"/> elements from <paramref name="start"/>, sharing memory (the byte range is scaled by <see cref="ItemSize"/>).</summary>
        /// <param name="start">The first element.</param>
        /// <param name="count">The number of elements.</param>
        /// <returns>A string slice over the same bytes.</returns>
        public IArraySlice Slice(long start, long count)
            => new StringArraySlice(Bytes.Slice(start * ItemSize, count * ItemSize), Descr);

        /// <summary>A deep copy of the elements (fresh buffer, same descriptor).</summary>
        /// <returns>The copy.</returns>
        public IArraySlice Clone() => new StringArraySlice(Bytes.Clone(), Descr);

        /// <inheritdoc/>
        object ICloneable.Clone() => Clone();

        /// <summary>A typed clone is only defined for the raw <c>byte</c> units.</summary>
        /// <typeparam name="T"><see cref="byte"/>.</typeparam>
        /// <returns>A copy of the raw bytes.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> is not <see cref="byte"/>.</exception>
        public ArraySlice<T> Clone<T>() where T : unmanaged
        {
            if (typeof(T) != typeof(byte))
                throw new NotSupportedException($"A {Descr.ToString(true)} buffer clones only as its raw bytes, not as {typeof(T).Name}.");
            return (ArraySlice<T>)(object)Bytes.Clone();
        }

        // ---------------------------------------------------------------------------------------------
        // Raw unit access (byte everywhere; uint for U)
        // ---------------------------------------------------------------------------------------------

        /// <summary>The raw units as a span: <c>byte</c> (any string class) or <c>uint</c> (a <c>U</c> buffer's code points).</summary>
        /// <typeparam name="T"><see cref="byte"/>, or <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <returns>A span over the whole buffer, no copy.</returns>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        public UnmanagedSpan<T> AsSpan<T>() where T : unmanaged
        {
            CheckUnitType<T>();
            return new UnmanagedSpan<T>((T*)Bytes.Address, Bytes.Count / sizeof(T));
        }

        /// <summary>
        ///     Copies the RAW bytes of the buffer into <paramref name="destination"/> (a byte-level copy, exactly like
        ///     <see cref="ArraySlice{T}"/>'s implementation of this member — the destination's element type only sizes it).
        /// </summary>
        /// <typeparam name="T">Any element type; the destination must hold at least <see cref="BytesLength"/> bytes.</typeparam>
        /// <param name="destination">The target span.</param>
        /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
        public void CopyTo<T>(Span<T> destination)
        {
            long sourceBytes = Bytes.Count;
            long destinationBytes = (long)destination.Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
            if ((ulong)sourceBytes > (ulong)destinationBytes)
                throw new ArgumentException("Destination was too short.");
            Buffer.MemoryCopy(Bytes.Address, System.Runtime.CompilerServices.Unsafe.AsPointer(ref destination.GetPinnableReference()), destinationBytes, sourceBytes);
        }

        /// <summary>Copies the RAW bytes of the buffer into <paramref name="destination"/> (see <see cref="CopyTo{T}(Span{T})"/>).</summary>
        /// <typeparam name="T">Any unmanaged type; the destination must hold at least <see cref="BytesLength"/> bytes.</typeparam>
        /// <param name="destination">The target span.</param>
        /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
        public void CopyTo<T>(UnmanagedSpan<T> destination) where T : unmanaged
        {
            long sourceBytes = Bytes.Count;
            long destinationBytes = destination.Length * sizeof(T);
            if ((ulong)sourceBytes > (ulong)destinationBytes)
                throw new ArgumentException("Destination was too short.");
            Buffer.MemoryCopy(Bytes.Address, System.Runtime.CompilerServices.Unsafe.AsPointer(ref destination.GetPinnableReference()), destinationBytes, sourceBytes);
        }

        /// <summary>A reference to the first raw unit.</summary>
        /// <typeparam name="T"><see cref="byte"/>, or <see cref="uint"/> for <c>U</c>.</typeparam>
        /// <returns>The reference.</returns>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        public ref T GetPinnableReference<T>() where T : unmanaged
        {
            CheckUnitType<T>();
            return ref *(T*)Bytes.Address;
        }

        /// <summary>Rejects every unit type but <c>byte</c> (and <c>uint</c> for a <c>U</c> buffer).</summary>
        /// <typeparam name="T">The requested unit type.</typeparam>
        /// <exception cref="NotSupportedException">Any other <typeparamref name="T"/>.</exception>
        private void CheckUnitType<T>() where T : unmanaged
        {
            if (typeof(T) != typeof(byte) && !(IsUnicode && typeof(T) == typeof(uint)))
                throw new NotSupportedException($"A {Descr.ToString(true)} buffer exposes only raw byte (or uint for U) units, not {typeof(T).Name}.");
        }

        /// <summary>
        ///     The elements as a managed array: <c>string[]</c> for <c>U</c>, <c>byte[][]</c> for <c>S</c> (each trailing-NUL
        ///     stripped).
        /// </summary>
        /// <returns>The values in buffer order.</returns>
        /// <exception cref="SystemError">A <c>U</c> unit is above <c>U+10FFFF</c>.</exception>
        public Array ToArray()
        {
            if (IsUnicode)
            {
                var result = new string[Count];
                for (long i = 0; i < Count; i++)
                    result[i] = NDStringCodec.ReadUcs4((uint*)(Bytes.Address + i * ItemSize), Chars);
                return result;
            }
            var bytes = new byte[Count][];
            for (long i = 0; i < Count; i++)
                bytes[i] = NDStringCodec.ReadBytes(Bytes.Address + i * ItemSize, Chars);
            return bytes;
        }

        // ---------------------------------------------------------------------------------------------
        // Lifetime — forwarded to the byte block
        // ---------------------------------------------------------------------------------------------

        /// <inheritdoc/>
        public void DangerousFree() => Bytes.DangerousFree();

        /// <inheritdoc/>
        public bool TryAddRef() => Bytes.TryAddRef();

        /// <inheritdoc/>
        public void Release() => Bytes.Release();

        /// <inheritdoc/>
        public void Abandon() => Bytes.Abandon();

        /// <inheritdoc/>
        public bool IsReleased => Bytes.IsReleased;

        /// <inheritdoc/>
        public bool IsUniquelyReferenced => Bytes.IsUniquelyReferenced;

        /// <summary>Enumerates the elements as C# values (see <see cref="GetIndex(long)"/>).</summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator GetEnumerator()
        {
            for (long i = 0; i < Count; i++)
                yield return GetIndex(i);
        }

        /// <inheritdoc/>
        public override string ToString() => $"StringArraySlice({Descr.ToString(true)}, Count={Count})";
    }
}
