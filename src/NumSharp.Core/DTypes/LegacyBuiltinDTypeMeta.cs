using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp
{
    /// <summary>
    ///     The DType class of a storage-backed NumSharp dtype — the 15 element types (plus the vestigial
    ///     <see cref="NPTypeCode.String"/>), wrapped the way NumPy's <c>dtypemeta_wrap_legacy_descriptor</c> wraps its
    ///     builtin descriptors into <c>numpy.dtypes.*DType</c> classes. Data-driven: the registry constructs one per
    ///     <see cref="NPTypeCode"/> with its NumPy type number, kind, char, size, alignment and aliases; the behaviour
    ///     slots are the builtin ones — the singleton is the default descriptor, and <see cref="common_dtype"/> is NumPy's
    ///     <c>default_builtin_common_dtype</c> over NumSharp's frozen promotion table
    ///     (<see cref="np._nptypemap_arr_arr"/>), so every existing <c>promote_types</c> answer is reproduced bit-for-bit.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Lane-sharing classes.</b> NumPy keeps a class per C type, and several C types can have one width: Windows'
    ///     <c>int</c> and <c>long</c> are both 32-bit (<c>IntDType</c> beside <c>Int32DType</c>), Linux's <c>long</c> and
    ///     <c>long long</c> both 64-bit (<c>LongLongDType</c> beside <c>Int64DType</c>), and NumSharp's <c>long double</c> is
    ///     a 64-bit double on every platform (<c>LongDoubleDType</c> beside <c>Float64DType</c>, <c>CLongDoubleDType</c>
    ///     beside <c>Complex128DType</c>). Such a class stores its elements in the storage lane of its width (the kernels
    ///     are the lane's) but does not OWN the lane: <see cref="DTypeRegistry.FromTypeCode"/> and
    ///     <see cref="DTypeRegistry.FromScalarType"/> keep returning the lane's own class, and an array keeps the
    ///     lane-sharing descriptor only where it is carried explicitly (creation with that dtype, views, copies, casts).
    ///     </para>
    ///     <para>
    ///     NumPy calls two such classes EQUIVALENT (<c>np.dtype('g') == np.dtype('d')</c> on Windows, a cast between them
    ///     is a no-op), and so does NumSharp: <see cref="DType.Equals(DType)"/>, <see cref="BuiltinCastingImpl"/> and the
    ///     promotion table treat every two NumPy classes of one kind and size that way.
    ///     </para>
    /// </remarks>
    public sealed class LegacyBuiltinDTypeMeta : DTypeMeta
    {
        /// <summary>Builds one builtin class; only <see cref="DTypeRegistry"/> calls this, once per class.</summary>
        /// <param name="name">NumPy's class name (<c>"Int32DType"</c>).</param>
        /// <param name="modulePath"><c>"numpy.dtypes"</c>, or <c>"numsharp.dtypes"</c> for the NumSharp-only classes.</param>
        /// <param name="typeNum">The class's <c>NPY_TYPES</c> number.</param>
        /// <param name="typeCode">The storage discriminator.</param>
        /// <param name="scalarType">The C# element type.</param>
        /// <param name="scalarName">NumPy's scalar type name (<c>"int32"</c>, <c>"intc"</c>, <c>"longdouble"</c>).</param>
        /// <param name="kind">NumPy's kind character.</param>
        /// <param name="typeChar">NumPy's type character.</param>
        /// <param name="itemSize">Element size in bytes.</param>
        /// <param name="alignment">Required alignment in bytes.</param>
        /// <param name="flags">The <c>NPY_DT_*</c> flags.</param>
        /// <param name="aliases">The alias class names NumPy exports for this class.</param>
        /// <param name="ownsLane">
        ///     True (the default) for the class that OWNS its storage lane (the registry maps the lane's
        ///     <see cref="NPTypeCode"/> and C# element type back to it); false for a lane-sharing C-type class (see the
        ///     remarks), which is reached by name, type character and type number only.
        /// </param>
        internal LegacyBuiltinDTypeMeta(string name, string modulePath, int typeNum, NPTypeCode typeCode, Type scalarType,
            string scalarName, char kind, char typeChar, int itemSize, int alignment, DTypeFlags flags, IReadOnlyList<string> aliases,
            bool ownsLane = true)
            : base(name, modulePath, typeNum, scalarType, scalarName, typeCode, kind, typeChar, itemSize, alignment, flags, aliases)
        {
            _ownsLane = ownsLane;
            // Decided once: DType.Equals / GetHashCode read it on every comparison of two descriptors.
            _isNumPyNumeric = modulePath == "numpy.dtypes" && (kind == 'b' || kind == 'i' || kind == 'u' || kind == 'f' || kind == 'c');
        }

        /// <summary>Whether this class owns its storage lane (see the constructor's <c>ownsLane</c>).</summary>
        private readonly bool _ownsLane;

        /// <summary>The cached answer of <see cref="IsNumPyNumeric"/> (the module and kind never change).</summary>
        private readonly bool _isNumPyNumeric;

        /// <summary>
        ///     The lane owner maps its <see cref="NPTypeCode"/> back to itself; a lane-sharing C-type class does not, so
        ///     <c>(DType)NPTypeCode.Double</c> stays <c>float64</c> and an array computed in the lane is labelled with the
        ///     lane's own class.
        /// </summary>
        internal override bool OwnsTypeCode => _ownsLane;

        /// <summary>
        ///     The lane owner claims the C# element type; a lane-sharing class does not, so <c>typeof(double)</c> stays
        ///     <c>float64</c> and <c>typeof(int)</c> stays <c>int32</c>.
        /// </summary>
        internal override bool OwnsScalarType => _ownsLane;

        /// <summary>
        ///     True for a class of NumPy's own legacy numeric set (module <c>numpy.dtypes</c>, kind <c>b</c> / <c>i</c> /
        ///     <c>u</c> / <c>f</c> / <c>c</c>): the classes among which NumPy's <c>add_numeric_cast</c> makes a pair of one
        ///     kind and one size a no-op cast, and <c>dtype.__eq__</c> calls them equal. NumSharp's own decimal and char are
        ///     not in it (char and uint16 stay two classes with two kernels).
        /// </summary>
        internal bool IsNumPyNumeric => _isNumPyNumeric;

        /// <summary>
        ///     Whether a cast from this class to <paramref name="other"/> is NumPy's <c>numeric_copy_or_byteswap</c>: the
        ///     same class, or two NumPy numeric classes of one kind and one item size (<c>intc</c> ↔ <c>int32</c> on Windows,
        ///     <c>longdouble</c> ↔ <c>float64</c>), whose elements have one bit layout.
        /// </summary>
        /// <param name="other">The other class.</param>
        /// <returns>True for an equivalent pair.</returns>
        internal bool IsEquivalentNumeric(LegacyBuiltinDTypeMeta other)
            => ReferenceEquals(this, other)
               || (other != null && IsNumPyNumeric && other.IsNumPyNumeric && Kind == other.Kind && ItemSize == other.ItemSize);

        /// <summary>NumPy's builtin <c>default_descr</c>: the singleton — a non-parametric class has exactly one canonical descriptor.</summary>
        /// <returns><see cref="DTypeMeta.singleton"/>.</returns>
        public override DType default_descr() => singleton;

        /// <summary>
        ///     NumPy's <c>default_builtin_common_dtype</c>: the NEP 50 Python-scalar rules first (a weak float adopts a
        ///     float/complex class, a weak int adopts any numeric class, a weak complex lifts a real float to complex),
        ///     then defer to the class with the larger type number, then the frozen promotion table.
        /// </summary>
        /// <param name="other">The other operand's class.</param>
        /// <returns>The common class, or null (NotImplemented) when the other class must answer.</returns>
        public override DTypeMeta common_dtype(DTypeMeta other) => DTypePromotion.DefaultBuiltinCommonDType(this, other);

        /// <summary>
        ///     The slot <c>is_known_scalar_type</c> for a builtin class: the C# element type, and a C# array of it
        ///     (<c>np.array(int[])</c> is int32). NumPy's legacy slot (<c>python_builtins_are_known_scalar_types</c>)
        ///     accepts Python's own scalars instead; C# has no such shared scalar set, so each class claims its element type.
        /// </summary>
        /// <param name="type">The C# type of the scalar.</param>
        /// <returns>True when <paramref name="type"/> belongs to this class.</returns>
        public override bool is_known_scalar_type(Type type)
        {
            if (type == null)
                return false;
            if (this.type != null && type == this.type)
                return true;
            // C# array types of the scalar map to the same class (np.array(int[]) is int32).
            return type.IsArray && type.GetTypeCode() == TypeCode && TypeCode != NPTypeCode.Empty;
        }

        // ---------------------------------------------------------------------------------------------
        // Element slots (np-layer dtype plan, P3's batch 1): NumPy's arraytypes.c.src functions of the numeric C types,
        // from NumSharp's PyArray_Pack port for setitem / getitem.
        // ---------------------------------------------------------------------------------------------

        /// <summary>The byte-order unit of an element: half the item for complex (real and imaginary swap apart), else the item.</summary>
        private int SwapUnit => TypeCode == NPTypeCode.Complex ? ItemSize / 2 : ItemSize;

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>@TYPE@_setitem</c>: NumSharp's <c>PyArray_Pack</c> port (<see cref="ReductionInitial.PackStore"/>:
        ///     a Python int range-checked with NumPy's texts, a float, a complex, a str parsed, None as NaN or False, a 0-d
        ///     array cast), then the swap of a non-native descriptor.
        /// </remarks>
        /// <exception cref="OverflowException">An integer that does not fit this class, a float infinity into an integer
        ///     class, a Python int past the float range (NumPy's <c>OverflowError</c> texts).</exception>
        /// <exception cref="ValueError">A float NaN into an integer class, an unparsable string, an array with dimensions
        ///     or a sequence.</exception>
        /// <exception cref="TypeError">None into an integer or decimal class, a complex into a real class, or a value NumPy
        ///     cannot convert.</exception>
        internal override unsafe void setitem(DType descr, object value, byte* item, IArraySlice buffer)
        {
            // The pack writes a whole 16-byte slot (it clears the slot first), wider than most elements: pack on the stack
            // and copy the element's own bytes, swapped first for a non-native descriptor.
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            ReductionInitial.PackStore(value, TypeCode, new Span<byte>(slot, ReductionInitial.SlotBytes));
            int size = ItemSize;
            if (!descr.isnative)
                ReverseUnits(slot, size, SwapUnit);
            Buffer.MemoryCopy(slot, item, size, size);
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>@TYPE@_getitem</c>: the element boxed as the class's exact CLR type
        ///     (<see cref="ReductionInitial.Box"/>), read from a swapped copy for a non-native descriptor.</remarks>
        internal override unsafe object getitem(DType descr, byte* item, IArraySlice buffer)
        {
            int size = ItemSize;
            if (descr.isnative)
                return ReductionInitial.Box(new ReadOnlySpan<byte>(item, size), TypeCode);
            // NumPy's getitem copyswaps a byte-swapped element before it boxes it; the element itself is left as it is.
            byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
            Buffer.MemoryCopy(item, slot, size, size);
            ReverseUnits(slot, size, SwapUnit);
            return ReductionInitial.Box(new ReadOnlySpan<byte>(slot, size), TypeCode);
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>@TYPE@_copyswapn</c>: a copy per element (overlap-safe, as NumPy's memmove), then each
        ///     element's units reversed — the whole item, or each half of a complex number.</remarks>
        internal override unsafe void copyswapn(DType descr, byte* dst, long dstride, byte* src, long sstride, long n, bool swap)
        {
            int size = ItemSize;
            if (src != null)
            {
                for (long i = 0; i < n; i++)
                    Buffer.MemoryCopy(src + i * sstride, dst + i * dstride, size, size);
            }
            if (swap && size > 1)
            {
                int unit = SwapUnit;
                for (long i = 0; i < n; i++)
                    ReverseUnits(dst + i * dstride, size, unit);
            }
        }

        /// <inheritdoc/>
        /// <remarks>Every numeric class has NumPy's fill function (bool's raises once called for more than two elements).</remarks>
        internal override bool HasFill => true;

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's <c>@NAME@_fill</c>, operation for operation: an integer class takes <c>delta</c> in its own type and
        ///     computes <c>start + i * delta</c> in C's promotion (a 64-bit product, wrapping, then stored truncated; uint64
        ///     in unsigned 64-bit arithmetic); float32 and float64 compute in their own precision with <c>i</c> converted;
        ///     float16 computes in float32 and rounds each element; complex computes the real and imaginary parts apart.
        ///     <c>decimal</c> (no NumPy class) computes in <see cref="decimal"/>, raising on overflow. The buffer is native
        ///     (np.arange allocates it).
        /// </remarks>
        /// <exception cref="TypeError">The bool class (NumPy's <c>BOOL_fill</c>: <c>arange() is only supported for booleans
        ///     when the result has at most length 2.</c>).</exception>
        /// <exception cref="OverflowException">A <c>decimal</c> progression past the decimal range.</exception>
        internal override unsafe bool fill(DType descr, byte* buffer, long length)
        {
            switch (TypeCode)
            {
                case NPTypeCode.Boolean:
                    throw new TypeError("arange() is only supported for booleans when the result has at most length 2.");
                case NPTypeCode.SByte:
                {
                    var b = (sbyte*)buffer;
                    sbyte start = b[0], delta = unchecked((sbyte)(b[1] - start));
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked((sbyte)(start + i * delta));
                    return true;
                }
                case NPTypeCode.Byte:
                {
                    byte start = buffer[0], delta = unchecked((byte)(buffer[1] - start));
                    for (long i = 2; i < length; i++)
                        buffer[i] = unchecked((byte)(start + i * delta));
                    return true;
                }
                case NPTypeCode.Int16:
                {
                    var b = (short*)buffer;
                    short start = b[0], delta = unchecked((short)(b[1] - start));
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked((short)(start + i * delta));
                    return true;
                }
                case NPTypeCode.UInt16:
                case NPTypeCode.Char: // NumSharp's char is the uint16 lane: NumPy's USHORT_fill
                {
                    var b = (ushort*)buffer;
                    ushort start = b[0], delta = unchecked((ushort)(b[1] - start));
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked((ushort)(start + i * delta));
                    return true;
                }
                case NPTypeCode.Int32:
                {
                    var b = (int*)buffer;
                    int start = b[0], delta = unchecked(b[1] - start);
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked((int)(start + i * delta));
                    return true;
                }
                case NPTypeCode.UInt32:
                {
                    // C promotes npy_uint against the 64-bit npy_intp to a SIGNED 64-bit product (every uint32 fits).
                    var b = (uint*)buffer;
                    uint start = b[0], delta = unchecked(b[1] - start);
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked((uint)(start + i * (long)delta));
                    return true;
                }
                case NPTypeCode.Int64:
                {
                    var b = (long*)buffer;
                    long start = b[0], delta = unchecked(b[1] - start);
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked(start + i * delta);
                    return true;
                }
                case NPTypeCode.UInt64:
                {
                    // C converts npy_intp to npy_ulonglong (the unsigned type of the same rank wins): modular arithmetic.
                    var b = (ulong*)buffer;
                    ulong start = b[0], delta = unchecked(b[1] - start);
                    for (long i = 2; i < length; i++)
                        b[i] = unchecked(start + (ulong)i * delta);
                    return true;
                }
                case NPTypeCode.Half:
                {
                    // HALF_fill: the progression in float32, each element rounded to float16.
                    var b = (Half*)buffer;
                    float start = (float)b[0], delta = (float)b[1];
                    delta -= start;
                    for (long i = 2; i < length; i++)
                        b[i] = (Half)(start + i * delta);
                    return true;
                }
                case NPTypeCode.Single:
                {
                    var b = (float*)buffer;
                    float start = b[0], delta = b[1];
                    delta -= start;
                    for (long i = 2; i < length; i++)
                        b[i] = start + i * delta;
                    return true;
                }
                case NPTypeCode.Double:
                {
                    var b = (double*)buffer;
                    double start = b[0], delta = b[1];
                    delta -= start;
                    for (long i = 2; i < length; i++)
                        b[i] = start + i * delta;
                    return true;
                }
                case NPTypeCode.Complex:
                {
                    // CDOUBLE_fill: the real and imaginary progressions apart.
                    var b = (double*)buffer;
                    double re = b[0], im = b[1];
                    double dre = b[2] - re, dim = b[3] - im;
                    for (long i = 2; i < length; i++)
                    {
                        b[2 * i] = re + i * dre;
                        b[2 * i + 1] = im + i * dim;
                    }
                    return true;
                }
                case NPTypeCode.Decimal:
                {
                    var b = (decimal*)buffer;
                    decimal start = b[0], delta = b[1] - start;
                    for (long i = 2; i < length; i++)
                        b[i] = start + i * delta;
                    return true;
                }
                default:
                    return false;
            }
        }

        /// <inheritdoc/>
        /// <remarks>NumPy's <c>@fname@_nonzero</c>: an element different from zero (a NaN is), a float16 by its bits
        ///     (<c>!npy_half_iszero</c>), a complex number when either part is; a float read from a swapped copy for a
        ///     non-native descriptor (a signed zero's sign byte moves with the order), an integer as it is (a swap cannot
        ///     make a zero nonzero).</remarks>
        internal override unsafe bool nonzero(DType descr, byte* item)
        {
            byte* p = item;
            if (!descr.isnative && TypeCode is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex)
            {
                byte* slot = stackalloc byte[ReductionInitial.SlotBytes];
                Buffer.MemoryCopy(item, slot, ItemSize, ItemSize);
                ReverseUnits(slot, ItemSize, SwapUnit);
                p = slot;
            }
            switch (TypeCode)
            {
                case NPTypeCode.Half: return (*(ushort*)p & 0x7fff) != 0;
                case NPTypeCode.Single: return *(float*)p != 0f;
                case NPTypeCode.Double: return *(double*)p != 0d;
                case NPTypeCode.Complex: return ((double*)p)[0] != 0d || ((double*)p)[1] != 0d;
                case NPTypeCode.Decimal: return *(decimal*)p != 0m;
                default:
                    // Booleans and every integer: any nonzero byte (the bytes of zero are all zero in either order).
                    for (int i = 0; i < ItemSize; i++)
                        if (p[i] != 0)
                            return true;
                    return false;
            }
        }
    }
}
