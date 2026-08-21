using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Aliasing

        /// <summary>
        /// Creates an alias (view) of this storage that shares the same underlying memory.
        /// </summary>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage.
        /// The returned storage's <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner:
        /// <list type="bullet">
        ///   <item>If this storage owns its data: <c>alias._baseStorage = this</c></item>
        ///   <item>If this storage is a view: <c>alias._baseStorage = this._baseStorage</c></item>
        /// </list>
        /// This ensures all views in a chain point to the original owner, not intermediate views.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public UnmanagedStorage Alias()
        {
            var r = new UnmanagedStorage();
            r._shape = _shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = _shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            r.Engine = Engine;
            return r;
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different shape.
        /// </summary>
        /// <param name="shape">The shape for the alias. Should be compatible with the storage size (not validated).</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage but has
        /// the specified shape. The returned storage's <see cref="_baseStorage"/> points to
        /// the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Shape Compatibility:</b> This method does NOT validate that the shape is
        /// compatible with the storage size. Use with caution.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public unsafe UnmanagedStorage Alias(Shape shape)
        {
            var r = new UnmanagedStorage();
            r._typecode = _typecode;
            r._dtype = _dtype;
            // Hot path: when this storage is already wired (InternalArray + Address
            // set), copy the IArraySlice surface and the *single* live type-specific
            // field directly instead of routing through SetInternalArray's full
            // 15-case typecode dispatch. The aliased storage exposes the same
            // backing buffer; the type-specific field is still needed for typed
            // accessors elsewhere in UnmanagedStorage, so we mirror parent's slot
            // via an IL-emitted delegate cached per dtype (no switch in hot path).
            if (InternalArray != null)
            {
                r.InternalArray = InternalArray;
                r.Address = Address;
                DirectILKernelGenerator.GetStorageAliasFieldCopier(_typecode)(r, this);
            }

            // A view inherits writeability from what it aliases: any view of a read-only array (a
            // broadcast, or an 'r' memmap) must stay read-only — otherwise a write would reach read-only
            // memory (a hard segfault on memory-mapped read-only pages). The externally-built `shape` may
            // have defaulted WRITEABLE back to true (transpose, expand_dims, newaxis all pass one here).
            if (!_shape.IsWriteable && shape.IsWriteable)
                shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            r._shape = shape;
            r.Count = shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            r.Engine = Engine;
            return r;
        }


        /// <summary>
        /// Creates an alias (view) of this storage with a different shape (by reference).
        /// </summary>
        /// <param name="shape">The shape for the alias. Should be compatible with the storage size (not validated).</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage but has
        /// the specified shape. The returned storage's <see cref="_baseStorage"/> points to
        /// the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Shape Compatibility:</b> This method does NOT validate that the shape is
        /// compatible with the storage size. Use with caution.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public UnmanagedStorage Alias(ref Shape shape)
        {
            var r = new UnmanagedStorage();
            // A view inherits writeability from what it aliases (see Alias(Shape)).
            r._shape = (!_shape.IsWriteable && shape.IsWriteable)
                ? shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE)
                : shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            r.Engine = Engine;
            return r;
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different dtype, reinterpreting bytes —
        /// the storage half of NumPy's <c>ndarray.view(dtype)</c>.
        /// </summary>
        /// <typeparam name="T">The new dtype to interpret the bytes as.</typeparam>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage but
        /// interprets the bytes as a different type. Shape is adjusted if type sizes differ.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Byte Reinterpretation:</b> This does NOT convert values. It reinterprets the raw
        /// bytes as a different type, like NumPy's view(). For example, viewing float64 as int64
        /// will show the IEEE 754 bit patterns, not converted values.
        /// </para>
        /// <para>
        /// <b>Same itemsize:</b> The dtype tag changes and the shape, strides and offset are kept
        /// verbatim, so ANY layout works (C/F-contiguous, sliced, strided, transposed, negative-stride,
        /// broadcast, 0-d). A read-only source (broadcast) stays read-only.
        /// </para>
        /// <para>
        /// <b>Different itemsize (NumPy 2.x rule):</b> only the LAST axis must be contiguous
        /// (byte-stride == old itemsize) — the whole array need NOT be C-contiguous. That last axis is
        /// resized by the size ratio while every outer axis keeps its byte stride, so
        /// <c>a[::2].view(...)</c> and other outer-strided/offset views are supported, matching NumPy.
        /// Read-only (broadcast) sources stay read-only.
        /// </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">
        /// If the last axis is not contiguous and type sizes differ (NumPy raises ValueError). NumSharp
        /// stores strides/offset in ELEMENTS rather than bytes, so the rare misaligned strided+offset
        /// view whose byte offset/stride is not a whole multiple of the new itemsize is also refused here
        /// (NumPy, which carries a byte pointer, would allow it).
        /// </exception>
        /// <exception cref="ArgumentException">
        /// If the array is 0-d, or the last axis's byte size is not divisible by the new itemsize
        /// (NumPy raises ValueError with the same wording).
        /// </exception>
        public unsafe UnmanagedStorage AliasAs<T>() where T : unmanaged
        {
            if (_dtype == typeof(T))
                return Alias();

            int oldSize = DTypeSize;
            int newSize = sizeof(T);

            // Same itemsize: pure reinterpret. Keep dims/strides/offset (any layout — NumPy allows the
            // same-size view on non-contiguous / broadcast / negative-stride arrays), only the dtype tag
            // changes. The wrap spans the WHOLE backing slice so a strided view that drops elements still
            // addresses every in-bounds coordinate via Shape.offset + strides.
            if (oldSize == newSize)
                return WrapReinterpreted<T>(_shape, InternalArray.Count);

            // Different itemsize — NumPy 2.x view(dtype) rules (numpy/_core/src/multiarray/getset.c
            // array_descr_set): only the LAST axis has to be contiguous; every outer axis keeps its byte
            // stride and the last axis is rescaled by the size ratio.
            var dims = _shape.dimensions;
            if (dims.Length == 0)
                throw new ArgumentException("Changing the dtype of a 0d array is only supported if the itemsize is unchanged");

            var oldStrides = _shape.strides; // element strides in the OLD dtype
            int last = dims.Length - 1;
            // The last axis is contiguous iff its byte-stride == old itemsize, i.e. its element stride is
            // 1. A length-≤1 last axis is trivially contiguous (its stride is never stepped).
            bool lastContiguous = dims[last] <= 1 || oldStrides[last] == 1;
            if (!lastContiguous)
                throw new InvalidOperationException("To change to a dtype of a different size, the last axis must be contiguous");

            long lastAxisBytes = dims[last] * oldSize;
            if (lastAxisBytes % newSize != 0)
                throw new ArgumentException("When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.");

            // Reinterpret the outer axes' byte strides and the base offset as new-dtype elements. NumSharp
            // stores these in ELEMENTS (NumPy in bytes), so a value that is not a whole multiple of the new
            // itemsize cannot be represented — refuse it (extremely rare; offset 0 and the common strided
            // cases always divide). This is strictly MORE permissive than before, never less.
            long byteOffset = _shape.offset * oldSize;
            if (byteOffset % newSize != 0)
                throw new InvalidOperationException("To change to a dtype of a different size, the last axis must be contiguous");

            var newDims = new long[dims.Length];
            var newStrides = new long[dims.Length];
            for (int i = 0; i < last; i++)
            {
                newDims[i] = dims[i];
                long outerByteStride = oldStrides[i] * oldSize;
                if (outerByteStride % newSize != 0)
                    throw new InvalidOperationException("To change to a dtype of a different size, the last axis must be contiguous");
                newStrides[i] = outerByteStride / newSize;
            }
            newDims[last] = lastAxisBytes / newSize;
            newStrides[last] = 1;
            long newOffset = byteOffset / newSize;

            // Span the whole backing buffer in new-dtype units (floor: a trailing partial element is never
            // addressed, since each last-axis run is a whole number of new elements).
            long newBufferCount = InternalArray.BytesLength / newSize;
            var newShape = new Shape(newDims, newStrides, newOffset, newBufferCount);
            return WrapReinterpreted<T>(newShape, newBufferCount);
        }

        /// <summary>
        /// Builds a byte-reinterpreting alias: a new storage of dtype <typeparamref name="T"/> over the
        /// SAME memory, carrying <paramref name="shape"/> (dims/strides/offset) and a non-owning wrap of
        /// <paramref name="wrapCount"/> new-dtype elements. A read-only source stays read-only, and the
        /// ultimate owner is rooted through <c>_baseStorage</c> so the shared buffer outlives the view.
        /// </summary>
        private unsafe UnmanagedStorage WrapReinterpreted<T>(Shape shape, long wrapCount) where T : unmanaged
        {
            // A view inherits non-writeability: a view of a read-only array (broadcast, 'r' memmap) must
            // stay read-only. The freshly-built strided/same-size shape may have defaulted WRITEABLE back on.
            if (!_shape.IsWriteable && shape.IsWriteable)
                shape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            var newSlice = ArraySlice.Wrap<T>((T*)InternalArray.Address, wrapCount);
            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = InfoOf<T>.NPTypeCode;
            r._dtype = typeof(T);
            r.SetInternalArray(newSlice);
            r.Count = shape.size;
            r._baseStorage = _baseStorage ?? this;
            r.Engine = Engine;
            return r;
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different dtype, reinterpreting bytes.
        /// </summary>
        /// <param name="dtype">The new dtype to interpret the bytes as.</param>
        /// <returns>A view with reinterpreted bytes.</returns>
        public unsafe UnmanagedStorage AliasAs(Type dtype)
        {
            if (dtype == _dtype)
                return Alias();

            var typeCode = dtype.GetTypeCode();
            return AliasAs(typeCode);
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different dtype, reinterpreting bytes.
        /// </summary>
        /// <param name="typeCode">The new dtype to interpret the bytes as.</param>
        /// <returns>A view with reinterpreted bytes.</returns>
        public unsafe UnmanagedStorage AliasAs(NPTypeCode typeCode)
        {
            if (typeCode == _typecode)
                return Alias();

            // Type switch to call the generic version
            switch (typeCode)
            {
                case NPTypeCode.Boolean: return AliasAs<bool>();
                case NPTypeCode.Byte: return AliasAs<byte>();
                case NPTypeCode.SByte: return AliasAs<sbyte>();
                case NPTypeCode.Int16: return AliasAs<short>();
                case NPTypeCode.UInt16: return AliasAs<ushort>();
                case NPTypeCode.Int32: return AliasAs<int>();
                case NPTypeCode.UInt32: return AliasAs<uint>();
                case NPTypeCode.Int64: return AliasAs<long>();
                case NPTypeCode.UInt64: return AliasAs<ulong>();
                case NPTypeCode.Char: return AliasAs<char>();
                case NPTypeCode.Half: return AliasAs<Half>();
                case NPTypeCode.Single: return AliasAs<float>();
                case NPTypeCode.Double: return AliasAs<double>();
                case NPTypeCode.Decimal: return AliasAs<decimal>();
                case NPTypeCode.Complex: return AliasAs<System.Numerics.Complex>();
                default:
                    throw new NotSupportedException($"Type code {typeCode} is not supported.");
            }
        }

        /// <summary>
        ///     Creates a <see cref="double"/> (float64) VIEW onto one lane — real or imaginary — of a
        ///     <see cref="System.Numerics.Complex"/> storage, reproducing NumPy's <c>a.real</c> / <c>a.imag</c>:
        ///     a strided float64 view that SHARES memory (and writeability) with the Complex base.
        /// </summary>
        /// <param name="imaginary"><c>false</c> selects the real lane, <c>true</c> the imaginary lane.</param>
        /// <returns>
        ///     A float64 <see cref="UnmanagedStorage"/> aliasing this storage's chosen lane. Writes through
        ///     to the Complex base; the base is kept alive via <c>_baseStorage</c>.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     A <see cref="System.Numerics.Complex"/> is two consecutive float64s (real, then imaginary),
        ///     laid out exactly like NumPy's <c>complex128</c>. A lane is therefore a plain strided float64
        ///     view: every Complex element stride is doubled (a Complex spans two float64s) and the base
        ///     offset is doubled, plus one for the imaginary lane. This works for EVERY layout
        ///     (contiguous / F-contiguous / transposed / strided / negative-stride / sliced-offset /
        ///     broadcast), because it operates on the raw contiguous backing buffer and mirrors whatever
        ///     strides/offset the Complex shape carries — the same strided-lane read the FFT driver uses.
        ///     </para>
        ///     <para>
        ///     A lane of a read-only Complex (a broadcast view, or an <c>'r'</c> memmap) stays read-only.
        ///     </para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">If this storage is not a Complex storage.</exception>
        public unsafe UnmanagedStorage AliasComplexLane(bool imaginary)
        {
            if (_typecode != NPTypeCode.Complex)
                throw new InvalidOperationException(
                    $"AliasComplexLane requires a Complex storage but got {_typecode}.");

            // The backing buffer holds `bufferSize` Complex values == 2*bufferSize float64 lanes.
            long doubleCount = _shape.bufferSize * 2;

            // float64 element strides = Complex element strides * 2 (a Complex is two float64s wide).
            long[] cstr = _shape.strides;
            long[] dstr = new long[cstr.Length];
            for (int i = 0; i < cstr.Length; i++)
                dstr[i] = cstr[i] * 2;

            long laneOffset = _shape.offset * 2 + (imaginary ? 1 : 0);
            var laneShape = new Shape((long[])_shape.dimensions.Clone(), dstr, laneOffset, doubleCount);

            // Inherit read-only-ness: a lane of a non-writeable Complex must not become writeable
            // (the internal Shape ctor defaults WRITEABLE on). Same guard as Alias(Shape).
            if (!_shape.IsWriteable && laneShape.IsWriteable)
                laneShape = laneShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            // Non-owning float64 slice over the SAME memory; lifetime handed to the Complex owner.
            var slice = ArraySlice.Wrap<double>((double*)InternalArray.Address, doubleCount);

            var r = new UnmanagedStorage();
            r._shape = laneShape;
            r._typecode = NPTypeCode.Double;
            r._dtype = typeof(double);
            r.SetInternalArray(slice);
            r.Count = laneShape.size; // logical element count (bufferSize may be larger for a sliced/broadcast view)
            r._baseStorage = _baseStorage ?? this;
            r.Engine = Engine;
            return r;
        }

        #endregion

        #region Casting

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast<T>() where T : unmanaged
        {
            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeof(T)) { Engine = Engine };

            if (_dtype == typeof(T))
                return Clone();

            // SIMD copy-with-cast via NDIter (materializes logical element order for strided /
            // F-contiguous views and converts in a single pass). Was: CloneData().CastTo<T>().
            return CastViaIterator(InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(NPTypeCode typeCode)
        {
            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeCode) { Engine = Engine };

            if (_typecode == typeCode)
                return Clone();

            // SIMD copy-with-cast via NDIter (materializes logical element order for strided /
            // F-contiguous views and converts in a single pass). Was: CloneData().CastTo(typeCode).
            return CastViaIterator(typeCode);
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(Type dtype)
        {
            return Cast(dtype.GetTypeCode());
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype only if necessary.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <typeparamref name="T"/></remarks>
        public UnmanagedStorage CastIfNecessary<T>() where T : unmanaged
        {
            if (_dtype == typeof(T))
                return this;

            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeof(T)) { Engine = Engine };

            // SIMD copy-with-cast via NDIter (materializes logical element order for strided /
            // F-contiguous views and converts in a single pass). Was: CloneData().CastTo<T>().
            return CastViaIterator(InfoOf<T>.NPTypeCode);
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype only if necessary
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(NPTypeCode typeCode)
        {
            if (_typecode == typeCode)
                return this;

            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeCode) { Engine = Engine };

            // SIMD copy-with-cast via NDIter (materializes logical element order for strided /
            // F-contiguous views and converts in a single pass). Was: CloneData().CastTo(typeCode).
            return CastViaIterator(typeCode);
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(Type dtype)
        {
            return CastIfNecessary(dtype.GetTypeCode());
        }

        /// <summary>
        ///     SIMD cast of this storage's logical data to <paramref name="typeCode"/> through the
        ///     unified <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/> core, into a
        ///     fresh C-contiguous storage of the same logical dimensions. Replaces the legacy
        ///     per-element <c>CloneData().CastTo</c> scalar loop (same NumPy-faithful values, now
        ///     vectorized and materialized + cast in a single pass). Preserves <see cref="Engine"/>.
        /// </summary>
        private UnmanagedStorage CastViaIterator(NPTypeCode typeCode)
        {
            Shape outShape = _shape.NDim == 0
                ? Shape.NewScalar()
                : new Shape((long[])_shape.dimensions.Clone(), 'C');

            var dst = new UnmanagedStorage(ArraySlice.Allocate(typeCode, outShape.size, false), outShape) { Engine = Engine };
            NDIter.Copy(dst, this);
            return dst;
        }

        /// <summary>
        ///     SIMD cast of a contiguous 1-D <paramref name="value"/> slice to <paramref name="typeCode"/>
        ///     through <see cref="NDIter.Copy(UnmanagedStorage, UnmanagedStorage)"/>. Replaces the legacy
        ///     scalar <c>IMemoryBlock.CastTo</c> loop at the slice level (indexed-assignment cast, typed
        ///     extraction). Returns a fresh owning slice of <paramref name="typeCode"/>.
        /// </summary>
        private static IArraySlice CastSliceViaIterator(IArraySlice value, NPTypeCode typeCode)
        {
            var src = new UnmanagedStorage(value, Shape.Vector(value.Count));
            var dst = new UnmanagedStorage(ArraySlice.Allocate(typeCode, value.Count, false), Shape.Vector(value.Count));
            NDIter.Copy(dst, src);
            return dst.InternalArray;
        }

        #endregion

        #region Cloning

        /// <summary>
        ///     Clone internal storage and return an owning <see cref="IArraySlice"/>
        ///     sized to <c>_shape.size</c> (NOT <c>InternalArray.Count</c>).
        /// </summary>
        /// <returns>
        ///     A freshly-allocated <see cref="IArraySlice"/> whose
        ///     <c>Count == _shape.size</c>. For contiguous shapes the
        ///     elements come from <c>InternalArray[_shape.offset.._shape.offset + _shape.size)</c>
        ///     via <see cref="IArraySlice.Slice(int, int)"/> + Clone; for strided /
        ///     broadcast / transposed shapes they are materialised via <see cref="NDIter.Copy"/>.
        /// </returns>
        /// <remarks>
        ///     Subtle: the C-contig branch must <b>always</b> slice to
        ///     <c>_shape.size</c> when <c>_shape.bufferSize &gt; _shape.size</c>,
        ///     even when <c>offset == 0</c>. A 1-D slice like
        ///     <c>arr[0:4]</c> on a 5-element source has offset 0 yet covers
        ///     only the first 4 elements; a previous version unconditionally
        ///     cloned the entire <c>InternalArray</c> in the <c>offset == 0</c>
        ///     branch, then handed the 5-element clone to <see cref="UnmanagedStorage(IArraySlice, Shape)"/>
        ///     paired with a (4,) shape, tripping
        ///     <see cref="IncorrectShapeException"/>.
        /// </remarks>
        public IArraySlice CloneData()
        {
            // An empty array (any dim == 0) has no elements to copy — return a fresh zero-length
            // buffer of this dtype. This MUST precede the contiguous branch: an empty slice keeps
            // its parent's offset while its own backing IArraySlice.Count has already collapsed to
            // 0, so InternalArray.Slice(offset, 0) below would trip start > Count == 0 and throw
            // (e.g. arr["1:,1:,1:1"].flatten()). NumPy's flatten / ravel / copy of an empty array
            // all yield an empty array.
            if (_shape.size == 0)
                return ArraySlice.Allocate(_typecode, 0L, false);

            // Contiguous shapes can copy directly from memory.
            // Must account for offset AND the size-vs-buffer mismatch — slice
            // to exactly _shape.size starting at _shape.offset so the cloned
            // IArraySlice matches the shape we'll pair it with downstream.
            if (_shape.IsContiguous)
            {
                if (_shape.offset == 0 && _shape.size == InternalArray.Count)
                    return InternalArray.Clone();
                return InternalArray.Slice(_shape.offset, _shape.size).Clone();
            }

            if (_shape.IsScalar)
                return ArraySlice.Scalar(GetValue(0), _typecode);

            // Scalar-broadcast (all strides 0): every element is the SAME single source value, so
            // materialize with a fast typed fill (1-byte -> InitBlock/memset, wider -> SIMD fill via
            // UnmanagedMemoryBlock<T>.Fill) instead of the general per-element NDIter.Copy walk.
            // Proven 6-8x for the same-type broadcast clone (bcast u8->u8 4M: 0.83->5.69x).
            // Bit-identical (same value in every slot).
            if (_shape.IsScalarBroadcast)
                return ArraySlice.Allocate(InternalArray.TypeCode, _shape.size, GetValue(0));

            //Linear copy of all the sliced items (non-contiguous: broadcast, stepped, transposed).
            var ret = ArraySlice.Allocate(InternalArray.TypeCode, _shape.size, false);
            var dst = new UnmanagedStorage(ret, _shape.Clean());
            NDIter.Copy(dst, this);

            return ret;
        }

        /// <summary>
        ///     Get all elements from cloned storage as <see cref="ArraySlice{T}"/> and cast if necessary.
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage and casted (if necessary) as <see cref="ArraySlice{T}"/></returns>
        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            if (_typecode == InfoOf<T>.NPTypeCode)
                return (ArraySlice<T>)CloneData();

            // SIMD materialize-and-cast in a single NDIter pass. Was: CloneData() (materialize)
            // followed by the scalar CastTo<T> loop — two passes over the data.
            return (ArraySlice<T>)CastViaIterator(InfoOf<T>.NPTypeCode).InternalArray;
        }

        /// <summary>
        ///     Perform a complete copy of this <see cref="UnmanagedStorage"/> and <see cref="InternalArray"/>.
        /// </summary>
        /// <remarks>If shape is sliced, discards any slicing properties but copies only the sliced data</remarks>
        public UnmanagedStorage Clone()
        {
            if (InternalArray == null)
                return new UnmanagedStorage(_typecode) { Engine = Engine };

            if (CanCloneRawLayout())
                return new UnmanagedStorage(InternalArray.Clone(), new Shape(_shape)) { Engine = Engine };

            return new UnmanagedStorage(CloneData(), _shape.Clone(true, true, true)) { Engine = Engine };
        }

        private bool CanCloneRawLayout()
        {
            if (_shape.IsEmpty || _shape.IsBroadcasted || _shape.offset != 0)
                return false;

            if (_shape.bufferSize > 0 && _shape.bufferSize != _shape.size)
                return false;

            return _shape.IsContiguous || _shape.IsFContiguous;
        }

        object ICloneable.Clone() => Clone();

        #endregion
    }
}
