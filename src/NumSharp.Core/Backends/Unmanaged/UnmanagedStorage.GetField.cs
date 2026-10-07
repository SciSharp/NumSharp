using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region GetField

        /// <summary>
        ///     Builds a byte-reinterpreting VIEW of one field of each element — the storage half of
        ///     NumPy's <c>ndarray.getfield(dtype, offset)</c> (<c>numpy/_core/src/multiarray/methods.c</c>,
        ///     <c>PyArray_GetField</c>). Unlike <see cref="AliasAs{T}"/> (<c>view(dtype)</c>, which rescales
        ///     the last axis), this keeps EVERY byte-stride and dimension verbatim and simply reads
        ///     <typeparamref name="T"/> out of the <paramref name="offset"/>-th byte of each element.
        /// </summary>
        /// <typeparam name="T">The field dtype to read; its itemsize must be ≤ this storage's itemsize.</typeparam>
        /// <param name="offset">Byte offset of the field within each element; <c>[0, oldItemsize - newItemsize]</c>.</param>
        /// <returns>
        ///     A new <see cref="UnmanagedStorage"/> of dtype <typeparamref name="T"/> that SHARES memory with
        ///     this storage (writes through, unless the source is read-only). The base is rooted through
        ///     <c>_baseStorage</c> so the shared buffer outlives the view.
        /// </returns>
        /// <remarks>
        ///     <para>
        ///     <b>Byte-strides are preserved.</b> NumPy carries strides in bytes; the getfield view keeps
        ///     them unchanged, so a C-contiguous <c>int32[4]</c> field-viewed as <c>int16</c> stays strided
        ///     (byte-stride 4, itemsize 2 → non-contiguous), reading the low or high half of each int32.
        ///     NumSharp stores strides in ELEMENTS, so each is recomputed as <c>oldByteStride / newItemsize</c>
        ///     — always exact, because every NumSharp itemsize (1/2/4/8/16) divides every larger one, so an
        ///     old byte-stride (a whole multiple of the old itemsize) is always a whole multiple of the new.
        ///     </para>
        ///     <para>
        ///     <b>Sub-element byte offset.</b> When <paramref name="offset"/> is not a multiple of the new
        ///     itemsize (e.g. <c>int32.getfield(int16, 1)</c>), the logical element start falls mid-element,
        ///     which the element-granular <see cref="Shape.offset"/> cannot express. The remainder
        ///     <c>r = offset % newItemsize</c> is absorbed into the WRAP POINTER (shifted by <c>r</c> bytes)
        ///     while <see cref="Shape.offset"/> carries the aligned part — so EVERY offset in range is
        ///     representable, matching NumPy's raw byte pointer.
        ///     </para>
        ///     <para>
        ///     Contiguity flags are recomputed from the resulting strides (NumPy does the same — a same-size
        ///     field of a C-contiguous array stays C-contiguous, a narrower one does not). A read-only source
        ///     (broadcast view, <c>'r'</c> memmap) yields a read-only field view.
        ///     </para>
        /// </remarks>
        /// <exception cref="ValueError">
        ///     <c>new type is larger than original type</c> (newItemsize &gt; oldItemsize),
        ///     <c>offset is negative</c>, or <c>new type plus offset is larger than original type</c>
        ///     (offset &gt; oldItemsize − newItemsize) — the three verbatim NumPy <c>ValueError</c> texts,
        ///     in NumPy's check order.
        /// </exception>
        public unsafe UnmanagedStorage GetFieldAlias<T>(int offset) where T : unmanaged
        {
            // A fixed-width string's item size (S3, U5) need not be a multiple of the field's, so the stride arithmetic
            // below (exact only for the numeric sizes 1/2/4/8/16) would silently truncate: the descriptor route checks it.
            if (_typecode == NPTypeCode.String)
                return GetFieldAliasDescr(global::NumSharp.DType.From(InfoOf<T>.NPTypeCode), offset);

            int oldSize = DTypeSize;
            int newSize = sizeof(T);

            // NumPy's PyArray_GetField validation, verbatim texts and order.
            if (newSize > oldSize)
                throw new ValueError("new type is larger than original type");
            if (offset < 0)
                throw new ValueError("offset is negative");
            if (offset > oldSize - newSize)
                throw new ValueError("new type plus offset is larger than original type");

            var dims = _shape.dimensions;
            int ndim = dims.Length;

            // Empty array: no bytes to alias. Keep the dims, change only the dtype (validation already
            // ran, since NumPy rejects an oversized field even on an empty array). A fresh C-order empty
            // storage of the new dtype is the observable result (shape preserved, dtype changed).
            if (_shape.size == 0 || InternalArray == null)
            {
                var emptyShape = ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])dims.Clone(), 'C');
                return new UnmanagedStorage(ArraySlice.Allocate(InfoOf<T>.NPTypeCode, emptyShape.size, false), emptyShape) { Engine = Engine };
            }

            // Element strides (old dtype) -> new-dtype element strides, keeping the BYTE stride constant.
            var oldStrides = _shape.strides;
            var newStrides = new long[ndim];
            for (int i = 0; i < ndim; i++)
                newStrides[i] = oldStrides[i] * oldSize / newSize; // == oldByteStride / newSize, always exact

            // Byte position of logical element [0,…,0] from the buffer base. _shape.offset*oldSize is a whole
            // multiple of newSize (newSize | oldSize), so the sub-element remainder is exactly offset % newSize.
            long byteStart = _shape.offset * oldSize + offset;
            long r = byteStart % newSize;            // 0 when the offset is aligned to the new itemsize
            long newOffset = (byteStart - r) / newSize;

            // Shift the wrap base by the sub-element remainder; Shape.offset carries the whole-element part.
            // r < newSize ≤ oldSize ≤ BytesLength, so the span stays positive.
            byte* wrapBase = (byte*)InternalArray.Address + r;
            long newBufferCount = (InternalArray.BytesLength - r) / newSize;

            var newShape = new Shape((long[])dims.Clone(), newStrides, newOffset, newBufferCount);

            // A field of a read-only array (broadcast / 'r' memmap) stays read-only. The freshly-built shape
            // may have defaulted WRITEABLE back on.
            if (!_shape.IsWriteable && newShape.IsWriteable)
                newShape = newShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            // The field alias counts its references on THIS storage's block (WrapShared): disposing the owner while the
            // field view lives does not free the buffer under it.
            var newSlice = ArraySlice.WrapShared<T>(wrapBase, newBufferCount, InternalArray);
            var r2 = new UnmanagedStorage();
            r2._shape = newShape;
            r2._typecode = InfoOf<T>.NPTypeCode;
            r2._dtype = typeof(T);
            r2.SetInternalArray(newSlice);
            r2.Count = newShape.size;
            r2._baseStorage = _baseStorage ?? this;
            r2.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
            r2.Engine = Engine;
            return r2;
        }

        /// <summary>
        ///     Dtype-code dispatcher for <see cref="GetFieldAlias{T}"/> — mirrors <see cref="AliasAs(NPTypeCode)"/>.
        /// </summary>
        /// <param name="typeCode">The field dtype to read.</param>
        /// <param name="offset">Byte offset of the field within each element.</param>
        public unsafe UnmanagedStorage GetFieldAlias(NPTypeCode typeCode, int offset)
        {
            switch (typeCode)
            {
                case NPTypeCode.Boolean: return GetFieldAlias<bool>(offset);
                case NPTypeCode.Byte: return GetFieldAlias<byte>(offset);
                case NPTypeCode.SByte: return GetFieldAlias<sbyte>(offset);
                case NPTypeCode.Int16: return GetFieldAlias<short>(offset);
                case NPTypeCode.UInt16: return GetFieldAlias<ushort>(offset);
                case NPTypeCode.Int32: return GetFieldAlias<int>(offset);
                case NPTypeCode.UInt32: return GetFieldAlias<uint>(offset);
                case NPTypeCode.Int64: return GetFieldAlias<long>(offset);
                case NPTypeCode.UInt64: return GetFieldAlias<ulong>(offset);
                case NPTypeCode.Char: return GetFieldAlias<char>(offset);
                case NPTypeCode.Half: return GetFieldAlias<Half>(offset);
                case NPTypeCode.Single: return GetFieldAlias<float>(offset);
                case NPTypeCode.Double: return GetFieldAlias<double>(offset);
                case NPTypeCode.Decimal: return GetFieldAlias<decimal>(offset);
                case NPTypeCode.Complex: return GetFieldAlias<System.Numerics.Complex>(offset);
                default:
                    throw new NotSupportedException($"Type code {typeCode} is not supported.");
            }
        }

        /// <summary>
        ///     <c>ndarray.getfield(dtype, offset)</c> when this array or the field is a fixed-width string (<c>S</c>/<c>U</c>)
        ///     — the same byte-reinterpreting VIEW as <see cref="GetFieldAlias{T}"/> (every dimension and BYTE stride kept,
        ///     the field read <paramref name="offset"/> bytes into each element), for item sizes that need not divide each
        ///     other: <c>S3 → S1</c>, <c>U2 → S4</c>, <c>int32 → S2</c>, <c>S3 → uint8</c>.
        /// </summary>
        /// <param name="descr">The field descriptor: a sized native string dtype, or any numeric dtype.</param>
        /// <param name="offset">Byte offset of the field within each element; <c>[0, oldItemsize - newItemsize]</c>.</param>
        /// <returns>A storage of the field dtype that SHARES this storage's memory (its references counted on this
        ///     storage's block) and does not own it; read-only when this storage is.</returns>
        /// <exception cref="ValueError">NumPy's three <c>PyArray_GetField</c> texts, in its order: <c>new type is larger
        ///     than original type</c>, <c>offset is negative</c>, <c>new type plus offset is larger than original
        ///     type</c>.</exception>
        /// <exception cref="NotSupportedException">The field is an UNSIZED string (NumPy builds a zero-width <c>S0</c> view
        ///     here, which a NumSharp string buffer cannot hold); the field descriptor is non-native (NumSharp has no
        ///     big-endian storage to view); or a byte stride of an axis longer than 1 is not a whole number of field
        ///     items (<c>S3 → S2</c>, <c>S3 → int16</c>: NumPy keeps byte strides, NumSharp stores them in whole
        ///     elements).</exception>
        /// <remarks>
        ///     The element start may fall inside a field item (<c>int32 → S2</c> at offset 1): as in
        ///     <see cref="GetFieldAlias{T}"/>, the remainder is absorbed into the wrap pointer. Every byte stride being a
        ///     whole number of field items makes every element address congruent to that remainder, so the wrapped slice
        ///     addresses each field exactly and never reads outside this storage's buffer (a field ends no later than the
        ///     element holding it).
        /// </remarks>
        internal unsafe UnmanagedStorage GetFieldAliasDescr(DType descr, int offset)
        {
            int oldSize = DTypeSize;
            int newSize = descr.itemsize;

            // NumPy's PyArray_GetField validation, verbatim texts and order.
            if (newSize > oldSize)
                throw new ValueError("new type is larger than original type");
            if (offset < 0)
                throw new ValueError("offset is negative");
            if (offset > oldSize - newSize)
                throw new ValueError("new type plus offset is larger than original type");

            bool stringField = descr.Meta is StringDTypeMeta;
            if (stringField && newSize == 0)
                throw new NotSupportedException(
                    $"getfield with the unsized string dtype {descr.ToString(true)}: NumPy builds a zero-width (itemsize 0) " +
                    "view here, which NumSharp's fixed-width string buffers cannot hold. Give a width: 'S1', '<U2'.");
            if (!descr.isnative)
                throw new NotSupportedException(
                    $"getfield with the non-native dtype {descr.ToString(true)}: NumPy reads the bytes in that byte order on " +
                    "every access; NumSharp stores every array in host order and has no big-endian view.");

            NPTypeCode fieldCode = descr.GetTypeCode();
            var dims = _shape.dimensions;
            int ndim = dims.Length;

            // Empty array: no bytes to alias - a fresh empty storage of the field dtype with the same dims (the numeric
            // path's answer; NumPy's empty view addresses nothing either).
            if (_shape.size == 0 || InternalArray == null)
            {
                var emptyShape = ndim == 0 ? Shape.NewScalar() : Shape.Adopt((long[])dims.Clone(), 'C');
                IArraySlice emptySlice = stringField
                    ? StringArraySlice.Allocate(descr, emptyShape.size, false)
                    : ArraySlice.Allocate(fieldCode, emptyShape.size, false);
                return new UnmanagedStorage(emptySlice, emptyShape) { Engine = Engine };
            }

            // Element strides (old dtype) -> field strides with the BYTE stride kept. An axis longer than 1 whose byte
            // stride is not a whole number of field items cannot be expressed in NumSharp's element strides; a length-1
            // axis is only read at index 0, so its stride is free (the truncated quotient is as good as any).
            var oldStrides = _shape.strides;
            var newStrides = new long[ndim];
            for (int i = 0; i < ndim; i++)
            {
                long byteStride = oldStrides[i] * oldSize;
                if (dims[i] > 1 && byteStride % newSize != 0)
                    throw new NotSupportedException(
                        $"getfield {descr.ToString(true)} of a {Descr.ToString(true)} array: the byte stride {byteStride} of axis " +
                        $"{i} is not a multiple of the field size {newSize}. NumSharp stores strides in whole elements, so it " +
                        "cannot lay out this view (NumPy keeps byte strides).");
                newStrides[i] = byteStride / newSize;
            }

            // Byte position of element [0,...,0] from the buffer base; its remainder modulo the field size goes into the
            // wrap pointer, the whole items into Shape.offset.
            long byteStart = _shape.offset * oldSize + offset;
            long r = byteStart % newSize;
            long newOffset = (byteStart - r) / newSize;
            byte* wrapBase = (byte*)InternalArray.Address + r;
            long newBufferCount = (InternalArray.BytesLength - r) / newSize;

            var newShape = new Shape((long[])dims.Clone(), newStrides, newOffset, newBufferCount);
            if (!_shape.IsWriteable && newShape.IsWriteable)
                newShape = newShape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);

            // The alias counts its references on THIS storage's block: disposing the owner while the field view lives
            // does not free the buffer under it. A string field is the span's bytes read as items of its descriptor.
            IArraySlice newSlice = stringField
                ? new StringArraySlice(ArraySlice.WrapShared<byte>(wrapBase, newBufferCount * newSize, InternalArray), descr)
                : ArraySlice.WrapShared(fieldCode, wrapBase, newBufferCount, InternalArray);
            var r2 = new UnmanagedStorage();
            r2._shape = newShape;
            r2._typecode = fieldCode;
            r2._dtype = fieldCode.AsType();
            r2.SetInternalArray(newSlice);
            r2.Count = newShape.size;
            r2._baseStorage = _baseStorage ?? this;
            r2.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL implies !OWNDATA)
            r2.Engine = Engine;
            return r2;
        }

        #endregion
    }
}
