/*
 * NumSharp
 * Copyright (C) 2018 Haiping Chen
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the Apache License 2.0 as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the Apache License 2.0
 * along with this program.  If not, see <http://www.apache.org/licenses/LICENSE-2.0/>.
 */

using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

// ReSharper disable once CheckNamespace
namespace NumSharp.Generic
{
    public partial class NDArray<TDType> : NDArray where TDType : unmanaged
    {
        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage) : base(storage)
        {
            if (storage.DType != typeof(TDType))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(TDType).Name}", nameof(storage));
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage, Shape shape) : base(storage, shape)
        {
            if (storage.DType != typeof(TDType))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(TDType).Name}", nameof(storage));
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage, ref Shape shape) : base(storage, ref shape)
        {
            if (storage.DType != typeof(TDType))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(TDType).Name}", nameof(storage));
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        protected internal NDArray(TensorEngine engine) : base(InfoOf<TDType>.NPTypeCode, engine) { }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray() : base(InfoOf<TDType>.NPTypeCode) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(int size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros)
        { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(long size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros)
        { }

        /// <summary>
        ///     Wraps a .NET array of <typeparamref name="TDType"/> as a typed array — NumPy's
        ///     <c>np.ndarray(shape, buffer=values, order=order)</c>, as <see cref="NumSharp.NDArray(System.Array, NumSharp.Shape, char)"/> describes:
        ///     <paramref name="shape"/> contributes its DIMENSIONS only, and the values are laid out densely in
        ///     <paramref name="order"/>.
        /// </summary>
        /// <param name="values">The elements; a one-dimensional array is wrapped without a copy, a multi-dimensional or
        ///     jagged one is flattened row by row first. Its element type must be <typeparamref name="TDType"/>.</param>
        /// <param name="shape">The dimensions of the result (a view's shape gives the same array as a fresh shape of the
        ///     same dimensions); <c>default</c> uses the dimensions of <paramref name="values"/>.</param>
        /// <param name="order"><c>'C'</c> (default) row-major, <c>'F'</c> column-major; <c>'A'</c> and <c>'K'</c> mean
        ///     <c>'C'</c>; lowercase is accepted.</param>
        /// <exception cref="NullReferenceException"><paramref name="values"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not 'C', 'F', 'A' or 'K' in either case.</exception>
        /// <exception cref="ArgumentException">The element count of <paramref name="values"/> is not the size of
        ///     <paramref name="shape"/>, or the element type of <paramref name="values"/> is not <typeparamref name="TDType"/>
        ///     (checked after the base constructor has wrapped the values).</exception>
        public NDArray(Array values, Shape shape = default, char order = 'C') : base(values, shape, order)
        {
            if (dtype != typeof(TDType))
                throw new ArgumentException($"Array type must be the same as T. {dtype.type.Name} != {typeof(TDType).Name}", nameof(values));
        }

        /// <summary>
        ///     Wraps an <see cref="IArraySlice"/> of <typeparamref name="TDType"/> as a typed array — NumPy's
        ///     <c>np.ndarray(shape, buffer=values, order=order)</c>, as <see cref="NumSharp.NDArray(NumSharp.Backends.Unmanaged.IArraySlice, NumSharp.Shape, char)"/>
        ///     describes: <paramref name="shape"/> contributes its DIMENSIONS only, and the slice is laid out densely in
        ///     <paramref name="order"/>.
        /// </summary>
        /// <param name="values">The buffer, wrapped without a copy. Its element type must be <typeparamref name="TDType"/>.</param>
        /// <param name="shape">The dimensions of the result; <c>default</c> makes a one-dimensional array of
        ///     <c>values.Count</c> elements.</param>
        /// <param name="order"><c>'C'</c> (default) row-major, <c>'F'</c> column-major; <c>'A'</c> and <c>'K'</c> mean
        ///     <c>'C'</c>; lowercase is accepted.</param>
        /// <exception cref="NullReferenceException"><paramref name="values"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="order"/> is not 'C', 'F', 'A' or 'K' in either case.</exception>
        /// <exception cref="ArgumentException">The element count of <paramref name="values"/> is not the size of
        ///     <paramref name="shape"/>, or the slice's element type is not <typeparamref name="TDType"/> (checked after the
        ///     base constructor has wrapped the slice).</exception>
        public NDArray(IArraySlice values, Shape shape = default, char order = 'C') : base(values, shape, order)
        {
            var underlying = values.GetType().GenericTypeArguments[0];
            if (underlying != typeof(TDType))
                throw new ArgumentException($"Array type must be the same as T. {underlying.Name} != {typeof(TDType).Name}", nameof(values));
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(Shape shape) : base(InfoOf<TDType>.NPTypeCode, shape) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(int size) : base(InfoOf<TDType>.NPTypeCode, size) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(long size) : base(InfoOf<TDType>.NPTypeCode, size) { }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="NumSharp.Backends.UnmanagedStorage.Allocate(NumSharp.Shape, NumSharp.DType, bool)"/></remarks>
        public NDArray(Shape shape, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, shape, fillZeros) { }

        /// <summary>
        ///     The allocation constructor behind <see cref="NumSharp.TensorEngine.NewArray{T}(Shape, bool)"/>: a new typed
        ///     array bound to <paramref name="engine"/> instead of <see cref="NumSharp.TensorEngine.Default"/>.
        /// </summary>
        /// <param name="shape">The dimensions (a view's layout is laid out again in C order, as for every allocation).</param>
        /// <param name="fillZeros">Zero the buffer; false leaves pool noise the caller must overwrite completely.</param>
        /// <param name="engine">The engine the array binds (the engine computing the result it will hold), or <c>null</c>
        ///     for <see cref="NumSharp.TensorEngine.Default"/> as it is now.</param>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows.</exception>
        /// <remarks>Footgun: <paramref name="fillZeros"/> <c>false</c> hands out whatever bytes the buffer pool held - the
        ///     caller must overwrite every element before anything reads the array.</remarks>
        public NDArray(Shape shape, bool fillZeros, TensorEngine engine) : base(InfoOf<TDType>.NPTypeCode, shape, fillZeros, engine) { }

        /// <summary>
        ///     A new typed array under a GIVEN descriptor of <typeparamref name="TDType"/>'s element type: the constructor for
        ///     a result that must keep a C-type class the type code alone does not name (an <c>np.intc</c> item, char
        ///     <c>'i'</c>, where <see cref="NDArray{TDType}(Shape, bool)"/> picks the type code's default class).
        /// </summary>
        /// <param name="dtype">The descriptor; its storage type must be <typeparamref name="TDType"/>.</param>
        /// <param name="shape">The dimensions (a view's layout is laid out again in C order, as for every allocation).</param>
        /// <param name="fillZeros">Zero the buffer; false leaves pool noise the caller must overwrite completely.</param>
        /// <exception cref="ArgumentException"><paramref name="dtype"/> stores another element type than
        ///     <typeparamref name="TDType"/> (a typed wrapper over it would read the wrong number of bytes per element).</exception>
        /// <exception cref="ValueError">A dimension is negative, or the byte count overflows.</exception>
        /// <remarks>Footgun: <paramref name="fillZeros"/> <c>false</c> hands out whatever bytes the buffer pool held - the
        ///     caller must overwrite every element before anything reads the array.</remarks>
        internal NDArray(DType dtype, Shape shape, bool fillZeros) : base(CheckedDescr(dtype), shape, fillZeros) { }

        /// <summary>
        ///     The guard of the descriptor constructor: a typed array must store exactly <typeparamref name="TDType"/>.
        /// </summary>
        /// <param name="dtype">The descriptor to allocate under.</param>
        /// <returns><paramref name="dtype"/>, unchanged.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dtype"/> is <c>null</c>.</exception>
        /// <exception cref="ArgumentException"><paramref name="dtype"/>'s element type is not <typeparamref name="TDType"/>.</exception>
        private static DType CheckedDescr(DType dtype)
        {
            if (dtype is null)
                throw new ArgumentNullException(nameof(dtype));
            // Checked before the base constructor allocates: a mismatch would hand out a typed view of differently sized
            // elements, every typed read past the element it means.
            if (dtype.typecode != InfoOf<TDType>.NPTypeCode)
                throw new ArgumentException(
                    $"An NDArray<{typeof(TDType).Name}> cannot be allocated under dtype {dtype}: its elements are not {typeof(TDType).Name}.",
                    nameof(dtype));
            return dtype;
        }

        /// <summary>
        /// Clone the array while preserving the typed NDArray wrapper.
        /// </summary>
        public override NDArray Clone() => new NDArray<TDType>(Storage.Clone()) { TensorEngine = TensorEngine };

        /// <summary>
        /// Array access to storage data - overridden on purpose
        /// </summary>
        /// <value></value>
        protected internal new ArraySlice<TDType> Array
        {
            [MethodImpl(Inline)]
            get => Storage.GetData<TDType>();
            [MethodImpl(Inline)]
            set => Storage.ReplaceData(value);
        }

        /// <summary>
        ///     Gets the address that this NDArray starts from.
        /// </summary>
        protected internal new unsafe TDType* Address
        {
            [MethodImpl(Inline)]
            get => (TDType*)Storage.Address;
        }

        /// <summary>
        ///     NumPy's <c>ndarray.__getitem__(key)</c> for a typed array: the selection <see cref="NDArray.this[object[]]"/>
        ///     makes, typed as this array's element type (<c>typed["1:3"]</c>, <c>typed[0]</c>, <c>typed[mask]</c> are all
        ///     <see cref="NDArray{TDType}"/>). Assigning uses <see cref="NDArray"/>'s setter (NumPy's <c>__setitem__</c>).
        /// </summary>
        /// <param name="key">The index tuple, as for <see cref="NDArray.this[object[]]"/>.</param>
        /// <value>
        ///     GET: the selection as an <see cref="NDArray{TDType}"/> — for one integer per axis NumPy's scalar, a read-only
        ///     0-d COPY of the element (read its value with a cast, <c>(double)typed[0]</c>, or
        ///     <see cref="NDArray.GetValue{T}(long[])"/>; write an element through the setter); a view for any other index of
        ///     integers and slices (a 0-d one is an explicit 0-d array); a new array for an advanced index.
        ///     SET (inherited): the value is stored by NumPy's rules. Through an <see cref="NDArray{TDType}"/> variable the
        ///     value must be an <see cref="NDArray{TDType}"/> or a <typeparamref name="TDType"/> (which converts implicitly
        ///     to a 0-d array); assign any other value through an <see cref="NDArray"/> reference or
        ///     <see cref="NDArray.__setitem__(object, object)"/>.
        /// </value>
        /// <exception cref="ArgumentNullException"><paramref name="key"/> or one of its entries is <c>null</c>.</exception>
        /// <exception cref="IndexError">The index is invalid for this array (NumPy's texts).</exception>
        /// <exception cref="InvalidOperationException">An advanced selection came back in another element type (the
        ///     indexer never changes the element type; the guard keeps a future change from returning a wrongly typed
        ///     array).</exception>
        /// <remarks>
        ///     <para>
        ///     The override is GETTER-ONLY on purpose: C# lets an override narrow its type only when it does not declare a
        ///     setter (a read-write override must keep <see cref="NDArray"/>, error CS1715), and the setter stays
        ///     <see cref="NDArray"/>'s. The typed scalar indexers <c>TDType this[params long[]]</c> /
        ///     <c>TDType this[int[]]</c> are gone with the rule of one NumPy signature: an element is a 0-d array, as for
        ///     every other array.
        ///     </para>
        ///     <para>
        ///     Virtual dispatch reaches this getter also through an <see cref="NDArray"/> reference: the selection of a typed
        ///     array is a typed array object at run time, with the same elements and the same view relationship as the
        ///     untyped selection (the basic index builds ONE typed object, an item's scalar copy or a view of the storage; an
        ///     advanced index wraps the untyped result's own storage object, so a gather still owns its data, and disposes
        ///     the untyped wrapper).
        ///     </para>
        /// </remarks>
        public override NDArray<TDType> this[params object[] key]
        {
            get
            {
                if (key is null)
                    throw new ArgumentNullException(nameof(key));
                // NDArray's three indexing rules, typed (see NDArray.this[object[]]): every key of a NumPy scalar indexes a
                // copy of its value (gentype_subscript), an item is NumPy's scalar (a read-only copy), and any other 0-d
                // view is an explicit 0-d array.
                if (IsScalarSubscript)
                    return TypedScalarSubscript(key);
                if (TryBasicIndex(key, out Slice[] slices))
                {
                    if (IsItemIndex(slices))
                        return TypedItemScalar(slices);
                    UnmanagedStorage view = Storage.GetView(slices);
                    // A trivial index may hand the storage itself back; the result must be its own object, or a reshape of
                    // it would relabel this array too.
                    if (ReferenceEquals(view, Storage))
                        view = Storage.Alias();
                    var typedView = new NDArray<TDType>(view);
                    if (typedView.ndim == 0)
                        typedView.MarkExplicitZeroDimArray();
                    return typedView;
                }
                return TypedSelection(key);
            }
        }

        /// <summary>
        ///     The typed item path: NumPy's scalar of the element a full integer index selects (<see cref="NDArray.ItemScalar"/>),
        ///     built directly as an <see cref="NDArray{TDType}"/> around the shared element copy
        ///     (<see cref="NDArray.CopyItem"/>), so an element read allocates one object, as the untyped path does.
        /// </summary>
        /// <param name="slices">One integer <see cref="Slice"/> per dimension (none for a 0-d array).</param>
        /// <returns>A new read-only 0-d typed array holding a copy of the element, under the descriptor of the element's
        ///     class (<see cref="NDArray.ScalarDescr"/>).</returns>
        /// <exception cref="IndexError">An integer is out of bounds for its axis (the storage's NumPy text).</exception>
        /// <remarks>
        ///     Anchored for the reason <see cref="NDArray.ItemScalar"/> is: the element's bytes are read after this array's last
        ///     use in the body. <see cref="NDArray.PyArray_Return"/> runs as a statement: it returns this same object (see its
        ///     remarks on the typed wrappers), and the typed reference is the one handed out.
        /// </remarks>
        [NDAnchored]
        private unsafe NDArray<TDType> TypedItemScalar(Slice[] slices)
        {
            // The offset (and NumPy's bounds error) before the allocation, as the untyped item path does: no view storage
            // is built for the one element read.
            long offset = ItemOffset(slices);
            // An NDArray<TDType> holds TDType's own type code (TypedSelection enforces it), never text. The descriptor is
            // the element CLASS's own (an np.intc item stays 'i'), not the type code's default class.
            var scalar = new NDArray<TDType>(ScalarDescr(Storage.Descr), Shape.Scalar, false);
            CopyItem(Storage, offset, (byte*)scalar.Address);
            scalar.PyArray_Return();
            return scalar;
        }

        /// <summary>
        ///     The typed <see cref="NDArray.ScalarSubscript"/>: any index of a read-only scalar indexes a writeable copy of
        ///     its value, typed as this array's element type.
        /// </summary>
        /// <param name="key">The index tuple (any key: basic or advanced).</param>
        /// <returns>What the 0-d array of the value gives for <paramref name="key"/> (a view of the copy, an item's scalar,
        ///     a gather): itself when the route already produced a typed array (the copy of a typed array may keep the
        ///     class, and its index re-enters this getter), else a typed wrapper over the SAME storage object the untyped
        ///     route returned.</returns>
        /// <exception cref="IndexError">The 0-d array of the value rejects the key: NumPy's
        ///     <c>invalid index to scalar variable.</c> (see <see cref="NDArray.ScalarSubscript"/>).</exception>
        /// <remarks>
        ///     Shares the untyped result's storage object and disposes the untyped wrapper, as <see cref="TypedSelection"/>
        ///     does: the result is this call's own, held by no one else, and the typed wrapper takes its own counted
        ///     reference before the untyped one is released. The result derives from a fresh copy, never this array's
        ///     storage, so no alias is needed. The explicit 0-d mark is per object, so it is carried over to the typed
        ///     wrapper exactly when the untyped result had it (a view's 0-d result is an explicit array, the item of
        ///     <c>s[()]</c> stays NumPy's scalar).
        /// </remarks>
        private NDArray<TDType> TypedScalarSubscript(object[] key)
        {
            NDArray result = ScalarSubscript(key);
            if (result is NDArray<TDType> typed)
                return typed;
            var wrapped = new NDArray<TDType>(result.Storage);
            if (result.IsExplicitZeroDimArray)
                wrapped.MarkExplicitZeroDimArray();
            result.Dispose();
            return wrapped;
        }

        /// <summary>
        ///     The advanced half of the typed getter: <see cref="NDArray"/>'s selection (a mask, an index array, a bool, a
        ///     mixed tuple), typed.
        /// </summary>
        /// <param name="key">The index tuple (not <c>null</c>, not basic).</param>
        /// <returns>The selection: itself when the untyped indexer already produced a typed array (its general path
        ///     re-enters this override for a nested tuple), else a typed wrapper over the SAME storage object.</returns>
        /// <exception cref="IndexError">The index is invalid for this array (NumPy's texts).</exception>
        /// <exception cref="InvalidOperationException">The selection is not of element type <typeparamref name="TDType"/>.</exception>
        /// <remarks>
        ///     The typed wrapper shares the result's storage object instead of aliasing it (<see cref="NDArray.MakeGeneric{T}"/>),
        ///     because an alias is a view: a fresh gather owns its data (NumPy's <c>flags.owndata</c> is true), and an alias of
        ///     it would read false. Sharing is safe here: the storage is this call's own result, held by no one else (only a
        ///     selection that handed back this array's own storage object is aliased). The untyped wrapper is disposed at
        ///     once: the typed wrapper holds its own counted reference on the data, and <see cref="NDArray.Dispose"/> releases
        ///     only the wrapper's reference, never the storage object. An explicit dispose, not an <see cref="NDScopedAttribute"/>
        ///     scope: the untyped wrapper is the only array this method creates. Cost: the second array object, about 0.1 µs
        ///     on a gather (a typed mask gather 1.02 µs against the untyped 0.91 µs, measured apart).
        /// </remarks>
        private NDArray<TDType> TypedSelection(object[] key)
        {
            // NDArray's indexer body after its basic-index test (which the caller already ran): the mask path or the general
            // advanced path, without sorting the key a second time.
            NDArray result = key.Length == 1 && key[0] is NDArray<bool> mask ? GetMasked(mask) : GetAdvanced(key);
            if (result is NDArray<TDType> typed)
                return typed;
            if (result.typecode != InfoOf<TDType>.NPTypeCode)
            {
                DType selected = result.dtype;
                result.Dispose();
                throw new InvalidOperationException(
                    $"An index of an NDArray<{typeof(TDType).Name}> selected an array of dtype {selected}: the indexer keeps the element type.");
            }
            var wrapped = ReferenceEquals(result.Storage, Storage)
                ? new NDArray<TDType>(Storage.Alias())
                : new NDArray<TDType>(result.Storage);
            result.Dispose();
            return wrapped;
        }

        public new TDType GetAtIndex(long index)
        {
            unsafe
            {
                return *(Address + Shape.TransformOffset(index));
            }
        }


        /// <summary>
        ///     A 1-D iterator over the array.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flat.html</remarks>
        public new NDArray<TDType> flat
        {
            [NDScoped]
            get
            {
                return base.flat.MakeGeneric<TDType>();
            }
        }

        /// <summary>
        ///     The transposed array. <br></br>
        ///     Same as self.transpose().
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.T.html</remarks>
        public new NDArray<TDType> T
        {
            [NDScoped]
            get
            {
                return transpose().MakeGeneric<TDType>();
            }
        }


        [MethodImpl(Inline)]
        public static implicit operator ArraySlice<TDType>(NDArray<TDType> nd) => nd.Array;

        /// <summary>
        ///     A <typeparamref name="TDType"/> value as a 0-d array of its own type: what lets an element be assigned through
        ///     a typed variable's indexer (<c>typed[0] = 3.14</c>, <c>mask[m] = false</c>), whose setter value is an
        ///     <see cref="NDArray{TDType}"/> (see <see cref="this[object[]]"/>).
        /// </summary>
        /// <param name="value">The element value.</param>
        /// <returns>A new writeable 0-d array holding <paramref name="value"/>, its dtype <typeparamref name="TDType"/>'s.</returns>
        /// <remarks>
        ///     The array is a plain (STRONG) typed scalar, not NumPy's Python scalar: it already has the destination's type,
        ///     so a store never range-checks or converts it. Where both this conversion and <see cref="NDArray"/>'s scalar
        ///     conversions apply, C# prefers this one only for an <see cref="NDArray{TDType}"/> parameter (the typed
        ///     operators <c>&amp;</c>, <c>|</c>, <c>^</c>), whose results are the same.
        /// </remarks>
        [MethodImpl(Inline)]
        public static implicit operator NDArray<TDType>(TDType value) => new NDArray<TDType>(UnmanagedStorage.Scalar(value));

        [MethodImpl(Inline), NDScoped]
        public static explicit operator NDArray<TDType>(TDType[] tArray) { return new NDArray(tArray).MakeGeneric<TDType>(); }

    }
}
