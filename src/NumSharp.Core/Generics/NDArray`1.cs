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
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(int size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros)
        { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
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
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Shape shape) : base(InfoOf<TDType>.NPTypeCode, shape) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(int size) : base(InfoOf<TDType>.NPTypeCode, size) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(long size) : base(InfoOf<TDType>.NPTypeCode, size) { }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Shape shape, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, shape, fillZeros) { }

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

        public new TDType this[int[] indices]
        {
            [MethodImpl(Inline)]
            get
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(TDType).Name}> to a non-scalar indices", nameof(indices));

                return Storage.GetValue<TDType>(indices);
            }
            [MethodImpl(Inline)]
            set
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(TDType).Name}> to a non-scalar indices", nameof(indices));

                Storage.SetValue<TDType>(value, indices);
            }
        }

        public new TDType this[params long[] indices]
        {
            [MethodImpl(Inline)]
            get
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(TDType).Name}> to a non-scalar indices", nameof(indices));

                return Storage.GetValue<TDType>(indices);
            }
            [MethodImpl(Inline)]
            set
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(TDType).Name}> to a non-scalar indices", nameof(indices));

                Storage.SetValue<TDType>(value, indices);
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        public new NDArray<TDType> this[string slice]
        {
            // NDScoped: the untyped base view is reclaimed once the yielded typed alias owns the data.
            [MethodImpl(Inline), NDScoped]
            get
            {
                return base[slice].MakeGeneric<TDType>();
            }

            [MethodImpl(Inline)]
            set
            {
                base[slice] = value;
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        public new NDArray<TDType> this[params Slice[] slices]
        {
            [MethodImpl(Inline), NDScoped]
            get { return base[slices].MakeGeneric<TDType>(); }

            [MethodImpl(Inline)]
            set => base[slices] = value;
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

        [MethodImpl(Inline), NDScoped]
        public static explicit operator NDArray<TDType>(TDType[] tArray) { return new NDArray(tArray).MakeGeneric<TDType>(); }

    }
}
