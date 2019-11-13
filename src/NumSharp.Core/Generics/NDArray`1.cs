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
        /// <param name="dtype">Internal data type</param>
        /// <param name="size">The size as a single dimension shape</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(int size, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, size, fillZeros)
        { }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape"></param>
        /// <param name="order"></param>
        /// <returns>Array with values</returns>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Array values, Shape shape = default, char order = 'C') : base(values, shape, order)
        {
            if (dtype != typeof(TDType))
                throw new ArgumentException($"Array type must be the same as T. {dtype.Name} != {typeof(TDType).Name}", nameof(values));
        }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <param name="shape"></param>
        /// <param name="order"></param>
        /// <returns>Array with values</returns>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
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
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Shape shape, bool fillZeros) : base(InfoOf<TDType>.NPTypeCode, shape, fillZeros) { }

        /// <summary>
        /// Array access to storage data - overridden on purpose
        /// </summary>
        /// <value></value>
        protected internal new ArraySlice<TDType> Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.GetData<TDType>();
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Storage.ReplaceData(value);
        }

        /// <summary>
        ///     Gets the address that this NDArray starts from.
        /// </summary>
        protected internal new unsafe TDType* Address
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (TDType*)Storage.Address;
        }

        public new TDType this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(TDType).Name}> to a non-scalar indices", nameof(indices));

                return Storage.GetValue<TDType>(indices);
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return base[slice].MakeGeneric<TDType>();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
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
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => base[slices].MakeGeneric<TDType>();

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => base[slices] = value;
        }

        public new TDType GetAtIndex(int index)
        {
            unsafe
            {
                return *(Address + Shape.TransformOffset(index));
            }
        }


        /// <summary>
        ///     A 1-D iterator over the array.
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html</remarks>
        public new NDArray<TDType> flat
        {
            get
            {
                return base.flat.MakeGeneric<TDType>();
            }
        }

        /// <summary>
        ///     The transposed array. <br></br>
        ///     Same as self.transpose().
        /// </summary>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html</remarks>
        public new NDArray<TDType> T
        {
            get
            {
                return transpose().MakeGeneric<TDType>();
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator ArraySlice<TDType>(NDArray<TDType> nd) => nd.Array;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator NDArray<TDType>(TDType[] tArray) => new NDArray(tArray).MakeGeneric<TDType>();

    }
}
