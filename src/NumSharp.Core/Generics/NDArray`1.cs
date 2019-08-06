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

// ReSharper disable once CheckNamespace
namespace NumSharp.Generic
{
    public class NDArray<T> : NDArray where T : unmanaged
    {
        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage) : base(storage)
        {
            if (storage.DType != typeof(T))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(T).Name}", nameof(storage));
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage, Shape shape) : base(storage, shape)
        {
            if (storage.DType != typeof(T))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(T).Name}", nameof(storage));
        }

        /// <summary>
        ///     Creates a new <see cref="NDArray"/> with this storage.
        /// </summary>
        /// <param name="storage"></param>
        protected internal NDArray(UnmanagedStorage storage, ref Shape shape) : base(storage, ref shape)
        {
            if (storage.DType != typeof(T))
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(T).Name}", nameof(storage));
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <param name="engine">The engine of this <see cref="NDArray"/></param>
        /// <remarks>This constructor does not call allocation/></remarks>
        protected internal NDArray(TensorEngine engine) : base(typeof(T).GetTypeCode(), engine) { }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        /// <remarks>This constructor does not call allocation/></remarks>
        public NDArray() : base(typeof(T).GetTypeCode()) { }

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
            if (dtype != typeof(T))
                throw new ArgumentException($"Array type must be the same as T. {dtype.Name} != {typeof(T).Name}", nameof(values));
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
            if (underlying != typeof(T))
                throw new ArgumentException($"Array type must be the same as T. {underlying.Name} != {typeof(T).Name}", nameof(values));
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Shape shape) : base(typeof(T).GetTypeCode(), shape) { }

        /// <summary>
        ///     Constructor which initialize elements with length of <paramref name="size"/>
        /// </summary>
        /// <param name="size">The size as a single dimension shape</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(int size) : base(typeof(T).GetTypeCode(), size) { }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <param name="fillZeros">Should set the values of the new allocation to default(dtype)? otherwise - old memory noise</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Shape shape, bool fillZeros) : base(typeof(T).GetTypeCode(), shape, fillZeros) { }

        /// <summary>
        /// Array access to storage data - overridden on purpose
        /// </summary>
        /// <value></value>
        protected internal new ArraySlice<T> Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Storage.GetData<T>();
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => Storage.ReplaceData(value);
        }

        /// <summary>
        ///     Gets the address that this NDArray starts from.
        /// </summary>
        protected internal new unsafe T* Address
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => (T*)Storage.Address;
        }

        public new T this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(T).Name}> to a non-scalar indices", nameof(indices));

                return Storage.GetValue<T>(indices);
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                if (Shape.IsScalar && indices.Length != 1 || !Shape.IsScalar && indices.Length != ndim)
                    throw new ArgumentException($"Unable to set an NDArray<{typeof(T).Name}> to a non-scalar indices", nameof(indices));

                Storage.SetValue<T>(value, indices);
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        public new NDArray<T> this[string slice]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return base[slice].MakeGeneric<T>();
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
        public new NDArray<T> this[params Slice[] slices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => base[slices].MakeGeneric<T>();

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => base[slices] = value;
        }

        public new T GetAtIndex(int index)
        {
            unsafe
            {
                return *(Address + Shape.TransformOffset(index));
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator ArraySlice<T>(NDArray<T> nd) => nd.Array;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator NDArray<T>(T[] tArray) => new NDArray(tArray).MakeGeneric<T>();
    }
}
