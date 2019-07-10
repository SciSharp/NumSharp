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
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Globalization;
using System.Collections;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

// ReSharper disable once CheckNamespace
namespace NumSharp.Generic
{
    public class NDArray<T> : NDArray where T : unmanaged
    {
        public NDArray() : base(typeof(T))
        { }

        public NDArray(UnmanagedStorage storage) : base(storage)
        {
            if (typeof(T) != storage.DType)
            {
                throw new ArgumentException($"Storage type must be the same as T. {storage.DType.Name} != {typeof(T).Name}", nameof(storage));
            }
        }

        public NDArray(Shape shape) : base(typeof(T))
        {
            Storage.Allocate(shape);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        /// <remarks>This constructor calls <see cref="IStorage.Allocate(NumSharp.Shape,System.Type)"/></remarks>
        public NDArray(Type dtype, Shape shape) : base(dtype, shape) { }


        public NDArray(Array array, Shape shape) : this(shape)
        {
            Storage.ReplaceData(array);
        }

        /// <summary>
        /// Array access to storage data - overridden on purpose
        /// </summary>
        /// <value></value>
        internal new ArraySlice<T> Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return Storage.GetData<T>();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                Storage.ReplaceData(value);
            }
        }

        /// <summary>
        /// indexing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        public new T this[params int[] select]
        {
            get
            {
                return Storage.GetData<T>(select);
            }

            set
            {
                Storage.SetData(value, select);
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
                this[slice].Array = value.Data<T>();
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        public new NDArray<T> this[params Slice[] slices]
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                return base[slices].MakeGeneric<T>();
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
                this[slice].Array = value.Data<T>();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator ArraySlice<T>(NDArray<T> nd)
        {
            return nd.Array;
        }

        public static explicit operator NDArray<T>(T[] tArray)
        {
            return new NDArray(tArray).MakeGeneric<T>(); //TODO! unit test it
        }
    }
}
