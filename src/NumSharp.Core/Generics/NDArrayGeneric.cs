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
using NumSharp.Backends;

namespace NumSharp.Generic
{
    public class NDArray<T> : NDArray where T : struct
    {
        public NDArray() : base(typeof(T))
        {
        }

        public NDArray(IStorage storage) : base(storage)
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

        public NDArray(Array array, Shape shape) : this(shape)
        {
            Storage.SetData(array);
        }

        /// <summary>
        /// Array access to storage data - overridden on purpose
        /// </summary>
        /// <value></value>
        new public T[] Array
        {
            get
            {
                return Storage.GetData<T>();
            }

            set
            {
                Storage.SetData<T>(value);
            }
        }
        /// <summary>
        /// indexing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        new public T this[params int[] select]
        {
            get
            {
                return Storage.GetData<T>(select);
            }

            set
            {
                Storage.SetData<T>(value, select);
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        new public NDArray<T> this[string slice]
        {
            get
            {
                return base[slice].MakeGeneric<T>();
            }

            set
            {
                Array = value.Data<T>();
            }
        }

        /// <summary>
        /// slicing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        new public NDArray<T> this[params Slice[] slices]
        {
            get
            {
                return base[slices].MakeGeneric<T>();
            }

            set
            {
                Array = value.Data<T>();
            }
        }

        public static implicit operator T[] (NDArray<T> nd)
        {
            return nd.Array;
        }
        /*
        public static implicit operator NDArray<T>(T[] tArray)
        {
            var genericArray = new NDArray<T>();
            genericArray.Array = tArray;
            return genericArray;
        }
        */

    }

}
