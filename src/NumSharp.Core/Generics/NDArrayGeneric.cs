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
        public NDArray()
        {
            Storage = new DefaultEngine(typeof(T));
            Storage.Allocate(this.dtype,new Shape(1));
        }
        public NDArray(Shape shape) : this()
        {
            Storage = new DefaultEngine(typeof(T));
            Storage.Allocate(this.dtype, shape);
        }
        /// <summary>
        /// indexing of generic - overridden on purpose
        /// </summary>
        /// <value></value>
        new public T this[params int[] select]
        {
            get
            {
                return (T) Storage.GetData(select);
            }

            set
            {
                Storage.SetData(value, select);
            }
        }
    }
}
 