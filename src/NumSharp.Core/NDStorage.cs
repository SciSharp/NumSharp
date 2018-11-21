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
using System.Text;

namespace NumSharp.Core
{
    /// <summary>
    /// Numerical dynamic storage
    /// </summary>
    public class NDStorage //: IComparable, IComparable<Double>, IConvertible, IEquatable<Double>, IFormattable
    {
        /// <summary>
        /// storage for Int32
        /// </summary>
        public int[] Int32 { get; set; }

        /// <summary>
        /// storage for double
        /// </summary>
        public double[] Double8 { get; set; }

        private Type dtype;

        public NDStorage(Type dtype)
        {
            this.dtype = dtype;
        }

        public void Set<T>(T[] value)
        {
            switch (value)
            {
                case int[] v:
                    Int32 = v;
                    break;
                case double[] v:
                    Double8 = v;
                    break;
            }
        }

        /// <summary>
        /// Allocate memory of size
        /// </summary>
        /// <param name="size"></param>
        public void Allocate(int size)
        {
            switch (dtype.Name)
            {
                case "Int32":
                    Int32 = new int[size];
                    break;
                case "Double":
                    Double8 = new double[size];
                    break;
            }
        }
    }
}
