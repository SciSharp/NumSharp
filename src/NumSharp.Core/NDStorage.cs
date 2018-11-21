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
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Core
{
    /// <summary>
    /// Numerical dynamic storage
    /// </summary>
    public class NDStorage : IEnumerable, IEnumerator//IComparable, IComparable<Double>, IConvertible, IEquatable<Double>, IFormattable
    {
        /// <summary>
        /// storage for Int32
        /// </summary>
        public int[] Int32 { get; set; }

        /// <summary>
        /// storage for double
        /// </summary>
        public double[] Double8 { get; set; }

        /// <summary>
        /// storage for string
        /// </summary>
        public string[] StringN { get; set; }

        /// <summary>
        /// storage for bytes
        /// </summary>
        public byte[] Bytes { get; set; }

        private Type dtype;

        public Shape Shape { get; set; }

        public NDStorage(Type dtype)
        {
            this.dtype = dtype;
        }

        /// <summary>
        /// It's convenint but not recommended due to low performance
        /// recommend to use NDStorage.Int32 directly
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>()
        {
            switch (dtype.Name)
            {
                case "Int32":
                    return Int32 as T[];
                case "Double":
                    return Double8 as T[];
            }

            return null;
        }

        /// <summary>
        /// It's convenint but not recommended due to low performance
        /// recommend to use NDStorage.Int32 directly
        /// </summary>
        public object[] Values
        {
            get
            {
                switch (dtype.Name)
                {
                    case "Int32":
                        return Int32.Cast<object>().ToArray();
                    case "Double":
                        return Double8.Cast<object>().ToArray();
                }

                return null;
            }
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

        private int pos = -1;
        public object Current
        {
            get
            {
                if (Shape.Length == 1)
                {
                    switch (dtype.Name)
                    {
                        case "Int32":
                            return Int32[pos];
                        case "Double":
                            return Double8[pos];
                    }

                    return null;
                }
                else
                {
                    return null;
                }
            }
        }

        public IEnumerator GetEnumerator()
        {
            return this;
        }

        public bool MoveNext()
        {
            pos++;
            return pos < Shape[0];
        }

        public void Reset()
        {
            pos = -1;
        }
    }
}
