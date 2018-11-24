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
        /// memory allocation
        /// </summary>
        private Array values { get; set; }

        private Type dtype;

        public Shape Shape { get; set; }

        public NDStorage(Type dtype)
        {
            this.dtype = dtype;
        }

        public object this[int idx]
        {
            get
            {
                switch (values)
                {
                    case int[] v:
                        return v[idx];
                    case float[] v:
                        return v[idx];
                    case double[] v:
                        return v[idx];
                }

                return null;
            }

            set
            {
                switch (values)
                {
                    case int[] v:
                        v[idx] = (int)value;
                        break;
                    case float[] v:
                        v[idx] = (float)value;
                        break;
                    case double[] v:
                        v[idx] = (double)value;
                        break;
                }
            }
        }

        public object Data()
        {
            return values;
        }

        /// <summary>
        /// It's convenint but not recommended due to low performance
        /// recommend to use NDStorage.Int32 directly
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>()
        {
            return values as T[];
        }

        public void Set<T>(T[] value)
        {
            values = value;
        }

        /// <summary>
        /// Allocate memory of size
        /// </summary>
        /// <param name="size"></param>
        internal void Allocate(int size)
        {
            switch (dtype.Name)
            {
                case "Int32":
                    values = new int[size];
                    break;
                case "Single":
                    values = new float[size];
                    break;
                case "Double":
                    values = new double[size];
                    break;
            }
        }

        private int pos = -1;
        public object Current
        {
            get
            {
                if (Shape.NDim == 1)
                {
                    switch (values)
                    {
                        case int[] a:
                            return a[pos];
                        case float[] a:
                            return a[pos];
                        case double[] a:
                            return a[pos];
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
