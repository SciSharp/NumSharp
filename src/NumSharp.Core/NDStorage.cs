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

        public int Length => values.Length;

        public NDStorage(Type dtype)
        {
            this.dtype = dtype;
        }

        /// <summary>
        /// Retrieve element
        /// low performance, use generic Data<T> method for performance sensitive invoke
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public object this[params int[] select]
        {
            get
            {
                if (select.Length == Shape.NDim)
                {
                    switch (values)
                    {
                        case double[] values:
                            return values[Shape.GetIndexInShape(select)];
                        case int[] values:
                            return values[Shape.GetIndexInShape(select)];
                    }

                    return null;
                }
                else
                {
                    int start = Shape.GetIndexInShape(select);
                    int length = Shape.DimOffset[select.Length - 1];

                    var nd = new NDArray(dtype);

                    switch (values)
                    {
                        case double[] values:
                            Span<double> double8 = Data<double>();
                            nd.Storage.Set(double8.Slice(start, length).ToArray());
                            break;
                        case int[] values:
                            Span<int> int32 = Data<int>();
                            nd.Storage.Set(int32.Slice(start, length).ToArray());
                            break;
                    }

                    int[] shape = new int[Shape.NDim - select.Length];
                    for (int i = select.Length; i < Shape.NDim; i++)
                    {
                        shape[i - select.Length] = Shape[i];
                    }
                    nd.Shape = new Shape(shape);

                    return nd;
                }
            }

            set
            {
                if (select.Length == Shape.NDim)
                {
                    switch (values)
                    {
                        case double[] values:
                            values[Shape.GetIndexInShape(select)] = (double)value;
                            break;
                        case int[] values:
                            values[Shape.GetIndexInShape(select)] = (int)value;
                            break;
                    }
                }
                else
                {
                    int start = Shape.GetIndexInShape(select);
                    int length = Shape.DimOffset[Shape.NDim - 1];

                    switch (value)
                    {
                        case double v:
                            Span<double> data1 = Data<double>();
                            var elements1 = data1.Slice(start, length);

                            for (int i = 0; i < elements1.Length; i++)
                            {
                                elements1[i] = v;
                            }

                            break;
                        case int v:
                            Span<int> data2 = Data<int>();
                            var elements2 = data2.Slice(start, length);

                            for (int i = 0; i < elements2.Length; i++)
                            {
                                elements2[i] = v;
                            }

                            break;
                    }


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

        public void Set<T>(Shape shape, T value)
        {
            if (shape.NDim == Shape.NDim)
            {
                throw new Exception("Please use NDArray[m, n] to access element.");
            }
            else
            {
                int start = Shape.GetIndexInShape(shape.Shapes.ToArray());
                int length = Shape.DimOffset[shape.NDim - 1];

                Span<T> data = Data<T>();
                var elements = data.Slice(start, length);

                for (int i = 0; i < elements.Length; i++)
                {
                    elements[i] = value;
                }
            }
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
