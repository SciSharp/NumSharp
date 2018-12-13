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
using System.Numerics;

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
        public Type dtype {get;set;}
        public Shape Shape { get; set; }
        public int Length => values.Length;
        public NDStorage()
        {
            dtype = typeof(ValueType);
            Shape = new Shape(1);
            values = new int[] {0};
        }
        public NDStorage(Type dtype)
        {
            this.dtype = dtype;
            Shape = new Shape(1);
            values = Array.CreateInstance(dtype,1);
        }
        /// <summary>
        /// Create a NDStorage by data type and array shape
        /// </summary>
        /// <param name="dtype">The type of arrays elements</param>
        /// <param name="shape">The shape of array/param>
        /// <returns>The constructed NDStorage</returns>
        public static NDStorage CreateByShapeAndType(Type dtype,Shape shape)
        {
            var storage = new NDStorage(dtype);
            storage.Shape = shape;
            storage.Allocate(shape.Size);
            return storage;
        }
        public static NDStorage CreateByArray(Array values)
        {
            Type dtype = null;

            if ( !values.GetType().GetElementType().IsArray  )
                dtype = values.GetType().GetElementType();
            else 
                throw new IncorrectShapeException();

            int[] dims = new int[values.Rank];

            for (int idx = 0; idx < dims.Length;idx++)
                dims[idx] = values.GetLength(idx);

            var storage = NDStorage.CreateByShapeAndType(dtype,new Shape(dims));
            storage.values = Array.CreateInstance(dtype,values.Length);

            storage.SetData(values);

            return storage;
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
                            Span<double> double8 = GetData<double>();
                            nd.Storage.Set(double8.Slice(start, length).ToArray());
                            break;
                        case int[] values:
                            Span<int> int32 = GetData<int>();
                            nd.Storage.Set(int32.Slice(start, length).ToArray());
                            break;
                    }

                    int[] shape = new int[Shape.NDim - select.Length];
                    for (int i = select.Length; i < Shape.NDim; i++)
                    {
                        shape[i - select.Length] = Shape[i];
                    }
                    nd.Storage.Shape = new Shape(shape);

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
                            Span<double> data1 = GetData<double>();
                            var elements1 = data1.Slice(start, length);

                            for (int i = 0; i < elements1.Length; i++)
                            {
                                elements1[i] = v;
                            }

                            break;
                        case int v:
                            Span<int> data2 = GetData<int>();
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
        /// <summary>
        /// Get all elements as one System.Array object without shape
        /// </summary>
        /// <returns></returns>
        public Array GetData()
        {
            return values;
        }
        /// <summary>
        /// Get all elements with correct type by parameter instead of generic
        /// </summary>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public Array GetData(Type dtype)
        {
            var methods = this.GetType().GetMethods().Where(x => x.Name.Equals("GetData") && x.IsGenericMethod && x.ReturnType.Name.Equals("T[]"));
            var genMethods = methods.First().MakeGenericMethod(dtype);

            return (Array) genMethods.Invoke(this,null);
        }
        public void SetData(Array values)
        {
            this.values = values;
        }
        public void SetData(object value, params int[] indexes)
        {
            this.values.SetValue(value,Shape.GetIndexInShape(indexes));
        }
        /// <summary>
        /// Get specific element depending on Shape of array
        /// </summary>
        /// <param name="indexes">The indexes of dimensions</param>
        /// <returns></returns>
        public object GetData(params int[] indexes)
        {
            object element = null;
            if (indexes.Length == Shape.NDim)
                element = values.GetValue(Shape.GetIndexInShape(indexes));
            else if (Shape.Shapes.Last() == 1)
                element = values.GetValue(Shape.GetIndexInShape(indexes));
            else
                throw new Exception("indexes must be equal to number of dimension.");
            return element;
        }
        /// <summary>
        /// Return all elements as one 1D .NET array but cast to specific type.
        /// </summary>
        /// <typeparam name="T">Data type of elements</typeparam>
        /// <returns>casted array</returns>
        public T[] GetData<T>()
        {
            T[] returnArray = null;

            if (values.GetType().GetElementType() == typeof(T))
                returnArray = values as T[];
            else 
            {  
                returnArray = new T[values.Length];
                switch (Type.GetTypeCode(typeof(T))) 
                {
                    case TypeCode.Double : 
                    {
                        var returnArray_ = returnArray as double[];
                        for(int idx = 0;idx < returnArray.Length;idx++)
                            returnArray_[idx] =  Convert.ToDouble(values.GetValue(idx));
                        break;
                    }
                    case TypeCode.Single : 
                    {
                        var returnArray_ = returnArray as float[];
                        for(int idx = 0;idx < returnArray.Length;idx++)
                            returnArray_[idx] =  Convert.ToSingle(values.GetValue(idx));
                        break;
                    }
                    case TypeCode.Decimal : 
                    {
                        var returnArray_ = returnArray as decimal[];
                        for(int idx = 0;idx < returnArray.Length;idx++)
                            returnArray_[idx] =  Convert.ToDecimal(values.GetValue(idx));
                        break;    
                    }
                    case TypeCode.Int32 : 
                    {
                        var returnArray_ = returnArray as int[];
                        for(int idx = 0;idx < returnArray.Length;idx++)
                            returnArray_[idx] =  Convert.ToInt32(values.GetValue(idx));
                        break;
                    }
                    case TypeCode.Int64 :
                    {
                        var returnArray_ = returnArray as Int64[];
                        for(int idx = 0;idx < returnArray.Length;idx++)
                            returnArray_[idx] =  Convert.ToInt64(values.GetValue(idx));
                        break;
                    }
                    case TypeCode.Object : 
                    {
                        if( typeof(T) == typeof(Complex) )
                        {
                            var returnArray_ = returnArray as Complex[];
                            for(int idx = 0;idx < returnArray.Length;idx++)
                                returnArray_[idx] = new Complex((double)values.GetValue(idx),0);
                        break;
                        }
                        else if ( typeof(T) == typeof(Quaternion) )
                        {
                            var returnArray_ = returnArray as Quaternion[];
                            for(int idx = 0;idx < returnArray.Length;idx++)
                                returnArray_[idx] = new Quaternion(new Vector3(0,0,0),(float)values.GetValue(idx));
                        break;
                        }
                        else 
                        {
                            var returnArray_ = returnArray as object[];
                            for(int idx = 0;idx < returnArray.Length;idx++)
                                returnArray_[idx] = values.GetValue(idx);
                        }
                        break;
                    }
                    default : 
                    {
                        break;
                    }
                } 
            }
            return returnArray;
        }
        public T GetData<T>(params int[] indexes)
        {
            T element;
            T[] elements = values as T[];
            if (indexes.Length == Shape.NDim)
                element = elements[Shape.GetIndexInShape(indexes)];
            else
                throw new Exception("indexes must be equal to number of dimension.");
            return element;
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
        public T[] Data<T>()
        {
            return values as T[];
        }

        /// <summary>
        /// Allocate memory of size
        /// </summary>
        /// <param name="size"></param>
        internal void Allocate(int size)
        {
            values = Array.CreateInstance(this.dtype,size);
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
