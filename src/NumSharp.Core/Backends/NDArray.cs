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
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArray : ICloneable, IEnumerable
    {
        /// <summary>
        /// Data type of NDArray
        /// </summary>
        public Type dtype => Storage.DType;
        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public int[] shape => Storage.Shape.Dimensions;

        public int len => Storage.Shape.NDim == 0 ? 1 : Storage.Shape.Dimensions[0];

        /// <summary>
        /// Dimension count
        /// </summary>
        public int ndim => Storage.Shape.NDim;
        /// <summary>
        /// Total of elements
        /// </summary>
        public int size => Storage.Shape.Size;

        public int dtypesize => Storage.DTypeSize;

        public string order => Storage.Shape.Order;

        public Slice slice => Storage.Slice;

        public int[] strides => Storage.Shape.Strides;

        /// <summary>
        /// The internal storage for elements of NDArray
        /// </summary>
        /// <value>Internal Storage</value>
        protected IStorage Storage { get; set; }

        public ITensorEngine TensorEngine { get; set; }

        /// <summary>
        /// Shortcut for access internal elements
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>()
        {
            if (slice is null)
                return Storage.GetData<T>();
            else if (Storage.SupportsSpan)
                return Storage.View<T>().ToArray();
            else 
               return Storage.GetData<T>();
        }

        public T Data<T>(params int[] indice) => Storage.GetData<T>(indice);

        public bool GetBoolean(params int[] indice) => Storage.GetBoolean(indice);
        public short GetInt16(params int[] indice) => Storage.GetInt16(indice);
        public int GetInt32(params int[] indice) => Storage.GetInt32(indice);
        public long GetInt64(params int[] indice) => Storage.GetInt64(indice);
        public float GetSingle(params int[] indice) => Storage.GetSingle(indice);
        public double GetDouble(params int[] indice) => Storage.GetDouble(indice);
        public decimal GetDecimal(params int[] indice) => Storage.GetDecimal(indice);
        public string GetString(params int[] indice) => Storage.GetString(indice);
        public NDArray GetNDArray(params int[] indice) => Storage.GetNDArray(indice);

        public void SetData<T>(T value, params int[] indice) => Storage.SetData(value, indice);

        public int GetIndexInShape(params int[] select) => Storage.Shape.GetIndexInShape(slice, select);

        public Array Array
        {
            get
            {
                 return Storage.GetData();
            }

            set
            {
                Storage.SetData(value);
            }
        }

        public T[] CloneData<T>() => Storage.CloneData() as T[];

        public T Max<T>() => Data<T>().Max();

        // NumPy Signature: ndarray.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)
        public NDArray astype(Type dtype, bool copy=true)
        {
            if (copy)
            {
                var result = new NDArray(Storage.DType, Storage.Shape);
                result.SetData(Storage.GetData());
                result.Storage.SetData(Storage.GetData(), dtype);
                return result;
            } else {
                Storage.SetData(Storage.GetData(), dtype);
                return this;
            }
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        public NDArray(Type dtype)
        {
            TensorEngine = BackendFactory.GetEngine();
            Storage = BackendFactory.GetStorage(dtype);
        }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Array with values</returns>
        public NDArray(Array values, Shape shape = null, string order = "C") : this(values.GetType().GetElementType())
        {
            if (shape is null)
                shape = new Shape(values.Length);

            shape.ChangeTensorLayout(order);
            Storage.Allocate(shape);
            Storage.SetData(values);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        public NDArray(Type dtype, Shape shape) : this(dtype)
        {
            Storage.Allocate(shape);
        }

        public NDArray(IStorage storage)
        {
            Storage=storage;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                var result = 1337;
                result = (result * 397) ^ this.ndim;
                result = (result * 397) ^ this.size;
                return result;
            }
        }

        /// <summary>
        /// Clone the whole NDArray
        /// internal storage is also cloned into 2nd memory area
        /// </summary>
        /// <returns>Cloned NDArray</returns>
        public object Clone()
        {
            var puffer = new NDArray(this.dtype);
            var shapePuffer = new Shape(this.shape);
            puffer.Storage.Allocate(shapePuffer);

            puffer.Storage.SetData(this.Storage.CloneData());

            return puffer;
        }

        public IEnumerator GetEnumerator()
        {
            switch (dtype.Name)
            {
                case "Int32":
                    if (ndim == 0)
                    {
                        throw new Exception("Can't iterate scalar ndarray.");
                    }
                    if (ndim == 1)
                    {
                        var data = Data<int>();
                        for (int i = 0; i < size; i++)
                            yield return data[i];
                    }
                    else
                    {
                        for (int i = 0; i < shape[0]; i++)
                            yield return this[i];
                    }

                    break;
                case "Single":
                    for (int i = 0; i < size; i++)
                        yield return Data<float>(i);
                    break;
                default:
                    throw new NotImplementedException("ndarray.GetEnumerator");
            }
            
        }

        /// <summary>
        /// New view of array with the same data.
        /// </summary>
        /// <returns></returns>
        public NDArray view(Type dtype = null)
        {
            if (dtype != null && dtype != this.dtype)
                Storage.SetData(Array, dtype);

            var nd = new NDArray(Array, shape);

            return nd;
        }
    }
}
