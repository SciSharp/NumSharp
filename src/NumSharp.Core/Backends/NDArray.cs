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
        /// <summary>
        /// Dimension count
        /// </summary>
        public int ndim => Storage.Shape.NDim;
        /// <summary>
        /// Total of elements
        /// </summary>
        public int size => Storage.Shape.Size;

        public int dtypesize => Storage.DTypeSize;

        /// <summary>
        /// The internal storage for elements of NDArray
        /// </summary>
        /// <value>Internal Storage</value>
        public IStorage Storage { get; set; }

        public ITensorEngine TensorEngine { get; set; }

        /// <summary>
        /// Shortcut for access internal elements
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>() => Storage.GetData<T>();

        public T Data<T>(int index) => Storage.GetData<T>()[index];

        public T Data<T>(params int[] indexes) => Storage.GetData<T>(indexes);

        public Array Data() => Storage.GetData(dtype);

        public T Max<T>() => Data<T>().Max();

        public void astype(Type dtype) => Storage.SetData(Storage.GetData(), dtype);

        /*public NDArray()
        {
            throw new Exception("Don't use 0 parameter constructor.");
        }*/

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        public NDArray(Type dtype)
        {
            TensorEngine = BackendFactory.GetEngine();
            Storage = new NDStorage(dtype);
        }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Array with values</returns>
        public NDArray(Array values, Shape shape = null) : this(values.GetType().GetElementType())
        {
            if (shape is null)
                shape = new Shape(values.Length);

            Storage = new NDStorage(dtype);
            Storage.Allocate(new Shape(shape));
            Storage.SetData(values);
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        public NDArray(Type dtype, Shape shape)
        {
            TensorEngine = BackendFactory.GetEngine();
            Storage = new NDStorage(dtype);
            Storage.Allocate(shape);
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
        /// Determines if NDArray references are the same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case NDArray safeCastObj:
                {
                    var thatData = safeCastObj.Storage?.GetData();
                    if (thatData == null)
                    {
                        return false;
                    }

                    var thisData = this.Storage?.GetData();
                    return thisData == thatData && safeCastObj.shape == this.shape;

                }
                // Other object is not of Type NDArray, return false immediately.
                default:
                    return false;
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
                    for (int i = 0; i < size; i++)
                        yield return Data<int>(i);
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
                Storage.SetData(Data(), dtype);

            return new NDArray(Data(), shape);
        }
    }
}
