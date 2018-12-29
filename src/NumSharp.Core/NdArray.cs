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
using NumSharp.Core.Interfaces;
using NumSharp.Core;

namespace NumSharp.Core
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArray : ICloneable
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
        public NDStorage Storage {get;set;}

        /// <summary>
        /// Shortcut for access internal elements
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T[] Data<T>() => Storage.GetData<T>();

        public Array Data() => Storage.GetData(dtype);

        /// <summary>
        /// Default constructor 
        /// Create a 1D double array with 1 element
        /// one element is 1
        /// </summary>
        public NDArray()
        {
            Storage = new NDStorage();
        }

        /// <summary>
        /// Constructor for init data type
        /// internal storage is 1D with 1 element
        /// </summary>
        /// <param name="dtype">Data type of elements</param>
        public NDArray(Type dtype)
        {
            Storage = new NDStorage(dtype);
        }

        /// <summary>
        /// Constructor which takes .NET array
        /// dtype and shape is determined from array
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Array with values</returns>
        public NDArray(Array values ) : this(values.GetType().GetElementType())
        {
            int[] strgDim = new int[values.Rank];

            for(int idx = 0; idx < strgDim.Length;idx++)
                strgDim[idx] = values.GetLength(idx);
            
            Storage.Allocate(Storage.DType,new Shape(strgDim),1);

            switch( values.Rank )
            {
                case 1 :
                {
                    Storage.SetData(values);
                    break;
                }
            } 
        }

        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and shape are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shape">Shape of NDArray</param>
        public NDArray(Type dtype, IShape shape)
        {
            Storage = new NDStorage();
            Storage.Allocate(dtype,shape,1);
        }
        /// <summary>
        /// Constructor which initialize elements with 0
        /// type and dimensions are given.
        /// </summary>
        /// <param name="dtype">internal data type</param>
        /// <param name="shapes">dimensions</param>
        /// <returns></returns>
        public NDArray(Type dtype, params int[] shapes) : this(dtype, new Shape(shapes) )
        {
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
        /// Determines if NDarray references are the same
        /// </summary>
        /// <param name="obj">NDArray to compare</param>
        /// <returns>if reference is same</returns>
        public override bool Equals(object obj)
        {
            bool isSame = false;
            try 
            {
                var objCast  = (NDArray) obj;
                
                if (objCast.Storage.GetData() == this.Storage.GetData())
                {
                    if (objCast.shape == this.shape)
                    {
                        isSame = true;
                    } 
                }
            }
            catch 
            {

            }
            return isSame;
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
            shapePuffer.ChangeTensorLayout(this.Storage.Shape.TensorLayout);

            puffer.Storage.Allocate(this.dtype, shapePuffer, this.Storage.TensorLayout );

            puffer.Storage.SetData(this.Storage.CloneData());

            return puffer;
        }
    }
}
