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

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArray<T>
    {
        /// <summary>
        /// 1 dim array data storage
        /// </summary>
        public T[] Data { get; set; }

        private IList<int> shape;

        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public IList<int> Shape
        {
            get
            {
                return shape;
            }

            set
            {
                shape = value;
                dimOffset = new List<int> { 1 };

                for (int s = Shape.Count-1; s >= 1; s--)
                {
                    dimOffset.Add(dimOffset[Shape.Count - 1 - s] * shape[s]);
                }
                dimOffset = dimOffset.Reverse().ToList();
            }
        }

        /// <summary>
        /// Speed up index accessor
        /// </summary>
        private IList<int> dimOffset { get; set; }

        /// <summary>
        /// Dimension count
        /// </summary>
        public int NDim { get { return Shape.Count; } }

        /// <summary>
        /// Total of elements
        /// </summary>
        public int Size { get { return Data.Length; } }

        /// <summary>
        /// Random reference
        /// </summary>
        public NDArrayRandom Random { get; set; }

        public NDArray()
        {
            // set default shape as 1 dim and 0 elements.
            Shape = new List<int>() { 0 };
            Data = new T[] { };
            Random = new NDArrayRandom();
        }

        /// <summary>
        /// Index accessor
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public T this[params int[] select]
        {
            get
            {
                return Data[GetIndexInShape(select)];
            }

            set
            {
                Data[GetIndexInShape(select)] = value;
            }
        }

        public NDArray<T> Vector(params int[] select)
        {
            if (select.Length == NDim)
            {
                throw new Exception("Please use NDArray[m, n] to access element.");
            }
            else
            {
                int start = GetIndexInShape(select);
                int length = dimOffset[select.Length - 1];

                var n = new NDArray<T>();
                n.Data = Data.Skip(start).Take(length).ToArray();
                n.Shape = shape.Skip(select.Length).ToList();

                return n;
            }
        }

        /// <summary>
        /// Filter specific elements through select.
        /// </summary>
        /// <param name="select"></param>
        /// <returns>Return a new NDArray with filterd elements.</returns>
        public NDArray<T> this[IEnumerable<int> select]
        {
            get
            {
                int i = 0;
                var array = Data.Where(x => select.Contains(i++)).ToList();
                return new NDArray<T>().Array(array);
            }
        }

        /// <summary>
        /// Overload
        /// </summary>
        /// <param name="select"></param>
        /// <returns></returns>
        public NDArray<T> this[NDArray<int> select]
        {
            get
            {
                int i = 0;
                var array = Data.Where(x => select.Data.Contains(i++)).ToList();
                return new NDArray<T>().Array(array);
            }
        }

        private int GetIndexInShape(params int[] select)
        {
            int idx = 0;
            for (int i = 0; i < select.Length; i++)
            {
                idx += dimOffset[i] * select[i];
            }

            return idx;
        }

        public override string ToString()
        {
            string output = "";

            if (this.NDim == 2)
            {
                output = this._ToMatrixString();
            }
            else
            {
                output = "array([";

                // loop
                for (int r = 0; r < Data.Length; r++)
                {
                    output += (r == 0) ? Data[r] + "" : ", " + Data[r];
                }

                output += "])";
            }

            return output;
        }

        public override bool Equals(object obj)
        {
            return Data[0].Equals(obj);
        }

        public static bool operator ==(NDArray<T> np, object obj)
        {
            return np.Data[0].Equals(obj);
        }

        public static bool operator !=(NDArray<T> np, object obj)
        {
            return np.Data[0].Equals(obj);
        }
        protected string _ToMatrixString()
        {
            string returnValue = "array([[";

            int dim0 = Shape[0];
            int dim1 = Shape[1];

            for (int idx = 0; idx < (dim0-1);idx++)
            {
                for (int jdx = 0;jdx < (dim1-1);jdx++)
                {
                    returnValue += (this[idx,jdx] + ", ");
                }
                returnValue += (this[idx,dim1-1] + "],   \n       [");
            }
            for (int jdx = 0; jdx < (dim1-1);jdx++)
            {
                returnValue += (this[dim0-1,jdx] + ", ");
            }
            returnValue += (this[dim0-1,dim1-1] + "]])");

            return returnValue;    
        }
    }
}
