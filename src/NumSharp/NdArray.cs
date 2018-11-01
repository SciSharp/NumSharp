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
    public class NDArray<T>
    {
        public T[] Data { get; set; }

        private IList<int> shape;
        public IList<int> Shape
        {
            get
            {
                return shape;
            }

            set
            {
                shape = value;
                ShapeOffset = new List<int> { 1 };

                for (int s = Shape.Count-1; s >= 1; s--)
                {
                    ShapeOffset.Add(ShapeOffset[Shape.Count - 1 - s] * shape[s]);
                }
                ShapeOffset = ShapeOffset.Reverse().ToList();
            }
        }

        public IList<int> ShapeOffset { get; set; }

        public int NDim { get { return Shape.Count; } }

        public int Size { get { return Data.Length; } }

        public int Length { get { return Shape[0]; } }

        public NDArrayRandom Random { get; set; }

        public NDArray()
        {
            // default as 1 dim
            Shape = new List<int>() { 0 };
            ShapeOffset = new List<int> { 0 };
            Data = new T[] { };
            Random = new NDArrayRandom();
        }

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

        public NDArray<T> this[IEnumerable<int> select]
        {
            get
            {
                int i = 0;
                var array = Data.Where(x => select.Contains(i++)).ToList();
                return new NDArray<T>().Array(array);
            }
        }

        public NDArray<T> this[NDArray<int> select]
        {
            get
            {
                int i = 0;
                var array = Data.Where(x => select.Data.Contains(i++)).ToList();
                return new NDArray<T>().Array(array);
            }
        }

        public NDArray<T> Get(params int[] select)
        {
            return new NDArray<T>();
        }

        private int GetIndexInShape(params int[] select)
        {
            int idx = 0;
            for (int i = 0; i < select.Length; i++)
            {
                idx += ShapeOffset[i] * select[i];
            }

            return idx;
        }

        public override string ToString()
        {
            string output = "array([";

            // loop
            for (int r = 0; r < Data.Length; r++)
            {
                output += (r == 0) ? Data[r] + "" : ", " + Data[r];
            }

            output += "])";

            return output;
        }
    }
}
