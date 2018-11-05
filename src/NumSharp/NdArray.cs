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
                //n.Data = Data.Skip(start).Take(length).ToArray();
                //n.Shape = shape.Skip(select.Length).ToList();
                n.Data = new Span<T>(Data, start, length).ToArray();
                // Since n.Shape is a IList it cannot be converted to Span<T>
                // This is a lot of hoops to jump throught to get it into a span
                // shape.Skip(select.Length).ToList() may be more efficient - not sure
                n.Shape = shape.ToArray().AsSpan().Slice(select.Length).ToArray().ToList();
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

                var n = new NDArray<T>();
                n.Data = Data.Where(x => select.Contains(i++)).ToArray();
                n.Shape[0] = n.Data.Length;

                return n;
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

                var n = new NDArray<T>();
                n.Data = Data.Where(x => select.Data.Contains(i++)).ToArray();
                n.Shape = shape;
                n.Shape[0] = select.shape[0];

                return n;
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

        public override int GetHashCode()
        {
            unchecked
            {
                var result = 1337;
                result = (result * 397) ^ this.NDim;
                result = (result * 397) ^ this.Size;
                return result;
            }
        }

        public TCast ToDotNetArray<TCast>()
        {
            dynamic dotNetArray = null;
            switch (this.NDim)
            {
                case 1 : dotNetArray = new T[this.Shape[0]].ToArray();break;
                case 2 : dotNetArray = new T[this.Shape[0]][].Select(x => new T[this.Shape[1]].ToArray()).ToArray();break;
                case 3 : dotNetArray = new T[this.Shape[0]][][].Select(x => new T[this.Shape[1]][].Select(y => new T[this.Shape[2]].ToArray().ToArray()).ToArray()).ToArray();break;
            }

            switch (this.NDim)
            {
                case 1 : 
                {
                    dotNetArray = this.Data.ToArray();
                    break;
                }
                case 2 : 
                {
                    for(int idx = 0; idx < this.Shape[0];idx++)
                    {
                        for(int jdx = 0; jdx < this.Shape[1];jdx++)
                        {
                            dotNetArray[idx][jdx] = this[idx,jdx];
                        }
                    }
                    break;
                }
                case 3 : 
                {
                    for(int idx = 0; idx < this.Shape[0];idx++)
                    {
                        for(int jdx = 0; jdx < this.Shape[1];jdx++)
                        {
                            for(int kdx = 0; kdx < this.Shape[2];kdx++)
                            {
                                dotNetArray[idx][jdx][kdx] = this[idx,jdx,kdx];
                            }
                        }
                    }
                    break;
                }
            }
            TCast castedDotNetArray = (TCast)dotNetArray;
            return castedDotNetArray;
        }
        protected string _ToMatrixString()
        {
            string returnValue = "array([[";

            int digitBefore = 0;
            int digitAfter = 0;

           var dataParsed =  Data.Select(x => _ParseNumber(x,ref digitBefore,ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < (Data.Length-1);idx++)
            {   
                missingDigits =  digitBefore - dataParsed[idx].Replace(" ","").Split('.')[0].Length;
                
                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "0." + elementFormatEnd; 

                if( ((idx+1) % Shape[1] ) == 0 )
                {
                    returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data[idx]) + "],   \n       [");    
                }
                else 
                {
                    returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data[idx]) + ", ");
                }
                
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data.Last()) + "]])");

            return returnValue;    
        }
        protected string _ParseNumber(T number, ref int  noBefore,ref int noAfter)
        {
            string parsed = string.Format(new CultureInfo("en-us"),"{0:0.00000000}",number);
            
            parsed = (parsed.StartsWith('-')) ? parsed : (" " + parsed); 

            int noBefore_local = parsed.Split('.')[0].Length;
            int noAfter_local = parsed.Split('.')[1].ToCharArray().Reverse().SkipWhile(x => x == '0').ToArray().Length;

            noBefore = (noBefore_local > noBefore) ? noBefore_local : noBefore;
            noAfter  = (noAfter_local  > noAfter ) ? noAfter_local  : noAfter;

            return parsed;
        }
    }
}
