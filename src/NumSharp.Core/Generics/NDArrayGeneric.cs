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
using NumSharp.Core;

namespace NumSharp.Generic
{
    public class NDArray<T> : NumSharp.Core.NDArray where T : struct
    {
        public NDArray()
        {
            Storage.dtype = typeof(T);
            Storage = NDStorage.CreateByShapeAndType(this.dtype,new Shape(1));
        }
        public NDArray(Shape shape) : this()
        {
            Storage = NDStorage.CreateByShapeAndType(Storage.dtype, shape);
        }
        public T this[params int[] select]
        {
            get
            {
                return (T)Storage[shape.GetIndexInShape(select)];
            }
            set
            {
                if (select.Length == ndim)
                {
                    Storage[shape.GetIndexInShape(select)] = value;
                }
                else
                {

                }
            }
        }
    }
}
 

namespace NumSharp.Core
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    /// <typeparam name="T">dtype</typeparam>
    [Obsolete("please use NDArray<T>")]
    public partial class NDArrayGeneric<T>
    {
        /// <summary>
        /// 1 dim array data storage
        /// </summary>
        public T[] Data { get; set; }

        private Shape shape;
        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public Shape Shape
        {
            get
            {
                return shape;
            }
            set
            {
                shape = value;
            }
        }

        /// <summary>
        /// Dimension count
        /// </summary>
        public int NDim => Shape.NDim;

        /// <summary>
        /// Total of elements
        /// </summary>
        public int Size => Shape.Size;

        public NDArrayGeneric()
        {
            // set default shape as 1 dim and 0 elements.
            Shape = new Shape(new int[] { 0 });
        }

        public void Set(Shape shape, T value)
        {
            if (shape.NDim == NDim)
            {
                throw new Exception("Please use NDArray[m, n] to access element.");
            }
            else
            {
                int start = GetIndexInShape(shape.Shapes.ToArray());
                int length = Shape.DimOffset[shape.NDim - 1];

                Span<T> data = Data;
                var elements = data.Slice(start, length);

                for (int i = 0; i < elements.Length; i++)
                {
                    elements[i] = value;
                }
            }
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
                output = this._ToVectorString();
            }

            return output;
        }

        public override bool Equals(object obj)
        {
            return Data[0].Equals(obj);
        }

        public static bool operator ==(NDArrayGeneric<T> np, object obj)
        {
            return np.Data[0].Equals(obj);
        }

        public static bool operator !=(NDArrayGeneric<T> np, object obj)
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
                case 1 : dotNetArray = new T[this.Shape.Shapes[0]].ToArray();break;
                case 2 : dotNetArray = new T[this.Shape.Shapes[0]][].Select(x => new T[this.Shape.Shapes[1]].ToArray()).ToArray();break;
                case 3 : dotNetArray = new T[this.Shape.Shapes[0]][][].Select(x => new T[this.Shape.Shapes[1]][].Select(y => new T[this.Shape.Shapes[2]].ToArray().ToArray()).ToArray()).ToArray();break;
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
                    for(int idx = 0; idx < this.Shape.Shapes[0];idx++)
                    {
                        for(int jdx = 0; jdx < this.Shape.Shapes[1];jdx++)
                        {
                            dotNetArray[idx][jdx] = this[idx,jdx];
                        }
                    }
                    break;
                }
                case 3 : 
                {
                    for(int idx = 0; idx < this.Shape.Shapes[0];idx++)
                    {
                        for(int jdx = 0; jdx < this.Shape.Shapes[1];jdx++)
                        {
                            for(int kdx = 0; kdx < this.Shape.Shapes[2];kdx++)
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
        protected string _ToVectorString()
        {
            string returnValue = "array([";

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

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data[idx]) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data.Last()) + "])");

            return returnValue;
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

                if( ((idx+1) % Shape.Shapes[1] ) == 0 )
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
            
            parsed = (parsed.StartsWith("-")) ? parsed : (" " + parsed); 

            int noBefore_local = parsed.Split('.')[0].Length;
            int noAfter_local = parsed.Split('.')[1].ToCharArray().Reverse().SkipWhile(x => x == '0').ToArray().Length;

            noBefore = (noBefore_local > noBefore) ? noBefore_local : noBefore;
            noAfter  = (noAfter_local  > noAfter ) ? noAfter_local  : noAfter;

            return parsed;
        }
    }
}
