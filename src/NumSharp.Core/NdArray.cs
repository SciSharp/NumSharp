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

namespace NumSharp.Core
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArray
    {
        public Type dtype { get; set; }

        public NDStorage Storage { get; set; }

        public T[] Data<T>() => Storage.Data<T>();

        public T Data<T>(params int[] shape) => Storage.Data<T>()[Shape.GetIndexInShape(shape)];

        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public Shape Shape { get; set; }

        /// <summary>
        /// Dimension count
        /// </summary>
        public int NDim => Shape.Length;

        /// <summary>
        /// Total of elements
        /// </summary>
        public int Size => Shape.Size;
        public NDArray()
        {

        }
        public NDArray(Type dtype)
        {
            this.dtype = dtype;

            // set default shape as 1 dim and 0 elements.
            Shape = new Shape(new int[] { 0 });
            Storage = new NDStorage(dtype);
        }

        public void Set<T>(T[] data)
        {
            Storage.Values = data;
        }

        public void Set<T>(Shape shape, T value)
        {
            if (shape.Length == NDim)
            {
                throw new Exception("Please use NDArray[m, n] to access element.");
            }
            else
            {
                int start = Shape.GetIndexInShape(shape.Shapes.ToArray());
                int length = Shape.DimOffset[shape.Length - 1];

                Span<T> data = Data<T>();
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
                if(dtype == typeof(int))
                {
                    output = this._ToMatrixString<int>();
                }
                else if(dtype == typeof(double))
                {
                    output = this._ToMatrixString<double>();
                }
                
            }
            else
            {
                if (dtype == typeof(int))
                {
                    output = this._ToVectorString<int>();
                }
                else if (dtype == typeof(double))
                {
                    output = this._ToVectorString<double>();
                }
            }

            return output;
        }

        public override bool Equals(object obj)
        {
            switch (obj)
            {
                case int o:
                    return o == Data<int>()[0];
            }

            return false;
        }

        public static bool operator ==(NDArray np, object obj)
        {
            switch (obj)
            {
                case int o:
                    return o == np.Data<int>()[0];
            }

            return false;
        }

        public static bool operator !=(NDArray np, object obj)
        {
            return !(np == obj);
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
        
        protected string _ToVectorString<T>()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            var dataParsed = Data<T>().Select(x => _ParseNumber(x,ref digitBefore,ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < (Storage.Shape.Size-1);idx++)
            {   
                missingDigits =  digitBefore - dataParsed[idx].Replace(" ","").Split('.')[0].Length;
                
                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "0." + elementFormatEnd; 

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage[idx]) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Data<T>().Last()) + "])");

            return returnValue;
        }
        protected string _ToMatrixString<T>()
        {
            string returnValue = "array([[";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = Data<T>().Select(x => _ParseNumber(x, ref digitBefore, ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < Shape.Length - 1; idx++)
            {
                missingDigits = digitBefore - dataParsed[idx].Replace(" ", "").Split('.')[0].Length;

                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ', missingDigits).ToArray()) + "0." + elementFormatEnd;

                if (((idx + 1) % Shape.Shapes[1]) == 0)
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, Storage[idx]) + "],   \n       [");
                }
                else
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, Storage[idx]) + ", ");
                }
            }

            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Data<T>().Last()) + "]])");

            return returnValue;    
        }
        protected string _ParseNumber(object number, ref int  noBefore,ref int noAfter)
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
