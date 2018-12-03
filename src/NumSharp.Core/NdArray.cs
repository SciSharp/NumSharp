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
        /// <summary>
        /// Data type of NDArray
        /// </summary>
        public Type dtype => Storage.dtype;
        /// <summary>
        /// Data length of every dimension
        /// </summary>
        public Shape shape => Storage.Shape;
        /// <summary>
        /// Dimension count
        /// </summary>
        public int ndim => shape.NDim;
        /// <summary>
        /// Total of elements
        /// </summary>
        public int size => shape.Size;
        /// <summary>
        /// The internal storage for elements of NDArray
        /// </summary>
        /// <value></value>
        public NDStorage Storage { get; set; }
        public NDArray()
        {
            Storage = NDStorage.CreateByShapeAndType(typeof(double),new Shape(1));
        }
        public NDArray(Type dtype)
        {
            Storage = NDStorage.CreateByShapeAndType(dtype,new Shape(1));
        }
        public NDArray(Type dtype, Shape shape)
        {
            Storage = NDStorage.CreateByShapeAndType(dtype,shape);
        }
        public NDArray(Type dtype, params int[] shapes)
        {
            Storage = NDStorage.CreateByShapeAndType(dtype,new Shape(shapes));
        }
        public override string ToString()
        {
            string output = "";

            if (this.ndim == 2)
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
                    //output = this._ToVectorString<int>();
                }
                else if (dtype == typeof(double))
                {
                    //output = this._ToVectorString<double>();
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

            for (int idx = 0; idx < shape.NDim - 1; idx++)
            {
                missingDigits = digitBefore - dataParsed[idx].Replace(" ", "").Split('.')[0].Length;

                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ', missingDigits).ToArray()) + "0." + elementFormatEnd;

                if (((idx + 1) % shape.Shapes[1]) == 0)
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
