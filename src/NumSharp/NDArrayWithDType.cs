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

namespace NumSharp
{
    /// <summary>
    /// A powerful N-dimensional array object
    /// Inspired from https://www.numpy.org/devdocs/user/quickstart.html
    /// </summary>
    public partial class NDArrayWithDType
    {
        public static Type int16 = typeof(short);
        public static Type double8 = typeof(double);
        public static Type decimal16 = typeof(decimal);

        public Type dtype { get; set; }

        public NDStorage Storage { get; set; }

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
        public int NDim => Shape.Length;

        /// <summary>
        /// Total of elements
        /// </summary>
        public int Size => Shape.Size;

        public NDArrayWithDType(Type dtype)
        {
            this.dtype = dtype;

            // set default shape as 1 dim and 0 elements.
            Shape = new Shape(new int[] { 0 });
            Storage = new NDStorage(this.dtype);
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
            return Storage.Int32[0].Equals(obj);
        }

        public static bool operator ==(NDArrayWithDType np, object obj)
        {
            return np.Storage.Int32[0].Equals(obj);
        }

        public static bool operator !=(NDArrayWithDType np, object obj)
        {
            return np.Storage.Int32[0].Equals(obj);
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

        protected string _ToVectorString()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            var dataParsed = Storage.Int32.Select(x => _ParseNumber(x,ref digitBefore,ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < (Storage.Int32.Length-1);idx++)
            {   
                missingDigits =  digitBefore - dataParsed[idx].Replace(" ","").Split('.')[0].Length;
                
                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "0." + elementFormatEnd; 

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Int32[idx]) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Int32.Last()) + "])");

            return returnValue;
        }
        protected string _ToMatrixString()
        {
            string returnValue = "array([[";

            int digitBefore = 0;
            int digitAfter = 0;

            var dataParsed = Storage.Int32.Select(x => _ParseNumber(x,ref digitBefore,ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < (Storage.Int32.Length-1);idx++)
            {   
                missingDigits =  digitBefore - dataParsed[idx].Replace(" ","").Split('.')[0].Length;
                
                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "0." + elementFormatEnd; 

                if( ((idx+1) % Shape.Shapes[1] ) == 0 )
                {
                    returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Int32[idx]) + "],   \n       [");    
                }
                else 
                {
                    returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Int32[idx]) + ", ");
                }
                
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.Int32.Last()) + "]])");

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
