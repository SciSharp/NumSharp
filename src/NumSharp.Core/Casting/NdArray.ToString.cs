using System;
using NumSharp.Core;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public override string ToString()
        {
            string output = "";
            if (this.ndim == 0)
            {
                switch (dtype.Name)
                {
                    case "Int16":
                        output = Data<short>()[0].ToString();
                        break;
                    case "Int32":
                        output = Data<int>()[0].ToString();
                        break;
                    case "Double":
                        output = Data<double>()[0].ToString();
                        break;
                    case "String":
                        output = Data<string>()[0].ToString();
                        break;
                    default:
                        throw new NotImplementedException("NDArray ToString()");
                }
            }
            else if (this.ndim == 2)
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
        protected string _ToVectorString<T>()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            var dataParsed = Storage.GetData<T>().Select(x => _ParseNumber(x,ref digitBefore,ref digitAfter)).ToArray();

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

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.GetData<T>()[idx]) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.GetData<T>().Last()) + "])");

            return returnValue;
        }
        protected string _ToMatrixString<T>()
        {
            string returnValue = "array([[";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = Storage.GetData<T>().Select(x => _ParseNumber(x, ref digitBefore, ref digitAfter)).ToArray();

            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < this.ndim - 1; idx++)
            {
                missingDigits = digitBefore - dataParsed[idx].Replace(" ", "").Split('.')[0].Length;

                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ', missingDigits).ToArray()) + "0." + elementFormatEnd;

                if (((idx + 1) % shape[1]) == 0)
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, Storage.GetData<T>()[idx]) + "],   \n       [");
                }
                else
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, Storage.GetData<T>()[idx]) + ", ");
                }
            }

            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, Storage.GetData<T>().Last()) + "]])");

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

