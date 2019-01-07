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

            if (this.ndim == 2)
            {
                output = this._ToMatrixString();
            }
            else
            {
                output = this._ToVectorString();
            }

            return output;
        }
        protected string _ToVectorString()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = new string[Storage.GetData().Length];

            Array strg = Storage.GetData();

            for (int idx = 0; idx < dataParsed.Length;idx++)
                dataParsed[idx] = _ParseNumber(strg.GetValue(idx),ref digitBefore, ref digitAfter);
            
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

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, strg.GetValue(idx)) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, strg.GetValue(strg.Length-1)) + "])");

            return returnValue;
        }
        protected string _ToMatrixString()
        {
            string returnValue = "array([[";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = new string[Storage.GetData().Length];

            Array strg = Storage.GetData();

            for (int idx = 0; idx < dataParsed.Length;idx++)
                dataParsed[idx] = _ParseNumber(strg.GetValue(idx),ref digitBefore, ref digitAfter);
            
            string elementFormatStart = "{0:";
            
            string elementFormatEnd = "";
            for(int idx = 0; idx < digitAfter;idx++)
                elementFormatEnd += "0";

            elementFormatEnd += "}";
            
            int missingDigits;
            string elementFormat;

            for (int idx = 0; idx < dataParsed.Length - 1; idx++)
            {
                missingDigits = digitBefore - dataParsed[idx].Replace(" ", "").Split('.')[0].Length;

                elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ', missingDigits).ToArray()) + "0." + elementFormatEnd;

                if (((idx + 1) % shape[1]) == 0)
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, strg.GetValue(idx)) + "],   \n       [");
                }
                else
                {
                    returnValue += (String.Format(new CultureInfo("en-us"), elementFormat, strg.GetValue(idx)) + ", ");
                }
            }

            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, strg.GetValue(strg.Length-1)) + "]])");

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

