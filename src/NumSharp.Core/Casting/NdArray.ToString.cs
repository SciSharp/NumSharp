using System;
using NumSharp;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;

namespace NumSharp
{
    public partial class NDArray
    {
        public override string ToString()
        {
            string output = "";

            if (ndim == 0)
                output = Storage.GetData().GetValue(0).ToString();
            else if (ndim == 2)
                output = _ToMatrixString();
            else
                output = _ToVectorString();

            return output;
        }
        protected string _ToVectorString()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = new string[size];

            Array a = null;
            switch (Type.GetTypeCode(dtype))
            {
                case TypeCode.Int32:
                    a = Data<int>();
                    break;
                case TypeCode.Int64:
                    a = Data<long>();
                    break;
                case TypeCode.Single:
                    a = Data<float>();
                    break;
                case TypeCode.Double:
                    a = Data<double>();
                    break;
                default:
                    break;
            }

            for (int idx = 0; idx < dataParsed.Length;idx++)
                dataParsed[idx] = _ParseNumber(a.GetValue(idx),ref digitBefore, ref digitAfter);
            
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

                returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, a.GetValue(idx)) + ", ");
            }
            missingDigits =  digitBefore - dataParsed.Last().Replace(" ","").Split('.')[0].Length;
                
            elementFormat = elementFormatStart + new string(Enumerable.Repeat<char>(' ',missingDigits).ToArray()) + "." + elementFormatEnd; 

            returnValue += (String.Format(new CultureInfo("en-us"),elementFormat, a.GetValue(a.Length-1)) + "])");

            return returnValue;
        }
        protected string _ToMatrixString()
        {
            string returnValue = "";

            switch (dtype.Name)
            {
                case "Int32":
                    {
                        string c1n = "[";
                        for (int c1 = 0; c1 < shape[0]; c1++)
                        {
                            string c2n = "[";
                            for (int c2 = 0; c2 < shape[1]; c2++)
                                c2n += (c2 == 0 ? "" : " ") + this[c1, c2].ToString();
                            c2n += "]";

                            c1n += (c1 > 0 && c1 < shape[0] ? "\r\n" : "") + c2n;
                        }
                        returnValue += c1n + "]";
                    }
                    break;

                case "Single":
                    {
                        string c1n = "[";
                        for (int c1 = 0; c1 < shape[0]; c1++)
                        {
                            string c2n = "[";
                            for (int c2 = 0; c2 < shape[1]; c2++)
                                c2n += (c2 == 0 ? "" : " ") + this[c1, c2].ToString();
                            c2n += "]";

                            c1n += (c1 > 0 && c1 < shape[0] ? "\r\n" : "") + c2n;
                        }
                        returnValue += c1n + "]";
                    }
                    break;

                case "Double":
                    {
                        string c1n = "[";
                        for (int c1 = 0; c1 < shape[0]; c1++)
                        {
                            string c2n = "[";
                            for (int c2 = 0; c2 < shape[1]; c2++)
                                c2n += (c2 == 0 ? "" : " ") + this[c1, c2].ToString();
                            c2n += "]";

                            c1n += (c1 > 0 && c1 < shape[0] ? "\r\n" : "") + c2n;
                        }
                        returnValue += c1n + "]";
                    }
                    break;
            }

            return returnValue + "";    
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

