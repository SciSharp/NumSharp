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
                    case "Single":
                        output = Data<float>()[0].ToString();
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
                output = _ToMatrixString();
            }
            else
            {
                output = _ToVectorString();
            }

            return output;
        }
        protected string _ToVectorString()
        {
            string returnValue = "array([";

            int digitBefore = 0;
            int digitAfter = 0;

            string[] dataParsed = new string[size];

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
                                c2n += (c2 == 0 ? "" : " ") + Data<int>(c1, c2).ToString();
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
                                c2n += (c2 == 0 ? "" : " ") + Data<float>(c1, c2).ToString();
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
                                c2n += (c2 == 0 ? "" : " ") + Data<double>(c1, c2).ToString();
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

