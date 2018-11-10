using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
{
    public partial class NDArray<T> 
    {
        public NDArray<T> log()
        {
            NDArray<T> logArray = new NDArray<T>();
            logArray.Data = new T[this.Data.Length];
            logArray.Shape = new Shape(this.Shape.Shapes);

            switch (logArray.Data)
            {
                case double[] logData : 
                {
                    double[] npData = this.Data as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        logData[idx] = Math.Log(npData[idx]);
                    }
                    break;
                }
                case float[] logData : 
                {
                    double[] npData = this.Data as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        // boxing necessary because Math.Sin just for double
                        logData[idx] = (float) Math.Log(npData[idx]);
                    }
                    break;
                }
                case Complex[] logData : 
                {
                    Complex[] npData = this.Data as Complex[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        // boxing necessary because Math.Sin just for double
                        logData[idx] = Complex.Log(  npData[idx]);
                    }
                    break;
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the");
                }
                
            }
            return logArray;
        }
    }
}
