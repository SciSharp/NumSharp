using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Extensions;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray log()
        {
            var logArray = new NDArray(this.dtype,this.Storage.Shape);   

            Array dataSysArr = this.Storage.GetData();
            Array logDataSysArr = logArray.Storage.GetData();

            switch (logDataSysArr)
            {
                case double[] logData : 
                {
                    double[] npData = dataSysArr as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        logData[idx] = Math.Log(npData[idx]);
                    }
                    break;
                }
                case float[] logData : 
                {
                    double[] npData = dataSysArr as double[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        // boxing necessary because Math.log just for double
                        logData[idx] = (float) Math.Log(npData[idx]);
                    }
                    break;
                }
                case Complex[] logData : 
                {
                    Complex[] npData = dataSysArr as Complex[];

                    for (int idx = 0; idx < npData.Length;idx++)
                    {
                        logData[idx] = Complex.Log(  npData[idx]);
                    }
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
                
            }
            return logArray; 
        }
    }
    public partial class NDArrayGeneric<T> 
    {
        public NDArrayGeneric<T> log()
        {
            NDArrayGeneric<T> logArray = new NDArrayGeneric<T>();
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
