using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Extensions;

namespace NumSharp
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
                    float[] npData = dataSysArr as float[];

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
}
