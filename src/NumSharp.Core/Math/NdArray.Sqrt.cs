using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray sqrt()
        {
            var sqrtArray = new NDArray(this.dtype,this.shape);

            Array dataSysArr = this.Storage.GetData();
            Array sqrtDataSysArray = sqrtArray.Storage.GetData();

            switch (dataSysArr)
            {
                case double[] data : 
                {
                    var sqrtData = sqrtDataSysArray as double[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sqrtData[idx] = Math.Sqrt(data[idx]);
                    
                    break;
                }
                case float[] data : 
                {
                    var sqrtData = sqrtDataSysArray as float[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sqrtData[idx] = (float) Math.Sqrt((double)data[idx]) ;
                    
                    break;
                }
                case Complex[] data : 
                {
                    var sqrtData = sqrtDataSysArray as Complex[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sqrtData[idx] = Complex.Sqrt(data[idx]);
                    
                    break;
                }
                case int[] data : 
                {
                    var sqrtData = sqrtDataSysArray as int[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sqrtData[idx] = (int) Math.Sqrt((double)data[idx]); 
                    
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }
            return sqrtArray;
        }

    }
}