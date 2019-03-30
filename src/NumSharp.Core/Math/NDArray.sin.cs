using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

namespace NumSharp
{
    public partial class NDArray
    {
        public NDArray sin()
        {
            var sinArray = new NDArray(this.dtype,this.shape);

            Array dataSysArr = this.Storage.GetData();
            Array sinDataSysArr = sinArray.Storage.GetData();

            switch (dataSysArr)
            {
                case double[] data : 
                {
                    var sinData = sinDataSysArr as double[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sinData[idx] = Math.Sin(data[idx]);
                    
                    break;
                }
                case float[] data : 
                {
                    var sinData = sinDataSysArr as float[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sinData[idx] = (float) Math.Sin((double)data[idx]) ;
                    
                    break;
                }
                case Complex[] data : 
                {
                    var sinData = sinDataSysArr as Complex[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sinData[idx] = Complex.Sin(data[idx]);
                    
                    break;
                }
                case int[] data : 
                {
                    var sinData = sinDataSysArr as int[];

                    for (int idx = 0; idx < data.Length;idx++)
                        sinData[idx] = (int) Math.Sin((double)data[idx]); 
                    
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
                
            }
            return sinArray;
        }
    }
}
