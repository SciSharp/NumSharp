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
        public NDArray power(ValueType exponent)
        {
            var powerArray = new NDArray(this.dtype,this.Shape);

            Array dataSysArr = this.Storage.GetData();
            Array powerDataSysArr = powerArray.Storage.GetData();

            switch (dataSysArr)
            {
                case double[] data : 
                {
                    var powerData = powerDataSysArr as double[];

                    for (int idx = 0; idx < data.Length;idx++)
                        powerData[idx] = Math.Pow(data[idx], (double)exponent);
                    
                    break;
                }
                case float[] data : 
                {
                    var powerData = powerDataSysArr as float[];

                    for (int idx = 0; idx < data.Length;idx++)
                        powerData[idx] = (float) Math.Pow((double)data[idx], (double)exponent);
                    
                    break;
                }
                case Complex[] data : 
                {
                    var powerData = powerDataSysArr as Complex[];

                    for (int idx = 0; idx < data.Length;idx++)
                        powerData[idx] = Complex.Pow(data[idx],(double)exponent);
                    
                    break;
                }
                case int[] data : 
                {
                    var powerData = powerDataSysArr as int[];

                    for (int idx = 0; idx < data.Length;idx++)
                        powerData[idx] = (int) Math.Pow((double)data[idx], (double)exponent);
                    
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
                
            }
            return powerArray;
        }
    }
    public partial class NDArrayGeneric<T> 
    {
        public NDArrayGeneric<T> power(T exponent)
        {
            NDArrayGeneric<T> sinArray = new NDArrayGeneric<T>();
            sinArray.Data = new T[this.Size];
            sinArray.Shape = new Shape(this.Shape.Shapes);

            switch (Data)
            {
                case double[] sinData : 
                {
                    for (int idx = 0; idx < sinData.Length;idx++)
                    {
                            sinArray[idx] = (T)(object)Math.Pow(sinData[idx], (double)(object)exponent);
                    }
                    break;
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the");
                }
                
            }
            return sinArray;
        }
    }
}
