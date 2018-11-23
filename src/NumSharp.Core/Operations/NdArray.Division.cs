using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;
using NumSharp.Core.Shared;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public static NDArray operator /(NDArray np1, NDArray np2)
        {
            var sum = new NDArray(np1.dtype, np1.Shape);
            
            switch (sum.Data())
            {
                case double[] sumArray : 
                {
                    double[] np1Array = np1.Data<double>();
                    double[] np2Array = np2.Data<double>();
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] / np2Array[idx];
                    break;
                }
                case float[] sumArray : 
                {
                    float[] np1Array = np1.Data<float>();
                    float[] np2Array = np2.Data<float>();
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] / np2Array[idx];
                    break; 
                }
                case Complex[] sumArray : 
                {
                    Complex[] np1Array = np1.Data<Complex>();
                    Complex[] np2Array = np2.Data<Complex>();
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] / np2Array[idx];
                    break; 
                }
                case Quaternion[] sumArray : 
                {
                    Quaternion[] np1Array = np1.Data<Quaternion>();
                    Quaternion[] np2Array = np2.Data<Quaternion>();
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] / np2Array[idx];
                    break; 
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + np1.dtype.Name);
                }
            }

            return sum;
        }

        public static NDArray operator /(NDArray np1, double scalar)
        {
            var sum = new NDArray(np1.dtype, np1.Shape);
            
            switch (np1.Data())
            {
                case double[] np1Array: 
                {
                    // for is faster than linq 
                    for (int idx = 0; idx < sum.Size;idx++)
                        sum[idx] = np1Array[idx] / scalar;
                    break;
                }
                case float[] np1Array: 
                {
                    // for is faster than linq 
                    for (int idx = 0; idx < sum.Size;idx++)
                        sum[idx] = np1Array[idx] / scalar;
                    break; 
                }
                case Complex[] np1Array: 
                {
                    // for is faster than linq 
                    for (int idx = 0; idx < sum.Size;idx++)
                        sum[idx] = np1Array[idx] / scalar;
                    break; 
                }
                /*case Quaternion[] np1Array: 
                {
                    // for is faster than linq 
                    for (int idx = 0; idx < sum.Size;idx++)
                        sum[idx] = np1Array[idx] / scalar;
                    break; 
                }*/
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + np1.dtype.Name);
                }
            }

            return sum;
        }
        
    }
}
