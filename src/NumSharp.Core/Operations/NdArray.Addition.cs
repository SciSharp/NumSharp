using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Numerics;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public static NDArray operator +(NDArray np1, NDArray np2)
        {
            var sum = new NDArray(np1.dtype, np1.Shape);

            switch (np1.dtype.Name)
            {
                case "Double":
                    {
                        double[] np1Array = np1.Data<double>();
                        double[] np2Array = np2.Data<double>();
                        // for is faster than linq 
                        for (int idx = 0; idx < sum.Size; idx++)
                            sum[idx] = np1Array[idx] + np2Array[idx];
                        break;
                    }
                /*case float[] sumArray:
                    {
                        float[] np1Array = np1.Data<float>();
                        float[] np2Array = np2.Data<float>();
                        // for is faster than linq 
                        for (int idx = 0; idx < sumArray.Length; idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                        break;
                    }
                case Complex[] sumArray:
                    {
                        Complex[] np1Array = np1.Data<Complex>();
                        Complex[] np2Array = np2.Data<Complex>();
                        // for is faster than linq 
                        for (int idx = 0; idx < sumArray.Length; idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                        break;
                    }
                case Quaternion[] sumArray:
                    {
                        Quaternion[] np1Array = np1.Data<Quaternion>();
                        Quaternion[] np2Array = np2.Data<Quaternion>();
                        // for is faster than linq 
                        for (int idx = 0; idx < sumArray.Length; idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                        break;
                    }*/
                default:
                    {
                        throw new Exception("The operation is not implemented for the " + np1.dtype.Name);
                    }
            }

            return sum;
        }

        public static NDArray operator +(NDArray np1, double scalar)
        {
            var sum = new NDArray(typeof(double), np1.Shape);

            // for is faster than linq 
            for (int idx = 0; idx < sum.Size; idx++)
            {
                switch (np1[idx])
                {
                    case int d:
                        sum[idx] = d + scalar;
                        break;
                    case double d:
                        sum[idx] = d + scalar;
                        break;
                }
            }

            return sum;
        }

        public static NDArray operator +(double scalar, NDArray np1)
        {
            return np1 + scalar;
        }
    }

    public partial class NDArrayGeneric<T>
    {
        public static NDArrayGeneric<T> operator +(NDArrayGeneric<T> np1, NDArrayGeneric<T> np2)
        {
            NDArrayGeneric<T> sum = new NDArrayGeneric<T>();
            sum.Shape = np1.Shape;
            sum.Data = new T[np1.Data.Length];
            
            switch (sum.Data)
            {
                case double[] sumArray : 
                {
                    double[] np1Array = np1.Data as double[];
                    double[] np2Array = np2.Data as double[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + np2Array[idx];
                    break;
                }
                case float[] sumArray : 
                {
                    float[] np1Array = np1.Data as float[];
                    float[] np2Array = np2.Data as float[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + np2Array[idx];
                    break; 
                }
                case Complex[] sumArray : 
                {
                    Complex[] np1Array = np1.Data as Complex[];
                    Complex[] np2Array = np2.Data as Complex[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + np2Array[idx];
                    break; 
                }
                case Quaternion[] sumArray : 
                {
                    Quaternion[] np1Array = np1.Data as Quaternion[];
                    Quaternion[] np2Array = np2.Data as Quaternion[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + np2Array[idx];
                    break; 
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + typeof(T).Name);
                }
            }

            return (NDArrayGeneric<T>) sum;
        }
        public static NDArrayGeneric<T> operator +(NDArrayGeneric<T> np1, T scalar)
        {
            NDArrayGeneric<T> sum = new NDArrayGeneric<T>();
            sum.Shape = np1.Shape;
            sum.Data = new T[np1.Data.Length];
            
            switch (scalar)
            {
                case double scalarDouble : 
                {
                    double[] np1Array = np1.Data as double[];
                    double[] sumArray = sum.Data as double[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + scalarDouble;
                    break;
                }
                case float scalarFloat : 
                {
                    float[] np1Array = np1.Data as float[];
                    float[] sumArray = sum.Data as float[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + scalarFloat;
                    break; 
                }
                case Complex scalarComplex : 
                {
                    Complex[] np1Array = np1.Data as Complex[];
                    Complex[] sumArray = sum.Data as Complex[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + scalarComplex;
                    break; 
                }
                case Quaternion scalarQuaternion : 
                {
                    Quaternion[] np1Array = np1.Data as Quaternion[];
                    Quaternion[] sumArray = sum.Data as Quaternion[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] + scalarQuaternion;
                    break; 
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + typeof(T).Name);
                }
            }

            return (NDArrayGeneric<T>) sum;
        }
        
    }
}
