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
       public static NDArray operator *(NDArray np1, NDArray np2)
        {
            NDArray sum = new NDArray(np1.dtype,np1.Storage.Shape);
            
            if (!Enumerable.SequenceEqual(np1.Storage.Shape.Shapes,np2.Storage.Shape.Shapes))
                throw new IncorrectShapeException();

            Array np1SysArr = np1.Storage.GetData();
            Array np2SysArr = np2.Storage.GetData();
            Array np3SysArr = sum.Storage.GetData();

            switch (np3SysArr)
            {
                case double[] sumArray : 
                {
                    double[] np1Array = np1SysArr as double[];
                    double[] np2Array = np2SysArr as double[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break;
                }
                case float[] sumArray : 
                {
                    float[] np1Array = np1SysArr as float[];
                    float[] np2Array = np2SysArr as float[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case int[] sumArray : 
                {
                    int[] np1Array = np1SysArr as int[];
                    int[] np2Array = np2SysArr as int[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case Int64[] sumArray : 
                {
                    Int64[] np1Array = np1SysArr as Int64[];
                    Int64[] np2Array = np2SysArr as Int64[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case Complex[] sumArray : 
                {
                    Complex[] np1Array = np1SysArr as Complex[];
                    Complex[] np2Array = np2SysArr as Complex[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case Quaternion[] sumArray : 
                {
                    Quaternion[] np1Array = np1SysArr as Quaternion[];
                    Quaternion[] np2Array = np2SysArr as Quaternion[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }

            return sum;
        }
        public static NDArray operator *(NDArray np1, ValueType scalar)
        {
            NDArray sum = new NDArray(np1.dtype,np1.shape);
            
            Array np1SysArr = np1.Storage.GetData();
            Array sumSysArr = sum.Storage.GetData();

            switch (sumSysArr)
            {
                case double[] sumArr : 
                {
                    double scalar_ = Convert.ToDouble(scalar);
                    double[] np1Array = np1SysArr as double[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;

                    break;
                }
                case float[] sumArr : 
                {
                    float scalar_ = Convert.ToSingle(scalar);
                    float[] np1Array = np1SysArr as float[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;
                        
                    break;
                }
                case int[] sumArr : 
                {
                    int scalar_ = Convert.ToInt32(scalar);
                    int[] np1Array = np1SysArr as int[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;
                        
                    break;
                }
                case Int64[] sumArr : 
                {
                    Int64 scalar_ = Convert.ToInt64(scalar);
                    Int64[] np1Array = np1SysArr as Int64[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;
                        
                    break;
                }
                case Complex[] sumArr : 
                {
                    Complex scalar_ = new Complex(0,0);

                    try 
                    {
                        scalar_ = (Complex) scalar;
                    }
                    catch (InvalidCastException e) 
                    {
                        TypeCode tc = Type.GetTypeCode(scalar.GetType()) ;

                        if (tc == TypeCode.Object)
                        {
                            Quaternion puffer = (Quaternion) scalar;

                            scalar_ = new Complex(puffer.W,puffer.X);
                        }
                        else 
                        {
                            scalar_ = new Complex(Convert.ToDouble(scalar),0 );
                        }
                    }
                    
                    Complex[] np1Array = np1SysArr as Complex[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;
                        
                    break;
                }
                case Quaternion[] sumArr : 
                {
                    Quaternion scalar_ = (Quaternion) scalar;
                    Quaternion[] np1Array = np1SysArr as Quaternion[];

                    for (int idx = 0;idx < np1Array.Length;idx++)
                        sumArr[idx] = np1Array[idx] * scalar_;
                        
                    break;
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }

            
            return sum;
        }
        
        public static NDArray operator *(double scalar, NDArray np1)
        {
            return np1 + scalar;
        }
    }
    public partial class NDArrayGeneric<T>
    {
        public static NDArrayGeneric<T> operator *(NDArrayGeneric<T> np1, NDArrayGeneric<T> np2)
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
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break;
                }
                case float[] sumArray : 
                {
                    float[] np1Array = np1.Data as float[];
                    float[] np2Array = np2.Data as float[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case Complex[] sumArray : 
                {
                    Complex[] np1Array = np1.Data as Complex[];
                    Complex[] np2Array = np2.Data as Complex[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                case Quaternion[] sumArray : 
                {
                    Quaternion[] np1Array = np1.Data as Quaternion[];
                    Quaternion[] np2Array = np2.Data as Quaternion[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * np2Array[idx];
                    break; 
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + typeof(T).Name);
                }
            }

            return (NDArrayGeneric<T>) sum;
        }
        public static NDArrayGeneric<T> operator *(NDArrayGeneric<T> np1, T scalar)
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
                        sumArray[idx] = np1Array[idx] * scalarDouble;
                    break;
                }
                case float scalarFloat : 
                {
                    float[] np1Array = np1.Data as float[];
                    float[] sumArray = sum.Data as float[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * scalarFloat;
                    break; 
                }
                case Complex scalarComplex : 
                {
                    Complex[] np1Array = np1.Data as Complex[];
                    Complex[] sumArray = sum.Data as Complex[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * scalarComplex;
                    break; 
                }
                case Quaternion scalarQuaternion : 
                {
                    Quaternion[] np1Array = np1.Data as Quaternion[];
                    Quaternion[] sumArray = sum.Data as Quaternion[];
                    // for is faster than linq 
                    for (int idx = 0; idx < sumArray.Length;idx++)
                        sumArray[idx] = np1Array[idx] * scalarQuaternion;
                    break; 
                }
                default : 
                {
                    throw new Exception("The operation is not implemented for the "  + typeof(T).Name);
                }
            }

            return (NDArrayGeneric<T>) sum;
        }

        public static NDArrayGeneric<T> operator *(T scalar, NDArrayGeneric<T> np1)
        {
            return np1 * scalar;
        }
    }
}