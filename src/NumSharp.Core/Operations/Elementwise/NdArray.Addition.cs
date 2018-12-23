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
            int scalarNo = -1;
            
            var sum = np1.DetermineEmptyResult(np2,ref scalarNo);
            
            Array np1SysArr = np1.Storage.GetData();
            Array np2SysArr = np2.Storage.GetData();
            Array np3SysArr = sum.Storage.GetData();

            switch (np3SysArr)
            {
                case double[] sumArray : 
                {
                    double[] np1Array = np1SysArr as double[];
                    double[] np2Array = np2SysArr as double[];

                    double scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<double>()[0] : np2.Storage.GetData<double>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break;
                }
                case float[] sumArray : 
                {
                    float[] np1Array = np1SysArr as float[];
                    float[] np2Array = np2SysArr as float[];

                    float scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<float>()[0] : np2.Storage.GetData<float>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break; 
                }
                case int[] sumArray : 
                {
                    int[] np1Array = np1SysArr as int[];
                    int[] np2Array = np2SysArr as int[];

                    int scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<int>()[0] : np2.Storage.GetData<int>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break;
                }
                case Int64[] sumArray : 
                {
                    Int64[] np1Array = np1SysArr as Int64[];
                    Int64[] np2Array = np2SysArr as Int64[];

                    Int64 scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<Int64>()[0] : np2.Storage.GetData<Int64>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break;
                }
                case Complex[] sumArray : 
                {
                    Complex[] np1Array = np1SysArr as Complex[];
                    Complex[] np2Array = np2SysArr as Complex[];

                    Complex scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<Complex>()[0] : np2.Storage.GetData<Complex>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break;
                }
                case Quaternion[] sumArray : 
                {
                    Quaternion[] np1Array = np1SysArr as Quaternion[];
                    Quaternion[] np2Array = np2SysArr as Quaternion[];

                    Quaternion scalar = ( scalarNo == 1 ) ? np1.Storage.GetData<Quaternion>()[0] : np2.Storage.GetData<Quaternion>()[0];

                    // for is faster than linq 
                    if( scalarNo == 0 )
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = np1Array[idx] + np2Array[idx];
                    else if (scalarNo == 1)
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np2Array[idx];
                    else
                        for (int idx = 0; idx < sumArray.Length;idx++)
                            sumArray[idx] = scalar + np1Array[idx];
                    break; 
                }
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }

            return sum;
        }
        
    }

}
