using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray<T>
    {
        public NDArray<T> linspace(double start, double stop,int num, bool entdpoint = true)
        {
            double steps = (stop - start)/((entdpoint) ? (double) num - 1.0 : (double) num);

            double[] doubleArray = new double[num];

            for (int idx = 0; idx < doubleArray.Length;idx++)
                doubleArray[idx] = start + idx * steps;

            Data = new T[doubleArray.Length];

            switch (Data)
            {
                case int[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = (int)doubleArray[idx];
                    break;
                }
                case long[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = (long) doubleArray[idx];
                    break;
                }
                case double[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = doubleArray[idx];
                    break;
                }
                case float[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = (float)doubleArray[idx];
                    break;
                }
                case Complex[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = (Complex)doubleArray[idx];
                    break;
                }
                case Quaternion[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = new Quaternion(new Vector3(0,0,0),(float)doubleArray[idx]);
                    break;
                }
                default : 
                {
                    throw new Exception("This method was not yet implemented for this type" + typeof(T).Name);
                }
            }

            this.Shape = new Shape(doubleArray.Length);

            return this;
        }
    }
}
