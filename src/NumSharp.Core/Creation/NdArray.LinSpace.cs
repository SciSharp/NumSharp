using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp.Core
{
    public partial class NDArray
    {
        public NDArray linspace(double start, double stop,int num, bool entdpoint = true)
        {
            double steps = (stop - start)/((entdpoint) ? (double) num - 1.0 : (double) num);

            double[] doubleArray = new double[num];

            for (int idx = 0; idx < doubleArray.Length;idx++)
                doubleArray[idx] = start + idx * steps;

            Storage.SetData(Array.CreateInstance(this.dtype,doubleArray.Length));

            var data = Storage.GetData();

            switch (data)
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
                    throw new IncorrectTypeException();
                }
            }

            this.Storage.Reshape(doubleArray.Length);

            return this;
        }
    }
}