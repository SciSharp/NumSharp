using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Numerics;
using System.Text;

namespace NumSharp
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
                /*case Quaternion[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = new Quaternion(new Vector3(0,0,0),(float)doubleArray[idx]);
                    break;
                }*/
                default : 
                {
                    throw new IncorrectTypeException();
                }
            }

            this.Storage.Reshape(doubleArray.Length);

            return this;
        }

        public NDArray linspace(float start, float stop, int num, bool entdpoint = true)
        {
            float steps = (stop - start) / ((entdpoint) ? (float)num - 1.0f : (float)num);

            float[] floatArray = new float[num];

            for (int idx = 0; idx < floatArray.Length; idx++)
                floatArray[idx] = start + idx * steps;

            Storage.SetData(Array.CreateInstance(this.dtype, floatArray.Length));

            var data = Storage.GetData();

            switch (data)
            {
                case int[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = (int)floatArray[idx];
                        break;
                    }
                case long[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = (long)floatArray[idx];
                        break;
                    }
                case double[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = floatArray[idx];
                        break;
                    }
                case float[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = (float)floatArray[idx];
                        break;
                    }
                case Complex[] dataArray:
                    {
                        for (int idx = 0; idx < dataArray.Length; idx++)
                            dataArray[idx] = (Complex)floatArray[idx];
                        break;
                    }
                /*case Quaternion[] dataArray : 
                {
                    for(int idx = 0; idx < dataArray.Length;idx++)
                        dataArray[idx] = new Quaternion(new Vector3(0,0,0),(float)doubleArray[idx]);
                    break;
                }*/
                default:
                    {
                        throw new IncorrectTypeException();
                    }
            }

            this.Storage.Reshape(floatArray.Length);

            return this;
        }
    }
}