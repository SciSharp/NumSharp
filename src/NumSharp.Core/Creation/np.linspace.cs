using System;
using System.Collections.Generic;
using System.Numerics;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray linspace(double start, double stop, int num, bool entdpoint = true, Type dtype = null)
        {
            dtype = (dtype == null) ? typeof(double) : dtype;

            return new NDArray(dtype).linspace(start,stop,num,entdpoint);
        }
        public static NDArray linspace(float start, float stop, int num, bool entdpoint = true, Type dtype = null)
        {
            dtype = (dtype == null) ? typeof(float) : dtype;

            return new NDArray(dtype).linspace(start, stop, num, entdpoint);
        }

        public static NDArray linspace<T>(double start, double stop, int num, bool entdpoint = true)
        {
            double steps = (stop - start) / ((entdpoint) ? (double)num - 1.0 : (double)num);

            double[] doubleArray = new double[num];

            for (int idx = 0; idx < doubleArray.Length; idx++)
                doubleArray[idx] = start + idx * steps;

            var nd = new NDArray(typeof(T), doubleArray.Length);
            nd.SetData(doubleArray);

            return nd;
        }
    }
}
