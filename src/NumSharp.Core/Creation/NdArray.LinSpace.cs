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
        public NDArray linspace(double start, double stop, int num, bool entdpoint = true)
        {
            double steps = (stop - start) / ((entdpoint) ? (double)num - 1.0 : (double)num);

            double[] doubleArray = new double[num];

            for (int idx = 0; idx < doubleArray.Length; idx++)
                doubleArray[idx] = start + idx * steps;

            Storage.ReplaceData(doubleArray);
            Storage.Reshape(doubleArray.Length);

            return this;
        }

        public NDArray linspace(float start, float stop, int num, bool entdpoint = true)
        {
            float steps = (stop - start) / ((entdpoint) ? (float)num - 1.0f : (float)num);

            float[] floatArray = new float[num];

            for (int idx = 0; idx < floatArray.Length; idx++)
                floatArray[idx] = start + idx * steps;

            Storage.ReplaceData(floatArray);
            Storage.Reshape(floatArray.Length);

            return this;
        }
    }
}
