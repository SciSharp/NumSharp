using System;
using System.Numerics;
using System.Linq;

namespace NumSharp.Shared
{
    internal static partial class ScalarProduct1D
    {
        //start 1 
        internal static double[] MuliplyScalarProd1DDouble(double[] np1, double[]np2)
        {
            double sum = np1.Select((x,idx) => x * np2[idx] ).Sum();

            return new double[]{sum};
        }
        //end 1
    }
}