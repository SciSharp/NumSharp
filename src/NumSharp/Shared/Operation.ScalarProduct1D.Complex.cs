using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Text;

namespace NumSharp.Shared
{
   internal static partial class ScalarProduct1D
   {
        //start 1 
        internal static Complex[] MuliplyScalarProd1DComplex(Complex[] np1, Complex[]np2)
        {
            Complex sum = new Complex(0,0);
            for (int idx = 0; idx < np1.Length;idx++)
                sum += np1[idx] * np2[idx];
                
            return new Complex[]{sum};
        }
        //end 1
   }
}
