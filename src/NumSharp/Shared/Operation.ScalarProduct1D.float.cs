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
        internal static float[] MuliplyScalarProd1Dfloat(float[] np1, float[]np2)
        {
            float sum = np1.Select((x,idx) => x * np2[idx] ).Sum();

            return new float[]{sum};
        }
        //end 1
   }
}
