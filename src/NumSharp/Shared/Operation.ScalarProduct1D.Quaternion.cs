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
        internal static Quaternion[] MuliplyScalarProd1DQuaternion(Quaternion[] np1, Quaternion[]np2)
        {
            Quaternion sum = new Quaternion(0,0,0,0);

            for (int idx = 0; idx < np1.Length;idx++)
                sum += np1[idx] * np2[idx];
            
            return new Quaternion[]{sum};
        }
        //end 1
   }
}
