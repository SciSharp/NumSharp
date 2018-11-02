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
        internal static Int64[] MuliplyScalarProd1DInt64(Int64[] np1, Int64[]np2)
        {
            Int64 sum = np1.Select((x,idx) => x * np2[idx] ).Sum();

            return new Int64[]{sum};
        }
        //end 1
   }
}
