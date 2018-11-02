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
        internal static decimal[] MuliplyScalarProd1Ddecimal(decimal[] np1, decimal[]np2)
        {
            decimal sum = np1.Select((x,idx) => x * np2[idx] ).Sum();

            return new decimal[]{sum};
        }
        //end 1
   }
}
