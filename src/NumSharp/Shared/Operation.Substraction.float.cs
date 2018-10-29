using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Text;

namespace NumSharp.Shared
{
   internal static partial class Substraction
   {
        //start 1 
        internal static float[] SubfloatArrayFromfloatArray(float[] np1, float[]np2)
        {
            return np1.Select((x,idx) => x - np2[idx]).ToArray();
        }
        //end 1
        //start 2
        internal static float[] SubfloatFromfloatArray(float[] np1, float np2)
        {
            return np1.Select((x) => x - np2).ToArray();
        }
        //end 2
        //start 3 
        internal static float[][] SubfloatMatrixFromfloatMatrix(float[][] np1, float[][]np2)
        {
            return np1.Select((x,idx) => x.Select((y,jdx) => y - np2[idx][jdx] ).ToArray()).ToArray();
        }
        //end 3
        //start 4
        internal static float[][] SubfloatFromfloatMatrix(float[][] np1, float np2)
        {
            return np1.Select((x) => x.Select((y) => y - np2 ).ToArray()).ToArray();
        }
        //end 4
   }
}
