using System;
using System.Collections.Generic;
using System.Numerics;
using System.Linq;
using System.Text;

namespace NumSharp.Shared
{
   internal static partial class MatrixMultiplication
   {
        //start 1 
        internal static float[] MatrixMultiplyfloatMatrix(float[]np1, float[]np2, int[] dimNp1, int[] dimNp2)
        {
            var result = new float[dimNp1[0] * dimNp2[1]];

            int iterator = dimNp1[1];

            for (int idx = 0; idx < result.Length;idx++)
            {
                int line = idx % dimNp1[0];
                int column = idx / dimNp2[1];
                
                for (int kdx = 0; kdx < iterator;kdx++)
                {
                    result[idx] += np1[line + kdx * dimNp1[0]] * np2[column * dimNp2[0] + kdx];    
                }
            }

            return result;
        }
        //end 1
   }
}
