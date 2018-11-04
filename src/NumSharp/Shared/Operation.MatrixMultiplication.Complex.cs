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
        internal static Complex[] MatrixMultiplyComplexMatrix(Complex[]np1, Complex[]np2, int[] dimNp1, int[] dimNp2)
        {
            var result = new Complex[dimNp1[0] * dimNp2[1]];

            int iterator = dimNp1[1];

            for (int idx = 0; idx < result.Length;idx++)
            {
                int line = idx / dimNp1[0];
                int column = idx % dimNp2[1];
                
                for (int kdx = 0; kdx < iterator;kdx++)
                {
                    int index1 = line * dimNp1[1] + kdx;
                    int index2 = dimNp2[1] * kdx + column;

                    result[idx] += np1[index1] * np2[index2];    
                }
            }

            return result;
        }
        //end 1
   }
}
