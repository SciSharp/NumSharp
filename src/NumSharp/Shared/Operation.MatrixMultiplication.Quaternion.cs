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
        internal static Quaternion[] MatrixMultiplyQuaternionMatrix(Quaternion[]np1, Quaternion[]np2, int dim0, int dim1, int iterator)
        {
            var result = new Quaternion[dim0 * dim1];

            for (int idx = 0; idx < result.Length;idx++)
            {
                int line = idx % dim0;
                int column = idx / dim1;

                result[idx] = new Quaternion(0,0,0,0);
                for (int kdx = 0; kdx < 3;kdx++)
                {
                    result[idx] += np1[line + kdx * dim0] * np2[3 * dim1 + kdx];    
                }
            }

            return result;
        }
        //end 1
   }
}
