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
        internal static decimal[][] MatrixMultiplydecimalMatrix(decimal[][] np1, decimal[][]np2)
        {
            int numOfLines = np1.GetLength(0);
            int numOfColumns = np2[0].GetLength(0);

            int iterator = np1[0].GetLength(0);

            var result = new decimal[numOfLines][];
            result = result.Select(x => new decimal[numOfColumns]).ToArray();

            for (int idx = 0; idx < numOfLines;idx++)
            {
                for( int jdx = 0; jdx < numOfColumns; jdx++)
                {
                    result[idx][jdx] = 0;
                    for (int kdx = 0; kdx < iterator; kdx++)
                    {
                        result[idx][jdx] += np1[idx][kdx] * np2[kdx][jdx];
                    }
                }
            }
            return result;
        }
        //end 1
   }
}
