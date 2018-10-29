using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumSharp.Extensions
{
    public static partial class MatrixExtensions
    {
        public static Matrix<double> Dot(this Matrix<double> np, Matrix<double> np2 )
        {
          int numOfLines = np.Data.GetLength(0);
          int numOfColumns = np2.Data.GetLength(1);

            var result = new Matrix<double>();
            result.Data = new double[numOfLines,numOfColumns];

            for (int idx = 0; idx < numOfLines;idx++)
            {
                for( int jdx = 0; jdx < numOfColumns; jdx++)
                {
                    result.Data[idx,jdx] = 0;
                    for (int kdx = 0; kdx < numOfColumns; kdx++)
                    {
                        result.Data[idx,jdx] += np.Data[idx,kdx] * np2.Data[kdx,jdx];
                    }
                }
            }

            return result;
        }
    }
}
        